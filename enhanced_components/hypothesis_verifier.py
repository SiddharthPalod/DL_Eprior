"""Hypothesis Verification via RAG + Semantic Entropy Filtering.

This module implements Priority 2: Full RAV (Retrieval-Augmented Verification)
pipeline with semantic entropy estimation to filter hallucinated hypotheses.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Sequence

import google.generativeai as genai
import numpy as np
from google.generativeai.types import GenerationConfig

from .rag_hyde import Hypothesis


@dataclass
class VerifiedHypothesis:
    """A hypothesis that has passed verification."""
    hypothesis: Hypothesis
    support_score: float  # p_support from verification
    semantic_entropy: float  # H_semantic
    evidence_snippets: List[str]
    verified: bool  # Whether it passed dual filtering


class HypothesisVerifier:
    """Verifies hypotheses using RAG + Semantic Entropy filtering.
    
    Implements the full RAV pipeline:
    1. Evidence retrieval via RAG for each hypothesis
    2. LLM self-check verification (f_verify)
    3. Semantic entropy estimation (multiple stochastic forward passes)
    4. Dual filtering: p_support > τ_support AND H_semantic < τ_entropy
    """
    
    def __init__(
        self,
        vector_store,  # SentenceVectorStore from src.vector_store
        api_key: str | None = None,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.7,  # Moderate temperature for entropy estimation
        api_key_env: str = "GEMINI_API_KEY",
        timeout_seconds: float = 30.0,
        k_rag: int = 3,  # Top-k RAG snippets for verification
        tau_support: float = 0.5,  # Support threshold
        tau_entropy: float = 0.45,  # Entropy threshold
        r_samples: int = 5,  # Number of independent samples for entropy
    ) -> None:
        """Initialize hypothesis verifier.
        
        Args:
            vector_store: SentenceVectorStore for RAG retrieval
            api_key: Gemini API key
            model_name: Gemini model name
            temperature: Temperature for stochastic sampling (0.6-0.8 recommended)
            api_key_env: Environment variable for API key
            timeout_seconds: API timeout
            k_rag: Number of RAG snippets to retrieve for verification
            tau_support: Support score threshold
            tau_entropy: Semantic entropy threshold
            r_samples: Number of independent samples for entropy estimation
        """
        env_name = api_key_env or "GEMINI_API_KEY"
        resolved_key = api_key or os.getenv(env_name)
        if not resolved_key:
            raise RuntimeError(
                f"Gemini API key is missing. Provide api_key or set the {env_name} environment variable."
            )
        
        genai.configure(api_key=resolved_key)
        self.model = genai.GenerativeModel(model_name)
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.vector_store = vector_store
        self.k_rag = k_rag
        self.tau_support = tau_support
        self.tau_entropy = tau_entropy
        self.r_samples = r_samples
    
    def verify_hypotheses(
        self,
        hypotheses: List[Hypothesis],
    ) -> List[VerifiedHypothesis]:
        """Verify a list of hypotheses using RAG + Semantic Entropy.
        
        Args:
            hypotheses: List of hypotheses to verify
        
        Returns:
            List of VerifiedHypothesis objects (only those that pass dual filtering)
        """
        verified_list: List[VerifiedHypothesis] = []
        
        for hypothesis in hypotheses:
            # Step 2: Evidence Retrieval via RAG
            evidence_snippets = self._retrieve_evidence(hypothesis.text)
            
            if not evidence_snippets:
                # No evidence found, skip this hypothesis
                continue
            
            # Step 3: LLM Self-Check Verification
            support_score = self._verify_support(hypothesis.text, evidence_snippets)
            
            if support_score is None:
                continue
            
            # Step 4: Semantic Entropy Estimation
            semantic_entropy = self._estimate_semantic_entropy(
                hypothesis.text,
                evidence_snippets,
            )
            
            # Step 5: Dual Filtering
            verified = (
                support_score > self.tau_support
                and semantic_entropy < self.tau_entropy
            )
            
            verified_hyp = VerifiedHypothesis(
                hypothesis=hypothesis,
                support_score=support_score,
                semantic_entropy=semantic_entropy,
                evidence_snippets=evidence_snippets,
                verified=verified,
            )
            
            if verified:
                verified_list.append(verified_hyp)
                print(
                    f"[verifier] Verified: support={support_score:.3f}, "
                    f"entropy={semantic_entropy:.3f}"
                )
            else:
                print(
                    f"[verifier] Rejected: support={support_score:.3f} "
                    f"(<{self.tau_support}), entropy={semantic_entropy:.3f} "
                    f"(>{self.tau_entropy})"
                )
        
        return verified_list
    
    def _retrieve_evidence(self, hypothesis_text: str) -> List[str]:
        """Step 2: Retrieve top-k RAG snippets for verification."""
        snippets = self.vector_store.query(hypothesis_text, self.k_rag)
        return [snippet.text for snippet in snippets]
    
    def _verify_support(
        self,
        hypothesis_text: str,
        evidence_snippets: List[str],
    ) -> Optional[float]:
        """Step 3: LLM self-check verification.
        
        Returns:
            Support score p_support in [0, 1], or None if verification fails
        """
        prompt = self._build_verification_prompt(hypothesis_text, evidence_snippets)
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=0.2,  # Lower temperature for deterministic verification
                    response_mime_type="application/json",
                ),
            )
            
            payload = self._extract_text(response)
            data = json.loads(payload)
            
            # Extract support score
            if isinstance(data, dict):
                # Try different possible keys
                support = data.get("support", data.get("score", data.get("confidence")))
                if support is not None:
                    return float(max(0.0, min(1.0, float(support))))
                
                # Try YES/NO format
                answer = data.get("answer", data.get("verification", "")).upper()
                if "YES" in answer:
                    return 0.9
                elif "NO" in answer:
                    return 0.1
            
            return None
            
        except Exception as e:
            print(f"[verifier] Verification failed: {type(e).__name__}: {e}")
            return None
    
    def _estimate_semantic_entropy(
        self,
        hypothesis_text: str,
        evidence_snippets: List[str],
    ) -> float:
        """Step 4: Estimate semantic entropy via multiple stochastic samples.
        
        Performs R independent forward passes with temperature-scaled sampling
        and computes: H_semantic = -Σ p_r log p_r
        """
        prompt = self._build_verification_prompt(hypothesis_text, evidence_snippets)
        
        responses: List[str] = []
        
        # Collect R independent samples
        for r in range(self.r_samples):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=GenerationConfig(
                        temperature=self.temperature,  # Use moderate temperature
                        response_mime_type="application/json",
                    ),
                )
                
                payload = self._extract_text(response)
                data = json.loads(payload)
                
                # Extract answer (YES/NO)
                if isinstance(data, dict):
                    answer = data.get("answer", data.get("verification", "")).upper()
                    if "YES" in answer:
                        responses.append("YES")
                    elif "NO" in answer:
                        responses.append("NO")
                    else:
                        # Try to extract from support score
                        support = data.get("support", data.get("score"))
                        if support is not None:
                            responses.append("YES" if float(support) > 0.5 else "NO")
                
            except Exception as e:
                print(f"[verifier] Entropy sample {r+1} failed: {type(e).__name__}: {e}")
                continue
        
        if not responses:
            # If all samples failed, return high entropy (uncertain)
            return 1.0
        
        # Compute semantic entropy: H = -Σ p_r log p_r
        # Count occurrences of each semantic outcome
        counter = Counter(responses)
        total = len(responses)
        
        entropy = 0.0
        for count in counter.values():
            p_r = count / total
            if p_r > 0:
                entropy -= p_r * np.log(p_r)
        
        # Normalize to [0, 1] range (max entropy for binary is log(2) ≈ 0.693)
        # We normalize by log(2) to get [0, 1]
        max_entropy = np.log(2)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return float(normalized_entropy)
    
    @staticmethod
    def _build_verification_prompt(
        hypothesis_text: str,
        evidence_snippets: List[str],
    ) -> str:
        """Build prompt for LLM verification."""
        evidence_text = "\n".join(
            f"{idx+1}. {snippet}" for idx, snippet in enumerate(evidence_snippets)
        )
        
        return (
            f"Claim: '{hypothesis_text}'\n\n"
            f"Evidence:\n{evidence_text}\n\n"
            f"Does the Evidence strongly support the Claim? "
            f"Respond with JSON: {{\"answer\": \"YES\" or \"NO\", \"support\": <float 0-1>}}"
        )
    
    @staticmethod
    def _extract_text(response) -> str:
        """Extract text from Gemini response."""
        if getattr(response, "text", None):
            return response.text
        chunks: list[str] = []
        for candidate in getattr(response, "candidates", []):
            content = getattr(candidate, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []):
                text = getattr(part, "text", "")
                if text:
                    chunks.append(text)
        if not chunks:
            raise ValueError("Model response did not contain text content.")
        return "".join(chunks)

