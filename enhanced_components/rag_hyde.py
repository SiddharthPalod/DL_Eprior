"""RAG-HyDE: Hypothetical Document Embeddings for hypothesis generation.

This module implements Priority 1: LLM-based hypothesis generation
replacing the template-based approach in the original semantic filter.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import google.generativeai as genai
from google.generativeai.types import GenerationConfig


@dataclass
class Hypothesis:
    """A generated hypothesis about a causal relationship."""
    text: str
    confidence: Optional[float] = None


class RAGHyDE:
    """Generates hypothetical documents using LLM for RAG-HyDE approach.
    
    Instead of using template-based hypotheses, this class queries an LLM
    to generate k hypothetical sentences describing plausible causal relationships
    between two nodes.
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.7,
        api_key_env: str = "GEMINI_API_KEY",
        timeout_seconds: float = 30.0,
        k_hypothetical: int = 4,
    ) -> None:
        """Initialize RAG-HyDE hypothesis generator.
        
        Args:
            api_key: Gemini API key (or use environment variable)
            model_name: Gemini model to use
            temperature: Sampling temperature (0.6-0.8 recommended for semantic entropy)
            api_key_env: Environment variable name for API key
            timeout_seconds: Timeout for API calls
            k_hypothetical: Number of hypotheses to generate per pair
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
        self.k_hypothetical = k_hypothetical
    
    def generate_hypotheses(
        self,
        node_i: str,
        node_j: str,
        k: Optional[int] = None,
    ) -> List[Hypothesis]:
        """Generate k hypothetical sentences describing a plausible relationship.
        
        Args:
            node_i: First node name
            node_j: Second node name
            k: Number of hypotheses to generate (defaults to self.k_hypothetical)
        
        Returns:
            List of Hypothesis objects with generated text
        """
        k = k or self.k_hypothetical
        
        prompt = self._build_prompt(node_i, node_j, k)
        
        try:
            start_time = time.time()
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=self.temperature,
                    response_mime_type="application/json",
                ),
            )
            elapsed = time.time() - start_time
            print(f"[rag-hyde] Generated {k} hypotheses in {elapsed:.1f}s")
            
            payload = self._extract_text(response)
            data = json.loads(payload)
            
            hypotheses = []
            if isinstance(data, dict) and "hypotheses" in data:
                for h_text in data["hypotheses"]:
                    if isinstance(h_text, str) and h_text.strip():
                        hypotheses.append(Hypothesis(text=h_text.strip()))
            elif isinstance(data, list):
                for h_text in data:
                    if isinstance(h_text, str) and h_text.strip():
                        hypotheses.append(Hypothesis(text=h_text.strip()))
            
            # Ensure we have at least k hypotheses (fallback to templates if needed)
            if len(hypotheses) < k:
                print(f"[rag-hyde] Warning: Only got {len(hypotheses)} hypotheses, expected {k}")
                # Fallback to template-based if LLM fails
                templates = [
                    f"{node_i} leads to {node_j}",
                    f"{node_i} affects {node_j}",
                    f"relationship between {node_i} and {node_j}",
                    f"{node_j} depends on {node_i}",
                ]
                for template in templates[:k - len(hypotheses)]:
                    if template not in [h.text for h in hypotheses]:
                        hypotheses.append(Hypothesis(text=template))
            
            return hypotheses[:k]
            
        except Exception as e:
            print(f"[rag-hyde] Error generating hypotheses: {type(e).__name__}: {e}")
            # Fallback to template-based hypotheses
            templates = [
                f"{node_i} leads to {node_j}",
                f"{node_i} affects {node_j}",
                f"relationship between {node_i} and {node_j}",
                f"{node_j} depends on {node_i}",
            ]
            return [Hypothesis(text=t) for t in templates[:k]]
    
    @staticmethod
    def _build_prompt(node_i: str, node_j: str, k: int) -> str:
        """Build the prompt for hypothesis generation."""
        return (
            f"Generate {k} hypothetical short sentences describing a plausible causal relationship "
            f"between '{node_i}' and '{node_j}'. "
            f"Each sentence should be concise (10-20 words) and describe a potential causal mechanism. "
            f"Return a JSON object with a 'hypotheses' array containing {k} strings.\n\n"
            f"Example format: {{\"hypotheses\": [\"sentence 1\", \"sentence 2\", ...]}}"
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

