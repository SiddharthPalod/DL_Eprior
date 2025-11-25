"""Example: Integrating enhanced components into the existing pipeline.

This script demonstrates how to use the new components to replace
the simplified implementations in src/semantic_filter.py.
"""

from pathlib import Path
from typing import List, Optional

import sys

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from enhanced_components.hypothesis_verifier import HypothesisVerifier, VerifiedHypothesis
from enhanced_components.rag_hyde import Hypothesis, RAGHyDE
from src.config import SemanticFilterConfig
from src.llm_verifier import GeminiVerifier
from src.node_extractor import NodeRecord
from src.semantic_filter import CandidatePair, SemanticCandidate
from src.vector_store import SentenceVectorStore


class EnhancedSemanticFilter:
    """Enhanced semantic filter using RAG-HyDE and hypothesis verification.
    
    This replaces the template-based hypothesis generation in SemanticFilter
    with LLM-generated hypotheses and adds verification with semantic entropy.
    """
    
    def __init__(
        self,
        vector_store: SentenceVectorStore,
        nodes: List[NodeRecord],
        config: SemanticFilterConfig,
        llm_verifier: Optional[GeminiVerifier] = None,
        use_enhanced: bool = True,
    ) -> None:
        """Initialize enhanced semantic filter.
        
        Args:
            vector_store: Sentence vector store for RAG
            nodes: Node records
            config: Semantic filter configuration
            llm_verifier: Optional LLM verifier for final scoring
            use_enhanced: Whether to use enhanced components (RAG-HyDE + verification)
        """
        self.store = vector_store
        self.config = config
        self.node_lookup = {node.node_id: node.label for node in nodes}
        self.llm_verifier = llm_verifier
        self.use_enhanced = use_enhanced
        
        if use_enhanced:
            # Initialize RAG-HyDE
            self.rag_hyde = RAGHyDE(
                api_key=config.gemini_api_key,
                model_name=config.gemini_model,
                temperature=0.7,
                k_hypothetical=4,
            )
            
            # Initialize hypothesis verifier
            self.hypothesis_verifier = HypothesisVerifier(
                vector_store=vector_store,
                api_key=config.gemini_api_key,
                model_name=config.gemini_model,
                temperature=0.7,
                tau_support=config.tau_plausible,
                tau_entropy=config.entropy_threshold,
            )
    
    def score_pair(self, pair: CandidatePair) -> Optional[SemanticCandidate]:
        """Score a candidate pair using enhanced pipeline."""
        label_i = self.node_lookup.get(pair.node_i)
        label_j = self.node_lookup.get(pair.node_j)
        if not label_i or not label_j:
            return None
        
        if self.use_enhanced:
            # Step 1: Generate hypotheses with RAG-HyDE
            hypotheses = self.rag_hyde.generate_hypotheses(label_i, label_j)
            
            # Step 2: Verify hypotheses (RAV pipeline)
            verified_hypotheses = self.hypothesis_verifier.verify_hypotheses(hypotheses)
            
            if not verified_hypotheses:
                # No verified hypotheses, skip this pair
                return None
            
            # Step 3: Use verified hypotheses for RRF (same as original)
            from collections import defaultdict
            import numpy as np
            
            rrf_scores: dict[int, float] = defaultdict(float)
            similarity_cache: dict[int, float] = {}
            
            for vh in verified_hypotheses:
                hypothesis_text = vh.hypothesis.text
                query_vec = self.store.embedder.encode([hypothesis_text])[0]
                
                # Adaptive pooling based on verification confidence
                base_results = self.store.query(hypothesis_text, self.config.pool_base)
                base_support = np.mean([
                    self._normalize_similarity(snippet.similarity)
                    for snippet in base_results
                ]) if base_results else 0.0
                
                # Use verification score for adaptive pooling
                confidence = vh.support_score * (1.0 - vh.semantic_entropy)
                k_pool = int(
                    self.config.pool_base
                    + (1.0 - confidence) * self.config.pool_expansion
                )
                
                pool_results = self.store.query(hypothesis_text, k_pool)
                candidate_indices = [snippet.index for snippet in pool_results]
                mmr_selected = self._mmr_select(query_vec, candidate_indices, self.config.rag_top_k)
                
                for rank, idx in enumerate(mmr_selected, start=1):
                    rrf_scores[idx] += 1.0 / (self.config.rrf_kappa + rank)
                
                for snippet in pool_results:
                    normed = self._normalize_similarity(snippet.similarity)
                    similarity_cache[snippet.index] = max(
                        similarity_cache.get(snippet.index, 0.0), normed
                    )
        else:
            # Fallback to original template-based approach
            hypotheses = self._generate_hypotheses_template(label_i, label_j)
            # ... (rest of original implementation)
            return None  # Simplified for example
        
        if not rrf_scores:
            return None
        
        # Get top contexts
        top_indices = [
            idx
            for idx, _ in sorted(
                rrf_scores.items(), key=lambda item: item[1], reverse=True
            )[: self.config.rag_top_k]
        ]
        contexts = [self.store.sentences[idx].text for idx in top_indices]
        support_values = [similarity_cache.get(idx, 0.0) for idx in top_indices]
        support = float(np.mean(support_values)) if support_values else 0.0
        
        temporal = self._marker_ratio(contexts, self.config.temporal_markers)
        mechanistic = self._marker_ratio(contexts, self.config.mechanistic_markers)
        rationale: Optional[str] = None
        
        # Optional: Final LLM verification
        if self.llm_verifier is not None and contexts:
            llm_score = self.llm_verifier.score(label_i, label_j, contexts)
            if llm_score:
                support = llm_score.support
                temporal = llm_score.temporal
                mechanistic = llm_score.mechanistic
                rationale = llm_score.rationale or rationale
        
        return SemanticCandidate(
            node_i=pair.node_i,
            node_j=pair.node_j,
            node_i_label=label_i,
            node_j_label=label_j,
            support=support,
            temporal=temporal,
            mechanistic=mechanistic,
            contexts=contexts,
            rationale=rationale,
        )
    
    def _mmr_select(self, query_vec, candidates, k: int) -> List[int]:
        """MMR selection (same as original)."""
        # Implementation from src/semantic_filter.py
        selected: List[int] = []
        candidate_set = list(dict.fromkeys(candidates))
        while candidate_set and len(selected) < k:
            best_idx = None
            best_score = float("-inf")
            for idx in candidate_set:
                doc_vec = self.store.embeddings[idx]
                relevance = float(np.dot(query_vec, doc_vec))
                diversity = 0.0
                if selected:
                    diversity = max(
                        float(np.dot(doc_vec, self.store.embeddings[sel_idx]))
                        for sel_idx in selected
                    )
                score = (
                    self.config.mmr_lambda * relevance
                    - (1.0 - self.config.mmr_lambda) * diversity
                )
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is None:
                break
            selected.append(best_idx)
            candidate_set.remove(best_idx)
        return selected
    
    def _marker_ratio(self, contexts, markers) -> float:
        """Marker ratio (same as original)."""
        if not contexts:
            return 0.0
        hits = 0
        for text in contexts:
            lowered = text.lower()
            if any(marker in lowered for marker in markers):
                hits += 1
        return hits / len(contexts)
    
    @staticmethod
    def _normalize_similarity(value: float) -> float:
        """Normalize similarity (same as original)."""
        return (value + 1.0) / 2.0
    
    @staticmethod
    def _generate_hypotheses_template(node_i: str, node_j: str) -> List[str]:
        """Template-based fallback."""
        templates = [
            f"{node_i} leads to {node_j}",
            f"{node_i} affects {node_j}",
            f"relationship between {node_i} and {node_j}",
            f"{node_j} depends on {node_i}",
        ]
        return templates


if __name__ == "__main__":
    print("Enhanced Semantic Filter Example")
    print("=" * 50)
    print("\nThis module demonstrates how to integrate:")
    print("1. RAG-HyDE for hypothesis generation")
    print("2. Hypothesis verification with semantic entropy")
    print("3. Enhanced RRF with verified hypotheses")
    print("\nSee README.md for full usage examples.")

