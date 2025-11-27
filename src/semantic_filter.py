from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .config import SemanticFilterConfig
from .llm_verifier import GeminiVerifier
from .node_extractor import NodeRecord
from .structural_filter import CandidatePair
from .vector_store import SentenceVectorStore

# Import RAG-HyDE from enhanced_components
try:
    import sys
    from pathlib import Path
    enhanced_path = Path(__file__).parent.parent / "enhanced_components"
    if enhanced_path.exists():
        sys.path.insert(0, str(enhanced_path.parent))
        from enhanced_components.rag_hyde import RAGHyDE
        RAG_HYDE_AVAILABLE = True
    else:
        RAG_HYDE_AVAILABLE = False
except ImportError:
    RAG_HYDE_AVAILABLE = False


@dataclass
class SemanticCandidate:
    node_i: int
    node_j: int
    node_i_label: str
    node_j_label: str
    support: float
    temporal: float
    mechanistic: float
    contexts: List[str]
    rationale: Optional[str] = None


class SemanticFilter:
    def __init__(
        self,
        vector_store: SentenceVectorStore,
        nodes: Sequence[NodeRecord],
        config: SemanticFilterConfig,
        llm_verifier: Optional[GeminiVerifier] = None,
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
    ) -> None:
        self.store = vector_store
        self.config = config
        self.node_lookup = {node.node_id: node.label for node in nodes}
        self.llm_verifier = llm_verifier
        self.progress_callback = progress_callback
        
        # Initialize RAG-HyDE if available and enabled
        self.rag_hyde: Optional[RAGHyDE] = None
        if RAG_HYDE_AVAILABLE and config.use_rag_hyde:
            try:
                # Try to get API key from config or environment
                api_key = config.gemini_api_key
                if not api_key:
                    import os
                    api_key = os.getenv(config.gemini_api_key_env)
                
                if api_key:
                    self.rag_hyde = RAGHyDE(
                        api_key=api_key,
                        model_name=config.gemini_model,
                        temperature=config.rag_hyde_temperature,
                        k_hypothetical=config.rag_hyde_k,
                    )
                    print("[semantic] RAG-HyDE initialized successfully")
                else:
                    print("[semantic] RAG-HyDE disabled: No API key available, falling back to templates")
            except Exception as e:
                print(f"[semantic] RAG-HyDE initialization failed: {e}, falling back to templates")
                self.rag_hyde = None

    def run(
        self,
        candidates: Sequence[CandidatePair],
        result_callback: Optional[Callable[[SemanticCandidate], None]] = None,
    ) -> List[SemanticCandidate]:
        results: List[SemanticCandidate] = []
        total = len(candidates)
        if self.progress_callback is not None and total > 0:
            self.progress_callback(0, total, 0)
        for index, pair in enumerate(candidates, start=1):
            score = self._score_pair(pair)
            if score is None:
                pass
            elif score.support < self.config.tau_plausible:
                pass
            elif score.temporal < self.config.tau_temporal and score.mechanistic < self.config.tau_mechanistic:
                pass
            else:
                results.append(score)
                if result_callback is not None:
                    result_callback(score)
            if (
                self.progress_callback is not None
                and self.config.progress_every > 0
                and (index % self.config.progress_every == 0 or index == total)
            ):
                self.progress_callback(index, total, len(results))
        results.sort(key=lambda item: item.support, reverse=True)
        return results

    def _score_pair(self, pair: CandidatePair) -> SemanticCandidate | None:
        label_i = self.node_lookup.get(pair.node_i)
        label_j = self.node_lookup.get(pair.node_j)
        if not label_i or not label_j:
            return None
        hypotheses = self._generate_hypotheses(label_i, label_j)
        rrf_scores: Dict[int, float] = defaultdict(float)
        similarity_cache: Dict[int, float] = {}

        for hypothesis in hypotheses:
            query_vec = self.store.embedder.encode([hypothesis])[0]
            base_results = self.store.query(hypothesis, self.config.pool_base)
            base_support = np.mean([self._normalize_similarity(snippet.similarity) for snippet in base_results]) if base_results else 0.0
            k_pool = int(self.config.pool_base + (1.0 - base_support) * self.config.pool_expansion)
            pool_results = self.store.query(hypothesis, k_pool)
            candidate_indices = [snippet.index for snippet in pool_results]
            mmr_selected = self._mmr_select(query_vec, candidate_indices, self.config.rag_top_k)
            for rank, idx in enumerate(mmr_selected, start=1):
                rrf_scores[idx] += 1.0 / (self.config.rrf_kappa + rank)
            for snippet in pool_results:
                normed = self._normalize_similarity(snippet.similarity)
                similarity_cache[snippet.index] = max(similarity_cache.get(snippet.index, 0.0), normed)

        if not rrf_scores:
            return None
        top_indices = [idx for idx, _ in sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[: self.config.rag_top_k]]
        contexts = [self.store.sentences[idx].text for idx in top_indices]
        support_values = [similarity_cache.get(idx, 0.0) for idx in top_indices]
        support = float(np.mean(support_values)) if support_values else 0.0
        temporal = self._marker_ratio(contexts, self.config.temporal_markers)
        mechanistic = self._marker_ratio(contexts, self.config.mechanistic_markers)
        rationale: Optional[str] = None

        if self.llm_verifier is not None and contexts:
            print(f"[semantic] Calling Gemini for pair ({label_i[:30]}.../{label_j[:30]}...)")
            llm_score = self.llm_verifier.score(label_i, label_j, contexts)
            if llm_score is None:
                print(f"[semantic] Gemini returned None, falling back to heuristic scores")
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

    def _mmr_select(self, query_vec: np.ndarray, candidates: Sequence[int], k: int) -> List[int]:
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
                score = self.config.mmr_lambda * relevance - (1.0 - self.config.mmr_lambda) * diversity
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is None:
                break
            selected.append(best_idx)
            candidate_set.remove(best_idx)
        return selected

    def _marker_ratio(self, contexts: Sequence[str], markers: Sequence[str]) -> float:
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
        return (value + 1.0) / 2.0

    def _generate_hypotheses(self, node_i: str, node_j: str) -> List[str]:
        """Generate hypotheses using RAG-HyDE if available, otherwise use templates."""
        # Try RAG-HyDE first if available
        if self.rag_hyde is not None:
            try:
                hypotheses = self.rag_hyde.generate_hypotheses(node_i, node_j, k=self.config.rag_hyde_k)
                # Extract text from Hypothesis objects
                hypothesis_texts = [hyp.text for hyp in hypotheses]
                if hypothesis_texts:
                    return hypothesis_texts
            except Exception as e:
                print(f"[semantic] RAG-HyDE generation failed for ({node_i}/{node_j}): {e}, using templates")
        
        # Fallback to template-based hypotheses
        templates = [
            "{a} leads to {b}",
            "{a} affects {b}",
            "relationship between {a} and {b}",
            "{b} depends on {a}",
        ]
        return [template.format(a=node_i, b=node_j) for template in templates]
