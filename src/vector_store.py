from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import faiss

from .embedder import SentenceEmbedder
from .node_extractor import SentenceRecord


@dataclass
class RetrievedSnippet:
    text: str
    similarity: float
    index: int


class SentenceVectorStore:
    def __init__(self, sentences: Sequence[SentenceRecord], embedder: SentenceEmbedder) -> None:
        self.sentences: List[SentenceRecord] = list(sentences)
        self.embedder = embedder
        if not self.sentences:
            raise ValueError("Sentence store cannot be empty")
        self.embeddings = embedder.encode([sent.text for sent in self.sentences])
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

    def query(self, text: str, top_k: int) -> List[RetrievedSnippet]:
        vec = self.embedder.encode([text])[0]
        distances, indices = self.index.search(vec[None, :], top_k)
        snippets: List[RetrievedSnippet] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            snippets.append(
                RetrievedSnippet(
                    text=self.sentences[idx].text,
                    similarity=float(score),
                    index=int(idx),
                )
            )
        return snippets
