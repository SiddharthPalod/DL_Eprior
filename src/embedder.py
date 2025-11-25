from __future__ import annotations

from typing import Iterable, List, Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceEmbedder:
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str], normalize: bool = True) -> np.ndarray:
        embeddings = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        if normalize:
            faiss.normalize_L2(embeddings)
        return embeddings.astype(np.float32)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    faiss.normalize_L2(vectors)
    return vectors
