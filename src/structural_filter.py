from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

import faiss
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from .config import StructuralFilterConfig


def _normalize_adjacency(adjacency: sp.coo_matrix) -> torch.Tensor:
    matrix = adjacency.tocsr()
    matrix = matrix + sp.eye(matrix.shape[0], format="csr")
    degree = np.array(matrix.sum(axis=1)).flatten()
    degree[degree == 0.0] = 1.0
    inv_sqrt = 1.0 / np.sqrt(degree)
    d_mat = sp.diags(inv_sqrt)
    normalized = d_mat @ matrix @ d_mat
    dense = normalized.toarray().astype(np.float32)
    return torch.from_numpy(dense)


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.01)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        support = x @ self.weight
        return adj_norm @ support


class GraphAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        h1 = torch.relu(self.gc1(x, adj_norm))
        return self.gc2(h1, adj_norm)


@dataclass
class CandidatePair:
    node_i: int
    node_j: int
    structural_score: float


def _sample_edges(adjacency: sp.coo_matrix) -> List[Tuple[int, int]]:
    matrix = adjacency.tocoo()
    positives: Set[Tuple[int, int]] = set()
    for i, j in zip(matrix.row, matrix.col):
        if i == j:
            continue
        ordered = (min(i, j), max(i, j))
        positives.add(ordered)
    return list(positives)


def _negative_samples(num_nodes: int, positives: Set[Tuple[int, int]], count: int) -> List[Tuple[int, int]]:
    negatives: Set[Tuple[int, int]] = set()
    while len(negatives) < count and len(negatives) < num_nodes * num_nodes:
        i = random.randrange(num_nodes)
        j = random.randrange(num_nodes)
        if i == j:
            continue
        ordered = (min(i, j), max(i, j))
        if ordered in positives:
            continue
        negatives.add(ordered)
    return list(negatives)


def train_structural_filter(
    adjacency: sp.coo_matrix,
    node_features: np.ndarray,
    config: StructuralFilterConfig,
) -> np.ndarray:
    num_nodes, feature_dim = node_features.shape
    if num_nodes < 2:
        return node_features

    adj_norm = _normalize_adjacency(adjacency)
    model = GraphAutoEncoder(feature_dim, config.hidden_dim, config.latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    x_tensor = torch.from_numpy(node_features.astype(np.float32))

    positives = _sample_edges(adjacency)
    if not positives:
        return node_features
    pos_tensor = torch.tensor(positives, dtype=torch.long)

    for _ in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        z = model(x_tensor, adj_norm)

        neg_samples = _negative_samples(num_nodes, set(positives), len(positives))
        if not neg_samples:
            break
        neg_tensor = torch.tensor(neg_samples, dtype=torch.long)

        pos_scores = (z[pos_tensor[:, 0]] * z[pos_tensor[:, 1]]).sum(dim=1)
        neg_scores = (z[neg_tensor[:, 0]] * z[neg_tensor[:, 1]]).sum(dim=1)

        loss = -torch.log(torch.sigmoid(pos_scores) + 1e-9).mean()
        loss -= torch.log(1 - torch.sigmoid(neg_scores) + 1e-9).mean()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        embeddings = model(x_tensor, adj_norm).cpu().numpy()
    faiss.normalize_L2(embeddings)
    return embeddings.astype(np.float32)


def build_structural_candidates(
    embeddings: np.ndarray,
    config: StructuralFilterConfig,
) -> List[CandidatePair]:
    if embeddings.shape[0] < 2:
        return []
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    _, neighbors = index.search(embeddings, config.top_k_neighbors + 1)

    candidates: Dict[Tuple[int, int], float] = {}
    for i, row in enumerate(neighbors):
        for j in row:
            if i == j or j == -1:
                continue
            ordered = tuple(sorted((i, int(j))))
            if ordered in candidates:
                continue
            score = float(np.dot(embeddings[ordered[0]], embeddings[ordered[1]]))
            candidates[ordered] = score
    result = [CandidatePair(node_i=i, node_j=j, structural_score=score) for (i, j), score in candidates.items()]
    result.sort(key=lambda item: item.structural_score, reverse=True)
    return result
