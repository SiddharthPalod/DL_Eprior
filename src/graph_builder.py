from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import scipy.sparse as sp

from .node_extractor import NodeRecord


@dataclass
class GraphArtifacts:
    adjacency: sp.coo_matrix
    occurrences: Dict[int, List[Tuple[int, int]]]
    paragraph_to_nodes: Dict[int, Set[int]]


def build_cooccurrence_graph(nodes: Sequence[NodeRecord]) -> GraphArtifacts:
    if not nodes:
        raise ValueError("Cannot build a graph without nodes")

    paragraph_map: Dict[int, Set[int]] = defaultdict(set)
    for node in nodes:
        for chunk_id, _ in node.occurrences:
            paragraph_map[chunk_id].add(node.node_id)

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    for node_ids in paragraph_map.values():
        ids = sorted(node_ids)
        for i_idx in range(len(ids)):
            for j_idx in range(i_idx + 1, len(ids)):
                i = ids[i_idx]
                j = ids[j_idx]
                rows.extend([i, j])
                cols.extend([j, i])
                data.extend([1.0, 1.0])

    size = len(nodes)
    adjacency = sp.coo_matrix((data, (rows, cols)), shape=(size, size))
    occurrences = {node.node_id: node.occurrences for node in nodes}
    return GraphArtifacts(adjacency=adjacency, occurrences=occurrences, paragraph_to_nodes=paragraph_map)
