from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class StructuralFilterConfig:
    hidden_dim: int = 128
    latent_dim: int = 64
    epochs: int = 200
    learning_rate: float = 1e-3
    top_k_neighbors: int = 20


@dataclass
class SemanticFilterConfig:
    rag_top_k: int = 5
    pool_base: int = 8
    pool_expansion: int = 8
    mmr_lambda: float = 0.65
    rrf_kappa: int = 60
    support_threshold: float = 0.55
    entropy_threshold: float = 0.45
    tau_plausible: float = 0.5
    tau_temporal: float = 0.35
    tau_mechanistic: float = 0.35
    temporal_markers: List[str] = field(default_factory=lambda: [
        "before",
        "after",
        "led to",
        "resulted",
        "subsequently",
        "following",
    ])
    mechanistic_markers: List[str] = field(default_factory=lambda: [
        "because",
        "by",
        "through",
        "using",
        "via",
        "enables",
        "causes",
    ])
    use_llm: bool = False
    gemini_model: str = "gemini-2.0-flash"
    gemini_temperature: float = 0.2
    gemini_api_key: str | None = None
    gemini_api_key_env: str = "API_KEY"
    gemini_base_url: str | None = None
    gemini_base_url_env: str | None = "BASE_URL"
    progress_every: int = 10


@dataclass
class PipelineConfig:
    pdf_embeddings_dir: Path = Path("pdf_embeddings")
    sentence_embedder: str = "all-MiniLM-L6-v2"
    structural: StructuralFilterConfig = field(default_factory=StructuralFilterConfig)
    semantic: SemanticFilterConfig = field(default_factory=SemanticFilterConfig)
    output_dir: Path = Path("outputs")
    max_pairs: int = 2000
    read_prep: Path | None = None
    write_prep: Path | None = None
