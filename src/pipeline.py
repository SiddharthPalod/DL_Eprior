from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import PipelineConfig
from .data_loader import load_chunks
from .embedder import SentenceEmbedder
from .graph_builder import build_cooccurrence_graph
from .llm_verifier import GeminiVerifier
from .node_extractor import NodeExtractor, NodeRecord, SentenceRecord
from .semantic_filter import SemanticCandidate, SemanticFilter
from .structural_filter import CandidatePair, build_structural_candidates, train_structural_filter
from .vector_store import SentenceVectorStore


def _log(step: str) -> None:
    print(f"[pipeline] {step}", flush=True)


def run_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    _log("initializing output directory")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    nodes: List[NodeRecord]
    sentences: List[SentenceRecord]
    structural_candidates: List[CandidatePair]

    embedder: Optional[SentenceEmbedder] = None

    if config.read_prep:
        _log(f"loading prepared artifacts from {config.read_prep}")
        (
            nodes,
            sentences,
            structural_candidates,
            embedder_override,
        ) = load_prepared_state(config.read_prep)
        if embedder_override and embedder_override != config.sentence_embedder:
            _log(f"overriding sentence embedder with cached value {embedder_override}")
            config.sentence_embedder = embedder_override
    else:
        _log("loading chunks from embeddings")
        chunks = load_chunks(config.pdf_embeddings_dir)
        extractor = NodeExtractor(min_frequency=1)
        _log("extracting nodes and sentences")
        nodes, sentences = extractor.extract(chunks)
        if not nodes:
            raise ValueError("No nodes were extracted from the document")

        _log("building co-occurrence graph")
        graph = build_cooccurrence_graph(nodes)

        _log("encoding node features")
        embedder = SentenceEmbedder(config.sentence_embedder)
        node_features = embedder.encode([node.label for node in nodes])

        _log("training structural filter (GAE)")
        structural_embeddings = train_structural_filter(
            adjacency=graph.adjacency,
            node_features=node_features,
            config=config.structural,
        )
        _log("building structural candidate pairs")
        structural_candidates = build_structural_candidates(structural_embeddings, config.structural)

        if config.write_prep:
            _log(f"saving prepared artifacts to {config.write_prep}")
            save_prepared_state(
                config.write_prep,
                nodes,
                sentences,
                structural_candidates,
                config.sentence_embedder,
            )

    _log("building sentence vector store")
    if embedder is None:
        embedder = SentenceEmbedder(config.sentence_embedder)
    store = SentenceVectorStore(sentences, embedder)
    llm_verifier = None
    if config.semantic.use_llm:
        _log("initializing Gemini verifier")
        llm_verifier = GeminiVerifier(
            api_key=config.semantic.gemini_api_key,
            model_name=config.semantic.gemini_model,
            temperature=config.semantic.gemini_temperature,
            api_key_env=config.semantic.gemini_api_key_env,
            base_url=config.semantic.gemini_base_url,
            base_url_env=config.semantic.gemini_base_url_env,
        )
        _log("testing Gemini with a simple example...")
        test_result = llm_verifier.score("test", "example", ["This is a test context."])
        if test_result is None:
            _log("ERROR: Gemini test failed - API call returned None. Check your API key and network connection.")
            _log("Continuing with heuristic scores only (Gemini disabled for this run)")
            llm_verifier = None
        else:
            _log(f"Gemini test passed! Got scores: support={test_result.support:.2f}, temporal={test_result.temporal:.2f}, mechanistic={test_result.mechanistic:.2f}")
    _log("running semantic filter")
    semantic_filter = SemanticFilter(
        store,
        nodes,
        config.semantic,
        llm_verifier=llm_verifier,
        progress_callback=lambda done, total, kept: _log(
            f"semantic filter progress {done}/{total} processed, kept {kept}"
        ),
    )
    semantic_candidates = semantic_filter.run(structural_candidates)
    semantic_candidates = semantic_candidates[: config.max_pairs]

    _log("writing outputs/E_prior.json")
    e_prior_path = config.output_dir / "E_prior.json"
    with e_prior_path.open("w", encoding="utf-8") as fp:
        json.dump([_serialize_candidate(candidate) for candidate in semantic_candidates], fp, indent=2)

    return {
        "nodes": len(nodes),
        "structural_candidates": len(structural_candidates),
        "semantic_candidates": len(semantic_candidates),
        "output": str(e_prior_path.resolve()),
    }


def _serialize_candidate(candidate: SemanticCandidate) -> Dict[str, Any]:
    payload = {
        "node_i": candidate.node_i,
        "node_j": candidate.node_j,
        "node_i_label": candidate.node_i_label,
        "node_j_label": candidate.node_j_label,
        "support": candidate.support,
        "temporal": candidate.temporal,
        "mechanistic": candidate.mechanistic,
        "contexts": candidate.contexts,
    }
    if candidate.rationale:
        payload["rationale"] = candidate.rationale
    return payload


def save_prepared_state(
    path: Path,
    nodes: Sequence[NodeRecord],
    sentences: Sequence[SentenceRecord],
    candidates: Sequence[CandidatePair],
    embedder_name: str,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    data = {
        "sentence_embedder": embedder_name,
        "nodes": [{"node_id": node.node_id, "label": node.label} for node in nodes],
        "sentences": [sentence.text for sentence in sentences],
        "candidates": [
            {
                "node_i": candidate.node_i,
                "node_j": candidate.node_j,
                "structural_score": candidate.structural_score,
            }
            for candidate in candidates
        ],
    }
    (path / "prep.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_prepared_state(path: Path) -> Tuple[
    List[NodeRecord],
    List[SentenceRecord],
    List[CandidatePair],
    Optional[str],
]:
    prep_file = path / "prep.json" if path.is_dir() else path
    payload = json.loads(prep_file.read_text(encoding="utf-8"))
    nodes = [
        NodeRecord(node_id=item["node_id"], label=item["label"], occurrences=[])
        for item in payload.get("nodes", [])
    ]
    sentences = [
        SentenceRecord(text=text, chunk_id=0, sentence_id=index)
        for index, text in enumerate(payload.get("sentences", []))
    ]
    candidates = [
        CandidatePair(
            node_i=item["node_i"],
            node_j=item["node_j"],
            structural_score=float(item.get("structural_score", 0.0)),
        )
        for item in payload.get("candidates", [])
    ]
    embedder_name = payload.get("sentence_embedder")
    return nodes, sentences, candidates, embedder_name


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def parse_args() -> argparse.Namespace:
    defaults = PipelineConfig()
    parser = argparse.ArgumentParser(description="Build the E_prior candidate set")
    parser.add_argument("--pdf-dir", type=Path, default=defaults.pdf_embeddings_dir, help="Folder that stores the Excel embeddings")
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir, help="Destination directory for outputs")
    parser.add_argument("--max-pairs", type=int, default=defaults.max_pairs, help="Maximum semantic pairs to emit")
    parser.add_argument("--env-file", type=Path, default=Path("env.txt"), help="Optional KEY=VALUE file to load before running")
    parser.add_argument("--use-llm", action="store_true", help="Enable Gemini-based semantic verification")
    parser.add_argument("--writeprep", type=Path, default=None, help="Save precomputed state to model folder (e.g., outputs/model)")
    parser.add_argument("--readprep", type=Path, default=None, help="Load precomputed state from model folder (e.g., outputs/model)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    # Debug: check if API_KEY was loaded
    if args.use_llm:
        api_key = os.getenv("API_KEY")
        base_url = os.getenv("BASE_URL")
        print(f"[debug] API_KEY loaded: {'Yes' if api_key else 'No'} (length: {len(api_key) if api_key else 0})")
        print(f"[debug] BASE_URL loaded: {base_url if base_url else 'No'}")
    config = PipelineConfig(
        pdf_embeddings_dir=args.pdf_dir,
        output_dir=args.output_dir,
        max_pairs=args.max_pairs,
    )
    config.semantic.use_llm = args.use_llm
    config.read_prep = args.readprep
    config.write_prep = args.writeprep
    stats = run_pipeline(config)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
