from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TextIO

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


class JsonChunkWriter:
    """Write semantic candidates to chunked JSONL files for crash-safe streaming."""

    def __init__(
        self,
        output_dir: Path,
        base_name: str = "E_prior_part",
        chunk_size: int = 500,
    ) -> None:
        self.output_dir = output_dir
        self.base_name = base_name
        self.chunk_size = max(1, chunk_size)
        self.file_index = 0
        self.current_count = 0
        self.current_file: Optional[TextIO] = None
        self.current_path: Optional[Path] = None
        self._open_new_file()

    def _open_new_file(self) -> None:
        if self.current_file:
            self.current_file.close()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file_index += 1
        filename = f"{self.base_name}_{self.file_index:03d}.jsonl"
        self.current_path = self.output_dir / filename
        self.current_file = self.current_path.open("w", encoding="utf-8")
        self.current_count = 0
        _log(f"streaming semantic pairs to {self.current_path}")

    def append(self, payload: Dict[str, Any]) -> None:
        if not self.current_file:
            self._open_new_file()
        json.dump(payload, self.current_file, ensure_ascii=False)
        self.current_file.write("\n")
        self.current_file.flush()
        self.current_count += 1
        if self.current_count >= self.chunk_size:
            self._open_new_file()

    def close(self) -> None:
        if self.current_file:
            self.current_file.close()
            self.current_file = None
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
            output_dir=config.output_dir,
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
    
    # Set up API key for RAG-HyDE and LLM verifier
    if config.semantic.use_rag_hyde or config.semantic.use_llm:
        if not config.semantic.gemini_api_key:
            # Try to get from environment
            api_key = os.getenv(config.semantic.gemini_api_key_env)
            if api_key:
                config.semantic.gemini_api_key = api_key
                _log(f"Loaded Gemini API key from environment variable {config.semantic.gemini_api_key_env}")
            else:
                _log(f"Warning: No API key found. RAG-HyDE and LLM verification will be disabled.")
                config.semantic.use_rag_hyde = False
                config.semantic.use_llm = False
    
    llm_verifier = None
    if config.semantic.use_llm:
        _log("initializing Gemini verifier")
        # Set max API calls to prevent quota issues
        max_calls = config.semantic.max_api_calls
        llm_verifier = GeminiVerifier(
            api_key=config.semantic.gemini_api_key,
            model_name=config.semantic.gemini_model,
            temperature=config.semantic.gemini_temperature,
            api_key_env=config.semantic.gemini_api_key_env,
            base_url=config.semantic.gemini_base_url,
            base_url_env=config.semantic.gemini_base_url_env,
            max_api_calls=max_calls,
        )
        _log(f"Gemini verifier initialized with max {max_calls} API calls")
        _log("testing Gemini with a simple example...")
        test_result = llm_verifier.score("test", "example", ["This is a test context."])
        if test_result is None:
            _log("ERROR: Gemini test failed - API call returned None. Check your API key and network connection.")
            _log("Continuing with heuristic scores only (Gemini disabled for this run)")
            llm_verifier = None
        else:
            _log(f"Gemini test passed! Got scores: support={test_result.support:.2f}, temporal={test_result.temporal:.2f}, mechanistic={test_result.mechanistic:.2f}")
    
    if config.semantic.use_rag_hyde:
        _log("RAG-HyDE enabled: Using LLM-generated hypotheses instead of templates")
    else:
        _log("RAG-HyDE disabled: Using template-based hypotheses")
    
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

    stream_dir = config.output_dir / "stream"
    chunk_writer = JsonChunkWriter(
        stream_dir,
        base_name="E_prior_part",
        chunk_size=config.semantic.stream_chunk_size,
    )
    try:
        semantic_candidates = semantic_filter.run(
            structural_candidates,
            result_callback=lambda candidate: chunk_writer.append(
                _serialize_candidate(candidate)
            ),
        )
    finally:
        chunk_writer.close()
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
    """Load environment variables from a file.
    
    Supports both .env and env.txt formats (KEY=VALUE).
    """
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


def load_dotenv(root_dir: Path | None = None) -> None:
    """Load .env file from project root if it exists.
    
    Args:
        root_dir: Root directory to search for .env (defaults to current working directory)
    """
    if root_dir is None:
        root_dir = Path.cwd()
    env_file = root_dir / ".env"
    if env_file.exists():
        load_env_file(env_file)


def parse_args() -> argparse.Namespace:
    defaults = PipelineConfig()
    parser = argparse.ArgumentParser(description="Build the E_prior candidate set")
    parser.add_argument("--pdf-dir", type=Path, default=defaults.pdf_embeddings_dir, help="Folder that stores the Excel embeddings")
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir, help="Destination directory for outputs")
    parser.add_argument("--max-pairs", type=int, default=defaults.max_pairs, help="Maximum semantic pairs to emit")
    parser.add_argument("--env-file", type=Path, default=Path("env.txt"), help="Optional KEY=VALUE file to load before running")
    parser.add_argument("--use-llm", action="store_true", help="Enable Gemini-based semantic verification")
    parser.add_argument("--use-rag-hyde", action="store_true", help="Enable RAG-HyDE for LLM-generated hypotheses (default: False, uses templates)")
    parser.add_argument("--no-rag-hyde", dest="use_rag_hyde", action="store_false", help="Disable RAG-HyDE and use template-based hypotheses")
    parser.add_argument("--gae-epochs", type=int, default=None, help="Number of epochs for GAE training (default: 200, use lower for faster testing)")
    parser.add_argument("--stream-chunk-size", type=int, default=defaults.semantic.stream_chunk_size, help="Semantic pairs per streamed chunk file (JSONL) for crash-safe outputs")
    parser.add_argument("--writeprep", type=Path, default=None, help="Save precomputed state to model folder (e.g., outputs/model)")
    parser.add_argument("--readprep", type=Path, default=None, help="Load precomputed state from model folder (e.g., outputs/model)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Load .env file from project root first (if it exists)
    load_dotenv()
    
    # Then load the specified env file (if provided)
    if args.env_file:
        load_env_file(args.env_file)
    
    # Debug: check if API_KEY was loaded
    if args.use_llm or args.use_rag_hyde:
        api_key = os.getenv("API_KEY")
        base_url = os.getenv("BASE_URL")
        print(f"[debug] API_KEY loaded: {'Yes' if api_key else 'No'} (length: {len(api_key) if api_key else 0})")
        print(f"[debug] BASE_URL loaded: {base_url if base_url else 'No'}")
    config = PipelineConfig(
        pdf_embeddings_dir=args.pdf_dir,
        output_dir=args.output_dir,
        max_pairs=args.max_pairs,
    )
    # use_llm and use_rag_hyde default to False (no LLM calls)
    config.semantic.use_llm = args.use_llm
    config.semantic.use_rag_hyde = args.use_rag_hyde
    config.semantic.stream_chunk_size = args.stream_chunk_size
    if args.gae_epochs is not None:
        config.structural.epochs = args.gae_epochs
        print(f"[pipeline] Using {args.gae_epochs} epochs for GAE training (instead of default 200)")
    config.read_prep = args.readprep
    config.write_prep = args.writeprep
    stats = run_pipeline(config)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
