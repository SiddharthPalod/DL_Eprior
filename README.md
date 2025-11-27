# E_prior Pipeline

This repository implements the ACE portion of the CausGT-HS plan up to the construction of E_prior.
The workflow ingests pre-computed PDF embeddings, discovers entity nodes, expands structural
candidates with a graph auto-encoder, and applies a retrieval-based semantic filter to retain the
highest-confidence pairs.

## Environment

```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Optional: create `env.txt` and add Gemini credentials so the CLI can load them automatically:

```
# env.txt
API_KEY="your-gemini-key"
BASE_URL="https://generativelanguage.googleapis.com/v1beta"
```

## Running the pipeline

```
python -m src.pipeline --pdf-dir pdf_embeddings --max-pairs 200 --output-dir outputs
```

To enable Gemini verification:

```
python -m src.pipeline --use-llm --env-file env.txt --max-pairs 200
```

This produces `outputs/E_prior.json`, a JSON list of tuples with support/temporal/mechanistic
scores plus the RAG snippets (and optional Gemini rationale) that justified the decision.

When `--use-llm` is on, the CLI prints `[pipeline] ...` stage markers and periodic
`semantic filter progress X/Y processed` updates so you can track the Gemini calls.

### Streaming semantic outputs

During the semantic step every accepted pair is also written immediately to chunked
JSONL files in `outputs/stream/` (e.g., `E_prior_part_001.jsonl`). These files are
crash-safe (each line is standalone JSON), so even if the run stops early you still
have the processed pairs. The chunk size defaults to 500 pairs but can be tuned with:

```
python -m src.pipeline --stream-chunk-size 200 ...
```

Use these chunked files directly for external tools or merge them later; the final
`outputs/E_prior.json` is still emitted after the pipeline completes.

### Reusing precomputed artifacts

If the graph + GAE prep steps take too long, cache the intermediate artifacts once and reuse them:

```
# first run: build everything and store the prep results
python -m src.pipeline --writeprep outputs/model --max-pairs 200

# later: skip graph/GAE, reuse cached nodes/sentences/candidates
python -m src.pipeline --readprep outputs/model --use-llm --env-file env.txt --max-pairs 200
```

The cache stores node IDs/labels, sentence texts, structural scores, and the sentence-embedder name
in `outputs/model/prep.json`. You can keep multiple model folders if you work with several documents.

## Implementation notes

- src/node_extractor.py: spaCy-based node & sentence extractor with paragraph-level indexing.
- src/structural_filter.py: 2-layer GAE that reconstructs the co-occurrence graph and uses FAISS
  to expand multi-hop structural candidates.
- src/vector_store.py & src/semantic_filter.py: sentence-level dense retriever with heuristic
  RAG-MMR + RRF fusion plus an optional Gemini-backed CPC surrogate mirroring the plan's thresholds.
- src/pipeline.py: CLI orchestrator that ties everything together and serializes E_prior.

The semantic filter defaults to embedding-based heuristics but can call Gemini for calibrated support/
temporal/mechanistic scores when `--use-llm` is supplied (or when `SemanticFilterConfig.use_llm=True`).
