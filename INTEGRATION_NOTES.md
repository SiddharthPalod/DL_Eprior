# RAG-HyDE Integration Notes

## Summary

RAG-HyDE has been successfully integrated into the main pipeline (`src/semantic_filter.py`). The pipeline now uses LLM-generated hypotheses instead of template-based ones when RAG-HyDE is enabled.

## Changes Made

### 1. Configuration (`src/config.py`)
- Added `use_rag_hyde: bool = True` (enabled by default)
- Added `rag_hyde_k: int = 4` (number of hypotheses to generate)
- Added `rag_hyde_temperature: float = 0.7` (temperature for generation)

### 2. Semantic Filter (`src/semantic_filter.py`)
- Added import for `RAGHyDE` from `enhanced_components`
- Modified `__init__` to initialize RAG-HyDE when enabled and API key is available
- Modified `_generate_hypotheses` to use RAG-HyDE first, with fallback to templates

### 3. Pipeline (`src/pipeline.py`)
- Added API key loading logic for RAG-HyDE
- Added logging to indicate when RAG-HyDE is enabled/disabled
- Added command-line argument `--use-rag-hyde` (default: True)
- Added command-line argument `--no-rag-hyde` to disable it

## Usage

### Basic Usage (RAG-HyDE enabled by default)
The pipeline automatically loads API keys from `.env` file in the project root:

```bash
# Create .env file in project root:
# API_KEY=your_gemini_api_key
# BASE_URL=https://generativelanguage.googleapis.com/v1beta

python -m src.pipeline --pdf-dir pdf_embeddings --output-dir outputs
```

### With API key in environment variable
```bash
export API_KEY=your_gemini_api_key
python -m src.pipeline --pdf-dir pdf_embeddings --output-dir outputs
```

### With custom env file
```bash
python -m src.pipeline --env-file env.txt --pdf-dir pdf_embeddings --output-dir outputs
```

### Disable RAG-HyDE (use templates)
```bash
python -m src.pipeline --no-rag-hyde --pdf-dir pdf_embeddings --output-dir outputs
```

### Enable both RAG-HyDE and LLM verification
```bash
python -m src.pipeline --use-llm --pdf-dir pdf_embeddings --output-dir outputs
```

## How It Works

1. **Initialization**: When `SemanticFilter` is created, it checks if RAG-HyDE is enabled and an API key is available
2. **Hypothesis Generation**: For each candidate pair:
   - If RAG-HyDE is available: Calls LLM to generate k hypothetical sentences
   - If RAG-HyDE fails or is disabled: Falls back to template-based hypotheses
3. **Rest of Pipeline**: Unchanged - uses hypotheses for RRF and context retrieval

## Fallback Behavior

- If API key is missing: Falls back to templates, logs warning
- If RAG-HyDE initialization fails: Falls back to templates, logs error
- If hypothesis generation fails for a specific pair: Falls back to templates for that pair only

## Next Steps

The generated `E_prior.json` can now be processed through CPC (Causal Plausibility Classifier) for further filtering. See `enhanced_components/example_cpc_training.py` for CPC usage.

## Notes

- RAG-HyDE requires a Gemini API key (set via `API_KEY` environment variable or config)
- RAG-HyDE is enabled by default but gracefully falls back to templates if unavailable
- The integration is backward compatible - existing code will work with templates if RAG-HyDE is disabled

