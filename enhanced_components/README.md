# Enhanced Components for ACE Pipeline

This folder contains implementations of the missing components from the plan, organized by priority:

## Priority 1: RAG-HyDE Hypothesis Generation

**File**: `rag_hyde.py`

Replaces template-based hypothesis generation with LLM-generated hypothetical documents.

### Usage

```python
from enhanced_components.rag_hyde import RAGHyDE

# Initialize
hyde = RAGHyDE(
    api_key="your-api-key",  # or set GEMINI_API_KEY env var
    model_name="gemini-2.0-flash",
    k_hypothetical=4,
)

# Generate hypotheses
hypotheses = hyde.generate_hypotheses("XGBoost", "accuracy")
for hyp in hypotheses:
    print(hyp.text)
```

## Priority 2: Hypothesis Verification with Semantic Entropy

**File**: `hypothesis_verifier.py`

Implements full RAV (Retrieval-Augmented Verification) pipeline:
1. Evidence retrieval via RAG
2. LLM self-check verification
3. Semantic entropy estimation
4. Dual filtering (support + entropy thresholds)

### Usage

```python
from enhanced_components.hypothesis_verifier import HypothesisVerifier
from enhanced_components.rag_hyde import Hypothesis
from src.vector_store import SentenceVectorStore

# Initialize
verifier = HypothesisVerifier(
    vector_store=your_vector_store,
    api_key="your-api-key",
    tau_support=0.5,
    tau_entropy=0.45,
    r_samples=5,  # Number of samples for entropy estimation
)

# Verify hypotheses
hypotheses = [Hypothesis(text="XGBoost leads to accuracy")]
verified = verifier.verify_hypotheses(hypotheses)

for vh in verified:
    print(f"Verified: {vh.hypothesis.text}")
    print(f"  Support: {vh.support_score:.3f}")
    print(f"  Entropy: {vh.semantic_entropy:.3f}")
```

## Priority 3: CPC (Causal Plausibility Classifier)

### Part A: Dataset Construction

**File**: `cpc_dataset.py`

Builds training dataset with Teacher LLM labeling and adversarial hard negatives.

### Usage

```python
from enhanced_components.cpc_dataset import CPCDatasetBuilder, CPCDatasetEntry

# Initialize
builder = CPCDatasetBuilder(api_key="your-api-key")

# Label a single example
entry = builder.label_with_teacher_llm(
    node_i="XGBoost",
    node_j="accuracy",
    context="We evaluated XGBoost on ImageNet and achieved high accuracy...",
)

# Build balanced dataset
balanced = builder.build_balanced_dataset(
    positive_entries=positives,
    easy_negative_entries=easy_negatives,
    hard_negative_entries=hard_negatives,
)

# Save dataset
builder.save_dataset(balanced, Path("traindataset.jsonl"))
```

### Part B: Model Training and Calibration

**File**: `cpc_model.py`

Trains DeBERTa-v3 cross-encoder with three heads and isotonic regression calibration.

### Usage

```python
from enhanced_components.cpc_model import CPCTrainer, CPCTrainingConfig
from enhanced_components.cpc_dataset import CPCDatasetBuilder
from pathlib import Path

# Load dataset
entries = CPCDatasetBuilder.load_dataset(Path("traindataset.jsonl"))

# Configure training
config = CPCTrainingConfig(
    batch_size=16,
    learning_rate=2e-5,
    num_epochs=3,
)

# Train
trainer = CPCTrainer(config)
history = trainer.train(entries)

# Fit calibrators (on validation set)
trainer.fit_calibrators(validation_entries)

# Save model
trainer.save(
    model_path=Path("CPCModel.bin"),
    calibrators_path=Path("Calibrators.pkl"),
)

# Use for inference
probs = trainer.predict(
    context="XGBoost achieves high accuracy...",
    node_i="XGBoost",
    node_j="accuracy",
    calibrated=True,
)
print(f"Plausible: {probs['plausible']:.3f}")
print(f"Temporal: {probs['temporal']:.3f}")
print(f"Mechanistic: {probs['mechanistic']:.3f}")
```

## Integration with Existing Pipeline

To integrate these components into the existing `src/semantic_filter.py`:

1. Replace `_generate_hypotheses` to use `RAGHyDE`
2. Add hypothesis verification step before RRF
3. Replace heuristic scoring with CPC predictions

See `example_integration.py` for a complete example.

## Dependencies

All components require:
- `google-generativeai` (for LLM calls)
- `torch` and `transformers` (for CPC model)
- `scikit-learn` (for isotonic regression)
- Existing dependencies from `src/` (vector_store, etc.)

## Configuration

Key parameters:
- **RAG-HyDE**: `k_hypothetical` (number of hypotheses, default 4)
- **Verifier**: `tau_support` (0.5), `tau_entropy` (0.45), `r_samples` (5)
- **CPC Training**: `batch_size` (16), `learning_rate` (2e-5), `num_epochs` (3)

## Notes

- All components include fallback mechanisms if LLM calls fail
- Semantic entropy uses temperature-scaled sampling (0.6-0.8 recommended)
- CPC calibration requires a hold-out validation set (10% recommended)
- Model training supports GPU acceleration if available

