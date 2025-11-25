# Enhanced Components Implementation Summary

This document summarizes what has been implemented in the `enhanced_components/` folder.

## Overview

All missing components from `IMPLEMENTATION_STATUS.md` have been implemented according to the priorities specified:

1. ✅ **Priority 1**: RAG-HyDE hypothesis generation
2. ✅ **Priority 2**: Hypothesis verification with semantic entropy
3. ✅ **Priority 3**: CPC dataset construction and training pipeline

---

## Priority 1: RAG-HyDE Hypothesis Generation

**File**: `rag_hyde.py`

### What it does:
- Replaces template-based hypothesis generation with LLM-generated hypothetical documents
- Queries Gemini to generate k hypothetical sentences describing causal relationships
- Includes fallback to templates if LLM fails

### Key Features:
- Configurable number of hypotheses (`k_hypothetical`, default 4)
- Temperature control for generation (0.7 default)
- Error handling with graceful fallback
- JSON-structured output parsing

### Usage:
```python
from enhanced_components.rag_hyde import RAGHyDE

hyde = RAGHyDE(api_key="...", k_hypothetical=4)
hypotheses = hyde.generate_hypotheses("XGBoost", "accuracy")
```

---

## Priority 2: Hypothesis Verification with Semantic Entropy

**File**: `hypothesis_verifier.py`

### What it does:
Implements the full RAV (Retrieval-Augmented Verification) pipeline:
1. **Evidence Retrieval**: Retrieves top-k RAG snippets for each hypothesis
2. **LLM Self-Check**: Verifies if evidence supports the hypothesis
3. **Semantic Entropy Estimation**: Performs R independent stochastic forward passes
4. **Dual Filtering**: Keeps only hypotheses with `p_support > τ_support AND H_semantic < τ_entropy`

### Key Features:
- RAG-based evidence retrieval using existing vector store
- Support score calculation (p_support)
- Semantic entropy via multiple temperature-scaled samples
- Configurable thresholds (tau_support=0.5, tau_entropy=0.45)
- Configurable number of samples for entropy (r_samples=5)

### Mathematical Details:
- **Semantic Entropy**: `H_semantic = -Σ p_r log p_r` across R independent samples
- **Normalization**: Entropy normalized by log(2) to [0, 1] range
- **Temperature**: Moderate temperature (0.6-0.8) for entropy estimation

### Usage:
```python
from enhanced_components.hypothesis_verifier import HypothesisVerifier

verifier = HypothesisVerifier(
    vector_store=store,
    tau_support=0.5,
    tau_entropy=0.45,
    r_samples=5,
)
verified = verifier.verify_hypotheses(hypotheses)
```

---

## Priority 3: CPC (Causal Plausibility Classifier)

### Part A: Dataset Construction

**File**: `cpc_dataset.py`

### What it does:
- Labels tuples (node_i, node_j, context) with Teacher LLM
- Generates three binary labels: plausible, temporal, mechanistic
- Builds balanced dataset (50% positive, 25% easy negative, 25% hard negative)
- Supports JSONL and JSON formats

### Key Features:
- Teacher LLM labeling with structured JSON output
- Rationale generation for each example
- Dataset balancing and shuffling
- Save/load functionality

### Usage:
```python
from enhanced_components.cpc_dataset import CPCDatasetBuilder

builder = CPCDatasetBuilder(api_key="...")
entry = builder.label_with_teacher_llm("XGBoost", "accuracy", context)
balanced = builder.build_balanced_dataset(positives, easy_neg, hard_neg)
builder.save_dataset(balanced, Path("traindataset.jsonl"))
```

### Part B: Model Training and Calibration

**File**: `cpc_model.py`

### What it does:
- Trains DeBERTa-v3 cross-encoder with three task-specific heads
- Multi-task BCE loss (plausible + temporal + mechanistic)
- Isotonic regression calibration for each head
- Inference with calibrated probabilities

### Architecture:
- **Encoder**: DeBERTa-v3 base (microsoft/deberta-v3-base)
- **Input Format**: `[CLS]context[SEP]node_i[SEP]node_j[SEP]`
- **Three Heads**: Linear layers producing logits for each task
- **Loss**: `L_total = L_BCE(plausible) + L_BCE(temporal) + L_BCE(mechanistic)`

### Calibration:
- Uses hold-out validation set (10% recommended)
- Fits IsotonicRegression for each head
- Converts raw logits → raw probabilities → calibrated probabilities
- Preserves ordering (monotonic mapping)

### Usage:
```python
from enhanced_components.cpc_model import CPCTrainer, CPCTrainingConfig

config = CPCTrainingConfig(batch_size=16, learning_rate=2e-5, num_epochs=3)
trainer = CPCTrainer(config)
history = trainer.train(entries)
trainer.fit_calibrators(validation_entries)
trainer.save(Path("CPCModel.bin"), Path("Calibrators.pkl"))

# Inference
probs = trainer.predict(context, node_i, node_j, calibrated=True)
```

---

## Integration

### Example Integration Script

**File**: `example_integration.py`

Demonstrates how to integrate enhanced components into existing `src/semantic_filter.py`:

1. Replace `_generate_hypotheses` with `RAGHyDE.generate_hypotheses`
2. Add `HypothesisVerifier.verify_hypotheses` step
3. Use verified hypotheses for RRF (instead of all hypotheses)
4. Adaptive pooling based on verification confidence

### Example CPC Training

**File**: `example_cpc_training.py`

Complete example of CPC training pipeline:
1. Prepare unlabeled dataset
2. Label with Teacher LLM
3. Build balanced dataset
4. Train model
5. Fit calibrators
6. Save model and calibrators
7. Test inference

---

## Files Structure

```
enhanced_components/
├── __init__.py                    # Module initialization
├── README.md                      # Usage documentation
├── IMPLEMENTATION_SUMMARY.md      # This file
├── requirements.txt               # Additional dependencies
├── rag_hyde.py                    # Priority 1: RAG-HyDE
├── hypothesis_verifier.py         # Priority 2: Verification
├── cpc_dataset.py                 # Priority 3A: Dataset construction
├── cpc_model.py                  # Priority 3B: Model training
├── example_integration.py        # Integration example
└── example_cpc_training.py       # CPC training example
```

---

## Dependencies

### Additional Requirements:
- `transformers>=4.30.0` (for DeBERTa-v3)
- `scikit-learn>=1.3.0` (for IsotonicRegression)

### Existing Requirements (from parent):
- `google-generativeai>=0.8.3` (for LLM calls)
- `torch>=2.0.0` (for model training)
- `numpy>=1.26.0` (for numerical operations)

---

## Configuration Parameters

### RAG-HyDE:
- `k_hypothetical`: Number of hypotheses (default: 4)
- `temperature`: Generation temperature (default: 0.7)

### Hypothesis Verifier:
- `tau_support`: Support threshold (default: 0.5)
- `tau_entropy`: Entropy threshold (default: 0.45)
- `r_samples`: Number of samples for entropy (default: 5)
- `k_rag`: RAG snippets for verification (default: 3)

### CPC Training:
- `batch_size`: Training batch size (default: 16)
- `learning_rate`: Learning rate (default: 2e-5)
- `num_epochs`: Training epochs (default: 3)
- `validation_split`: Hold-out for calibration (default: 0.1)

---

## Next Steps

To fully integrate these components:

1. **Update `src/semantic_filter.py`**:
   - Import enhanced components
   - Replace `_generate_hypotheses` with RAG-HyDE
   - Add verification step before RRF

2. **Train CPC Model**:
   - Run dataset construction pipeline
   - Train model on balanced dataset
   - Save model and calibrators

3. **Replace Heuristic Scoring**:
   - Load trained CPC model
   - Use CPC predictions instead of marker-based scoring
   - Apply calibrated probabilities for filtering

4. **Update Pipeline Config**:
   - Add flags for enhanced components
   - Add CPC model path configuration
   - Add verification thresholds

---

## Status

✅ **All priorities implemented and ready for integration**

All components include:
- Error handling and fallbacks
- Comprehensive documentation
- Example usage scripts
- Integration guidance

The implementation follows the plan specifications from `plan.txt` and addresses all missing components identified in `IMPLEMENTATION_STATUS.md`.

