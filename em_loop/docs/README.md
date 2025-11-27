# EM-Refinement Loop Implementation

This module implements the Expectation-Maximization (EM) refinement loop with Mean-Teacher architecture as described in the PDF specification (DLF (4)-50-53.pdf).

## Overview

The EM-Refinement Loop refines pre-trained models on real, unlabeled document corpus by using their own predictions as a self-supervised signal. It consists of:

1. **E-Step (Expectation)**: Generate pseudo-labels using teacher models
2. **M-Step (Maximization)**: Update student models with pseudo-labels and consistency regularization
3. **Teacher Update**: Update teacher models via exponential moving average (EMA)
4. **Stopping Criteria**: Label stability (Jaccard similarity) and validation plateau

## Components

### Models

- **FusionModel**: Maps feature vectors `v_ij` to calibrated scores in (0,1)
- **LPAModel**: Learned Path Aggregation model with attention-based aggregator
- **StudentSurrogate**: Lightweight model that distills LLM scores (round 1 only)

### Loss Functions

- **Confidence-Weighted BCE**: For fusion and LPA model updates
- **Huber Loss**: For LPA pre-training regression (not used in refinement)
- **Consistency Loss**: MSE between augmented predictions
- **Distillation Loss**: MSE for student surrogate training

### Data Structures

- **PseudoLabel**: Contains node pair, scores, p-values, and feature vectors
- **AnswerKey**: Sparse answer key `C_prior^(r)` for round `r`
- **PseudoLabelDataset**: PyTorch dataset for training

## Usage

### Basic Example

```python
from em_loop import (
    EMRefinementConfig,
    EMRefinementLoop,
    AnswerKey,
    PseudoLabel,
    FusionModel,
    LPAModel,
)

# Configure
config = EMRefinementConfig(
    num_rounds=10,
    num_epochs_per_round=5,
    batch_size=32,
    learning_rate=1e-3,
)

# Define pipeline callback (must implement CoCaD pipeline)
def pipeline_callback(round_num, fusion_teacher, lpa_teacher, student_surrogate):
    # Run CoCaD pipeline with teacher models
    # Return AnswerKey with pseudo-labels
    labels = []
    # ... generate pseudo-labels ...
    return AnswerKey(round=round_num, labels=labels)

# Initialize and run
em_loop = EMRefinementLoop(config, pipeline_callback=pipeline_callback)
results = em_loop.run()

# Access final models
fusion_teacher = results["fusion_teacher"]
lpa_teacher = results["lpa_teacher"]
student_surrogate = results["student_surrogate"]
```

### Pipeline Callback Interface

The pipeline callback should:

1. Run the complete CoCaD pipeline (Phase 3: Steps 3.1-3.3) using teacher models
2. For each candidate pair `(i, j)`, compute:
   - `finalscore_ij^(r)`: Final score in [0,1]
   - `p(i, j)`: Empirical p-value
   - `v_ij`: Feature vector
   - Path features (for LPA model)
   - LLM score (round 1 only, for distillation)
3. Return an `AnswerKey` object with all pseudo-labels

### Configuration

Key configuration parameters:

- `num_rounds`: Maximum number of EM rounds (default: 10)
- `num_epochs_per_round`: Epochs per M-step (default: 5)
- `batch_size`: Training batch size (default: 32)
- `learning_rate`: Learning rate (default: 1e-3)
- `teacher_ema_alpha`: EMA momentum (default: 0.999)
- `lambda_consistency`: Consistency regularization weight (default: 0.1)
- `jaccard_threshold`: Stop if Jaccard similarity exceeds this (default: 0.99)
- `confidence_threshold`: Threshold for high-confidence links (default: 0.7)

## Testing

Run individual test files:

```bash
# Test models
python -m em_loop.tests.test_models

# Test loss functions
python -m em_loop.tests.test_losses

# Test EM loop
python -m em_loop.tests.test_em_loop

# Integration test
python -m em_loop.tests.test_integration

# Run all tests
python -m em_loop.tests.run_tests

# Quick validation
python -m em_loop.tests.validate_implementation
```

See `docs/TESTING_GUIDE.txt` for detailed testing instructions.

## Implementation Details

### E-Step

1. Run CoCaD pipeline with teacher models `θ_T^(r-1)`
2. For each pair `(i, j)`, compute final score and statistics
3. In round 1, store LLM scores for distillation
4. Return sparse answer key `C_prior^(r)`

### M-Step

1. **Fusion Model Update**:
   - Confidence-weighted BCE loss: `L_fusion-real = Σ w_ij · BCE(f_fusion,S(v_ij), y_ij^soft)`
   - Consistency regularization: MSE between augmented predictions

2. **LPA Model Update**:
   - Binary classification: `L_LPA-real = Σ w_ij · BCE(I_learned,S(i,j), y_ij^bin)`

3. **Student Surrogate Distillation** (round 1 only):
   - MSE loss: `L_distill = Σ ||f_student(v_ij) - t_ij||²`

4. **Total Loss**: `L_M = L_pseudo-label + λ_consist · L_consistency`

### Teacher Update

Exponential moving average:
```
θ_T^(r) ← α · θ_T^(r-1) + (1-α) · θ_S^(r)
```
where `α = 0.999` (typically).

### Stopping Criteria

1. **Label Stability**: Jaccard similarity between high-confidence label sets
   ```
   J^(r) = |H^(r) ∩ H^(r-1)| / |H^(r) ∪ H^(r-1)|
   ```
   Stop if `J^(r) >= threshold` (default: 0.99)

2. **Validation Plateau**: Stop if validation loss doesn't improve for N rounds (default: 3)

## Project Structure

```
em_loop/
├── __init__.py              # Package exports
├── src/                     # Core source code
│   ├── __init__.py
│   ├── config.py           # Configuration dataclass
│   ├── models.py           # Model architectures
│   └── em_refinement.py    # Main EM loop implementation
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── test_models.py       # Model tests
│   ├── test_losses.py      # Loss function tests
│   ├── test_em_loop.py     # EM loop tests
│   ├── test_integration.py # Integration tests
│   ├── validate_implementation.py  # Quick validation
│   └── run_tests.py        # Test runner
├── examples/                # Example scripts
│   ├── __init__.py
│   └── example_usage.py   # Usage example
└── docs/                    # Documentation
    ├── README.md           # This file
    ├── PROJECT_STRUCTURE.md
    ├── IMPLEMENTATION_SUMMARY.txt
    └── TESTING_GUIDE.txt
```

See `docs/PROJECT_STRUCTURE.md` for detailed structure documentation.

## References

See `DLF (4)-50-53.pdf` for the complete specification of the EM-Refinement Loop.

