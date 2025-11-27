# EM-Refinement Loop Project Structure

This document describes the organization of the EM-Refinement Loop project.

## Directory Structure

```
em_loop/
├── __init__.py              # Main package exports
├── src/                      # Core source code
│   ├── __init__.py          # Source package exports
│   ├── config.py            # Configuration dataclass
│   ├── models.py            # Model architectures
│   └── em_refinement.py     # Main EM loop implementation
├── tests/                    # Test suite
│   ├── __init__.py
│   ├── test_models.py        # Model architecture tests
│   ├── test_losses.py       # Loss function tests
│   ├── test_em_loop.py       # EM loop component tests
│   ├── test_integration.py  # Integration tests
│   ├── validate_implementation.py  # Quick validation
│   └── run_tests.py         # Test runner
├── examples/                 # Example scripts
│   ├── __init__.py
│   └── example_usage.py     # Usage example
├── docs/                     # Documentation
│   ├── __init__.py
│   ├── README.md            # Main documentation
│   ├── PROJECT_STRUCTURE.md # This file
│   ├── IMPLEMENTATION_SUMMARY.txt
│   └── TESTING_GUIDE.txt
└── DLF (4)-50-53.pdf        # Specification PDF
```

## Module Organization

### Source Code (`src/`)

**config.py**
- `EMRefinementConfig`: Configuration dataclass with all hyperparameters

**models.py**
- `FusionModel`: MLP for mapping feature vectors to scores
- `LPAModel`: Path encoder + attention aggregator
- `StudentSurrogate`: Lightweight LLM surrogate
- Helper functions: `copy_model_weights`, `update_teacher_ema`

**em_refinement.py**
- `PseudoLabel`: Data structure for pseudo-labels
- `AnswerKey`: Sparse answer key container
- `PseudoLabelDataset`: PyTorch dataset with augmentation
- `EMRefinementLoop`: Main EM loop class

### Tests (`tests/`)

All test files follow the naming convention `test_*.py`:
- Unit tests for individual components
- Integration tests for full workflows
- Validation scripts for quick checks

### Examples (`examples/`)

Example scripts demonstrating:
- Basic usage of the EM loop
- Pipeline callback implementation
- Configuration and customization

### Documentation (`docs/`)

- **README.md**: Main documentation with usage guide
- **PROJECT_STRUCTURE.md**: This file
- **IMPLEMENTATION_SUMMARY.txt**: Implementation overview
- **TESTING_GUIDE.txt**: Testing instructions

## Import Structure

### From Package Root

```python
from em_loop import (
    EMRefinementConfig,
    FusionModel,
    LPAModel,
    StudentSurrogate,
    EMRefinementLoop,
    PseudoLabel,
    AnswerKey,
)
```

### From Source Directly

```python
from em_loop.src import (
    EMRefinementConfig,
    FusionModel,
    LPAModel,
    EMRefinementLoop,
    # ... etc
)
```

### Internal Imports (within src/)

Source files use relative imports:
```python
from .config import EMRefinementConfig
from .models import FusionModel, LPAModel
```

## Running Tests

From project root:
```bash
# Individual tests
python -m em_loop.tests.test_models
python -m em_loop.tests.test_losses
python -m em_loop.tests.test_em_loop
python -m em_loop.tests.test_integration

# All tests
python -m em_loop.tests.run_tests

# Quick validation
python -m em_loop.tests.validate_implementation
```

## Running Examples

```bash
python -m em_loop.examples.example_usage
```

## Benefits of This Structure

1. **Separation of Concerns**: Source, tests, examples, and docs are clearly separated
2. **Maintainability**: Easy to find and modify specific components
3. **Scalability**: Easy to add new modules, tests, or examples
4. **Professional**: Follows Python packaging best practices
5. **Clear Dependencies**: Import paths make dependencies explicit

## Adding New Components

### Adding a New Model

1. Add to `src/models.py` or create `src/new_model.py`
2. Export from `src/__init__.py`
3. Export from `em_loop/__init__.py`
4. Add tests to `tests/test_new_model.py`

### Adding a New Test

1. Create `tests/test_new_feature.py`
2. Use imports: `from em_loop.src import ...`
3. Add to `tests/run_tests.py` if needed

### Adding a New Example

1. Create `examples/new_example.py`
2. Use imports: `from em_loop.src import ...` or `from em_loop import ...`
3. Document in `docs/README.md`

