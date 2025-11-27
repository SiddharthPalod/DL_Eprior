"""Quick validation script for EM-Refinement Loop implementation.

This script validates that all components can be imported and basic functionality works.
Run from project root: python -m em_loop.validate_implementation
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from em_loop.src.config import EMRefinementConfig
    from em_loop.src.models import FusionModel, LPAModel, StudentSurrogate
    from em_loop.src.em_refinement import (
        EMRefinementLoop,
        PseudoLabel,
        AnswerKey,
        PseudoLabelDataset,
    )
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

try:
    # Test configuration
    config = EMRefinementConfig()
    print(f"✓ Configuration created: {config.num_rounds} rounds")
except Exception as e:
    print(f"✗ Configuration error: {e}")
    sys.exit(1)

try:
    # Test model creation
    fusion = FusionModel(
        input_dim=config.fusion_input_dim,
        hidden_dims=config.fusion_hidden_dims,
    )
    lpa = LPAModel(
        path_feature_dim=config.lpa_path_feature_dim,
        hidden_dim=config.lpa_hidden_dim,
    )
    surrogate = StudentSurrogate(
        input_dim=config.student_surrogate_input_dim,
        hidden_dims=config.student_surrogate_hidden_dims,
    )
    print("✓ All models created successfully")
except Exception as e:
    print(f"✗ Model creation error: {e}")
    sys.exit(1)

try:
    # Test data structures
    label = PseudoLabel(
        node_i=0,
        node_j=1,
        final_score=0.8,
        p_value=0.1,
        feature_vector=([0.0] * config.fusion_input_dim),
    )
    answer_key = AnswerKey(round=1, labels=[label])
    print(f"✓ Data structures work: {len(answer_key.labels)} label(s)")
except Exception as e:
    print(f"✗ Data structure error: {e}")
    sys.exit(1)

try:
    # Test dataset
    dataset = PseudoLabelDataset([label], augment=False, config=config)
    sample = dataset[0]
    print(f"✓ Dataset works: {len(sample)} keys in sample")
except Exception as e:
    print(f"✗ Dataset error: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All basic components validated successfully!")
print("=" * 60)
print("\nTo run full tests:")
print("  python -m em_loop.test_models")
print("  python -m em_loop.test_losses")
print("  python -m em_loop.test_em_loop")
print("  python -m em_loop.test_integration")
print("\nTo see usage example:")
print("  python -m em_loop.example_usage")

