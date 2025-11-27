"""Test models for EM-Refinement Loop."""

import numpy as np
import torch
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from em_loop.src.models import FusionModel, LPAModel, StudentSurrogate, update_teacher_ema
from em_loop.src.config import EMRefinementConfig


def test_fusion_model():
    """Test FusionModel forward pass."""
    print("Testing FusionModel...")
    
    config = EMRefinementConfig()
    model = FusionModel(
        input_dim=config.fusion_input_dim,
        hidden_dims=config.fusion_hidden_dims,
        dropout=config.fusion_dropout,
    )
    
    batch_size = 16
    v_ij = torch.randn(batch_size, config.fusion_input_dim)
    
    output = model(v_ij)
    
    assert output.shape == (batch_size,), f"Expected shape ({batch_size},), got {output.shape}"
    assert torch.all(output >= 0) and torch.all(output <= 1), "Output should be in [0,1]"
    
    print(f"  ✓ FusionModel output shape: {output.shape}")
    print(f"  ✓ Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    print("  ✓ FusionModel test passed!\n")


def test_lpa_model():
    """Test LPAModel forward pass."""
    print("Testing LPAModel...")
    
    config = EMRefinementConfig()
    model = LPAModel(
        path_feature_dim=config.lpa_path_feature_dim,
        hidden_dim=config.lpa_hidden_dim,
    )
    
    batch_size = 8
    num_paths = 5
    path_features = torch.randn(batch_size, num_paths, config.lpa_path_feature_dim)
    
    output = model(path_features)
    
    assert output.shape == (batch_size,), f"Expected shape ({batch_size},), got {output.shape}"
    assert torch.all(output >= 0) and torch.all(output <= 1), "Output should be in [0,1]"
    
    print(f"  ✓ LPAModel output shape: {output.shape}")
    print(f"  ✓ Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    print("  ✓ LPAModel test passed!\n")


def test_student_surrogate():
    """Test StudentSurrogate forward pass."""
    print("Testing StudentSurrogate...")
    
    config = EMRefinementConfig()
    model = StudentSurrogate(
        input_dim=config.student_surrogate_input_dim,
        hidden_dims=config.student_surrogate_hidden_dims,
        dropout=config.student_surrogate_dropout,
    )
    
    batch_size = 16
    v_ij = torch.randn(batch_size, config.student_surrogate_input_dim)
    
    output = model(v_ij)
    
    assert output.shape == (batch_size,), f"Expected shape ({batch_size},), got {output.shape}"
    
    print(f"  ✓ StudentSurrogate output shape: {output.shape}")
    print(f"  ✓ Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    print("  ✓ StudentSurrogate test passed!\n")


def test_teacher_ema_update():
    """Test EMA update for teacher models."""
    print("Testing Teacher EMA Update...")
    
    config = EMRefinementConfig()
    
    student = FusionModel(
        input_dim=config.fusion_input_dim,
        hidden_dims=[64, 32],
    )
    teacher = FusionModel(
        input_dim=config.fusion_input_dim,
        hidden_dims=[64, 32],
    )
    
    # Initialize teacher with student weights
    teacher.load_state_dict(student.state_dict())
    
    # Get initial teacher params
    initial_params = [p.clone() for p in teacher.parameters()]
    
    # Modify student (simulate training step)
    with torch.no_grad():
        for param in student.parameters():
            param.add_(torch.randn_like(param) * 0.1)
    
    # Update teacher via EMA
    alpha = 0.9
    update_teacher_ema(teacher, student, alpha=alpha)
    
    # Check that teacher changed but not as much as student
    for initial, teacher_param, student_param in zip(
        initial_params, teacher.parameters(), student.parameters()
    ):
        teacher_diff = (teacher_param - initial).abs().mean()
        student_diff = (student_param - initial).abs().mean()
        
        # Teacher should change less than student (due to EMA)
        assert teacher_diff < student_diff, "EMA should smooth updates"
    
    print(f"  ✓ EMA update applied successfully")
    print(f"  ✓ Teacher updates are smoother than student updates")
    print("  ✓ Teacher EMA update test passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing EM-Refinement Loop Models")
    print("=" * 60 + "\n")
    
    test_fusion_model()
    test_lpa_model()
    test_student_surrogate()
    test_teacher_ema_update()
    
    print("=" * 60)
    print("All model tests passed! ✓")
    print("=" * 60)

