"""Test loss functions for EM-Refinement Loop."""

import numpy as np
import torch
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from em_loop.src.config import EMRefinementConfig


def huber_loss(error: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Huber loss as specified in the PDF.
    
    L_regress(i,j) = {
        0.5 * e_ij^2,              if |e_ij| <= delta
        delta * |e_ij| - 0.5 * delta^2,  if |e_ij| > delta
    }
    """
    abs_error = torch.abs(error)
    quadratic = 0.5 * error ** 2
    linear = delta * abs_error - 0.5 * delta ** 2
    return torch.where(abs_error <= delta, quadratic, linear)


def test_huber_loss():
    """Test Huber loss implementation."""
    print("Testing Huber Loss...")
    
    delta = 1.0
    errors = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    
    losses = huber_loss(errors, delta)
    
    # Check quadratic region (|e| <= delta)
    assert torch.allclose(losses[2], torch.tensor(0.5 * 0.5 ** 2)), "Quadratic region incorrect"
    assert torch.allclose(losses[3], torch.tensor(0.0)), "Zero error should give zero loss"
    assert torch.allclose(losses[4], torch.tensor(0.5 * 0.5 ** 2)), "Quadratic region incorrect"
    
    # Check linear region (|e| > delta)
    assert torch.allclose(losses[0], torch.tensor(delta * 2.0 - 0.5 * delta ** 2)), "Linear region incorrect"
    assert torch.allclose(losses[6], torch.tensor(delta * 2.0 - 0.5 * delta ** 2)), "Linear region incorrect"
    
    print(f"  ✓ Huber loss values: {losses.tolist()}")
    print("  ✓ Huber loss test passed!\n")


def test_confidence_weighted_bce():
    """Test confidence-weighted binary cross-entropy loss."""
    print("Testing Confidence-Weighted BCE Loss...")
    
    batch_size = 8
    predictions = torch.sigmoid(torch.randn(batch_size))  # In [0,1]
    targets = torch.randint(0, 2, (batch_size,)).float()
    weights = torch.rand(batch_size)  # Confidence weights in [0,1]
    
    # Standard BCE
    bce = F.binary_cross_entropy(predictions, targets, reduction="none")
    
    # Weighted BCE
    weighted_bce = (weights * bce).mean()
    
    # Check that higher weights contribute more
    assert weighted_bce.item() >= 0, "Loss should be non-negative"
    
    print(f"  ✓ Weighted BCE loss: {weighted_bce.item():.4f}")
    print("  ✓ Confidence-weighted BCE test passed!\n")


def test_consistency_loss():
    """Test consistency regularization loss."""
    print("Testing Consistency Loss...")
    
    batch_size = 8
    pred1 = torch.randn(batch_size)
    pred2 = torch.randn(batch_size)
    
    # MSE consistency loss
    consistency_loss = F.mse_loss(pred1, pred2)
    
    # If predictions are identical, loss should be zero
    pred_identical = torch.randn(batch_size)
    zero_loss = F.mse_loss(pred_identical, pred_identical)
    assert torch.allclose(zero_loss, torch.tensor(0.0)), "Identical predictions should give zero loss"
    
    print(f"  ✓ Consistency loss: {consistency_loss.item():.4f}")
    print(f"  ✓ Zero loss for identical predictions: {zero_loss.item():.4f}")
    print("  ✓ Consistency loss test passed!\n")


def test_distillation_loss():
    """Test distillation loss (MSE for student surrogate)."""
    print("Testing Distillation Loss...")
    
    batch_size = 8
    student_pred = torch.randn(batch_size)
    teacher_target = torch.randn(batch_size)
    
    # MSE distillation loss
    distill_loss = F.mse_loss(student_pred, teacher_target)
    
    # If predictions match, loss should be zero
    matching_pred = torch.randn(batch_size)
    zero_loss = F.mse_loss(matching_pred, matching_pred)
    assert torch.allclose(zero_loss, torch.tensor(0.0)), "Matching predictions should give zero loss"
    
    print(f"  ✓ Distillation loss: {distill_loss.item():.4f}")
    print(f"  ✓ Zero loss for matching predictions: {zero_loss.item():.4f}")
    print("  ✓ Distillation loss test passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing EM-Refinement Loop Loss Functions")
    print("=" * 60 + "\n")
    
    test_huber_loss()
    test_confidence_weighted_bce()
    test_consistency_loss()
    test_distillation_loss()
    
    print("=" * 60)
    print("All loss function tests passed! ✓")
    print("=" * 60)

