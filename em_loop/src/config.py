"""Configuration for EM-Refinement Loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class EMRefinementConfig:
    """Configuration for EM-Refinement Loop."""
    
    # Model architecture
    fusion_input_dim: int = 128  # Dimension of feature vectors v_ij
    fusion_hidden_dims: List[int] = None  # Will default to [256, 128, 64]
    fusion_dropout: float = 0.2
    
    lpa_path_feature_dim: int = 64  # Dimension of path features
    lpa_hidden_dim: int = 64
    
    student_surrogate_input_dim: int = 128
    student_surrogate_hidden_dims: List[int] = None  # Will default to [128, 64]
    student_surrogate_dropout: float = 0.1
    
    # Training hyperparameters
    num_rounds: int = 10  # Maximum number of EM rounds
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs_per_round: int = 5  # Epochs per M-step
    
    # EMA parameters
    teacher_ema_alpha: float = 0.999  # EMA momentum (typically close to 1)
    
    # Loss weights
    lambda_consistency: float = 0.1  # Weight for consistency regularization
    
    # Huber loss parameter
    huber_delta: float = 1.0  # For regression loss in LPA pre-training
    
    # Stopping criteria
    jaccard_threshold: float = 0.99  # Stop if Jaccard similarity exceeds this
    confidence_threshold: float = 0.7  # Threshold for high-confidence links
    validation_plateau_rounds: int = 3  # Stop if no improvement for N rounds
    
    # Data augmentation for consistency
    augmentation_dropout_rate: float = 0.1  # Feature dropout rate
    augmentation_noise_std: float = 0.01  # Noise standard deviation
    
    # Paths
    output_dir: Path = Path("em_loop_outputs")
    model_save_dir: Path = Path("em_loop_outputs/models")
    
    # Device
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    
    def __post_init__(self):
        """Set default values for lists."""
        if self.fusion_hidden_dims is None:
            self.fusion_hidden_dims = [256, 128, 64]
        if self.student_surrogate_hidden_dims is None:
            self.student_surrogate_hidden_dims = [128, 64]
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

