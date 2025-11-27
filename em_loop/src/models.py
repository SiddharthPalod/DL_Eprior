"""Model architectures for EM-Refinement Loop.

This module implements:
- Fusion Model: Maps feature vectors to calibrated scores
- LPA (Learned Path Aggregation) Model: Attention-based path aggregator
- Student Surrogate: Distills LLM scores to a lightweight model
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class FusionModel(nn.Module):
    """Fusion model that maps feature vectors to scores in (0,1).
    
    Architecture: Multi-layer perceptron with sigmoid output.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2,
    ) -> None:
        """Initialize fusion model.
        
        Args:
            input_dim: Dimension of input feature vectors
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer with sigmoid to ensure (0,1) range
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, v_ij: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            v_ij: Feature vectors [batch_size, input_dim]
        
        Returns:
            Scores in (0,1) [batch_size, 1]
        """
        return self.network(v_ij).squeeze(-1)


class PathEncoder(nn.Module):
    """Encodes individual paths for LPA model."""
    
    def __init__(
        self,
        path_feature_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        """Initialize path encoder.
        
        Args:
            path_feature_dim: Dimension of path features
            hidden_dim: Hidden dimension for path encoding
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(path_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, path_features: torch.Tensor) -> torch.Tensor:
        """Encode paths.
        
        Args:
            path_features: [batch_size, num_paths, path_feature_dim]
        
        Returns:
            Encoded paths [batch_size, num_paths, hidden_dim]
        """
        return self.encoder(path_features)


class AttentionAggregator(nn.Module):
    """Attention-based aggregator for paths."""
    
    def __init__(
        self,
        hidden_dim: int,
    ) -> None:
        """Initialize attention aggregator.
        
        Args:
            hidden_dim: Hidden dimension of path encodings
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
        )
        self.output_proj = nn.Linear(hidden_dim, 1)
    
    def forward(self, encoded_paths: torch.Tensor) -> torch.Tensor:
        """Aggregate paths with attention.
        
        Args:
            encoded_paths: [batch_size, num_paths, hidden_dim]
        
        Returns:
            Aggregated scores [batch_size]
        """
        # Self-attention
        attn_out, _ = self.attention(
            encoded_paths, encoded_paths, encoded_paths
        )
        
        # Global average pooling
        pooled = attn_out.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Output projection with sigmoid
        score = torch.sigmoid(self.output_proj(pooled)).squeeze(-1)
        
        return score


class LPAModel(nn.Module):
    """Learned Path Aggregation (LPA) model.
    
    Consists of a path encoder and an attention-based aggregator.
    Maps sets of paths to calibrated indirect scores.
    """
    
    def __init__(
        self,
        path_feature_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        """Initialize LPA model.
        
        Args:
            path_feature_dim: Dimension of path features
            hidden_dim: Hidden dimension for path encoding
        """
        super().__init__()
        self.path_encoder = PathEncoder(path_feature_dim, hidden_dim)
        self.aggregator = AttentionAggregator(hidden_dim)
    
    def forward(
        self,
        path_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            path_features: [batch_size, num_paths, path_feature_dim]
                If num_paths varies, use padding
        
        Returns:
            Indirect scores in (0,1) [batch_size]
        """
        # Encode paths
        encoded = self.path_encoder(path_features)  # [batch_size, num_paths, hidden_dim]
        
        # Aggregate with attention
        score = self.aggregator(encoded)  # [batch_size]
        
        return score


class StudentSurrogate(nn.Module):
    """Student surrogate model for LLM distillation.
    
    Lightweight model that approximates LLM conditional scores.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.1,
    ) -> None:
        """Initialize student surrogate.
        
        Args:
            input_dim: Dimension of input feature vectors
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer (no sigmoid - can be negative)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, v_ij: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            v_ij: Feature vectors [batch_size, input_dim]
        
        Returns:
            Predicted LLM scores [batch_size]
        """
        return self.network(v_ij).squeeze(-1)


def copy_model_weights(source: nn.Module, target: nn.Module) -> None:
    """Copy weights from source model to target model.
    
    Args:
        source: Source model
        target: Target model
    """
    target.load_state_dict(source.state_dict())


def update_teacher_ema(
    teacher: nn.Module,
    student: nn.Module,
    alpha: float = 0.999,
) -> None:
    """Update teacher parameters via exponential moving average.
    
    Args:
        teacher: Teacher model (updated in-place)
        student: Student model
        alpha: EMA momentum coefficient (typically close to 1)
    """
    with torch.no_grad():
        for teacher_param, student_param in zip(
            teacher.parameters(), student.parameters()
        ):
            teacher_param.data.mul_(alpha).add_(
                student_param.data, alpha=1 - alpha
            )

