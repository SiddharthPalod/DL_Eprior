"""EM-Refinement Loop Implementation.

This module implements the Expectation-Maximization refinement loop with
Mean-Teacher architecture as described in the PDF specification.
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .models import (
    FusionModel,
    LPAModel,
    StudentSurrogate,
    copy_model_weights,
    update_teacher_ema,
)
from .config import EMRefinementConfig


@dataclass
class PseudoLabel:
    """Pseudo-label for a candidate pair."""
    node_i: int
    node_j: int
    final_score: float  # finalscore_ij^(r) in [0,1]
    p_value: float  # Empirical p-value from Step 3.3
    feature_vector: np.ndarray  # v_ij
    path_features: Optional[np.ndarray] = None  # For LPA model
    llm_score: Optional[float] = None  # For distillation in round 1


@dataclass
class AnswerKey:
    """Sparse answer key C_prior^(r)."""
    round: int
    labels: List[PseudoLabel]
    
    def get_high_confidence_labels(
        self,
        confidence_threshold: float = 0.7,
    ) -> List[PseudoLabel]:
        """Get high-confidence labels (w_ij > threshold)."""
        return [
            label for label in self.labels
            if (1 - label.p_value) >= confidence_threshold
        ]


class PseudoLabelDataset(Dataset):
    """Dataset for pseudo-labels."""
    
    def __init__(
        self,
        labels: List[PseudoLabel],
        augment: bool = False,
        config: Optional[EMRefinementConfig] = None,
    ) -> None:
        """Initialize dataset.
        
        Args:
            labels: List of pseudo-labels
            augment: Whether to apply augmentation
            config: Configuration for augmentation
        """
        self.labels = labels
        self.augment = augment
        self.config = config
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> dict:
        label = self.labels[idx]
        
        v_ij = torch.tensor(label.feature_vector, dtype=torch.float32)
        
        # Apply augmentation if enabled
        if self.augment and self.config:
            v_ij = self._augment_features(v_ij)
        
        return {
            "v_ij": v_ij,
            "final_score": torch.tensor(label.final_score, dtype=torch.float32),
            "p_value": torch.tensor(label.p_value, dtype=torch.float32),
            "confidence_weight": torch.tensor(1.0 - label.p_value, dtype=torch.float32),
            "node_i": label.node_i,
            "node_j": label.node_j,
            "path_features": (
                torch.tensor(label.path_features, dtype=torch.float32)
                if label.path_features is not None
                else None
            ),
        }
    
    def _augment_features(self, v_ij: torch.Tensor) -> torch.Tensor:
        """Apply feature augmentation.
        
        Args:
            v_ij: Feature vector
        
        Returns:
            Augmented feature vector
        """
        if self.config is None:
            return v_ij
        
        # Feature dropout
        if self.config.augmentation_dropout_rate > 0:
            mask = torch.rand_like(v_ij) > self.config.augmentation_dropout_rate
            v_ij = v_ij * mask
        
        # Add noise
        if self.config.augmentation_noise_std > 0:
            noise = torch.randn_like(v_ij) * self.config.augmentation_noise_std
            v_ij = v_ij + noise
        
        return v_ij


class EMRefinementLoop:
    """EM-Refinement Loop with Mean-Teacher architecture."""
    
    def __init__(
        self,
        config: EMRefinementConfig,
        pipeline_callback: Optional[Callable] = None,
    ) -> None:
        """Initialize EM refinement loop.
        
        Args:
            config: Configuration
            pipeline_callback: Callback function to run CoCaD pipeline
                Should accept teacher models and return AnswerKey
        """
        self.config = config
        self.device = torch.device(config.device)
        self.pipeline_callback = pipeline_callback
        
        # Initialize models (student and teacher)
        self.fusion_student = FusionModel(
            input_dim=config.fusion_input_dim,
            hidden_dims=config.fusion_hidden_dims,
            dropout=config.fusion_dropout,
        ).to(self.device)
        
        self.fusion_teacher = FusionModel(
            input_dim=config.fusion_input_dim,
            hidden_dims=config.fusion_hidden_dims,
            dropout=config.fusion_dropout,
        ).to(self.device)
        
        # Initialize teacher with student weights
        copy_model_weights(self.fusion_student, self.fusion_teacher)
        self.fusion_teacher.eval()
        
        self.lpa_student = LPAModel(
            path_feature_dim=config.lpa_path_feature_dim,
            hidden_dim=config.lpa_hidden_dim,
        ).to(self.device)
        
        self.lpa_teacher = LPAModel(
            path_feature_dim=config.lpa_path_feature_dim,
            hidden_dim=config.lpa_hidden_dim,
        ).to(self.device)
        
        copy_model_weights(self.lpa_student, self.lpa_teacher)
        self.lpa_teacher.eval()
        
        # Student surrogate for LLM distillation
        self.student_surrogate = StudentSurrogate(
            input_dim=config.student_surrogate_input_dim,
            hidden_dims=config.student_surrogate_hidden_dims,
            dropout=config.student_surrogate_dropout,
        ).to(self.device)
        
        # Optimizers
        self.fusion_optimizer = torch.optim.Adam(
            self.fusion_student.parameters(),
            lr=config.learning_rate,
        )
        
        self.lpa_optimizer = torch.optim.Adam(
            self.lpa_student.parameters(),
            lr=config.learning_rate,
        )
        
        self.surrogate_optimizer = torch.optim.Adam(
            self.student_surrogate.parameters(),
            lr=config.learning_rate,
        )
        
        # History
        self.history: List[Dict] = []
        self.distillation_data: List[Tuple[np.ndarray, float]] = []
    
    def e_step(self, round_num: int) -> AnswerKey:
        """E-Step: Generate pseudo-labels using teacher models.
        
        Args:
            round_num: Current round number
        
        Returns:
            Answer key with pseudo-labels
        """
        print(f"[EM-Loop] E-Step Round {round_num}: Generating pseudo-labels...")
        
        if self.pipeline_callback is None:
            raise ValueError("pipeline_callback must be provided for E-step")
        
        # Run CoCaD pipeline with teacher models
        answer_key = self.pipeline_callback(
            round_num=round_num,
            fusion_teacher=self.fusion_teacher,
            lpa_teacher=self.lpa_teacher,
            student_surrogate=(
                self.student_surrogate if round_num > 1 else None
            ),
        )
        
        # Store distillation data in round 1
        if round_num == 1:
            for label in answer_key.labels:
                if label.llm_score is not None:
                    self.distillation_data.append(
                        (label.feature_vector, label.llm_score)
                    )
        
        print(
            f"[EM-Loop] E-Step Round {round_num}: "
            f"Generated {len(answer_key.labels)} pseudo-labels"
        )
        
        return answer_key
    
    def m_step(
        self,
        answer_key: AnswerKey,
        round_num: int,
    ) -> Dict[str, float]:
        """M-Step: Update student models with pseudo-labels.
        
        Args:
            answer_key: Answer key with pseudo-labels
            round_num: Current round number
        
        Returns:
            Dictionary with loss values
        """
        print(f"[EM-Loop] M-Step Round {round_num}: Updating student models...")
        
        # Create datasets
        dataset = PseudoLabelDataset(answer_key.labels, augment=False, config=self.config)
        augmented_dataset = PseudoLabelDataset(
            answer_key.labels, augment=True, config=self.config
        )
        
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        augmented_loader = DataLoader(
            augmented_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        
        losses = {
            "fusion_loss": [],
            "lpa_loss": [],
            "consistency_loss": [],
            "distill_loss": [],
        }
        
        # Train fusion model
        self.fusion_student.train()
        for epoch in range(self.config.num_epochs_per_round):
            epoch_losses = {"fusion": [], "consistency": []}
            
            for batch, aug_batch in zip(loader, augmented_loader):
                v_ij = batch["v_ij"].to(self.device)
                y_soft = batch["final_score"].to(self.device)
                w_ij = batch["confidence_weight"].to(self.device)
                
                # Pseudo-label loss (confidence-weighted BCE)
                pred = self.fusion_student(v_ij)
                bce_loss = F.binary_cross_entropy(
                    pred, y_soft, reduction="none"
                )
                pseudo_loss = (w_ij * bce_loss).mean()
                
                # Consistency loss
                v_ij_aug1 = aug_batch["v_ij"].to(self.device)
                v_ij_aug2 = batch["v_ij"].to(self.device)  # Use original as second view
                
                pred_aug1 = self.fusion_student(v_ij_aug1)
                pred_aug2 = self.fusion_student(v_ij_aug2)
                
                consistency_loss = F.mse_loss(pred_aug1, pred_aug2)
                
                # Total loss
                total_loss = (
                    pseudo_loss
                    + self.config.lambda_consistency * consistency_loss
                )
                
                self.fusion_optimizer.zero_grad()
                total_loss.backward()
                self.fusion_optimizer.step()
                
                epoch_losses["fusion"].append(pseudo_loss.item())
                epoch_losses["consistency"].append(consistency_loss.item())
            
            losses["fusion_loss"].append(np.mean(epoch_losses["fusion"]))
            losses["consistency_loss"].append(np.mean(epoch_losses["consistency"]))
        
        # Train LPA model
        self.lpa_student.train()
        for epoch in range(self.config.num_epochs_per_round):
            epoch_losses = []
            
            for batch in loader:
                if batch["path_features"][0] is None:
                    continue  # Skip if no path features
                
                # Stack path features (assuming fixed number of paths)
                path_features = torch.stack([
                    p if p is not None else torch.zeros(self.config.lpa_path_feature_dim)
                    for p in batch["path_features"]
                ]).to(self.device)
                
                # Reshape for LPA: [batch_size, num_paths, path_feature_dim]
                # For simplicity, assume single path per pair
                path_features = path_features.unsqueeze(1)  # [batch_size, 1, path_feature_dim]
                
                y_bin = (batch["final_score"] > 0).float().to(self.device)
                w_ij = batch["confidence_weight"].to(self.device)
                
                pred = self.lpa_student(path_features).squeeze()
                
                bce_loss = F.binary_cross_entropy(
                    pred, y_bin, reduction="none"
                )
                loss = (w_ij * bce_loss).mean()
                
                self.lpa_optimizer.zero_grad()
                loss.backward()
                self.lpa_optimizer.step()
                
                epoch_losses.append(loss.item())
            
            if epoch_losses:
                losses["lpa_loss"].append(np.mean(epoch_losses))
        
        # Train student surrogate (round 1 only)
        if round_num == 1 and self.distillation_data:
            self.student_surrogate.train()
            distill_dataset = [
                (torch.tensor(v, dtype=torch.float32), torch.tensor(t, dtype=torch.float32))
                for v, t in self.distillation_data
            ]
            
            for epoch in range(self.config.num_epochs_per_round):
                epoch_losses = []
                
                for v_ij, t_ij in distill_dataset:
                    v_ij = v_ij.unsqueeze(0).to(self.device)
                    t_ij = t_ij.unsqueeze(0).to(self.device)
                    
                    pred = self.student_surrogate(v_ij)
                    loss = F.mse_loss(pred, t_ij)
                    
                    self.surrogate_optimizer.zero_grad()
                    loss.backward()
                    self.surrogate_optimizer.step()
                    
                    epoch_losses.append(loss.item())
                
                if epoch_losses:
                    losses["distill_loss"].append(np.mean(epoch_losses))
        
        avg_losses = {
            k: np.mean(v) if v else 0.0
            for k, v in losses.items()
        }
        
        print(
            f"[EM-Loop] M-Step Round {round_num} complete. "
            f"Losses: {avg_losses}"
        )
        
        return avg_losses
    
    def update_teachers(self) -> None:
        """Update teacher models via EMA."""
        update_teacher_ema(
            self.fusion_teacher,
            self.fusion_student,
            alpha=self.config.teacher_ema_alpha,
        )
        update_teacher_ema(
            self.lpa_teacher,
            self.lpa_student,
            alpha=self.config.teacher_ema_alpha,
        )
        self.fusion_teacher.eval()
        self.lpa_teacher.eval()
    
    def compute_jaccard_similarity(
        self,
        answer_key_prev: AnswerKey,
        answer_key_curr: AnswerKey,
    ) -> float:
        """Compute Jaccard similarity between high-confidence label sets.
        
        Args:
            answer_key_prev: Previous round answer key
            answer_key_curr: Current round answer key
        
        Returns:
            Jaccard similarity
        """
        prev_high_conf = set(
            (label.node_i, label.node_j)
            for label in answer_key_prev.get_high_confidence_labels(
                self.config.confidence_threshold
            )
        )
        
        curr_high_conf = set(
            (label.node_i, label.node_j)
            for label in answer_key_curr.get_high_confidence_labels(
                self.config.confidence_threshold
            )
        )
        
        intersection = len(prev_high_conf & curr_high_conf)
        union = len(prev_high_conf | curr_high_conf)
        
        if union == 0:
            return 1.0
        
        return intersection / union
    
    def run(
        self,
        validation_data: Optional[List[PseudoLabel]] = None,
    ) -> Dict:
        """Run the complete EM-Refinement Loop.
        
        Args:
            validation_data: Optional validation data for monitoring
        
        Returns:
            Dictionary with final models and history
        """
        print("[EM-Loop] Starting EM-Refinement Loop...")
        
        answer_key_prev: Optional[AnswerKey] = None
        best_val_loss = float("inf")
        plateau_count = 0
        
        for round_num in range(1, self.config.num_rounds + 1):
            print(f"\n{'='*60}")
            print(f"EM Round {round_num}/{self.config.num_rounds}")
            print(f"{'='*60}")
            
            # E-Step: Generate pseudo-labels
            answer_key_curr = self.e_step(round_num)
            
            # M-Step: Update student models
            losses = self.m_step(answer_key_curr, round_num)
            
            # Update teachers
            self.update_teachers()
            
            # Compute metrics
            metrics = {
                "round": round_num,
                "num_labels": len(answer_key_curr.labels),
                "num_high_conf": len(
                    answer_key_curr.get_high_confidence_labels(
                        self.config.confidence_threshold
                    )
                ),
                **losses,
            }
            
            # Label stability (Jaccard similarity)
            if answer_key_prev is not None:
                jaccard = self.compute_jaccard_similarity(
                    answer_key_prev, answer_key_curr
                )
                metrics["jaccard_similarity"] = jaccard
                
                print(f"[EM-Loop] Jaccard similarity: {jaccard:.4f}")
                
                # Check stopping criterion: label stability
                if jaccard >= self.config.jaccard_threshold:
                    print(
                        f"[EM-Loop] Stopping: Label stability reached "
                        f"(Jaccard >= {self.config.jaccard_threshold})"
                    )
                    break
            
            # Validation plateau check
            if validation_data:
                val_loss = self._compute_validation_loss(validation_data)
                metrics["validation_loss"] = val_loss
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    plateau_count = 0
                else:
                    plateau_count += 1
                
                if plateau_count >= self.config.validation_plateau_rounds:
                    print(
                        f"[EM-Loop] Stopping: Validation plateau "
                        f"({plateau_count} rounds without improvement)"
                    )
                    break
            
            self.history.append(metrics)
            answer_key_prev = answer_key_curr
            
            # Save checkpoint
            self.save_checkpoint(round_num)
        
        print("\n[EM-Loop] EM-Refinement Loop complete!")
        
        return {
            "fusion_teacher": self.fusion_teacher,
            "lpa_teacher": self.lpa_teacher,
            "student_surrogate": self.student_surrogate,
            "history": self.history,
        }
    
    def _compute_validation_loss(
        self,
        validation_data: List[PseudoLabel],
    ) -> float:
        """Compute validation loss on held-out data.
        
        Args:
            validation_data: Validation pseudo-labels
        
        Returns:
            Average validation loss
        """
        self.fusion_student.eval()
        
        total_loss = 0.0
        count = 0
        
        with torch.no_grad():
            for label in validation_data:
                v_ij = torch.tensor(
                    label.feature_vector, dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                
                y_soft = torch.tensor(
                    label.final_score, dtype=torch.float32
                ).to(self.device)
                
                pred = self.fusion_student(v_ij)
                loss = F.binary_cross_entropy(pred, y_soft.unsqueeze(0))
                
                total_loss += loss.item()
                count += 1
        
        return total_loss / count if count > 0 else float("inf")
    
    def save_checkpoint(self, round_num: int) -> None:
        """Save model checkpoints.
        
        Args:
            round_num: Current round number
        """
        checkpoint_dir = self.config.model_save_dir / f"round_{round_num}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        torch.save(
            self.fusion_student.state_dict(),
            checkpoint_dir / "fusion_student.pt",
        )
        torch.save(
            self.fusion_teacher.state_dict(),
            checkpoint_dir / "fusion_teacher.pt",
        )
        torch.save(
            self.lpa_student.state_dict(),
            checkpoint_dir / "lpa_student.pt",
        )
        torch.save(
            self.lpa_teacher.state_dict(),
            checkpoint_dir / "lpa_teacher.pt",
        )
        torch.save(
            self.student_surrogate.state_dict(),
            checkpoint_dir / "student_surrogate.pt",
        )
        
        # Save history
        with open(checkpoint_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
    
    def load_checkpoint(self, round_num: int) -> None:
        """Load model checkpoints.
        
        Args:
            round_num: Round number to load
        """
        checkpoint_dir = self.config.model_save_dir / f"round_{round_num}"
        
        self.fusion_student.load_state_dict(
            torch.load(checkpoint_dir / "fusion_student.pt", map_location=self.device)
        )
        self.fusion_teacher.load_state_dict(
            torch.load(checkpoint_dir / "fusion_teacher.pt", map_location=self.device)
        )
        self.lpa_student.load_state_dict(
            torch.load(checkpoint_dir / "lpa_student.pt", map_location=self.device)
        )
        self.lpa_teacher.load_state_dict(
            torch.load(checkpoint_dir / "lpa_teacher.pt", map_location=self.device)
        )
        self.student_surrogate.load_state_dict(
            torch.load(checkpoint_dir / "student_surrogate.pt", map_location=self.device)
        )
        
        with open(checkpoint_dir / "history.json", "r") as f:
            self.history = json.load(f)

