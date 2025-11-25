"""CPC (Causal Plausibility Classifier) Model Training.

This module implements Priority 3, Part B: Model training and calibration.

Architecture:
- DeBERTa-v3 base encoder
- Three task-specific heads (Plausible, Temporal, Mechanistic)
- Multi-task BCE loss
- Isotonic regression calibration
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .cpc_dataset import CPCDatasetEntry


class CPCDataset(Dataset):
    """PyTorch Dataset for CPC training."""
    
    def __init__(
        self,
        entries: List[CPCDatasetEntry],
        tokenizer,
        max_length: int = 512,
    ) -> None:
        """Initialize dataset.
        
        Args:
            entries: List of dataset entries
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.entries = entries
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        
        # Format: [CLS]context[SEP]node_i[SEP]node_j[SEP]
        text = f"{entry.context} [SEP] {entry.node_i} [SEP] {entry.node_j}"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label_plausible": torch.tensor(1.0 if entry.label_plausible else 0.0),
            "label_temporal": torch.tensor(1.0 if entry.label_temporal else 0.0),
            "label_mechanistic": torch.tensor(1.0 if entry.label_mechanistic else 0.0),
        }


class CPCModel(nn.Module):
    """Causal Plausibility Classifier with three task-specific heads."""
    
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        hidden_dim: int = 768,
    ) -> None:
        """Initialize CPC model.
        
        Args:
            model_name: HuggingFace model name
            hidden_dim: Hidden dimension of the encoder
        """
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_dim = hidden_dim
        
        # Three independent heads
        self.head_plausible = nn.Linear(hidden_dim, 1)
        self.head_temporal = nn.Linear(hidden_dim, 1)
        self.head_mechanistic = nn.Linear(hidden_dim, 1)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Returns:
            Tuple of (logit_plausible, logit_temporal, logit_mechanistic)
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        cls_representation = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
        
        logit_plausible = self.head_plausible(cls_representation).squeeze(-1)
        logit_temporal = self.head_temporal(cls_representation).squeeze(-1)
        logit_mechanistic = self.head_mechanistic(cls_representation).squeeze(-1)
        
        return logit_plausible, logit_temporal, logit_mechanistic


@dataclass
class CPCTrainingConfig:
    """Configuration for CPC training."""
    model_name: str = "microsoft/deberta-v3-base"
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    max_length: int = 512
    validation_split: float = 0.1  # For calibrator training
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class CPCTrainer:
    """Trains and calibrates the CPC model."""
    
    def __init__(self, config: CPCTrainingConfig) -> None:
        """Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = CPCModel(model_name=config.model_name).to(self.device)
        
        # Calibrators (fitted after training)
        self.calibrator_plausible: Optional[IsotonicRegression] = None
        self.calibrator_temporal: Optional[IsotonicRegression] = None
        self.calibrator_mechanistic: Optional[IsotonicRegression] = None
    
    def train(
        self,
        train_entries: List[CPCDatasetEntry],
        validation_entries: Optional[List[CPCDatasetEntry]] = None,
    ) -> dict:
        """Train the CPC model.
        
        Args:
            train_entries: Training dataset entries
            validation_entries: Optional validation entries (if None, split from train)
        
        Returns:
            Training history dictionary
        """
        # Split validation set if needed
        if validation_entries is None:
            split_idx = int(len(train_entries) * (1 - self.config.validation_split))
            train_entries, validation_entries = (
                train_entries[:split_idx],
                train_entries[split_idx:],
            )
        
        # Create datasets
        train_dataset = CPCDataset(train_entries, self.tokenizer, self.config.max_length)
        val_dataset = CPCDataset(validation_entries, self.tokenizer, self.config.max_length)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )
        
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )
        
        # Loss function (BCE for each head)
        criterion = nn.BCEWithLogitsLoss()
        
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_auroc_plausible": [],
            "val_auroc_temporal": [],
            "val_auroc_mechanistic": [],
        }
        
        best_val_loss = float("inf")
        
        for epoch in range(self.config.num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                label_plausible = batch["label_plausible"].to(self.device)
                label_temporal = batch["label_temporal"].to(self.device)
                label_mechanistic = batch["label_mechanistic"].to(self.device)
                
                optimizer.zero_grad()
                
                logit_plausible, logit_temporal, logit_mechanistic = self.model(
                    input_ids, attention_mask
                )
                
                # Multi-task loss
                loss_plausible = criterion(logit_plausible, label_plausible)
                loss_temporal = criterion(logit_temporal, label_temporal)
                loss_mechanistic = criterion(logit_mechanistic, label_mechanistic)
                loss = loss_plausible + loss_temporal + loss_mechanistic
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            val_loss, val_aurocs = self._validate(val_loader, criterion)
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_auroc_plausible"].append(val_aurocs["plausible"])
            history["val_auroc_temporal"].append(val_aurocs["temporal"])
            history["val_auroc_mechanistic"].append(val_aurocs["mechanistic"])
            
            print(
                f"Epoch {epoch+1}/{self.config.num_epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"val_auroc_plausible={val_aurocs['plausible']:.4f}, "
                f"val_auroc_temporal={val_aurocs['temporal']:.4f}, "
                f"val_auroc_mechanistic={val_aurocs['mechanistic']:.4f}"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Model state will be saved separately
        
        return history
    
    def _validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, dict]:
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        
        all_logits_plausible = []
        all_logits_temporal = []
        all_logits_mechanistic = []
        all_labels_plausible = []
        all_labels_temporal = []
        all_labels_mechanistic = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                label_plausible = batch["label_plausible"].to(self.device)
                label_temporal = batch["label_temporal"].to(self.device)
                label_mechanistic = batch["label_mechanistic"].to(self.device)
                
                logit_plausible, logit_temporal, logit_mechanistic = self.model(
                    input_ids, attention_mask
                )
                
                loss_plausible = criterion(logit_plausible, label_plausible)
                loss_temporal = criterion(logit_temporal, label_temporal)
                loss_mechanistic = criterion(logit_mechanistic, label_mechanistic)
                loss = loss_plausible + loss_temporal + loss_mechanistic
                
                val_loss += loss.item()
                
                all_logits_plausible.extend(logit_plausible.cpu().numpy())
                all_logits_temporal.extend(logit_temporal.cpu().numpy())
                all_logits_mechanistic.extend(logit_mechanistic.cpu().numpy())
                all_labels_plausible.extend(label_plausible.cpu().numpy())
                all_labels_temporal.extend(label_temporal.cpu().numpy())
                all_labels_mechanistic.extend(label_mechanistic.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        # Compute AUROC
        auroc_plausible = roc_auc_score(
            all_labels_plausible,
            torch.sigmoid(torch.tensor(all_logits_plausible)).numpy(),
        )
        auroc_temporal = roc_auc_score(
            all_labels_temporal,
            torch.sigmoid(torch.tensor(all_logits_temporal)).numpy(),
        )
        auroc_mechanistic = roc_auc_score(
            all_labels_mechanistic,
            torch.sigmoid(torch.tensor(all_logits_mechanistic)).numpy(),
        )
        
        return val_loss, {
            "plausible": auroc_plausible,
            "temporal": auroc_temporal,
            "mechanistic": auroc_mechanistic,
        }
    
    def fit_calibrators(
        self,
        validation_entries: List[CPCDatasetEntry],
    ) -> None:
        """Step 5: Fit isotonic regression calibrators on hold-out set.
        
        Args:
            validation_entries: Hold-out validation entries
        """
        val_dataset = CPCDataset(validation_entries, self.tokenizer, self.config.max_length)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        self.model.eval()
        
        all_logits_plausible = []
        all_logits_temporal = []
        all_logits_mechanistic = []
        all_labels_plausible = []
        all_labels_temporal = []
        all_labels_mechanistic = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                label_plausible = batch["label_plausible"].to(self.device)
                label_temporal = batch["label_temporal"].to(self.device)
                label_mechanistic = batch["label_mechanistic"].to(self.device)
                
                logit_plausible, logit_temporal, logit_mechanistic = self.model(
                    input_ids, attention_mask
                )
                
                all_logits_plausible.extend(logit_plausible.cpu().numpy())
                all_logits_temporal.extend(logit_temporal.cpu().numpy())
                all_logits_mechanistic.extend(logit_mechanistic.cpu().numpy())
                all_labels_plausible.extend(label_plausible.cpu().numpy())
                all_labels_temporal.extend(label_temporal.cpu().numpy())
                all_labels_mechanistic.extend(label_mechanistic.cpu().numpy())
        
        # Convert logits to raw probabilities
        raw_probs_plausible = torch.sigmoid(torch.tensor(all_logits_plausible)).numpy()
        raw_probs_temporal = torch.sigmoid(torch.tensor(all_logits_temporal)).numpy()
        raw_probs_mechanistic = torch.sigmoid(torch.tensor(all_logits_mechanistic)).numpy()
        
        # Fit calibrators
        self.calibrator_plausible = IsotonicRegression(out_of_bounds="clip")
        self.calibrator_plausible.fit(raw_probs_plausible, all_labels_plausible)
        
        self.calibrator_temporal = IsotonicRegression(out_of_bounds="clip")
        self.calibrator_temporal.fit(raw_probs_temporal, all_labels_temporal)
        
        self.calibrator_mechanistic = IsotonicRegression(out_of_bounds="clip")
        self.calibrator_mechanistic.fit(raw_probs_mechanistic, all_labels_mechanistic)
        
        print("[cpc-trainer] Calibrators fitted successfully")
    
    def predict(
        self,
        context: str,
        node_i: str,
        node_j: str,
        calibrated: bool = True,
    ) -> dict:
        """Predict probabilities for a single example.
        
        Args:
            context: Context string
            node_i: First node
            node_j: Second node
            calibrated: Whether to use calibrated probabilities
        
        Returns:
            Dictionary with probabilities for each head
        """
        text = f"{context} [SEP] {node_i} [SEP] {node_j}"
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            logit_plausible, logit_temporal, logit_mechanistic = self.model(
                input_ids, attention_mask
            )
        
        # Convert to probabilities
        p_plausible = torch.sigmoid(logit_plausible).item()
        p_temporal = torch.sigmoid(logit_temporal).item()
        p_mechanistic = torch.sigmoid(logit_mechanistic).item()
        
        # Apply calibration if available
        if calibrated:
            if self.calibrator_plausible is not None:
                p_plausible = self.calibrator_plausible.predict([p_plausible])[0]
            if self.calibrator_temporal is not None:
                p_temporal = self.calibrator_temporal.predict([p_temporal])[0]
            if self.calibrator_mechanistic is not None:
                p_mechanistic = self.calibrator_mechanistic.predict([p_mechanistic])[0]
        
        return {
            "plausible": float(p_plausible),
            "temporal": float(p_temporal),
            "mechanistic": float(p_mechanistic),
        }
    
    def save(
        self,
        model_path: Path,
        calibrators_path: Path,
    ) -> None:
        """Save model and calibrators.
        
        Args:
            model_path: Path to save model weights
            calibrators_path: Path to save calibrators
        """
        # Save model
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        
        # Save calibrators
        calibrators = {
            "plausible": self.calibrator_plausible,
            "temporal": self.calibrator_temporal,
            "mechanistic": self.calibrator_mechanistic,
        }
        with calibrators_path.open("wb") as f:
            pickle.dump(calibrators, f)
        
        print(f"[cpc-trainer] Saved model to {model_path}")
        print(f"[cpc-trainer] Saved calibrators to {calibrators_path}")
    
    def load(
        self,
        model_path: Path,
        calibrators_path: Path,
    ) -> None:
        """Load model and calibrators.
        
        Args:
            model_path: Path to model weights
            calibrators_path: Path to calibrators
        """
        # Load model
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Load calibrators
        with calibrators_path.open("rb") as f:
            calibrators = pickle.load(f)
            self.calibrator_plausible = calibrators["plausible"]
            self.calibrator_temporal = calibrators["temporal"]
            self.calibrator_mechanistic = calibrators["mechanistic"]
        
        print(f"[cpc-trainer] Loaded model from {model_path}")
        print(f"[cpc-trainer] Loaded calibrators from {calibrators_path}")

