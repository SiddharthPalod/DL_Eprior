"""Example script for training the CPC model.

This demonstrates the full CPC training pipeline:
1. Dataset construction with Teacher LLM
2. Model training
3. Calibration
"""

from pathlib import Path

from enhanced_components.cpc_dataset import CPCDatasetBuilder, CPCDatasetEntry
from enhanced_components.cpc_model import CPCTrainer, CPCTrainingConfig


def main():
    """Example CPC training pipeline."""
    
    # Step 1: Prepare unlabeled dataset
    # In practice, this would come from the ACE pipeline (structural candidates + RAG contexts)
    unlabeled_examples = [
        {
            "node_i": "XGBoost",
            "node_j": "accuracy",
            "context": "We evaluated XGBoost on the ImageNet dataset. The performance achieved high accuracy...",
        },
        {
            "node_i": "learning_rate",
            "node_j": "convergence",
            "context": "A lower learning rate leads to slower but more stable convergence...",
        },
        # ... more examples
    ]
    
    # Step 2: Label with Teacher LLM
    print("Step 2: Labeling with Teacher LLM...")
    builder = CPCDatasetBuilder(api_key=None)  # Uses GEMINI_API_KEY env var
    
    labeled_entries: list[CPCDatasetEntry] = []
    for example in unlabeled_examples:
        entry = builder.label_with_teacher_llm(
            node_i=example["node_i"],
            node_j=example["node_j"],
            context=example["context"],
        )
        if entry:
            labeled_entries.append(entry)
    
    print(f"Labeled {len(labeled_entries)} examples")
    
    # Step 3: Separate into positive/negative
    positive_entries = [e for e in labeled_entries if e.label_plausible]
    negative_entries = [e for e in labeled_entries if not e.label_plausible]
    
    # Step 4: Generate adversarial hard negatives (simplified example)
    # In practice, use GAE embeddings to find semantically similar but structurally distant pairs
    hard_negative_entries = negative_entries[:len(negative_entries) // 2]  # Simplified
    
    # Step 5: Build balanced dataset
    print("Step 5: Building balanced dataset...")
    balanced = builder.build_balanced_dataset(
        positive_entries=positive_entries,
        easy_negative_entries=negative_entries[len(negative_entries) // 2:],
        hard_negative_entries=hard_negative_entries,
    )
    
    # Save dataset
    dataset_path = Path("outputs/traindataset.jsonl")
    builder.save_dataset(balanced, dataset_path)
    
    # Step 6: Train CPC model
    print("Step 6: Training CPC model...")
    config = CPCTrainingConfig(
        batch_size=16,
        learning_rate=2e-5,
        num_epochs=3,
    )
    
    trainer = CPCTrainer(config)
    history = trainer.train(balanced)
    
    # Step 7: Fit calibrators
    print("Step 7: Fitting calibrators...")
    # Split validation set (10% for calibration)
    split_idx = int(len(balanced) * 0.9)
    train_entries, val_entries = balanced[:split_idx], balanced[split_idx:]
    trainer.fit_calibrators(val_entries)
    
    # Step 8: Save model and calibrators
    print("Step 8: Saving model...")
    trainer.save(
        model_path=Path("outputs/CPCModel.bin"),
        calibrators_path=Path("outputs/Calibrators.pkl"),
    )
    
    # Step 9: Test inference
    print("Step 9: Testing inference...")
    probs = trainer.predict(
        context="XGBoost achieves high accuracy through gradient boosting...",
        node_i="XGBoost",
        node_j="accuracy",
        calibrated=True,
    )
    
    print(f"\nPredicted probabilities:")
    print(f"  Plausible: {probs['plausible']:.3f}")
    print(f"  Temporal: {probs['temporal']:.3f}")
    print(f"  Mechanistic: {probs['mechanistic']:.3f}")
    
    # Apply filtering rule
    tau_plausible = 0.5
    tau_temporal = 0.3
    tau_mechanistic = 0.3
    
    if probs["plausible"] > tau_plausible and (
        probs["temporal"] > tau_temporal or probs["mechanistic"] > tau_mechanistic
    ):
        print("\n✓ Pair passes CPC filtering")
    else:
        print("\n✗ Pair rejected by CPC filtering")


if __name__ == "__main__":
    main()

