"""Example usage of EM-Refinement Loop.

This script demonstrates how to use the EM-Refinement Loop with a mock pipeline.
"""

import numpy as np
import torch
from pathlib import Path

from em_loop.src.config import EMRefinementConfig
from em_loop.src.em_refinement import EMRefinementLoop, AnswerKey, PseudoLabel
from em_loop.src.models import FusionModel, LPAModel


def create_example_pipeline_callback(num_pairs: int = 100):
    """Create an example pipeline callback.
    
    In a real implementation, this would:
    1. Load the document corpus
    2. Run the CoCaD pipeline (Steps 3.1-3.3) with teacher models
    3. Compute final scores, p-values, and feature vectors
    4. Return an AnswerKey with pseudo-labels
    """
    
    def pipeline_callback(
        round_num: int,
        fusion_teacher: FusionModel,
        lpa_teacher: LPAModel,
        student_surrogate=None,
    ) -> AnswerKey:
        """Example pipeline callback."""
        print(f"\n  [Pipeline] Round {round_num}: Running CoCaD pipeline...")
        print(f"  [Pipeline] Using teacher models to generate pseudo-labels...")
        
        labels = []
        config = EMRefinementConfig()
        device = next(fusion_teacher.parameters()).device
        
        # Simulate running pipeline on candidate pairs
        for i in range(num_pairs):
            node_i = i
            node_j = (i + 1) % num_pairs
            
            # Generate feature vector (in real pipeline, this comes from Step 3.2)
            feature_vector = np.random.randn(config.fusion_input_dim).astype(np.float32)
            
            # Use teacher fusion model to get score
            with torch.no_grad():
                v_ij_tensor = torch.tensor(feature_vector).unsqueeze(0).to(device)
                final_score = fusion_teacher(v_ij_tensor).item()
                final_score = max(0.0, min(1.0, final_score))
            
            # Compute p-value (in real pipeline, from Step 3.3)
            # Higher scores should have lower p-values
            p_value = max(0.0, min(1.0, (1.0 - final_score) + np.random.normal(0, 0.1)))
            
            # Path features for LPA model
            path_features = np.random.randn(config.lpa_path_feature_dim).astype(np.float32)
            
            # LLM score for distillation (only in round 1)
            llm_score = None
            if round_num == 1 and i % 2 == 0:  # 50% of pairs
                # In real pipeline, this comes from Step 3.1 (conditional LLM)
                llm_score = final_score + np.random.normal(0, 0.05)
                llm_score = max(0.0, min(1.0, llm_score))
            
            label = PseudoLabel(
                node_i=node_i,
                node_j=node_j,
                final_score=final_score,
                p_value=p_value,
                feature_vector=feature_vector,
                path_features=path_features,
                llm_score=llm_score,
            )
            labels.append(label)
        
        print(f"  [Pipeline] Generated {len(labels)} pseudo-labels")
        return AnswerKey(round=round_num, labels=labels)
    
    return pipeline_callback


def main():
    """Run example EM-Refinement Loop."""
    print("=" * 70)
    print("EM-Refinement Loop Example")
    print("=" * 70)
    print()
    
    # Configure EM loop
    config = EMRefinementConfig(
        num_rounds=5,
        num_epochs_per_round=3,
        batch_size=32,
        learning_rate=1e-3,
        jaccard_threshold=0.98,
        confidence_threshold=0.7,
    )
    
    print("Configuration:")
    print(f"  Rounds: {config.num_rounds}")
    print(f"  Epochs per round: {config.num_epochs_per_round}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Jaccard threshold: {config.jaccard_threshold}")
    print(f"  Confidence threshold: {config.confidence_threshold}")
    print()
    
    # Create pipeline callback
    pipeline_callback = create_example_pipeline_callback(num_pairs=100)
    
    # Initialize EM loop
    print("Initializing EM-Refinement Loop...")
    em_loop = EMRefinementLoop(config, pipeline_callback=pipeline_callback)
    print("  ✓ Models initialized (student and teacher)")
    print()
    
    # Run EM loop
    print("Running EM-Refinement Loop...")
    print("=" * 70)
    results = em_loop.run()
    print("=" * 70)
    
    # Print results
    print("\nResults:")
    print(f"  Completed {len(results['history'])} rounds")
    print(f"  Final models saved to: {config.model_save_dir}")
    
    # Print round-by-round summary
    print("\nRound Summary:")
    for entry in results["history"]:
        round_num = entry["round"]
        num_labels = entry["num_labels"]
        num_high_conf = entry["num_high_conf"]
        fusion_loss = entry.get("fusion_loss", 0.0)
        lpa_loss = entry.get("lpa_loss", 0.0)
        jaccard = entry.get("jaccard_similarity", None)
        
        print(f"\n  Round {round_num}:")
        print(f"    Labels: {num_labels} (high-confidence: {num_high_conf})")
        print(f"    Fusion loss: {fusion_loss:.4f}")
        print(f"    LPA loss: {lpa_loss:.4f}")
        if jaccard is not None:
            print(f"    Jaccard similarity: {jaccard:.4f}")
    
    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)
    
    # Access final models
    print("\nFinal models available:")
    print("  - results['fusion_teacher']: Final fusion model")
    print("  - results['lpa_teacher']: Final LPA model")
    print("  - results['student_surrogate']: Final student surrogate")
    print("  - results['history']: Training history")


if __name__ == "__main__":
    main()

