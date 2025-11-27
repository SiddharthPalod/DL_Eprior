"""Integration test for complete EM-Refinement Loop."""

import numpy as np
import torch
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from em_loop.src.config import EMRefinementConfig
from em_loop.src.em_refinement import EMRefinementLoop, PseudoLabel, AnswerKey
from em_loop.src.models import FusionModel, LPAModel


def create_synthetic_pipeline_callback(num_pairs: int = 200):
    """Create a synthetic pipeline callback that simulates realistic behavior."""
    
    def pipeline_callback(
        round_num: int,
        fusion_teacher: FusionModel,
        lpa_teacher: LPAModel,
        student_surrogate=None,
    ) -> AnswerKey:
        """Synthetic pipeline callback with round-dependent behavior."""
        print(f"  [Synthetic Pipeline] Round {round_num}: Processing {num_pairs} pairs...")
        
        labels = []
        config = EMRefinementConfig()
        device = next(fusion_teacher.parameters()).device
        
        for i in range(num_pairs):
            node_i = i
            node_j = (i + 1) % num_pairs
            
            # Generate feature vector
            feature_vector = np.random.randn(config.fusion_input_dim).astype(np.float32)
            
            # Use teacher model to get score
            with torch.no_grad():
                v_ij_tensor = torch.tensor(feature_vector).unsqueeze(0).to(device)
                final_score = fusion_teacher(v_ij_tensor).item()
                final_score = max(0.0, min(1.0, final_score))  # Clamp to [0,1]
            
            # P-value: inversely related to score (higher score -> lower p-value)
            # Add some noise to make it realistic
            p_value = max(0.0, min(1.0, (1.0 - final_score) + np.random.normal(0, 0.1)))
            
            # Path features
            path_features = np.random.randn(config.lpa_path_feature_dim).astype(np.float32)
            
            # LLM score for distillation (round 1 only)
            llm_score = None
            if round_num == 1 and i % 2 == 0:  # 50% of pairs
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
        
        return AnswerKey(round=round_num, labels=labels)
    
    return pipeline_callback


def test_full_em_loop():
    """Test the complete EM-Refinement Loop."""
    print("=" * 60)
    print("Integration Test: Full EM-Refinement Loop")
    print("=" * 60 + "\n")
    
    # Configure for testing
    config = EMRefinementConfig()
    config.num_rounds = 5
    config.num_epochs_per_round = 3
    config.batch_size = 32
    config.learning_rate = 1e-3
    config.jaccard_threshold = 0.98  # High threshold
    config.confidence_threshold = 0.7
    config.validation_plateau_rounds = 3
    
    # Create pipeline callback
    pipeline_callback = create_synthetic_pipeline_callback(num_pairs=100)
    
    # Initialize EM loop
    print("Initializing EM-Refinement Loop...")
    em_loop = EMRefinementLoop(config, pipeline_callback=pipeline_callback)
    
    # Run EM loop
    print("\nRunning EM-Refinement Loop...")
    results = em_loop.run(validation_data=None)
    
    # Check results
    assert "fusion_teacher" in results, "Results should contain fusion_teacher"
    assert "lpa_teacher" in results, "Results should contain lpa_teacher"
    assert "student_surrogate" in results, "Results should contain student_surrogate"
    assert "history" in results, "Results should contain history"
    assert len(results["history"]) > 0, "History should not be empty"
    
    print("\n" + "=" * 60)
    print("Integration Test Results:")
    print("=" * 60)
    print(f"  ✓ Completed {len(results['history'])} rounds")
    print(f"  ✓ Final models saved")
    
    # Print history summary
    print("\nRound History:")
    for entry in results["history"]:
        round_num = entry["round"]
        num_labels = entry["num_labels"]
        num_high_conf = entry["num_high_conf"]
        fusion_loss = entry.get("fusion_loss", 0.0)
        jaccard = entry.get("jaccard_similarity", None)
        
        print(f"  Round {round_num}: {num_labels} labels, {num_high_conf} high-conf, "
              f"fusion_loss={fusion_loss:.4f}", end="")
        if jaccard is not None:
            print(f", jaccard={jaccard:.4f}")
        else:
            print()
    
    print("\n" + "=" * 60)
    print("Integration test passed! ✓")
    print("=" * 60)


def test_checkpoint_save_load():
    """Test checkpoint saving and loading."""
    print("\n" + "=" * 60)
    print("Testing Checkpoint Save/Load")
    print("=" * 60 + "\n")
    
    config = EMRefinementConfig()
    config.num_rounds = 2
    config.num_epochs_per_round = 1
    config.batch_size = 16
    
    pipeline_callback = create_synthetic_pipeline_callback(num_pairs=30)
    
    # Create and run one round
    em_loop1 = EMRefinementLoop(config, pipeline_callback=pipeline_callback)
    answer_key = em_loop1.e_step(round_num=1)
    em_loop1.m_step(answer_key, round_num=1)
    em_loop1.update_teachers()
    em_loop1.save_checkpoint(round_num=1)
    
    # Create new loop and load checkpoint
    em_loop2 = EMRefinementLoop(config, pipeline_callback=pipeline_callback)
    em_loop2.load_checkpoint(round_num=1)
    
    # Check that models match
    for p1, p2 in zip(
        em_loop1.fusion_student.parameters(),
        em_loop2.fusion_student.parameters()
    ):
        assert torch.allclose(p1, p2), "Loaded model should match saved model"
    
    print("  ✓ Checkpoint saved successfully")
    print("  ✓ Checkpoint loaded successfully")
    print("  ✓ Model parameters match")
    print("\n  ✓ Checkpoint save/load test passed!")


if __name__ == "__main__":
    test_full_em_loop()
    test_checkpoint_save_load()
    
    print("\n" + "=" * 60)
    print("All integration tests passed! ✓")
    print("=" * 60)

