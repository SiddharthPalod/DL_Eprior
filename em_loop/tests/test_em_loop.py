"""Test EM-Refinement Loop with mock data."""

import numpy as np
import torch
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from em_loop.src.config import EMRefinementConfig
from em_loop.src.em_refinement import (
    EMRefinementLoop,
    PseudoLabel,
    AnswerKey,
    PseudoLabelDataset,
)
from em_loop.src.models import FusionModel, LPAModel


def create_mock_pipeline_callback(num_pairs: int = 100):
    """Create a mock pipeline callback for testing.
    
    This simulates the CoCaD pipeline that would normally run
    with teacher models to generate pseudo-labels.
    """
    def pipeline_callback(
        round_num: int,
        fusion_teacher: FusionModel,
        lpa_teacher: LPAModel,
        student_surrogate=None,
    ) -> AnswerKey:
        """Mock pipeline callback."""
        print(f"  [Mock Pipeline] Round {round_num}: Generating {num_pairs} pseudo-labels...")
        
        labels = []
        config = EMRefinementConfig()
        
        for i in range(num_pairs):
            node_i = i
            node_j = (i + 1) % num_pairs
            
            # Generate random feature vector
            feature_vector = np.random.randn(config.fusion_input_dim).astype(np.float32)
            
            # Use teacher model to get score (simulate pipeline)
            with torch.no_grad():
                device = next(fusion_teacher.parameters()).device
                v_ij_tensor = torch.tensor(feature_vector).unsqueeze(0).to(device)
                final_score = fusion_teacher(v_ij_tensor).item()
            
            # Generate random p-value (lower is better)
            p_value = np.random.uniform(0.0, 1.0)
            
            # Generate path features for LPA
            path_features = np.random.randn(config.lpa_path_feature_dim).astype(np.float32)
            
            # In round 1, include LLM score for distillation
            llm_score = None
            if round_num == 1 and np.random.rand() > 0.5:  # 50% chance
                llm_score = np.random.uniform(0.0, 1.0)
            
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


def test_pseudo_label_dataset():
    """Test PseudoLabelDataset."""
    print("Testing PseudoLabelDataset...")
    
    config = EMRefinementConfig()
    num_labels = 20
    
    labels = []
    for i in range(num_labels):
        label = PseudoLabel(
            node_i=i,
            node_j=(i + 1) % num_labels,
            final_score=np.random.uniform(0.0, 1.0),
            p_value=np.random.uniform(0.0, 1.0),
            feature_vector=np.random.randn(config.fusion_input_dim).astype(np.float32),
            path_features=np.random.randn(config.lpa_path_feature_dim).astype(np.float32),
        )
        labels.append(label)
    
    dataset = PseudoLabelDataset(labels, augment=False, config=config)
    
    assert len(dataset) == num_labels, f"Expected {num_labels} samples, got {len(dataset)}"
    
    sample = dataset[0]
    assert "v_ij" in sample, "Missing v_ij in sample"
    assert "final_score" in sample, "Missing final_score in sample"
    assert "p_value" in sample, "Missing p_value in sample"
    
    print(f"  ✓ Dataset size: {len(dataset)}")
    print(f"  ✓ Sample keys: {list(sample.keys())}")
    print("  ✓ PseudoLabelDataset test passed!\n")


def test_answer_key():
    """Test AnswerKey functionality."""
    print("Testing AnswerKey...")
    
    num_labels = 50
    labels = []
    
    for i in range(num_labels):
        label = PseudoLabel(
            node_i=i,
            node_j=(i + 1) % num_labels,
            final_score=np.random.uniform(0.0, 1.0),
            p_value=np.random.uniform(0.0, 1.0),
            feature_vector=np.random.randn(128).astype(np.float32),
        )
        labels.append(label)
    
    answer_key = AnswerKey(round=1, labels=labels)
    
    assert answer_key.round == 1, "Round number incorrect"
    assert len(answer_key.labels) == num_labels, "Label count incorrect"
    
    # Test high-confidence filtering
    confidence_threshold = 0.7
    high_conf = answer_key.get_high_confidence_labels(confidence_threshold)
    
    # Count how many should be high-confidence
    expected_count = sum(1 for label in labels if (1 - label.p_value) >= confidence_threshold)
    assert len(high_conf) == expected_count, "High-confidence filtering incorrect"
    
    print(f"  ✓ Answer key round: {answer_key.round}")
    print(f"  ✓ Total labels: {len(answer_key.labels)}")
    print(f"  ✓ High-confidence labels (threshold={confidence_threshold}): {len(high_conf)}")
    print("  ✓ AnswerKey test passed!\n")


def test_em_loop_single_round():
    """Test a single round of EM loop."""
    print("Testing EM Loop - Single Round...")
    
    config = EMRefinementConfig()
    config.num_rounds = 1
    config.num_epochs_per_round = 2  # Fewer epochs for testing
    config.batch_size = 16
    
    pipeline_callback = create_mock_pipeline_callback(num_pairs=50)
    
    em_loop = EMRefinementLoop(config, pipeline_callback=pipeline_callback)
    
    # Run one round
    answer_key = em_loop.e_step(round_num=1)
    losses = em_loop.m_step(answer_key, round_num=1)
    em_loop.update_teachers()
    
    assert len(answer_key.labels) > 0, "Should have generated pseudo-labels"
    assert "fusion_loss" in losses, "Should have fusion loss"
    assert "lpa_loss" in losses, "Should have LPA loss"
    
    print(f"  ✓ Generated {len(answer_key.labels)} pseudo-labels")
    print(f"  ✓ Losses: {losses}")
    print("  ✓ Single round test passed!\n")


def test_em_loop_multiple_rounds():
    """Test multiple rounds of EM loop."""
    print("Testing EM Loop - Multiple Rounds...")
    
    config = EMRefinementConfig()
    config.num_rounds = 3
    config.num_epochs_per_round = 2
    config.batch_size = 16
    config.jaccard_threshold = 0.95  # Lower threshold for testing
    
    pipeline_callback = create_mock_pipeline_callback(num_pairs=50)
    
    em_loop = EMRefinementLoop(config, pipeline_callback=pipeline_callback)
    
    answer_key_prev = None
    for round_num in range(1, config.num_rounds + 1):
        answer_key_curr = em_loop.e_step(round_num)
        losses = em_loop.m_step(answer_key_curr, round_num)
        em_loop.update_teachers()
        
        if answer_key_prev is not None:
            jaccard = em_loop.compute_jaccard_similarity(answer_key_prev, answer_key_curr)
            print(f"  Round {round_num}: Jaccard similarity = {jaccard:.4f}")
        
        answer_key_prev = answer_key_curr
    
    print(f"  ✓ Completed {config.num_rounds} rounds")
    print("  ✓ Multiple rounds test passed!\n")


def test_jaccard_similarity():
    """Test Jaccard similarity computation."""
    print("Testing Jaccard Similarity...")
    
    config = EMRefinementConfig()
    config.confidence_threshold = 0.7
    
    # Create two answer keys with overlapping high-confidence labels
    labels1 = []
    labels2 = []
    
    # Create some overlapping pairs
    for i in range(20):
        p_val1 = 0.1 if i < 10 else 0.9  # First 10 are high-confidence
        p_val2 = 0.1 if i < 8 else 0.9  # First 8 are high-confidence (overlap)
        
        label1 = PseudoLabel(
            node_i=i,
            node_j=(i + 1) % 20,
            final_score=0.8,
            p_value=p_val1,
            feature_vector=np.random.randn(128).astype(np.float32),
        )
        labels1.append(label1)
        
        label2 = PseudoLabel(
            node_i=i,
            node_j=(i + 1) % 20,
            final_score=0.8,
            p_value=p_val2,
            feature_vector=np.random.randn(128).astype(np.float32),
        )
        labels2.append(label2)
    
    answer_key1 = AnswerKey(round=1, labels=labels1)
    answer_key2 = AnswerKey(round=2, labels=labels2)
    
    em_loop = EMRefinementLoop(config, pipeline_callback=None)
    jaccard = em_loop.compute_jaccard_similarity(answer_key1, answer_key2)
    
    # Expected: 8 overlapping pairs out of 10 total (union = 10)
    expected_jaccard = 8 / 10
    assert abs(jaccard - expected_jaccard) < 0.01, f"Jaccard should be ~{expected_jaccard}, got {jaccard}"
    
    print(f"  ✓ Jaccard similarity: {jaccard:.4f} (expected ~{expected_jaccard:.4f})")
    print("  ✓ Jaccard similarity test passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing EM-Refinement Loop")
    print("=" * 60 + "\n")
    
    test_pseudo_label_dataset()
    test_answer_key()
    test_jaccard_similarity()
    test_em_loop_single_round()
    test_em_loop_multiple_rounds()
    
    print("=" * 60)
    print("All EM loop tests passed! ✓")
    print("=" * 60)

