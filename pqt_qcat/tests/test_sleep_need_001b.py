"""Unit tests for EXP-SLEEP-NEED-001B stand and overlapping B matrices."""
import copy
import numpy as np
import pytest

from pqt_qcat.memory.exp_sleep_need_001b import (
    build_overlapping_B,
    setup_core_with_B,
    evaluate_joint_learnability,
    run_wake_history_with_B,
)
from pqt_qcat.memory.exp_sleep_need_001 import generate_task_panel, FORBIDDEN_SEEDS


def test_b_overlap_norm_and_orthogonality():
    """Verify B matrix construction preserves equal norm 8.0 and exact cosine similarities."""
    for rho in [0.0, 0.25, 0.50, 0.70, 0.80]:
        B = build_overlapping_B(rho=rho, norm_B=8.0)
        assert B.shape == (48, 8)
        
        # Norms
        for k in range(1, 7):
            norm_k = float(np.linalg.norm(B[:, k]))
            assert np.isclose(norm_k, 8.0, atol=1e-12), f"Token {k} norm {norm_k} != 8.0 for rho={rho}"
            
        # Cosine similarities
        for k in range(1, 4):
            cos_sim = float(np.dot(B[:, k], B[:, k + 3]) / (8.0 * 8.0))
            assert np.isclose(cos_sim, rho, atol=1e-12), f"Cosine sim between {k} and {k+3} is {cos_sim} != {rho}"
            
        # Non-paired tokens must be orthogonal
        for k1 in range(1, 7):
            for k2 in range(1, 7):
                if k1 != k2 and abs(k1 - k2) != 3:
                    dot = float(np.dot(B[:, k1], B[:, k2]))
                    assert np.isclose(dot, 0.0, atol=1e-12), f"Tokens {k1} and {k2} not orthogonal: dot={dot}"


def test_forbidden_seeds_rejected_001b():
    """Verify sealed seeds 101-105 are rejected before any execution on a full valid schema."""
    valid_base = {
        "protocol_id": "EXP-SLEEP-NEED-001",
        "version": "0.1",
        "forbidden_seeds": [101, 102, 103, 104, 105],
        "calibration_seeds": [42],
        "panel": {
            "total_distinct_mappings": 6,
            "old_sources": [1, 2, 3],
            "new_sources": [4, 5, 6],
        },
        "wake": {
            "events_per_life": 201,
            "prefix_old_epochs": 25,
            "day_epochs_per_group": 20,
            "tail_epochs_per_group": 1,
            "eta": 0.8,
            "ticks_per_query": 8,
            "coordinate_clip": 0.5,
        },
        "sleep": {
            "candidate_slots": 25,
            "active_ticks": 1000,
            "anchor_count": 6,
            "eta": 0.05,
        },
        "evaluation": {
            "prefix_old_accuracy_threshold": 0.80,
            "joint_positive_control_accuracy_threshold": 0.75,
            "ticks_per_query": 8,
        },
        "analysis": {},
    }
    for s in FORBIDDEN_SEEDS:
        bad_cfg = copy.deepcopy(valid_base)
        bad_cfg["calibration_seeds"] = [s]
        with pytest.raises(ValueError, match="FORBIDDEN_FINAL_SEEDS|sealed M5 seeds"):
            from pqt_qcat.memory.exp_sleep_need_001 import validate_config
            validate_config(bad_cfg)


def test_evaluation_modes_structural_reset_vs_carry():
    """Verify structural reset and carry state behave differently and evaluate accurately."""
    B = build_overlapping_B(rho=0.25, norm_B=8.0)
    task_map = {1: 1, 2: 2, 3: 3, 4: 4, 5: 1, 6: 2}
    
    res = run_wake_history_with_B(
        seed=42,
        task_id=0,
        task_map=task_map,
        history_type="blocked",
        B_matrix=B,
        run_seed=12345,
        eval_modes=["structural_reset", "carry_state"],
    )
    
    assert "structural_reset" in res["modes"]
    assert "carry_state" in res["modes"]
    sr = res["modes"]["structural_reset"]
    cs = res["modes"]["carry_state"]
    
    assert "I_pre_tail" in sr
    assert "I_pre_tail" in cs
    assert sr["endpoints"]["pre_new"]["acc_old"] >= 0.80
