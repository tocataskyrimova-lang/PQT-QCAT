"""Smoke tests for Stage R3 (exp_stateful_online_001) and Stage R4 (exp_long_life_001)."""
from __future__ import annotations

import numpy as np
import pytest

from pqt_qcat.core.security import FORBIDDEN_FINAL_SEEDS, QUARANTINED_SEEDS
from pqt_qcat.memory.exp_stateful_online_001 import (
    setup_context_core,
    generate_context_task_pairs,
    StatefulOnlineAgent,
    run_joint_learnability_check,
)
from pqt_qcat.memory.exp_long_life_001 import (
    generate_c1_mapping_pool,
    run_long_continuous_life,
)


def test_setup_context_core_interface():
    core = setup_context_core(seed=42, enable_vortex=True)
    assert core.B.shape == (48, 8)
    assert core.B[8, 0] == 8.0 # Context 0 -> E2 unit 8
    assert core.B[9, 7] == 8.0 # Context 1 -> E2 unit 9
    for k in range(1, 7):
        assert core.B[k - 1, k] == 8.0
    assert np.count_nonzero(core.B) == 8


def test_generate_context_task_pairs():
    pairs = generate_context_task_pairs(generator_seed=740003, total_pairs=3)
    assert len(pairs) == 3
    for m0, m1 in pairs:
        assert len(m0) == 6
        assert len(m1) == 6
        assert len(set(m0.values())) == 4
        assert len(set(m1.values())) == 4
        diff = sum(m0[t] != m1[t] for t in range(1, 7))
        assert diff >= 2


def test_stateful_agent_prediction_before_feedback():
    core = setup_context_core(seed=42, enable_vortex=True)
    agent = StatefulOnlineAgent(core, learning_mode="CARRY_TRAIN", t_context=2, t_query=4)
    
    # 1. Prediction commit
    pred, logits = agent.begin_event(context_id=0, query_token=2)
    assert 1 <= pred <= 4
    assert logits.shape == (4,)
    assert agent.event_active is True
    
    # 2. Cannot begin another event before observing feedback
    with pytest.raises(AssertionError):
        agent.observe_feedback_and_update(target_class=5) # Invalid class
        
    # 3. Observe feedback and update
    res = agent.observe_feedback_and_update(target_class=3)
    assert isinstance(res["correct"], bool)
    assert np.isfinite(res["signed_margin"])
    assert res["delta_k_norm"] >= 0.0
    assert agent.event_active is False


def test_stateful_agent_clone_isolation():
    core = setup_context_core(seed=42, enable_vortex=True)
    agent = StatefulOnlineAgent(core, learning_mode="CARRY_TRAIN")
    
    Z_before = agent.core.Z.copy()
    K_before = agent.core.K.copy()
    
    eval_res = agent.evaluate_clone(context_id=1, query_token=3, target_class=2, enable_vortex_read=True)
    assert "correct" in eval_res
    assert "margin" in eval_res
    
    # Ensure clone evaluation did not mutate agent live state
    assert np.array_equal(agent.core.Z, Z_before)
    assert np.array_equal(agent.core.K, K_before)


def test_c1_pool_generation():
    pool = generate_c1_mapping_pool(generator_seed=740004, pool_size=5)
    assert len(pool) == 5
    for m in pool:
        assert len(set(m.values())) == 4
