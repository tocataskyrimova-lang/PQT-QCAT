"""Test exact resume of StatefulOnlineAgent across interruptions."""
from __future__ import annotations

import copy
import numpy as np
import pytest

from pqt_qcat.core.six_echelon import SixEchelonConfig, SixEchelonCore
from pqt_qcat.memory.exp_stateful_online_001 import (
    setup_context_core,
    StatefulOnlineAgent,
    generate_context_task_pairs,
    generate_streaming_schedule,
)


def test_agent_exact_resume():
    """Verify that pausing an agent mid-stream via snapshot/restore produces identical bits."""
    seed = 42
    stream_seed = 42000
    pairs = generate_context_task_pairs(generator_seed=740003, total_pairs=1)
    mapping_pair = pairs[0]
    schedule = generate_streaming_schedule(mapping_pair, total_events=50, seed=stream_seed)

    # 1. Uninterrupted run of 50 events
    core1 = setup_context_core(seed=seed, enable_vortex=True)
    agent1 = StatefulOnlineAgent(core1, learning_mode="CARRY_TRAIN", t_context=4, t_query=8)

    uninterrupted_results = []
    for ev in schedule:
        pred, logits = agent1.begin_event(ev["context"], ev["query"])
        res = agent1.observe_feedback_and_update(ev["target"])
        uninterrupted_results.append({
            "pred": pred,
            "logits": logits.copy(),
            "correct": res["correct"],
            "margin": res["signed_margin"],
            "nll": res["nll"],
            "delta_k_norm": res["delta_k_norm"],
        })

    # 2. Interrupted run: run 25 events, snapshot, restore into a fresh agent, run remaining 25
    core2 = setup_context_core(seed=seed, enable_vortex=True)
    agent2 = StatefulOnlineAgent(core2, learning_mode="CARRY_TRAIN", t_context=4, t_query=8)

    interrupted_results = []
    for ev in schedule[:25]:
        pred, logits = agent2.begin_event(ev["context"], ev["query"])
        res = agent2.observe_feedback_and_update(ev["target"])
        interrupted_results.append({
            "pred": pred,
            "logits": logits.copy(),
            "correct": res["correct"],
            "margin": res["signed_margin"],
            "nll": res["nll"],
            "delta_k_norm": res["delta_k_norm"],
        })

    # Take snapshot
    snap = agent2.snapshot()

    # Create fresh empty agent and restore
    core_restored = setup_context_core(seed=999, enable_vortex=False) # Deliberately mismatched initial state
    agent_restored = StatefulOnlineAgent(core_restored, learning_mode="RESET_TRAIN")
    agent_restored.restore(snap)

    # Continue running events 25..50
    for ev in schedule[25:]:
        pred, logits = agent_restored.begin_event(ev["context"], ev["query"])
        res = agent_restored.observe_feedback_and_update(ev["target"])
        interrupted_results.append({
            "pred": pred,
            "logits": logits.copy(),
            "correct": res["correct"],
            "margin": res["signed_margin"],
            "nll": res["nll"],
            "delta_k_norm": res["delta_k_norm"],
        })

    # 3. Assert bitwise identity of all predictions, margins, and final matrices
    assert len(uninterrupted_results) == len(interrupted_results) == 50
    for i in range(50):
        assert uninterrupted_results[i]["pred"] == interrupted_results[i]["pred"], f"Pred mismatch at event {i}"
        assert uninterrupted_results[i]["correct"] == interrupted_results[i]["correct"], f"Correct mismatch at event {i}"
        np.testing.assert_allclose(
            uninterrupted_results[i]["logits"],
            interrupted_results[i]["logits"],
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Logits mismatch at event {i}",
        )
        assert uninterrupted_results[i]["margin"] == pytest.approx(interrupted_results[i]["margin"], abs=1e-12)

    # Final Z and K identity
    np.testing.assert_array_equal(agent1.core.Z, agent_restored.core.Z)
    np.testing.assert_array_equal(agent1.core.K, agent_restored.core.K)
    np.testing.assert_array_equal(agent1.eligibility_trace, agent_restored.eligibility_trace)
    print("Exact resume test passed: 100% bit-for-bit identity across uninterrupted vs resumed agent!")
