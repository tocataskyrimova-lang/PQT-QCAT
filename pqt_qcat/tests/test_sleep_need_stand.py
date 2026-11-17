"""Unit tests and invariant verifications for EXP-SLEEP-NEED-001 stand."""
import numpy as np
import pytest

from pqt_qcat.core.six_echelon import SixEchelonCore
from pqt_qcat.memory.exp_sleep_need_001 import (
    generate_task_panel,
    setup_fresh_core,
    query_core_prediction,
    evaluate_query_accuracy,
    wake_train_step,
    SleepAdapterNeed001,
    ObservedEvent,
)
from pqt_qcat.memory.m3_diagnostics import _get_quiescent_psi


def test_panel_generator_properties():
    """Verify panel generation determinism, distinctness, and conditioning on all 4 classes."""
    maps = generate_task_panel(generator_seed=170700, total_maps=12)
    assert len(maps) == 12, "Should generate exactly 12 mappings"
    
    seen = set()
    for idx, m in enumerate(maps):
        # All 6 tokens present
        assert set(m.keys()) == {1, 2, 3, 4, 5, 6}
        # All 4 classes present
        classes_present = set(m.values())
        assert classes_present == {1, 2, 3, 4}, f"Map {idx} missing classes: {classes_present}"
        
        sig = tuple(m[x] for x in range(1, 7))
        assert sig not in seen, f"Duplicate mapping found at index {idx}"
        seen.add(sig)
        
    # Test determinism
    maps2 = generate_task_panel(generator_seed=170700, total_maps=12)
    for m1, m2 in zip(maps, maps2):
        assert m1 == m2


def test_b_w_bitwise_invariance():
    """Verify B and W matrices are bitwise invariant across wake steps and sleep."""
    core = setup_fresh_core(seed=42, enable_vortex=True)
    B_init = core.B.copy()
    W_init = core.W_readout.copy()
    
    psi_0 = _get_quiescent_psi()
    
    # Run 10 wake training steps
    for step in range(10):
        tok = (step % 6) + 1
        target = ((step + 1) % 4) + 1
        wake_train_step(core, tok, target, psi_0)
        
    assert np.array_equal(core.B, B_init), "B matrix modified during wake training!"
    assert np.array_equal(core.W_readout, W_init), "W_readout matrix modified during wake training!"
    
    # Run sleep session
    adapter = SleepAdapterNeed001(candidate_slots=5, anchor_count=6)
    obs = [ObservedEvent(1, 1, 2, 1, "test"), ObservedEvent(2, 2, 3, 1, "test")]
    anchors = {1: 2, 2: 3, 3: 4, 4: 1, 5: 2, 6: 3}
    
    adapter.run_sleep_session("dream", core, obs, anchors, run_seed=123, sleep_log_records=[])
    
    assert np.array_equal(core.B, B_init), "B matrix modified during sleep session!"
    assert np.array_equal(core.W_readout, W_init), "W_readout matrix modified during sleep session!"


def test_diagnostic_probes_do_not_mutate():
    """Verify diagnostic probing does not mutate the core's state or parameters."""
    core = setup_fresh_core(seed=42, enable_vortex=True)
    psi_0 = _get_quiescent_psi()
    wake_train_step(core, 1, 2, psi_0)
    
    K_before = core.K.copy()
    Z_before = core.Z.copy()
    
    # Evaluate queries
    task_map = {1: 2, 2: 3, 3: 4, 4: 1, 5: 2, 6: 3}
    acc_o, _, _ = evaluate_query_accuracy(core, [1, 2, 3], task_map)
    acc_n, _, _ = evaluate_query_accuracy(core, [4, 5, 6], task_map)
    
    assert np.array_equal(core.K, K_before), "K matrix modified during evaluation!"
    # Note: query resets Z, but K and weights must be completely untouched


def test_exact_compute_budgets():
    """Verify sleep session strictly adheres to 1000 active ticks and 2400 shadow ticks."""
    core = setup_fresh_core(seed=42, enable_vortex=True)
    adapter = SleepAdapterNeed001(candidate_slots=25, anchor_count=6)
    obs = [ObservedEvent(x, x, ((x % 4) + 1), 1, "test") for x in range(1, 7)]
    anchors = {x: ((x % 4) + 1) for x in range(1, 7)}
    
    for pol in ["dream", "replay", "sham_dream"]:
        c = SixEchelonCore(config=core.config)
        c.restore(core.snapshot())
        res = adapter.run_sleep_session(pol, c, obs, anchors, run_seed=42, sleep_log_records=[])
        assert res["active_ticks"] == 1000, f"{pol} active ticks {res['active_ticks']} != 1000"
        assert res["shadow_ticks"] == 2400, f"{pol} shadow ticks {res['shadow_ticks']} != 2400"
        
    # Idle policy
    c = SixEchelonCore(config=core.config)
    c.restore(core.snapshot())
    res_idle = adapter.run_sleep_session("idle", c, obs, anchors, run_seed=42, sleep_log_records=[])
    assert res_idle["active_ticks"] == 1000, f"idle active ticks {res_idle['active_ticks']} != 1000"


def test_sham_policy_cancellation():
    """Verify sham_dream cancels proposals, leaving K bitwise identical to pre-sleep."""
    core = setup_fresh_core(seed=42, enable_vortex=True)
    K_before = core.K.copy()
    
    adapter = SleepAdapterNeed001(candidate_slots=25, anchor_count=6)
    obs = [ObservedEvent(x, x, ((x % 4) + 1), 1, "test") for x in range(1, 7)]
    anchors = {x: ((x % 4) + 1) for x in range(1, 7)}
    
    logs = []
    res = adapter.run_sleep_session("sham_dream", core, obs, anchors, run_seed=42, sleep_log_records=logs)
    
    assert np.array_equal(core.K, K_before), "sham_dream modified K!"
    assert res["accepted_count"] == 0, "sham_dream accepted updates!"
    assert res["delta_K_norm"] == 0.0, "sham_dream delta_K_norm != 0"
    
    # Check that proposals had policy_cancel or rejected
    assert all(r["rejection_reason"] in ("policy_cancel", "provenance_or_contradiction", "structural_bounds", "anchor_guard_rollback") for r in logs)


def test_rollback_exactness():
    """Verify rollback K_post -> K_pre -> K_post exactly restores logits and predictions."""
    core = setup_fresh_core(seed=42, enable_vortex=True)
    K_pre = core.K.copy()
    
    adapter = SleepAdapterNeed001(candidate_slots=25, anchor_count=6)
    obs = [ObservedEvent(x, x, ((x % 4) + 1), 1, "test") for x in range(1, 7)]
    anchors = {x: ((x % 4) + 1) for x in range(1, 7)}
    
    res = adapter.run_sleep_session("dream", core, obs, anchors, run_seed=42, sleep_log_records=[])
    K_post = core.K.copy()
    
    pred_post, logits_post, _, _ = query_core_prediction(core, 1)
    
    # Rollback to pre
    core.K = K_pre.copy()
    pred_pre, logits_pre, _, _ = query_core_prediction(core, 1)
    
    # Restore to post
    core.K = K_post.copy()
    pred_restored, logits_restored, _, _ = query_core_prediction(core, 1)
    
    assert np.array_equal(logits_post, logits_restored), "Logits did not match bit-for-bit after rollback restoration!"
    assert pred_post == pred_restored
