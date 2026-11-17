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
    assert np.array_equal(core.Z, Z_before), "Evaluation changed the live Z"


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


def test_carry_query_preserves_and_uses_state():
    import copy
    core = setup_fresh_core(42)
    core.Z[32:35] = [0.7, -0.3, 0.1]
    snapshot = copy.deepcopy(core.snapshot())
    manual = copy.deepcopy(core)
    x = np.zeros(8)
    x[1] = 1
    for _ in range(8):
        manual.step(x, write_gate=1)
    carried = query_core_prediction(core, 1, reset_state=False)[1]
    reset = query_core_prediction(core, 1, reset_state=True)[1]
    assert np.array_equal(carried, manual.predict()[1:5])
    assert not np.array_equal(carried, reset)
    for name in ('Z', 'K', 'B', 'W_readout'):
        assert np.array_equal(getattr(core, name), snapshot[name])
    assert core.rng.bit_generator.state == snapshot['rng_state']
    # Each token begins from the same snapshot, irrespective of query order.
    before = query_core_prediction(core, 2, reset_state=False)[1]
    query_core_prediction(core, 1, reset_state=False)
    assert np.array_equal(before, query_core_prediction(core, 2, reset_state=False)[1])


def test_target_margin_is_negative_for_wrong_answer():
    core = setup_fresh_core(42)
    predicted, logits, _, _ = query_core_prediction(core, 1)
    wrong_target = predicted % 4 + 1
    accuracy, margin, _ = evaluate_query_accuracy(core, [1], {1: wrong_target})
    assert accuracy == 0
    assert margin < 0
    assert margin == logits[wrong_target - 1] - max(np.delete(logits, wrong_target - 1))


def test_instrumented_budget_and_paired_randomness(monkeypatch):
    import copy
    calls = 0
    original = SixEchelonCore.step
    def counted(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)
    monkeypatch.setattr(SixEchelonCore, 'step', counted)
    base = setup_fresh_core(42)
    events = [ObservedEvent(t, t, t % 4 + 1, 1, 'test') for t in range(1, 7)]
    anchors = {e.token_in: e.target_token for e in events}
    adapter = SleepAdapterNeed001(candidate_slots=2)
    fingerprints = []
    for policy in ('dream', 'replay', 'sham_dream', 'idle'):
        calls = 0
        records = []
        core = copy.deepcopy(base)
        metrics = adapter.run_sleep_session(policy, core, events, anchors, 987, records)
        assert metrics['active_ticks'] == 1000
        assert metrics['diagnostic_ticks'] == 96
        assert calls == metrics['total_compute_ticks']
        assert calls == (1096 if policy == 'idle' else 1288)
        if policy != 'idle':
            fingerprints.append([(r['grounding_event_id'], r['noise_sha256']) for r in records])
        else:
            manual = copy.deepcopy(base)
            for _ in range(1000):
                original(manual, x_t=None, write_gate=0)
            assert np.array_equal(core.Z, manual.Z), 'Anchor checks changed idle dynamics'
    assert fingerprints[0] == fingerprints[1] == fingerprints[2]


def test_history_integration_pairs_policies_and_saves_actual_state(tmp_path):
    import json
    from pqt_qcat.memory.exp_sleep_need_001 import run_single_history_experiment
    class CapturingAdapter(SleepAdapterNeed001):
        def __init__(self):
            super().__init__(candidate_slots=1)
            self.end_states = {}
        def run_sleep_session(self, **kwargs):
            result = super().run_sleep_session(**kwargs)
            self.end_states[kwargs['policy']] = kwargs['core'].Z.copy()
            return result
    adapter = CapturingAdapter()
    logs = []
    result = run_single_history_experiment(42, 0, generate_task_panel()[0], 'blocked',
        12345, adapter, [], [], logs, [], tmp_path)
    assert all(r['rollback_exact'] for r in result['policies'].values())
    assert {r['run_seed'] for r in logs} == {12345}
    assert len({(r['grounding_event_id'], r['noise_sha256']) for r in logs}) == 1
    for policy, expected_z in adapter.end_states.items():
        with np.load(tmp_path / f'post_seed42_task0_blocked_{policy}.npz', allow_pickle=False) as data:
            assert np.array_equal(data['Z'], expected_z)
            metadata = json.loads(str(data['metadata_json']))
            assert metadata['run_seed'] == 12345
            assert len(metadata['observed_events']) == 201


def test_sealed_seeds_rejected_before_output(tmp_path):
    from pqt_qcat.memory.exp_sleep_need_001 import run_stage_experiment
    output = tmp_path / 'forbidden'
    with pytest.raises(ValueError, match='sealed'):
        run_stage_experiment('calibration', [101], generate_task_panel()[:1], output, {})
    assert not output.exists()


def test_config_validation_schema_and_sealed_seeds():
    import copy
    from pqt_qcat.memory.exp_sleep_need_001 import validate_config
    valid_cfg = {
        "protocol_id": "EXP-SLEEP-NEED-001",
        "version": "0.1",
        "calibration_seeds": [42, 11],
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
        },
        "sleep": {
            "candidate_slots": 25,
            "active_ticks": 1000,
            "anchor_count": 6,
            "eta": 0.05,
        },
        "evaluation": {},
        "analysis": {},
    }
    validate_config(valid_cfg)

    # Missing section
    bad_cfg = copy.deepcopy(valid_cfg)
    del bad_cfg["sleep"]
    with pytest.raises(ValueError, match="Missing required configuration section"):
        validate_config(bad_cfg)

    # Sealed seed
    bad_cfg2 = copy.deepcopy(valid_cfg)
    bad_cfg2["calibration_seeds"] = [101]
    with pytest.raises(ValueError, match="sealed M5 seeds"):
        validate_config(bad_cfg2)

    # run_m5 True
    bad_cfg3 = copy.deepcopy(valid_cfg)
    bad_cfg3["run_m5"] = True
    with pytest.raises(ValueError, match="run_m5 must be False"):
        validate_config(bad_cfg3)

    # invalid slots
    bad_cfg4 = copy.deepcopy(valid_cfg)
    bad_cfg4["sleep"]["candidate_slots"] = 100
    with pytest.raises(ValueError, match="candidate_slots"):
        validate_config(bad_cfg4)


def test_schedule_generator_decoupling_and_endpoints():
    from pqt_qcat.memory.exp_sleep_need_001 import generate_wake_schedule
    task_map = {1: 1, 2: 2, 3: 3, 4: 4, 5: 1, 6: 2}
    
    sched_b = generate_wake_schedule(seed=42, task_id=0, task_map=task_map, history_type="blocked")
    sched_i = generate_wake_schedule(seed=42, task_id=0, task_map=task_map, history_type="interleaved")
    
    assert len(sched_b) == 201
    assert len(sched_i) == 201
    
    # Check identical multiset of events
    tokens_b = sorted([e.token_in for e in sched_b])
    tokens_i = sorted([e.token_in for e in sched_i])
    assert tokens_b == tokens_i, "BLOCKED and INTERLEAVED have different event multisets!"
    
    # Check endpoints in BLOCKED
    pre_new_events = [e for e in sched_b if e.is_pre_new]
    assert len(pre_new_events) == 3
    assert all(e.epoch_idx == 45 for e in pre_new_events)
    
    post_new_events = [e for e in sched_b if e.is_post_new_pre_tail]
    assert len(post_new_events) == 3
    assert all(e.epoch_idx == 65 for e in post_new_events)
    
    tail_events = [e for e in sched_b if e.is_tail]
    assert len(tail_events) == 6


def test_joint_positive_control_execution():
    from pqt_qcat.memory.exp_sleep_need_001 import evaluate_joint_positive_control
    maps = generate_task_panel(generator_seed=170700, total_maps=2)
    res = evaluate_joint_positive_control(seeds=[11, 42], panel_maps=maps, epochs=15)
    assert "mean_joint_accuracy" in res
    assert res["mean_joint_accuracy"] >= 0.70
    assert "ci_95" in res


def test_semantic_grounding_matching_across_histories():
    import copy
    from pqt_qcat.memory.exp_sleep_need_001 import (
        SleepAdapterNeed001, ObservedEvent, setup_fresh_core
    )
    adapter = SleepAdapterNeed001(candidate_slots=10, anchor_count=6)
    core = setup_fresh_core(seed=42)
    anchors = {x: ((x % 4) + 1) for x in range(1, 7)}
    
    # Create observed events in two different orders (simulating BLOCKED vs INTERLEAVED)
    events_1 = [ObservedEvent(i, (i % 6) + 1, ((i % 4) + 1), i, "grp1") for i in range(30)]
    events_2 = list(reversed(events_1))
    
    logs_1, logs_2 = [], []
    c1 = copy.deepcopy(core)
    c2 = copy.deepcopy(core)
    
    adapter.run_sleep_session("dream", c1, events_1, anchors, run_seed=777, sleep_log_records=logs_1)
    adapter.run_sleep_session("dream", c2, events_2, anchors, run_seed=777, sleep_log_records=logs_2)
    
    tokens_1 = [(r["slot"], r["token_in"], r["target_token"]) for r in logs_1]
    tokens_2 = [(r["slot"], r["token_in"], r["target_token"]) for r in logs_2]
    assert tokens_1 == tokens_2, "Semantic grounding matching failed across different presentation orders!"


def test_separate_compute_accounting():
    from pqt_qcat.memory.exp_sleep_need_001 import run_single_history_experiment, SleepAdapterNeed001
    adapter = SleepAdapterNeed001(candidate_slots=5, anchor_count=6)
    task_map = {1: 1, 2: 2, 3: 3, 4: 4, 5: 1, 6: 2}
    
    res = run_single_history_experiment(
        seed=42, task_id=0, task_map=task_map, history_type="blocked",
        run_seed=123, adapter=adapter,
        wake_log_records=[], probe_log_records=[], sleep_log_records=[], query_log_records=[],
        save_snapshots_dir=None, execute_sleep=True,
    )
    
    pol_dr = res["policies"]["dream"]
    assert pol_dr["wake_ticks"] == 201 * 8
    assert pol_dr["guard_ticks"] == 5 * 2 * 6 * 8
    assert pol_dr["sleep_active_ticks"] == 1000
    assert pol_dr["delay_ticks"] == 1000
    assert pol_dr["rollback_ticks"] == 96
    assert pol_dr["total_life_ticks"] > pol_dr["sleep_session_compute_ticks"]

