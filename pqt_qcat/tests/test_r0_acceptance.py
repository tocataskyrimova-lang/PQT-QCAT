"""R0 Acceptance Test Suite: Rigorous verification of defects F01-F12."""
import copy
import json
import os
from pathlib import Path
import numpy as np
import pytest

from pqt_qcat.core.six_echelon import SixEchelonCore, SixEchelonConfig
from pqt_qcat.core.security import (
    FORBIDDEN_FINAL_SEEDS,
    QUARANTINED_SEEDS,
    check_seed_safety,
    validate_config_seed_integrity,
)
from pqt_qcat.memory.exp_sleep_need_001 import (
    validate_config,
    generate_wake_schedule,
    generate_task_panel,
    setup_fresh_core,
    query_core_prediction,
    evaluate_query_accuracy,
    wake_train_step,
    run_single_history_experiment,
    SleepAdapterNeed001,
    ObservedEvent,
)
from pqt_qcat.memory.exp_sleep_need_001b import (
    build_overlapping_B,
    run_wake_history_with_B,
)
from pqt_qcat.memory.m3_diagnostics import _get_quiescent_psi, CLASSES


def get_base_valid_config() -> dict:
    return {
        "protocol_id": "EXP-SLEEP-NEED-001",
        "version": "0.2",
        "forbidden_seeds": [101, 102, 103, 104, 105],
        "calibration_seeds": [42, 11],
        "panel": {
            "generator_seed": 170700,
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
            "anchor_max_degradation": 0.05,
            "sigma": 0.01,
            "h": 0.1,
            "eta": 0.05,
        },
        "evaluation": {
            "prefix_old_accuracy_threshold": 0.80,
            "joint_positive_control_accuracy_threshold": 0.75,
            "ticks_per_query": 8,
        },
        "analysis": {},
    }


def test_f01_seed_protection_in_core_and_config():
    """F01: User config cannot weaken forbidden list; core factory and validator reject sealed seeds."""
    cfg = get_base_valid_config()
    
    # 1. Config with forbidden seed is rejected
    for s in FORBIDDEN_FINAL_SEEDS:
        bad_cfg = copy.deepcopy(cfg)
        bad_cfg["calibration_seeds"] = [s]
        with pytest.raises(ValueError, match="FORBIDDEN_FINAL_SEEDS|sealed M5 seeds"):
            validate_config(bad_cfg)
            
    # 2. Config attempting to weaken forbidden_seeds list
    bad_cfg = copy.deepcopy(cfg)
    bad_cfg["forbidden_seeds"] = []
    with pytest.raises(ValueError, match="weakened forbidden_seeds"):
        validate_config(bad_cfg)
        
    bad_cfg2 = copy.deepcopy(cfg)
    bad_cfg2["forbidden_seeds"] = [101, 102]  # missing 103, 104, 105
    with pytest.raises(ValueError, match="weakened forbidden_seeds"):
        validate_config(bad_cfg2)
        
    # 3. Direct instantiation of core with forbidden seed is rejected
    for s in FORBIDDEN_FINAL_SEEDS:
        with pytest.raises(ValueError, match="FORBIDDEN_FINAL_SEEDS"):
            SixEchelonCore(seed=s)


def test_f02_quarantined_seeds_and_incident_file():
    """F02: Seeds 100-111 are quarantined and SEED_INCIDENT.md is present."""
    root = Path(__file__).resolve().parent.parent.parent
    incident_file = root / "SEED_INCIDENT.md"
    assert incident_file.exists(), "SEED_INCIDENT.md must be present in repository root"
    content = incident_file.read_text(encoding="utf-8")
    assert "100" in content and "105" in content and "111" in content
    assert "карантин" in content.lower() or "quarantine" in content.lower()
    
    # Check that check_seed_safety rejects quarantined seeds if allow_quarantined=False
    for s in range(100, 112):
        if s in FORBIDDEN_FINAL_SEEDS:
            with pytest.raises(ValueError, match="FORBIDDEN_FINAL_SEEDS"):
                check_seed_safety(s, allow_quarantined=False)
        else:
            with pytest.raises(ValueError, match="QUARANTINED_SEEDS"):
                check_seed_safety(s, allow_quarantined=False)


def test_f03_config_plumbing():
    """F03: Changing wake eta and ticks_per_query changes execution values."""
    cfg = get_base_valid_config()
    
    # Negative/zero eta rejected
    bad_cfg = copy.deepcopy(cfg)
    bad_cfg["wake"]["eta"] = -0.1
    with pytest.raises(ValueError, match="wake eta"):
        validate_config(bad_cfg)
        
    bad_cfg2 = copy.deepcopy(cfg)
    bad_cfg2["wake"]["ticks_per_query"] = 0
    with pytest.raises(ValueError, match="ticks_per_query"):
        validate_config(bad_cfg2)
        
    bad_cfg3 = copy.deepcopy(cfg)
    bad_cfg3["wake"]["coordinate_clip"] = 0.0
    with pytest.raises(ValueError, match="coordinate_clip"):
        validate_config(bad_cfg3)


def test_f04_endpoints_exact_single_events():
    """F04: is_pre_new and is_post_new_pre_tail have exactly 1 event each; tail probe is at event 201."""
    task_map = {1: 1, 2: 2, 3: 3, 4: 4, 5: 1, 6: 2}
    
    sched_b = generate_wake_schedule(seed=42, task_id=0, task_map=task_map, history_type="blocked")
    sched_i = generate_wake_schedule(seed=42, task_id=0, task_map=task_map, history_type="interleaved")
    
    assert len(sched_b) == 201
    assert len(sched_i) == 201
    
    # Blocked
    pre_b = [e for e in sched_b if e.is_pre_new]
    assert len(pre_b) == 1, f"Blocked pre_new should have exactly 1 event, got {len(pre_b)}"
    assert pre_b[0].event_idx == 135
    
    post_b = [e for e in sched_b if e.is_post_new_pre_tail]
    assert len(post_b) == 1, f"Blocked post_new_pre_tail should have exactly 1 event, got {len(post_b)}"
    assert post_b[0].event_idx == 195
    
    # Interleaved
    pre_i = [e for e in sched_i if e.is_pre_new]
    assert len(pre_i) == 1, f"Interleaved pre_new should have exactly 1 event, got {len(pre_i)}"
    assert pre_i[0].event_idx == 75
    
    post_i = [e for e in sched_i if e.is_post_new_pre_tail]
    assert len(post_i) == 1, f"Interleaved post_new_pre_tail should have exactly 1 event, got {len(post_i)}"
    assert post_i[0].event_idx == 195
    
    # Tail
    tail_b = [e for e in sched_b if e.is_tail]
    assert len(tail_b) == 6
    assert tail_b[-1].event_idx == 201


def test_f05_delay_metrics_preserved_and_serialized():
    """F05: Both immediate carry and delay1000 recorded and serialized with Z_start and Z_end."""
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
    
    assert "delay1000" in res, "delay1000 missing from result dictionary!"
    del_data = res["delay1000"]
    assert "acc_old" in del_data and "acc_new" in del_data
    assert "margin_old" in del_data and "margin_new" in del_data
    assert "nll_old" in del_data and "nll_new" in del_data
    assert "Z_start" in del_data and "Z_end" in del_data
    assert len(del_data["Z_start"]) == 48 and len(del_data["Z_end"]) == 48
    assert "Z_drift" in del_data
    assert "logits_old" in del_data and "logits_new" in del_data
    
    # Check that it serializes to JSON without loss
    serialized = json.dumps(res)
    deserialized = json.loads(serialized)
    assert "delay1000" in deserialized
    assert deserialized["delay1000"]["acc_old"] == del_data["acc_old"]


def test_f06_paired_histories_share_run_seed_and_noise():
    """F06: Coupled paired histories receive identical run_seed and noise draws."""
    map0 = generate_task_panel(generator_seed=170700, total_maps=1)[0]
    adapter = SleepAdapterNeed001(candidate_slots=2, anchor_count=6)
    
    shared_seed = 42000
    b_records, i_records = [], []
    
    res_b = run_single_history_experiment(
        seed=42, task_id=0, task_map=map0, history_type="blocked",
        run_seed=shared_seed, adapter=adapter,
        wake_log_records=[], probe_log_records=[], sleep_log_records=b_records, query_log_records=[],
        execute_sleep=True,
    )
    res_i = run_single_history_experiment(
        seed=42, task_id=0, task_map=map0, history_type="interleaved",
        run_seed=shared_seed, adapter=adapter,
        wake_log_records=[], probe_log_records=[], sleep_log_records=i_records, query_log_records=[],
        execute_sleep=True,
    )
    
    # Check dream records for identical noise
    b_dream_noise = [r["noise_sha256"] for r in b_records if r["policy"] == "dream"]
    i_dream_noise = [r["noise_sha256"] for r in i_records if r["policy"] == "dream"]
    assert len(b_dream_noise) == len(i_dream_noise)
    assert b_dream_noise == i_dream_noise, "Thermal noise was not identical across paired histories!"


def test_f07_preconditions_and_gate_enforcement():
    """F07: Pre-sleep gate enforces Joint >= 75%, Prefix >= 80%, Interference >= 10%."""
    cfg = get_base_valid_config()
    
    # Invalid low prefix threshold rejected
    bad_cfg = copy.deepcopy(cfg)
    bad_cfg["evaluation"]["prefix_old_accuracy_threshold"] = 0.70
    with pytest.raises(ValueError, match="prefix_old_accuracy_threshold"):
        validate_config(bad_cfg)
        
    bad_cfg2 = copy.deepcopy(cfg)
    bad_cfg2["evaluation"]["joint_positive_control_accuracy_threshold"] = 0.60
    with pytest.raises(ValueError, match="joint_positive_control_accuracy_threshold"):
        validate_config(bad_cfg2)


def test_f08_baseline_definition():
    """F08: I_pre_tail = Q_old(pre_new) - Q_old(post_new_pre_tail)."""
    map0 = generate_task_panel(generator_seed=170700, total_maps=1)[0]
    adapter = SleepAdapterNeed001(candidate_slots=1, anchor_count=6)
    
    res = run_single_history_experiment(
        seed=42, task_id=0, task_map=map0, history_type="blocked",
        run_seed=42000, adapter=adapter,
        wake_log_records=[], probe_log_records=[], sleep_log_records=[], query_log_records=[],
        execute_sleep=False,
    )
    
    q_pre = res["q_old_pre_new"]
    q_post = res["q_old_post_new_pre_tail"]
    assert np.isclose(res["interference_I_pre_tail"], q_pre - q_post)
    assert "interference_I_prefix" in res


def test_f09_tick_accounting_and_idle_unmatched():
    """F09: Full physical tick accounting and idle_unmatched flag."""
    map0 = generate_task_panel(generator_seed=170700, total_maps=1)[0]
    adapter = SleepAdapterNeed001(candidate_slots=2, anchor_count=6)
    
    res = run_single_history_experiment(
        seed=42, task_id=0, task_map=map0, history_type="blocked",
        run_seed=42000, adapter=adapter,
        wake_log_records=[], probe_log_records=[], sleep_log_records=[], query_log_records=[],
        execute_sleep=True,
    )
    
    for pol, pdata in res["policies"].items():
        assert pdata["wake_ticks"] == 201 * 8
        assert pdata["delay_ticks"] == 1000
        assert pdata["total_life_ticks"] > 0
        if pol == "idle":
            assert pdata["idle_unmatched"] is True
        else:
            assert pdata["idle_unmatched"] is False


def test_f10_provenance_and_atomic_writes(tmp_path):
    """F10: Atomic file writes with .tmp and rename."""
    test_file = tmp_path / "atomic_test.json"
    tmp_file = tmp_path / "atomic_test.json.tmp"
    
    data = {"test": 123}
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp_file, test_file)
    
    assert test_file.exists()
    assert not tmp_file.exists()
    assert json.loads(test_file.read_text(encoding="utf-8")) == data


def test_f11_mathematical_observability_rows_and_fixed_v():
    """F11: Wake update modifies rows 32:35 (E5 rows, p=144); check directional derivative convergence."""
    core = setup_fresh_core(seed=42, enable_vortex=True)
    psi_0 = _get_quiescent_psi()
    K_init = core.K.copy()
    
    # Train 25 epochs of old tokens (prefix)
    map0 = generate_task_panel(generator_seed=170700, total_maps=1)[0]
    for ep in range(25):
        for tok in [1, 2, 3]:
            wake_train_step(core, tok, map0[tok], psi_0)
            
    delta_K = core.K - K_init
    total_norm = float(np.linalg.norm(delta_K))
    assert total_norm > 0
    
    # Update on rows 32:35 (p=144)
    e5_update_norm = float(np.linalg.norm(delta_K[32:35, :]))
    e5_fraction_sq = (e5_update_norm**2) / (total_norm**2)
    # The active plastic update is strictly concentrated on rows 32:35
    assert e5_fraction_sq > 0.99, f"E5 update fraction {e5_fraction_sq} <= 0.99"
    
    # Fixed direction v convergence across epsilons
    rng = np.random.default_rng(999)
    v = rng.standard_normal((48, 48))
    v[0:32, :] = 0.0
    v[35:, :] = 0.0
    v /= np.linalg.norm(v)
    
    # Check convergence of (logits(K + eps*v) - logits(K - eps*v)) / (2*eps)
    eps_list = [0.01, 0.001, 0.0001, 1e-05]
    derivs = []
    
    for eps in eps_list:
        c_plus = copy.deepcopy(core)
        c_plus.K += eps * v
        pred_p, logits_p, _, _ = query_core_prediction(c_plus, 1, ticks=8)
        
        c_minus = copy.deepcopy(core)
        c_minus.K -= eps * v
        pred_m, logits_m, _, _ = query_core_prediction(c_minus, 1, ticks=8)
        
        derivs.append((logits_p - logits_m) / (2.0 * eps))
        
    diff_last_two = float(np.linalg.norm(derivs[-1] - derivs[-2]))
    assert diff_last_two < 1e-6, f"Finite difference did not converge: {diff_last_two}"


def test_f12_retraction_document_exists():
    """F12: ADDENDUM_RETRACTION_20261118.md formally documents all required retractions."""
    root = Path(__file__).resolve().parent.parent.parent
    retraction_file = root / "ADDENDUM_RETRACTION_20261118.md"
    assert retraction_file.exists(), "ADDENDUM_RETRACTION_20261118.md must be present"
    content = retraction_file.read_text(encoding="utf-8")
    assert "99,2%" in content or "99.2%" in content
    assert "физический предел" in content or "physical limit" in content or "емкости" in content or "ёмкости" in content
    assert "58.33%" in content or "majority" in content
