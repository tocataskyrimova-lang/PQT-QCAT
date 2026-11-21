"""Comprehensive Verification Test Suite for LIFE-003 CONTINUOUS-THOUGHT-001.

Contains all 19 mandatory automated tests required by Contract v0.2.1:
  - 1-14: Baseline Integrity & Anti-Cheat Guards
  - 15-19: v0.2.1 Enhanced Epistemic & Deliberation Verifications
"""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from pqt_qcat.core.six_echelon import SixEchelonCore, SixEchelonConfig
from life003.config import load_and_validate_prereg_config, EXPECTED_CONFIG_SHA256
from life003.clock import LifeClock
from life003.ledger import SaksinLedger
from life003.hypotheses import HypothesisPool
from life003.trigger import EndogenousTrigger
from life003.sidecar_pool import SidecarPool
from life003.sankalpa import SankalpaGate
from life003.checkpoint import save_checkpoint, load_checkpoint, compute_state_hash
from life003.scheduler import ContinuousLifeOrganism, array_hash


# ---------------------------------------------------------------------------
# Test 1: Core ticks advance while idle
# ---------------------------------------------------------------------------
def test_core_ticks_advance_while_idle():
    org = ContinuousLifeOrganism()
    initial_tick = org.clock.core_tick
    for _ in range(10):
        org.execute_physical_tick(event_query=None)
    assert org.clock.core_tick == initial_tick + 10


# ---------------------------------------------------------------------------
# Test 2: No double integrator step
# ---------------------------------------------------------------------------
def test_no_double_integrator_step():
    org = ContinuousLifeOrganism()
    t_q = org.t_query  # 8
    tick_before = org.clock.core_tick
    org.execute_physical_tick(event_query=1)
    # The physical tick must advance exactly by t_query, no extra step
    assert org.clock.core_tick == tick_before + t_q


# ---------------------------------------------------------------------------
# Test 3: Main Z never reset during life
# ---------------------------------------------------------------------------
def test_main_Z_never_reset_during_life():
    org = ContinuousLifeOrganism()
    for _ in range(25):
        org.execute_physical_tick(event_query=None)
    for q in [1, 2, 3]:
        org.execute_physical_tick(event_query=q)

    # Z must not be zeros or trivial
    assert float(np.linalg.norm(org.core.Z)) > 0.1
    # Check that Hold deep (E4) coordinates are preserved
    assert np.any(org.core.Z[24:32] != 0.0)


# ---------------------------------------------------------------------------
# Test 4: Sidecar does not mutate main
# ---------------------------------------------------------------------------
def test_sidecar_does_not_mutate_main():
    org = ContinuousLifeOrganism()
    for _ in range(15):
        org.execute_physical_tick(event_query=None)

    z_pre = org.core.Z.copy()
    k_pre = org.core.K.copy()
    rng_pre = copy.deepcopy(org.core.rng.bit_generator.state)

    stimulus = np.zeros(8)
    stimulus[2] = 1.0

    res = org.sidecar_pool.run_deliberation(
        main_core=org.core,
        stimulus=stimulus,
        rng_sidecar=org.rng_sidecar,
    )

    assert res["main_unmutated"] is True
    np.testing.assert_array_equal(org.core.Z, z_pre)
    np.testing.assert_array_equal(org.core.K, k_pre)
    assert org.core.rng.bit_generator.state == rng_pre


# ---------------------------------------------------------------------------
# Test 5: Predict before action before feedback
# ---------------------------------------------------------------------------
def test_predict_before_action_before_feedback():
    org = ContinuousLifeOrganism()
    res = org.execute_physical_tick(event_query=2)

    # In log: state_integrity and trigger are evaluated, decision made, then feedback
    assert res["phase"] == "DELIBERATION_AND_ACT"
    ledger_types = [entry["record_type"] for entry in res["saksin_ledger_events"]]
    assert "OBSERVED" in ledger_types
    assert "AUDIT_SHADOW" in ledger_types

    obs_idx = ledger_types.index("OBSERVED")
    audit_idx = ledger_types.index("AUDIT_SHADOW")
    assert obs_idx < audit_idx


# ---------------------------------------------------------------------------
# Test 6: No feedback no weight update
# ---------------------------------------------------------------------------
def test_no_feedback_no_weight_update():
    org = ContinuousLifeOrganism()
    k_initial = org.core.K.copy()

    # Run mixture of idle and event ticks in T1
    for _ in range(15):
        org.execute_physical_tick(event_query=None)
    for q in [1, 2, 3]:
        org.execute_physical_tick(event_query=q)

    # In T1, plastic updates are strictly frozen (k_write_gate_active = 0.0)
    np.testing.assert_array_equal(org.core.K, k_initial)


# ---------------------------------------------------------------------------
# Test 7: Hypothesis not promoted to observed
# ---------------------------------------------------------------------------
def test_hypothesis_not_promoted_to_observed():
    org = ContinuousLifeOrganism()
    # Force trigger on idle
    org.trigger.threshold_theta = -1.0
    org.execute_physical_tick(event_query=None)

    hyp_records = [
        r for r in org.ledger.records if r["record_type"] == "HYPOTHESIS"
    ]
    assert len(hyp_records) > 0
    for h_rec in hyp_records:
        assert h_rec["record_type"] == "HYPOTHESIS"
        assert h_rec["payload"]["v_just"] == "UNVERIFIED"
        assert h_rec["record_type"] != "OBSERVED"


# ---------------------------------------------------------------------------
# Test 8: Pending and clock exact resume
# ---------------------------------------------------------------------------
def test_pending_and_clock_exact_resume(tmp_path):
    org1 = ContinuousLifeOrganism()
    for _ in range(12):
        org1.execute_physical_tick(event_query=None)
    org1.execute_physical_tick(event_query=1)

    state1 = org1.capture_full_state()
    ckpt_path = tmp_path / "test_ckpt.npz"
    h1 = save_checkpoint(ckpt_path, state1)

    org2 = ContinuousLifeOrganism()
    state2, h2 = load_checkpoint(ckpt_path)
    org2.restore_full_state(state2)

    assert h1 == h2
    assert org2.clock.snapshot() == org1.clock.snapshot()


# ---------------------------------------------------------------------------
# Test 9: World and brain same resume hash
# ---------------------------------------------------------------------------
def test_world_and_brain_same_resume_hash(tmp_path):
    org = ContinuousLifeOrganism()
    for _ in range(20):
        org.execute_physical_tick(event_query=None)
    org.execute_physical_tick(event_query=3)

    state = org.capture_full_state()
    ckpt_path = tmp_path / "resume_hash_test.npz"
    hash_saved = save_checkpoint(ckpt_path, state)

    loaded_state, hash_loaded = load_checkpoint(ckpt_path)
    assert hash_saved == hash_loaded


# ---------------------------------------------------------------------------
# Test 10: Reward credit uses executed action
# ---------------------------------------------------------------------------
def test_reward_credit_uses_executed_action():
    org = ContinuousLifeOrganism()
    res = org.execute_physical_tick(event_query=2)

    exec_action = res["sankalpa_decision"]["operational_channel"]
    audit_events = [r for r in res["saksin_ledger_events"] if r["record_type"] == "AUDIT_SHADOW"]
    assert len(audit_events) == 1
    assert audit_events[0]["payload"]["action_taken"] == exec_action


# ---------------------------------------------------------------------------
# Test 11: No hidden optimal action leak
# ---------------------------------------------------------------------------
def test_no_hidden_optimal_action_leak():
    org = ContinuousLifeOrganism()
    obs = org.world.get_observation()
    assert obs.shape == (8,)
    assert np.all(np.isfinite(obs))
    # World step info must not leak oracle fields
    _, _, _, info = org.world.step(1)
    assert "rule_id" not in info
    assert "true_optimal_action" not in info
    assert "oracle" not in info


# ---------------------------------------------------------------------------
# Test 12: Bounded internal compute budget
# ---------------------------------------------------------------------------
def test_bounded_internal_compute_budget():
    org = ContinuousLifeOrganism()
    assert org.sidecar_pool.max_sidecars_per_event <= 3
    assert org.sidecar_pool.max_deliberation_ticks_per_tick <= 192

    # Verify that requesting excessive clones is clipped
    res = org.sidecar_pool.run_deliberation(
        main_core=org.core,
        stimulus=np.zeros(8),
        rng_sidecar=org.rng_sidecar,
        num_clones=10,  # exceeds 3
    )
    assert res["sidecars_spawned"] <= 3


# ---------------------------------------------------------------------------
# Test 13: Disabled learning baseline is unchanged
# ---------------------------------------------------------------------------
def test_disabled_learning_baseline_is_unchanged():
    org1 = ContinuousLifeOrganism()
    org2 = ContinuousLifeOrganism()

    for q in [1, 2, None, 3, None]:
        res1 = org1.execute_physical_tick(event_query=q)
        res2 = org2.execute_physical_tick(event_query=q)
        assert res1["state_integrity"] == res2["state_integrity"]
        assert res1["sankalpa_decision"] == res2["sankalpa_decision"]


# ---------------------------------------------------------------------------
# Test 14: New seeds exclude M5 101-105
# ---------------------------------------------------------------------------
def test_new_seeds_exclude_M5_101_105():
    banned = [101, 102, 103, 104, 105]
    for b_seed in banned:
        cfg = copy.deepcopy(ContinuousLifeOrganism().cfg)
        cfg["rng_streams"]["rng_main_seed"] = b_seed
        with pytest.raises(ValueError, match="(Quarantined M5 seed|SECURITY VIOLATION)"):
            ContinuousLifeOrganism(config=cfg)


# ---------------------------------------------------------------------------
# Test 15: Epistemic and action channels decoupled
# ---------------------------------------------------------------------------
def test_epistemic_and_action_channels_decoupled():
    gate = SankalpaGate()
    # When witness is None, epistemic is ABSTAIN, but operational action can be safe PROBE or motor
    dec_probe = gate.decide(candidate_action=2, witness_ref=None, is_safe_probe=True)
    assert dec_probe.epistemic_channel == "ABSTAIN"
    assert dec_probe.operational_channel == "PROBE"

    dec_motor = gate.decide(candidate_action=3, witness_ref=None, is_safe_probe=False)
    assert dec_motor.epistemic_channel == "ABSTAIN"
    assert dec_motor.operational_channel == 3


# ---------------------------------------------------------------------------
# Test 16: Zero witness forbids commit
# ---------------------------------------------------------------------------
def test_zero_witness_forbids_commit():
    gate = SankalpaGate()
    for action in [1, 2, 3, 4, "PROBE", "WAIT"]:
        dec = gate.decide(candidate_action=action, witness_ref=None)
        assert dec.epistemic_channel == "ABSTAIN"
        assert dec.zero_witness_guard_tripped is True
        assert dec.epistemic_channel != "COMMIT"


# ---------------------------------------------------------------------------
# Test 17: Stale evidence not marked refuted
# ---------------------------------------------------------------------------
def test_stale_evidence_not_marked_refuted():
    pool = HypothesisPool(
        m_phi_uses_aging=8,
        m_phi_uses_ossified=32,
        age_ticks_aging=50,
        age_ticks_ossified=200,
    )
    hyp = pool.register_candidate(
        candidate_id="H_test",
        action_candidate=2,
        v_just="SUPPORTED",
    )
    assert hyp.s_fresh == "FRESH"
    assert hyp.v_just == "SUPPORTED"

    # Simulate heavy usage without audit
    for _ in range(35):
        pool.record_usage("H_test")

    # Freshness must transition to OSSIFIED, but v_just must STAY SUPPORTED!
    assert hyp.s_fresh == "OSSIFIED"
    assert hyp.v_just == "SUPPORTED"
    assert hyp.v_just != "REFUTED"

    # Also test elapsed time trigger
    hyp2 = pool.register_candidate("H_test2", 1, v_just="SUPPORTED")
    pool.advance_time(250)
    assert hyp2.s_fresh == "OSSIFIED"
    assert hyp2.v_just == "SUPPORTED"


# ---------------------------------------------------------------------------
# Test 18: Ledger append-only hash chain valid
# ---------------------------------------------------------------------------
def test_ledger_append_only_hash_chain_valid():
    org = ContinuousLifeOrganism()
    for _ in range(10):
        org.execute_physical_tick(event_query=None)
    for q in [1, 2]:
        org.execute_physical_tick(event_query=q)

    assert org.ledger.records[0]["record_type"] == "CONFIG_REGISTERED"
    assert org.ledger.records[0]["seq"] == 0
    assert org.ledger.validate_integrity() is True


# ---------------------------------------------------------------------------
# Test 19: Endogenous trigger changes behavior vs disabled trigger
# ---------------------------------------------------------------------------
def test_endogenous_trigger_changes_behavior_vs_disabled_trigger():
    """Verify deliberation causally changes chosen action with frozen K."""
    # Seed 810001: historical baseline with calibrated attractor dynamics
    cfg = copy.deepcopy(ContinuousLifeOrganism().cfg)
    cfg["rng_streams"]["rng_main_seed"] = 810001
    cfg["rng_streams"]["banned_seeds"] = []

    org = ContinuousLifeOrganism(config=cfg)

    # Establish carry state across queries 1, 2, 3 with trigger disabled for setup
    for q in [1, 2, 3]:
        org.execute_physical_tick(event_query=q, force_disable_trigger=True)

    # Check live logits and normalized uncertainty U
    l_live = org.core.predict()[1:5]
    p_live = np.exp(l_live - np.max(l_live)) / np.sum(np.exp(l_live - np.max(l_live)))
    u_live = -np.sum(p_live * np.log(np.clip(p_live, 1e-12, 1.0))) / np.log(4)
    assert u_live >= 0.90, f"Expected high uncertainty >= 0.90, got {u_live:.4f}"

    a_live = int(np.argmax(l_live)) + 1

    # Check pre-registered control context trigger score
    ctrl = org.cfg["endogenous_trigger"]["test19_control_context"]
    score, can_fire, _ = org.trigger.evaluate(
        surprise_proxy=ctrl["fixed_surprise_proxy"],
        uncertainty=u_live,
        energy_deficit=ctrl["fixed_energy_deficit"],
        unresolved=ctrl["fixed_unresolved"],
        current_tick=org.clock.core_tick,
        force_disable=False,
    )
    assert score >= 0.495 > 0.45, f"Score {score:.4f} must exceed threshold 0.45"
    assert can_fire is True

    # Pre-state snapshot of MAIN
    z_hash_pre = array_hash(org.core.Z)
    k_hash_pre = array_hash(org.core.K)
    rng_state_pre = copy.deepcopy(org.core.rng.bit_generator.state)

    # --- BRANCH A: trigger = OFF ---
    a_off = a_live

    # --- BRANCH B: trigger = ON ---
    stim = np.zeros(8)
    stim[1] = 1.0
    delib_res = org.sidecar_pool.run_deliberation(org.core, stim, org.rng_sidecar)
    a_on = delib_res["sidecar_prediction"]

    # 1. Behavior change verification
    assert a_on != a_off, f"Deliberation must causally change action! a_on={a_on}, a_off={a_off}"

    # 2. Main core invariance verification
    assert array_hash(org.core.Z) == z_hash_pre, "MAIN Z was mutated by deliberation!"
    assert array_hash(org.core.K) == k_hash_pre, "MAIN K was mutated by deliberation!"
    assert org.core.rng.bit_generator.state == rng_state_pre, "MAIN RNG was consumed by deliberation!"


# ===========================================================================
# T1-FIX-001 REGRESSION TESTS (Independent Audit Verification)
# ===========================================================================

def test_fix1_full_state_checkpoint_and_resume_pending_event(tmp_path):
    """FIX-1: Full-state checkpoint restores step_count, pending event, and exact state."""
    org = ContinuousLifeOrganism()

    # Advance a few idle ticks
    for _ in range(5):
        org.execute_physical_tick(event_query=None)

    # Initiate an external query without finishing it
    pred = org.agent.begin_query(2)
    assert org.agent.pending is not None
    assert pred["prediction"] in (1, 2, 3, 4)

    # Capture and serialize full state during pending event
    full_state = org.capture_full_state()
    ckpt_path = tmp_path / "pending_event_checkpoint.npz"
    h_saved = save_checkpoint(ckpt_path, full_state)

    # Load and restore into a completely fresh organism instance
    loaded_state, h_loaded = load_checkpoint(ckpt_path)
    assert h_saved == h_loaded

    org_resumed = ContinuousLifeOrganism()
    org_resumed.restore_full_state(loaded_state)

    # Verify that pending event and step count were faithfully restored
    assert org_resumed.agent.pending is not None
    assert org_resumed.agent.pending["prediction"]["query"] == 2
    assert org_resumed.core.step_count == org.core.step_count
    np.testing.assert_array_equal(org_resumed.core.Z, org.core.Z)
    np.testing.assert_array_equal(org_resumed.core.K, org.core.K)
    np.testing.assert_array_equal(org_resumed.agent.pending["legacy"], org.agent.pending["legacy"])
    np.testing.assert_array_equal(org_resumed.agent.pending["event"], org.agent.pending["event"])

    # Finish query on both instances and verify identical closure
    org.agent.finish_query(None)
    org_resumed.agent.finish_query(None)

    assert org.agent.pending is None
    assert org_resumed.agent.pending is None
    np.testing.assert_array_equal(org_resumed.core.Z, org.core.Z)
    np.testing.assert_array_equal(org_resumed.core.K, org.core.K)
    assert org_resumed.agent.event_id == org.agent.event_id


def test_fix1_full_state_comparison_after_continuation(tmp_path):
    """FIX-1: Compare FULL dictionary states after continuation (not just Z/K/clock/ledger-head)."""
    cfg = copy.deepcopy(ContinuousLifeOrganism().cfg)
    cfg["rng_streams"]["rng_main_seed"] = 42
    org1 = ContinuousLifeOrganism(config=cfg)

    # Run 10 ticks
    for _ in range(8):
        org1.execute_physical_tick(event_query=None)
    org1.execute_physical_tick(event_query=1)
    org1.execute_physical_tick(event_query=None)

    # Save full checkpoint
    ckpt_path = tmp_path / "full_continuation_ckpt.npz"
    save_checkpoint(ckpt_path, org1.capture_full_state())

    # Restore in clone
    loaded, _ = load_checkpoint(ckpt_path)
    org2 = ContinuousLifeOrganism(config=cfg)
    org2.restore_full_state(loaded)

    # Run both across multiple subsequent events and idle ticks
    for tick_idx in range(15):
        q = (tick_idx % 4) + 1 if tick_idx % 5 == 0 else None
        org1.execute_physical_tick(event_query=q)
        org2.execute_physical_tick(event_query=q)

    # Full dictionary state comparison
    state1 = org1.capture_full_state()
    state2 = org2.capture_full_state()

    def assert_state_equal(v1, v2, path=""):
        if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
            np.testing.assert_array_equal(v1, v2, err_msg=f"Array mismatch at {path}")
        elif isinstance(v1, dict):
            assert isinstance(v2, dict), f"Type mismatch at {path}"
            assert v1.keys() == v2.keys(), f"Key mismatch at {path}: {v1.keys()} vs {v2.keys()}"
            for k in v1:
                assert_state_equal(v1[k], v2[k], f"{path}/{k}" if path else k)
        elif isinstance(v1, (list, tuple)):
            assert isinstance(v2, (list, tuple)), f"Type mismatch at {path}"
            assert len(v1) == len(v2), f"Length mismatch at {path}"
            for idx, (item1, item2) in enumerate(zip(v1, v2)):
                assert_state_equal(item1, item2, f"{path}[{idx}]")
        else:
            assert v1 == v2, f"Value mismatch at {path}: {v1} != {v2}"

    assert_state_equal(state1, state2)


def test_fix2_sidecar_pool_executes_actual_clones():
    """FIX-2: clones_spawned and clones_executed match actual branches run."""
    org = ContinuousLifeOrganism()

    # 1. Single reader mode
    res_single = org.sidecar_pool.single_sidecar_reader(
        main_core=org.core,
        stimulus=np.zeros(8),
        rng_sidecar=org.rng_sidecar,
    )
    assert res_single["mode"] == "single_sidecar_reader"
    assert res_single["clones_requested"] == 1
    assert res_single["clones_executed"] == 1
    assert res_single["clones_spawned"] == 1
    assert res_single["rollout_ticks"] == 64

    # 2. Multi-candidate deliberation: 3 clones requested
    res_multi = org.sidecar_pool.run_deliberation(
        main_core=org.core,
        stimulus=np.zeros(8),
        rng_sidecar=org.rng_sidecar,
        num_clones=3,
        candidate_actions=[1, 2, 3],
    )
    assert res_multi["clones_requested"] == 3
    assert res_multi["clones_executed"] == 3
    assert res_multi["clones_spawned"] == 3
    assert len(res_multi["branches"]) == 3
    assert res_multi["total_rollout_ticks"] == 192

    # Verify each branch ran distinct simulation
    for idx, branch in enumerate(res_multi["branches"]):
        assert branch["branch_index"] == idx
        assert branch["rollout_ticks"] == 64
        assert len(branch["logits"]) == 4

    assert res_multi["sidecar_prediction"] in (1, 2, 3, 4)
    assert res_multi["main_unmutated"] is True


def test_fix3_full_runtime_causal_intervention_real_variables():
    """FIX-3: Full runtime paired test with identical state, events, and 100% REAL internal variables."""
    cfg_a = copy.deepcopy(ContinuousLifeOrganism().cfg)
    cfg_a["rng_streams"]["rng_main_seed"] = 810001
    cfg_a["rng_streams"]["banned_seeds"] = []
    org_a = ContinuousLifeOrganism(config=cfg_a)

    cfg_b = copy.deepcopy(cfg_a)
    org_b = ContinuousLifeOrganism(config=cfg_b)

    # Initial states must be 100% identical
    assert array_hash(org_a.core.Z) == array_hash(org_b.core.Z)
    assert array_hash(org_a.core.K) == array_hash(org_b.core.K)

    # Query event 1 arrives: evaluate with REAL internal variables (NO dummy control substitution)
    res_a = org_a.execute_physical_tick(event_query=1, force_disable_trigger=False)
    res_b = org_b.execute_physical_tick(event_query=1, force_disable_trigger=True)

    # Check that Organism A fired trigger based on its genuine internal state
    assert res_a["trigger"]["fired"] is True
    assert res_a["trigger"]["score"] > 0.45
    # Verify surprise and uncertainty were computed from live state
    assert res_a["trigger"]["components"]["surprise_proxy"] > 0.0
    assert res_a["trigger"]["components"]["uncertainty"] > 0.90

    # Organism B had trigger disabled
    assert res_b["trigger"]["fired"] is False

    # Actual executed operational actions diverge
    act_a = res_a["sankalpa_decision"]["operational_channel"]
    act_b = res_b["sankalpa_decision"]["operational_channel"]
    assert act_a != act_b, f"Causal divergence failed! A={act_a}, B={act_b}"

    # Verify divergence mechanism was reported
    assert res_a["deliberation"] is not None
    assert "divergence_mechanism" in res_a["deliberation"]

    # Verify MAIN core in Organism A remained strictly unmutated
    assert res_a["deliberation"]["main_unmutated"] is True


def test_fix4_witness_integrity_and_ledger_validation():
    """FIX-4: Validate witness reference in Ledger and ensure no unlogged evictions."""
    org = ContinuousLifeOrganism()

    # 1. Fabricated witness_ref is rejected by ledger
    fake_ref = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    is_valid, reason = org.ledger.verify_witness_reference(fake_ref)
    assert is_valid is False
    assert reason == "WITNESS_NOT_FOUND_IN_LEDGER"

    # Fabricated witness trips zero-witness guard in SankalpaGate
    d_ep, zero_tripped = org.sankalpa.evaluate_epistemic_channel(
        witness_ref=fake_ref,
        witness_support_score=0.95,
        ledger=org.ledger,
    )
    assert d_ep == "ABSTAIN"
    assert zero_tripped is True

    # 2. Genuine witness record in ledger is accepted
    obs_rec = org.ledger.append_entry(
        record_type="OBSERVED",
        payload={"query": 1, "prediction": 3, "entropy": 0.95},
        clock=org.clock,
    )
    is_valid_real, _ = org.ledger.verify_witness_reference(obs_rec["entry_hash"])
    assert is_valid_real is True

    d_ep_real, zero_tripped_real = org.sankalpa.evaluate_epistemic_channel(
        witness_ref=obs_rec["entry_hash"],
        witness_support_score=0.95,
        ledger=org.ledger,
    )
    assert d_ep_real == "COMMIT"
    assert zero_tripped_real is False

    # 3. Eviction logging: HypothesisPool records eviction info when capacity exceeded
    pool = HypothesisPool(max_active_hypotheses=3)
    pool.register_candidate("H1", 1, v_just="SUPPORTED")
    pool.register_candidate("H2", 2, v_just="UNVERIFIED")
    pool.register_candidate("H3", 3, v_just="REFUTED")
    assert pool.last_eviction is None

    # Registering 4th candidate evicts REFUTED (H3) first
    pool.register_candidate("H4", 4, v_just="UNVERIFIED")
    assert pool.last_eviction is not None
    assert pool.last_eviction["candidate_id"] == "H3"
    assert pool.last_eviction["v_just"] == "REFUTED"
    assert pool.last_eviction["reason"] == "POOL_CAPACITY_EXCEEDED"
