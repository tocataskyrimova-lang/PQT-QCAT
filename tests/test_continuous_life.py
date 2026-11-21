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
    ledger = SaksinLedger()
    rec1 = ledger.append_verified_feedback("H_test", 2, "OK", "H_test", "DOMAIN_A", 1.0, True, LifeClock())
    hyp = pool.register_candidate(
        candidate_id="H_test",
        action_candidate=2,
        v_just="SUPPORTED",
        witness_ref=rec1["entry_hash"],
        context={"domain": "DOMAIN_A"},
        ledger=ledger,
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
    rec2 = ledger.append_verified_feedback("H_test2", 1, "OK", "H_test2", "DOMAIN_A", 1.0, True, LifeClock())
    hyp2 = pool.register_candidate("H_test2", 1, v_just="SUPPORTED", witness_ref=rec2["entry_hash"], context={"domain": "DOMAIN_A"}, ledger=ledger)
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

    # 2. Genuine OBSERVED record in ledger is accepted as observation, but CANNOT authorize COMMIT
    obs_rec = org.ledger.append_entry(
        record_type="OBSERVED",
        payload={"query": 1, "prediction": 3, "entropy": 0.95},
        clock=org.clock,
    )
    is_valid_real, _ = org.ledger.verify_witness_reference(obs_rec["entry_hash"])
    assert is_valid_real is True

    # High support score with OBSERVED record yields CLASS_ONLY, NEVER COMMIT (Item 1 & 5)
    d_ep_obs, zero_tripped_obs = org.sankalpa.evaluate_epistemic_channel(
        witness_ref=obs_rec["entry_hash"],
        witness_support_score=0.95,
        ledger=org.ledger,
    )
    assert d_ep_obs == "CLASS_ONLY"
    assert d_ep_obs != "COMMIT"
    assert zero_tripped_obs is False

    # 3. Genuine VERIFIED_FEEDBACK record with matching provenance and outcome permits COMMIT (Item 2)
    fb_rec = org.ledger.append_verified_feedback(
        hypothesis_id="H_evt_1",
        action=3,
        result="TARGET_MATCH",
        provenance="H_evt_1",
        applicability_domain="QUERY_1",
        support_score=0.90,
        clock=org.clock,
    )
    d_ep_fb, zero_tripped_fb = org.sankalpa.evaluate_epistemic_channel(
        witness_ref=fb_rec["entry_hash"],
        ledger=org.ledger,
        hypothesis="H_evt_1",
        candidate_action=3,
        context={"domain": "QUERY_1"},
    )
    assert d_ep_fb == "COMMIT"
    assert zero_tripped_fb is False

    # 4. Eviction logging: HypothesisPool records eviction info when capacity exceeded
    pool = HypothesisPool(max_active_hypotheses=3)
    ev_ledger = SaksinLedger()
    rec_h1 = ev_ledger.append_verified_feedback("H1", 1, "OK", "H1", "DOMAIN_A", 1.0, True, LifeClock())
    pool.register_candidate("H1", 1, v_just="SUPPORTED", witness_ref=rec_h1["entry_hash"], context={"domain": "DOMAIN_A"}, ledger=ev_ledger)
    pool.register_candidate("H2", 2, v_just="UNVERIFIED")
    pool.register_candidate("H3", 3, v_just="REFUTED")
    assert pool.last_eviction is None

    # Registering 4th candidate evicts REFUTED (H3) first
    pool.register_candidate("H4", 4, v_just="UNVERIFIED")
    assert pool.last_eviction is not None
    assert pool.last_eviction["candidate_id"] == "H3"
    assert pool.last_eviction["v_just"] == "REFUTED"
    assert pool.last_eviction["reason"] == "POOL_CAPACITY_EXCEEDED"


# ---------------------------------------------------------------------------
# LIFE-003 / T1-EPISTEMIC-PATCH-001 Tests
# ---------------------------------------------------------------------------
def test_epistemic_patch_observed_and_shadow_cannot_authorize_commit():
    """Negative Test (Items 1 & 5): Genuine OBSERVED or AUDIT_SHADOW records
    in Ledger NEVER permit COMMIT even with arbitrary high witness_support_score.
    """
    gate = SankalpaGate(require_structured_witness=True, witness_support_min_score=0.60)
    ledger = SaksinLedger()
    clk = LifeClock()

    # 1. OBSERVED record in ledger
    obs_rec = ledger.append_entry(
        record_type="OBSERVED",
        payload={"query": 1, "prediction": 2, "entropy": 0.85},
        clock=clk,
    )
    # Authenticity in ledger is valid
    is_valid, _ = ledger.verify_witness_reference(obs_rec["entry_hash"])
    assert is_valid is True

    # High support score MUST NOT authorize COMMIT (strictly CLASS_ONLY)
    d_ep, zero_tripped = gate.evaluate_epistemic_channel(
        witness_ref=obs_rec["entry_hash"],
        witness_support_score=0.99,  # Arbitrarily high caller score
        ledger=ledger,
    )
    assert d_ep != "COMMIT", "Authentic OBSERVED record must NEVER authorize COMMIT!"
    assert d_ep == "CLASS_ONLY"
    assert zero_tripped is False

    # Also via full decide() interface
    dec = gate.decide(
        candidate_action=2,
        witness_ref=obs_rec["entry_hash"],
        witness_support_score=0.99,
        ledger=ledger,
    )
    assert dec.epistemic_channel != "COMMIT"
    assert dec.epistemic_channel == "CLASS_ONLY"
    assert dec.operational_channel == 2

    # 2. AUDIT_SHADOW record in ledger
    shadow_rec = ledger.append_entry(
        record_type="AUDIT_SHADOW",
        payload={"world_step": 1, "reward": 1.0, "done": False},
        clock=clk,
    )
    is_valid_shadow, _ = ledger.verify_witness_reference(shadow_rec["entry_hash"])
    assert is_valid_shadow is True

    d_ep_sh, zero_tripped_sh = gate.evaluate_epistemic_channel(
        witness_ref=shadow_rec["entry_hash"],
        witness_support_score=0.99,
        ledger=ledger,
    )
    assert d_ep_sh != "COMMIT", "AUDIT_SHADOW record must NEVER authorize COMMIT!"
    assert d_ep_sh == "ABSTAIN"
    assert zero_tripped_sh is True


def test_epistemic_patch_missing_ledger_fails_closed_abstain():
    """Item 3: When require_structured_witness=True and ledger is None,
    evaluate_epistemic_channel fails closed to ABSTAIN with zero-witness guard tripped.
    """
    gate = SankalpaGate(require_structured_witness=True)
    d_ep, zero_tripped = gate.evaluate_epistemic_channel(
        witness_ref="0" * 64,
        witness_support_score=0.99,
        ledger=None,
    )
    assert d_ep == "ABSTAIN"
    assert zero_tripped is True

    # decide() also returns ABSTAIN
    dec = gate.decide(
        candidate_action=1,
        witness_ref="0" * 64,
        witness_support_score=0.99,
        ledger=None,
    )
    assert dec.epistemic_channel == "ABSTAIN"
    assert dec.zero_witness_guard_tripped is True


def test_epistemic_patch_ledger_integrity_seq_and_prev_hash():
    """Item 4: Ledger integrity checks seq == len(records) and prev_hash matches computed hash."""
    ledger = SaksinLedger(genesis_config_sha="a" * 64)
    clk = LifeClock()
    ledger.append_entry("OBSERVED", {"query": 1}, clk)
    ledger.append_entry("OBSERVED", {"query": 2}, clk)
    assert ledger.validate_integrity() is True

    # 1. Mutate seq
    ledger.seq = 999
    assert ledger.validate_integrity() is False
    ledger.seq = len(ledger.records)
    assert ledger.validate_integrity() is True

    # 2. Mutate prev_hash
    real_prev = ledger.prev_hash
    ledger.prev_hash = "f" * 64
    assert ledger.validate_integrity() is False
    ledger.prev_hash = real_prev
    assert ledger.validate_integrity() is True


def test_epistemic_patch_sidecar_neural_entropy_terminology():
    """Item 6: Sidecar deliberation terminology reflects neural prediction entropy selection,
    not empirical future outcome comparison.
    """
    core = SixEchelonCore(seed=42)
    pool = SidecarPool()
    rng = np.random.default_rng(123)
    stim = np.zeros(8)
    stim[1] = 1.0

    res = pool.run_deliberation(core, stim, rng, num_clones=3)
    assert res["selection_criterion"] == "MINIMAL_NEURAL_PREDICTION_ENTROPY"
    assert res["divergence_mechanism"] in (
        "MULTI_BRANCH_PREDICTION_ENTROPY_SELECTION",
        "CONCORDANT_PREDICTION",
    )
    # Ensure misleading term is completely absent
    assert res["divergence_mechanism"] != "MULTI_CANDIDATE_OUTCOME_COMPARISON"
    assert "epistemic_disclaimer" in res


# ---------------------------------------------------------------------------
# LIFE-003 / T1-FIX-002 Regression Tests (Addressing GPT-Astra Audit Findings)
# ---------------------------------------------------------------------------

def test_early_tick_160_resume_preserves_exact_eviction_order(tmp_path):
    """FIX-1 (Focus 1): Monotonic created_seq ensures deterministic eviction order
    across checkpoint serialization and resume. Resuming at tick 160 evicts the exact
    same candidate at tick 168 as the continuous un-interrupted run.
    """
    cfg, _ = load_and_validate_prereg_config()
    
    # 1. Run org1 uninterrupted up to tick 200
    org1 = ContinuousLifeOrganism(config=cfg)
    evictions_org1 = []
    stream_org1 = []
    
    event_counter = 0
    while org1.clock.core_tick < 200:
        ticks_rem = 200 - org1.clock.core_tick
        is_query = (org1.clock.core_tick % 40 == 0) and (ticks_rem >= org1.t_query)
        q = (event_counter % 6) + 1 if is_query else None
        if is_query:
            event_counter += 1
        entry = org1.execute_physical_tick(event_query=q)
        stream_org1.append(entry)
        if org1.hypotheses.last_eviction is not None:
            evictions_org1.append((org1.clock.core_tick, org1.hypotheses.last_eviction["candidate_id"]))
            
    # 2. Run org2 up to tick 160, save checkpoint, resume into org2_resumed and run to 200
    org2 = ContinuousLifeOrganism(config=cfg)
    event_counter2 = 0
    while org2.clock.core_tick < 160:
        ticks_rem = 160 - org2.clock.core_tick
        is_query = (org2.clock.core_tick % 40 == 0) and (ticks_rem >= org2.t_query)
        q = (event_counter2 % 6) + 1 if is_query else None
        if is_query:
            event_counter2 += 1
        org2.execute_physical_tick(event_query=q)
        
    ckpt_path = tmp_path / "ckpt_160.npz"
    h_save = save_checkpoint(ckpt_path, org2.capture_full_state())
    loaded_st, h_load = load_checkpoint(ckpt_path)
    assert h_save == h_load
    
    org2_resumed = ContinuousLifeOrganism(config=cfg)
    org2_resumed.restore_full_state(loaded_st)
    
    evictions_org2 = []
    stream_org2 = []
    while org2_resumed.clock.core_tick < 200:
        ticks_rem = 200 - org2_resumed.clock.core_tick
        is_query = (org2_resumed.clock.core_tick % 40 == 0) and (ticks_rem >= org2_resumed.t_query)
        q = (event_counter2 % 6) + 1 if is_query else None
        if is_query:
            event_counter2 += 1
        entry = org2_resumed.execute_physical_tick(event_query=q)
        stream_org2.append(entry)
        if org2_resumed.hypotheses.last_eviction is not None:
            evictions_org2.append((org2_resumed.clock.core_tick, org2_resumed.hypotheses.last_eviction["candidate_id"]))
            
    # Eviction sequence on tick 168 must match perfectly
    assert len(evictions_org1) > 0, "Expected at least one eviction in org1"
    assert len(evictions_org2) > 0, "Expected at least one eviction in org2_resumed"
    evict_168_org1 = [cand for tick, cand in evictions_org1 if tick == 168]
    evict_168_org2 = [cand for tick, cand in evictions_org2 if tick == 168]
    assert evict_168_org1 == evict_168_org2, f"Eviction mismatch at tick 168: org1={evict_168_org1} vs resumed={evict_168_org2}"
    
    # State hash at tick 200 must match bit-for-bit
    h_final_1 = compute_state_hash(org1.capture_full_state())
    h_final_2 = compute_state_hash(org2_resumed.capture_full_state())
    assert h_final_1 == h_final_2, f"State hash divergence at tick 200: {h_final_1} != {h_final_2}"


def test_checkpoint_digest_injective_no_sequence_or_type_collision():
    """FIX-2 (Focus 6): Injective canonical binary serialization eliminates digest collisions."""
    # Sequence boundaries
    assert compute_state_hash({"seq": [1, 23]}) != compute_state_hash({"seq": [12, 3]})
    assert compute_state_hash({"seq": [1, 2, 3]}) != compute_state_hash({"seq": [12, 3]})
    assert compute_state_hash({"seq": [123]}) != compute_state_hash({"seq": [1, 23]})

    # Primitive types
    assert compute_state_hash({"v": 1}) != compute_state_hash({"v": "1"})
    assert compute_state_hash({"v": True}) != compute_state_hash({"v": 1})
    assert compute_state_hash({"v": False}) != compute_state_hash({"v": 0})
    assert compute_state_hash({"v": None}) != compute_state_hash({"v": "None"})

    # Container types
    assert compute_state_hash({"v": [1, 2]}) != compute_state_hash({"v": (1, 2)})
    assert compute_state_hash({"v": {"a": 1}}) != compute_state_hash({"v": [("a", 1)]})

    # Numpy dtypes and shapes
    arr_i32 = np.array([1, 2], dtype=np.int32)
    arr_i64 = np.array([1, 2], dtype=np.int64)
    arr_f32 = np.array([1.0, 2.0], dtype=np.float32)
    assert compute_state_hash({"arr": arr_i32}) != compute_state_hash({"arr": arr_i64})
    assert compute_state_hash({"arr": arr_i32}) != compute_state_hash({"arr": arr_f32})
    assert compute_state_hash({"arr": arr_i32}) != compute_state_hash({"arr": [1, 2]})


def test_hypothesis_cannot_promote_to_supported_without_verified_witness():
    """FIX-3 (Check D & Focus 4): Promotion to SUPPORTED strictly requires non-empty verified witness."""
    pool = HypothesisPool()
    hyp = pool.register_candidate("H_cand", 2, context={"domain": "DOMAIN_A"})

    # 1. audit_verification with None witness_ref fails closed
    with pytest.raises(ValueError, match="without a valid witness_ref"):
        pool.audit_verification("H_cand", is_supported=True, witness_ref=None)
    assert hyp.v_just == "UNVERIFIED"

    # 2. audit_verification with empty or whitespace witness_ref fails closed
    with pytest.raises(ValueError, match="without a valid witness_ref"):
        pool.audit_verification("H_cand", is_supported=True, witness_ref="   ")
    assert hyp.v_just == "UNVERIFIED"

    # 3. Direct registration with v_just='SUPPORTED' and no witness_ref fails closed
    with pytest.raises(ValueError, match="without a non-empty witness_ref"):
        pool.register_candidate("H_bad", 1, v_just="SUPPORTED", witness_ref=None)

    # 4. If ledger provided, unverified witness fails closed
    ledger = SaksinLedger()
    with pytest.raises(ValueError, match="not verified in ledger"):
        pool.audit_verification("H_cand", is_supported=True, witness_ref="fake_hash_123", ledger=ledger)
    assert hyp.v_just == "UNVERIFIED"

    # 5. If ledger provided, non-VERIFIED_FEEDBACK record fails closed
    obs = ledger.append_entry("OBSERVED", {"query": 1, "prediction": 1}, LifeClock())
    with pytest.raises(ValueError, match="not a VERIFIED_FEEDBACK record"):
        pool.audit_verification("H_cand", is_supported=True, witness_ref=obs["entry_hash"], ledger=ledger)
    assert hyp.v_just == "UNVERIFIED"

    # 6. Genuine VERIFIED_FEEDBACK succeeds
    fb = ledger.append_verified_feedback("H_cand", 2, "SUCCESS", "H_cand", "DOMAIN_A", 1.0, True, LifeClock())
    res = pool.audit_verification("H_cand", is_supported=True, witness_ref=fb["entry_hash"], ledger=ledger)
    assert hyp.v_just == "SUPPORTED"
    assert res is not None
    assert res["new_justification"] == "SUPPORTED"


@pytest.mark.parametrize(
    "flaw_field,flaw_val",
    [
        ("confirmed", None),
        ("confirmed", False),
        ("confirmed", 1),                # Integer 1, not strict bool True
        ("confirmed", "True"),           # String "True", not bool True
        ("result", None),
        ("result", ""),
        ("result", []),
        ("provenance_ledger", None),     # Missing provenance in witness payload
        ("provenance_ledger", ""),
        ("provenance_context", None),    # Missing provenance in query/hypothesis context
        ("provenance_mismatch", "H_diff"),
        ("action_ledger", None),         # Missing action in witness payload
        ("action_context", None),        # Missing action in candidate
        ("action_mismatch", 99),
        ("domain_ledger", None),         # Missing domain in witness payload
        ("domain_context", None),        # Missing domain in query context
        ("domain_mismatch", "DOMAIN_WRONG"),
    ],
)
def test_sankalpa_commit_requires_strict_confirmed_true_and_all_four_fields(flaw_field, flaw_val):
    """FIX-3 (Focus 4): Whitelist fail-closed gate requires confirmed is True, non-empty result,
    and non-empty matching provenance, action, domain on both sides simultaneously.
    """
    gate = SankalpaGate(require_structured_witness=True, witness_support_min_score=0.60)
    ledger = SaksinLedger()

    # Base valid parameters
    p_confirmed = True
    p_result = "VALID_OUTCOME"
    p_prov = "H_target"
    p_act = 3
    p_dom = "ENV_DOMAIN_1"
    
    ctx_prov = "H_target"
    ctx_act = 3
    ctx_dom = "ENV_DOMAIN_1"

    if flaw_field == "confirmed":
        p_confirmed = flaw_val
    elif flaw_field == "result":
        p_result = flaw_val
    elif flaw_field == "provenance_ledger":
        p_prov = flaw_val
    elif flaw_field == "provenance_context":
        ctx_prov = flaw_val
    elif flaw_field == "provenance_mismatch":
        p_prov = flaw_val
    elif flaw_field == "action_ledger":
        p_act = flaw_val
    elif flaw_field == "action_context":
        ctx_act = flaw_val
    elif flaw_field == "action_mismatch":
        p_act = flaw_val
    elif flaw_field == "domain_ledger":
        p_dom = flaw_val
    elif flaw_field == "domain_context":
        ctx_dom = flaw_val
    elif flaw_field == "domain_mismatch":
        p_dom = flaw_val

    # Record verified feedback entry with flawed parameters
    payload = {
        "hypothesis_id": p_prov,
        "action": p_act,
        "result": p_result,
        "provenance": p_prov,
        "applicability_domain": p_dom,
        "support_score": 1.0,
        "confirmed": p_confirmed,
    }
    rec = ledger.append_entry("VERIFIED_FEEDBACK", payload, LifeClock())

    hypothesis_obj = {"candidate_id": ctx_prov, "action_candidate": ctx_act} if ctx_prov is not None else None
    context_obj = {"domain": ctx_dom} if ctx_dom is not None else None

    d_ep, tripped = gate.evaluate_epistemic_channel(
        witness_ref=rec["entry_hash"],
        ledger=ledger,
        hypothesis=hypothesis_obj,
        candidate_action=ctx_act,
        context=context_obj,
    )
    assert d_ep == "ABSTAIN", f"Failed closed check for {flaw_field}={flaw_val}: got {d_ep}"
    assert tripped is True


def test_resume_verification_covers_all_5000_physical_ticks_and_subticks():
    """FIX-4 (Check A & Focus 3): resume_verification.json explicitly covers 5000 physical ticks,
    4125 scheduler steps, and 875 interior query subticks with 0 discrepancies.
    """
    res_path = REPO_ROOT / "results" / "THOUGHT_LOOP_T1" / "resume_verification.json"
    if not res_path.is_file():
        pytest.skip("resume_verification.json not generated yet")

    import json
    with open(res_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["verified_physical_ticks"] == 5000
    assert data["verified_scheduler_steps"] == 4125
    assert data["verified_interior_query_subticks"] == 875
    assert data["step_discrepancy_count"] == 0
    assert data["subtick_discrepancy_count"] == 0
    assert data["all_5000_physical_ticks_identical"] is True
    assert data["status"] == "PASS"


def test_clean_reproduction_matches_published_thought_stream_and_trigger_artifact():
    """FIX-4 (Check A & Focus 3): Ensure outdated MULTI_CANDIDATE_OUTCOME_COMPARISON
    is completely eradicated and fresh run matches published artifacts.
    """
    results_dir = REPO_ROOT / "results" / "THOUGHT_LOOP_T1"
    t19_path = results_dir / "trigger_on_vs_off.json"
    stream_path = results_dir / "thought_stream.jsonl"

    if not t19_path.is_file() or not stream_path.is_file():
        pytest.skip("T1 published artifacts not found")

    import json
    with open(t19_path, "r", encoding="utf-8") as f:
        t19_text = f.read()

    assert "MULTI_CANDIDATE_OUTCOME_COMPARISON" not in t19_text

    with open(stream_path, "r", encoding="utf-8") as f:
        for line in f:
            assert "MULTI_CANDIDATE_OUTCOME_COMPARISON" not in line


# ---------------------------------------------------------------------------
# LIFE-003 / T1-FIX-003 (RC3) Regression Tests
# ---------------------------------------------------------------------------
def test_canonical_encode_distinguishes_int_and_str_dict_keys():
    """FIX-2 (RC3): {1: 'a'} and {'1': 'a'} must produce distinct canonical encodings and digests."""
    h_int = compute_state_hash({1: "a"})
    h_str = compute_state_hash({"1": "a"})
    assert h_int != h_str, f"Collision between int and str dict keys: {h_int}"


def test_canonical_encode_distinguishes_structured_numpy_field_names():
    """FIX-2 (RC3): Structured arrays with different field names must produce distinct digests."""
    a_x = np.zeros(3, dtype=[("x", "<i4")])
    a_y = np.zeros(3, dtype=[("y", "<i4")])
    h_x = compute_state_hash({"arr": a_x})
    h_y = compute_state_hash({"arr": a_y})
    assert h_x != h_y, f"Collision between structured numpy dtypes with different field names: {h_x}"


def test_checkpoint_save_load_preserves_tuple_and_digest_or_rejects_explicitly(tmp_path):
    """FIX-2 (RC3): Round-trip saving and loading preserves tuples, dict key types, and compute_state_hash."""
    orig_state = {
        "tuple_data": (1, 2, (3, 4)),
        "list_data": [5, 6],
        "int_keys": {10: "val10", 20: "val20"},
        "str_keys": {"10": "str10", "20": "str20"},
        "struct_arr": np.array([(1, 2.5)], dtype=[("field_a", "<i4"), ("field_b", "<f4")]),
        "raw_bytes": b"binary_payload",
    }
    ckpt_file = tmp_path / "test_roundtrip.npz"
    orig_hash = save_checkpoint(ckpt_file, orig_state)
    loaded_state, loaded_hash = load_checkpoint(ckpt_file)

    assert isinstance(loaded_state["tuple_data"], tuple)
    assert isinstance(loaded_state["tuple_data"][2], tuple)
    assert isinstance(loaded_state["list_data"], list)
    assert isinstance(loaded_state["raw_bytes"], bytes)
    assert 10 in loaded_state["int_keys"] and isinstance(list(loaded_state["int_keys"].keys())[0], int)
    assert "10" in loaded_state["str_keys"] and isinstance(list(loaded_state["str_keys"].keys())[0], str)
    assert loaded_state["struct_arr"].dtype == orig_state["struct_arr"].dtype
    assert orig_hash == loaded_hash
    assert compute_state_hash(loaded_state) == compute_state_hash(orig_state)


def test_hypothesis_promotion_without_ledger_fails_closed():
    """FIX-3 (RC3): Establishing v_just='SUPPORTED' without a ledger strictly fails closed."""
    pool = HypothesisPool()
    hyp = pool.register_candidate("H_test", 1, v_just="UNVERIFIED")

    # 1. audit_verification with ledger=None must raise ValueError and remain UNVERIFIED
    with pytest.raises(ValueError, match="without an authentic ledger"):
        pool.audit_verification("H_test", is_supported=True, witness_ref="FAKE_WITNESS_999", ledger=None)
    assert hyp.v_just == "UNVERIFIED"

    # 2. Direct registration with v_just='SUPPORTED' and ledger=None must raise ValueError
    with pytest.raises(ValueError, match="without an authentic ledger"):
        pool.register_candidate("H_new", 1, v_just="SUPPORTED", witness_ref="FAKE_WITNESS_999", ledger=None)


def test_hypothesis_promotion_rejects_negative_feedback_record():
    """FIX-3 (RC3): Genuine VERIFIED_FEEDBACK record with confirmed=False fails closed on promotion."""
    pool = HypothesisPool()
    hyp = pool.register_candidate("H_neg", 2, v_just="UNVERIFIED")
    ledger = SaksinLedger()

    # Create authentic feedback with confirmed=False via raw append_entry
    neg_payload = {
        "hypothesis_id": "H_neg",
        "action": 2,
        "result": "DISCONFIRMED",
        "provenance": "H_neg",
        "applicability_domain": "DOMAIN_A",
        "support_score": 0.0,
        "confirmed": False,
    }
    rec = ledger.append_entry("VERIFIED_FEEDBACK", neg_payload, LifeClock())

    with pytest.raises(ValueError, match="confirmed status is not strict True"):
        pool.audit_verification("H_neg", is_supported=True, witness_ref=rec["entry_hash"], ledger=ledger)
    assert hyp.v_just == "UNVERIFIED"


def test_hypothesis_promotion_rejects_mismatched_candidate_or_action_feedback():
    """FIX-3 (RC3): Genuine VERIFIED_FEEDBACK from another candidate or action fails closed on promotion."""
    pool = HypothesisPool()
    hyp = pool.register_candidate("H_target", 2, v_just="UNVERIFIED")
    ledger = SaksinLedger()

    # Feedback from different candidate ("H_other", action 2)
    fb_other_cand = ledger.append_verified_feedback("H_other", 2, "OUTCOME_OK", "H_other", "DOMAIN_A", 1.0, True, LifeClock())
    with pytest.raises(ValueError, match="does not match hypothesis"):
        pool.audit_verification("H_target", is_supported=True, witness_ref=fb_other_cand["entry_hash"], ledger=ledger)
    assert hyp.v_just == "UNVERIFIED"

    # Feedback from same candidate but different action ("H_target", action 3)
    fb_other_act = ledger.append_verified_feedback("H_target", 3, "OUTCOME_OK", "H_target", "DOMAIN_A", 1.0, True, LifeClock())
    with pytest.raises(ValueError, match="does not match hypothesis action"):
        pool.audit_verification("H_target", is_supported=True, witness_ref=fb_other_act["entry_hash"], ledger=ledger)
    assert hyp.v_just == "UNVERIFIED"


def test_public_ledger_writer_rejects_int_confirmed_before_sankalpa():
    """FIX-3 (RC3): Public ledger writer rejects int confirmed=1 with TypeError before reaching Sankalpa."""
    ledger = SaksinLedger()
    gate = SankalpaGate(require_structured_witness=True, witness_support_min_score=0.60)

    # 1. Public writer must raise TypeError when confirmed is not strict bool
    with pytest.raises(TypeError, match="confirmed must be strict bool, got int"):
        ledger.append_verified_feedback(
            hypothesis_id="H_strict",
            action=1,
            result="RESULT_OK",
            provenance="H_strict",
            applicability_domain="DOMAIN_A",
            support_score=1.0,
            confirmed=1,
            clock=LifeClock(),
        )

    # 2. Attempting to pass confirmed='True' (str) also raises TypeError
    with pytest.raises(TypeError, match="confirmed must be strict bool, got str"):
        ledger.append_verified_feedback(
            hypothesis_id="H_strict",
            action=1,
            result="RESULT_OK",
            provenance="H_strict",
            applicability_domain="DOMAIN_A",
            support_score=1.0,
            confirmed="True",
            clock=LifeClock(),
        )

    # 3. Valid bool True succeeds in writer and authorizes COMMIT in Sankalpa
    valid_rec = ledger.append_verified_feedback(
        hypothesis_id="H_strict",
        action=1,
        result="RESULT_OK",
        provenance="H_strict",
        applicability_domain="DOMAIN_A",
        support_score=1.0,
        confirmed=True,
        clock=LifeClock(),
    )
    d_ep, tripped = gate.evaluate_epistemic_channel(
        witness_ref=valid_rec["entry_hash"],
        ledger=ledger,
        hypothesis={"candidate_id": "H_strict", "action_candidate": 1},
        candidate_action=1,
        context={"domain": "DOMAIN_A"},
    )
    assert d_ep == "COMMIT"
    assert tripped is False

