"""Regression tests for CONTROL-FIX-001 and Control Models Empirical Audit.

Verifies:
1. no_replay injects strictly zero external sensory drive during sleep (u_ext = 0, x_t = None).
2. noA_dream maintains A=0 (vortex force == 0.0) across the ENTIRE trial.
3. audit_control_models evaluates empirical execution and computes COMPLIANT.
4. Symmetric PQT vs noA dream safety comparison across tuning seeds.
"""
from __future__ import annotations
import numpy as np
import pytest

from pqt_qcat.core.six_echelon import SixEchelonCore, SixEchelonConfig
from pqt_qcat.core.vortex import VortexParameters
from pqt_qcat.memory.witness import WitnessLedger, ObservedEvent
from pqt_qcat.dream.consolidation import DreamConsolidationEngine, ConsolidationConfig
from pqt_qcat.dream.control_audit import audit_single_policy, run_empirical_control_audit

CALIBRATION_SEEDS = [42, 11, 12, 13]
PROTECTED_SEEDS = {101, 102, 103, 104, 105}


def test_seed_isolation():
    """Ensure seeds 101-105 remain strictly sealed and forbidden."""
    for s in CALIBRATION_SEEDS:
        assert s not in PROTECTED_SEEDS


def test_control_fix_001_no_replay_zero_sensory_drive():
    """Verify no_replay provides strictly zero external sensory input (u_ext = 0) during sleep."""
    core = SixEchelonCore(seed=42)
    ledger = WitnessLedger()
    ledger.record_observation(1, 1, 1, 1, 0.99)

    calls_during_sleep = []
    original_step = core.step

    def instrumented_step(x_t=None, write_gate=1.0, dt=None):
        if write_gate == 0.0:
            calls_during_sleep.append((x_t, write_gate))
        return original_step(x_t=x_t, write_gate=write_gate, dt=dt)

    core.step = instrumented_step

    engine = DreamConsolidationEngine(seed=42)
    res = engine.run_sleep_policy("no_replay", core, ledger, idle_ticks=100)

    # Invariant 1: Exactly 100 idle steps were taken with write_gate=0.0
    assert len(calls_during_sleep) == 100

    # Invariant 2: None of the sleep steps had any sensory input
    sensory_inputs_during_sleep = [c[0] for c in calls_during_sleep if c[0] is not None]
    assert len(sensory_inputs_during_sleep) == 0, f"Sensory inputs found during sleep: {sensory_inputs_during_sleep}"

    # Invariant 3: Parameters are 100% untouched
    assert res["delta_K_norm"] == 0.0
    assert res["delta_B_norm"] == 0.0


def test_control_fix_001_noA_dream_zero_vortex_throughout_entire_trial():
    """Verify noA_dream maintains enable_vortex=False and F_vortex=0 throughout the entire trial."""
    cfg = SixEchelonConfig(enable_vortex=False, integrator_mode="fixed")
    core = SixEchelonCore(config=cfg, seed=42)
    ledger = WitnessLedger()
    ledger.record_observation(1, 1, 2, 2, 0.99)
    ledger.record_observation(2, 2, 3, 3, 0.99)

    vortex_states_during_steps = []
    original_step = core.step

    def instrumented_step(x_t=None, write_gate=1.0, dt=None):
        vortex_states_during_steps.append(bool(core.config.enable_vortex))
        return original_step(x_t=x_t, write_gate=write_gate, dt=dt)

    core.step = instrumented_step

    engine = DreamConsolidationEngine(seed=42)
    res = engine.run_sleep_policy("noA_dream", core, ledger, idle_ticks=150)

    # Invariant 1: enable_vortex was False on EVERY single continuous step
    assert len(vortex_states_during_steps) >= 150
    assert all(v is False for v in vortex_states_during_steps), "enable_vortex was temporarily True during noA_dream!"

    # Invariant 2: After sleep, enable_vortex is STILL False
    assert core.config.enable_vortex is False

    # Invariant 3: Rotational vortex force is identically zero on non-zero gradient
    z_test = np.array([0.8, -0.5, 0.4, 0.8, -0.4, 0.5, -0.5, 0.2], dtype=np.float64)
    for c in range(6):
        core.Z[8 * c : 8 * c + 8] = z_test.copy()
        f_vort = core.compute_cell_vortex_force(c)
        assert np.max(np.abs(f_vort)) < 1e-15, f"Cell {c}: non-zero vortex force in noA: {f_vort}"


def test_control_fix_001_empirical_audit_computes_compliance():
    """Verify that audit_control_models evaluates empirical execution and computes COMPLIANT."""
    audit_report = run_empirical_control_audit(seed=42)
    assert audit_report["audit_type"] == "EMPIRICAL_RUNTIME_VERIFICATION"
    assert audit_report["all_policies_compliant"] is True
    assert audit_report["overall_control_verdict"] == "COMPLIANT"

    policies = ["no_replay", "state_only", "naive_replay", "guarded_replay", "noA_dream", "PQT_dream"]
    for pol in policies:
        pol_data = audit_report["policies"][pol]
        assert pol_data["audit_verdict"] == "COMPLIANT"
        emp = pol_data["empirical_checks"]
        assert emp["step_budget_exact"] is True
        assert emp["sensory_drive_strictly_zero"] is True
        assert emp["readout_strictly_frozen"] is True
        assert emp["vortex_behavior_verified"] is True
        assert emp["gate_behavior_verified"] is True
        assert emp["provenance_isolated_verified"] is True


def test_control_fix_001_symmetric_multi_seed_comparison():
    """Verify symmetric PQT vs noA dream safety across tuning seeds."""
    engine = DreamConsolidationEngine(seed=42)

    for seed in CALIBRATION_SEEDS:
        core_pqt = SixEchelonCore(SixEchelonConfig(enable_vortex=True, integrator_mode="fixed"), seed=seed)
        core_noa = SixEchelonCore(SixEchelonConfig(enable_vortex=False, integrator_mode="fixed"), seed=seed)

        ledger_pqt = WitnessLedger()
        ledger_noa = WitnessLedger()

        for k in range(1, 5):
            ledger_pqt.record_observation(k, k, k, k, 0.98)
            ledger_noa.record_observation(k, k, k, k, 0.98)
            core_pqt.B[31 + k, k] = 4.0
            core_noa.B[31 + k, k] = 4.0

        diag_pqt = engine.run_sleep_policy("PQT_dream", core_pqt, ledger_pqt, idle_ticks=100)
        diag_noa = engine.run_sleep_policy("noA_dream", core_noa, ledger_noa, idle_ticks=100)

        # Both policies must pass sleep safety (degradation <= 2%)
        assert diag_pqt["passed_h3_sleep_safe"] is True
        assert diag_noa["passed_h3_sleep_safe"] is True
        assert diag_pqt["anchor_degradation"] <= 0.02
        assert diag_noa["anchor_degradation"] <= 0.02
        assert core_noa.config.enable_vortex is False
        assert core_pqt.config.enable_vortex is True


def test_control_fix_002_exact_sleep_budget_across_all_policies():
    """Verify that every policy executes EXACTLY idle_ticks physical integration steps (CONTROL-FIX-002).
    
    Replay controls must NOT execute 750 ticks instead of 1000, and matrix updates
    are NOT counted as integration steps.
    """
    policies = ["no_replay", "state_only", "naive_replay", "guarded_replay", "noA_dream", "PQT_dream"]
    idle_target = 1000

    for pol in policies:
        enable_v = bool("noA" not in pol)
        core = SixEchelonCore(SixEchelonConfig(enable_vortex=enable_v, integrator_mode="fixed"), seed=42)
        ledger = WitnessLedger()
        for k in range(1, 5):
            ledger.record_observation(k, k, k, k, 0.98)
            core.B[31 + k, k] = 4.0

        physical_ticks_counted = []
        orig_step = core.step

        def tracked_step(x_t=None, write_gate=1.0, dt=None):
            if write_gate == 0.0:
                physical_ticks_counted.append(1)
            return orig_step(x_t=x_t, write_gate=write_gate, dt=dt)

        core.step = tracked_step
        engine = DreamConsolidationEngine(seed=42)
        res = engine.run_sleep_policy(pol, core, ledger, idle_ticks=idle_target, dream_seed=42)
        core.step = orig_step

        # Assertion 1: Physical integration ticks measured in runtime is EXACTLY 1000
        assert len(physical_ticks_counted) == idle_target, (
            f"Policy {pol} executed {len(physical_ticks_counted)} physical ticks, expected {idle_target}!"
        )

        # Assertion 2: Reported budget metrics match exact measurement
        b_metrics = res["budget_metrics"]
        assert b_metrics["physical_integration_ticks"] == idle_target
        assert b_metrics["idle_ticks_target"] == idle_target


def test_control_fix_002_vortex_audit_strict_across_all_policies():
    """Verify that vortex audit checks actual enable_vortex and physical forces without default True."""
    audit_report = run_empirical_control_audit(seed=42)
    policies = ["no_replay", "state_only", "naive_replay", "guarded_replay", "noA_dream", "PQT_dream"]

    for pol in policies:
        emp = audit_report["policies"][pol]["empirical_checks"]
        expected_v = bool("noA" not in pol)

        # Assertion 1: Expected vortex matches configuration
        assert emp["expected_enable_vortex"] == expected_v
        assert emp["enable_vortex_flag"] == expected_v
        assert emp["all_sleep_steps_vortex_matched"] is True

        # Assertion 2: Physical force measurement
        if expected_v:
            assert emp["max_measured_vortex_force"] > 1e-3, (
                f"Policy {pol} expected active vortex, but force was {emp['max_measured_vortex_force']}"
            )
        else:
            assert emp["max_measured_vortex_force"] < 1e-15, (
                f"Policy {pol} expected ablated vortex, but force was {emp['max_measured_vortex_force']}"
            )

        # Assertion 3: Verified flag is strictly True based on measured values
        assert emp["vortex_behavior_verified"] is True


def test_control_fix_002_paired_pqt_noa_rng_fingerprints_identical():
    """Verify that paired PQT vs noA dream trials share bit-for-bit identical RNG fingerprints.
    
    Guarantees:
    - Identical initial states
    - Identical training observations
    - Identical dream seed events
    - Identical thermal noise perturbations
    The sole causal difference is A=A_I vs A=0.
    """
    for seed in CALIBRATION_SEEDS:
        core_pqt = SixEchelonCore(SixEchelonConfig(enable_vortex=True, integrator_mode="fixed"), seed=seed)
        core_noa = SixEchelonCore(SixEchelonConfig(enable_vortex=False, integrator_mode="fixed"), seed=seed)

        ledger_pqt = WitnessLedger()
        ledger_noa = WitnessLedger()

        for k in range(1, 5):
            ledger_pqt.record_observation(k, k, k, k, 0.98)
            ledger_noa.record_observation(k, k, k, k, 0.98)
            core_pqt.B[31 + k, k] = 4.0
            core_noa.B[31 + k, k] = 4.0

        engine_pqt = DreamConsolidationEngine(seed=seed)
        engine_noa = DreamConsolidationEngine(seed=seed)

        diag_pqt = engine_pqt.run_sleep_policy("PQT_dream", core_pqt, ledger_pqt, idle_ticks=100, dream_seed=seed)
        diag_noa = engine_noa.run_sleep_policy("noA_dream", core_noa, ledger_noa, idle_ticks=100, dream_seed=seed)

        fp_pqt = diag_pqt["rng_fingerprints"]
        fp_noa = diag_noa["rng_fingerprints"]

        # 1. Identical initial state hash
        assert fp_pqt["initial_state_hash"] == fp_noa["initial_state_hash"]
        assert len(fp_pqt["initial_state_hash"]) == 64

        # 2. Identical training events hash
        assert fp_pqt["training_events_hash"] == fp_noa["training_events_hash"]
        assert len(fp_pqt["training_events_hash"]) == 64

        # 3. Identical dream seed events hash
        assert fp_pqt["dream_seed_events_hash"] == fp_noa["dream_seed_events_hash"]
        assert len(fp_pqt["dream_seed_events_hash"]) == 64

        # 4. Identical dream noise stream hash
        assert fp_pqt["dream_noise_stream_hash"] == fp_noa["dream_noise_stream_hash"]
        assert len(fp_pqt["dream_noise_stream_hash"]) == 64

        # 5. Sole operational difference is vortex operator A
        assert core_pqt.config.enable_vortex is True
        assert core_noa.config.enable_vortex is False
