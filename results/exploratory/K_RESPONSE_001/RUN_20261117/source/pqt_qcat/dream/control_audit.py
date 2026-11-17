"""Dynamic Empirical Audit of Control Models (CONTROL-FIX-001).

Implements exhaustive empirical checks across all 6 sleep policies:
- 'no_replay': passive baseline, u_ext = 0, delta_theta = 0
- 'state_only': thermal baseline, u_ext = 0, delta_theta = 0
- 'naive_replay': negative control, replay without anchor guard
- 'guarded_replay': replay of real events with anchor guard protection
- 'noA_dream': counterfactual dream with A=0 throughout entire trial
- 'PQT_dream': counterfactual dream with A=A_I and 3-gate validation

All compliance verdicts ('COMPLIANT' vs 'NON_COMPLIANT') are strictly COMPUTED
from measured runtime properties, never assigned statically.
"""
from __future__ import annotations
import numpy as np

from ..core.six_echelon import SixEchelonCore, SixEchelonConfig
from ..core.vortex import VortexParameters
from ..memory.witness import WitnessLedger, ObservedEvent, AnchorGuardGate
from .consolidation import DreamConsolidationEngine, ConsolidationConfig

CLASSES = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [-1.0, -1.0, 0.0],
], dtype=np.float64)


def audit_single_policy(policy: str, seed: int = 42, idle_ticks: int = 1000) -> dict:
    """Empirically evaluate and audit a single sleep policy against all control invariants."""
    enable_vortex_init = bool("PQT" in policy or policy not in ("noA_dream", "noA_dream_K"))
    cfg = SixEchelonConfig(enable_vortex=enable_vortex_init, integrator_mode="fixed")
    core = SixEchelonCore(config=cfg, seed=seed)

    # Set up fixed readout and anchors
    core.W_readout.fill(0.0)
    for c in range(1, 5):
        core.W_readout[c, 32:35] = CLASSES[c]
    initial_W = core.W_readout.copy()

    for k in range(1, 5):
        core.B[31 + k, k] = 4.0
    initial_B = core.B.copy()
    initial_K = core.K.copy()

    ledger = WitnessLedger()
    for k in range(1, 5):
        ledger.record_observation(
            timestamp=k,
            input_token=k,
            target_token=k,
            pred_token=k,
            confidence=0.98,
            context_basin=1,
        )

    # Step tracking instrumentation
    calls_recorded: list[tuple[object, float]] = []
    vortex_states_during_sleep: list[bool] = []
    original_step = core.step

    def tracked_step(x_t=None, write_gate=1.0, dt=None):
        calls_recorded.append((None if x_t is None else np.asarray(x_t, dtype=np.float64).copy(), float(write_gate)))
        if write_gate == 0.0:
            vortex_states_during_sleep.append(bool(core.config.enable_vortex))
        return original_step(x_t=x_t, write_gate=write_gate, dt=dt)

    core.step = tracked_step

    engine = DreamConsolidationEngine(
        ConsolidationConfig(anchor_max_degradation=0.02, dream_cycles=25),
        seed=seed,
    )

    # Execute sleep policy
    res = engine.run_sleep_policy(policy=policy, core=core, ledger=ledger, idle_ticks=idle_ticks)
    core.step = original_step  # restore

    # 1. Check sensory drive and physical sleep budget during sleep (calls where write_gate == 0.0)
    # Under protocol, idle sleep periods must have strictly zero sensory input
    sleep_calls = [c for c in calls_recorded if c[1] == 0.0]
    sleep_calls_with_sensory = [c for c in sleep_calls if c[0] is not None and np.any(c[0] != 0.0)]
    sensory_drive_strictly_zero = bool(len(sleep_calls_with_sensory) == 0)

    # EXACT PHYSICAL BUDGET: Every policy must execute exactly idle_ticks physical integration steps (CONTROL-FIX-002)
    physical_integration_ticks = len(sleep_calls)
    step_budget_exact = bool(physical_integration_ticks == idle_ticks)

    # 2. Check readout strictly frozen
    readout_strictly_frozen = bool(np.array_equal(core.W_readout, initial_W))
    delta_W_norm = float(np.linalg.norm(core.W_readout - initial_W))

    # 3. Check vortex configuration and forces (CONTROL-FIX-002: no default vortex_verified=True)
    expected_enable_vortex = bool("noA" not in policy)
    actual_enable_vortex = bool(core.config.enable_vortex)
    all_sleep_steps_match = bool(
        len(vortex_states_during_sleep) > 0
        and all(v == expected_enable_vortex for v in vortex_states_during_sleep)
    )

    z_test = np.array([0.8, -0.5, 0.4, 0.8, -0.4, 0.5, -0.5, 0.2], dtype=np.float64)
    vortex_forces = []
    for c in range(6):
        core.Z[8 * c : 8 * c + 8] = z_test.copy()
        vortex_forces.append(core.compute_cell_vortex_force(c))

    max_vortex_force = float(np.max([np.linalg.norm(f) for f in vortex_forces]))

    if expected_enable_vortex:
        vortex_verified = bool(
            actual_enable_vortex is True
            and all_sleep_steps_match
            and max_vortex_force > 1e-3
        )
    else:
        vortex_verified = bool(
            actual_enable_vortex is False
            and all_sleep_steps_match
            and max_vortex_force < 1e-15
        )

    # 4. Check gate behavior and anchor guard
    anchors = ledger.get_anchors()
    guard = AnchorGuardGate(max_degradation=0.02)
    acc_current = guard.evaluate_anchors(core, anchors)

    # Test shadow rollback capability on core
    shadow_passed, _, _ = guard.check_and_apply(
        core,
        anchors,
        proposed_delta_K=np.zeros_like(core.K),
        proposed_delta_B=-core.B.copy(),  # destructive update
    )
    rollback_works = bool(shadow_passed is False and guard.evaluate_anchors(core, anchors) == acc_current)

    if policy in ("guarded_replay", "noA_dream", "PQT_dream"):
        gate_behavior_verified = bool(
            res["passed_h3_sleep_safe"] is True
            and res["anchor_degradation"] <= 0.02
            and rollback_works
        )
    elif policy in ("no_replay", "state_only"):
        gate_behavior_verified = bool(
            res["delta_K_norm"] == 0.0
            and res["delta_B_norm"] == 0.0
            and res["anchor_degradation"] == 0.0
        )
    elif policy == "naive_replay":
        gate_behavior_verified = bool(res["accepted_updates"] > 0)
    else:
        gate_behavior_verified = False

    # 5. Check provenance isolation
    try:
        ledger.assert_no_unverified_promotions()
        provenance_isolated = True
    except AssertionError:
        provenance_isolated = False

    # COMPUTED COMPLIANCE (strictly logical conjunction of measured evidence)
    compliant = bool(
        step_budget_exact
        and sensory_drive_strictly_zero
        and readout_strictly_frozen
        and vortex_verified
        and gate_behavior_verified
        and provenance_isolated
    )

    return {
        "policy": policy,
        "empirical_checks": {
            "sleep_ticks_recorded": len(sleep_calls),
            "idle_ticks_budget": idle_ticks,
            "step_budget_exact": step_budget_exact,
            "sensory_drive_strictly_zero": sensory_drive_strictly_zero,
            "readout_strictly_frozen": readout_strictly_frozen,
            "delta_W_norm": delta_W_norm,
            "expected_enable_vortex": expected_enable_vortex,
            "enable_vortex_flag": core.config.enable_vortex,
            "all_sleep_steps_vortex_matched": all_sleep_steps_match,
            "max_measured_vortex_force": max_vortex_force,
            "vortex_behavior_verified": vortex_verified,
            "anchor_accuracy_before": res["anchor_accuracy_before"],
            "anchor_accuracy_after": res["anchor_accuracy_after"],
            "measured_anchor_degradation": res["anchor_degradation"],
            "rollback_mechanism_verified": rollback_works,
            "gate_behavior_verified": gate_behavior_verified,
            "provenance_isolated_verified": provenance_isolated,
            "delta_K_norm": res["delta_K_norm"],
            "delta_B_norm": res["delta_B_norm"],
            "budget_metrics": res.get("budget_metrics", {}),
            "rng_fingerprints": res.get("rng_fingerprints", {}),
        },
        "audit_verdict": "COMPLIANT" if compliant else "NON_COMPLIANT",
    }


def run_empirical_control_audit(seed: int = 42) -> dict:
    """Run empirical compliance audit across all 6 pre-registered sleep policies."""
    policies = ["no_replay", "state_only", "naive_replay", "guarded_replay", "noA_dream", "PQT_dream"]
    results = {}
    all_compliant = True

    for pol in policies:
        audit_res = audit_single_policy(pol, seed=seed)
        if audit_res["audit_verdict"] != "COMPLIANT":
            all_compliant = False
        results[pol] = audit_res

    return {
        "audit_type": "EMPIRICAL_RUNTIME_VERIFICATION",
        "calibration_seed": seed,
        "all_policies_compliant": all_compliant,
        "overall_control_verdict": "COMPLIANT" if all_compliant else "NON_COMPLIANT",
        "policies": results,
    }
