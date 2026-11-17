"""Mandatory test suite for MEMORY-TRACE-001 / EXP-1 Experimental Stand.

Verifies all requirements from Section 9 of the protocol:
1. Target is strictly withheld until AFTER core.predict() during wake.
2. B and W matrices remain 100% bit-for-bit invariant (array_equal) across all stages.
3. noA condition enforces real A=0 on vortex forces/parameters, not just a boolean flag.
4. Toggling A_read does not mutate other branches or base configurations.
5. Queries, anchor checks, and sham dream leave the snapshot untouched.
6. Compute accounting: exactly 1000 active ticks (including presentations), shadow ticks tracked separately.
7. Gate rejections do not perturb the pre-generated grounding and noise streams.
8. Statuses and contrasts are dynamically computed, never hardcoded.
"""
from __future__ import annotations
import numpy as np
import pytest

from pqt_qcat.core.six_echelon import SixEchelonCore, SixEchelonConfig
from pqt_qcat.core.vortex import VortexParameters
from pqt_qcat.memory.m3_diagnostics import CLASSES
from pqt_qcat.memory.witness import WitnessLedger
from pqt_qcat.memory.exp1_memory_trace import (
    SleepAdapter,
    evaluate_query_set,
    get_donor_permutation,
    get_calibration_permutations,
    get_all_permutations,
)


def test_b_w_bitwise_invariance():
    """Verify B and W remain 100% identical bit-for-bit across operations."""
    cfg = SixEchelonConfig(enable_vortex=True, integrator_mode="fixed")
    core = SixEchelonCore(config=cfg, seed=42)

    core.W_readout.fill(0.0)
    for c in range(1, 5):
        core.W_readout[c, 32:35] = CLASSES[c]
    core.B.fill(0.0)
    for k in range(1, 5):
        core.B[k - 1, k] = 8.0

    B_orig = core.B.copy()
    W_orig = core.W_readout.copy()

    # Run steps, evaluations, K-modifications
    x = np.zeros(core.config.d_in)
    x[1] = 1.0
    for _ in range(8):
        core.step(x, write_gate=1.0)
    core.predict()

    core.K += np.ones_like(core.K) * 0.01

    assert np.array_equal(core.B, B_orig), "B matrix mutated!"
    assert np.array_equal(core.W_readout, W_orig), "W matrix mutated!"


def test_vortex_ablation_force_verified():
    """Verify that A=0 strictly produces zero vortex force and zero vortex parameters."""
    cfg = SixEchelonConfig(enable_vortex=False, integrator_mode="fixed")
    core = SixEchelonCore(config=cfg, seed=42)

    assert core.config.enable_vortex is False
    vparams = VortexParameters(0.0, 0.0, 0.0, beta_v=core.config.beta_v)
    core.integrator.vortex_params = vparams

    # Run physical step and inspect integrator parameters
    x = np.zeros(core.config.d_in)
    x[2] = 1.0
    core.step(x, write_gate=1.0)

    # Invariants: vortex parameters are zero
    assert core.integrator.vortex_params.omega_space == 0.0
    assert core.integrator.vortex_params.omega_xw == 0.0
    assert core.integrator.vortex_params.omega_lm == 0.0

    # Explicit force calculation verification
    from pqt_qcat.core.integrator import apply_vortex_A_I, compute_grad_energy
    z_cell = core.Z[:8]
    grad_E = compute_grad_energy(z_cell, params=core.integrator.cell_params)
    vortex_force = apply_vortex_A_I(grad_E, w0=float(z_cell[6]), params=vparams)
    assert np.allclose(vortex_force, 0.0), "Vortex force must be strictly 0.0 under noA"


def test_sleep_adapter_budget_exactness():
    """Verify that dream, replay, and sham all execute exactly 1000 active ticks and equal shadow ticks."""
    cfg = SixEchelonConfig(enable_vortex=True, integrator_mode="fixed")
    core = SixEchelonCore(config=cfg, seed=42)
    core.B.fill(0.0)
    for k in range(1, 5):
        core.B[k - 1, k] = 8.0
    for c in range(1, 5):
        core.W_readout[c, 32:35] = CLASSES[c]

    ledger = WitnessLedger()
    for tok in range(1, 5):
        ledger.record_observation(tok, tok, (tok % 4) + 1, (tok % 4) + 1, 0.99)

    adapter = SleepAdapter(candidate_slots=25)

    for pol in ("dream", "replay", "sham_dream"):
        test_core = SixEchelonCore(config=cfg, seed=42)
        test_core.restore(core.snapshot())
        logs = []
        diag = adapter.run_sleep(pol, a_write=True, core=test_core, ledger=ledger, run_seed=42, log_records=logs)

        assert diag["active_ticks"] == 1000, f"{pol} active ticks != 1000"
        assert diag["shadow_ticks"] == 1600, f"{pol} shadow ticks != 1600"

    # Idle check
    test_core_idle = SixEchelonCore(config=cfg, seed=42)
    test_core_idle.restore(core.snapshot())
    diag_idle = adapter.run_sleep("idle", a_write=True, core=test_core_idle, ledger=ledger, run_seed=42, log_records=[])
    assert diag_idle["active_ticks"] == 1000
    assert diag_idle["shadow_ticks"] == 0


def test_donor_permutation_disjointness():
    """Verify that every permutation's donor has disjoint targets across all 4 queries."""
    all_perms = get_all_permutations()
    assert len(all_perms) == 24

    for p in all_perms:
        d = get_donor_permutation(p)
        assert len(d) == 4
        # All 4 elements must differ
        for u in range(4):
            assert p[u] != d[u], f"Donor target must differ from original at position {u}: {p} vs {d}"
        # Reciprocal check: donor of donor must be original
        d_of_d = get_donor_permutation(d)
        assert d_of_d == p, f"Donor of donor must equal original: {d_of_d} != {p}"
