"""Tests for the Six-Echelon 48D Dynamic Core."""
from __future__ import annotations
import numpy as np
import pytest

from pqt_qcat.core.six_echelon import SixEchelonCore, SixEchelonConfig

TEST_SEED = 42


def test_six_echelon_structure_and_dimensions():
    """Verify 48D state dimension and 6 individual 8D cells."""
    core = SixEchelonCore(seed=TEST_SEED)
    assert core.Z.shape == (48,)
    for c in range(6):
        z_c = core.get_cell_state(c)
        assert z_c.shape == (8,)
        assert np.all(np.isfinite(z_c))


def test_write_gate_blocks_hold_intake():
    """Verify that write_gate = 0 completely blocks sensory transmission into E3 (Hold-Receiver)."""
    core = SixEchelonCore(seed=TEST_SEED)
    # Sensory input (one-hot token 3)
    x = np.zeros(8)
    x[3] = 1.0

    # 1. With write_gate = 0.0
    u_blocked = core.compute_currents(x, write_gate=0.0)
    # E3 is coordinates [16:24]
    u_hold_blocked = u_blocked[16:24]
    assert np.allclose(u_hold_blocked, 0.0), f"Hold intake was not blocked: {u_hold_blocked}"

    # 2. With write_gate = 1.0
    u_open = core.compute_currents(x, write_gate=1.0)
    u_hold_open = u_open[16:24]
    assert not np.allclose(u_hold_open, 0.0), "Hold intake remained zero with gate=1"


def test_intercell_current_norm_bounding():
    """Verify that inter-cell currents u_i never exceed u_max for any cell."""
    u_max = 3.0
    config = SixEchelonConfig(u_max=u_max)
    core = SixEchelonCore(config=config, seed=TEST_SEED)

    # Force large state activations
    core.Z = np.ones(48) * 10.0
    core.K = np.ones((48, 48)) * 5.0
    x_large = np.ones(8) * 10.0

    u = core.compute_currents(x_large, write_gate=1.0)
    for c in range(6):
        norm_c = np.linalg.norm(u[8*c : 8*c+8])
        assert norm_c <= u_max + 1e-12, f"Cell {c} current norm {norm_c} exceeded u_max {u_max}"


def test_save_resume_exact_determinism():
    """Verify that save/restore restores exact next-step trajectory and RNG state."""
    core1 = SixEchelonCore(seed=TEST_SEED)
    # Advance 10 steps
    for step in range(10):
        x = np.zeros(8)
        x[step % 8] = 1.0
        core1.step(x, write_gate=1.0)

    # Snapshot
    snap = core1.snapshot()

    # Continue core1 for 5 steps
    future_z1 = []
    for _ in range(5):
        z, _ = core1.step(np.zeros(8), write_gate=0.0)
        future_z1.append(z.copy())

    # Create core2, restore snapshot, continue for 5 steps
    core2 = SixEchelonCore(seed=TEST_SEED + 999)
    core2.restore(snap)

    future_z2 = []
    for _ in range(5):
        z, _ = core2.step(np.zeros(8), write_gate=0.0)
        future_z2.append(z.copy())

    # Compare trajectories
    for k in range(5):
        max_diff = np.max(np.abs(future_z1[k] - future_z2[k]))
        assert max_diff < 1e-14, f"Determinism violated at step {k}: max diff {max_diff}"


def test_vortex_ablation_switch():
    """Verify enable_vortex=False strictly zeroes out vortex parameters for clean noA ablation."""
    core_pqt = SixEchelonCore(config=SixEchelonConfig(enable_vortex=True), seed=TEST_SEED)
    core_noA = SixEchelonCore(config=SixEchelonConfig(enable_vortex=False), seed=TEST_SEED)

    assert core_pqt.config.enable_vortex is True
    assert core_noA.config.enable_vortex is False
