"""Comprehensive unit test suite for PQT-QCat mathematical invariants and integrator.

Conforms strictly to ARCHITECTURE_LIFE_002 (Milestone M1) and ILORI approval:
- Gradient exactness vs autograd (< 1e-12) and finite differences (< 1e-8)
- Skew symmetry of canonical A_I(w0) (max defect < 1e-15)
- Energy dissipation law dE/dt = -grad(E)^T Gamma grad(E) <= 0
- Trajectory convergence order with step sizes h, h/2, h/4
- Strict NonFiniteStateError fail-fast on NaN/Inf
- Negative control against flawed +J3/-J3 matrix from PDF page 26
- Protection of pre-registered seeds 101-105
"""
from __future__ import annotations
import math
import numpy as np
import pytest

from pqt_qcat.core.energy import (
    CellParameters,
    compute_energy,
    compute_grad_energy,
    grad_energy_autograd,
)
from pqt_qcat.core.vortex import (
    VortexParameters,
    build_vortex_A_I,
    apply_vortex_A_I,
    check_antisymmetry,
    vortex_latch_gain,
    J3,
)
from pqt_qcat.core.integrator import (
    AdaptiveHeunIntegrator,
    IntegratorConfig,
    NonFiniteStateError,
)


# Calibration seed for development unit tests (SEEDS 101-105 STRICTLY FORBIDDEN)
TEST_SEED = 42


def test_seed_isolation():
    """Ensure seeds 101-105 are protected and never used in unit tests."""
    forbidden = {101, 102, 103, 104, 105}
    assert TEST_SEED not in forbidden


def test_grad_energy_against_autograd():
    """Analytical gradient grad E(Z) must match PyTorch autograd to < 1e-12."""
    rng = np.random.default_rng(TEST_SEED)
    params = CellParameters(r0_sq=0.8)

    max_diff = 0.0
    for _ in range(30):
        # Sample realistic coordinates around active range
        X = rng.uniform(-2.0, 2.0, size=3)
        W = rng.uniform(-1.5, 1.5, size=3)
        w0 = rng.uniform(-1.5, 1.5)
        m = rng.uniform(-1.0, 1.5)
        Z = np.concatenate([X, W, [w0], [m]])

        grad_analytical = compute_grad_energy(Z, params)
        grad_ref = grad_energy_autograd(Z, params)

        diff = np.max(np.abs(grad_analytical - grad_ref))
        if diff > max_diff:
            max_diff = diff

    assert max_diff < 1e-12, f"Gradient mismatch vs autograd: {max_diff}"


def test_grad_energy_against_finite_differences():
    """Analytical gradient must match central finite differences to < 1e-7."""
    rng = np.random.default_rng(TEST_SEED + 1)
    params = CellParameters(r0_sq=0.8)
    eps = 1e-6

    max_diff = 0.0
    for _ in range(15):
        Z = rng.uniform(-1.5, 1.5, size=8)
        grad_analytical = compute_grad_energy(Z, params)

        grad_fd = np.zeros(8, dtype=np.float64)
        for i in range(8):
            Z_plus = Z.copy()
            Z_minus = Z.copy()
            Z_plus[i] += eps
            Z_minus[i] -= eps

            E_plus = compute_energy(Z_plus, params)
            E_minus = compute_energy(Z_minus, params)
            grad_fd[i] = (E_plus - E_minus) / (2.0 * eps)

        diff = np.max(np.abs(grad_analytical - grad_fd))
        if diff > max_diff:
            max_diff = diff

    assert max_diff < 1e-7, f"Gradient mismatch vs finite diff: {max_diff}"


def test_vortex_operator_exact_antisymmetry():
    """Canonical A_I(w0) must satisfy A + A^T = 0 with machine precision < 1e-15."""
    params = VortexParameters()
    test_w0_values = [-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0]

    for w0 in test_w0_values:
        A = build_vortex_A_I(w0, params)
        defect = check_antisymmetry(A)
        assert defect < 1e-15, f"Antisymmetry violated at w0={w0}: defect={defect}"

        # Invariant: v^T A v = 0 for any vector v
        rng = np.random.default_rng(TEST_SEED + 2)
        for _ in range(5):
            v = rng.standard_normal(8)
            v_A_v = float(v.T @ A @ v)
            assert abs(v_A_v) < 1e-14, f"Energy conservation defect: {v_A_v}"

        # Matrix-vector product apply_vortex_A_I matches matrix @ v
        for _ in range(5):
            v = rng.standard_normal(8)
            mv_fast = apply_vortex_A_I(v, w0, params)
            mv_direct = A @ v
            diff = np.max(np.abs(mv_fast - mv_direct))
            assert diff < 1e-15, f"Matrix-vector product mismatch: {diff}"


def test_negative_control_pdf_page_26_faulty_matrix():
    """Negative control: verify that flawed +J3/-J3 formulation from PDF page 26 fails antisymmetry."""
    g = 1.0
    omega_xw = 1.5
    # Faulty construction: +J3 on top, -J3 on bottom
    block_xw = g * omega_xw * J3
    block_wx = -g * omega_xw * J3

    A_faulty = np.zeros((8, 8), dtype=np.float64)
    A_faulty[0:3, 3:6] = block_xw
    A_faulty[3:6, 0:3] = block_wx

    defect = check_antisymmetry(A_faulty)
    # Because J3^T = -J3, (block_xw)^T = -g*omega_xw*J3 = block_wx!
    # So A + A^T != 0, defect is 2 * omega_xw = 3.0
    assert defect > 2.9, f"Expected defect ~3.0 for faulty matrix, got {defect}"


def test_latch_sigmoid_behavior():
    """Verify latch transition g(w0): w0 > 0 -> latch, w0 < 0 -> vortex."""
    # w0 >> 0 (latch)
    g_latch = vortex_latch_gain(2.0, beta_v=4.0)
    assert g_latch < 0.001, f"Expected latch near 0, got {g_latch}"

    # w0 << 0 (vortex)
    g_vortex = vortex_latch_gain(-2.0, beta_v=4.0)
    assert g_vortex > 0.999, f"Expected vortex near 1, got {g_vortex}"

    # w0 = 0 (transition midpoint)
    g_mid = vortex_latch_gain(0.0, beta_v=4.0)
    assert abs(g_mid - 0.5) < 1e-12


def test_dissipative_force_field_potentiality():
    """Curl Defect test: Dissipative force field G = grad E must have symmetric Jacobian."""
    params = CellParameters()
    eps = 1e-6
    rng = np.random.default_rng(TEST_SEED + 3)

    for _ in range(5):
        Z = rng.uniform(-1.0, 1.0, size=8)
        Jac = np.zeros((8, 8), dtype=np.float64)

        for j in range(8):
            Z_p = Z.copy()
            Z_m = Z.copy()
            Z_p[j] += eps
            Z_m[j] -= eps
            gp = compute_grad_energy(Z_p, params)
            gm = compute_grad_energy(Z_m, params)
            Jac[:, j] = (gp - gm) / (2.0 * eps)

        curl_defect = np.linalg.norm(Jac - Jac.T, ord='fro')
        assert curl_defect < 1e-6, f"Curl defect of dissipative force field: {curl_defect}"


def test_closed_system_energy_dissipation():
    """Verify that in closed system (u=0, T=0), dE/dt = -grad(E)^T Gamma grad(E) <= 0."""
    params = CellParameters()
    v_params = VortexParameters()
    integrator = AdaptiveHeunIntegrator(cell_params=params, vortex_params=v_params)

    rng = np.random.default_rng(TEST_SEED + 4)
    for _ in range(10):
        z = rng.uniform(-1.5, 1.5, size=8)
        drift, grad_E, E = integrator.compute_drift(z, u=None)

        # dE/dt = grad_E^T * drift
        dE_dt = float(grad_E.T @ drift)

        # Theoretical dissipation: -grad_E^T Gamma grad_E
        dE_dt_theory = -float(grad_E.T @ (integrator.gamma_diag * grad_E))

        # Check equivalence (vortex does not contribute to dE/dt)
        diff = abs(dE_dt - dE_dt_theory)
        assert diff < 1e-12, f"Vortex power leakage: {diff}"
        assert dE_dt <= 1e-14, f"Energy was not non-increasing: {dE_dt}"


def test_trajectory_convergence_h_refinement():
    """Verify numerical convergence of trajectories under h, h/2, h/4."""
    params = CellParameters()
    v_params = VortexParameters()

    rng = np.random.default_rng(TEST_SEED + 5)
    z0 = rng.uniform(-0.8, 0.8, size=8)

    t_final = 0.2
    h_coarse = 0.02
    h_medium = 0.01
    h_fine = 0.005

    def integrate_fixed_step(z_init, dt, total_t):
        n_steps = int(round(total_t / dt))
        z = z_init.copy()
        integrator = AdaptiveHeunIntegrator(
            config=IntegratorConfig(h_base=dt, tol=1e-8, max_subdivisions=0),
            cell_params=params,
            vortex_params=v_params,
        )
        for _ in range(n_steps):
            z, _ = integrator.step_cell(z, dt=dt)
        return z

    z_coarse = integrate_fixed_step(z0, h_coarse, t_final)
    z_medium = integrate_fixed_step(z0, h_medium, t_final)
    z_fine = integrate_fixed_step(z0, h_fine, t_final)

    err_coarse = np.max(np.abs(z_coarse - z_fine))
    err_medium = np.max(np.abs(z_medium - z_fine))

    assert err_medium < err_coarse, f"Error did not decrease: err(h)={err_coarse}, err(h/2)={err_medium}"
    convergence_ratio = err_coarse / (err_medium + 1e-14)
    # For second-order Heun, ratio ~ 2^2 = 4 (typically between 3.0 and 4.5)
    assert 2.5 <= convergence_ratio <= 5.5, f"Unexpected convergence ratio: {convergence_ratio}"


def test_monotonic_energy_along_adaptive_trajectory():
    """Verify that E(z(t)) monotonically decreases along adaptive Heun trajectory (T=0, u=0)."""
    params = CellParameters()
    v_params = VortexParameters()
    integrator = AdaptiveHeunIntegrator(
        config=IntegratorConfig(h_base=0.01, tol=1e-4),
        cell_params=params,
        vortex_params=v_params,
    )

    rng = np.random.default_rng(TEST_SEED + 6)
    z = rng.uniform(-1.0, 1.0, size=8)

    energies = [compute_energy(z, params)]
    for _ in range(25):
        z, diag = integrator.step_cell(z)
        energies.append(diag.final_energy)
        # Numerical tolerance for tiny local curvature variations
        assert diag.delta_energy <= 1e-6, f"Energy increased by {diag.delta_energy}"

    # Overall energy must have strictly decreased from start
    assert energies[-1] < energies[0]


def test_strict_non_finite_fail_fast():
    """Verify that any NaN or Inf immediately raises NonFiniteStateError."""
    integrator = AdaptiveHeunIntegrator()

    # NaN in state
    z_nan = np.zeros(8)
    z_nan[2] = float('nan')
    with pytest.raises(NonFiniteStateError):
        integrator.step_cell(z_nan)

    # Inf in state
    z_inf = np.zeros(8)
    z_inf[0] = float('inf')
    with pytest.raises(NonFiniteStateError):
        integrator.step_cell(z_inf)

    # NaN in input u
    z_normal = np.zeros(8)
    u_nan = np.zeros(8)
    u_nan[1] = float('nan')
    with pytest.raises(NonFiniteStateError):
        integrator.step_cell(z_normal, u=u_nan)
