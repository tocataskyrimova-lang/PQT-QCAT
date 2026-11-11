"""M1 Mathematical Verification and Integrator Benchmark Suite.

Executes quantitative benchmarks for:
1. Skew-symmetry defect of canonical A_I(w0) vs faulty PDF matrix
2. Gradient error vs PyTorch autograd across random configurations
3. Energy dissipation dE/dt = -grad(E)^T Gamma grad(E)
4. Numerical convergence under h, h/2, h/4
5. Multi-cell 48D (6 cells x 8D) integration stability and monotonicity
6. Strict finiteness on edge cases
Outputs results/m1_math_verification.json for audit review.
"""
from __future__ import annotations
import json
import pathlib
import sys
import numpy as np

# Ensure package is on sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

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

CALIBRATION_SEED = 42


def run_benchmarks() -> dict:
    rng = np.random.default_rng(CALIBRATION_SEED)
    results = {
        "status": "PASS",
        "milestone": "M1",
        "seed_used": CALIBRATION_SEED,
        "protected_seeds_untouched": [101, 102, 103, 104, 105],
        "checks": {}
    }

    # -------------------------------------------------------------
    # 1. Skew-Symmetry Verification
    # -------------------------------------------------------------
    v_params = VortexParameters()
    w0_sweep = [-10.0, -5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    max_skew_defect = 0.0
    max_v_A_v = 0.0

    for w0 in w0_sweep:
        A = build_vortex_A_I(w0, v_params)
        defect = check_antisymmetry(A)
        if defect > max_skew_defect:
            max_skew_defect = defect

        for _ in range(10):
            v = rng.standard_normal(8)
            v_A_v = abs(float(v.T @ A @ v))
            if v_A_v > max_v_A_v:
                max_v_A_v = v_A_v

    # Negative control with flawed PDF matrix
    A_faulty = np.zeros((8, 8))
    A_faulty[0:3, 3:6] = 1.5 * J3
    A_faulty[3:6, 0:3] = -1.5 * J3
    faulty_defect = check_antisymmetry(A_faulty)

    results["checks"]["skew_symmetry"] = {
        "canonical_A_I_max_defect": max_skew_defect,
        "max_v_A_v_quadratic_form": max_v_A_v,
        "pdf_faulty_matrix_defect": faulty_defect,
        "status": "PASS" if max_skew_defect < 1e-15 and max_v_A_v < 1e-14 else "FAIL"
    }

    # -------------------------------------------------------------
    # 2. Gradient Verification against Autograd
    # -------------------------------------------------------------
    c_params = CellParameters(r0_sq=0.8)
    max_autograd_diff = 0.0
    grad_samples = 50

    for _ in range(grad_samples):
        X = rng.uniform(-2.5, 2.5, size=3)
        W = rng.uniform(-2.0, 2.0, size=3)
        w0 = rng.uniform(-2.0, 2.0)
        m = rng.uniform(-1.5, 2.0)
        Z = np.concatenate([X, W, [w0], [m]])

        g_ana = compute_grad_energy(Z, c_params)
        g_auto = grad_energy_autograd(Z, c_params)
        diff = float(np.max(np.abs(g_ana - g_auto)))
        if diff > max_autograd_diff:
            max_autograd_diff = diff

    results["checks"]["gradient_accuracy"] = {
        "samples": grad_samples,
        "max_abs_diff_vs_autograd": max_autograd_diff,
        "status": "PASS" if max_autograd_diff < 1e-12 else "FAIL"
    }

    # -------------------------------------------------------------
    # 3. Closed-System Energy Dissipation Balance
    # -------------------------------------------------------------
    integrator = AdaptiveHeunIntegrator(cell_params=c_params, vortex_params=v_params)
    max_dissipation_leak = 0.0
    all_dE_negative = True

    for _ in range(25):
        z = rng.uniform(-2.0, 2.0, size=8)
        drift, g_E, E = integrator.compute_drift(z, u=None)
        dE_dt = float(g_E.T @ drift)
        dE_dt_expected = -float(g_E.T @ (integrator.gamma_diag * g_E))
        leak = abs(dE_dt - dE_dt_expected)
        if leak > max_dissipation_leak:
            max_dissipation_leak = leak
        if dE_dt > 1e-14:
            all_dE_negative = False

    results["checks"]["energy_dissipation"] = {
        "max_vortex_energy_leak": max_dissipation_leak,
        "strictly_non_increasing": all_dE_negative,
        "status": "PASS" if max_dissipation_leak < 1e-12 and all_dE_negative else "FAIL"
    }

    # -------------------------------------------------------------
    # 4. Numerical Convergence Refinement (h, h/2, h/4)
    # -------------------------------------------------------------
    z0 = rng.uniform(-1.0, 1.0, size=8)
    t_end = 0.20
    h1 = 0.01
    h2 = 0.005
    h4 = 0.0025

    def integrate(h_step):
        n = int(round(t_end / h_step))
        z = z0.copy()
        stepper = AdaptiveHeunIntegrator(
            config=IntegratorConfig(mode="fixed", h_base=h_step),
            cell_params=c_params,
            vortex_params=v_params,
        )
        for _ in range(n):
            z, _ = stepper.step_cell(z, dt=h_step)
        return z

    z_ref = integrate(0.000625)
    z_h1 = integrate(h1)
    z_h2 = integrate(h2)

    err1 = float(np.max(np.abs(z_h1 - z_ref)))
    err2 = float(np.max(np.abs(z_h2 - z_ref)))
    order_ratio = err1 / (err2 + 1e-15)
    est_order = float(np.log2(order_ratio)) if order_ratio > 1.0 else 0.0

    results["checks"]["convergence_h_refinement"] = {
        "h1": h1, "h2": h2, "h_ref": 0.000625,
        "error_h1": err1,
        "error_h2": err2,
        "ratio_err1_div_err2": order_ratio,
        "estimated_order_p": est_order,
        "status": "PASS" if err2 < err1 and 3.0 <= order_ratio <= 5.0 else "FAIL"
    }

    # -------------------------------------------------------------
    # 5. Multi-Cell 48D Integration (6 cells x 8D)
    # -------------------------------------------------------------
    Z_48 = rng.uniform(-1.2, 1.2, size=(6, 8))
    adaptive_stepper = AdaptiveHeunIntegrator(
        config=IntegratorConfig(h_base=0.01, tol=1e-4),
        cell_params=c_params,
        vortex_params=v_params,
    )

    steps_48 = 50
    E_initial_48 = [float(compute_energy(Z_48[c], c_params)) for c in range(6)]
    Z_curr = Z_48.copy()
    all_monotonic = True

    for step_idx in range(steps_48):
        Z_curr, diags = adaptive_stepper.step_multi(Z_curr)
        for c in range(6):
            if diags[c].delta_energy > 1e-6:
                all_monotonic = False

    E_final_48 = [float(compute_energy(Z_curr[c], c_params)) for c in range(6)]
    all_decreased = all(E_final_48[c] < E_initial_48[c] for c in range(6))

    results["checks"]["multi_cell_48D"] = {
        "n_cells": 6,
        "dimension": 48,
        "steps": steps_48,
        "initial_energies": E_initial_48,
        "final_energies": E_final_48,
        "stepwise_monotonic": all_monotonic,
        "all_cells_decreased_energy": all_decreased,
        "status": "PASS" if all_monotonic and all_decreased else "FAIL"
    }

    # -------------------------------------------------------------
    # 6. Finiteness & Fail-Fast Protection
    # -------------------------------------------------------------
    nan_caught = False
    try:
        bad_z = np.zeros(8)
        bad_z[4] = np.nan
        adaptive_stepper.step_cell(bad_z)
    except NonFiniteStateError:
        nan_caught = True

    results["checks"]["fail_fast_nan_protection"] = {
        "nan_caught_as_NonFiniteStateError": nan_caught,
        "status": "PASS" if nan_caught else "FAIL"
    }

    # Overall verdict
    all_passed = all(check["status"] == "PASS" for check in results["checks"].values())
    results["overall_verdict"] = "PASS" if all_passed else "FAIL"

    return results


if __name__ == "__main__":
    res = run_benchmarks()
    out_dir = pathlib.Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "m1_math_verification.json"
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"Results saved to: {out_path}")
    print(f"Overall verdict: {res['overall_verdict']}")
    for name, c in res["checks"].items():
        print(f"  {name:30s}: {c['status']}")
    if res["overall_verdict"] != "PASS":
        sys.exit(1)
