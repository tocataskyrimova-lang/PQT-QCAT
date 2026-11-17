"""EXP-K-OBSERVABILITY-001: Numerical and Spectral Analysis of Memory Matrix K Observability.

Implements rigorous Jacobian computation, central-difference verification across multiple epsilons,
SVD spectrum analysis, accuracy guard polyhedral cone geometry, external reader decoding,
and paired causal vortex intervention (J_A vs J_A=0).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.linalg as linalg

from pqt_qcat.core.six_echelon import SixEchelonConfig, SixEchelonCore
from pqt_qcat.memory.m3_diagnostics import CLASSES, _get_quiescent_psi
from pqt_qcat.memory.exp_sleep_need_001 import (
    generate_task_panel,
    setup_fresh_core,
    wake_train_step,
    FORBIDDEN_SEEDS,
)


def compute_forward_rollout(
    core: SixEchelonCore,
    token: int,
    ticks: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Execute forward rollout for query token without mutating input core.
    
    Returns (Z_T in R^48, logits in R^4 for classes 1..4).
    """
    c = copy.deepcopy(core)
    # Start each query from quiescent psi_0 latch state
    c.Z.fill(0.0)
    for k in range(6):
        c.Z[8 * k + 6] = 1.0
        
    x = np.zeros(8, dtype=np.float64)
    x[token] = 1.0
    for _ in range(ticks):
        c.step(x, write_gate=1.0)
        
    pred_logits = c.predict()[1:5].copy()
    return c.Z.copy(), pred_logits


def compute_directional_derivative(
    core: SixEchelonCore,
    token: int,
    delta_K: np.ndarray,
    epsilon: float,
    ticks: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute central difference approximation along direction delta_K with step epsilon.
    
    Returns (dZ_approx in R^48, dlogits_approx in R^4).
    """
    c_plus = copy.deepcopy(core)
    c_plus.K += epsilon * delta_K
    Z_plus, logits_plus = compute_forward_rollout(c_plus, token, ticks=ticks)
    
    c_minus = copy.deepcopy(core)
    c_minus.K -= epsilon * delta_K
    Z_minus, logits_minus = compute_forward_rollout(c_minus, token, ticks=ticks)
    
    dZ = (Z_plus - Z_minus) / (2.0 * epsilon)
    dlogits = (logits_plus - logits_minus) / (2.0 * epsilon)
    return dZ, dlogits


def compute_full_jacobians_numerical(
    core: SixEchelonCore,
    token: int,
    epsilon: float = 1e-4,
    ticks: int = 8,
    active_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute numerical Jacobian S_q (48 x 2304) and J_q (4 x 2304) via central differences.
    
    If active_mask (48 x 48 boolean) is provided, only evaluates active entries; others set to 0.
    """
    S_q = np.zeros((48, 2304), dtype=np.float64)
    J_q = np.zeros((4, 2304), dtype=np.float64)
    
    Z_base, logits_base = compute_forward_rollout(core, token, ticks=ticks)
    
    # Iterate over parameters
    for i in range(48):
        for j in range(48):
            param_idx = i * 48 + j
            if active_mask is not None and not active_mask[i, j]:
                continue
                
            delta_K = np.zeros((48, 48), dtype=np.float64)
            delta_K[i, j] = 1.0
            
            dZ, dlogits = compute_directional_derivative(core, token, delta_K, epsilon, ticks=ticks)
            S_q[:, param_idx] = dZ
            J_q[:, param_idx] = dlogits
            
    return S_q, J_q


def get_active_echelon_mask() -> np.ndarray:
    """Boolean mask of active inter-cell synapses in SixEchelon architecture."""
    mask = np.zeros((48, 48), dtype=bool)
    mask[16:24, 0:8] = True    # E1 -> E3
    mask[24:32, 16:24] = True  # E3 -> E4
    mask[32:40, 24:32] = True  # E4 -> E5
    mask[40:48, 8:16] = True   # E2 -> E6
    mask[32:40, 40:48] = True  # E6 -> E5
    return mask


def analyze_svd_spectrum(J_stack: np.ndarray) -> dict[str, Any]:
    """Perform exact SVD spectrum analysis of stacked Jacobian J_stack (24 x 2304)."""
    U, s, Vt = linalg.svd(J_stack, full_matrices=False)
    
    s_max = float(s[0]) if len(s) > 0 else 0.0
    s_min = float(s[-1]) if len(s) > 0 else 0.0
    cond_num = float(s_max / s_min) if s_min > 1e-15 else float("inf")
    
    # Numerical rank with threshold max(shape) * eps * s_max
    tol = max(J_stack.shape) * np.finfo(np.float64).eps * s_max
    rank = int(np.sum(s > tol))
    null_dim = J_stack.shape[1] - rank
    
    energy_90 = int(np.searchsorted(np.cumsum(s**2) / np.sum(s**2), 0.90)) + 1
    energy_99 = int(np.searchsorted(np.cumsum(s**2) / np.sum(s**2), 0.99)) + 1
    
    return {
        "shape": list(J_stack.shape),
        "singular_values": [float(x) for x in s],
        "s_max": s_max,
        "s_min": s_min,
        "condition_number": cond_num,
        "numerical_rank": rank,
        "null_space_dimension": null_dim,
        "effective_rank_90_energy": energy_90,
        "effective_rank_99_energy": energy_99,
        "frobenius_norm": float(linalg.norm(J_stack, "fro")),
    }


def analyze_accuracy_guard_geometry(
    core: SixEchelonCore,
    J_stack: np.ndarray,
    task_map: dict[int, int],
    n_samples: int = 50,
) -> dict[str, Any]:
    """Examine polyhedral cone of the accuracy guard vs linear nullspace."""
    U, s, Vt = linalg.svd(J_stack, full_matrices=True)
    null_basis = Vt[24:, :] # (2280 x 2304) basis of nullspace
    row_basis = Vt[:24, :]  # (24 x 2304) basis of row space
    
    rng = np.random.default_rng(12345)
    
    # 1. Nullspace Perturbations: Delta logits must be identically 0
    null_max_dlogits = []
    null_acc_retained = 0
    scales = [0.01, 0.1, 0.5, 1.0]
    
    for scale in scales:
        for _ in range(n_samples):
            # Sample random direction in nullspace
            coeffs = rng.standard_normal(null_basis.shape[0])
            coeffs /= np.linalg.norm(coeffs)
            delta_theta = scale * (coeffs @ null_basis)
            delta_K = delta_theta.reshape(48, 48)
            
            c_pert = copy.deepcopy(core)
            c_pert.K += delta_K
            
            dlog_max = 0.0
            all_correct = True
            for q in range(1, 7):
                _, orig_log = compute_forward_rollout(core, q)
                _, pert_log = compute_forward_rollout(c_pert, q)
                dlog = np.max(np.abs(pert_log - orig_log))
                if dlog > dlog_max:
                    dlog_max = dlog
                if int(np.argmax(pert_log)) + 1 != task_map[q]:
                    all_correct = False
            null_max_dlogits.append(dlog_max)
            if all_correct:
                null_acc_retained += 1
                
    # 2. Rowspace (Orthogonal) Perturbations: Test distance to decision boundary
    row_acc_by_scale = {}
    for scale in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
        correct_count = 0
        for _ in range(n_samples):
            coeffs = rng.standard_normal(row_basis.shape[0])
            coeffs /= np.linalg.norm(coeffs)
            delta_theta = scale * (coeffs @ row_basis)
            delta_K = delta_theta.reshape(48, 48)
            
            c_pert = copy.deepcopy(core)
            c_pert.K += delta_K
            all_correct = True
            for q in range(1, 7):
                _, pert_log = compute_forward_rollout(c_pert, q)
                if int(np.argmax(pert_log)) + 1 != task_map[q]:
                    all_correct = False
            if all_correct:
                correct_count += 1
        row_acc_by_scale[str(scale)] = correct_count / n_samples
        
    return {
        "null_space_samples_tested": len(null_max_dlogits),
        "null_space_max_delta_logits": float(np.max(null_max_dlogits)),
        "null_space_mean_delta_logits": float(np.mean(null_max_dlogits)),
        "null_space_accuracy_retention": float(null_acc_retained / len(null_max_dlogits)),
        "row_space_accuracy_retention_by_scale": row_acc_by_scale,
    }


def run_paired_vortex_intervention(core: SixEchelonCore) -> dict[str, Any]:
    """Compare Jacobians under paired vortex intervention: J_A vs J_A=0."""
    mask = get_active_echelon_mask()
    
    # 1. With vortex (A enabled)
    c_A = copy.deepcopy(core)
    c_A.config = SixEchelonConfig(enable_vortex=True, integrator_mode="fixed")
    
    J_list_A = []
    for q in range(1, 7):
        _, J_q = compute_full_jacobians_numerical(c_A, q, epsilon=1e-4, active_mask=mask)
        J_list_A.append(J_q)
    J_stack_A = np.vstack(J_list_A)
    
    # 2. Without vortex (A=0 ablated)
    c_noA = copy.deepcopy(core)
    c_noA.config = SixEchelonConfig(enable_vortex=False, integrator_mode="fixed")
    
    J_list_noA = []
    for q in range(1, 7):
        _, J_q = compute_full_jacobians_numerical(c_noA, q, epsilon=1e-4, active_mask=mask)
        J_list_noA.append(J_q)
    J_stack_noA = np.vstack(J_list_noA)
    
    svd_A = analyze_svd_spectrum(J_stack_A)
    svd_noA = analyze_svd_spectrum(J_stack_noA)
    
    norm_A = float(linalg.norm(J_stack_A, "fro"))
    norm_noA = float(linalg.norm(J_stack_noA, "fro"))
    frob_ratio = float(norm_A / norm_noA) if norm_noA > 0 else float("inf")
    
    return {
        "frobenius_norm_A": norm_A,
        "frobenius_norm_noA": norm_noA,
        "frobenius_ratio_A_over_noA": frob_ratio,
        "svd_A": svd_A,
        "svd_noA": svd_noA,
    }


def evaluate_external_reader_decoding(
    trained_cores: list[dict[str, Any]],
    n_splits: int = 5,
) -> dict[str, Any]:
    """Train external linear ridge classifier on vec(K) to decode task labels with label-shuffle control."""
    from sklearn.linear_model import RidgeClassifier
    
    X = []
    y = []
    
    for item in trained_cores:
        K_vec = item["K"].flatten()
        X.append(K_vec)
        # Classify target for token 1 as primary test
        y.append(item["task_map"][1])
        
    X = np.array(X)
    y = np.array(y)
    
    if len(np.unique(y)) < 2 or len(X) < n_splits:
        return {"status": "insufficient_samples_for_external_reader", "n_samples": len(X)}
        
    rng = np.random.default_rng(42)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    
    fold_size = len(X) // n_splits
    true_accs = []
    shuff_accs = []
    
    for fold in range(n_splits):
        test_idx = indices[fold * fold_size : (fold + 1) * fold_size]
        train_idx = np.setdiff1d(indices, test_idx)
        
        clf_true = RidgeClassifier(alpha=1.0)
        clf_true.fit(X[train_idx], y[train_idx])
        pred_true = clf_true.predict(X[test_idx])
        true_accs.append(float(np.mean(pred_true == y[test_idx])))
        
        # Label shuffle control
        clf_shuff = RidgeClassifier(alpha=1.0)
        y_shuff = rng.permutation(y[train_idx])
        clf_shuff.fit(X[train_idx], y_shuff)
        pred_shuff = clf_shuff.predict(X[test_idx])
        shuff_accs.append(float(np.mean(pred_shuff == y[test_idx])))
        
    return {
        "n_samples": len(X),
        "n_splits": n_splits,
        "mean_reader_accuracy": float(np.mean(true_accs)),
        "mean_shuffled_baseline_accuracy": float(np.mean(shuff_accs)),
        "chance_level": 0.25,
        "excess_decodability": float(np.mean(true_accs) - np.mean(shuff_accs)),
    }


def main():
    parser = argparse.ArgumentParser(description="Run EXP-K-OBSERVABILITY-001 analysis.")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else Path(f"results/exploratory/K_OBSERVABILITY_001/RUN_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=================================================================")
    print("EXP-K-OBSERVABILITY-001: MEMORY MATRIX K OBSERVABILITY ANALYSIS")
    print("=================================================================\n")
    
    # 1. Setup trained core from seed 42 on task map 0
    panel = generate_task_panel(generator_seed=170700, total_maps=1)
    task_map = panel[0]
    core = setup_fresh_core(seed=42, enable_vortex=True)
    psi_0 = _get_quiescent_psi()
    
    # Train core on tokens 1..3 for 25 epochs
    for _ in range(25):
        for tok in [1, 2, 3]:
            wake_train_step(core, tok, task_map[tok], psi_0, eta_K=0.8, delta_core_max=0.5)
            
    # 2. Central difference verification across epsilons
    print("--- 1. CENTRAL DIFFERENCE JACOBIAN VERIFICATION ---")
    rng = np.random.default_rng(999)
    test_epsilons = [1e-2, 1e-3, 1e-4, 1e-5]
    diff_results = {}
    
    for eps in test_epsilons:
        v_rand = rng.standard_normal((48, 48))
        v_rand /= np.linalg.norm(v_rand)
        
        dZ_num, dlog_num = compute_directional_derivative(core, 1, v_rand, epsilon=eps)
        diff_results[str(eps)] = {
            "norm_dZ": float(np.linalg.norm(dZ_num)),
            "norm_dlogits": float(np.linalg.norm(dlog_num)),
            "dlogits": [float(x) for x in dlog_num],
        }
        print(f"  Epsilon {eps:e}: ||dlogits|| = {np.linalg.norm(dlog_num):.6f}")
        
    # 3. Full Jacobian computation on active synapse mask
    print("\n--- 2. COMPUTING SYSTEM STACKED JACOBIAN J_stack (24 x 2304) ---")
    t0 = time.perf_counter()
    mask = get_active_echelon_mask()
    active_count = int(np.sum(mask))
    print(f"Active inter-cell synapses in echelon mask: {active_count} of 2304 (p = {active_count})")
    
    J_queries = []
    for q in range(1, 7):
        S_q, J_q = compute_full_jacobians_numerical(core, q, epsilon=1e-4, active_mask=mask)
        J_queries.append(J_q)
    J_stack = np.vstack(J_queries)
    print(f"J_stack computed in {time.perf_counter() - t0:.2f}s. Shape: {J_stack.shape}")
    
    # 4. SVD Spectrum Analysis
    print("\n--- 3. SVD SPECTRUM & NULL-SPACE ANALYSIS ---")
    svd_res = analyze_svd_spectrum(J_stack)
    print(f"  Condition number: {svd_res['condition_number']:.2e}")
    print(f"  Numerical rank: {svd_res['numerical_rank']} / 24")
    print(f"  Nullspace dimension: {svd_res['null_space_dimension']} / 2304")
    print(f"  Top 5 singular values: {[round(x, 4) for x in svd_res['singular_values'][:5]]}")
    print(f"  Bottom 3 singular values: {[round(x, 6) for x in svd_res['singular_values'][-3:]]}")
    
    # 5. Accuracy Guard Geometry (Polyhedral Cone vs Nullspace)
    print("\n--- 4. ACCURACY GUARD GEOMETRY (CONE VS NULLSPACE) ---")
    geom_res = analyze_accuracy_guard_geometry(core, J_stack, task_map, n_samples=30)
    print(f"  Nullspace perturbations tested: {geom_res['null_space_samples_tested']}")
    print(f"  Nullspace max delta logits: {geom_res['null_space_max_delta_logits']:.2e}")
    print(f"  Nullspace accuracy retention: {geom_res['null_space_accuracy_retention']*100:.1f}%")
    print(f"  Rowspace accuracy retention by scale: {geom_res['row_space_accuracy_retention_by_scale']}")
    
    # 6. Paired Vortex Intervention (J_A vs J_noA)
    print("\n--- 5. PAIRED VORTEX CAUSALITY INTERVENTION (A vs noA) ---")
    vortex_res = run_paired_vortex_intervention(core)
    print(f"  ||J_stack||_F (A enabled): {vortex_res['frobenius_norm_A']:.4f}")
    print(f"  ||J_stack||_F (A=0 ablated): {vortex_res['frobenius_norm_noA']:.4f}")
    print(f"  Sensitivity ratio (A / noA): {vortex_res['frobenius_ratio_A_over_noA']:.4f}")
    print(f"  Rank A: {vortex_res['svd_A']['numerical_rank']} | Rank noA: {vortex_res['svd_noA']['numerical_rank']}")
    
    # 7. External Reader on K
    print("\n--- 6. EXTERNAL READER DECODABILITY ON K ---")
    # Collect 12 trained cores across panel mappings
    panel_12 = generate_task_panel(generator_seed=170700, total_maps=12)
    trained_cores_dataset = []
    for idx, tmap in enumerate(panel_12):
        c_train = setup_fresh_core(seed=100 + idx, enable_vortex=True)
        for _ in range(25):
            for tok in [1, 2, 3]:
                wake_train_step(c_train, tok, tmap[tok], psi_0, eta_K=0.8, delta_core_max=0.5)
        trained_cores_dataset.append({"K": c_train.K.copy(), "task_map": tmap})
        
    reader_res = evaluate_external_reader_decoding(trained_cores_dataset, n_splits=4)
    print(f"  External Reader Mean Accuracy: {reader_res.get('mean_reader_accuracy', 0.0):.4f}")
    print(f"  Label-Shuffle Baseline: {reader_res.get('mean_shuffled_baseline_accuracy', 0.0):.4f}")
    print(f"  Excess Decodability: {reader_res.get('excess_decodability', 0.0):.4f}")
    
    # Save complete artifacts
    full_results = {
        "protocol_id": "EXP-K-OBSERVABILITY-001",
        "timestamp": timestamp,
        "verification_epsilons": diff_results,
        "svd_spectrum": svd_res,
        "accuracy_guard_geometry": geom_res,
        "vortex_intervention": vortex_res,
        "external_reader": reader_res,
    }
    
    with open(out_dir / "k_observability_results.json", "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
        
    print(f"\nCompleted EXP-K-OBSERVABILITY-001. All results saved to {out_dir}")


if __name__ == "__main__":
    main()
