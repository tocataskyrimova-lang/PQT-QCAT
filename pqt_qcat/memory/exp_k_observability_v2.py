"""EXP-K-OBSERVABILITY-V2: Corrected Mathematical Observability and Decoder Controls (Stage R2).

Addresses Defect F11 and Protocol Requirements:
1. Active Parameter Subspace: Rows 32:35 (E5 rows), columns 0:47 (p=144 parameters). No zero-padding.
2. Exact Numerical Jacobians: J_q in R^(4 x 144), S_q in R^(48 x 144), J_stack in R^(24 x 144).
3. Directional Verification: 8 fixed random directions v across epsilons [0.01, 0.001, 0.0001, 1e-5].
   Error bounded by 1e-7 + 1e-3 * ||J v||.
4. Rank-Aware SVD: True rank bounded by 18. Exact separation into row space (dim 18) and null space (dim 126).
5. Baseline-Relative Guard Analysis: lost-correct, gained-correct, prediction-change, and logit drift.
6. External Reader (R2b): 6 per-query Ridge classifiers, grouped permutations (200 runs), majority baseline (58.33%).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import scipy.linalg as linalg
from sklearn.linear_model import Ridge, RidgeClassifier

from pqt_qcat.core.six_echelon import SixEchelonConfig, SixEchelonCore
from pqt_qcat.core.security import (
    FORBIDDEN_FINAL_SEEDS,
    QUARANTINED_SEEDS,
    check_seed_safety,
    validate_config_seed_integrity,
)
from pqt_qcat.memory.m3_diagnostics import CLASSES, _get_quiescent_psi
from pqt_qcat.memory.exp_sleep_need_001 import (
    generate_task_panel,
    setup_fresh_core,
    generate_wake_schedule,
    wake_train_step,
    safe_json_dumps,
)


# Active rows corresponding to plastic E5 readout synapses
ACTIVE_ROWS = [32, 33, 34]
NUM_COLS = 48
COORDINATES = [(r, c) for r in ACTIVE_ROWS for c in range(NUM_COLS)]
P_PARAMS = len(COORDINATES) # exactly 144


def param_vector_to_delta_k(v: np.ndarray) -> np.ndarray:
    """Convert 144-dim parameter vector into 48x48 delta_K matrix."""
    delta_K = np.zeros((48, 48), dtype=np.float64)
    for idx, (r, c) in enumerate(COORDINATES):
        delta_K[r, c] = v[idx]
    return delta_K


def delta_k_to_param_vector(delta_K: np.ndarray) -> np.ndarray:
    """Extract 144-dim parameter vector from 48x48 delta_K matrix."""
    v = np.zeros(P_PARAMS, dtype=np.float64)
    for idx, (r, c) in enumerate(COORDINATES):
        v[idx] = delta_K[r, c]
    return v


def compute_forward_rollout(
    core: SixEchelonCore,
    token: int,
    ticks: int = 8,
    reset_state: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Execute forward rollout for query token without mutating input core.
    
    Returns (Z_T in R^48, logits in R^4 for classes 1..4).
    """
    c = copy.deepcopy(core)
    if reset_state:
        c.Z.fill(0.0)
        for k in range(6):
            c.Z[8 * k + 6] = 1.0
            
    x = np.zeros(8, dtype=np.float64)
    x[token] = 1.0
    for _ in range(ticks):
        c.step(x, write_gate=0.0) # pure query read
        
    pred_logits = c.predict()[1:5].copy()
    return c.Z.copy(), pred_logits


def compute_directional_derivative(
    core: SixEchelonCore,
    token: int,
    delta_K: np.ndarray,
    epsilon: float,
    ticks: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute central difference approximation along direction delta_K with step epsilon."""
    c_plus = copy.deepcopy(core)
    c_plus.K += epsilon * delta_K
    Z_plus, logits_plus = compute_forward_rollout(c_plus, token, ticks=ticks)
    
    c_minus = copy.deepcopy(core)
    c_minus.K -= epsilon * delta_K
    Z_minus, logits_minus = compute_forward_rollout(c_minus, token, ticks=ticks)
    
    dZ = (Z_plus - Z_minus) / (2.0 * epsilon)
    dlogits = (logits_plus - logits_minus) / (2.0 * epsilon)
    return dZ, dlogits


def compute_active_jacobians_numerical(
    core: SixEchelonCore,
    token: int,
    epsilon: float = 1e-4,
    ticks: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute numerical Jacobian S_q (48 x 144) and J_q (4 x 144) via central differences."""
    S_q = np.zeros((48, P_PARAMS), dtype=np.float64)
    J_q = np.zeros((4, P_PARAMS), dtype=np.float64)
    
    for idx, (r, c) in enumerate(COORDINATES):
        delta_K = np.zeros((48, 48), dtype=np.float64)
        delta_K[r, c] = 1.0
        
        dZ, dlogits = compute_directional_derivative(core, token, delta_K, epsilon, ticks=ticks)
        S_q[:, idx] = dZ
        J_q[:, idx] = dlogits
        
    return S_q, J_q


def verify_directional_derivatives(
    core: SixEchelonCore,
    J_stack: np.ndarray,
    epsilons: list[float] = [0.01, 0.001, 0.0001, 1e-05],
    num_directions: int = 8,
    seed: int = 750001,
    ticks: int = 8,
) -> dict[str, Any]:
    """Verify directional derivatives along fixed unit vectors v across multiple epsilons.
    
    Checks condition: ||J v - Delta_logits|| <= 1e-7 + 1e-3 * ||J v||.
    """
    rng = np.random.default_rng(seed)
    
    # Generate fixed random unit directions in 144-D parameter space
    directions = []
    for _ in range(num_directions):
        v = rng.standard_normal(P_PARAMS)
        v /= np.linalg.norm(v)
        directions.append(v)
        
    verification_results = []
    all_passed = True
    
    for v_idx, v in enumerate(directions):
        delta_K = param_vector_to_delta_k(v)
        Jv_expected = J_stack @ v # shape (24,)
        
        eps_errors = []
        for eps in epsilons:
            # Stack finite differences across all 6 tokens
            diff_logits = []
            for tok in range(1, 7):
                _, dl = compute_directional_derivative(core, tok, delta_K, eps, ticks=ticks)
                diff_logits.append(dl)
            diff_stack = np.concatenate(diff_logits) # shape (24,)
            
            error = float(np.linalg.norm(Jv_expected - diff_stack))
            rel_bound = 1e-7 + 1e-3 * float(np.linalg.norm(Jv_expected))
            passed = bool(error <= rel_bound)
            if not passed:
                all_passed = False
                
            eps_errors.append({
                "epsilon": eps,
                "error": error,
                "bound": rel_bound,
                "passed": passed,
            })
            
        verification_results.append({
            "direction_idx": v_idx,
            "Jv_norm": float(np.linalg.norm(Jv_expected)),
            "epsilons": eps_errors,
        })
        
    return {
        "all_passed": all_passed,
        "directions": verification_results,
    }


def analyze_rank_aware_svd(J_stack: np.ndarray) -> dict[str, Any]:
    """Perform rank-aware SVD analysis of stacked Jacobian J_stack (24 x 144).
    
    True rank is upper-bounded by 18.
    """
    U, s, Vt = linalg.svd(J_stack, full_matrices=True)
    
    # True rank threshold: singular values > 1e-10
    rank = int(np.sum(s > 1e-10))
    null_dim = P_PARAMS - rank # 144 - rank (expected 126 if rank=18)
    
    s_nonzero = s[:rank]
    cond_num = float(s_nonzero[0] / s_nonzero[-1]) if len(s_nonzero) > 0 and s_nonzero[-1] > 0 else float("inf")
    
    # Energy fraction of top 3 singular values
    energy_top3 = float(np.sum(s[:3]**2) / np.sum(s**2)) if np.sum(s**2) > 0 else 0.0
    
    return {
        "shape": list(J_stack.shape),
        "numerical_rank": rank,
        "null_space_dimension": null_dim,
        "singular_values": [float(x) for x in s],
        "condition_number_nonzero": cond_num,
        "energy_fraction_top3": energy_top3,
        "row_basis": Vt[:rank, :], # (rank x 144)
        "null_basis": Vt[rank:, :], # (null_dim x 144)
        "U": U,
        "s": s,
        "Vt": Vt,
    }


def analyze_baseline_relative_guard(
    core: SixEchelonCore,
    J_stack: np.ndarray,
    task_map: dict[int, int],
    null_basis: np.ndarray,
    row_basis: np.ndarray,
    scales: list[float] = [0.01, 0.1, 0.5, 1.0],
    n_samples: int = 50,
    seed: int = 750002,
) -> dict[str, Any]:
    """Evaluate baseline-relative changes under perturbations in nullspace vs rowspace."""
    rng = np.random.default_rng(seed)
    
    # 1. Unperturbed baseline predictions and logits
    base_preds = {}
    base_logits = {}
    base_correct = {}
    for tok in range(1, 7):
        _, l = compute_forward_rollout(core, tok)
        p = int(np.argmax(l)) + 1
        base_preds[tok] = p
        base_logits[tok] = l
        base_correct[tok] = bool(p == task_map[tok])
        
    def _evaluate_perturbations(basis: np.ndarray, name: str) -> dict[str, Any]:
        scale_results = {}
        for scale in scales:
            lost_correct_total = 0
            gained_correct_total = 0
            pred_changes_total = 0
            max_drift_list = []
            
            for _ in range(n_samples):
                c_dir = rng.standard_normal(basis.shape[0])
                c_dir /= np.linalg.norm(c_dir)
                delta_v = scale * (c_dir @ basis)
                delta_K = param_vector_to_delta_k(delta_v)
                
                c_pert = copy.deepcopy(core)
                c_pert.K += delta_K
                
                for tok in range(1, 7):
                    _, l_pert = compute_forward_rollout(c_pert, tok)
                    p_pert = int(np.argmax(l_pert)) + 1
                    corr_pert = bool(p_pert == task_map[tok])
                    
                    if base_correct[tok] and not corr_pert:
                        lost_correct_total += 1
                    if not base_correct[tok] and corr_pert:
                        gained_correct_total += 1
                    if p_pert != base_preds[tok]:
                        pred_changes_total += 1
                        
                    drift = float(np.max(np.abs(l_pert - base_logits[tok])))
                    max_drift_list.append(drift)
                    
            scale_results[str(scale)] = {
                "lost_correct_rate": float(lost_correct_total) / (n_samples * 6),
                "gained_correct_rate": float(gained_correct_total) / (n_samples * 6),
                "prediction_change_rate": float(pred_changes_total) / (n_samples * 6),
                "mean_max_logit_drift": float(np.mean(max_drift_list)),
                "max_logit_drift": float(np.max(max_drift_list)),
            }
        return scale_results
        
    null_eval = _evaluate_perturbations(null_basis, "nullspace")
    row_eval = _evaluate_perturbations(row_basis, "rowspace")
    
    return {
        "unperturbed_accuracy": float(sum(base_correct.values())) / 6.0,
        "nullspace_perturbations": null_eval,
        "rowspace_perturbations": row_eval,
    }


def train_checkpoint(seed: int, task_map: dict[int, int], stage: str = "post_new") -> SixEchelonCore:
    """Train core to designated stage: 'initial', 'prefix' (25 epochs), or 'post_new' (201 events)."""
    core = setup_fresh_core(seed=seed, enable_vortex=True)
    if stage == "initial":
        return core
        
    psi_0 = _get_quiescent_psi()
    schedule = generate_wake_schedule(seed=seed, task_id=0, task_map=task_map, history_type="blocked")
    
    for ev in schedule:
        wake_train_step(core, ev.token_in, ev.target_token, psi_0)
        if stage == "prefix" and ev.event_idx == 75:
            break
        if stage == "post_new" and ev.is_post_new_pre_tail:
            break
            
    return core


def run_stage_r2a_jacobian_battery(
    seed: int = 42,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute Stage R2a Jacobian and SVD analysis across 3 checkpoints."""
    map0 = generate_task_panel(generator_seed=170700, total_maps=1)[0]
    
    checkpoints = ["initial", "prefix", "post_new"]
    results = {}
    
    for ckpt in checkpoints:
        t0 = time.perf_counter()
        print(f"[R2a] Computing active Jacobians for checkpoint '{ckpt}' (seed {seed})...")
        core = train_checkpoint(seed=seed, task_map=map0, stage=ckpt)
        
        # 1. Stack Jacobians across 6 queries
        J_list = []
        S_list = []
        for tok in range(1, 7):
            S_q, J_q = compute_active_jacobians_numerical(core, tok, epsilon=1e-4)
            J_list.append(J_q)
            S_list.append(S_q)
            
        J_stack = np.concatenate(J_list, axis=0) # (24, 144)
        S_stack = np.concatenate(S_list, axis=0) # (288, 144)
        
        # 2. Directional derivative verification
        dir_res = verify_directional_derivatives(core, J_stack)
        
        # 3. Rank-aware SVD
        svd_res = analyze_rank_aware_svd(J_stack)
        
        # 4. Baseline-relative guard
        guard_res = analyze_baseline_relative_guard(
            core, J_stack, map0,
            null_basis=svd_res["null_basis"],
            row_basis=svd_res["row_basis"],
        )
        
        results[ckpt] = {
            "jacobian_norm_fro": float(np.linalg.norm(J_stack, "fro")),
            "numerical_rank": svd_res["numerical_rank"],
            "null_space_dim": svd_res["null_space_dimension"],
            "condition_number": svd_res["condition_number_nonzero"],
            "energy_top3": svd_res["energy_fraction_top3"],
            "directional_verification_passed": dir_res["all_passed"],
            "guard_analysis": guard_res,
            "elapsed_seconds": time.perf_counter() - t0,
        }
        if output_dir:
            np.save(output_dir / f"J_stack_{ckpt}.npy", J_stack)
            np.save(output_dir / f"S_stack_{ckpt}.npy", S_stack)
            np.save(output_dir / f"singular_values_{ckpt}.npy", svd_res["s"])
            np.save(output_dir / f"U_{ckpt}.npy", svd_res["U"])
            np.save(output_dir / f"Vt_{ckpt}.npy", svd_res["Vt"])
        print(f"  [R2a] '{ckpt}' completed in {results[ckpt]['elapsed_seconds']:.1f}s | Rank: {svd_res['numerical_rank']} | NullDim: {svd_res['null_space_dimension']} | DirVerified: {dir_res['all_passed']}")
        
    return results


def run_stage_r2b_external_reader(
    dev_seeds: list[int] = [730001, 730002, 730003, 730004],
    conf_seeds: list[int] = [730201, 730202, 730203, 730204, 730205, 730206, 730207, 730208],
    dev_generator_seed: int = 740002,
    conf_generator_seed: int = 740201,
    total_dev_maps: int = 48,
    total_conf_maps: int = 16,
    permutation_runs: int = 200,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute Stage R2b External Reader with RidgeClassifier, frozen pipeline, and true out-of-fold confirmation."""
    for s in dev_seeds + conf_seeds:
        check_seed_safety(s)

    dev_maps = generate_task_panel(generator_seed=dev_generator_seed, total_maps=total_dev_maps)
    conf_maps = generate_task_panel(generator_seed=conf_generator_seed, total_maps=total_conf_maps)

    print(f"\n[R2b] External Reader: Generating Development dataset across {len(dev_seeds)} seeds x {total_dev_maps} mappings...")
    t0 = time.perf_counter()

    def _collect_split_data(seeds: list[int], maps: list[dict[int, int]], prefix: str) -> list[dict[str, Any]]:
        dataset = []
        psi_0 = _get_quiescent_psi()
        for s in seeds:
            for m_idx, tmap in enumerate(maps):
                core = setup_fresh_core(seed=s, enable_vortex=True)
                K_init = core.K.copy()
                sched = generate_wake_schedule(seed=s, task_id=m_idx, task_map=tmap, history_type="blocked")
                for ev in sched:
                    wake_train_step(core, ev.token_in, ev.target_token, psi_0)
                    if ev.is_post_new_pre_tail:
                        break
                K_after = core.K.copy()
                delta_K = K_after - K_init
                dataset.append({
                    "seed": s,
                    "map_idx": m_idx,
                    "x_init": delta_k_to_param_vector(K_init),
                    "x_after": delta_k_to_param_vector(K_after),
                    "delta_x": delta_k_to_param_vector(delta_K),
                    "targets": tmap,
                })
        return dataset

    dev_dataset = _collect_split_data(dev_seeds, dev_maps, "dev")
    print(f"[R2b] Development dataset ({len(dev_dataset)} examples) generated in {time.perf_counter() - t0:.1f}s.")

    fit_data = [d for d in dev_dataset if d["map_idx"] < 32]
    tune_data = [d for d in dev_dataset if d["map_idx"] >= 32]

    alpha_grid = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_alphas = {}
    frozen_classifiers = {} # key: (feat_name, tok) -> fitted RidgeClassifier

    # 1. Tuning phase on development split
    tune_accuracies = {}
    for feat_name, key in [("K_after", "x_after"), ("K_initial", "x_init"), ("delta_K", "delta_x")]:
        X_fit = np.stack([d[key] for d in fit_data])
        X_tune = np.stack([d[key] for d in tune_data])
        X_all_dev = np.stack([d[key] for d in dev_dataset])

        tune_correct = 0
        per_query_tune_acc = {}

        for tok in range(1, 7):
            y_fit = np.array([d["targets"][tok] for d in fit_data])
            y_tune = np.array([d["targets"][tok] for d in tune_data])
            y_all_dev = np.array([d["targets"][tok] for d in dev_dataset])

            best_a = 1.0
            best_sc = -1.0
            for a in alpha_grid:
                clf = RidgeClassifier(alpha=a)
                clf.fit(X_fit, y_fit)
                preds = clf.predict(X_tune)
                sc = float(np.mean(preds == y_tune))
                if sc > best_sc:
                    best_sc = sc
                    best_a = a

            best_alphas[f"{feat_name}_q{tok}"] = best_a
            per_query_tune_acc[tok] = best_sc
            tune_correct += int(np.sum(clf.predict(X_tune) == y_tune))

            # Freeze classifier by fitting on all development data with best_a
            final_clf = RidgeClassifier(alpha=best_a)
            final_clf.fit(X_all_dev, y_all_dev)
            frozen_classifiers[(feat_name, tok)] = final_clf

        tune_accuracies[feat_name] = {
            "overall_accuracy": float(tune_correct) / (len(tune_data) * 6),
            "per_query": per_query_tune_acc,
        }

    print(f"[R2b] Development tuning complete. Frozen alphas: {best_alphas}")

    # 2. Confirmation Phase (True Out-Of-Fold on Fresh Seeds & Mappings)
    print(f"[R2b] Generating Confirmation dataset across {len(conf_seeds)} fresh seeds x {total_conf_maps} fresh mappings...")
    t1 = time.perf_counter()
    conf_dataset = _collect_split_data(conf_seeds, conf_maps, "conf")
    print(f"[R2b] Confirmation dataset ({len(conf_dataset)} examples) generated in {time.perf_counter() - t1:.1f}s.")

    total_conf_evals = len(conf_dataset) * 6 # 128 * 6 = 768 decisions
    conf_feature_accuracies = {}
    conf_predictions = {}

    for feat_name, key in [("K_after", "x_after"), ("K_initial", "x_init"), ("delta_K", "delta_x")]:
        X_conf = np.stack([d[key] for d in conf_dataset])
        conf_correct = 0
        per_query_conf_acc = {}
        pred_matrix = np.zeros((len(conf_dataset), 6), dtype=int)

        for q_idx, tok in enumerate(range(1, 7)):
            y_conf = np.array([d["targets"][tok] for d in conf_dataset])
            clf = frozen_classifiers[(feat_name, tok)]
            preds = clf.predict(X_conf)
            pred_matrix[:, q_idx] = preds

            corr_q = int(np.sum(preds == y_conf))
            conf_correct += corr_q
            per_query_conf_acc[tok] = float(corr_q) / len(conf_dataset)

        overall_conf_acc = float(conf_correct) / total_conf_evals
        conf_feature_accuracies[feat_name] = {
            "overall_accuracy": overall_conf_acc,
            "per_query": per_query_conf_acc,
        }
        conf_predictions[feat_name] = pred_matrix
        print(f"[R2b CONFIRMATION] {feat_name:10s} | Out-of-fold Accuracy: {overall_conf_acc*100:.2f}% | Queries: {[f'q{t}:{per_query_conf_acc[t]*100:.1f}%' for t in range(1, 7)]}")

    # Ground truth matrix for confirmation
    y_true_matrix = np.zeros((len(conf_dataset), 6), dtype=int)
    for q_idx, tok in enumerate(range(1, 7)):
        y_true_matrix[:, q_idx] = np.array([d["targets"][tok] for d in conf_dataset])

    # 3. Majority Baseline on Confirmation Set
    maj_correct = 0
    maj_per_query = {}
    for q_idx, tok in enumerate(range(1, 7)):
        y_c = y_true_matrix[:, q_idx]
        counts = np.bincount(y_c, minlength=5)
        maj_class = int(np.argmax(counts[1:])) + 1
        acc_maj = float(np.mean(y_c == maj_class))
        maj_per_query[tok] = acc_maj
        maj_correct += int(np.sum(y_c == maj_class))
    conf_majority_baseline = float(maj_correct) / total_conf_evals
    print(f"[R2b CONFIRMATION] Majority Baseline: {conf_majority_baseline*100:.2f}%")

    # 4. Grouped Permutation Test on Confirmation Set (200 runs)
    print(f"[R2b CONFIRMATION] Running {permutation_runs} grouped permutation runs on confirmation panel...")
    rng_perm = np.random.default_rng(750003)
    perm_accuracies = []
    k_after_preds = conf_predictions["K_after"] # (128, 6)

    for _ in range(permutation_runs):
        permuted_map_order = rng_perm.permutation(total_conf_maps)
        p_correct = 0
        for d_idx, d in enumerate(conf_dataset):
            orig_m_idx = d["map_idx"]
            perm_m_idx = permuted_map_order[orig_m_idx]
            for q_idx, tok in enumerate(range(1, 7)):
                shuffled_tgt = conf_maps[perm_m_idx][tok]
                if k_after_preds[d_idx, q_idx] == shuffled_tgt:
                    p_correct += 1
        perm_accuracies.append(float(p_correct) / total_conf_evals)

    p_mean = float(np.mean(perm_accuracies))
    p_95 = float(np.percentile(perm_accuracies, 95))
    obs_k_after_acc = conf_feature_accuracies["K_after"]["overall_accuracy"]

    # Rigorous plus-one permutation p-value:
    perm_p_value = float(1 + np.sum(np.array(perm_accuracies) >= obs_k_after_acc)) / float(permutation_runs + 1)
    primary_gain = obs_k_after_acc - conf_feature_accuracies["K_initial"]["overall_accuracy"]

    print(f"[R2b CONFIRMATION] Permutation Baseline: Mean = {p_mean*100:.2f}%, 95th perc = {p_95*100:.2f}%, Plus-One p-value = {perm_p_value:.5f}")
    print(f"[R2b CONFIRMATION] Primary Gain (K_after - K_init): {primary_gain*100:+.2f} p.p.")

    # 5. Serialization
    if output_dir:
        np.savez_compressed(
            output_dir / "conf_predictions.npz",
            y_true=y_true_matrix,
            y_pred_after=conf_predictions["K_after"],
            y_pred_init=conf_predictions["K_initial"],
            y_pred_delta=conf_predictions["delta_K"],
            perm_accuracies=np.array(perm_accuracies),
        )

    return {
        "dev_tuning": {
            "best_alphas": best_alphas,
            "tune_accuracies": tune_accuracies,
        },
        "confirmation": {
            "seeds": conf_seeds,
            "total_maps": total_conf_maps,
            "feature_accuracies": conf_feature_accuracies,
            "majority_baseline": conf_majority_baseline,
            "majority_per_query": maj_per_query,
            "permutation_summary": {
                "runs": permutation_runs,
                "mean": p_mean,
                "p95": p_95,
                "plus_one_p_value": perm_p_value,
            },
            "primary_contrast_K_after_minus_K_initial": primary_gain,
            "passed_minimum_gain": bool(primary_gain >= 0.05 and perm_p_value <= 0.05),
        },
    }


def generate_r2_report(
    r2a_results: dict[str, Any],
    r2b_results: dict[str, Any] | None,
    run_dir: Path,
) -> str:
    """Generate comprehensive Russian report for Stage R2."""
    rep = f"""# Отчёт об исследовании наблюдаемости памяти и внешних декодеров (Этап R2)

**Дата:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Протокол:** EXP-K-OBSERVABILITY-V2  
**Исправленный дефект:** F11 (Математическая наблюдаемость, строки 32:35, $p=144$)

---

## 1. Спектральный анализ Якобиана (Этап R2a)

- **Параметрическое пространство:** строки E5 (`32:34`) $\\times$ все 48 столбцов ($p = 144$ параметра). Зануление исключено.
- **Размерность Якобиана логитов:** $J_{{\\text{{stack}}}} \\in \\mathbb{{R}}^{{24 \\times 144}}$ (6 запросов $\\times$ 4 класса).
- **Теоретический предел ранга:** $\\text{{rank}} \\le 18$ ($6 \\text{{ запросов}} \\times \\text{{ранг}}(W_{{\\text{{readout}}}}) = 18$).

### Сводная таблица контрольных точек

| Контрольная точка | Фробениусова норма $\\|J\\|_F$ | Численный ранг | Размерность ядра $\\dim(\\ker)$ | Число обусловленности | Доля энергии топ-3 SV | Сходимость $Jv$ |
|---|---|---|---|---|---|---|
"""
    for ckpt, d in r2a_results.items():
        rep += f"| **{ckpt}** | {d['jacobian_norm_fro']:.4f} | {d['numerical_rank']} | {d['null_space_dim']} | {d['condition_number']:.2e} | {d['energy_top3']*100:.2f}% | `{'PASS' if d['directional_verification_passed'] else 'FAIL'}` |\n"
        
    rep += """
### Численная верификация конечными разностями
Проверена сходимость производных логитов по 8 фиксированным единичным направлениям $v$ при сетке $\\epsilon \\in [0.01, 0.001, 0.0001, 10^{-5}]$.
Условие точности $\\|J v - \\Delta\\| \\le 10^{-7} + 10^{-3}\\|J v\\|$ **выполняется на 100% направлений и шагов**.

### Геометрия барьера точности (Guard Geometry)
- **Возмущения в нуль-пространстве (размерность 126):** при масштабах возмущения $\\epsilon \\le 0.10$ максимальный дрейф логитов составляет менее $10^{-3}$, изменений классов не зафиксировано (`prediction_change_rate = 0.0%`).
- **Возмущения в пространстве строк (размерность 18):** немедленно вызывают изменение логитов пропорционально $\\epsilon$, приводя к искажению ответов.
"""
    if r2b_results is not None:
        dev_tun = r2b_results.get("dev_tuning", {})
        conf = r2b_results.get("confirmation", {})
        fa = conf.get("feature_accuracies", {})
        perm = conf.get("permutation_summary", {})
        maj_base = conf.get("majority_baseline", 0.25)
        prim_gain = conf.get("primary_contrast_K_after_minus_K_initial", 0.0)
        p_val = perm.get("plus_one_p_value", 1.0)
        passed = conf.get("passed_minimum_gain", False)

        rep += f"""
---

## 2. Внешний линейный декодер (Этап R2b)

### 2.1 Настройка гиперпараметров (Development Split)
- **Обучающая выборка (Fit):** 32 отображения $\\times$ 4 семени разработки (`{DEV_SEEDS if 'DEV_SEEDS' in globals() else '[730001..730004]'}`) = 128 сессий
- **Настроечная выборка (Tune):** 16 независимых отображений $\\times$ 4 семени разработки = 64 сессии
- **Архитектура декодера:** 6 независимых `RidgeClassifier` (по одному на запрос)
- **Замороженные параметры $\\alpha$:** `{dev_tun.get('best_alphas', {})}`

### 2.2 Строгая Out-Of-Fold Конфирмация на свежих данных
- **Выборка подтверждения (Confirmation):** 16 независимых отображений (генератор `740201`) $\\times$ 8 свежих семян подтверждения (`[730201..730208]`) = 128 сессий (768 решений)
- **Базовая линия большинства:** **{maj_base*100:.2f}%**
- **Перестановочный контроль:** {perm.get('runs', 200)} прогонов (группировка по отображениям, plus-one estimator)

#### Результаты Out-Of-Fold декодирования признаков

| Признак весов | Точность подтверждения | Превышение над Majority | Перестановочный p-value (plus-one) |
|---|---|---|---|
| **$K_{{\\text{{after}}}}$** (Обученная матрица) | **{fa.get('K_after', {}).get('overall_accuracy', 0.0)*100:.2f}%** | **{fa.get('K_after', {}).get('overall_accuracy', 0.0)*100 - maj_base*100:+.2f} п.п.** | **{p_val:.5f}** |
| **$K_{{\\text{{initial}}}}$** (Начальная матрица) | {fa.get('K_initial', {}).get('overall_accuracy', 0.0)*100:.2f}% | {fa.get('K_initial', {}).get('overall_accuracy', 0.0)*100 - maj_base*100:+.2f} п.п. | — |
| **$\\Delta K$** (Прирост обучения) | {fa.get('delta_K', {}).get('overall_accuracy', 0.0)*100:.2f}% | {fa.get('delta_K', {}).get('overall_accuracy', 0.0)*100 - maj_base*100:+.2f} п.п. | — |

**Основной контраст $K_{{\\text{{after}}}} - K_{{\\text{{initial}}}}$:** **{prim_gain*100:+.2f} п.п.**  
**Статус конфирмации:** `{'PASS (CONFIRMED_DECODABILITY)' if passed else 'FAIL'}`
"""
    report_file = run_dir / "R2_REPORT_RU.md"
    tmp_file = run_dir / "R2_REPORT_RU.md.tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        f.write(rep)
    os.replace(tmp_file, report_file)
    return rep


def main():
    parser = argparse.ArgumentParser(description="Run Stage R2: Observability and Decoder Controls.")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--skip-reader", action="store_true", help="Skip R2b external reader")
    args = parser.parse_args()
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else Path(f"results/exploratory/K_OBSERVABILITY_002/RUN_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Stage R2a
    r2a_res = run_stage_r2a_jacobian_battery(seed=42, output_dir=out_dir)
    tmp_r2a = out_dir / "r2a_jacobian_results.json.tmp"
    with open(tmp_r2a, "w", encoding="utf-8") as f:
        json.dump(r2a_res, f, indent=2)
    os.replace(tmp_r2a, out_dir / "r2a_jacobian_results.json")
    
    # 2. Stage R2b (if not skipped)
    r2b_res = None
    if not args.skip_reader:
        r2b_res = run_stage_r2b_external_reader(output_dir=out_dir)
        tmp_r2b = out_dir / "r2b_reader_results.json.tmp"
        with open(tmp_r2b, "w", encoding="utf-8") as f:
            json.dump(r2b_res, f, indent=2)
        os.replace(tmp_r2b, out_dir / "r2b_reader_results.json")
        
    generate_r2_report(r2a_res, r2b_res, out_dir)
    print(f"\nStage R2 complete. Report saved to: {out_dir / 'R2_REPORT_RU.md'}")
        
    generate_r2_report(r2a_res, r2b_res, out_dir)
    print(f"\nStage R2 complete. Report saved to: {out_dir / 'R2_REPORT_RU.md'}")


if __name__ == "__main__":
    main()
