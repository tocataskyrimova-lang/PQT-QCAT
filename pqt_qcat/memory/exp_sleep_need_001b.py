"""EXP-SLEEP-NEED-001B: Wake-Only B-Code Overlap Calibration Experiment.

Investigates partially overlapping B-codes between old and new tokens with equal norm 8.0.
Decouples selection seeds [11, 12] from held-out validation seeds [13, 42].
Strictly enforces no sealed seeds 101-105 and run_m5=False.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.stats as stats

from pqt_qcat.core.six_echelon import SixEchelonConfig, SixEchelonCore
from pqt_qcat.core.security import FORBIDDEN_FINAL_SEEDS, validate_config_seed_integrity
from pqt_qcat.memory.exp_sleep_need_001 import (
    CLASSES,
    ObservedEvent,
    generate_task_panel,
    generate_wake_schedule,
    query_core_prediction,
    evaluate_query_accuracy,
    wake_train_step,
    FORBIDDEN_SEEDS,
)
from pqt_qcat.memory.m3_diagnostics import _get_quiescent_psi


def build_overlapping_B(rho: float, norm_B: float = 8.0) -> np.ndarray:
    """Construct (48, 8) input matrix B with equal L2 norm 8.0 and controlled overlap rho.
    
    Old tokens 1..3 map to orthogonal coordinates 0..2.
    New tokens 4..6 map to coordinates 3..5 with fraction rho projecting onto corresponding old coordinates 0..2.
    For each token k in 1..6: ||B[:, k]||_2 == norm_B exactly.
    Cosine similarity between old token i and new token i+3 is rho.
    All other pairs have cosine similarity 0.0.
    """
    if not (0.0 <= rho < 1.0):
        raise ValueError(f"rho must be in [0.0, 1.0), got {rho}")
    
    B = np.zeros((48, 8), dtype=np.float64)
    # Old tokens 1, 2, 3 -> rows 0, 1, 2
    for k in range(1, 4):
        B[k - 1, k] = norm_B
        
    # New tokens 4, 5, 6 -> rows 3, 4, 5 (orthogonal component) + rows 0, 1, 2 (overlapping component)
    w_ortho = norm_B * np.sqrt(max(0.0, 1.0 - rho ** 2))
    w_overlap = norm_B * rho
    for k in range(4, 7):
        old_k = k - 3
        B[k - 1, k] = w_ortho
        B[old_k - 1, k] = w_overlap
        
    # Verify norms
    for k in range(1, 7):
        n = np.linalg.norm(B[:, k])
        if not np.isclose(n, norm_B, atol=1e-12):
            raise ValueError(f"Norm mismatch for token {k}: {n} != {norm_B}")
            
    return B


def setup_core_with_B(seed: int, B_matrix: np.ndarray, enable_vortex: bool = True) -> SixEchelonCore:
    """Initialize fresh 48D core with given B matrix and fixed EXP-0 W_readout."""
    cfg = SixEchelonConfig(enable_vortex=enable_vortex, integrator_mode="fixed")
    core = SixEchelonCore(config=cfg, seed=seed)
    core.W_readout.fill(0.0)
    for c in range(1, 5):
        core.W_readout[c, 32:35] = CLASSES[c]
    core.B = B_matrix.copy()
    return core


def evaluate_joint_learnability(
    seeds: list[int],
    panel_maps: list[dict[int, int]],
    B_matrix: np.ndarray,
    epochs: int = 25,
    eta_K: float = 0.8,
    delta_core_max: float = 0.5,
    ticks: int = 8,
) -> dict[str, Any]:
    """Train core on all 6 tokens jointly in random interleaved order and evaluate accuracy."""
    psi_0 = _get_quiescent_psi()
    accs = []
    
    for seed in seeds:
        for m_idx, task_map in enumerate(panel_maps):
            core = setup_core_with_B(seed=seed, B_matrix=B_matrix, enable_vortex=True)
            rng = np.random.default_rng(seed * 1000 + m_idx)
            tokens = [1, 2, 3, 4, 5, 6]
            
            for _ in range(epochs):
                shuffled = rng.permutation(tokens)
                for tok in shuffled:
                    wake_train_step(
                        core,
                        tok,
                        task_map[tok],
                        psi_0,
                        eta_K=eta_K,
                        delta_core_max=delta_core_max,
                        ticks=ticks,
                    )
            acc, _, _ = evaluate_query_accuracy(core, tokens, task_map, ticks=ticks, reset_state=True)
            accs.append(acc)
            
    mean_acc = float(np.mean(accs))
    ci = (
        float(stats.t.ppf(0.975, len(accs) - 1) * stats.sem(accs))
        if len(accs) > 1 and np.std(accs) > 0
        else 0.0
    )
    return {
        "mean_joint_accuracy": mean_acc,
        "ci_95": ci,
        "sample_size": len(accs),
        "passed": bool(mean_acc >= 0.75),
    }


def run_wake_history_with_B(
    seed: int,
    task_id: int,
    task_map: dict[int, int],
    history_type: str,
    B_matrix: np.ndarray,
    run_seed: int,
    eval_modes: list[str] = ["structural_reset", "carry_state"],
) -> dict[str, Any]:
    """Run single wake history (201 steps) with custom B and measure endpoints in specified modes."""
    core = setup_core_with_B(seed=seed, B_matrix=B_matrix, enable_vortex=True)
    psi_0 = _get_quiescent_psi()
    
    # Generate schedule
    schedule = generate_wake_schedule(
        seed=seed,
        task_id=task_id,
        task_map=task_map,
        history_type=history_type,
    )
    
    endpoints: dict[str, Any] = {}
    
    def probe_at(label: str):
        endpoints[label] = {}
        for mode in eval_modes:
            reset_flag = (mode == "structural_reset")
            acc_old, margin_old, nll_old = evaluate_query_accuracy(
                core, [1, 2, 3], task_map, reset_state=reset_flag
            )
            acc_new, margin_new, nll_new = evaluate_query_accuracy(
                core, [4, 5, 6], task_map, reset_state=reset_flag
            )
            endpoints[label][mode] = {
                "acc_old": acc_old,
                "margin_old": margin_old,
                "nll_old": nll_old,
                "acc_new": acc_new,
                "margin_new": margin_new,
                "nll_new": nll_new,
            }

    # Execute wake schedule
    for event in schedule:
        wake_train_step(
            core,
            event.token_in,
            event.target_token,
            psi_0,
            eta_K=0.8,
            delta_core_max=0.5,
            ticks=8,
        )
        if event.is_pre_new:
            probe_at("pre_new")
        elif event.is_post_new_pre_tail:
            probe_at("post_new_pre_tail")
            # Fixed delay: 1000 ticks of free integration from saved live Z
            z_start = core.Z.copy()
            delay_core = copy.deepcopy(core)
            x_zero = np.zeros(8, dtype=np.float64)
            for _ in range(1000):
                delay_core.step(x_zero, write_gate=0.0)
            z_end = delay_core.Z.copy()
            
            acc_del_o, margin_del_o, nll_del_o = evaluate_query_accuracy(
                delay_core, [1, 2, 3], task_map, reset_state=False
            )
            acc_del_n, margin_del_n, nll_del_n = evaluate_query_accuracy(
                delay_core, [4, 5, 6], task_map, reset_state=False
            )
            logits_o = [query_core_prediction(delay_core, t, reset_state=False)[1].tolist() for t in [1, 2, 3]]
            logits_n = [query_core_prediction(delay_core, t, reset_state=False)[1].tolist() for t in [4, 5, 6]]
            
            delay1000_data = {
                "acc_old": acc_del_o,
                "acc_new": acc_del_n,
                "margin_old": margin_del_o,
                "margin_new": margin_del_n,
                "nll_old": nll_del_o,
                "nll_new": nll_del_n,
                "logits_old": logits_o,
                "logits_new": logits_n,
                "Z_start": z_start.tolist(),
                "Z_end": z_end.tolist(),
                "Z_drift": float(np.linalg.norm(z_end - z_start)),
            }
            endpoints["post_new_pre_tail"]["delay1000"] = delay1000_data
        elif event.is_tail and event.event_idx == 201:
            probe_at("post_tail")
            
    # Compute interference metrics (primary: pre_tail, secondary: post_tail)
    res_modes: dict[str, Any] = {}
    for mode in eval_modes:
        q_old_pre = endpoints["pre_new"][mode]["acc_old"]
        q_old_post_pre_tail = endpoints["post_new_pre_tail"][mode]["acc_old"]
        q_old_post_tail = endpoints["post_tail"][mode]["acc_old"]
        
        i_pre_tail = float(q_old_pre - q_old_post_pre_tail)
        i_post_tail = float(q_old_pre - q_old_post_tail)
        
        mode_data = {
            "endpoints": {k: endpoints[k][mode] for k in endpoints},
            "I_pre_tail": i_pre_tail,
            "I_post_tail": i_post_tail,
        }
        if mode == "carry_state" and "delay1000" in endpoints["post_new_pre_tail"]:
            mode_data["endpoints"]["post_new_pre_tail"]["delay1000"] = endpoints["post_new_pre_tail"]["delay1000"]
        res_modes[mode] = mode_data
        
    return {
        "seed": seed,
        "task_id": task_id,
        "history_type": history_type,
        "modes": res_modes,
        "delay1000": endpoints["post_new_pre_tail"].get("delay1000"),
    }


def run_load_experiment(
    load_cfg: dict[str, Any],
    seeds: list[int],
    panel_maps: list[dict[int, int]],
    eval_modes: list[str],
) -> dict[str, Any]:
    """Execute complete wake experiment for a given B-code load across all seed-task pairs."""
    rho = float(load_cfg["rho"])
    load_id = load_cfg["load_id"]
    B_matrix = build_overlapping_B(rho=rho, norm_B=8.0)
    
    # 1. Evaluate Joint Learnability Gate
    jl_res = evaluate_joint_learnability(
        seeds=seeds,
        panel_maps=panel_maps,
        B_matrix=B_matrix,
        epochs=25,
    )
    
    # 2. Run wake calibration across histories
    histories_res = []
    for s_idx, seed in enumerate(seeds):
        for task_id, task_map in enumerate(panel_maps):
            pair_run_seed = 100000 + s_idx * 100 + task_id * 10
            for h_type in ["blocked", "interleaved"]:
                h_res = run_wake_history_with_B(
                    seed=seed,
                    task_id=task_id,
                    task_map=task_map,
                    history_type=h_type,
                    B_matrix=B_matrix,
                    run_seed=pair_run_seed,
                    eval_modes=eval_modes,
                )
                histories_res.append(h_res)
                
    # Aggregate metrics
    summary: dict[str, Any] = {"load_id": load_id, "rho": rho, "joint_learnability": jl_res}
    for mode in eval_modes:
        blocked_i = [
            h["modes"][mode]["I_pre_tail"] for h in histories_res if h["history_type"] == "blocked"
        ]
        interleaved_i = [
            h["modes"][mode]["I_pre_tail"] for h in histories_res if h["history_type"] == "interleaved"
        ]
        old_acc_pre = [
            h["modes"][mode]["endpoints"]["pre_new"]["acc_old"] for h in histories_res if h["history_type"] == "blocked"
        ]
        old_acc_post = [
            h["modes"][mode]["endpoints"]["post_new_pre_tail"]["acc_old"] for h in histories_res if h["history_type"] == "blocked"
        ]
        new_acc_post = [
            h["modes"][mode]["endpoints"]["post_new_pre_tail"]["acc_new"] for h in histories_res if h["history_type"] == "blocked"
        ]
        margin_old_pre = [
            h["modes"][mode]["endpoints"]["pre_new"]["margin_old"] for h in histories_res if h["history_type"] == "blocked"
        ]
        margin_old_post = [
            h["modes"][mode]["endpoints"]["post_new_pre_tail"]["margin_old"] for h in histories_res if h["history_type"] == "blocked"
        ]
        nll_old_pre = [
            h["modes"][mode]["endpoints"]["pre_new"]["nll_old"] for h in histories_res if h["history_type"] == "blocked"
        ]
        nll_old_post = [
            h["modes"][mode]["endpoints"]["post_new_pre_tail"]["nll_old"] for h in histories_res if h["history_type"] == "blocked"
        ]
        
        summary[mode] = {
            "mean_old_acc_pre": float(np.mean(old_acc_pre)),
            "mean_old_acc_post_pre_tail": float(np.mean(old_acc_post)),
            "mean_new_acc_post_pre_tail": float(np.mean(new_acc_post)),
            "mean_I_pre_tail_blocked": float(np.mean(blocked_i)),
            "mean_I_pre_tail_interleaved": float(np.mean(interleaved_i)),
            "pressure_contrast": float(np.mean(blocked_i) - np.mean(interleaved_i)),
            "mean_margin_old_pre": float(np.mean(margin_old_pre)),
            "mean_margin_old_post": float(np.mean(margin_old_post)),
            "delta_margin_old": float(np.mean(margin_old_post) - np.mean(margin_old_pre)),
            "mean_nll_old_pre": float(np.mean(nll_old_pre)),
            "mean_nll_old_post": float(np.mean(nll_old_post)),
            "delta_nll_old": float(np.mean(nll_old_post) - np.mean(nll_old_pre)),
        }
        
    # Fixed delay summary
    delays_old = [
        h.get("delay1000", {}).get("acc_old")
        for h in histories_res
        if h["history_type"] == "blocked" and h.get("delay1000") is not None
    ]
    delays_new = [
        h.get("delay1000", {}).get("acc_new")
        for h in histories_res
        if h["history_type"] == "blocked" and h.get("delay1000") is not None
    ]
    if delays_old and all(d is not None for d in delays_old):
        summary["mean_acc_old_delay1000"] = float(np.mean(delays_old))
    if delays_new and all(d is not None for d in delays_new):
        summary["mean_acc_new_delay1000"] = float(np.mean(delays_new))
        
    return {
        "load_id": load_id,
        "rho": rho,
        "summary": summary,
        "histories": histories_res,
    }


def main():
    parser = argparse.ArgumentParser(description="Run EXP-SLEEP-NEED-001B B-code overlap calibration.")
    parser.add_argument("--config", type=str, default="pqt_qcat/memory/SLEEP_NEED_001B_CONFIG.json", help="Path to config JSON")
    parser.add_argument("--output", type=str, default=None, help="Output directory path")
    args = parser.parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    validate_config_seed_integrity(config)
            
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else Path(f"results/exploratory/SLEEP_NEED_001B/RUN_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save provenance
    with open(out_dir / "config_resolved.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        
    # Generate panel
    panel_maps = generate_task_panel(
        generator_seed=config["panel"]["generator_seed"],
        total_maps=config["panel"]["total_calibration_maps"],
    )
    with open(out_dir / "panel_mappings.json", "w", encoding="utf-8") as f:
        json.dump(panel_maps, f, indent=2)
        
    eval_modes = config["evaluation"]["modes"]
    selection_seeds = config["selection_seeds"]
    held_out_seeds = config["held_out_seeds"]
    loads = config["loads"]
    
    print("=================================================================")
    print("EXP-SLEEP-NEED-001B: WAKE B-CODE OVERLAP CALIBRATION")
    print(f"Selection seeds: {selection_seeds} | Held-out seeds: {held_out_seeds}")
    print(f"Loads: {[l['load_id'] for l in loads]}")
    print("=================================================================\n")
    
    # Phase 1: Selection Phase
    print("--- STAGE 1: SELECTION SWEEP ON SEEDS [11, 12] ---")
    selection_results = {}
    eligible_candidates = []
    
    for load_cfg in loads:
        load_id = load_cfg["load_id"]
        print(f"\nEvaluating {load_id} (rho={load_cfg['rho']})...")
        res = run_load_experiment(load_cfg, selection_seeds, panel_maps, eval_modes)
        selection_results[load_id] = res
        
        summ = res["summary"]
        jl = summ["joint_learnability"]
        sr = summ["structural_reset"]
        print(f"  Joint Learnability: {jl['mean_joint_accuracy']:.4f} (passed >= 0.75: {jl['passed']})")
        print(f"  Structural Reset: Old Pre Acc={sr['mean_old_acc_pre']:.4f}, Post Acc={sr['mean_old_acc_post_pre_tail']:.4f}, I={sr['mean_I_pre_tail_blocked']:.4f}")
        print(f"  New Post Acc: {sr['mean_new_acc_post_pre_tail']:.4f}")
        
        # Check eligibility: Joint learnability >= 75% AND Prefix Old Acc >= 80%
        if jl["passed"] and sr["mean_old_acc_pre"] >= 0.80:
            eligible_candidates.append({
                "load_id": load_id,
                "rho": load_cfg["rho"],
                "I_blocked": sr["mean_I_pre_tail_blocked"],
                "contrast": sr["pressure_contrast"],
                "new_acc": sr["mean_new_acc_post_pre_tail"],
                "res": res,
            })
            print(f"  -> Candidate ELIGIBLE.")
        else:
            print(f"  -> Candidate INELIGIBLE (failed gate).")
            
    with open(out_dir / "selection_phase_results.json", "w", encoding="utf-8") as f:
        json.dump(selection_results, f, indent=2)
        
    # Phase 2: Selection Decision
    print("\n--- STAGE 2: SELECTION DECISION ---")
    best_candidate = None
    if eligible_candidates:
        # Sort by forgetting in blocked history
        eligible_candidates.sort(key=lambda x: x["I_blocked"], reverse=True)
        top = eligible_candidates[0]
        if top["I_blocked"] > 0.0:
            best_candidate = top
            print(f"Selected candidate: {best_candidate['load_id']} with I_blocked = {best_candidate['I_blocked']:.4f}")
        else:
            print("No eligible candidate demonstrated catastrophic forgetting (all eligible have I <= 0).")
    else:
        print("Zero candidates passed the Joint Learnability gate (>= 75%).")
        
    # Phase 3: Held-Out Evaluation
    validation_result = None
    calibration_verdict = "CALIBRATION_NEGATIVE"
    
    if best_candidate is not None:
        print(f"\n--- STAGE 3: HELD-OUT VALIDATION ON SEEDS {held_out_seeds} ---")
        best_load_cfg = [l for l in loads if l["load_id"] == best_candidate["load_id"]][0]
        val_res = run_load_experiment(best_load_cfg, held_out_seeds, panel_maps, eval_modes)
        validation_result = val_res
        
        v_summ = val_res["summary"]
        v_jl = v_summ["joint_learnability"]
        v_sr = v_summ["structural_reset"]
        print(f"Held-out Joint Learnability: {v_jl['mean_joint_accuracy']:.4f}")
        print(f"Held-out Structural Reset: I_blocked = {v_sr['mean_I_pre_tail_blocked']:.4f}, New Acc = {v_sr['mean_new_acc_post_pre_tail']:.4f}")
        
        # Check held-out criteria
        rule = config["selection_rule"]
        if (
            v_jl["passed"]
            and v_sr["mean_I_pre_tail_blocked"] >= rule["held_out_interference_min"]
            and v_sr["mean_new_acc_post_pre_tail"] >= rule["held_out_new_acc_min"]
        ):
            calibration_verdict = "CALIBRATION_SUCCESSFUL"
            print("Held-out validation SUCCEEDED!")
        else:
            print("Held-out validation FAILED criteria.")
    else:
        print("\nSkipping Phase 3 due to no eligible candidate.")
        
    final_report = {
        "protocol_id": config["protocol_id"],
        "timestamp": timestamp,
        "verdict": calibration_verdict,
        "selection_seeds": selection_seeds,
        "held_out_seeds": held_out_seeds,
        "best_candidate": best_candidate["load_id"] if best_candidate else None,
        "eligible_candidates_count": len(eligible_candidates),
        "selection_summaries": {k: v["summary"] for k, v in selection_results.items()},
        "validation_summary": validation_result["summary"] if validation_result else None,
    }
    
    with open(out_dir / "sleep_need_001b_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
        
    print("\n=================================================================")
    print(f"EXP-SLEEP-NEED-001B COMPLETED | VERDICT: {calibration_verdict}")
    print(f"Results saved to: {out_dir}")
    print("=================================================================")


if __name__ == "__main__":
    main()
