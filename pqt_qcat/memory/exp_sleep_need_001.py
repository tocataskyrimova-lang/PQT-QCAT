"""EXP-SLEEP-NEED-001: Empirical Investigation of Consolidation Benefit vs Prior Interference.

Protocol v0.1 (17 Nov 2026, Astra/Codex).
Investigates whether consolidation benefit (DREAM vs matched Replay) depends on measured
interference arising during prior learning (BLOCKED vs INTERLEAVED).
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np

from pqt_qcat.core.six_echelon import SixEchelonCore, SixEchelonConfig
from pqt_qcat.memory.m3_diagnostics import CLASSES, _get_quiescent_psi


def safe_json_dumps(obj: dict | list | any) -> str:
    """Serialize object to JSON, safely converting numpy types."""
    def default(o):
        if isinstance(o, (np.integer, np.int64, np.int32)):
            return int(o)
        if isinstance(o, (np.floating, np.float64, np.float32)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    return json.dumps(obj, default=default)


def compute_softmax(logits: np.ndarray) -> np.ndarray:
    """Compute numerically stable softmax over classes 1..4."""
    z = logits[1:5] - np.max(logits[1:5])
    p = np.exp(z)
    return p / np.sum(p)


def generate_task_panel(generator_seed: int = 170700, total_maps: int = 12) -> list[dict[int, int]]:
    """Generate fixed panel of 12 mappings for sources 1..6 with all 4 classes present.
    
    Rule: uniform i.i.d. targets from {1, 2, 3, 4}, conditioned on all 4 classes present,
    reject duplicate mappings.
    """
    rng = np.random.default_rng(generator_seed)
    maps: list[dict[int, int]] = []
    seen: set[tuple[int, ...]] = set()
    
    while len(maps) < total_maps:
        t = tuple(rng.integers(1, 5, size=6).tolist())
        if len(set(t)) == 4 and t not in seen:
            seen.add(t)
            maps.append({x: t[x - 1] for x in range(1, 7)})
    return maps


def setup_fresh_core(seed: int, enable_vortex: bool = True) -> SixEchelonCore:
    """Initialize fresh 48D core with fixed B and W matrices per protocol."""
    cfg = SixEchelonConfig(enable_vortex=enable_vortex, integrator_mode="fixed")
    core = SixEchelonCore(config=cfg, seed=seed)
    
    # Fixed W_readout on coordinates 32:35 matching CLASSES for classes 1..4
    core.W_readout.fill(0.0)
    for c in range(1, 5):
        core.W_readout[c, 32:35] = CLASSES[c]
        
    # Fixed B: B=0, B[x-1, x] = 8.0 for x in 1..6
    core.B.fill(0.0)
    for k in range(1, 7):
        core.B[k - 1, k] = 8.0
        
    return core


def query_core_prediction(
    core: SixEchelonCore,
    token: int,
    ticks: int = 8,
) -> tuple[int, np.ndarray, float, float]:
    """Execute evaluation query on token for 8 ticks from quiescent Z.
    
    Returns: (predicted_class, logits_1_to_4, margin, nll).
    """
    core.Z.fill(0.0)
    for c in range(6):
        core.Z[8 * c + 6] = 1.0
        
    x = np.zeros(8)
    x[token] = 1.0
    
    for _ in range(ticks):
        core.step(x, write_gate=1.0)
        
    logits = core.predict()
    logits_4 = logits[1:5].copy()
    pred = int(np.argmax(logits_4)) + 1
    
    # Softmax probabilities
    z = logits_4 - np.max(logits_4)
    p = np.exp(z) / np.sum(np.exp(z))
    
    # Margin between top and second logit
    sorted_logits = np.sort(logits_4)
    margin = float(sorted_logits[-1] - sorted_logits[-2])
    
    return pred, logits_4, margin, float(np.max(p))


def evaluate_query_accuracy(
    core: SixEchelonCore,
    tokens: list[int],
    task_map: dict[int, int],
    ticks: int = 8,
) -> tuple[float, float, float]:
    """Evaluate accuracy, mean margin, and mean NLL on a set of tokens."""
    correct = 0
    margins = []
    nlls = []
    
    for tok in tokens:
        pred, logits_4, margin, _ = query_core_prediction(core, tok, ticks=ticks)
        target = task_map[tok]
        if pred == target:
            correct += 1
        margins.append(margin)
        
        # Target NLL
        z = logits_4 - np.max(logits_4)
        p = np.exp(z) / np.sum(np.exp(z))
        nlls.append(float(-np.log(p[target - 1] + 1e-12)))
        
    acc = correct / float(len(tokens))
    return acc, float(np.mean(margins)), float(np.mean(nlls))


def wake_train_step(
    core: SixEchelonCore,
    token: int,
    target: int,
    psi_0: np.ndarray,
    eta_K: float = 0.8,
    delta_core_max: float = 0.5,
    ticks: int = 8,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Execute single wake training step with EXP-0 target-only error rule."""
    core.Z.fill(0.0)
    for c in range(6):
        core.Z[8 * c + 6] = 1.0
        
    x = np.zeros(8)
    x[token] = 1.0
    
    for _ in range(ticks):
        core.step(x, write_gate=1.0)
        
    logits = core.predict()
    z = logits[1:5] - np.max(logits[1:5])
    p = np.exp(z) / np.sum(np.exp(z))
    pred = int(np.argmax(logits[1:5])) + 1
    
    # Target-only error: err[target] = 1 - p[target - 1]
    err = np.zeros(8)
    err[target] = 1.0 - p[target - 1]
    
    delta_Z = core.W_readout.T @ err
    d_psi = np.tanh(core.Z) - psi_0
    delta_K = np.clip(eta_K * np.outer(delta_Z, d_psi), -delta_core_max, delta_core_max)
    core.K += delta_K
    
    return pred, logits[1:5].copy(), delta_K


@dataclass
class ObservedEvent:
    event_id: int
    token_in: int
    target_token: int
    epoch: int
    group: str  # 'prefix_old', 'day_old', 'day_new', 'tail_old', 'tail_new'


class SleepAdapterNeed001:
    """Matched-compute sleep adapter for EXP-SLEEP-NEED-001.
    
    Supports 4 policies: 'dream', 'replay', 'sham_dream', 'idle'.
    Budgets:
      - Active ticks: exactly 1000 ticks.
      - Shadow ticks: exactly 2400 ticks (25 slots * 2 evaluations * 6 anchors * 8 ticks).
      - Anchors: all 6 available real mappings.
    """

    def __init__(
        self,
        candidate_slots: int = 25,
        anchor_count: int = 6,
        anchor_max_degradation: float = 0.02,
        thermal_noise_sigma: float = 0.05,
        h_base: float = 0.01,
        eta_sleep: float = 0.05,
    ):
        self.candidate_slots = candidate_slots
        self.anchor_count = anchor_count
        self.anchor_max_degradation = anchor_max_degradation
        self.thermal_noise_sigma = thermal_noise_sigma
        self.h_base = h_base
        self.eta_sleep = eta_sleep

    def evaluate_anchors_accuracy(
        self,
        core: SixEchelonCore,
        anchors: dict[int, int],
        ticks_per_token: int = 8,
    ) -> float:
        """Evaluate accuracy across the 6 anchors."""
        correct = 0
        for tok, tgt in anchors.items():
            pred, _, _, _ = query_core_prediction(core, tok, ticks=ticks_per_token)
            if pred == tgt:
                correct += 1
        return correct / float(len(anchors))

    def run_sleep_session(
        self,
        policy: str,
        core: SixEchelonCore,
        observed_events: list[ObservedEvent],
        anchors: dict[int, int],
        run_seed: int,
        sleep_log_records: list[dict],
    ) -> dict:
        """Execute autonomous sleep session under strict matched compute budget."""
        t_start = time.perf_counter()
        initial_K = core.K.copy()
        
        acc_before = self.evaluate_anchors_accuracy(core, anchors, ticks_per_token=8)
        
        active_ticks = 0
        shadow_ticks = 0
        accepted_count = 0
        rejected_count = 0
        rejection_reasons: list[str] = []
        
        # Independent, deterministic pre-generation of grounding events and thermal noise stream
        rng_grounding = np.random.default_rng(run_seed + 10007)
        rng_noise = np.random.default_rng(run_seed + 20011)
        
        grounding_choices = [rng_grounding.choice(observed_events) for _ in range(self.candidate_slots)]
        noise_stream = [rng_noise.normal(0.0, self.thermal_noise_sigma, size=(10, 48)) for _ in range(self.candidate_slots)]
        
        # Observed transition mapping for provenance verification
        observed_map = {}
        for ev in observed_events:
            observed_map[ev.token_in] = ev.target_token
            
        if policy == "idle":
            # Idle control: 1000 physical ticks with write_gate=0.0, zero presentations, zero updates
            for _ in range(1000):
                core.step(x_t=None, write_gate=0.0)
                active_ticks += 1
                
        elif policy in ("dream", "replay", "sham_dream"):
            for slot_idx in range(self.candidate_slots):
                seed_event = grounding_choices[slot_idx]
                token_in = seed_event.token_in
                noise_block = noise_stream[slot_idx]
                
                # Reset to quiescent state
                core.Z.fill(0.0)
                for c in range(6):
                    core.Z[8 * c + 6] = 1.0
                    
                # 1 presentation tick
                x_seed = np.zeros(core.config.d_in)
                x_seed[token_in] = 1.0
                core.step(x_seed, write_gate=1.0)
                active_ticks += 1
                
                # 10 thermal relaxation ticks
                for tick_idx in range(10):
                    core.Z += noise_block[tick_idx] * np.sqrt(self.h_base)
                    core.step(x_t=None, write_gate=0.0)
                    active_ticks += 1
                    
                # Read candidate proposal from internal dynamic state
                logits = core.predict() * 4.0
                p = compute_softmax(logits)
                pred_token = int(np.argmax(p)) + 1
                conf = float(np.max(p))
                
                if policy == "replay":
                    target_token = seed_event.target_token
                    target_source = "OBSERVED_EVENT"
                else:
                    target_token = pred_token
                    target_source = "DREAM_DYNAMIC"
                    
                # Gate 1: Provenance & Consistency (must match observed history for this token)
                gate1_provenance = (token_in in observed_map and observed_map[token_in] == target_token)
                
                # Gate 2: Structural Bounds (target must be valid class 1..4)
                gate2_structural = (1 <= target_token <= 4 and 1 <= token_in <= 6)
                
                # Construct proposed delta_K
                src_idx = token_in - 1
                tgt_idx = 31 + target_token
                delta_K = np.zeros_like(core.K)
                delta_K[tgt_idx, src_idx] = self.eta_sleep * conf
                
                # Gate 3: Anchor Guard Evaluation
                # Always run exactly 96 shadow ticks (48 baseline + 48 proposed) to equalize shadow compute!
                shadow_core = core.snapshot()
                acc_anc_before = self.evaluate_anchors_accuracy(core, anchors, ticks_per_token=8)
                shadow_ticks += len(anchors) * 8
                
                test_core = SixEchelonCore(config=core.config)
                test_core.restore(shadow_core)
                test_core.K += delta_K
                acc_anc_after = self.evaluate_anchors_accuracy(test_core, anchors, ticks_per_token=8)
                shadow_ticks += len(anchors) * 8
                
                degradation = acc_anc_before - acc_anc_after
                anchor_ok = bool(degradation <= self.anchor_max_degradation)
                
                slot_passed = False
                rejection_reason = "NONE"
                
                if not gate1_provenance:
                    rejection_reason = "provenance_or_contradiction"
                elif not gate2_structural:
                    rejection_reason = "structural_bounds"
                elif not anchor_ok:
                    rejection_reason = "anchor_guard_rollback"
                else:
                    slot_passed = True
                    
                if slot_passed:
                    if policy in ("dream", "replay"):
                        core.K += delta_K
                        accepted_count += 1
                    elif policy == "sham_dream":
                        rejected_count += 1
                        rejection_reason = "policy_cancel"
                else:
                    rejected_count += 1
                    rejection_reasons.append(rejection_reason)
                    
                sleep_log_records.append({
                    "slot": slot_idx,
                    "policy": policy,
                    "token_in": token_in,
                    "target_token": target_token,
                    "target_source": target_source,
                    "confidence": conf,
                    "gate1_provenance": gate1_provenance,
                    "gate2_structural": gate2_structural,
                    "gate3_anchor": anchor_ok,
                    "anchor_degradation": degradation,
                    "passed": slot_passed,
                    "rejection_reason": rejection_reason,
                })
                
            # 725 free relaxation padding ticks to reach EXACTLY 1000 active ticks
            for _ in range(725):
                core.step(x_t=None, write_gate=0.0)
                active_ticks += 1
                
        else:
            raise ValueError(f"Unknown sleep policy: {policy}")
            
        acc_after = self.evaluate_anchors_accuracy(core, anchors, ticks_per_token=8)
        retention_degradation = acc_before - acc_after
        delta_K_norm = float(np.linalg.norm(core.K - initial_K))
        delta_K_ratio = float(delta_K_norm / max(np.linalg.norm(initial_K), 1e-12))
        
        return {
            "policy": policy,
            "anchor_acc_before": acc_before,
            "anchor_acc_after": acc_after,
            "retention_degradation": retention_degradation,
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "rejection_reasons": rejection_reasons,
            "active_ticks": active_ticks,
            "shadow_ticks": shadow_ticks,
            "delta_K_norm": delta_K_norm,
            "delta_K_ratio": delta_K_ratio,
            "wall_time": time.perf_counter() - t_start,
        }


def run_single_history_experiment(
    seed: int,
    task_id: int,
    task_map: dict[int, int],
    history_type: str,  # 'blocked' or 'interleaved'
    run_seed: int,
    adapter: SleepAdapterNeed001,
    wake_log_records: list[dict],
    probe_log_records: list[dict],
    sleep_log_records: list[dict],
    query_log_records: list[dict],
    save_snapshots_dir: Path | None = None,
) -> dict:
    """Execute complete life for one (seed, task, history) combination across all 4 sleep policies."""
    psi_0 = _get_quiescent_psi()
    core = setup_fresh_core(seed=seed, enable_vortex=True)
    K_initial = core.K.copy()
    
    # Epoch permutations pre-generation
    rng_ep = np.random.default_rng(seed * 10000 + task_id * 100 + 42)
    prefix_perms = [rng_ep.permutation([1, 2, 3]).tolist() for _ in range(25)]
    day_O_perms = [rng_ep.permutation([1, 2, 3]).tolist() for _ in range(20)]
    day_N_perms = [rng_ep.permutation([4, 5, 6]).tolist() for _ in range(20)]
    tail_O_perm = rng_ep.permutation([1, 2, 3]).tolist()
    tail_N_perm = rng_ep.permutation([4, 5, 6]).tolist()
    
    observed_events: list[ObservedEvent] = []
    event_id = 0
    
    def log_wake_step(token, target, group, epoch_idx):
        nonlocal event_id
        pred, logits_4, delta_K = wake_train_step(core, token, target, psi_0)
        event_id += 1
        ev = ObservedEvent(event_id, token, target, epoch_idx, group)
        observed_events.append(ev)
        wake_log_records.append({
            "event_id": event_id,
            "seed": seed,
            "task_id": task_id,
            "history": history_type,
            "epoch": epoch_idx,
            "group": group,
            "token_in": token,
            "target": target,
            "predicted": pred,
            "delta_K_norm": float(np.linalg.norm(delta_K)),
        })
        return ev
        
    def take_diagnostic_probe(stage_name: str, epoch_num: int):
        probe_core = SixEchelonCore(config=core.config)
        probe_core.restore(core.snapshot())
        acc_o, m_o, nll_o = evaluate_query_accuracy(probe_core, [1, 2, 3], task_map)
        acc_n, m_n, nll_n = evaluate_query_accuracy(probe_core, [4, 5, 6], task_map)
        k_hash = hashlib.sha256(probe_core.K.tobytes()).hexdigest()
        probe_log_records.append({
            "seed": seed,
            "task_id": task_id,
            "history": history_type,
            "stage": stage_name,
            "epoch": epoch_num,
            "acc_old": acc_o,
            "acc_new": acc_n,
            "margin_old": m_o,
            "margin_new": m_n,
            "nll_old": nll_o,
            "nll_new": nll_n,
            "K_sha256": k_hash,
        })
        return acc_o, acc_n
        
    # Phase 1: Prefix (25 epochs on Old 1, 2, 3)
    for ep in range(25):
        for tok in prefix_perms[ep]:
            log_wake_step(tok, task_map[tok], "prefix_old", ep + 1)
            
    K_prefix = core.K.copy()
    q_old_prefix, q_new_prefix = take_diagnostic_probe("prefix_end", 25)
    
    # Phase 2: Main Day (120 events)
    if history_type == "blocked":
        # 20 epochs Old consecutively
        for ep in range(20):
            for tok in day_O_perms[ep]:
                log_wake_step(tok, task_map[tok], "day_old", 25 + ep + 1)
            if (ep + 1) % 6 == 0 or (ep + 1) == 20:
                take_diagnostic_probe(f"day_O_ep_{ep+1}", 25 + ep + 1)
        # 20 epochs New consecutively
        for ep in range(20):
            for tok in day_N_perms[ep]:
                log_wake_step(tok, task_map[tok], "day_new", 45 + ep + 1)
            if (ep + 1) % 6 == 0 or (ep + 1) == 20:
                take_diagnostic_probe(f"day_N_ep_{ep+1}", 45 + ep + 1)
                
    elif history_type == "interleaved":
        # 20 pairs of (1 epoch Old, 1 epoch New)
        for ep in range(20):
            for tok in day_O_perms[ep]:
                log_wake_step(tok, task_map[tok], "day_old", 25 + 2 * ep + 1)
            for tok in day_N_perms[ep]:
                log_wake_step(tok, task_map[tok], "day_new", 25 + 2 * ep + 2)
            if (ep + 1) % 6 == 0 or (ep + 1) == 20:
                take_diagnostic_probe(f"day_inter_pair_{ep+1}", 25 + 2 * (ep + 1))
    else:
        raise ValueError(f"Unknown history type: {history_type}")
        
    take_diagnostic_probe("pre_tail", 65)
    
    # Phase 3: Tail (1 epoch Old, 1 epoch New = 6 events)
    for tok in tail_O_perm:
        log_wake_step(tok, task_map[tok], "tail_old", 66)
    for tok in tail_N_perm:
        log_wake_step(tok, task_map[tok], "tail_new", 67)
        
    K_pre_sleep = core.K.copy()
    q_old_pre, q_new_pre = take_diagnostic_probe("post_tail_pre_sleep", 67)
    
    # Interference metric I_h
    I_h = q_old_prefix - q_old_pre
    
    # Save snapshot before sleep
    pre_sleep_snapshot = core.snapshot()
    
    anchors = {x: task_map[x] for x in range(1, 7)}
    
    policy_results = {}
    
    # Execute 4 sleep policies
    for pol_idx, policy in enumerate(["dream", "replay", "sham_dream", "idle"]):
        core_sleep = SixEchelonCore(config=core.config)
        core_sleep.restore(pre_sleep_snapshot)
        
        pol_seed = run_seed + pol_idx * 503
        sleep_metrics = adapter.run_sleep_session(
            policy=policy,
            core=core_sleep,
            observed_events=observed_events,
            anchors=anchors,
            run_seed=pol_seed,
            sleep_log_records=sleep_log_records,
        )
        
        K_post = core_sleep.K.copy()
        
        # Rollback check: K_post -> K_pre -> K_post
        core_sleep.K = K_pre_sleep.copy()
        roll_acc, _, _ = evaluate_query_accuracy(core_sleep, [1, 2, 3], task_map)
        core_sleep.K = K_post.copy()
        restored_acc, _, _ = evaluate_query_accuracy(core_sleep, [1, 2, 3], task_map)
        rollback_exact = bool(np.isclose(restored_acc, sleep_metrics["anchor_acc_after"]))
        
        # Evaluation 1: Structural Recipient (Primary)
        # Clean recipient core with initial B, W, Z0, but K_post
        struct_core = setup_fresh_core(seed=seed, enable_vortex=True)
        struct_core.K = K_post.copy()
        acc_o_post_struct, m_o_post_struct, nll_o_post_struct = evaluate_query_accuracy(struct_core, [1, 2, 3], task_map)
        acc_n_post_struct, m_n_post_struct, nll_n_post_struct = evaluate_query_accuracy(struct_core, [4, 5, 6], task_map)
        
        # Log queries for structural recipient
        for tok in range(1, 7):
            p_pred, p_logits, p_margin, _ = query_core_prediction(struct_core, tok)
            tgt = task_map[tok]
            query_log_records.append({
                "seed": seed,
                "task_id": task_id,
                "history": history_type,
                "policy": policy,
                "eval_mode": "structural_recipient",
                "token": tok,
                "target": tgt,
                "predicted": p_pred,
                "correct": bool(p_pred == tgt),
                "margin": p_margin,
                "logits": p_logits.tolist(),
            })
            
        # Evaluation 2: Carry-State (Secondary)
        # Snapshot of sleep core evaluated directly
        acc_o_post_carry, m_o_post_carry, nll_o_post_carry = evaluate_query_accuracy(core_sleep, [1, 2, 3], task_map)
        acc_n_post_carry, m_n_post_carry, nll_n_post_carry = evaluate_query_accuracy(core_sleep, [4, 5, 6], task_map)
        
        # Evaluation 3: 1000 Idle ticks delay stability check
        delay_core = SixEchelonCore(config=core.config)
        delay_core.restore(core_sleep.snapshot())
        for _ in range(1000):
            delay_core.step(x_t=None, write_gate=0.0)
        acc_o_delay, _, _ = evaluate_query_accuracy(delay_core, [1, 2, 3], task_map)
        
        gain_old_struct = acc_o_post_struct - q_old_pre
        gain_new_struct = acc_n_post_struct - q_new_pre
        
        policy_results[policy] = {
            "acc_old_pre": q_old_pre,
            "acc_new_pre": q_new_pre,
            "acc_old_post_struct": acc_o_post_struct,
            "acc_new_post_struct": acc_n_post_struct,
            "acc_old_post_carry": acc_o_post_carry,
            "acc_new_post_carry": acc_n_post_carry,
            "acc_old_delay1000": acc_o_delay,
            "gain_old_struct": gain_old_struct,
            "gain_new_struct": gain_new_struct,
            "margin_old_struct": m_o_post_struct,
            "nll_old_struct": nll_o_post_struct,
            "delta_K_norm": sleep_metrics["delta_K_norm"],
            "accepted_updates": sleep_metrics["accepted_count"],
            "rejected_updates": sleep_metrics["rejected_count"],
            "active_ticks": sleep_metrics["active_ticks"],
            "shadow_ticks": sleep_metrics["shadow_ticks"],
            "rollback_exact": rollback_exact,
        }
        
        if save_snapshots_dir:
            snap_path = save_snapshots_dir / f"post_seed{seed}_task{task_id}_{history_type}_{policy}.npz"
            np.savez_compressed(
                snap_path,
                K=K_post,
                B=core_sleep.B,
                W_readout=core_sleep.W_readout,
                Z=core_sleep.Z,
            )
            
    return {
        "seed": seed,
        "task_id": task_id,
        "history": history_type,
        "q_old_prefix": q_old_prefix,
        "q_new_prefix": q_new_prefix,
        "q_old_pre_sleep": q_old_pre,
        "q_new_pre_sleep": q_new_pre,
        "interference_I": I_h,
        "policies": policy_results,
    }


def compute_student_t_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float, list[float]]:
    """Compute mean, std error, and two-sided Student-t confidence interval."""
    from scipy import stats
    n = len(values)
    mean = float(np.mean(values))
    if n <= 1:
        return mean, 0.0, [mean, mean]
    std_err = float(stats.sem(values))
    t_crit = float(stats.t.ppf((1 + confidence) / 2.0, df=n - 1))
    ci = [float(mean - t_crit * std_err), float(mean + t_crit * std_err)]
    return mean, std_err, ci


def run_stage_experiment(
    stage_name: str,
    seeds: list[int],
    panel_maps: list[dict[int, int]],
    output_stage_dir: Path,
    config: dict,
) -> dict:
    """Execute complete stage (Calibration or Replication) across all histories and policies."""
    t_start = time.perf_counter()
    output_stage_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = output_stage_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    
    wake_records: list[dict] = []
    probe_records: list[dict] = []
    sleep_records: list[dict] = []
    query_records: list[dict] = []
    history_metrics_rows: list[dict] = []
    
    adapter = SleepAdapterNeed001(
        candidate_slots=config["sleep"]["candidate_slots"],
        anchor_count=config["sleep"]["anchor_count"],
        anchor_max_degradation=config["sleep"]["anchor_max_degradation"],
        thermal_noise_sigma=config["sleep"]["sigma"],
        h_base=config["sleep"]["h"],
        eta_sleep=config["sleep"]["eta"],
    )
    
    run_seed_counter = 42000
    all_history_results = []
    
    print(f"[{stage_name.upper()}] Starting execution: {len(seeds)} seeds x {len(panel_maps)} maps x 2 histories x 4 policies = {len(seeds) * len(panel_maps) * 2 * 4} sleep branches.")
    
    for s_idx, seed in enumerate(seeds):
        t_seed_start = time.perf_counter()
        for m_idx, task_map in enumerate(panel_maps):
            for hist in ["blocked", "interleaved"]:
                run_seed_counter += 1
                res = run_single_history_experiment(
                    seed=seed,
                    task_id=m_idx,
                    task_map=task_map,
                    history_type=hist,
                    run_seed=run_seed_counter,
                    adapter=adapter,
                    wake_log_records=wake_records,
                    probe_log_records=probe_records,
                    sleep_log_records=sleep_records,
                    query_log_records=query_records,
                    save_snapshots_dir=snapshots_dir,
                )
                all_history_results.append(res)
                
                # Append rows for each policy
                for pol, pdata in res["policies"].items():
                    history_metrics_rows.append({
                        "seed": seed,
                        "task_id": m_idx,
                        "history": hist,
                        "policy": pol,
                        "q_old_prefix": res["q_old_prefix"],
                        "q_new_prefix": res["q_new_prefix"],
                        "q_old_pre": pdata["acc_old_pre"],
                        "q_new_pre": pdata["acc_new_pre"],
                        "interference_I": res["interference_I"],
                        "acc_old_post_struct": pdata["acc_old_post_struct"],
                        "acc_new_post_struct": pdata["acc_new_post_struct"],
                        "gain_old_struct": pdata["gain_old_struct"],
                        "gain_new_struct": pdata["gain_new_struct"],
                        "acc_old_post_carry": pdata["acc_old_post_carry"],
                        "acc_new_post_carry": pdata["acc_new_post_carry"],
                        "acc_old_delay1000": pdata["acc_old_delay1000"],
                        "margin_old_struct": pdata["margin_old_struct"],
                        "nll_old_struct": pdata["nll_old_struct"],
                        "delta_K_norm": pdata["delta_K_norm"],
                        "accepted_updates": pdata["accepted_updates"],
                        "rejected_updates": pdata["rejected_updates"],
                        "active_ticks": pdata["active_ticks"],
                        "shadow_ticks": pdata["shadow_ticks"],
                        "rollback_exact": pdata["rollback_exact"],
                    })
                    
        print(f"  Completed seed {seed} ({s_idx+1}/{len(seeds)}) in {time.perf_counter() - t_seed_start:.1f}s")
        
    # Write JSONL logs
    print(f"[{stage_name.upper()}] Writing logs...")
    with open(output_stage_dir / "wake.jsonl", "w", encoding="utf-8") as f:
        for r in wake_records:
            f.write(safe_json_dumps(r) + "\n")
            
    with open(output_stage_dir / "pressure_probes.jsonl", "w", encoding="utf-8") as f:
        for r in probe_records:
            f.write(safe_json_dumps(r) + "\n")
            
    with open(output_stage_dir / "sleep.jsonl", "w", encoding="utf-8") as f:
        for r in sleep_records:
            f.write(safe_json_dumps(r) + "\n")
            
    with open(output_stage_dir / "queries.jsonl", "w", encoding="utf-8") as f:
        for r in query_records:
            f.write(safe_json_dumps(r) + "\n")
            
    # Write CSV
    csv_file = output_stage_dir / "metrics_per_history.csv"
    if history_metrics_rows:
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history_metrics_rows[0].keys()))
            writer.writeheader()
            writer.writerows(history_metrics_rows)
            
    # Aggregation per seed and contrasts
    seed_stats = []
    
    # Analyze contrast D = (gain_dream_old - gain_replay_old)_blocked - (gain_dream_old - gain_replay_old)_interleaved
    D_per_seed = []
    I_contrast_per_seed = []
    I_blocked_per_seed = []
    I_interleaved_per_seed = []
    gain_dream_blocked_per_seed = []
    gain_replay_blocked_per_seed = []
    
    for seed in seeds:
        seed_rows = [r for r in history_metrics_rows if r["seed"] == seed]
        
        # Paired comparisons per task
        D_tasks = []
        I_contrast_tasks = []
        I_b_tasks = []
        I_i_tasks = []
        g_dr_b_tasks = []
        g_rp_b_tasks = []
        
        for m_idx in range(len(panel_maps)):
            t_rows = [r for r in seed_rows if r["task_id"] == m_idx]
            
            row_dr_b = next((r for r in t_rows if r["history"] == "blocked" and r["policy"] == "dream"), None)
            row_rp_b = next((r for r in t_rows if r["history"] == "blocked" and r["policy"] == "replay"), None)
            row_dr_i = next((r for r in t_rows if r["history"] == "interleaved" and r["policy"] == "dream"), None)
            row_rp_i = next((r for r in t_rows if r["history"] == "interleaved" and r["policy"] == "replay"), None)
            
            if row_dr_b and row_rp_b and row_dr_i and row_rp_i:
                R_blocked = row_dr_b["gain_old_struct"] - row_rp_b["gain_old_struct"]
                R_interleaved = row_dr_i["gain_old_struct"] - row_rp_i["gain_old_struct"]
                D_task = R_blocked - R_interleaved
                D_tasks.append(D_task)
                
                I_b = row_dr_b["interference_I"]
                I_i = row_dr_i["interference_I"]
                I_b_tasks.append(I_b)
                I_i_tasks.append(I_i)
                I_contrast_tasks.append(I_b - I_i)
                
                g_dr_b_tasks.append(row_dr_b["gain_old_struct"])
                g_rp_b_tasks.append(row_rp_b["gain_old_struct"])
                
        D_seed_mean = float(np.mean(D_tasks)) if D_tasks else 0.0
        I_cont_mean = float(np.mean(I_contrast_tasks)) if I_contrast_tasks else 0.0
        I_b_mean = float(np.mean(I_b_tasks)) if I_b_tasks else 0.0
        I_i_mean = float(np.mean(I_i_tasks)) if I_i_tasks else 0.0
        g_dr_b_mean = float(np.mean(g_dr_b_tasks)) if g_dr_b_tasks else 0.0
        g_rp_b_mean = float(np.mean(g_rp_b_tasks)) if g_rp_b_tasks else 0.0
        
        D_per_seed.append(D_seed_mean)
        I_contrast_per_seed.append(I_cont_mean)
        I_blocked_per_seed.append(I_b_mean)
        I_interleaved_per_seed.append(I_i_mean)
        gain_dream_blocked_per_seed.append(g_dr_b_mean)
        gain_replay_blocked_per_seed.append(g_rp_b_mean)
        
        seed_stats.append({
            "seed": seed,
            "mean_D": D_seed_mean,
            "mean_I_contrast": I_cont_mean,
            "mean_I_blocked": I_b_mean,
            "mean_I_interleaved": I_i_mean,
            "mean_gain_dream_blocked": g_dr_b_mean,
            "mean_gain_replay_blocked": g_rp_b_mean,
        })
        
    # Write metrics_per_seed.csv
    with open(output_stage_dir / "metrics_per_seed.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(seed_stats[0].keys()))
        writer.writeheader()
        writer.writerows(seed_stats)
        
    # Statistical contrasts
    D_mean, D_sem, D_ci = compute_student_t_ci(D_per_seed)
    I_mean, I_sem, I_ci = compute_student_t_ci(I_contrast_per_seed)
    Ib_mean, Ib_sem, Ib_ci = compute_student_t_ci(I_blocked_per_seed)
    Ii_mean, Ii_sem, Ii_ci = compute_student_t_ci(I_interleaved_per_seed)
    Gdr_b_mean, Gdr_b_sem, Gdr_b_ci = compute_student_t_ci(gain_dream_blocked_per_seed)
    Grp_b_mean, Grp_b_sem, Grp_b_ci = compute_student_t_ci(gain_replay_blocked_per_seed)
    
    # Verdict computation
    # Tolerance / equivalence bound: [-0.05, +0.05]
    if D_ci[0] > 0.05:
        D_verdict = "DREAM_NEED_SUPPORTED"
    elif D_ci[0] >= -0.05 and D_ci[1] <= 0.05:
        D_verdict = "EQUIVALENT"
    else:
        D_verdict = "INCONCLUSIVE"
        
    contrasts_data = {
        "D_primary_contrast": {
            "definition": "D = (G_dream - G_replay)_blocked - (G_dream - G_replay)_interleaved",
            "mean": D_mean,
            "sem": D_sem,
            "ci_95": D_ci,
            "seed_means": D_per_seed,
            "verdict": D_verdict,
        },
        "interference_contrast": {
            "definition": "I_contrast = I_blocked - I_interleaved",
            "mean": I_mean,
            "sem": I_sem,
            "ci_95": I_ci,
            "seed_means": I_contrast_per_seed,
            "target_threshold": 0.10,
            "passed_threshold": bool(I_mean >= 0.10 and I_ci[0] > 0.0),
        },
        "I_blocked": {
            "mean": Ib_mean,
            "sem": Ib_sem,
            "ci_95": Ib_ci,
            "seed_means": I_blocked_per_seed,
        },
        "I_interleaved": {
            "mean": Ii_mean,
            "sem": Ii_sem,
            "ci_95": Ii_ci,
            "seed_means": I_interleaved_per_seed,
        },
        "gain_dream_blocked": {
            "mean": Gdr_b_mean,
            "sem": Gdr_b_sem,
            "ci_95": Gdr_b_ci,
            "seed_means": gain_dream_blocked_per_seed,
        },
        "gain_replay_blocked": {
            "mean": Grp_b_mean,
            "sem": Grp_b_sem,
            "ci_95": Grp_b_ci,
            "seed_means": gain_replay_blocked_per_seed,
        },
    }
    
    with open(output_stage_dir / "contrasts.json", "w", encoding="utf-8") as f:
        json.dump(contrasts_data, f, indent=2)
        
    print(f"[{stage_name.upper()}] Done in {time.perf_counter() - t_start:.1f}s. Primary D={D_mean:.4f} {D_ci}, I_contrast={I_mean:.4f} {I_ci}")
    return contrasts_data


def generate_russian_report(stage_name: str, contrasts: dict, run_dir: Path, n_branches: int) -> str:
    """Generate Russian summary report markdown for the stage."""
    d_data = contrasts["D_primary_contrast"]
    i_data = contrasts["interference_contrast"]
    
    rep = f"""# Отчёт о результатах этапа: {stage_name.upper()} (EXP-SLEEP-NEED-001)

- **Объём ветвей сна:** {n_branches} ветвей
- **Основной контраст D:** {d_data['mean']:+.4f} (95% ДИ: [{d_data['ci_95'][0]:+.4f}, {d_data['ci_95'][1]:+.4f}])
- **Вердикт по D:** `{d_data['verdict']}`
- **Интерференция I_contrast (BLOCKED - INTERLEAVED):** {i_data['mean']:+.4f} (95% ДИ: [{i_data['ci_95'][0]:+.4f}, {i_data['ci_95'][1]:+.4f}])
- **Порог интерференции (>= 10 п.п.):** {'ДОСТИГНУТ' if i_data['passed_threshold'] else 'НЕ ДОСТИГНУТ'}

## Сводные средние по seeds
- D по seeds: {d_data['seed_means']}
- I_contrast по seeds: {i_data['seed_means']}
"""
    with open(run_dir / f"report_{stage_name}_ru.md", "w", encoding="utf-8") as f:
        f.write(rep)
    return rep
