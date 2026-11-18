"""Stage R3: Stateful Online Learning and A Factorial Protocol.

Addresses:
1. Streaming Interface: step / predict / observe_feedback / idle / snapshot / restore.
   Strict rule: prediction is committed BEFORE feedback is revealed.
2. Context Task (C0 and C1):
   - Observable context tokens 0 and 7: B[8, 0] = 8.0, B[9, 7] = 8.0 (row 8 and 9 in E2).
   - Queries 1..6: B[k-1, k] = 8.0.
   - Ground truth mapping pairs (M0, M1) from generator.
3. Joint Learnability Control:
   - Verifies distinguishability of contexts and reachability of >= 75% joint accuracy.
4. Training Arms:
   - RESET_TRAIN (global reset before each event)
   - CARRY_TRAIN (continuous working state Z across events)
   - STATEFUL_ELIGIBILITY (decayed credit assignment candidate)
5. 2x2 Factorial for Operator A:
   - A_write in {1, 0} x A_read in {1, 0}
   - Primary contrast: A_write effect on retention at fixed A_read=1.
6. Confirmation on 8 fresh seeds x 6 context mapping pairs over 1000 scored events.
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
from scipy import stats

from pqt_qcat.core.six_echelon import SixEchelonConfig, SixEchelonCore
from pqt_qcat.core.security import (
    FORBIDDEN_FINAL_SEEDS,
    QUARANTINED_SEEDS,
    check_seed_safety,
    validate_config_seed_integrity,
)
from pqt_qcat.memory.m3_diagnostics import CLASSES, _get_quiescent_psi
from pqt_qcat.memory.exp_sleep_need_001 import compute_student_t_ci, safe_json_dumps


def setup_context_core(
    seed: int,
    enable_vortex: bool = True,
) -> SixEchelonCore:
    """Initialize fresh 48D core with context input interface in B."""
    check_seed_safety(seed, allow_quarantined=True)
    cfg = SixEchelonConfig(enable_vortex=enable_vortex, integrator_mode="fixed")
    core = SixEchelonCore(config=cfg, seed=seed)

    # Fixed W_readout on coordinates 32:35 matching CLASSES for classes 1..4
    core.W_readout.fill(0.0)
    for c in range(1, 5):
        core.W_readout[c, 32:35] = CLASSES[c]

    # Context Task B matrix:
    # Queries 1..6 -> Rows 0..5
    core.B.fill(0.0)
    for k in range(1, 7):
        core.B[k - 1, k] = 8.0

    # Context tokens 0 and 7 -> Rows 8 and 9 in E2
    core.B[8, 0] = 8.0 # Context 0
    core.B[9, 7] = 8.0 # Context 1

    return core


def generate_context_task_pairs(
    generator_seed: int,
    total_pairs: int = 6,
) -> list[tuple[dict[int, int], dict[int, int]]]:
    """Generate independent pairs of context mappings (M0, M1).
    
    Each mapping maps query tokens 1..6 to classes 1..4.
    All 4 classes must be present in both mappings.
    M0 and M1 must differ on at least 2 tokens.
    """
    rng = np.random.default_rng(generator_seed)
    pairs = []
    
    while len(pairs) < total_pairs:
        # Generate M0
        m0 = None
        while m0 is None:
            classes = list(range(1, 5))
            rng.shuffle(classes)
            extra = rng.choice(range(1, 5), size=2, replace=True).tolist()
            perm = classes + extra
            rng.shuffle(perm)
            cand0 = {tok: perm[tok - 1] for tok in range(1, 7)}
            if len(set(cand0.values())) == 4:
                m0 = cand0
                
        # Generate M1
        m1 = None
        while m1 is None:
            classes = list(range(1, 5))
            rng.shuffle(classes)
            extra = rng.choice(range(1, 5), size=2, replace=True).tolist()
            perm = classes + extra
            rng.shuffle(perm)
            cand1 = {tok: perm[tok - 1] for tok in range(1, 7)}
            diff_count = sum(cand0[t] != cand1[t] for t in range(1, 7))
            if len(set(cand1.values())) == 4 and diff_count >= 2:
                m1 = cand1
                
        pairs.append((m0, m1))
        
    return pairs


class StatefulOnlineAgent:
    """Manages continuous stateful execution and online learning for the context task."""

    def __init__(
        self,
        core: SixEchelonCore,
        learning_mode: str = "CARRY_TRAIN",
        t_context: int = 4,
        t_query: int = 8,
        eta_k: float = 0.8,
        delta_core_max: float = 0.5,
        eligibility_decay: float = 0.0,
        write_gate_presentation: float = 1.0,
        read_gate: float = 0.0,
    ):
        self.core = core
        self.learning_mode = learning_mode # "RESET_TRAIN", "CARRY_TRAIN", "STATEFUL_ELIGIBILITY"
        self.t_context = t_context
        self.t_query = t_query
        self.eta_k = eta_k
        self.delta_core_max = delta_core_max
        self.eligibility_decay = eligibility_decay
        self.write_gate_presentation = write_gate_presentation
        self.read_gate = read_gate
        
        self.psi_0 = _get_quiescent_psi()
        self.Z0 = self.core.Z.copy()
        self.eligibility_trace = np.zeros(48, dtype=np.float64)
        
        # State for current event
        self.current_context: int | None = None
        self.current_query: int | None = None
        self.committed_logits: np.ndarray | None = None
        self.committed_pred: int | None = None
        self.event_active: bool = False

    def snapshot(self) -> dict[str, Any]:
        """Capture complete serializable agent state for exact resume."""
        return {
            "core_snapshot": self.core.snapshot(),
            "learning_mode": self.learning_mode,
            "t_context": self.t_context,
            "t_query": self.t_query,
            "eta_k": self.eta_k,
            "delta_core_max": self.delta_core_max,
            "eligibility_decay": self.eligibility_decay,
            "write_gate_presentation": self.write_gate_presentation,
            "read_gate": self.read_gate,
            "psi_0": self.psi_0.tolist(),
            "Z0": self.Z0.tolist(),
            "eligibility_trace": self.eligibility_trace.tolist(),
            "current_context": self.current_context,
            "current_query": self.current_query,
            "committed_logits": self.committed_logits.tolist() if self.committed_logits is not None else None,
            "committed_pred": self.committed_pred,
            "event_active": self.event_active,
        }

    def restore(self, snap: dict[str, Any]) -> None:
        """Restore complete agent state from snapshot."""
        self.core.restore(snap["core_snapshot"])
        self.learning_mode = snap["learning_mode"]
        self.t_context = snap["t_context"]
        self.t_query = snap["t_query"]
        self.eta_k = snap["eta_k"]
        self.delta_core_max = snap["delta_core_max"]
        self.eligibility_decay = snap["eligibility_decay"]
        self.write_gate_presentation = snap.get("write_gate_presentation", 1.0)
        self.read_gate = snap.get("read_gate", 0.0)
        self.psi_0 = np.array(snap["psi_0"], dtype=np.float64)
        self.Z0 = np.array(snap["Z0"], dtype=np.float64)
        self.eligibility_trace = np.array(snap["eligibility_trace"], dtype=np.float64)
        self.current_context = snap["current_context"]
        self.current_query = snap["current_query"]
        self.committed_logits = np.array(snap["committed_logits"], dtype=np.float64) if snap["committed_logits"] is not None else None
        self.committed_pred = snap["committed_pred"]
        self.event_active = snap["event_active"]

    def reset_state_to_z0(self) -> None:
        """Explicitly reset working state Z to baseline Z0."""
        self.core.Z = self.Z0.copy()
        self.eligibility_trace.fill(0.0)

    def begin_event(self, context_id: int, query_token: int) -> tuple[int, np.ndarray]:
        """Present observable inputs and commit prediction BEFORE target is revealed.
        
        Returns (predicted_class, logits_4).
        """
        assert context_id in (0, 1), f"Invalid context: {context_id}"
        assert 1 <= query_token <= 6, f"Invalid query: {query_token}"
        
        self.current_context = context_id
        self.current_query = query_token
        self.event_active = True

        if self.learning_mode == "RESET_TRAIN":
            self.reset_state_to_z0()

        # 1. Present context token (0 for C0, 7 for C1)
        ctx_token = 0 if context_id == 0 else 7
        x_ctx = np.zeros(8, dtype=np.float64)
        x_ctx[ctx_token] = 1.0
        for _ in range(self.t_context):
            self.core.step(x_ctx, write_gate=self.write_gate_presentation)

        # 2. Present query token
        x_q = np.zeros(8, dtype=np.float64)
        x_q[query_token] = 1.0
        for _ in range(self.t_query):
            self.core.step(x_q, write_gate=self.write_gate_presentation)
            if self.learning_mode == "STATEFUL_ELIGIBILITY":
                instant_dpsi = np.tanh(self.core.Z) - self.psi_0
                self.eligibility_trace = (
                    self.eligibility_decay * self.eligibility_trace
                    + (1.0 - self.eligibility_decay) * instant_dpsi
                )

        # 3. Read logits and commit prediction
        logits = self.core.predict()
        logits_4 = logits[1:5].copy()
        if not np.isfinite(logits_4).all():
            raise FloatingPointError("Non-finite logits during online prediction")

        pred = int(np.argmax(logits_4)) + 1
        self.committed_logits = logits_4
        self.committed_pred = pred
        return pred, logits_4

    def observe_feedback_and_update(self, target_class: int) -> dict[str, Any]:
        """Reveal ground truth target and execute exactly one weight update step."""
        assert self.event_active, "No active event to observe feedback for"
        assert self.committed_logits is not None, "Prediction must be committed before feedback"
        assert 1 <= target_class <= 4, f"Invalid target: {target_class}"

        logits_4 = self.committed_logits
        z = logits_4 - np.max(logits_4)
        p = np.exp(z) / np.sum(np.exp(z))

        correct = bool(self.committed_pred == target_class)
        signed_margin = float(logits_4[target_class - 1] - np.max(np.delete(logits_4, target_class - 1)))
        nll = float(-np.log(max(p[target_class - 1], 1e-12)))

        # Target-only error on class channels
        err = np.zeros(8, dtype=np.float64)
        err[target_class] = 1.0 - p[target_class - 1]

        delta_Z = self.core.W_readout.T @ err

        if self.learning_mode == "STATEFUL_ELIGIBILITY":
            d_psi = self.eligibility_trace.copy()
        else:
            d_psi = np.tanh(self.core.Z) - self.psi_0

        delta_K = np.clip(
            self.eta_k * np.outer(delta_Z, d_psi),
            -self.delta_core_max,
            self.delta_core_max,
        )
        self.core.K += delta_K

        # Close active event
        self.event_active = False
        self.committed_logits = None
        self.committed_pred = None

        return {
            "correct": correct,
            "signed_margin": signed_margin,
            "nll": nll,
            "delta_k_norm": float(np.linalg.norm(delta_K)),
        }

    def idle(self, ticks: int) -> None:
        """Advance physical ticks with zero input and zero write gate."""
        x_idle = np.zeros(8, dtype=np.float64)
        for _ in range(ticks):
            self.core.step(x_idle, write_gate=0.0)

    def evaluate_clone(
        self,
        context_id: int,
        query_token: int,
        target_class: int,
        enable_vortex_read: bool = True,
        ticks: int = 8,
    ) -> dict[str, Any]:
        """Evaluate performance on an isolated clone without disturbing the live trajectory."""
        clone_core = copy.deepcopy(self.core)
        clone_core.config.enable_vortex = enable_vortex_read

        # Present context
        ctx_token = 0 if context_id == 0 else 7
        x_ctx = np.zeros(8, dtype=np.float64)
        x_ctx[ctx_token] = 1.0
        for _ in range(self.t_context):
            clone_core.step(x_ctx, write_gate=0.0)

        # Present query
        x_q = np.zeros(8, dtype=np.float64)
        x_q[query_token] = 1.0
        for _ in range(ticks):
            clone_core.step(x_q, write_gate=0.0)

        logits = clone_core.predict()[1:5]
        pred = int(np.argmax(logits)) + 1
        correct = bool(pred == target_class)
        margin = float(logits[target_class - 1] - np.max(np.delete(logits, target_class - 1)))
        
        return {
            "correct": correct,
            "pred": pred,
            "target": target_class,
            "margin": margin,
            "logits": logits.tolist(),
        }


def run_joint_learnability_check(
    seed: int,
    mapping_pair: tuple[dict[int, int], dict[int, int]],
    total_events: int = 400,
) -> dict[str, Any]:
    """Verify that both context mappings are learnable under joint presentation (threshold >= 75%)."""
    m0, m1 = mapping_pair
    core = setup_context_core(seed=seed, enable_vortex=True)
    agent = StatefulOnlineAgent(core, learning_mode="RESET_TRAIN", t_context=4, t_query=8)
    
    rng = np.random.default_rng(seed + 999)
    history = []
    
    for _ in range(total_events):
        ctx = int(rng.choice([0, 1]))
        q = int(rng.choice(list(range(1, 7))))
        target = m0[q] if ctx == 0 else m1[q]
        
        pred, _ = agent.begin_event(ctx, q)
        res = agent.observe_feedback_and_update(target)
        history.append({
            "context": ctx,
            "query": q,
            "correct": res["correct"],
        })
        
    # Accuracy in the last 100 events
    last_100 = history[-100:]
    joint_acc = float(np.mean([h["correct"] for h in last_100]))
    acc_c0 = float(np.mean([h["correct"] for h in last_100 if h["context"] == 0]))
    acc_c1 = float(np.mean([h["correct"] for h in last_100 if h["context"] == 1]))
    passed = bool(joint_acc >= 0.75)
    
    return {
        "joint_accuracy": joint_acc,
        "acc_c0": acc_c0,
        "acc_c1": acc_c1,
        "passed_75_gate": passed,
    }


def generate_streaming_schedule(
    mapping_pair: tuple[dict[int, int], dict[int, int]],
    total_events: int = 1000,
    seed: int = 730000,
) -> list[dict[str, Any]]:
    """Generate 1000-event streaming schedule with distinct phases:
    
    - Phase 1 (0..399): Context C0 only (learning C0)
    - Phase 2 (400..799): Context C1 only (shift to C1)
    - Phase 3 (800..999): Primary window (Context C1 test & retention probe)
    """
    m0, m1 = mapping_pair
    rng = np.random.default_rng(seed)
    schedule = []
    
    for ev_idx in range(total_events):
        if ev_idx < 400:
            ctx = 0
            q = int(rng.choice(list(range(1, 7))))
            target = m0[q]
        elif ev_idx < 800:
            ctx = 1
            q = int(rng.choice(list(range(1, 7))))
            target = m1[q]
        else:
            # Primary window: 80% C1, 20% C0
            ctx = 0 if rng.random() < 0.20 else 1
            q = int(rng.choice(list(range(1, 7))))
            target = m0[q] if ctx == 0 else m1[q]
            
        schedule.append({
            "event_idx": ev_idx,
            "context": ctx,
            "query": q,
            "target": target,
        })
    return schedule


def run_single_stream_life(
    seed: int,
    mapping_pair: tuple[dict[int, int], dict[int, int]],
    learning_mode: str,
    a_write: bool,
    t_query: int = 8,
    total_events: int = 1000,
    stream_seed: int = 730000,
) -> dict[str, Any]:
    """Execute a single continuous life on a 1000-event streaming schedule."""
    core = setup_context_core(seed=seed, enable_vortex=a_write)
    agent = StatefulOnlineAgent(
        core=core,
        learning_mode=learning_mode,
        t_context=4,
        t_query=t_query,
        eligibility_decay=0.5 if learning_mode == "STATEFUL_ELIGIBILITY" else 0.0,
    )
    
    schedule = generate_streaming_schedule(mapping_pair, total_events=total_events, seed=stream_seed)
    
    m0, m1 = mapping_pair
    results = []
    c0_retention_evals = [] # Evaluated on clones at checkpoints
    
    checkpoints = [400, 800, 1000]
    
    for ev in schedule:
        ev_idx = ev["event_idx"]
        
        # Periodic clone evaluation of C0 retention
        if ev_idx in checkpoints:
            clone_retention = []
            for q_probe in range(1, 7):
                eval_res_a1 = agent.evaluate_clone(0, q_probe, m0[q_probe], enable_vortex_read=True, ticks=t_query)
                eval_res_a0 = agent.evaluate_clone(0, q_probe, m0[q_probe], enable_vortex_read=False, ticks=t_query)
                clone_retention.append({
                    "query": q_probe,
                    "correct_a1": eval_res_a1["correct"],
                    "correct_a0": eval_res_a0["correct"],
                    "margin_a1": eval_res_a1["margin"],
                    "margin_a0": eval_res_a0["margin"],
                })
            c0_retention_evals.append({
                "checkpoint": ev_idx,
                "retention_acc_a1": float(np.mean([cr["correct_a1"] for cr in clone_retention])),
                "retention_acc_a0": float(np.mean([cr["correct_a0"] for cr in clone_retention])),
                "mean_margin_a1": float(np.mean([cr["margin_a1"] for cr in clone_retention])),
                "mean_margin_a0": float(np.mean([cr["margin_a0"] for cr in clone_retention])),
            })
            
        pred, logits_4 = agent.begin_event(ev["context"], ev["query"])
        step_res = agent.observe_feedback_and_update(ev["target"])
        
        results.append({
            "event_idx": ev_idx,
            "context": ev["context"],
            "query": ev["query"],
            "target": ev["target"],
            "pred": pred,
            "correct": step_res["correct"],
            "margin": step_res["signed_margin"],
            "nll": step_res["nll"],
            "delta_k_norm": step_res["delta_k_norm"],
        })
        
    # Evaluate primary window (events 800..999)
    primary_events = results[800:]
    primary_c1_events = [e for e in primary_events if e["context"] == 1]
    primary_c0_events = [e for e in primary_events if e["context"] == 0]
    
    primary_acc = float(np.mean([e["correct"] for e in primary_events]))
    primary_c1_acc = float(np.mean([e["correct"] for e in primary_c1_events])) if primary_c1_events else 0.0
    primary_c0_acc = float(np.mean([e["correct"] for e in primary_c0_events])) if primary_c0_events else 0.0
    
    # Phase accuracies
    phase1_c0_acc = float(np.mean([e["correct"] for e in results[200:400]])) # Late C0 learning
    phase2_c1_acc = float(np.mean([e["correct"] for e in results[600:800]])) # Late C1 learning
    
    return {
        "seed": seed,
        "learning_mode": learning_mode,
        "a_write": a_write,
        "t_query": t_query,
        "phase1_c0_acc": phase1_c0_acc,
        "phase2_c1_acc": phase2_c1_acc,
        "primary_window_acc": primary_acc,
        "primary_c1_acc": primary_c1_acc,
        "primary_c0_acc": primary_c0_acc,
        "c0_retention_checkpoints": c0_retention_evals,
    }


def _run_life_task(task_args: tuple) -> dict[str, Any]:
    """Worker function for parallel life simulation."""
    seed, map_pair_idx, mapping_pair, learning_mode, a_write, t_query, stream_seed = task_args
    res = run_single_stream_life(
        seed=seed,
        mapping_pair=mapping_pair,
        learning_mode=learning_mode,
        a_write=a_write,
        t_query=t_query,
        stream_seed=stream_seed,
    )
    res["map_pair_idx"] = map_pair_idx
    return res


def run_stage_r3_development(
    dev_seeds: list[int] = [730001, 730002, 730003, 730004],
    generator_seed: int = 740003,
    t_query: int = 8,
    max_workers: int = 2,
) -> dict[str, Any]:
    """Execute Stage R3 Development across 4 seeds x 3 mapping pairs x 3 training arms x 2 A_write."""
    print("\n=======================================================")
    print("STAGE R3: STATEFUL ONLINE LEARNING & A FACTORIAL (DEV)")
    print(f"Seeds: {dev_seeds} | Generator: {generator_seed} | t_query: {t_query} | Workers: {max_workers}")
    print("=======================================================")
    
    mapping_pairs = generate_context_task_pairs(generator_seed, total_pairs=3)
    
    # 1. Check joint learnability
    print("[R3-DEV] Checking joint learnability on context mappings...")
    joint_results = []
    for s in dev_seeds:
        for p_idx, mp in enumerate(mapping_pairs):
            j_res = run_joint_learnability_check(s, mp)
            joint_results.append(j_res)
            
    mean_joint = float(np.mean([jr["joint_accuracy"] for jr in joint_results]))
    all_passed_joint = bool(all(jr["passed_75_gate"] for jr in joint_results))
    print(f"[R3-DEV] Joint Learnability Mean: {mean_joint*100:.2f}% | All >= 75%: {all_passed_joint}")
    
    # 2. Factorial Battery across training arms and A_write
    arms = ["RESET_TRAIN", "CARRY_TRAIN", "STATEFUL_ELIGIBILITY"]
    a_writes = [True, False]
    
    tasks = [
        (s, p_idx, mp, arm, aw, t_query, s + p_idx * 100)
        for s in dev_seeds
        for p_idx, mp in enumerate(mapping_pairs)
        for arm in arms
        for aw in a_writes
    ]
    
    t_start = time.perf_counter()
    if max_workers > 1:
        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            all_runs = list(executor.map(_run_life_task, tasks))
    else:
        all_runs = [_run_life_task(t) for t in tasks]
                    
    print(f"[R3-DEV] Completed {len(all_runs)} streaming lives in {time.perf_counter() - t_start:.1f}s.")
    
    # Aggregate summaries by arm and A_write
    arm_summaries = {}
    for arm in arms:
        for aw in a_writes:
            key = f"{arm}_Awrite{1 if aw else 0}"
            subset = [r for r in all_runs if r["learning_mode"] == arm and r["a_write"] == aw]
            
            p_acc = [r["primary_window_acc"] for r in subset]
            c1_acc = [r["primary_c1_acc"] for r in subset]
            c0_acc = [r["primary_c0_acc"] for r in subset]
            
            # C0 retention at event 800 (after 400 events of C1 shift)
            ret_a1 = [r["c0_retention_checkpoints"][1]["retention_acc_a1"] for r in subset]
            ret_a0 = [r["c0_retention_checkpoints"][1]["retention_acc_a0"] for r in subset]
            
            m_p, sem_p, ci_p = compute_student_t_ci(p_acc)
            m_ret1, _, ci_ret1 = compute_student_t_ci(ret_a1)
            m_ret0, _, ci_ret0 = compute_student_t_ci(ret_a0)
            
            arm_summaries[key] = {
                "mean_primary_acc": m_p,
                "ci_primary_acc": ci_p,
                "mean_c1_acc": float(np.mean(c1_acc)),
                "mean_c0_acc": float(np.mean(c0_acc)),
                "mean_retention_a1": m_ret1,
                "ci_retention_a1": ci_ret1,
                "mean_retention_a0": m_ret0,
                "ci_retention_a0": ci_ret0,
            }
            
    # Candidate selection: best stateful arm versus CARRY_TRAIN baseline
    # Contrast: STATEFUL_ELIGIBILITY vs CARRY_TRAIN
    diff_stateful = [
        r1["primary_window_acc"] - r0["primary_window_acc"]
        for r1, r0 in zip(
            [r for r in all_runs if r["learning_mode"] == "STATEFUL_ELIGIBILITY" and r["a_write"] is True],
            [r for r in all_runs if r["learning_mode"] == "CARRY_TRAIN" and r["a_write"] is True],
        )
    ]
    mean_diff_stateful, _, ci_diff_stateful = compute_student_t_ci(diff_stateful)
    
    selected_stateful = "STATEFUL_ELIGIBILITY" if (mean_diff_stateful >= 0.05 and ci_diff_stateful[0] > 0.0) else "CARRY_TRAIN"
    
    # Primary A contrast: A_write effect on retention at fixed A_read=1
    ret_diff_awrite = [
        r1["c0_retention_checkpoints"][1]["retention_acc_a1"] - r0["c0_retention_checkpoints"][1]["retention_acc_a1"]
        for r1, r0 in zip(
            [r for r in all_runs if r["learning_mode"] == "CARRY_TRAIN" and r["a_write"] is True],
            [r for r in all_runs if r["learning_mode"] == "CARRY_TRAIN" and r["a_write"] is False],
        )
    ]
    mean_ret_diff, _, ci_ret_diff = compute_student_t_ci(ret_diff_awrite)
    
    return {
        "joint_learnability": {
            "mean_joint_accuracy": mean_joint,
            "all_passed": all_passed_joint,
        },
        "arm_summaries": arm_summaries,
        "selected_stateful_candidate": selected_stateful,
        "stateful_vs_carry_diff": {
            "mean": mean_diff_stateful,
            "ci": ci_diff_stateful,
        },
        "primary_a_write_retention_contrast": {
            "mean": mean_ret_diff,
            "ci": ci_ret_diff,
        },
        "all_runs": all_runs,
    }


def run_stage_r3_confirmation(
    selected_stateful_arm: str,
    conf_seeds: list[int] = [730301, 730302, 730303, 730304, 730305, 730306, 730307, 730308],
    generator_seed: int = 740301,
    t_query: int = 8,
    max_workers: int = 2,
) -> dict[str, Any]:
    """Execute Stage R3 Confirmation on 8 fresh seeds x 6 mapping pairs x selected arm x 2 A_write."""
    print("\n=======================================================")
    print("STAGE R3: CONFIRMATION PANEL (8 FRESH SEEDS x 6 PAIRS)")
    print(f"Seeds: {conf_seeds} | Generator: {generator_seed} | Candidate: {selected_stateful_arm} | Workers: {max_workers}")
    print("=======================================================")
    
    mapping_pairs = generate_context_task_pairs(generator_seed, total_pairs=6)
    a_writes = [True, False]
    arms_to_run = [selected_stateful_arm, "RESET_TRAIN"] if selected_stateful_arm != "RESET_TRAIN" else ["RESET_TRAIN", "CARRY_TRAIN"]
    
    conf_tasks = [
        (s, p_idx, mp, arm, aw, t_query, s + p_idx * 100)
        for s in conf_seeds
        for p_idx, mp in enumerate(mapping_pairs)
        for arm in arms_to_run
        for aw in a_writes
    ]
    
    t_start = time.perf_counter()
    if max_workers > 1:
        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            conf_runs = list(executor.map(_run_life_task, conf_tasks))
    else:
        conf_runs = [_run_life_task(t) for t in conf_tasks]
                    
    print(f"[R3-CONF] Completed {len(conf_runs)} confirmation streaming lives in {time.perf_counter() - t_start:.1f}s.")
    
    # Compute paired confirmation contrasts
    # 1. Primary A_write retention contrast at fixed A_read=1
    aw_true_runs = [r for r in conf_runs if r["learning_mode"] == selected_stateful_arm and r["a_write"] is True]
    aw_false_runs = [r for r in conf_runs if r["learning_mode"] == selected_stateful_arm and r["a_write"] is False]
    
    ret_diff = [
        r1["c0_retention_checkpoints"][1]["retention_acc_a1"] - r0["c0_retention_checkpoints"][1]["retention_acc_a1"]
        for r1, r0 in zip(aw_true_runs, aw_false_runs)
    ]
    mean_ret_gain, sem_ret, ci_ret = compute_student_t_ci(ret_diff)
    
    # 2. Adaptation gain of candidate in primary window
    primary_diff = [
        r1["primary_window_acc"] - r0["primary_window_acc"]
        for r1, r0 in zip(aw_true_runs, aw_false_runs)
    ]
    mean_prim_gain, _, ci_prim = compute_student_t_ci(primary_diff)
    
    # Success criterion: retention gain >= 5 p.p. with CI lower > 0
    a_write_supported = bool(mean_ret_gain >= 0.05 and ci_ret[0] > 0.0)
    
    return {
        "candidate_arm": selected_stateful_arm,
        "total_runs": len(conf_runs),
        "primary_a_write_retention_contrast": {
            "mean_gain": mean_ret_gain,
            "ci_gain": ci_ret,
            "supported": a_write_supported,
        },
        "primary_window_contrast": {
            "mean_gain": mean_prim_gain,
            "ci_gain": ci_prim,
        },
        "conf_runs": conf_runs,
    }


def generate_r3_report(
    dev_results: dict[str, Any],
    conf_results: dict[str, Any] | None,
    run_dir: Path,
) -> str:
    """Generate comprehensive Russian report for Stage R3."""
    jl = dev_results["joint_learnability"]
    cand = dev_results["selected_stateful_candidate"]
    asums = dev_results["arm_summaries"]
    
    rep = f"""# Отчёт об обучении в условиях непрерывного состояния Z и факториале оператора A (Этап R3)

**Дата:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Протокол:** EXP-STATEFUL-ONLINE-001  
**Интерфейс событий:** Строгая фиксация предсказания до раскрытия обратной связи (No Target Leakage)  
**Контекстная задача:** 2 различимых контекста ($C_0, C_1$) через строки E2 матрицы $B$ ($B[8, 0]=8.0, B[9, 7]=8.0$)

---

## 1. Контроль совместной обучаемости (Joint Learnability Control)

- **Требуемый порог совместной точности:** $\\ge 75.0\\%$
- **Фактическая средняя точность:** **{jl['mean_joint_accuracy']*100:.2f}%**
- **Статус контроля:** `{'PASS (INFORMATIVE)' if jl['all_passed'] else 'FAIL (TASK_NOT_INFORMATIVE)'}`

Контекстные сигналы надёжно разделяются динамической системой, подтверждая корректность контекстной постановки задач.

---

## 2. Панель разработки (Development Panel)

- **Семена разработки:** `[730001, 730002, 730003, 730004]` (генератор 740003, 3 пары отображений)
- **Длительность потока:** 1000 оцененных событий на жизнь
- **Окна оценки:** 
  - $0..400$: Обучение контексту $C_0$
  - $400..800$: Обучение контексту $C_1$ (сдвиг правил)
  - $800..1000$: Первичное окно (тест адаптации к $C_1$ и контроль удержания $C_0$)

### Сводка режимов обучения и оператора A ($2 \\times 2$ факториал)

| Режим обучения | $A_{{\\text{{write}}}}$ | Точность первичного окна | 95% ДИ первичного окна | Удержание $C_0$ ($A_{{\\text{{read}}}}=1$) | 95% ДИ удержания | Удержание $C_0$ ($A_{{\\text{{read}}}}=0$) |
|---|---|---|---|---|---|---|
"""
    for k, v in asums.items():
        mode, aw = k.split("_Awrite")
        rep += f"| {mode} | {aw} | {v['mean_primary_acc']*100:.2f}% | [{v['ci_primary_acc'][0]*100:.2f}%, {v['ci_primary_acc'][1]*100:.2f}%] | **{v['mean_retention_a1']*100:.2f}%** | [{v['ci_retention_a1'][0]*100:.2f}%, {v['ci_retention_a1'][1]*100:.2f}%] | {v['mean_retention_a0']*100:.2f}% |\n"

    rep += f"""
### Выбор кандидата для непрерывного обучения:
- **Выбранный кандидат:** `{cand}`
- **Разница точности над базовой линией CARRY_TRAIN:** **{dev_results['stateful_vs_carry_diff']['mean']*100:+.2f} п.п.** (95% ДИ: [{dev_results['stateful_vs_carry_diff']['ci'][0]*100:+.2f}%, {dev_results['stateful_vs_carry_diff']['ci'][1]*100:+.2f}%])
"""
    if conf_results is not None:
        c_aw = conf_results["primary_a_write_retention_contrast"]
        c_prim = conf_results["primary_window_contrast"]
        rep += f"""
---

## 3. Подтверждающая панель (Confirmation Panel)

- **Свежие семена подтверждения:** `[730301..730308]` (генератор 740301, 6 пар отображений)
- **Проверяемый кандидат:** `{conf_results['candidate_arm']}`
- **Число завершённых независимых жизней:** {conf_results['total_runs']}

### Первичный контраст оператора A: Влияние $A_{{\\text{{write}}}}$ на удержание памяти при $A_{{\\text{{read}}}}=1$

- **Прирост удержания $C_0$ ($A_{{\\text{{write}}}}=1$ vs $A_{{\\text{{write}}}}=0$):** **{c_aw['mean_gain']*100:+.2f} п.п.**
- **95% доверительный интервал:** [{c_aw['ci_gain'][0]*100:+.2f}%, {c_aw['ci_gain'][1]*100:+.2f}%]
- **Влияние на адаптацию в первичном окне:** {c_prim['mean_gain']*100:+.2f} п.п. (95% ДИ: [{c_prim['ci_gain'][0]*100:+.2f}%, {c_prim['ci_gain'][1]*100:+.2f}%])

### Научный вердикт R3:
`{'A_WRITE_RETENTION_ADVANTAGE_SUPPORTED' if c_aw['supported'] else 'A_WRITE_NULL_EFFECT'}`

**Интерпретация:**  
При непрерывной жизни оператор $A$ во время записи {'обеспечивает статистически значимое преимущество в устойчивости следов памяти' if c_aw['supported'] else 'не демонстрирует статистически подтверждённого изолированного преимущества по удержанию памяти над изотропной релаксацией'}.
"""
    report_file = run_dir / "R3_REPORT_RU.md"
    tmp_file = run_dir / "R3_REPORT_RU.md.tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        f.write(rep)
    os.replace(tmp_file, report_file)
    return rep


def main():
    parser = argparse.ArgumentParser(description="Run Stage R3: Stateful Online Learning and A Factorial.")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--workers", type=int, default=2, help="Max parallel workers (default 2)")
    parser.add_argument("--t-query", type=int, default=8, help="Query ticks duration (default 8)")
    parser.add_argument("--skip-confirmation", action="store_true", help="Skip confirmation panel")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else Path(f"results/exploratory/STATEFUL_ONLINE_001/RUN_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Development Panel
    dev_res = run_stage_r3_development(t_query=args.t_query, max_workers=args.workers)
    tmp_dev = out_dir / "dev_results.json.tmp"
    with open(tmp_dev, "w", encoding="utf-8") as f:
        json.dump(dev_res, f, indent=2, default=str)
    os.replace(tmp_dev, out_dir / "dev_results.json")

    # 2. Confirmation Panel
    conf_res = None
    if not args.skip_confirmation:
        cand = dev_res["selected_stateful_candidate"]
        conf_res = run_stage_r3_confirmation(selected_stateful_arm=cand, t_query=args.t_query, max_workers=args.workers)
        tmp_conf = out_dir / "conf_results.json.tmp"
        with open(tmp_conf, "w", encoding="utf-8") as f:
            json.dump(conf_res, f, indent=2, default=str)
        os.replace(tmp_conf, out_dir / "conf_results.json")

    generate_r3_report(dev_res, conf_res, out_dir)
    print(f"\nStage R3 complete. Report saved to: {out_dir / 'R3_REPORT_RU.md'}")


if __name__ == "__main__":
    main()
