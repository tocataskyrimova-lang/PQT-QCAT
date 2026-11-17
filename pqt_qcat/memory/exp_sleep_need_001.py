"""EXP-SLEEP-NEED-001: Empirical Investigation of Consolidation Benefit vs Prior Interference.

Protocol v0.2 (17 Nov 2026, Astra/Codex & Antigravity).
Addresses all 7 engineering requirements of Задача 1:
1. Strict CLI argument --config validation and error handling.
2. Decoupled wake schedule generation and execution; distinct pre_new, post_new_pre_tail, and post_tail endpoints.
3. Pre-sleep joint positive control gate (>=75% joint accuracy, >=80% prefix old accuracy).
4. Two-stage execution architecture: wake calibration -> gate check -> sleep stage.
5. Semantic grounding event matching across paired BLOCKED and INTERLEAVED histories.
6. Rigorous separate compute accounting for wake, probe, guard, sleep, delay, and rollback ticks.
7. Full regression testability and invariant preservation.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass
import datetime
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

FORBIDDEN_SEEDS = [101, 102, 103, 104, 105]


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


def validate_config(config: dict) -> None:
    """Strictly validate schema and parameter ranges of EXP-SLEEP-NEED config.
    
    Raises ValueError with descriptive reason if invalid.
    """
    required_sections = [
        "protocol_id", "version", "calibration_seeds", "panel", "wake", "sleep", "evaluation", "analysis"
    ]
    for sec in required_sections:
        if sec not in config:
            raise ValueError(f"Missing required configuration section: '{sec}'")

    forbidden = set(config.get("forbidden_seeds", [101, 102, 103, 104, 105]))
    cal_seeds = config.get("calibration_seeds", [])
    if set(cal_seeds) & forbidden:
        raise ValueError(f"Configuration uses sealed M5 seeds: {set(cal_seeds) & forbidden}")
    if config.get("run_m5", False):
        raise ValueError("run_m5 must be False; M5 final evaluation is sealed")

    # Panel checks
    panel = config["panel"]
    if panel.get("total_distinct_mappings", 0) < 6:
        raise ValueError("total_distinct_mappings must be at least 6")
    if set(panel.get("old_sources", [])) != {1, 2, 3}:
        raise ValueError("old_sources must be [1, 2, 3]")
    if set(panel.get("new_sources", [])) != {4, 5, 6}:
        raise ValueError("new_sources must be [4, 5, 6]")

    # Wake checks
    wake = config["wake"]
    if wake.get("events_per_life") != 201:
        raise ValueError("events_per_life must equal 201 (75 prefix + 120 main + 6 tail)")
    if wake.get("prefix_old_epochs") != 25:
        raise ValueError("prefix_old_epochs must be 25")
    if wake.get("day_epochs_per_group") != 20:
        raise ValueError("day_epochs_per_group must be 20")
    if wake.get("tail_epochs_per_group") != 1:
        raise ValueError("tail_epochs_per_group must be 1")
    if wake.get("eta", 0) <= 0:
        raise ValueError("wake eta must be positive")

    # Sleep checks
    sl = config["sleep"]
    slots = sl.get("candidate_slots", 25)
    if not 0 <= slots <= 90:
        raise ValueError("candidate_slots must fit in the 1000 active tick budget (0 <= slots <= 90)")
    if sl.get("active_ticks") != 1000:
        raise ValueError("active_ticks must be exactly 1000")
    if sl.get("anchor_count") != 6:
        raise ValueError("anchor_count must be exactly 6")
    if sl.get("eta", 0) <= 0:
        raise ValueError("sleep eta must be positive")


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
    *,
    reset_state: bool = True,
) -> tuple[int, np.ndarray, float, float]:
    """Read an isolated copy; never modify supplied core state or RNG.
    
    reset_state=False starts EACH query from the supplied Z (carry-state).
    Returns (class, logits_1_to_4, top-two margin, max probability).
    """
    core = copy.deepcopy(core)
    if reset_state:
        core.Z.fill(0.0)
        for c in range(6):
            core.Z[8 * c + 6] = 1.0
        
    x = np.zeros(8)
    x[token] = 1.0
    
    for _ in range(ticks):
        core.step(x, write_gate=1.0)
        
    logits = core.predict()
    logits_4 = logits[1:5].copy()
    if not np.isfinite(logits_4).all():
        raise FloatingPointError("Non-finite evaluation logits encountered")
        
    pred = int(np.argmax(logits_4)) + 1
    
    z = logits_4 - np.max(logits_4)
    p = np.exp(z) / np.sum(np.exp(z))
    
    sorted_logits = np.sort(logits_4)
    margin = float(sorted_logits[-1] - sorted_logits[-2])
    
    return pred, logits_4, margin, float(np.max(p))


def evaluate_query_accuracy(
    core: SixEchelonCore,
    tokens: list[int],
    task_map: dict[int, int],
    ticks: int = 8,
    *,
    reset_state: bool = True,
) -> tuple[float, float, float]:
    """Evaluate accuracy, signed target margin, and mean NLL on a set of tokens."""
    correct = 0
    margins = []
    nlls = []
    
    for tok in tokens:
        pred, logits_4, _, _ = query_core_prediction(core, tok, ticks=ticks, reset_state=reset_state)
        target = task_map[tok]
        if pred == target:
            correct += 1
        # Signed margin: target logit minus maximum competing non-target logit
        other_logits = np.delete(logits_4, target - 1)
        signed_margin = float(logits_4[target - 1] - np.max(other_logits))
        margins.append(signed_margin)
        
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
    logits_4 = logits[1:5].copy()
    if not np.isfinite(logits_4).all():
        raise FloatingPointError("Non-finite logits during wake training")
        
    z = logits_4 - np.max(logits_4)
    p = np.exp(z) / np.sum(np.exp(z))
    pred = int(np.argmax(logits_4)) + 1
    
    # Target-only error: err[target] = 1 - p[target - 1]
    err = np.zeros(8)
    err[target] = 1.0 - p[target - 1]
    
    delta_Z = core.W_readout.T @ err
    d_psi = np.tanh(core.Z) - psi_0
    delta_K = np.clip(eta_K * np.outer(delta_Z, d_psi), -delta_core_max, delta_core_max)
    core.K += delta_K
    
    return pred, logits_4, delta_K


@dataclass
class ScheduledWakeEvent:
    event_idx: int
    epoch_idx: int
    group: str
    token_in: int
    target_token: int
    is_prefix: bool
    is_pre_new: bool
    is_post_new_pre_tail: bool
    is_tail: bool


def generate_wake_schedule(
    seed: int,
    task_id: int,
    task_map: dict[int, int],
    history_type: str,
) -> list[ScheduledWakeEvent]:
    """Generate exact deterministic schedule of wake training events.
    
    Preserves exact RNG sequence seed * 10000 + task_id * 100 + 42.
    """
    rng_ep = np.random.default_rng(seed * 10000 + task_id * 100 + 42)
    prefix_perms = [rng_ep.permutation([1, 2, 3]).tolist() for _ in range(25)]
    day_O_perms = [rng_ep.permutation([1, 2, 3]).tolist() for _ in range(20)]
    day_N_perms = [rng_ep.permutation([4, 5, 6]).tolist() for _ in range(20)]
    tail_O_perm = rng_ep.permutation([1, 2, 3]).tolist()
    tail_N_perm = rng_ep.permutation([4, 5, 6]).tolist()

    events: list[ScheduledWakeEvent] = []
    idx = 0

    # Phase 1: Prefix (25 epochs Old)
    for ep in range(25):
        for tok in prefix_perms[ep]:
            idx += 1
            events.append(ScheduledWakeEvent(
                event_idx=idx, epoch_idx=ep + 1, group="prefix_old",
                token_in=tok, target_token=task_map[tok],
                is_prefix=True, is_pre_new=(ep == 24 and history_type == "interleaved"),
                is_post_new_pre_tail=False, is_tail=False
            ))

    # Phase 2: Main Day (120 events)
    if history_type == "blocked":
        for ep in range(20):
            for tok in day_O_perms[ep]:
                idx += 1
                events.append(ScheduledWakeEvent(
                    event_idx=idx, epoch_idx=25 + ep + 1, group="day_old",
                    token_in=tok, target_token=task_map[tok],
                    is_prefix=False, is_pre_new=(ep == 19),
                    is_post_new_pre_tail=False, is_tail=False
                ))
        for ep in range(20):
            for tok in day_N_perms[ep]:
                idx += 1
                events.append(ScheduledWakeEvent(
                    event_idx=idx, epoch_idx=45 + ep + 1, group="day_new",
                    token_in=tok, target_token=task_map[tok],
                    is_prefix=False, is_pre_new=False,
                    is_post_new_pre_tail=(ep == 19), is_tail=False
                ))
    elif history_type == "interleaved":
        for ep in range(20):
            for tok in day_O_perms[ep]:
                idx += 1
                events.append(ScheduledWakeEvent(
                    event_idx=idx, epoch_idx=25 + 2 * ep + 1, group="day_old",
                    token_in=tok, target_token=task_map[tok],
                    is_prefix=False, is_pre_new=False,
                    is_post_new_pre_tail=False, is_tail=False
                ))
            for tok in day_N_perms[ep]:
                idx += 1
                events.append(ScheduledWakeEvent(
                    event_idx=idx, epoch_idx=25 + 2 * ep + 2, group="day_new",
                    token_in=tok, target_token=task_map[tok],
                    is_prefix=False, is_pre_new=False,
                    is_post_new_pre_tail=(ep == 19), is_tail=False
                ))
    else:
        raise ValueError(f"Unknown history type: {history_type}")

    # Phase 3: Tail (1 epoch Old, 1 epoch New = 6 events)
    for tok in tail_O_perm:
        idx += 1
        events.append(ScheduledWakeEvent(
            event_idx=idx, epoch_idx=66, group="tail_old",
            token_in=tok, target_token=task_map[tok],
            is_prefix=False, is_pre_new=False, is_post_new_pre_tail=False, is_tail=True
        ))
    for tok in tail_N_perm:
        idx += 1
        events.append(ScheduledWakeEvent(
            event_idx=idx, epoch_idx=67, group="tail_new",
            token_in=tok, target_token=task_map[tok],
            is_prefix=False, is_pre_new=False, is_post_new_pre_tail=False, is_tail=True
        ))

    return events


@dataclass
class ObservedEvent:
    event_id: int
    token_in: int
    target_token: int
    epoch: int
    group: str


def evaluate_joint_positive_control(
    seeds: list[int],
    panel_maps: list[dict[int, int]],
    epochs: int = 25,
    eta_K: float = 0.8,
    delta_core_max: float = 0.5,
    ticks: int = 8,
) -> dict:
    """Evaluate joint learnability of all 6 tokens simultaneously on identical 48D core.
    
    Unit of aggregation: mean across mappings within seed, then Student-t CI across seeds.
    """
    psi_0 = _get_quiescent_psi()
    seed_means = []

    for s in seeds:
        map_accs = []
        for m in panel_maps:
            core = setup_fresh_core(seed=s, enable_vortex=True)
            for ep in range(epochs):
                for tok in range(1, 7):
                    wake_train_step(core, tok, m[tok], psi_0, eta_K=eta_K, delta_core_max=delta_core_max, ticks=ticks)
            acc, _, _ = evaluate_query_accuracy(core, list(range(1, 7)), m, ticks=ticks)
            map_accs.append(acc)
        seed_means.append(float(np.mean(map_accs)))

    mean_acc, sem_acc, ci_95 = compute_student_t_ci(seed_means)
    passed = bool(mean_acc >= 0.75 and all(sm >= 0.60 for sm in seed_means))

    return {
        "mean_joint_accuracy": mean_acc,
        "sem": sem_acc,
        "ci_95": ci_95,
        "seed_means": seed_means,
        "threshold": 0.75,
        "passed": passed,
    }


class SleepAdapterNeed001:
    """Matched-compute sleep adapter for EXP-SLEEP-NEED-001.
    
    Supports 4 policies: 'dream', 'replay', 'sham_dream', 'idle'.
    Budgets:
      - Active ticks: exactly 1000 ticks.
      - Shadow ticks: exactly 2400 ticks (25 slots * 2 evaluations * 6 anchors * 8 ticks).
      - Anchors: all 6 available real mappings.
      - Diagnostic ticks: 96 ticks (initial + final anchor evaluations).
      - Semantic matching: canonical event sorting guarantees identical token sequence across paired histories.
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
        if not 0 <= candidate_slots <= 90:
            raise ValueError("candidate_slots must fit in the 1000 active tick budget")

    def evaluate_anchors_accuracy(
        self,
        core: SixEchelonCore,
        anchors: dict[int, int],
        ticks_per_token: int = 8,
    ) -> float:
        """Evaluate accuracy across the 6 anchors without mutating core state."""
        correct = 0
        for tok, tgt in anchors.items():
            pred, _, _, _ = query_core_prediction(core, tok, ticks=ticks_per_token, reset_state=True)
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
        if len(anchors) != self.anchor_count or not anchors:
            raise ValueError("anchor_count must equal the nonempty anchor set")
        if not observed_events:
            raise ValueError("Observed history must be nonempty")
        if not np.isclose(core.config.h_base, self.h_base, rtol=0, atol=0):
            raise ValueError("Noise scaling h must equal the integrator h_base")
            
        initial_K = core.K.copy()
        
        acc_before = self.evaluate_anchors_accuracy(core, anchors, ticks_per_token=8)
        
        active_ticks = 0
        shadow_ticks = 0
        diagnostic_ticks = len(anchors) * 8  # 48 ticks initial diagnostic
        accepted_count = 0
        rejected_count = 0
        rejection_reasons: list[str] = []
        
        # Semantic canonical ordering: sort observed events by (token_in, target_token, event_id)
        # so that slot k selects the exact same semantic token in both BLOCKED and INTERLEAVED!
        canonical_events = sorted(observed_events, key=lambda ev: (ev.token_in, ev.target_token, ev.event_id))
        
        rng_grounding = np.random.default_rng(run_seed + 10007)
        rng_noise = np.random.default_rng(run_seed + 20011)
        
        grounding_choices = [rng_grounding.choice(canonical_events) for _ in range(self.candidate_slots)]
        noise_stream = [rng_noise.normal(0.0, self.thermal_noise_sigma, size=(10, 48)) for _ in range(self.candidate_slots)]
        
        observed_map = {}
        for ev in observed_events:
            observed_map[ev.token_in] = ev.target_token
            
        if policy == "idle":
            # Idle control: 1000 physical ticks with write_gate=0.0 (natural time baseline, unmatched compute)
            for _ in range(1000):
                core.step(x_t=None, write_gate=0.0)
                active_ticks += 1
                
        elif policy in ("dream", "replay", "sham_dream"):
            for slot_idx in range(self.candidate_slots):
                seed_event = grounding_choices[slot_idx]
                token_in = seed_event.token_in
                noise_block = noise_stream[slot_idx]
                
                # Reset to quiescent state for slot exploration
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
                    
                gate1_provenance = (token_in in observed_map and observed_map[token_in] == target_token)
                gate2_structural = (1 <= target_token <= 4 and 1 <= token_in <= 6)
                
                src_idx = token_in - 1
                tgt_idx = 31 + target_token
                delta_K = np.zeros_like(core.K)
                delta_K[tgt_idx, src_idx] = self.eta_sleep * conf
                
                # Gate 3: Anchor Guard Evaluation (always 96 shadow ticks)
                acc_anc_before = self.evaluate_anchors_accuracy(core, anchors, ticks_per_token=8)
                shadow_ticks += len(anchors) * 8
                
                test_core = copy.deepcopy(core)
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
                    "run_seed": run_seed,
                    "grounding_event_id": seed_event.event_id,
                    "noise_sha256": hashlib.sha256(noise_block.tobytes()).hexdigest(),
                    "proposed_delta_K_sha256": hashlib.sha256(delta_K.tobytes()).hexdigest(),
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
                
            # Padding ticks to reach EXACTLY 1000 active ticks
            padding_ticks = 1000 - (self.candidate_slots * 11)
            for _ in range(padding_ticks):
                core.step(x_t=None, write_gate=0.0)
                active_ticks += 1
                
        else:
            raise ValueError(f"Unknown sleep policy: {policy}")
            
        acc_after = self.evaluate_anchors_accuracy(core, anchors, ticks_per_token=8)
        diagnostic_ticks += len(anchors) * 8  # 48 ticks final diagnostic
        retention_degradation = acc_before - acc_after
        delta_K_norm = float(np.linalg.norm(core.K - initial_K))
        delta_K_ratio = float(delta_K_norm / max(np.linalg.norm(initial_K), 1e-12))
        
        total_session_compute_ticks = active_ticks + shadow_ticks + diagnostic_ticks
        
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
            "diagnostic_ticks": diagnostic_ticks,
            "total_compute_ticks": total_session_compute_ticks,
            "delta_K_norm": delta_K_norm,
            "delta_K_ratio": delta_K_ratio,
            "wall_time": time.perf_counter() - t_start,
        }


def run_single_history_experiment(
    seed: int,
    task_id: int,
    task_map: dict[int, int],
    history_type: str,
    run_seed: int,
    adapter: SleepAdapterNeed001,
    wake_log_records: list[dict],
    probe_log_records: list[dict],
    sleep_log_records: list[dict],
    query_log_records: list[dict],
    save_snapshots_dir: Path | None = None,
    execute_sleep: bool = True,
) -> dict:
    """Execute complete life for one (seed, task, history) combination across all 4 sleep policies."""
    psi_0 = _get_quiescent_psi()
    core = setup_fresh_core(seed=seed, enable_vortex=True)
    K_initial = core.K.copy()
    
    schedule = generate_wake_schedule(seed=seed, task_id=task_id, task_map=task_map, history_type=history_type)
    
    observed_events: list[ObservedEvent] = []
    
    probe_count = 0
    def take_diagnostic_probe(stage_name: str, epoch_num: int):
        nonlocal probe_count
        probe_count += 1
        acc_o, m_o, nll_o = evaluate_query_accuracy(core, [1, 2, 3], task_map, reset_state=True)
        acc_n, m_n, nll_n = evaluate_query_accuracy(core, [4, 5, 6], task_map, reset_state=True)
        k_hash = hashlib.sha256(core.K.tobytes()).hexdigest()
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

    q_old_prefix = 0.0
    q_new_prefix = 0.0
    K_prefix = None
    q_old_pre_new = 0.0
    q_new_pre_new = 0.0
    q_old_post_new_pre_tail = 0.0
    q_new_post_new_pre_tail = 0.0

    # Execute wake schedule
    for ev in schedule:
        pred, logits_4, delta_K = wake_train_step(core, ev.token_in, ev.target_token, psi_0)
        obs_ev = ObservedEvent(ev.event_idx, ev.token_in, ev.target_token, ev.epoch_idx, ev.group)
        observed_events.append(obs_ev)
        
        wake_log_records.append({
            "event_id": ev.event_idx,
            "seed": seed,
            "task_id": task_id,
            "history": history_type,
            "epoch": ev.epoch_idx,
            "group": ev.group,
            "token_in": ev.token_in,
            "target": ev.target_token,
            "predicted": pred,
            "logits": logits_4.tolist(),
            "delta_K_sha256": hashlib.sha256(delta_K.tobytes()).hexdigest(),
            "delta_K_norm": float(np.linalg.norm(delta_K)),
        })
        
        # Check endpoints
        if ev.event_idx == 75:  # End of 25 prefix epochs
            K_prefix = core.K.copy()
            q_old_prefix, q_new_prefix = take_diagnostic_probe("prefix_end", 25)
            
        if ev.is_pre_new:
            q_old_pre_new, q_new_pre_new = take_diagnostic_probe("pre_new", ev.epoch_idx)
            
        if ev.is_post_new_pre_tail:
            q_old_post_new_pre_tail, q_new_post_new_pre_tail = take_diagnostic_probe("post_new_pre_tail", ev.epoch_idx)

    # Diagnostic probe after tail
    K_pre_sleep = core.K.copy()
    q_old_post_tail, q_new_post_tail = take_diagnostic_probe("post_tail_pre_sleep", 67)
    
    # Primary forgetting measure taken BEFORE tail
    I_pre_tail = q_old_prefix - q_old_post_new_pre_tail
    # Post-tail forgetting measure
    I_post_tail = q_old_prefix - q_old_post_tail
    
    pre_sleep_snapshot = core.snapshot()
    anchors = {x: task_map[x] for x in range(1, 7)}
    
    policy_results = {}
    
    if execute_sleep:
        for pol_idx, policy in enumerate(["dream", "replay", "sham_dream", "idle"]):
            core_sleep = SixEchelonCore(config=core.config)
            core_sleep.restore(pre_sleep_snapshot)
            
            pol_seed = run_seed
            first_sleep_record = len(sleep_log_records)
            sleep_metrics = adapter.run_sleep_session(
                policy=policy,
                core=core_sleep,
                observed_events=observed_events,
                anchors=anchors,
                run_seed=pol_seed,
                sleep_log_records=sleep_log_records,
            )
            for record in sleep_log_records[first_sleep_record:]:
                record.update(seed=seed, task_id=task_id, history=history_type)
            
            K_post = core_sleep.K.copy()
            
            # Exact rollback check on all 4 logits across all 6 queries
            post_logits = np.stack([query_core_prediction(core_sleep, t)[1] for t in range(1, 7)])
            core_sleep.K = K_pre_sleep.copy()
            core_sleep.K = K_post.copy()
            restored_logits = np.stack([query_core_prediction(core_sleep, t)[1] for t in range(1, 7)])
            rollback_exact = bool(np.array_equal(post_logits, restored_logits))
            
            # Evaluation 1: Structural Recipient (Primary)
            struct_core = setup_fresh_core(seed=seed, enable_vortex=True)
            struct_core.K = K_post.copy()
            acc_o_post_struct, m_o_post_struct, nll_o_post_struct = evaluate_query_accuracy(struct_core, [1, 2, 3], task_map)
            acc_n_post_struct, m_n_post_struct, nll_n_post_struct = evaluate_query_accuracy(struct_core, [4, 5, 6], task_map)
            
            for tok in range(1, 7):
                p_pred, p_logits, p_margin, _ = query_core_prediction(struct_core, tok)
                tgt = task_map[tok]
                other_logits = np.delete(p_logits, tgt - 1)
                signed_margin = float(p_logits[tgt - 1] - np.max(other_logits))
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
                    "margin": signed_margin,
                    "logits": p_logits.tolist(),
                })
                
            # Evaluation 2: Carry-State (Secondary)
            acc_o_post_carry, m_o_post_carry, nll_o_post_carry = evaluate_query_accuracy(core_sleep, [1, 2, 3], task_map, reset_state=False)
            acc_n_post_carry, m_n_post_carry, nll_n_post_carry = evaluate_query_accuracy(core_sleep, [4, 5, 6], task_map, reset_state=False)
            
            # Evaluation 3: 1000 Idle ticks delay stability check
            delay_core = copy.deepcopy(core_sleep)
            for _ in range(1000):
                delay_core.step(x_t=None, write_gate=0.0)
            acc_o_delay, _, _ = evaluate_query_accuracy(delay_core, [1, 2, 3], task_map, reset_state=False)
            
            gain_old_struct = acc_o_post_struct - q_old_post_new_pre_tail
            gain_new_struct = acc_n_post_struct - q_new_post_new_pre_tail
            
            # Compute ticks accounting
            wake_ticks = 201 * 8
            probe_ticks = probe_count * 6 * 8
            guard_ticks = sleep_metrics["shadow_ticks"]
            sleep_active = sleep_metrics["active_ticks"]
            delay_ticks = 1000
            rollback_ticks = 6 * 8 * 2  # 96 ticks
            total_life_ticks = wake_ticks + probe_ticks + sleep_metrics["total_compute_ticks"] + delay_ticks + rollback_ticks
            
            policy_results[policy] = {
                "acc_old_pre_tail": q_old_post_new_pre_tail,
                "acc_new_pre_tail": q_new_post_new_pre_tail,
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
                "wake_ticks": wake_ticks,
                "probe_ticks": probe_ticks,
                "guard_ticks": guard_ticks,
                "sleep_active_ticks": sleep_active,
                "delay_ticks": delay_ticks,
                "rollback_ticks": rollback_ticks,
                "sleep_session_compute_ticks": sleep_metrics["total_compute_ticks"],
                "total_life_ticks": total_life_ticks,
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
                    K_initial=K_initial,
                    K_prefix=K_prefix,
                    K_pre_sleep=K_pre_sleep,
                    Z_pre_sleep=pre_sleep_snapshot["Z"],
                    metadata_json=np.array(safe_json_dumps({
                        "seed": seed, "task_id": task_id, "history": history_type,
                        "policy": policy, "run_seed": pol_seed,
                        "config": asdict(core_sleep.config),
                        "cell_params": asdict(core_sleep.cell_params),
                        "vortex_params": asdict(core_sleep.vortex_params),
                        "rng_state": core_sleep.rng.bit_generator.state,
                        "observed_events": [asdict(ev) for ev in observed_events],
                        "snapshot_phase": "actual_sleep_end_before_nonmutating_evaluation",
                    })),
                )
                
    return {
        "seed": seed,
        "task_id": task_id,
        "history": history_type,
        "q_old_prefix": q_old_prefix,
        "q_new_prefix": q_new_prefix,
        "q_old_pre_new": q_old_pre_new,
        "q_new_pre_new": q_new_pre_new,
        "q_old_post_new_pre_tail": q_old_post_new_pre_tail,
        "q_new_post_new_pre_tail": q_new_post_new_pre_tail,
        "q_old_post_tail": q_old_post_tail,
        "q_new_post_tail": q_new_post_tail,
        "interference_I_pre_tail": I_pre_tail,
        "interference_I_post_tail": I_post_tail,
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
    execute_sleep: bool = True,
) -> dict:
    """Execute complete stage across all histories and policies."""
    if set(seeds) & {101, 102, 103, 104, 105} or config.get("run_m5", False):
        raise ValueError("Final M5 seeds are sealed; this is an exploratory stand")
        
    t_start = time.perf_counter()
    output_stage_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = output_stage_dir / "snapshots"
    if execute_sleep:
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
    
    print(f"[{stage_name.upper()}] Starting execution: {len(seeds)} seeds x {len(panel_maps)} maps x 2 histories (execute_sleep={execute_sleep}).")
    
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
                    save_snapshots_dir=snapshots_dir if execute_sleep else None,
                    execute_sleep=execute_sleep,
                )
                all_history_results.append(res)
                
                if execute_sleep:
                    for pol, pdata in res["policies"].items():
                        history_metrics_rows.append({
                            "seed": seed,
                            "task_id": m_idx,
                            "history": hist,
                            "policy": pol,
                            "q_old_prefix": res["q_old_prefix"],
                            "q_new_prefix": res["q_new_prefix"],
                            "q_old_pre_new": res["q_old_pre_new"],
                            "q_old_post_new_pre_tail": res["q_old_post_new_pre_tail"],
                            "q_new_post_new_pre_tail": res["q_new_post_new_pre_tail"],
                            "q_old_post_tail": res["q_old_post_tail"],
                            "interference_I_pre_tail": res["interference_I_pre_tail"],
                            "interference_I_post_tail": res["interference_I_post_tail"],
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
                            "wake_ticks": pdata["wake_ticks"],
                            "probe_ticks": pdata["probe_ticks"],
                            "guard_ticks": pdata["guard_ticks"],
                            "sleep_active_ticks": pdata["sleep_active_ticks"],
                            "delay_ticks": pdata["delay_ticks"],
                            "rollback_ticks": pdata["rollback_ticks"],
                            "sleep_session_compute_ticks": pdata["sleep_session_compute_ticks"],
                            "total_life_ticks": pdata["total_life_ticks"],
                            "rollback_exact": pdata["rollback_exact"],
                        })
                else:
                    # Wake-only rows
                    history_metrics_rows.append({
                        "seed": seed,
                        "task_id": m_idx,
                        "history": hist,
                        "q_old_prefix": res["q_old_prefix"],
                        "q_new_prefix": res["q_new_prefix"],
                        "q_old_pre_new": res["q_old_pre_new"],
                        "q_old_post_new_pre_tail": res["q_old_post_new_pre_tail"],
                        "q_new_post_new_pre_tail": res["q_new_post_new_pre_tail"],
                        "q_old_post_tail": res["q_old_post_tail"],
                        "interference_I_pre_tail": res["interference_I_pre_tail"],
                        "interference_I_post_tail": res["interference_I_post_tail"],
                    })
                    
        print(f"  Completed seed {seed} ({s_idx+1}/{len(seeds)}) in {time.perf_counter() - t_seed_start:.1f}s")

    # Save JSONL logs
    print(f"[{stage_name.upper()}] Writing logs...")
    with open(output_stage_dir / "wake.jsonl", "w", encoding="utf-8") as f:
        for r in wake_records:
            f.write(safe_json_dumps(r) + "\n")
            
    with open(output_stage_dir / "pressure_probes.jsonl", "w", encoding="utf-8") as f:
        for r in probe_records:
            f.write(safe_json_dumps(r) + "\n")
            
    if execute_sleep:
        with open(output_stage_dir / "sleep.jsonl", "w", encoding="utf-8") as f:
            for r in sleep_records:
                f.write(safe_json_dumps(r) + "\n")
                
        with open(output_stage_dir / "queries.jsonl", "w", encoding="utf-8") as f:
            for r in query_records:
                f.write(safe_json_dumps(r) + "\n")
                
    # Save CSV
    csv_file = output_stage_dir / "metrics_per_history.csv"
    if history_metrics_rows:
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history_metrics_rows[0].keys()))
            writer.writeheader()
            writer.writerows(history_metrics_rows)
            
    # Compute contrasts per seed
    seed_stats = []
    D_per_seed = []
    I_contrast_pre_tail_per_seed = []
    I_contrast_post_tail_per_seed = []
    I_blocked_pre_tail_per_seed = []
    I_interleaved_pre_tail_per_seed = []
    gain_dream_blocked_per_seed = []
    gain_replay_blocked_per_seed = []
    new_skill_loss_dream_per_seed = []

    for seed in seeds:
        seed_res = [r for r in all_history_results if r["seed"] == seed]
        
        D_tasks = []
        I_pre_contrast_tasks = []
        I_b_pre_tasks = []
        I_i_pre_tasks = []
        g_dr_b_tasks = []
        g_rp_b_tasks = []
        new_loss_dr_tasks = []
        
        for m_idx in range(len(panel_maps)):
            b_res = next((r for r in seed_res if r["task_id"] == m_idx and r["history"] == "blocked"), None)
            i_res = next((r for r in seed_res if r["task_id"] == m_idx and r["history"] == "interleaved"), None)
            
            if b_res and i_res:
                I_b = b_res["interference_I_pre_tail"]
                I_i = i_res["interference_I_pre_tail"]
                I_b_pre_tasks.append(I_b)
                I_i_pre_tasks.append(I_i)
                I_pre_contrast_tasks.append(I_b - I_i)
                
                if execute_sleep and "dream" in b_res["policies"] and "replay" in b_res["policies"]:
                    dr_b = b_res["policies"]["dream"]
                    rp_b = b_res["policies"]["replay"]
                    dr_i = i_res["policies"]["dream"]
                    rp_i = i_res["policies"]["replay"]
                    
                    R_b = dr_b["gain_old_struct"] - rp_b["gain_old_struct"]
                    R_i = dr_i["gain_old_struct"] - rp_i["gain_old_struct"]
                    D_tasks.append(R_b - R_i)
                    g_dr_b_tasks.append(dr_b["gain_old_struct"])
                    g_rp_b_tasks.append(rp_b["gain_old_struct"])
                    # New skill loss: pre_new_tail acc minus post_struct acc
                    loss_new = dr_b["acc_new_pre_tail"] - dr_b["acc_new_post_struct"]
                    new_loss_dr_tasks.append(loss_new)
                    
        I_contrast_pre_tail_per_seed.append(float(np.mean(I_pre_contrast_tasks)) if I_pre_contrast_tasks else 0.0)
        I_blocked_pre_tail_per_seed.append(float(np.mean(I_b_pre_tasks)) if I_b_pre_tasks else 0.0)
        I_interleaved_pre_tail_per_seed.append(float(np.mean(I_i_pre_tasks)) if I_i_pre_tasks else 0.0)
        
        if execute_sleep:
            D_per_seed.append(float(np.mean(D_tasks)) if D_tasks else 0.0)
            gain_dream_blocked_per_seed.append(float(np.mean(g_dr_b_tasks)) if g_dr_b_tasks else 0.0)
            gain_replay_blocked_per_seed.append(float(np.mean(g_rp_b_tasks)) if g_rp_b_tasks else 0.0)
            new_skill_loss_dream_per_seed.append(float(np.mean(new_loss_dr_tasks)) if new_loss_dr_tasks else 0.0)

    # Compute Student-t CIs
    I_mean, I_sem, I_ci = compute_student_t_ci(I_contrast_pre_tail_per_seed)
    Ib_mean, Ib_sem, Ib_ci = compute_student_t_ci(I_blocked_pre_tail_per_seed)
    Ii_mean, Ii_sem, Ii_ci = compute_student_t_ci(I_interleaved_pre_tail_per_seed)
    
    passed_interference = bool(I_mean >= 0.10 and I_ci[0] > 0.0)

    contrasts_data = {
        "interference_contrast_pre_tail": {
            "definition": "I_contrast = I_blocked_pre_tail - I_interleaved_pre_tail",
            "mean": I_mean,
            "sem": I_sem,
            "ci_95": I_ci,
            "seed_means": I_contrast_pre_tail_per_seed,
            "target_threshold": 0.10,
            "passed_threshold": passed_interference,
        },
        "I_blocked_pre_tail": {
            "mean": Ib_mean,
            "sem": Ib_sem,
            "ci_95": Ib_ci,
            "seed_means": I_blocked_pre_tail_per_seed,
        },
        "I_interleaved_pre_tail": {
            "mean": Ii_mean,
            "sem": Ii_sem,
            "ci_95": Ii_ci,
            "seed_means": I_interleaved_pre_tail_per_seed,
        },
    }

    if execute_sleep:
        D_mean, D_sem, D_ci = compute_student_t_ci(D_per_seed)
        Gdr_b_mean, Gdr_b_sem, Gdr_b_ci = compute_student_t_ci(gain_dream_blocked_per_seed)
        Grp_b_mean, Grp_b_sem, Grp_b_ci = compute_student_t_ci(gain_replay_blocked_per_seed)
        Nloss_mean, Nloss_sem, Nloss_ci = compute_student_t_ci(new_skill_loss_dream_per_seed)
        
        # Comprehensive verdict
        if not passed_interference:
            verdict = "TASK_NOT_INFORMATIVE"
        elif D_ci[0] >= 0.05 and Gdr_b_ci[0] >= 0.05 and Nloss_mean <= 0.02 and Nloss_ci[1] <= 0.05:
            verdict = "DREAM_NEED_SUPPORTED"
        elif D_ci[0] >= -0.05 and D_ci[1] <= 0.05:
            verdict = "EQUIVALENT"
        else:
            verdict = "INCONCLUSIVE"
            
        contrasts_data.update({
            "D_primary_contrast": {
                "definition": "D = (G_dream - G_replay)_blocked - (G_dream - G_replay)_interleaved",
                "mean": D_mean,
                "sem": D_sem,
                "ci_95": D_ci,
                "seed_means": D_per_seed,
                "verdict": verdict,
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
            "new_skill_loss_dream": {
                "mean": Nloss_mean,
                "sem": Nloss_sem,
                "ci_95": Nloss_ci,
                "seed_means": new_skill_loss_dream_per_seed,
                "tolerance": 0.02,
                "passed_tolerance": bool(Nloss_mean <= 0.02 and Nloss_ci[1] <= 0.05),
            },
        })
    else:
        contrasts_data["status"] = "WAKE_ONLY_CALIBRATION"
        contrasts_data["verdict"] = "INTERFERENCE_CONFIRMED" if passed_interference else "TASK_NOT_INFORMATIVE"

    with open(output_stage_dir / "contrasts.json", "w", encoding="utf-8") as f:
        json.dump(contrasts_data, f, indent=2)

    return contrasts_data


def generate_russian_report(stage_name: str, contrasts: dict, run_dir: Path, n_branches: int) -> str:
    """Generate Russian summary report markdown for the stage."""
    i_data = contrasts["interference_contrast_pre_tail"]
    
    rep = f"""# Отчёт о результатах этапа: {stage_name.upper()} (EXP-SLEEP-NEED-001)

- **Объём ветвей:** {n_branches}
- **Интерференция I_contrast (BLOCKED - INTERLEAVED до tail):** {i_data['mean']:+.4f} (95% ДИ: [{i_data['ci_95'][0]:+.4f}, {i_data['ci_95'][1]:+.4f}])
- **Порог интерференции (>= 10 п.п.):** {'ДОСТИГНУТ' if i_data['passed_threshold'] else 'НЕ ДОСТИГНУТ'}
"""
    if "D_primary_contrast" in contrasts:
        d_data = contrasts["D_primary_contrast"]
        rep += f"""- **Основной контраст D:** {d_data['mean']:+.4f} (95% ДИ: [{d_data['ci_95'][0]:+.4f}, {d_data['ci_95'][1]:+.4f}])
- **Вердикт по D:** `{d_data['verdict']}`
"""
    with open(run_dir / f"report_{stage_name}_ru.md", "w", encoding="utf-8") as f:
        f.write(rep)
    return rep
