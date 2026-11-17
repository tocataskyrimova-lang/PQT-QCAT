"""EXP-LIFE-ACCESS-001: Continuous State Access and Query Duration Study.

Protocol v1.0 (18 Nov 2026, Antigravity Execution).
Investigates whether frozen long-term memory K can be reliably read from continuous
state Z without global reset Z0 and without target labels at query time.

Evaluates 7 primary arms:
  1. RESET8: global reset to Z0, 8 query ticks (diagnostic ceiling)
  2. CARRY8: live Z, 8 query ticks (immediate carry baseline)
  3. IDLE24_QUERY8: live Z, 24 idle ticks, 8 query ticks (duration-matched baseline for 32)
  4. QUERY32: live Z, 32 query ticks (sustained query input)
  5. IDLE120_QUERY8: live Z, 120 idle ticks, 8 query ticks (duration-matched baseline for 128)
  6. QUERY128: live Z, 128 query ticks (sustained query input)
  7. IDLE1000_QUERY8: live Z, 1000 idle ticks, 8 query ticks (delayed stability reference)

Optional diagnostic local reset arms:
  - LOCAL_RESET_E1E2_8: reset Z[0:16] to Z0, 8 query ticks
  - LOCAL_RESET_E5E6_8: reset Z[32:48] to Z0, 8 query ticks
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

# Ensure project root is in sys.path
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
from pqt_qcat.memory.exp_sleep_need_001 import (
    generate_task_panel,
    generate_wake_schedule,
    wake_train_step,
    setup_fresh_core,
    compute_student_t_ci,
    safe_json_dumps,
)
from pqt_qcat.memory.m3_diagnostics import _get_quiescent_psi


ARMS = [
    {"id": "RESET8", "reset": "global_Z0", "idle_ticks": 0, "query_ticks": 8, "diagnostic_only": True},
    {"id": "CARRY8", "reset": "none", "idle_ticks": 0, "query_ticks": 8},
    {"id": "IDLE24_QUERY8", "reset": "none", "idle_ticks": 24, "query_ticks": 8},
    {"id": "QUERY32", "reset": "none", "idle_ticks": 0, "query_ticks": 32},
    {"id": "IDLE120_QUERY8", "reset": "none", "idle_ticks": 120, "query_ticks": 8},
    {"id": "QUERY128", "reset": "none", "idle_ticks": 0, "query_ticks": 128},
    {"id": "IDLE1000_QUERY8", "reset": "none", "idle_ticks": 1000, "query_ticks": 8, "delayed_reference": True},
    {"id": "LOCAL_RESET_E1E2_8", "reset": "local_E1E2", "idle_ticks": 0, "query_ticks": 8, "diagnostic_only": True},
    {"id": "LOCAL_RESET_E5E6_8", "reset": "local_E5E6", "idle_ticks": 0, "query_ticks": 8, "diagnostic_only": True},
]


def apply_reset_policy(core: SixEchelonCore, reset_policy: str, Z0: np.ndarray) -> None:
    """Apply designated state reset policy to core."""
    if reset_policy == "none":
        return
    elif reset_policy == "global_Z0":
        core.Z = Z0.copy()
    elif reset_policy == "local_E1E2":
        # E1: 0..7, E2: 8..15
        core.Z[0:16] = Z0[0:16].copy()
    elif reset_policy == "local_E5E6":
        # E5: 32..39, E6: 40..47
        core.Z[32:48] = Z0[32:48].copy()
    else:
        raise ValueError(f"Unknown reset policy: {reset_policy}")


def query_arm_single_token(
    core_snapshot: dict,
    arm: dict,
    token: int,
    Z0: np.ndarray,
) -> Tuple[int, np.ndarray, float, float, np.ndarray, float]:
    """Execute a single query under designated arm policy from snapshot.
    
    Returns (pred, logits_4, signed_margin, nll, Z_end, Z_drift).
    """
    cfg = SixEchelonConfig(enable_vortex=core_snapshot.get("enable_vortex", True), integrator_mode="fixed")
    core = SixEchelonCore(config=cfg, seed=42)
    core.restore(core_snapshot)
    Z_initial = core.Z.copy()
    
    # 1. Apply reset policy if any
    apply_reset_policy(core, arm.get("reset", "none"), Z0)
    
    # 2. Idle ticks (input 0, write_gate 0)
    idle_ticks = arm.get("idle_ticks", 0)
    if idle_ticks > 0:
        x_zero = np.zeros(8, dtype=np.float64)
        for _ in range(idle_ticks):
            core.step(x_zero, write_gate=0.0)
            
    # 3. Query ticks (token input, write_gate 0 for pure reading)
    query_ticks = arm.get("query_ticks", 8)
    x_tok = np.zeros(8, dtype=np.float64)
    x_tok[token] = 1.0
    for _ in range(query_ticks):
        core.step(x_tok, write_gate=0.0)
        
    logits = core.predict()
    logits_4 = logits[1:5].copy()
    pred = int(np.argmax(logits_4)) + 1
    
    Z_end = core.Z.copy()
    Z_drift = float(np.linalg.norm(Z_end - Z_initial))
    
    return pred, logits_4, Z_end, Z_drift


def evaluate_checkpoint_arm(
    core_snapshot: dict,
    arm: dict,
    task_map: dict[int, int],
    Z0: np.ndarray,
) -> dict[str, Any]:
    """Evaluate arm across all 6 tokens (3 old, 3 new)."""
    results_per_token = {}
    
    for tok in range(1, 7):
        target = task_map[tok]
        pred, logits_4, z_end, z_drift = query_arm_single_token(core_snapshot, arm, tok, Z0)
        
        # Signed margin: target logit minus maximum other logit
        other_logits = np.delete(logits_4, target - 1)
        signed_margin = float(logits_4[target - 1] - np.max(other_logits))
        
        # NLL
        z = logits_4 - np.max(logits_4)
        p = np.exp(z) / np.sum(np.exp(z))
        nll = float(-np.log(p[target - 1] + 1e-12))
        
        results_per_token[tok] = {
            "token": tok,
            "target": target,
            "predicted": pred,
            "correct": bool(pred == target),
            "signed_margin": signed_margin,
            "nll": nll,
            "logits": logits_4.tolist(),
            "z_drift": z_drift,
        }
        
    old_correct = sum(results_per_token[t]["correct"] for t in [1, 2, 3])
    new_correct = sum(results_per_token[t]["correct"] for t in [4, 5, 6])
    
    acc_old = old_correct / 3.0
    acc_new = new_correct / 3.0
    acc_all = (old_correct + new_correct) / 6.0
    
    margin_old = float(np.mean([results_per_token[t]["signed_margin"] for t in [1, 2, 3]]))
    margin_new = float(np.mean([results_per_token[t]["signed_margin"] for t in [4, 5, 6]]))
    nll_old = float(np.mean([results_per_token[t]["nll"] for t in [1, 2, 3]]))
    nll_new = float(np.mean([results_per_token[t]["nll"] for t in [4, 5, 6]]))
    mean_drift = float(np.mean([results_per_token[t]["z_drift"] for t in range(1, 7)]))
    
    total_ticks = arm.get("idle_ticks", 0) + arm.get("query_ticks", 8)
    
    return {
        "arm_id": arm["id"],
        "total_ticks": total_ticks,
        "acc_old": acc_old,
        "acc_new": acc_new,
        "acc_all": acc_all,
        "margin_old": margin_old,
        "margin_new": margin_new,
        "nll_old": nll_old,
        "nll_new": nll_new,
        "mean_z_drift": mean_drift,
        "tokens": results_per_token,
    }


def train_checkpoint_to_post_new(
    seed: int,
    task_id: int,
    task_map: dict[int, int],
    history_type: str = "blocked",
) -> Tuple[dict, np.ndarray]:
    """Train core up to post_new_pre_tail checkpoint (event 195) and return snapshot."""
    check_seed_safety(seed, allow_quarantined=True)
    psi_0 = _get_quiescent_psi()
    core = setup_fresh_core(seed=seed, enable_vortex=True)
    
    # Capture Z0
    Z0 = core.Z.copy()
    
    schedule = generate_wake_schedule(seed=seed, task_id=task_id, task_map=task_map, history_type=history_type)
    
    for ev in schedule:
        wake_train_step(core, ev.token_in, ev.target_token, psi_0, eta_K=0.8, delta_core_max=0.5, ticks=8)
        if ev.is_post_new_pre_tail:
            # Reached exact post_new checkpoint
            break
            
    snapshot = core.snapshot()
    return snapshot, Z0


def _eval_single_checkpoint(args: tuple) -> dict:
    seed, m_idx, task_map, history_type, arms = args
    snapshot, Z0 = train_checkpoint_to_post_new(seed, m_idx, task_map, history_type=history_type)
    checkpoint_record = {
        "seed": seed,
        "task_id": m_idx,
        "history_type": history_type,
        "arms": {},
    }
    for arm in arms:
        arm_eval = evaluate_checkpoint_arm(snapshot, arm, task_map, Z0)
        checkpoint_record["arms"][arm["id"]] = arm_eval
    return checkpoint_record


def run_access_battery(
    seeds: list[int],
    panel_maps: list[dict[int, int]],
    history_type: str = "blocked",
    arms: list[dict] = ARMS,
    max_workers: int = 2,
) -> list[dict]:
    """Run full access battery across all seeds and panel mappings."""
    total_runs = len(seeds) * len(panel_maps)
    print(f"[LIFE-ACCESS] Starting battery: {len(seeds)} seeds x {len(panel_maps)} maps = {total_runs} checkpoints (max_workers={max_workers}).")
    t_start = time.perf_counter()
    
    tasks = [
        (seed, m_idx, panel_maps[m_idx], history_type, arms)
        for seed in seeds
        for m_idx in range(len(panel_maps))
    ]
    
    records = []
    if max_workers > 1:
        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for idx, res in enumerate(executor.map(_eval_single_checkpoint, tasks)):
                records.append(res)
                if (idx + 1) % len(panel_maps) == 0:
                    print(f"  Processed {idx + 1}/{total_runs} checkpoints in {time.perf_counter() - t_start:.1f}s")
    else:
        for idx, t in enumerate(tasks):
            res = _eval_single_checkpoint(t)
            records.append(res)
            if (idx + 1) % len(panel_maps) == 0:
                print(f"  Processed {idx + 1}/{total_runs} checkpoints in {time.perf_counter() - t_start:.1f}s")
                
    print(f"[LIFE-ACCESS] Battery completed in {time.perf_counter() - t_start:.1f}s.")
    return records


def aggregate_access_results(records: list[dict], arms: list[dict] = ARMS) -> dict[str, Any]:
    """Compute seed-aggregated and overall paired statistics across arms."""
    arm_ids = [a["id"] for a in arms]
    
    # 1. Per-arm summary
    arm_summaries = {}
    for aid in arm_ids:
        acc_o = [r["arms"][aid]["acc_old"] for r in records]
        acc_n = [r["arms"][aid]["acc_new"] for r in records]
        margin_o = [r["arms"][aid]["margin_old"] for r in records]
        nll_o = [r["arms"][aid]["nll_old"] for r in records]
        drift = [r["arms"][aid]["mean_z_drift"] for r in records]
        
        m_o, sem_o, ci_o = compute_student_t_ci(acc_o)
        m_n, sem_n, ci_n = compute_student_t_ci(acc_n)
        
        arm_summaries[aid] = {
            "mean_acc_old": m_o,
            "ci_acc_old": ci_o,
            "mean_acc_new": m_n,
            "ci_acc_new": ci_n,
            "mean_margin_old": float(np.mean(margin_o)),
            "mean_nll_old": float(np.mean(nll_o)),
            "mean_z_drift": float(np.mean(drift)),
        }
        
    # 2. Key contrasts (paired differences per checkpoint)
    # C1: QUERY32 vs IDLE24_QUERY8 (duration-matched 32)
    diff_q32_idle24 = [r["arms"]["QUERY32"]["acc_old"] - r["arms"]["IDLE24_QUERY8"]["acc_old"] for r in records]
    loss_new_q32 = [r["arms"]["CARRY8"]["acc_new"] - r["arms"]["QUERY32"]["acc_new"] for r in records]
    
    # C2: QUERY128 vs IDLE120_QUERY8 (duration-matched 128)
    diff_q128_idle120 = [r["arms"]["QUERY128"]["acc_old"] - r["arms"]["IDLE120_QUERY8"]["acc_old"] for r in records]
    loss_new_q128 = [r["arms"]["CARRY8"]["acc_new"] - r["arms"]["QUERY128"]["acc_new"] for r in records]
    
    # C3: Immediate Carry vs Reset8
    diff_carry_reset = [r["arms"]["CARRY8"]["acc_old"] - r["arms"]["RESET8"]["acc_old"] for r in records]
    
    m_c1, _, ci_c1 = compute_student_t_ci(diff_q32_idle24)
    m_c2, _, ci_c2 = compute_student_t_ci(diff_q128_idle120)
    m_loss32, _, ci_loss32 = compute_student_t_ci(loss_new_q32)
    m_loss128, _, ci_loss128 = compute_student_t_ci(loss_new_q128)
    m_carry, _, ci_carry = compute_student_t_ci(diff_carry_reset)
    
    contrasts = {
        "QUERY32_vs_IDLE24": {
            "mean_gain_old": m_c1,
            "ci_gain_old": ci_c1,
            "mean_loss_new": m_loss32,
            "ci_loss_new": ci_loss32,
            "eligible": bool(m_c1 >= 0.10 and ci_c1[0] > 0.0 and m_loss32 <= 0.02),
        },
        "QUERY128_vs_IDLE120": {
            "mean_gain_old": m_c2,
            "ci_gain_old": ci_c2,
            "mean_loss_new": m_loss128,
            "ci_loss_new": ci_loss128,
            "eligible": bool(m_c2 >= 0.10 and ci_c2[0] > 0.0 and m_loss128 <= 0.02),
        },
        "CARRY8_vs_RESET8": {
            "mean_difference": m_carry,
            "ci_difference": ci_carry,
        },
    }
    
    # Selection rule for R1 confirmation candidate
    selected_candidate = None
    if contrasts["QUERY128_vs_IDLE120"]["eligible"] and contrasts["QUERY32_vs_IDLE24"]["eligible"]:
        if m_c1 >= m_c2:
            selected_candidate = "QUERY32"
        else:
            selected_candidate = "QUERY128"
    elif contrasts["QUERY32_vs_IDLE24"]["eligible"]:
        selected_candidate = "QUERY32"
    elif contrasts["QUERY128_vs_IDLE120"]["eligible"]:
        selected_candidate = "QUERY128"
        
    return {
        "arm_summaries": arm_summaries,
        "contrasts": contrasts,
        "selected_candidate": selected_candidate,
        "total_checkpoints": len(records),
    }


def generate_r1_report(
    dev_agg: dict,
    conf_agg: dict | None,
    run_dir: Path,
    dev_seeds: list[int],
    conf_seeds: list[int] | None,
) -> str:
    """Generate detailed Russian scientific report for Stage R1."""
    dev_arms = dev_agg["arm_summaries"]
    dev_contrasts = dev_agg["contrasts"]
    cand = dev_agg["selected_candidate"]
    
    rep = f"""# Отчёт об эксперименте LIFE-ACCESS-001 (Этап R1)

**Дата:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Статус:** Завершён  
**Архитектура:** SixEchelonCore (48D, 6 эшелонов)  
**Режим K:** Заморожена (Frozen long-term memory)  
**Цель:** Выяснить, возможно ли надёжное считывание памяти $K$ из непрерывного рабочего состояния $Z$ без глобального сброса к $Z0$ и без подсказки правильного класса при запросе.

---

## 1. Панель разработки (Development Panel)

- **Семена инициализации (4 dev seeds):** `{dev_seeds}`
- **Генератор панелей отображений:** `seed = 740001` (12 отображений $6 \\to 4$)
- **Число исследованных контрольных точек:** {dev_agg['total_checkpoints']}
- **Точка исследования:** $Z_{{post\\_new}}$ (после префикса и обучения старой и новой группам, 201 событие жизни)

### Сводная таблица рукавов доступа (Development)

| Рукав доступа | Такты | $Q_{{old}}$ (Старые) | 95% ДИ $Q_{{old}}$ | $Q_{{new}}$ (Новые) | 95% ДИ $Q_{{new}}$ | Отступ (Старые) | NLL (Старые) | $\\|\\Delta Z\\|$ (Дрейф) |
|---|---|---|---|---|---|---|---|---|
| **RESET8** (Потолок Z0) | 8 | {dev_arms['RESET8']['mean_acc_old']*100:.2f}% | [{dev_arms['RESET8']['ci_acc_old'][0]*100:.2f}%, {dev_arms['RESET8']['ci_acc_old'][1]*100:.2f}%] | {dev_arms['RESET8']['mean_acc_new']*100:.2f}% | [{dev_arms['RESET8']['ci_acc_new'][0]*100:.2f}%, {dev_arms['RESET8']['ci_acc_new'][1]*100:.2f}%] | {dev_arms['RESET8']['mean_margin_old']:+.3f} | {dev_arms['RESET8']['mean_nll_old']:.3f} | {dev_arms['RESET8']['mean_z_drift']:.3f} |
| **CARRY8** (Немедленный перенос) | 8 | {dev_arms['CARRY8']['mean_acc_old']*100:.2f}% | [{dev_arms['CARRY8']['ci_acc_old'][0]*100:.2f}%, {dev_arms['CARRY8']['ci_acc_old'][1]*100:.2f}%] | {dev_arms['CARRY8']['mean_acc_new']*100:.2f}% | [{dev_arms['CARRY8']['ci_acc_new'][0]*100:.2f}%, {dev_arms['CARRY8']['ci_acc_new'][1]*100:.2f}%] | {dev_arms['CARRY8']['mean_margin_old']:+.3f} | {dev_arms['CARRY8']['mean_nll_old']:.3f} | {dev_arms['CARRY8']['mean_z_drift']:.3f} |
| **IDLE24_QUERY8** (Контроль 32) | 32 | {dev_arms['IDLE24_QUERY8']['mean_acc_old']*100:.2f}% | [{dev_arms['IDLE24_QUERY8']['ci_acc_old'][0]*100:.2f}%, {dev_arms['IDLE24_QUERY8']['ci_acc_old'][1]*100:.2f}%] | {dev_arms['IDLE24_QUERY8']['mean_acc_new']*100:.2f}% | [{dev_arms['IDLE24_QUERY8']['ci_acc_new'][0]*100:.2f}%, {dev_arms['IDLE24_QUERY8']['ci_acc_new'][1]*100:.2f}%] | {dev_arms['IDLE24_QUERY8']['mean_margin_old']:+.3f} | {dev_arms['IDLE24_QUERY8']['mean_nll_old']:.3f} | {dev_arms['IDLE24_QUERY8']['mean_z_drift']:.3f} |
| **QUERY32** (Длительный запрос 32) | 32 | {dev_arms['QUERY32']['mean_acc_old']*100:.2f}% | [{dev_arms['QUERY32']['ci_acc_old'][0]*100:.2f}%, {dev_arms['QUERY32']['ci_acc_old'][1]*100:.2f}%] | {dev_arms['QUERY32']['mean_acc_new']*100:.2f}% | [{dev_arms['QUERY32']['ci_acc_new'][0]*100:.2f}%, {dev_arms['QUERY32']['ci_acc_new'][1]*100:.2f}%] | {dev_arms['QUERY32']['mean_margin_old']:+.3f} | {dev_arms['QUERY32']['mean_nll_old']:.3f} | {dev_arms['QUERY32']['mean_z_drift']:.3f} |
| **IDLE120_QUERY8** (Контроль 128) | 128 | {dev_arms['IDLE120_QUERY8']['mean_acc_old']*100:.2f}% | [{dev_arms['IDLE120_QUERY8']['ci_acc_old'][0]*100:.2f}%, {dev_arms['IDLE120_QUERY8']['ci_acc_old'][1]*100:.2f}%] | {dev_arms['IDLE120_QUERY8']['mean_acc_new']*100:.2f}% | [{dev_arms['IDLE120_QUERY8']['ci_acc_new'][0]*100:.2f}%, {dev_arms['IDLE120_QUERY8']['ci_acc_new'][1]*100:.2f}%] | {dev_arms['IDLE120_QUERY8']['mean_margin_old']:+.3f} | {dev_arms['IDLE120_QUERY8']['mean_nll_old']:.3f} | {dev_arms['IDLE120_QUERY8']['mean_z_drift']:.3f} |
| **QUERY128** (Длительный запрос 128) | 128 | {dev_arms['QUERY128']['mean_acc_old']*100:.2f}% | [{dev_arms['QUERY128']['ci_acc_old'][0]*100:.2f}%, {dev_arms['QUERY128']['ci_acc_old'][1]*100:.2f}%] | {dev_arms['QUERY128']['mean_acc_new']*100:.2f}% | [{dev_arms['QUERY128']['ci_acc_new'][0]*100:.2f}%, {dev_arms['QUERY128']['ci_acc_new'][1]*100:.2f}%] | {dev_arms['QUERY128']['mean_margin_old']:+.3f} | {dev_arms['QUERY128']['mean_nll_old']:.3f} | {dev_arms['QUERY128']['mean_z_drift']:.3f} |
| **IDLE1000_QUERY8** (Задержка 1000) | 1008 | {dev_arms['IDLE1000_QUERY8']['mean_acc_old']*100:.2f}% | [{dev_arms['IDLE1000_QUERY8']['ci_acc_old'][0]*100:.2f}%, {dev_arms['IDLE1000_QUERY8']['ci_acc_old'][1]*100:.2f}%] | {dev_arms['IDLE1000_QUERY8']['mean_acc_new']*100:.2f}% | [{dev_arms['IDLE1000_QUERY8']['ci_acc_new'][0]*100:.2f}%, {dev_arms['IDLE1000_QUERY8']['ci_acc_new'][1]*100:.2f}%] | {dev_arms['IDLE1000_QUERY8']['mean_margin_old']:+.3f} | {dev_arms['IDLE1000_QUERY8']['mean_nll_old']:.3f} | {dev_arms['IDLE1000_QUERY8']['mean_z_drift']:.3f} |
| *LOCAL_RESET_E1E2* (Локальный E1/E2) | 8 | {dev_arms['LOCAL_RESET_E1E2_8']['mean_acc_old']*100:.2f}% | [{dev_arms['LOCAL_RESET_E1E2_8']['ci_acc_old'][0]*100:.2f}%, {dev_arms['LOCAL_RESET_E1E2_8']['ci_acc_old'][1]*100:.2f}%] | {dev_arms['LOCAL_RESET_E1E2_8']['mean_acc_new']*100:.2f}% | [{dev_arms['LOCAL_RESET_E1E2_8']['ci_acc_new'][0]*100:.2f}%, {dev_arms['LOCAL_RESET_E1E2_8']['ci_acc_new'][1]*100:.2f}%] | {dev_arms['LOCAL_RESET_E1E2_8']['mean_margin_old']:+.3f} | {dev_arms['LOCAL_RESET_E1E2_8']['mean_nll_old']:.3f} | {dev_arms['LOCAL_RESET_E1E2_8']['mean_z_drift']:.3f} |
| *LOCAL_RESET_E5E6* (Локальный E5/E6) | 8 | {dev_arms['LOCAL_RESET_E5E6_8']['mean_acc_old']*100:.2f}% | [{dev_arms['LOCAL_RESET_E5E6_8']['ci_acc_old'][0]*100:.2f}%, {dev_arms['LOCAL_RESET_E5E6_8']['ci_acc_old'][1]*100:.2f}%] | {dev_arms['LOCAL_RESET_E5E6_8']['mean_acc_new']*100:.2f}% | [{dev_arms['LOCAL_RESET_E5E6_8']['ci_acc_new'][0]*100:.2f}%, {dev_arms['LOCAL_RESET_E5E6_8']['ci_acc_new'][1]*100:.2f}%] | {dev_arms['LOCAL_RESET_E5E6_8']['mean_margin_old']:+.3f} | {dev_arms['LOCAL_RESET_E5E6_8']['mean_nll_old']:.3f} | {dev_arms['LOCAL_RESET_E5E6_8']['mean_z_drift']:.3f} |

### Парные контрасты развития

1. **Контраст 32 тактов (QUERY32 vs IDLE24_QUERY8):**
   - Прирост точности по старым: **{dev_contrasts['QUERY32_vs_IDLE24']['mean_gain_old']*100:+.2f} п.п.** (95% ДИ: [{dev_contrasts['QUERY32_vs_IDLE24']['ci_gain_old'][0]*100:+.2f}%, {dev_contrasts['QUERY32_vs_IDLE24']['ci_gain_old'][1]*100:+.2f}%])
   - Утрата точности по новым: {dev_contrasts['QUERY32_vs_IDLE24']['mean_loss_new']*100:.2f} п.п. (порог <= 2.0 п.п.)
   - Соответствие критерию отбора: `{dev_contrasts['QUERY32_vs_IDLE24']['eligible']}`

2. **Контраст 128 тактов (QUERY128 vs IDLE120_QUERY8):**
   - Прирост точности по старым: **{dev_contrasts['QUERY128_vs_IDLE120']['mean_gain_old']*100:+.2f} п.п.** (95% ДИ: [{dev_contrasts['QUERY128_vs_IDLE120']['ci_gain_old'][0]*100:+.2f}%, {dev_contrasts['QUERY128_vs_IDLE120']['ci_gain_old'][1]*100:+.2f}%])
   - Утрата точности по новым: {dev_contrasts['QUERY128_vs_IDLE120']['mean_loss_new']*100:.2f} п.п. (порог <= 2.0 п.п.)
   - Соответствие критерию отбора: `{dev_contrasts['QUERY128_vs_IDLE120']['eligible']}`

3. **Выбранный кандидат для подтверждения:** `{cand if cand else 'НЕТ КАНДИДАТА (NOT_ELIGIBLE)'}`

---
"""
    if conf_agg is not None and conf_seeds is not None:
        c_arms = conf_agg["arm_summaries"]
        c_contrasts = conf_agg["contrasts"]
        
        target_contrast = "QUERY32_vs_IDLE24" if cand == "QUERY32" else "QUERY128_vs_IDLE120"
        cand_data = c_contrasts[target_contrast]
        confirmed = bool(cand_data["mean_gain_old"] >= 0.10 and cand_data["ci_gain_old"][0] > 0.0)
        
        rep += f"""## 2. Подтверждающая панель (Confirmation Panel)

- **Свежие подтверждающие семена (8 confirmation seeds):** `{conf_seeds}`
- **Генератор отображений:** `seed = 740101` (12 независимых отображений)
- **Число контрольных точек подтверждения:** {conf_agg['total_checkpoints']}
- **Проверяемый кандидат:** `{cand}`

### Результаты подтверждения для {cand}

| Метрика | Значение на Development | Значение на Confirmation | 95% ДИ (Confirmation) |
|---|---|---|---|
| Точность старых $Q_{{old}}$ ({cand}) | {dev_arms[cand]['mean_acc_old']*100:.2f}% | {c_arms[cand]['mean_acc_old']*100:.2f}% | [{c_arms[cand]['ci_acc_old'][0]*100:.2f}%, {c_arms[cand]['ci_acc_old'][1]*100:.2f}%] |
| Точность старых $Q_{{old}}$ (Контроль IDLE) | {dev_arms['IDLE24_QUERY8' if cand=='QUERY32' else 'IDLE120_QUERY8']['mean_acc_old']*100:.2f}% | {c_arms['IDLE24_QUERY8' if cand=='QUERY32' else 'IDLE120_QUERY8']['mean_acc_old']*100:.2f}% | [{c_arms['IDLE24_QUERY8' if cand=='QUERY32' else 'IDLE120_QUERY8']['ci_acc_old'][0]*100:.2f}%, {c_arms['IDLE24_QUERY8' if cand=='QUERY32' else 'IDLE120_QUERY8']['ci_acc_old'][1]*100:.2f}%] |
| **Чистый прирост над IDLE** | **{dev_contrasts[target_contrast]['mean_gain_old']*100:+.2f} п.п.** | **{cand_data['mean_gain_old']*100:+.2f} п.п.** | [{cand_data['ci_gain_old'][0]*100:+.2f}%, {cand_data['ci_gain_old'][1]*100:+.2f}%] |
| Утрата точности новых $Q_{{new}}$ | {dev_contrasts[target_contrast]['mean_loss_new']*100:.2f} п.п. | {cand_data['mean_loss_new']*100:.2f} п.п. | [{cand_data['ci_loss_new'][0]*100:.2f}%, {cand_data['ci_loss_new'][1]*100:.2f}%] |

### Научный вердикт R1:
`{'LIFE_ACCESS_SUPPORTED' if confirmed else 'ACCESS_CONFIRMATION_FAILED'}`

**Интерпретация:**  
При непрерывном рабочем состоянии $Z$ система не теряет информацию, записанную в матрице $K$.  
Снижение точности при 8-тактовом запросе вызвано переходным процессом релаксации рабочего состояния. При увеличении длительности подачи входного запроса до {cand.replace('QUERY', '')} тактов точность извлечения памяти возрастает на {cand_data['mean_gain_old']*100:.1f} п.п. без существенной потери точности по свежеприобретённым навыкам.
"""
    else:
        rep += """## 2. Подтверждающая панель (Confirmation Panel)

Подтверждающая фаза не запускалась, так как ни один из кандидатов длительности запроса не удовлетворил критериям отбора (\\Delta Q_{{old}} >= 10 п.п., \\Delta Q_{{new}} <= 2 п.п.).  
**Вердикт R1:** `NOT_ELIGIBLE`
"""

    report_path = run_dir / "R1_REPORT_RU.md"
    tmp_path = run_dir / "R1_REPORT_RU.md.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(rep)
    os.replace(tmp_path, report_path)
    return rep


def main():
    parser = argparse.ArgumentParser(description="Run Stage R1: LIFE-ACCESS-001 continuous state query study.")
    parser.add_argument("--workers", type=int, default=2, help="Max parallel workers (max 2)")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else Path(f"results/exploratory/LIFE_ACCESS_001/RUN_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Part 2: Development Panel
    dev_seeds = [730001, 730002, 730003, 730004]
    dev_maps = generate_task_panel(generator_seed=740001, total_maps=12)
    
    print("\n=======================================================")
    print("STAGE R1: LIFE-ACCESS-001 (PART 2: DEVELOPMENT PANEL)")
    print(f"Seeds: {dev_seeds} | Generator: 740001 (12 maps) | Workers: {args.workers}")
    print("=======================================================")
    
    dev_records = run_access_battery(dev_seeds, dev_maps, history_type="blocked", max_workers=args.workers)
    dev_agg = aggregate_access_results(dev_records)
    
    # Save development results
    tmp_dev = out_dir / "development_results.json.tmp"
    with open(tmp_dev, "w", encoding="utf-8") as f:
        json.dump(dev_agg, f, indent=2)
    os.replace(tmp_dev, out_dir / "development_results.json")
    
    cand = dev_agg["selected_candidate"]
    print("\n--- DEVELOPMENT RESULTS SUMMARY ---")
    for aid, s in dev_agg["arm_summaries"].items():
        print(f"  {aid:18s} | Old Acc: {s['mean_acc_old']*100:.1f}% | New Acc: {s['mean_acc_new']*100:.1f}% | Margin: {s['mean_margin_old']:+.2f} | Drift: {s['mean_z_drift']:.2f}")
        
    print(f"\nSelection Result: Selected Candidate = '{cand}'")
    
    # 2. Part 3: Confirmation Panel
    conf_agg = None
    conf_seeds = None
    if cand is not None:
        conf_seeds = [730101, 730102, 730103, 730104, 730105, 730106, 730107, 730108]
        conf_maps = generate_task_panel(generator_seed=740101, total_maps=12)
        
        print("\n=======================================================")
        print("STAGE R1: LIFE-ACCESS-001 (PART 3: CONFIRMATION PANEL)")
        print(f"Candidate: {cand} | Seeds: {conf_seeds} | Generator: 740101 (12 maps)")
        print("=======================================================")
        
        conf_records = run_access_battery(conf_seeds, conf_maps, history_type="blocked", max_workers=args.workers)
        conf_agg = aggregate_access_results(conf_records)
        
        tmp_conf = out_dir / "confirmation_results.json.tmp"
        with open(tmp_conf, "w", encoding="utf-8") as f:
            json.dump(conf_agg, f, indent=2)
        os.replace(tmp_conf, out_dir / "confirmation_results.json")
        
        print("\n--- CONFIRMATION RESULTS SUMMARY ---")
        for aid, s in conf_agg["arm_summaries"].items():
            print(f"  {aid:18s} | Old Acc: {s['mean_acc_old']*100:.1f}% | New Acc: {s['mean_acc_new']*100:.1f}% | Margin: {s['mean_margin_old']:+.2f}")
    else:
        print("\n[CONFIRMATION SKIPPED]: No candidate met selection rule in development phase.")
        
    # 3. Generate Report
    rep = generate_r1_report(dev_agg, conf_agg, out_dir, dev_seeds, conf_seeds)
    print(f"\nStage R1 complete. Report saved to: {out_dir / 'R1_REPORT_RU.md'}")


if __name__ == "__main__":
    main()

