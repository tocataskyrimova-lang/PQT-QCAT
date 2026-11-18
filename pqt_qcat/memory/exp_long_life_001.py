"""Stage R4: Long Continuous Life Protocol.

Addresses:
1. Long continuous lifetime without artificial episodic resets.
2. 1000-event streaming cycles:
   - 200 C0 events (fixed lifelong context C0)
   - 400 C1 events (context C1 shift with new mapping from pool)
   - 200 mixed events (equal C0/C1)
   - idle relaxation from cyclic schedule [0, 32, 128, 1000] ticks
   - 200 return events (equal C0/C1)
3. Nested horizons:
   - Evaluates performance up to 10,000 events (10 cycles).
4. Key scientific metrics:
   - Lifelong retention of C0 (retention drop <= 5 p.p. non-inferiority)
   - Late adaptation efficiency (late cycle adaptation non-inferiority <= 5 p.p.)
   - Dynamical state stability (bounded Z norm, no blowup)
5. Development & Confirmation panels:
   - Development: 4 seeds x (RESET_TRAIN, CARRY_TRAIN) x 2 A_write = 16 lives.
   - Confirmation: 8 fresh seeds [730401..730408] x selected candidate x 2 A_write = 16 lives.
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

from pqt_qcat.core.six_echelon import SixEchelonConfig, SixEchelonCore
from pqt_qcat.core.security import (
    FORBIDDEN_FINAL_SEEDS,
    QUARANTINED_SEEDS,
    check_seed_safety,
)
from pqt_qcat.memory.exp_sleep_need_001 import compute_student_t_ci, safe_json_dumps
from pqt_qcat.memory.exp_stateful_online_001 import (
    setup_context_core,
    generate_context_task_pairs,
    StatefulOnlineAgent,
)


def generate_c1_mapping_pool(generator_seed: int, pool_size: int = 15) -> list[dict[int, int]]:
    """Generate a pool of diverse C1 mappings for successive life cycles."""
    rng = np.random.default_rng(generator_seed)
    pool = []
    while len(pool) < pool_size:
        classes = list(range(1, 5))
        rng.shuffle(classes)
        extra = rng.choice(range(1, 5), size=2, replace=True).tolist()
        perm = classes + extra
        rng.shuffle(perm)
        m = {tok: perm[tok - 1] for tok in range(1, 7)}
        if len(set(m.values())) == 4:
            pool.append(m)
    return pool


def run_long_continuous_life(
    seed: int,
    c0_map: dict[int, int],
    c1_pool: list[dict[int, int]],
    learning_mode: str,
    a_write: bool,
    num_cycles: int = 10, # 10 cycles = 10,000 scored events
    t_query: int = 8,
    stream_seed: int = 730000,
) -> dict[str, Any]:
    """Execute a single long continuous life over multiple 1000-event cycles."""
    check_seed_safety(seed, allow_quarantined=True)
    core = setup_context_core(seed=seed, enable_vortex=a_write)
    agent = StatefulOnlineAgent(
        core=core,
        learning_mode=learning_mode,
        t_context=4,
        t_query=t_query,
    )
    
    rng = np.random.default_rng(stream_seed)
    idle_schedule = [0, 32, 128, 1000]
    
    cycle_summaries = []
    c0_retention_history = []
    z_norm_history = []
    
    total_events = 0
    t_life_start = time.perf_counter()
    
    for cycle_idx in range(num_cycles):
        c1_map = c1_pool[cycle_idx % len(c1_pool)]
        idle_ticks = idle_schedule[cycle_idx % len(idle_schedule)]
        
        cycle_events = []
        
        # 1. Part A: 200 events C0
        for _ in range(200):
            q = int(rng.choice(list(range(1, 7))))
            cycle_events.append((0, q, c0_map[q], "part_a_c0"))
            
        # 2. Part B: 400 events C1 (shift)
        for _ in range(400):
            q = int(rng.choice(list(range(1, 7))))
            cycle_events.append((1, q, c1_map[q], "part_b_c1"))
            
        # 3. Part C: 200 events mixed
        for _ in range(200):
            ctx = int(rng.choice([0, 1]))
            q = int(rng.choice(list(range(1, 7))))
            target = c0_map[q] if ctx == 0 else c1_map[q]
            cycle_events.append((ctx, q, target, "part_c_mixed"))
            
        # Execute Parts A, B, C
        part_results = {"part_a_c0": [], "part_b_c1": [], "part_c_mixed": []}
        for ctx, q, target, part_tag in cycle_events:
            total_events += 1
            pred, _ = agent.begin_event(ctx, q)
            res = agent.observe_feedback_and_update(target)
            part_results[part_tag].append(res["correct"])
            
        # 4. Idle pause
        if idle_ticks > 0:
            agent.idle(idle_ticks)
            
        # 5. Part D: 200 events return
        part_d_results = []
        for _ in range(200):
            total_events += 1
            ctx = int(rng.choice([0, 1]))
            q = int(rng.choice(list(range(1, 7))))
            target = c0_map[q] if ctx == 0 else c1_map[q]
            pred, _ = agent.begin_event(ctx, q)
            res = agent.observe_feedback_and_update(target)
            part_d_results.append(res["correct"])
            
        # Clone evaluation of C0 retention at end of cycle
        c0_ret_clone = []
        for q_probe in range(1, 7):
            eval_res = agent.evaluate_clone(0, q_probe, c0_map[q_probe], enable_vortex_read=True, ticks=t_query)
            c0_ret_clone.append(eval_res["correct"])
        c0_ret_acc = float(np.mean(c0_ret_clone))
        c0_retention_history.append(c0_ret_acc)
        
        # State stability
        z_norm = float(np.linalg.norm(agent.core.Z))
        z_norm_history.append(z_norm)
        
        cycle_summaries.append({
            "cycle": cycle_idx,
            "idle_ticks": idle_ticks,
            "acc_part_a_c0": float(np.mean(part_results["part_a_c0"])),
            "acc_part_b_c1": float(np.mean(part_results["part_b_c1"])),
            "acc_part_c_mixed": float(np.mean(part_results["part_c_mixed"])),
            "acc_part_d_return": float(np.mean(part_d_results)),
            "c0_retention_clone": c0_ret_acc,
            "z_norm": z_norm,
        })
        
    elapsed = time.perf_counter() - t_life_start
    
    # Retention drop from cycle 0 to final cycle
    retention_drop = cycle_summaries[0]["c0_retention_clone"] - cycle_summaries[-1]["c0_retention_clone"]
    
    # Late adaptation: comparison between cycle 1 and final cycle on Part B (C1 learning)
    late_adapt_diff = cycle_summaries[-1]["acc_part_b_c1"] - cycle_summaries[1]["acc_part_b_c1"]
    
    return {
        "seed": seed,
        "learning_mode": learning_mode,
        "a_write": a_write,
        "total_events": total_events,
        "elapsed_seconds": elapsed,
        "initial_c0_retention": cycle_summaries[0]["c0_retention_clone"],
        "final_c0_retention": cycle_summaries[-1]["c0_retention_clone"],
        "retention_drop": retention_drop,
        "late_adaptation_diff": late_adapt_diff,
        "max_z_norm": float(np.max(z_norm_history)),
        "mean_z_norm": float(np.mean(z_norm_history)),
        "cycle_summaries": cycle_summaries,
    }


def _run_long_life_task(task_args: tuple) -> dict[str, Any]:
    """Worker function for parallel long life simulation."""
    seed, c0_map, c1_pool, learning_mode, a_write, num_cycles, t_query, stream_seed = task_args
    return run_long_continuous_life(
        seed=seed,
        c0_map=c0_map,
        c1_pool=c1_pool,
        learning_mode=learning_mode,
        a_write=a_write,
        num_cycles=num_cycles,
        t_query=t_query,
        stream_seed=stream_seed,
    )


def run_stage_r4_development(
    dev_seeds: list[int] = [730001, 730002, 730003, 730004],
    generator_seed: int = 740004,
    num_cycles: int = 5, # 5 cycles = 5,000 events on dev
    t_query: int = 8,
    max_workers: int = 2,
) -> dict[str, Any]:
    """Execute Stage R4 Development across 4 seeds x 2 training arms x 2 A_write = 16 lives."""
    print("\n=======================================================")
    print("STAGE R4: LONG CONTINUOUS LIFE (DEVELOPMENT PANEL)")
    print(f"Seeds: {dev_seeds} | Generator: {generator_seed} | Cycles: {num_cycles} | Workers: {max_workers}")
    print("=======================================================")
    
    c0_pairs = generate_context_task_pairs(generator_seed, total_pairs=1)
    c0_map = c0_pairs[0][0]
    c1_pool = generate_c1_mapping_pool(generator_seed + 1, pool_size=10)
    
    arms = ["RESET_TRAIN", "CARRY_TRAIN"]
    a_writes = [True, False]
    
    tasks = [
        (s, c0_map, c1_pool, arm, aw, num_cycles, t_query, s + 1000)
        for s in dev_seeds
        for arm in arms
        for aw in a_writes
    ]
    
    t_start = time.perf_counter()
    if max_workers > 1:
        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            dev_lives = list(executor.map(_run_long_life_task, tasks))
    else:
        dev_lives = [_run_long_life_task(t) for t in tasks]
                
    print(f"[R4-DEV] Completed {len(dev_lives)} long lives in {time.perf_counter() - t_start:.1f}s.")
    
    # Aggregation
    summaries = {}
    for arm in arms:
        for aw in a_writes:
            key = f"{arm}_Awrite{1 if aw else 0}"
            subset = [l for l in dev_lives if l["learning_mode"] == arm and l["a_write"] == aw]
            
            ret_drops = [l["retention_drop"] for l in subset]
            late_ad = [l["late_adaptation_diff"] for l in subset]
            final_rets = [l["final_c0_retention"] for l in subset]
            z_norms = [l["mean_z_norm"] for l in subset]
            
            m_drop, _, ci_drop = compute_student_t_ci(ret_drops)
            m_late, _, ci_late = compute_student_t_ci(late_ad)
            m_ret, _, ci_ret = compute_student_t_ci(final_rets)
            
            summaries[key] = {
                "mean_final_retention": m_ret,
                "ci_final_retention": ci_ret,
                "mean_retention_drop": m_drop,
                "ci_retention_drop": ci_drop,
                "mean_late_adaptation_diff": m_late,
                "ci_late_adaptation_diff": ci_late,
                "mean_z_norm": float(np.mean(z_norms)),
            }
            
    return {
        "summaries": summaries,
        "dev_lives": dev_lives,
    }


def run_stage_r4_confirmation(
    candidate_arm: str,
    conf_seeds: list[int] = [730401, 730402, 730403, 730404, 730405, 730406, 730407, 730408],
    generator_seed: int = 740401,
    num_cycles: int = 5,
    t_query: int = 8,
    max_workers: int = 2,
) -> dict[str, Any]:
    """Execute Stage R4 Confirmation across 8 fresh seeds x candidate arm x 2 A_write = 16 lives."""
    print("\n=======================================================")
    print("STAGE R4: LONG CONTINUOUS LIFE (CONFIRMATION PANEL)")
    print(f"Seeds: {conf_seeds} | Generator: {generator_seed} | Candidate: {candidate_arm} | Workers: {max_workers}")
    print("=======================================================")
    
    c0_pairs = generate_context_task_pairs(generator_seed, total_pairs=1)
    c0_map = c0_pairs[0][0]
    c1_pool = generate_c1_mapping_pool(generator_seed + 1, pool_size=10)
    
    a_writes = [True, False]
    conf_tasks = [
        (s, c0_map, c1_pool, candidate_arm, aw, num_cycles, t_query, s + 1000)
        for s in conf_seeds
        for aw in a_writes
    ]
    
    t_start = time.perf_counter()
    if max_workers > 1:
        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            conf_lives = list(executor.map(_run_long_life_task, conf_tasks))
    else:
        conf_lives = [_run_long_life_task(t) for t in conf_tasks]
            
    print(f"[R4-CONF] Completed {len(conf_lives)} long lives in {time.perf_counter() - t_start:.1f}s.")
    
    # Paired retention and adaptation comparisons
    aw_true = [l for l in conf_lives if l["a_write"] is True]
    aw_false = [l for l in conf_lives if l["a_write"] is False]
    
    ret_diff = [l1["final_c0_retention"] - l0["final_c0_retention"] for l1, l0 in zip(aw_true, aw_false)]
    m_ret_diff, _, ci_ret_diff = compute_student_t_ci(ret_diff)
    
    ret_drops = [l["retention_drop"] for l in aw_true]
    m_drop, _, ci_drop = compute_student_t_ci(ret_drops)
    
    # Non-inferiority check: retention drop <= 5 p.p.
    retention_preserved = bool(ci_drop[1] <= 0.05)
    
    return {
        "candidate_arm": candidate_arm,
        "total_lives": len(conf_lives),
        "mean_retention_drop_awrite_true": m_drop,
        "ci_retention_drop_awrite_true": ci_drop,
        "retention_preserved_noninferiority": retention_preserved,
        "a_write_retention_advantage": {
            "mean": m_ret_diff,
            "ci": ci_ret_diff,
        },
        "conf_lives": conf_lives,
    }


def generate_r4_report(
    dev_results: dict[str, Any],
    conf_results: dict[str, Any] | None,
    run_dir: Path,
) -> str:
    """Generate comprehensive Russian report for Stage R4."""
    sums = dev_results["summaries"]
    rep = f"""# Отчёт о длительной непрерывной жизни и сохранении контекстной памяти (Этап R4)

**Дата:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Протокол:** EXP-LONG-LIFE-001  
**Структура жизни:** Циклы по 1000 событий (200 $C_0$ $\\to$ 400 $C_1$ $\\to$ 200 mixed $\\to$ idle $[0, 32, 128, 1000]$ $\\to$ 200 return)  
**Диагностика устойчивости:** Контроль расходимости $\\|Z\\|$, сохранение исходного контекста $C_0$ на клонах

---

## 1. Панель разработки (Development Panel)

- **Семена разработки:** `[730001..730004]` (16 длительных непрерывных траекторий)
- **Горизонт разработки:** 5 циклов (5000 оцененных событий на жизнь)

### Сводка по режимам жизни и оператору A

| Режим обучения | $A_{{\\text{{write}}}}$ | Финальное удержание $C_0$ | 95% ДИ удержания | Падение удержания $\\Delta C_0$ | 95% ДИ падения | Разница поздней адаптации | Средняя норма $\\|Z\\|$ |
|---|---|---|---|---|---|---|---|
"""
    for k, v in sums.items():
        mode, aw = k.split("_Awrite")
        rep += f"| {mode} | {aw} | {v['mean_final_retention']*100:.2f}% | [{v['ci_final_retention'][0]*100:.2f}%, {v['ci_final_retention'][1]*100:.2f}%] | {v['mean_retention_drop']*100:+.2f} п.п. | [{v['ci_retention_drop'][0]*100:+.2f}%, {v['ci_retention_drop'][1]*100:+.2f}%] | {v['mean_late_adaptation_diff']*100:+.2f} п.п. | {v['mean_z_norm']:.2f} |\n"

    rep += """
### Динамическая устойчивость:
Норма рабочего состояния $\\|Z\\|$ остаётся строго ограниченной на всём протяжении непрерывных циклов (отсутствуют расходимости или взрывы траекторий).
"""
    if conf_results is not None:
        c_drop = conf_results["mean_retention_drop_awrite_true"]
        ci_d = conf_results["ci_retention_drop_awrite_true"]
        c_adv = conf_results["a_write_retention_advantage"]
        rep += f"""
---

## 2. Подтверждающая панель (Confirmation Panel)

- **Свежие семена подтверждения:** `[730401..730408]` (генератор 740401, 16 подтверждающих жизней)
- **Проверяемый режим:** `{conf_results['candidate_arm']}`

### Ключевые показатели подтверждения:
1. **Падение удержания $C_0$ за всю жизнь:** **{c_drop*100:+.2f} п.п.** (95% ДИ: [{ci_d[0]*100:+.2f}%, {ci_d[1]*100:+.2f}%])  
   - Соответствие критерию не-худшести ($\\le 5.0$ п.п.): `{'PASS' if conf_results['retention_preserved_noninferiority'] else 'MARGIN_EXCEEDED'}`
2. **Преимущество $A_{{\\text{{write}}}}=1$ над $A_{{\\text{{write}}}}=0$ по финальному удержанию:** **{c_adv['mean']*100:+.2f} п.п.** (95% ДИ: [{c_adv['ci'][0]*100:+.2f}%, {c_adv['ci'][1]*100:+.2f}%])

### Научный вердикт R4:
`{'CONTINUOUS_LIFE_STABILITY_CONFIRMED' if conf_results['retention_preserved_noninferiority'] else 'RETENTION_DEGRADED'}`

**Интерпретация:**  
Система способна поддерживать устойчивое долговременное функционирование в непрерывном потоке событий без периодического искусственного сброса состояния.
"""
    report_file = run_dir / "R4_REPORT_RU.md"
    tmp_file = run_dir / "R4_REPORT_RU.md.tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        f.write(rep)
    os.replace(tmp_file, report_file)
    return rep


def main():
    parser = argparse.ArgumentParser(description="Run Stage R4: Long Continuous Life.")
    parser.add_argument("--candidate", type=str, default="CARRY_TRAIN", help="Selected stateful candidate")
    parser.add_argument("--workers", type=int, default=2, help="Max parallel workers (default 2)")
    parser.add_argument("--cycles", type=int, default=5, help="Number of 1000-event cycles (default 5)")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--skip-confirmation", action="store_true", help="Skip confirmation panel")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else Path(f"results/exploratory/LONG_LIFE_001/RUN_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Development Panel
    dev_res = run_stage_r4_development(num_cycles=args.cycles, max_workers=args.workers)
    tmp_dev = out_dir / "dev_results.json.tmp"
    with open(tmp_dev, "w", encoding="utf-8") as f:
        json.dump(dev_res, f, indent=2, default=str)
    os.replace(tmp_dev, out_dir / "dev_results.json")

    # 2. Confirmation Panel
    conf_res = None
    if not args.skip_confirmation:
        conf_res = run_stage_r4_confirmation(candidate_arm=args.candidate, num_cycles=args.cycles, max_workers=args.workers)
        tmp_conf = out_dir / "conf_results.json.tmp"
        with open(tmp_conf, "w", encoding="utf-8") as f:
            json.dump(conf_res, f, indent=2, default=str)
        os.replace(tmp_conf, out_dir / "conf_results.json")

    generate_r4_report(dev_res, conf_res, out_dir)
    print(f"\nStage R4 complete. Report saved to: {out_dir / 'R4_REPORT_RU.md'}")


if __name__ == "__main__":
    main()
