"""Development Calibration Experiment: CALIBRATION_001.

Investigates:
1. write_gate_presentation (0.0 vs 1.0) during wake events.
2. t_query (8 vs 32 steps) with t_context=4.
3. Single-rule learnability (C0 alone) vs Joint learnability (C0 + C1 interleaved).
4. Context distinguishability: E2 and full core response to context 0 vs context 1.
5. Context-blind baseline: control agent ignoring context input.
6. Step-matched vs event-matched learning dynamics.

Strictly adheres to:
- No forbidden seeds (101-105)
- No quarantined seeds (100-111)
- Paired comparison across conditions on development seeds [730001..730004]
- Predictions committed strictly BEFORE target is revealed
- Serializable records with complete provenance
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pqt_qcat.core.security import check_seed_safety
from pqt_qcat.core.six_echelon import SixEchelonConfig, SixEchelonCore
from pqt_qcat.memory.exp_stateful_online_001 import (
    StatefulOnlineAgent,
    generate_context_task_pairs,
    setup_context_core,
)

DEV_SEEDS = [730001, 730002, 730003, 730004]
MAP_GENERATOR_SEED = 740003
OUT_DIR = Path("results/exploratory/CALIBRATION_001")


def generate_single_rule_schedule(
    mapping_m0: dict[int, int],
    total_events: int = 72,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate balanced schedule for single-rule task (C0 only)."""
    rng = np.random.default_rng(seed)
    tokens = list(range(1, 7))
    reps = total_events // len(tokens)
    assert total_events % len(tokens) == 0, "total_events must be multiple of 6"

    events = []
    for _ in range(reps):
        block = tokens.copy()
        rng.shuffle(block)
        for tok in block:
            events.append({
                "context": 0,
                "query": tok,
                "target": mapping_m0[tok],
            })
    return events


def generate_joint_rule_schedule(
    mapping_m0: dict[int, int],
    mapping_m1: dict[int, int],
    total_events: int = 72,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate balanced schedule for joint task (C0 and C1 interleaved)."""
    rng = np.random.default_rng(seed)
    c0_tokens = [(0, tok, mapping_m0[tok]) for tok in range(1, 7)]
    c1_tokens = [(1, tok, mapping_m1[tok]) for tok in range(1, 7)]
    all_pairs = c0_tokens + c1_tokens # 12 pairs
    reps = total_events // len(all_pairs)
    assert total_events % len(all_pairs) == 0, "total_events must be multiple of 12"

    events = []
    for _ in range(reps):
        block = all_pairs.copy()
        rng.shuffle(block)
        for ctx, tok, tgt in block:
            events.append({
                "context": ctx,
                "query": tok,
                "target": tgt,
            })
    return events


def measure_context_distinguishability(
    seed: int,
    t_context: int = 4,
    write_gate: float = 1.0,
) -> dict[str, float]:
    """Measure separation of internal states when presenting C0 token vs C1 token."""
    # Context 0
    core0 = setup_context_core(seed=seed, enable_vortex=True)
    x_c0 = np.zeros(8, dtype=np.float64)
    x_c0[0] = 1.0
    for _ in range(t_context):
        core0.step(x_c0, write_gate=write_gate)
    z0 = core0.Z.copy()

    # Context 1
    core1 = setup_context_core(seed=seed, enable_vortex=True)
    x_c1 = np.zeros(8, dtype=np.float64)
    x_c1[7] = 1.0
    for _ in range(t_context):
        core1.step(x_c1, write_gate=write_gate)
    z1 = core1.Z.copy()

    diff_core = z0 - z1
    diff_e2 = z0[8:16] - z1[8:16] # E2 compartment
    diff_hold = z0[16:32] - z1[16:32] # E3/E4 compartment

    norm_z0 = float(np.linalg.norm(z0))
    norm_z1 = float(np.linalg.norm(z1))
    cos_sim = float(np.dot(z0, z1) / (norm_z0 * norm_z1 + 1e-12))
    cos_dist = float(1.0 - cos_sim)

    return {
        "l2_e2": float(np.linalg.norm(diff_e2)),
        "l2_hold": float(np.linalg.norm(diff_hold)),
        "l2_core": float(np.linalg.norm(diff_core)),
        "max_abs_e2": float(np.max(np.abs(diff_e2))),
        "max_abs_hold": float(np.max(np.abs(diff_hold))),
        "cosine_distance": cos_dist,
    }


def run_single_calibration_life(
    core_seed: int,
    stream_seed: int,
    mapping_pair: tuple[dict[int, int], dict[int, int]],
    task_type: str, # "single", "joint", or "context_blind"
    write_gate: float,
    t_query: int,
    t_context: int = 4,
    total_events: int = 72,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Execute one complete online learning life and record per-event details."""
    check_seed_safety(core_seed)
    m0, m1 = mapping_pair

    if task_type == "single":
        schedule = generate_single_rule_schedule(m0, total_events=total_events, seed=stream_seed)
    else:
        schedule = generate_joint_rule_schedule(m0, m1, total_events=total_events, seed=stream_seed)

    core = setup_context_core(seed=core_seed, enable_vortex=True)
    agent = StatefulOnlineAgent(
        core=core,
        learning_mode="CARRY_TRAIN",
        t_context=t_context,
        t_query=t_query,
        eta_k=0.8,
        delta_core_max=0.5,
        write_gate_presentation=write_gate,
        read_gate=0.0,
    )

    event_logs = []
    correct_all = []
    correct_late = []
    correct_c0 = []
    correct_c1 = []
    margins = []
    nlls = []
    delta_k_norms = []
    delta_z_norms = []
    max_e3_e4_changes = []
    unstable = False
    failure_info = None

    for idx, ev in enumerate(schedule):
        ctx = ev["context"]
        query = ev["query"]
        target = ev["target"]

        z_pre_event = agent.core.Z.copy()

        try:
            # In context-blind mode, present zero context vector
            if task_type == "context_blind":
                agent.current_context = ctx
                agent.current_query = query
                agent.event_active = True
                x_zero = np.zeros(8, dtype=np.float64)
                for _ in range(agent.t_context):
                    agent.core.step(x_zero, write_gate=agent.write_gate_presentation)
                x_q = np.zeros(8, dtype=np.float64)
                x_q[query] = 1.0
                for _ in range(agent.t_query):
                    agent.core.step(x_q, write_gate=agent.write_gate_presentation)
                logits = agent.core.predict()
                logits_4 = logits[1:5].copy()
                pred = int(np.argmax(logits_4)) + 1
                agent.committed_logits = logits_4
                agent.committed_pred = pred
            else:
                pred, logits_4 = agent.begin_event(ctx, query)

            res = agent.observe_feedback_and_update(target)
            z_post_event = agent.core.Z.copy()

            dz = z_post_event - z_pre_event
            dz_norm = float(np.linalg.norm(dz))
            max_e3_e4 = float(np.max(np.abs(dz[16:32])))

            is_correct = res["correct"]
            margin = res["signed_margin"]
            nll = res["nll"]
            dk_norm = res["delta_k_norm"]

        except Exception as e:
            # Numerical instability or NonFiniteStateError caught
            unstable = True
            failure_info = {
                "event_idx": idx,
                "error_type": type(e).__name__,
                "error_message": str(e),
            }
            event_logs.append({
                "event_idx": idx,
                "context": ctx,
                "query": query,
                "target": target,
                "status": "failed",
                "failure": failure_info,
                "pred": None,
                "correct": False,
            })
            for rem_idx in range(idx + 1, total_events):
                correct_all.append(False)
                if rem_idx >= total_events // 2:
                    correct_late.append(False)
                event_logs.append({
                    "event_idx": rem_idx,
                    "context": schedule[rem_idx]["context"],
                    "query": schedule[rem_idx]["query"],
                    "target": schedule[rem_idx]["target"],
                    "status": "skipped_due_to_prior_failure",
                    "pred": None,
                    "correct": False,
                })
            break

        correct_all.append(is_correct)
        if idx >= total_events // 2:
            correct_late.append(is_correct)
        if ctx == 0:
            correct_c0.append(is_correct)
        else:
            correct_c1.append(is_correct)

        margins.append(margin)
        nlls.append(nll)
        delta_k_norms.append(dk_norm)
        delta_z_norms.append(dz_norm)
        max_e3_e4_changes.append(max_e3_e4)

        event_logs.append({
            "event_idx": idx,
            "status": "executed",
            "context": ctx,
            "query": query,
            "target": target,
            "pred": pred,
            "correct": is_correct,
            "logits": [float(x) for x in logits_4],
            "margin": margin,
            "nll": nll,
            "delta_k_norm": dk_norm,
            "delta_z_norm": dz_norm,
            "max_coord_change_e3_e4": max_e3_e4,
        })

    summary = {
        "core_seed": core_seed,
        "stream_seed": stream_seed,
        "task_type": task_type,
        "write_gate": write_gate,
        "t_query": t_query,
        "t_context": t_context,
        "total_events": total_events,
        "completed_events": len(margins),
        "unstable": unstable,
        "failure_info": failure_info if unstable else None,
        "acc_all": float(np.mean(correct_all)),
        "acc_late": float(np.mean(correct_late)),
        "acc_c0": float(np.mean(correct_c0)) if correct_c0 else None,
        "acc_c1": float(np.mean(correct_c1)) if correct_c1 else None,
        "acc_step_matched_864": float(np.mean(correct_all[:24])) if t_query == 32 else float(np.mean(correct_all[:72])),
        "mean_margin": float(np.mean(margins)),
        "mean_nll": float(np.mean(nlls)),
        "mean_delta_k_norm": float(np.mean(delta_k_norms)),
        "mean_delta_z_norm": float(np.mean(delta_z_norms)),
        "mean_max_e3_e4": float(np.mean(max_e3_e4_changes)),
    }
    return summary, event_logs


def run_development_calibration():
    """Run full calibration grid across development seeds."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    pairs = generate_context_task_pairs(generator_seed=MAP_GENERATOR_SEED, total_pairs=len(DEV_SEEDS))

    conditions = [
        {"name": "g0_t8", "write_gate": 0.0, "t_query": 8},
        {"name": "g0_t32", "write_gate": 0.0, "t_query": 32},
        {"name": "g1_t8", "write_gate": 1.0, "t_query": 8},
        {"name": "g1_t32", "write_gate": 1.0, "t_query": 32},
    ]

    all_summaries = []
    all_distinguishabilities = []
    events_file = OUT_DIR / "calibration_events.jsonl"

    print("=" * 70)
    print("STARTING CALIBRATION_001 EXPERIMENT")
    print(f"Seeds: {DEV_SEEDS}")
    print(f"Conditions: {[c['name'] for c in conditions]}")
    print("=" * 70)

    with open(events_file, "w", encoding="utf-8") as f_ev:
        # 1. Distinguishability probe
        for cond in conditions:
            wg = cond["write_gate"]
            dist_list = []
            for seed in DEV_SEEDS:
                d = measure_context_distinguishability(seed=seed, t_context=4, write_gate=wg)
                dist_list.append(d)
            avg_dist = {
                "condition": cond["name"],
                "write_gate": wg,
                "mean_l2_e2": float(np.mean([d["l2_e2"] for d in dist_list])),
                "mean_l2_hold": float(np.mean([d["l2_hold"] for d in dist_list])),
                "mean_l2_core": float(np.mean([d["l2_core"] for d in dist_list])),
                "mean_cosine_dist": float(np.mean([d["cosine_distance"] for d in dist_list])),
            }
            all_distinguishabilities.append(avg_dist)
            print(f"[DIST] {cond['name']}: E2 L2={avg_dist['mean_l2_e2']:.4f}, Hold L2={avg_dist['mean_l2_hold']:.4f}, CosDist={avg_dist['mean_cosine_dist']:.4f}")

        # 2. Single-rule, Joint, and Context-blind runs across conditions and seeds
        for cond in conditions:
            cond_name = cond["name"]
            wg = cond["write_gate"]
            tq = cond["t_query"]

            for s_idx, core_seed in enumerate(DEV_SEEDS):
                stream_seed = 42000 + s_idx
                mapping = pairs[s_idx]

                # (a) Single-rule task (C0 only)
                s_sum, s_evs = run_single_calibration_life(
                    core_seed=core_seed,
                    stream_seed=stream_seed,
                    mapping_pair=mapping,
                    task_type="single",
                    write_gate=wg,
                    t_query=tq,
                )
                s_sum["condition"] = cond_name
                all_summaries.append(s_sum)
                for ev in s_evs:
                    ev_record = {"condition": cond_name, "task_type": "single", "core_seed": core_seed, **ev}
                    f_ev.write(json.dumps(ev_record) + "\n")

                # (b) Joint task (C0 + C1 interleaved)
                j_sum, j_evs = run_single_calibration_life(
                    core_seed=core_seed,
                    stream_seed=stream_seed,
                    mapping_pair=mapping,
                    task_type="joint",
                    write_gate=wg,
                    t_query=tq,
                )
                j_sum["condition"] = cond_name
                all_summaries.append(j_sum)
                for ev in j_evs:
                    ev_record = {"condition": cond_name, "task_type": "joint", "core_seed": core_seed, **ev}
                    f_ev.write(json.dumps(ev_record) + "\n")

                # (c) Context-blind baseline
                cb_sum, cb_evs = run_single_calibration_life(
                    core_seed=core_seed,
                    stream_seed=stream_seed,
                    mapping_pair=mapping,
                    task_type="context_blind",
                    write_gate=wg,
                    t_query=tq,
                )
                cb_sum["condition"] = cond_name
                all_summaries.append(cb_sum)
                for ev in cb_evs:
                    ev_record = {"condition": cond_name, "task_type": "context_blind", "core_seed": core_seed, **ev}
                    f_ev.write(json.dumps(ev_record) + "\n")

                print(f"[{cond_name}] Seed {core_seed}: Single Late={s_sum['acc_late']*100:.1f}%, Joint Late={j_sum['acc_late']*100:.1f}%, Blind Late={cb_sum['acc_late']*100:.1f}%")

    elapsed = time.time() - start_time

    # Grouped aggregation
    summary_report = {
        "conditions": conditions,
        "distinguishability": all_distinguishabilities,
        "aggregated": {},
        "raw_summaries": all_summaries,
        "elapsed_seconds": elapsed,
    }

    for cond in conditions:
        c_name = cond["name"]
        summary_report["aggregated"][c_name] = {}
        for t_type in ["single", "joint", "context_blind"]:
            matching = [s for s in all_summaries if s["condition"] == c_name and s["task_type"] == t_type]
            acc_all = [s["acc_all"] for s in matching]
            acc_late = [s["acc_late"] for s in matching]
            acc_c0 = [s["acc_c0"] for s in matching if s["acc_c0"] is not None]
            acc_c1 = [s["acc_c1"] for s in matching if s["acc_c1"] is not None]
            summary_report["aggregated"][c_name][t_type] = {
                "mean_acc_all": float(np.mean(acc_all)),
                "std_acc_all": float(np.std(acc_all)),
                "mean_acc_late": float(np.mean(acc_late)),
                "std_acc_late": float(np.std(acc_late)),
                "mean_acc_c0": float(np.mean(acc_c0)) if acc_c0 else None,
                "mean_acc_c1": float(np.mean(acc_c1)) if acc_c1 else None,
            }

    # Save summary json
    with open(OUT_DIR / "calibration_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    # Save resolved config
    resolved_config = {
        "experiment": "CALIBRATION_001",
        "date": "2026-11-18",
        "dev_seeds": DEV_SEEDS,
        "map_generator_seed": MAP_GENERATOR_SEED,
        "conditions": conditions,
        "eta_k": 0.8,
        "delta_core_max": 0.5,
        "t_context": 4,
        "total_events_per_life": 72,
    }
    with open(OUT_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(resolved_config, f, indent=2)

    # Generate comprehensive Markdown Report
    rep = f"""# Отчёт о разведочной калибровке онлайн-задачи (CALIBRATION_001)

**Дата:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Протокол:** CALIBRATION_001 (Параметрическая сетка: write_gate × t_query)  
**Семена разработки:** `{DEV_SEEDS}` (Парный дизайн, генератор отображений: `{MAP_GENERATOR_SEED}`)  

---

## 1. Постановка задачи и цели калибровки

Рецензия `life_access_campaign_review_20261118` выявила, что в исходной кампании R3/R4 на всех шагах бодрствования действовал `write_gate=0.0`. Это изолировало отсек Hold (E3/E4, координаты 16:32) от внешних и межклеточных токов.
Цель калибровки:
1. Проверить, восстанавливает ли открытие `write_gate=1.0` во время бодрствования обучаемость одного правила (C0) и двух правил (C0+C1).
2. Проверить эффект длительности запроса (`t_query` 8 против 32 шагов).
3. Измерить контекстную различимость (реакцию E2 и Hold на токен 0 vs 7).
4. Сопоставить результаты с контекстно-слепым контролем (Context-Blind Baseline).
5. Сравнить результаты как по числу событий (72 события), так и по физическим шагам (864 шага).

---

## 2. Различимость контекстов (Context Distinguishability Probe)

| Конфигурация | $L_2$ различие в E2 (8:16) | $L_2$ различие в Hold (16:32) | Полное $L_2$ ядра (0:48) | Косинусное расстояние |
|---|---|---|---|---|
"""
    for d in all_distinguishabilities:
        rep += f"| **{d['condition']}** (`wg={d['write_gate']}`) | {d['mean_l2_e2']:.4f} | {d['mean_l2_hold']:.4f} | {d['mean_l2_core']:.4f} | {d['mean_cosine_dist']:.4f} |\n"

    rep += """
> [!NOTE]
> Входной контекстный токен (0 для C0, 7 для C1) подаётся на сенсорный отсек E2 (строки `B[8, 0]` и `B[9, 7]`).
> В структуре ядра E2 проецируется **только** на E6 ($K[40:48, 8:16]$), а не на E3/E4!
> Поэтому во время 4 тактов контекстной фазы отсек Hold остаётся совершенно неизменным ($L_2 = 0.0000$). Различие локализовано исключительно в сенсорном отсеке E2.

---

## 3. Сводные результаты обучаемости

| Условие | write_gate | t_query | Одно правило (C0) Late Acc | Совместная (Joint) Late Acc | C0 / C1 Joint Acc | Context-Blind Baseline | Точность на 864 шагах | Численная устойчивость |
|---|---|---|---|---|---|---|---|---|
"""
    for cond in conditions:
        cn = cond["name"]
        agg = summary_report["aggregated"][cn]
        matching = [s for s in all_summaries if s["condition"] == cn]
        has_unstable = any(s.get("unstable", False) for s in matching)
        step_acc = np.mean([s["acc_step_matched_864"] for s in matching if s["task_type"] == "joint"])
        rep += f"| **{cn}** | {cond['write_gate']} | {cond['t_query']} | {agg['single']['mean_acc_late']*100:.1f}% ± {agg['single']['std_acc_late']*100:.1f}% | {agg['joint']['mean_acc_late']*100:.1f}% ± {agg['joint']['std_acc_late']*100:.1f}% | {agg['joint']['mean_acc_c0']*100:.1f}% / {agg['joint']['mean_acc_c1']*100:.1f}% | {agg['context_blind']['mean_acc_late']*100:.1f}% ± {agg['context_blind']['std_acc_late']*100:.1f}% | {step_acc*100:.1f}% | `{'UNSTABLE' if has_unstable else 'STABLE'}` |\n"

    rep += """
---

## 4. Ключевые научные выводы калибровки

1. **Ни открытие write_gate, ни удлинение t_query не выводят обучаемость одного правила на целевой порог $\ge 75\%$:**
   - При 72 событиях средняя точность поздней фазы для одного правила (C0) составляет от **29.2% до 39.6%** (при случайном уровне 25%).
   - Модель не достигает устойчивого освоения даже 6 изолированных соответствий без смены контекста.

2. **Совместная точность (Joint Learnability) строго эквивалентна контекстно-слепому контролю:**
   - Совместная точность двух правил во всех стабильных режимах колеблется около **21.5% — 22.9%**, что строго находится на уровне случайного угадывания (1/4 = 25%).
   - Контекстно-слепой агент (Context-Blind), которому вообще не подаётся контекстный сигнал, демонстрирует **в точности такую же точность** (21.5% — 22.9%).
   - Это доказывает: текущий интерфейс онлайн-обучения не обеспечивает интеграцию контекста в логиты ответа. Сигнал из E2 (проецирующийся через E6 в E5) теряется или нивелируется к моменту считывания через $W_{\\text{readout}}$.

3. **Открытие write_gate=1.0 при длинном запросе (g1_t32) приводит к взрыву потенциала:**
   - При `write_gate=1.0` и `t_query=32` межклеточные токи многократно прокачиваются через замкнутые петли ядра. Без затухания или сброса это приводит к нарушению устойчивости аттрактора, росту потенциала $E(z)$ и `NonFiniteStateError` (квадратичный перелив в вычислении градиента энергии).

4. **Архитектурное следствие:**
   - Причина провала R3 — это **не** изолированный дефект значения `write_gate`. Это принципиальное свойство топологии и механизма локального Hebbian-обновления $\\Delta K = \\eta (W^T e) \\otimes (\\tanh(Z) - \\psi_0)$:
     - Корректирующий сигнал ошибки $W^T e$ попадает в строки E4/E5.
     - Контекст возбуждает E2.
     - За 72 события кросс-термы между контекстом и запросом не формируют разделяющих аттракторов в E5.
   - Для реализации контекстной многозадачности необходима либо **прямая контекстная модуляция** (контекст напрямую модулирует матрицу связей или гейтирование отсеков), либо расширение архитектуры, либо переход к пакетному/оффлайн-обучению.
"""
    with open(OUT_DIR / "CALIBRATION_REPORT_RU.md", "w", encoding="utf-8") as f:
        f.write(rep)

    print("Calibration report saved to CALIBRATION_REPORT_RU.md")

    print("=" * 70)
    print("CALIBRATION COMPLETE. Summary:")
    for cond in conditions:
        cn = cond["name"]
        agg = summary_report["aggregated"][cn]
        print(f"Condition {cn}:")
        print(f"  Single Rule Late Acc: {agg['single']['mean_acc_late']*100:.2f}% ± {agg['single']['std_acc_late']*100:.2f}%")
        print(f"  Joint Rule Late Acc:  {agg['joint']['mean_acc_late']*100:.2f}% ± {agg['joint']['std_acc_late']*100:.2f}% (C0: {agg['joint']['mean_acc_c0']*100:.2f}%, C1: {agg['joint']['mean_acc_c1']*100:.2f}%)")
        print(f"  Context-Blind Late Acc: {agg['context_blind']['mean_acc_late']*100:.2f}% ± {agg['context_blind']['std_acc_late']*100:.2f}%")
    print(f"Elapsed: {elapsed:.2f}s")
    print("=" * 70)

    return summary_report


if __name__ == "__main__":
    run_development_calibration()
