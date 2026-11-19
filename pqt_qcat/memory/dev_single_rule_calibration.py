"""Development investigation: Single-rule learnability calibration in 6-echelon core.

Diagnoses and establishes conditions for achieving >= 75% learnability on a single rule:
1. Compares RESET_TRAIN (canonical M3 reference) vs CARRY_TRAIN (continuous life).
2. Investigates query duration: t_query in {8, 16, 32}.
3. Investigates context interface drive: t_context in {0, 4}.
4. Evaluates across multiple development seeds (e.g., 42, 730001, 730002).
5. Tracks accuracy trajectory over 72 events, final 6-token predictions, and class entropy.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np

from pqt_qcat.core.security import check_seed_safety
from pqt_qcat.memory.dev_context_pathway_ablation import make_stable_core
from pqt_qcat.memory.exp_stateful_online_001 import (
    StatefulOnlineAgent,
    generate_context_task_pairs,
)
from pqt_qcat.memory.exp_online_calibration_001 import generate_single_rule_schedule


def run_single_rule_experiment(
    core_seed: int = 42,
    task_seed: int = 740003,
    stream_seed: int = 42,
    learning_mode: str = "RESET_TRAIN",
    t_context: int = 0,
    t_query: int = 8,
    eta_k: float = 0.8,
    total_events: int = 72,
    idle_ticks: int = 0,
) -> dict[str, Any]:
    """Execute a single-rule training life and measure accuracy trajectory."""
    check_seed_safety(core_seed, allow_quarantined=True)
    pair = generate_context_task_pairs(task_seed, 4)[0]
    m0 = pair[0] # Single mapping for Context 0
    schedule = generate_single_rule_schedule(m0, total_events=total_events, seed=stream_seed)

    core = make_stable_core(core_seed, substeps=2)
    agent = StatefulOnlineAgent(
        core=core,
        learning_mode=learning_mode,
        t_context=t_context,
        t_query=t_query,
        eta_k=eta_k,
        write_gate_presentation=1.0,
    )

    history = []
    corrects = []
    failure = None
    for i, ev in enumerate(schedule):
        try:
            p, logits = agent.begin_event(ev["context"], ev["query"])
            res = agent.observe_feedback_and_update(ev["target"])
            corrects.append(res["correct"])
            if idle_ticks > 0:
                agent.idle(idle_ticks)

            history.append({
                "event_idx": i,
                "query": ev["query"],
                "target": ev["target"],
                "pred": p,
                "correct": res["correct"],
                "margin": res["signed_margin"],
            })
        except Exception as exc:
            failure = {"event_idx": i, "error_type": type(exc).__name__, "error_message": str(exc)}
            break

    # Evaluate final clean accuracy on all 6 tokens
    final_preds = []
    if failure is None:
        try:
            for q in range(1, 7):
                p, _ = agent.begin_event(0, q)
                final_preds.append(p)
                if learning_mode == "RESET_TRAIN":
                    agent.reset_state_to_z0()
        except Exception as exc:
            failure = {"event_idx": len(corrects), "error_type": type(exc).__name__, "error_message": str(exc)}
            final_preds = [-1] * 6
    else:
        final_preds = [-1] * 6

    targets = [m0[q] for q in range(1, 7)]
    final_acc = float(np.mean([p == t for p, t in zip(final_preds, targets)])) if failure is None else 0.0
    last_24_acc = float(np.mean(corrects[-24:])) if len(corrects) >= 24 else (float(np.mean(corrects)) if corrects else 0.0)
    overall_acc = float(np.mean(corrects)) if corrects else 0.0

    # Compute prediction entropy (to detect attractor collapse)
    valid_preds = [p for p in final_preds if p > 0]
    if valid_preds:
        counts = np.bincount(valid_preds, minlength=5)[1:]
        probs = counts / np.sum(counts)
        nonzero_p = probs[probs > 0]
        pred_entropy = float(-np.sum(nonzero_p * np.log2(nonzero_p))) if len(nonzero_p) > 0 else 0.0
    else:
        pred_entropy = 0.0

    return {
        "core_seed": core_seed,
        "learning_mode": learning_mode,
        "t_context": t_context,
        "t_query": t_query,
        "idle_ticks": idle_ticks,
        "total_events": total_events,
        "failure": failure,
        "overall_accuracy": overall_acc,
        "last_24_accuracy": last_24_acc,
        "final_accuracy": final_acc,
        "final_predictions": final_preds,
        "targets": targets,
        "prediction_entropy": pred_entropy,
        "collapsed": bool(pred_entropy < 0.1),
    }


def run_calibration_battery(
    seeds: list[int] = [42, 730001, 730002],
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run comprehensive single-rule calibration battery."""
    configs = [
        # 1. Canonical M3 reference: RESET_TRAIN, t_query=8, t_context=0
        {"name": "M3_Reference_RESET_t8_ctx0", "mode": "RESET_TRAIN", "t_context": 0, "t_query": 8, "idle": 0},
        # 2. RESET_TRAIN with context pulse: t_context=4, t_query=8
        {"name": "RESET_t8_ctx4", "mode": "RESET_TRAIN", "t_context": 4, "t_query": 8, "idle": 0},
        # 3. RESET_TRAIN with long query: t_context=0, t_query=32
        {"name": "RESET_t32_ctx0", "mode": "RESET_TRAIN", "t_context": 0, "t_query": 32, "idle": 0},
        # 4. CARRY_TRAIN: t_context=0, t_query=8, idle=0
        {"name": "CARRY_t8_ctx0_idle0", "mode": "CARRY_TRAIN", "t_context": 0, "t_query": 8, "idle": 0},
        # 5. CARRY_TRAIN with idle relaxation: t_context=0, t_query=8, idle=16
        {"name": "CARRY_t8_ctx0_idle16", "mode": "CARRY_TRAIN", "t_context": 0, "t_query": 8, "idle": 16},
        # 6. Prior calibration setup: CARRY_TRAIN, t_context=4, t_query=32, idle=0
        {"name": "Prior_Calibration_CARRY_t32_ctx4", "mode": "CARRY_TRAIN", "t_context": 4, "t_query": 32, "idle": 0},
    ]

    results_by_config = {}

    for cfg in configs:
        name = cfg["name"]
        print(f"\n[SingleRule] Running config: {name}...")
        runs = []
        for s in seeds:
            res = run_single_rule_experiment(
                core_seed=s,
                learning_mode=cfg["mode"],
                t_context=cfg["t_context"],
                t_query=cfg["t_query"],
                idle_ticks=cfg["idle"],
                total_events=72,
            )
            runs.append(res)
            print(f"  Seed {s}: final_acc={res['final_accuracy']*100:.1f}%, last24={res['last_24_accuracy']*100:.1f}%, preds={res['final_predictions']}")

        mean_final_acc = float(np.mean([r["final_accuracy"] for r in runs]))
        mean_last24_acc = float(np.mean([r["last_24_accuracy"] for r in runs]))
        collapsed_count = sum(r["collapsed"] for r in runs)

        results_by_config[name] = {
            "config": cfg,
            "mean_final_accuracy": mean_final_acc,
            "mean_last_24_accuracy": mean_last24_acc,
            "collapsed_runs": f"{collapsed_count}/{len(seeds)}",
            "runs": runs,
        }

    output_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": seeds,
        "configs": results_by_config,
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_json = output_dir / "single_rule_results.json"
        tmp_json = output_dir / "single_rule_results.json.tmp"
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)
        os.replace(tmp_json, out_json)

        generate_single_rule_report(output_payload, output_dir / "SINGLE_RULE_CALIBRATION_REPORT_RU.md")

    return output_payload


def generate_single_rule_report(data: dict[str, Any], filepath: Path) -> None:
    """Generate Markdown diagnostic report for single rule learnability."""
    rep = rf"""# Отчёт о калибровке обучаемости одного правила (Single-Rule Learnability)

**Дата:** {data['timestamp']}  
**Семя ядра:** {data['seeds']}  
**Бюджет:** 72 события (12 повторений каждого из 6 токенов).  
**Целевой научный критерий:** $\ge 75\%$ точности на одном правиле.

## 1. Сводные результаты конфигураций

| Конфигурация | Режим | $t_{{\text{{context}}}}$ | $t_{{\text{{query}}}}$ | Idle | Средняя финальная точность | Точность последних 24 событий | Коллапс аттрактора |
|---|---|---|---|---|---|---|---|
"""
    for name, c_res in data["configs"].items():
        c = c_res["config"]
        rep += f"| `{name}` | `{c['mode']}` | {c['t_context']} | {c['t_query']} | {c['idle']} | **{c_res['mean_final_accuracy']*100:.1f}%** | {c_res['mean_last_24_accuracy']*100:.1f}% | {c_res['collapsed_runs']} |\n"

    rep += """
## 2. Ограничения интерпретации

Числа выше относятся к указанному режиму RESET/CARRY и данной development-панели.
Высокая точность со сбросами не устанавливает непрерывную обучаемость.
Энтропия предсказанных классов не доказывает существование единственного аттрактора.
Причины различий (контекст, длительность, самосвязи) требуют отдельных парных интервенций.
Порог или изменение архитектуры не выбираются автоматически из этой таблицы.
"""
    tmp_path = filepath.with_suffix(".md.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(rep)
    os.replace(tmp_path, filepath)
    print(f"[SingleRule] Report written to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Run single-rule learnability calibration.")
    parser.add_argument("--output", type=str, default="results/exploratory/SINGLE_RULE_CALIBRATION_001")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 730001, 730002])
    args = parser.parse_args()

    out_dir = Path(args.output)
    run_calibration_battery(seeds=args.seeds, output_dir=out_dir)


if __name__ == "__main__":
    main()
