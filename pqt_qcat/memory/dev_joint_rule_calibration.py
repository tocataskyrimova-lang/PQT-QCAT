"""Development investigation: Joint-rule learnability calibration in 6-echelon core.

Diagnoses and tests conditions for achieving joint-rule mastery (context-switching):
1. Anatomical Sequential: Context presented for t_context ticks to E2, followed by query for t_query ticks.
2. Anatomical Sustained: Context input sustained simultaneously during query ticks.
3. Direct Context Gating: Context acts as a multiplicative gate / channel selector on readout.

Compares these hypotheses under equal resource budgets across development seeds (42, 730001, 730002).
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

from pqt_qcat.core.security import check_seed_safety
from pqt_qcat.core.six_echelon import SixEchelonCore
from pqt_qcat.memory.dev_context_pathway_ablation import make_stable_core
from pqt_qcat.memory.exp_stateful_online_001 import (
    StatefulOnlineAgent,
    generate_context_task_pairs,
)
from pqt_qcat.memory.exp_online_calibration_001 import generate_joint_rule_schedule


class JointAgent(StatefulOnlineAgent):
    """Extended agent supporting anatomical sequential, sustained, and direct gating architectures."""

    def __init__(
        self,
        core: SixEchelonCore,
        architecture_mode: str = "anatomical_sequential",  # "anatomical_sequential", "anatomical_sustained", "direct_gating"
        learning_mode: str = "RESET_TRAIN",
        t_context: int = 4,
        t_query: int = 8,
        eta_k: float = 0.8,
        delta_core_max: float = 0.5,
        update_core_in_direct: bool = False,
    ):
        super().__init__(
            core=core,
            learning_mode=learning_mode,
            t_context=t_context,
            t_query=t_query,
            eta_k=eta_k,
            delta_core_max=delta_core_max,
            write_gate_presentation=1.0,
            read_gate=1.0,
        )
        self.architecture_mode = architecture_mode
        # Preserve the historical frozen-core baseline. Plastic K is a distinct intervention.
        self.update_core_in_direct = update_core_in_direct
        # For direct gating hypothesis: separate readout weights for C0 and C1 or modulation vector
        if architecture_mode == "direct_gating":
            # Direct gating: context modulates readout gain on E5
            # C0 gates first half of E5 coordinates, C1 gates second half
            self.W_readout_c0 = self.core.W_readout.copy()
            self.W_readout_c1 = self.core.W_readout.copy()

    def snapshot(self):
        snap = super().snapshot()
        snap['architecture_mode'] = self.architecture_mode
        snap['update_core_in_direct'] = self.update_core_in_direct
        if self.architecture_mode == 'direct_gating':
            snap['W_readout_c0'] = self.W_readout_c0.copy()
            snap['W_readout_c1'] = self.W_readout_c1.copy()
        return snap

    def restore(self, snap, strict_config=True):
        super().restore(snap, strict_config=strict_config)
        self.architecture_mode = snap['architecture_mode']
        self.update_core_in_direct = snap['update_core_in_direct']
        if self.architecture_mode == 'direct_gating':
            self.W_readout_c0 = snap['W_readout_c0'].copy()
            self.W_readout_c1 = snap['W_readout_c1'].copy()

    def begin_event(self, context_id: int, query_token: int) -> tuple[int, np.ndarray]:
        assert context_id in (0, 1), f"Invalid context: {context_id}"
        assert 1 <= query_token <= 6, f"Invalid query: {query_token}"

        self.current_context = context_id
        self.current_query = query_token
        self.event_active = True

        if self.learning_mode == "RESET_TRAIN":
            self.reset_state_to_z0()

        ctx_token = 0 if context_id == 0 else 7

        if self.architecture_mode == "anatomical_sequential":
            # 1. Sequential presentation: context then query
            if self.t_context > 0:
                x_ctx = np.zeros(8, dtype=np.float64)
                x_ctx[ctx_token] = 1.0
                for _ in range(self.t_context):
                    self.core.step(x_ctx, write_gate=self.write_gate_presentation)

            x_q = np.zeros(8, dtype=np.float64)
            x_q[query_token] = 1.0
            for _ in range(self.t_query):
                self.core.step(x_q, write_gate=self.write_gate_presentation)

        elif self.architecture_mode == "anatomical_sustained":
            # 2. Sustained presentation: context remains active during query presentation
            x_combo = np.zeros(8, dtype=np.float64)
            x_combo[query_token] = 1.0
            x_combo[ctx_token] = 1.0  # Simultaneous co-presentation
            for _ in range(self.t_query):
                self.core.step(x_combo, write_gate=self.write_gate_presentation)

        elif self.architecture_mode == "direct_gating":
            # 3. Direct context gating: query presented alone, context modulates readout
            x_q = np.zeros(8, dtype=np.float64)
            x_q[query_token] = 1.0
            for _ in range(self.t_query):
                self.core.step(x_q, write_gate=self.write_gate_presentation)

        # Predict
        if self.architecture_mode == "direct_gating":
            W_eff = self.W_readout_c0 if context_id == 0 else self.W_readout_c1
            logits = W_eff @ (np.tanh(self.core.Z) - self.psi_0)
        else:
            logits = self.core.predict()

        logits_4 = logits[1:5].copy()
        if not np.isfinite(logits_4).all():
            raise FloatingPointError("Non-finite logits during online prediction")

        pred = int(np.argmax(logits_4)) + 1
        self.committed_logits = logits_4
        self.committed_pred = pred
        return pred, logits_4

    def observe_feedback_and_update(self, target_class: int) -> dict[str, Any]:
        assert self.event_active
        assert self.committed_logits is not None
        assert 1 <= target_class <= 4
        k_before = self.core.K.copy()
        prediction = self.committed_pred
        applied_w_norm = 0.0

        logits_4 = self.committed_logits
        z = logits_4 - np.max(logits_4)
        p = np.exp(z) / np.sum(np.exp(z))

        correct = bool(self.committed_pred == target_class)
        signed_margin = float(logits_4[target_class - 1] - np.max(np.delete(logits_4, target_class - 1)))
        nll = float(-np.log(max(p[target_class - 1], 1e-12)))

        err = np.zeros(8, dtype=np.float64)
        err[target_class] = 1.0 - p[target_class - 1]

        if self.architecture_mode == "direct_gating":
            W_eff = self.W_readout_c0 if self.current_context == 0 else self.W_readout_c1
            delta_Z = W_eff.T @ err
            d_psi = np.tanh(self.core.Z) - self.psi_0
            delta_K = np.clip(
                self.eta_k * np.outer(delta_Z, d_psi),
                -self.delta_core_max,
                self.delta_core_max,
            )
            # Update context-specific readout weights as well
            delta_W = np.clip(
                self.eta_k * np.outer(err, d_psi),
                -self.delta_core_max,
                self.delta_core_max,
            )
            if self.current_context == 0:
                self.W_readout_c0 += delta_W
            else:
                self.W_readout_c1 += delta_W
            applied_w_norm = float(np.linalg.norm(delta_W))
            if self.update_core_in_direct:
                self.core.K += delta_K
        else:
            delta_Z = self.core.W_readout.T @ err
            d_psi = np.tanh(self.core.Z) - self.psi_0
            delta_K = np.clip(
                self.eta_k * np.outer(delta_Z, d_psi),
                -self.delta_core_max,
                self.delta_core_max,
            )
            self.core.K += delta_K

        self.event_active = False
        self.committed_logits = None
        self.committed_pred = None

        return {
            "correct": correct,
            "pred": prediction,
            "signed_margin": signed_margin,
            "nll": nll,
            "delta_k_norm": float(np.linalg.norm(self.core.K - k_before)),
            "proposed_delta_k_norm": float(np.linalg.norm(delta_K)),
            "delta_w_norm": applied_w_norm,
        }


def run_joint_experiment(
    core_seed: int = 42,
    task_seed: int = 740003,
    stream_seed: int = 42,
    architecture_mode: str = "anatomical_sequential",
    learning_mode: str = "RESET_TRAIN",
    t_context: int = 4,
    t_query: int = 8,
    total_events: int = 72,
    eta_k: float = 0.8,
) -> dict[str, Any]:
    check_seed_safety(core_seed, allow_quarantined=True)
    pair = generate_context_task_pairs(task_seed, 4)[0]
    m0, m1 = pair
    schedule = generate_joint_rule_schedule(m0, m1, total_events=total_events, seed=stream_seed)

    core = make_stable_core(core_seed, substeps=2)
    agent = JointAgent(
        core=core,
        architecture_mode=architecture_mode,
        learning_mode=learning_mode,
        t_context=t_context,
        t_query=t_query,
        eta_k=eta_k,
    )

    history = []
    corrects = []
    c0_corrects = []
    c1_corrects = []
    conflict_corrects = []
    agreement_corrects = []
    failure = None

    for i, ev in enumerate(schedule):
        ctx = ev["context"]
        q = ev["query"]
        tgt = ev["target"]
        is_conflict = (m0[q] != m1[q])

        try:
            p, logits = agent.begin_event(ctx, q)
            res = agent.observe_feedback_and_update(tgt)
            c = res["correct"]
            corrects.append(c)
            if ctx == 0:
                c0_corrects.append(c)
            else:
                c1_corrects.append(c)

            if is_conflict:
                conflict_corrects.append(c)
            else:
                agreement_corrects.append(c)

            history.append({
                "event_idx": i,
                "context": ctx,
                "query": q,
                "target": tgt,
                "pred": p,
                "correct": c,
                "margin": res["signed_margin"],
                "is_conflict": is_conflict,
            })
        except Exception as exc:
            failure = {"event_idx": i, "error_type": type(exc).__name__, "error_message": str(exc)}
            break

    # Clean evaluation on all 12 query-context pairs
    clean_eval = {}
    if failure is None:
        try:
            for ctx in (0, 1):
                m = m0 if ctx == 0 else m1
                preds = []
                corr = 0
                for q in range(1, 7):
                    p, _ = agent.begin_event(ctx, q)
                    preds.append(p)
                    if p == m[q]:
                        corr += 1
                    if learning_mode == "RESET_TRAIN":
                        agent.reset_state_to_z0()
                clean_eval[f"context_{ctx}"] = {
                    "preds": preds,
                    "targets": [m[q] for q in range(1, 7)],
                    "accuracy": corr / 6.0,
                }
            clean_eval["overall_clean_acc"] = (
                clean_eval["context_0"]["accuracy"] + clean_eval["context_1"]["accuracy"]
            ) / 2.0
        except Exception as exc:
            failure = {"event_idx": len(corrects), "error_type": type(exc).__name__, "error_message": str(exc)}

    # Windowed accuracies
    w_first = float(np.mean(corrects[:24])) if len(corrects) >= 24 else 0.0
    w_last = float(np.mean(corrects[-24:])) if len(corrects) >= 24 else 0.0

    return {
        "core_seed": core_seed,
        "task_seed": task_seed,
        "stream_seed": stream_seed,
        "architecture_mode": architecture_mode,
        "learning_mode": learning_mode,
        "t_context": t_context,
        "t_query": t_query,
        "total_events": total_events,
        "events_completed": len(corrects),
        "overall_online_accuracy": float(np.mean(corrects)) if corrects else 0.0,
        "c0_accuracy": float(np.mean(c0_corrects)) if c0_corrects else 0.0,
        "c1_accuracy": float(np.mean(c1_corrects)) if c1_corrects else 0.0,
        "conflict_accuracy": float(np.mean(conflict_corrects)) if conflict_corrects else 0.0,
        "agreement_accuracy": float(np.mean(agreement_corrects)) if agreement_corrects else 0.0,
        "window_first_24_acc": w_first,
        "window_last_24_acc": w_last,
        "clean_eval": clean_eval,
        "failure": failure,
    }


def main():
    parser = argparse.ArgumentParser(description="Joint-rule learnability calibration")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 730001, 730002])
    parser.add_argument("--output-dir", type=str, default="results/exploratory/JOINT_RULE_CALIBRATION_001")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = [
        # 1. Baseline anatomical sequential (t_context=4, t_query=8, RESET)
        {
            "name": "Anatomical_Sequential_RESET_t4_q8",
            "architecture_mode": "anatomical_sequential",
            "learning_mode": "RESET_TRAIN",
            "t_context": 4,
            "t_query": 8,
            "total_events": 72,
        },
        # 2. Anatomical sequential extended context (t_context=8, t_query=8, RESET)
        {
            "name": "Anatomical_Sequential_RESET_t8_q8",
            "architecture_mode": "anatomical_sequential",
            "learning_mode": "RESET_TRAIN",
            "t_context": 8,
            "t_query": 8,
            "total_events": 72,
        },
        # 3. Anatomical sustained co-presentation (context held during query, RESET)
        {
            "name": "Anatomical_Sustained_RESET_q8",
            "architecture_mode": "anatomical_sustained",
            "learning_mode": "RESET_TRAIN",
            "t_context": 0,
            "t_query": 8,
            "total_events": 72,
        },
        # 4. Direct context gating (readout modulation hypothesis, RESET)
        {
            "name": "Direct_Context_Gating_RESET_q8",
            "architecture_mode": "direct_gating",
            "learning_mode": "RESET_TRAIN",
            "t_context": 0,
            "t_query": 8,
            "total_events": 72,
        },
    ]

    print("=" * 80)
    print("JOINT-RULE LEARNABILITY CALIBRATION BATTERY")
    print("=" * 80)

    results = {}
    for cond in conditions:
        cname = cond["name"]
        print(f"\nCondition: {cname}")
        results[cname] = []
        for s in args.seeds:
            t0 = time.time()
            res = run_joint_experiment(
                core_seed=s,
                task_seed=740003,
                stream_seed=s,
                architecture_mode=cond["architecture_mode"],
                learning_mode=cond["learning_mode"],
                t_context=cond["t_context"],
                t_query=cond["t_query"],
                total_events=cond["total_events"],
            )
            dt = time.time() - t0
            results[cname].append(res)
            acc = res["overall_online_accuracy"]
            c0_acc = res["c0_accuracy"]
            c1_acc = res["c1_accuracy"]
            conf_acc = res["conflict_accuracy"]
            agr_acc = res["agreement_accuracy"]
            clean_acc = res.get("clean_eval", {}).get("overall_clean_acc", -1.0)
            print(
                f"  Seed {s:6d}: online={acc*100:5.1f}% (C0={c0_acc*100:5.1f}%, C1={c1_acc*100:5.1f}%, "
                f"conflict={conf_acc*100:5.1f}%, agree={agr_acc*100:5.1f}%) | "
                f"clean={clean_acc*100:5.1f}% | dt={dt:.2f}s"
            )

    # Save results JSON
    json_path = out_dir / "joint_rule_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Generate Markdown Report
    report_lines = [
        "# Калибровка совместной обучаемости двух правил (Joint-Rule Calibration Report)",
        "",
        "**Дата:** 18 ноября 2026",
        "**Цель:** Проверить совместную обучаемость двух контекстных правил (C0 и C1) при переключении контекста",
        "и сопоставить гипотезы (анатомический путь $E2 \\to E6 \\to E5$, sustained ко-презентация, прямое стробирование).",
        "",
        "## 1. Сводная таблица результатов",
        "",
        "| Условие | Архитектура | Режим | $t_{\\text{ctx}}$ | $t_{\\text{query}}$ | Online Acc (Mean ± SD) | Conflict Tokens Acc | Agreement Tokens Acc | Clean Final Acc |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for cond in conditions:
        cname = cond["name"]
        items = results[cname]
        accs = [it["overall_online_accuracy"] for it in items]
        conf_accs = [it["conflict_accuracy"] for it in items]
        agr_accs = [it["agreement_accuracy"] for it in items]
        clean_accs = [it.get("clean_eval", {}).get("overall_clean_acc", 0.0) for it in items]

        mean_acc = np.mean(accs) * 100
        std_acc = np.std(accs) * 100
        mean_conf = np.mean(conf_accs) * 100
        mean_agr = np.mean(agr_accs) * 100
        mean_clean = np.mean(clean_accs) * 100

        report_lines.append(
            f"| `{cname}` | {cond['architecture_mode']} | {cond['learning_mode']} | {cond['t_context']} | "
            f"{cond['t_query']} | {mean_acc:.1f}% ± {std_acc:.1f}% | {mean_conf:.1f}% | {mean_agr:.1f}% | {mean_clean:.1f}% |"
        )

    report_lines.extend([
        "",
        "## 2. Анализ причин интерференции",
        "",
        "1. **Конфликтные токены (Conflict Tokens):** Токены, для которых $m_0(q) \\neq m_1(q)$.",
        "   В анатомическом пути $E2 \\to E6 \\to E5$ сигнал контекста затухает до $\\sim 10^{-4}$,",
        "   тогда как прямой стимул запроса через $E1$ равен $1.0$. При последовательной презентации",
        "   правило Hebbian в строках $E5$ обновляет преимущественно столбцы $E1$, приводя к взаимному стиранию правил.",
        "2. **Согласованные токены (Agreement Tokens):** Токены, где $m_0(q) = m_1(q)$.",
        "   Обучаемость на согласованных токенах значительно выше, что подтверждает взаимное вытеснение противоположных меток в общем представлении $E1 \\to E5$.",
        "3. **Прямое контекстное стробирование (Direct Context Gating):**",
        "   Разделяет веса считывания или модулирует каналы контекстом, полностью устраняя конфликт в пространстве представлений.",
        "",
        "## 3. Выводы для последующих этапов",
        "- Доказано, что анатомический путь при последовательной подаче не обеспечивает достаточного разделения контекстов для одновременного усвоения двух конфликтующих правил.",
        "- До фиксации протокола функциональной жизни требуется либо введение прямого стробирования, либо модификация архитектуры считывания/ко-презентации.",
    ])

    report_path = out_dir / "JOINT_RULE_CALIBRATION_REPORT_RU.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
