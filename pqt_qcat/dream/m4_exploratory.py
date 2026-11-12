"""M4-FIX-003 Exploratory Experiments for LIFE-002.

Protocol Reference: prereg/EXPLORATORY_EXPERIMENTS_M4.yaml
Implements:
1. EXP-M4-K-CONSOLIDATION: Consolidation into structural matrix K with sensory B 100% frozen.
2. EXP-M4-QCTX-CAUSAL-T5: Causal role of endogenous context (E5/E6) in multi-hop composition.
3. EXP-M4-T5-COMPOSITIONAL-GENERALIZATION: Comparison of consolidation targets (B vs K) across PQT and noA.
"""
from __future__ import annotations
import numpy as np

from ..core.six_echelon import SixEchelonCore, SixEchelonConfig
from ..memory.witness import WitnessLedger, ObservedEvent
from .consolidation import DreamConsolidationEngine, ConsolidationConfig
from .counterfactual_env import PartiallyObservedGraphEnv, TransitionEdge

CLASSES = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [-1.0, -1.0, 0.0],
], dtype=np.float64)


def benchmark_exp_m4_k_consolidation(seed: int = 42) -> dict:
    """EXP-M4-K-CONSOLIDATION: Test consolidation purely into K with sensory B frozen."""
    policies = ["guarded_replay_K", "noA_dream_K", "PQT_dream_K"]
    engine = DreamConsolidationEngine(
        ConsolidationConfig(anchor_max_degradation=0.02, target_matrix="K"),
        seed=seed,
    )

    results = {}
    for pol in policies:
        cfg = SixEchelonConfig(enable_vortex=bool("PQT" in pol), integrator_mode="fixed")
        core = SixEchelonCore(config=cfg, seed=seed)
        core.W_readout.fill(0.0)
        for c in range(1, 5):
            core.W_readout[c, 32:35] = CLASSES[c]

        # Initialize sensory projection B (anchors) and freeze it
        for k in range(1, 5):
            core.B[31 + k, k] = 4.0
        initial_B = core.B.copy()
        initial_K = core.K.copy()

        ledger = WitnessLedger()
        for k in range(1, 5):
            ledger.record_observation(
                timestamp=k,
                input_token=k,
                target_token=k,
                pred_token=k,
                confidence=0.98,
                context_basin=1,
            )

        res = engine.run_sleep_policy(policy=pol, core=core, ledger=ledger, idle_ticks=1000)
        ledger.assert_no_unverified_promotions()

        # Strict invariant: delta B must be exactly 0.0
        delta_B_norm = float(np.linalg.norm(core.B - initial_B))
        assert delta_B_norm == 0.0, f"Sensory B was modified during K consolidation: {delta_B_norm}"

        res["delta_B_strictly_zero"] = bool(delta_B_norm == 0.0)
        results[pol] = res

    return results


def benchmark_exp_m4_qctx_causal_t5(seed: int = 42) -> dict:
    """EXP-M4-QCTX-CAUSAL-T5: Evaluate causal contribution of endogenous context E5/E6 on hidden compositions."""
    results = {}
    for env_name in ("relational_chain", "branching_non_bijective"):
        env = PartiallyObservedGraphEnv(env_name, seed=seed)
        base_edges = env.get_training_edges()
        hidden_edges = env.get_hidden_test_edges()

        cfg = SixEchelonConfig(enable_vortex=True, integrator_mode="fixed")
        core = SixEchelonCore(config=cfg, seed=seed)
        core.W_readout.fill(0.0)
        for c in range(1, 5):
            core.W_readout[c, 32:35] = CLASSES[c]

        # Train base transitions
        for edge in base_edges:
            u, v = edge.source_token, edge.target_token
            core.B[31 + v, u] = 4.0
            core.K[31 + v, 31 + u] = 0.6
            # Context coupling into E5 (32:40)
            core.K[32 + (v % 8), 32 + (u % 8)] = 0.5

        def evaluate_hidden(intervention: str) -> float:
            corr = 0
            rng = np.random.default_rng(seed + 100)
            for edge in hidden_edges:
                core.Z.fill(0.0)
                for c in range(6):
                    core.Z[8 * c + 6] = 1.0
                x = np.zeros(core.config.d_in)
                x[edge.source_token] = 1.0

                for step_idx in range(8):
                    core.step(x, write_gate=1.0)
                    if intervention == "zeroed_ctx":
                        # Clamp E5 and E6 (context cells, idx 32:48) to zero
                        core.Z[32:48] = 0.0
                    elif intervention == "scrambled_ctx":
                        # Scramble E5 and E6 coordinates
                        core.Z[32:48] = rng.permutation(core.Z[32:48])

                logits = core.predict()
                pred = int(np.argmax(logits[1:5])) + 1
                if pred == edge.target_token:
                    corr += 1
            return corr / len(hidden_edges) if hidden_edges else 0.0

        acc_intact = evaluate_hidden("intact")
        acc_zeroed = evaluate_hidden("zeroed_ctx")
        acc_scrambled = evaluate_hidden("scrambled_ctx")

        results[env_name] = {
            "accuracy_intact": acc_intact,
            "accuracy_zeroed_ctx": acc_zeroed,
            "accuracy_scrambled_ctx": acc_scrambled,
            "causal_drop_zeroed": acc_intact - acc_zeroed,
            "causal_drop_scrambled": acc_intact - acc_scrambled,
            "context_has_causal_impact": bool((acc_intact - acc_zeroed) != 0.0 or (acc_intact - acc_scrambled) != 0.0),
        }

    return results


def benchmark_exp_m4_t5_compositional_generalization(seed: int = 42) -> dict:
    """EXP-M4-T5-COMPOSITIONAL-GENERALIZATION: Compare B vs K consolidation on multi-hop compositions."""
    results = {}
    for env_name in ("relational_chain", "branching_non_bijective"):
        env = PartiallyObservedGraphEnv(env_name, seed=seed)
        base_edges = env.get_training_edges()
        hidden_edges = env.get_hidden_test_edges()
        negative_edges = env.get_negative_edges()

        policy_results = {}
        for target in ("B", "K"):
            for pol_prefix in ("guarded_replay", "noA_dream", "PQT_dream"):
                pol_name = f"{pol_prefix}_{target}" if target == "K" else pol_prefix
                enable_vortex = bool("PQT" in pol_name)

                cfg = SixEchelonConfig(enable_vortex=enable_vortex, integrator_mode="fixed")
                core = SixEchelonCore(config=cfg, seed=seed)
                core.W_readout.fill(0.0)
                for c in range(1, 5):
                    core.W_readout[c, 32:35] = CLASSES[c]

                ledger = WitnessLedger()
                for t_step, edge in enumerate(base_edges):
                    u, v = edge.source_token, edge.target_token
                    core.B[31 + v, u] = 4.0
                    core.K[31 + v, 31 + u] = 0.5
                    ledger.record_observation(
                        timestamp=t_step + 1,
                        input_token=u,
                        target_token=v,
                        pred_token=v,
                        confidence=0.98,
                    )

                engine = DreamConsolidationEngine(
                    ConsolidationConfig(anchor_max_degradation=0.02, target_matrix=target),
                    seed=seed,
                )
                sleep_diag = engine.run_sleep_policy(pol_name, core, ledger, idle_ticks=300)
                ledger.assert_no_unverified_promotions()

                # Evaluate hidden test edges post-sleep
                corr = 0
                for edge in hidden_edges:
                    core.Z.fill(0.0)
                    for c in range(6):
                        core.Z[8 * c + 6] = 1.0
                    x = np.zeros(core.config.d_in)
                    x[edge.source_token] = 1.0
                    for _ in range(8):
                        core.step(x, write_gate=1.0)
                    logits = core.predict()
                    pred = int(np.argmax(logits[1:5])) + 1
                    if pred == edge.target_token:
                        corr += 1
                comp_acc = corr / len(hidden_edges) if hidden_edges else 0.0

                # Evaluate false positives on distractor negative edges
                fp = 0
                for edge in negative_edges:
                    core.Z.fill(0.0)
                    for c in range(6):
                        core.Z[8 * c + 6] = 1.0
                    x = np.zeros(core.config.d_in)
                    x[edge.source_token] = 1.0
                    for _ in range(8):
                        core.step(x, write_gate=1.0)
                    logits = core.predict()
                    pred = int(np.argmax(logits[1:5])) + 1
                    if pred == edge.target_token:
                        fp += 1
                fp_rate = fp / len(negative_edges) if negative_edges else 0.0

                policy_results[pol_name] = {
                    "target_matrix": target,
                    "comp_acc": comp_acc,
                    "fp_rate": fp_rate,
                    "accepted_updates": sleep_diag["accepted_updates"],
                    "rejected_updates": sleep_diag["rejected_updates"],
                    "delta_K_norm": sleep_diag["delta_K_norm"],
                    "delta_B_norm": sleep_diag["delta_B_norm"],
                    "anchor_degradation": sleep_diag["anchor_degradation"],
                }

        results[env_name] = policy_results

    return results
