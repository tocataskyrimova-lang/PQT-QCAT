"""Bounded Sidecar (VIMARSA) deliberation pool for LIFE-003.

Supports:
1. single_sidecar_reader: strictly single Z0-64 branch evaluation.
2. run_deliberation: multi-candidate branch deliberation where requested clone count
   strictly matches the actual number of executed branches.

Epistemic Status (EPISTEMIC-PATCH-001):
Sidecar deliberation compares internal neural prediction entropy across alternative
candidate rollouts (selection_criterion: MINIMAL_NEURAL_PREDICTION_ENTROPY).
This is an internal neural prediction selection, NOT a proven causal comparison
of actual future world consequences or outcomes.
"""
from __future__ import annotations
import copy
import hashlib
import numpy as np
from typing import Any

from pqt_qcat.core.six_echelon import SixEchelonCore


class SidecarPool:
    """Manages temporary isolated Z0-64 deliberation clones."""

    def __init__(
        self,
        max_sidecars_per_event: int = 3,
        sidecar_rollout_ticks: int = 64,
        max_deliberation_ticks_per_tick: int = 192,
    ):
        self.max_sidecars_per_event = int(max_sidecars_per_event)
        self.sidecar_rollout_ticks = int(sidecar_rollout_ticks)
        self.max_deliberation_ticks_per_tick = int(max_deliberation_ticks_per_tick)

    def single_sidecar_reader(
        self,
        main_core: SixEchelonCore,
        stimulus: np.ndarray | None,
        rng_sidecar: np.random.Generator,
        rollout_ticks: int | None = None,
    ) -> dict[str, Any]:
        """Explicitly run a single isolated Z0-64 sidecar branch.
        
        Guarantees that exactly 1 clone is spawned and evaluated.
        """
        ticks = min(
            rollout_ticks or self.sidecar_rollout_ticks,
            self.max_deliberation_ticks_per_tick,
        )

        z_pre = main_core.Z.copy()
        k_pre = main_core.K.copy()
        rng_pre = copy.deepcopy(main_core.rng.bit_generator.state)

        sidecar_core = SixEchelonCore(
            config=copy.deepcopy(main_core.config),
            cell_params=copy.deepcopy(main_core.cell_params),
            vortex_params=copy.deepcopy(main_core.vortex_params),
            seed=None,
        )
        sidecar_core.rng = rng_sidecar
        sidecar_core.K = k_pre.copy()
        sidecar_core.B = main_core.B.copy()
        sidecar_core.W_readout = main_core.W_readout.copy()
        sidecar_core._init_state()

        ticks_run = 0
        for _ in range(ticks):
            sidecar_core.step(stimulus, write_gate=0.0)
            ticks_run += 1

        all_logits = sidecar_core.predict()
        sidecar_logits = all_logits[1:5].copy()
        shifted = sidecar_logits - np.max(sidecar_logits)
        probs = np.exp(shifted) / np.sum(np.exp(shifted))
        pred_action = int(np.argmax(sidecar_logits)) + 1
        entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)))) / float(np.log(4))

        np.testing.assert_array_equal(main_core.Z, z_pre)
        np.testing.assert_array_equal(main_core.K, k_pre)
        assert main_core.rng.bit_generator.state == rng_pre

        del sidecar_core

        return {
            "mode": "single_sidecar_reader",
            "clones_requested": 1,
            "clones_spawned": 1,
            "clones_executed": 1,
            "rollout_ticks": ticks_run,
            "sidecar_prediction": pred_action,
            "sidecar_logits": sidecar_logits.tolist(),
            "sidecar_probabilities": probs.tolist(),
            "sidecar_entropy": float(entropy),
            "main_unmutated": True,
        }

    def run_deliberation(
        self,
        main_core: SixEchelonCore,
        stimulus: np.ndarray | None,
        rng_sidecar: np.random.Generator,
        num_clones: int = 1,
        candidate_actions: list[int] | None = None,
    ) -> dict[str, Any]:
        """Spawn bounded Z0-64 clones, evaluate candidate branches, verify MAIN invariance.
        
        CRITICAL FIX-2: clones_spawned and clones_executed strictly reflect the actual number
        of candidate branches executed in code. Never misrepresent requested count as executed count.
        """
        clones_to_run = min(max(1, int(num_clones)), self.max_sidecars_per_event)
        ticks_per_clone = min(
            self.sidecar_rollout_ticks,
            self.max_deliberation_ticks_per_tick // clones_to_run,
        )

        z_pre = main_core.Z.copy()
        k_pre = main_core.K.copy()
        rng_pre = copy.deepcopy(main_core.rng.bit_generator.state)

        # Default candidate motor actions if not explicitly supplied
        if candidate_actions is not None:
            actions_to_eval = candidate_actions[:clones_to_run]
        else:
            # When evaluating multiple candidates, evaluate distinct candidate actions [1, 2, ...]
            actions_to_eval = [((i % 4) + 1) for i in range(clones_to_run)]

        branches: list[dict[str, Any]] = []
        total_ticks_run = 0

        for idx, cand_action in enumerate(actions_to_eval):
            clone_core = SixEchelonCore(
                config=copy.deepcopy(main_core.config),
                cell_params=copy.deepcopy(main_core.cell_params),
                vortex_params=copy.deepcopy(main_core.vortex_params),
                seed=None,
            )
            clone_core.rng = rng_sidecar
            clone_core.K = k_pre.copy()
            clone_core.B = main_core.B.copy()
            clone_core.W_readout = main_core.W_readout.copy()
            clone_core._init_state()

            # Construct branch stimulus
            branch_stim = stimulus.copy() if stimulus is not None else np.zeros(8)
            if clones_to_run > 1:
                # Modulate candidate action into candidate branch context
                branch_stim = branch_stim.copy()
                branch_stim[cand_action] = 1.0

            ticks_this_branch = 0
            for _ in range(ticks_per_clone):
                clone_core.step(branch_stim, write_gate=0.0)
                ticks_this_branch += 1

            total_ticks_run += ticks_this_branch

            logits = clone_core.predict()[1:5].copy()
            shifted = logits - np.max(logits)
            probs = np.exp(shifted) / np.sum(np.exp(shifted))
            pred = int(np.argmax(logits)) + 1
            entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)))) / float(np.log(4))

            branches.append({
                "branch_index": idx,
                "candidate_action_tested": cand_action,
                "rollout_ticks": ticks_this_branch,
                "prediction": pred,
                "logits": logits.tolist(),
                "probabilities": probs.tolist(),
                "entropy": float(entropy),
            })

            del clone_core

        # Check invariance of MAIN
        np.testing.assert_array_equal(main_core.Z, z_pre)
        np.testing.assert_array_equal(main_core.K, k_pre)
        assert main_core.rng.bit_generator.state == rng_pre

        # Select best branch (lowest entropy / highest certainty prediction)
        best_branch = min(branches, key=lambda b: b["entropy"])

        # Determine divergence mechanism
        live_logits = main_core.predict()[1:5]
        live_pred = int(np.argmax(live_logits)) + 1
        if best_branch["prediction"] != live_pred:
            if clones_to_run > 1:
                mechanism = "MULTI_BRANCH_PREDICTION_ENTROPY_SELECTION"
            else:
                mechanism = "READOUT_RELAXATION_Z0_64"
        else:
            mechanism = "CONCORDANT_PREDICTION"

        return {
            "mode": "multi_candidate_deliberation" if clones_to_run > 1 else "single_sidecar_deliberation",
            "clones_requested": int(num_clones),
            "clones_spawned": len(branches),
            "clones_executed": len(branches),
            "sidecars_spawned": len(branches),
            "total_rollout_ticks": total_ticks_run,
            "branches": branches,
            "selected_branch": best_branch["branch_index"],
            "sidecar_prediction": best_branch["prediction"],
            "sidecar_logits": best_branch["logits"],
            "sidecar_probabilities": best_branch["probabilities"],
            "sidecar_entropy": best_branch["entropy"],
            "divergence_mechanism": mechanism,
            "selection_criterion": "MINIMAL_NEURAL_PREDICTION_ENTROPY",
            "epistemic_disclaimer": "Comparison of candidate neural prediction entropies; not a validated world outcome comparison.",
            "main_unmutated": True,
        }
