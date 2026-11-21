"""Bounded Sidecar (VIMARSA) deliberation pool for LIFE-003."""
from __future__ import annotations
import copy
import hashlib
import numpy as np
from typing import Any

from pqt_qcat.core.six_echelon import SixEchelonCore, SixEchelonConfig


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

    def run_deliberation(
        self,
        main_core: SixEchelonCore,
        stimulus: np.ndarray | None,
        rng_sidecar: np.random.Generator,
        num_clones: int = 1,
    ) -> dict[str, Any]:
        """Spawn bounded Z0-64 clones, evaluate candidate action, verify MAIN invariance.
        
        CRITICAL: main_core.Z, main_core.K, and main_core.rng must be bit-exact unmutated!
        """
        clones_to_run = min(max(1, num_clones), self.max_sidecars_per_event)
        
        # Capture pre-state hashes of MAIN
        z_pre = main_core.Z.copy()
        k_pre = main_core.K.copy()
        rng_pre = copy.deepcopy(main_core.rng.bit_generator.state)

        # Create temporary Z0-64 clone
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
        # sidecar starts from clean Z0 (stable latch configuration)
        sidecar_core._init_state()

        # Run rollout ticks
        ticks_run = 0
        for _ in range(self.sidecar_rollout_ticks):
            sidecar_core.step(stimulus, write_gate=0.0)
            ticks_run += 1
            if ticks_run >= self.max_deliberation_ticks_per_tick:
                break

        # Read out logits over classes 1..4
        all_logits = sidecar_core.predict()
        sidecar_logits = all_logits[1:5].copy()
        shifted = sidecar_logits - np.max(sidecar_logits)
        probs = np.exp(shifted) / np.sum(np.exp(shifted))
        pred_action = int(np.argmax(sidecar_logits)) + 1
        entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)))) / float(np.log(4))

        # Check invariance of MAIN
        np.testing.assert_array_equal(main_core.Z, z_pre)
        np.testing.assert_array_equal(main_core.K, k_pre)
        assert main_core.rng.bit_generator.state == rng_pre

        # Destroy clone explicitly
        del sidecar_core

        return {
            "sidecars_spawned": clones_to_run,
            "rollout_ticks": ticks_run,
            "sidecar_prediction": pred_action,
            "sidecar_logits": sidecar_logits.tolist(),
            "sidecar_probabilities": probs.tolist(),
            "sidecar_entropy": float(entropy),
            "main_unmutated": True,
        }
