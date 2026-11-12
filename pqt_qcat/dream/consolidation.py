"""Gated Consolidation and Policy Ablation Engine (ARCHITECTURE_LIFE_002 Section 5.3).

Implements:
1. Three-Gate Validation for each candidate:
   - ProvenanceConsistencyGate (provenance in B_real, non-contradictory)
   - StructuralCheckGate (finite, valid token ranges)
   - AnchorGuardGate (shadow test on real anchors, rollback if degradation > threshold)
2. Safe orthogonal / Maya-weighted update on core parameters theta_core = {K, B}
3. Full spectrum of pre-registered dream policies:
   - 'no_replay': 1000 idle ticks with zero replay or updates
   - 'state_only': internal dynamic state evolves, core parameters frozen
   - 'naive_replay': samples from B_real without anchor protection
   - 'guarded_replay': replay from B_real with anchor protection (no counterfactuals)
   - 'noA_dream': counterfactual generation with A=0 and three-gate consolidation
   - 'PQT_dream': counterfactual generation with A=A_I and three-gate consolidation
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..core.six_echelon import SixEchelonCore
from ..memory.witness import (
    WitnessLedger,
    DreamHypothesis,
    ProvenanceConsistencyGate,
    StructuralCheckGate,
    AnchorGuardGate,
)
from .generator import DreamGenerator, DreamGeneratorConfig


@dataclass
class ConsolidationConfig:
    """Configuration for dream consolidation."""
    eta_dream: float = 0.05             # Learning rate for dream consolidation
    anchor_max_degradation: float = 0.02 # Strict anchor protection threshold (<= 2%)
    dream_cycles: int = 25              # Number of dream exploration cycles
    target_matrix: str = "B"            # Target matrix for plasticity: 'B' (sensory) or 'K' (inter-cell coupling)


def _get_quiescent_psi() -> np.ndarray:
    """Quiescent state tanh(Z) where w0 = 1.0 for all 6 cells and other coords are 0."""
    z_0 = np.zeros(48, dtype=np.float64)
    for c in range(6):
        z_0[8 * c + 6] = 1.0
    return np.tanh(z_0)


class DreamConsolidationEngine:
    """Manages the full pipeline of generation, validation, shadow testing, and consolidation."""

    def __init__(
        self,
        config: ConsolidationConfig | None = None,
        seed: int = 42,
    ):
        self.config = config or ConsolidationConfig()
        self.seed = seed
        self.provenance_gate = ProvenanceConsistencyGate()
        self.structural_gate = StructuralCheckGate()
        self.anchor_guard = AnchorGuardGate(max_degradation=self.config.anchor_max_degradation)
        self.generator = DreamGenerator(seed=seed)

    def run_sleep_policy(
        self,
        policy: str,
        core: SixEchelonCore,
        ledger: WitnessLedger,
        idle_ticks: int = 1000,
    ) -> dict:
        """Run an autonomous sleep session under a specified policy.
        
        Parameters
        ----------
        policy : str
            One of {'no_replay', 'state_only', 'naive_replay', 'guarded_replay', 'noA_dream', 'PQT_dream'}.
        core : SixEchelonCore
            The agent's 48D core.
        ledger : WitnessLedger
            Audit ledger containing B_real observations and anchors.
        idle_ticks : int
            Total physical idle duration (default 1000 ticks per protocol H3-SLEEP-SAFE).
            
        Returns
        -------
        dict
            Execution diagnostics including anchor retention, accepted/rejected updates, and core drift.
        """
        anchors = ledger.get_anchors()
        acc_before = self.anchor_guard.evaluate_anchors(core, anchors)

        initial_K = core.K.copy()
        initial_B = core.B.copy()

        accepted_count = 0
        rejected_count = 0
        rejection_reasons: list[str] = []

        is_k_consolidation = bool(
            self.config.target_matrix == "K"
            or policy.endswith("_K")
        )

        if policy == "no_replay":
            # Pure idle ticks with write_gate=0.0: baseline passive forgetting
            x_neutral = np.zeros(core.config.d_in)
            x_neutral[0] = 1.0
            for _ in range(idle_ticks):
                core.step(x_neutral, write_gate=0.0)

        elif policy == "state_only":
            # Dynamic state runs with thermal noise, but core parameters are untouched
            rng = np.random.default_rng(self.seed)
            for _ in range(idle_ticks):
                noise = rng.normal(0.0, 0.05, size=48) * np.sqrt(core.config.h_base)
                core.Z += noise
                core.step(x_t=None, write_gate=0.0)

        elif policy == "naive_replay":
            # Replay observed events without anchor guard protection (negative control)
            rng = np.random.default_rng(self.seed)
            real_events = ledger.real_events
            if real_events:
                for _ in range(self.config.dream_cycles):
                    ev = rng.choice(real_events)
                    # Naive gradient update without gates
                    if is_k_consolidation:
                        delta_K = np.zeros_like(core.K)
                        src_idx = ev.input_token - 1 if ev.input_token > 0 else 0
                        tgt_idx = 31 + ev.target_token
                        delta_K[tgt_idx, src_idx] = self.config.eta_dream
                        core.K += delta_K
                    else:
                        delta_B = np.zeros_like(core.B)
                        delta_B[31 + ev.target_token, ev.input_token] = self.config.eta_dream
                        core.B += delta_B
                    accepted_count += 1
            # Remaining idle ticks
            for _ in range(max(0, idle_ticks - self.config.dream_cycles * 10)):
                core.step(x_t=None, write_gate=0.0)

        elif policy in ("guarded_replay", "guarded_replay_K"):
            # Replay exclusively real events from B_real with anchor protection
            rng = np.random.default_rng(self.seed)
            real_events = ledger.real_events
            if real_events:
                for _ in range(self.config.dream_cycles):
                    ev = rng.choice(real_events)
                    if is_k_consolidation:
                        delta_K = np.zeros_like(core.K)
                        src_idx = ev.input_token - 1 if ev.input_token > 0 else 0
                        tgt_idx = 31 + ev.target_token
                        delta_K[tgt_idx, src_idx] = self.config.eta_dream * 0.5
                        passed, _, _ = self.anchor_guard.check_and_apply(
                            core, anchors, proposed_delta_K=delta_K, proposed_delta_B=np.zeros_like(core.B)
                        )
                    else:
                        delta_B = np.zeros_like(core.B)
                        delta_B[31 + ev.target_token, ev.input_token] = self.config.eta_dream * 0.5
                        passed, _, _ = self.anchor_guard.check_and_apply(
                            core, anchors, proposed_delta_K=np.zeros_like(core.K), proposed_delta_B=delta_B
                        )
                    if passed:
                        accepted_count += 1
                    else:
                        rejected_count += 1
                        rejection_reasons.append("anchor_degradation")
            for _ in range(max(0, idle_ticks - self.config.dream_cycles * 10)):
                core.step(x_t=None, write_gate=0.0)

        elif policy in ("noA_dream", "PQT_dream", "noA_dream_K", "PQT_dream_K"):
            # Counterfactual Dream Generation & Gated Consolidation
            # Set core vortex according to policy
            saved_vortex = core.config.enable_vortex
            core.config.enable_vortex = bool("PQT" in policy)

            # Generate counterfactual candidates via symmetric generator
            candidates = self.generator.generate_candidate_hypotheses(
                core, ledger, cycle_count=self.config.dream_cycles
            )

            # Process candidates through the 3 gates
            for cand in candidates:
                # Gate 1: Provenance and consistency
                if not self.provenance_gate.check(cand, ledger):
                    rejected_count += 1
                    rejection_reasons.append("provenance_or_contradiction")
                    continue

                # Gate 2: Structural checks
                if not self.structural_gate.check(cand):
                    rejected_count += 1
                    rejection_reasons.append("structural_bounds")
                    continue

                # Propose candidate parameter adjustment weighted by w_Maya(C)
                update_scale = self.config.eta_dream * cand.maya_weight
                if is_k_consolidation:
                    delta_K = np.zeros_like(core.K)
                    src_idx = cand.input_token - 1 if cand.input_token > 0 else 0
                    tgt_idx = 31 + cand.pred_token
                    delta_K[tgt_idx, src_idx] = update_scale
                    passed, _, _ = self.anchor_guard.check_and_apply(
                        core, anchors, proposed_delta_K=delta_K, proposed_delta_B=np.zeros_like(core.B)
                    )
                else:
                    delta_B = np.zeros_like(core.B)
                    delta_B[31 + cand.pred_token, cand.input_token] = update_scale
                    passed, _, _ = self.anchor_guard.check_and_apply(
                        core, anchors, proposed_delta_K=np.zeros_like(core.K), proposed_delta_B=delta_B
                    )

                if passed:
                    accepted_count += 1
                else:
                    rejected_count += 1
                    rejection_reasons.append("anchor_guard_rollback")

            core.config.enable_vortex = saved_vortex
            # Remaining idle ticks
            for _ in range(max(0, idle_ticks - self.config.dream_cycles * 10)):
                core.step(x_t=None, write_gate=0.0)

        else:
            raise ValueError(f"Unknown dream sleep policy: {policy}")

        acc_after = self.anchor_guard.evaluate_anchors(core, anchors)
        degradation = acc_before - acc_after
        delta_K_norm = float(np.linalg.norm(core.K - initial_K))
        delta_B_norm = float(np.linalg.norm(core.B - initial_B))

        return {
            "policy": policy,
            "anchor_accuracy_before": acc_before,
            "anchor_accuracy_after": acc_after,
            "anchor_degradation": degradation,
            "passed_h3_sleep_safe": bool(degradation <= self.config.anchor_max_degradation),
            "accepted_updates": accepted_count,
            "rejected_updates": rejected_count,
            "rejection_reasons": rejection_reasons,
            "delta_K_norm": delta_K_norm,
            "delta_B_norm": delta_B_norm,
        }
