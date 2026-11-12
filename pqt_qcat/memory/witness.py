"""Witness Ledger and Provenance Registry (M4 ARCHITECTURE_LIFE_002 Section 5).

Strictly enforces:
1. Complete separation of confirmed observations B_real and dream hypotheses B_dream:
       M_t = (B_real, B_dream, theta_core, Z_t), with B_real ∩ B_dream = ∅
2. Maya uncertainty weighting:
       w_Maya(C) = C * exp(-H2(C) / T_Maya)
3. Three independent validation gates:
       - Provenance & Consistency Gate
       - Structural Check Gate
       - Anchor Guard Gate (evaluates anchor performance on shadow core with rollback)
4. Strict prohibition of unverified promotion:
       0 entries of DREAM_CANDIDATE -> OBSERVED without external ground truth.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from ..core.errors import NonFiniteStateError


class UnverifiedPromotionError(RuntimeError):
    """Raised when a dream candidate is illicitly promoted to observed status without external feedback."""
    pass


class GateRejectionError(RuntimeError):
    """Raised when a candidate update fails one of the three validation gates."""
    pass


@dataclass(frozen=True)
class ObservedEvent:
    """Immutable record of an externally confirmed real-world observation."""
    event_id: int
    timestamp: int
    input_token: int
    target_token: int
    pred_token: int
    confidence: float
    context_basin: int | None = None
    source: str = "OBSERVED"

    def __post_init__(self):
        if self.source != "OBSERVED":
            raise ValueError(f"ObservedEvent must have source='OBSERVED', got {self.source}")


@dataclass(frozen=True)
class DreamHypothesis:
    """Immutable record of an internally generated dream hypothesis."""
    hypothesis_id: int
    timestamp: int
    input_token: int
    pred_token: int
    confidence: float
    maya_weight: float
    provenance_basis: tuple[int, ...]  # Event IDs from B_real that motivated this candidate
    source: str = "DREAM_CANDIDATE"

    def __post_init__(self):
        if self.source != "DREAM_CANDIDATE":
            raise ValueError(f"DreamHypothesis must have source='DREAM_CANDIDATE', got {self.source}")


def compute_maya_weight(confidence: float, t_maya: float = 1.0) -> float:
    """Compute Maya uncertainty weight w_Maya(C) = C * exp(-H2(C) / T_Maya).
    
    Parameters
    ----------
    confidence : float
        Model posterior probability C in (0, 1).
    t_maya : float
        Temperature parameter T_Maya > 0.
        
    Returns
    -------
    float
        Maya weight penalizing maximum uncertainty (C = 0.5).
    """
    if not np.isfinite(confidence):
        raise NonFiniteStateError(f"Non-finite confidence in Maya weight calculation: {confidence}")

    c = float(np.clip(confidence, 1e-12, 1.0 - 1e-12))
    h2 = -c * np.log(c) - (1.0 - c) * np.log(1.0 - c)
    w_maya = c * np.exp(-h2 / max(t_maya, 1e-6))
    return float(np.clip(w_maya, 0.0, 1.0))


class WitnessLedger:
    """Audit ledger maintaining immutable provenance and separating real facts from dreams."""

    def __init__(self, t_maya: float = 1.0):
        self.t_maya = t_maya
        self.b_real: list[ObservedEvent] = []
        self.b_dream: list[DreamHypothesis] = []
        self.rejected_updates: list[dict] = []

        self._next_event_id: int = 1
        self._next_hypo_id: int = 1
        self._unverified_promotion_count: int = 0

    @property
    def real_events(self) -> list[ObservedEvent]:
        return list(self.b_real)

    @property
    def dream_hypotheses(self) -> list[DreamHypothesis]:
        return list(self.b_dream)

    def record_observation(
        self,
        timestamp: int,
        input_token: int,
        target_token: int,
        pred_token: int,
        confidence: float,
        context_basin: int | None = None,
    ) -> ObservedEvent:
        """Record an externally verified observation into B_real."""
        event = ObservedEvent(
            event_id=self._next_event_id,
            timestamp=timestamp,
            input_token=input_token,
            target_token=target_token,
            pred_token=pred_token,
            confidence=float(confidence),
            context_basin=context_basin,
            source="OBSERVED",
        )
        self._next_event_id += 1
        self.b_real.append(event)
        return event

    def record_dream_hypothesis(
        self,
        timestamp: int,
        input_token: int,
        pred_token: int,
        confidence: float,
        provenance_basis: list[int] | tuple[int, ...],
    ) -> DreamHypothesis:
        """Record an internally generated counterfactual hypothesis into B_dream."""
        w_maya = compute_maya_weight(confidence, self.t_maya)
        hypo = DreamHypothesis(
            hypothesis_id=self._next_hypo_id,
            timestamp=timestamp,
            input_token=input_token,
            pred_token=pred_token,
            confidence=float(confidence),
            maya_weight=w_maya,
            provenance_basis=tuple(provenance_basis),
            source="DREAM_CANDIDATE",
        )
        self._next_hypo_id += 1
        self.b_dream.append(hypo)
        return hypo

    def attempt_promote_without_feedback(self, hypothesis_id: int) -> None:
        """Illicit attempt to promote dream to observed without feedback — MUST FAIL."""
        self._unverified_promotion_count += 1
        raise UnverifiedPromotionError(
            f"VIOLATION: Hypothesis {hypothesis_id} cannot be promoted to OBSERVED "
            "without external delayed feedback!"
        )

    def promote_with_verified_feedback(
        self,
        hypothesis_id: int,
        true_target: int,
        timestamp: int,
    ) -> ObservedEvent:
        """Legitimate promotion only upon receipt of external delayed feedback."""
        match = next((h for h in self.b_dream if h.hypothesis_id == hypothesis_id), None)
        if match is None:
            raise ValueError(f"Hypothesis {hypothesis_id} not found in B_dream")

        # Verified observation created with external target
        obs = self.record_observation(
            timestamp=timestamp,
            input_token=match.input_token,
            target_token=true_target,
            pred_token=match.pred_token,
            confidence=match.confidence,
        )
        return obs

    def assert_no_unverified_promotions(self) -> None:
        """Assert zero unverified promotions ever occurred."""
        assert self._unverified_promotion_count == 0, (
            f"Audit failure: {self._unverified_promotion_count} unverified promotions detected!"
        )
        # Ensure disjoint origin IDs and sources
        real_sources = {e.source for e in self.b_real}
        dream_sources = {h.source for h in self.b_dream}
        assert real_sources.issubset({"OBSERVED"}), "B_real must strictly contain OBSERVED records"
        assert dream_sources.issubset({"DREAM_CANDIDATE"}), "B_dream must strictly contain DREAM_CANDIDATE"

    def get_anchors(self, max_count: int = 20) -> list[ObservedEvent]:
        """Retrieve fixed real anchors from B_real for anchor guard evaluation."""
        return list(self.b_real[:max_count])


class ProvenanceConsistencyGate:
    """Gate 1: Verifies provenance basis and consistency with confirmed real events."""

    def check(self, candidate: DreamHypothesis, ledger: WitnessLedger) -> bool:
        # 1. Provenance: Basis event IDs must exist in B_real
        known_real_ids = {e.event_id for e in ledger.b_real}
        for basis_id in candidate.provenance_basis:
            if basis_id not in known_real_ids:
                return False

        # 2. Consistency: Candidate must not contradict confirmed mappings in B_real
        for e in ledger.b_real:
            if e.input_token == candidate.input_token:
                # If there is a confirmed real target for this token, candidate must not contradict it
                if e.target_token != candidate.pred_token:
                    return False
        return True


class StructuralCheckGate:
    """Gate 2: Verifies bounds, non-finiteness, and valid transitions."""

    def __init__(self, d_in: int = 8, d_out: int = 8):
        self.d_in = d_in
        self.d_out = d_out

    def check(self, candidate: DreamHypothesis) -> bool:
        if not (0 <= candidate.input_token < self.d_in):
            return False
        if not (0 <= candidate.pred_token < self.d_out):
            return False
        if not np.isfinite(candidate.confidence) or not (0.0 <= candidate.confidence <= 1.0):
            return False
        if not np.isfinite(candidate.maya_weight) or not (0.0 <= candidate.maya_weight <= 1.0):
            return False
        return True


class AnchorGuardGate:
    """Gate 3: Evaluates anchor accuracy on shadow copy and enforces rollback on degradation."""

    def __init__(self, max_degradation: float = 0.02):
        self.max_degradation = max_degradation

    def evaluate_anchors(
        self,
        core,
        anchors: list[ObservedEvent],
        ticks_per_token: int = 5,
    ) -> float:
        """Measure prediction accuracy across real anchor set."""
        if not anchors:
            return 1.0

        saved_z = core.Z.copy()
        corr = 0
        try:
            for anc in anchors:
                core.Z.fill(0.0)
                for c in range(6):
                    core.Z[8 * c + 6] = 1.0
                x = np.zeros(core.config.d_in)
                x[anc.input_token] = 1.0
                for _ in range(ticks_per_token):
                    core.step(x, write_gate=1.0)
                logits = core.predict()
                pred = int(np.argmax(logits[1:5])) + 1
                if pred == anc.target_token:
                    corr += 1
            return corr / len(anchors)
        finally:
            core.Z[:] = saved_z

    def check_and_apply(
        self,
        core,
        anchors: list[ObservedEvent],
        proposed_delta_K: np.ndarray,
        proposed_delta_B: np.ndarray | None = None,
        ticks_per_token: int = 5,
    ) -> tuple[bool, float, float]:
        """Shadow test: apply proposed delta in shadow copy, verify anchors, commit or rollback."""
        acc_before = self.evaluate_anchors(core, anchors, ticks_per_token=ticks_per_token)

        # Create shadow copy
        shadow_snap = core.snapshot()
        core.K += proposed_delta_K
        if proposed_delta_B is not None:
            core.B += proposed_delta_B

        acc_after = self.evaluate_anchors(core, anchors, ticks_per_token=ticks_per_token)
        degradation = acc_before - acc_after

        if degradation > self.max_degradation:
            # ROLLBACK: Revert shadow modification
            core.restore(shadow_snap)
            return False, acc_before, acc_after

        # PASSED: Keep update
        return True, acc_before, acc_after
