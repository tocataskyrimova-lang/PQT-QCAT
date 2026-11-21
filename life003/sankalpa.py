"""Two-channel SANKALPA policy gate for LIFE-003 (PQT-KC-001 v0.3)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SankalpaDecision:
    """Orthogonal Epistemic and Operational channel outputs."""
    epistemic_channel: str          # "COMMIT" | "CLASS_ONLY" | "ABSTAIN"
    operational_channel: int | str  # 1..4 | "PROBE" | "WAIT" | "IDLE"
    witness_ref: str | None
    zero_witness_guard_tripped: bool


class SankalpaGate:
    """Separates epistemic assertion from physical action execution."""

    def __init__(
        self,
        require_structured_witness: bool = True,
        witness_support_min_score: float = 0.60,
        allow_zero_witness_commit: bool = False,
        default_idle_action: str = "IDLE",
        safe_probe_action: str = "PROBE",
        wait_action: str = "WAIT",
    ):
        self.require_structured_witness = require_structured_witness
        self.witness_support_min_score = float(witness_support_min_score)
        self.allow_zero_witness_commit = allow_zero_witness_commit
        self.default_idle_action = default_idle_action
        self.safe_probe_action = safe_probe_action
        self.wait_action = wait_action

    def evaluate_epistemic_channel(
        self,
        witness_ref: str | None,
        witness_support_score: float = 0.0,
    ) -> tuple[str, bool]:
        """Evaluate epistemic channel d_ep.
        
        ZeroWitnessCommit == 0: If witness is missing, epistemic assertion must be ABSTAIN.
        """
        if witness_ref is None:
            # HonestPause: missing witness strictly forbids COMMIT
            return "ABSTAIN", True

        if witness_support_score >= self.witness_support_min_score:
            return "COMMIT", False

        return "CLASS_ONLY", False

    def decide(
        self,
        candidate_action: int | str,
        witness_ref: str | None = None,
        witness_support_score: float = 0.0,
        is_safe_probe: bool = False,
        is_idle: bool = False,
    ) -> SankalpaDecision:
        """Compute joint decision over decoupled channels.
        
        CRITICAL: Epistemic ABSTAIN does NOT forbid safe PROBE or motor action.
        """
        if is_idle:
            d_ep, tripped = self.evaluate_epistemic_channel(witness_ref, witness_support_score)
            return SankalpaDecision(
                epistemic_channel="ABSTAIN",
                operational_channel=self.default_idle_action,
                witness_ref=None,
                zero_witness_guard_tripped=False,
            )

        d_ep, zero_witness_tripped = self.evaluate_epistemic_channel(witness_ref, witness_support_score)

        if is_safe_probe:
            a_act = self.safe_probe_action
        else:
            a_act = candidate_action

        return SankalpaDecision(
            epistemic_channel=d_ep,
            operational_channel=a_act,
            witness_ref=witness_ref,
            zero_witness_guard_tripped=zero_witness_tripped,
        )
