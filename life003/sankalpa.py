"""Two-channel SANKALPA policy gate for LIFE-003 (PQT-KC-001 v0.3)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from .ledger import DEFAULT_SUPPORT_MIN_SCORE, feedback_score

if TYPE_CHECKING:
    from .ledger import SaksinLedger


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
        witness_support_min_score: float = DEFAULT_SUPPORT_MIN_SCORE,
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
        ledger: Any | None = None,
        hypothesis: Any | None = None,
        candidate_action: int | str | None = None,
        context: dict[str, Any] | None = None,
    ) -> tuple[str, bool]:
        """Evaluate epistemic channel d_ep.
        
        ZeroWitnessCommit == 0: If witness is missing or unverified in ledger,
        epistemic assertion must be ABSTAIN.
        
        EPISTEMIC-PATCH-001:
        1. Ledger authenticity != Evidentiary support:
           Authentic OBSERVED records allow at most CLASS_ONLY, never COMMIT.
           AUDIT_SHADOW records strictly result in ABSTAIN.
        2. COMMIT requires VERIFIED_FEEDBACK matching:
           - Provenance (hypothesis / event basis)
           - Outcome / Result (action match and confirmation status)
           - Applicability domain
           - Verified support score from ledger payload (never arbitrary caller score).
        3. Fail-closed: If require_structured_witness=True and ledger is None,
           returns ABSTAIN with zero-witness guard tripped.
        """
        if witness_ref is None:
            # HonestPause: missing witness strictly forbids COMMIT
            return "ABSTAIN", True

        # Requirement 3: Fail-closed if require_structured_witness is enabled and ledger is missing
        if self.require_structured_witness and ledger is None:
            return "ABSTAIN", True

        if ledger is not None:
            is_valid, _ = ledger.verify_witness_reference(witness_ref)
            if not is_valid:
                # Invalid or unrecorded witness reference trips the zero-witness guard
                return "ABSTAIN", True

            rec = ledger.get_entry(witness_ref)
            if rec is None:
                return "ABSTAIN", True

            rec_type = rec.get("record_type")
            payload = rec.get("payload", {})

            # Requirement 1: OBSERVED and AUDIT_SHADOW do not authorize COMMIT
            if rec_type != "VERIFIED_FEEDBACK":
                if rec_type == "OBSERVED":
                    # An observation registers an event without verified feedback confirmation
                    return "CLASS_ONLY", False
                else:
                    # AUDIT_SHADOW, HYPOTHESIS, EVICTED, etc. cannot act as epistemic witness
                    return "ABSTAIN", True

            if isinstance(hypothesis, str):
                candidate_id = hypothesis
            elif isinstance(hypothesis, dict):
                candidate_id = hypothesis.get("candidate_id")
                if candidate_action is None:
                    candidate_action = hypothesis.get("action_candidate")
                if context is None:
                    context = hypothesis.get("context")
            else:
                candidate_id = getattr(hypothesis, "candidate_id", None)
                if candidate_action is None:
                    candidate_action = getattr(hypothesis, "action_candidate", None)
                if context is None:
                    context = getattr(hypothesis, "context", None)
            ctx = context or {}
            if hypothesis is None:
                candidate_id = ctx.get("candidate_id", ctx.get("hypothesis_id", ctx.get("provenance")))
                if candidate_action is None:
                    candidate_action = ctx.get("action", ctx.get("candidate_action"))
            domain = ctx.get("applicability_domain", ctx.get("domain", ctx.get("query")))
            try:
                verified_score = feedback_score(payload, candidate_id, candidate_action, domain)
            except (TypeError, ValueError):
                return "ABSTAIN", True
            if verified_score >= self.witness_support_min_score:
                return "COMMIT", False
            else:
                return "CLASS_ONLY", False

        # If ledger is None and require_structured_witness is False:
        if not self.require_structured_witness:
            if witness_support_score >= self.witness_support_min_score:
                return "COMMIT", False
            return "CLASS_ONLY", False

        return "ABSTAIN", True

    def decide(
        self,
        candidate_action: int | str,
        witness_ref: str | None = None,
        witness_support_score: float = 0.0,
        is_safe_probe: bool = False,
        is_idle: bool = False,
        ledger: Any | None = None,
        hypothesis: Any | None = None,
        context: dict[str, Any] | None = None,
    ) -> SankalpaDecision:
        """Compute joint decision over decoupled channels.
        
        CRITICAL: Epistemic ABSTAIN does NOT forbid safe PROBE or motor action.
        """
        if is_idle:
            d_ep, tripped = self.evaluate_epistemic_channel(
                witness_ref=witness_ref,
                witness_support_score=witness_support_score,
                ledger=ledger,
                hypothesis=hypothesis,
                candidate_action=candidate_action,
                context=context,
            )
            return SankalpaDecision(
                epistemic_channel="ABSTAIN",
                operational_channel=self.default_idle_action,
                witness_ref=None,
                zero_witness_guard_tripped=False,
            )

        d_ep, zero_witness_tripped = self.evaluate_epistemic_channel(
            witness_ref=witness_ref,
            witness_support_score=witness_support_score,
            ledger=ledger,
            hypothesis=hypothesis,
            candidate_action=candidate_action,
            context=context,
        )

        if is_safe_probe:
            a_act = self.safe_probe_action
        else:
            a_act = candidate_action

        return SankalpaDecision(
            epistemic_channel=d_ep,
            operational_channel=a_act,
            witness_ref=witness_ref if (not zero_witness_tripped and d_ep in ("COMMIT", "CLASS_ONLY")) else None,
            zero_witness_guard_tripped=zero_witness_tripped,
        )
