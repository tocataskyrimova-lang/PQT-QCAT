"""Two-channel SANKALPA policy gate for LIFE-003 (PQT-KC-001 v0.3)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

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

            # Requirement 2: Strict whitelist fail-closed VERIFIED_FEEDBACK validation
            # All 4 conditions are mandatory simultaneously for COMMIT authorization:
            
            # Condition 1: Strict boolean confirmed is True
            confirmed = payload.get("confirmed")
            if not (isinstance(confirmed, bool) and confirmed is True):
                return "ABSTAIN", True
            if payload.get("status") in ("REFUTED", "REJECTED", "FAILED", "CONFLICTED"):
                return "ABSTAIN", True

            # Condition 2: result present and non-empty
            result = payload.get("result")
            if result is None or (isinstance(result, (str, list, tuple, dict, set)) and len(result) == 0):
                return "ABSTAIN", True

            # Condition 3: provenance, action, domain ALL present, non-empty on both sides and match strictly
            # 3a. Provenance check
            rec_prov = payload.get("provenance") or payload.get("hypothesis_id") or payload.get("candidate_id")
            hyp_id = None
            if hypothesis is not None:
                hyp_id = getattr(hypothesis, "candidate_id", None)
                if hyp_id is None and isinstance(hypothesis, dict):
                    hyp_id = hypothesis.get("candidate_id")
                elif hyp_id is None and isinstance(hypothesis, str):
                    hyp_id = hypothesis
            elif context is not None:
                hyp_id = context.get("provenance") or context.get("candidate_id") or context.get("hypothesis_id")

            if not rec_prov or not hyp_id:
                return "ABSTAIN", True
            if isinstance(rec_prov, str) and not rec_prov.strip():
                return "ABSTAIN", True
            if isinstance(hyp_id, str) and not str(hyp_id).strip():
                return "ABSTAIN", True

            if isinstance(rec_prov, (list, tuple, set)):
                if hyp_id not in rec_prov:
                    return "ABSTAIN", True
            else:
                if str(rec_prov) != str(hyp_id):
                    return "ABSTAIN", True

            # 3b. Outcome / action check
            rec_act = payload.get("action")
            if rec_act is None:
                rec_act = payload.get("candidate_action")

            target_act = candidate_action
            if target_act is None and hypothesis is not None:
                target_act = getattr(hypothesis, "action_candidate", None)
                if target_act is None and isinstance(hypothesis, dict):
                    target_act = hypothesis.get("action_candidate")
            elif target_act is None and context is not None:
                target_act = context.get("action") or context.get("candidate_action")

            if rec_act is None or target_act is None:
                return "ABSTAIN", True
            if isinstance(rec_act, str) and not rec_act.strip():
                return "ABSTAIN", True
            if isinstance(target_act, str) and not str(target_act).strip():
                return "ABSTAIN", True

            if str(target_act) != str(rec_act):
                return "ABSTAIN", True

            # 3c. Applicability domain check
            rec_domain = payload.get("applicability_domain") or payload.get("domain") or payload.get("context")
            hyp_domain = None
            if context is not None:
                hyp_domain = context.get("applicability_domain") or context.get("domain") or context.get("query")
            elif hypothesis is not None:
                hyp_ctx = getattr(hypothesis, "context", None)
                if isinstance(hyp_ctx, dict):
                    hyp_domain = hyp_ctx.get("applicability_domain") or hyp_ctx.get("domain") or hyp_ctx.get("query")
                elif isinstance(hyp_ctx, str):
                    hyp_domain = hyp_ctx

            if not rec_domain or not hyp_domain:
                return "ABSTAIN", True
            if isinstance(rec_domain, str) and not str(rec_domain).strip():
                return "ABSTAIN", True
            if isinstance(hyp_domain, str) and not str(hyp_domain).strip():
                return "ABSTAIN", True

            if str(rec_domain) != str(hyp_domain):
                return "ABSTAIN", True

            # Condition 4: Verified evidence score (must come from ledger payload, not caller parameter)
            if "support_score" in payload:
                verified_score = float(payload["support_score"])
            elif "confidence" in payload:
                verified_score = float(payload["confidence"])
            elif "score" in payload:
                verified_score = float(payload["score"])
            else:
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
