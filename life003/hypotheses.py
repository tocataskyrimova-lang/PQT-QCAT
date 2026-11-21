"""Two-coordinate hypothesis space H for LIFE-003 (PQT-KC-001 v0.3)."""
from __future__ import annotations
import copy
from dataclasses import dataclass, asdict
from typing import Any

JUSTIFICATION_STATUSES = {"UNVERIFIED", "SUPPORTED", "CONFLICTED", "REFUTED"}
FRESHNESS_STATUSES = {"FRESH", "AGING", "OSSIFIED"}


@dataclass
class Hypothesis:
    """Hypothesis candidate with orthogonal truth justification and Spanda freshness coordinates."""
    candidate_id: str
    action_candidate: int | str
    v_just: str = "UNVERIFIED"           # Truth justification
    s_fresh: str = "FRESH"               # Spanda freshness
    inertia_m_phi_uses: int = 0          # Uses without audit verification
    ticks_since_audit: int = 0           # Elapsed ticks since last verified audit
    witness_ref: str | None = None       # External witness handle if confirmed
    context: dict[str, Any] | None = None
    created_seq: int = 0                 # Monotonic creation sequence for deterministic eviction

    def __post_init__(self):
        if self.v_just not in JUSTIFICATION_STATUSES:
            raise ValueError(f"Invalid justification status: {self.v_just}")
        if self.s_fresh not in FRESHNESS_STATUSES:
            raise ValueError(f"Invalid freshness status: {self.s_fresh}")
        if self.v_just == "SUPPORTED" and (not self.witness_ref or not str(self.witness_ref).strip()):
            raise ValueError("Hypothesis cannot have v_just='SUPPORTED' without a non-empty witness_ref")

    def update_freshness(
        self,
        m_phi_uses_aging: int = 8,
        m_phi_uses_ossified: int = 32,
        age_ticks_aging: int = 50,
        age_ticks_ossified: int = 200,
    ) -> None:
        """Update Spanda freshness via OR-rule over usage count and elapsed ticks.
        
        CRITICAL: Freshness aging NEVER changes truth justification v_just to REFUTED.
        """
        if self.inertia_m_phi_uses >= m_phi_uses_ossified or self.ticks_since_audit >= age_ticks_ossified:
            self.s_fresh = "OSSIFIED"
        elif self.inertia_m_phi_uses >= m_phi_uses_aging or self.ticks_since_audit >= age_ticks_aging:
            self.s_fresh = "AGING"
        else:
            self.s_fresh = "FRESH"


def _validate_supported_witness(
    candidate_id: str,
    action_candidate: int | str,
    witness_ref: str | None,
    ledger: Any | None,
    context: dict[str, Any] | None = None,
    is_audit: bool = False,
) -> None:
    """Enforce fail-closed verification for any candidate promoting to or registering as SUPPORTED.
    
    All conditions must hold simultaneously:
    1. witness_ref must be non-empty.
    2. ledger must be provided and authentic.
    3. witness_ref must exist in ledger with valid cryptographic hash chain.
    4. record_type must be VERIFIED_FEEDBACK.
    5. payload['confirmed'] must be strict boolean True (False or truthy int rejected).
    6. payload['result'] must be present and non-empty.
    7. Candidate action, id, and domain must match strictly if present.
    """
    if not witness_ref or not str(witness_ref).strip():
        if is_audit:
            raise ValueError(f"Cannot promote candidate '{candidate_id}' to SUPPORTED without a valid witness_ref")
        else:
            raise ValueError(f"Cannot register hypothesis with v_just='SUPPORTED' without a non-empty witness_ref")
    if ledger is None:
        raise ValueError(f"Cannot establish v_just='SUPPORTED' for candidate '{candidate_id}' without an authentic ledger")

    is_valid, reason = ledger.verify_witness_reference(witness_ref)
    if not is_valid:
        raise ValueError(f"Witness reference '{witness_ref}' not verified in ledger: {reason}")
    rec = ledger.get_entry(witness_ref)
    if rec is None or rec.get("record_type") != "VERIFIED_FEEDBACK":
        raise ValueError(f"Witness reference '{witness_ref}' is not a VERIFIED_FEEDBACK record")

    payload = rec.get("payload", {})
    if type(payload.get("confirmed")) is not bool or payload.get("confirmed") is not True:
        raise ValueError(f"Witness record '{witness_ref}' confirmed status is not strict True bool")

    result = payload.get("result")
    if result is None or (isinstance(result, (str, list, tuple, dict, set)) and len(result) == 0):
        raise ValueError(f"Witness record '{witness_ref}' has empty result")

    rec_act = payload.get("action") if payload.get("action") is not None else payload.get("candidate_action")
    if rec_act is not None and str(rec_act) != str(action_candidate):
        raise ValueError(f"Witness action '{rec_act}' does not match hypothesis action '{action_candidate}'")

    rec_cid = payload.get("hypothesis_id") or payload.get("candidate_id") or payload.get("provenance")
    if rec_cid is not None and isinstance(rec_cid, str) and rec_cid.strip() and str(rec_cid) != str(candidate_id):
        raise ValueError(f"Witness candidate '{rec_cid}' does not match hypothesis '{candidate_id}'")

    rec_dom = payload.get("applicability_domain") or payload.get("domain")
    ctx_dom = context.get("applicability_domain") or context.get("domain") if isinstance(context, dict) else None
    if rec_dom is not None and ctx_dom is not None and str(rec_dom) != str(ctx_dom):
        raise ValueError(f"Witness domain '{rec_dom}' does not match hypothesis context domain '{ctx_dom}'")


class HypothesisPool:
    """Bounded set of hypotheses |H| <= max_active_hypotheses with auditable lifecycle."""

    def __init__(
        self,
        max_active_hypotheses: int = 8,
        m_phi_uses_aging: int = 8,
        m_phi_uses_ossified: int = 32,
        age_ticks_aging: int = 50,
        age_ticks_ossified: int = 200,
    ):
        self.max_capacity = max_active_hypotheses
        self.m_phi_uses_aging = m_phi_uses_aging
        self.m_phi_uses_ossified = m_phi_uses_ossified
        self.age_ticks_aging = age_ticks_aging
        self.age_ticks_ossified = age_ticks_ossified
        self.hypotheses: dict[str, Hypothesis] = {}
        self.last_eviction: dict[str, Any] | None = None
        self._next_seq: int = 0

    def register_candidate(
        self,
        candidate_id: str,
        action_candidate: int | str,
        v_just: str = "UNVERIFIED",
        witness_ref: str | None = None,
        context: dict[str, Any] | None = None,
        ledger: Any | None = None,
    ) -> Hypothesis:
        """Register a new candidate.
        
        CRITICAL FIX-1 & FIX-3:
        - If eviction is required, choose candidate strictly by deterministic (created_seq, candidate_id).
        - v_just='SUPPORTED' strictly requires a non-empty confirmed witness_ref verified against an authentic ledger.
        """
        if v_just == "SUPPORTED":
            _validate_supported_witness(
                candidate_id=candidate_id,
                action_candidate=action_candidate,
                witness_ref=witness_ref,
                ledger=ledger,
                context=context,
            )

        eviction_info = None

        if len(self.hypotheses) >= self.max_capacity and candidate_id not in self.hypotheses:
            # Hierarchical eviction priority: REFUTED -> UNVERIFIED -> oldest OSSIFIED supported
            # Eviction tie-breaking is strictly deterministic by (created_seq, candidate_id)
            refuted = [h for h in self.hypotheses.values() if h.v_just == "REFUTED"]
            unverified = [h for h in self.hypotheses.values() if h.v_just == "UNVERIFIED"]

            if refuted:
                evicted_hyp = min(refuted, key=lambda h: (h.created_seq, h.candidate_id))
            elif unverified:
                evicted_hyp = min(unverified, key=lambda h: (h.created_seq, h.candidate_id))
            else:
                # All hypotheses are SUPPORTED: evict oldest ossified/stale
                evicted_hyp = max(self.hypotheses.values(), key=lambda h: (h.ticks_since_audit, -h.created_seq, h.candidate_id))

            evict_id = evicted_hyp.candidate_id
            eviction_info = {
                "candidate_id": evict_id,
                "action": evicted_hyp.action_candidate,
                "v_just": evicted_hyp.v_just,
                "s_fresh": evicted_hyp.s_fresh,
                "witness_ref": evicted_hyp.witness_ref,
                "reason": "POOL_CAPACITY_EXCEEDED",
            }
            del self.hypotheses[evict_id]

        seq = self._next_seq
        self._next_seq += 1

        hyp = Hypothesis(
            candidate_id=candidate_id,
            action_candidate=action_candidate,
            v_just=v_just,
            s_fresh="FRESH",
            inertia_m_phi_uses=0,
            ticks_since_audit=0,
            witness_ref=witness_ref,
            context=copy.deepcopy(context),
            created_seq=seq,
        )
        self.hypotheses[candidate_id] = hyp
        self.last_eviction = eviction_info
        return hyp

    def record_usage(self, candidate_id: str) -> None:
        """Record usage of a hypothesis without re-verification."""
        if candidate_id in self.hypotheses:
            hyp = self.hypotheses[candidate_id]
            hyp.inertia_m_phi_uses += 1
            hyp.update_freshness(
                self.m_phi_uses_aging,
                self.m_phi_uses_ossified,
                self.age_ticks_aging,
                self.age_ticks_ossified,
            )

    def advance_time(self, delta_ticks: int = 1) -> None:
        """Advance time since audit for all active hypotheses."""
        for hyp in self.hypotheses.values():
            hyp.ticks_since_audit += delta_ticks
            hyp.update_freshness(
                self.m_phi_uses_aging,
                self.m_phi_uses_ossified,
                self.age_ticks_aging,
                self.age_ticks_ossified,
            )

    def audit_verification(
        self,
        candidate_id: str,
        is_supported: bool,
        witness_ref: str | None = None,
        ledger: Any | None = None,
    ) -> dict[str, Any] | None:
        """External feedback verifies or refutes hypothesis; resets Spanda staleness.
        
        CRITICAL FIX-3: Promotion to SUPPORTED strictly requires an authentic ledger
        and a valid non-empty matching VERIFIED_FEEDBACK record.
        """
        if candidate_id in self.hypotheses:
            hyp = self.hypotheses[candidate_id]
            prev_v = hyp.v_just
            if is_supported:
                eff_witness = witness_ref if witness_ref is not None else hyp.witness_ref
                _validate_supported_witness(
                    candidate_id=candidate_id,
                    action_candidate=hyp.action_candidate,
                    witness_ref=eff_witness,
                    ledger=ledger,
                    context=hyp.context,
                    is_audit=True,
                )
                hyp.v_just = "SUPPORTED"
                hyp.witness_ref = str(eff_witness)
            else:
                hyp.v_just = "REFUTED"

            hyp.s_fresh = "FRESH"
            hyp.inertia_m_phi_uses = 0
            hyp.ticks_since_audit = 0

            return {
                "candidate_id": candidate_id,
                "previous_justification": prev_v,
                "new_justification": hyp.v_just,
                "witness_ref": hyp.witness_ref,
            }
        return None

    def snapshot(self) -> dict[str, Any]:
        return {
            "max_capacity": self.max_capacity,
            "m_phi_uses_aging": self.m_phi_uses_aging,
            "m_phi_uses_ossified": self.m_phi_uses_ossified,
            "age_ticks_aging": self.age_ticks_aging,
            "age_ticks_ossified": self.age_ticks_ossified,
            "_next_seq": self._next_seq,
            "hypotheses": {hid: asdict(hyp) for hid, hyp in self.hypotheses.items()},
        }

    @classmethod
    def restore(cls, data: dict[str, Any]) -> HypothesisPool:
        pool = cls(
            max_active_hypotheses=int(data["max_capacity"]),
            m_phi_uses_aging=int(data["m_phi_uses_aging"]),
            m_phi_uses_ossified=int(data["m_phi_uses_ossified"]),
            age_ticks_aging=int(data["age_ticks_aging"]),
            age_ticks_ossified=int(data["age_ticks_ossified"]),
        )
        pool._next_seq = int(data.get("_next_seq", 0))
        loaded = [Hypothesis(**hdata) for hdata in data["hypotheses"].values()]
        loaded.sort(key=lambda h: (h.created_seq, h.candidate_id))
        for hyp in loaded:
            pool.hypotheses[hyp.candidate_id] = hyp
        if loaded:
            max_c = max(h.created_seq for h in loaded)
            if pool._next_seq <= max_c:
                pool._next_seq = max_c + 1
        return pool
