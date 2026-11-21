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

    def __post_init__(self):
        if self.v_just not in JUSTIFICATION_STATUSES:
            raise ValueError(f"Invalid justification status: {self.v_just}")
        if self.s_fresh not in FRESHNESS_STATUSES:
            raise ValueError(f"Invalid freshness status: {self.s_fresh}")

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

    def register_candidate(
        self,
        candidate_id: str,
        action_candidate: int | str,
        v_just: str = "UNVERIFIED",
        witness_ref: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> Hypothesis:
        """Register a new candidate.
        
        CRITICAL FIX-4: If eviction is required, never silently drop confirmed hypotheses
        if unverified or refuted ones exist, and record eviction details on self.last_eviction
        for Ledger audit logging.
        """
        eviction_info = None

        if len(self.hypotheses) >= self.max_capacity and candidate_id not in self.hypotheses:
            # Hierarchical eviction priority: REFUTED -> UNVERIFIED -> oldest OSSIFIED supported
            refuted = [hid for hid, h in self.hypotheses.items() if h.v_just == "REFUTED"]
            unverified = [hid for hid, h in self.hypotheses.items() if h.v_just == "UNVERIFIED"]

            if refuted:
                evict_id = refuted[0]
            elif unverified:
                evict_id = unverified[0]
            else:
                # All hypotheses are SUPPORTED: evict oldest ossified/stale
                evict_id = max(self.hypotheses.keys(), key=lambda hid: self.hypotheses[hid].ticks_since_audit)

            evicted_hyp = self.hypotheses[evict_id]
            eviction_info = {
                "candidate_id": evict_id,
                "action": evicted_hyp.action_candidate,
                "v_just": evicted_hyp.v_just,
                "s_fresh": evicted_hyp.s_fresh,
                "witness_ref": evicted_hyp.witness_ref,
                "reason": "POOL_CAPACITY_EXCEEDED",
            }
            del self.hypotheses[evict_id]

        hyp = Hypothesis(
            candidate_id=candidate_id,
            action_candidate=action_candidate,
            v_just=v_just,
            s_fresh="FRESH",
            inertia_m_phi_uses=0,
            ticks_since_audit=0,
            witness_ref=witness_ref,
            context=copy.deepcopy(context),
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
    ) -> dict[str, Any] | None:
        """External feedback verifies or refutes hypothesis; resets Spanda staleness."""
        if candidate_id in self.hypotheses:
            hyp = self.hypotheses[candidate_id]
            prev_v = hyp.v_just
            hyp.v_just = "SUPPORTED" if is_supported else "REFUTED"
            hyp.s_fresh = "FRESH"
            hyp.inertia_m_phi_uses = 0
            hyp.ticks_since_audit = 0
            if witness_ref is not None:
                hyp.witness_ref = witness_ref

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
        for hid, hdata in data["hypotheses"].items():
            pool.hypotheses[hid] = Hypothesis(**hdata)
        return pool
