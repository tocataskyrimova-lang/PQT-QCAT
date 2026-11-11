"""Tests for Witness Ledger, Provenance Invariants, and Three-Gate Consolidation.

Validates:
1. Complete isolation of B_real and B_dream.
2. UnverifiedPromotionError when attempting to promote without external feedback.
3. Zero false promotions invariant (assert_no_unverified_promotions).
4. Maya uncertainty weighting w_Maya(C) = C * exp(-H2(C) / T_Maya).
5. Provenance & Consistency Gate checks.
6. Structural Check Gate checks.
7. Anchor Guard Gate: shadow update, rollback on degradation > 2%, commit on safe update.
"""
from __future__ import annotations
import numpy as np
import pytest

from pqt_qcat.memory.witness import (
    WitnessLedger,
    ObservedEvent,
    DreamHypothesis,
    compute_maya_weight,
    UnverifiedPromotionError,
    ProvenanceConsistencyGate,
    StructuralCheckGate,
    AnchorGuardGate,
)
from pqt_qcat.core.six_echelon import SixEchelonCore, SixEchelonConfig


def test_witness_ledger_isolation_and_immutability():
    """Verify B_real and B_dream are strictly isolated and immutable."""
    ledger = WitnessLedger(t_maya=1.0)

    # 1. Record real observation
    obs = ledger.record_observation(
        timestamp=10,
        input_token=1,
        target_token=2,
        pred_token=2,
        confidence=0.95,
        context_basin=1,
    )
    assert obs.source == "OBSERVED"
    assert len(ledger.real_events) == 1

    # 2. Record dream hypothesis
    hypo = ledger.record_dream_hypothesis(
        timestamp=15,
        input_token=2,
        pred_token=3,
        confidence=0.88,
        provenance_basis=[obs.event_id],
    )
    assert hypo.source == "DREAM_CANDIDATE"
    assert len(ledger.dream_hypotheses) == 1

    # Invariant: Disjoint collections
    assert set(ledger.real_events).isdisjoint(set(ledger.dream_hypotheses))
    ledger.assert_no_unverified_promotions()


def test_witness_ledger_prohibits_unverified_promotion():
    """Verify any attempt to promote dream candidate without feedback raises UnverifiedPromotionError."""
    ledger = WitnessLedger()
    obs = ledger.record_observation(1, 1, 2, 2, 0.99)
    hypo = ledger.record_dream_hypothesis(2, 2, 3, 0.85, provenance_basis=[obs.event_id])

    # Direct promotion attempt must fail
    with pytest.raises(UnverifiedPromotionError):
        ledger.attempt_promote_without_feedback(hypo.hypothesis_id)

    # Legitimate promotion with external feedback succeeds
    verified_obs = ledger.promote_with_verified_feedback(
        hypothesis_id=hypo.hypothesis_id,
        true_target=3,
        timestamp=20,
    )
    assert verified_obs.source == "OBSERVED"
    assert verified_obs.target_token == 3


def test_maya_uncertainty_weighting():
    """Verify Maya weighting penalizes maximum entropy (C=0.5) and handles boundaries."""
    # Near certainty (C -> 1.0): H2 -> 0, w_Maya -> 1.0
    w_high = compute_maya_weight(0.99, t_maya=1.0)
    assert w_high > 0.90

    # High uncertainty (C = 0.5): H2 = ln(2) = 0.693, w_Maya = 0.5 * exp(-0.693) = 0.25
    w_mid = compute_maya_weight(0.50, t_maya=1.0)
    assert 0.24 < w_mid < 0.26

    # Low confidence (C = 0.05): w_Maya is small
    w_low = compute_maya_weight(0.05, t_maya=1.0)
    assert w_low < 0.05

    # Monotonicity test: w_Maya(0.95) > w_Maya(0.5)
    assert w_high > w_mid > w_low


def test_provenance_and_structural_gates():
    """Verify Gates 1 and 2 reject ungrounded or contradictory candidates."""
    ledger = WitnessLedger()
    obs1 = ledger.record_observation(1, 1, 2, 2, 0.95)

    gate1 = ProvenanceConsistencyGate()
    gate2 = StructuralCheckGate(d_in=8, d_out=8)

    # Candidate grounded in obs1 and consistent (token 2 -> 3)
    c_valid = ledger.record_dream_hypothesis(5, 2, 3, 0.90, provenance_basis=[obs1.event_id])
    assert gate1.check(c_valid, ledger) is True
    assert gate2.check(c_valid) is True

    # Candidate with non-existent provenance ID must be rejected
    c_fake_prov = ledger.record_dream_hypothesis(6, 3, 4, 0.80, provenance_basis=[9999])
    assert gate1.check(c_fake_prov, ledger) is False

    # Candidate that contradicts confirmed observation (token 1 -> 4 instead of confirmed 2)
    c_contradict = ledger.record_dream_hypothesis(7, 1, 4, 0.70, provenance_basis=[obs1.event_id])
    assert gate1.check(c_contradict, ledger) is False

    # Structural check rejects out-of-bounds tokens
    c_bad_token = ledger.record_dream_hypothesis(8, 99, 1, 0.5, provenance_basis=[obs1.event_id])
    assert gate2.check(c_bad_token) is False


def test_anchor_guard_shadow_update_and_rollback():
    """Verify Gate 3 applies updates on shadow core and rolls back upon anchor degradation > 2%."""
    core = SixEchelonCore(seed=42)
    core.W_readout.fill(0.0)
    CLASSES = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, -1.0, 0.0],
    ])
    for c in range(1, 5):
        core.W_readout[c, 32:35] = CLASSES[c]

    # Set B to give 100% on anchors (token 1 -> target 1, token 2 -> target 2)
    core.B[32, 1] = 5.0
    core.B[33, 2] = 5.0

    anchors = [
        ObservedEvent(1, 0, 1, 1, 1, 0.99),
        ObservedEvent(2, 0, 2, 2, 2, 0.99),
    ]

    guard = AnchorGuardGate(max_degradation=0.02)
    initial_acc = guard.evaluate_anchors(core, anchors)
    assert initial_acc == 1.0

    # 1. Proposed harmful delta B (wipes out learned anchor mapping)
    harmful_delta_B = -core.B.copy()
    B_before = core.B.copy()

    passed, acc_before, acc_after = guard.check_and_apply(
        core,
        anchors,
        proposed_delta_K=np.zeros((48, 48)),
        proposed_delta_B=harmful_delta_B,
    )

    # Must be rejected and rolled back!
    assert passed is False
    assert np.allclose(core.B, B_before), "Core parameters B must be strictly restored after rollback!"

    # 2. Proposed safe delta (zero update)
    safe_delta_K = np.zeros((48, 48))
    passed_safe, _, _ = guard.check_and_apply(core, anchors, proposed_delta_K=safe_delta_K)
    assert passed_safe is True
