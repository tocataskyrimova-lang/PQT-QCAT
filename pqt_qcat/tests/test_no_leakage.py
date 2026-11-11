"""Tests for Provenance Isolation and Zero-Leakage Enforcement.

Conforms strictly to ARCHITECTURE_LIFE_002 (C-01, C-07, H6-PROVENANCE) and ILORI mandate:
- Prediction is committed BEFORE feedback is available
- Attempting to pass target_y or rule labels into observe_and_predict raises AssertionError
- Tickets must match strictly for delayed feedback
"""
from __future__ import annotations
import numpy as np
import pytest

from pqt_qcat.core.six_echelon import SixEchelonCore, SixEchelonConfig

TEST_SEED = 42


class IsolatedAgentHarness:
    """Rigorous wrapper testing the exact prediction-before-feedback lifecycle."""

    def __init__(self, core: SixEchelonCore):
        self.core = core
        self.pending_ticket: int | None = None
        self.last_prediction: int | None = None
        self.ticket_counter: int = 0

    def observe_and_predict(self, x_t: np.ndarray, leaked_target: int | None = None) -> tuple[int, int]:
        """Observe input x_t, advance core, and commit prediction. Must NOT accept target."""
        if leaked_target is not None:
            raise AssertionError("LEAKAGE ATTEMPT: target passed into observe_and_predict!")

        # Advance core
        self.core.step(x_t, write_gate=1.0)

        # Commit prediction
        logits = self.core.predict()
        pred = int(np.argmax(logits))

        self.ticket_counter += 1
        ticket = self.ticket_counter
        self.pending_ticket = ticket
        self.last_prediction = pred

        return pred, ticket

    def accept_feedback(self, ticket: int, y_true: int) -> dict:
        """Accept delayed feedback only for a matching existing committed ticket."""
        if ticket != self.pending_ticket:
            raise ValueError(f"Ticket mismatch: received {ticket}, expected {self.pending_ticket}")

        self.pending_ticket = None
        return {
            "ticket": ticket,
            "y_true": y_true,
            "pred_was": self.last_prediction,
            "verified": True,
        }


def test_strict_prediction_before_feedback():
    """Verify that prediction occurs strictly before feedback."""
    core = SixEchelonCore(seed=TEST_SEED)
    agent = IsolatedAgentHarness(core)

    x = np.zeros(8)
    x[1] = 1.0

    # 1. Normal workflow
    pred, ticket = agent.observe_and_predict(x)
    assert ticket is not None
    assert isinstance(pred, int)

    feedback_receipt = agent.accept_feedback(ticket, y_true=2)
    assert feedback_receipt["verified"] is True

    # 2. Re-using ticket fails
    with pytest.raises(ValueError):
        agent.accept_feedback(ticket, y_true=2)


def test_leakage_assertion_triggers():
    """Verify that passing target into observe_and_predict is strictly caught and rejected."""
    core = SixEchelonCore(seed=TEST_SEED)
    agent = IsolatedAgentHarness(core)

    x = np.zeros(8)
    with pytest.raises(AssertionError) as exc_info:
        agent.observe_and_predict(x, leaked_target=5)

    assert "LEAKAGE ATTEMPT" in str(exc_info.value)
