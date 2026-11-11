"""Autonomous Twin Agent for T2 Blind Twin-Histories (PQT-HN002-TRI-003).

Strictly adheres to ARCHITECTURE_LIFE_002 and ILORI mandates:
- Zero external context_id, zero hidden rule hints, zero target access before predict()
- Endogenous context inferred purely from dynamic core state Z_ctx (E5/E6)
- Readout projection W_readout is strictly frozen
- Plasticity operates exclusively on internal core parameters theta_core = {K, B}
- Vortex transition (w0 < 0) driven strictly by accumulated prediction neviazka
"""
from __future__ import annotations
import numpy as np

from ..core.six_echelon import SixEchelonCore, SixEchelonConfig
from ..core.errors import NonFiniteStateError
from ..plasticity.eligibility import CoreEligibilityTraces
from ..plasticity.core_update import CorePlasticityEngine, CorePlasticityConfig
from .basins import CANONICAL_BASINS, compute_q_ctx, get_dominant_basin


class TwinAgent:
    """Autonomous continuous dynamic agent for blind twin-history experiments."""

    def __init__(
        self,
        agent_id: str,
        config: SixEchelonConfig | None = None,
        lr_B: float = 0.35,
        lr_K: float = 0.05,
        shift_threshold: float = 0.40,
        error_alpha: float = 0.25,
        ticks_per_event: int = 5,
        seed: int = 42,
    ):
        self.agent_id = agent_id
        self.config = config or SixEchelonConfig()
        self.core = SixEchelonCore(config=self.config, seed=seed)
        self.traces = CoreEligibilityTraces()
        self.plasticity = CorePlasticityEngine(CorePlasticityConfig(eta_input=lr_B, eta_core=lr_K))

        self.ticks_per_event = ticks_per_event
        self.shift_threshold = shift_threshold
        self.error_alpha = error_alpha

        self.rolling_error = 0.0
        self.last_event_id = 0
        self.pending_events: dict[int, tuple[int, np.ndarray, int, np.ndarray]] = {}

        # Log of operational metrics
        self.history: list[dict] = []

    def predict(self, token: int) -> dict:
        """Issue autonomous prediction strictly BEFORE observing true feedback or target.
        
        Parameters
        ----------
        token : int
            Sensory input token in [0, d_in - 1]. No context hint provided!
            
        Returns
        -------
        dict
            Prediction payload containing ticket event_id, pred_token, probabilities, q_ctx.
        """
        if not (0 <= token < self.config.d_in):
            raise ValueError(f"Token {token} outside allowed range [0, {self.config.d_in - 1}]")

        # Encode sensory token into one-hot input
        x = np.zeros(self.config.d_in, dtype=np.float64)
        x[token] = 1.0

        # Step core forward in continuous time and update eligibility traces
        for _ in range(self.ticks_per_event):
            self.core.step(x_t=x, write_gate=1.0)
            self.traces.update(self.core.Z, x)

        # Output logits from frozen readout
        logits = self.core.predict()
        shifted = logits - np.max(logits)
        exp_l = np.exp(shifted)
        probs = exp_l / np.sum(exp_l)
        pred_token = int(np.argmax(probs))

        # Endogenous context inferred from E5 state (Z[32:40])
        z_e5 = self.core.get_cell_state(4)
        q_ctx = compute_q_ctx(z_e5)
        dom_basin, dom_prob, dom_dist = get_dominant_basin(z_e5)

        self.last_event_id += 1
        event_id = self.last_event_id
        self.pending_events[event_id] = (token, probs.copy(), pred_token, x.copy())

        return {
            "event_id": event_id,
            "pred_token": pred_token,
            "probabilities": probs,
            "q_ctx": q_ctx,
            "dominant_basin": dom_basin,
            "basin_confidence": dom_prob,
            "basin_distance": dom_dist,
        }

    def feedback(self, event_id: int, target_token: int) -> dict:
        """Process delayed ground-truth feedback strictly AFTER prediction is committed.
        
        Parameters
        ----------
        event_id : int
            Ticket id returned by predict().
        target_token : int
            True target class in [0, d_out - 1].
            
        Returns
        -------
        dict
            Feedback diagnostics including correctness, discrepancy, and core plasticity norm.
        """
        if event_id not in self.pending_events:
            raise ValueError(f"Event ID {event_id} not pending or already resolved")

        token, probs, pred_token, x_input = self.pending_events.pop(event_id)
        correct = bool(pred_token == target_token)

        # Target one-hot vector and error signal
        target_y = np.zeros(self.config.d_out, dtype=np.float64)
        target_y[target_token] = 1.0

        err_vec = target_y - probs
        instant_err = float(np.linalg.norm(err_vec))

        # Update rolling prediction neviazka (discrepancy)
        self.rolling_error = (1.0 - self.error_alpha) * self.rolling_error + self.error_alpha * instant_err

        # Rule shift detection: if neviazka exceeds threshold, trigger vortex
        is_shift = bool(self.rolling_error > self.shift_threshold)

        if is_shift and self.config.enable_vortex:
            # Vortex unlocked in E5 (Context Gate) and E6 (Rule Tracking)
            # Sets w0 < 0 to activate canonical A_I(w0) for barrier crossing
            self.core.Z[38] = -1.0  # E5 w0
            self.core.Z[46] = -1.0  # E6 w0
        else:
            # Latch mode (conservative attractor well)
            self.core.Z[38] = 1.0
            self.core.Z[46] = 1.0

        # Plasticity update strictly on theta_core = {K, B} with frozen W_readout
        error_signal = np.zeros(self.config.d_out, dtype=np.float64)
        error_signal[target_token] = 1.0 - probs[target_token]

        diag = self.plasticity.apply_feedback(
            self.core,
            self.traces,
            error_signal=error_signal,
            x_last=x_input,
        )

        record = {
            "event_id": event_id,
            "token": token,
            "target": target_token,
            "prediction": pred_token,
            "correct": correct,
            "error_norm": instant_err,
            "rolling_error": self.rolling_error,
            "is_shift": is_shift,
            "core_delta_norm": diag["norm_delta_B"] + diag["norm_delta_K"],
        }
        self.history.append(record)
        return record

    # =========================================================================
    # Causal Interventions for Disentanglement (Requirement 3)
    # =========================================================================

    def reset_state_to_neutral(self) -> None:
        """Reset fast dynamic state Z to neutral, leaving plastic core {K, B} intact."""
        self.core.Z.fill(0.0)
        for c in range(6):
            self.core.Z[8 * c + 6] = 1.0  # w0 = 1 (latch)
        self.rolling_error = 0.0

    def swap_dynamic_state(self, other: TwinAgent) -> None:
        """Swap fast dynamic state Z between two agents, preserving parameters {K, B}."""
        self.core.Z, other.core.Z = other.core.Z.copy(), self.core.Z.copy()

    def swap_core_parameters(self, other: TwinAgent) -> None:
        """Swap plastic core parameters {K, B} between two agents, preserving dynamic states."""
        self.core.K, other.core.K = other.core.K.copy(), self.core.K.copy()
        self.core.B, other.core.B = other.core.B.copy(), self.core.B.copy()

    def swap_context_router(self, other: TwinAgent) -> None:
        """Swap only context router echelons E5 and E6 (Z[32:48] and associated K rows/cols)."""
        # Swap E5/E6 states
        self.core.Z[32:48], other.core.Z[32:48] = other.core.Z[32:48].copy(), self.core.Z[32:48].copy()
        # Swap E5/E6 coupling blocks
        self.core.K[32:48, :], other.core.K[32:48, :] = other.core.K[32:48, :].copy(), self.core.K[32:48, :].copy()
        self.core.K[:, 32:48], other.core.K[:, 32:48] = other.core.K[:, 32:48].copy(), self.core.K[:, 32:48].copy()
