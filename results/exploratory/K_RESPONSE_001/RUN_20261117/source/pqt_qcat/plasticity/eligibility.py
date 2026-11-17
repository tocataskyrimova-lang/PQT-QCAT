"""Eligibility trace tracking for delayed feedback plasticity.

Maintains temporal eligibility traces e_t of core activations and inputs,
allowing delayed error signals to update inter-cell coupling parameters K and B
without backpropagation through time (BPTT).
"""
from __future__ import annotations
import numpy as np


class CoreEligibilityTraces:
    """Maintains eligibility traces for core coupling K and input B."""

    def __init__(self, lambda_trace: float = 0.95):
        self.lambda_trace = lambda_trace
        # Traces for K (48, 48) and B (48, d_in)
        self.trace_K = np.zeros((48, 48), dtype=np.float64)
        self.trace_B: np.ndarray | None = None

    def reset(self) -> None:
        """Reset traces."""
        self.trace_K.fill(0.0)
        if self.trace_B is not None:
            self.trace_B.fill(0.0)

    def update(self, Z: np.ndarray, x_t: np.ndarray | None) -> None:
        """Advance eligibility trace with current state Z and sensory input x_t."""
        psi = np.tanh(Z)

        # Outer product of post-synaptic and pre-synaptic activations: psi @ psi^T
        outer_K = np.outer(psi, psi)
        self.trace_K = self.lambda_trace * self.trace_K + outer_K

        # Trace for B if x_t is provided
        if x_t is not None:
            if self.trace_B is None or self.trace_B.shape != (48, len(x_t)):
                self.trace_B = np.zeros((48, len(x_t)), dtype=np.float64)
            outer_B = np.outer(psi, x_t)
            self.trace_B = self.lambda_trace * self.trace_B + outer_B

    def snapshot(self) -> dict:
        return {
            "trace_K": self.trace_K.copy(),
            "trace_B": self.trace_B.copy() if self.trace_B is not None else None,
        }

    def restore(self, snap: dict) -> None:
        self.trace_K = snap["trace_K"].copy()
        self.trace_B = snap["trace_B"].copy() if snap["trace_B"] is not None else None
