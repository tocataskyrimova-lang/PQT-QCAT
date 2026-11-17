"""Bounded core parameter plasticity engine (core_only mode).

Updates structural core parameters theta_core = {K, B} while keeping
W_readout strictly frozen, proving causal learning in the dynamic core.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..core.six_echelon import SixEchelonCore
from .eligibility import CoreEligibilityTraces


@dataclass
class CorePlasticityConfig:
    """Configuration for core plasticity."""
    eta_core: float = 1e-2          # Learning rate for inter-cell coupling K
    eta_input: float = 2e-1         # Learning rate for input projection B
    delta_core_max: float = 0.5     # Max absolute parameter update per step
    weight_decay: float = 0.0       # Passive weight decay on core parameters


class CorePlasticityEngine:
    """Applies local eligibility-based updates to theta_core."""

    def __init__(self, config: CorePlasticityConfig | None = None):
        self.config = config or CorePlasticityConfig()

    def apply_feedback(
        self,
        core: SixEchelonCore,
        traces: CoreEligibilityTraces,
        error_signal: np.ndarray,
        x_last: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Update core parameters theta_core from error signal using frozen readout feedback.
        
        Args:
            core: The 48D core whose K and B are updated. W_readout is FROZEN.
            traces: The eligibility traces accumulated before feedback.
            error_signal: Target error in output space: r_t = (y_true - y_pred) in R^d_out.
            x_last: Latest sensory input token vector.
            
        Returns:
            Diagnostics dict with norms of parameter changes.
        """
        error_arr = np.asarray(error_signal, dtype=np.float64)

        # 1. Project output error back to 48D core space via frozen readout transpose
        delta_Z = core.W_readout.T @ error_arr

        # 2. Update input projection B: delta_B = eta * outer(delta_Z, x)
        delta_B_norm = 0.0
        if x_last is not None:
            delta_B = self.config.eta_input * np.outer(delta_Z, x_last)
            delta_B = np.clip(delta_B, -self.config.delta_core_max, self.config.delta_core_max)
            core.B = core.B * (1.0 - self.config.weight_decay) + delta_B
            delta_B_norm = float(np.linalg.norm(delta_B))

        # 3. Update coupling K: delta_K = eta * outer(delta_Z, psi_pre)
        psi = np.tanh(core.Z)
        delta_K = self.config.eta_core * np.outer(delta_Z, psi)
        delta_K = np.clip(delta_K, -self.config.delta_core_max, self.config.delta_core_max)
        core.K = core.K * (1.0 - self.config.weight_decay) + delta_K
        delta_K_norm = float(np.linalg.norm(delta_K))

        return {
            "norm_delta_K": delta_K_norm,
            "norm_delta_B": delta_B_norm,
            "core_updated": True,
            "readout_frozen": True,
        }
