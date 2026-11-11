"""Canonical antisymmetric vortex operator A_I(w0) for PQT-QCat 8D cell.

Conforms strictly to ARCHITECTURE_LIFE_002 (Section 3.2) and ILORI approval:
- w0 > 0: Latch phase (g -> 0), low mobility
- w0 < 0: Vortex phase (g -> 1), exploration without factual promotion
- A_I(w0) is strictly antisymmetric: A_I + A_I^T = 0 (machine precision < 1e-15)
- For any vector g, g^T A_I(w0) g = 0
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# Standard cyclic antisymmetric 3x3 permutation matrix
# J3 v = roll(v, 1) - roll(v, -1)
J3 = np.array([
    [0.0, -1.0,  1.0],
    [1.0,  0.0, -1.0],
    [-1.0, 1.0,  0.0]
], dtype=np.float64)


@dataclass(frozen=True)
class VortexParameters:
    """Parameters of the vortex operator A_I(w0)."""
    omega_space: float = 1.2
    omega_xw: float = 1.5
    omega_lm: float = 0.35
    beta_v: float = 4.0        # Sharpness of sigmoid latch transition


def vortex_latch_gain(w0: float | np.ndarray, beta_v: float = 4.0) -> float | np.ndarray:
    """Compute g(w0) = sigma(-beta_v * w0).
    
    When w0 > 0: g -> 0 (Latch / frozen memory)
    When w0 < 0: g -> 1 (Vortex / active exploration)
    """
    # Numerically stable sigmoid: 1 / (1 + exp(beta_v * w0))
    val = beta_v * np.asarray(w0, dtype=np.float64)
    # Clip to prevent overflow in exp
    val_clipped = np.clip(val, -60.0, 60.0)
    g = 1.0 / (1.0 + np.exp(val_clipped))
    return float(g) if np.ndim(w0) == 0 else g


def build_vortex_A_I(w0: float, params: VortexParameters | None = None) -> np.ndarray:
    """Construct the 8x8 canonical antisymmetric matrix A_I(w0)."""
    if params is None:
        params = VortexParameters()

    g = float(vortex_latch_gain(w0, params.beta_v))
    I3 = np.eye(3, dtype=np.float64)

    # 1. Block X-X: g * omega_space * J3 (3x3)
    block_xx = g * params.omega_space * J3
    # 2. Block X-W: +g * omega_xw * I3 (3x3)
    block_xw = g * params.omega_xw * I3
    # 3. Block W-X: -g * omega_xw * I3 (3x3)
    block_wx = -g * params.omega_xw * I3
    # 4. Block W-W: g * omega_space * J3 (3x3)
    block_ww = g * params.omega_space * J3

    A = np.zeros((8, 8), dtype=np.float64)
    A[0:3, 0:3] = block_xx
    A[0:3, 3:6] = block_xw
    A[3:6, 0:3] = block_wx
    A[3:6, 3:6] = block_ww

    # 5. Slow-axis coupling between w0 (idx 6) and m_phi (idx 7):
    # A[w0, m] = -omega_lm, A[m, w0] = +omega_lm
    A[6, 7] = -params.omega_lm
    A[7, 6] = params.omega_lm

    return A


def apply_vortex_A_I(vec: np.ndarray, w0: float, params: VortexParameters | None = None) -> np.ndarray:
    """Efficient matrix-vector product A_I(w0) @ vec without full 8x8 construction."""
    if params is None:
        params = VortexParameters()

    vec = np.asarray(vec, dtype=np.float64)
    g = float(vortex_latch_gain(w0, params.beta_v))

    X_v = vec[0:3]
    W_v = vec[3:6]
    w0_v = vec[6]
    m_v = vec[7]

    # J3 @ v = roll(v, 1) - roll(v, -1)
    J3_X = np.array([X_v[2] - X_v[1], X_v[0] - X_v[2], X_v[1] - X_v[0]], dtype=np.float64)
    J3_W = np.array([W_v[2] - W_v[1], W_v[0] - W_v[2], W_v[1] - W_v[0]], dtype=np.float64)

    out_X = g * (params.omega_space * J3_X + params.omega_xw * W_v)
    out_W = g * (-params.omega_xw * X_v + params.omega_space * J3_W)
    out_w0 = -params.omega_lm * m_v
    out_m = params.omega_lm * w0_v

    return np.concatenate([out_X, out_W, [out_w0], [out_m]])


def check_antisymmetry(A: np.ndarray) -> float:
    """Return max absolute defect ||A + A^T||_inf."""
    return float(np.max(np.abs(A + A.T)))
