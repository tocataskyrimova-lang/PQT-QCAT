"""Energy functional and exact analytical gradients for PQT-QCat 8D cell.

Conforms strictly to AUDIT_REPORT_RU and ARCHITECTURE_LIFE_002:
- State Z = (X, W, w0, m_phi) in R^8
- Explicit r0_sq parameter (default 0.8) without implicit conversions
- Finite check on all inputs and outputs
- Analytical gradient matching autograd to < 1e-12 and finite diff to < 1e-8
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch

from .errors import NonFiniteStateError


@dataclass(frozen=True)
class CellParameters:
    """Parameters of the single 8D PQT cell potential."""
    alpha: float = 1.0
    kappa: float = 1.5
    lambda_w: float = 0.5
    alpha_0: float = 1.2
    w_star: float = 1.0
    h_eff: float = 0.45
    lambda_m: float = 1.5
    m_max: float = 1.0
    r0_sq: float = 0.8      # Canonical denominator r0^2 per PDF code page 18
    gamma_wire: float = 1.8


def compute_energy(Z: np.ndarray | torch.Tensor, params: CellParameters | None = None) -> np.ndarray | float:
    """Compute E(Z) for single state (8,) or batch (B, 8)."""
    if params is None:
        params = CellParameters()

    is_torch = isinstance(Z, torch.Tensor)
    if is_torch:
        z_np = Z.detach().cpu().numpy()
    else:
        z_np = np.asarray(Z, dtype=np.float64)

    if not np.all(np.isfinite(z_np)):
        raise NonFiniteStateError("Non-finite values encountered in state Z during energy computation")

    single = z_np.ndim == 1
    if single:
        z_np = z_np[np.newaxis, :]

    X = z_np[:, 0:3]
    W = z_np[:, 3:6]
    w0 = z_np[:, 6]
    m = z_np[:, 7]

    # Cyclic next along axis 1: [x1, x2, x0]
    X_next = np.roll(X, -1, axis=1)
    e = X - W * np.tanh(X_next)

    # Barycentric radius squared: r^2 = 1/3 * sum (x_i - x_{i+1})^2
    r_sq = np.sum((X - X_next) ** 2, axis=1) / 3.0
    m_eq = params.m_max * np.exp(-r_sq / params.r0_sq)

    term_x = (params.alpha / 4.0) * np.sum((X ** 2 - 1.0) ** 2, axis=1)
    term_e = (params.kappa / 2.0) * np.sum(e ** 2, axis=1)
    term_w = (params.lambda_w / 2.0) * np.sum((W - w0[:, np.newaxis]) ** 2, axis=1)
    term_w0 = (params.alpha_0 / 4.0) * ((w0 ** 2 - params.w_star ** 2) ** 2) - params.h_eff * w0
    term_m = (params.lambda_m / 2.0) * ((m - m_eq) ** 2) + params.gamma_wire * m * w0

    E = term_x + term_e + term_w + term_w0 + term_m
    return float(E[0]) if single else E


def compute_grad_energy(Z: np.ndarray, params: CellParameters | None = None) -> np.ndarray:
    """Compute analytical grad E(Z) for state (8,) or batch (B, 8)."""
    if params is None:
        params = CellParameters()

    z_np = np.asarray(Z, dtype=np.float64)
    if not np.all(np.isfinite(z_np)):
        raise NonFiniteStateError("Non-finite values encountered in state Z during gradient computation")

    single = z_np.ndim == 1
    if single:
        z_np = z_np[np.newaxis, :]

    X = z_np[:, 0:3]
    W = z_np[:, 3:6]
    w0 = z_np[:, 6]
    m = z_np[:, 7]

    # Cyclic rolls along axis 1:
    # X_next: [x1, x2, x0]
    # X_prev: [x2, x0, x1]
    X_next = np.roll(X, -1, axis=1)
    X_prev = np.roll(X, 1, axis=1)

    tanh_next = np.tanh(X_next)
    sech2_X = 1.0 - np.tanh(X) ** 2

    e = X - W * tanh_next
    e_prev = np.roll(e, 1, axis=1)
    W_prev = np.roll(W, 1, axis=1)

    r_sq = np.sum((X - X_next) ** 2, axis=1) / 3.0
    m_eq = params.m_max * np.exp(-r_sq / params.r0_sq)

    # 1. grad w.r.t X (3 components)
    # term_geom: 2*lambda_m / (3*r0_sq) * (m - m_eq) * m_eq * (2*X - X_next - X_prev)
    m_diff = (m - m_eq)[:, np.newaxis]
    m_eq_col = m_eq[:, np.newaxis]
    geom_factor = (2.0 * params.lambda_m / (3.0 * params.r0_sq)) * m_diff * m_eq_col
    geom_vector = 2.0 * X - X_next - X_prev

    g_X = (params.alpha * X * (X ** 2 - 1.0)
           + params.kappa * (e - W_prev * e_prev * sech2_X)
           + geom_factor * geom_vector)

    # 2. grad w.r.t W (3 components)
    g_W = params.lambda_w * (W - w0[:, np.newaxis]) - params.kappa * e * tanh_next

    # 3. grad w.r.t w0 (1 component)
    sum_W = np.sum(W, axis=1)
    g_w0 = (params.lambda_w * (3.0 * w0 - sum_W)
            + params.alpha_0 * w0 * (w0 ** 2 - params.w_star ** 2)
            - params.h_eff
            + params.gamma_wire * m)

    # 4. grad w.r.t m (1 component)
    g_m = params.lambda_m * (m - m_eq) + params.gamma_wire * w0

    grad = np.concatenate([g_X, g_W, g_w0[:, np.newaxis], g_m[:, np.newaxis]], axis=1)
    if not np.all(np.isfinite(grad)):
        raise NonFiniteStateError("Non-finite values encountered in analytical gradient")

    return grad[0] if single else grad


def grad_energy_autograd(Z: np.ndarray, params: CellParameters | None = None) -> np.ndarray:
    """Compute grad E(Z) via PyTorch autograd for exact reference."""
    if params is None:
        params = CellParameters()

    z_np = np.asarray(Z, dtype=np.float64)
    single = z_np.ndim == 1
    if single:
        z_np = z_np[np.newaxis, :]

    z_tensor = torch.tensor(z_np, dtype=torch.float64, requires_grad=True)
    X = z_tensor[:, 0:3]
    W = z_tensor[:, 3:6]
    w0 = z_tensor[:, 6]
    m = z_tensor[:, 7]

    X_next = torch.roll(X, shifts=-1, dims=1)
    e = X - W * torch.tanh(X_next)

    r_sq = torch.sum((X - X_next) ** 2, dim=1) / 3.0
    m_eq = params.m_max * torch.exp(-r_sq / params.r0_sq)

    term_x = (params.alpha / 4.0) * torch.sum((X ** 2 - 1.0) ** 2, dim=1)
    term_e = (params.kappa / 2.0) * torch.sum(e ** 2, dim=1)
    term_w = (params.lambda_w / 2.0) * torch.sum((W - w0.unsqueeze(1)) ** 2, dim=1)
    term_w0 = (params.alpha_0 / 4.0) * ((w0 ** 2 - params.w_star ** 2) ** 2) - params.h_eff * w0
    term_m = (params.lambda_m / 2.0) * ((m - m_eq) ** 2) + params.gamma_wire * m * w0

    E = term_x + term_e + term_w + term_w0 + term_m
    E_sum = torch.sum(E)
    E_sum.backward()

    grad_np = z_tensor.grad.detach().cpu().numpy()
    return grad_np[0] if single else grad_np
