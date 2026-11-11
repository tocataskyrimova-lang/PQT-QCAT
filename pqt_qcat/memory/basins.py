"""Endogenous Context Attractor Basins and Routing (PQT-HN002-TRI-003).

Provides:
- 8 canonical topological attractor basins of V_8 (6 hexagonal + 2 synchronous)
- Endogenous context distribution q_ctx(Z_ctx) without external context_id
- Energy barrier computation Delta E_barrier between basins
- Rule-shift detection through feedback discrepancy (neviazka)
"""
from __future__ import annotations
import numpy as np

from ..core.energy import CellParameters, compute_energy, compute_grad_energy
from ..core.errors import NonFiniteStateError

# 8 Canonical 8D topological minima of the PQT potential V_8:
# Coords: [x1, x2, x3, w1, w2, w3, w0, m]
CANONICAL_BASINS = np.array([
    # Basin 0: Synchronous negative well (E = 0.0531)
    [-1.0072, -1.0072, -1.0072,  1.3459,  1.3459,  1.3459,  1.3962, -0.6754],
    # Basins 1-6: Hexagonal cyclic minima (E = -0.5720)
    [-1.4220, -0.7340, -0.2140,  1.9650,  1.8220,  0.6430,  1.6010, -1.5220],
    [-0.7340, -0.2140, -1.4220,  1.8220,  0.6430,  1.9650,  1.6010, -1.5220],
    [ 0.2140,  1.4220,  0.7340,  0.6430,  1.9650,  1.8220,  1.6010, -1.5220],
    [-0.2140, -1.4220, -0.7340,  0.6430,  1.9650,  1.8220,  1.6010, -1.5220],
    [ 0.7340,  0.2140,  1.4220,  1.8220,  0.6430,  1.9650,  1.6010, -1.5220],
    [ 1.4220,  0.7340,  0.2140,  1.9650,  1.8220,  0.6430,  1.6010, -1.5220],
    # Basin 7: Synchronous positive well (E = 0.0531)
    [ 1.0072,  1.0072,  1.0072,  1.3459,  1.3459,  1.3459,  1.3962, -0.6754],
], dtype=np.float64)


def compute_q_ctx(
    z_ctx: np.ndarray,
    temperature: float = 0.75,
    basins: np.ndarray | None = None,
) -> np.ndarray:
    """Compute endogenous context routing distribution q_ctx(Z_ctx) over attractor basins.
    
    Zero oracle input: strictly derived from internal dynamical coordinates.
    
    Parameters
    ----------
    z_ctx : np.ndarray
        8D context state vector (e.g. from E5 Z_ctx_1). If 16D, uses first 8 coordinates.
    temperature : float
        Softmax temperature for distance weighting.
    basins : np.ndarray, optional
        Attractor basin coordinates (N_basins, 8). Defaults to CANONICAL_BASINS.
        
    Returns
    -------
    np.ndarray
        Normalized posterior probability vector of shape (N_basins,).
    """
    z_arr = np.asarray(z_ctx, dtype=np.float64)
    if not np.all(np.isfinite(z_arr)):
        raise NonFiniteStateError(f"Non-finite context state encountered in compute_q_ctx: {z_arr}")
        
    if z_arr.ndim == 1 and len(z_arr) >= 8:
        z_8d = z_arr[:8]
    else:
        raise ValueError(f"Expected at least 8 coordinates for context state, got {z_arr.shape}")
        
    basin_pts = CANONICAL_BASINS if basins is None else basins
    
    # Euclidean distance squared to each basin
    diffs = basin_pts - z_8d[np.newaxis, :]
    dist_sq = np.sum(diffs ** 2, axis=1)
    
    # Softmax over negative distance
    scaled = -dist_sq / (2.0 * (temperature ** 2))
    shifted = scaled - np.max(scaled)
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals)
    
    if sum_exp <= 0.0 or not np.isfinite(sum_exp):
        # Fallback to uniform if extreme underflow
        return np.ones(len(basin_pts), dtype=np.float64) / len(basin_pts)
        
    return exp_vals / sum_exp


def get_dominant_basin(
    z_ctx: np.ndarray,
    temperature: float = 0.75,
) -> tuple[int, float, float]:
    """Identify dominant basin index, confidence probability, and distance.
    
    Returns
    -------
    tuple[int, float, float]
        (dominant_index, posterior_probability, euclidean_distance)
    """
    q = compute_q_ctx(z_ctx, temperature=temperature)
    idx = int(np.argmax(q))
    prob = float(q[idx])
    dist = float(np.linalg.norm(z_ctx[:8] - CANONICAL_BASINS[idx]))
    return idx, prob, dist


def compute_energy_barrier(
    basin_idx_a: int,
    basin_idx_b: int,
    n_points: int = 50,
    params: CellParameters | None = None,
) -> dict[str, float]:
    """Measure energy barrier along the straight-line interpolation between two basins."""
    b_a = CANONICAL_BASINS[basin_idx_a]
    b_b = CANONICAL_BASINS[basin_idx_b]
    
    e_a = compute_energy(b_a, params=params)
    e_b = compute_energy(b_b, params=params)
    
    s_vals = np.linspace(0.0, 1.0, n_points)
    path_energies = []
    for s in s_vals:
        z_s = (1.0 - s) * b_a + s * b_b
        path_energies.append(compute_energy(z_s, params=params))
        
    e_max = float(np.max(path_energies))
    barrier_a = e_max - e_a
    barrier_b = e_max - e_b
    
    return {
        "energy_basin_a": float(e_a),
        "energy_basin_b": float(e_b),
        "energy_saddle_max": e_max,
        "barrier_from_a": float(barrier_a),
        "barrier_from_b": float(barrier_b),
    }
