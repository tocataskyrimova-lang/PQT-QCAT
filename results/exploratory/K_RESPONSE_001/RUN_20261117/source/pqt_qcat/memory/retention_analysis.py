"""Rigorous Numerical & Physical Investigation of Gap Retention (Requirement 5).

Provides analytical tools to quantify:
1. Discretization drift across integrator step sizes h in {0.01, 0.005, 0.002, 0.001} over 1000 idle ticks
2. Energy barrier profile between canonical topological attractor basins
3. Perturbation relaxation and basin basin-of-attraction radius under Gaussian noise sigma in [0.01, 0.5]
4. Contrast between heuristic non-equilibrium initialization vs canonical topological minima
"""
from __future__ import annotations
import numpy as np

from .basins import CANONICAL_BASINS, compute_energy_barrier
from ..core.energy import compute_energy
from ..core.integrator import AdaptiveHeunIntegrator, IntegratorConfig


def analyze_step_refinement_drift(
    basin_idx: int = 1,
    h_values: list[float] = [0.01, 0.005, 0.002, 0.001],
    total_physical_time: float = 10.0,  # equivalent to 1000 ticks at h=0.01
) -> dict[float, dict[str, float]]:
    """Measure exact numerical drift from a canonical minimum across refined integration steps."""
    b_target = CANONICAL_BASINS[basin_idx].copy()
    results = {}
    
    for h in h_values:
        cfg = IntegratorConfig(mode="fixed", h_base=h)
        integrator = AdaptiveHeunIntegrator(config=cfg)
        
        z = b_target.copy()
        n_steps = int(round(total_physical_time / h))
        
        for _ in range(n_steps):
            z, _ = integrator.step_cell(z, dt=h, u=np.zeros(8))
            
        drift_norm = float(np.linalg.norm(z - b_target))
        energy_final = float(compute_energy(z))
        
        results[h] = {
            "steps": n_steps,
            "drift_norm": drift_norm,
            "energy_final": energy_final,
            "stable_in_basin": bool(drift_norm < 0.1),
        }
        
    return results


def analyze_perturbation_resilience(
    basin_idx: int = 1,
    sigmas: list[float] = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
    trials_per_sigma: int = 25,
    relaxation_steps: int = 300,
    h: float = 0.01,
    seed: int = 42,
) -> dict[float, float]:
    """Measure the recovery rate of a canonical attractor well under random perturbations."""
    b_target = CANONICAL_BASINS[basin_idx].copy()
    cfg = IntegratorConfig(mode="fixed", h_base=h)
    integrator = AdaptiveHeunIntegrator(config=cfg)
    rng = np.random.default_rng(seed)
    
    recovery_rates = {}
    for sigma in sigmas:
        recovered = 0
        for _ in range(trials_per_sigma):
            noise = rng.normal(0.0, sigma, size=8)
            z_perturbed = b_target + noise
            
            z = z_perturbed.copy()
            for _ in range(relaxation_steps):
                z, _ = integrator.step_cell(z, dt=h, u=np.zeros(8))
                
            # If relaxed within 0.15 of true minimum, classified as recovered
            if np.linalg.norm(z - b_target) < 0.15:
                recovered += 1
                
        recovery_rates[sigma] = recovered / trials_per_sigma
        
    return recovery_rates


def analyze_all_energy_barriers() -> dict[str, dict[str, float]]:
    """Compute energy barrier profile between adjacent canonical hexagonal basins (1->2, 2->3, etc.)."""
    barriers = {}
    pairs = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1), (0, 7), (1, 7)]
    for a, b in pairs:
        barriers[f"basin_{a}_to_{b}"] = compute_energy_barrier(a, b, n_points=50)
    return barriers
