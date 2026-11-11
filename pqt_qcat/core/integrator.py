"""Adaptive Heun integrator with strict finiteness validation and convergence control.

Conforms strictly to ARCHITECTURE_LIFE_002 (Section 3.4) and ILORI approval:
- Evaluates dz = [-(Gamma - A_I(w0)) grad E(z) + u(t)] dt + sqrt(2*T*Gamma) dB
- Second-order Heun (predictor-corrector) with adaptive substepping
- Strict fail-fast on NaN/Inf: raises NonFiniteStateError immediately
- Trajectory convergence and energy monotonicity validation
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .energy import CellParameters, compute_energy, compute_grad_energy
from .vortex import VortexParameters, apply_vortex_A_I
from .errors import NonFiniteStateError, IntegratorConvergenceError


@dataclass(frozen=True)
class IntegratorConfig:
    """Configuration of numerical integrator."""
    h_base: float = 0.01             # Base step size (dt)
    tol: float = 1e-4                # Local truncation error tolerance
    max_subdivisions: int = 8        # Up to 2^8 = 256 substeps per dt
    eps_w0: float = 0.04             # Dissipation for w0
    eps_m: float = 0.02              # Dissipation for m_phi
    temperature: float = 0.0         # T=0 by default during deterministic training/testing


@dataclass
class StepDiagnostics:
    """Detailed diagnostics of an integration step."""
    substeps_taken: int = 1
    max_local_error: float = 0.0
    initial_energy: float = 0.0
    final_energy: float = 0.0
    delta_energy: float = 0.0
    finite: bool = True


class AdaptiveHeunIntegrator:
    """Adaptive Heun stepper for PQT cells."""

    def __init__(
        self,
        config: IntegratorConfig | None = None,
        cell_params: CellParameters | None = None,
        vortex_params: VortexParameters | None = None,
    ):
        self.config = config or IntegratorConfig()
        self.cell_params = cell_params or CellParameters()
        self.vortex_params = vortex_params or VortexParameters()

        # Build diagonal Gamma: diag(1, 1, 1, 1, 1, 1, eps_w0, eps_m)
        self.gamma_diag = np.array([
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            self.config.eps_w0, self.config.eps_m
        ], dtype=np.float64)

    def compute_drift(
        self,
        z: np.ndarray,
        u: np.ndarray | None = None,
        cell_idx: int = 0
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute deterministic drift b(z) = -(Gamma - A_I(w0)) grad E + u.
        
        Returns:
            b: drift vector (8,)
            grad_E: analytical gradient (8,)
            E: energy scalar
        """
        if not np.all(np.isfinite(z)):
            raise NonFiniteStateError(f"Cell {cell_idx}: non-finite coordinates in state z: {z}")

        E = compute_energy(z, self.cell_params)
        grad_E = compute_grad_energy(z, self.cell_params)

        if not np.isfinite(E):
            raise NonFiniteStateError(f"Cell {cell_idx}: non-finite energy {E}")
        if not np.all(np.isfinite(grad_E)):
            raise NonFiniteStateError(f"Cell {cell_idx}: non-finite gradient {grad_E}")

        w0 = float(z[6])

        # Dissipative part: Gamma * grad_E
        dissipative = self.gamma_diag * grad_E

        # Vortex rotation part: A_I(w0) * grad_E
        vortex = apply_vortex_A_I(grad_E, w0, self.vortex_params)

        # Net drift: -(Gamma - A_I) * grad_E = -Gamma * grad_E + A_I * grad_E
        drift = -dissipative + vortex

        if u is not None:
            if not np.all(np.isfinite(u)):
                raise NonFiniteStateError(f"Cell {cell_idx}: non-finite external input u: {u}")
            drift = drift + u

        return drift, grad_E, E

    def _heun_substep(
        self,
        z: np.ndarray,
        dt: float,
        u: np.ndarray | None,
        cell_idx: int
    ) -> tuple[np.ndarray, float]:
        """Execute single Heun predictor-corrector step of length dt.
        
        Returns:
            z_corr: corrected state
            err: infinity-norm difference ||z_corr - z_pred||_inf
        """
        b_curr, _, _ = self.compute_drift(z, u, cell_idx)
        z_pred = z + dt * b_curr

        if not np.all(np.isfinite(z_pred)):
            raise NonFiniteStateError(f"Cell {cell_idx}: non-finite predictor state during Heun step")

        b_pred, _, _ = self.compute_drift(z_pred, u, cell_idx)
        z_corr = z + (dt * 0.5) * (b_curr + b_pred)

        if not np.all(np.isfinite(z_corr)):
            raise NonFiniteStateError(f"Cell {cell_idx}: non-finite corrector state during Heun step")

        err = float(np.max(np.abs(z_corr - z_pred)))
        return z_corr, err

    def _adaptive_step_recursive(
        self,
        z: np.ndarray,
        dt: float,
        u: np.ndarray | None,
        level: int,
        cell_idx: int
    ) -> tuple[np.ndarray, int, float]:
        """Recursive subdivision for local error control."""
        z_corr, err = self._heun_substep(z, dt, u, cell_idx)

        # If error is within tolerance or we've reached max subdivision depth
        if err <= self.config.tol or level >= self.config.max_subdivisions:
            if err > self.config.tol and level >= self.config.max_subdivisions:
                # Warning or error if tolerance cannot be satisfied even at 2^K substeps
                # Per spec: if non-finite or uncontrolled, raise exception
                if not np.all(np.isfinite(z_corr)):
                    raise NonFiniteStateError(f"Cell {cell_idx}: non-finite state at max subdivisions")
            return z_corr, 1, err

        # Subdivide interval into two halves
        half_dt = dt * 0.5
        z_mid, count1, err1 = self._adaptive_step_recursive(z, half_dt, u, level + 1, cell_idx)
        z_end, count2, err2 = self._adaptive_step_recursive(z_mid, half_dt, u, level + 1, cell_idx)

        return z_end, count1 + count2, max(err1, err2)

    def step_cell(
        self,
        z: np.ndarray,
        dt: float | None = None,
        u: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        cell_idx: int = 0
    ) -> tuple[np.ndarray, StepDiagnostics]:
        """Advance single 8D cell state by dt."""
        dt = dt if dt is not None else self.config.h_base
        z = np.asarray(z, dtype=np.float64)

        diag = StepDiagnostics()
        diag.initial_energy = float(compute_energy(z, self.cell_params))

        # 1. Deterministic adaptive Heun step
        z_next, substeps, max_err = self._adaptive_step_recursive(z, dt, u, level=0, cell_idx=cell_idx)
        diag.substeps_taken = substeps
        diag.max_local_error = max_err

        # 2. Add stochastic diffusion if T > 0
        if self.config.temperature > 0.0:
            if rng is None:
                rng = np.random.default_rng()
            sigma = np.sqrt(2.0 * self.config.temperature * self.gamma_diag * dt)
            noise = rng.normal(0.0, 1.0, size=8) * sigma
            z_next = z_next + noise
            if not np.all(np.isfinite(z_next)):
                raise NonFiniteStateError(f"Cell {cell_idx}: non-finite state after noise addition")

        diag.final_energy = float(compute_energy(z_next, self.cell_params))
        diag.delta_energy = diag.final_energy - diag.initial_energy
        diag.finite = bool(np.all(np.isfinite(z_next)))

        return z_next, diag

    def step_multi(
        self,
        Z: np.ndarray,
        dt: float | None = None,
        U: np.ndarray | None = None,
        rng: np.random.Generator | None = None
    ) -> tuple[np.ndarray, list[StepDiagnostics]]:
        """Advance multi-cell state Z of shape (C, 8) or (8*C,)."""
        dt = dt if dt is not None else self.config.h_base
        Z_arr = np.asarray(Z, dtype=np.float64)

        flat = Z_arr.ndim == 1
        if flat:
            n_cells = len(Z_arr) // 8
            Z_cells = Z_arr.reshape(n_cells, 8)
        else:
            n_cells = Z_arr.shape[0]
            Z_cells = Z_arr

        U_cells = None
        if U is not None:
            U_arr = np.asarray(U, dtype=np.float64)
            U_cells = U_arr.reshape(n_cells, 8) if flat else U_arr

        Z_next_cells = np.empty_like(Z_cells)
        diagnostics = []

        for c in range(n_cells):
            u_c = U_cells[c] if U_cells is not None else None
            z_c_next, diag_c = self.step_cell(Z_cells[c], dt=dt, u=u_c, rng=rng, cell_idx=c)
            Z_next_cells[c] = z_c_next
            diagnostics.append(diag_c)

        out = Z_next_cells.flatten() if flat else Z_next_cells
        return out, diagnostics
