"""Adaptive Heun integrator with strict finiteness validation, convergence control, and energy monotonicity.

Mathematical Schemes:
1. Deterministic Heun Stepper (T = 0):
   Second-order Runge-Kutta / Heun predictor-corrector:
     b(z) = -(Gamma - A_I(w0)) grad E(z) + u(t)
     z_pred = z + dt * b(z)
     z_corr = z + (dt / 2) * [b(z) + b(z_pred)]
   Local truncation error: err = ||z_corr - z_pred||_inf.
   In 'adaptive' mode, if err > tol, the step is recursively bisected up to
   max_subdivisions levels. If tolerance cannot be met within max_subdivisions,
   an IntegratorConvergenceError is strictly raised (M1-FIX-001).
   In 'fixed' mode, a single Heun step of size dt is executed without substepping.

2. Stochastic Langevin SDE Stepper (T > 0):
   Predictor-corrector Heun for drift plus Euler-Maruyama diffusion:
     z(t + dt) = z_corr + sqrt(2 * T * Gamma * dt) * eta,  eta ~ N(0, I).
   Note: Diffusion is evaluated with fixed noise increments at the macro step dt.
   Formal Gibbs stationarity holds for closed continuous-time systems (u=0, fixed parameters);
   finite dt introduces standard O(dt) discretization bias in stationary distributions.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np

from .energy import CellParameters, compute_energy, compute_grad_energy
from .vortex import VortexParameters, apply_vortex_A_I
from .errors import NonFiniteStateError, IntegratorConvergenceError


@dataclass(frozen=True)
class IntegratorConfig:
    """Configuration of numerical integrator."""
    mode: Literal["adaptive", "fixed"] = "adaptive"  # 'adaptive' or 'fixed'
    h_base: float = 0.01                              # Base step size (dt)
    tol: float = 1e-4                                 # Local truncation error tolerance
    max_subdivisions: int = 8                         # Up to 2^8 = 256 substeps per dt
    eps_w0: float = 0.04                              # Dissipation for w0
    eps_m: float = 0.02                               # Dissipation for m_phi
    temperature: float = 0.0                          # T=0 by default during deterministic operation
    max_u_norm: float | None = 10.0                   # Max norm for external forcing ||u||_2
    clip_u: bool = True                               # Whether to clip external input norm
    enforce_monotonic_energy: bool = True             # Check dE <= 0 in closed system (u=0, T=0)
    mono_energy_tol: float = 1e-6                     # Numerical slack for discrete energy monotonicity


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

        # Diagonal dissipation Gamma = diag(1, 1, 1, 1, 1, 1, eps_w0, eps_m)
        self.gamma_diag = np.array([
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            self.config.eps_w0, self.config.eps_m
        ], dtype=np.float64)

    def _sanitize_input(self, u: np.ndarray | None, cell_idx: int) -> np.ndarray | None:
        """Validate and optionally clip external input u."""
        if u is None:
            return None
        u_arr = np.asarray(u, dtype=np.float64)
        if u_arr.shape != (8,):
            raise ValueError(f"Cell {cell_idx}: invalid input u shape {u_arr.shape}, expected (8,)")
        if not np.all(np.isfinite(u_arr)):
            raise NonFiniteStateError(f"Cell {cell_idx}: non-finite external input u: {u_arr}")

        norm_u = float(np.linalg.norm(u_arr))
        if self.config.max_u_norm is not None and norm_u > self.config.max_u_norm:
            if self.config.clip_u:
                u_arr = u_arr * (self.config.max_u_norm / norm_u)
            else:
                raise ValueError(
                    f"Cell {cell_idx}: input norm {norm_u:.4f} exceeds limit {self.config.max_u_norm:.4f}"
                )
        return u_arr

    def compute_drift(
        self,
        z: np.ndarray,
        u: np.ndarray | None = None,
        cell_idx: int = 0
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute deterministic drift b(z) = -(Gamma - A_I(w0)) grad E + u."""
        if z.shape != (8,):
            raise ValueError(f"Cell {cell_idx}: invalid state shape {z.shape}, expected (8,)")
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
            drift = drift + u

        return drift, grad_E, E

    def _heun_substep(
        self,
        z: np.ndarray,
        dt: float,
        u: np.ndarray | None,
        cell_idx: int
    ) -> tuple[np.ndarray, float]:
        """Execute single Heun predictor-corrector step of length dt."""
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
        """Recursive subdivision for local error control with strict convergence gate."""
        z_corr, err = self._heun_substep(z, dt, u, cell_idx)

        # If error is within tolerance, step is accepted
        if err <= self.config.tol:
            return z_corr, 1, err

        # M1-FIX-001: If max subdivision depth reached and error still exceeds tolerance -> STRICT FAIL
        if level >= self.config.max_subdivisions:
            raise IntegratorConvergenceError(
                f"Cell {cell_idx}: local truncation error {err:.4e} exceeds tol {self.config.tol:.4e} "
                f"at max subdivisions {self.config.max_subdivisions} (dt={dt})"
            )

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
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"Cell {cell_idx}: invalid dt={dt}. Must be finite and strictly positive.")

        z = np.asarray(z, dtype=np.float64)
        if z.shape != (8,):
            raise ValueError(f"Cell {cell_idx}: invalid state shape {z.shape}, expected (8,)")

        u_sanitized = self._sanitize_input(u, cell_idx)

        diag = StepDiagnostics()
        diag.initial_energy = float(compute_energy(z, self.cell_params))

        # 1. Choose stepper mode: adaptive or fixed
        if self.config.mode == "fixed":
            z_next, max_err = self._heun_substep(z, dt, u_sanitized, cell_idx)
            substeps = 1
        elif self.config.mode == "adaptive":
            z_next, substeps, max_err = self._adaptive_step_recursive(
                z, dt, u_sanitized, level=0, cell_idx=cell_idx
            )
        else:
            raise ValueError(f"Unknown integrator mode '{self.config.mode}'. Use 'adaptive' or 'fixed'.")

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

        # 3. Monotonic energy check in closed deterministic system (u=0, T=0)
        if (
            self.config.enforce_monotonic_energy
            and u_sanitized is None
            and self.config.temperature == 0.0
            and diag.delta_energy > self.config.mono_energy_tol
        ):
            raise IntegratorConvergenceError(
                f"Cell {cell_idx}: discrete energy increased by {diag.delta_energy:.4e} "
                f"(initial={diag.initial_energy:.4f}, final={diag.final_energy:.4f}) in closed system"
            )

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
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"Invalid dt={dt}. Must be finite and strictly positive.")

        Z_arr = np.asarray(Z, dtype=np.float64)
        flat = Z_arr.ndim == 1
        if flat:
            if len(Z_arr) % 8 != 0:
                raise ValueError(f"State vector length {len(Z_arr)} is not divisible by 8.")
            n_cells = len(Z_arr) // 8
            Z_cells = Z_arr.reshape(n_cells, 8)
        else:
            if Z_arr.ndim != 2 or Z_arr.shape[1] != 8:
                raise ValueError(f"Multi-cell array shape {Z_arr.shape} must be (C, 8).")
            n_cells = Z_arr.shape[0]
            Z_cells = Z_arr

        U_cells = None
        if U is not None:
            U_arr = np.asarray(U, dtype=np.float64)
            if flat:
                if len(U_arr) != len(Z_arr):
                    raise ValueError(f"Input U length {len(U_arr)} does not match state {len(Z_arr)}.")
                U_cells = U_arr.reshape(n_cells, 8)
            else:
                if U_arr.shape != (n_cells, 8):
                    raise ValueError(f"Input U shape {U_arr.shape} does not match (C, 8).")
                U_cells = U_arr

        Z_next_cells = np.empty_like(Z_cells)
        diagnostics = []

        for c in range(n_cells):
            u_c = U_cells[c] if U_cells is not None else None
            z_c_next, diag_c = self.step_cell(Z_cells[c], dt=dt, u=u_c, rng=rng, cell_idx=c)
            Z_next_cells[c] = z_c_next
            diagnostics.append(diag_c)

        out = Z_next_cells.flatten() if flat else Z_next_cells
        return out, diagnostics
