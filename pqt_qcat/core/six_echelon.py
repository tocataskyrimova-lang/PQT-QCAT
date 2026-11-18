"""Six-Echelon 48D Dynamic Core (PQT-HN002-TRI-003).

Conforms strictly to ARCHITECTURE_LIFE_002 (Section 2 & 2.1) and ILORI approval:
- 6 functional cells x 8D coordinates = 48D continuous dynamic core:
    E1 (idx 0): Z_sens_1 - primary sensory receiver
    E2 (idx 1): Z_sens_2 - prediction residual & trace
    E3 (idx 2): Z_hold_1 - write-gated cue latch (gated by write_gate)
    E4 (idx 3): Z_hold_2 - deep protected retention with low mobility Gamma_hold
    E5 (idx 4): Z_ctx_1  - endogenous context attractor basins
    E6 (idx 5): Z_ctx_2  - rule shift detector & exploration trigger
- Inter-cell coupling matrix K in R^(48x48) with norm bounding ||u_i|| <= U_max
- Parameter theta_core = {K, B} separates core plasticity from readout
- Full ablation support: enable_vortex=False for exact A=0 (noA) comparison
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .energy import CellParameters, compute_energy, compute_grad_energy
from .vortex import VortexParameters, apply_vortex_A_I
from .integrator import AdaptiveHeunIntegrator, IntegratorConfig, StepDiagnostics
from .errors import NonFiniteStateError
from .security import check_seed_safety


@dataclass
class SixEchelonConfig:
    """Configuration of the 6-echelon 48D core."""
    d_in: int = 8                      # Dimension of external input x_t (8 one-hot tokens)
    d_out: int = 8                     # Dimension of output target
    u_max: float = 5.0                 # Maximum norm of inter-cell current ||u_i||_2
    hold_mobility_scale: float = 0.001 # Reduced mobility Gamma_hold for E4 deep retention
    enable_vortex: bool = True         # True for PQT (A != 0), False for noA ablation (A = 0)
    beta_v: float = 4.0                # Vortex latch sigmoid slope
    h_base: float = 0.01               # Base integration step
    integrator_mode: str = "fixed"     # 'fixed' or 'adaptive'
    integrator_tol: float = 1e-4       # Local truncation error tolerance


class SixEchelonCore:
    """Six-Echelon 48D PQT-QCat continuous core."""

    def __init__(
        self,
        config: SixEchelonConfig | None = None,
        cell_params: CellParameters | None = None,
        vortex_params: VortexParameters | None = None,
        seed: int = 42,
    ):
        self.config = config or SixEchelonConfig()
        if seed is not None:
            check_seed_safety(seed, allow_quarantined=True)
        self.cell_params = cell_params or CellParameters()
        self.vortex_params = vortex_params or VortexParameters(beta_v=self.config.beta_v)
        self.rng = np.random.default_rng(seed)

        # 48D state: 6 cells x 8 coordinates
        self.Z = np.zeros(48, dtype=np.float64)
        self._init_state()

        # Structural Core Parameters theta_core = {K, B}
        self.K = np.zeros((48, 48), dtype=np.float64)
        self.B = np.zeros((48, self.config.d_in), dtype=np.float64)
        self._init_core_parameters()

        # Frozen Readout Projection W_readout (d_out, 48)
        self.W_readout = np.zeros((self.config.d_out, 48), dtype=np.float64)
        self._init_readout()

        # Steppers
        int_cfg = IntegratorConfig(
            mode=self.config.integrator_mode,
            h_base=self.config.h_base,
            tol=self.config.integrator_tol,
            enforce_monotonic_energy=False, # open system
        )
        vparams_init = (
            self.vortex_params if self.config.enable_vortex
            else VortexParameters(0.0, 0.0, 0.0, beta_v=self.config.beta_v)
        )
        self.integrator = AdaptiveHeunIntegrator(
            config=int_cfg,
            cell_params=self.cell_params,
            vortex_params=vparams_init,
        )

        # Dissipation matrices: normal vs hold
        self.gamma_diag_normal = self.integrator.gamma_diag.copy()
        self.gamma_diag_hold = self.gamma_diag_normal.copy() * self.config.hold_mobility_scale

    def _init_state(self) -> None:
        """Initialize 48D state with stable latch configuration."""
        self.Z.fill(0.0)
        for c in range(6):
            idx = 8 * c
            # w0 set to +1.0 (Latch phase) for stability
            self.Z[idx+6] = 1.0

    def _init_core_parameters(self) -> None:
        """Initialize default structural inter-cell coupling K and sensory projection B."""
        scale_b = 1.0 / np.sqrt(self.config.d_in)
        self.B[0:8, :] = self.rng.normal(0.0, scale_b, size=(8, self.config.d_in))
        self.B[8:16, :] = self.rng.normal(0.0, scale_b, size=(8, self.config.d_in))

        scale_k = 0.25
        self.K[16:24, 0:8] = self.rng.normal(0.0, scale_k, size=(8, 8))
        self.K[24:32, 16:24] = np.eye(8) * 0.5
        self.K[32:40, 24:32] = self.rng.normal(0.0, scale_k, size=(8, 8))
        self.K[40:48, 8:16] = self.rng.normal(0.0, scale_k, size=(8, 8))
        self.K[32:40, 40:48] = self.rng.normal(0.0, scale_k, size=(8, 8))

    def _init_readout(self) -> None:
        """Initialize frozen readout matrix W_readout (d_out, 48)."""
        scale_r = 1.0 / np.sqrt(16)
        self.W_readout[:, 24:40] = self.rng.normal(0.0, scale_r, size=(self.config.d_out, 16))

    def compute_currents(
        self,
        x_t: np.ndarray | None,
        write_gate: float = 1.0,
    ) -> np.ndarray:
        """Compute inter-cell currents u in R^48 with strict norm bounding and write gating."""
        psi = np.tanh(self.Z)
        u_inter = self.K @ psi

        # Apply write_gate to the Hold compartment (E3 and E4, indices 16:32):
        # When write_gate = 0 (idle/pause), Hold compartment is completely isolated from write currents!
        u_inter[16:32] *= write_gate

        # External sensory drive: u_ext = B @ x_t
        u_ext = np.zeros(48, dtype=np.float64)
        if x_t is not None:
            x_arr = np.asarray(x_t, dtype=np.float64)
            u_ext = self.B @ x_arr

        # External drive into Hold is also governed by write_gate
        u_ext[16:32] *= write_gate

        u_total = u_inter + u_ext

        # Norm-bounding per cell: ||u_c|| <= u_max
        u_bounded = np.empty_like(u_total)
        for c in range(6):
            idx = 8 * c
            u_c = u_total[idx:idx+8]
            norm_c = float(np.linalg.norm(u_c))
            if norm_c > self.config.u_max:
                u_bounded[idx:idx+8] = u_c * (self.config.u_max / norm_c)
            else:
                u_bounded[idx:idx+8] = u_c

        return u_bounded

    def step(
        self,
        x_t: np.ndarray | None = None,
        write_gate: float = 1.0,
        dt: float | None = None,
    ) -> tuple[np.ndarray, list[StepDiagnostics]]:
        """Advance the full 48D core by physical time dt."""
        dt = dt if dt is not None else self.config.h_base
        u_48 = self.compute_currents(x_t, write_gate=write_gate)

        Z_next = np.empty(48, dtype=np.float64)
        diags = []

        for c in range(6):
            idx = 8 * c
            z_c = self.Z[idx:idx+8]
            u_c = u_48[idx:idx+8]

            # E4 (Hold-Deep, idx 3) uses reduced mobility Gamma_hold
            self.integrator.gamma_diag = (
                self.gamma_diag_hold if c == 3 else self.gamma_diag_normal
            )

            # Vortex parameters handling (noA ablation vs PQT)
            if not self.config.enable_vortex:
                vparams_c = VortexParameters(0.0, 0.0, 0.0, beta_v=self.config.beta_v)
            elif c == 3 and write_gate == 0.0:
                # In pause, Hold-Deep operates in low-vortex regime per specification
                vparams_c = VortexParameters(0.1, 0.1, 0.0, beta_v=self.config.beta_v)
            else:
                vparams_c = self.vortex_params

            self.integrator.vortex_params = vparams_c
            z_c_next, diag_c = self.integrator.step_cell(
                z_c, dt=dt, u=u_c, cell_idx=c, vortex_params=vparams_c
            )

            Z_next[idx:idx+8] = z_c_next
            diags.append(diag_c)

        self.Z = Z_next
        return self.Z.copy(), diags

    def compute_cell_vortex_force(self, cell_idx: int) -> np.ndarray:
        """Compute the vortex force A_I(w0) * grad_E for cell c given current state Z."""
        idx = 8 * cell_idx
        z_c = self.Z[idx:idx+8]
        if not self.config.enable_vortex:
            vparams = VortexParameters(0.0, 0.0, 0.0, beta_v=self.config.beta_v)
        else:
            vparams = self.vortex_params
        return self.integrator.compute_vortex_force(z_c, vortex_params=vparams)

    def predict(self) -> np.ndarray:
        """Compute output logits via frozen readout projection W_readout @ Z."""
        logits = self.W_readout @ self.Z
        if not np.all(np.isfinite(logits)):
            raise NonFiniteStateError(f"Non-finite logits produced in readout: {logits}")
        return logits

    def get_cell_state(self, cell_idx: int) -> np.ndarray:
        """Return 8D state vector of cell c."""
        idx = 8 * cell_idx
        return self.Z[idx:idx+8].copy()

    def snapshot(self) -> dict:
        """Export full persistent state for exact deterministic save/resume."""
        return {
            "Z": self.Z.copy(),
            "K": self.K.copy(),
            "B": self.B.copy(),
            "W_readout": self.W_readout.copy(),
            "enable_vortex": self.config.enable_vortex,
            "h_base": self.config.h_base,
            "beta_v": self.config.beta_v,
            "u_max": self.config.u_max,
            "hold_mobility_scale": self.config.hold_mobility_scale,
            "integrator_mode": self.config.integrator_mode,
            "rng_state": self.rng.bit_generator.state,
        }

    def restore(self, snap: dict, strict_config: bool = True) -> None:
        """Restore exact persistent state.
        
        If strict_config is True, raises ValueError if essential configuration (e.g. h_base) differs.
        If strict_config is False, synchronizes configuration parameters to match the snapshot.
        """
        if "h_base" in snap and not np.isclose(snap["h_base"], self.config.h_base):
            if strict_config:
                raise ValueError(
                    f"Incompatible core configuration: snapshot h_base={snap['h_base']} != current {self.config.h_base}"
                )
            self.config.h_base = snap["h_base"]

        self.Z = snap["Z"].copy()
        self.K = snap["K"].copy()
        self.B = snap["B"].copy()
        self.W_readout = snap["W_readout"].copy()
        if "enable_vortex" in snap:
            self.config.enable_vortex = snap["enable_vortex"]
        if "beta_v" in snap and not strict_config:
            self.config.beta_v = snap["beta_v"]
        if "u_max" in snap and not strict_config:
            self.config.u_max = snap["u_max"]
        self.rng.bit_generator.state = snap["rng_state"]
