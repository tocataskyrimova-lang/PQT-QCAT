"""PQT-QCat Core Mathematical Engine: Energy, Vortex operator, and Integrator."""
from .errors import NonFiniteStateError, IntegratorConvergenceError
from .energy import CellParameters, compute_energy, compute_grad_energy, grad_energy_autograd
from .vortex import VortexParameters, build_vortex_A_I, vortex_latch_gain, J3
from .integrator import IntegratorConfig, AdaptiveHeunIntegrator, StepDiagnostics

__all__ = [
    "CellParameters",
    "compute_energy",
    "compute_grad_energy",
    "grad_energy_autograd",
    "VortexParameters",
    "build_vortex_A_I",
    "vortex_latch_gain",
    "J3",
    "IntegratorConfig",
    "AdaptiveHeunIntegrator",
    "NonFiniteStateError",
    "IntegratorConvergenceError",
    "StepDiagnostics",
]
