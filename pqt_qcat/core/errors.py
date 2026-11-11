"""Exceptions for PQT-QCat core mathematical engine."""

class NonFiniteStateError(FloatingPointError, ValueError):
    """Raised when NaN or Inf is encountered in state, energy, gradient, or step."""
    pass


class IntegratorConvergenceError(RuntimeError):
    """Raised when adaptive substepping exceeds max subdivision limit."""
    pass
