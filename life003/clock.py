"""Independent monotonic counters for LIFE-003 multiscale time."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class LifeClock:
    """Monotonic clocks separating physical core integration from environment/deliberation."""
    core_tick: int = 0      # Physical core integration steps (dt = 0.01)
    world_tick: int = 0     # Discrete world interaction steps
    decision_tick: int = 0  # SANKALPA action deliberations
    feedback_tick: int = 0  # Verified outcome / feedback steps

    def advance_core(self, ticks: int = 1) -> int:
        if ticks < 0:
            raise ValueError("Ticks to advance cannot be negative")
        self.core_tick += ticks
        return self.core_tick

    def advance_world(self, ticks: int = 1) -> int:
        if ticks < 0:
            raise ValueError("Ticks to advance cannot be negative")
        self.world_tick += ticks
        return self.world_tick

    def advance_decision(self, ticks: int = 1) -> int:
        if ticks < 0:
            raise ValueError("Ticks to advance cannot be negative")
        self.decision_tick += ticks
        return self.decision_tick

    def advance_feedback(self, ticks: int = 1) -> int:
        if ticks < 0:
            raise ValueError("Ticks to advance cannot be negative")
        self.feedback_tick += ticks
        return self.feedback_tick

    def snapshot(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def restore(cls, data: dict[str, Any]) -> LifeClock:
        return cls(
            core_tick=int(data["core_tick"]),
            world_tick=int(data["world_tick"]),
            decision_tick=int(data["decision_tick"]),
            feedback_tick=int(data["feedback_tick"]),
        )
