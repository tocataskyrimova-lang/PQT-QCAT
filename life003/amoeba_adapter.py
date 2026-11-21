"""Deterministic Body & Adapter for Digital Amoeba (Brain/World API)."""
from __future__ import annotations
import copy
from typing import Any
import numpy as np


class AmoebaWorld:
    """Deterministic 2D grid world with homeostasis, hazards, food, and sensory tokens."""

    def __init__(self, size: int = 5, seed: int = 2026):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.agent_pos = [size // 2, size // 2]
        self.energy = 1.0
        self.food_pos = [0, 0]
        self.hazard_pos = [size - 1, size - 1]
        self.step_count = 0
        self._spawn_items()

    def _spawn_items(self) -> None:
        """Place food and hazard deterministically."""
        self.food_pos = [int(self.rng.integers(0, self.size)), int(self.rng.integers(0, self.size))]
        self.hazard_pos = [int(self.rng.integers(0, self.size)), int(self.rng.integers(0, self.size))]
        while self.hazard_pos == self.food_pos or self.hazard_pos == self.agent_pos:
            self.hazard_pos = [int(self.rng.integers(0, self.size)), int(self.rng.integers(0, self.size))]

    def get_observation(self) -> np.ndarray:
        """Return 8D sensory vector without oracle leakage of optimal action."""
        obs = np.zeros(8, dtype=np.float64)
        # Encode relative quadrant and proximity
        dr = self.food_pos[0] - self.agent_pos[0]
        dc = self.food_pos[1] - self.agent_pos[1]
        
        # Token 1..4 indicates primary directional cue
        if abs(dr) >= abs(dc):
            obs[1 if dr < 0 else 2] = 1.0
        else:
            obs[3 if dc < 0 else 4] = 1.0

        # Sensor 5: proximity cue
        dist = float(abs(dr) + abs(dc))
        obs[5] = float(np.clip(1.0 / (dist + 1.0), 0.0, 1.0))
        # Sensor 6: hazard proximity cue
        dist_h = float(abs(self.hazard_pos[0] - self.agent_pos[0]) + abs(self.hazard_pos[1] - self.agent_pos[1]))
        obs[6] = float(np.clip(1.0 / (dist_h + 1.0), 0.0, 1.0))
        obs[7] = float(self.energy)
        return obs

    def step(self, action: int | str) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Advance world by one discrete action step."""
        self.step_count += 1
        reward = -0.05  # Base metabolism cost
        self.energy = max(0.0, self.energy - 0.01)

        if action == 1 or action == "UP":
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 2 or action == "DOWN":
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        elif action == 3 or action == "LEFT":
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 4 or action == "RIGHT":
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        elif action in ("WAIT", "IDLE"):
            pass  # No movement
        elif action == "PROBE":
            reward -= 0.02  # Specific probing cost
        else:
            raise ValueError(f"Unknown action: {action}")

        # Check interaction
        if self.agent_pos == self.food_pos:
            reward += 1.0
            self.energy = min(1.0, self.energy + 0.3)
            self._spawn_items()
        elif self.agent_pos == self.hazard_pos:
            reward -= 0.5
            self.energy = max(0.0, self.energy - 0.2)

        done = self.energy <= 0.0 or self.step_count >= 10000
        obs_next = self.get_observation()
        info = {
            "energy": self.energy,
            "agent_pos": list(self.agent_pos),
            "step_count": self.step_count,
        }
        return obs_next, reward, done, info

    def snapshot(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "agent_pos": list(self.agent_pos),
            "energy": float(self.energy),
            "food_pos": list(self.food_pos),
            "hazard_pos": list(self.hazard_pos),
            "step_count": int(self.step_count),
        }

    @classmethod
    def restore(cls, data: dict[str, Any]) -> AmoebaWorld:
        world = cls(size=int(data["size"]))
        world.rng.bit_generator.state = copy.deepcopy(data["rng_state"])
        world.agent_pos = list(data["agent_pos"])
        world.energy = float(data["energy"])
        world.food_pos = list(data["food_pos"])
        world.hazard_pos = list(data["hazard_pos"])
        world.step_count = int(data["step_count"])
        return world
