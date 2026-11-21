"""Endogenous thought trigger for LIFE-003 VIMARSA deliberation."""
from __future__ import annotations
import numpy as np
from typing import Any


class EndogenousTrigger:
    """Computes endogenous deliberation trigger score from internal states."""

    def __init__(
        self,
        w_surprise: float = 0.35,
        w_uncertainty: float = 0.35,
        w_energy_deficit: float = 0.15,
        w_unresolved: float = 0.15,
        threshold_theta: float = 0.45,
        norm_scale_surprise: float = 2.5,
        cooldown_ticks: int = 20,
        max_total_deliberations: int = 150,
    ):
        self.w_surprise = float(w_surprise)
        self.w_uncertainty = float(w_uncertainty)
        self.w_energy_deficit = float(w_energy_deficit)
        self.w_unresolved = float(w_unresolved)
        self.threshold_theta = float(threshold_theta)
        self.norm_scale_surprise = float(norm_scale_surprise)
        self.cooldown_ticks = int(cooldown_ticks)
        self.max_total_deliberations = int(max_total_deliberations)

        self.last_trigger_tick: int = -10000
        self.total_triggers_count: int = 0

    @staticmethod
    def compute_entropy_uncertainty(logits_or_probs: np.ndarray, is_probs: bool = False) -> float:
        """Compute normalized entropy H(p)/log(4) over 4 action classes in [0, 1]."""
        if is_probs:
            p = np.asarray(logits_or_probs, dtype=np.float64)
        else:
            logits = np.asarray(logits_or_probs, dtype=np.float64)
            shifted = logits - np.max(logits)
            p = np.exp(shifted) / np.sum(np.exp(shifted))

        p = np.clip(p, 1e-12, 1.0)
        p = p / np.sum(p)
        h = -float(np.sum(p * np.log(p)))
        max_h = float(np.log(len(p)))
        if max_h <= 0:
            return 0.0
        return float(np.clip(h / max_h, 0.0, 1.0))

    def evaluate(
        self,
        surprise_proxy: float,
        uncertainty: float,
        energy_deficit: float,
        unresolved: float,
        current_tick: int,
        force_disable: bool = False,
    ) -> tuple[float, bool, dict[str, float]]:
        """Evaluate trigger score and determine if VIMARSA deliberation fires.
        
        Returns:
            (score, fired_bool, components_dict)
        """
        s = float(np.clip(surprise_proxy, 0.0, 1.0))
        u = float(np.clip(uncertainty, 0.0, 1.0))
        e = float(np.clip(energy_deficit, 0.0, 1.0))
        r = float(np.clip(unresolved, 0.0, 1.0))

        score = float(
            self.w_surprise * s
            + self.w_uncertainty * u
            + self.w_energy_deficit * e
            + self.w_unresolved * r
        )

        components = {
            "surprise_proxy": s,
            "uncertainty": u,
            "energy_deficit": e,
            "unresolved_plan": r,
        }

        if force_disable:
            return score, False, components

        can_trigger = (
            score > self.threshold_theta
            and (current_tick - self.last_trigger_tick) >= self.cooldown_ticks
            and self.total_triggers_count < self.max_total_deliberations
        )

        if can_trigger:
            self.last_trigger_tick = current_tick
            self.total_triggers_count += 1
            return score, True, components

        return score, False, components

    def snapshot(self) -> dict[str, Any]:
        return {
            "w_surprise": self.w_surprise,
            "w_uncertainty": self.w_uncertainty,
            "w_energy_deficit": self.w_energy_deficit,
            "w_unresolved": self.w_unresolved,
            "threshold_theta": self.threshold_theta,
            "norm_scale_surprise": self.norm_scale_surprise,
            "cooldown_ticks": self.cooldown_ticks,
            "max_total_deliberations": self.max_total_deliberations,
            "last_trigger_tick": self.last_trigger_tick,
            "total_triggers_count": self.total_triggers_count,
        }

    @classmethod
    def restore(cls, data: dict[str, Any]) -> EndogenousTrigger:
        trig = cls(
            w_surprise=data["w_surprise"],
            w_uncertainty=data["w_uncertainty"],
            w_energy_deficit=data["w_energy_deficit"],
            w_unresolved=data["w_unresolved"],
            threshold_theta=data["threshold_theta"],
            norm_scale_surprise=data["norm_scale_surprise"],
            cooldown_ticks=data["cooldown_ticks"],
            max_total_deliberations=data["max_total_deliberations"],
        )
        trig.last_trigger_tick = int(data["last_trigger_tick"])
        trig.total_triggers_count = int(data["total_triggers_count"])
        return trig
