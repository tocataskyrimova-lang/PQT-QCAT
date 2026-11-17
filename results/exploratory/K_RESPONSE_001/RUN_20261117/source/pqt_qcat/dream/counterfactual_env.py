"""Partially Observed Graph Environment for Counterfactual Synthesis (Scenario T5).

Defines structured relational worlds where:
1. Base transitions are shown to the agent during wake training.
2. Multi-step compositions (e.g. 1 -> 2 and 2 -> 3 implies compositional relation 1 -> 3)
   are STRICTLY HIDDEN from the agent during training and sleep.
3. Hidden test transitions are evaluated strictly post-sleep without feedback.
4. Includes both reversible and non-bijective/branching environments to test generality.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TransitionEdge:
    source_token: int
    target_token: int
    is_composition: bool = False
    is_branching: bool = False


class PartiallyObservedGraphEnv:
    """Relational environment with base edges (train) and hidden compositional edges (test)."""

    def __init__(self, mode: str = "relational_chain", seed: int = 42):
        self.mode = mode
        self.rng = np.random.default_rng(seed)

        if mode == "relational_chain":
            # Chain 1: 1 -> 2 -> 3; Chain 2: 4 -> 5 -> 6
            # Base edges shown during training:
            self.base_edges = [
                TransitionEdge(1, 2),
                TransitionEdge(2, 3),
                TransitionEdge(3, 4),
            ]
            # Hidden compositional edges (never shown during training or sleep!):
            self.hidden_compositions = [
                TransitionEdge(1, 3, is_composition=True),  # 2-step hop (1->2->3)
                TransitionEdge(2, 4, is_composition=True),  # 2-step hop (2->3->4)
            ]
            # Invalid negative edges (to check false positive promotions):
            self.negative_edges = [
                TransitionEdge(1, 4),
                TransitionEdge(4, 1),
                TransitionEdge(3, 1),
            ]

        elif mode == "branching_non_bijective":
            # Non-bijective graph (merging and branching)
            self.base_edges = [
                TransitionEdge(1, 3, is_branching=True),
                TransitionEdge(2, 3, is_branching=True),  # Many-to-one merge
                TransitionEdge(3, 4),
            ]
            self.hidden_compositions = [
                TransitionEdge(1, 4, is_composition=True),  # 1 -> 3 -> 4
                TransitionEdge(2, 4, is_composition=True),  # 2 -> 3 -> 4
            ]
            self.negative_edges = [
                TransitionEdge(4, 1),
                TransitionEdge(4, 2),
            ]
        else:
            raise ValueError(f"Unknown graph mode: {mode}")

    def get_training_edges(self) -> list[TransitionEdge]:
        """Base edges accessible during wake observation."""
        return list(self.base_edges)

    def get_hidden_test_edges(self) -> list[TransitionEdge]:
        """Hidden edges accessible ONLY to the evaluator post-sleep."""
        return list(self.hidden_compositions)

    def get_negative_edges(self) -> list[TransitionEdge]:
        """Distractor/invalid pairs to evaluate false discovery rate."""
        return list(self.negative_edges)
