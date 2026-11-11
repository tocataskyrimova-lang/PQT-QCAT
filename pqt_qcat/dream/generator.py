"""Symmetric Counterfactual Dream Generator (ARCHITECTURE_LIFE_002 Section 5.1).

Conforms strictly to ILORI mandates:
- Identical generator code, step budget, noise schedule, and candidate extraction
  for both PQT (A = A_I) and noA (A = 0)
- Explores internal state space during idle/sleep ticks with zero sensory input (u_ext = 0)
- ZERO access to external oracle, test labels, or hidden environment truth
- All generated hypotheses are recorded in B_dream with provenance pointing to B_real
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..core.six_echelon import SixEchelonCore, SixEchelonConfig
from ..memory.witness import WitnessLedger, DreamHypothesis, compute_maya_weight
from ..core.errors import NonFiniteStateError


@dataclass
class DreamGeneratorConfig:
    """Configuration for counterfactual dream generator."""
    dream_ticks_per_cycle: int = 10     # Continuous integration steps per dream cycle
    thermal_noise_sigma: float = 0.05   # Amplitude of thermal exploration noise sqrt(2 T Gamma)
    min_confidence_threshold: float = 0.25 # Candidate discarded if model posterior < threshold
    inverse_temp: float = 4.0           # Softmax inverse temperature beta for candidate readout
    t_maya: float = 1.0                # Maya temperature


class DreamGenerator:
    """Autonomous dream trajectory generator operating identically for PQT and noA."""

    def __init__(
        self,
        config: DreamGeneratorConfig | None = None,
        seed: int = 42,
    ):
        self.config = config or DreamGeneratorConfig()
        self.rng = np.random.default_rng(seed)

    def generate_candidate_hypotheses(
        self,
        core: SixEchelonCore,
        ledger: WitnessLedger,
        cycle_count: int = 20,
    ) -> list[DreamHypothesis]:
        """Generate counterfactual transition candidates during offline sleep.
        
        Parameters
        ----------
        core : SixEchelonCore
            The agent's 48D core (either PQT with enable_vortex=True or noA with enable_vortex=False).
        ledger : WitnessLedger
            Witness ledger containing confirmed prior observations in B_real.
            
        Returns
        -------
        list[DreamHypothesis]
            Internally synthesized candidates appended to B_dream with immutable provenance.
        """
        real_events = ledger.real_events
        if not real_events:
            return []

        generated: list[DreamHypothesis] = []

        # Grounding: Dream exploration starts from representations seeded by real observations
        for cycle in range(cycle_count):
            # Select a real grounding event from B_real as the seed
            seed_event = self.rng.choice(real_events)
            token_in = seed_event.input_token

            # Initialize seed activation
            core.Z.fill(0.0)
            for c in range(6):
                core.Z[8 * c + 6] = 1.0  # Quiescent latch
            
            # Brief internal sensory activation from seed
            x_seed = np.zeros(core.config.d_in)
            x_seed[token_in] = 1.0
            core.step(x_seed, write_gate=1.0)

            # Offline autonomous trajectory with zero external input (u_ext = 0)
            # Both PQT and noA experience identical noise and step counts
            for _ in range(self.config.dream_ticks_per_cycle):
                noise = self.rng.normal(0.0, self.config.thermal_noise_sigma, size=48)
                core.Z += noise * np.sqrt(core.config.h_base)
                # Free evolution (no sensory input)
                core.step(x_t=None, write_gate=0.0)

            # Readout candidate from internal state with inverse temperature beta
            logits = core.predict() * self.config.inverse_temp
            p = np.exp(logits[1:5] - np.max(logits[1:5]))
            p /= np.sum(p)

            pred_token = int(np.argmax(p)) + 1
            conf = float(np.max(p))

            if conf >= self.config.min_confidence_threshold:
                hypo = ledger.record_dream_hypothesis(
                    timestamp=seed_event.timestamp + cycle + 1,
                    input_token=token_in,
                    pred_token=pred_token,
                    confidence=conf,
                    provenance_basis=[seed_event.event_id],
                )
                generated.append(hypo)

        return generated
