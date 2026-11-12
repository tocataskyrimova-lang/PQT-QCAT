"""Symmetric Counterfactual Dream Generator (ARCHITECTURE_LIFE_002 Section 5.1).

Conforms strictly to ILORI mandates:
- Identical generator code, step budget, noise schedule, and candidate extraction
  for both PQT (A = A_I) and noA (A = 0)
- Explores internal state space during idle/sleep ticks with zero sensory input (u_ext = 0)
- ZERO access to external oracle, test labels, or hidden environment truth
- All generated hypotheses are recorded in B_dream with provenance pointing to B_real
- Deterministic RNG synchronization and verifiable RNG fingerprints (CONTROL-FIX-002)
"""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
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
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.last_seed_event_ids: list[str] = []
        self.last_noise_stream: list[np.ndarray] = []
        self.last_rng_fingerprints: dict[str, str | int] = {}
        self.last_integrated_ticks: int = 0
        self.last_presentation_ticks: int = 0

    def reset_rng(self, seed: int | None = None) -> None:
        """Reset internal RNG to ensure identical random trajectory across paired runs."""
        s = self.seed if seed is None else seed
        self.rng = np.random.default_rng(s)

    def generate_candidate_hypotheses(
        self,
        core: SixEchelonCore,
        ledger: WitnessLedger,
        cycle_count: int = 20,
        reset_seed: int | None = None,
    ) -> list[DreamHypothesis]:
        """Generate counterfactual transition candidates during offline sleep.
        
        Parameters
        ----------
        core : SixEchelonCore
            The agent's 48D core (either PQT with enable_vortex=True or noA with enable_vortex=False).
        ledger : WitnessLedger
            Witness ledger containing confirmed prior observations in B_real.
        cycle_count : int
            Number of dream exploration cycles.
        reset_seed : int | None
            If provided, reseed the generator RNG before sampling to ensure identical paired comparisons.
            
        Returns
        -------
        list[DreamHypothesis]
            Internally synthesized candidates appended to B_dream with immutable provenance.
        """
        if reset_seed is not None:
            self.reset_rng(reset_seed)

        real_events = ledger.real_events
        if not real_events:
            self.last_seed_event_ids = []
            self.last_noise_stream = []
            self.last_integrated_ticks = 0
            self.last_presentation_ticks = 0
            self.last_rng_fingerprints = {
                "dream_seed_events_hash": hashlib.sha256(b"").hexdigest(),
                "dream_noise_stream_hash": hashlib.sha256(b"").hexdigest(),
                "seed_events_count": 0,
                "noise_vectors_count": 0,
            }
            return []

        self.last_seed_event_ids = []
        self.last_noise_stream = []
        self.last_integrated_ticks = 0
        self.last_presentation_ticks = 0

        generated: list[DreamHypothesis] = []

        # Grounding: Dream exploration starts from representations seeded by real observations
        for cycle in range(cycle_count):
            # Select a real grounding event from B_real as the seed
            seed_event = self.rng.choice(real_events)
            self.last_seed_event_ids.append(str(seed_event.event_id))
            token_in = seed_event.input_token

            # Initialize seed activation
            core.Z.fill(0.0)
            for c in range(6):
                core.Z[8 * c + 6] = 1.0  # Quiescent latch
            
            # Brief internal sensory activation from seed
            x_seed = np.zeros(core.config.d_in)
            x_seed[token_in] = 1.0
            core.step(x_seed, write_gate=1.0)
            self.last_presentation_ticks += 1

            # Offline autonomous trajectory with zero external input (u_ext = 0)
            # Both PQT and noA experience identical noise and step counts
            for _ in range(self.config.dream_ticks_per_cycle):
                noise = self.rng.normal(0.0, self.config.thermal_noise_sigma, size=48)
                self.last_noise_stream.append(noise.copy())
                core.Z += noise * np.sqrt(core.config.h_base)
                # Free evolution (no sensory input)
                core.step(x_t=None, write_gate=0.0)
                self.last_integrated_ticks += 1

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

        # Compute deterministic cryptographic fingerprints of the stochastic trajectory
        seeds_bytes = ",".join(self.last_seed_event_ids).encode("utf-8")
        seeds_hash = hashlib.sha256(seeds_bytes).hexdigest()

        if self.last_noise_stream:
            noise_stacked = np.ascontiguousarray(np.stack(self.last_noise_stream, axis=0))
            noise_hash = hashlib.sha256(noise_stacked.tobytes()).hexdigest()
        else:
            noise_hash = hashlib.sha256(b"").hexdigest()

        self.last_rng_fingerprints = {
            "dream_seed_events_hash": seeds_hash,
            "dream_noise_stream_hash": noise_hash,
            "seed_events_count": len(self.last_seed_event_ids),
            "noise_vectors_count": len(self.last_noise_stream),
        }

        return generated
