"""Stage 0/1 event-local plasticity on the existing LIFE core."""
from dataclasses import dataclass
import copy
import hashlib
import json
import numpy as np

from .exp_stateful_online_001 import StatefulOnlineAgent


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def rng_hash(core):
    data = json.dumps(core.rng.bit_generator.state, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()


@dataclass(frozen=True)
class ObservedEvidence:
    event_id: int
    target: int
    source: str = 'observed_stream'


@dataclass(frozen=True)
class FixationAssessment:
    """Stage 2 interface. A caller must supply losses from past observed anchors."""
    current_improvement: float
    prior_anchor_loss_deltas: tuple[float, ...]

    def accepts(self, max_delta, min_improvement=0.0):
        deltas = self.prior_anchor_loss_deltas
        return bool(deltas and np.isfinite([self.current_improvement, *deltas]).all()
                    and self.current_improvement >= min_improvement
                    and np.mean(deltas) <= max_delta and max(deltas) <= max_delta)


def need_accepts(evidence, target_probability, novelty, threshold):
    """Stage 2 policy interface. Even maximal novelty requires observed feedback."""
    return evidence is not None and (novelty or 1 - target_probability > threshold)


class QNUFAgent(StatefulOnlineAgent):
    """One query per event, one matched shadow, and immediate commit after feedback."""

    def __init__(self, core, eligibility='legacy', error='target', norm_cap=None, **kwargs):
        super().__init__(core, t_context=0, **kwargs)
        if eligibility not in ('legacy', 'event') or error not in ('target', 'full'):
            raise ValueError('Unknown eligibility or class error')
        if norm_cap is not None and (not np.isfinite(norm_cap) or norm_cap < 0):
            raise ValueError('Invalid update norm cap')
        if self.learning_mode not in ('CARRY_TRAIN', 'RESET_TRAIN'):
            raise ValueError('Expected CARRY or RESET')
        self.eligibility = eligibility
        self.error = error
        self.norm_cap = norm_cap
        self.event_id = 0
        self.pending = None
        self.witness_ledger = []
        self.observed_counts = [0] * 7
        self.shadow_steps = 0
        self.update_candidates = 0
        self.last_decision = None

    def begin_query(self, query):
        if self.pending is not None:
            raise RuntimeError('Resolve pending feedback before the next query')
        if query not in range(7):
            raise ValueError('Expected neutral query 0 or a token in 1..6')
        if self.learning_mode == 'RESET_TRAIN':
            self.reset_state_to_z0()
        shadow = copy.deepcopy(self.core)
        z_pre = self.core.Z.copy()
        k_pre = array_hash(self.core.K)
        rng_pre = rng_hash(self.core)
        stimulus = np.zeros(8)
        if query:
            stimulus[query] = 1.0
        for _ in range(self.t_query):
            self.core.step(stimulus, write_gate=self.write_gate_presentation)
        shadow_start = shadow.step_count
        for _ in range(self.t_query):
            shadow.step(None, write_gate=self.write_gate_presentation)
        self.shadow_steps += shadow.step_count - shadow_start
        if rng_hash(self.core) != rng_hash(shadow):
            raise RuntimeError('Input and no-input RNG diverged')
        if k_pre != array_hash(self.core.K) or k_pre != array_hash(shadow.K):
            raise RuntimeError('Prediction modified K')
        logits = self.core.predict()[1:5].copy()
        probabilities = np.exp(logits - np.max(logits))
        probabilities /= np.sum(probabilities)
        positive = probabilities[probabilities > 0]
        record = {
            'event_id': self.event_id, 'query': query, 'input': stimulus.tolist(),
            'Z_pre': z_pre.tolist(), 'Z_post': self.core.Z.tolist(), 'Z_noinput': shadow.Z.tolist(),
            'Z_pre_hash': array_hash(z_pre), 'Z_post_hash': array_hash(self.core.Z),
            'Z_noinput_hash': array_hash(shadow.Z), 'K_pre_hash': k_pre,
            'Z_norms': [float(np.linalg.norm(z)) for z in (z_pre, self.core.Z, shadow.Z)],
            'rng_pre_hash': rng_pre, 'rng_post_hash': rng_hash(self.core),
            'rng_shadow_post_hash': rng_hash(shadow), 'T': self.t_query,
            'logits': logits.tolist(), 'probabilities': probabilities.tolist(),
            'prediction': int(np.argmax(logits)) + 1,
            'entropy': float(-np.sum(positive * np.log(positive)) / np.log(4)),
            'novelty': self.observed_counts[query] == 0,
        }
        psi = np.tanh(self.core.Z)
        self.pending = {'prediction': record, 'legacy': psi - self.psi_0,
                        'event': psi - np.tanh(shadow.Z)}
        if not query and np.any(self.pending['event'] != 0):
            raise RuntimeError('Neutral paired branches diverged')
        return copy.deepcopy(record)

    def finish_query(self, evidence):
        if self.pending is None:
            raise RuntimeError('Feedback requires a committed prediction')
        prediction = self.pending['prediction']
        if evidence is not None and (evidence.event_id != self.event_id
                or evidence.target not in range(1, 5) or evidence.source != 'observed_stream'):
            raise ValueError('Feedback must match the pending event and observed source')
        phi = self.pending[self.eligibility]
        info = {
            'event_id': self.event_id, 'phi_kind': self.eligibility, 'error_kind': self.error,
            'phi_legacy': self.pending['legacy'].tolist(), 'phi_event': self.pending['event'].tolist(),
            'phi_norm': float(np.linalg.norm(phi)), 'target': None, 'evidence_source': None,
            'error_vector': None, 'error_norm': None, 'N_score': None, 'F_gate': None,
            'anchor_loss_diff': None, 'N_gate': None, 'decision': 'MISSING_FEEDBACK',
            'proposed_delta_norm': 0.0, 'candidate_norm_before_cap': 0.0,
            'applied_delta_norm': 0.0, 'nonfinite': False,
            'K_proposed_hash': None,
        }
        if evidence is not None:
            p = np.array(prediction['probabilities'])
            error = np.zeros(8)
            if self.error == 'full':
                error[1:5] = -p
            error[evidence.target] = 1 - p[evidence.target - 1]
            # Retain the exact 8D matmul operation of the historical baseline.
            returned_error = self.core.W_readout.T @ error
            delta = np.clip(self.eta_k * np.outer(returned_error, phi),
                            -self.delta_core_max, self.delta_core_max)
            raw_norm = float(np.linalg.norm(delta))
            if self.norm_cap is not None and raw_norm > self.norm_cap:
                delta *= self.norm_cap / raw_norm
            before = self.core.K.copy()
            proposed = before + delta
            if not np.isfinite(proposed).all():
                raise FloatingPointError('Nonfinite weight proposal')
            self.core.K[:] = proposed
            logits = np.array(prediction['logits'])
            index = evidence.target - 1
            info.update(target=evidence.target, evidence_source=evidence.source,
                        error_vector=error.tolist(), error_norm=float(np.linalg.norm(error)),
                        proposed_delta_norm=float(np.linalg.norm(delta)), candidate_norm_before_cap=raw_norm,
                        applied_delta_norm=float(np.linalg.norm(self.core.K - before)),
                        K_proposed_hash=array_hash(proposed), N_gate=True, decision='COMMITTED_UNCHECKED',
                        correct=prediction['prediction'] == evidence.target,
                        nll=float(np.logaddexp.reduce(logits) - logits[index]),
                        margin=float(logits[index] - np.max(np.delete(logits, index))))
            self.observed_counts[prediction['query']] += 1
            self.update_candidates += 1
            self.witness_ledger.append({'event_id': self.event_id, 'query': prediction['query'],
                                        'target': evidence.target, 'source': evidence.source})
        info['K_post_hash'] = array_hash(self.core.K)
        if array_hash(self.core.Z) != prediction['Z_post_hash']:
            raise RuntimeError('Feedback changed the live trajectory')
        self.last_decision = copy.deepcopy(info)
        self.pending = None
        self.event_id += 1
        return info

    def snapshot(self):
        snapshot = super().snapshot()
        names = ('eligibility', 'error', 'norm_cap', 'event_id', 'pending', 'witness_ledger',
                 'observed_counts', 'shadow_steps', 'update_candidates', 'last_decision')
        snapshot['qnuf'] = {name: copy.deepcopy(getattr(self, name)) for name in names}
        return snapshot

    def restore(self, snapshot, strict_config=True):
        super().restore(snapshot, strict_config=strict_config)
        for name, value in snapshot['qnuf'].items():
            setattr(self, name, copy.deepcopy(value))
