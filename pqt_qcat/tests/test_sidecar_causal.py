"""Check independent credit routing, the scale intervention, and reaction guards."""
import copy

import numpy as np
import pytest

from pqt_qcat.memory import sidecar_causal as causal
from pqt_qcat.memory.functional_sidecar import ActionJournal, state_hash
from pqt_qcat.memory.life_qnuf_001 import new_agent
from pqt_qcat.memory.qnuf import ObservedEvidence


def test_action_dependent_advance_never_retries_and_reveals_after_reaction(tmp_path):
    results = []
    for action in (1, 2):
        environment = causal.AdvanceEnvironment([1, 2, 3, 4, 1, 2], [5], 1)
        journal = ActionJournal(tmp_path / f'{action}.jsonl')
        with pytest.raises(RuntimeError):
            environment.reveal()
        receipt = journal.publish({'query': environment.observe(), 'cursor': 0, 'action': action,
                                   'main': {'event_id': 48}})
        result = environment.react(journal, receipt)
        assert environment.reveal() == ObservedEvidence(48, 1)
        results.append((result['next_query'], result['reward']))
    assert results == [(6, 1.), (2, -.25)]


def test_sources_are_orthogonal_and_norm_is_exact():
    origin = new_agent('B3', 821101, dict(substeps=2, h_base=.01, query_ticks=8, eta=.8, clip=.5, gate=1.))
    packet = causal.b2.forecast(origin, 4, 'Z0-64', 'aligned', None, {})
    start_hash = state_hash(origin.snapshot())
    updates = {}
    for arm in causal.SOURCES:
        branch = copy.deepcopy(origin)
        for norm in (None, .05, .50):
            trial = copy.deepcopy(branch)
            update, _ = causal.apply_credit(trial, packet, ObservedEvidence(0, 3), arm, norm)
            if norm is None:
                updates[arm] = update
            else:
                assert update['applied_delta_norm'] == pytest.approx(norm, abs=2e-14, rel=0.)
            np.testing.assert_array_equal(trial.core.Z, origin.core.Z)
    assert start_hash == state_hash(origin.snapshot())
    for a, b in [('MM', 'SM'), ('MS', 'SS')]:
        assert updates[a]['phi_event'] == updates[b]['phi_event']
        assert updates[a]['error_vector'] != updates[b]['error_vector']
    for a, b in [('MM', 'MS'), ('SM', 'SS')]:
        assert updates[a]['phi_event'] != updates[b]['phi_event']
        assert updates[a]['error_vector'] == updates[b]['error_vector']
