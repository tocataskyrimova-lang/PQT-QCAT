"""Regression tests for action-dependent evidence and coherent credit."""
import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from pqt_qcat.memory import functional_sidecar_b2a as b2
from pqt_qcat.memory.access_state import exact, load_agent
from pqt_qcat.memory.functional_sidecar import ActionJournal
from pqt_qcat.memory.qnuf import ObservedEvidence


def test_transition_and_evidence_order(tmp_path):
    branches = []
    for action in (2, 4):
        env = b2.RetryEnvironment([2, 4, 1, 3, 2, 4], [3], 1)
        journal = ActionJournal(tmp_path / f'{action}.jsonl')
        with pytest.raises(RuntimeError):
            env.reveal()
        with pytest.raises(RuntimeError):
            env.react(journal, None)
        q = env.observe()
        receipt = journal.publish({'query': q, 'cursor': 0, 'action': action, 'main': {'event_id': 48}})
        reaction = env.react(journal, receipt)
        with pytest.raises(RuntimeError):
            env.observe()
        evidence = env.reveal()
        assert evidence.target == 2
        branches.append((env.query, reaction['reward']))
    assert branches == [(4, 1.), (1, -.25)]
    assert b2.RetryEnvironment.transition(1, 4, 2, 3, 9)['reward'] == -1.


def test_credit_matches_dispatched_read_branch():
    root = Path(os.environ['LIFE_B2A_TEST_RUN'])
    plan = json.loads((root / 'PLAN.json').read_text())
    seed, mapping = plan['units'][0]
    main = load_agent(plan, 'B3', seed, mapping, 48)
    aligned = copy.deepcopy(main)
    current = plan['streams'][f'{seed}_{mapping}']['start_query']
    original_k = main.core.K.copy()
    a = b2.forecast(main, current, 'Z0-64', 'main', None, {})
    b = b2.forecast(aligned, current, 'Z0-64', 'aligned', None, {})
    exact(a['main'], b['main'])
    exact(a['logits'], b['logits'])
    exact(main.snapshot(), aligned.snapshot())
    evidence = ObservedEvidence(48, plan['parent']['stage1_mappings'][mapping][current - 1])
    source, update = b2.learn(aligned, b, evidence, 'aligned')
    assert source == 'external'
    e = np.array(update['error_vector'])
    phi = np.array(b['external']['eligibility'])
    p = np.exp(np.array(b['logits']) - max(b['logits']))
    expected = -p / p.sum()
    expected[evidence.target - 1] += 1.
    np.testing.assert_array_equal(e[1:5], expected)
    delta = np.clip(aligned.eta_k * np.outer(aligned.core.W_readout.T @ e, phi),
                    -aligned.delta_core_max, aligned.delta_core_max)
    np.testing.assert_array_equal(aligned.core.K, original_k + delta)
    b2.learn(main, a, evidence, 'main')
    assert not np.array_equal(main.core.K, aligned.core.K)
    np.testing.assert_array_equal(main.core.Z, aligned.core.Z)


def test_forbidden_seeds_cannot_be_routed():
    with pytest.raises(ValueError):
        b2.validate_unit({'units': [[101, 0]], 'forbidden_seeds': []}, 101, 0)
