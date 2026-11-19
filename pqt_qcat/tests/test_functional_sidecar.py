"""Focused B0 tests of action gating, donor isolation, and checkpoint recovery."""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
import pytest

from pqt_qcat.memory.access_state import exact, load_agent
from pqt_qcat.memory.checkpoint_io import load_state
from pqt_qcat.memory.functional_sidecar import (
    ARMS, ActionJournal, FixedEnvironment, archived_events, predict_packet, proceed,
    state_hash, validate_unit, wrong_trajectory,
)


@pytest.fixture(scope='module')
def plan():
    output = Path(os.environ['LIFE_SIDECAR_TEST_RUN'])
    return json.loads((output / 'PLAN.json').read_text())


def test_label_is_released_after_committed_action_only(tmp_path):
    env = FixedEnvironment([{'query': 2, 'target': 3}], 0)
    assert env.observe() == 2
    journal = ActionJournal(tmp_path / 'actions.jsonl')
    with pytest.raises(RuntimeError):
        env.feedback(journal, None)
    wrong = journal.publish({'query': 1, 'event_id': 0, 'deployed_action': 3})
    with pytest.raises(RuntimeError):
        env.feedback(journal, wrong)
    assert env.cursor == 0
    good = ActionJournal(tmp_path / 'good.jsonl')
    receipt = good.publish({'query': 2, 'event_id': 0, 'deployed_action': 2})
    evidence, reward = env.feedback(good, receipt)
    assert evidence.target == 3 and reward == 0
    with pytest.raises(RuntimeError):
        env.feedback(good, receipt)


def test_altered_action_bytes_are_rejected(tmp_path):
    path = tmp_path / 'actions.jsonl'
    journal = ActionJournal(path)
    receipt = journal.publish({'query': 2, 'event_id': 0, 'deployed_action': 3})
    path.write_bytes(path.read_bytes().replace(b'":3', b'":2'))
    with pytest.raises(RuntimeError, match='Action bytes'):
        journal.committed(receipt)


def test_exact_rng_isolation_and_Z0(plan):
    seed, mapping = plan['units'][0]
    agent = load_agent(plan, 'B3', seed, mapping, 48)
    agent.core.integrator.config = replace(agent.core.integrator.config, temperature=.001)
    reference = copy.deepcopy(agent)
    query = 2
    packet, _, _ = predict_packet(agent, query, 'S-Z0-64', None, ARMS['S-Z0-64'])
    reference.begin_query(query)
    exact(agent.snapshot(), reference.snapshot())
    read = packet['sidecar_or_padding']
    assert read['Z_start_hash'] != packet['main']['Z_pre_hash']
    from pqt_qcat.memory.qnuf import array_hash
    assert read['Z_start_hash'] == array_hash(agent.Z0)
    assert read['rng_pre_hash'] != read['rng_post_hash']


def test_pending_checkpoint_and_feedback_use_main_prediction(tmp_path, plan):
    seed, mapping = plan['units'][0]
    agent = load_agent(plan, 'B3', seed, mapping, 48)
    rows = archived_events(plan, seed, mapping)
    env = FixedEnvironment(rows, 48)
    packet, _, _ = predict_packet(agent, env.observe(), 'S-Z0-64', None, ARMS['S-Z0-64'])
    journal = ActionJournal(tmp_path / 'actions.jsonl')
    receipt = journal.publish(packet)
    path = tmp_path / 'pending.npz'
    agent.save_checkpoint(path, {'cursor': 48, 'receipt': receipt})
    assert not list(tmp_path.glob('*.tmp'))
    restored = load_agent(plan, 'B3', seed, mapping, 48)
    saved = load_state(path)
    restored.restore(saved['agent'])
    assert state_hash(restored.snapshot()) == packet['main_pending_sha256']
    restored_journal = ActionJournal(journal.path)
    restored_journal.restore_receipt(receipt)
    a = proceed(agent, env, journal, receipt, rows[48])
    assert a['feedback']['reward'] == int(packet['deployed_action'] == rows[48]['target'])
    b = proceed(restored, FixedEnvironment(rows, 48), restored_journal, receipt, rows[48])
    exact(a['update'], b['update'])
    exact(agent.snapshot(), restored.snapshot())


@pytest.mark.parametrize('arm', list(ARMS))
def test_budget_no_plasticity_and_donor_matching(plan, arm):
    seed, mapping = plan['units'][0]
    agent = load_agent(plan, 'B3', seed, mapping, 48)
    spec = ARMS[arm]
    wrong_mapping, states, wrong_core = wrong_trajectory(plan, seed, mapping)
    assert wrong_mapping != mapping and len(states) == 24
    k = states[48] if spec['donor'] == 'wrong' else None
    if spec['donor'] == 'initial':
        from pqt_qcat.memory.functional_sidecar import checkpoint_path
        k = load_state(checkpoint_path(plan, 'B0', seed, mapping, 0))['agent']['core_snapshot']['K']
    exact(agent.core.W_readout, wrong_core['W_readout'])
    packet, pre, _ = predict_packet(agent, 1, arm, k, spec)
    from pqt_qcat.memory.qnuf import array_hash
    exact(agent.core.K, pre['core_snapshot']['K'])
    assert packet['sidecar_or_padding']['K_hash'] == array_hash(agent.core.K if k is None else k)
    assert packet['substeps_main'] == packet['substeps_shadow'] == 16
    assert packet['substeps_sidecar'] == 2 * spec['ticks']
    assert packet['substeps_padding'] == 2 * spec['padding']


@pytest.mark.parametrize('seed', [100, 101, 105, 999999])
def test_runtime_guard_cannot_be_disabled(plan, seed):
    altered = {**plan, 'forbidden_seeds': []}
    with pytest.raises(ValueError):
        validate_unit(altered, seed, 0)
