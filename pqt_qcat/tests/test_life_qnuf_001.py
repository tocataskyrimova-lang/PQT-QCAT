"""Regression checks for the causal boundaries of the QNUF experiment."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest

from pqt_qcat.memory.checkpoint_io import load_state
from pqt_qcat.memory.life_qnuf_001 import ARMS, new_agent, read_probes
from pqt_qcat.memory.qnuf import FixationAssessment, ObservedEvidence, need_accepts


PLAN = {'substeps': 2, 'h_base': .01, 'query_ticks': 8, 'eta': .8, 'clip': .5, 'gate': 1.}
BASE_RUN = Path(r'C:\AGI\PQT_QCAT_AUDIT\CODEX_MEMORY_LOCATION_001_RUN_B')


@pytest.mark.parametrize('arm,mode', [('B0','CARRY_TRAIN'), ('RESET','RESET_TRAIN')])
def test_baseline_bitwise_matches_frozen_source(arm, mode):
    folder = BASE_RUN / 'lives' / f'{mode}_810001_00'
    records = json.loads((folder / 'events.json').read_text())
    for_clone = load_state(folder / 'checkpoint_072.npz')['agent']['core_snapshot']
    agent = new_agent(arm, 810001, PLAN)
    for index, expected in enumerate(records):
        observed = agent.begin_query(expected['query'])
        np.testing.assert_array_equal(observed['logits'], expected['logits_before_feedback'])
        agent.finish_query(ObservedEvidence(index, expected['target_after_prediction']))
    np.testing.assert_array_equal(agent.core.K, for_clone['K'])
    np.testing.assert_array_equal(agent.core.Z, for_clone['Z'])
    archived = json.loads((folder / 'result.json').read_text())
    probes, _ = read_probes(agent, range(1, 7), archived['targets'], 0)
    final = archived['probes']['72']
    for mode in ('live', 'reset'):
        np.testing.assert_array_equal(probes[mode]['logits'], final[mode]['logits'])


def test_q_pair_same_snapshot_and_rng():
    agent = new_agent('B1', 821001, PLAN)
    agent.core.integrator.config = replace(agent.core.integrator.config, temperature=.001)
    core_reference = copy.deepcopy(agent.core)
    for _ in range(8):
        core_reference.step(np.eye(8)[2], write_gate=1.)
    event = agent.begin_query(2)
    assert event['rng_pre_hash'] != event['rng_post_hash'] == event['rng_shadow_post_hash']
    np.testing.assert_array_equal(agent.core.Z, core_reference.Z)
    agent.finish_query(None)
    agent.begin_query(0)
    np.testing.assert_array_equal(agent.pending['event'], np.zeros(48))
    agent.finish_query(None)


@pytest.mark.parametrize('arm', list(ARMS))
def test_feedback_freezes_prediction_and_preserves_physics(arm):
    agent = new_agent(arm, 821001, PLAN)
    other = copy.deepcopy(agent)
    k0, b0, w0 = (m.copy() for m in (agent.core.K, agent.core.B, agent.core.W_readout))
    with pytest.raises(RuntimeError):
        agent.finish_query(ObservedEvidence(0, 1))
    pred = agent.begin_query(3)
    assert pred == other.begin_query(3)
    with pytest.raises(RuntimeError):
        agent.begin_query(4)
    with pytest.raises(ValueError):
        agent.finish_query(ObservedEvidence(1, 2))
    with pytest.raises(ValueError):
        agent.finish_query(ObservedEvidence(0, 2, 'test_label'))
    np.testing.assert_array_equal(agent.core.K, k0)
    missing = agent.finish_query(None)
    assert missing['applied_delta_norm'] == 0 and agent.witness_ledger == []
    agent.begin_query(4)
    result = agent.finish_query(ObservedEvidence(1, 3))
    assert result['applied_delta_norm'] == np.linalg.norm(agent.core.K - k0)
    assert np.max(np.abs(agent.core.K - k0)) <= .5 + 1e-15
    np.testing.assert_array_equal(agent.core.B, b0)
    np.testing.assert_array_equal(agent.core.W_readout, w0)
    changed = other.finish_query(ObservedEvidence(0, 2))
    assert changed['target'] != result['target'] and pred['prediction'] in range(1,5)


def test_pending_checkpoint_and_no_shadow_contamination(tmp_path):
    agent = new_agent('B3', 821001, PLAN)
    agent.begin_query(1)
    agent.finish_query(ObservedEvidence(0, 1))
    previous_z = agent.core.Z.copy()
    record = agent.begin_query(2)
    np.testing.assert_array_equal(record['Z_pre'], previous_z)
    assert agent.shadow_steps == agent.core.step_count == 32
    path = tmp_path / 'pending.npz'
    agent.save_checkpoint(path, {'cursor': 1})
    restored = new_agent('B3', 821002, PLAN)
    assert restored.load_checkpoint(path)['cursor'] == 1
    result = agent.finish_query(ObservedEvidence(1, 3))
    assert result == restored.finish_query(ObservedEvidence(1, 3))
    np.testing.assert_array_equal(agent.core.K, restored.core.K)
    assert agent.witness_ledger == restored.witness_ledger
    assert agent.begin_query(4) == restored.begin_query(4)
    z = restored.core.Z.copy()
    read_probes(restored, [1, 2], [1, 2], 4)
    np.testing.assert_array_equal(z, restored.core.Z)


@pytest.mark.parametrize('seed', [101, 100])
def test_seed_guard(seed):
    with pytest.raises(ValueError):
        new_agent('B0', seed, PLAN)


def test_stage2_interfaces_only():
    assert not need_accepts(None, .0, True, .1)
    assert not need_accepts(ObservedEvidence(0, 1), 1., False, .1)
    assert not FixationAssessment(.5, (-.2, .01)).accepts(0.)
    assert not FixationAssessment(.5, ()).accepts(0.)
    assert FixationAssessment(.5, (-.2, 0.)).accepts(0.)
