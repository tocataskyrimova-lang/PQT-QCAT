"""Behavioral regressions for defects demonstrated by independent audit."""
import copy
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from pqt_qcat.core.energy import compute_energy
from pqt_qcat.memory.dev_context_pathway_ablation import make_stable_core
from pqt_qcat.memory.dev_joint_rule_calibration import JointAgent
from pqt_qcat.memory.exp_stateful_online_001 import StatefulOnlineAgent


def make_joint():
    return JointAgent(make_stable_core(42), architecture_mode='direct_gating',
                      learning_mode='RESET_TRAIN', t_context=0)


def test_joint_applied_update_and_process_resume(tmp_path):
    agent = make_joint()
    initial = agent.core.K.copy()
    for i in range(12):
        agent.begin_event(i % 2, i % 6 + 1)
        stats = agent.observe_feedback_and_update(i % 4 + 1)
        assert stats['delta_k_norm'] == 0.0
        assert stats['delta_w_norm'] > 0
        assert stats['pred'] is not None
    np.testing.assert_array_equal(initial, agent.core.K)
    # Persist with an outstanding prediction (before feedback) and external stream RNG.
    agent.begin_event(1, 4)
    stream = np.random.default_rng(42000)
    stream.random(9)
    ckpt = tmp_path / 'pending.npz'
    agent.save_checkpoint(ckpt, {'cursor': 12, 'rng': stream.bit_generator.state})
    script = tmp_path / 'child.py'
    script.write_text('''import sys, numpy as np
from pqt_qcat.memory.dev_context_pathway_ablation import make_stable_core
from pqt_qcat.memory.dev_joint_rule_calibration import JointAgent
ag=JointAgent(make_stable_core(42),architecture_mode="direct_gating",t_context=0)
runner=ag.load_checkpoint(sys.argv[1])
rng=np.random.default_rng();rng.bit_generator.state=runner['rng']
assert runner['cursor']==12
ag.observe_feedback_and_update(3)
logits=[]
for i in range(5):
 p,l=ag.begin_event(i%2,int(rng.integers(1,7)));logits.append(l)
 ag.observe_feedback_and_update(i%4+1)
ag.save_checkpoint(sys.argv[2], {'logits':np.array(logits),'cursor':17,'rng':rng.bit_generator.state})
''', encoding='utf-8')
    import os
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])
    subprocess.run([sys.executable, str(script), str(ckpt), str(tmp_path/'child.npz')],
                   check=True, env=env, capture_output=True, text=True)
    agent.observe_feedback_and_update(3)
    expected=[]
    for i in range(5):
        _, l=agent.begin_event(i%2, int(stream.integers(1,7)));expected.append(l)
        agent.observe_feedback_and_update(i%4+1)
    restored=make_joint();runner=restored.load_checkpoint(tmp_path/'child.npz')
    for name in ('W_readout_c0','W_readout_c1'):
        np.testing.assert_array_equal(getattr(agent,name),getattr(restored,name))
    np.testing.assert_array_equal(agent.core.Z,restored.core.Z)
    np.testing.assert_array_equal(agent.core.K,restored.core.K)
    np.testing.assert_array_equal(expected,runner['logits'])


@pytest.mark.parametrize('field,value', [('substeps',4),('u_max',1.),('beta_v',8.),('integrator_tol',1e-8)])
def test_strict_restore_checks_physics(field,value):
    source=make_stable_core(42);dest=make_stable_core(42)
    if field=='substeps': dest.substeps=value
    else: setattr(dest.config,field,value)
    before=dest.Z.copy()
    with pytest.raises(ValueError,match='Incompatible core configuration'):
        dest.restore(source.snapshot())
    np.testing.assert_array_equal(before,dest.Z)
    dest.restore(source.snapshot(),strict_config=False)
    x=np.eye(8)[1]
    source.step(x);dest.step(x)
    np.testing.assert_array_equal(source.Z,dest.Z)


def test_energy_matches_six_cells_and_is_not_zero_placeholder():
    core=make_stable_core(42)
    core.step(np.eye(8)[1])
    expected=sum(float(compute_energy(core.Z[c:c+8],core.cell_params)) for c in range(0,48,8))
    assert core.total_energy()==pytest.approx(expected)
    assert core.total_energy()!=0.0
    assert np.isfinite(core.max_total_energy)
    assert core.step_count==2


def test_clone_read_gate_is_applied_and_preserves_live_state():
    agent=StatefulOnlineAgent(make_stable_core(42),t_context=0)
    # Construct a detectable Hold-to-readout path, so ignoring the gate must fail.
    agent.core.K[16,0]=4.;agent.core.K[24,16]=4.;agent.core.K[32,24]=4.
    before=agent.snapshot()
    a=agent.evaluate_clone(0,1,1,ticks=32,read_gate=0.)
    b=agent.evaluate_clone(0,1,1,ticks=32,read_gate=1.)
    assert np.max(np.abs(np.array(a['logits'])-b['logits']))>1e-8
    np.testing.assert_array_equal(agent.core.Z,before['core_snapshot']['Z'])


def test_nonzero_eligibility_survives_pending_feedback(tmp_path):
    a=StatefulOnlineAgent(make_stable_core(42),learning_mode='STATEFUL_ELIGIBILITY',eligibility_decay=.8)
    a.begin_event(0,2)
    assert np.linalg.norm(a.eligibility_trace)>0
    a.save_checkpoint(tmp_path/'trace.npz')
    b=StatefulOnlineAgent(make_stable_core(42));b.load_checkpoint(tmp_path/'trace.npz')
    a.observe_feedback_and_update(3);b.observe_feedback_and_update(3)
    np.testing.assert_array_equal(a.core.K,b.core.K)
    np.testing.assert_array_equal(a.eligibility_trace,b.eligibility_trace)


def test_stochastic_rng_exact_after_restore():
    from dataclasses import replace
    source=make_stable_core(42)
    source.integrator.config=replace(source.integrator.config,temperature=.001)
    source.step(np.eye(8)[1])
    destination=make_stable_core(999)
    destination.restore(source.snapshot(),strict_config=False)
    for q in (2,4,1):
        source.step(np.eye(8)[q]);destination.step(np.eye(8)[q])
        np.testing.assert_array_equal(source.Z,destination.Z)
