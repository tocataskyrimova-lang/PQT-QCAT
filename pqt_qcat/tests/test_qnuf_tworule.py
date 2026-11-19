"""Exercise the shared token code, legacy no-input shadow, and feedback boundary."""
import copy

import numpy as np
import pytest

from pqt_qcat.memory import qnuf_tworule as c
from pqt_qcat.memory.access_state import exact
from pqt_qcat.memory.functional_sidecar import ActionJournal, state_hash
from pqt_qcat.memory.functional_sidecar_b2a import forecast


CONFIG = dict(substeps=2, h_base=.01, query_ticks=8, eta=.8, clip=.5, gate=1.)


@pytest.mark.parametrize('mode', c.MODES)
def test_context_only_enters_existing_e2_slots_and_keeps_zero_shadow(mode):
    first = c.new_agent('B3', c.SEEDS[0], CONFIG)
    second = copy.deepcopy(first)
    a, b = (c.vector(mode, ctx, 4) for ctx in ('A','B'))
    np.testing.assert_array_equal(a[1:7], b[1:7])
    assert abs(np.linalg.norm(a[[0,7]]) - 1.) < 1e-15
    drive = first.core.B @ (a-b)
    assert np.linalg.norm(drive[8:16]) > 0
    assert not np.any(np.r_[drive[:8],drive[16:]])
    pa, pb = (forecast(agent,4,'Z0-64','aligned',None,{},stimulus=x)
              for agent,x in ((first,a),(second,b)))
    exact(pa['main']['Z_noinput'], pb['main']['Z_noinput'])
    exact(pa['external']['Z_no_input'], pb['external']['Z_no_input'])
    assert pa['budget'] == pb['budget']


def test_explicit_token_only_is_bitwise_legacy():
    first = c.new_agent('B3', c.SEEDS[0], CONFIG)
    second = copy.deepcopy(first)
    a = forecast(first,3,'Z0-64','aligned',None,{})
    b = forecast(second,3,'Z0-64','aligned',None,{},stimulus=np.eye(8)[3])
    exact(a,b)
    assert state_hash(first.snapshot()) == state_hash(second.snapshot())


def test_label_follows_action_and_context_belongs_to_observation(tmp_path):
    env = c.TeachingStream([['B',1]], {'A':[1]*6,'B':[2]*6})
    journal = ActionJournal(tmp_path/'events.jsonl')
    with pytest.raises(RuntimeError):
        env.feedback(journal, None)
    log = {'observation':env.observe(),'query':1,'action':1}
    assert env.feedback(journal,journal.publish(log)).target == 2
    assert env.cursor == 1
