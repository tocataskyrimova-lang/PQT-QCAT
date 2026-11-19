import numpy as np

from pqt_qcat.memory.qnuf_f_gate import acquisition, memory_gate, proposals, select


def score(nll,correct=1):
    return {'nll':nll,'correct':correct}


def test_f_is_per_anchor_and_rejects_class_loss():
    base = {'A:1':score(.4),'A:2':score(.3)}
    assert memory_gate(base,{'A:1':score(.405),'A:2':score(.295)},.01)['passes']
    assert not memory_gate(base,{'A:1':score(.42),'A:2':score(.10)},.01)['passes']
    assert not memory_gate(base,{'A:1':score(.39,0),'A:2':score(.29)},.01)['passes']


def test_current_witness_filter_and_first_admissible_repair():
    assert acquisition(score(.9,0),score(.78,0),.10)['passes']
    assert not acquisition(score(.9,0),score(.85,0),.10)['passes']
    assert not acquisition(score(.9),score(.78,0),.10)['passes']
    assert select({'RAW':False,'MIX25':True,'CTX':True},['RAW','MIX25','CTX'])=={
        'selected':'MIX25','decision':'REPAIRED'}
    assert select({'RAW':False},['RAW'])['decision']=='NO_FEASIBLE_UPDATE'


def test_mixes_keep_cap_and_match_the_small_radius():
    k = np.array([[.1,.2],[.3,.4]])
    raw,ctx = np.array([[.5,.3],[0.,0.]]),np.array([[.02,.35],[0.,0.]])
    got = proposals(k,raw,ctx,.5)
    for name in ('RAW_SMALL','MIX25','MIX50','MIX75','CTX'):
        proposal,end,info = got[name]
        assert info['status']=='OK' and np.max(np.abs(proposal))<=.5
        np.testing.assert_allclose(np.linalg.norm(end-k),np.linalg.norm(ctx),rtol=1e-12)
        np.testing.assert_array_equal(end[1],k[1])
