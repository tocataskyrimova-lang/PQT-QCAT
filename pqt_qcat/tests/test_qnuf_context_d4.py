import numpy as np

from pqt_qcat.memory.qnuf_context_d4 import match_update, within_norm


def test_ray_match_preserves_zeros_exact_norm_and_direction():
    k = np.array([[1.6, .12], [.2, -.97]])
    direction = np.array([[.03, -.06], [0., .1]])
    u, end, info = match_update(k,direction,.29,.5)
    assert info['status']=='OK'
    assert within_norm(np.linalg.norm(end-k),.29)
    np.testing.assert_allclose(u,direction*(.29/np.linalg.norm(direction)),rtol=1e-15,atol=0)
    assert end[1,0]==k[1,0]


def test_component_cap_on_ray_is_never_reclipped():
    k = np.zeros((2,2))
    delta = np.diag([.5,.25])
    u, end, info = match_update(k,delta,1.,.5)
    assert u is end is None and info['status']=='INFEASIBLE_MATCH' and info['reason']=='component_cap_on_ray'
    target = np.linalg.norm(delta)
    u, end, info = match_update(k,delta,target,.5,original=True)
    assert info['status']=='OK' and info['peak_proposed']==.5
    np.testing.assert_array_equal(u,delta)


def test_zero_and_nonfinite_candidates_are_explicit():
    k = np.zeros((2,2))
    _, end, info = match_update(k,k,.2,.5)
    assert end is None and info['reason']=='zero_direction'
    _, end, info = match_update(k,k,0.,.5,original=True)
    assert info['status']=='ZERO_RADIUS'
    np.testing.assert_array_equal(end,k)
    _, end, info = match_update(k,np.full_like(k,np.nan),.2,.5)
    assert end is None and info['status']=='NONFINITE_CANDIDATE'
