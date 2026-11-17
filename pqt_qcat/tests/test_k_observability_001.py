"""Unit tests for EXP-K-OBSERVABILITY-001."""
import numpy as np
import pytest

from pqt_qcat.memory.exp_k_observability_001 import (
    compute_forward_rollout,
    compute_directional_derivative,
    analyze_svd_spectrum,
    get_active_echelon_mask,
    compute_full_jacobians_numerical,
)
from pqt_qcat.memory.exp_sleep_need_001 import setup_fresh_core


def test_active_echelon_mask_structure():
    """Verify active echelon mask covers exactly the 5 structural 8x8 blocks (320 synapses)."""
    mask = get_active_echelon_mask()
    assert mask.shape == (48, 48)
    assert np.sum(mask) == 320
    
    # Check specific blocks
    assert np.all(mask[16:24, 0:8])    # E1 -> E3
    assert np.all(mask[24:32, 16:24])  # E3 -> E4
    assert np.all(mask[32:40, 24:32])  # E4 -> E5
    assert np.all(mask[40:48, 8:16])   # E2 -> E6
    assert np.all(mask[32:40, 40:48])  # E6 -> E5


def test_forward_rollout_does_not_mutate():
    """Verify compute_forward_rollout preserves core parameters and live state."""
    core = setup_fresh_core(seed=42)
    K_before = core.K.copy()
    Z_before = core.Z.copy()
    
    Z_out, logits = compute_forward_rollout(core, token=1, ticks=8)
    assert len(Z_out) == 48
    assert len(logits) == 4
    assert np.array_equal(core.K, K_before)
    assert np.array_equal(core.Z, Z_before)


def test_svd_spectrum_properties():
    """Verify SVD spectrum calculation invariants on synthetic and real matrices."""
    # Synthetic matrix
    A = np.random.default_rng(42).standard_normal((24, 2304))
    res = analyze_svd_spectrum(A)
    assert res["shape"] == [24, 2304]
    assert len(res["singular_values"]) == 24
    assert res["numerical_rank"] == 24
    assert res["null_space_dimension"] == 2304 - 24
    assert res["condition_number"] >= 1.0
    
    # Check monotonic descent of singular values
    s = res["singular_values"]
    assert all(s[i] >= s[i+1] for i in range(len(s)-1))
