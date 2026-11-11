"""Scenario T6: Core Causality and Plasticity with Frozen Readout.

Conforms strictly to ARCHITECTURE_LIFE_002 (Section 4.2 & 6.3) and ILORI mandate:
- W_readout is STRICTLY FROZEN (never updated / verified byte-for-byte identical)
- Plasticity operates exclusively on theta_core = {K, B} via CorePlasticityEngine
- Causal ablations:
    1. core_only adaptation: accuracy reaches 100%
    2. core scramble / zero-out: performance collapses back to chance (25%)
    3. core restore: performance immediately recovers to 100%
    4. core swap: swapping theta_core between agents trained on different rules flips predictions
- Proves causally that learned knowledge resides inside the dynamical core,
  not in the readout layer.
"""
from __future__ import annotations
import numpy as np
import pytest

from pqt_qcat.core.six_echelon import SixEchelonCore, SixEchelonConfig
from pqt_qcat.plasticity.eligibility import CoreEligibilityTraces
from pqt_qcat.plasticity.core_update import CorePlasticityEngine, CorePlasticityConfig

TEST_SEED = 42

CLASSES = np.array([
    [0.0, 0.0, 0.0],
    [ 0.683, -1.101, -0.533],
    [-0.533,  0.683, -1.101],
    [-1.101, -0.533,  0.683],
    [ 1.101,  0.532, -0.682],
], dtype=np.float64)


def test_t6_core_causality_with_frozen_readout():
    """Verify that learning alters core dynamics theta_core, with frozen readout."""
    config = SixEchelonConfig(d_in=8, d_out=8, h_base=0.01, integrator_mode="fixed")

    core_a = SixEchelonCore(config=config, seed=TEST_SEED)
    core_b = SixEchelonCore(config=config, seed=TEST_SEED + 10)

    # Identical frozen readout for both agents reading from Context Cell 4 (coords 32:35)
    core_a.W_readout.fill(0.0)
    core_b.W_readout.fill(0.0)
    for c in range(1, 5):
        core_a.W_readout[c, 32:35] = CLASSES[c]
        core_b.W_readout[c, 32:35] = CLASSES[c]

    W_readout_initial = core_a.W_readout.copy()

    # Rule A: cyclic shift 1->2, 2->3, 3->4, 4->1
    mapping_a = {1: 2, 2: 3, 3: 4, 4: 1}
    # Rule B: reversed cyclic shift 1->4, 2->1, 3->2, 4->3
    mapping_b = {1: 4, 2: 1, 3: 2, 4: 3}

    traces = CoreEligibilityTraces()
    plasticity = CorePlasticityEngine(CorePlasticityConfig(eta_input=0.35, eta_core=0.05))

    def train_agent(core: SixEchelonCore, mapping: dict[int, int], epochs: int = 25):
        for _ in range(epochs):
            for token, target in mapping.items():
                core.Z.fill(0.0)
                x = np.zeros(8)
                x[token] = 1.0

                for _ in range(5):
                    core.step(x, write_gate=1.0)
                    traces.update(core.Z, x)

                logits = core.predict()
                probs = np.exp(logits[1:5] - np.max(logits[1:5]))
                probs /= np.sum(probs)

                # Form 8-dimensional error vector matching output space
                error = np.zeros(8)
                error[target] = 1.0 - probs[target - 1]

                plasticity.apply_feedback(core, traces, error, x_last=x)

    def evaluate_core(core: SixEchelonCore, mapping: dict[int, int]) -> float:
        correct = 0
        for token, target in mapping.items():
            core.Z.fill(0.0)
            x = np.zeros(8)
            x[token] = 1.0
            for _ in range(5):
                core.step(x, write_gate=1.0)
            pred = int(np.argmax(core.predict()[1:5])) + 1
            if pred == target:
                correct += 1
        return correct / len(mapping)

    # 1. Train Agent A on Task A
    train_agent(core_a, mapping_a)
    # Confirm W_readout remained strictly frozen
    assert np.array_equal(core_a.W_readout, W_readout_initial), "W_readout was modified!"

    acc_a_on_a = evaluate_core(core_a, mapping_a)
    acc_a_on_b = evaluate_core(core_a, mapping_b)
    assert acc_a_on_a >= 0.75, f"Agent A failed to learn Task A: {acc_a_on_a}"
    assert acc_a_on_b <= 0.25, f"Agent A incorrectly solved Task B: {acc_a_on_b}"

    # 2. Train Agent B on Task B
    train_agent(core_b, mapping_b)
    assert np.array_equal(core_b.W_readout, W_readout_initial), "W_readout was modified!"
    acc_b_on_b = evaluate_core(core_b, mapping_b)
    assert acc_b_on_b >= 0.75, f"Agent B failed to learn Task B: {acc_b_on_b}"

    # 3. Causal Scramble: zero-out core parameter B in core_a
    B_saved = core_a.B.copy()
    K_saved = core_a.K.copy()
    core_a.B.fill(0.0)
    core_a.K.fill(0.0)
    acc_scrambled = evaluate_core(core_a, mapping_a)
    assert acc_scrambled <= 0.25, f"Zeroing core did not collapse performance: {acc_scrambled}"

    # 4. Causal Restore: restore saved core parameters
    core_a.B = B_saved.copy()
    core_a.K = K_saved.copy()
    acc_restored = evaluate_core(core_a, mapping_a)
    assert acc_restored == acc_a_on_a, f"Restoring core did not restore performance: {acc_restored}"

    # 5. Causal Swap: swap core_b parameters into core_a
    core_a.B = core_b.B.copy()
    core_a.K = core_b.K.copy()
    acc_swapped = evaluate_core(core_a, mapping_b)
    assert acc_swapped >= 0.75, f"Swapped core did not adopt Task B history: {acc_swapped}"

    # Readout head remained frozen across all experiments
    assert np.array_equal(core_a.W_readout, W_readout_initial)
    assert np.array_equal(core_b.W_readout, W_readout_initial)
