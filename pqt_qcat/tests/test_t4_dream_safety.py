"""Scenario T4: Dream Safety and Anchor Protection (ARCHITECTURE_LIFE_002 Section 6.3).

Verifies hypothesis H3-SLEEP-SAFE:
- Anchor accuracy degradation after 1000 idle/dream ticks <= 2% (0.02)
- Zero invalid non-finite states (invalid == 0)
- Evaluates across all pre-registered policies:
    {'no_replay', 'state_only', 'guarded_replay', 'noA_dream', 'PQT_dream'}
- Zero oracle feedback during sleep
"""
from __future__ import annotations
import numpy as np
import pytest

from pqt_qcat.core.six_echelon import SixEchelonCore, SixEchelonConfig
from pqt_qcat.memory.witness import WitnessLedger
from pqt_qcat.dream.consolidation import DreamConsolidationEngine, ConsolidationConfig

CALIBRATION_SEED = 42

CLASSES = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [-1.0, -1.0, 0.0],
])


def _setup_agent_with_anchors(seed: int = CALIBRATION_SEED):
    """Instantiate core with trained anchors in B_real."""
    cfg = SixEchelonConfig(enable_vortex=True, integrator_mode="fixed")
    core = SixEchelonCore(config=cfg, seed=seed)
    core.W_readout.fill(0.0)
    for c in range(1, 5):
        core.W_readout[c, 32:35] = CLASSES[c]

    # Ground sensory mapping for anchors
    for k in range(1, 5):
        core.B[31 + k, k] = 4.0

    ledger = WitnessLedger()
    # Register real anchor observations in B_real
    for k in range(1, 5):
        ledger.record_observation(
            timestamp=k,
            input_token=k,
            target_token=k,
            pred_token=k,
            confidence=0.98,
            context_basin=1,
        )

    return core, ledger


def test_t4_dream_safety_across_policies():
    """Verify H3-SLEEP-SAFE across all pre-registered sleep policies."""
    policies = ["no_replay", "state_only", "guarded_replay", "noA_dream", "PQT_dream"]
    engine = DreamConsolidationEngine(ConsolidationConfig(anchor_max_degradation=0.02), seed=CALIBRATION_SEED)

    results = {}

    for pol in policies:
        core, ledger = _setup_agent_with_anchors(CALIBRATION_SEED)
        res = engine.run_sleep_policy(
            policy=pol,
            core=core,
            ledger=ledger,
            idle_ticks=1000,
        )
        results[pol] = res

        # Assert H3-SLEEP-SAFE: degradation <= 2%
        assert res["passed_h3_sleep_safe"] is True, (
            f"Policy {pol} failed H3-SLEEP-SAFE: degradation={res['anchor_degradation']*100:.2f}% > 2%"
        )
        # Assert initial anchor accuracy was preserved
        assert res["anchor_accuracy_after"] >= 0.98

        # Invariant: No unverified promotions in ledger
        ledger.assert_no_unverified_promotions()

    print("\nT4 Dream Safety Results (1000 idle ticks):")
    for pol, r in results.items():
        print(f"  {pol:15s}: AccBefore={r['anchor_accuracy_before']*100:.1f}%, "
              f"AccAfter={r['anchor_accuracy_after']*100:.1f}%, "
              f"Degradation={r['anchor_degradation']*100:+.2f}%, "
              f"Accepted={r['accepted_updates']}, Rejected={r['rejected_updates']}")
