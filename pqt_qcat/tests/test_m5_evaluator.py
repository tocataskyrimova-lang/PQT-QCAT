"""Unit and regression tests for Milestone M5 Independent Evaluator and Manifest."""
from __future__ import annotations
import json
from pathlib import Path
import pytest
import numpy as np

from pqt_qcat.evaluator.independent_evaluator import (
    IndependentEvaluator,
    EvaluationConfig,
    SEALED_SEEDS,
    CALIBRATION_SEEDS,
)
from pqt_qcat.evaluator.manifest import generate_sha256_manifest, verify_sha256_manifest


def test_sealed_seeds_protection():
    """Verify that evaluator strictly blocks sealed seeds 101-105 without authorization."""
    evaluator = IndependentEvaluator(EvaluationConfig(unseal_authorized=False))
    for s in SEALED_SEEDS:
        with pytest.raises(PermissionError, match="STRICT GATE VIOLATION"):
            evaluator.evaluate_h0(s, n_steps=10)


def test_h0_stability_evaluation():
    """Verify H0 numerical stability evaluation on calibration seed."""
    evaluator = IndependentEvaluator(EvaluationConfig(save_raw_logs=False))
    res = evaluator.evaluate_h0(42, n_steps=100)
    assert res["hypothesis"] == "H0_STABLE"
    assert res["invalid_count"] == 0
    assert res["max_skew_residual"] < 1e-10
    assert res["passed"] is True
    assert res["status"] == "CONFIRMED"


def test_h1_cue_latching_evaluation():
    """Verify H1 cue latching evaluation on calibration seed."""
    evaluator = IndependentEvaluator(EvaluationConfig(save_raw_logs=False))
    res = evaluator.evaluate_h1(42, n_trials=50)
    assert res["hypothesis"] == "H1_HOLD"
    assert res["gap"] == 300
    assert res["accuracy_gap_300"] >= 0.90
    assert res["passed"] is True


def test_h4_budget_symmetry_verified():
    """Verify H4 verifies budget symmetry between PQT_dream and guarded_replay."""
    evaluator = IndependentEvaluator(EvaluationConfig(save_raw_logs=False))
    res = evaluator.evaluate_h4(42, n_episodes=50)
    assert res["hypothesis"] == "H4_DREAM_GAIN"
    assert res["budget_symmetry_verified"] is True


def test_h5_rng_fingerprints_verified():
    """Verify H5 verifies identical RNG fingerprints between PQT_dream and noA_dream."""
    evaluator = IndependentEvaluator(EvaluationConfig(save_raw_logs=False))
    res = evaluator.evaluate_h5(42, n_episodes=50)
    assert res["hypothesis"] == "H5_A_CAUSAL"
    assert res["rng_fingerprints_verified"] is True


def test_h6_provenance_isolation():
    """Verify H6 asserts zero unverified promotions."""
    evaluator = IndependentEvaluator(EvaluationConfig(save_raw_logs=False))
    res = evaluator.evaluate_h6(42, n_queries=100)
    assert res["hypothesis"] == "H6_PROVENANCE"
    assert res["false_promotions_detected"] == 0
    assert res["passed"] is True


def test_sha256_manifest_and_verification():
    """Verify that generating and verifying SHA-256 manifest completes with 100% match."""
    gen_res = generate_sha256_manifest()
    assert gen_res["total_files_hashed"] > 20

    success, count, mismatches = verify_sha256_manifest()
    assert success is True
    assert len(mismatches) == 0
    assert count == gen_res["total_files_hashed"]
