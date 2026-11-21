"""Public API regressions for the RC3 checkpoint and witness counterexamples."""
import json

import numpy as np
import pytest

from life003.checkpoint import canonical_encode, compute_state_hash, load_checkpoint, save_checkpoint
from life003.clock import LifeClock
from life003.hypotheses import Hypothesis, HypothesisPool
from life003.ledger import SaksinLedger
from life003.sankalpa import SankalpaGate
from life003.scheduler import ContinuousLifeOrganism


@pytest.mark.parametrize("value", [
    {"__tuple__": [1, 2]}, {"__bytes__": "4142"},
    {"__dict_pairs__": [["k", 1]]}, {"__ndarray__": "x", "descr": "", "shape": [1]},
    {"__scalar__": "array_0"}, {"__bytearray__": "4142"},
    bytearray(b"AB"), b"AB", (1, 2), {1: "a", "1": "b"}, np.int16(42),
    np.array([(17,)], dtype=[("x", "<i4")]),
])
def test_round_trip_preserves_literal_types(tmp_path, value):
    p = tmp_path / "state.npz"
    original = {"value": value}
    saved = save_checkpoint(p, original)
    state, loaded = load_checkpoint(p)
    assert type(state["value"]) is type(value)
    assert canonical_encode(state) == canonical_encode(original)
    assert loaded == saved == compute_state_hash(original)
    if isinstance(value, np.ndarray):
        assert state["value"].dtype == value.dtype


def test_array_member_paths_cannot_alias(tmp_path):
    state = {"a/b": np.array([1]), "a": {"b": np.array([2])},
             "__meta_json__": np.array([3]), "a[0]": np.array([4])}
    p = tmp_path / "state.npz"
    saved = save_checkpoint(p, state)
    restored, digest = load_checkpoint(p)
    assert saved == digest
    for actual, expected in ((restored["a/b"], [1]), (restored["a"]["b"], [2]),
                             (restored["__meta_json__"], [3]), (restored["a[0]"], [4])):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("damage", ["state_hash", "array", "state", "missing_hash"])
def test_load_rejects_inconsistent_checkpoint(tmp_path, damage):
    p = tmp_path / "state.npz"
    save_checkpoint(p, {"a": np.array([1])})
    with np.load(p, allow_pickle=False) as archive:
        members = dict(archive)
    envelope = json.loads(members["__meta_json__"].tobytes())
    if damage == "array":
        members["array_0"][0] += 1
    elif damage == "state":
        envelope["state"] = {"__dict_pairs__": []}
    elif damage == "missing_hash":
        del envelope["state_hash"]
    else:
        envelope["state_hash"] = "0" * 64
    members["__meta_json__"] = np.frombuffer(json.dumps(envelope).encode(), dtype=np.uint8)
    np.savez_compressed(p, **members)
    with pytest.raises(ValueError):
        load_checkpoint(p)


def test_unrepresentable_nested_dtype_preserves_previous_archive(tmp_path):
    p = tmp_path / "state.npz"
    save_checkpoint(p, {"value": 7})
    previous = p.read_bytes()
    dtype = np.dtype([("x", np.dtype("i4", metadata={"unit": "ms"}))])
    with pytest.raises(TypeError):
        save_checkpoint(p, {"value": np.zeros(1, dtype=dtype)})
    assert p.read_bytes() == previous


def evidence():
    return dict(hypothesis_id="H", action=2, result="MATCH", provenance="H",
                applicability_domain="D", confirmed=True, support_score=0.60)


@pytest.mark.parametrize("field", [
    "hypothesis_id", "action", "result", "provenance", "applicability_domain", "confirmed", "support_score",
])
@pytest.mark.parametrize("invalid", ["absent", "null", "blank"])
def test_incomplete_record_denied_at_all_entry_points(field, invalid):
    payload = evidence()
    if invalid == "absent":
        del payload[field]
    else:
        payload[field] = None if invalid == "null" else "   "
    ledger = SaksinLedger()
    ref = ledger.append_entry("VERIFIED_FEEDBACK", payload, LifeClock())["entry_hash"]
    pool = HypothesisPool()
    ctx = {"domain": "D"}
    with pytest.raises(ValueError):
        pool.register_candidate("H", 2, "SUPPORTED", ref, ctx, ledger)
    h = pool.register_candidate("H", 2, context=ctx)
    with pytest.raises(ValueError):
        pool.audit_verification("H", True, ref, ledger)
    assert h.v_just == "UNVERIFIED" and h.witness_ref is None
    assert SankalpaGate().decide(2, ref, ledger=ledger, hypothesis=h, context=ctx).epistemic_channel == "ABSTAIN"


@pytest.mark.parametrize(("candidate", "action", "domain"), [
    (None, 2, "D"), ("", 2, "D"), ("H", None, "D"), ("H", "", "D"),
    ("H", 2, None), ("H", 2, "  "), ("other", 2, "D"), ("H", "2", "D"), ("H", 2, "other"),
])
def test_missing_or_different_request_identity(candidate, action, domain):
    ledger = SaksinLedger()
    ref = ledger.append_verified_feedback(**evidence())["entry_hash"]
    ctx = {"domain": domain}
    pool = HypothesisPool()
    with pytest.raises(ValueError):
        pool.register_candidate(candidate, action, "SUPPORTED", ref, ctx, ledger)
    pool.register_candidate(candidate, action, context=ctx)
    with pytest.raises(ValueError):
        pool.audit_verification(candidate, True, ref, ledger)


@pytest.mark.parametrize("missing", ["hypothesis_id", "applicability_domain", "provenance", "result", "action"])
def test_writer_rejects_incomplete_record_before_insert(missing):
    ledger = SaksinLedger()
    with pytest.raises(ValueError):
        ledger.append_verified_feedback(**(evidence() | {missing: ""}))
    assert ledger.seq == 0


@pytest.mark.parametrize("score", [0.59, 0.60, 0.79, 0.80])
def test_supported_uses_same_configured_threshold_as_commit(score):
    threshold = 0.80
    ledger = SaksinLedger()
    ref = ledger.append_verified_feedback(**(evidence() | {"support_score": score}))["entry_hash"]
    pool = HypothesisPool(witness_support_min_score=threshold)
    ctx = {"domain": "D"}
    h = pool.register_candidate("H", 2, context=ctx)
    decision = SankalpaGate(witness_support_min_score=threshold).decide(
        2, ref, ledger=ledger, hypothesis=h, context=ctx)
    if score < threshold:
        assert decision.epistemic_channel == "CLASS_ONLY"
        with pytest.raises(ValueError):
            pool.audit_verification("H", True, ref, ledger)
    else:
        assert decision.epistemic_channel == "COMMIT"
        pool.audit_verification("H", True, ref, ledger)
        assert h.v_just == "SUPPORTED"


def test_direct_supported_construction_requires_ledger():
    with pytest.raises(ValueError):
        Hypothesis("H", 2, "SUPPORTED", witness_ref="FAKE_WITNESS_999", context={"domain": "D"})


def test_restore_revalidates_with_receiving_ledger(tmp_path):
    source = ContinuousLifeOrganism()
    ref = source.ledger.append_verified_feedback(**evidence())["entry_hash"]
    source.hypotheses.register_candidate("H", 2, "SUPPORTED", ref, {"domain": "D"}, source.ledger)
    snapshot = source.hypotheses.snapshot()
    for other in (None, SaksinLedger()):
        with pytest.raises(ValueError):
            HypothesisPool.restore(snapshot, other)
    archive = tmp_path / "supported.npz"
    save_checkpoint(archive, source.capture_full_state())
    state, _ = load_checkpoint(archive)
    receiving = ContinuousLifeOrganism()
    receiving.restore_full_state(state)
    assert receiving.hypotheses.hypotheses["H"].v_just == "SUPPORTED"
    state["ledger"] = SaksinLedger().snapshot()
    previous = compute_state_hash(receiving.capture_full_state())
    with pytest.raises(ValueError):
        receiving.restore_full_state(state)
    assert compute_state_hash(receiving.capture_full_state()) == previous



def test_legacy_checkpoint_requires_external_original_hash(tmp_path):
    state = {"clock": 7}
    p = tmp_path / "rc3.npz"
    meta = np.frombuffer(json.dumps(state).encode(), dtype=np.uint8)
    np.savez_compressed(p, __meta_json__=meta)
    with pytest.raises(ValueError, match="original state hash"):
        load_checkpoint(p)
    with pytest.raises(ValueError, match="state hash mismatch"):
        load_checkpoint(p, expected_state_hash="0" * 64)
    assert load_checkpoint(p, expected_state_hash=compute_state_hash(state))[0] == state


@pytest.mark.parametrize(("field", "value"), [
    ("provenance", "unrelated"), ("confirmed", 1), ("confirmed", False),
    ("support_score", 0.59), ("support_score", True), ("action", 3),
    ("applicability_domain", "other"), ("hypothesis_id", "other"),
])
def test_restore_rejects_inapplicable_support(field, value):
    source = SaksinLedger()
    ref = source.append_verified_feedback(**evidence())["entry_hash"]
    pool = HypothesisPool()
    pool.register_candidate("H", 2, "SUPPORTED", ref, {"domain": "D"}, source)
    other = SaksinLedger()
    record = other.append_entry("VERIFIED_FEEDBACK", evidence() | {field: value}, LifeClock())
    snapshot = pool.snapshot()
    snapshot["hypotheses"]["H"]["witness_ref"] = record["entry_hash"]
    with pytest.raises(ValueError):
        HypothesisPool.restore(snapshot, ledger=other)
