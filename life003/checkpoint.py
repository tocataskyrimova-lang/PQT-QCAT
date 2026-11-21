"""Bit-exact Checkpoint / Resume IO for LIFE-003 complete organism state.

Supports arbitrary nested dictionaries, lists, and numpy arrays without requiring pickle
(strict allow_pickle=False compliance).
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import struct
import tempfile
from typing import Any
import numpy as np


def _check_dtype(dtype: np.dtype) -> None:
    # NPY drops dtype metadata and alignment flags, including on nested fields.
    if dtype.hasobject or dtype.metadata is not None or dtype.isalignedstruct or dtype.type is np.record:
        raise TypeError(f"Unsupported checkpoint dtype: {dtype!r}")
    for name in dtype.names or ():
        _check_dtype(dtype.fields[name][0])
    if dtype.subdtype is not None:
        _check_dtype(dtype.subdtype[0])


def pack_state(val: Any, arrays: dict[str, np.ndarray]) -> Any:
    """Encode every dictionary as pairs; only the encoder emits tag objects."""
    kind = type(val)
    if kind is np.ndarray or isinstance(val, np.generic):
        arr = np.asarray(val)
        _check_dtype(arr.dtype)
        if kind is not np.ndarray and (kind is not arr.dtype.type or kind is np.record):
            raise TypeError(f"Unsupported NumPy scalar subclass: {kind}")
        key = f"array_{len(arrays)}"
        arrays[key] = arr
        return {"__ndarray__" if kind is np.ndarray else "__scalar__": key}
    if kind is dict:
        return {"__dict_pairs__": [
            [pack_state(k, arrays), pack_state(v, arrays)] for k, v in val.items()
        ]}
    if kind is tuple:
        return {"__tuple__": [pack_state(item, arrays) for item in val]}
    if kind is list:
        return [pack_state(item, arrays) for item in val]
    if kind in (bytes, bytearray):
        return {"__bytes__" if kind is bytes else "__bytearray__": val.hex()}
    if val is None or kind in (bool, int, float, str):
        return val
    raise TypeError(f"Unsupported checkpoint value: {kind}")


def unpack_state(val: Any, arrays: Any) -> Any:
    """Recursively reconstruct state structure, preserving tuple/dict/bytes types and numpy ndarrays."""
    if isinstance(val, dict):
        if "__scalar__" in val and len(val) == 1:
            return arrays[val["__scalar__"]][()]
        if "__bytearray__" in val and len(val) == 1:
            return bytearray.fromhex(val["__bytearray__"])
        if "__ndarray__" in val and len(val) in (1, 3):
            key = val["__ndarray__"]
            return arrays[key]
        if "__bytes__" in val and len(val) == 1:
            return bytes.fromhex(val["__bytes__"])
        if "__tuple__" in val and len(val) == 1:
            return tuple(unpack_state(item, arrays) for item in val["__tuple__"])
        if "__dict_pairs__" in val and len(val) == 1:
            return {
                unpack_state(pair[0], arrays): unpack_state(pair[1], arrays)
                for pair in val["__dict_pairs__"]
            }
        return {k: unpack_state(v, arrays) for k, v in val.items()}
    elif isinstance(val, list):
        return [unpack_state(item, arrays) for item in val]
    else:
        return val


def canonical_encode(val: Any) -> bytes:
    """Type-tagged, length-prefixed canonical binary serialization.
    
    Guarantees injectivity across types, boundaries, and sequence elements:
    - [1, 23] vs [12, 3] -> distinct digests
    - 1 vs "1" -> distinct digests
    - True vs 1 -> distinct digests
    - [1, 2] vs (1, 2) -> distinct digests
    - None vs "None" -> distinct digests
    - {1: "a"} vs {"1": "a"} -> distinct digests (recursive key canonical_encode)
    - numpy structured dtypes with different field names -> distinct digests (canonical_encode of descr)
    - numpy.ndarray -> (tag="A", descr_enc, shape, len(bytes), raw_bytes)
    - dict -> sorted by canonical bytes of keys, (tag="D", len(pairs), key_enc + val_enc)
    """
    if val is None:
        return b"N;"
    if isinstance(val, np.generic):
        return b"G:" + canonical_encode(np.asarray(val))
    if isinstance(val, bool):
        return b"B:1;" if val else b"B:0;"
    if isinstance(val, (int, np.integer)):
        s = str(int(val)).encode("ascii")
        return b"I:" + str(len(s)).encode("ascii") + b":" + s + b";"
    if isinstance(val, (float, np.floating)):
        return b"F:" + struct.pack(">d", float(val)) + b";"
    if isinstance(val, str):
        s_bytes = val.encode("utf-8")
        return b"S:" + str(len(s_bytes)).encode("ascii") + b":" + s_bytes + b";"
    if isinstance(val, (bytes, bytearray)):
        b_bytes = bytes(val)
        tag = b"Y:" if isinstance(val, bytes) else b"J:"
        return tag + str(len(b_bytes)).encode("ascii") + b":" + b_bytes + b";"
    if isinstance(val, np.ndarray):
        dt_descr = canonical_encode(val.dtype.descr)
        sh = ",".join(str(x) for x in val.shape).encode("ascii")
        raw = val.tobytes()
        return (
            b"A:" + str(len(dt_descr)).encode("ascii") + b":" + dt_descr +
            b":" + str(len(sh)).encode("ascii") + b":" + sh +
            b":" + str(len(raw)).encode("ascii") + b":" + raw + b";"
        )
    if isinstance(val, list):
        items = b"".join(canonical_encode(x) for x in val)
        return b"L:" + str(len(val)).encode("ascii") + b"[" + items + b"];"
    if isinstance(val, tuple):
        items = b"".join(canonical_encode(x) for x in val)
        return b"T:" + str(len(val)).encode("ascii") + b"(" + items + b");"
    if isinstance(val, dict):
        encoded_pairs = [(canonical_encode(k), canonical_encode(v)) for k, v in val.items()]
        encoded_pairs.sort(key=lambda pair: pair[0])
        content = b"".join(k_bytes + v_bytes for k_bytes, v_bytes in encoded_pairs)
        return b"D:" + str(len(val)).encode("ascii") + b"{" + content + b"};"
    raise TypeError(f"Unsupported type for canonical encoding: {type(val)}")


def compute_state_hash(state: dict[str, Any]) -> str:
    """Hash the canonical, type-tagged encoding of the organism state."""
    return hashlib.sha256(canonical_encode(state)).hexdigest()


def save_checkpoint(filepath: str | Path, state: dict[str, Any]) -> str:
    """Save complete organism state to atomic npz archive and return its state hash."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    packed = pack_state(state, arrays)
    original_hash = compute_state_hash(state)
    envelope = {"format": "life003-rc4", "state_hash": original_hash, "state": packed}
    meta = json.dumps(envelope, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    arrays["__meta_json__"] = np.frombuffer(meta.encode("utf-8"), dtype=np.uint8)

    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        np.savez_compressed(temp_path, **arrays)
        load_checkpoint(temp_path, expected_state_hash=original_hash)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)
    return original_hash


def load_checkpoint(filepath: str | Path, *, expected_state_hash: str | None = None) -> tuple[dict[str, Any], str]:
    """Verify the recorded digest. RC3 archives require an external original digest."""
    with np.load(filepath, allow_pickle=False) as npz:
        meta = json.loads(npz["__meta_json__"].tobytes())
        if meta.get("format") == "life003-rc4":
            recorded_hash = meta.get("state_hash")
            if not isinstance(recorded_hash, str):
                raise ValueError("Missing original checkpoint state hash")
            packed = meta["state"]
        elif expected_state_hash is not None:
            recorded_hash = expected_state_hash
            packed = meta
        else:
            raise ValueError("Missing checkpoint format or original state hash")
        state = unpack_state(packed, npz)
    actual_hash = compute_state_hash(state)
    if actual_hash != recorded_hash or (expected_state_hash is not None and actual_hash != expected_state_hash):
        raise ValueError("Checkpoint state hash mismatch")
    return state, actual_hash
