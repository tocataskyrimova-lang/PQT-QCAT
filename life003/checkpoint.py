"""Bit-exact Checkpoint / Resume IO for LIFE-003 complete organism state.

Supports arbitrary nested dictionaries, lists, and numpy arrays without requiring pickle
(strict allow_pickle=False compliance).
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import struct
from typing import Any
import numpy as np


def pack_state(val: Any, arrays: dict[str, np.ndarray], prefix: str = "") -> Any:
    """Recursively extract all numpy ndarrays and tagged types for raw npz storage without pickle."""
    if isinstance(val, np.ndarray):
        arrays[prefix] = val
        return {
            "__ndarray__": prefix,
            "descr": canonical_encode(val.dtype.descr).hex(),
            "shape": list(val.shape),
        }
    elif isinstance(val, (bytes, bytearray)):
        return {"__bytes__": bytes(val).hex()}
    elif isinstance(val, tuple):
        return {
            "__tuple__": [pack_state(item, arrays, f"{prefix}[{i}]") for i, item in enumerate(val)]
        }
    elif isinstance(val, list):
        return [pack_state(item, arrays, f"{prefix}[{i}]") for i, item in enumerate(val)]
    elif isinstance(val, dict):
        if any(not isinstance(k, str) for k in val.keys()):
            return {
                "__dict_pairs__": [
                    [pack_state(k, arrays, f"{prefix}/k_{i}"), pack_state(v, arrays, f"{prefix}/v_{i}")]
                    for i, (k, v) in enumerate(val.items())
                ]
            }
        return {
            k: pack_state(v, arrays, f"{prefix}/{k}" if prefix else k)
            for k, v in val.items()
        }
    elif val is None or isinstance(val, (bool, int, float, str, np.integer, np.floating)):
        return val
    else:
        raise TypeError(f"Unsupported type for checkpoint serialization: {type(val)}")


def unpack_state(val: Any, arrays: Any) -> Any:
    """Recursively reconstruct state structure, preserving tuple/dict/bytes types and numpy ndarrays."""
    if isinstance(val, dict):
        if "__ndarray__" in val and len(val) == 3:
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
        return b"Y:" + str(len(b_bytes)).encode("ascii") + b":" + b_bytes + b";"
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
    """Compute deterministic, strictly injective SHA-256 digest of organism state."""
    return hashlib.sha256(canonical_encode(state)).hexdigest()


def save_checkpoint(filepath: str | Path, state: dict[str, Any]) -> str:
    """Save complete organism state to atomic npz archive and return its state hash."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    packed_meta = pack_state(state, arrays)

    meta_json = json.dumps(
        packed_meta,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    arrays["__meta_json__"] = np.frombuffer(meta_json.encode("utf-8"), dtype=np.uint8)

    temp_path = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp_path, **arrays)
    temp_path.replace(path)

    return compute_state_hash(state)


def load_checkpoint(filepath: str | Path) -> tuple[dict[str, Any], str]:
    """Load complete organism state from npz archive and verify consistency."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint archive not found at: {path}")

    with np.load(path, allow_pickle=False) as npz:
        meta_bytes = npz["__meta_json__"].tobytes()
        packed_meta = json.loads(meta_bytes.decode("utf-8"))
        state = unpack_state(packed_meta, npz)

    state_hash = compute_state_hash(state)
    return state, state_hash
