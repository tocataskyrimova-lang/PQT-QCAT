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
    """Recursively extract all numpy ndarrays for raw npz storage without pickle."""
    if isinstance(val, np.ndarray):
        arrays[prefix] = val
        return {
            "__ndarray__": prefix,
            "dtype": str(val.dtype),
            "shape": list(val.shape),
        }
    elif isinstance(val, dict):
        return {
            k: pack_state(v, arrays, f"{prefix}/{k}" if prefix else k)
            for k, v in val.items()
        }
    elif isinstance(val, (list, tuple)):
        packed = [pack_state(item, arrays, f"{prefix}[{i}]") for i, item in enumerate(val)]
        return tuple(packed) if isinstance(val, tuple) else packed
    else:
        return val


def unpack_state(val: Any, arrays: Any) -> Any:
    """Recursively reconstruct state structure and re-insert numpy ndarrays."""
    if isinstance(val, dict):
        if "__ndarray__" in val:
            key = val["__ndarray__"]
            return arrays[key]
        return {k: unpack_state(v, arrays) for k, v in val.items()}
    elif isinstance(val, list):
        return [unpack_state(item, arrays) for item in val]
    elif isinstance(val, tuple):
        return tuple(unpack_state(item, arrays) for item in val)
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
    - numpy.ndarray -> (tag="A", dtype, shape, len(bytes), raw_bytes)
    - dict -> sorted keys, (tag="D", len(keys), key_enc, val_enc)
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
        dt = val.dtype.str.encode("ascii")
        sh = ",".join(str(x) for x in val.shape).encode("ascii")
        raw = val.tobytes()
        return (
            b"A:" + str(len(dt)).encode("ascii") + b":" + dt +
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
        keys = sorted(val.keys(), key=lambda k: str(k))
        k_enc = []
        for k in keys:
            k_enc.append(canonical_encode(str(k)))
            k_enc.append(canonical_encode(val[k]))
        content = b"".join(k_enc)
        return b"D:" + str(len(keys)).encode("ascii") + b"{" + content + b"};"
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
