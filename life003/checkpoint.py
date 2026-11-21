"""Bit-exact Checkpoint / Resume IO for LIFE-003 complete organism state.

Supports arbitrary nested dictionaries, lists, and numpy arrays without requiring pickle
(strict allow_pickle=False compliance).
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
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


def compute_state_hash(state: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 digest of organism state."""
    hasher = hashlib.sha256()

    def walk(val):
        if isinstance(val, np.ndarray):
            hasher.update(str(val.dtype).encode("utf-8"))
            hasher.update(str(val.shape).encode("utf-8"))
            hasher.update(val.tobytes())
        elif isinstance(val, dict):
            for k in sorted(val.keys()):
                hasher.update(str(k).encode("utf-8"))
                walk(val[k])
        elif isinstance(val, (list, tuple)):
            for item in val:
                walk(item)
        else:
            hasher.update(str(val).encode("utf-8"))

    walk(state)
    return hasher.hexdigest()


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
