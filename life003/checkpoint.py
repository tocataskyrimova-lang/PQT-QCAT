"""Bit-exact Checkpoint / Resume IO for LIFE-003 complete organism state."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np


def compute_state_hash(state: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 digest of organism state."""
    hasher = hashlib.sha256()

    def walk(val):
        if isinstance(val, np.ndarray):
            hasher.update(str(val.dtype).encode())
            hasher.update(str(val.shape).encode())
            hasher.update(val.tobytes())
        elif isinstance(val, dict):
            for k in sorted(val.keys()):
                hasher.update(str(k).encode())
                walk(val[k])
        elif isinstance(val, (list, tuple)):
            for item in val:
                walk(item)
        else:
            hasher.update(str(val).encode())

    walk(state)
    return hasher.hexdigest()


def save_checkpoint(filepath: str | Path, state: dict[str, Any]) -> str:
    """Save complete organism state to atomic npz archive and return its state hash."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {}
    metadata = {}

    for k, v in state.items():
        if isinstance(v, np.ndarray):
            arrays[k] = v
        else:
            metadata[k] = v

    meta_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    meta_array = np.frombuffer(meta_json.encode("utf-8"), dtype=np.uint8)
    arrays["__meta_json__"] = meta_array

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
        metadata = json.loads(meta_bytes.decode("utf-8"))

        state = {}
        for k in npz.files:
            if k != "__meta_json__":
                state[k] = npz[k]

        state.update(metadata)

    state_hash = compute_state_hash(state)
    return state, state_hash
