"""Pre-registered configuration loader & validator for LIFE-003 T1."""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "life003_t1_prereg.yaml"
EXPECTED_CONFIG_SHA256 = "09efa0a3a497dbc33637fd99921d23e42f799576555deb6d85af4a9642ea683e"


def compute_file_sha256(path: str | Path) -> str:
    """Compute standard SHA-256 hex digest of file bytes."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_and_validate_prereg_config(
    path: str | Path | None = None,
    enforce_sha: bool = True,
    expected_sha: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Load, validate, and check safety bounds of configs/life003_t1_prereg.yaml.
    
    Returns:
        tuple of (config_dict, sha256_hash)
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Prereg configuration file not found at: {cfg_path}")

    actual_sha = compute_file_sha256(cfg_path)
    target_sha = expected_sha or EXPECTED_CONFIG_SHA256

    if enforce_sha and actual_sha != target_sha:
        raise ValueError(
            f"Config SHA-256 verification failed! Expected: {target_sha}, got: {actual_sha}"
        )

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Validate mandatory contract invariants
    if cfg.get("contract_version") != "v0.2.1":
        raise ValueError(f"Unexpected contract version: {cfg.get('contract_version')}")

    # Check seed isolation and banned quarantine seeds
    rng_cfg = cfg.get("rng_streams", {})
    banned = set(rng_cfg.get("banned_seeds", [101, 102, 103, 104, 105]))

    for stream in ("rng_main_seed", "rng_sidecar_seed", "rng_world_seed"):
        seed_val = rng_cfg.get(stream)
        if seed_val in banned:
            raise ValueError(f"Quarantined M5 seed {seed_val} detected in stream {stream}!")

    # Guard against accidental plastic modification in T1
    guards = cfg.get("frozen_core_guards", {})
    if guards.get("k_write_gate_active", 0.0) != 0.0:
        raise ValueError("k_write_gate_active must be 0.0 in T1!")

    return cfg, actual_sha
