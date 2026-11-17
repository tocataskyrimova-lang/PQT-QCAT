"""Core security, seed isolation, and immutability validation for PQT-QCat."""
from __future__ import annotations

from typing import Any, Iterable, Set

# Strict immutable set of final sealed M5 seeds that must NEVER be accessed for exploratory training
FORBIDDEN_FINAL_SEEDS: frozenset[int] = frozenset([101, 102, 103, 104, 105])

# Quarantined seeds from the November 17 external reader incident (must not be used for confirmation)
QUARANTINED_SEEDS: frozenset[int] = frozenset(range(100, 112))

# Debug / historical reproducibility seeds
DEBUG_REPRODUCTION_SEEDS: frozenset[int] = frozenset([11, 12, 13, 42])


def check_seed_safety(seed: int, allow_quarantined: bool = False) -> None:
    """Validate that a seed is strictly allowed for use.
    
    Raises:
        ValueError: If seed is in FORBIDDEN_FINAL_SEEDS or QUARANTINED_SEEDS.
    """
    if seed in FORBIDDEN_FINAL_SEEDS:
        raise ValueError(
            f"SECURITY VIOLATION: Seed {seed} is in FORBIDDEN_FINAL_SEEDS {sorted(FORBIDDEN_FINAL_SEEDS)}. "
            f"sealed M5 seeds are strictly prohibited from all exploratory and development training."
        )
    if not allow_quarantined and seed in QUARANTINED_SEEDS:
        raise ValueError(
            f"SECURITY VIOLATION: Seed {seed} is in QUARANTINED_SEEDS (100-111). "
            "This seed range was compromised in an earlier exploratory reader experiment and is quarantined."
        )


def enforce_seed_list_safety(seeds: Iterable[int], allow_quarantined: bool = False) -> list[int]:
    """Validate a list of seeds and return them if safe."""
    seed_list = list(seeds)
    for s in seed_list:
        check_seed_safety(s, allow_quarantined=allow_quarantined)
    return seed_list


def validate_config_seed_integrity(config: dict, allow_quarantined: bool = True) -> None:
    """Ensure user configuration cannot weaken the built-in forbidden list.
    
    Even if config sets 'forbidden_seeds': [] or omits it, the built-in list is enforced.
    Recursively scans the configuration structure to reject any forbidden seeds.
    """
    if config.get("run_m5", False):
        raise ValueError("SECURITY VIOLATION: run_m5 must be False; M5 final evaluation is sealed.")
        
    user_forbidden = set(config.get("forbidden_seeds", []))
    # Check that user config did not attempt to weaken the forbidden list
    missing_forbidden = FORBIDDEN_FINAL_SEEDS - user_forbidden
    if missing_forbidden and "forbidden_seeds" in config:
        raise ValueError(
            f"SECURITY VIOLATION: User config weakened forbidden_seeds list. "
            f"Missing required final seeds: {sorted(missing_forbidden)}"
        )

    def _scan_obj(obj: Any, key_name: str = "") -> None:
        if key_name == "forbidden_seeds":
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                _scan_obj(v, key_name=str(k))
        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                _scan_obj(item, key_name=key_name)
        elif isinstance(obj, int) and not isinstance(obj, bool):
            if "seed" in key_name.lower():
                check_seed_safety(obj, allow_quarantined=allow_quarantined)

    _scan_obj(config)
