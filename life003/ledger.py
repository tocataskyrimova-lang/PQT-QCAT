"""Immutable hash-chained Sākṣin Ledger for LIFE-003 (PQT-KC-001 v0.3)."""
from __future__ import annotations
import copy
import hashlib
import json
from typing import Any

from .clock import LifeClock

ALLOWED_RECORD_TYPES = {
    "CONFIG_REGISTERED",
    "OBSERVED",
    "VERIFIED_FEEDBACK",
    "HYPOTHESIS",
    "REJECTED",
    "REVISED",
    "AUDIT_SHADOW",
    "EVICTED",
}


def canonical_json_bytes(obj: Any) -> bytes:
    """Deterministic JSON serialization without whitespace, sorted keys, no floats NaN."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


class SaksinLedger:
    """Append-only witness and belief ledger with strict SHA-256 hash chaining."""

    def __init__(self, genesis_config_sha: str | None = None, genesis_clock: LifeClock | None = None):
        self.records: list[dict[str, Any]] = []
        self.prev_hash: str = "0" * 64
        self.seq: int = 0

        # Register genesis record if config hash is provided
        if genesis_config_sha is not None:
            clk = genesis_clock or LifeClock()
            self.append_entry(
                record_type="CONFIG_REGISTERED",
                payload={"prereg_manifest_hash": genesis_config_sha, "contract_version": "v0.2.1"},
                clock=clk,
            )

    def append_entry(
        self,
        record_type: str,
        payload: dict[str, Any],
        clock: LifeClock,
    ) -> dict[str, Any]:
        """Append a new strictly hashed entry to the ledger."""
        if record_type not in ALLOWED_RECORD_TYPES:
            raise ValueError(f"Record type '{record_type}' is not permitted in SaksinLedger!")

        # Hash payload excludes any wall-clock or random non-reproducible tokens
        canonical_payload = copy.deepcopy(payload)
        
        entry_meta = {
            "seq": self.seq,
            "prev_hash": self.prev_hash,
            "core_tick": clock.core_tick,
            "world_tick": clock.world_tick,
            "record_type": record_type,
            "payload": canonical_payload,
        }

        entry_bytes = canonical_json_bytes(entry_meta)
        entry_hash = hashlib.sha256(entry_bytes).hexdigest()

        record = {
            **entry_meta,
            "entry_hash": entry_hash,
        }

        self.records.append(record)
        self.prev_hash = entry_hash
        self.seq += 1
        return record

    def append_verified_feedback(
        self,
        hypothesis_id: str | None,
        action: int | str,
        result: Any,
        provenance: str | dict[str, Any],
        applicability_domain: str | dict[str, Any],
        support_score: float = 1.0,
        confirmed: bool = True,
        clock: LifeClock | None = None,
    ) -> dict[str, Any]:
        """Append a strictly verified feedback record to the ledger.
        
        Requires explicit provenance, action/result outcome, and applicability domain.
        """
        payload = {
            "hypothesis_id": hypothesis_id,
            "action": action,
            "result": result,
            "provenance": provenance,
            "applicability_domain": applicability_domain,
            "support_score": float(support_score),
            "confirmed": bool(confirmed),
        }
        clk = clock or LifeClock()
        return self.append_entry(
            record_type="VERIFIED_FEEDBACK",
            payload=payload,
            clock=clk,
        )

    def get_entry(self, entry_hash: str) -> dict[str, Any] | None:
        """Find record by its entry_hash."""
        for rec in self.records:
            if rec["entry_hash"] == entry_hash:
                return rec
        return None

    def verify_witness_reference(
        self,
        witness_ref: str | None,
        required_types: set[str] | None = None,
    ) -> tuple[bool, str]:
        """FIX-4: Cryptographically verify that witness_ref points to an authentic ledger record.
        
        Returns:
            (is_valid, reason)
        """
        if witness_ref is None:
            return False, "WITNESS_NONE"

        rec = self.get_entry(witness_ref)
        if rec is None:
            return False, "WITNESS_NOT_FOUND_IN_LEDGER"

        # Check sequence index
        if not (0 <= rec["seq"] < len(self.records)):
            return False, "WITNESS_INVALID_SEQ"

        # Re-verify cryptographic hash
        entry_meta = {
            "seq": rec["seq"],
            "prev_hash": rec["prev_hash"],
            "core_tick": rec["core_tick"],
            "world_tick": rec["world_tick"],
            "record_type": rec["record_type"],
            "payload": rec["payload"],
        }
        expected_hash = hashlib.sha256(canonical_json_bytes(entry_meta)).hexdigest()
        if expected_hash != rec["entry_hash"]:
            return False, "WITNESS_HASH_CHAIN_CORRUPT"

        valid_types = required_types or {"VERIFIED_FEEDBACK", "OBSERVED", "AUDIT_SHADOW"}
        if rec["record_type"] not in valid_types:
            return False, f"WITNESS_TYPE_NOT_ALLOWED:{rec['record_type']}"

        return True, "WITNESS_VALID"

    def validate_integrity(self) -> bool:
        """Verify the cryptographic hash-chain across all historical records.
        
        CRITICAL EPISTEMIC CHECK (Item 4):
        - self.seq must strictly equal len(self.records).
        - self.prev_hash must strictly match the computed final record hash
          (or '0'*64 if empty).
        """
        if self.seq != len(self.records):
            return False

        expected_prev = "0" * 64
        for idx, rec in enumerate(self.records):
            if rec["seq"] != idx:
                return False
            if rec["prev_hash"] != expected_prev:
                return False

            # Recompute hash
            entry_meta = {
                "seq": rec["seq"],
                "prev_hash": rec["prev_hash"],
                "core_tick": rec["core_tick"],
                "world_tick": rec["world_tick"],
                "record_type": rec["record_type"],
                "payload": rec["payload"],
            }
            recomputed = hashlib.sha256(canonical_json_bytes(entry_meta)).hexdigest()
            if recomputed != rec["entry_hash"]:
                return False

            expected_prev = rec["entry_hash"]

        if self.prev_hash != expected_prev:
            return False

        return True

    def snapshot(self) -> dict[str, Any]:
        return {
            "records": copy.deepcopy(self.records),
            "prev_hash": self.prev_hash,
            "seq": self.seq,
        }

    @classmethod
    def restore(cls, data: dict[str, Any]) -> SaksinLedger:
        ledger = cls()
        ledger.records = copy.deepcopy(data["records"])
        ledger.prev_hash = str(data["prev_hash"])
        ledger.seq = int(data["seq"])
        if not ledger.validate_integrity():
            raise ValueError("Restored ledger failed cryptographic integrity validation!")
        return ledger
