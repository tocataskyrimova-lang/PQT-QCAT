"""SHA-256 Cryptographic Manifest Generator and Verifier for LIFE-002 M5 Preparation.

Computes and verifies SHA-256 checksums across all code, configurations,
benchmarks, generators, and protocol definitions.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json


EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "results",
    "raw_logs",
    "legacy_life_001",
}

EXCLUDE_EXTS = {
    ".pyc",
    ".tmp",
    ".log",
    ".jsonl",
}

TEXT_EXTS = {
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".md",
    ".txt",
    ".sh",
    ".gitignore",
    ".gitattributes",
}


def is_text_file(path: Path) -> bool:
    """Check if file is a known text file or contains no null bytes."""
    p = Path(path)
    if p.suffix.lower() in TEXT_EXTS or p.name in {".gitignore", ".gitattributes"}:
        return True
    try:
        with open(p, "rb") as f:
            chunk = f.read(1024)
            return b"\x00" not in chunk
    except Exception:
        return False


def compute_file_sha256(path: Path) -> str:
    """Compute canonical SHA-256 hash of a file.
    
    For text files, normalizes CRLF to LF to guarantee consistent cross-platform
    hashes regardless of checkout environment (Ubuntu vs Windows).
    For binary files, hashes raw bytes directly.
    """
    p = Path(path)
    if is_text_file(p):
        with open(p, "rb") as f:
            content = f.read()
        canonical_content = content.replace(b"\r\n", b"\n")
        return hashlib.sha256(canonical_content).hexdigest()
    else:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()


def generate_sha256_manifest(root_dir: Path | str | None = None) -> dict:
    """Scan all codebase, configuration, and protocol files and generate SHA-256 manifest."""
    root = Path(root_dir or Path(__file__).resolve().parent.parent.parent)
    manifest = {}

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(root)
        parts = rel_path.parts

        # Check directory exclusions
        if any(part in EXCLUDE_DIRS for part in parts):
            continue
        if path.suffix in EXCLUDE_EXTS:
            continue
        # Skip manifests themselves to avoid circular dependencies
        if "MANIFEST_SHA256" in path.name:
            continue

        sha256_hash = compute_file_sha256(path)
        manifest[str(rel_path).replace("\\", "/")] = sha256_hash

    # Save to JSON and TXT with explicit LF newlines
    json_path = root / "prereg" / "MANIFEST_SHA256_LIFE002.json"
    txt_path = root / "prereg" / "MANIFEST_SHA256_LIFE002.txt"

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    with open(txt_path, "w", encoding="utf-8", newline="\n") as f:
        for file_path, h in manifest.items():
            f.write(f"{h}  {file_path}\n")

    return {
        "manifest_path_json": str(json_path),
        "manifest_path_txt": str(txt_path),
        "total_files_hashed": len(manifest),
        "manifest": manifest,
    }


def verify_sha256_manifest(manifest_path: Path | str | None = None, root_dir: Path | str | None = None) -> tuple[bool, int, list[str]]:
    """Verify that all files in the manifest match their recorded SHA-256 hash."""
    root = Path(root_dir or Path(__file__).resolve().parent.parent.parent)
    mpath = Path(manifest_path or (root / "prereg" / "MANIFEST_SHA256_LIFE002.json"))

    if not mpath.exists():
        raise FileNotFoundError(f"Manifest not found: {mpath}")

    with open(mpath, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    mismatches = []
    verified_count = 0

    for rel_path_str, expected_hash in manifest.items():
        file_path = root / rel_path_str
        if not file_path.exists():
            mismatches.append(f"MISSING: {rel_path_str}")
            continue
        actual_hash = compute_file_sha256(file_path)
        if actual_hash != expected_hash:
            mismatches.append(f"MISMATCH: {rel_path_str} (expected {expected_hash}, got {actual_hash})")
        else:
            verified_count += 1

    success = (len(mismatches) == 0)
    return success, verified_count, mismatches
