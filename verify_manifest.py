"""CLI verification script for LIFE-002 SHA-256 Cryptographic Manifest."""
import sys
from pathlib import Path

# Add directory to sys.path
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

from pqt_qcat.evaluator.manifest import verify_sha256_manifest


def main():
    print("=" * 80)
    print("Verifying LIFE-002 Cryptographic Manifest (prereg/MANIFEST_SHA256_LIFE002.json)...")
    print("=" * 80)

    success, verified_count, mismatches = verify_sha256_manifest(root_dir=root)

    print(f"Total files checked: {verified_count + len(mismatches)}")
    print(f"Verified matching files: {verified_count}")
    print(f"Mismatches / Missing: {len(mismatches)}")

    if not success:
        print("\nMANIFEST VERIFICATION FAILED:")
        for m in mismatches:
            print(f"  [X] {m}")
        sys.exit(1)
    else:
        print("\nALL FILES VERIFIED 100% UNCHANGED AND CRYPTOGRAPHICALLY VALID!")
        sys.exit(0)


if __name__ == "__main__":
    main()
