import hashlib
from pathlib import Path

root = Path("C:/AGI/PQT_QCAT_AUDIT/PQT_QCat_LIFE_ACCESS_001")
files_to_hash = []

# Core campaign reports & manifests
for fname in [
    "STATUS.json",
    "RUN_INDEX.json",
    "FINAL_REPORT_RU.md",
    "NEXT_DECISION.md",
    "IMPLEMENTATION_CHANGELOG.md",
    "SEED_INCIDENT.md",
    "ADDENDUM_RETRACTION_20261118.md",
    "R0_ACCEPTANCE.json",
]:
    p = root / fname
    if p.exists():
        files_to_hash.append(p)

# Source modules created / updated in campaign
for mod in [
    "pqt_qcat/core/security.py",
    "pqt_qcat/core/six_echelon.py",
    "pqt_qcat/memory/exp_life_access_001.py",
    "pqt_qcat/memory/exp_k_observability_v2.py",
    "pqt_qcat/memory/exp_stateful_online_001.py",
    "pqt_qcat/memory/exp_long_life_001.py",
    "pqt_qcat/tests/test_r0_acceptance.py",
    "pqt_qcat/tests/test_r3_r4_smoke.py",
]:
    p = root / mod
    if p.exists():
        files_to_hash.append(p)

# Results from R1, R2, R3, R4
for pattern in [
    "results/exploratory/LIFE_ACCESS_001/RUN_20261118_040220/*",
    "results/exploratory/K_OBSERVABILITY_002/RUN_20261118_043850/*",
    "results/exploratory/STATEFUL_ONLINE_001/RUN_20261118_051107/*",
    "results/exploratory/LONG_LIFE_001/RUN_20261118_064048/*",
]:
    for p in root.glob(pattern):
        if p.is_file() and not p.name.endswith(".tmp"):
            files_to_hash.append(p)

manifest_lines = []
for p in sorted(set(files_to_hash)):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    rel_path = p.relative_to(root).as_posix()
    manifest_lines.append(f"{h.hexdigest()}  {rel_path}")

out_path = root / "ARTIFACT_MANIFEST.sha256"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(manifest_lines) + "\n")
print(f"Generated ARTIFACT_MANIFEST.sha256 with {len(manifest_lines)} entries.")
