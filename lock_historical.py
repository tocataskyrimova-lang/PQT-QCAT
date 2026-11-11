"""Lock and verify all historical files before starting LIFE-002."""
import hashlib
import json
import pathlib
import sys

def build_lock():
    root = pathlib.Path(__file__).resolve().parent.parent
    dirs_to_lock = {
        'PQT_QCat_LIFE_001': root / 'PQT_QCat_LIFE_001',
        'independent_nn_audit_20261111': root / 'independent_nn_audit_20261111',
        'PQT_QCat_Neural_Network': root / 'PQT_QCat_Neural_Network'
    }

    lock_data = {'algorithm': 'sha256', 'directories': {}, 'root_files': {}}

    for name, d in dirs_to_lock.items():
        file_hashes = {}
        for p in sorted(d.rglob('*')):
            if p.is_file() and not ({'.git', '__pycache__'} & set(p.parts)):
                rel = p.relative_to(d).as_posix()
                file_hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
        lock_data['directories'][name] = file_hashes
        print(f"{name}: locked {len(file_hashes)} files")

    root_files = [
        'DELIVERY_README_RU.md', 'PQT_QCAT_AUDIT_2026-11-10.md', 
        'ADDENDUM_v1_1_REVIEW_2026-11-11.md', 'package_delivery.py', 
        'verify_qcat.py', 'addendum_m1_check.py', 'PQT_QCAT2.pdf'
    ]
    for rf in root_files:
        p = root / rf
        if p.exists():
            lock_data['root_files'][rf] = hashlib.sha256(p.read_bytes()).hexdigest()
    print(f"Root files locked: {len(lock_data['root_files'])}")

    out_file = pathlib.Path(__file__).resolve().parent / 'prereg' / 'HISTORICAL_LOCK.json'
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(lock_data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f"Saved: {out_file}")

def verify_lock():
    root = pathlib.Path(__file__).resolve().parent.parent
    lock_file = pathlib.Path(__file__).resolve().parent / 'prereg' / 'HISTORICAL_LOCK.json'
    if not lock_file.exists():
        print("Lock file not found!")
        sys.exit(1)
    lock_data = json.loads(lock_file.read_text(encoding='utf-8'))
    
    mismatches = []
    for dir_name, files in lock_data['directories'].items():
        base = root / dir_name
        for rel, expected in files.items():
            fp = base / rel
            if not fp.exists():
                mismatches.append(f"MISSING: {dir_name}/{rel}")
            else:
                h = hashlib.sha256(fp.read_bytes()).hexdigest()
                if h != expected:
                    mismatches.append(f"CHANGED: {dir_name}/{rel}")

    for rf, expected in lock_data['root_files'].items():
        fp = root / rf
        if not fp.exists():
            mismatches.append(f"MISSING root file: {rf}")
        else:
            h = hashlib.sha256(fp.read_bytes()).hexdigest()
            if h != expected:
                mismatches.append(f"CHANGED root file: {rf}")

    if mismatches:
        print(f"FAILED: {len(mismatches)} mismatches detected!")
        for m in mismatches[:10]:
            print("  ", m)
        sys.exit(1)
    else:
        print("ALL HISTORICAL FILES VERIFIED 100% UNCHANGED!")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--verify':
        verify_lock()
    else:
        build_lock()
        verify_lock()
