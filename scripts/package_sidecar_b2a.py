"""Archive B2A inputs, evidence and the exact source history, then verify every member."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile


REPO = Path(__file__).resolve().parents[1]
REPORT = 'LIFE_ACCESS_STATE_001_STAGE_B2A_RU.md'
PROTOCOL = 'STAGE_B2A_ENVIRONMENT_RU.md'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO, text=True).strip()


def verify_archive(path, manifest):
    with ZipFile(path) as archive:
        for name, expected in manifest.items():
            with archive.open(name) as member:
                if hashlib.file_digest(member, 'sha256').hexdigest() != expected:
                    raise ValueError(f'Archive hash mismatch: {name}')


def archive_folder(destination, root, manifest):
    with ZipFile(destination, 'x', compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for name in manifest:
            archive.write(root / name, name)
    verify_archive(destination, manifest)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    root, archive = args.run.resolve(), args.archive.resolve()
    plan = read(root / 'PLAN.json')
    source = read(root / 'SOURCE.json')
    summary = read(root / 'ANALYSIS.json')
    branch = git('branch', '--show-current')
    commit = git('rev-parse', 'HEAD')
    if git('status', '--porcelain') or branch != 'exp/life-functional-sidecar-b2a':
        raise ValueError('Package from the clean B2A delivery branch')
    if archive.exists() or archive.is_relative_to(root):
        raise ValueError('Choose a new ZIP outside the run folder')
    for phase in ('factorial', 'controls', 'replay', 'resume'):
        if read(root / f'STATUS_{phase}.json')['phase'] != 'COMPLETED':
            raise ValueError(f'Incomplete phase: {phase}')
    for name, key, manifest_name in [('INPUT_QNUF.zip', 'historical_source', 'sha256_manifest.json'),
                                     ('INPUT_B1.zip', 'b1_source', 'SHA256.json')]:
        original = Path(plan[key])
        original_manifest = read(original / manifest_name)
        selected = plan['inputs'][str(original)]
        if any(original_manifest[name] != digest for name, digest in selected.items()):
            raise ValueError(f'Historical manifest differs from the frozen selection: {manifest_name}')
        manifest = {**selected, manifest_name: sha(original / manifest_name)}
        archive_folder(root / name, original, manifest)
    frozen = root / 'FROZEN_CODE.zip'
    git('archive', '--format=zip', '-o', str(frozen), source['commit'])
    verify_archive(frozen, {name.replace('\\', '/'): value for name, value in source['files'].items()})
    bundle = root / 'B2A_SOURCE.bundle'
    git('bundle', 'create', str(bundle), branch)
    git('bundle', 'verify', str(bundle))
    shutil.copy2(REPO / 'reports' / REPORT, root / REPORT)
    shutil.copy2(REPO / 'protocol' / PROTOCOL, root / PROTOCOL)
    delivery = {'base_sha': plan['base'], 'frozen_executable_sha': source['commit'],
                'delivery_commit': commit, 'branch': branch,
                'analysis_dependencies': {p: importlib.metadata.version(p)
                                          for p in ('scikit-learn', 'matplotlib')},
                **{k: summary[k] for k in ('engineering', 'scientific', 'table_gate', 'reward_gate',
                                          'selectivity_gate', 'main_learning_gate', 'trajectories_replayed_exact',
                                          'action_records_verified', 'pending_resume_exact')}}
    (root / 'DELIVERY.json').write_text(json.dumps(delivery, indent=2) + '\n', encoding='utf-8')
    paths = sorted(p for p in root.rglob('*') if p.is_file())
    manifest = {p.relative_to(root).as_posix(): sha(p) for p in paths}
    manifest_path = root / 'ARTIFACT_MANIFEST.sha256'
    with manifest_path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.writelines(f'{h}  {name}\n' for name, h in manifest.items())
    archive_folder(archive, root, {**manifest, manifest_path.name: sha(manifest_path)})
    checksum = sha(archive)
    with archive.with_suffix('.zip.sha256').open('x', encoding='ascii') as stream:
        stream.write(f'{checksum}  {archive.name}\n')
    print(f'{len(manifest)} evidence files + manifest verified in {archive}\nSHA-256 {checksum}')


if __name__ == '__main__':
    main()
