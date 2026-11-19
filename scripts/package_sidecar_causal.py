"""Preserve B2A provenance, causal-analysis artifacts and exact source history."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil

from package_sidecar_b2a import REPO, archive_folder, git, read, sha, verify_archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    root, archive = args.run.resolve(), args.archive.resolve()
    if archive.is_relative_to(root) or archive.exists() or git('status', '--porcelain'):
        raise ValueError('Use a clean branch and a new archive path outside the run folder')
    plan = read(root / 'PLAN.json')
    source = read(root / 'SOURCE.json')
    analysis = read(root / 'ANALYSIS.json')
    branch = git('branch', '--show-current')
    assert branch == 'exp/life-b2a-causal-001'
    assert analysis['engineering'] == 'CAUSAL_VALID'
    for stage in ('primary', 'controls', 'one-step', 'resume'):
        assert read(root / f'STATUS_{stage}.json')['phase'] == 'COMPLETED'

    original = root / 'B2A_ORIGINAL.zip'
    shutil.copy2(plan['b2a_archive'], original)
    assert sha(original) == plan['b2a_archive_sha256']
    executable = root / 'FROZEN_CODE.zip'
    git('archive', '--format=zip', '-o', str(executable), source['commit'])
    verify_archive(executable, {name.replace('\\', '/'): value for name, value in source['files'].items()})
    bundle = root / 'CAUSAL_SOURCE.bundle'
    git('bundle', 'create', str(bundle), branch)
    git('bundle', 'verify', str(bundle))
    shutil.copy2(REPO / 'reports' / 'LIFE_B2A_CAUSAL_001_RU.md', root)
    shutil.copy2(REPO / 'protocol' / 'LIFE_B2A_CAUSAL_001_RU.md', root / 'PROTOCOL_RU.md')
    provenance = {'baseline': plan['base'], 'baseline_zip_sha256': plan['b2a_archive_sha256'],
                  'baseline_manifest_sha256': plan['b2a_manifest_sha256'], 'baseline_files_checked': plan['b2a_artifacts_verified'],
                  'frozen_executable': source['commit'], 'delivery_commit': git('rev-parse', 'HEAD'),
                  'branch': branch, 'matplotlib': importlib.metadata.version('matplotlib'),
                  **{k: analysis[k] for k in ('engineering', 'lives_verified', 'events_verified', 'historical_endpoints_exact',
                                             'resumes_exact', 'one_step_anchors', 'summary_of_accuracy_and_return_tests')}}
    (root / 'DELIVERY.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    manifest = {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
    manifest_path = root / 'ARTIFACT_MANIFEST.sha256'
    with manifest_path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.writelines(f'{checksum}  {name}\n' for name, checksum in manifest.items())
    archive_folder(archive, root, {**manifest, manifest_path.name: sha(manifest_path)})
    checksum = sha(archive)
    with archive.with_suffix('.zip.sha256').open('x', encoding='ascii') as stream:
        stream.write(f'{checksum}  {archive.name}\n')
    print(f'{len(manifest)} evidence files + manifest verified in {archive}\nSHA-256 {checksum}')


if __name__ == '__main__':
    main()
