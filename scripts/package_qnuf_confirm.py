"""Archive the confirmation, historical inputs, and exact Git history."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil

from package_sidecar_b2a import REPO, archive_folder, git, read, sha, verify_archive


NAME = 'LIFE_QNUF_CONFIRM_001_RU.md'
FOLLOW_ON = 'LIFE_QNUF_TWO_RULE_PROPOSAL_RU.md'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    root, archive = args.run.resolve(), args.archive.resolve()
    if archive.exists() or archive.is_relative_to(root) or git('status', '--porcelain'):
        raise ValueError('Package from a clean branch with a new ZIP destination')
    branch = git('branch', '--show-current')
    assert branch == 'exp/life-qnuf-confirm-001'
    plan, source, result = (read(root / name) for name in ('PLAN.json', 'SOURCE.json', 'ANALYSIS.json'))
    old = root / 'B2A_CAUSAL_ORIGINAL.zip'
    shutil.copy2(plan['baseline_archive'], old)
    assert sha(old) == plan['baseline_archive_sha256']
    frozen = root / 'FROZEN_CODE.zip'
    git('archive', '--format=zip', '-o', str(frozen), source['git'])
    verify_archive(frozen, {name.replace('\\', '/'): value for name, value in source['files'].items()})
    bundle = root / 'CONFIRM_SOURCE.bundle'
    git('bundle', 'create', str(bundle), branch)
    git('bundle', 'verify', str(bundle))
    shutil.copy2(REPO / 'reports' / NAME, root)
    shutil.copy2(REPO / 'protocol' / NAME, root / 'PROTOCOL_RU.md')
    if result['accuracy_verdict'] == 'CONFIRMED':
        shutil.copy2(REPO / 'protocol' / FOLLOW_ON, root)
    delivery = {'branch': branch, 'baseline_sha': plan['base'], 'baseline_archive_sha256': plan['baseline_archive_sha256'],
                'baseline_manifest_sha256': plan['baseline_manifest_sha256'], 'baseline_files_checked': plan['baseline_files_checked'],
                'frozen_executable_sha': source['git'], 'delivery_commit': git('rev-parse', 'HEAD'),
                'plan_file_sha256': sha(root / 'PLAN.json'), 'matplotlib': importlib.metadata.version('matplotlib'),
                **{field: result[field] for field in ('engineering', 'accuracy_verdict', 'reward_verdict', 'primary',
                                                       'lives_verified', 'actions_verified', 'resumes_exact', 'missing')}}
    (root / 'DELIVERY.json').write_text(json.dumps(delivery, indent=2) + '\n', encoding='utf-8')
    manifest = {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
    manifest_path = root / 'ARTIFACT_MANIFEST.sha256'
    with manifest_path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.writelines(f'{value}  {name}\n' for name, value in manifest.items())
    archive_folder(archive, root, {**manifest, manifest_path.name: sha(manifest_path)})
    checksum = sha(archive)
    with archive.with_suffix('.zip.sha256').open('x', encoding='ascii') as stream:
        stream.write(f'{checksum}  {archive.name}\n')
    print(f'{len(manifest)} evidence files + manifest verified in {archive}\nSHA-256 {checksum}')


if __name__ == '__main__':
    main()
