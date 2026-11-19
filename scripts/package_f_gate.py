"""Ship the F0 audit trail and verify it against the frozen source and inputs."""
import argparse
import json
from pathlib import Path
import shutil
from xml.etree.ElementTree import parse
from zipfile import ZipFile

from package_sidecar_b2a import REPO, archive_folder, git, read, sha, verify_archive


BRANCH = 'exp/life-qnuf-f-gate-001'
REPORT = 'F0_REPORT_RU.md'


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('run', type=Path)
    cli.add_argument('archive', type=Path)
    args = cli.parse_args()
    root, archive = args.run.resolve(), args.archive.resolve()
    checksum_path = archive.with_suffix('.zip.sha256')
    if git('status', '--porcelain') or git('branch', '--show-current') != BRANCH:
        raise ValueError('Package from the clean F0 delivery branch')
    if archive.exists() or checksum_path.exists() or archive.is_relative_to(root):
        raise ValueError('Choose a new ZIP outside the run folder')

    plan, source, summary = (read(root / name) for name in ('PLAN.json', 'SOURCE.json', 'ANALYSIS.json'))
    suite = parse(root / 'TESTS.xml').getroot().find('testsuite')
    if suite is None or any(suite.get(attr) != '0' for attr in ('errors', 'failures', 'skipped')):
        raise ValueError('Regression tests did not all pass')
    if summary['snapshots'] != 9 or summary['cases'] != 54:
        raise ValueError('Incomplete F0 development panel')
    original = root / 'D4_ORIGINAL.zip'
    shutil.copyfile(Path(plan['base_archive']), original)
    if sha(original) != plan['base_archive_sha256']:
        raise ValueError('Original D4 archive differs from frozen input')
    frozen = root / 'FROZEN_CODE.zip'
    git('archive', '--format=zip', '-o', str(frozen), source['git'])
    verify_archive(frozen, {name.replace('\\', '/'): value for name, value in source['files'].items()})
    with ZipFile(frozen) as packed:
        protocol = packed.read('protocol/LIFE_QNUF_F_GATE_F0_METHOD.md')
        if protocol != (root / 'LIFE_QNUF_F_GATE_F0_METHOD.md').read_bytes():
            raise ValueError('Protocol differs from frozen source')
    bundle = root / 'F0_SOURCE.bundle'
    git('bundle', 'create', str(bundle), BRANCH)
    git('bundle', 'verify', str(bundle))
    shutil.copyfile(REPO / 'reports' / REPORT, root / REPORT)
    delivery = {'branch': BRANCH, 'base_sha': plan['base'], 'base_executable_sha': plan['base_executable'],
                'base_archive_sha256': sha(original), 'frozen_executable_sha': source['git'],
                'delivery_commit': git('rev-parse', 'HEAD'), 'frozen_code_sha256': sha(frozen),
                'engineering': 'F0_VALID', 'regression_tests_passed': int(suite.get('tests')),
                'independent_blocks': False, 'F1_executed': False,
                **{k: summary[k] for k in ('verdict', 'F1_recommended', 'snapshots', 'cases', 'decision_counts',
                                           'A_outside_violations', 'selected_safe_new_B', 'offline_safe_new_B_cases')}}
    (root / 'DELIVERY.json').write_text(json.dumps(delivery, indent=2) + '\n', encoding='utf-8')
    paths = sorted(p for p in root.rglob('*') if p.is_file())
    manifest = {p.relative_to(root).as_posix(): sha(p) for p in paths}
    manifest_path = root / 'ARTIFACT_MANIFEST.sha256'
    with manifest_path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.writelines(f'{h}  {name}\n' for name, h in manifest.items())
    archive_folder(archive, root, {**manifest, manifest_path.name: sha(manifest_path)})
    checksum = sha(archive)
    with checksum_path.open('x', encoding='ascii') as stream:
        stream.write(f'{checksum}  {archive.name}\n')
    print(f'{len(manifest)} evidence files plus manifest verified in {archive}\nSHA-256 {checksum}')


if __name__ == '__main__':
    main()
