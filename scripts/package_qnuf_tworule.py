"""Archive the C0 results, locked source, and unchanged confirmation input."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil

from package_sidecar_b2a import REPO, archive_folder, git, read, sha, verify_archive


REPORT = 'LIFE_QNUF_TWORULE_C0_RU.md'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('archive',type=Path)
    args = parser.parse_args()
    root, archive = args.run.resolve(), args.archive.resolve()
    if archive.is_relative_to(root) or archive.exists() or git('status','--porcelain'):
        raise ValueError('Use a clean C0 branch and a fresh archive path outside the run')
    branch = git('branch','--show-current')
    assert branch == 'exp/life-qnuf-tworule-001'
    plan, source, result = (read(root/name) for name in ('PLAN.json','SOURCE.json','ANALYSIS.json'))
    baseline = root/'CONFIRM_ORIGINAL.zip'
    shutil.copy2(plan['base_archive'],baseline)
    assert sha(baseline) == plan['base_archive_sha256']
    frozen = root/'FROZEN_CODE.zip'
    git('archive','--format=zip','-o',str(frozen),source['git'])
    verify_archive(frozen,{name.replace('\\','/'):value for name,value in source['files'].items()})
    bundle = root/'TWORULE_C0_SOURCE.bundle'
    git('bundle','create',str(bundle),branch)
    git('bundle','verify',str(bundle))
    shutil.copy2(REPO/'protocol'/REPORT,root/'PROTOCOL_RU.md')
    shutil.copy2(REPO/'reports'/REPORT,root)
    delivery = {'branch':branch,'delivery_commit':git('rev-parse','HEAD'),'frozen_executable_sha':source['git'],
                'baseline_delivery_sha':plan['base_delivery_git'],'baseline_executable_sha':plan['base_executable_git'],
                'baseline_archive_sha256':plan['base_archive_sha256'],'baseline_source_zip_sha256':plan['base_source_zip_sha256'],
                'baseline_manifest_sha256':plan['base_manifest_sha256'],'baseline_files_verified':plan['base_files_verified'],
                'plan_sha256':sha(root/'PLAN.json'),'matplotlib':importlib.metadata.version('matplotlib'),
                **{k:result[k] for k in ('verdicts','gate','lives','training_events','exact_resumes','missing')}}
    (root/'DELIVERY.json').write_text(json.dumps(delivery,indent=2)+'\n',encoding='utf-8')
    manifest = {p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
    manifest_path = root/'ARTIFACT_MANIFEST.sha256'
    with manifest_path.open('x',encoding='utf-8',newline='\n') as stream:
        stream.writelines(f'{checksum}  {name}\n' for name,checksum in manifest.items())
    archive_folder(archive,root,{**manifest,manifest_path.name:sha(manifest_path)})
    checksum = sha(archive)
    with archive.with_suffix('.zip.sha256').open('x',encoding='ascii') as stream:
        stream.write(f'{checksum}  {archive.name}\n')
    print(f'{len(manifest)} evidence files + manifest verified in {archive}\nSHA-256 {checksum}')


if __name__ == '__main__':
    main()
