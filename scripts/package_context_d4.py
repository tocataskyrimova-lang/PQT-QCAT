"""Package the verified D4 result, exact source and untouched handoff/base."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil

from package_sidecar_b2a import REPO,archive_folder,git,read,sha,verify_archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('archive',type=Path)
    parser.add_argument('--handoff',type=Path,required=True)
    args = parser.parse_args()
    root,archive = args.run.resolve(),args.archive.resolve()
    if archive.is_relative_to(root) or archive.exists() or git('status','--porcelain'):
        raise ValueError('Use a clean branch and a new destination outside the run')
    branch = git('branch','--show-current')
    assert branch=='exp/life-qnuf-context-bind-d4'
    plan,source,result = (read(root/f) for f in ('PLAN.json','SOURCE.json','ANALYSIS.json'))
    old = root/'D0_D3_ORIGINAL.zip'
    handoff = root/'D4_HANDOFF_ORIGINAL.zip'
    shutil.copy2(plan['base_archive'],old)
    shutil.copy2(args.handoff,handoff)
    assert sha(old)==plan['base_archive_sha256'] and sha(handoff)==plan['handoff_sha256']
    frozen = root/'FROZEN_CODE.zip'
    git('archive','--format=zip','-o',str(frozen),source['git'])
    verify_archive(frozen,{k.replace('\\','/'):v for k,v in source['files'].items()})
    bundle = root/'D4_SOURCE.bundle'
    git('bundle','create',str(bundle),branch)
    git('bundle','verify',str(bundle))
    shutil.copy2(REPO/'reports'/'D4_REPORT_RU.md',root)
    delivery = {'branch':branch,'delivery_commit':git('rev-parse','HEAD'),'executable_sha':source['git'],
                'base_delivery':plan['base'],'base_executable':plan['base_executable'],
                'base_archive_sha256':plan['base_archive_sha256'],'base_source_zip_sha256':plan['base_source_zip_sha256'],
                'base_manifest_sha256':plan['base_manifest_sha256'],'handoff_sha256':plan['handoff_sha256'],
                'matplotlib':importlib.metadata.version('matplotlib'),
                **{key:result[key] for key in ('verdicts','snapshot_count','single_writes','counts','panels_restored',
                                             'max_pair_relative_norm_error')}}
    (root/'DELIVERY.json').write_text(json.dumps(delivery,indent=2)+'\n',encoding='utf-8')
    manifest = {p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
    file = root/'ARTIFACT_MANIFEST.sha256'
    with file.open('x',encoding='utf-8',newline='\n') as stream:
        stream.writelines(f'{value}  {name}\n' for name,value in manifest.items())
    archive_folder(archive,root,{**manifest,file.name:sha(file)})
    checksum = sha(archive)
    with archive.with_suffix('.zip.sha256').open('x',encoding='ascii') as stream:
        stream.write(f'{checksum}  {archive.name}\n')
    print(f'{len(manifest)} evidence files + manifest verified in {archive}\nSHA-256 {checksum}')


if __name__=='__main__':
    main()
