"""Package verified context binding evidence with the original C0 input."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil

from package_sidecar_b2a import REPO,archive_folder,git,read,sha,verify_archive


NAME = 'LIFE_QNUF_CONTEXT_BIND_001_RU.md'


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('run',type=Path)
    cli.add_argument('archive',type=Path)
    args = cli.parse_args()
    root,archive = args.run.resolve(),args.archive.resolve()
    if archive.exists() or archive.is_relative_to(root) or git('status','--porcelain'):
        raise ValueError('Use a clean branch and a new archive output outside the run')
    branch = git('branch','--show-current')
    assert branch=='exp/life-qnuf-context-bind-001'
    plan, source, analysis = (read(root/name) for name in ('PLAN.json','SOURCE.json','ANALYSIS.json'))
    baseline = root/'C0_ORIGINAL.zip'
    shutil.copy2(plan['base_archive'],baseline)
    assert sha(baseline)==plan['base_zip_sha256']
    frozen = root/'FROZEN_CODE.zip'
    git('archive','--format=zip','-o',str(frozen),source['git'])
    verify_archive(frozen,{k.replace('\\','/'):v for k,v in source['files'].items()})
    bundle = root/'CONTEXT_BIND_SOURCE.bundle'
    git('bundle','create',str(bundle),branch)
    git('bundle','verify',str(bundle))
    shutil.copy2(REPO/'reports'/NAME,root)
    shutil.copy2(REPO/'protocol'/NAME,root/'PROTOCOL_RU.md')
    delivery = {'branch':branch,'delivery_commit':git('rev-parse','HEAD'),'executable_sha':source['git'],
                'base_delivery':plan['base'],'base_executable':plan['base_executable'],
                'base_source_zip_sha256':plan['base_source_zip_sha256'],'base_zip_sha256':plan['base_zip_sha256'],
                'dependencies':{name:importlib.metadata.version(name)
                                for name in ('numpy','torch','scipy','scikit-learn','pytest','matplotlib')},
                **{k:analysis[k] for k in ('verdict','mechanisms','solo_verified_actions','exact_resumes',
                                            'diagnostic_states','decoder_controls_exact_restore')}}
    (root/'DELIVERY.json').write_text(json.dumps(delivery,indent=2)+'\n',encoding='utf-8')
    manifest = {f.relative_to(root).as_posix():sha(f) for f in sorted(root.rglob('*')) if f.is_file()}
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
