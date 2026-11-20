"""Plot and package a completed F0B run with its original evidence bundle."""
import argparse
import csv
import json
from pathlib import Path
import shutil
from zipfile import ZipFile

from package_sidecar_b2a import REPO, archive_folder, git, read, sha, verify_archive


REPORT = 'F0B_COVERAGE_REPORT_RU.md'
BRANCH = 'exp/life-qnuf-f0b-coverage-001'


def plot(run):
    from matplotlib.figure import Figure

    with (run/'TRADEOFF.csv').open(newline='',encoding='utf-8') as stream:
        records = list(csv.DictReader(stream))
    x = [float(row['bound']) for row in records]
    fig = Figure(figsize=(9.4,6.4),layout='constrained')
    top,bottom = fig.subplots(2,1,sharex=True)
    top.plot(x,[int(r['available_new_B']) for r in records],'o-',color='#003c7a',label='Есть допустимый кандидат с новым правильным B')
    top.plot(x,[int(r['new_B']) for r in records],'s--',color='#906000',label='Выбранная запись исправляет B')
    top.set(ylabel='Случаи из 54',title='Исправление B и ущерб A при полном банке')
    top.legend(loc='upper left',fontsize=9)
    bottom.plot(x,[float(r['A_max']) for r in records],'o-',color='#a03536',label='Наибольшая ΔNLL A среди выбранных записей')
    bottom.plot(x,[float(r['A_mean']) for r in records],'s:',color='#226c52',label='Средняя ΔNLL A, включая отказы')
    bottom.set(xlabel='Допустимый рост NLL отдельного запроса A (порог)',ylabel='Фактическая ΔNLL A')
    bottom.set_xscale('symlog',linthresh=.001)
    bottom.legend(loc='upper left',fontsize=9)
    for ax in (top,bottom):
        ax.axvline(.01,linewidth=1,color='0.5',linestyle=':')
        ax.grid(alpha=.2)
    fig.savefig(run/'TRADEOFF.png',dpi=160)


def package(run,original,archive):
    if git('status','--porcelain') or git('branch','--show-current')!=BRANCH:
        raise ValueError('Package from the clean F0B delivery branch')
    checksum = archive.with_suffix('.zip.sha256')
    if archive.exists() or checksum.exists() or archive.is_relative_to(run):
        raise ValueError('Choose a new ZIP outside the run folder')
    plan,summary = (read(run/name) for name in ('PLAN.json','SUMMARY.json'))
    if (summary['cases'],summary['restores_exact'],summary['fresh_reads'])!=(54,324,189):
        raise ValueError('Missing development cases or snapshot control')
    baseline = run/'F0_ORIGINAL.zip'
    shutil.copyfile(original,baseline)
    assert sha(baseline)==plan['base_archive_sha256']
    frozen = run/'FROZEN_CODE.zip'
    git('archive','--format=zip','-o',str(frozen),plan['source_commit'])
    verify_archive(frozen,{'scripts/f0b_coverage.py':plan['source_sha256'],
                           'protocol/LIFE_QNUF_F0B_COVERAGE_METHOD_RU.md':read(run/'LOCK.json')['LIFE_QNUF_F0B_COVERAGE_METHOD_RU.md']})
    bundle = run/'F0B_SOURCE.bundle'
    git('bundle','create',str(bundle),BRANCH)
    git('bundle','verify',str(bundle))
    shutil.copyfile(REPO/'reports'/REPORT,run/REPORT)
    delivery = {'base_sha':plan['base_sha'],'base_archive_sha256':plan['base_archive_sha256'],
                'source_commit':plan['source_commit'],'delivery_commit':git('rev-parse','HEAD'),
                'frozen_code_sha256':sha(frozen),'engineering':'F0B_VALID','branch':BRANCH,
                **{k:summary[k] for k in ('verdict','F1_executed','C1_executed','local_L','joint_J','G3','G6')}}
    (run/'DELIVERY.json').write_text(json.dumps(delivery,indent=2)+'\n',encoding='utf-8')
    hashes = {p.relative_to(run).as_posix():sha(p) for p in sorted(run.rglob('*')) if p.is_file()}
    manifest = run/'ARTIFACT_MANIFEST.sha256'
    with manifest.open('x',encoding='utf-8',newline='\n') as stream:
        stream.writelines(f'{value}  {name}\n' for name,value in hashes.items())
    archive_folder(archive,run,{**hashes,manifest.name:sha(manifest)})
    with ZipFile(archive) as z:
        assert set(z.namelist())=={*hashes,manifest.name}
    digest = sha(archive)
    checksum.write_text(f'{digest}  {archive.name}\n',encoding='ascii')
    print(f'{len(hashes)} files and manifest verified\nSHA-256 {digest}')


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('run',type=Path)
    cli.add_argument('--archive',type=Path)
    cli.add_argument('--original',type=Path)
    args = cli.parse_args()
    if args.archive:
        package(args.run.resolve(),args.original.resolve(),args.archive.resolve())
    else:
        plot(args.run.resolve())


if __name__=='__main__':
    main()
