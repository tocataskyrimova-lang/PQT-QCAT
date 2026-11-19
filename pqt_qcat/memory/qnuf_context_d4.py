"""A single-write, norm-matched intervention on archived D1/D2 checkpoints."""
from . import qnuf_context_bind as parent  # pin numerical threads before NumPy

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import importlib.metadata
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zipfile import ZipFile

import numpy as np

from .access_state import exact
from .checkpoint_io import load_state, save_state
from .codex_memory_location import atomic_json, digest, source_hashes
from .functional_sidecar import ActionJournal, read_json, state_hash
from .functional_sidecar_b2a import forecast
from .qnuf import array_hash, rng_hash


BASE = 'b8fc818cdcface65d1410d9197aaaf60773a5e9d'
BASE_EXEC = '0d4644a2c36147e5f110f275641ab8780d7f32dd'
BASE_ARCHIVE = '87bbf0083239e1eb7e7773d1f9e13fabfbb704c8115d1494fce8c337360dce85'
ARMS = {'RAW@raw': ('raw', 'raw'), 'CTX@raw': ('ctx', 'raw'),
        'RAW@ctx': ('raw', 'ctx'), 'CTX@ctx': ('ctx', 'ctx')}
TOL = 1e-12
WORKERS = 6
BRIEF = 'LIFE_QNUF_CONTEXT_BIND_D4_BRIEF.md'
METHOD = 'LIFE_QNUF_CONTEXT_BIND_D4_METHOD.md'


def init(root, reference, archive, handoff):
    repo = parent.c0.causal.b2.b1.bridge.ROOT
    subprocess.run(['git','merge-base','--is-ancestor',BASE,'HEAD'],cwd=repo,check=True)
    subprocess.run(['git','diff','--quiet',BASE_EXEC,BASE,'--','pqt_qcat'],cwd=repo,check=True)
    delivery, source = (read_json(reference / name) for name in ('DELIVERY.json','SOURCE.json'))
    exact([delivery['delivery_commit'],delivery['executable_sha'],source['git']], [BASE,BASE_EXEC,BASE_EXEC])
    assert delivery['verdict']=='CONTEXT_BINDING_DIAGNOSED'
    assert parent.c0.causal.file_sha(archive)==BASE_ARCHIVE
    manifest = reference/'ARTIFACT_MANIFEST.sha256'
    inputs = {name:sha for sha,name in (line.split('  ',1) for line in manifest.read_text(encoding='utf-8').splitlines())}
    for file, expected in inputs.items():
        assert parent.c0.causal.file_sha(reference/file)==expected, file
    with ZipFile(archive) as packed:
        old_code = packed.read('FROZEN_CODE.zip')
    assert old_code == subprocess.check_output(['git','archive','--format=zip',BASE_EXEC],cwd=repo)
    assert old_code == (reference/'FROZEN_CODE.zip').read_bytes()
    with ZipFile(handoff) as packed:
        brief = packed.read(BRIEF)
    prior = read_json(reference/'PLAN.json')
    plan = {'base':BASE, 'base_executable':BASE_EXEC, 'base_archive':str(archive), 'base_archive_sha256':BASE_ARCHIVE,
            'base_source_zip_sha256':hashlib.sha256(old_code).hexdigest(), 'base_inputs':inputs,
            'base_manifest_sha256':parent.c0.causal.file_sha(manifest), 'reference':str(reference),
            'base_plan':prior, 'handoff_sha256':parent.c0.causal.file_sha(handoff),
            'brief_sha256':hashlib.sha256(brief).hexdigest(), 'units':prior['units'],
            'mode':'separate','policy':'SS','write_context':'B','anchor':'A','horizon':64,
            'arms':['NONE',*ARMS], 'relative_tolerance':TOL, 'workers':WORKERS,
            'preflight_unit':prior['units'][0], 'clipping':'strict proposal cap; only for applied float subtraction use 4 ulps of the relevant scale',
            'fallback':'paired direction comparison at min(raw,ctx) from existing arms; no upward extrapolation',
            'scope':'D4 only; nine old development checkpoints; one write per independent clone',
            'bootstrap_seed':prior['bootstrap_seed'],'bootstrap_replicates':prior['bootstrap_replicates'],
            'scale_only_equivalence_nll':0.01}
    exact(plan['units'],[[s,m] for s in parent.c0.SEEDS for m in range(3)])
    root.mkdir(parents=True,exist_ok=False)
    (root/BRIEF).write_bytes(brief)
    shutil.copy2(repo/'protocol'/METHOD, root/METHOD)
    for name in ('DELIVERY.json','SOURCE.json','ARTIFACT_MANIFEST.sha256'):
        shutil.copy2(reference/name,root/('BASE_'+name))
    atomic_json(root/'PLAN.json',plan)
    atomic_json(root/'SOURCE.json',{'git':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
                'files':source_hashes(), 'plan_sha256':digest(plan), 'python':sys.version,
                'dependencies':{name:importlib.metadata.version(name) for name in ('numpy','torch','scipy','pytest','scikit-learn')}})
    atomic_json(root/'LOCK.json',{name:parent.c0.causal.file_sha(root/name) for name in ('PLAN.json','SOURCE.json',BRIEF,METHOD)})


def verify_plan(root):
    locks = read_json(root/'LOCK.json')
    assert all(parent.c0.causal.file_sha(root/name)==value for name,value in locks.items())
    plan,source = (read_json(root/name) for name in ('PLAN.json','SOURCE.json'))
    assert digest(plan)==source['plan_sha256'] and source_hashes()==source['files']
    assert parent.c0.causal.file_sha(root/'BASE_ARTIFACT_MANIFEST.sha256')==plan['base_manifest_sha256']
    return plan


def within_norm(actual, target):
    return abs(actual-target) <= TOL*target


def match_update(k, direction, target, cap, original=False):
    """Return proposal, implemented K, and physical feasibility without reclipping."""
    peak = float(np.max(np.abs(direction)))
    native = float(np.linalg.norm(direction))
    meta = {'target_norm':target if np.isfinite(target) else None, 'native_norm':native if np.isfinite(native) else None,
            'capacity':None, 'factor':None, 'actual_norm':None, 'peak_proposed':None,
            'peak_applied':None, 'applied_roundoff_bound':None}
    if not np.isfinite([peak,native,target]).all():
        return None,None,dict(meta,status='NONFINITE_CANDIDATE')
    if target < 0 or peak > cap:
        raise AssertionError('Input direction was not an admissible clipped candidate')
    if not native and target:
        return None,None,dict(meta,status='INFEASIBLE_MATCH',reason='zero_direction')
    maximum = (cap/peak)*native if peak else 0.
    meta['capacity'] = maximum
    if target > maximum and not within_norm(maximum,target):
        return None,None,dict(meta,status='INFEASIBLE_MATCH',reason='component_cap_on_ray')
    factor = min(target/native,cap/peak) if native else 0.
    proposal = direction.copy() if original else direction*factor
    if not original and float(np.max(np.abs(proposal))) > cap:
        factor = float(np.nextafter(factor,0.))
        proposal = direction*factor
    peak_proposed = float(np.max(np.abs(proposal)))
    assert peak_proposed <= cap
    k_after = k+proposal
    applied = k_after-k
    actual = float(np.linalg.norm(applied))
    roundoff_bound = float(4*np.spacing(max(1.,cap,float(np.max(np.abs(k))),float(np.max(np.abs(k_after))))))
    peak_applied = float(np.max(np.abs(applied)))
    assert peak_applied <= cap+roundoff_bound
    assert not np.any(applied[direction==0.])
    meta.update(factor=1. if original else factor,actual_norm=actual,peak_proposed=peak_proposed,
                peak_applied=peak_applied,applied_roundoff_bound=roundoff_bound)
    if not within_norm(actual,target):
        return None,None,dict(meta,status='INFEASIBLE_MATCH',reason='stored_norm_outside_tolerance')
    return proposal,k_after,dict(meta,status='OK' if target else 'ZERO_RADIUS')


def source_paths(plan, seed, mapping):
    ref = Path(plan['reference'])
    solo = parent.path_for(ref,plan['base_plan'],'A',seed,mapping,'separate','SS')
    diagnostic = ref/'diagnostics'/'A'/f'{seed}_{mapping}'/'separate'/'SS'
    donor = parent.path_for(ref,plan['base_plan'],'A',seed,(mapping+1)%3,'separate','SS')
    return solo, diagnostic, donor


def checked_input(plan, path):
    name = path.relative_to(Path(plan['reference'])).as_posix()
    assert parent.c0.causal.file_sha(path)==plan['base_inputs'][name], name
    return path


def read_panel(agent, anchor, k):
    scratch = copy.deepcopy(agent)
    scratch.core.K[:] = k
    locked = state_hash(scratch.snapshot())
    logits,post_rng,steps = {},{},0
    for ctx in ('A','B'):
        for q in range(1,7):
            tag = f'{ctx}:{q}'
            values,_,rnd,count = parent.neutral_read(scratch,q,parent.c0.vector('separate',ctx,q))
            logits[tag],post_rng[tag] = values,rnd
            steps += count
    assert locked==state_hash(scratch.snapshot())
    scratch.restore(anchor)
    assert state_hash(scratch.snapshot())==state_hash(anchor)
    return {'logits':logits,'rng_post':post_rng,'K_hash':array_hash(k),'substeps':steps,
            'restored_sha256':state_hash(scratch.snapshot())}


def run_block(job):
    root,plan,seed,mapping,preflight = job
    assert [seed,mapping] in plan['units']
    _,rules = parent.c0.block(plan['base_plan']['c0'],seed,mapping)
    solo,diag,donor = source_paths(plan,seed,mapping)
    for directory,names in [(solo,('start.npz','end.npz','PROBES.json')),
                           (diag,('RESULTS.json','DELTAS.npz','pre_feedback.jsonl')), (donor,('end.npz',))]:
        for name in names:
            checked_input(plan,directory/name)
    output = root/('preflight' if preflight else 'blocks')/f'{seed}_{mapping}'
    output.mkdir(parents=True,exist_ok=False)
    saved = load_state(solo/'end.npz')
    agent = parent.agent_for(solo,plan['base_plan'],seed)
    anchor = agent.snapshot()
    anchor_sha = state_hash(anchor)
    assert anchor_sha==state_hash(saved['agent'])
    save_state(output/'START.npz',saved)
    initial = load_state(solo/'start.npz')['agent']['core_snapshot']['K']
    wrong = load_state(donor/'end.npz')['agent']['core_snapshot']['K']
    k = anchor['core_snapshot']['K']
    limit = agent.delta_core_max
    assert limit == plan['base_plan']['c0']['parent']['clip']
    log = ActionJournal(output/'ACTIONS_BEFORE_FEEDBACK.jsonl')
    historical = ActionJournal(diag/'pre_feedback.jsonl')
    earlier = read_json(diag/'RESULTS.json')
    matrices = load_state(diag/'DELTAS.npz')
    exact(matrices['K'],k)
    assert earlier['snapshot_sha256']==anchor_sha
    panels = {'NONE':read_panel(agent,anchor,k),'K0':read_panel(agent,anchor,initial),'WRONG_K':read_panel(agent,anchor,wrong)}
    base_rng = panels['NONE']['rng_post']
    assert all(p['rng_post']==base_rng for p in panels.values())
    exact(panels['NONE']['logits'],read_json(solo/'PROBES.json')['logits'])
    budget = sum(x['substeps'] for x in panels.values())
    saved_proposals,saved_actual,rows = [],[],[]
    index = 0
    for q, previous in enumerate(earlier['rows'],1):
        assert previous['token']==q
        scratch = copy.deepcopy(agent)
        packet = forecast(scratch,q,'Z0-64','aligned',None,{},stimulus=parent.c0.vector('separate','B',q))
        packet['observation'] = {'context':'B','query':q}
        prior = previous['before']['B']
        exact(packet,historical.read_verified(prior['receipt']))
        receipt = log.publish(packet)
        exact(log.committed(receipt),packet)
        target = rules['B'][q-1]
        assert target==previous['truth_after']['B']
        log.last_receipt = None
        error = parent.error_for(packet['logits'],target)
        scratch.restore(anchor)
        assert state_hash(scratch.snapshot())==anchor_sha
        x_token = parent.c0.vector('separate',None,q)
        _,no_cue,shadow_rng,substeps = parent.neutral_read(agent,q,x_token)
        exact(no_cue.tolist(),previous['controls']['no_cue']['Z'])
        assert shadow_rng == packet['external']['rng_post'] == base_rng[f'B:{q}']
        budget += 288+substeps
        external = packet['external']
        features = {'raw':np.array(external['eligibility']),
                    'ctx':np.tanh(external['Z_after'])-np.tanh(no_cue)}
        candidates = {key:parent.candidate(agent,error,phi) for key,phi in features.items()}
        radii = {name:float(np.linalg.norm(value)) for name,value in candidates.items()}
        profiles = {}
        archived = {}
        for historical_write in previous['interventions']:
            if historical_write['write_context']=='B':
                archived[historical_write['feature']] = historical_write,matrices['delta'][index]
            index += 1
        for arm,(direction,radius) in ARMS.items():
            original = direction==radius
            old_name = 'raw' if direction=='raw' else 'context_given_token'
            if original:
                exact(candidates[direction],archived[old_name][1])
            proposal,k_after,info = match_update(k,candidates[direction],radii[radius],limit,original)
            profiles[arm] = info
            if k_after is None:
                continue
            item = read_panel(agent,anchor,k_after)
            exact(base_rng,item['rng_post'])
            if original:
                historical_write = archived[old_name][0]
                assert item['K_hash']==historical_write['K_post_hash']
                for ctx in ('A','B'):
                    exact(item['logits'][f'{ctx}:{q}'],historical_write['logits'][ctx])
            info.update(probe=item, vector_index=len(saved_proposals))
            saved_proposals.append(proposal)
            saved_actual.append(k_after-k)
            budget += item['substeps']
        rows.append({'write_token':q,'receipt':receipt,'target_after':target,'error':error.tolist(),
                     'features':{name:phi.tolist() for name,phi in features.items()},
                     'radii':{name:r if np.isfinite(r) else None for name,r in radii.items()},'arms':profiles})
    assert index==len(matrices['delta']) and anchor_sha==state_hash(agent.snapshot())
    empty = np.empty((0,*k.shape),dtype=k.dtype)
    save_state(output/'DELTAS.npz',{'proposed':np.stack(saved_proposals) if saved_proposals else empty,
                                   'applied':np.stack(saved_actual) if saved_actual else empty})
    result = {'unit':[seed,mapping],'start_sha256':anchor_sha,'start_rng_sha256':rng_hash(agent.core),
              'baseline_panels':panels,'events':rows,'substeps':budget,'clip':limit}
    atomic_json(output/'RESULTS.json',result)
    return [seed,mapping]


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('phase',choices=('init','preflight','run'))
    for name in ('output','reference','archive','handoff'):
        cli.add_argument('--'+name,type=Path,required=name=='output')
    args = cli.parse_args()
    root,phase = args.output.resolve(),args.phase
    if phase=='init':
        init(root,args.reference.resolve(),args.archive.resolve(),args.handoff.resolve())
        return
    plan = verify_plan(root)
    started = time.perf_counter()
    if phase=='preflight':
        unit = run_block((root,plan,*plan['preflight_unit'],True))
        atomic_json(root/'STATUS_preflight.json',{'phase':'PASSED','unit':unit,
                    'wall_seconds':time.perf_counter()-started})
        return
    assert read_json(root/'STATUS_preflight.json')['phase']=='PASSED'
    status = root/'STATUS_run.json'
    atomic_json(status,{'phase':'RUNNING','completed':0,'total':len(plan['units'])})
    failures = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        jobs = {pool.submit(run_block,(root,plan,*unit,False)):unit for unit in plan['units']}
        for count, future in enumerate(as_completed(jobs),1):
            try:
                print(count,len(jobs),future.result(),flush=True)
            except Exception as exc:
                failures.append({'unit':jobs[future],'type':type(exc).__name__,'message':str(exc)})
                print(failures[-1],flush=True)
            atomic_json(status,{'phase':'RUNNING','completed':count,'total':len(jobs),'failures':failures})
    atomic_json(status,{'phase':'COMPLETED' if not failures else 'COMPLETED_WITH_FAILURES',
                       'total':len(jobs),'failures':failures,'wall_seconds':time.perf_counter()-started})
    if failures:
        sys.exit(2)


if __name__=='__main__':
    main()
