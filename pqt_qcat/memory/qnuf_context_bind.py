"""D0–D3 causal diagnostics on the locked C0 development panel."""
from . import qnuf_tworule as c0  # cap numerical threads first

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zipfile import ZipFile

import numpy as np

from .access_state import exact
from .checkpoint_io import load_state, save_state
from .codex_memory_location import atomic_json,digest,source_hashes
from .functional_sidecar import ActionJournal,read_json,state_hash
from .functional_sidecar_b2a import forecast,read_branch
from .qnuf import ObservedEvidence,array_hash,rng_hash


BASE = '1b9eaaff981308f869dd58bf372b3cd4b7857b46'
BASE_EXECUTABLE = '89482a1cfb5dde2e4c01af4203cf9fa55812847b'
BASE_ZIP = '65f232c6cd7f2a66354669a69b0821d7c3f253cc3ddbb4de615db19c76b6c50d'
PRIMARY = ('separate','SS')
CONTEXTS = ('A','B')
WORKERS = 6


def init(root, reference, archive):
    repo = c0.causal.b2.b1.bridge.ROOT
    subprocess.run(['git','merge-base','--is-ancestor',BASE,'HEAD'],cwd=repo,check=True)
    subprocess.run(['git','diff','--quiet',BASE_EXECUTABLE,BASE,'--','pqt_qcat'],cwd=repo,check=True)
    delivery, source = (read_json(reference/name) for name in ('DELIVERY.json','SOURCE.json'))
    exact([delivery['delivery_commit'],delivery['frozen_executable_sha'],source['git']],
          [BASE,BASE_EXECUTABLE,BASE_EXECUTABLE])
    assert delivery['verdicts']['ENGINEERING_VALIDITY'] == 'C0_VALID'
    assert c0.causal.file_sha(archive) == BASE_ZIP
    manifest = reference/'ARTIFACT_MANIFEST.sha256'
    entries = [row.split('  ',1) for row in manifest.read_text(encoding='utf-8').splitlines()]
    for expected,name in entries:
        assert c0.causal.file_sha(reference/name) == expected, name
    with ZipFile(archive) as packed:
        frozen = packed.read('FROZEN_CODE.zip')
    tracked = subprocess.check_output(['git','archive','--format=zip',BASE_EXECUTABLE],cwd=repo)
    assert frozen == tracked == (reference/'FROZEN_CODE.zip').read_bytes()
    with ZipFile(reference/'FROZEN_CODE.zip') as packed:
        assert packed.comment.decode('ascii') == BASE_EXECUTABLE
        for name, expected in source['files'].items():
            assert hashlib.sha256(packed.read(name.replace('\\','/'))).hexdigest() == expected, name
    inherited = read_json(reference/'PLAN.json')
    plan = {'base':BASE,'base_executable':BASE_EXECUTABLE,'base_zip_sha256':BASE_ZIP,
            'base_source_zip_sha256':hashlib.sha256(frozen).hexdigest(),
            'base_archive':str(archive),'reference':str(reference),'reference_inputs':{name:h for h,name in entries},
            'c0':inherited,'units':inherited['development_units'],'policies':c0.NEURAL,'primary':PRIMARY,
            'solo_gate':.75,'solo_events':48,'solo_cut':24,'horizon128':PRIMARY,
            'diagnostic_anchors':'joint and A-only for all arms; B-only additionally for separate/SS',
            'diagnostic_features':'raw for all; token given context and context given token only for separate/SS',
            'decoder':'fixed class-wise standardized Ridge alpha=1; fit excludes both test seed and mapping',
            'decoder_control':'fixed K0, wrong same-seed K, swapped cue, and blind context',
            'bootstrap_seed':821204,'bootstrap_replicates':5000,
            'scope':'C0 development only; C1, independent/holdout, M5, DREAM, N/F and adaptive configuration excluded'}
    root.mkdir(parents=True,exist_ok=False)
    for name in ('ANALYSIS.json','DELIVERY.json','SOURCE.json','ARTIFACT_MANIFEST.sha256'):
        shutil.copy2(reference/name,root/('BASE_C0_'+name))
    atomic_json(root/'PLAN.json',plan)
    atomic_json(root/'SOURCE.json',{'git':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
                'files':source_hashes(),'plan_sha256':digest(plan),'python':sys.version})
    atomic_json(root/'LOCK.json',{name:c0.causal.file_sha(root/name) for name in ('PLAN.json','SOURCE.json')})


def verify_plan(root):
    locks = read_json(root/'LOCK.json')
    assert all(c0.causal.file_sha(root/name)==h for name,h in locks.items())
    plan,source = read_json(root/'PLAN.json'),read_json(root/'SOURCE.json')
    assert digest(plan)==source['plan_sha256'] and source_hashes()==source['files']
    return plan


def single_plan(plan, context):
    c = plan['c0']
    return {**c,'schedules':{k:[event for event in sequence if event[0]==context] for k,sequence in c['schedules'].items()},
            'events':plan['solo_events'],'cut':plan['solo_cut']}


def path_for(root,plan,anchor,seed,mapping,mode,arm):
    folder = Path(plan['reference']) if anchor=='joint' else root/'single'/anchor
    return c0.where(folder,seed,mapping,mode,arm)


def agent_for(path,plan,seed):
    saved = load_state(path/'end.npz')['agent']
    agent = c0.new_agent('B3',seed,plan['c0']['parent'])
    agent.restore(saved)
    return agent


def neutral_read(agent, token, sensory, k=None, horizon=64):
    if horizon==64:
        out = read_branch(copy.deepcopy(agent.core),agent.Z0,token,'Z0-64',k,False,agent.read_gate,stimulus=sensory)
        return out['logits'], np.array(out['Z_after']), out['rng_post'], out['side_substeps']
    item = c0.read_trace(agent,sensory,agent.Z0,agent.core.K if k is None else k,horizon)
    final = item['trace'][str(horizon)]
    return final['logits'],np.array(final['Z']),item['rng_post'],item['substeps']


def solo(job):
    root,plan,seed,mapping,mode,arm,context,resume = job
    folder = root/'single'/context
    c0.train((folder,single_plan(plan,context),seed,mapping,mode,arm,resume))
    if not resume:
        path = path_for(root,plan,context,seed,mapping,mode,arm)
        agent = agent_for(path,plan,seed)
        before = state_hash(agent.snapshot())
        rows = {f'{ctx}:{q}':neutral_read(agent,q,c0.vector(mode,ctx,q))[0] for ctx in CONTEXTS for q in range(1,7)}
        assert before==state_hash(agent.snapshot())
        atomic_json(path/'PROBES.json',{'logits':rows,'snapshot_hash':before,'substeps':12*128})
    return [context,seed,mapping,mode,arm]


def error_for(logits,target):
    logits = np.array(logits)
    p = np.exp(logits-logits.max())
    p /= p.sum()
    e = np.zeros(8)
    e[1:5] = -p
    e[target] = 1-p[target-1]
    return e


def candidate(agent,error,phi):
    return np.clip(agent.eta_k*np.outer(agent.core.W_readout.T@error,phi),-agent.delta_core_max,agent.delta_core_max)


def diagnostics(job):
    root,plan,seed,mapping,mode,arm,anchor = job
    c0.block(plan['c0'],seed,mapping)
    path = path_for(root,plan,anchor,seed,mapping,mode,arm)
    agent = agent_for(path,plan,seed)
    original = state_hash(agent.snapshot())
    k = agent.core.K.copy()
    output = root/'diagnostics'/anchor/c0.causal.unit_name(seed,mapping)/mode/arm
    output.mkdir(parents=True,exist_ok=False)
    journal = ActionJournal(output/'pre_feedback.jsonl')
    rows, vectors, budget = [], [], 0
    extra = (mode,arm)==PRIMARY
    for q in range(1,7):
        packets, targets, errors, features, states = {}, {}, {}, {}, {}
        specs = {}
        for ctx in CONTEXTS:
            branch = copy.deepcopy(agent)
            packet = forecast(branch,q,'Z0-64','aligned',None,{},stimulus=c0.vector(mode,ctx,q))
            packet['observation'] = {'context':ctx,'query':q}
            receipt = journal.publish(packet)
            exact(journal.committed(receipt),packet)
            target = plan['c0']['mappings'][mapping][ctx][q-1]
            journal.last_receipt = None
            logits,raw = c0.causal.credit_values(packet,arm)
            error = error_for(logits,target)
            applied, _ = c0.causal.apply_credit(branch,packet,ObservedEvidence(packet['main']['event_id'],target),arm,None)
            delta = candidate(agent,error,raw)
            exact(k+delta,branch.core.K)
            assert applied['K_post_hash']==array_hash(branch.core.K)
            packets[ctx], targets[ctx], errors[ctx] = {'receipt':receipt,'packet':packet},target,error
            features[ctx] = {'raw':raw}
            specs[('raw',ctx)] = delta
            budget += 288
        if extra:
            x_token = c0.vector(mode,None,q)
            _, no_cue, common_rng, steps = neutral_read(agent,q,x_token)
            budget += steps
            states['no_cue'] = {'input':x_token.tolist(),'Z':no_cue.tolist(),'rng_post':common_rng}
            for ctx in CONTEXTS:
                x_cue_only = c0.vector(mode,ctx,q)-x_token
                _, no_token, rnd, steps = neutral_read(agent,q,x_cue_only)
                budget += steps
                ext = packets[ctx]['packet']['external']
                assert common_rng==rnd==ext['rng_post']
                full = np.tanh(ext['Z_after'])
                states[ctx] = {'input':x_cue_only.tolist(),'Z':no_token.tolist(),'rng_post':rnd}
                features[ctx].update(token_given_context=full-np.tanh(no_token),context_given_token=full-np.tanh(no_cue))
                for name in ('token_given_context','context_given_token'):
                    specs[(name,ctx)] = candidate(agent,errors[ctx],features[ctx][name])
        interventions = []
        for (name,ctx), proposed in specs.items():
            predictions = {}
            for read_ctx in CONTEXTS:
                logits,_,rnd,steps = neutral_read(agent,q,c0.vector(mode,read_ctx,q),k+proposed)
                assert rnd==packets[read_ctx]['packet']['external']['rng_post']
                predictions[read_ctx] = logits
                budget += steps
            interventions.append({'feature':name,'write_context':ctx,'logits':predictions,'K_post_hash':array_hash(k+proposed)})
            vectors.append(proposed)
        fixed_error = {name:[candidate(agent,errors['A'],features[ctx][name]) for ctx in CONTEXTS]
                       for name in features['A']}
        rows.append({'token':q,'before':packets,'truth_after':targets,'controls':states,'interventions':interventions,
                     'features':{ctx:{name:phi.tolist() for name,phi in group.items()} for ctx,group in features.items()},
                     'fixed_error_cos':{name:float(np.vdot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
                                        if np.linalg.norm(a)*np.linalg.norm(b) else None for name,(a,b) in fixed_error.items()}})
    assert original==state_hash(agent.snapshot())
    save_state(output/'DELTAS.npz',{'K':k,'delta':np.stack(vectors)})
    atomic_json(output/'RESULTS.json',{'anchor':anchor,'snapshot_sha256':original,'rows':rows,'substeps':budget})
    return [anchor,seed,mapping,mode,arm]


def horizon(job):
    root,plan,seed,mapping = job
    mode,arm = PRIMARY
    c0.block(plan['c0'],seed,mapping)
    agent = agent_for(path_for(root,plan,'joint',seed,mapping,mode,arm),plan,seed)
    before = state_hash(agent.snapshot())
    logits = {f'{ctx}:{q}':neutral_read(agent,q,c0.vector(mode,ctx,q),horizon=128)[0]
              for ctx in CONTEXTS for q in range(1,7)}
    assert before==state_hash(agent.snapshot())
    atomic_json(root/'horizon128'/f'{seed}_{mapping}.json',{'logits':logits,'snapshot_sha256':before,'substeps':12*256})
    return [seed,mapping]


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('phase',choices=('init','single','resume','diagnostics','horizon'))
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--reference',type=Path)
    cli.add_argument('--archive',type=Path)
    args = cli.parse_args()
    root,phase = args.output.resolve(),args.phase
    if phase=='init':
        init(root,args.reference.resolve(),args.archive.resolve())
        return
    plan = verify_plan(root)
    if phase in ('single','resume'):
        function = solo
        units = plan['units'] if phase=='single' else plan['units'][:1]
        jobs = [(root,plan,*unit,*arm,context,phase=='resume') for unit in units for arm in c0.NEURAL for context in CONTEXTS]
    elif phase=='diagnostics':
        function = diagnostics
        jobs = [(root,plan,*unit,*arm,anchor) for unit in plan['units'] for arm in c0.NEURAL
                for anchor in (('joint','A','B') if arm==PRIMARY else ('joint','A'))]
    else:
        function,jobs = horizon,[(root,plan,*unit) for unit in plan['units']]
    status = root/f'STATUS_{phase}.json'
    start = time.perf_counter()
    atomic_json(status,{'phase':'RUNNING','completed':0,'total':len(jobs)})
    failures = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(function,job):job[2:] for job in jobs}
        for count,future in enumerate(as_completed(futures),1):
            try:
                print(count,len(jobs),future.result(),flush=True)
            except Exception as exc:
                error = {'job':futures[future],'type':type(exc).__name__,'message':str(exc)}
                failures.append(error)
                print('FAILED',error,flush=True)
            atomic_json(status,{'phase':'RUNNING','completed':count,'total':len(jobs),'failures':failures})
    atomic_json(status,{'phase':'COMPLETED_WITH_FAILURES' if failures else 'COMPLETED', 'total':len(jobs),
                        'failures':failures,'wall_seconds':time.perf_counter()-start})
    if failures:
        sys.exit(2)


if __name__=='__main__':
    main()
