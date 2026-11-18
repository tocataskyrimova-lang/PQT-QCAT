"""MEMORY-LOCATION-001: three readers of identical CARRY checkpoints.

Single-rule diagnostic, not a claim of joint-rule or lifelong mastery. The external
reader is frozen using disjoint development mappings before holdout lives run.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from sklearn.linear_model import RidgeClassifier
from pqt_qcat.core.security import check_seed_safety
from pqt_qcat.memory.checkpoint_io import save_state, load_state
from pqt_qcat.memory.dev_context_pathway_ablation import make_stable_core
from pqt_qcat.memory.exp_stateful_online_001 import StatefulOnlineAgent

ROOT=Path(__file__).resolve().parents[2]


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def source_hashes():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT/'pqt_qcat').rglob('*.py'))}


def init_plan(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(810201)
    maps=[]
    while len(maps)<24:
        row=np.array([1,2,3,4,*rng.integers(1,5,2)]);rng.shuffle(row)
        row=row.tolist()
        if row not in maps:maps.append(row)
    # No fit-class omission should masquerade as absence of encoded information.
    if any(len(set(np.array(maps[:12])[:,q]))<4 for q in range(6)):
        raise ValueError('Fit mapping panel lacks class coverage; declare a new plan before running')
    seeds={'development':[810001,810002], 'holdout':[810101,810102,810103,810104]}
    for group in seeds.values():
        for s in group:check_seed_safety(s)
    plan={'protocol':'CODEX_MEMORY_LOCATION_001','version':1,'events':72,'query_ticks':8,
          'context_ticks':0,'gate':1.0,'substeps':2,'eta_k':.8,'clip':.5,
          'checkpoint_events':[0,24,48,72], 'task':'single_rule_no_context_switch',
          'modes':['CARRY_TRAIN','RESET_TRAIN'],'seeds':seeds,
          'maps':maps,'splits':{'fit':list(range(12)),'tune':list(range(12,16)),'holdout':list(range(16,24))},
          'alpha_grid':[.01,.1,1.,10.,100.], 'stream_seed_root':810301,
          'bootstrap_seed':810401,'bootstrap_replicates':5000,
          'permutation_seed':810402,'permutations':999,
          'workers':2,'wall_cap_hours':2.0,
          'primary_checkpoint':72,'primary_metrics':['live_read','reset_read','external_K_after'],
          'positive_control':'RESET_TRAIN with independently fitted decoder',
          'interpretation':'External decode failure does not establish information absence; no M5, no R4 claim.'}
    atomic_json(output/'PLAN.json',plan)
    atomic_json(output/'SOURCE.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'dirty_diff':subprocess.check_output(['git','diff'],cwd=ROOT,text=True),
        'files':source_hashes(),'python':sys.version,'numpy':np.__version__})
    atomic_json(output/'STATUS.json',{'phase':'PLANNED','plan_hash':digest(plan)})
    return plan


def make_agent(plan,seed,mode):
    return StatefulOnlineAgent(make_stable_core(seed,substeps=plan['substeps']),learning_mode=mode,
        t_context=plan['context_ticks'],t_query=plan['query_ticks'],eta_k=plan['eta_k'],
        delta_core_max=plan['clip'],write_gate_presentation=plan['gate'],read_gate=plan['gate'])


def metrics(logits,targets):
    logits=np.asarray(logits);targets=np.asarray(targets,dtype=int)-1
    shifted=logits-logits.max(axis=1,keepdims=True)
    logp=shifted-np.log(np.exp(shifted).sum(axis=1,keepdims=True))
    other=logits.copy();other[np.arange(len(targets)),targets]=-np.inf
    return {'accuracy':float(np.mean(logits.argmax(axis=1)==targets)),
            'nll':float(-np.mean(logp[np.arange(len(targets)),targets])),
            'margin':float(np.mean(logits[np.arange(len(targets)),targets]-other.max(axis=1))),
            'predictions':(logits.argmax(axis=1)+1).tolist(),'logits':logits.tolist()}


def probe(agent,targets):
    # Each query starts from the same snapshot. Probe order cannot alter training.
    before=agent.core.Z.copy(); k=agent.core.K.copy()
    out={}
    for read_mode in ('live','reset'):
        logits=[]
        for q in range(1,7):
            clone=copy.deepcopy(agent)
            clone.learning_mode='CARRY_TRAIN'
            if read_mode=='reset':clone.reset_state_to_z0()
            _,l=clone.begin_event(0,q);logits.append(l)
        out[read_mode]=metrics(logits,targets)
    assert np.array_equal(agent.core.Z,before) and np.array_equal(agent.core.K,k)
    return out


def life_task(args):
    plan,output,mode,seed,map_id=args
    folder=Path(output)/'lives'/f'{mode}_{seed}_{map_id:02d}'
    folder.mkdir(parents=True,exist_ok=True)
    done=folder/'result.json'
    if done.exists():
        stored=json.loads(done.read_text())
        if stored['plan_hash']!=digest(plan):raise ValueError('Completed life protocol mismatch')
        return stored
    start=time.perf_counter();cpu=time.process_time()
    target=plan['maps'][map_id]
    schedule_rng=np.random.default_rng(np.random.SeedSequence([plan['stream_seed_root'],seed,map_id]))
    agent=make_agent(plan,seed,mode)
    initial_k=agent.core.K.copy();records=[];probes={};cursor=0;schedule=[]
    statefile=folder/'latest.npz'
    if statefile.exists():
        state=agent.load_checkpoint(statefile)
        if state['plan_hash']!=digest(plan):raise ValueError('Checkpoint protocol mismatch')
        cursor=state['cursor'];schedule=state['schedule'];records=state['records'];probes=state['probes']
        schedule_rng.bit_generator.state=state['stream_rng']
        initial_k=state['initial_k']
    else:
        while len(schedule)<plan['events']:
            schedule.extend((schedule_rng.permutation(6)+1).tolist())
        schedule=schedule[:plan['events']]
    failure=None

    def checkpoint(event):
        probes[str(event)]=probe(agent,target)
        runner={'cursor':event,'schedule':schedule,'stream_rng':schedule_rng.bit_generator.state,
                'records':records,'probes':probes,'initial_k':initial_k,'plan_hash':digest(plan),
                'seed':seed,'map_id':map_id,'mode':mode}
        agent.save_checkpoint(folder/f'checkpoint_{event:03d}.npz',runner)
        agent.save_checkpoint(statefile,runner)

    try:
        if cursor==0:checkpoint(0)
        for i in range(cursor,plan['events']):
            q=schedule[i]
            pred,logits=agent.begin_event(0,q)  # no target passed to forward
            k_before=agent.core.K.copy()
            feedback=target[q-1]
            stats=agent.observe_feedback_and_update(feedback)
            actual=float(np.linalg.norm(agent.core.K-k_before))
            records.append({'event':i,'query':q,'prediction_before_feedback':pred,'logits_before_feedback':logits.tolist(),
                            'target_after_prediction':feedback,**stats,'actual_delta_k_norm':actual,
                            'z_norm':float(np.linalg.norm(agent.core.Z)),'total_cell_energy':agent.core.total_energy()})
            if i+1 in plan['checkpoint_events']:checkpoint(i+1)
    except Exception as exc:
        failure={'type':type(exc).__name__,'message':str(exc),'completed_events':len(records)}
    result={'plan_hash':digest(plan),'mode':mode,'seed':seed,'map_id':map_id,'targets':target,'failure':failure,
            'events_completed':len(records),'online_accuracy':float(np.mean([r['correct'] for r in records])) if records else None,
            'probes':probes,'actual_K_change_norm':float(np.linalg.norm(agent.core.K-initial_k)),
            'training_substeps':agent.core.step_count,'max_z_norm_all_substeps':agent.core.max_state_norm,
            'max_total_cell_energy_all_substeps':agent.core.max_total_energy,
            'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-start}
    save_state(folder/'features.npz',{'K_initial':initial_k[32:35].reshape(-1),
               'K_after':agent.core.K[32:35].reshape(-1),'delta_K':(agent.core.K-initial_k)[32:35].reshape(-1)})
    atomic_json(folder/'events.json',records);atomic_json(done,result)
    return result


def load_features(output,mode,seeds,maps,kind):
    rows=[];y=[]
    for s in seeds:
        for m in maps:
            folder=Path(output)/'lives'/f'{mode}_{s}_{m:02d}'
            result=json.loads((folder/'result.json').read_text())
            if result['failure']:raise ValueError('Failed life cannot silently enter decoder')
            rows.append(load_state(folder/'features.npz')[kind]);y.append(result['targets'])
    return np.array(rows),np.array(y)


def fit_readers(plan,output):
    saved={}
    for mode in plan['modes']:
        saved[mode]={}
        for feature in ('K_after','K_initial','delta_K'):
            xf,yf=load_features(output,mode,plan['seeds']['development'],plan['splits']['fit'],feature)
            xt,yt=load_features(output,mode,plan['seeds']['development'],plan['splits']['tune'],feature)
            models=[]
            for q in range(6):
                scores=[]
                for alpha in plan['alpha_grid']:
                    model=RidgeClassifier(alpha=alpha).fit(xf,yf[:,q])
                    scores.append(float(np.mean(model.predict(xt)==yt[:,q])))
                alpha=plan['alpha_grid'][int(np.argmax(scores))]
                model=RidgeClassifier(alpha=alpha).fit(np.vstack([xf,xt]),np.concatenate([yf[:,q],yt[:,q]]))
                models.append({'alpha':alpha,'tune_scores':scores,'classes':model.classes_.tolist(),
                               'coef':model.coef_.tolist(),'intercept':model.intercept_.tolist()})
            saved[mode][feature]=models
        dev_y=np.array([plan['maps'][i] for i in plan['splits']['fit']+plan['splits']['tune']])
        saved[mode]['majority']=[int(np.argmax(np.bincount(dev_y[:,q],minlength=5)[1:])+1) for q in range(6)]
    atomic_json(Path(output)/'FROZEN_READERS.json',saved)
    atomic_json(Path(output)/'READER_FREEZE.json',{'utc':datetime.now(timezone.utc).isoformat(),
        'sha256':hashlib.sha256((Path(output)/'FROZEN_READERS.json').read_bytes()).hexdigest(),
        'holdout_started':False,'plan_hash':digest(plan)})
    return saved


def predict_reader(models,x):
    return np.array([np.array(m['classes'])[np.argmax(x@np.array(m['coef']).T+np.array(m['intercept']),axis=1)] for m in models]).T


def crossed_ci(values,seed,reps):
    values=np.asarray(values);rng=np.random.default_rng(seed);means=[]
    for _ in range(reps):
        ix=rng.integers(0,values.shape[0],values.shape[0]);iy=rng.integers(0,values.shape[1],values.shape[1])
        means.append(float(values[ix[:,None],iy].mean()))
    return [float(x) for x in np.quantile(means,[.025,.975])]


def analyze(plan,output,readers):
    output=Path(output);result={};saved={};nseed=len(plan['seeds']['holdout']);nmaps=len(plan['splits']['holdout'])
    for mode in plan['modes']:
        x,y=load_features(output,mode,plan['seeds']['holdout'],plan['splits']['holdout'],'K_after')
        predictions={}
        for kind in ('K_after','K_initial','delta_K'):
            x,_=load_features(output,mode,plan['seeds']['holdout'],plan['splits']['holdout'],kind)
            predictions[kind]=predict_reader(readers[mode][kind],x)
        predictions['majority']=np.tile(readers[mode]['majority'],(len(y),1))
        rows=[json.loads((output/'lives'/f'{mode}_{s}_{m:02d}'/'result.json').read_text())
              for s in plan['seeds']['holdout'] for m in plan['splits']['holdout']]
        for read in ('live','reset'):
            predictions[read]=np.array([r['probes'][str(plan['primary_checkpoint'])][read]['predictions'] for r in rows])
        acc={key:np.mean(pred==y,axis=1).reshape(nseed,nmaps) for key,pred in predictions.items()}
        result[mode]={'accuracy':{k:float(a.mean()) for k,a in acc.items()},
                      'online_accuracy':float(np.mean([r['online_accuracy'] for r in rows])),
                      'contrasts':{}}
        for a,b in [('reset','live'),('K_after','live'),('K_after','K_initial')]:
            diff=acc[a]-acc[b]
            result[mode]['contrasts'][a+'-minus-'+b]={'mean':float(diff.mean()),
                    'ci95_crossed':crossed_ci(diff,plan['bootstrap_seed'],plan['bootstrap_replicates'])}
        # Conditional association of a frozen reader with held-out mappings; maps permuted jointly across seeds.
        yp=predictions['K_after'].reshape(nseed,nmaps,6);yt=y.reshape(nseed,nmaps,6)
        rng=np.random.default_rng(plan['permutation_seed']);stats=[]
        for _ in range(plan['permutations']):stats.append(float(np.mean(yp==yt[:,rng.permutation(nmaps),:])))
        obs=acc['K_after'].mean()
        result[mode]['permutation_p']=(1+sum(v>=obs for v in stats))/(1+len(stats))
        result[mode]['permutation_mean']=float(np.mean(stats))
        result[mode]['learned_live_gate_75']=bool(acc['live'].mean()>=.75)
        result[mode]['unit_counts']={'seeds':nseed,'maps':nmaps,'queries_per_life':6}
        saved[mode]={'predictions':predictions,'targets':y,'permutation_statistics':np.array(stats),
                     'seeds':plan['seeds']['holdout'],'map_ids':plan['splits']['holdout']}
    save_state(output/'predictions.npz',saved)
    atomic_json(output/'RESULTS.json',result)
    return result


def run(output,workers=2):
    output=Path(output)
    plan=json.loads((output/'PLAN.json').read_text())
    source=json.loads((output/'SOURCE.json').read_text())
    if source['files']!=source_hashes():raise ValueError('Source changed after plan freeze; use a new run directory')
    if not 1<=workers<=plan['workers']:raise ValueError('Worker cap exceeded')
    started=time.perf_counter()
    for phase in ('development','holdout'):
        if phase=='holdout':
            if not (output/'FROZEN_READERS.json').exists():fit_readers(plan,output)
            readers=json.loads((output/'FROZEN_READERS.json').read_text())
            freeze=json.loads((output/'READER_FREEZE.json').read_text())
            if freeze['sha256']!=hashlib.sha256((output/'FROZEN_READERS.json').read_bytes()).hexdigest():
                raise ValueError('Frozen reader has changed')
        seeds=plan['seeds'][phase]
        maps=plan['splits']['fit']+plan['splits']['tune'] if phase=='development' else plan['splits']['holdout']
        tasks=[(plan,str(output),mode,s,m) for mode in plan['modes'] for s in seeds for m in maps]
        atomic_json(output/'STATUS.json',{'phase':phase,'total_lives':len(tasks),'completed':0})
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(life_task,t):t for t in tasks}
            for i,f in enumerate(as_completed(futures),1):
                r=f.result()
                atomic_json(output/'STATUS.json',{'phase':phase,'total_lives':len(tasks),'completed':i,'last_failure':r['failure']})
                print(f'{phase} {i}/{len(tasks)} {r["mode"]} seed={r["seed"]} map={r["map_id"]} failed={r["failure"] is not None}',flush=True)
                if r['failure']:raise RuntimeError('Life failed: keep evidence and stop dependent inference')
                if time.perf_counter()-started>plan['wall_cap_hours']*3600:raise TimeoutError('Wall budget exhausted')
    result=analyze(plan,output,readers)
    atomic_json(output/'STATUS.json',{'phase':'COMPLETED','wall_seconds':time.perf_counter()-started,
                                     'reader_sha256':hashlib.sha256((output/'FROZEN_READERS.json').read_bytes()).hexdigest()})
    print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['init','run']);parser.add_argument('--output',required=True)
    parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    if args.command=='init':init_plan(args.output)
    else:
        try:run(args.output,args.workers)
        except Exception as exc:
            atomic_json(Path(args.output)/'STATUS.json',{'phase':'STOPPED_ERROR','type':type(exc).__name__,'message':str(exc)})
            raise


if __name__=='__main__':main()
