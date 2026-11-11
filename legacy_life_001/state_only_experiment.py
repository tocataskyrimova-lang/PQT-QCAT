"""New seeds801-803; separate internal replay from replay weight updates.
Predefined follow-up after anchor guard did not remove all idle harm.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from life import Life,Config
from experiment import ROOT,save,train_stream,query_metrics,VARIANTS,run_life

def run(seed):
    agent=Life(Config(seed=seed,consolidation=False));rng=np.random.default_rng(200000+seed)
    rules={c:rng.permutation(8) for c in ['A','B','C','D','A_v2']};logs=[];pending=[]
    for c in ['A','B','C','D']:train_stream(agent,c,rules[c],rng,500,c,logs,pending)
    while pending:
        eid,y,_=pending.pop(0);agent.feedback(eid,y)
    old={c:rules[c] for c in ['A','B','C','D']};query_metrics(agent,old,300000+seed);snap=agent.snapshot();rows={}
    for mode in ('no_idle','naive','state_only','no_replay'):
        branch=Life.restore(snap);before=branch.status()
        if mode=='state_only':branch.cfg.replay_weight_updates=False
        if mode=='no_replay':branch.cfg.replay=False
        if mode!='no_idle':branch.idle(10000)
        after=branch.status()
        rows[mode]={'queries':query_metrics(branch,old,310000+seed),'status':after,
                    'idle_path_length':after['core_path_length']-before['core_path_length']}
    VARIANTS['state_only_PQT']={'replay_weight_updates':False}
    full=run_life((seed,'state_only_PQT'))
    return {'seed':seed,'policies':rows,'full_state_only_life':full}

if __name__=='__main__':
    rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(run,s) for s in (801,802,803)]):
            r=f.result();rows.append(r);save(ROOT/'state_only_results.json',rows)
            print('STATE ONLY',r['seed'],{k:np.mean([v['accuracy'] for v in z['queries'].values()]) for k,z in r['policies'].items()},flush=True)
