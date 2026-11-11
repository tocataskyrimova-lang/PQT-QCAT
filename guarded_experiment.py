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
    old={c:rules[c] for c in ['A','B','C','D']};query_metrics(agent,old,300000+seed);snapshot=agent.snapshot();rows={}
    for mode in ('no_idle','naive','guarded','no_replay'):
        branch=Life.restore(snapshot)
        if mode=='guarded':branch.cfg.guarded_replay=True
        if mode=='no_replay':branch.cfg.replay=False
        if mode!='no_idle':branch.idle(10000)
        rows[mode]={'queries':query_metrics(branch,old,310000+seed),'status':branch.status()}
    VARIANTS['guarded_PQT']={'guarded_replay':True}
    full=run_life((seed,'guarded_PQT'))
    return {'seed':seed,'policies':rows,'full_guarded_life':full}

if __name__=='__main__':
    result=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(run,s) for s in (701,702,703)]):
            row=f.result();result.append(row);save(ROOT/'guarded_results.json',result)
            print('GUARDED',row['seed'],{k:np.mean([v['accuracy'] for v in z['queries'].values()]) for k,z in row['policies'].items()},flush=True)
