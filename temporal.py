"""History-dependent task: query0 alone contains no cue information."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
import numpy as np
from life import Life,Config
from experiment import ROOT,VARIANTS,save

def run(job):
    seed,variant=job;agent=Life(replace(Config(seed=seed),**VARIANTS[variant]));rng=np.random.default_rng(500000+seed)
    rule=rng.permutation(np.arange(1,8));scores=[];pending=[]
    for i in range(1200):
        cue=int(rng.integers(1,8));agent.predict(cue,'temporal')
        agent.idle(int(rng.integers(0,21)))
        r=agent.predict(0,'temporal');target=int(rule[cue-1]);scores.append(r['token']==target);pending.append((r['event_id'],target))
        if len(pending)>3:agent.feedback(*pending.pop(0))
    for item in pending:agent.feedback(*item)
    snapshot=agent.snapshot();tests={}
    for delay in (0,10,20,50,100,300):
        # Paired continuations of one learned life, each delay has its own cloned branch.
        branch=Life.restore(snapshot);correct=[];gen=np.random.default_rng(510000+seed)
        for i in range(280):
            cue=int(gen.integers(1,8));branch.predict(cue,'temporal');branch.idle(delay);answer=branch.predict(0,'temporal')
            correct.append(answer['token']==int(rule[cue-1]))
        tests[str(delay)]={'accuracy':float(np.mean(correct)),'n':280,'status':branch.status()}
    row={'seed':seed,'variant':variant,'first100':float(np.mean(scores[:100])),'last100':float(np.mean(scores[-100:])),
         'tests':tests,'status':agent.status(),'chance':1/7}
    save(ROOT/'temporal_results'/variant/str(seed)/'result.json',row)
    return row

if __name__=='__main__':
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(run,(s,k)) for s in (301,302,303) for k in ('PQT','noA','leaky','reset_core')]
        for f in as_completed(fs):
            r=f.result();rows.append(r);save(ROOT/'temporal_results.json',rows)
            print('TEMPORAL',r['seed'],r['variant'],r['last100'],{k:v['accuracy'] for k,v in r['tests'].items()},flush=True)
