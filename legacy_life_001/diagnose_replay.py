"""Posthoc causal follow-up on harmful idle without consolidation.
New seeds601-603; no primary hyperparameters or results are changed.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from life import Life,Config
from experiment import ROOT,save,train_stream,query_metrics

def run(seed):
    agent=Life(Config(seed=seed,consolidation=False));rng=np.random.default_rng(200000+seed)
    rules={c:rng.permutation(8) for c in ['A','B','C','D','A_v2']};logs=[];pending=[]
    for c in ['A','B','C','D']:train_stream(agent,c,rules[c],rng,500,c,logs,pending)
    while pending:
        eid,y,_=pending.pop(0);agent.feedback(eid,y)
    old={c:rules[c] for c in ['A','B','C','D']};query_metrics(agent,old,300000+seed)
    snapshot=agent.snapshot();rows={}
    for mode in ['no_idle','replay_on','replay_off','replay_without_drive','consolidation_enabled','freeze_core_during_idle']:
        branch=Life.restore(snapshot)
        if mode=='replay_off':branch.cfg.replay=False
        if mode=='replay_without_drive':branch.cfg.internal_drive=False
        if mode=='consolidation_enabled':branch.cfg.consolidation=True
        original_tick=branch.core.tick
        if mode=='freeze_core_during_idle':branch.core.tick=lambda drive:None
        if mode!='no_idle':branch.idle(10000)
        branch.core.tick=original_tick
        rows[mode]={'queries':query_metrics(branch,old,310000+seed),'status':branch.status()}
    return {'seed':seed,'policies':rows}

if __name__=='__main__':
    results=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(run,s) for s in (601,602,603)]):
            row=f.result();results.append(row);save(ROOT/'replay_diagnosis.json',results)
            print('REPLAY DIAGNOSIS',row['seed'],{k:np.mean([q['accuracy'] for q in v['queries'].values()]) for k,v in row['policies'].items()},flush=True)
