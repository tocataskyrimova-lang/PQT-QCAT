"""Prequential continuous lives; teacher exists only in this environment."""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace,asdict
from pathlib import Path
import argparse,copy,json,time
import numpy as np
from life import Life,Config

ROOT=Path(__file__).resolve().parent
VARIANTS={
    'PQT':{},'noA':{'core':'noA'},'leaky':{'core':'leaky'},
    'frozen':{'learning':False},'no_replay':{'replay':False},
    'no_consolidation':{'consolidation':False},'reset_core':{'reset_core':True},
    'shared_context':{'context_gates':False},'fixed_encoder':{'input_learning_rate':0.},
    'no_replay_no_consolidation':{'replay':False,'consolidation':False}}

def finite_json(obj):
    if isinstance(obj,np.ndarray):return obj.tolist()
    if isinstance(obj,np.generic):return obj.item()
    raise TypeError(type(obj))
def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,default=finite_json,indent=2,allow_nan=False),encoding='utf8');os.replace(tmp,path)

def query_metrics(agent,rules,seed,n=64):
    rng=np.random.default_rng(seed);scores={}
    # Queries have no feedback, but physical ticks/consolidation never pause.
    for context,rule in rules.items():
        correct=[];loss=[];confidence=[]
        for token in rng.integers(0,8,n):
            r=agent.predict(int(token),context);y=int(rule[token]);correct.append(r['token']==y)
            loss.append(-np.log(max(r['probabilities'][y],1e-12)));confidence.append(max(r['probabilities']))
        scores[context]={'accuracy':float(np.mean(correct)),'cross_entropy':float(np.mean(loss)),'confidence':float(np.mean(confidence))}
    return scores

def generation_metrics(agent,rules):
    rows=[]
    for context,rule in rules.items():
        for start in range(8):
            emitted=agent.generate(start,context,8);got=[r['token'] for r in emitted];target=[];token=start
            for _ in range(8):token=int(rule[token]);target.append(token)
            rows.append({'context':context,'start':start,'generated':got,'target':target,
                         'exact':got==target,'prefix_tokens':next((i for i,(a,b) in enumerate(zip(got,target)) if a!=b),8)})
    return {'exact_match':float(np.mean([r['exact'] for r in rows])),'rows':rows}

def train_stream(agent,context,rule,rng,n,stage,event_log,pending):
    scores=[]
    for i in range(n):
        token=int(rng.integers(0,8));answer=agent.predict(token,context);target=int(rule[token]);ok=answer['token']==target;scores.append(ok)
        row={'stage':stage,'context':context,'event_id':answer['event_id'],'input':token,'prediction':answer['token'],
             'target_received_later':target,'correct_before_feedback':ok,'prediction_tick':answer['tick'],'feedback_tick':None}
        event_log.append(row);pending.append((answer['event_id'],target,row))
        if len(pending)>3:
            eid,y,old=pending.pop(0);agent.feedback(eid,y);old['feedback_tick']=agent.clock
    adaptation=next((i+1 for i in range(99,len(scores)) if np.mean(scores[i-99:i+1])>=.90),None)
    return {'stage':stage,'context':context,'n':n,'first100':float(np.mean(scores[:100])),
            'last100':float(np.mean(scores[-100:])),'overall':float(np.mean(scores)),'events_to_rolling100_90pct':adaptation}

def run_life(job):
    seed,variant=job;cfg=replace(Config(seed=seed),**VARIANTS[variant]);agent=Life(cfg)
    rng=np.random.default_rng(200000+seed);rules={c:rng.permutation(8) for c in ['A','B','C','D','A_v2']}
    logs=[];pending=[];stages=[];initial_encoder=agent.encoder.copy();tic=time.perf_counter()
    for c in ['A','B','C','D']:
        stages.append(train_stream(agent,c,rules[c],rng,500,'learn_'+c,logs,pending))
    while pending:
        eid,y,row=pending.pop(0);agent.feedback(eid,y);row['feedback_tick']=agent.clock
    oldrules={c:rules[c] for c in ['A','B','C','D']}
    before=query_metrics(agent,oldrules,300000+seed)
    before_status=agent.status();snapshot=agent.snapshot();state_before=agent.core.z.copy();physical_w_before=agent.core.z[:,3:6].copy()
    no_idle=Life.restore(snapshot);paired_noidle=query_metrics(no_idle,oldrules,310000+seed)
    agent.idle(10000)
    idle_status=agent.status();idle_motion=float(np.linalg.norm(agent.core.z-state_before));idle_w=float(np.linalg.norm(agent.core.z[:,3:6]-physical_w_before))
    after=query_metrics(agent,oldrules,310000+seed)
    generated=generation_metrics(agent,oldrules)
    stages.append(train_stream(agent,'A_v2',rules['A_v2'],rng,500,'new_version_A',logs,pending))
    while pending:
        eid,y,row=pending.pop(0);agent.feedback(eid,y);row['feedback_tick']=agent.clock
    old_after_new=query_metrics(agent,oldrules,320000+seed)
    new_rule=query_metrics(agent,{'A_v2':rules['A_v2']},330000+seed)
    # Contradictory rule under the same observed context: a separate stress test.
    replacement=rng.permutation(8)
    stages.append(train_stream(agent,'A',replacement,rng,500,'unannounced_overwrite_A',logs,pending))
    while pending:
        eid,y,row=pending.pop(0);agent.feedback(eid,y);row['feedback_tick']=agent.clock
    conflict={'new_A':query_metrics(agent,{'A':replacement},340000+seed),
              'old_A':query_metrics(agent,{'A':rules['A']},340000+seed)}
    status=agent.status()
    result={'seed':seed,'variant':variant,'config':asdict(cfg),'seconds':time.perf_counter()-tic,
            'stages':stages,'before_idle':before,'paired_noidle':paired_noidle,'after_idle':after,
            'generation':generated,'old_after_new_context':old_after_new,'new_rule':new_rule,'same_context_conflict':conflict,
            'idle_metrics':{'external_events_delta':idle_status['external_events']-before_status['external_events'],
                'teacher_feedback_delta':idle_status['feedback']-before_status['feedback'],
                'replay_updates':idle_status['replay_updates']-before_status['replay_updates'],
                'consolidations':idle_status['consolidations']-before_status['consolidations'],
                'core_motion':idle_motion,'physical_W_motion':idle_w,
                'path_length':idle_status['core_path_length']-before_status['core_path_length']},
            'encoder_delta_norm':float(np.linalg.norm(agent.encoder-initial_encoder)),
            'status':status,'all_feedback_follows_prediction':all(r['feedback_tick']>=r['prediction_tick'] for r in logs)}
    folder=ROOT/'results'/variant/str(seed);save(folder/'result.json',result);save(folder/'events.json',logs)
    agent.save(folder/'final_life.json.gz')
    return result

def long_life(job):
    seed,variant=job;agent=Life(replace(Config(seed=seed),**VARIANTS[variant]));rng=np.random.default_rng(400000+seed)
    stages=[];logs=[];pending=[];tic=time.perf_counter()
    for episode in range(60):
        context=f'context_{episode%3}';rule=rng.permutation(8)
        stages.append(train_stream(agent,context,rule,rng,300,f'change_{episode}',logs,pending))
        while pending:
            eid,y,row=pending.pop(0);agent.feedback(eid,y);row['feedback_tick']=agent.clock
        agent.idle(100)
    result={'seed':seed,'variant':variant,'stages':stages,'seconds':time.perf_counter()-tic,'status':agent.status()}
    folder=ROOT/'long_results'/variant/str(seed);save(folder/'result.json',result);agent.save(folder/'final_life.json.gz')
    # Avoid a huge repeated target ledger here; main life contains full causal logs.
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['development','main','long'],default='main');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if a.mode=='development':
        result=run_life((12,'PQT'));save(ROOT/'development_life12.json',result)
        print('DEV12',result['stages'],result['after_idle'],flush=True);return
    jobs=[(s,k) for s in range(101,106) for k in VARIANTS] if a.mode=='main' else [(s,k) for s in (201,202,203) for k in ('PQT','noA','fixed_encoder')]
    fn=run_life if a.mode=='main' else long_life
    rows=[];target=ROOT/('main_results.json' if a.mode=='main' else 'long_results.json')
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(fn,job):job for job in jobs}
        for future in as_completed(futures):
            row=future.result();rows.append(row);save(target,rows)
            print(a.mode,row['seed'],row['variant'],'last',row['stages'][-1]['last100'],'finite',row['status']['finite'],'seconds',round(row['seconds'],1),flush=True)
    print('COMPLETE',len(rows),flush=True)

if __name__=='__main__':main()
