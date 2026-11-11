"""Development only. These seeds are never used in final results."""
import json,time
from pathlib import Path
import numpy as np
from life import Life,Config
ROOT=Path(__file__).resolve().parent
out=[]
for lr in (.3,.8,1.5):
    agent=Life(Config(seed=11,learning_rate=lr));rng=np.random.default_rng(1811);pending=[];rows=[];start=time.perf_counter()
    for context in ('A','B'):
        rule=rng.permutation(8);correct=[]
        for i in range(500):
            token=int(rng.integers(8));reply=agent.predict(token,context);target=int(rule[token]);correct.append(reply['token']==target)
            pending.append((reply['event_id'],target))
            if len(pending)>3:agent.feedback(*pending.pop(0))
        rows.append({'context':context,'first100':np.mean(correct[:100]),'last100':np.mean(correct[-100:])})
    for item in pending:agent.feedback(*item)
    row={'lr':lr,'rows':rows,'seconds':time.perf_counter()-start,'status':agent.status()};out.append(row)
    print(row,flush=True)
(ROOT/'development.json').write_text(json.dumps(out,indent=2))
