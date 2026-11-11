"""One small, continuous life with a readable symbolic demonstration."""
import json
from pathlib import Path
import numpy as np
from life import Life,Config

ROOT=Path(__file__).resolve().parent
def main():
    agent=Life(Config(seed=901));rng=np.random.default_rng(902);stages=[];pending=[]
    rules={'forward':np.array([1,2,3,4,5,6,7,0]),'reverse':np.array([7,0,1,2,3,4,5,6])}
    for context,rule in rules.items():
        correct=[]
        for i in range(500):
            token=int(rng.integers(8));reply=agent.predict(token,context);y=int(rule[token]);correct.append(reply['token']==y)
            pending.append((reply['event_id'],y))
            if len(pending)>3:agent.feedback(*pending.pop(0))
        stages.append({'context':context,'first100_accuracy':float(np.mean(correct[:100])),'last100_accuracy':float(np.mean(correct[-100:]))})
    for item in pending:agent.feedback(*item)
    before=agent.status();agent.idle(10000);after=agent.status();outputs={}
    for context in rules:outputs[context]=[r['token'] for r in agent.generate(0,context,8)]
    expected={'forward':[1,2,3,4,5,6,7,0],'reverse':[7,6,5,4,3,2,1,0]}
    result={'stages':stages,'idle_ticks':10000,'teacher_labels_during_idle':after['feedback']-before['feedback'],
            'replay_updates_during_idle':after['replay_updates']-before['replay_updates'],
            'generated':outputs,'expected':expected,'status':agent.status()}
    (ROOT/'demo_result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    agent.save(ROOT/'runtime/demo_life.json.gz')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
