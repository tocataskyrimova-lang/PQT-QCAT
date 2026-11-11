"""Numerical refinement and actual running JSONL server smoke check."""
import json,sys,subprocess,time,threading,queue,tempfile
from pathlib import Path
import numpy as np
from life import Life,Config,PhysicalCore
from experiment import ROOT,save

def main():
    result={};q=np.random.default_rng(777).normal(0,.4,(6,8));q[:,6]=.7
    rng=np.random.default_rng(778);drives=rng.normal(0,.7,(1000,6,3));states={}
    for k in (1,2,4,8):
        core=PhysicalCore(Config(integration_substeps=k));core.z=q.copy()
        for force in drives:core.tick(force)
        states[k]=core.z.copy()
    result['refinement']={str(k):{'rms_to_k8':float(np.sqrt(np.mean((v-states[8])**2))),'finite':bool(np.isfinite(v).all())} for k,v in states.items()}
    life=Life(Config(seed=779))
    for i in range(128):r=life.predict(i%8,'stress');life.feedback(r['event_id'],(i+1)%8)
    before=life.status();start=time.perf_counter();life.idle(100000)
    result['long_idle']={'before':before,'after':life.status(),'seconds':time.perf_counter()-start}
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint=Path(tmp)/'live.json.gz'
        proc=subprocess.Popen([sys.executable,'-X','utf8','-B',str(ROOT/'life.py'),'--serve','--checkpoint',str(checkpoint),'--idle-hz','40'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,creationflags=0x08000000 if sys.platform=='win32' else 0)
        inbox=queue.Queue()
        def reader():
            for line in proc.stdout:inbox.put(line)
        threading.Thread(target=reader,daemon=True).start()
        ready=json.loads(inbox.get(timeout=15))
        def send(data):
            proc.stdin.write(json.dumps(data)+'\n');proc.stdin.flush();return json.loads(inbox.get(timeout=15))
        first=send({'op':'predict','token':2,'context':'live'});time.sleep(.4)
        status=send({'op':'status'});ack=send({'op':'feedback','event_id':first['event_id'],'target':3})
        proc.stdin.write('{"op":"quit"}\n');proc.stdin.flush();proc.wait(timeout=15)
        result['server']={'ready':ready['ready'],'prediction':first,'status_after_silence':status,'feedback_ack':ack,
                          'exit_code':proc.returncode,'checkpoint_saved':checkpoint.exists(),'continued_during_silence':status['clock']>first['tick']}
        assert proc.returncode==0 and checkpoint.exists() and status['clock']>first['tick']
        resumed=Life.load(checkpoint);result['server']['resumed_status']=resumed.status()
    save(ROOT/'runtime_validation.json',result)
    print('RUNTIME VALIDATION',result['refinement'],result['server']['continued_during_silence'],flush=True)

if __name__=='__main__':main()
