"""Resume an actual experiment checkpoint in a separate process and compare all events."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import json, shutil, subprocess, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pqt_qcat.memory.checkpoint_io import load_state


def main():
    run=Path(sys.argv[1]);plan=json.loads((run/'PLAN.json').read_text())
    name='CARRY_TRAIN_810001_00';source=run/'lives'/name
    replay=run/'resume_verification'
    destination=replay/'lives'/name
    destination.mkdir(parents=True,exist_ok=False)
    shutil.copy2(source/'checkpoint_024.npz',destination/'latest.npz')
    script=replay/'child.py'
    script.write_text('''import json,sys
from pathlib import Path
from pqt_qcat.memory.codex_memory_location import life_task
plan=json.loads(Path(sys.argv[1]).read_text())
life_task((plan,sys.argv[2],"CARRY_TRAIN",810001,0))
''',encoding='utf-8')
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT)
    subprocess.run([sys.executable,str(script),str(run/'PLAN.json'),str(replay)],check=True,env=env)
    assert json.loads((source/'events.json').read_text())==json.loads((destination/'events.json').read_text())
    a=load_state(source/'checkpoint_072.npz');b=load_state(destination/'checkpoint_072.npz')
    def exact(x,y):
        if isinstance(x,np.ndarray):np.testing.assert_array_equal(x,y)
        elif isinstance(x,dict):
            assert set(x)==set(y)
            for k in x:exact(x[k],y[k])
        elif isinstance(x,list):
            assert len(x)==len(y)
            for p,q in zip(x,y):exact(p,q)
        else:assert x==y
    exact(a,b)
    result={'source_life':name,'resume_after_event':24,'completed_events':72,
            'new_process':True,'all_event_records_exact':True,'all_checkpoint_fields_exact':True}
    (run/'RUNNER_RESUME_CHECK.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
