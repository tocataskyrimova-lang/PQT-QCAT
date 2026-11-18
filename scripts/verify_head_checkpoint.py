"""Replay the exact 100%-to-16.67% regression from the audit, after the fix."""
from pathlib import Path
import copy, json, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pqt_qcat.memory.dev_joint_rule_calibration import JointAgent
from pqt_qcat.memory.dev_context_pathway_ablation import make_stable_core
from pqt_qcat.memory.exp_stateful_online_001 import generate_context_task_pairs
from pqt_qcat.memory.exp_online_calibration_001 import generate_joint_rule_schedule

out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
pair=generate_context_task_pairs(740003,4)[0]
def make():return JointAgent(make_stable_core(42),architecture_mode='direct_gating',learning_mode='RESET_TRAIN',t_context=0)
agent=make();initial=agent.core.K.copy()
for ev in generate_joint_rule_schedule(*pair,total_events=72,seed=42):
    agent.begin_event(ev['context'],ev['query']);agent.observe_feedback_and_update(ev['target'])
agent.save_checkpoint(out/'heads.npz')
restored=make();restored.load_checkpoint(out/'heads.npz')
def evaluate(a):
    predictions=[];targets=[];logits=[]
    for ctx in (0,1):
        for q in range(1,7):
            p,l=a.begin_event(ctx,q);predictions.append(p);targets.append(pair[ctx][q]);logits.append(l)
    return np.array(predictions),np.array(targets),np.array(logits)
pa,y,la=evaluate(copy.deepcopy(agent));pb,_,lb=evaluate(restored)
np.testing.assert_array_equal(la,lb)
np.testing.assert_array_equal(initial,agent.core.K)
result={'seed':42,'events':72,'accuracy_before':float(np.mean(pa==y)),
        'accuracy_after':float(np.mean(pb==y)),'logits_bitwise_identical':True,'K_frozen':True}
(out/'HEAD_CHECKPOINT_CHECK.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
