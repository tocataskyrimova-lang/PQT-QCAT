"""Ensure later optional guards did not change the primary default trajectory."""
import json
import numpy as np
import experiment
from life import Life

ROOT=experiment.ROOT
original=json.loads((ROOT/'results/PQT/101/events.json').read_text())
original_life=Life.load(ROOT/'results/PQT/101/final_life.json.gz')
experiment.ROOT=ROOT/'verification_replay'
result=experiment.run_life((101,'PQT'))
new=json.loads((experiment.ROOT/'results/PQT/101/events.json').read_text())
new_life=Life.load(experiment.ROOT/'results/PQT/101/final_life.json.gz')
diffs={'z':float(np.max(np.abs(original_life.core.z-new_life.core.z))),
       'encoder':float(np.max(np.abs(original_life.encoder-new_life.encoder)))}
for key in original_life.contexts:
    for kind in ('fast','slow'):
        diffs[f'{key}_{kind}']=float(np.max(np.abs(original_life.contexts[key][kind]-new_life.contexts[key][kind])))
proof={'same_3000_event_log':original==new,'max_absolute_differences':diffs,
       'same_clock':original_life.clock==new_life.clock,
       'note':'Optional guards and finite-error hardening added after primary runs; default guarded_replay=False preserved.'}
(ROOT/'reproduction_check.json').write_text(json.dumps(proof,indent=2))
assert proof['same_3000_event_log'] and proof['same_clock'] and max(diffs.values())==0,proof
print('EXACT REPRODUCTION PASS',proof)
