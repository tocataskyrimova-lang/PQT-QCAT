"""One fixed static K decoder, crossed by development seed and mapping."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pqt_qcat.memory import qnuf_context_bind as b
from pqt_qcat.memory.checkpoint_io import load_state
from pqt_qcat.memory.functional_sidecar import state_hash
from pqt_qcat.memory.access_state import exact
from scripts.analyze_qnuf_tworule import score


def class_features(k, initial, w, cue, token):
    columns = []
    for weights in (k, k-initial):
        projected = w[1:5] @ weights
        token_drive = projected[:,token-1]
        cue_drive = projected[:,8:16] @ cue
        # Each row represents one of the four candidate classes. No labels,
        # unit identifiers, dynamical state, or class-specific fitted heads.
        columns.extend([token_drive, cue_drive, token_drive*cue_drive,
                        token_drive-token_drive.mean(), cue_drive-cue_drive.mean(),
                        projected[:,:6].mean(axis=1), projected[:,:6].std(axis=1)])
    return np.stack(columns,axis=1)


def rows(snapshot, mode, k, initial, blind=False, swap=False):
    for ctx in b.CONTEXTS:
        cue_label = ({'A':'B','B':'A'}[ctx] if swap else ctx) if not blind else None
        for token in range(1,7):
            sensory = b.c0.vector(mode,cue_label,token) - b.c0.vector(mode,None,token)
            cue = (snapshot['B'] @ sensory)[8:16]/8.
            yield f'{ctx}:{token}', class_features(k,initial,snapshot['W_readout'],cue,token)


def select_folds(units, test):
    seed,mapping = test
    return [tuple(unit) for unit in units if unit[0]!=seed and unit[1]!=mapping]


def fit(blocks, train, mode, blind):
    x, y = [], []
    for core, initial, rules in (blocks[key] for key in train):
        for key, features in rows(core,mode,core['K'],initial,blind=blind):
            ctx, token = key.split(':')
            target = rules[ctx][int(token)-1]
            x.extend(features)
            y.extend(int(label==target) for label in range(1,5))
    model = make_pipeline(StandardScaler(),Ridge(alpha=1.,solver='svd'))
    model.fit(x,y)
    return model,len(y)


def evaluate(root, plan):
    results, predictions, folds = [], [], []
    for mode,arm in b.c0.NEURAL:
        blocks, verified = {}, {}
        for seed,mapping in plan['units']:
            path = b.path_for(root,plan,'joint',seed,mapping,mode,arm)
            initial = load_state(path/'start.npz')['agent']['core_snapshot']['K']
            end = load_state(path/'end.npz')['agent']
            agent = b.agent_for(path,plan,seed)
            snapshot = state_hash(agent.snapshot())
            assert snapshot==state_hash(end)
            # Exercise an actual donor substitution and exact restoration.
            agent.core.K = initial.copy()
            agent.restore(end)
            assert snapshot==state_hash(agent.snapshot())
            exact(end['core_snapshot']['K'],agent.core.K)
            key = seed,mapping
            blocks[key] = (end['core_snapshot'],initial,plan['c0']['mappings'][mapping])
            verified[key] = snapshot
        for seed,mapping in plan['units']:
            test = seed,mapping
            train = select_folds(plan['units'],test)
            assert len(train)==4 and all(s!=seed and m!=mapping for s,m in train)
            full,n = fit(blocks,train,mode,False)
            blind,n_blind = fit(blocks,train,mode,True)
            folds.append({'mode':mode,'arm':arm,'test':list(test),'fit':[list(key) for key in train],
                          'fit_class_rows':n,'blind_class_rows':n_blind,'restored_sha256':verified[test]})
            core, initial, rules = blocks[test]
            donor = blocks[(seed,(mapping+1)%len(plan['c0']['mappings']))][0]['K']
            predictors = [('decoded',full,core['K'],False,False),('decoded_K0',full,initial,False,False),
                          ('decoded_wrong',full,donor,False,False),('decoded_swap',full,core['K'],False,True),
                          ('decoded_blind',blind,core['K'],True,False)]
            before = state_hash(core)
            for name, model, k, no_cue, swap in predictors:
                logits = {key:model.predict(features).tolist() for key,features in
                          rows(core,mode,k,initial,blind=no_cue,swap=swap)}
                if no_cue:
                    assert all(logits[f'A:{q}']==logits[f'B:{q}'] for q in range(1,7))
                results.append({'seed':seed,'mapping':mapping,'mode':mode,'arm':arm,'probe':name,
                                **score(logits,rules)})
                predictions.append({'unit':list(test),'mode':mode,'arm':arm,'probe':name,'scores':logits})
            assert before==state_hash(core)
    return results,predictions,folds
