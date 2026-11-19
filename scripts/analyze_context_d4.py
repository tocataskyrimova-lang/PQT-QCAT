"""Verify D4 physical pairs and contrast NLL, margins and decisions by block."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pqt_qcat.memory import qnuf_context_d4 as d
from pqt_qcat.memory.access_state import exact
from pqt_qcat.memory.checkpoint_io import load_state
from pqt_qcat.memory.codex_memory_location import atomic_json
from pqt_qcat.memory.functional_sidecar import ActionJournal,read_json,state_hash
from pqt_qcat.memory.qnuf import array_hash
from scripts.analyze_access_bridge import paired
from scripts.analyze_context_bind import cosine,norm_blocks
from scripts.analyze_functional_sidecar import has_target,write_csv

import numpy as np


SLICES = ('A_same','B_same','A_other','B_other')
CONTRASTS = {'RAW@raw-NONE':('RAW@raw','NONE'), 'CTX@raw-NONE':('CTX@raw','NONE'),
             'RAW@ctx-NONE':('RAW@ctx','NONE'), 'CTX@ctx-NONE':('CTX@ctx','NONE'),
             'direction@raw':('CTX@raw','RAW@raw'), 'direction@ctx':('CTX@ctx','RAW@ctx'),
             'scale_RAW':('RAW@raw','RAW@ctx'), 'scale_CTX':('CTX@raw','CTX@ctx'),
             'direction@min':('CTX@min','RAW@min'), 'RAW@min-NONE':('RAW@min','NONE'),
             'CTX@min-NONE':('CTX@min','NONE'), 'interaction':('CTX@raw','RAW@raw','CTX@ctx','RAW@ctx')}


def metric(logits,target):
    vec = np.array(logits)
    other = np.delete(vec,target-1)
    return (float(np.logaddexp.reduce(vec)-vec[target-1]),float(vec[target-1]-other.max()),
            int(np.argmax(vec))+1)


def score_panel(panel,rules):
    return {name:metric(logits,rules[name[0]][int(name[2:])-1]) for name,logits in panel['logits'].items()}


def check_panel(panel,k,anchor_sha,rng):
    assert panel['K_hash']==array_hash(k) and panel['restored_sha256']==anchor_sha
    assert panel['substeps']==12*128 and panel['rng_post']==rng
    assert set(panel['logits'])=={f'{c}:{q}' for c in ('A','B') for q in range(1,7)}
    assert all(len(x)==4 and np.isfinite(x).all() for x in panel['logits'].values())


def append_panel(out,panel,baseline,rules,seed,mapping,write,arm):
    for (key,value),now in zip(panel['logits'].items(),score_panel(panel,rules).values(),strict=True):
        earlier = baseline[key]
        ctx,q = key[0],int(key[2:])
        target = rules[ctx][q-1]
        out.append({'seed':seed,'mapping':mapping,'write_token':write,'condition':arm,'read_context':ctx,
                    'read_token':q,'same_token':int(q==write) if write else None,'target':target,
                    'logits':json.dumps(value),'nll':now[0],'margin':now[1],'prediction':now[2],
                    'correct':int(now[2]==target),'before_prediction':earlier[2],'before_correct':int(earlier[2]==target),
                    'before_nll':earlier[0],'before_margin':earlier[1],
                    'delta_nll':now[0]-earlier[0],'delta_margin':now[1]-earlier[1],
                    'changed_argmax':int(now[2]!=earlier[2])})


def verify_block(root,plan,seed,mapping):
    path = root/'blocks'/f'{seed}_{mapping}'
    result = read_json(path/'RESULTS.json')
    solo,diag,donor = d.source_paths(plan,seed,mapping)
    _,rules = d.parent.c0.block(plan['base_plan']['c0'],seed,mapping)
    anchor = load_state(path/'START.npz')['agent']
    exact(state_hash(anchor),state_hash(load_state(d.checked_input(plan,solo/'end.npz'))['agent']))
    core = anchor['core_snapshot']
    k = core['K']
    assert state_hash(anchor)==result['start_sha256']
    archived = read_json(d.checked_input(plan,diag/'RESULTS.json'))
    old_matrices = load_state(d.checked_input(plan,diag/'DELTAS.npz'))['delta']
    arrays = load_state(path/'DELTAS.npz')
    empty = result['baseline_panels']['NONE']
    exact(empty['logits'],read_json(d.checked_input(plan,solo/'PROBES.json'))['logits'])
    initial = load_state(d.checked_input(plan,solo/'start.npz'))['agent']['core_snapshot']['K']
    foreign = load_state(d.checked_input(plan,donor/'end.npz'))['agent']['core_snapshot']['K']
    baseline = score_panel(empty,rules)
    probes, matches, layers, data = [], [], [], []
    control_k = {'NONE':k, 'K0':initial,'WRONG_K':foreign}
    rng = empty['rng_post']
    for name,panel in result['baseline_panels'].items():
        check_panel(panel,control_k[name],result['start_sha256'],rng)
        if name!='NONE':
            append_panel(probes,panel,baseline,rules,seed,mapping,None,name)
    ordered = ActionJournal(path/'ACTIONS_BEFORE_FEEDBACK.jsonl')
    history = ActionJournal(d.checked_input(plan,diag/'pre_feedback.jsonl'))
    offset,vectors,historical_index = 0,0,0
    cap = result['clip']
    assert cap==plan['base_plan']['c0']['parent']['clip']
    budget = 3*12*128
    receiving = np.any(core['W_readout'][1:5]!=0.,axis=0)
    for q,(record,previous) in enumerate(zip(result['events'],archived['rows'],strict=True),1):
        packet = ordered.read_verified(record['receipt'])
        assert record['receipt']['offset']==offset and not has_target(packet)
        offset += record['receipt']['length']
        exact(packet,history.read_verified(previous['before']['B']['receipt']))
        assert record['write_token']==q and packet['pre_sha256']==result['start_sha256']
        exact(packet['observation'],{'context':'B','query':q})
        assert packet['action']==np.argmax(packet['logits'])+1 and record['target_after']==rules['B'][q-1]
        exact(packet['logits'],empty['logits'][f'B:{q}'])
        logits = np.array(packet['logits'])
        p = np.exp(logits-logits.max())
        p /= p.sum()
        target = record['target_after']
        err = np.zeros(8)
        err[1:5],err[target] = -p,1-p[target-1]
        exact(record['error'],err.tolist())
        ext = packet['external']
        full = np.tanh(ext['Z_after'])
        raw,ctx = full-np.tanh(ext['Z_no_input']),full-np.tanh(previous['controls']['no_cue']['Z'])
        exact(record['features'],{'raw':raw.tolist(),'ctx':ctx.tolist()})
        eta = plan['base_plan']['c0']['parent']['eta']
        candidates = {name:np.clip(eta*np.outer(core['W_readout'].T @ err,phi),-cap,cap)
                      for name,phi in [('raw',raw),('ctx',ctx)]}
        radii = {name:float(np.linalg.norm(x)) for name,x in candidates.items()}
        exact(record['radii'],radii)
        old = {}
        for item in previous['interventions']:
            if item['write_context']=='B':
                old[item['feature']] = (item,old_matrices[historical_index])
            historical_index += 1
        for name,feature in [('raw','raw'),('ctx','context_given_token')]:
            exact(candidates[name],old[feature][1])
        point = {'seed':seed,'mapping':mapping,'write_token':q}
        scored = {'NONE':baseline}
        append_panel(probes,empty,baseline,rules,seed,mapping,q,'NONE')
        budget += 416
        for name,(direction,radius) in d.ARMS.items():
            info = record['arms'][name]
            u = candidates[direction]
            target_norm = radii[radius]
            assert info['target_norm']==target_norm and info['native_norm']==radii[direction]
            common_fields = {key:val for key,val in info.items() if key not in ('probe','vector_index','reason')}
            matches.append({**point,'arm':name,'direction':direction,'radius':radius,
                            **common_fields,'reason':info.get('reason'), 'cosine_applied_vs_native':None})
            if info['status']=='INFEASIBLE_MATCH':
                if info['reason']=='component_cap_on_ray':
                    assert target_norm>(cap/np.max(np.abs(u)))*radii[direction]*(1+d.TOL)
                elif info['reason']=='zero_direction':
                    assert radii[direction]==0 and target_norm>0
                else:
                    assert not d.within_norm(info['actual_norm'],target_norm)
                assert 'probe' not in info
                continue
            assert info['status'] in ('OK','ZERO_RADIUS') and info['vector_index']==vectors
            delta = arrays['proposed'][vectors]
            effective = arrays['applied'][vectors]
            vectors += 1
            exact(delta,u if direction==radius else u*info['factor'])
            new = k+delta
            exact(new-k,effective)
            distance = float(np.linalg.norm(effective))
            assert info['actual_norm']==distance and d.within_norm(distance,target_norm)
            assert np.max(np.abs(delta))<=cap and info['peak_proposed']==np.max(np.abs(delta))
            assert np.max(np.abs(effective))<=cap+info['applied_roundoff_bound']
            assert not np.any(effective[~receiving]) and not np.any(effective[u==0])
            check_panel(info['probe'],new,result['start_sha256'],rng)
            if direction==radius:
                old_key = 'raw' if direction=='raw' else 'context_given_token'
                for c in ('A','B'):
                    exact(info['probe']['logits'][f'{c}:{q}'],old[old_key][0]['logits'][c])
            matches[-1]['cosine_applied_vs_native'] = cosine(effective,u)
            append_panel(probes,info['probe'],baseline,rules,seed,mapping,q,name)
            scored[name] = score_panel(info['probe'],rules)
            budget += info['probe']['substeps']
        for name in ('raw','ctx'):
            if radii[name]==0:
                for arm in d.ARMS:
                    if d.ARMS[arm][1]==name:
                        scored.pop(arm,None)
        for name,(nr,_) in d.ARMS.items():
            if radii[nr]==0:
                scored.pop(name,None)
        magnitudes = {name:norm_blocks(value) for name,value in candidates.items()}
        for i in range(6):
            for j in range(6):
                raw_n,raw_cells = magnitudes['raw']
                ctx_n,ctx_cells = magnitudes['ctx']
                layers.append({**point,'receiving':i+1,'sending':j+1,'raw_norm':float(raw_n[i,j]),
                               'ctx_norm':float(ctx_n[i,j]),'cosine':cosine(raw_cells[i,j],ctx_cells[i,j]),
                               'dot':float(np.vdot(raw_cells[i,j],ctx_cells[i,j]))})
        norms = {name:r['actual_norm'] for name,r in record['arms'].items() if r['status']=='OK'}
        for rr in ('raw','ctx'):
            if 'RAW@'+rr in norms and 'CTX@'+rr in norms:
                assert d.within_norm(norms['RAW@'+rr],norms['CTX@'+rr])
        data.append({**point,'r_raw':radii['raw'],'r_ctx':radii['ctx'],'direction_cos':cosine(*candidates.values()),
                     'norms':norms,
                     'arm_outputs':scored,'targets':{f'{c}:{i}':rules[c][i-1] for c in ('A','B') for i in range(1,7)}})
    assert budget==result['substeps'] and offset==ordered.path.stat().st_size
    assert vectors==len(arrays['proposed'])==len(arrays['applied']) and historical_index==len(old_matrices)
    return probes,matches,layers,data,budget


def paired_rows(events, name):
    rows = []
    for event in events:
        radius = 'raw' if event['r_raw']<=event['r_ctx'] else 'ctx'
        arms = [tag.replace('@min','@'+radius) for tag in CONTRASTS[name]]
        outputs = event['arm_outputs']
        if not all(arm in outputs for arm in arms):
            continue
        for key,target in event['targets'].items():
            left,right,*second = [outputs[arm][key] for arm in arms]
            dnll,dmargin = left[0]-right[0], left[1]-right[1]
            l_ok,r_ok = int(left[2]==target),int(right[2]==target)
            dacc = l_ok-r_ok
            if second:
                dnll -= second[0][0]-second[1][0]
                dmargin -= second[0][1]-second[1][1]
                dacc -= int(second[0][2]==target)-int(second[1][2]==target)
            rows.append({'slice':key[0]+('_same' if int(key[2:])==event['write_token'] else '_other'),
                         'nll':dnll,'margin':dmargin,'accuracy_delta':dacc,
                         'left_correct':None if second else l_ok,'right_correct':None if second else r_ok,
                         'changed':None if second else int(left[2]!=right[2]),
                         'gained':None if second else int(l_ok and not r_ok),
                         'lost':None if second else int(not l_ok and r_ok),'nll_worse':int(dnll>0)})
    return rows


def summarize(data,plan):
    boot = {'parent':plan['base_plan']['c0']['parent'],'bootstrap_seed':plan['bootstrap_seed'],
            'bootstrap_replicates':plan['bootstrap_replicates']}
    blocked,contrasts = [],{}
    for name in CONTRASTS:
        for seed,mapping in plan['units']:
            rows = paired_rows([e for e in data if (e['seed'],e['mapping'])==(seed,mapping)],name)
            for tag in SLICES:
                subset = [r for r in rows if r['slice']==tag]
                averages = {key:float(np.mean(v)) if (v := [r[key] for r in subset if r[key] is not None]) else None
                            for key in ('nll','margin','accuracy_delta','left_correct','right_correct','changed','nll_worse')}
                counts = {key:sum(v) if (v := [r[origin] for r in subset if r[origin] is not None]) else None
                          for key,origin in [('gained','gained'),('lost','lost'),('left_correct_count','left_correct'),
                                             ('right_correct_count','right_correct')]}
                blocked.append({'contrast':name,'slice':tag,'seed':seed,'mapping':mapping,'n':len(subset),**averages,
                                **counts})
        contrasts[name] = {}
        for tag in SLICES:
            subset = [r for r in blocked if r['contrast']==name and r['slice']==tag]
            stats = {k:paired([r[k] if r[k] is not None else np.nan for r in subset],boot)
                     for k in ('nll','margin','accuracy_delta','nll_worse','changed')}
            stats.update({k:sum(v) if (v := [r[k] for r in subset if r[k] is not None]) else None for k in
                          ('n','gained','lost','left_correct_count','right_correct_count')})
            contrasts[name][tag] = stats
    return blocked,contrasts


def status(contrasts,plan):
    primary = contrasts['direction@min']['A_same']
    a = primary['nll']['ci95'] if primary['n']==6*len(plan['units']) else None
    b = contrasts['direction@min']['B_same']['nll']['ci95']
    acquired = contrasts['CTX@min-NONE']['B_same']
    acquisition = acquired['nll']['ci95']
    direction = 'DIRECTION_SUPPORT' if a and a[1]<0 else (
        'SCALE_ONLY' if a and max(abs(a[0]),abs(a[1]))<plan['scale_only_equivalence_nll'] else 'UNDECIDED')
    tradeoff = 'TRADEOFF' if direction=='DIRECTION_SUPPORT' and b and b[0]>0 else 'TRADEOFF_NOT_ESTABLISHED'
    return {'direction':direction,'tradeoff':tradeoff,'B_NLL_improves_vs_NONE':bool(acquisition and acquisition[1]<0),
            'B_NLL_worse_than_RAW_at_matched_norm':bool(b and b[0]>0),
            'B_new_correct_vs_NONE_at_matched_norm':acquired['gained']}


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('run',type=Path)
    cli.add_argument('--reference',type=Path)
    cli.add_argument('--verify-only',action='store_true')
    args = cli.parse_args()
    root = args.run.resolve()
    plan = d.verify_plan(root)
    if args.reference:
        plan['reference'] = str(args.reference.resolve())
    stages = {name:read_json(root/f'STATUS_{name}.json') for name in ('preflight','run')}
    assert stages['preflight']['phase']=='PASSED' and stages['run']['phase']=='COMPLETED'
    unit = '_'.join(map(str,plan['preflight_unit']))
    original,repeated = root/'preflight'/unit,root/'blocks'/unit
    for name in ('RESULTS.json','ACTIONS_BEFORE_FEEDBACK.jsonl'):
        assert (original/name).read_bytes()==(repeated/name).read_bytes()
    exact(load_state(original/'START.npz'),load_state(repeated/'START.npz'))
    exact(load_state(original/'DELTAS.npz'),load_state(repeated/'DELTAS.npz'))
    probes,matches,layers,data,substeps = [],[],[],[],0
    for seed,mapping in plan['units']:
        p,m,l,v,budget = verify_block(root,plan,seed,mapping)
        probes.extend(p)
        matches.extend(m)
        layers.extend(l)
        data.extend(v)
        substeps += budget
    blocked,contrasts = summarize(data,plan)
    counts = {arm:{name:sum(m['arm']==arm and m['status']==name for m in matches) for name in
                  ('OK','INFEASIBLE_MATCH','ZERO_RADIUS','NONFINITE_CANDIDATE')} for arm in d.ARMS}
    controls = {arm:{ctx:float(np.mean([r['correct'] for r in probes if r['condition']==arm and r['read_context']==ctx]))
                     for ctx in ('A','B')} for arm in ('NONE','K0','WRONG_K')}
    rel_errors = [abs(e['norms']['RAW@'+rr]-e['norms']['CTX@'+rr])/e['norms']['CTX@'+rr]
                  for e in data for rr in ('raw','ctx') if all(tag+'@'+rr in e['norms'] for tag in ('RAW','CTX'))]
    result = {'verdicts':status(contrasts,plan),'stages':stages,'snapshot_count':len(plan['units']),'single_writes':len(data),
              'panels_restored':3*len(plan['units'])+sum(m['status']=='OK' or m['status']=='ZERO_RADIUS' for m in matches),
              'r_raw_larger':sum(e['r_raw']>e['r_ctx'] for e in data),'counts':counts,
              'substeps':substeps,'control_accuracy':controls,'max_pair_relative_norm_error':max(rel_errors,default=None),
              'contrasts':contrasts}
    if args.verify_only:
        exact(result,read_json(root/'ANALYSIS.json'))
    else:
        atomic_json(root/'ANALYSIS.json',result)
        event_metrics = [{k:v for k,v in e.items() if k not in ('arm_outputs','targets','norms')} for e in data]
        for file,rows in [('D4_PER_TOKEN',probes),('D4_PER_BLOCK',blocked),('D4_MATCHES',matches),
                          ('D4_ECHELONS',layers),('D4_DIRECTIONS',event_metrics)]:
            write_csv(root/(file+'.csv'),rows)
    print(json.dumps({k:result[k] for k in ('verdicts','snapshot_count','single_writes','panels_restored','counts')},indent=2))


if __name__=='__main__':
    main()
