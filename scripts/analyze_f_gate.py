"""Audit the F0 evidence boundary and evaluate outcomes beyond its bank."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pqt_qcat.memory import qnuf_f_gate as f
from pqt_qcat.memory.access_state import exact
from pqt_qcat.memory.checkpoint_io import load_state
from pqt_qcat.memory.codex_memory_location import atomic_json
from pqt_qcat.memory.functional_sidecar import ActionJournal,read_json,state_hash
from pqt_qcat.memory.qnuf import array_hash
from scripts.analyze_access_bridge import paired
from scripts.analyze_functional_sidecar import has_target,write_csv

import numpy as np


POLICIES = ('NONE',*f.NAMES,'F_GATE')
SLICES = ('A_bank','A_outside','A_all','B_current','B_other','B_all')


def probe_score(logits,label):
    data = np.array(logits)
    prediction = int(data.argmax())+1
    return {'nll':float(np.logaddexp.reduce(data)-data[label-1]),
            'margin':float(data[label-1]-max(data[i] for i in range(4) if i!=label-1)),
            'argmax':prediction,'correct':int(prediction==label)}


def metric_rows(point,name,logits,baseline,rules,bank_tokens):
    out = []
    for key,values in logits.items():
        ctx,token = key[0],int(key[2:])
        target = rules[ctx][token-1]
        old,new = probe_score(baseline[key],target),probe_score(values,target)
        leaf = ('A_bank' if token in bank_tokens else 'A_outside') if ctx=='A' else (
            'B_current' if token==point['write_token'] else 'B_other')
        out.append({**point,'policy':name,'slice':leaf,'context':ctx,'read_token':token,'target':target,
                    'logits':json.dumps(values),'baseline_logits':json.dumps(baseline[key]),
                    'nll':new['nll'],'margin':new['margin'],'argmax':new['argmax'],'correct':new['correct'],
                    'before_correct':old['correct'],'before_argmax':old['argmax'],
                    'delta_nll':new['nll']-old['nll'],'delta_margin':new['margin']-old['margin']})
    return out


def full_safety(rows,plan):
    a = [r for r in rows if r['context']=='A']
    return (len(a)==6 and all(r['delta_nll']<=plan['max_A_nll_increase']
                               and (not r['before_correct'] or r['correct']) for r in a))


def qualifying_B(rows,plan):
    q = next(r for r in rows if r['slice']=='B_current')
    return not q['before_correct'] and bool(q['correct']) and -q['delta_nll']>=plan['min_B_nll_gain']


def verify_unit(root,plan,seed,mapping,phase='blocks'):
    name = f'{seed}_{mapping}'
    relative = f'blocks/{name}/'
    path = root/phase/name
    saved = read_json(path/'RESULTS.json')
    assert saved['unit']==[seed,mapping]
    source = load_state(f.input_path(plan,relative+'START.npz'))
    clone = load_state(path/'START.npz')
    exact(source,clone)
    anchor = source['agent']
    core = anchor['core_snapshot']
    k = core['K']
    assert saved['snapshot_sha256']==saved['restored_sha256']==state_hash(anchor)
    bank = f.historical_bank(f.input_path(plan,'D0_D3_ORIGINAL.zip'),seed,mapping,anchor,plan['bank_size'])
    exact(bank,saved['historical_bank'])
    bank_tokens = [w['query'] for w in bank]
    assert len(set(bank_tokens))==3 and all(w['presented_at']<48 for w in bank)
    bank_labels = {f"A:{w['query']}":w['label'] for w in bank}
    assert set(saved['before_bank'])==set(bank_labels)
    before_A = {tag:probe_score(saved['before_bank'][tag]['logits'],label) for tag,label in bank_labels.items()}
    old = read_json(f.input_path(plan,relative+'RESULTS.json'))
    exact(saved['diagnostic_none'],old['baseline_panels']['NONE']['logits'])
    baseline = saved['diagnostic_none']
    for tag in bank_labels:
        exact(saved['before_bank'][tag]['logits'],baseline[tag])
    frozen_rng = old['baseline_panels']['NONE']['rng_post']
    outputs = load_state(path/'DELTAS.npz')
    actions = ActionJournal(path/'ACTIONS_BEFORE_FEEDBACK.jsonl')
    sealed = ActionJournal(path/'DECISIONS_BEFORE_DIAGNOSTICS.jsonl')
    historical_actions = f.input_path(plan,relative+'ACTIONS_BEFORE_FEEDBACK.jsonl').read_bytes()
    assert actions.path.read_bytes()==historical_actions
    _,rules = f.d4.parent.c0.block(plan['d4']['base_plan']['c0'],seed,mapping)
    results,candidate_rows,decisions = [],[],[]
    offset_action,offset_decision,num = 0,0,0
    gate_cost,diag_cost,reused = 3*128,0,0
    for q,(event,prior) in enumerate(zip(saved['events'],old['events'],strict=True),1):
        receipt,decision_receipt = event['action_receipt'],event['decision_receipt']
        assert receipt['offset']==offset_action and decision_receipt['offset']==offset_decision
        offset_action += receipt['length']
        offset_decision += decision_receipt['length']
        packet = actions.read_verified(receipt)
        assert not has_target(packet) and packet['pre_sha256']==state_hash(anchor)
        exact(packet['observation'],{'context':'B','query':q})
        assert q==event['write_token']==prior['write_token'] and packet['action']==np.argmax(packet['logits'])+1
        assert event['target_after']==rules['B'][q-1]==prior['target_after']
        current = f'B:{q}'
        exact(packet['logits'],baseline[current])
        forecast_steps = sum(packet['budget'][k] for k in ('main','main_shadow','side','side_shadow'))
        assert forecast_steps==288
        before_b = probe_score(packet['logits'],event['target_after'])
        exact(event['B_before'],before_b)
        w = core['W_readout']
        logits = np.array(packet['logits'])
        probabilities = np.exp(logits-logits.max())
        probabilities /= probabilities.sum()
        err = np.zeros(8)
        err[1:5] = -probabilities
        err[event['target_after']] += 1.
        exact(err.tolist(),event['err'])
        side = packet['external']
        tanh = np.tanh(side['Z_after'])
        phis = {'raw':tanh-np.tanh(side['Z_no_input']), 'ctx':tanh-np.tanh(event['token_Z'])}
        for tag,value in phis.items():
            exact(event['phis'][tag],value.tolist())
        eta = plan['d4']['base_plan']['c0']['parent']['eta']
        initial_vectors = [np.clip(eta*np.outer(w.T@err,phis[tag]),-plan['clip'],plan['clip']) for tag in ('raw','ctx')]
        expected = f.proposals(k,*initial_vectors,plan['clip'])
        point = {'seed':seed,'mapping':mapping,'write_token':q}
        admitted,panels = {},{'NONE':baseline}
        gate_cost += forecast_steps+128
        exact(list(event['candidates']),list(f.NAMES))
        for label,seen in event['candidates'].items():
            proposal,end,geometry = expected[label]
            exact(geometry,seen['geometry'])
            if end is None:
                assert 'diagnostic_logits' not in seen and 'reads' not in seen
                admitted[label] = False
                candidate_rows.append({**point,'candidate':label,'physical':geometry['status'],'norm':None,
                                       'bank_passes':None,'B_passes':None,'admitted':False,'B_gain':None,
                                       'bank_max_delta_nll':None,'A_outside_violation':None,'safe_new_B':False})
                continue
            assert seen['index']==num and seen['K_hash']==array_hash(end)
            exact(outputs['proposed'][num],proposal)
            exact(outputs['K_after'][num],end)
            exact(outputs['applied'][num],end-k)
            assert np.max(np.abs(proposal))<=plan['clip']
            assert np.max(np.abs(end-k))<=plan['clip']+geometry['applied_roundoff_bound']
            assert f.d4.within_norm(np.linalg.norm(end-k),geometry['target_norm'])
            assert not np.any((end-k)[~np.any(w[1:5]!=0.,axis=0)])
            num += 1
            reads = seen['reads']
            assert set(reads)==set(bank_labels)|{current}
            assert all(val['substeps']==128 and val['rng_post']==frozen_rng[k] for k,val in reads.items())
            gate_cost += 4*128
            after_a = {tag:probe_score(reads[tag]['logits'],label) for tag,label in bank_labels.items()}
            factual_B = probe_score(reads[current]['logits'],event['target_after'])
            gain = before_b['nll']-factual_B['nll']
            bank_max = max(after_a[tag]['nll']-before_A[tag]['nll'] for tag in before_A)
            lost_A = sum(before_A[tag]['correct'] and not after_a[tag]['correct'] for tag in before_A)
            lost_B = bool(before_b['correct'] and not factual_B['correct'])
            B_passes,F_passes = gain>=plan['min_B_nll_gain'] and not lost_B, bank_max<=plan['max_A_nll_increase'] and not lost_A
            exact(seen['B'],{'nll_gain':gain,'lost_correct':lost_B,'passes':B_passes})
            exact(seen['F'],{'max_delta_nll':bank_max,'lost_correct':lost_A,'passes':F_passes})
            admitted[label] = B_passes and F_passes
            panel = seen['diagnostic_logits']
            assert set(panel)==set(baseline) and all(len(x)==4 and np.isfinite(x).all() for x in panel.values())
            for tag,probe in reads.items():
                exact(panel[tag],probe['logits'])
            if label in f.REUSED:
                archived = prior['arms'][f.REUSED[label]]['probe']
                exact(panel,archived['logits'])
                assert seen['K_hash']==archived['K_hash']
                reused += 8
            else:
                diag_cost += 8*128
            panels[label] = panel
            rolled = metric_rows(point,label,panel,baseline,rules,bank_tokens)
            harm_outside = any(r['delta_nll']>plan['max_A_nll_increase'] or (r['before_correct'] and not r['correct'])
                               for r in rolled if r['slice']=='A_outside')
            candidate_rows.append({**point,'candidate':label,'physical':geometry['status'],'norm':geometry['actual_norm'],
                                   'bank_passes':F_passes,'B_passes':B_passes,'admitted':admitted[label],
                                   'B_gain':gain,'bank_max_delta_nll':bank_max,'A_outside_violation':harm_outside,
                                   'safe_new_B':full_safety(rolled,plan) and qualifying_B(rolled,plan)})
        selected = next((name for name in plan['selection_order'] if admitted[name]),None)
        action_taken = 'ACCEPTED_RAW' if selected=='RAW' else 'REPAIRED' if selected else 'NO_FEASIBLE_UPDATE'
        expected_seal = {'write_token':q,'selected':selected,'decision':action_taken,
                         'source_action_sha256':receipt['sha256'],'admissible':admitted,
                         'already_confirmed_A_tokens':bank_tokens}
        exact(event['decision'],expected_seal)
        exact(sealed.read_verified(decision_receipt),expected_seal)
        panels['F_GATE'] = panels[selected] if selected else baseline
        current_candidates = [r for r in candidate_rows if r['write_token']==q]
        candidate = next((r for r in current_candidates if r['candidate']==selected),None)
        decisions.append({**point,'decision':action_taken,'selected':selected,'bank_tokens':json.dumps(bank_tokens),
                          'chosen_norm':candidate['norm'] if candidate else 0.,
                          'A_outside_violation':candidate['A_outside_violation'] if candidate else False,
                          'safe_new_B':candidate['safe_new_B'] if candidate else False,
                          'any_fully_safe_new_B':any(r['safe_new_B'] for r in current_candidates),
                          'safe_new_B_candidates':json.dumps([r['candidate'] for r in current_candidates if r['safe_new_B']])})
        for label,panel in panels.items():
            results.extend(metric_rows(point,label,panel,baseline,rules,bank_tokens))
    assert offset_action==actions.path.stat().st_size and offset_decision==sealed.path.stat().st_size
    assert num==len(outputs['proposed'])==len(outputs['applied'])==len(outputs['K_after'])
    exact(saved['cost'],{'gate_substeps':gate_cost,'diagnostic_substeps':diag_cost,'reused_diagnostic_reads':reused})
    return results,candidate_rows,decisions,saved['cost']


def summarize(plan,rows):
    boot = {'parent':plan['d4']['base_plan']['c0']['parent'],'bootstrap_seed':plan['bootstrap_seed'],
            'bootstrap_replicates':plan['bootstrap_replicates']}
    per_block,summary = [],{}
    for policy in POLICIES:
        summary[policy] = {}
        for tag in SLICES:
            points = []
            for seed,mapping in plan['units']:
                selected = [r for r in rows if (r['seed'],r['mapping'],r['policy'])==(seed,mapping,policy)
                            and (r['slice']==tag or (tag.endswith('_all') and r['context']==tag[0]))]
                row = {'policy':policy,'slice':tag,'seed':seed,'mapping':mapping,'n':len(selected),
                       **{k:float(np.mean([p[k] for p in selected])) if selected else None
                          for k in ('correct','before_correct','delta_nll','delta_margin')},
                       'correct_n':sum(p['correct'] for p in selected),'before_correct_n':sum(p['before_correct'] for p in selected),
                       'new_correct':sum(p['correct'] and not p['before_correct'] for p in selected),
                       'lost_correct':sum(p['before_correct'] and not p['correct'] for p in selected),
                       'changed_argmax':sum(p['before_argmax']!=p['argmax'] for p in selected)}
                per_block.append(row)
                points.append(row)
            summary[policy][tag] = {**{k:paired([p[k] if p[k] is not None else np.nan for p in points],boot)
                                      for k in ('correct','before_correct','delta_nll','delta_margin')},
                                    **{k:sum(p[k] for p in points) for k in ('n','correct_n','before_correct_n',
                                                            'new_correct','lost_correct','changed_argmax')}}
    return summary,per_block


def compare(plan,per_block):
    boot = {'parent':plan['d4']['base_plan']['c0']['parent'],'bootstrap_seed':plan['bootstrap_seed'],
            'bootstrap_replicates':plan['bootstrap_replicates']}
    by_key = {(p['policy'],p['slice'],p['seed'],p['mapping']):p for p in per_block}
    contrasts = {}
    for arm,reference in (('F_GATE','RAW'),('F_GATE','NONE'),('CTX','RAW_SMALL'),('RAW_SMALL','RAW')):
        contrasts[f'{arm} vs {reference}'] = {}
        for tag in SLICES:
            pairs = [(by_key[arm,tag,s,m],by_key[reference,tag,s,m]) for s,m in plan['units']]
            contrasts[f'{arm} vs {reference}'][tag] = {
                k:paired([a[k]-b[k] if a[k] is not None and b[k] is not None else np.nan for a,b in pairs],boot)
                for k in ('correct','delta_nll','delta_margin')}
    return contrasts


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('run',type=Path)
    cli.add_argument('--reference',type=Path)
    cli.add_argument('--verify-only',action='store_true')
    args = cli.parse_args()
    root = args.run.resolve()
    plan = f.verify_plan(root)
    if args.reference:
        plan['reference'] = str(args.reference.resolve())
    phases = {name:read_json(root/f'STATUS_{name}.json') for name in ('preflight','run')}
    assert phases['preflight']['phase']=='PASSED' and phases['run']['phase']=='COMPLETED'
    unit = '_'.join(map(str,plan['preflight_unit']))
    first,second = root/'preflight'/unit,root/'blocks'/unit
    for part in ('RESULTS.json','ACTIONS_BEFORE_FEEDBACK.jsonl','DECISIONS_BEFORE_DIAGNOSTICS.jsonl'):
        assert (first/part).read_bytes()==(second/part).read_bytes()
    for part in ('START.npz','DELTAS.npz'):
        exact(load_state(first/part),load_state(second/part))
    rows,candidates,choices,costs = [],[],[],[]
    for seed,mapping in plan['units']:
        probe,props,decisions,cost = verify_unit(root,plan,seed,mapping)
        rows.extend(probe)
        candidates.extend(props)
        choices.extend(decisions)
        costs.append(cost)
    summary,per_block = summarize(plan,rows)
    decision_counts = {name:sum(x['decision']==name for x in choices) for name in
                       ('ACCEPTED_RAW','REPAIRED','NO_FEASIBLE_UPDATE')}
    held_out = sum(x['A_outside_violation'] for x in choices)
    new_B = sum(x['safe_new_B'] for x in choices)
    selected = sum(x['selected'] is not None for x in choices)
    assert len(choices)==54 and len(candidates)==54*len(f.NAMES)
    verdict = ('NO_FEASIBLE_UPDATE' if not selected else 'SAFE_UPDATE_FEASIBLE' if new_B and not held_out
               else 'F_GATE_INSUFFICIENT')
    out = {'verdict':verdict,'F1_recommended':verdict=='SAFE_UPDATE_FEASIBLE','snapshots':len(plan['units']),
           'cases':len(choices),'decision_counts':decision_counts,'A_outside_violations':held_out,
           'selected_safe_new_B':new_B,'offline_safe_new_B_cases':sum(x['any_fully_safe_new_B'] for x in choices),
           'candidate_counts':{name:{'physical':sum(x['norm'] is not None and x['candidate']==name for x in candidates),
                                      'bank_passes':sum(x['bank_passes'] is True and x['candidate']==name for x in candidates),
                                      'B_passes':sum(x['B_passes'] is True and x['candidate']==name for x in candidates),
                                      'admitted':sum(x['candidate']==name and x['admitted'] for x in candidates),
                                      'safe_new_B':sum(x['candidate']==name and x['safe_new_B'] for x in candidates),
                                      'selected':sum(x['selected']==name for x in choices)} for name in f.NAMES},
           'cost':{k:sum(r[k] for r in costs) for k in costs[0]},'phases':phases,'summary':summary,
           'paired_contrasts':compare(plan,per_block)}
    if args.verify_only:
        exact(out,read_json(root/'ANALYSIS.json'))
    else:
        atomic_json(root/'ANALYSIS.json',out)
        for file, data in [('F0_PER_QUERY',rows),('F0_PER_BLOCK',per_block),('F0_CANDIDATES',candidates),('F0_DECISIONS',choices)]:
            write_csv(root/(file+'.csv'),data)
    print(json.dumps({k:v for k,v in out.items() if k not in ('summary','phases','paired_contrasts')},indent=2))


if __name__=='__main__':
    main()
