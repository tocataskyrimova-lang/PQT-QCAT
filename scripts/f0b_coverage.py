"""Replay all F0 witness-bank masks and audit the finite candidate set."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import json
from pathlib import Path
import shutil
from statistics import mean
import time
from zipfile import ZipFile

from analyze_f_gate import (ActionJournal, atomic_json, exact, f, full_safety, load_state,
                            probe_score, qualifying_B, read_json, state_hash, verify_unit, write_csv)
from package_sidecar_b2a import REPO, git, sha


BASE = '72379f393ffa35909712e80311008930203f3f5a'
BASE_ZIP_SHA = '7edee4a19e59598d785fd96f8038a642cb990fa4fcc54c8bb676b1e1e792e350'
METHOD = 'LIFE_QNUF_F0B_COVERAGE_METHOD_RU.md'
TOKENS = tuple(range(1,7))
SIZES = (3,4,5,6)
MASKS = tuple(mask for k in SIZES for mask in combinations(TOKENS,k))
BOUNDS = (0.,.001,.0025,.005,.01,.02,.05,.1,.25,.5,1.)
REPLAY = ('A:1','A:6','B:1')
NAMES = ('NONE',*f.NAMES)


def manifest(path):
    return {name:digest for digest,name in (line.split('  ',1) for line in path.read_text(encoding='utf-8').splitlines())}


def base_inputs(root, archive, reference):
    if sha(archive)!=BASE_ZIP_SHA:
        raise ValueError('F0 delivery archive SHA-256 mismatch')
    manifest_path = root/'ARTIFACT_MANIFEST.sha256'
    with ZipFile(archive) as z:
        assert z.read(manifest_path.name)==manifest_path.read_bytes()
    entries = manifest(manifest_path)
    for name,digest in entries.items():
        if sha(root/name)!=digest:
            raise ValueError(f'F0 input differs from packaged evidence: {name}')
    delivery = read_json(root/'DELIVERY.json')
    exact((delivery['delivery_commit'],delivery['engineering'],delivery['verdict']),
          (BASE,'F0_VALID','F_GATE_INSUFFICIENT'))
    baseline = f.verify_plan(root)
    exact((baseline['max_A_nll_increase'],baseline['min_B_nll_gain'],baseline['clip'],baseline['selection_order']),
          (.01,.10,.50,list(f.NAMES)))
    assert len(baseline['units'])==9 and sha(reference/'ARTIFACT_MANIFEST.sha256')==baseline['base_manifest_sha256']
    baseline['reference'] = str(reference)
    return baseline


def freeze(out, root, archive, reference):
    base = base_inputs(root,archive,reference)
    assert git('merge-base','--is-ancestor',BASE,'HEAD')==''
    blocks = {}
    for seed,mapping in base['units']:
        tag = f'{seed}_{mapping}'
        anchor = load_state(root/'blocks'/tag/'START.npz')['agent']
        receipts = f.historical_bank(f.input_path(base,'D0_D3_ORIGINAL.zip'),seed,mapping,anchor,6)
        tokens = [w['query'] for w in receipts]
        assert sorted(tokens)==list(TOKENS)
        blocks[tag] = {'witnesses':receipts,'snapshot':state_hash(anchor),
                       'primary':{str(k):sorted(tokens[:k]) for k in SIZES}}
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(REPO/'protocol'/METHOD,out/METHOD)
    plan = {'base_sha':BASE,'base_archive_sha256':BASE_ZIP_SHA,'base_manifest_sha256':sha(root/'ARTIFACT_MANIFEST.sha256'),
            'source_commit':git('rev-parse','HEAD'),'source_sha256':sha(Path(__file__)),
            'units':base['units'],'candidate_order':base['selection_order'],'max_A_nll_increase':.01,
            'min_B_nll_gain':.10,'clip':.50,'masks':MASKS,'bounds':BOUNDS,'blocks':blocks,
            'fresh_replay':{'write_token':1,'queries':REPLAY,'policies':NAMES}}
    atomic_json(out/'PLAN.json',plan)
    atomic_json(out/'LOCK.json',{name:sha(out/name) for name in ('PLAN.json',METHOD)})


def choose(candidates,bank_before,b_pass,order,bound):
    """The bank and the current, already receipted B pass bit are the only inputs."""
    admitted = {name:bool(name in candidates and b_pass[name] and
                          f.memory_gate(bank_before,candidates[name],bound)['passes']) for name in order}
    choice = f.select(admitted,order)
    status = {'ACCEPTED_RAW':'ALLOW','REPAIRED':'REPAIR_PASS','NO_FEASIBLE_UPDATE':'BLOCK'}[choice['decision']]
    return choice['selected'] or 'NONE',status,admitted


def snapshot_replay(job):
    path, base, seed, mapping = job
    block = f'{seed}_{mapping}'
    anchor = load_state(path/'START.npz')['agent']
    target = load_state(path/'DELTAS.npz')['K_after']
    saved = read_json(path/'RESULTS.json')
    agent = f.d4.parent.c0.new_agent('B3',seed,base['d4']['base_plan']['c0']['parent'])
    prior = read_json(f.input_path(base,f'blocks/{block}/RESULTS.json'))
    rng = prior['baseline_panels']['NONE']['rng_post']
    records, restored = [], []
    for q,event in enumerate(saved['events'],1):
        if q==1:
            agent.restore(anchor)
            for tag in REPLAY:
                item = f.observe(agent,agent.core.K,tag)
                exact(item['logits'],saved['diagnostic_none'][tag])
                exact(item['rng_post'],rng[tag])
                records.append({'candidate':'NONE','query':tag,**item})
            exact(agent.snapshot(),anchor)
        for name,candidate in event['candidates'].items():
            if 'index' not in candidate:
                continue
            agent.restore(anchor)
            exact(agent.snapshot(),anchor)
            k_after = target[candidate['index']]
            agent.core.K[:] = k_after
            expected = {**anchor,'core_snapshot':{**anchor['core_snapshot'],'K':k_after}}
            exact(agent.snapshot(),expected)
            if q==1:
                for tag in REPLAY:
                    item = f.observe(agent,k_after,tag)
                    exact(item['logits'],candidate['diagnostic_logits'][tag])
                    exact(item['rng_post'],rng[tag])
                    records.append({'candidate':name,'query':tag,**item})
                exact(agent.snapshot(),expected)
            agent.restore(anchor)
            exact(agent.snapshot(),anchor)
            restored.append({'write_token':q,'candidate':name,'start_sha256':state_hash(anchor),
                             'restored_sha256':state_hash(agent.snapshot()),'K_sha256':candidate['K_hash']})
    return block,{'checkpoint_sha256':state_hash(anchor),'exact_restores':restored,'fresh_reads':records,
                  'fresh_substeps':sum(row['substeps'] for row in records)}


def decisions_for_block(root,base,plan,seed,mapping):
    tag = f'{seed}_{mapping}'
    saved = read_json(root/'blocks'/tag/'RESULTS.json')
    archive = ActionJournal(root/'blocks'/tag/'ACTIONS_BEFORE_FEEDBACK.jsonl')
    witnesses = {w['query']:w['label'] for w in plan['blocks'][tag]['witnesses']}
    selections = []
    order = plan['candidate_order']
    for event in saved['events']:
        q = event['write_token']
        receipt = event['action_receipt']
        packet = archive.read_verified(receipt)
        exact(packet['observation'],{'context':'B','query':q})
        base_B = probe_score(packet['logits'],event['target_after'])
        variants = {name:row['diagnostic_logits'] for name,row in event['candidates'].items() if 'index' in row}
        b_pass = {name:f.acquisition(base_B,probe_score(z[f'B:{q}'],event['target_after']),
                                     plan['min_B_nll_gain'])['passes'] for name,z in variants.items()}
        for mask in plan['masks']:
            bank_before = {t:probe_score(saved['diagnostic_none'][f'A:{t}'],witnesses[t]) for t in mask}
            bank_after = {n:{t:probe_score(z[f'A:{t}'],witnesses[t]) for t in mask} for n,z in variants.items()}
            chosen,status,admitted = choose(bank_after,bank_before,b_pass,order,plan['max_A_nll_increase'])
            primary = mask==plan['blocks'][tag]['primary'][str(len(mask))]
            if primary and len(mask)==3:
                exact((None if chosen=='NONE' else chosen,admitted),
                      (event['decision']['selected'],event['decision']['admissible']))
            selections.append({'seed':seed,'mapping':mapping,'write_token':q,'mask':mask,'primary':primary,
                               'choice':chosen,'status':status,'admitted':admitted,'action_sha256':receipt['sha256']})
    return selections


def outcome(selection,probes,plan):
    a = [p for p in probes if p['context']=='A']
    by_bank = [p for p in a if p['read_token'] in selection['mask']]
    outside = [p for p in a if p['read_token'] not in selection['mask']]
    b = next(p for p in probes if p['slice']=='B_current')
    assert all(p['delta_nll']<=plan['max_A_nll_increase'] and p['correct']>=p['before_correct'] for p in by_bank)
    return {**{key:value for key,value in selection.items() if key not in ('admitted','mask')},
            'k':len(by_bank),'mask':''.join(map(str,selection['mask'])),'new_B':qualifying_B(probes,plan),
            'B_before_correct':b['before_correct'],'B_correct':b['correct'],'B_delta_nll':b['delta_nll'],
            'B_other_correct':sum(p['correct'] for p in probes if p['slice']=='B_other'),
            'A_mean':mean(p['delta_nll'] for p in a),'A_max':max(p['delta_nll'] for p in a),
            'A_lost':sum(p['before_correct'] and not p['correct'] for p in a),
            'bank_mean':mean(p['delta_nll'] for p in by_bank),'bank_max':max(p['delta_nll'] for p in by_bank),
            'outside_mean':mean(p['delta_nll'] for p in outside) if outside else None,
            'outside_max':max(p['delta_nll'] for p in outside) if outside else None,
            'full_safe':full_safety(probes,plan),
            'hidden':not full_safety(probes,plan) if outside else None}


def rollup(rows):
    counts = Counter(row['status'] for row in rows)
    accepted = [r for r in rows if r['status']!='BLOCK']
    hidden = [r['hidden'] for r in accepted if r['hidden'] is not None]
    return {'cases':len(rows),**{key:counts[key] for key in ('ALLOW','REPAIR_PASS','BLOCK')},
            'new_B':sum(r['new_B'] for r in rows),'safe_new_B':sum(r['new_B'] and r['full_safe'] for r in rows),
            'B_correct':sum(r['B_correct'] for r in rows),'B_other_correct':sum(r['B_other_correct'] for r in rows),
            'B_mean_delta_nll':mean(r['B_delta_nll'] for r in rows),'A_lost':sum(r['A_lost'] for r in rows),
            'hidden':sum(hidden) if rows[0]['hidden'] is not None else None,
            'hidden_accepted_denominator':len(hidden) if rows[0]['hidden'] is not None else None,
            **{f'{side}_{suffix}':(fn(values) if (values:=[r[f'{side}_{suffix}'] for r in rows
                                                          if r[f'{side}_{suffix}'] is not None]) else None)
               for side in ('A','bank','outside') for suffix,fn in (('mean',mean),('max',max))}}


def search(cases,plan):
    points, existence = [], []
    for (seed,mapping,q),arms in cases.items():
        b = next(p for p in arms['NONE'] if p['slice']=='B_current')
        pass_B = {n:f.acquisition(b,next(p for p in arms[n] if p['slice']=='B_current'),
                                  plan['min_B_nll_gain'])['passes'] for n in plan['candidate_order']}
        local = [n for n in plan['candidate_order'] if qualifying_B(arms[n],plan)]
        safe = [n for n in local if full_safety(arms[n],plan)]
        existence.append({'seed':seed,'mapping':mapping,'write_token':q,'B_before_correct':b['correct'],
                          'local_count':len(local),'joint_count':len(safe),
                          'local_candidates':json.dumps(local),'joint_candidates':json.dumps(safe)})
        for bound in plan['bounds']:
            trial = {**plan,'max_A_nll_increase':bound}
            allowed = [n for n in plan['candidate_order'] if pass_B[n] and full_safety(arms[n],trial)]
            name = next(iter(allowed),'NONE')
            a = [p['delta_nll'] for p in arms[name] if p['context']=='A']
            points.append({'bound':bound,'seed':seed,'mapping':mapping,'write_token':q,'choice':name,
                           'new_B':qualifying_B(arms[name],plan),'available_new_B':bool(set(local).intersection(allowed)),
                           'A_mean':mean(a),'A_max':max(a),'reference_safe':full_safety(arms[name],plan)})
    return points,existence


def classify(local,joint,g3,g6):
    set_loss,covered,rank_loss = local-joint,g6-g3,joint-g6
    assert min(set_loss,covered,rank_loss)>=0
    if not local or set_loss>max(covered,rank_loss):
        return 'CANDIDATE_SET_LIMITED'
    if covered>max(set_loss,rank_loss):
        return 'COVERAGE_LIMITED'
    if set_loss==covered==rank_loss==0 and g6:
        return 'SAFE_JOINT_UPDATE_FOUND'
    return 'UNRESOLVED'


def analyze(out,root,base,plan):
    start = time.perf_counter()
    order = plan['candidate_order']
    choices = [row for seed,mapping in plan['units'] for row in decisions_for_block(root,base,plan,seed,mapping)]
    assert len(choices)==54*len(MASKS)
    atomic_json(out/'DECISIONS_BEFORE_DIAGNOSTICS.json',choices)
    cases, norms = defaultdict(dict), {}
    jobs = []
    for seed,mapping in plan['units']:
        rows,candidates,_,_ = verify_unit(root,base,seed,mapping)
        tag = f'{seed}_{mapping}'
        exact(read_json(root/'blocks'/tag/'RESULTS.json')['snapshot_sha256'],plan['blocks'][tag]['snapshot'])
        jobs.append((root/'blocks'/tag,base,seed,mapping))
        for p in rows:
            if p['policy']!='F_GATE':
                cases[(seed,mapping,p['write_token'])].setdefault(p['policy'],[]).append(p)
        for row in candidates:
            assert row['physical']=='OK'
            norms[(seed,mapping,row['write_token'],row['candidate'])] = row['norm']
    assert len(norms)==54*len(order)
    rows = []
    for chosen in read_json(out/'DECISIONS_BEFORE_DIAGNOSTICS.json'):
        key = tuple(chosen[k] for k in ('seed','mapping','write_token'))
        row = outcome(chosen,cases[key][chosen['choice']],plan)
        row['actual_delta_K_norm'] = norms.get((*key,chosen['choice']),0.)
        rows.append(row)
    points,existence = search(cases,plan)
    by_mask = [{'mask':''.join(map(str,mask)),'k':len(mask),
                **rollup([r for r in rows if r['mask']==''.join(map(str,mask))])} for mask in MASKS]
    primary = {str(k):rollup([r for r in rows if r['primary'] and r['k']==k]) for k in SIZES}
    g3,g6 = (primary[str(k)]['safe_new_B'] for k in (3,6))
    local,joint = (sum(r[col]>0 for r in existence) for col in ('local_count','joint_count'))
    exact(primary['3']['hidden'],15)
    exact((g3,primary['3']['ALLOW'],primary['3']['REPAIR_PASS'],primary['3']['BLOCK']),(2,10,16,28))
    curve = []
    for bound in plan['bounds']:
        subset = [p for p in points if p['bound']==bound]
        curve.append({'bound':bound,'selected':sum(p['choice']!='NONE' for p in subset),
                      'new_B':sum(p['new_B'] for p in subset),'available_new_B':sum(p['available_new_B'] for p in subset),
                      'violations_at_0_01':sum(not p['reference_safe'] for p in subset),
                      'A_mean':mean(p['A_mean'] for p in subset),'A_max':max(p['A_max'] for p in subset)})
    at_original = next(p for p in curve if p['bound']==plan['max_A_nll_increase'])
    exact((at_original['new_B'],at_original['available_new_B'],at_original['violations_at_0_01']),(g6,joint,0))
    with ProcessPoolExecutor(max_workers=6) as pool:
        replay = dict(pool.map(snapshot_replay,jobs))
    routes = {str(k):{'initial_reads':9*k,'candidate_reads':54*len(order)*(k+1),
                      'forecast_and_feature_substeps':54*416,
                      'total_substeps':54*416+(9*k+54*len(order)*(k+1))*128} for k in SIZES}
    exact(routes['3']['total_substeps'],191808)
    restores = sum(len(r['exact_restores']) for r in replay.values())
    actual_steps = sum(r['fresh_substeps'] for r in replay.values())
    exact((restores,actual_steps),(324,9*len(NAMES)*len(REPLAY)*128))
    atomic_json(out/'SNAPSHOT_REPLAY.json',replay)
    for name,data in [('MASK_EVENTS',rows),('MASK_SUMMARY',by_mask),('CANDIDATE_EXISTENCE',existence),
                      ('TRADEOFF_EVENTS',points),('TRADEOFF',curve)]:
        write_csv(out/(name+'.csv'),data)
    total_wrong = sum(not r['B_before_correct'] for r in existence)
    summary = {'verdict':classify(local,joint,g3,g6),'cases':len(existence),'initially_wrong_B':total_wrong,
               'local_L':local,'joint_J':joint,'G3':g3,'G6':g6,'no_local_among_wrong':total_wrong-local,
               'L_minus_J':local-joint,'G6_minus_G3':g6-g3,'J_minus_G6':joint-g6,'primary':primary,
               'cost':routes,'restores_exact':restores,'fresh_substeps':actual_steps,'fresh_reads':actual_steps//128,
               'wall_seconds':time.perf_counter()-start,'F1_executed':False,'C1_executed':False}
    atomic_json(out/'SUMMARY.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('primary','cost')},indent=2))


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('phase',choices=('freeze','run'))
    for option in ('out','input','archive','reference'):
        cli.add_argument('--'+option,type=Path,required=True)
    args = cli.parse_args()
    out,root,archive,reference = (p.resolve() for p in (args.out,args.input,args.archive,args.reference))
    if args.phase=='freeze':
        freeze(out,root,archive,reference)
        return
    exact(read_json(out/'LOCK.json'),{name:sha(out/name) for name in ('PLAN.json',METHOD)})
    plan = read_json(out/'PLAN.json')
    if sha(Path(__file__))!=plan['source_sha256'] or (out/'SUMMARY.json').exists():
        raise ValueError('Use the locked implementation and an uncomputed run folder')
    base = base_inputs(root,archive,reference)
    assert plan['base_manifest_sha256']==sha(root/'ARTIFACT_MANIFEST.sha256')
    exact(plan['units'],base['units'])
    analyze(out,root,base,plan)


if __name__=='__main__':
    main()
