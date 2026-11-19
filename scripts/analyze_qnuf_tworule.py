"""Audit the fixed C0 teaching streams and frozen contextual reads."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pqt_qcat.memory import qnuf_tworule as c
from pqt_qcat.memory.access_state import exact
from pqt_qcat.memory.checkpoint_io import load_state
from pqt_qcat.memory.codex_memory_location import atomic_json, digest, metrics
from pqt_qcat.memory.functional_sidecar import ActionJournal, read_json, state_hash
from pqt_qcat.memory.qnuf import array_hash
from scripts.analyze_functional_sidecar import has_target, write_csv
from scripts.analyze_qnuf_confirm import check_snapshot, rng_hash, validate_update

import numpy as np


def verify_schedule(schedule, rules):
    assert len(schedule) == c.EXPOSURES
    expected = [(ctx, q) for ctx in ('A','B') for q in range(1,7)]
    for i in range(0, len(schedule), 12):
        exact(sorted(map(tuple, schedule[i:i+12])), expected)
    assert len(rules['A']) == len(rules['B']) == 6
    assert all(a != b for a,b in zip(rules['A'], rules['B'], strict=True))
    assert set(rules['A']) == set(rules['B']) == set(range(1,5))


def table_scores(table, arm, ctx, q):
    label = table.get(f'{ctx}:{q}' if arm == 'keyed' else str(q))
    return [8. if k == label else 0. for k in range(1,5)]


def verify_train(root, plan, seed, mapping, mode, arm):
    path = c.where(root, seed, mapping, mode, arm)
    schedule, rules = c.block(plan, seed, mapping)
    events, timing = read_json(path/'events.json'), read_json(path/'runtime.json')
    assert len(events) == len(timing) == len(schedule)
    result = load_state(path/'end.npz')
    assert result['cursor'] == len(events) and result['plan_sha256'] == digest(plan)
    exact(result['condition'], [mode, arm])
    journal = ActionJournal(path/'pre_feedback.jsonl')
    start = None if mode == 'table' else load_state(path/'start.npz')['agent']
    if start is not None:
        core = start['core_snapshot']
        k, z, rnd = core['K'], array_hash(core['Z']), rng_hash(core['rng_state'])
    byte_end, table = 0, {}
    rows = []
    for age, (event, clock, (ctx,q)) in enumerate(zip(events, timing, schedule, strict=True)):
        p, receipt = event['packet'], clock['receipt']
        assert not has_target(p) and receipt['offset'] == byte_end
        byte_end += receipt['length']
        exact(journal.read_verified(receipt), p)
        assert clock['issued_ns'] <= clock['revealed_ns'] <= clock['updated_ns']
        exact(p['observation'], {'context':ctx,'query':q,'cursor':age})
        assert q == p['query'] and p['action'] == np.argmax(p['logits']) + 1
        target = rules[ctx][q-1]
        assert event['target_after'] == target
        delta, state_norm = None, None
        if start is None:
            exact(p['logits'], table_scores(table,arm,ctx,q))
            assert p['pending_sha256'] == state_hash(table)
            table[f'{ctx}:{q}' if arm == 'keyed' else str(q)] = target
            assert event['end_state'] == state_hash(table)
        else:
            assert p['pre_sha256'] == (state_hash(start) if age == 0 else events[age-1]['end_state'])
            m, s = p['main'], p['external']
            assert m['event_id'] == age and m['query'] == q
            exact(m['input'], c.vector(mode,ctx,q).tolist())
            assert m['K_pre_hash'] == array_hash(k)
            assert m['Z_pre_hash'] == array_hash(m['Z_pre']) == z
            assert m['Z_noinput_hash'] == array_hash(m['Z_noinput'])
            assert m['Z_post_hash'] == array_hash(m['Z_post'])
            assert m['rng_pre_hash'] == s['rng_pre'] == rnd and m['rng_post_hash'] == m['rng_shadow_post_hash']
            z, rnd = m['Z_post_hash'], m['rng_post_hash']
            exact(start['Z0'], s['Z_start'])
            exact((core['W_readout'] @ m['Z_post'])[1:5].tolist(), m['logits'])
            exact(s['logits'], p['logits'])
            exact((core['W_readout'] @ s['Z_after'])[1:5].tolist(), s['logits'])
            exact(s['hashes'], {'K':array_hash(k),'B':array_hash(core['B']),'W_readout':array_hash(core['W_readout'])})
            phi_side = np.tanh(s['Z_after']) - np.tanh(s['Z_no_input'])
            exact(phi_side.tolist(), s['eligibility'])
            phi = phi_side if arm == 'SS' else np.tanh(m['Z_post']) - np.tanh(m['Z_noinput'])
            logits = p['logits'] if arm == 'SS' else m['logits']
            k, delta = validate_update(event['update'],logits,phi,target,k,core,plan,None)
            exact(p['budget'], {'main':16,'main_shadow':16,'side':128,'side_shadow':128,'snapshots':3,'clones':3})
            state_norm = m['Z_norms'][1]
        rows.append({'seed':seed,'mapping':mapping,'encoder':mode,'policy':arm,'age':age,'context':ctx,'query':q,
                     'action':p['action'],'target_after':target,'correct':int(p['action']==target),
                     'logits':json.dumps(p['logits']),'K_step_norm':delta,'Z_post_norm':state_norm,
                     'substeps':288 if start is not None else 0,'wall_ms':clock['wall_ns']/1e6})
    assert byte_end == journal.path.stat().st_size
    exact(table, result['table'])
    cost = read_json(path/'COMPLETE.json')
    output = {'substeps':sum(r['substeps'] for r in rows), 'wall_seconds':cost['wall_seconds'],
              'cpu_seconds':cost['cpu_seconds'],'K_delta_norm':None,'K_step_mean':None}
    if start is not None:
        ending = result['agent']['core_snapshot']
        check_snapshot(core, ending, k, z, rnd)
        exact(core['vortex_params'],ending['vortex_params'])
        assert state_hash(result['agent']) == events[-1]['end_state']
        assert ending['step_count'] - core['step_count'] == len(events)*16
        output.update(K_delta_norm=float(np.linalg.norm(k - core['K'])),
                      K_step_mean=float(np.mean([r['K_step_norm'] for r in rows])))
    else:
        assert result['agent'] is None
    return start, result, rows, output, events[0]['packet']


def score(reads, rules):
    raw = {ctx: metrics([reads[f'{ctx}:{q}'] for q in range(1,7)],rules[ctx]) for ctx in ('A','B')}
    a,b = raw['A'],raw['B']
    ac = np.asarray(a['predictions']) == rules['A']
    bc = np.asarray(b['predictions']) == rules['B']
    cue_difference = np.asarray(a['logits']) - b['logits']
    cue_norm = np.linalg.norm(cue_difference)
    return {'acc_A':a['accuracy'],'acc_B':b['accuracy'],'macro':(a['accuracy']+b['accuracy'])/2,
            'nll_A':a['nll'],'nll_B':b['nll'],'nll':(a['nll']+b['nll'])/2,
            'correct_pairs':int(np.sum(ac & bc)),
            'changed_actions':sum(i!=j for i,j in zip(a['predictions'],b['predictions'],strict=True)),
            'cue_logit_rms':float(cue_norm/np.sqrt(cue_difference.size)),
            'token_dependent_cue_fraction':float(np.linalg.norm(cue_difference - cue_difference.mean(axis=0))/cue_norm)
            if cue_norm else 0.}


def swap(reads):
    return {f'{ctx}:{q}':reads[f'{other}:{q}'] for ctx,other in (('A','B'),('B','A')) for q in range(1,7)}


def verify_probes(root, plan, seed, mapping, mode, arm, start, end):
    rules = plan['mappings'][mapping]
    file = read_json(c.where(root,seed,mapping,mode,arm)/'PROBES.json')
    if mode == 'table':
        reads = {f'{ctx}:{q}':table_scores(end['table'],arm,ctx,q) for ctx in ('A','B') for q in range(1,7)}
        exact(reads,file['table'])
        assert file['substeps'] == 0
        return {'table':score(reads,rules),'swapped_cue':score(swap(reads),rules)}, [], 0
    assert state_hash(end['agent']) == file['source_hash']
    core = end['agent']['core_snapshot']
    initial, learned = start['core_snapshot']['K'], core['K']
    wrong = (mapping+1) % len(plan['mappings'])
    assert file['wrong_mapping'] == wrong
    wrong_k = load_state(c.where(root,seed,wrong,mode,arm)/'end.npz')['agent']['core_snapshot']['K']
    matrices = {'z0':learned,'live':learned,'main_live':learned,'initial_K':initial,'wrong_K':wrong_k,'no_cue':learned}
    if [seed,mapping] == plan['trace_unit']:
        matrices.update(initial_cut_E2E6=c.cut_edges(initial,False,True),learned_cut_E2E6=c.cut_edges(learned,False,True),
                        learned_cut_E2E5=c.cut_edges(learned,True,False),learned_cut_both=c.cut_edges(learned,True,True))
    assert set(matrices)==set(file['reads'])
    scores, traces, cost, logits = {}, [], 0, {}
    for name, group in file['reads'].items():
        horizon = 8 if name == 'main_live' else 64
        start_z = core['Z'].tolist() if name in ('live','main_live') else end['agent']['Z0']
        logits[name] = {}
        for ctx in ('A','B'):
            for q in range(1,7):
                token = f'{ctx}:{q}'
                read = group[token]
                expected_x = c.vector(mode,ctx if name != 'no_cue' else None,q)
                exact(read['input'],expected_x.tolist())
                exact(read['B_x'],(core['B'] @ expected_x).tolist())
                exact(read['trace']['0']['Z'],start_z)
                exact(read['hashes'],{'K':array_hash(matrices[name]),'B':array_hash(core['B']),
                                      'W_readout':array_hash(core['W_readout'])})
                assert read['rng_pre']==rng_hash(core['rng_state']) and read['substeps']==2*horizon
                cost += read['substeps']
                for step in read['trace'].values():
                    exact(step['logits'],(core['W_readout'] @ step['Z'])[1:5].tolist())
                logits[name][token] = read['trace'][str(horizon)]['logits']
        for q in range(1,7):
            a,b = (group[f'{ctx}:{q}'] for ctx in ('A','B'))
            exact(a['rng_post'],b['rng_post'])
            raw = (np.array(a['B_x']) - b['B_x']).reshape(6,8)
            assert not np.any(raw[[0,2,3,4,5]])
            for tick in a['trace']:
                x,y = a['trace'][tick],b['trace'][tick]
                difference = (np.array(x['Z']) - y['Z']).reshape(6,8)
                if name=='no_cue':
                    exact(x,y)
                traces.append({'seed':seed,'mapping':mapping,'encoder':mode,'policy':arm,'probe':name,'query':q,
                               'tick':int(tick),'B_delta_E2':float(np.linalg.norm(raw[1])),
                               **{f'E{i+1}_l2':float(np.linalg.norm(v)) for i,v in enumerate(difference)},
                               'read_coordinates_l2':float(np.linalg.norm(difference[4,:3])),
                               'logits_linf':float(np.max(np.abs(np.array(x['logits'])-y['logits'])))})
        scores[name] = score(logits[name],rules)
    scores['swapped_cue'] = score(swap(logits['z0']),rules)
    assert cost == file['substeps']
    return scores, traces, cost


def summary(rows, heads, keys):
    groups = {}
    for row in rows:
        group = tuple(row[k] for k in heads)
        groups.setdefault(group,[]).append(row)
    return [{**dict(zip(heads,group,strict=True)),'blocks':len(members),
             **{k:float(np.mean([r[k] for r in members])) for k in keys}} for group,members in groups.items()]


def joint_gate(row, threshold):
    return min(row['acc_A'],row['acc_B']) >= threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--verify-only',action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    plan = c.verify_plan(root)
    phases = {phase:read_json(root/f'STATUS_{phase}.json') for phase in ('train','probe','resume')}
    event_rows, blocks, legs, trajectories, missing = [], [], [], [], []
    for seed,mapping in plan['development_units']:
        verify_schedule(*c.block(plan,seed,mapping))
        starts, first = {}, {}
        for mode,arm in c.CONDITIONS:
            path = c.where(root,seed,mapping,mode,arm)
            if not all((path/name).is_file() for name in ('COMPLETE.json','PROBES.json')):
                missing.append([seed,mapping,mode,arm]); continue
            start,end,events,cost,packet = verify_train(root,plan,seed,mapping,mode,arm)
            scores,traces,probe_steps = verify_probes(root,plan,seed,mapping,mode,arm,start,end)
            if start is not None:
                starts[(mode,arm)] = state_hash(start)
                first.setdefault(mode,packet)
                exact(first[mode],packet)
            event_rows.extend(events)
            trajectories.extend(traces)
            legs.append({'seed':seed,'mapping':mapping,'encoder':mode,'policy':arm,'probe_substeps':probe_steps,**cost})
            for channel, values in scores.items():
                blocks.append({'seed':seed,'mapping':mapping,'encoder':mode,'policy':arm,'probe':channel,**values})
        assert len(set(starts.values())) <= 1
    resumes = 0
    seed,mapping = plan['resume_unit']
    for mode,arm in c.CONDITIONS:
        orig,restored = (c.where(path,seed,mapping,mode,arm) for path in (root,root/'resume'))
        if not (restored/'COMPLETE.json').exists():
            missing.append([seed,mapping,mode,arm,'resume']); continue
        status = read_json(restored/'COMPLETE.json')
        assert status['exact'] and status['events'] == plan['events']-plan['cut']
        exact(read_json(orig/'events.json')[plan['cut']:],read_json(restored/'events.json'))
        assert state_hash(load_state(orig/'end.npz')) == state_hash(load_state(restored/'end.npz'))
        assert (orig/'pre_feedback.jsonl').read_bytes().splitlines(keepends=True)[plan['cut']:] == (
            (restored/'pre_feedback.jsonl').read_bytes().splitlines(keepends=True))
        resumes += 1
    keyed = [r for r in blocks if r['policy']=='keyed' and r['probe']=='table']
    blind = [r for r in blocks if r['policy']=='blind' and r['probe']=='table']
    table = len(keyed)==len(blind)==len(plan['development_units']) and all(joint_gate(r,1.) for r in keyed)
    assert all(r['macro']==.5 for r in blind)
    valid = not missing and table and all(s['phase']=='COMPLETED' for s in phases.values())
    keys = ('acc_A','acc_B','macro','nll_A','nll_B','nll','correct_pairs','changed_actions',
            'cue_logit_rms','token_dependent_cue_fraction')
    group = ('encoder','policy','probe')
    means = summary(blocks,group,keys)
    frozen = {(r['encoder'],r['policy'],r['probe']):r for r in means}
    passing = {probe:[[mode,arm] for mode in plan['chosen_mode_order'] for arm in plan['policy_order']
                     if (mode,arm,probe) in frozen and joint_gate(frozen[(mode,arm,probe)],plan['threshold_each_rule'])]
               for probe in ('z0','live','main_live')}
    joint_blocks = {f'{mode}/{arm}/{probe}':sum(joint_gate(r,plan['threshold_each_rule'])
                   for r in blocks if (r['encoder'],r['policy'],r['probe']) == (mode,arm,probe))
                    for mode,arm in c.NEURAL for probe in ('z0','live','main_live')}
    chosen = passing['z0'][0] if valid and passing['z0'] else None
    verdicts = {'JOINT_LEARNABILITY': ('JOINT_LEARNABILITY_ESTABLISHED' if chosen else 'JOINT_LEARNABILITY_NOT_ESTABLISHED')
                if valid else 'INCONCLUSIVE_INVALID_OR_INCOMPLETE',
                'CONTEXT_SWITCHING':'NOT_EVALUATED_C0_ONLY','RETENTION':'NOT_EVALUATED_C0_ONLY',
                'LIVE_ACCESS': ('LIVE_ACCESS_ESTABLISHED_C0' if passing['live'] else 'LIVE_ACCESS_NOT_ESTABLISHED_C0')
                if valid else 'INCONCLUSIVE_INVALID_OR_INCOMPLETE',
                'ENGINEERING_VALIDITY':'C0_VALID' if valid else 'C0_INVALID_OR_INCOMPLETE'}
    path_summary = summary(trajectories,(*group,'tick'),('B_delta_E2',*(f'E{i}_l2' for i in range(1,7)),
                                                         'read_coordinates_l2','logits_linf'))
    for row in path_summary:
        row['token_pairs'] = row.pop('blocks')
    result = {'verdicts':verdicts,'gate':{'selected':chosen,'passing':passing,'joint_blocks':joint_blocks,'table':table,
                                        'threshold':plan['threshold_each_rule']},
              'training_events':len(event_rows),'lives':len(legs),'exact_resumes':resumes,'missing':missing,
              'training_substeps':sum(x['substeps'] for x in legs),'probe_substeps':sum(x['probe_substeps'] for x in legs),
              'phase_status':phases,'probe_means':means,'by_seed':summary(blocks,('seed',*group),keys),
              'by_mapping':summary(blocks,('mapping',*group),keys),'path_summary':path_summary}
    if args.verify_only:
        exact(result,read_json(root/'ANALYSIS.json'))
    else:
        atomic_json(root/'ANALYSIS.json',result)
        for name,data in [('events',event_rows),('life_costs',legs),('per_block_probes',blocks),
                          ('per_seed',result['by_seed']),('per_mapping',result['by_mapping']),('path_trace',trajectories)]:
            if data:
                write_csv(root/f'{name}.csv',data)
    print(json.dumps({'verdicts':verdicts,'gate':result['gate'],'training_events':len(event_rows),
                      'exact_resumes':resumes},indent=2))


if __name__ == '__main__':
    main()
