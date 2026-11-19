"""Independently audit the locked confirmation and evaluate its two criteria."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pqt_qcat.memory import qnuf_confirm as c
from pqt_qcat.memory.access_state import exact
from pqt_qcat.memory.checkpoint_io import load_state
from pqt_qcat.memory.codex_memory_location import atomic_json, digest, metrics
from pqt_qcat.memory.functional_sidecar import ActionJournal, read_json, state_hash
from pqt_qcat.memory.qnuf import array_hash
from scripts.analyze_access_bridge import paired
from scripts.analyze_functional_sidecar import has_target, write_csv
from scripts.analyze_sidecar_causal import expected_update, rng_hash

import numpy as np


ALL_ARMS = ('TABLE', *c.POLICIES, *c.CONTROLS)


def table_scores(table, token):
    assigned = table.get(str(token))
    return [8. if i == assigned else 0. for i in (1, 2, 3, 4)]


def verify_physics(main, k, z, rnd, core, query, event_id):
    assert main['event_id'] == event_id and main['query'] == query
    exact(main['input'], [float(i == query) for i in range(8)])
    assert main['K_pre_hash'] == array_hash(k)
    assert main['Z_pre_hash'] == array_hash(main['Z_pre']) == z
    assert main['Z_post_hash'] == array_hash(main['Z_post'])
    assert main['Z_noinput_hash'] == array_hash(main['Z_noinput'])
    assert main['rng_pre_hash'] == rnd
    assert main['rng_post_hash'] == main['rng_shadow_post_hash']
    exact((core['W_readout'] @ main['Z_post'])[1:5].tolist(), main['logits'])
    return main['Z_post_hash'], main['rng_post_hash']


def validate_update(update, logits, phi, target, k, core, plan, norm):
    assert update['target'] == target and update['evidence_source'] == 'observed_stream'
    error, proposed, candidate, _ = expected_update(logits, phi, target, k, core['W_readout'],
                                                    plan['parent']['eta'], plan['parent']['clip'], norm)
    exact(phi.tolist(), update['phi_event'])
    exact(error.tolist(), update['error_vector'])
    assert candidate == update['candidate_norm_before_cap']
    assert array_hash(proposed) == update['K_post_hash'] == update['K_proposed_hash']
    applied = float(np.linalg.norm(proposed - k))
    assert applied == update['applied_delta_norm']
    if norm is not None:
        assert abs(applied - (norm if candidate else 0.)) <= 2e-14
    return proposed, applied


def check_snapshot(core, ending, k, z, rnd):
    exact(k, ending['K'])
    assert z == array_hash(ending['Z']) and rnd == rng_hash(ending['rng_state'])
    for key in ('B', 'W_readout', 'enable_vortex', 'config', 'integrator_config'):
        exact(core[key], ending[key])


def verify_probe(record, truth):
    assert record['substeps'] == 192
    for state in ('live8', 'Z0_8'):
        exact(record[state], metrics(record[state]['logits'], truth))


def verify_preparation(root, plan, seed, mapping):
    folder, truth, stream = c.panel(root, plan, seed, mapping)
    checkpoint = load_state(folder / 'checkpoint_048.npz')
    exact(checkpoint['unit'], [seed, mapping])
    assert checkpoint['plan_sha256'] == digest(plan)
    core = checkpoint['initial']['core_snapshot']
    k, z, rnd = core['K'], array_hash(core['Z']), rng_hash(core['rng_state'])
    journal = ActionJournal(folder / 'pre_feedback.jsonl')
    table, offset = {}, 0
    events = read_json(folder / 'events.json')
    assert len(events) == len(stream['prefix']) == plan['prefix_events']
    for i in range(0, len(events), 6):
        exact(sorted(stream['prefix'][i:i+6]), list(range(1, 7)))
    for i, (row, q) in enumerate(zip(events, stream['prefix'], strict=True)):
        packet, receipt = row['packet'], row['receipt']
        assert receipt['offset'] == offset and not has_target(packet)
        offset += receipt['length']
        exact(journal.read_verified(receipt), packet)
        assert packet['query'] == q and packet['index'] == i
        main = packet['neural_prediction']
        z, rnd = verify_physics(main, k, z, rnd, core, q, i)
        logits = table_scores(table, q)
        exact(logits, packet['table_logits'])
        assert np.argmax(logits) + 1 == packet['table_action']
        target = truth[q - 1]
        assert target == row['target_after']
        phi = np.tanh(main['Z_post']) - np.tanh(main['Z_noinput'])
        k, _ = validate_update(row['update'], main['logits'], phi, target, k, core, plan, None)
        table[str(q)] = target
    assert journal.path.stat().st_size == offset
    check_snapshot(core, checkpoint['agent']['core_snapshot'], k, z, rnd)
    assert checkpoint['agent']['core_snapshot']['step_count'] - core['step_count'] == plan['prefix_events'] * 16
    exact(checkpoint['table'], table)
    complete = read_json(folder / 'COMPLETE.json')
    assert complete['substeps'] == plan['prefix_events'] * 32
    verify_probe(complete['carry_probe'], truth)
    return checkpoint, complete


def verify_life(root, plan, seed, mapping, env, arm, checkpoint, prefix):
    folder = c.track(root, seed, mapping, env, arm)
    events, timing = read_json(folder/'events.json'), read_json(folder/'runtime.json')
    end = load_state(folder / 'end.npz')
    exact(end['condition'], [env, arm])
    assert end['plan_sha256'] == digest(plan)
    _, truth, stream = c.panel(root, plan, seed, mapping)
    assert len(events) == len(timing) == len(stream['offsets']) == plan['steps']
    table = checkpoint['table'].copy() if arm == 'TABLE' else {}
    core = checkpoint['agent']['core_snapshot']
    k, z, rnd = core['K'], array_hash(core['Z']), rng_hash(core['rng_state'])
    donors = None
    if arm == 'wrong_SS':
        other = (mapping + 1) % len(plan['parent']['stage1_mappings'])
        donor_folder = c.track(root, seed, other, env, 'raw_SS')
        donors = load_state(donor_folder / 'donors.npz')['K']
        donor_events = read_json(donor_folder / 'events.json')
    own_matrices = load_state(folder / 'donors.npz')['K'] if arm == 'raw_SS' else None
    journal = ActionJournal(folder / 'pre_feedback.jsonl')
    query, streak, receipt_end = stream['query'], 0, 0
    rows, targets, deployed, live = [], [], [], []
    start_hash = state_hash(checkpoint['agent'])
    zero_budget = {'main': 0, 'main_shadow': 0, 'side': 0, 'side_shadow': 0}
    neural_budget = {'main': 16, 'main_shadow': 16, 'side': 128, 'side_shadow': 128,
                     'snapshots': 3, 'clones': 3}
    for age, (row, clock) in enumerate(zip(events, timing, strict=True)):
        p, receipt = row['packet'], clock['receipt']
        assert receipt['offset'] == receipt_end and not has_target(p)
        receipt_end += receipt['length']
        exact(journal.read_verified(receipt), p)
        assert clock['publish_ns'] <= clock['reaction_ns'] <= clock['reveal_ns'] <= clock['update_ns']
        exact(p['observation'], {'query': query, 'cursor': age, 'streak': streak})
        assert p['query'] == query and p['cursor'] == age and p['main']['event_id'] == plan['prefix_events'] + age
        assert p['action'] == np.argmax(p['logits']) + 1
        target, offset = truth[query - 1], stream['offsets'][age]
        assert target == row['target_after']
        correct = p['action'] == target
        hit = (query - 1 + offset) % 6 + 1
        miss = query if env == 'retry' else (query + offset % 5) % 6 + 1
        assert hit != miss
        next_query = hit if correct else miss
        next_streak = 0 if correct else streak + 1
        reward = 1. if correct else -min(4, 1 + streak) / 4
        exact(row['reaction'], {'next_query': next_query, 'next_streak': next_streak, 'correct': correct, 'reward': reward})
        k_delta, z_norm, applied = None, None, None
        if arm == 'TABLE':
            exact(p['budget'], zero_budget)
            exact(p['logits'], table_scores(table, query))
            assert p['pending_sha256'] == state_hash(table)
            table[str(query)] = target
            assert row['end_state'] == state_hash(table)
        else:
            credit, norm, donor = (c.POLICIES | c.CONTROLS)[arm]
            exact(p['budget'], neural_budget)
            assert p['pre_sha256'] == (start_hash if age == 0 else events[age-1]['end_state'])
            main, side = p['main'], p['external']
            z, rnd = verify_physics(main, k, z, rnd, core, query, plan['prefix_events'] + age)
            exact(checkpoint['agent']['Z0'], side['Z_start'])
            assert side['rng_pre'] == main['rng_pre_hash']
            if donor == 'initial':
                read_k = checkpoint['initial']['core_snapshot']['K']
            elif donor == 'wrong':
                read_k = donors[age]
                assert array_hash(read_k) == donor_events[age]['packet']['main']['K_pre_hash']
            else:
                read_k = k
            if own_matrices is not None:
                exact(k, own_matrices[age])
            exact(side['hashes'], {'K': array_hash(read_k), 'B': array_hash(core['B']),
                                   'W_readout': array_hash(core['W_readout'])})
            exact(side['logits'], p['logits'])
            exact((core['W_readout'] @ side['Z_after'])[1:5].tolist(), p['logits'])
            side_phi = np.tanh(side['Z_after']) - np.tanh(side['Z_no_input'])
            exact(side_phi.tolist(), side['eligibility'])
            logit_source, feature_source = c.causal.SOURCES[credit]
            logits = p['logits'] if logit_source == 'sidecar' else main['logits']
            phi = side_phi if feature_source == 'sidecar' else np.tanh(main['Z_post']) - np.tanh(main['Z_noinput'])
            k_delta = float(np.linalg.norm(read_k - k))
            k, applied = validate_update(row['update'], logits, phi, target, k, core, plan, norm)
            z_norm = main['Z_norms'][1]
            live.append(main['logits'])
        deployed.append(p['logits'])
        targets.append(target)
        rows.append({'seed': seed, 'mapping': mapping, 'environment': env, 'arm': arm, 'age': age,
                     'query': query, 'action': p['action'], 'target_after': target, 'correct': int(correct),
                     'reward': reward, 'next_query': next_query, 'streak_after': next_streak, 'offset': offset,
                     'receipt_sha256': receipt['sha256'], 'deployed_logits': json.dumps(p['logits']),
                     'main_logits': json.dumps(live[-1]) if live else None, 'K_update_norm': applied,
                     'K_donor_distance': k_delta, 'Z_post_norm': z_norm, 'physical_substeps': 0 if arm == 'TABLE' else 288,
                     'event_wall_ms': clock['event_wall_ns'] / 1e6})
        query, streak = next_query, next_streak
    assert receipt_end == journal.path.stat().st_size
    exact(end['environment'], {'query': query, 'cursor': plan['steps'], 'streak': streak})
    exact(end['table'], table)
    complete = read_json(folder / 'COMPLETE.json')
    score = metrics(deployed, targets)
    summary = {'accuracy': score['accuracy'], 'nll': score['nll'], 'margin': score['margin'],
               'reward': sum(x['reward'] for x in rows), 'tail_accuracy': sum(x['correct'] for x in rows[-8:]) / 8,
               'unique_queries': len({x['query'] for x in rows}), 'max_error_streak': max(x['streak_after'] for x in rows),
               'substeps': sum(x['physical_substeps'] for x in rows), 'probe_substeps': 0,
               'event_wall_ms': float(np.mean([x['event_wall_ms'] for x in rows])), 'cpu_seconds': complete['cpu_seconds'],
               'K_step_norm': None, 'K_delta_norm': None, 'carry_accuracy': None, 'carry_nll': None,
               'carry_fixed_accuracy': None, 'carry_fixed_nll': None, 'carry_fixed_change': None, 'Z_end_norm': None}
    if arm != 'TABLE':
        check_snapshot(core, end['agent']['core_snapshot'], k, z, rnd)
        assert state_hash(end['agent']) == events[-1]['end_state']
        assert float(np.linalg.norm(k - core['K'])) == complete['delta_K_norm']
        verify_probe(complete['carry_probe'], truth)
        fixed, previous = complete['carry_probe']['live8'], prefix['carry_probe']['live8']
        internal = metrics(live, targets)
        summary.update(K_step_norm=float(np.mean([x['K_update_norm'] for x in rows])),
                       K_delta_norm=complete['delta_K_norm'], carry_accuracy=internal['accuracy'], carry_nll=internal['nll'],
                       carry_fixed_accuracy=fixed['accuracy'], carry_fixed_nll=fixed['nll'],
                       carry_fixed_change=fixed['accuracy'] - previous['accuracy'], Z_end_norm=rows[-1]['Z_post_norm'],
                       probe_substeps=complete['carry_probe']['substeps'])
    else:
        assert end['agent'] is None
    return rows, summary, events[0]['packet']


def difference(a, b, plan):
    res = {}
    keys = next((v.keys() for v in a + b if v is not None), ())
    for key in keys:
        if any(left is not None and right is not None and left[key] is not None and right[key] is not None
               for left, right in zip(a, b, strict=True)):
            res[key] = paired([left[key] - right[key] if left is not None and right is not None
                               and left[key] is not None and right[key] is not None else np.nan
                               for left, right in zip(a, b, strict=True)], plan)
    if 'accuracy' in res:
        # Keep zero in the bootstrap support exact by contrasting integer counts first.
        n = plan['steps']
        counts = paired([round(left['accuracy'] * n) - round(right['accuracy'] * n)
                         if left is not None and right is not None else np.nan
                         for left, right in zip(a, b, strict=True)], plan)
        result = {**counts, 'mean': counts['mean'] / n,
                  'seed_means': [x / n if x is not None else None for x in counts['seed_means']]}
        if counts['ci95'] is not None:
            result['ci95'] = [x / n for x in counts['ci95']]
        res['accuracy'] = result
    return res


def verdict(contrast, threshold, valid):
    if not valid or contrast is None or contrast['ci95'] is None:
        return 'INCONCLUSIVE_INVALID_OR_INCOMPLETE'
    reaches_threshold = contrast['mean'] >= threshold or abs(contrast['mean'] - threshold) <= 2 * np.spacing(threshold)
    if reaches_threshold and contrast['ci95'][0] > 0:
        return 'CONFIRMED'
    return 'NOT_CONFIRMED'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    plan = c.verify_plan(root)
    stages = {phase: read_json(root / f'STATUS_{phase}.json') for phase in ('prepare', 'primary', 'controls', 'resume')}
    complete = all(s['phase'] in ('COMPLETED', 'COMPLETED_WITH_FAILURES') for s in stages.values())
    assert complete
    event_rows, lives, missing = [], [], []
    groups = {(env, arm): [None] * len(plan['units']) for env in plan['environments'] for arm in ALL_ARMS}
    swaps = {(env, control): [] for env in plan['environments'] for control in c.CONTROLS}
    starts = {}
    for index, (seed, mapping) in enumerate(plan['units']):
        folder, _, _ = c.panel(root, plan, seed, mapping)
        if not (folder / 'COMPLETE.json').is_file():
            missing.append([seed, mapping, 'prepare'])
            continue
        checkpoint, prefix = verify_preparation(root, plan, seed, mapping)
        first_hash = state_hash(checkpoint['initial'])
        assert starts.setdefault(seed, first_hash) == first_hash
        first = {}
        for env, arm in groups:
            if not (c.track(root, seed, mapping, env, arm) / 'COMPLETE.json').is_file():
                missing.append([seed, mapping, env, arm])
                continue
            rows, summary, first[(env, arm)] = verify_life(root, plan, seed, mapping, env, arm, checkpoint, prefix)
            groups[(env, arm)][index] = summary
            event_rows.extend(rows)
            lives.append({'seed': seed, 'mapping': mapping, 'environment': env, 'arm': arm, **summary})
        own = [p for (env, arm), p in first.items() if arm in c.POLICIES]
        if own:
            for packet in own:
                exact(own[0], packet)
        for env, control in swaps:
            ref, donor = first.get((env, 'raw_SS')), first.get((env, control))
            if ref is None or donor is None:
                continue
            for name in ('main', 'pre_sha256', 'query'):
                exact(ref[name], donor[name])
            for name in ('Z_start', 'rng_pre', 'rng_post'):
                exact(ref['external'][name], donor['external'][name])
            for name in ('B', 'W_readout'):
                exact(ref['external']['hashes'][name], donor['external']['hashes'][name])
            swaps[(env, control)].append({'action_changed': bool(ref['action'] != donor['action']),
                                         'K_changed': bool(ref['external']['hashes']['K'] != donor['external']['hashes']['K']),
                                         'logits_linf': float(np.max(np.abs(np.array(ref['logits']) - donor['logits'])))})
    resumes = 0
    for env, arm in groups:
        seed, mapping = plan['resume_unit']
        restored = c.track(root / 'resume', seed, mapping, env, arm)
        original = c.track(root, seed, mapping, env, arm)
        if not (restored / 'COMPLETE.json').is_file():
            missing.append([seed, mapping, env, arm, 'resume'])
            continue
        status = read_json(restored / 'COMPLETE.json')
        assert status['exact'] and status['events'] == plan['steps'] - plan['resume_age']
        exact(read_json(original / 'events.json')[plan['resume_age']:], read_json(restored / 'events.json'))
        assert state_hash(load_state(original/'end.npz')) == state_hash(load_state(restored/'end.npz'))
        assert (original/'pre_feedback.jsonl').read_bytes().splitlines(keepends=True)[plan['resume_age']:] == (
            (restored/'pre_feedback.jsonl').read_bytes().splitlines(keepends=True))
        resumes += 1

    table_valid = all(row is not None and row['accuracy'] == plan['table_threshold']
                      for env in plan['environments'] for row in groups[(env, 'TABLE')])
    valid = not missing and table_valid and all(s['phase'] == 'COMPLETED' for s in stages.values())
    stats = {}
    for (env, arm), rows in groups.items():
        keys = next((row.keys() for row in rows if row is not None), ())
        stats[f'{env}/{arm}'] = {key: paired([row[key] if row is not None and row[key] is not None else np.nan
                                            for row in rows], plan) for key in keys}
    comparisons = {env: {arm: difference(groups[(env, arm)], groups[(env, 'raw_MM')], plan)
                          for arm in (*c.POLICIES, 'TABLE') if arm != 'raw_MM'} for env in plan['environments']}
    for env in plan['environments']:
        for arm in c.CONTROLS:
            comparisons[env]['raw_SS_minus_' + arm] = difference(groups[(env, 'raw_SS')], groups[(env, arm)], plan)
    main_stats = comparisons['advance']['raw_SS']
    analysis = {'engineering': 'CONFIRM_VALID' if valid else 'CONFIRM_INVALID_OR_INCOMPLETE',
                'accuracy_verdict': verdict(main_stats.get('accuracy'), plan['accuracy_threshold'], valid),
                'reward_verdict': verdict(main_stats.get('reward'), plan['reward_threshold'], valid),
                'primary': {key: main_stats.get(key) for key in ('accuracy', 'reward')},
                'stage_statuses': stages, 'table_valid': table_valid, 'missing': missing, 'resumes_exact': resumes,
                'lives_verified': len(lives), 'actions_verified': len(event_rows), 'prefix_blocks_verified': len({(r['seed'], r['mapping']) for r in lives}),
                'stats': stats, 'comparisons': comparisons, 'first_event_K_only': {
                    f'{env}/{control}': {'blocks': len(rows), 'K_changed': sum(r['K_changed'] for r in rows),
                                          'action_changed': sum(r['action_changed'] for r in rows),
                                          'mean_delta_logits_linf': float(np.mean([r['logits_linf'] for r in rows])) if rows else None}
                    for (env, control), rows in swaps.items()}}
    if args.verify_only:
        exact(analysis, read_json(root / 'ANALYSIS.json'))
    else:
        atomic_json(root / 'ANALYSIS.json', analysis)
        if event_rows:
            write_csv(root / 'per_event.csv', event_rows)
            write_csv(root / 'per_life.csv', lives)
    print(json.dumps({k: analysis[k] for k in ('engineering', 'accuracy_verdict', 'reward_verdict', 'primary',
                                             'lives_verified', 'actions_verified', 'resumes_exact')}, indent=2))


if __name__ == '__main__':
    main()
