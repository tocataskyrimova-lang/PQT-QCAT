"""Check the action journals and reconstruct the frozen factorial contrasts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pqt_qcat.memory import sidecar_causal as c
from pqt_qcat.memory.access_state import exact
from pqt_qcat.memory.checkpoint_io import load_state
from pqt_qcat.memory.codex_memory_location import atomic_json, digest, metrics, source_hashes
from pqt_qcat.memory.functional_sidecar import ActionJournal, checkpoint_path, read_json, state_hash
from pqt_qcat.memory.qnuf import array_hash
from scripts.analyze_access_bridge import paired
from scripts.analyze_functional_sidecar import has_target, write_csv

import numpy as np


def rng_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def conditions():
    return [(stage, arm) for stage in c.STAGES for arm in (('SS',) if stage in ('initial', 'wrong') else c.SOURCES)]


def provenance(root, reference, history, b1):
    original = read_json(root / 'PLAN.json')
    source = read_json(root / 'SOURCE.json')
    assert source['plan_sha256'] == digest(original)
    assert {p.replace('\\', '/'): h for p, h in source['files'].items()} == {
        p.replace('\\', '/'): h for p, h in source_hashes().items()}
    plan = {**original, 'inputs': {}}
    for key, location in [('historical_source', history), ('b1_source', b1)]:
        plan[key] = str(location.resolve()) if location else original[key]
        plan['inputs'][plan[key]] = original['inputs'][original[key]]
    c.b2.b1.verify_artifacts(plan)
    reference = reference.resolve() if reference else Path(original['b2a_reference'])
    plan['b2a_reference'] = str(reference)
    manifest = reference / 'ARTIFACT_MANIFEST.sha256'
    assert c.file_sha(manifest) == original['b2a_manifest_sha256']
    indexed = {name: checksum for checksum, name in
               [line.split('  ', 1) for line in manifest.read_text(encoding='utf-8').splitlines()]}
    for seed, mapping in plan['units']:
        for name in ('Z0-64_main', 'Z0-64_aligned'):
            relative = f'lives/{seed}_{mapping}/{name}/events.json'
            assert c.file_sha(reference / relative) == indexed[relative]
    return plan


def expected_update(logits, phi, target, k, w, eta, clip, norm):
    p = np.exp(logits - np.max(logits))
    p /= np.sum(p)
    e = np.zeros(8)
    e[1:5] = -p
    e[target] = 1 - p[target - 1]
    delta = np.clip(eta * np.outer(w.T @ e, phi), -clip, clip)
    magnitude = np.linalg.norm(delta)
    raw_norm = np.linalg.norm((k + delta) - k)
    if norm is not None and magnitude != 0.:
        delta *= norm / magnitude
    return e, k + delta, float(magnitude), float(raw_norm)


def verify_life(root, plan, seed, mapping, stage, arm, origin):
    folder = c.lived(root, seed, mapping, stage, arm)
    events = read_json(folder / 'events.json')
    timing = read_json(folder / 'runtime.json')
    assert len(events) == len(timing) == plan['steps']
    journal = ActionJournal(folder / 'pre_feedback.jsonl')
    stream = plan['streams'][c.unit_name(seed, mapping)]
    target_map = plan['parent']['stage1_mappings'][mapping]
    core = origin['core_snapshot']
    w = core['W_readout']
    k = core['K'].copy()
    z = array_hash(core['Z'])
    rnd = rng_hash(core['rng_state'])
    _, scale, donor = c.STAGES[stage]
    norm = c.SCALES[scale]
    if donor == 'initial':
        initial_k = load_state(checkpoint_path(plan, 'B0', seed, mapping, 0))['agent']['core_snapshot']['K']
    if donor == 'wrong':
        wrong = c.lived(root, seed, (mapping + 1) % 4, 'raw', 'SS')
        donor_k = load_state(wrong / 'donors.npz')['K']
        donor_history = read_json(wrong / 'events.json')
    reference = None
    if stage == 'raw' and arm in ('MM', 'SS'):
        old_arm = 'Z0-64_main' if arm == 'MM' else 'Z0-64_aligned'
        reference = read_json(Path(plan['b2a_reference']) / 'lives' / c.unit_name(seed, mapping) / old_arm / 'events.json')
    if stage == 'raw' and arm == 'SS':
        recorded_k = load_state(folder / 'donors.npz')['K']
    query, streak, byte_offset = stream['start_query'], 0, 0
    rows = []
    event_logits, main_logits, targets = [], [], []
    for age, (event, clock) in enumerate(zip(events, timing, strict=True)):
        packet = event['packet']
        assert clock['receipt']['offset'] == byte_offset and not has_target(packet)
        byte_offset += clock['receipt']['length']
        exact(journal.read_verified(clock['receipt']), packet)
        assert clock['publish_ns'] <= clock['reaction_ns'] <= clock['reveal_ns'] <= clock['update_ns']
        exact(packet['observation'], {'query': query, 'cursor': age, 'streak': streak})
        assert packet['query'] == query and packet['cursor'] == age
        exact(packet['budget'], {'main': 16, 'main_shadow': 16, 'side': 128, 'side_shadow': 128,
                                 'snapshots': 3, 'clones': 3})
        assert packet['pre_sha256'] == (state_hash(origin) if age == 0 else events[age-1]['end_state'])
        main, side = packet['main'], packet['external']
        assert main['event_id'] == plan['cursor_start'] + age and main['query'] == query
        assert main['Z_pre_hash'] == array_hash(main['Z_pre']) == z
        assert main['Z_post_hash'] == array_hash(main['Z_post'])
        assert main['Z_noinput_hash'] == array_hash(main['Z_noinput'])
        assert main['rng_pre_hash'] == side['rng_pre'] == rnd
        assert main['rng_post_hash'] == main['rng_shadow_post_hash']
        z, rnd = main['Z_post_hash'], main['rng_post_hash']
        exact(origin['Z0'], side['Z_start'])
        exact((w @ main['Z_post'])[1:5].tolist(), main['logits'])
        exact((w @ side['Z_after'])[1:5].tolist(), packet['logits'])
        exact(side['logits'], packet['logits'])
        assert main['K_pre_hash'] == array_hash(k)
        read_k = k
        if donor == 'initial':
            read_k = initial_k
        elif donor == 'wrong':
            read_k = donor_k[age]
            assert array_hash(read_k) == donor_history[age]['packet']['main']['K_pre_hash']
        if stage == 'raw' and arm == 'SS':
            np.testing.assert_array_equal(recorded_k[age], k)
        exact(side['hashes'], {'K': array_hash(read_k), 'B': array_hash(core['B']), 'W_readout': array_hash(w)})
        target = target_map[query - 1]
        action = np.argmax(packet['logits']).item() + 1
        assert action == packet['action'] and event['evidence_target'] == target
        correct = action == target
        offset = stream['offsets'][age]
        success_query = (query - 1 + offset) % 6 + 1
        fail_query = (query + offset % 5) % 6 + 1 if stage == 'advance' else query
        assert success_query != fail_query
        next_query = success_query if correct else fail_query
        next_streak = 0 if correct else streak + 1
        reward = 1. if correct else -min(4, 1 + streak) / 4
        exact(event['reaction'], {'next_query': next_query, 'correct': correct,
                                  'next_streak': next_streak, 'reward': reward})
        side_phi = np.tanh(side['Z_after']) - np.tanh(side['Z_no_input'])
        exact(side_phi.tolist(), side['eligibility'])
        logits_source, phi_source = c.SOURCES[arm]
        exact(event['credit'], [logits_source, phi_source])
        phi = np.tanh(main['Z_post']) - np.tanh(main['Z_noinput']) if phi_source == 'main' else side_phi
        logits = main['logits'] if logits_source == 'main' else packet['logits']
        update = event['update']
        assert update['target'] == target and update['evidence_source'] == 'observed_stream'
        e, proposed, magnitude, raw_norm = expected_update(
            logits, phi, target, k, w, plan['parent']['eta'], plan['parent']['clip'], norm)
        exact(e.tolist(), update['error_vector'])
        exact(phi.tolist(), update['phi_event'])
        exact(event['norm_target'], norm)
        assert update['candidate_norm_before_cap'] == magnitude and raw_norm == event['raw_applied']
        observed_norm = float(np.linalg.norm(proposed - k))
        assert update['applied_delta_norm'] == observed_norm
        assert update['K_post_hash'] == update['K_proposed_hash'] == array_hash(proposed)
        if norm is not None:
            assert abs(observed_norm - (0. if magnitude == 0. else norm)) <= 2e-14
        rows.append({'seed': seed, 'mapping': mapping, 'stage': stage, 'arm': arm, 'age': age,
                     'query': query, 'action': action, 'target': target, 'correct': int(correct),
                     'reward': reward, 'next_query': next_query, 'streak': next_streak, 'offset': offset,
                     'logits_source': logits_source, 'phi_source': phi_source,
                     'deployed_logits': json.dumps(packet['logits']), 'main_logits': json.dumps(main['logits']),
                     'pre_receipt': clock['receipt']['sha256'], 'norm_candidate': magnitude,
                     'norm_applied': observed_norm, 'phi_norm': float(np.linalg.norm(phi)),
                     'donor_distance': float(np.linalg.norm(k - read_k)), 'Z_post_norm': main['Z_norms'][1],
                     'substeps': 288, 'wall_ms': clock['event_wall_ns']/1e6})
        if reference is not None:
            for name in ('reaction', 'evidence_target', 'update', 'end_state'):
                exact(event[name], reference[age][name])
            for name in ('query', 'main', 'logits', 'action', 'pre_sha256', 'pending_sha256'):
                exact(packet[name], reference[age]['packet'][name])
        k, query, streak = proposed, next_query, next_streak
        targets.append(target)
        event_logits.append(packet['logits'])
        main_logits.append(main['logits'])
    assert journal.path.stat().st_size == byte_offset
    end = load_state(folder / 'end.npz')
    assert state_hash(end['agent']) == events[-1]['end_state']
    ending = end['agent']['core_snapshot']
    assert array_hash(ending['Z']) == z and rng_hash(ending['rng_state']) == rnd
    for field in ('B', 'W_readout', 'config', 'integrator_config'):
        exact(ending[field], core[field])
    exact(k, ending['K'])
    exact(end['environment'], {'query': query, 'cursor': plan['steps'], 'streak': streak})
    outcome = metrics(event_logits, targets)
    carry = metrics(main_logits, targets)
    complete = read_json(folder / 'COMPLETE.json')
    assert complete['delta_K_norm'] == float(np.linalg.norm(k - core['K']))
    accuracy = [r['correct'] for r in rows]
    summary = {'reward': sum(r['reward'] for r in rows), 'accuracy': outcome['accuracy'],
               'nll': outcome['nll'], 'margin': outcome['margin'], 'tail_accuracy': float(np.mean(accuracy[-8:])),
               'after_first_accuracy': float(np.mean(accuracy[1:])), 'carry_accuracy': carry['accuracy'],
               'carry_nll': carry['nll'], 'carry_fixed': complete['retention']['live8']['accuracy'],
               'carry_fixed_nll': complete['retention']['live8']['nll'],
               'reset_fixed': complete['retention']['Z0_8']['accuracy'],
               'norm_applied_mean': float(np.mean([r['norm_applied'] for r in rows])),
               'norm_candidate_mean': float(np.mean([r['norm_candidate'] for r in rows])),
               'zero_updates': sum(r['norm_candidate'] == 0. for r in rows),
               'donor_distance': float(np.mean([r['donor_distance'] for r in rows])),
               'delta_K_norm': complete['delta_K_norm'], 'unique_queries': len({r['query'] for r in rows}),
               'unique_transitions': len({(r['query'], r['next_query']) for r in rows}),
               'max_streak': max(r['streak'] for r in rows), 'Z_end_norm': rows[-1]['Z_post_norm'],
               'substeps': sum(r['substeps'] for r in rows), 'probe_substeps': complete['retention']['substeps'],
               'event_wall_ms': float(np.mean([r['wall_ms'] for r in rows])), 'cpu_seconds': complete['cpu_seconds']}
    return rows, summary


def compare(arrays, terms, plan):
    metrics_keys = next(iter(arrays.values()))[0].keys()
    return {key: paired([sum(weight * arrays[arm][i][key] for arm, weight in terms.items())
                         for i in range(len(plan['units']))], plan) for key in metrics_keys}


def factorial(arrays, plan):
    tests = {'logits_mean': {'SM': .5, 'MM': -.5, 'SS': .5, 'MS': -.5},
             'eligibility_mean': {'MS': .5, 'MM': -.5, 'SS': .5, 'SM': -.5},
             'interaction': {'SS': 1, 'MS': -1, 'SM': -1, 'MM': 1},
             'logits_at_main_phi': {'SM': 1, 'MM': -1}, 'logits_at_side_phi': {'SS': 1, 'MS': -1},
             'phi_at_main_logits': {'MS': 1, 'MM': -1}, 'phi_at_side_logits': {'SS': 1, 'SM': -1},
             'joint': {'SS': 1, 'MM': -1}}
    return {name: compare(arrays, terms, plan) for name, terms in tests.items()}


def classify(result, threshold):
    mean, ci = result['mean'], result['ci95']
    if ci is not None:
        if mean >= threshold and ci[0] > 0:
            return 'POSITIVE'
        if mean <= -threshold and ci[1] < 0:
            return 'NEGATIVE'
        if ci[0] > -threshold and ci[1] < threshold:
            return 'WITHIN_MARGIN'
    return 'INCONCLUSIVE'


def verify_one_step(root, plan):
    details, arrays = [], {f'{scale}/{arm}': [] for scale in c.SCALES for arm in c.SOURCES}
    arrays.update({name: [] for name in ('none/MM', 'read-initial/SS', 'read-wrong/SS')})
    for seed, mapping in plan['units']:
        folder = root / 'one_step' / c.unit_name(seed, mapping)
        anchors = read_json(folder / 'RESULTS.json')['anchors']
        exact([[a['anchor'], a['age']] for a in anchors], plan['counterfactual_anchors'])
        units = {name: [] for name in arrays}
        for anchor in anchors:
            age, arm = anchor['age'], anchor['anchor']
            raw = c.lived(root, seed, mapping, 'raw', arm)
            cut = load_state(raw / f'pending_{age:02}.npz')
            packet = ActionJournal(raw / 'pre_feedback.jsonl').read_verified(cut['receipt'])
            assert (folder / f'{arm}_{age:02}_pre_feedback.jsonl').read_bytes() == c.b2.b1.canonical(packet) + b'\n'
            assert anchor['pending_sha256'] == state_hash(cut['agent']) == packet['pending_sha256']
            event = read_json(raw / 'events.json')[age]
            exact(event['reaction'], anchor['reaction'])
            assert anchor['issued_action'] == packet['action'] and anchor['queries'][0] == packet['query']
            exact(anchor['queries'], list(dict.fromkeys((packet['query'], event['reaction']['next_query']))))
            targets = [plan['parent']['stage1_mappings'][mapping][q-1] for q in anchor['queries']]
            exact(targets, anchor['targets_for_scoring'])
            saved = load_state(folder / f'{arm}_{age:02}_deltas.npz')
            core = cut['agent']['core_snapshot']
            exact(saved['K'], core['K'])
            predictions = anchor['predictions']
            lookup = {(r['scale'], r['arm']): delta for r, delta in zip(predictions, saved['delta'], strict=True)}
            assert len(lookup) == len(arrays)
            for row, delta in zip(predictions, saved['delta'], strict=True):
                scale, choice = row['scale'], row['arm']
                norm = float(np.linalg.norm(delta))
                assert norm == row['applied_delta_norm'] and row['substeps'] == 128 * len(targets)
                if scale in c.SCALES:
                    logits, phi = c.credit_values(packet, choice)
                    error, proposed, _, _ = expected_update(logits, phi, targets[0], saved['K'], core['W_readout'],
                                                          plan['parent']['eta'], plan['parent']['clip'], c.SCALES[scale])
                    exact(proposed - saved['K'], delta)
                    assert row['K_hash'] == row['update']['K_post_hash'] == array_hash(proposed)
                    exact(error.tolist(), row['update']['error_vector'])
                    ref = lookup[(scale, 'MM')]
                    raw_ref = lookup[('raw', choice)]
                else:
                    assert norm == 0. and row['K_hash'] == array_hash(saved['K'])
                    if scale.startswith('read-'):
                        donor = c.donor_matrices(root, plan, seed, mapping, scale.removeprefix('read-'))[age]
                        assert row['donor_hash'] == array_hash(donor)
                    ref = raw_ref = delta
                values = {'current_nll': metrics(row['logits'][:1], targets[:1])['nll'],
                          'current_acc': int(np.argmax(row['logits'][0])+1 == targets[0]), 'norm': norm,
                          'cos_to_MM': float(np.vdot(ref, delta)/(norm*np.linalg.norm(ref))) if norm and np.linalg.norm(ref) else None,
                          'cos_to_raw': float(np.vdot(raw_ref, delta)/(norm*np.linalg.norm(raw_ref)))
                          if norm and np.linalg.norm(raw_ref) else None}
                if len(targets) == 2:
                    values['distinct_next_nll'] = metrics(row['logits'][1:], targets[1:])['nll']
                    values['distinct_next_acc'] = int(np.argmax(row['logits'][1])+1 == targets[1])
                details.append({'seed': seed, 'mapping': mapping, 'anchor': arm, 'age': age,
                                'scale': scale, 'arm': choice, **{k: values.get(k) for k in ('current_nll', 'current_acc',
                                'distinct_next_nll', 'distinct_next_acc', 'norm', 'cos_to_MM', 'cos_to_raw')}})
                units[f'{scale}/{choice}'].append(values)
        for key, data in units.items():
            stats = {}
            for name in ('current_nll','current_acc','distinct_next_nll','distinct_next_acc','norm','cos_to_MM','cos_to_raw'):
                valid = [r[name] for r in data if r.get(name) is not None]
                stats[name] = float(np.mean(valid)) if valid else np.nan
            arrays[key].append(stats)
    means = {name: {k: paired([r[k] for r in data], plan) for k in data[0]} for name, data in arrays.items()}
    contrasts = {'high_minus_low': {arm: compare(arrays, {f'high/{arm}': 1, f'low/{arm}': -1}, plan) for arm in c.SOURCES},
                 'factorial': {scale: factorial({arm: arrays[f'{scale}/{arm}'] for arm in c.SOURCES}, plan)
                               for scale in c.SCALES},
                 'vs_no_update': {key: compare(arrays, {key: 1, 'none/MM': -1}, plan) for key in arrays if key != 'none/MM'}}
    return means, contrasts, details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--history', type=Path)
    parser.add_argument('--b1', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    plan = provenance(root, args.reference, args.history, args.b1)
    for phase in ('primary', 'controls', 'one-step', 'resume'):
        assert read_json(root / f'STATUS_{phase}.json')['phase'] == 'COMPLETED'
    lives, event_rows, pairs = [], [], {f'{a}/{b}': [] for a, b in conditions()}
    for seed, mapping in plan['units']:
        origin = load_state(checkpoint_path(plan, 'B3', seed, mapping, plan['cursor_start']))['agent']
        first = None
        for stage, arm in conditions():
            events, scores = verify_life(root, plan, seed, mapping, stage, arm, origin)
            if stage not in ('initial', 'wrong'):
                initial = (events[0]['query'], events[0]['deployed_logits'], events[0]['main_logits'])
                if first is None:
                    first = initial
                else:
                    exact(first, initial)
            event_rows.extend(events)
            pairs[f'{stage}/{arm}'].append(scores)
            lives.append({'seed': seed, 'mapping': mapping, 'stage': stage, 'arm': arm, **scores})
    for stage, arm in conditions():
        seed, mapping = plan['resume_unit']
        original = c.lived(root, seed, mapping, stage, arm)
        resume = c.lived(root / 'resume', seed, mapping, stage, arm)
        status = read_json(resume / 'COMPLETE.json')
        assert status['exact'] and status['events'] == plan['steps'] - plan['resume_age']
        assert state_hash(load_state(original / 'end.npz')) == state_hash(load_state(resume / 'end.npz'))
        assert (original / 'pre_feedback.jsonl').read_bytes().splitlines(keepends=True)[plan['resume_age']:] == (
            (resume / 'pre_feedback.jsonl').read_bytes().splitlines(keepends=True))
    factor = {stage: factorial({arm: pairs[f'{stage}/{arm}'] for arm in c.SOURCES}, plan)
              for stage in ('raw', 'low', 'high', 'advance')}
    stats = {name: {k: paired([r[k] for r in data], plan) for k in data[0]} for name, data in pairs.items()}
    no_retry = compare(pairs, {'advance/SS': 1, 'advance/MM': -1, 'raw/SS': -1, 'raw/MM': 1}, plan)
    scale = {arm: compare(pairs, {f'high/{arm}': 1, f'low/{arm}': -1}, plan) for arm in c.SOURCES}
    donors = {donor: compare(pairs, {'raw/SS': 1, f'{donor}/SS': -1}, plan) for donor in ('initial', 'wrong')}
    one_step, one_step_contrasts, step_rows = verify_one_step(root, plan)
    scores = {'factor': {stage: {name: {m: classify(stats[m], plan[rule]) for m, rule in
              [('accuracy','accuracy_threshold'), ('reward','return_threshold')]} for name, stats in tests.items()}
              for stage, tests in factor.items()}, 'scale': {arm: classify(value['accuracy'], plan['accuracy_threshold'])
              for arm, value in scale.items()}}
    result = {'engineering': 'CAUSAL_VALID', 'lives_verified': len(lives), 'events_verified': len(event_rows),
              'historical_endpoints_exact': 2 * len(plan['units']), 'resumes_exact': len(conditions()),
              'one_step_anchors': len(step_rows) // len(one_step), 'factorial': factor, 'standalone_arm_stats': stats,
              'matched_high_minus_low': scale, 'summary_of_accuracy_and_return_tests': scores,
              'no_retry_interaction': no_retry, 'donor_contrasts': donors, 'one_step': one_step,
              'one_step_contrasts': one_step_contrasts}
    if args.verify_only:
        exact(result, read_json(root / 'ANALYSIS.json'))
    else:
        atomic_json(root / 'ANALYSIS.json', result)
        write_csv(root / 'per_event.csv', event_rows)
        write_csv(root / 'per_life.csv', lives)
        write_csv(root / 'one_step.csv', step_rows)
    print(f"Verified {len(lives)} lives, {len(event_rows)} events, {len(conditions())} exact resumes")


if __name__ == '__main__':
    main()
