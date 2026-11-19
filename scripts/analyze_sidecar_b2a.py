"""Verify every B2A journal and summarize the frozen paired comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pqt_qcat.memory import functional_sidecar_b2a as b2
from pqt_qcat.memory.functional_sidecar import ActionJournal, checkpoint_path, read_json
from pqt_qcat.memory.checkpoint_io import load_state
from pqt_qcat.memory.access_state import exact
from pqt_qcat.memory.codex_memory_location import atomic_json, digest, metrics, source_hashes
from pqt_qcat.memory.qnuf import array_hash
from scripts.analyze_access_bridge import paired
from scripts.analyze_functional_sidecar import has_target, write_csv
import numpy as np


def verify_provenance(root, plan, history, b1_source):
    frozen = read_json(root / 'SOURCE.json')
    assert frozen['plan_sha256'] == digest(plan)
    recorded = {name.replace('\\', '/'): h for name, h in frozen['files'].items()}
    working = {name.replace('\\', '/'): h for name, h in source_hashes().items()}
    changed = {name for name in recorded.keys() | working.keys() if recorded.get(name) != working.get(name)}
    if changed:
        assert all('/tests/' in name for name in changed)
        with ZipFile(root / 'FROZEN_CODE.zip') as archive:
            for name, expected in recorded.items():
                assert hashlib.sha256(archive.read(name)).hexdigest() == expected
    relocated = {**plan, 'inputs': {}}
    for key, local in [('historical_source', history), ('b1_source', b1_source)]:
        original = plan[key]
        relocated[key] = str(local.resolve()) if local else original
        relocated['inputs'][relocated[key]] = plan['inputs'][original]
    b2.b1.verify_artifacts(relocated)
    return relocated


def verify_trajectory(root, plan, seed, mapping, arm):
    reader, credit, donor_mode = b2.ARMS[arm]
    unit_name = f'{seed}_{mapping}'
    life = root / 'lives' / unit_name / arm
    rows = read_json(life / 'events.json')
    timing = read_json(life / 'runtime.json')
    replay_folder = root / 'replay' / 'lives' / unit_name / arm
    replay = read_json(replay_folder / 'COMPLETE.json')
    assert replay['exact'] and replay['events'] == len(rows) == len(timing) == plan['steps']
    for name in ('pre_feedback.jsonl', 'events.json'):
        assert (life / name).read_bytes() == (replay_folder / name).read_bytes()
    clock = read_json(life / 'COMPLETE.json')
    journal = ActionJournal(life / 'pre_feedback.jsonl')
    target_map = plan['parent']['stage1_mappings'][mapping]
    stream = plan['streams'][unit_name]
    state = {'query': stream['start_query'], 'cursor': 0, 'streak': 0}
    initial_agent = load_state(checkpoint_path(plan, 'B3', seed, mapping, 48))['agent']
    assert rows[0]['packet']['pre_sha256'] == b2.b1.state_hash(initial_agent)
    initial = initial_agent['core_snapshot']
    k = initial['K'].copy()
    initial_k = load_state(checkpoint_path(plan, 'B0', seed, mapping, 0))['agent']['core_snapshot']['K']
    expected_z = array_hash(initial['Z'])
    expected_rng = hashlib.sha256(json.dumps(initial['rng_state'], sort_keys=True).encode()).hexdigest()
    w = initial['W_readout']
    tab = {}
    receipt_end = 0
    serial = []
    for age, (row, time) in enumerate(zip(rows, timing)):
        packet = row['packet']
        assert packet['cursor'] == age and packet['query'] == state['query']
        assert packet['main']['event_id'] == age + plan['cursor_start']
        assert not has_target(packet)
        assert time['receipt']['offset'] == receipt_end
        receipt_end += time['receipt']['length']
        exact(journal.read_verified(time['receipt']), packet)
        exact(state, packet['observation'])
        assert time['publish_ns'] <= time['reaction_ns'] <= time['reveal_ns'] <= time['update_ns']
        target = target_map[state['query'] - 1]
        assert row['evidence_target'] == target
        assert row['exogenous_offset'] == stream['offsets'][age]
        assert packet['action'] == np.argmax(packet['logits']) + 1
        success = packet['action'] == target
        advance = (state['query'] - 1 + stream['offsets'][age]) % 6 + 1
        outcome = {'next_query': advance if success else state['query'], 'correct': success,
                   'reward': 1. if success else -min(state['streak'] + 1, 4) / 4,
                   'next_streak': 0 if success else state['streak'] + 1}
        exact(outcome, row['reaction'])
        exact([advance, state['query']], row['counterfactual_next_queries'])
        assert (outcome['next_query'] != state['query']) == outcome['correct']
        q = state['query']
        state = {'query': outcome['next_query'], 'cursor': age + 1, 'streak': outcome['next_streak']}
        main = packet['main']
        exact([int(i == q) for i in range(8)], main['input'])
        for serialized, hashed in (('Z_pre', 'Z_pre_hash'), ('Z_post', 'Z_post_hash'),
                                   ('Z_noinput', 'Z_noinput_hash')):
            assert array_hash(np.array(main[serialized])) == main[hashed]
        assert main['Z_pre_hash'] == expected_z
        expected_z = main['Z_post_hash']
        assert main['rng_pre_hash'] == expected_rng
        assert main['rng_post_hash'] == main['rng_shadow_post_hash']
        expected_rng = main['rng_post_hash']
        assert main['K_pre_hash'] == array_hash(k)
        exact((w @ main['Z_post'])[1:5].tolist(), main['logits'])
        before_k = k.copy()
        external = packet['external']
        is_sidecar = reader in ('LIVE64', 'Z0-64')
        aligned = is_sidecar and credit == 'aligned'
        assert packet['budget'] == {'main': 16, 'main_shadow': 16, 'side': 128 * is_sidecar,
                                    'side_shadow': 128 * aligned, 'snapshots': 3,
                                    'clones': 1 + is_sidecar + aligned}
        if external:
            expected_donor = (
                initial_k if donor_mode == 'initial' else load_state(
                    root / 'lives' / f'{seed}_{(mapping + 1) % 4}' / 'Z0-64_main' / f'pre_{age:02d}.npz'
                )['agent']['core_snapshot']['K']
                if donor_mode == 'wrong' else k)
            assert external['hashes'] == dict(K=array_hash(expected_donor), B=array_hash(initial['B']),
                                              W_readout=array_hash(w))
            assert external['rng_pre'] == main['rng_pre_hash']
            exact(external['Z_start'], initial_agent['Z0'] if reader == 'Z0-64' else main['Z_pre'])
            exact(external['logits'], packet['logits'])
            exact((w @ external['Z_after'])[1:5].tolist(), external['logits'])
        else:
            expected_donor = k
        read_k_distance = float(np.linalg.norm(expected_donor - before_k))
        if reader == 'TABLE':
            target_seen = tab.get(q)
            assert packet['action'] == (target_seen if target_seen is not None else 1)
            tab[q] = target
        if aligned:
            feature = np.tanh(external['Z_after']) - np.tanh(external['Z_no_input'])
            exact(feature.tolist(), external['eligibility'])
        else:
            feature = np.tanh(main['Z_post']) - np.tanh(main['Z_noinput'])
        credit_logits = np.array(packet['logits'] if aligned else main['logits'])
        probabilities = np.exp(credit_logits - credit_logits.max())
        probabilities /= probabilities.sum()
        if not aligned:
            exact(probabilities.tolist(), main['probabilities'])
        assert row['update_source'] == ('external' if aligned else 'main')
        update = row['update']
        np.testing.assert_array_equal(feature, update['phi_event'])
        error = np.zeros(8)
        error[1:5] = -probabilities
        error[target] = 1 - probabilities[target - 1]
        np.testing.assert_array_equal(error, update['error_vector'])
        k += np.clip(plan['parent']['eta'] * np.outer(w.T @ error, feature),
                     -plan['parent']['clip'], plan['parent']['clip'])
        assert array_hash(k) == update['K_post_hash']
        assert float(np.linalg.norm(k - before_k)) == update['applied_delta_norm']
        serial.append({'seed': seed, 'mapping': mapping, 'arm': arm, 'event': age + 1, 'query': q,
                       'action': packet['action'], 'correct': int(outcome['correct']), 'reward': outcome['reward'],
                       'next_query': state['query'], 'offset': stream['offsets'][age], 'target': target,
                       'main_prediction': main['prediction'], 'update_source': row['update_source'],
                       'logits': json.dumps(packet['logits']), 'main_logits': json.dumps(main['logits']),
                       'pre_sha256': packet['pre_sha256'], 'receipt_sha256': time['receipt']['sha256'],
                       'applied_K_norm': update['applied_delta_norm'],
                       'read_K_distance': read_k_distance,
                       'Z_pre_norm': main['Z_norms'][0], 'Z_post_norm': main['Z_norms'][1],
                       'substeps': sum(packet['budget'][x] for x in ('main', 'main_shadow', 'side', 'side_shadow')),
                       'event_wall_ns': time['event_wall_ns']})
    assert receipt_end == journal.path.stat().st_size
    end = load_state(life / 'end.npz')
    assert b2.b1.state_hash(end) == b2.b1.state_hash(load_state(replay_folder / 'end.npz'))
    end_core = end['agent']['core_snapshot']
    np.testing.assert_array_equal(k, end_core['K'])
    assert expected_z == array_hash(end_core['Z'])
    assert expected_rng == hashlib.sha256(json.dumps(end_core['rng_state'], sort_keys=True).encode()).hexdigest()
    exact(state, end['environment'])
    exact(initial['B'], end_core['B'])
    exact(w, end_core['W_readout'])
    deployed = metrics([r['packet']['logits'] for r in rows], [r['evidence_target'] for r in rows])
    main_scores = metrics([r['packet']['main']['logits'] for r in rows], [r['evidence_target'] for r in rows])
    beginning = clock['retention_before']
    ending = clock['retention_after']
    accuracy = [r['correct'] for r in serial]
    summary = {
        'seed': seed, 'mapping': mapping, 'arm': arm, 'reward_total': float(sum(r['reward'] for r in serial)),
        'accuracy': deployed['accuracy'], 'nll': deployed['nll'],
        'main_accuracy': main_scores['accuracy'], 'main_nll': main_scores['nll'],
        'after_first_accuracy': float(np.mean(accuracy[1:])), 'tail_accuracy': float(np.mean(accuracy[-8:])),
        'unique_queries': len({r['query'] for r in serial}),
        'max_error_streak': max(r['reaction']['next_streak'] for r in rows),
        'switches': sum(x['query'] != x['next_query'] for x in serial),
        'retention_live_acc': ending['live8']['accuracy'], 'retention_live_nll': ending['live8']['nll'],
        'retention_reset_acc': ending['Z0_8']['accuracy'], 'retention_reset_nll': ending['Z0_8']['nll'],
        'retention_live_acc_delta': ending['live8']['accuracy'] - beginning['live8']['accuracy'],
        'retention_reset_acc_delta': ending['Z0_8']['accuracy'] - beginning['Z0_8']['accuracy'],
        'delta_K_norm': clock['delta_K_norm'], 'final_Z_norm': float(np.linalg.norm(end_core['Z'])),
        'applied_K_path_sum': sum(x['applied_K_norm'] for x in serial),
        'mean_read_K_distance': float(np.mean([x['read_K_distance'] for x in serial])),
        'episode_max_boundary_Z': max(max(x['Z_pre_norm'], x['Z_post_norm']) for x in serial),
        'max_state_norm': clock['max_state_norm'], 'total_substeps': sum(x['substeps'] for x in serial),
        'probe_substeps': beginning['substeps'] + ending['substeps'],
        'wall_seconds': clock['wall_seconds'], 'cpu_seconds': clock['cpu_seconds'],
    }
    return serial, summary


def meets(result, threshold):
    return result['mean'] >= threshold and result['ci95'][0] > 0


def first_event_controls(root, plan):
    pairs = {arm: [] for arm in b2.CONTROLS}
    for seed, mapping in plan['units']:
        folder = root / 'lives' / f'{seed}_{mapping}'
        packets = {arm: read_json(folder / arm / 'events.json')[0]['packet'] for arm in plan['order']}
        for reader in b2.READERS:
            main, aligned = (packets[f'{reader}_{name}'] for name in ('main', 'aligned'))
            for key in ('main', 'query', 'observation', 'logits', 'action', 'pre_sha256'):
                exact(main[key], aligned[key])
        own = packets['Z0-64_main']
        for arm, results in pairs.items():
            donor = packets[arm]
            for key in ('main', 'query', 'observation', 'pre_sha256'):
                exact(own[key], donor[key])
            a, b = own['external'], donor['external']
            for key in ('Z_start', 'rng_pre', 'rng_post'):
                exact(a[key], b[key])
            for key in ('B', 'W_readout'):
                exact(a['hashes'][key], b['hashes'][key])
            results.append({'changed_K': a['hashes']['K'] != b['hashes']['K'],
                            'changed_action': own['action'] != donor['action'],
                            'abs_delta_logits_max': float(np.max(np.abs(np.array(a['logits']) - b['logits'])))})
    return {arm: {key: paired([r[key] for r in data], plan) for key in data[0]}
            for arm, data in pairs.items()}


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('output', type=Path)
    cli.add_argument('--history', type=Path)
    cli.add_argument('--b1', type=Path)
    cli.add_argument('--verify-only', action='store_true')
    args = cli.parse_args()
    root = args.output.resolve()
    plan = verify_provenance(root, read_json(root / 'PLAN.json'), args.history, args.b1)
    event_rows, life_rows = [], []
    for seed, mapping in plan['units']:
        for arm in plan['order']:
            rows, life = verify_trajectory(root, plan, seed, mapping, arm)
            event_rows.extend(rows)
            life_rows.append(life)
        life_root = root / 'lives' / f'{seed}_{mapping}'
        exact(read_json(life_root / 'CARRY_main' / 'events.json'),
              read_json(life_root / 'CARRY_aligned' / 'events.json'))
        exact(load_state(life_root / 'CARRY_main' / 'end.npz'),
              load_state(life_root / 'CARRY_aligned' / 'end.npz'))
    first_unit = '_'.join(map(str, plan['units'][0]))
    for arm in plan['order']:
        original = root / 'lives' / first_unit / arm
        continued = root / 'resume' / 'lives' / first_unit / arm
        resume = read_json(continued / 'COMPLETE.json')
        assert resume['exact'] and resume['events'] == 12
        assert (original / 'pre_feedback.jsonl').read_bytes().splitlines(keepends=True)[12:] == (
            continued / 'pre_feedback.jsonl').read_bytes().splitlines(keepends=True)
        assert b2.b1.state_hash(load_state(original / 'end.npz')) == b2.b1.state_hash(
            load_state(continued / 'end.npz'))
    rows = {arm: [x for x in life_rows if x['arm'] == arm] for arm in plan['order']}
    columns = [k for k in life_rows[0] if k not in ('seed', 'mapping', 'arm')]
    stats = {arm: {k: paired([r[k] for r in data], plan) for k in columns} for arm, data in rows.items()}

    def contrast(first, second):
        return {key: paired([a[key] - b[key] for a, b in zip(rows[first], rows[second], strict=True)], plan)
                for key in columns}

    pairs = [(a, b) for a, b in [
        ('Z0-64_main', 'CARRY_main'), ('Z0-64_main', 'LIVE64_main'),
        ('Z0-64_main', 'Z0-64_K0'), ('Z0-64_main', 'Z0-64_WRONG'),
        ('Z0-64_aligned', 'CARRY_aligned'), ('Z0-64_aligned', 'LIVE64_aligned'),
        ('Z0-64_aligned', 'TABLE'),
        *[(name + '_aligned', name + '_main') for name in b2.READERS]]]
    contrasts = {f'{a} minus {b}': contrast(a, b) for a, b in pairs}
    interaction = {}
    for key in ('reward_total', 'main_accuracy', 'retention_live_acc'):
        interaction[key] = paired([
            (x[key] - y[key]) - (u[key] - v[key])
            for x, y, u, v in zip(rows['Z0-64_aligned'], rows['Z0-64_main'],
                                  rows['LIVE64_aligned'], rows['LIVE64_main'], strict=True)], plan)
    table_ok = min(x['accuracy'] for x in rows['TABLE']) >= plan['table_gate']
    return_ok = all(meets(contrasts[f'Z0-64_main minus {name}']['reward_total'], plan['threshold_return'])
                    for name in ('CARRY_main', 'LIVE64_main'))
    selective = all(meets(contrasts[f'Z0-64_main minus {name}']['reward_total'], plan['threshold_return'])
                    for name in ('Z0-64_K0', 'Z0-64_WRONG'))
    own_learning = meets(contrasts['Z0-64_aligned minus Z0-64_main']['retention_live_acc'],
                         plan['main_retention_gain'])
    output = {'engineering': 'B2A_VALID' if table_ok else 'B2A_INVALID_TASK_CONTROL',
              'scientific': 'ACTION_DEPENDENT_DEPLOYED_SIGNAL' if table_ok and return_ok and selective
              else 'NO_CONFIRMED_ACTION_DEPENDENT_GAIN',
              'table_gate': table_ok, 'reward_gate': return_ok, 'selectivity_gate': selective,
              'main_learning_gate': own_learning, 'trajectories_replayed_exact': len(life_rows),
              'action_records_verified': len(event_rows), 'pending_resume_exact': len(plan['order']),
              'carry_policy_equal_exact': len(plan['units']),
              'first_event_K_only': first_event_controls(root, plan),
              'stats': stats, 'contrasts': contrasts,
              'reader_credit_interaction_Z0_vs_LIVE': interaction}
    if args.verify_only:
        exact(output, read_json(root / 'ANALYSIS.json'))
    else:
        atomic_json(root / 'ANALYSIS.json', output)
        write_csv(root / 'per_event.csv', event_rows)
        write_csv(root / 'per_life.csv', life_rows)
    print(json.dumps({k: v for k, v in output.items() if k not in ('stats', 'contrasts', 'first_event_K_only')},
                     indent=2))


if __name__ == '__main__':
    main()
