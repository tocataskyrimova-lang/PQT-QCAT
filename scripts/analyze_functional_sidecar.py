"""Independently verify action commitments and report paired B1 outcomes."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pqt_qcat.memory import functional_sidecar as b1
from pqt_qcat.memory.access_state import exact
from pqt_qcat.memory.checkpoint_io import load_state
from pqt_qcat.memory.codex_memory_location import atomic_json
from pqt_qcat.memory.qnuf import array_hash
from analyze_access_bridge import paired
import numpy as np


def score(logits, target):
    logits = np.array(logits)
    index = target - 1
    margin = logits[index] - np.max(np.delete(logits, index))
    return float(np.logaddexp.reduce(logits) - logits[index]), float(margin)


def has_target(value):
    """Reject supervised feedback fields anywhere in the pre-feedback payload."""
    if isinstance(value, dict):
        return any(k in ('target', 'label', 'reward', 'feedback', 'y_true') or has_target(v)
                   for k, v in value.items())
    return isinstance(value, list) and any(has_target(v) for v in value)


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_life(plan, path, arm, seed, mapping, pre_states, wrong, truth):
    folder = path / arm
    rows = b1.read_json(folder / 'events.json')
    assert len(rows) == 24
    life = b1.read_json(folder / 'life.json')
    assert life['identity_exact']
    journal = b1.ActionJournal(folder / 'actions_before_feedback.jsonl')
    initial_k = load_state(b1.checkpoint_path(plan, 'B0', seed, mapping, 0))['agent']['core_snapshot']['K']
    start = load_state(folder / 'checkpoint_048.npz')['agent']
    final = load_state(folder / 'checkpoint_072.npz')['agent']
    exact(start, pre_states[0])
    pending = load_state(folder / 'pending_057.npz')
    assert pending['runner']['cursor'] == 56
    exact(journal.read_verified(pending['runner']['receipt']), rows[8]['packet'])
    exact(pending['agent']['qnuf']['pending']['prediction'], rows[8]['packet']['main'])
    spec = plan['arms'][arm]
    expected_offset = 0
    events = []
    for pos, (row, pre) in enumerate(zip(rows, pre_states)):
        packet = row['packet']
        num = pos + 48
        reference = truth[num]
        assert packet['event_id'] == num and packet['query'] == reference['query']
        assert not has_target(packet)
        assert row['receipt']['offset'] == expected_offset
        exact(packet, journal.read_verified(row['receipt']))
        expected_offset += row['receipt']['length']
        assert row['committed_ns'] < row['feedback']['label_reveal_ns']
        exact({**packet['main'], **row['update']}, reference)
        assert b1.state_hash(pre) == packet['pre_state_sha256']
        target = row['feedback']['target']
        assert target == reference['target']
        action = int(np.argmax(packet['deployed_logits'])) + 1
        assert action == packet['deployed_action']
        assert row['feedback']['reward'] == int(action == target)
        read = packet['sidecar_or_padding']
        expected_k = initial_k if spec['donor'] == 'initial' else (
            wrong[1][num] if spec['donor'] == 'wrong' else pre['core_snapshot']['K'])
        assert read['K_hash'] == array_hash(expected_k)
        expected_z = pre['Z0'] if spec['start'] == 'Z0' else pre['core_snapshot']['Z']
        assert read['Z_start_hash'] == array_hash(np.array(expected_z))
        assert read['rng_pre_hash'] == packet['main']['rng_pre_hash']
        assert packet['substeps_main'] == packet['substeps_shadow'] == 16
        assert packet['substeps_sidecar'] == 2 * spec['ticks']
        assert packet['substeps_padding'] == 2 * spec['padding']
        if spec['ticks']:
            exact(read['logits'], packet['deployed_logits'])
        else:
            exact(packet['main']['logits'], packet['deployed_logits'])
        nll, margin = score(packet['deployed_logits'], target)
        events.append({
            'seed': seed, 'mapping': mapping, 'arm': arm, 'event': num + 1, 'query': packet['query'],
            'deployed_action': action, 'main_prediction': packet['main']['prediction'], 'target': target,
            'reward': row['feedback']['reward'], 'main_correct': int(reference['correct']),
            'deployed_nll': nll, 'main_nll': reference['nll'], 'deployed_margin': margin,
            'main_margin': reference['margin'], 'deployed_logits': json.dumps(packet['deployed_logits']),
            'main_logits': json.dumps(packet['main']['logits']), 'pre_state_sha256': packet['pre_state_sha256'],
            'pending_sha256': packet['main_pending_sha256'], 'sidecar_k_sha256': read['K_hash'],
            'receipt_sha256': row['receipt']['sha256'], 'committed_ns': row['committed_ns'],
            'label_reveal_ns': row['feedback']['label_reveal_ns'],
            **{k: packet[k] for k in ('substeps_main', 'substeps_sidecar', 'substeps_padding', 'substeps_shadow',
                                     'core_clones', 'main_snapshot_copies')},
            **row['cost_ns'],
        })
    assert expected_offset == journal.path.stat().st_size
    actual = load_state(b1.checkpoint_path(plan, 'B3', seed, mapping, 72))['agent']
    exact(final, actual)
    aggregate = {'seed': seed, 'mapping': mapping, 'arm': arm,
                 'deployed_accuracy': float(np.mean([r['reward'] for r in events])),
                 'main_accuracy': float(np.mean([r['main_correct'] for r in events])),
                 'deployed_nll': float(np.mean([r['deployed_nll'] for r in events])),
                 'main_nll': float(np.mean([r['main_nll'] for r in events])),
                 'deployed_margin': float(np.mean([r['deployed_margin'] for r in events])),
                 'main_margin': float(np.mean([r['main_margin'] for r in events])),
                 'reward_total': sum(r['reward'] for r in events),
                 'exact_final': True, 'full_snapshot_copies_recorded': life['snapshot_copies'],
                 'checkpoint_writes': len(life['checkpoint_io']),
                 'checkpoint_bytes': sum(x['bytes'] for x in life['checkpoint_io']),
                 'checkpoint_io_ns': sum(x['ns'] for x in life['checkpoint_io']),
                 'archival_checkpoint_loads': 3}
    return events, aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    plan = b1.read_json(out / 'PLAN.json')
    b1.verify(out, plan)
    p = b1.read_json(out / 'RESUME_CHECKS.json')
    assert p['separate_process'] and len(p['checks']) == 12 and all(r['exact'] for r in p['checks'])
    events, life_rows, block_costs = [], [], []
    for seed, mapping in plan['units']:
        path = out / 'blocks' / f'{seed}_{mapping}'
        block_costs.append(b1.read_json(path / 'COMPLETE.json'))
        pre = [load_state(path / 'pre_event' / f'pre_{i:03d}.npz')['agent'] for i in range(49, 73)]
        wrong = b1.wrong_trajectory(plan, seed, mapping)
        truth = b1.archived_events(plan, seed, mapping)
        for arm in plan['arm_order']:
            e, r = verify_life(plan, path, arm, seed, mapping, pre, wrong, truth)
            events.extend(e)
            life_rows.append(r)
    results = out / 'results' / 'stage_b1'
    results.mkdir(parents=True, exist_ok=True)
    write_csv(results / 'per_event.csv', events)
    write_csv(results / 'per_life.csv', life_rows)
    main_metrics = ('deployed_accuracy', 'main_accuracy', 'deployed_nll', 'main_nll',
                    'deployed_margin', 'main_margin', 'reward_total')
    by_arm = {a: [r for r in life_rows if r['arm'] == a] for a in plan['arm_order']}
    scores, confusion, cost, series, segments = {}, {}, {}, {}, {}
    for arm, rows in by_arm.items():
        active = [r for r in events if r['arm'] == arm]
        scores[arm] = {key: paired([r[key] for r in rows], plan) for key in main_metrics}
        scores[arm]['correct_count'] = sum(r['reward'] for r in active)
        scores[arm]['decisions'] = len(active)
        conf = np.zeros((4, 4), dtype=int)
        for r in active:
            conf[r['target'] - 1, r['deployed_action'] - 1] += 1
        confusion[arm] = conf.tolist()
        series[arm] = {str(t): float(np.mean([r['reward'] for r in active if r['event'] == t]))
                       for t in range(49, 73)}
        segments[arm] = {f'{start}-{start + 7}': float(np.mean([
            r['reward'] for r in active if start <= r['event'] < start + 8])) for start in (49, 57, 65)}
        totals = {k: sum(r[k] for r in active)
                  for k in ('substeps_main', 'substeps_shadow', 'substeps_sidecar', 'substeps_padding',
                            'core_clones', 'main_snapshot_copies')}
        useful = sum(totals[k] for k in ('substeps_main', 'substeps_shadow', 'substeps_sidecar'))
        hits = scores[arm]['correct_count']
        cost[arm] = {**totals, 'useful_substeps': useful, 'audited_substeps': useful + totals['substeps_padding'],
                     'substeps_per_correct': useful / hits if hits else None,
                     'checkpoints': {k: sum(r[k] for r in rows) for k in (
                         'checkpoint_writes', 'checkpoint_bytes', 'checkpoint_io_ns', 'archival_checkpoint_loads',
                         'full_snapshot_copies_recorded')}}
        for key in ('main_predict_ns', 'sidecar_ns', 'padding_ns', 'snapshot_setup_ns', 'guard_ns',
                    'commit_io_ns', 'compute_action_ns', 'audited_action_ns'):
            vals = np.array([r[key] for r in active]) / 1e6
            cost[arm][key.replace('_ns', '_ms')] = {'mean': float(vals.mean()),
                                                    'p95': float(np.quantile(vals, .95))}
    contrasts = {arm: {metric: paired(
        [a[metric] - b[metric] for a, b in zip(by_arm['S-Z0-64'], by_arm[arm], strict=True)], plan)
        for metric in main_metrics} for arm in by_arm if arm != 'S-Z0-64'}
    accuracy = {arm: statistics['deployed_accuracy'] for arm, statistics in contrasts.items()}
    bound = plan['threshold_pp'] / 100
    primary = accuracy['S-LIVE64']
    signal = (primary['mean'] >= bound and primary['ci95'][0] > 0
              and all(accuracy[k]['mean'] >= bound for k in ('S-Z0-64-K0', 'S-Z0-64-WRONG')))
    clock = {'worker_cpu_seconds': sum(x['cpu_seconds'] for x in block_costs),
             'wall_seconds': b1.read_json(out / 'STATUS.json')['wall_seconds'],
             'shared_wrong_donor_matrices': 24 * 18, 'shared_wrong_donor_start_loads': 18,
             'probe_or_guard_integration_substeps': 0}
    atomic_json(results / 'contrasts.json', {'arms': scores, 'S-Z0-64_minus': contrasts, 'segments': segments,
                                           'per_event_accuracy': series, 'class_confusion': confusion})
    atomic_json(results / 'compute_budget.json', {'arms': cost, **clock})
    atomic_json(results / 'continuity.json', {'engineering': 'B1_VALID', 'blocks': len(block_costs),
                                            'exact_main_lives': len(life_rows),
                                            'verified_pre_feedback_records': len(events),
                                            'resume_checks': p, 'main_learning': 'UNCHANGED'})
    atomic_json(results / 'negative_controls.json', {
        'wrong_uses_same_age_and_seed': True,
        'contrasts': {arm: contrasts[arm] for arm in ('S-Z0-64-K0', 'S-Z0-64-WRONG')},
        'main_padding_matched': cost['M-LIVE8']['audited_substeps'] == cost['S-Z0-64']['audited_substeps'],
        'scientific': 'B1_DEPLOYED_SIGNAL' if signal else 'NO_REPEATED_DEPLOYED_GAIN'})
    print(json.dumps({'engineering': 'B1_VALID', 'scientific': 'B1_DEPLOYED_SIGNAL' if signal
                      else 'NO_REPEATED_DEPLOYED_GAIN', 'primary': primary, 'events_verified': len(events)}, indent=2))


if __name__ == '__main__':
    main()
