"""Verify matched budgets and reconstruct plasticity from committed event records."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pqt_qcat.memory.checkpoint_io import load_state
from pqt_qcat.memory.qnuf import array_hash


def query_geometry(rows, name):
    features = np.array([r[name] for r in rows])
    queries = np.array([r['query'] for r in rows])
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    unit = features / np.maximum(norms, 1e-15)
    cosine = unit @ unit.T
    i, j = np.triu_indices(len(rows), k=1)
    same = queries[i] == queries[j]
    if not same.any() or same.all():
        return None
    return float(cosine[i[same], j[same]].mean() - cosine[i[~same], j[~same]].mean())


def feature_energy_fraction(rows, name, start, stop):
    features = np.array([row[name] for row in rows])
    fraction = np.sum(features[:, start:stop] ** 2, axis=1)
    total = np.sum(features ** 2, axis=1)
    return float(np.mean(fraction / np.maximum(total, 1e-30)))


def verify_life(folder, plan):
    result = json.loads((folder / 'result.json').read_text())
    rows = json.loads((folder / 'event_records.json').read_text())
    initial = load_state(folder / 'checkpoint_000.npz')['agent']
    final = load_state(folder / f"checkpoint_{len(rows):03d}.npz")['agent']
    start, end = initial['core_snapshot'], final['core_snapshot']
    assert result['complete'] and result['events'] == len(rows)
    assert all(np.array_equal(start[key], end[key]) for key in ('B', 'W_readout'))
    assert all(start[key] == end[key] for key in ('config', 'vortex_params', 'integrator_config', 'substeps'))
    journal = {}
    for path in sorted(folder.glob('predictions_resume_*.jsonl')):
        for line in path.read_text().splitlines():
            pred = json.loads(line)
            assert 'target' not in pred and 'evidence_source' not in pred
            previous = journal.setdefault(pred['event_id'], pred)
            assert previous == pred
    k = start['K'].copy()
    continuity, hits = 0, 0
    previous = None
    seen = {}
    lookup_scores = []
    for index, row in enumerate(rows):
        assert row['event_id'] == index
        assert all(row[key] == value for key, value in journal[index].items())
        assert row['K_pre_hash'] == array_hash(k)
        for name in ('Z_pre', 'Z_post', 'Z_noinput'):
            assert row[f'{name}_hash'] == array_hash(np.array(row[name]))
        assert row['rng_post_hash'] == row['rng_shadow_post_hash']
        assert row['prediction'] == 1 + np.argmax(row['logits'])
        assert row['decision'] == 'COMMITTED_UNCHECKED'
        assert row['target'] == result['targets'][row['query'] - 1]
        assert row['correct'] == (row['target'] == row['prediction'])
        hits += row['correct']
        if result['stage'] == 'stage1' and result['arm'] != 'RESET' and previous is not None:
            assert row['Z_pre_hash'] == previous
            continuity += 1
        previous = row['Z_post_hash']
        expected_phi = np.tanh(row['Z_post']) - np.tanh(row['Z_noinput'])
        np.testing.assert_array_equal(expected_phi, row['phi_event'])
        delta = np.clip(plan['eta'] * np.outer(start['W_readout'].T @ row['error_vector'],
                                             row['phi_' + row['phi_kind']]),
                        -plan['clip'], plan['clip'])
        norm = np.linalg.norm(delta)
        if result['cap'] is not None and norm > result['cap']:
            delta *= result['cap'] / norm
        k += delta
        assert array_hash(k) == row['K_proposed_hash'] == row['K_post_hash']
        # Independent positive control: remember only the previous observed label per query.
        lookup_scores.append(seen.get(row['query'], 1) == row['target'])
        seen[row['query']] = row['target']
    assert np.array_equal(k, end['K'])
    assert len(final['qnuf']['witness_ledger']) == len(rows) == result['update_candidates']
    assert hits / len(rows) == result['online']
    steps = len(rows) * plan['query_ticks'] * plan['substeps']
    assert result['shadow_substeps'] == steps
    assert result['live_substeps'] - result['neutral_substeps'] == steps
    late = rows[-24:]
    return {
        'stage': result['stage'], 'arm': result['arm'], 'seed': result['seed'], 'map': result['map_id'],
        'events': len(rows), 'continuity_count': continuity,
        'stream': [(r['query'], r['target'], r['rng_pre_hash'], r['rng_post_hash']) for r in rows],
        'initial_weights': [array_hash(start[key]) for key in ('K', 'B', 'W_readout')],
        'budget': [result[k] for k in ('live_substeps','shadow_substeps','eval_substeps','update_candidates')],
        'query_similarity_gap_legacy': query_geometry(late, 'phi_legacy'),
        'query_similarity_gap_event': query_geometry(late, 'phi_event'),
        'legacy_E1_fraction': feature_energy_fraction(late, 'phi_legacy', 0, 8),
        'event_E1_fraction': feature_energy_fraction(late, 'phi_event', 0, 8),
        'first_legacy_phi_norm': float(np.linalg.norm(rows[0]['phi_legacy'])),
        'first_event_phi_norm': float(np.linalg.norm(rows[0]['phi_event'])),
        'late_class_count': len({r['prediction'] for r in late}),
        'lookup_online': float(np.mean(lookup_scores)),
        'pause_change': result['probe']['pause']['accuracy'] - result['probe']['live']['accuracy'],
        'checkpoint0_live': load_state(folder / 'checkpoint_000.npz')['runner']['probes']['0']['live']['accuracy'],
        'max_energy': result['max_live_energy'],
        'cpu': result['cpu_seconds'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    root = args.run
    plan = json.loads((root / 'frozen_config.json').read_text())
    by_pair = defaultdict(list)
    by_arm = defaultdict(list)
    for stage in ('stage0', 'stage1'):
        for folder in sorted((root / stage / 'lives').glob('*')):
            life = verify_life(folder, plan)
            by_pair[stage, life['seed'], life['map']].append(life)
            by_arm[stage, life['arm']].append(life)
    for group in by_pair.values():
        for key in ('stream', 'initial_weights', 'budget'):
            assert all(life[key] == group[0][key] for life in group)
    summary = {}
    for (stage, arm), group in by_arm.items():
        data = {'lives': len(group), 'events': sum(life['events'] for life in group),
                'exact_continuity_boundaries': sum(life['continuity_count'] for life in group),
                'equal_budget_vector_per_life': group[0]['budget']}
        for key in ('query_similarity_gap_legacy', 'query_similarity_gap_event', 'late_class_count',
                    'lookup_online', 'pause_change', 'checkpoint0_live', 'max_energy', 'cpu',
                    'legacy_E1_fraction', 'event_E1_fraction', 'first_legacy_phi_norm', 'first_event_phi_norm'):
            values = [life[key] for life in group if life[key] is not None]
            data[key + '_mean'] = float(np.mean(values)) if values else None
        summary.setdefault(stage, {})[arm] = data
    (root / 'verification.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
