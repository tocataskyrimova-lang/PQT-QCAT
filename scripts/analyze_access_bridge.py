"""Summarize and independently check the frozen ACCESS-BRIDGE experiment."""
import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import RidgeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pqt_qcat.memory.access_bridge import HORIZONS, file_hash, score_rows
from pqt_qcat.memory.checkpoint_io import load_state
from pqt_qcat.memory.codex_memory_location import atomic_json
from pqt_qcat.memory.qnuf import array_hash


@lru_cache
def bootstrap_indices(seeds, maps, random_seed, repeats):
    rng = np.random.default_rng(random_seed)
    seed_indices, map_indices = [], []
    for _ in range(repeats):
        seed_indices.append(rng.integers(0, seeds, seeds))
        map_indices.append(rng.integers(0, maps, maps))
    return np.array(seed_indices)[:, :, None], np.array(map_indices)[:, None, :]


def paired(values, plan):
    array = np.array(values, dtype=float)
    count = int(np.isfinite(array).sum())
    if not count:
        return {'mean': None, 'blocks': 0, 'ci95': None}
    seeds = len(plan['parent']['stage1_seeds'])
    matrix = array.reshape(seeds, -1)
    seed_means = [float(row[np.isfinite(row)].mean()) if np.isfinite(row).any() else None
                  for row in matrix]
    interval = None
    if count == array.size:
        indices = bootstrap_indices(*matrix.shape, plan['bootstrap_seed'], plan['bootstrap_replicates'])
        bootstrap = matrix[indices].mean(axis=(1, 2))
        interval = np.quantile(bootstrap, [.025, .975]).tolist()
    return {'mean': float(np.nanmean(array)), 'blocks': count, 'seed_means': seed_means,
            'ci95': interval}


def scalar(scores, metric):
    return np.nan if scores is None else scores[metric]


def mechanistic_pair(after, baseline, horizon):
    tag = str(horizon)
    if any(tag not in r['horizons'] for r in [*after, *baseline]):
        return {'E5_l2': None, 'logits_linf': None, 'argmax_changed': None}
    z = np.array([row['horizons'][tag]['Z'][32:40] for row in after])
    z_initial = np.array([row['horizons'][tag]['Z'][32:40] for row in baseline])
    logits = np.array([row['horizons'][tag]['logits'] for row in after])
    baseline_logits = np.array([row['horizons'][tag]['logits'] for row in baseline])
    return {'E5_l2': float(np.linalg.norm(z - z_initial, axis=1).mean()),
            'logits_linf': float(np.max(np.abs(logits - baseline_logits), axis=1).mean()),
            'argmax_changed': float(np.mean(logits.argmax(axis=1) != baseline_logits.argmax(axis=1)))}


def verify_norm(folder, name, plan):
    subdir = folder / name
    data = [json.loads((subdir / f'{kind}_events.json').read_text()) for kind in ('legacy', 'event')]
    journal = [json.loads(line) for line in (subdir / 'pre_feedback.jsonl').read_text().splitlines()]
    checkpoints = [load_state(subdir / f'{kind}_start.npz')['agent']['core_snapshot']
                   for kind in ('legacy', 'event')]
    for key in ('K', 'Z', 'B', 'W_readout'):
        np.testing.assert_array_equal(checkpoints[0][key], checkpoints[1][key])
    assert len(data[0]) == len(data[1]) == len(journal) == plan['norm_events']
    assert data[0][0]['logits'] == data[1][0]['logits']
    norm_gap = 0.
    for index, (legacy, event) in enumerate(zip(*data)):
        assert legacy['query'] == event['query'] and legacy['target'] == event['target']
        assert legacy['rng_pre_hash'] == event['rng_pre_hash']
        norm_gap = max(norm_gap, abs(legacy['applied_delta_norm'] - event['applied_delta_norm']))
        for branch, row in enumerate((legacy, event)):
            assert 'target' not in journal[index][branch]
            assert all(row[key] == value for key, value in journal[index][branch].items())
            if index:
                assert row['Z_pre_hash'] == data[branch][index - 1]['Z_post_hash']
    for rows, snapshot, kind in zip(data, checkpoints, ('legacy', 'event')):
        k = snapshot['K'].copy()
        for row in rows:
            assert array_hash(k) == row['K_pre_hash']
            e = snapshot['W_readout'].T @ np.array(row['error_vector'])
            phi = np.array(row[f'phi_{kind}'])
            outer = np.outer(e, phi)
            eta = row['radius'] / np.linalg.norm(outer) if row['radius'] else 0.
            k += np.clip(eta * outer, -plan['parent']['clip'], plan['parent']['clip'])
            assert array_hash(k) == row['K_post_hash']
        end = load_state(subdir / f'{kind}_end.npz')['agent']['core_snapshot']
        np.testing.assert_array_equal(end['K'], k)
        np.testing.assert_array_equal(end['B'], snapshot['B'])
        np.testing.assert_array_equal(end['W_readout'], snapshot['W_readout'])
        assert end['step_count'] - snapshot['step_count'] == (
            plan['norm_events'] * plan['parent']['query_ticks'] * plan['parent']['substeps'])
    assert norm_gap < 1e-12
    return norm_gap


def norm_reader(output, units, targets, label, branch, available):
    """Same fixed-alpha, doubly excluded development decoder as Stage 1."""
    indices = np.flatnonzero(available)
    panels = []
    for index in indices:
        seed, mapping = units[index]
        directory = output / f'{seed}_{mapping}' / label
        start, end = [load_state(directory / f'{branch}_{timepoint}.npz')
                      ['agent']['core_snapshot']['K'][32:35].reshape(-1)
                      for timepoint in ('start', 'end')]
        panels.append([start, end, end - start])
    if not panels:
        return None
    x = np.array(panels)
    y = np.array(targets)[indices]
    predicted = np.zeros((len(indices), 3, y.shape[1]), dtype=int)
    covered = []
    for i, index in enumerate(indices):
        seed, mapping = units[index]
        train = [j for j, k in enumerate(indices)
                 if units[k][0] != seed and units[k][1] != mapping]
        if not train:
            return None
        for q in range(y.shape[1]):
            labels = y[train, q]
            classes = np.unique(labels)
            covered.append(y[i, q] in classes)
            for feature in range(x.shape[1]):
                if len(classes) == 1:
                    predicted[i, feature, q] = classes[0]
                else:
                    classifier = RidgeClassifier(alpha=1).fit(x[train, feature], labels)
                    predicted[i, feature, q] = classifier.predict(x[i:i+1, feature])[0]
    scores = np.mean(predicted == y[:, None, :], axis=2)
    return {'accuracy': dict(zip(('start', 'end', 'delta'), scores.mean(axis=0).tolist())),
            'per_block_accuracy': scores.tolist(), 'prediction': predicted.tolist(),
            'units': [units[index] for index in indices], 'target_classes_covered': float(np.mean(covered))}


def analyze(output, historical_source=None):
    plan = json.loads((output / 'PLAN.json').read_text())
    source = historical_source or Path(plan['source_path'])
    manifest = json.loads((source / 'sha256_manifest.json').read_text())
    for path, expected in plan['historical_files'].items():
        assert file_hash(source / path) == expected
    reads, norms, targets, initial_snapshots = [], [], [], []
    counters = Counter()
    norm_gap = 0.
    for seed, mapping in plan['units']:
        folder = output / f'{seed}_{mapping}'
        snapshot_path = source / 'stage1' / 'lives' / f'B0_{seed}_{mapping}'
        initial = load_state(snapshot_path / 'checkpoint_000.npz')['agent']['core_snapshot']
        initial_snapshots.append(initial)
        reads.append(json.loads((folder / 'readouts.json').read_text()))
        norms.append(json.loads((folder / 'norm_results.json').read_text()))
        targets.append(plan['parent']['stage1_mappings'][mapping])
        counters['cpu_seconds'] += json.loads((folder / 'complete.json').read_text())['cpu_seconds']
        for arm in ('B1', 'B3'):
            assert reads[-1][f'{arm}_live/{arm}'] == reads[-1][f'{arm}_live/{arm}_sham']
            archived_result = source / 'stage1' / 'lives' / f'{arm}_{seed}_{mapping}' / 'result.json'
            assert file_hash(archived_result) == manifest[archived_result.relative_to(source).as_posix()]
            reference = json.loads(archived_result.read_text())['probe']
            for recipient, mode in ((f'{arm}_live', 'live'), ('Z0', 'reset')):
                observed = [row['horizons']['8']['logits'] for row in reads[-1][f'{recipient}/{arm}']]
                np.testing.assert_array_equal(observed, reference[mode]['logits'])
                counters['exact_historical_logit_vectors'] += len(observed)
        for name, group in reads[-1].items():
            recipient = name.split('/')[0]
            paired_initial = reads[-1][f'{recipient}/initial']
            for query, row in enumerate(group):
                counters['frozen_trajectories'] += 1
                counters['frozen_substeps'] += row['substeps']
                counters['frozen_failures'] += int(row['failure'] is not None)
                if row['failure'] is None:
                    assert row['substeps'] == HORIZONS[-1] * plan['parent']['substeps']
                for horizon, result in row['horizons'].items():
                    np.testing.assert_array_equal((initial['W_readout'] @ result['Z'])[1:5], result['logits'])
                    reference_tick = paired_initial[query]['horizons'].get(horizon)
                    assert reference_tick is None or reference_tick['rng_hash'] == result['rng_hash']
        for name, pair in norms[-1].items():
            if 'failure' in pair:
                counters['norm_failed_pairs'] += 1
                continue
            norm_gap = max(norm_gap, verify_norm(folder, name, plan))
            counters['norm_pairs'] += 1
            for arm in pair.values():
                counters['norm_event_updates'] += len(arm['update_norms'])
                counters['norm_zero_updates'] += sum(n == 0 for n in arm['update_norms'])
                counters['norm_carry_boundaries'] += len(arm['update_norms']) - 1
                counters['norm_train_substeps'] += arm['training_substeps']
                counters['norm_shadow_substeps'] += arm['shadow_substeps']
                for row in arm['readouts']:
                    counters['norm_readout_substeps'] += row['substeps']
                    counters['norm_readout_failures'] += int(row['failure'] is not None)

    # The same Z0/B/W/RNG is shared among mappings of a seed. A run of donor
    # K from the next mapping is therefore the exact wrong-K intervention at
    # Z0, evaluated against this mapping's targets without another integration.
    unit_index = {tuple(pair): i for i, pair in enumerate(plan['units'])}
    map_count = len(plan['parent']['stage1_mappings'])
    for i, (seed, mapping) in enumerate(plan['units']):
        j = unit_index[seed, (mapping + 1) % map_count]
        a, b = initial_snapshots[i], initial_snapshots[j]
        for field in ('Z', 'B', 'W_readout', 'gamma_diag_normal', 'gamma_diag_hold'):
            np.testing.assert_array_equal(a[field], b[field])
        for field in ('config', 'integrator_config', 'vortex_params', 'cell_params', 'rng_state', 'substeps'):
            assert a[field] == b[field]
        for arm in ('B1', 'B3'):
            reads[i][f'Z0/{arm}_wrong'] = reads[j][f'Z0/{arm}']
            counters['reused_exact_Z0_wrong_trajectories'] += len(reads[i][f'Z0/{arm}_wrong'])

    read_metrics = {}
    for name in reads[0]:
        read_metrics[name] = {str(h): [score_rows(r[name], y, h) for r, y in zip(reads, targets)]
                              for h in HORIZONS}
    curves = {label: {h: {metric: paired([scalar(v, metric) for v in values], plan)
                           for metric in ('accuracy', 'nll', 'margin')}
                      for h, values in by_horizon.items()} for label, by_horizon in read_metrics.items()}
    interventions = {}
    for label, by_horizon in read_metrics.items():
        rec, donor = label.split('/')
        if donor == 'initial' or donor.endswith('_sham'):
            continue
        comparisons = [f'{rec}/initial']
        if donor in ('B1', 'B3') and rec in (f'{donor}_live', 'Z0'):
            comparisons.append(f'{rec}/{donor}_wrong')
        for ref in comparisons:
            key = f'{label} minus {ref}'
            per_horizon = {}
            delta_by_horizon = {}
            for horizon in HORIZONS:
                tag = str(horizon)
                delta = {metric: [scalar(a, metric) - scalar(b, metric)
                                  for a, b in zip(by_horizon[tag], read_metrics[ref][tag])]
                         for metric in ('accuracy', 'nll', 'margin')}
                delta_by_horizon[horizon] = delta
                table = {metric: paired(val, plan) for metric, val in delta.items()}
                mechanical = [mechanistic_pair(r[label], r[ref], horizon) for r in reads]
                for metric in ('E5_l2', 'logits_linf', 'argmax_changed'):
                    table[metric] = paired([v[metric] if v[metric] is not None else np.nan
                                            for v in mechanical], plan)
                per_horizon[tag] = table
            interventions[key] = {'horizons': per_horizon, '64_minus_8_of_K_effect': {
                metric: paired(np.array(delta_by_horizon[64][metric]) - delta_by_horizon[8][metric], plan)
                for metric in ('accuracy', 'nll', 'margin')},
                'first_E5_bounded_current_difference': paired([
                    np.linalg.norm(np.array(a[label][0]['first_E5_bounded_current'])
                                   - a[ref][0]['first_E5_bounded_current']) for a in reads], plan)}

    normalized = {}
    norm_details = {}
    for name in norms[0]:
        normalized[name] = {}
        norm_details[name] = {}
        pairs = [r[name] for r in norms]
        for arm in ('legacy', 'event'):
            detail = []
            for index, pair in enumerate(pairs):
                if 'failure' in pair:
                    detail.append(None)
                else:
                    detail.append({'online': pair[arm]['online'], **{
                        str(h): score_rows(pair[arm]['readouts'], targets[index], h) for h in HORIZONS}})
            norm_details[name][arm] = detail
        for timepoint in ('online', *map(str, HORIZONS)):
            metrics = {}
            for metric in ('accuracy', 'nll', 'margin'):
                values = {arm: [scalar(r[timepoint], metric) if r is not None else np.nan
                                for r in norm_details[name][arm]] for arm in ('legacy', 'event')}
                metrics[metric] = {arm: paired(x, plan) for arm, x in values.items()}
                metrics[metric]['event_minus_legacy'] = paired(
                    np.array(values['event']) - values['legacy'], plan)
            normalized[name][timepoint] = metrics
        normalized[name]['first_direction_cosine'] = paired([
            p['event']['direction_cosines'][0] if 'failure' not in p else np.nan for p in pairs], plan)
        available = ['failure' not in p for p in pairs]
        readers = {branch: norm_reader(output, plan['units'], targets, name, branch, available)
                   for branch in ('legacy', 'event')}
        if all(reader is not None for reader in readers.values()):
            scores = {branch: np.array(reader['per_block_accuracy']) for branch, reader in readers.items()}
            difference = np.full((len(pairs), 3), np.nan)
            difference[available] = scores['event'] - scores['legacy']
            readers['contrasts'] = {feature: paired(difference[:, j], plan)
                                    for j, feature in enumerate(('start', 'end', 'delta'))}
        normalized[name]['diagnostic_reader'] = readers
    dose_contrasts = {}
    for error in ('target', 'full'):
        for arm in ('legacy', 'event'):
            low, high = (norm_details[f'{error}_{r}'][arm] for r in plan['radii'])
            dose_contrasts[f'{error}/{arm}'] = {timepoint: {metric: paired([
                scalar(b[timepoint] if b else None, metric) - scalar(a[timepoint] if a else None, metric)
                for a, b in zip(low, high)], plan) for metric in ('accuracy', 'nll')}
                for timepoint in ('online', '8', '64')}
    return {'cost': dict(counters), 'max_norm_pair_gap': norm_gap, 'curves': curves,
            'interventions': interventions, 'normalized': normalized,
            'radius_high_minus_low': dose_contrasts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--source', type=Path, help='Relocated historical inputs, verified by hash')
    args = parser.parse_args()
    summary = analyze(args.output, args.source)
    atomic_json(args.output / 'ANALYSIS.json', summary)
    print(json.dumps({'cost': summary['cost'], 'max_norm_pair_gap': summary['max_norm_pair_gap']}, indent=2))


if __name__ == '__main__':
    main()
