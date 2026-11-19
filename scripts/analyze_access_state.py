"""Paired Stage A scoring, frozen selection, and three independent verdicts."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pqt_qcat.memory.access_state import DONORS, GROUPS, exact, shifted
from pqt_qcat.memory.access_bridge import score_rows
from pqt_qcat.memory.codex_memory_location import atomic_json, digest
from pqt_qcat.memory.checkpoint_io import load_state
from pqt_qcat.memory.qnuf import array_hash
from analyze_access_bridge import paired


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def store_selection(path, value):
    if path.exists():
        if load(path) != value:
            raise RuntimeError('Frozen selection differs from recomputed decision')
    else:
        atomic_json(path, value)


def difference(first, second, key, plan):
    return paired([a[key] - b[key] if a is not None and b is not None else np.nan
                   for a, b in zip(first, second)], plan)


def measure_curves(plan, raw, baseline, targets, hypotheses=None):
    result = {}
    for name, rows in raw.items():
        result[name] = {}
        for tick in plan['horizons']:
            scored = {k: [score_rows(r[k], y, tick) for r, y in zip(rows, targets)] for k in DONORS}
            base = [score_rows(b, y, tick) for b, y in zip(baseline, targets)]
            result[name][str(tick)] = {
                'donors': {label: {metric: paired([x[metric] if x else np.nan for x in group], plan)
                                   for metric in ('accuracy', 'nll', 'margin')}
                           for label, group in scored.items()},
                'contrasts': {control: {metric: difference(scored['own'], group, metric, plan)
                                         for metric in ('accuracy', 'nll', 'margin')}
                              for control, group in {'live': base, **scored}.items() if control != 'own'},
            }
        primary = result[name][str(plan['primary_horizon'])]['contrasts']
        result[name]['ACCESS'] = gate(primary['live']['accuracy'], plan['access_min'])
        strongest = [gate(primary[k]['accuracy'], plan['selectivity_min']) for k in ('initial', 'wrong')]
        result[name]['SELECTIVITY'] = 'PASS' if all(v == 'PASS' for v in strongest) else (
            'FAIL' if 'FAIL' in strongest else 'INCONCLUSIVE')
    return result


def gate(contrast, threshold):
    if contrast['ci95'] is None:
        return 'INCONCLUSIVE'
    low, high = contrast['ci95']
    if contrast['mean'] >= threshold and low > 0.:
        return 'PASS'
    return 'FAIL' if high < threshold else 'INCONCLUSIVE'


def verify_frozen(phase, unit, plan, output, history, z0_state):
    arm, seed, mapping = unit
    p = output / f'{arm}_{seed}_{mapping}' / f'{phase}.json'
    if not p.exists():
        return {}, {}
    data = load(p)
    counters = Counter()
    assert data['unit'] == unit
    original = load_state(history / 'stage1' / 'lives' / f'{arm}_{seed}_{mapping}'
                          / 'checkpoint_072.npz')['agent']['core_snapshot']
    wrong_map = (mapping + 1) % len(plan['parent']['stage1_mappings'])
    wrong = load_state(history / 'stage1' / 'lives' / f'{arm}_{seed}_{wrong_map}'
                       / 'checkpoint_072.npz')['agent']['core_snapshot']
    donor = {'own': original['K'], 'initial': z0_state['K'], 'wrong': wrong['K']}
    for name, raw in data['readings'].items():
        spec = data['specs'][name]
        wanted = shifted(original['Z'], z0_state['Z'], spec['cells'], spec['depth'])
        for k_type, rows in raw.items():
            assert len(rows) == 6
            for q, row in enumerate(rows, 1):
                assert row['query'] == q
                np.testing.assert_array_equal(row['Z_start'], original['Z'])
                np.testing.assert_array_equal(row['Z_intervened'], wanted)
                assert row['frozen_hashes'] == dict(K=array_hash(donor[k_type]), B=array_hash(original['B']),
                                                     W_readout=array_hash(original['W_readout']))
                for h, result in row['horizons'].items():
                    np.testing.assert_array_equal(result['logits'], (original['W_readout'] @ result['Z'])[1:5])
                    assert result['rng_hash'] == raw['own'][q-1]['horizons'][h]['rng_hash']
                if row['failure'] is None:
                    assert row['substeps'] == 128
                counters['substeps'] += row['substeps']
                counters['readout_trajectories'] += 1
                counters['failures'] += int(row['failure'] is not None)
    counters['cpu_seconds'] = data['cpu_seconds']
    return data['readings'], counters


def frozen_summary(plan, output, phase):
    combined = {}
    counts = Counter()
    history = Path(plan['historical_source'])
    bridge = Path(plan['bridge_source'])
    hypotheses = ('screen', 'refine') if phase == 'select-final' else ('screen',)
    for arm in ('B1', 'B3'):
        rows, live, targets = {}, [], []
        for unit in plan['units']:
            if unit[0] != arm:
                continue
            _, seed, mapping = unit
            archived = load(bridge / f'{seed}_{mapping}' / 'readouts.json')
            wrong_map = (mapping + 1) % len(plan['parent']['stage1_mappings'])
            wrong = load(bridge / f'{seed}_{wrong_map}' / 'readouts.json')
            originals = {
                'live': {'own': archived[f'{arm}_live/{arm}'],
                         'initial': archived[f'{arm}_live/initial'], 'wrong': archived[f'{arm}_live/{arm}_wrong']},
                'Z0': {'own': archived[f'Z0/{arm}'],
                       'initial': archived['Z0/initial'], 'wrong': wrong[f'Z0/{arm}']},
            }
            initial = load_state(history / 'stage1' / 'lives' / f'B0_{seed}_{mapping}'
                                 / 'checkpoint_000.npz')['agent']['core_snapshot']
            targets.append(plan['parent']['stage1_mappings'][mapping])
            live.append(originals['live']['own'])
            options = originals.copy()
            for stage in hypotheses:
                results, budget = verify_frozen(stage, unit, plan, output, history, initial)
                if stage == 'screen' and not results:
                    raise FileNotFoundError(f'No screen results for {unit}')
                options.update(results)
                counts.update(budget)
            for name, readout in options.items():
                rows.setdefault(name, []).append(readout)
        combined[arm] = measure_curves(plan, rows, live, targets)
    return combined, dict(counts)


def select(plan, data, output, phase):
    tick = str(plan['primary_horizon'])
    selections = {}
    for arm, rows in data.items():
        def key(name):
            value = rows[name][tick]['contrasts']['live']['accuracy']['mean']
            return value if value is not None else -np.inf
        best_group = max(GROUPS, key=key)
        if phase == 'select':
            cells = GROUPS[best_group]
            variants = {f'E{cell}': {'cells': [cell], 'depth': 1.} for cell in cells}
            variants.update({f'{best_group}@{depth}': {'cells': cells, 'depth': depth}
                             for depth in plan['refinement_depths'] if depth < 1.})
            selections[arm] = {'group': best_group, 'observed_gain': key(best_group),
                               'refinements': variants if key(best_group) >= plan['refinement_trigger_gain'] else {}}
        else:
            coarse = {k: {'cells': v, 'depth': 1.} for k, v in GROUPS.items()}
            refined = load(output / 'SELECTION.json')[arm]['refinements']
            all_specs = {**coarse, **refined}
            passing = [k for k in all_specs if rows[k]['ACCESS'] == rows[k]['SELECTIVITY'] == 'PASS']
            selected = max(passing, key=key) if passing else best_group
            selections[arm] = {'selected': selected, 'sidecar': all_specs[selected],
                               'direct': {**coarse, selected: all_specs[selected]},
                               'ACCESS': rows[selected]['ACCESS'], 'SELECTIVITY': rows[selected]['SELECTIVITY']}
    store_selection(output / ('SELECTION.json' if phase == 'select' else 'FINAL_SELECTION.json'), selections)
    return selections


def continuity(plan, output):
    results = {}
    cost = Counter()
    for arm in ('B1', 'B3'):
        observations = []
        for label, seed, mapping in plan['units']:
            if label != arm:
                continue
            folder = output / f'{arm}_{seed}_{mapping}'
            data = load(folder / 'continuity.json')
            cost['cpu_seconds'] += data['cpu_seconds']
            observations.append(data['results'])
            root = folder / 'continuity'
            historical = load_state(root / 'historical_live/end.npz')
            for scenario, row in data['results'].items():
                for category in ('live_substeps', 'shadow_substeps', 'sidecar_substeps'):
                    cost[category] += row[category]
                cost['failures'] += int(row['failure'] is not None or row['sidecar_failure'] is not None)
                if row['failure'] is not None:
                    continue
                expected = 320 if scenario.startswith('direct/') else 208
                assert row['live_substeps'] == row['shadow_substeps'] == expected
                assert row['sidecar_substeps'] == (128 if scenario.startswith('sidecar/') else 0)
                end = load_state(root / scenario.replace('/', '_') / 'end.npz')
                start = load_state(root / scenario.replace('/', '_') / 'start.npz')
                for key in ('B', 'W_readout'):
                    np.testing.assert_array_equal(start['agent']['core_snapshot'][key], end['agent']['core_snapshot'][key])
                if scenario.startswith('sidecar/'):
                    exact(historical['agent'], end['agent'])
                    cost['sidecar_exact_mainline_cases'] += 1
                for first, second in zip(row['records'], row['records'][1:]):
                    assert first['Z_post_hash'] == second['Z_pre_hash']
                    cost['exact_continuation_boundaries'] += 1
        results[arm] = {}
        for name in observations[0]:
            group = [unit[name] for unit in observations]
            control = 'direct/live64' if name.startswith('direct/') else 'historical_live'
            baseline = [unit[control] for unit in observations]
            historical = [unit['historical_live'] for unit in observations]
            result = {'first_read': paired([x['first_read']['accuracy'] if x['first_read'] else np.nan
                                           for x in group], plan),
                      'future': {metric: paired([x['future'][metric] if x['future'] else np.nan for x in group], plan)
                                 for metric in ('accuracy', 'nll', 'margin')},
                      'future_delta_matched': {key: difference([x['future'] for x in group],
                                  [x['future'] for x in baseline], key, plan) for key in ('accuracy', 'nll')},
                      'future_delta_historical': {key: difference([x['future'] for x in group],
                                  [x['future'] for x in historical], key, plan) for key in ('accuracy', 'nll')},
                      'final_Z_deviation': paired([x.get('final_Z_deviation_from_historical', np.nan) for x in group], plan),
                      'failures': sum(x['failure'] is not None or x['sidecar_failure'] is not None for x in group)}
            acc = result['future_delta_historical']['accuracy']['ci95']
            nll = result['future_delta_historical']['nll']['ci95']
            matched_acc = result['future_delta_matched']['accuracy']['ci95']
            matched_nll = result['future_delta_matched']['nll']['ci95']
            if result['failures']:
                verdict = 'FAIL'
            elif any(x is None for x in (acc, nll, matched_acc, matched_nll)):
                verdict = 'INCONCLUSIVE'
            elif min(acc[0], matched_acc[0]) >= -plan['continuity_accuracy_margin'] and max(
                    nll[1], matched_nll[1]) <= plan['continuity_nll_margin']:
                verdict = 'PASS'
            elif min(acc[1], matched_acc[1]) < -plan['continuity_accuracy_margin'] or max(
                    nll[0], matched_nll[0]) > plan['continuity_nll_margin']:
                verdict = 'FAIL'
            else:
                verdict = 'INCONCLUSIVE'
            result['CONTINUITY'] = verdict
            result['comparison'] = control
            result['sidecar_bitwise_exact'] = all(x.get('mainline_bitwise_exact', False) for x in group)
            results[arm][name] = result
    return {'results': results, 'cost': dict(cost)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('select', 'select-final', 'continuity'))
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    plan = load(output / 'PLAN.json')
    assert load(output / 'SOURCE.json')['plan_sha256'] == digest(plan)
    if args.action == 'continuity':
        summary = continuity(plan, output)
        atomic_json(output / 'CONTINUITY_ANALYSIS.json', summary)
        print(json.dumps(summary['cost'], indent=2))
    else:
        summary, cost = frozen_summary(plan, output, args.action)
        filename = 'SCREEN_ANALYSIS.json' if args.action == 'select' else 'READ_ANALYSIS.json'
        decisions = select(plan, summary, output, args.action)
        atomic_json(output / filename, {'results': summary, 'cost': cost})
        print(json.dumps(decisions, indent=2))


if __name__ == '__main__':
    main()
