"""Test mapping selectivity of the event-49 sidecar with intact mainline."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pqt_qcat.memory import access_state as stage
from pqt_qcat.memory.access_bridge import checkpoint
from pqt_qcat.memory.checkpoint_io import load_state
from pqt_qcat.memory.codex_memory_location import atomic_json, metrics
from analyze_access_state import difference, gate, load, paired


def score(read, target):
    row = read['horizons'].get('64')
    return metrics([row['logits']], [target]) if row else None


def run_unit(task):
    plan, output, unit = task
    arm, seed, map_id = unit
    directory = output / f'{arm}_{seed}_{map_id}'
    cursor = plan['continuity_checkpoint']
    history = Path(plan['historical_source'])
    next_map = (map_id + 1) % len(plan['parent']['stage1_mappings'])
    donors = {
        'initial': load_state(checkpoint(history, 'B0', seed, map_id, 0))['agent']['core_snapshot']['K'],
        'wrong': load_state(checkpoint(history, arm, seed, next_map, cursor))['agent']['core_snapshot']['K'],
    }
    start = time.process_time()
    continuity = load(directory / 'continuity.json')['results']
    control = {}
    for scenario in ('sidecar/local', 'sidecar/Z0', 'sidecar/live64'):
        saved = directory / 'continuity' / scenario.replace('/', '_')
        query = continuity[scenario]['records'][0]['query']
        agent = stage.load_agent(plan, arm, seed, map_id, cursor)
        original = load(saved / 'sidecar_pre_feedback.json')
        target = continuity[scenario]['records'][0]['target']
        results = {donor: stage.capture(agent, query, continuity[scenario]['spec'], matrix, prove_isolation=True)
                   for donor, matrix in donors.items()}
        for row in results.values():
            if row['failure'] is None:
                assert row['substeps'] == original['substeps'] == 128
        control[scenario] = {'readouts': results,
                             'scored': {name: score(read, target) for name, read in
                                        {'own': original, **results}.items()}}
    data = {'controls': control, 'unit': unit, 'cpu_seconds': time.process_time() - start}
    atomic_json(directory / 'sidecar_event49_donors.json', data)
    return data


def summarize(rows, plan, output):
    result = {}
    for arm in ('B1', 'B3'):
        group = [row['controls'] for row in rows if row['unit'][0] == arm]
        result[arm] = {}
        for scenario in group[0]:
            cases = [row[scenario]['scored'] for row in group]
            contrasts = {'live': [row['sidecar/live64']['scored']['own'] for row in group],
                         'initial': [row['initial'] for row in cases],
                         'wrong': [row['wrong'] for row in cases]}
            own = [row['own'] for row in cases]
            stats = {name: {metric: difference(own, alternate, metric, plan)
                            for metric in ('accuracy', 'nll')} for name, alternate in contrasts.items()}
            result[arm][scenario] = {
                'own': {metric: paired([row[metric] if row else np.nan for row in own], plan)
                        for metric in ('accuracy', 'nll')},
                'contrasts': stats,
                'ACCESS': gate(stats['live']['accuracy'], plan['access_min']),
                'SELECTIVITY_components': {name: gate(stats[name]['accuracy'], plan['selectivity_min'])
                                           for name in ('initial', 'wrong')},
            }
    substeps = sum(read['substeps'] for row in rows for case in row['controls'].values()
                   for read in case['readouts'].values())
    atomic_json(output / 'EVENT49_DONOR_ANALYSIS.json', {
        'results': result, 'substeps': substeps, 'cpu_seconds': sum(row['cpu_seconds'] for row in rows)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    plan = load(output / 'PLAN.json')
    stage.verify_inputs(output, plan)
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(run_unit, [(plan, output, unit) for unit in plan['units']]))
    summarize(rows, plan, output)


if __name__ == '__main__':
    main()
