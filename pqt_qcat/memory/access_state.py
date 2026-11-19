"""Stage A interventions and isolated readout on historical LIFE snapshots."""
from . import access_bridge as bridge  # sets the numerical worker thread caps

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from .checkpoint_io import load_state, save_state
from .codex_memory_location import atomic_json, digest, metrics, source_hashes
from .life_qnuf_001 import new_agent
from .qnuf import ObservedEvidence, array_hash


BASE = '8643bcff3c539298821a28040f64f28e0ed06cd3'
ARMS = ('B1', 'B3')
DONORS = ('own', 'initial', 'wrong')
GROUPS = {'E1-E2': [1, 2], 'E3-E4': [3, 4], 'E5-E6': [5, 6]}


def exact(a, b):
    if isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            exact(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            exact(left, right)
    else:
        assert a == b


def initialize(history, bridge_run, output):
    output.mkdir(parents=True, exist_ok=False)
    parent = json.loads((history / 'frozen_config.json').read_text())
    manifests = {str(root): json.loads((root / filename).read_text()) for root, filename in (
        (history, 'sha256_manifest.json'), (bridge_run, 'SHA256.json'))}
    units = [(arm, seed, map_id) for arm in ARMS for seed in parent['stage1_seeds']
             for map_id in range(len(parent['stage1_mappings']))]
    source_files = {str(history): {'frozen_config.json'}, str(bridge_run): {'PLAN.json', 'ANALYSIS.json'}}
    expected_bridge = json.loads((bridge_run / 'PLAN.json').read_text())
    if expected_bridge['parent_config_sha256'] != digest(parent):
        raise ValueError('ACCESS-BRIDGE is from a different historical panel')
    for arm, seed, mapping in units:
        folder = bridge.life_path(history, arm, seed, mapping)
        for name in ('checkpoint_048.npz', 'checkpoint_072.npz', 'event_records.json'):
            source_files[str(history)].add((folder / name).relative_to(history).as_posix())
        source_files[str(history)].add(bridge.checkpoint(history, 'B0', seed, mapping, 0)
                                       .relative_to(history).as_posix())
        source_files[str(bridge_run)].add(f'{seed}_{mapping}/readouts.json')
    hashes = {root: {name: manifests[root][name] for name in sorted(names)} for root, names in source_files.items()}
    plan = {
        'base': BASE, 'parent': parent, 'units': units, 'historical_source': str(history),
        'bridge_source': str(bridge_run), 'inputs': hashes, 'groups': GROUPS,
        'horizons': list(bridge.HORIZONS), 'primary_horizon': 64, 'refinement_depths': [.25, .5, 1.],
        'refinement_trigger_gain': .05, 'access_min': .10, 'selectivity_min': .10,
        'continuity_accuracy_margin': .05, 'continuity_nll_margin': .05,
        'continuity_checkpoint': 48, 'subsequent_events': 12,
        'bootstrap_seed': parent['bootstrap_seed'], 'bootstrap_replicates': 5000,
        'brief': 'Named file unavailable. Stage A instructions from user message govern.',
        'selection': 'Per arm, refine best group by 64-tick own-K gain if >=5pp. Then select best local probe '
                     'that meets ACCESS and SELECTIVITY, otherwise strongest group for continuity diagnosis. Tie: insertion order.',
    }
    for directory, files in hashes.items():
        for name, expected in files.items():
            if bridge.file_hash(Path(directory) / name) != expected:
                raise ValueError(f'Historical hash mismatch: {directory}/{name}')
    atomic_json(output / 'PLAN.json', plan)
    atomic_json(output / 'SOURCE.json', {
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=bridge.ROOT, text=True).strip(),
        'files': source_hashes(), 'plan_sha256': digest(plan),
    })


def shifted(z, z0, cells, depth):
    result = z.copy()
    for cell in cells:
        segment = slice((cell - 1) * 8, cell * 8)
        if depth == 1.:
            result[segment] = z0[segment]
        else:
            result[segment] = z[segment] + depth * (z0[segment] - z[segment])
    return result


def load_agent(plan, arm, seed, mapping, cursor):
    agent = new_agent(arm, seed, plan['parent'])
    checkpoint = bridge.checkpoint(Path(plan['historical_source']), arm, seed, mapping, cursor)
    agent.load_checkpoint(checkpoint)
    return agent


def capture(agent, query, specification, replacement_k=None, prove_isolation=False):
    original = agent.snapshot() if prove_isolation else None
    clone = copy.deepcopy(agent.core)
    start = clone.Z.copy()
    clone.Z = shifted(start, agent.Z0, specification['cells'], specification['depth'])
    shifted_z = clone.Z.copy()
    if replacement_k is not None:
        clone.K = replacement_k.copy()
    hashes = {key: array_hash(getattr(clone, key)) for key in ('K', 'B', 'W_readout')}
    row = bridge.integrate_read(clone, query, agent.read_gate)
    assert all(array_hash(getattr(clone, key)) == value for key, value in hashes.items())
    if original is not None:
        exact(original, agent.snapshot())
    row.update(query=query, Z_start=start.tolist(), Z_intervened=shifted_z.tolist(), frozen_hashes=hashes)
    return row


def read_task(args):
    plan, output, phase, unit, specs = args
    arm, seed, mapping = unit
    if not specs:
        return None
    folder = output / f'{arm}_{seed}_{mapping}'
    folder.mkdir(exist_ok=True)
    path = folder / f'{phase}.json'
    if path.exists():
        return json.loads(path.read_text())
    start = time.perf_counter()
    cpu = time.process_time()
    agent = load_agent(plan, arm, seed, mapping, 72)
    source = Path(plan['historical_source'])
    initial = load_state(bridge.checkpoint(source, 'B0', seed, mapping, 0))['agent']['core_snapshot']
    wrong_id = (mapping + 1) % len(plan['parent']['stage1_mappings'])
    wrong = load_state(bridge.checkpoint(source, arm, seed, wrong_id, 72))['agent']['core_snapshot']
    donor = {'own': agent.core.K, 'initial': initial['K'], 'wrong': wrong['K']}
    before = agent.snapshot()
    readings = {name: {label: [capture(agent, q, spec, donor[label]) for q in range(1, 7)]
                       for label in DONORS} for name, spec in specs.items()}
    exact(before, agent.snapshot())
    result = {'unit': unit, 'readings': readings, 'specs': specs,
              'wall_seconds': time.perf_counter() - start, 'cpu_seconds': time.process_time() - cpu}
    atomic_json(path, result)
    return result


def play(agent, event, factor, journal):
    query = event['query']
    original_ticks = agent.t_query
    agent.t_query = factor
    try:
        pred = agent.begin_query(query)
    finally:
        agent.t_query = original_ticks
    journal.write(json.dumps(pred, allow_nan=False) + '\n')
    journal.flush()
    update = agent.finish_query(ObservedEvidence(agent.event_id, event['target']))
    return {**pred, **update}


def continuity_task(args):
    plan, output, unit, selection = args
    arm, seed, mapping = unit
    folder = output / f'{arm}_{seed}_{mapping}'
    result_path = folder / 'continuity.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    cpu = time.process_time()
    cursor = plan['continuity_checkpoint']
    source = Path(plan['historical_source'])
    archive = json.loads((bridge.life_path(source, arm, seed, mapping) / 'event_records.json').read_text())
    schedule = archive[cursor:cursor + 1 + plan['subsequent_events']]
    live_spec = {'cells': [], 'depth': 0.}
    full_spec = {'cells': list(range(1, 7)), 'depth': 1.}
    native = {f'direct/{name}': spec for name, spec in selection['direct'].items()}
    scenarios = {'historical_live': live_spec, 'direct/live64': live_spec,
                 'direct/Z0': full_spec, **native, 'sidecar/live64': live_spec,
                 'sidecar/Z0': full_spec, 'sidecar/local': selection['sidecar']}
    results = {}
    for name, spec in scenarios.items():
        agent = load_agent(plan, arm, seed, mapping, cursor)
        label = name.replace('/', '_')
        saved = folder / 'continuity' / label
        saved.mkdir(parents=True, exist_ok=True)
        agent.save_checkpoint(saved / 'start.npz', {'cursor': cursor})
        old = agent.core.Z.copy()
        first = schedule[0]
        sidecar = None
        records = []
        error = None
        step_start, shadow_start = agent.core.step_count, agent.shadow_steps
        try:
            if name.startswith('sidecar/'):
                sidecar = capture(agent, first['query'], spec, prove_isolation=True)
                # The archived mainline protocol consumes its own short-read
                # prediction for feedback. The sidecar answer stays separate.
                atomic_json(saved / 'sidecar_pre_feedback.json', sidecar)
            if name.startswith('direct/'):
                agent.core.Z = shifted(old, agent.Z0, spec['cells'], spec['depth'])
            intervention_norm = float(np.linalg.norm(agent.core.Z - old))
            with (saved / 'main_pre_feedback.jsonl').open('w', encoding='utf-8') as journal:
                for index, event in enumerate(schedule):
                    ticks = 64 if index == 0 and name.startswith('direct/') else plan['parent']['query_ticks']
                    records.append(play(agent, event, ticks, journal))
            agent.save_checkpoint(saved / 'end.npz', {'cursor': cursor + len(records), 'specification': spec})
        except FloatingPointError as exc:
            intervention_norm = float(np.linalg.norm(agent.core.Z - old))
            error = {'type': type(exc).__name__, 'message': str(exc), 'committed': len(records)}
        if name == 'historical_live' or name.startswith('sidecar/'):
            if error is not None:
                raise RuntimeError('Historical mainline failed to reproduce')
            for observed, expected in zip(records, schedule):
                for field in ('Z_pre_hash', 'Z_post_hash', 'K_pre_hash', 'K_post_hash', 'rng_post_hash', 'logits'):
                    exact(observed[field], expected[field])
        for prev, row in zip(records, records[1:]):
            assert prev['Z_post_hash'] == row['Z_pre_hash']
        future = records[1:]
        scores = (metrics([r['logits'] for r in future], [r['target'] for r in future])
                  if len(future) == plan['subsequent_events'] else None)
        first_logits = (sidecar['horizons'].get('64', {}).get('logits') if sidecar is not None
                        else records[0]['logits'] if records else None)
        results[name] = {'spec': spec, 'first_read': metrics([first_logits], [first['target']])
                        if first_logits is not None else None, 'future': scores, 'failure': error,
                        'sidecar_failure': sidecar['failure'] if sidecar is not None else None,
                        'intervention_norm': intervention_norm,
                        'live_substeps': agent.core.step_count - step_start,
                        'shadow_substeps': agent.shadow_steps - shadow_start,
                        'sidecar_substeps': sidecar['substeps'] if sidecar is not None else 0,
                        'records': records}
    historical = load_state(folder / 'continuity' / 'historical_live' / 'end.npz')['agent']
    for name, row in results.items():
        if row['failure'] is not None:
            continue
        final = load_state(folder / 'continuity' / name.replace('/', '_') / 'end.npz')['agent']
        row['final_Z_deviation_from_historical'] = float(np.linalg.norm(
            final['core_snapshot']['Z'] - historical['core_snapshot']['Z']))
        row['final_K_deviation_from_historical'] = float(np.linalg.norm(
            final['core_snapshot']['K'] - historical['core_snapshot']['K']))
        if name.startswith('sidecar/'):
            exact(historical, final)
            row['mainline_bitwise_exact'] = True
    data = {'unit': unit, 'results': results, 'cpu_seconds': time.process_time() - cpu}
    atomic_json(result_path, data)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('init', 'screen', 'refine', 'continuity'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--history', type=Path)
    parser.add_argument('--bridge', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == 'init':
        initialize(args.history.resolve(), args.bridge.resolve(), output)
        return
    if not 1 <= args.workers <= 2:
        parser.error('Use one or two workers')
    plan = json.loads((output / 'PLAN.json').read_text())
    verify_inputs(output, plan)
    selection = (json.loads((output / 'SELECTION.json').read_text())
                 if args.action in ('refine', 'continuity') else None)
    if args.action == 'screen':
        specs = {name: {'cells': cells, 'depth': 1.} for name, cells in GROUPS.items()}
        jobs = [(plan, output, 'screen', unit, specs) for unit in plan['units']]
        function = read_task
    elif args.action == 'refine':
        jobs = [(plan, output, 'refine', unit, selection[unit[0]]['refinements']) for unit in plan['units']]
        function = read_task
    else:
        final = json.loads((output / 'FINAL_SELECTION.json').read_text())
        jobs = [(plan, output, unit, final[unit[0]]) for unit in plan['units']]
        function = continuity_task
    status_path = output / f'STATUS_{args.action}.json'
    start = time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for n, _ in enumerate(pool.map(function, jobs), 1):
                atomic_json(status_path, {'phase': 'RUNNING', 'completed': n, 'total': len(jobs)})
                print(f'{args.action} {n}/{len(jobs)}', flush=True)
    except Exception as exc:
        atomic_json(status_path, {'phase': 'ERROR', 'type': type(exc).__name__, 'message': str(exc)})
        raise
    atomic_json(status_path, {'phase': 'COMPLETED', 'total': len(jobs), 'wall_seconds': time.perf_counter() - start})


def verify_inputs(output, plan):
    source = json.loads((output / 'SOURCE.json').read_text())
    if source['files'] != source_hashes() or source['plan_sha256'] != digest(plan):
        raise ValueError('Frozen source or plan changed')
    for root, files in plan['inputs'].items():
        for name, expected in files.items():
            if bridge.file_hash(Path(root) / name) != expected:
                raise ValueError(f'Historical input changed: {root}/{name}')


if __name__ == '__main__':
    main()
