"""Paired development interventions on existing LIFE-QNUF-001 snapshots."""
from __future__ import annotations

import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from .checkpoint_io import load_state, save_state
from .codex_memory_location import atomic_json, digest, metrics, source_hashes
from .life_qnuf_001 import new_agent
from .qnuf import ObservedEvidence, array_hash, rng_hash


ROOT = Path(__file__).resolve().parents[2]
BASE = '86b56cb177f4818c0daf68c007cc1c9fbc991c45'
HORIZONS = (8, 16, 32, 64)
DONORS = ('initial', 'B0', 'B1', 'B2', 'B3')
RECIPIENTS = ('B1_live', 'B3_live', 'Z0')
RADII = (.08, .32)
PAIRINGS = {'target': ('B0', 'B1'), 'full': ('B2', 'B3')}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def life_path(source, arm, seed, mapping):
    return source / 'stage1' / 'lives' / f'{arm}_{seed}_{mapping}'


def checkpoint(source, arm, seed, mapping, event):
    return life_path(source, arm, seed, mapping) / f'checkpoint_{event:03d}.npz'


def initialize(source, output):
    output.mkdir(parents=True, exist_ok=False)
    parent = json.loads((source / 'frozen_config.json').read_text())
    manifest = json.loads((source / 'sha256_manifest.json').read_text())
    units = [(seed, mapping) for seed in parent['stage1_seeds']
             for mapping in range(len(parent['stage1_mappings']))]
    files = {'frozen_config.json', 'STATUS.json', 'stage1/contrasts.json'}
    for seed, mapping in units:
        for arm in DONORS[1:]:
            files.add(checkpoint(source, arm, seed, mapping, 72).relative_to(source).as_posix())
        for event in (0, 24):
            files.add(checkpoint(source, 'B0', seed, mapping, event).relative_to(source).as_posix())
        files.add((life_path(source, 'B0', seed, mapping) / 'event_records.json').relative_to(source).as_posix())
    for name in files:
        if file_hash(source / name) != manifest[name]:
            raise ValueError(f'Historical artifact hash mismatch: {name}')
    plan = {
        'baseline_commit': BASE, 'parent_config_sha256': digest(parent),
        'source_path': str(source), 'historical_files': {name: manifest[name] for name in sorted(files)},
        'parent': parent, 'units': units, 'horizons': list(HORIZONS), 'radii': list(RADII),
        'readout_gate': parent['gate'], 'norm_start': 24, 'norm_events': 24,
        'wrong_mapping': 'next mapping cyclically within the same seed and donor arm',
        'bootstrap_replicates': 5000, 'bootstrap_seed': parent['bootstrap_seed'],
        'worker_limit': 2, 'scope': 'development only',
        'primary_k_contrast': 'B1/B3 trained K vs initial and wrong-map K on own live state',
        'primary_horizon_contrast': 'paired K effect at 64 minus paired K effect at 8',
        'primary_norm_contrast': 'event minus legacy with per-event shared norm at each radius and error',
    }
    atomic_json(output / 'PLAN.json', plan)
    atomic_json(output / 'SOURCE.json', {
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'files': source_hashes(), 'plan_sha256': digest(plan),
        'historical_manifest_sha256': file_hash(source / 'sha256_manifest.json'),
    })


def score_rows(rows, targets, horizon):
    logits = [row['horizons'][str(horizon)]['logits'] for row in rows
              if str(horizon) in row['horizons']]
    valid_targets = [targets[q] for q, row in enumerate(rows) if str(horizon) in row['horizons']]
    return metrics(logits, valid_targets) if len(logits) == len(targets) else None


def integrate_read(core, query, gate):
    stimulus = np.eye(8)[query]
    row = {'horizons': {}, 'failure': None, 'substeps': 0,
           'first_E5_raw_current': (core.K @ np.tanh(core.Z))[32:40].tolist(),
           'first_E5_bounded_current': core.compute_currents(stimulus, write_gate=gate)[32:40].tolist()}
    initial_steps = core.step_count
    for tick in range(1, HORIZONS[-1] + 1):
        try:
            core.step(stimulus, write_gate=gate)
            if tick in HORIZONS:
                row['horizons'][str(tick)] = {
                    'logits': core.predict()[1:5].tolist(), 'Z': core.Z.tolist(),
                    'rng_hash': rng_hash(core),
                }
        except FloatingPointError as exc:
            row['failure'] = {'tick': tick, 'type': type(exc).__name__, 'message': str(exc)}
            break
    row['substeps'] = core.step_count - initial_steps
    return row


def read_intervention(plan, source, seed, mapping, originals, result_dir):
    parent = plan['parent']
    core = new_agent('B0', seed, parent).core
    gate = plan['readout_gate']
    targets = parent['stage1_mappings'][mapping]
    origin = originals['initial']['agent']
    donor_k = {name: state['agent']['core_snapshot']['K'] for name, state in originals.items()}
    wrong_mapping = (mapping + 1) % len(parent['stage1_mappings'])
    for arm in ('B1', 'B3'):
        wrong = load_state(checkpoint(source, arm, seed, wrong_mapping, 72))
        donor_k[f'{arm}_wrong'] = wrong['agent']['core_snapshot']['K']
    result = {}
    for recipient in RECIPIENTS:
        identity = recipient.split('_')[0] if recipient.endswith('_live') else 'initial'
        anchor = originals[identity]['agent']['core_snapshot']
        test_donors = list(DONORS)
        if identity in ('B1', 'B3'):
            test_donors += [f'{identity}_wrong', f'{identity}_sham']
        for donor in test_donors:
            chosen = donor_k[identity] if donor.endswith('_sham') else donor_k[donor]
            rows = []
            for key in ('B', 'W_readout'):
                np.testing.assert_array_equal(anchor[key], origin['core_snapshot'][key])
            for q in range(1, 7):
                core.restore(anchor)
                before = core.snapshot()
                core.K = chosen.copy()
                k_hash = array_hash(core.K)
                try:
                    record = integrate_read(core, q, gate)
                    if array_hash(core.K) != k_hash:
                        raise RuntimeError('Plasticity detected during read')
                finally:
                    core.restore(before)
                restored = core.snapshot()
                for name in ('K', 'B', 'W_readout', 'Z'):
                    np.testing.assert_array_equal(restored[name], before[name])
                assert restored['rng_state'] == before['rng_state']
                assert restored['step_count'] == before['step_count']
                rows.append(record)
            result[f'{recipient}/{donor}'] = rows
    for arm in ('B1', 'B3'):
        assert result[f'{arm}_live/{arm}'] == result[f'{arm}_live/{arm}_sham']
    atomic_json(result_dir / 'readouts.json', result)
    return result


def update_directions(agent, evidence):
    p = np.array(agent.pending['prediction']['probabilities'])
    error = np.zeros(8)
    if agent.error == 'full':
        error[1:5] = -p
    error[evidence.target] = 1 - p[evidence.target - 1]
    delta_z = agent.core.W_readout.T @ error
    return np.outer(delta_z, agent.pending[agent.eligibility])


def commit_at_norm(agent, evidence, direction, radius):
    norm = float(np.linalg.norm(direction))
    eta = agent.eta_k
    # The largest requested radius is below the coordinate clip. By normalizing
    # before clipping, the outer-product direction is preserved.
    agent.eta_k = radius / norm if norm else 0.0
    before = agent.core.K.copy()
    try:
        row = agent.finish_query(evidence)
    finally:
        agent.eta_k = eta
    applied = agent.core.K - before
    expected = direction * (radius / norm) if norm else np.zeros_like(direction)
    np.testing.assert_allclose(applied, expected, rtol=0, atol=2e-15)
    if abs(row['applied_delta_norm'] - radius) > 1e-12:
        raise RuntimeError('Applied norm did not match paired experimental radius')
    return row


def norm_intervention(plan, seed, mapping, start, result_dir):
    parent = plan['parent']
    targets = parent['stage1_mappings'][mapping]
    reference_path = life_path(Path(plan['source_path']), 'B0', seed, mapping) / 'event_records.json'
    reference = json.loads(reference_path.read_text())
    cursor = plan['norm_start']
    count = plan['norm_events']
    schedule = [row['query'] for row in reference[cursor:cursor + count]]
    result = {}
    for error, arms in PAIRINGS.items():
        for radius in RADII:
            agents = [new_agent(arm, seed, parent) for arm in arms]
            for agent, eligibility in zip(agents, ('legacy', 'event')):
                agent.restore(start['agent'])
                agent.eligibility, agent.error = eligibility, error
            name = f'{error}_{radius}'
            folder = result_dir / name
            folder.mkdir(exist_ok=True)
            stream = {'schedule': schedule, 'targets': targets, 'source_cursor': cursor,
                      'radius': radius, 'error': error}
            for agent in agents:
                agent.save_checkpoint(folder / f'{agent.eligibility}_start.npz', stream)
            records = [[], []]
            try:
                with (folder / 'pre_feedback.jsonl').open('w', encoding='utf-8') as journal:
                    for q in schedule:
                        predictions = [agent.begin_query(q) for agent in agents]
                        journal.write(json.dumps(predictions, allow_nan=False) + '\n')
                        journal.flush()
                        for record, pred in zip(records, predictions):
                            if record and record[-1]['Z_post_hash'] != pred['Z_pre_hash']:
                                raise RuntimeError('State discontinuity in normalized CARRY')
                        if predictions[0]['rng_post_hash'] != predictions[1]['rng_post_hash']:
                            raise RuntimeError('Unpaired RNG in normalized CARRY')
                        evidence = ObservedEvidence(agents[0].event_id, targets[q - 1])
                        directions = [update_directions(agent, evidence) for agent in agents]
                        norms = [float(np.linalg.norm(direction)) for direction in directions]
                        common_radius = radius if min(norms) > 1e-15 else 0.
                        cosine = (float(np.sum(directions[0] * directions[1]) / np.prod(norms))
                                  if common_radius else None)
                        for agent, record, pred, direction, raw_norm in zip(
                                agents, records, predictions, directions, norms):
                            update = commit_at_norm(agent, evidence, direction, common_radius)
                            record.append({**pred, **update, 'unscaled_direction_norm': raw_norm,
                                           'paired_direction_cosine': cosine, 'radius': common_radius})
                pair = {}
                for agent, record in zip(agents, records):
                    # Same 64-tick budget as frozen reads, from the actual final live state.
                    reads = [integrate_read(copy.deepcopy(agent.core), q, plan['readout_gate']) for q in range(1, 7)]
                    atomic_json(folder / f'{agent.eligibility}_events.json', record)
                    agent.save_checkpoint(folder / f'{agent.eligibility}_end.npz', {
                        **stream, 'cursor': cursor + count, 'event_records': record})
                    pair[agent.eligibility] = {
                        'online': metrics([r['logits'] for r in record], [r['target'] for r in record]),
                        'readouts': reads,
                        'update_norms': [r['applied_delta_norm'] for r in record],
                        'direction_cosines': [r['paired_direction_cosine'] for r in record],
                        'max_Z': agent.core.max_state_norm,
                        'training_substeps': count * parent['query_ticks'] * parent['substeps'],
                        'shadow_substeps': count * parent['query_ticks'] * parent['substeps'],
                    }
                result[name] = pair
            except FloatingPointError as exc:
                for agent, record in zip(agents, records):
                    atomic_json(folder / f'{agent.eligibility}_events.json', record)
                result[name] = {'failure': {'type': type(exc).__name__, 'message': str(exc),
                                             'commit_counts': [len(r) for r in records]}}
    atomic_json(result_dir / 'norm_results.json', result)
    return result


def work(task):
    plan, output, seed, mapping = task
    start = time.perf_counter()
    cpu = time.process_time()
    source = Path(plan['source_path'])
    folder = output / f'{seed}_{mapping}'
    folder.mkdir(exist_ok=True)
    result_file = folder / 'complete.json'
    if result_file.exists():
        saved = json.loads(result_file.read_text())
        if saved['plan_sha256'] != digest(plan):
            raise RuntimeError('Saved block conflicts with protocol')
        return saved
    original = {arm: load_state(checkpoint(source, arm, seed, mapping, 72)) for arm in DONORS[1:]}
    original['initial'] = load_state(checkpoint(source, 'B0', seed, mapping, 0))
    norm_start = load_state(checkpoint(source, 'B0', seed, mapping, plan['norm_start']))
    read_intervention(plan, source, seed, mapping, original, folder)
    norm_intervention(plan, seed, mapping, norm_start, folder)
    data = {'seed': seed, 'mapping': mapping, 'plan_sha256': digest(plan),
            'cpu_seconds': time.process_time() - cpu, 'wall_seconds': time.perf_counter() - start}
    atomic_json(result_file, data)
    return data


def run(output, workers):
    plan = json.loads((output / 'PLAN.json').read_text())
    source = json.loads((output / 'SOURCE.json').read_text())
    if not 1 <= workers <= plan['worker_limit']:
        raise ValueError('Worker limit exceeded')
    if source['files'] != source_hashes() or source['plan_sha256'] != digest(plan):
        raise RuntimeError('Frozen source or protocol changed')
    for path, expected in plan['historical_files'].items():
        if file_hash(Path(plan['source_path']) / path) != expected:
            raise RuntimeError(f'Historical input changed: {path}')
    status = output / 'STATUS.json'
    begin = time.perf_counter()
    atomic_json(status, {'phase': 'RUNNING', 'blocks': 0})
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs = [(plan, output, seed, mapping) for seed, mapping in plan['units']]
            for index, row in enumerate(pool.map(work, jobs), 1):
                atomic_json(status, {'phase': 'RUNNING', 'blocks': index, 'total': len(jobs)})
                print(f"{index}/{len(jobs)} seed={row['seed']} map={row['mapping']}", flush=True)
    except Exception as exc:
        atomic_json(status, {'phase': 'STOP_INVALID', 'error': str(exc), 'type': type(exc).__name__})
        raise
    atomic_json(status, {'phase': 'COMPLETED', 'blocks': len(jobs), 'wall_seconds': time.perf_counter() - begin})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('init', 'run'))
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.command == 'init':
        if args.source is None:
            parser.error('init requires --source')
        initialize(args.source.resolve(), args.output.resolve())
    else:
        run(args.output.resolve(), args.workers)


if __name__ == '__main__':
    main()
