"""Run only the development stages of LIFE-QNUF-001."""
from __future__ import annotations
import os
for _variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_variable] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from sklearn.linear_model import RidgeClassifier

from .codex_memory_location import atomic_json, crossed_ci, digest, metrics, source_hashes
from .checkpoint_io import load_state
from .dev_context_pathway_ablation import make_stable_core
from ..core.security import check_seed_safety
from .qnuf import ObservedEvidence, QNUFAgent, array_hash

ROOT = Path(__file__).resolve().parents[2]
BASE = 'dd92af26641f2e701ee9e2c4fa80fcad92caf376'
ARMS = {
    'B0': ('legacy', 'target', 'CARRY_TRAIN'),
    'B1': ('event', 'target', 'CARRY_TRAIN'),
    'B2': ('legacy', 'full', 'CARRY_TRAIN'),
    'B3': ('event', 'full', 'CARRY_TRAIN'),
    'RESET': ('legacy', 'target', 'RESET_TRAIN'),
    'B0_cap': ('legacy', 'target', 'CARRY_TRAIN'),
}


def init_plan(output):
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(821202)
    maps = []
    while len(maps) < 6:
        labels = [1, 2, 3, 4, *rng.integers(1, 5, 2).tolist()]
        rng.shuffle(labels)
        if labels not in maps:
            maps.append(labels)
    plan = {
        'base_sha': BASE, 'stage0_seeds': [821001, 821002],
        'stage1_seeds': [821101, 821102, 821103], 'stage1_mappings': maps,
        'map_generator': 821202, 'stream_seed': 821203, 'bootstrap_seed': 821204,
        'permutation_seed': 821205,
        'worker_cap': 2, 'stage0_events': 24, 'stage0_neutral_ticks': 4,
        'stage1_events': 72, 'query_ticks': 8, 'pause_ticks': 32, 'substeps': 2,
        'checkpoint_events_stage1': [0, 24, 48, 72],
        'h_base': .01, 'gate': 1.0, 'eta': .8, 'clip': .5,
        'gate0_late_accuracy': .5, 'gate0_positive_both_seeds': True,
        'gate1_online': .75, 'gate1_gain': .10, 'gate1_max_pause_loss': .05,
        'gate1_live_gain': .10, 'gate1_repeatability': 'positive online and live gain on every seed',
        'norm_control_selection': 'event arm with highest mean live accuracy; ties select B1',
        'reader': 'fixed alpha=1; development cross-fit excludes evaluated seed and mapping',
        'arms': {k: list(v) for k, v in ARMS.items()}, 'n_gate': 'all observed feedback',
        'f_gate': 'off', 'scope': 'stage0_and_stage1_development_only',
    }
    for seed in plan['stage0_seeds'] + plan['stage1_seeds'] + [
            plan['map_generator'], plan['stream_seed'], plan['bootstrap_seed'], plan['permutation_seed']]:
        check_seed_safety(seed)
    atomic_json(output / 'frozen_config.json', plan)
    atomic_json(output / 'source.json', {
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'files': source_hashes(), 'config_sha256': digest(plan),
        'maps_sha256': digest(maps),
    })
    status(output, 'PLANNED')


def status(output, phase, **details):
    atomic_json(output / 'STATUS.json', {'phase': phase, **details})


def new_agent(arm, seed, plan, cap=None):
    check_seed_safety(seed)
    eligibility, error, mode = ARMS[arm]
    core = make_stable_core(seed, substeps=plan['substeps'])
    if core.config.h_base != plan['h_base']:
        raise ValueError('Historical core step changed')
    return QNUFAgent(core, eligibility=eligibility, error=error, norm_cap=cap, learning_mode=mode,
                     t_query=plan['query_ticks'], eta_k=plan['eta'], delta_core_max=plan['clip'],
                     write_gate_presentation=plan['gate'], read_gate=plan['gate'])


def read_probes(agent, queries, targets, pause_ticks):
    """Generate label-free logits on clones, then score the frozen outputs."""
    origin = copy.deepcopy(agent.core)
    paused = copy.deepcopy(origin)
    for _ in range(pause_ticks):
        paused.step(None, write_gate=0.0)  # historical idle convention
    cost = paused.step_count - origin.step_count
    result = {}
    for mode in ('live', 'reset', 'pause'):
        starts = paused if mode == 'pause' else origin
        logits = []
        for query in queries:
            clone = copy.deepcopy(starts)
            if mode == 'reset':
                clone.Z = agent.Z0.copy()
            stimulus = np.eye(8)[query]
            for _ in range(agent.t_query):
                clone.step(stimulus, write_gate=agent.read_gate)
            cost += clone.step_count - starts.step_count
            logits.append(clone.predict()[1:5].copy())
        result[mode] = metrics(logits, targets)
    return result, cost


def life_task(task):
    plan, output, stage, arm, seed, map_id, cap = task
    folder = Path(output) / stage / 'lives' / f'{arm}_{seed}_{map_id}'
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    begin, cpu = time.perf_counter(), time.process_time()
    one_query = stage == 'stage0'
    targets = [map_id] if one_query else plan['stage1_mappings'][map_id]
    queries = list(range(1, len(targets) + 1))
    count = plan['stage0_events' if one_query else 'stage1_events']
    rng = np.random.default_rng(np.random.SeedSequence([plan['stream_seed'], seed, map_id]))
    schedule = [int(q) for _ in range(count // len(queries)) for q in rng.permutation(queries)]
    neutral = plan['stage0_neutral_ticks'] if one_query else 0
    agent = new_agent(arm, seed, plan, cap)
    frozen = [array_hash(m) for m in (agent.core.B, agent.core.W_readout)]
    initial_k = agent.core.K.copy()
    initial_rng = rng.bit_generator.state
    path = folder / 'latest.npz'
    cursor = 0
    results, probes = [], {}
    eval_steps = neutral_steps = 0
    if path.exists():
        saved = agent.load_checkpoint(path)
        if (saved['schedule'] != schedule or saved['plan_hash'] != digest(plan)
                or saved['cap'] != cap or saved['arm'] != arm):
            raise ValueError('Mismatched life checkpoint')
        cursor, results, probes = saved['cursor'], saved['records'], saved['probes']
        eval_steps, neutral_steps = saved['eval_steps'], saved['neutral_steps']
    checkpoints = [0, count] if one_query else plan['checkpoint_events_stage1']

    def save(event):
        nonlocal eval_steps
        probes[str(event)], ticks = read_probes(agent, queries, targets, plan['pause_ticks'])
        eval_steps += ticks
        state = {'cursor': event, 'schedule': schedule, 'stream_rng': initial_rng, 'cap': cap,
                 'arm': arm, 'plan_hash': digest(plan), 'records': results, 'probes': probes,
                 'eval_steps': eval_steps, 'neutral_steps': neutral_steps}
        agent.save_checkpoint(folder / f'checkpoint_{event:03d}.npz', state)
        agent.save_checkpoint(path, state)

    # If interrupted after a prediction journal write, restore its prior checkpoint and
    # preserve that partial journal. Committed rows remain independently verifiable.
    journal = folder / f'predictions_resume_{cursor:03d}_{time.time_ns()}.jsonl'
    if not path.exists():
        save(0)
    failure = None
    with journal.open('x', encoding='utf-8') as pre_feedback:
        for i in range(cursor, count):
            query = schedule[i]
            if neutral and i:
                agent.idle(neutral)
                neutral_steps += neutral * plan['substeps']
            try:
                before = array_hash(agent.core.Z)
                prediction = agent.begin_query(query)
                if arm != 'RESET' and prediction['Z_pre_hash'] != before:
                    raise RuntimeError('Global reset detected in CARRY')
                if not neutral and results and arm != 'RESET':
                    if results[-1]['Z_post_hash'] != prediction['Z_pre_hash']:
                        raise RuntimeError('CARRY continuity violated')
                pre_feedback.write(json.dumps(prediction, allow_nan=False) + '\n')
                pre_feedback.flush()
                feedback = ObservedEvidence(i, targets[query - 1])
                row = {**prediction, **agent.finish_query(feedback)}
                results.append(row)
                if i + 1 in checkpoints:
                    save(i + 1)
            except FloatingPointError as exc:
                failure = {'event_id': i, 'type': type(exc).__name__, 'message': str(exc)}
                break
    if [array_hash(m) for m in (agent.core.B, agent.core.W_readout)] != frozen:
        raise RuntimeError('Frozen B/W changed')
    final = str(count)
    complete = failure is None and len(results) == count and final in probes
    outcome = {
        'stage': stage, 'arm': arm, 'seed': seed, 'map_id': map_id, 'targets': targets,
        'events': len(results), 'failure': failure, 'complete': complete, 'cap': cap,
        'probe': probes.get(final), 'mode': agent.learning_mode,
        'online': float(np.mean([r['correct'] for r in results])) if results else None,
        'late_online': float(np.mean([r['correct'] for r in results[-8 if one_query else -24:]])) if results else None,
        'nll': float(np.mean([r['nll'] for r in results])) if results else None,
        'mean_phi_legacy': float(np.mean([np.linalg.norm(r['phi_legacy']) for r in results])) if results else None,
        'mean_phi_event': float(np.mean([np.linalg.norm(r['phi_event']) for r in results])) if results else None,
        'mean_update': float(np.mean([r['applied_delta_norm'] for r in results])) if results else None,
        'max_proposed_update': max((r['proposed_delta_norm'] for r in results), default=0.),
        'final_K_change': float(np.linalg.norm(agent.core.K - initial_k)),
        'max_live_Z': agent.core.max_state_norm, 'max_live_energy': agent.core.max_total_energy,
        'update_candidates': agent.update_candidates, 'updates_accepted': len(results),
        'live_substeps': agent.core.step_count, 'shadow_substeps': agent.shadow_steps,
        'eval_substeps': eval_steps, 'neutral_substeps': neutral_steps,
        'wall_seconds': time.perf_counter() - begin, 'cpu_seconds': time.process_time() - cpu,
        'B_W_hashes': frozen,
    }
    atomic_json(folder / 'event_records.json', results)
    atomic_json(result_path, outcome)
    return outcome


def run_tasks(output, plan, stage, jobs, workers):
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        iterator = pool.map(life_task, [(plan, str(output), stage, *job) for job in jobs])
        for i, result in enumerate(iterator, 1):
            rows.append(result)
            status(output, stage.upper(), complete=i, total=len(jobs))
            print(stage, f'{i}/{len(jobs)}', result['arm'], result['seed'], result['map_id'],
                  result['online'], flush=True)
    return rows


def matrix(rows, plan, key):
    values = {(row['seed'], row['map_id']): key(row) for row in rows}
    return np.array([[values[seed, m] for m in range(len(plan['stage1_mappings']))]
                     for seed in plan['stage1_seeds']])


def reader_diagnostic(output, group, permutation_seed):
    """Development-only cross-fit, excluding BOTH observed seed and mapping."""
    features = []
    targets = np.array([r['targets'] for r in group])
    for row in group:
        folder = output / 'stage1' / 'lives' / f"{row['arm']}_{row['seed']}_{row['map_id']}"
        original = load_state(folder / 'checkpoint_000.npz')['agent']['core_snapshot']['K'][32:35]
        trained = load_state(folder / 'checkpoint_072.npz')['agent']['core_snapshot']['K'][32:35]
        features.append(np.array([original.flatten(), trained.flatten()]))
    features = np.array(features)
    predictions = np.empty((len(group), 2, 6), dtype=int)
    coverage = []
    for i, row in enumerate(group):
        train = [j for j, r in enumerate(group) if r['seed'] != row['seed'] and r['map_id'] != row['map_id']]
        for q in range(6):
            classes = np.unique(targets[train, q])
            coverage.append(row['targets'][q] in classes)
            for f in range(2):
                if len(classes) == 1:
                    predictions[i, f, q] = classes[0]
                else:
                    model = RidgeClassifier(alpha=1).fit(features[train, f], targets[train, q])
                    predictions[i, f, q] = model.predict(features[i:i + 1, f])[0]
    observed = float(np.mean(predictions[:, 1] == targets))
    rng = np.random.default_rng(permutation_seed)
    scores = []
    # Descriptive shuffled-label reference. The overlapping cross-fit training
    # folds make this different from a permutation test of a frozen holdout model.
    map_count = len({r['map_id'] for r in group})
    by_seed = targets.reshape(-1, map_count, 6)
    for _ in range(999):
        shuffled = by_seed[:, rng.permutation(map_count), :]
        scores.append(float(np.mean(predictions[:, 1] == shuffled.reshape(-1, 6))))
    return {'K_initial': float(np.mean(predictions[:, 0] == targets)),
            'K_after': observed, 'shuffled_reference_mean': float(np.mean(scores)),
            'target_class_coverage': float(np.mean(coverage)), 'predictions': predictions.tolist(),
            'targets': targets.tolist(), 'units': [[r['seed'], r['map_id']] for r in group]}


def report_stage1(output, plan, rows):
    group = {arm: [r for r in rows if r['arm'] == arm] for arm in ARMS}
    grid = {}
    stats = {}
    for arm, items in group.items():
        grid[arm] = {
            'online': matrix(items, plan, lambda x: x['online']),
            'live': matrix(items, plan, lambda x: x['probe']['live']['accuracy']),
            'reset': matrix(items, plan, lambda x: x['probe']['reset']['accuracy']),
            'pause': matrix(items, plan, lambda x: x['probe']['pause']['accuracy']),
            'nll': matrix(items, plan, lambda x: x['nll']),
        }
        stats[arm] = {key: float(val.mean()) for key, val in grid[arm].items()}
        stats[arm].update({key: float(np.mean([x[key] for x in items])) for key in
                           ('late_online', 'mean_phi_legacy', 'mean_phi_event', 'mean_update', 'final_K_change',
                            'live_substeps', 'eval_substeps', 'shadow_substeps', 'max_live_Z')})
        stats[arm]['reader'] = reader_diagnostic(output, items, plan['permutation_seed'])
    contrasts = {}
    for metric in ('online', 'live', 'pause', 'nll'):
        g = {arm: matrix_[metric] for arm, matrix_ in grid.items()}
        pairs = {'Q_target': g['B1'] - g['B0'], 'Q_full': g['B3'] - g['B2'],
                 'U_legacy': g['B2'] - g['B0'], 'U_event': g['B3'] - g['B1'],
                 'QxU': g['B3'] - g['B2'] - g['B1'] + g['B0'],
                 'norm_cap_minus_B0': g['B0_cap'] - g['B0']}
        contrasts[metric] = {key: {'mean': float(delta.mean()),
                                    'seed_means': delta.mean(axis=1).tolist(),
                                    'ci95_crossed': crossed_ci(delta, plan['bootstrap_seed'], 5000)}
                             for key, delta in pairs.items()}
    decisions = {}
    for arm in ('B1', 'B2', 'B3'):
        online = grid[arm]['online'] - grid['B0']['online']
        live = grid[arm]['live'] - grid['B0']['live']
        pause_loss = stats[arm]['live'] - stats[arm]['pause']
        decisions[arm] = bool(stats[arm]['online'] >= plan['gate1_online']
            and online.mean() >= plan['gate1_gain'] and live.mean() >= plan['gate1_live_gain']
            and np.all(online.mean(axis=1) > 0) and np.all(live.mean(axis=1) > 0)
            and pause_loss <= plan['gate1_max_pause_loss'])
    result = {'arms': stats, 'contrasts': contrasts, 'development_gate': decisions,
              'stage2_authorized': False, 'holdout_run': False,
              'status': 'GO' if any(decisions.values()) else 'PARTIAL'}
    atomic_json(output / 'stage1' / 'contrasts.json', result)
    with (output / 'stage1' / 'per_life.csv').open('w', newline='', encoding='utf-8') as f:
        fields = [k for k, v in rows[0].items() if not isinstance(v, (dict, list))]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    return result


def run(output, workers):
    plan = json.loads((output / 'frozen_config.json').read_text())
    source = json.loads((output / 'source.json').read_text())
    if source['files'] != source_hashes() or source['config_sha256'] != digest(plan):
        raise ValueError('Frozen experiment source or configuration changed')
    if not (output / 'preflight' / 'PASS.json').exists():
        raise ValueError('Run preflight before collecting Stage 0 data')
    if not 1 <= workers <= plan['worker_cap']:
        raise ValueError('Worker cap exceeded')
    arms = list(ARMS)[:-1]
    jobs = [(a, s, y, None) for a in arms for s in plan['stage0_seeds'] for y in range(1, 5)]
    stage0 = run_tasks(output, plan, 'stage0', jobs, workers)
    stage0_stats = {}
    for arm in arms:
        group = [r for r in stage0 if r['arm'] == arm]
        if not all(r['complete'] for r in group):
            status(output, 'STOP', reason='Incomplete Stage 0 life', rows=stage0)
            return
        seed_means = [float(np.mean([r['late_online'] for r in group if r['seed'] == seed]))
                      for seed in plan['stage0_seeds']]
        # Score incompatible labels taken from the three other balanced mapping lives.
        shuffle = []
        for row in group:
            path = output / 'stage0' / 'lives' / f"{arm}_{row['seed']}_{row['map_id']}" / 'event_records.json'
            events = json.loads(path.read_text())[-8:]
            shuffle.extend(e['prediction'] == y for e in events for y in range(1, 5) if y != row['map_id'])
        stage0_stats[arm] = {'late_accuracy_by_seed': seed_means,
                             'late_accuracy': float(np.mean(seed_means)),
                             'other_life_label_accuracy': float(np.mean(shuffle))}
    atomic_json(output / 'stage0' / 'controls.json', stage0_stats)
    if not any(min(stage0_stats[a]['late_accuracy_by_seed']) >= plan['gate0_late_accuracy']
               for a in arms if a != 'RESET'):
        status(output, 'STOP', reason='Stage 0 CARRY positive control below fixed gate')
        return
    panel = [(seed, m) for seed in plan['stage1_seeds'] for m in range(len(plan['stage1_mappings']))]
    stage1 = run_tasks(output, plan, 'stage1', [(a, s, m, None) for a in arms for s, m in panel], workers)
    if not all(r['complete'] for r in stage1):
        status(output, 'STOP', reason='Incomplete Stage 1 life', rows=stage1)
        return
    event_arms = {a: [r for r in stage1 if r['arm'] == a] for a in ('B1', 'B3')}
    winner = max(event_arms, key=lambda a: np.mean([r['probe']['live']['accuracy'] for r in event_arms[a]]))
    cap = max(r['max_proposed_update'] for r in event_arms[winner])
    atomic_json(output / 'stage1' / 'norm_control.json', {'selected_event_arm': winner, 'cap': cap})
    stage1 += run_tasks(output, plan, 'stage1', [('B0_cap', s, m, cap) for s, m in panel], workers)
    if not all(r['complete'] for r in stage1):
        status(output, 'STOP', reason='Incomplete norm control life', rows=stage1)
        return
    fixed_cost = {(r['live_substeps'], r['shadow_substeps'], r['eval_substeps'], r['update_candidates']) for r in stage1}
    if len(fixed_cost) != 1:
        raise RuntimeError(f'Unequal compute under Stage 1: {fixed_cost}')
    result = report_stage1(output, plan, stage1)
    status(output, 'COMPLETED', stage1_lives=len(stage1), gate=result['development_gate'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('init', 'run'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.command == 'init':
        init_plan(args.output)
    else:
        try:
            run(args.output, args.workers)
        except Exception as exc:
            status(args.output, 'STOP_INVALID', error=type(exc).__name__, message=str(exc))
            raise


if __name__ == '__main__':
    main()
