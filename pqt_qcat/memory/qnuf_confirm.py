"""Frozen independent confirmation of the B2A causal Sidecar policies."""
from . import sidecar_causal as causal  # numerical thread caps before NumPy

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.metadata
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from .access_state import exact
from .checkpoint_io import load_state, save_state
from .codex_memory_location import atomic_json, digest, source_hashes
from .functional_sidecar import ActionJournal, read_json, state_hash
from .functional_sidecar_b2a import RetryEnvironment, forecast, probe
from .life_qnuf_001 import new_agent
from .qnuf import ObservedEvidence
from ..core.security import check_seed_safety


BASE = '01fa3ae7847f7e5bf8c0397702bf31f386d73690'
CAUSAL_ZIP = '358ba2e09c0db539ac851847bcbd8f92ed424edcefb2a5d9a83eccccb47f4975'
SEEDS = (931731, 931732, 931733, 931734, 931735)
GENERATORS = dict(mapping=931741, prefix=931742, offsets=931743, bootstrap=931744)
POLICIES = {'raw_MM': ('MM', None, 'own'), 'raw_SS': ('SS', None, 'own'),
            'norm050_SM': ('SM', .50, 'own'), 'norm050_SS': ('SS', .50, 'own')}
CONTROLS = {'initial_SS': ('SS', None, 'initial'), 'wrong_SS': ('SS', None, 'wrong')}
ENVIRONMENTS = {'retry': RetryEnvironment, 'advance': causal.AdvanceEnvironment}
PREFIX = 48
STEPS = 24
RESUME = 12
WORKERS = 6


def init(output, reference, archive, old_memory):
    repo = causal.b2.b1.bridge.ROOT
    subprocess.run(['git', 'merge-base', '--is-ancestor', BASE, 'HEAD'], cwd=repo, check=True)
    delivery = read_json(reference / 'DELIVERY.json')
    assert delivery['delivery_commit'] == BASE and delivery['engineering'] == 'CAUSAL_VALID'
    assert causal.file_sha(archive) == CAUSAL_ZIP
    manifest = reference / 'ARTIFACT_MANIFEST.sha256'
    entries = [line.split('  ', 1) for line in manifest.read_text(encoding='utf-8').splitlines()]
    for expected, name in entries:
        if causal.file_sha(reference / name) != expected:
            raise ValueError(f'Causal artifact differs from its manifest: {name}')
    prior = read_json(reference / 'PLAN.json')['parent']
    memory = read_json(old_memory)
    for seed in (*SEEDS, *GENERATORS.values()):
        check_seed_safety(seed)
        assert seed not in prior['stage1_seeds'] + prior['stage0_seeds']
        assert all(seed not in values for values in memory['seeds'].values())
    old_mappings = prior['stage1_mappings'] + memory['maps']
    rng = np.random.default_rng(GENERATORS['mapping'])
    mappings = []
    while len(mappings) < 6:
        labels = [1, 2, 3, 4, *rng.integers(1, 5, 2).tolist()]
        rng.shuffle(labels)
        if labels not in old_mappings and labels not in mappings:
            mappings.append(labels)
    units = [[seed, mapping] for seed in SEEDS for mapping in range(len(mappings))]
    streams = {}
    for seed, mapping in units:
        rng = np.random.default_rng(np.random.SeedSequence([GENERATORS['prefix'], seed, mapping]))
        sequence = [int(q) for _ in range(9) for q in rng.permutation(np.arange(1, 7))]
        rng = np.random.default_rng(np.random.SeedSequence([GENERATORS['offsets'], seed, mapping]))
        streams[causal.unit_name(seed, mapping)] = {'prefix': sequence[:PREFIX], 'query': sequence[PREFIX],
                                                    'offsets': rng.integers(1, 6, STEPS).tolist()}
    plan = {'base': BASE, 'base_engineering': 'CAUSAL_VALID', 'base_scientific': 'DEVELOPMENT_CAUSAL_EVIDENCE',
            'baseline_archive': str(archive), 'baseline_archive_sha256': CAUSAL_ZIP,
            'baseline_manifest_sha256': causal.file_sha(manifest), 'baseline_files_checked': len(entries),
            'historical_memory_plan_sha256': causal.file_sha(old_memory),
            'parent': {**{k: prior[k] for k in ('substeps','h_base','query_ticks','gate','eta','clip')},
                       'stage1_seeds': list(SEEDS), 'stage1_mappings': mappings},
            'generators': GENERATORS, 'units': units, 'streams': streams,
            'prefix_events': PREFIX, 'steps': STEPS, 'resume_age': RESUME, 'workers': WORKERS,
            'resume_unit': units[0], 'policies': POLICIES, 'controls': CONTROLS,
            'environments': list(ENVIRONMENTS), 'primary_contrast': 'advance raw_SS minus raw_MM',
            'accuracy_threshold': .05, 'reward_threshold': 2., 'table_threshold': 1.,
            'bootstrap_seed': GENERATORS['bootstrap'], 'bootstrap_replicates': 10000,
            'ci': '95% crossed pairs percentile bootstrap; resample seeds and mappings independently',
            'gate': 'point estimate reaches the stated threshold and lower 95% CI endpoint > 0',
            'missing': 'primary cannot pass if a registered block is missing; no exclusions or replacement seed',
            'negative_control': 'SS read/shadow swaps K; wrong donor is same seed, environment and age of next mapping raw_SS',
            'table': 'same 48 preset examples, no labels before durable action; class 1 until a query is observed',
            'scope': 'independent one-rule confirmation only; M5, DREAM, N/F and multiple rules excluded'}
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(old_memory, output / 'PREVIOUS_MEMORY_PLAN.json')
    for name in ('PLAN.json', 'DELIVERY.json', 'ANALYSIS.json', 'ARTIFACT_MANIFEST.sha256'):
        shutil.copy2(reference / name, output / ('BASELINE_' + name))
    atomic_json(output / 'PLAN.json', plan)
    atomic_json(output / 'SOURCE.json', {
        'git': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        'files': source_hashes(), 'plan_sha256': digest(plan), 'python': sys.version,
        'dependencies': {p: importlib.metadata.version(p) for p in ('numpy','torch','scipy','scikit-learn','pytest')},
    })
    atomic_json(output / 'LOCK.json', {'plan_file_sha256': causal.file_sha(output / 'PLAN.json'),
                                      'source_file_sha256': causal.file_sha(output / 'SOURCE.json'),
                                      'frozen_before_phase': 'prepare'})


def verify_plan(root):
    lock = read_json(root / 'LOCK.json')
    for document, key in [('PLAN.json', 'plan_file_sha256'), ('SOURCE.json', 'source_file_sha256')]:
        assert causal.file_sha(root / document) == lock[key]
    plan, source = read_json(root / 'PLAN.json'), read_json(root / 'SOURCE.json')
    assert digest(plan) == source['plan_sha256'] and source_hashes() == source['files']
    assert causal.file_sha(root / 'BASELINE_ARTIFACT_MANIFEST.sha256') == plan['baseline_manifest_sha256']
    assert causal.file_sha(root / 'PREVIOUS_MEMORY_PLAN.json') == plan['historical_memory_plan_sha256']
    return plan


def track(root, seed, mapping, env, arm):
    return root / 'lives' / causal.unit_name(seed, mapping) / env / arm


def panel(root, plan, seed, mapping):
    check_seed_safety(seed)
    if seed not in SEEDS or [seed, mapping] not in plan['units']:
        raise ValueError('Block outside the locked independent panel')
    name = causal.unit_name(seed, mapping)
    return root / 'preparation' / name, plan['parent']['stage1_mappings'][mapping], plan['streams'][name]


def table_logits(table, query):
    result = [0.] * 4
    if str(query) in table:
        result[table[str(query)] - 1] = 8.
    return result


def prepare(job):
    root, plan, seed, mapping = job
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    folder, truth, stream = panel(root, plan, seed, mapping)
    folder.mkdir(parents=True, exist_ok=False)
    agent = new_agent('B3', seed, plan['parent'])
    table = {}
    initial = agent.snapshot()
    journal = ActionJournal(folder / 'pre_feedback.jsonl')
    events = []
    for index, query in enumerate(stream['prefix']):
        main = agent.begin_query(query)
        logits = table_logits(table, query)
        packet = {'index': index, 'query': query, 'neural_prediction': main, 'table_logits': logits,
                  'table_action': int(np.argmax(logits)) + 1}
        receipt = journal.publish(packet)
        exact(journal.committed(receipt), packet)
        journal.last_receipt = None
        target = truth[query - 1]
        update = agent.finish_query(ObservedEvidence(index, target))
        table[str(query)] = target
        events.append({'packet': packet, 'receipt': receipt, 'target_after': target, 'update': update})
    checkpoint = {'initial': initial, 'agent': agent.snapshot(), 'table': table, 'unit': [seed, mapping],
                  'plan_sha256': digest(plan)}
    save_state(folder / 'checkpoint_048.npz', checkpoint)
    atomic_json(folder / 'events.json', events)
    result = {'unit': [seed, mapping], 'events': len(events), 'substeps': PREFIX * 32,
              'carry_probe': probe(agent, truth), 'wall_seconds': time.perf_counter() - start_wall,
              'cpu_seconds': time.process_time() - start_cpu}
    atomic_json(folder / 'COMPLETE.json', result)
    return [seed, mapping]


def donor_set(root, plan, seed, mapping, env, kind, checkpoint):
    if kind == 'wrong':
        other = (mapping + 1) % len(plan['parent']['stage1_mappings'])
        return load_state(track(root, seed, other, env, 'raw_SS') / 'donors.npz')['K']
    if kind == 'initial':
        k = checkpoint['initial']['core_snapshot']['K']
        return np.broadcast_to(k, (STEPS, *k.shape))
    return [None] * STEPS


def run_life(job):
    root, plan, seed, mapping, env, arm, resume = job
    clock0, cpu0 = time.perf_counter(), time.process_time()
    prefix, truth, stream = panel(root, plan, seed, mapping)
    source = track(root, seed, mapping, env, arm)
    folder = track(root / 'resume', seed, mapping, env, arm) if resume else source
    folder.mkdir(parents=True, exist_ok=False)
    checkpoint = load_state(prefix / 'checkpoint_048.npz')
    assert checkpoint['plan_sha256'] == digest(plan)
    is_table = arm == 'TABLE'
    table = checkpoint['table'] if is_table else {}
    agent = None
    if not is_table:
        source_credit, norm, donor = (POLICIES | CONTROLS)[arm]
        agent = new_agent('B3', seed, plan['parent'])
        agent.restore(checkpoint['agent'])
        donors = donor_set(root, plan, seed, mapping, env, donor, checkpoint)
    environment = ENVIRONMENTS[env](truth, stream['offsets'], stream['query'])
    journal = ActionJournal(folder / 'pre_feedback.jsonl')
    pending = None
    if resume:
        cut = load_state(source / 'pending_12.npz')
        assert cut['plan_sha256'] == digest(plan) and cut['condition'] == [env, arm]
        table = cut['table']
        if agent is not None:
            agent.restore(cut['agent'])
        environment = ENVIRONMENTS[env](truth, stream['offsets'], **cut['environment'])
        pending = ActionJournal(source / 'pre_feedback.jsonl').read_verified(cut['receipt'])
        assert state_hash(table if is_table else agent.snapshot()) == pending['pending_sha256']
        receipt = journal.publish(pending)

    rows, clocks, matrices = [], [], []
    while environment.cursor < STEPS:
        started = time.perf_counter_ns()
        age = environment.cursor
        observation = environment.snapshot()
        if pending is None:
            query = environment.observe()
            if is_table:
                logits = table_logits(table, query)
                pending = {'query': query, 'main': {'event_id': PREFIX + age}, 'logits': logits,
                           'action': int(np.argmax(logits)) + 1, 'pending_sha256': state_hash(table),
                           'budget': {'main': 0, 'main_shadow': 0, 'side': 0, 'side_shadow': 0}}
            else:
                if arm == 'raw_SS' and not resume:
                    matrices.append(agent.core.K.copy())
                pending = forecast(agent, query, 'Z0-64', 'aligned', donors[age], {})
            pending.update(cursor=age, observation=observation)
            receipt = journal.publish(pending)
        published = time.perf_counter_ns()
        if age == RESUME and not resume:
            save_state(folder / 'pending_12.npz', {'agent': None if is_table else agent.snapshot(),
                       'table': table, 'environment': observation, 'receipt': receipt,
                       'condition': [env, arm], 'plan_sha256': digest(plan)})
        reaction = environment.react(journal, receipt)
        reacted = time.perf_counter_ns()
        evidence = environment.reveal()
        revealed = time.perf_counter_ns()
        if is_table:
            table[str(pending['query'])] = evidence.target
            update = {'table_write': True}
            end_hash = state_hash(table)
        else:
            update, _ = causal.apply_credit(agent, pending, evidence, source_credit, norm)
            end_hash = state_hash(agent.snapshot())
        last = time.perf_counter_ns()
        rows.append({'packet': pending, 'reaction': reaction, 'target_after': evidence.target,
                     'update': update, 'end_state': end_hash})
        clocks.append({'receipt': receipt, 'publish_ns': published, 'reaction_ns': reacted, 'reveal_ns': revealed,
                       'update_ns': last, 'event_wall_ns': last - started})
        pending = None

    end = {'agent': None if is_table else agent.snapshot(), 'table': table,
           'environment': environment.snapshot(), 'condition': [env, arm], 'plan_sha256': digest(plan)}
    save_state(folder / 'end.npz', end)
    atomic_json(folder / 'events.json', rows)
    atomic_json(folder / 'runtime.json', clocks)
    result = {'unit': [seed, mapping], 'condition': [env, arm], 'events': len(rows)}
    if resume:
        exact(rows, read_json(source / 'events.json')[RESUME:])
        assert state_hash(end) == state_hash(load_state(source / 'end.npz'))
        assert journal.path.read_bytes().splitlines(keepends=True) == (
            (source / 'pre_feedback.jsonl').read_bytes().splitlines(keepends=True)[RESUME:])
        result['exact'] = True
    elif not is_table:
        if matrices:
            save_state(folder / 'donors.npz', {'K': np.stack(matrices)})
        result['carry_probe'] = probe(agent, truth)
        result['delta_K_norm'] = float(np.linalg.norm(agent.core.K - checkpoint['agent']['core_snapshot']['K']))
    result.update(wall_seconds=time.perf_counter() - clock0, cpu_seconds=time.process_time() - cpu0)
    atomic_json(folder / 'COMPLETE.json', result)
    return [seed, mapping, env, arm]


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('phase', choices=('init', 'prepare', 'primary', 'controls', 'resume'))
    for name in ('output', 'reference', 'archive', 'old-memory'):
        cli.add_argument('--' + name, type=Path, required=name == 'output')
    args = cli.parse_args()
    root = args.output.resolve()
    phase = args.phase
    if phase == 'init':
        init(root, args.reference.resolve(), args.archive.resolve(), args.old_memory.resolve())
        return
    plan = verify_plan(root)
    if phase == 'prepare':
        function = prepare
        jobs = [(root, plan, *unit) for unit in plan['units']]
    else:
        function = run_life
        choices = ('TABLE', *POLICIES) if phase == 'primary' else tuple(CONTROLS) if phase == 'controls' else (
            'TABLE', *POLICIES, *CONTROLS)
        units = [plan['resume_unit']] if phase == 'resume' else plan['units']
        jobs = [(root, plan, *unit, env, arm, phase == 'resume') for unit in units
                for env in ENVIRONMENTS for arm in choices]
    status = root / f'STATUS_{phase}.json'
    start = time.perf_counter()
    atomic_json(status, {'phase': 'RUNNING', 'completed': 0, 'total': len(jobs)})
    failures = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(function, job): job[2:] for job in jobs}
        for n, future in enumerate(as_completed(futures), 1):
            try:
                print(n, len(jobs), future.result(), flush=True)
            except Exception as exc:
                detail = {'job': futures[future], 'type': type(exc).__name__, 'message': str(exc)}
                failures.append(detail)
                print('FAILED', detail, flush=True)
            atomic_json(status, {'phase': 'RUNNING', 'completed': n, 'total': len(jobs), 'failures': failures})
    atomic_json(status, {'phase': 'COMPLETED_WITH_FAILURES' if failures else 'COMPLETED', 'total': len(jobs),
                         'failures': failures, 'wall_seconds': time.perf_counter() - start})


if __name__ == '__main__':
    main()
