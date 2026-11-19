"""Development-only Stage C0 for conflicting mappings on the QNUF core."""
from . import sidecar_causal as causal  # cap numerical threads first

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import importlib.metadata
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zipfile import ZipFile

import numpy as np

from .access_state import exact
from .checkpoint_io import load_state, save_state
from .codex_memory_location import atomic_json, digest, source_hashes
from .functional_sidecar import ActionJournal, read_json, state_hash
from .functional_sidecar_b2a import forecast
from .life_qnuf_001 import new_agent
from .qnuf import ObservedEvidence, array_hash, rng_hash
from ..core.security import check_seed_safety


BASE = '42750836c797bd6fdfd4df560ff4c55ed8353ad5'
BASE_EXECUTABLE = '43297fce8d24eb783c0cb10d38fbb9662b872a02'
BASE_ZIP = '2d01da82b364240937500b72cf3eb88800692ef5a63894cb3c77bd5b510257eb'
SEEDS = (821101, 821102, 821103)
MODES = ('separate', 'opponent')
POLICIES = ('MM', 'SS')
TABLES = ('keyed', 'blind')
EXPOSURES, CUT, WORKERS = 96, 48, 6
MANIFEST_COPIES = ('DELIVERY.json', 'SOURCE.json', 'PLAN.json', 'ANALYSIS.json', 'ARTIFACT_MANIFEST.sha256')
NEURAL = tuple((mode, policy) for mode in MODES for policy in POLICIES)
CONDITIONS = (*NEURAL, *(('table', arm) for arm in TABLES))


def init(root, reference, archive):
    repo = causal.b2.b1.bridge.ROOT
    subprocess.run(['git', 'merge-base', '--is-ancestor', BASE, 'HEAD'], cwd=repo, check=True)
    subprocess.run(['git', 'diff', '--quiet', BASE_EXECUTABLE, BASE, '--', 'pqt_qcat'], cwd=repo, check=True)
    delivery, source = (read_json(reference / filename) for filename in ('DELIVERY.json', 'SOURCE.json'))
    exact([delivery['delivery_commit'], delivery['frozen_executable_sha'], source['git']],
          [BASE, BASE_EXECUTABLE, BASE_EXECUTABLE])
    exact([delivery['engineering'], delivery['accuracy_verdict'], delivery['reward_verdict']],
          ['CONFIRM_VALID', 'CONFIRMED', 'CONFIRMED'])
    assert causal.file_sha(archive) == BASE_ZIP
    manifest = reference / 'ARTIFACT_MANIFEST.sha256'
    entries = [row.split('  ', 1) for row in manifest.read_text(encoding='utf-8').splitlines()]
    for expected, name in entries:
        if causal.file_sha(reference / name) != expected:
            raise ValueError(f'Confirmation artifact hash differs: {name}')
    with ZipFile(reference / 'FROZEN_CODE.zip') as archived:
        assert archived.comment.decode('ascii') == BASE_EXECUTABLE
        for name, expected in source['files'].items():
            assert hashlib.sha256(archived.read(name.replace('\\', '/'))).hexdigest() == expected
    confirmed = read_json(reference / 'PLAN.json')
    old = read_json(reference / 'BASELINE_PLAN.json')['parent']
    exact(list(SEEDS), old['stage1_seeds'])
    assert not set(SEEDS).intersection(confirmed['parent']['stage1_seeds'])
    maps = [{'A': rule, 'B': [1 + (label - 1 + i + 1) % 4 for label in rule]}
            for i, rule in enumerate(old['stage1_mappings'][:3])]
    units = [[seed, i] for seed in SEEDS for i in range(len(maps))]
    schedules = {}
    for seed, mapping in units:
        check_seed_safety(seed)
        rng = np.random.default_rng(np.random.SeedSequence([old['stream_seed'], seed, mapping]))
        examples = [[ctx, q] for ctx in ('A', 'B') for q in range(1, 7)]
        schedules[causal.unit_name(seed, mapping)] = [examples[int(i)] for _ in range(8) for i in rng.permutation(12)]
    plan = {'base_delivery_git': BASE, 'base_executable_git': BASE_EXECUTABLE,
            'base_archive': str(archive), 'base_archive_sha256': BASE_ZIP,
            'base_manifest_sha256': causal.file_sha(manifest), 'base_archived_sources': len(source['files']),
            'base_files_verified': len(entries), 'base_source_zip_sha256': causal.file_sha(reference / 'FROZEN_CODE.zip'),
            'parent': {**{k: confirmed['parent'][k] for k in ('substeps','h_base','query_ticks','gate','eta','clip')},
                       'stage1_seeds': list(SEEDS)},
            'development_units': units, 'mappings': maps, 'schedules': schedules,
            'conditions': CONDITIONS, 'encoders': {'separate': 'e_q + e_0 for A; e_q + e_7 for B',
                                                    'opponent': 'e_q +/- (e_0-e_7)/sqrt(2)'},
            'chosen_mode_order': list(MODES), 'policy_order': ['SS', 'MM'], 'threshold_each_rule': .75,
            'table_threshold': 1., 'events': EXPOSURES, 'cut': CUT, 'workers': WORKERS,
            'resume_unit': units[0], 'trace_unit': units[0], 'donor': 'same seed, encoder and policy; next mapping',
            'probe_order': ['A', 'B'], 'probe_ticks': [0,1,8,32,64],
            'negative_probe_types': ['initial_K', 'wrong_K', 'no_cue', 'swapped_cue'],
            'gate': 'first registered mode and policy with frozen Z0-64 >= .75 in A and B separately; keyed table perfect',
            'scope': 'development C0 only, fixed teaching stream; no C1 or independent panel'}
    root.mkdir(parents=True, exist_ok=False)
    for name in MANIFEST_COPIES:
        shutil.copy2(reference / name, root / ('BASELINE_' + name))
    atomic_json(root/'PLAN.json', plan)
    atomic_json(root/'SOURCE.json', {
        'git': subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip(),
        'files': source_hashes(), 'plan_sha256': digest(plan), 'python': sys.version,
        'dependencies': {p: importlib.metadata.version(p) for p in ('numpy','torch','scipy','scikit-learn','pytest')},
    })
    atomic_json(root/'LOCK.json', {'plan': causal.file_sha(root/'PLAN.json'), 'source': causal.file_sha(root/'SOURCE.json'),
                                  'frozen_before': 'train'})


def verify_plan(root):
    locks = read_json(root / 'LOCK.json')
    assert causal.file_sha(root/'PLAN.json') == locks['plan'] and causal.file_sha(root/'SOURCE.json') == locks['source']
    plan, source = read_json(root/'PLAN.json'), read_json(root/'SOURCE.json')
    assert digest(plan) == source['plan_sha256'] and source_hashes() == source['files']
    assert causal.file_sha(root/'BASELINE_ARTIFACT_MANIFEST.sha256') == plan['base_manifest_sha256']
    return plan


def vector(mode, cue, q):
    if q not in range(1, 7) or cue not in ('A','B',None) or mode not in MODES:
        raise ValueError('Unknown technical encoder, cue or token')
    x = np.eye(8)[q]
    if cue is None:
        return x
    if mode == 'separate':
        x[0 if cue == 'A' else 7] = 1.
    else:
        x[0] = (1. if cue == 'A' else -1.) / np.sqrt(2.)
        x[7] = -x[0]
    return x


def table_key(arm, ctx, q):
    return f'{ctx}:{q}' if arm == 'keyed' else str(q)


def table_logits(table, arm, ctx, q):
    label = table.get(table_key(arm, ctx, q))
    return [8. if k == label else 0. for k in range(1, 5)]


class TeachingStream:
    def __init__(self, schedule, rules, cursor=0):
        self._schedule, self._rules = schedule, rules
        self.cursor = cursor

    def observe(self):
        ctx, q = self._schedule[self.cursor]
        return {'context': ctx, 'query': q, 'cursor': self.cursor}

    def feedback(self, journal, receipt):
        packet = journal.committed(receipt)
        exact(packet['observation'], self.observe())
        if packet['action'] not in range(1,5):
            raise ValueError('Invalid committed action')
        label = self._rules[packet['observation']['context']][packet['query'] - 1]
        evidence = ObservedEvidence(self.cursor, label)
        journal.last_receipt = None
        self.cursor += 1
        return evidence


def block(plan, seed, mapping):
    check_seed_safety(seed)
    if seed not in SEEDS or [seed, mapping] not in plan['development_units']:
        raise ValueError('Only the registered C0 development panel is available')
    return plan['schedules'][causal.unit_name(seed, mapping)], plan['mappings'][mapping]


def where(root, seed, mapping, mode, arm):
    return root / 'lives' / causal.unit_name(seed, mapping) / mode / arm


def train(job):
    root, plan, seed, mapping, mode, arm, resume = job
    start, cpu = time.perf_counter(), time.process_time()
    order, rules = block(plan, seed, mapping)
    cut_age = plan['cut']
    pending_name = f'pending_{cut_age:02d}.npz'
    origin = where(root, seed, mapping, mode, arm)
    path = where(root/'resume', seed, mapping, mode, arm) if resume else origin
    path.mkdir(parents=True, exist_ok=False)
    agent = new_agent('B3', seed, plan['parent']) if mode != 'table' else None
    table, pending = {}, None
    environment = TeachingStream(order, rules)
    journal = ActionJournal(path/'pre_feedback.jsonl')
    if resume:
        cut = load_state(origin/pending_name)
        assert cut['plan_sha256'] == digest(plan) and cut['condition'] == [mode, arm]
        table = cut['table']
        if agent is not None:
            agent.restore(cut['agent'])
        environment = TeachingStream(order, rules, cut['cursor'])
        pending = ActionJournal(origin/'pre_feedback.jsonl').read_verified(cut['receipt'])
        assert state_hash(agent.snapshot() if agent is not None else table) == pending['pending_sha256']
        receipt = journal.publish(pending)
    elif agent is not None:
        save_state(path/'start.npz', {'agent': agent.snapshot(), 'plan_sha256': digest(plan)})
    rows, timing = [], []
    while environment.cursor < len(order):
        begin = time.perf_counter_ns()
        observation = environment.observe()
        ctx, q, age = observation['context'], observation['query'], observation['cursor']
        if pending is None:
            if agent is not None:
                pending = forecast(agent, q, 'Z0-64', 'aligned', None, {}, stimulus=vector(mode, ctx, q))
            else:
                logits = table_logits(table, arm, ctx, q)
                pending = {'query': q, 'logits': logits, 'action': int(np.argmax(logits))+1,
                           'pending_sha256': state_hash(table)}
            pending['observation'] = observation
            receipt = journal.publish(pending)
        issued = time.perf_counter_ns()
        if age == cut_age and not resume:
            save_state(path/pending_name, {'agent': agent.snapshot() if agent is not None else None,
                       'table': table, 'receipt': receipt, 'cursor': age, 'condition': [mode,arm], 'plan_sha256': digest(plan)})
        evidence = environment.feedback(journal, receipt)
        revealed = time.perf_counter_ns()
        if agent is not None:
            update, _ = causal.apply_credit(agent, pending, evidence, arm, None)
        else:
            table[table_key(arm, ctx, q)] = evidence.target
            update = {'table_write': True}
        updated = time.perf_counter_ns()
        rows.append({'packet': pending, 'target_after': evidence.target, 'update': update,
                     'end_state': state_hash(agent.snapshot() if agent is not None else table)})
        timing.append({'receipt': receipt, 'issued_ns': issued, 'revealed_ns': revealed, 'updated_ns': updated,
                       'wall_ns': updated - begin})
        pending = None
    end = {'agent': agent.snapshot() if agent is not None else None, 'table': table, 'cursor': environment.cursor,
           'condition': [mode,arm], 'plan_sha256': digest(plan)}
    save_state(path/'end.npz', end)
    atomic_json(path/'events.json', rows)
    atomic_json(path/'runtime.json', timing)
    complete = {'seed': seed, 'mapping': mapping, 'condition': [mode,arm], 'events': len(rows),
                'wall_seconds': time.perf_counter()-start, 'cpu_seconds': time.process_time()-cpu}
    if resume:
        exact(rows, read_json(origin/'events.json')[cut_age:])
        assert state_hash(end) == state_hash(load_state(origin/'end.npz'))
        assert journal.path.read_bytes().splitlines(keepends=True) == (
            (origin/'pre_feedback.jsonl').read_bytes().splitlines(keepends=True)[cut_age:])
        complete['exact'] = True
    atomic_json(path/'COMPLETE.json', complete)
    return [seed, mapping, mode, arm]


def read_trace(agent, stimulus, start, k, ticks):
    core = copy.deepcopy(agent.core)
    core.Z = start.copy()
    core.K = k.copy()
    initial_rng = rng_hash(core)
    frozen = {name: array_hash(getattr(core, name)) for name in ('K','B','W_readout')}
    trace = {}
    before = core.step_count
    for tick in range(ticks + 1):
        if tick in (0, 1, 8, 32, ticks):
            trace[str(tick)] = {'Z': core.Z.tolist(), 'logits': core.predict()[1:5].tolist()}
        if tick != ticks:
            core.step(stimulus, write_gate=agent.read_gate)
    assert frozen == {name: array_hash(getattr(core, name)) for name in frozen}
    return {'trace': trace, 'hashes': frozen, 'rng_pre': initial_rng, 'rng_post': rng_hash(core),
            'substeps': core.step_count - before}


def cut_edges(k, direct, indirect):
    candidate = k.copy()
    if direct:
        candidate[32:40, 8:16] = 0.
    if indirect:
        candidate[40:48, 8:16] = 0.
    return candidate


def probe(job):
    root, plan, seed, mapping, mode, arm = job
    block(plan, seed, mapping)
    path = where(root, seed, mapping, mode, arm)
    end = load_state(path/'end.npz')
    assert end['plan_sha256'] == digest(plan)
    reads = {}
    if mode == 'table':
        for ctx in ('A','B'):
            for q in range(1,7):
                reads[f'{ctx}:{q}'] = table_logits(end['table'], arm, ctx, q)
        result = {'table': reads, 'substeps': 0}
    else:
        agent = new_agent('B3', seed, plan['parent'])
        agent.restore(end['agent'])
        full_hash = state_hash(agent.snapshot())
        before = load_state(path/'start.npz')['agent']['core_snapshot']
        donor_map = (mapping + 1) % len(plan['mappings'])
        donor = load_state(where(root, seed, donor_map, mode, arm)/'end.npz')['agent']['core_snapshot']['K']
        live, z0, learned, initial = agent.core.Z, agent.Z0, agent.core.K, before['K']
        variants = {'z0': (z0, learned, 64), 'live': (live, learned, 64), 'main_live': (live, learned, 8),
                    'initial_K': (z0, initial, 64), 'wrong_K': (z0, donor, 64), 'no_cue': (z0, learned, 64)}
        if [seed,mapping] == plan['trace_unit']:
            variants.update(initial_cut_E2E6=(z0, cut_edges(initial, False, True), 64),
                            learned_cut_E2E6=(z0, cut_edges(learned, False, True), 64),
                            learned_cut_E2E5=(z0, cut_edges(learned, True, False), 64),
                            learned_cut_both=(z0, cut_edges(learned, True, True), 64))
        for tag, (start, k, horizon) in variants.items():
            reads[tag] = {}
            for ctx in ('A','B'):
                for q in range(1,7):
                    x = vector(mode, ctx if tag != 'no_cue' else None, q)
                    trace = read_trace(agent, x, start, k, horizon)
                    trace['input'] = x.tolist()
                    trace['B_x'] = (agent.core.B @ x).tolist()
                    reads[tag][f'{ctx}:{q}'] = trace
        assert state_hash(agent.snapshot()) == full_hash
        result = {'reads': reads, 'source_hash': full_hash, 'wrong_mapping': donor_map,
                  'substeps': sum(r['substeps'] for group in reads.values() for r in group.values())}
    atomic_json(path/'PROBES.json', result)
    return [seed,mapping,mode,arm]


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('phase', choices=('init','train','probe','resume'))
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--reference', type=Path)
    cli.add_argument('--archive', type=Path)
    args = cli.parse_args()
    root, phase = args.output.resolve(), args.phase
    if phase == 'init':
        init(root, args.reference.resolve(), args.archive.resolve())
        return
    plan = verify_plan(root)
    units = [plan['resume_unit']] if phase == 'resume' else plan['development_units']
    jobs = [(root, plan, *unit, *condition) for unit in units for condition in CONDITIONS]
    function = probe if phase == 'probe' else train
    if function is train:
        jobs = [(*job, phase == 'resume') for job in jobs]
    status = root / f'STATUS_{phase}.json'
    start = time.perf_counter()
    atomic_json(status, {'phase':'RUNNING','completed':0,'total':len(jobs)})
    failures = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(function, job): job[2:] for job in jobs}
        for count, future in enumerate(as_completed(futures), 1):
            try:
                print(count, len(jobs), future.result(), flush=True)
            except Exception as exc:
                error = {'job':futures[future], 'type':type(exc).__name__, 'message':str(exc)}
                failures.append(error)
                print('FAILED', error, flush=True)
            atomic_json(status, {'phase':'RUNNING','completed':count,'total':len(jobs),'failures':failures})
    atomic_json(status, {'phase':'COMPLETED_WITH_FAILURES' if failures else 'COMPLETED', 'total':len(jobs),
                         'failures':failures, 'wall_seconds':time.perf_counter()-start})


if __name__ == '__main__':
    main()
