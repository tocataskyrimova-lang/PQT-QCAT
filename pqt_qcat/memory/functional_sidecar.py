"""Functional Sidecar B0/B1 on the fixed B3 development stream."""
from . import access_bridge as bridge  # set numerical thread caps before imports

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from .access_state import exact, load_agent
from ..core.security import check_seed_safety
from .checkpoint_io import load_state, save_state
from .codex_memory_location import atomic_json, digest, source_hashes
from .qnuf import ObservedEvidence, array_hash, rng_hash


BASE = 'c32e8892f4aa5a37c659b89e23c58eefe09ae024'
ARMS = {
    'M-LIVE8': {'ticks': 0, 'start': 'live', 'donor': 'own', 'padding': 64},
    'S-LIVE64': {'ticks': 64, 'start': 'live', 'donor': 'own', 'padding': 0},
    'S-Z0-64': {'ticks': 64, 'start': 'Z0', 'donor': 'own', 'padding': 0},
    'S-Z0-8': {'ticks': 8, 'start': 'Z0', 'donor': 'own', 'padding': 0},
    'S-Z0-64-K0': {'ticks': 64, 'start': 'Z0', 'donor': 'initial', 'padding': 0},
    'S-Z0-64-WRONG': {'ticks': 64, 'start': 'Z0', 'donor': 'wrong', 'padding': 0},
}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def state_hash(state):
    def walk(v):
        if isinstance(v, np.ndarray):
            return {'dtype': str(v.dtype), 'shape': list(v.shape), 'sha256': array_hash(v)}
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [walk(x) for x in v]
        if isinstance(v, np.generic):
            return v.item()
        return v
    return digest(walk(state))


class ActionJournal:
    """Append and fsync the public answer before permitting environment feedback."""

    def __init__(self, path):
        self.path = path
        self.last_receipt = None

    def publish(self, packet):
        if self.last_receipt is not None:
            raise RuntimeError('Outstanding action')
        line = canonical(packet) + b'\n'
        with self.path.open('ab') as stream:
            offset = stream.tell()
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        self.last_receipt = {'offset': offset, 'length': len(line), 'sha256': hashlib.sha256(line).hexdigest()}
        return self.last_receipt.copy()

    def restore_receipt(self, receipt):
        if self.last_receipt is not None:
            raise RuntimeError('Outstanding action')
        self.read_verified(receipt)
        self.last_receipt = receipt.copy()

    def read_verified(self, receipt):
        with self.path.open('rb') as stream:
            stream.seek(receipt['offset'])
            line = stream.read(receipt['length'])
        if hashlib.sha256(line).hexdigest() != receipt['sha256']:
            raise RuntimeError('Action bytes differ from commit receipt')
        return json.loads(line)

    def committed(self, receipt):
        if receipt is None or receipt != self.last_receipt:
            raise RuntimeError('No matching durable action')
        return self.read_verified(receipt)


class FixedEnvironment:
    """Action determines reward, while the archived labels and queries stay fixed."""

    def __init__(self, rows, cursor):
        self._rows = rows
        self.cursor = cursor

    def observe(self):
        return self._rows[self.cursor]['query']

    def feedback(self, journal, receipt):
        packet = journal.committed(receipt)
        if packet['event_id'] != self.cursor or packet['query'] != self.observe():
            raise RuntimeError('Action does not match pending environment event')
        action = packet['deployed_action']
        if action not in range(1, 5):
            raise ValueError('Action outside declared alphabet')
        target = self._rows[self.cursor]['target']
        evidence = ObservedEvidence(self.cursor, target)
        self.cursor += 1
        journal.last_receipt = None
        return evidence, int(action == target)


def checkpoint_path(plan, arm, seed, mapping, cursor):
    return bridge.checkpoint(Path(plan['historical_source']), arm, seed, mapping, cursor)


def initialize(history, stage_a, brief, output):
    import importlib.metadata
    import sys
    import zipfile

    delivery = read_json(stage_a / 'DELIVERY.json')
    source_a = read_json(stage_a / 'SOURCE.json')
    if delivery['commit'] != BASE:
        raise ValueError('Stage A delivery SHA differs')
    root = bridge.ROOT
    subprocess.run(['git', 'merge-base', '--is-ancestor', BASE, 'HEAD'], cwd=root, check=True)
    manifest_a = read_json(stage_a / 'SHA256.json')
    for name, expected in manifest_a.items():
        if bridge.file_hash(stage_a / name) != expected:
            raise ValueError(f'Stage A manifest mismatch: {name}')
    with zipfile.ZipFile(stage_a / 'FROZEN_CODE.zip') as archive:
        for name, expected in source_a['files'].items():
            if hashlib.sha256(archive.read(name.replace(chr(92), '/'))).hexdigest() != expected:
                raise ValueError(f'Stage A frozen source mismatch: {name}')
    a_plan = read_json(stage_a / 'PLAN.json')
    parent = read_json(history / 'frozen_config.json')
    exact(a_plan['parent'], parent)
    units = [(seed, mapping) for arm, seed, mapping in a_plan['units'] if arm == 'B3']
    if len(units) != 18 or len({tuple(u) for u in units}) != 18:
        raise ValueError('Expected the exact Stage A B3 panel')
    output.mkdir(parents=True, exist_ok=False)
    qnuf_manifest = read_json(history / 'sha256_manifest.json')
    files = {'frozen_config.json'}
    a_files = {'PLAN.json', 'DELIVERY.json', 'SOURCE.json', 'ENVIRONMENT.json', 'EVENT49_DONOR_ANALYSIS.json'}
    for seed, mapping in units:
        check_seed_safety(seed)
        for name, cursor in [('B3', 48), ('B3', 72), ('B0', 0)]:
            files.add(bridge.checkpoint(history, name, seed, mapping, cursor).relative_to(history).as_posix())
        lib = bridge.life_path(history, 'B3', seed, mapping)
        files.add((lib / 'event_records.json').relative_to(history).as_posix())
        for a in ('sidecar_live64', 'sidecar_Z0'):
            a_files.add(f'B3_{seed}_{mapping}/continuity/{a}/sidecar_pre_feedback.json')
    plan = {
        'base': BASE, 'stage_a': str(stage_a), 'historical_source': str(history),
        'parent': parent, 'units': units, 'arm_order': list(ARMS), 'arms': ARMS,
        'cursor_start': 48, 'cursor_stop': 72, 'resume_after': 56,
        'wrong_mapping': 'cyclic +1, same seed and same pre-event age',
        'feedback': 'fixed paired supervised labels after durable action, reward=1[action==target]',
        'bootstrap_replicates': 5000, 'bootstrap_seed': parent['bootstrap_seed'],
        'primary': 'S-Z0-64 minus S-LIVE64 on events 49–72, paired seed/mapping',
        'threshold_pp': 10., 'worker_limit': 2,
        'inputs': {str(history): {f: qnuf_manifest[f] for f in sorted(files)},
                   str(stage_a): {f: manifest_a[f] for f in sorted(a_files)}},
    }
    verify_artifacts(plan)
    versions = {key: importlib.metadata.version(key) for key in ('numpy', 'scipy', 'torch', 'scikit-learn', 'pytest')}
    environment_a = read_json(stage_a / 'ENVIRONMENT.json')
    for key in ('numpy', 'scipy', 'torch', 'scikit-learn'):
        exact(environment_a['packages'][key], versions[key])
    atomic_json(output / 'PLAN.json', plan)
    atomic_json(output / 'SOURCE.json', {
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'files': source_hashes(), 'plan_sha256': digest(plan), 'dependencies': versions, 'python': sys.version,
        'stage_a_manifest_sha256': bridge.file_hash(stage_a / 'SHA256.json'),
        'stage_a_manifest_verified_files': len(manifest_a), 'brief_sha256': bridge.file_hash(brief)})
    (output / 'BRIEF.zip').write_bytes(brief.read_bytes())
    atomic_json(output / 'seed_registry.json', {'development': sorted({s for s, _ in units}),
                 'new_seeds': [], 'forbidden': [101, 102, 103, 104, 105],
                 'guard': 'hard-coded core check and exact Stage A panel, not configurable'})


def verify_artifacts(plan):
    for directory, entries in plan['inputs'].items():
        for name, expected in entries.items():
            if bridge.file_hash(Path(directory) / name) != expected:
                raise ValueError(f'Input hash mismatch: {directory}/{name}')


def verify(output, plan):
    source = read_json(output / 'SOURCE.json')
    if source['files'] != source_hashes() or source['plan_sha256'] != digest(plan):
        raise ValueError('Frozen source or plan changed')
    verify_artifacts(plan)


def validate_unit(plan, seed, mapping):
    check_seed_safety(seed)
    if [seed, mapping] not in plan['units'] or not 0 <= mapping < len(plan['parent']['stage1_mappings']):
        raise ValueError('Unit is outside the historical B3 panel')


def archived_events(plan, seed, mapping):
    return read_json(bridge.life_path(Path(plan['historical_source']), 'B3', seed, mapping) / 'event_records.json')


def wrong_trajectory(plan, seed, mapping):
    wrong_map = (mapping + 1) % len(plan['parent']['stage1_mappings'])
    validate_unit(plan, seed, wrong_map)
    start = load_state(checkpoint_path(plan, 'B3', seed, wrong_map, plan['cursor_start']))['agent']
    k = start['core_snapshot']['K'].copy()
    w = start['core_snapshot']['W_readout']
    rows = archived_events(plan, seed, wrong_map)
    states = {}
    for i in range(plan['cursor_start'], plan['cursor_stop']):
        row = rows[i]
        if row['event_id'] != i or row['K_pre_hash'] != array_hash(k):
            raise RuntimeError('Wrong-map pre-event matrix does not match archive')
        states[i] = k.copy()
        e = w.T @ np.array(row['error_vector'])
        delta = np.clip(plan['parent']['eta'] * np.outer(e, row['phi_event']),
                        -plan['parent']['clip'], plan['parent']['clip'])
        k += delta
        if row['K_post_hash'] != array_hash(k):
            raise RuntimeError('Wrong-map replay does not match archive')
    return wrong_map, states, start['core_snapshot']


def core_read(core, z0, query, spec, gate, donor_k):
    """Compute a frozen read on a disposable core. Padding publishes no logits."""
    if spec['start'] == 'Z0':
        core.Z = z0.copy()
    if donor_k is not None:
        core.K = donor_k.copy()
    k, b, w = (array_hash(value) for value in (core.K, core.B, core.W_readout))
    physical = (copy.deepcopy(core.config), core.vortex_params, core.integrator.config)
    z_hash, rng_pre = array_hash(core.Z), rng_hash(core)
    start = core.step_count
    stimulus = np.eye(8)[query] if spec['ticks'] else None
    for _ in range(spec['ticks'] + spec['padding']):
        core.step(stimulus, write_gate=gate)
    assert (k, b, w) == tuple(array_hash(value) for value in (core.K, core.B, core.W_readout))
    assert physical == (core.config, core.vortex_params, core.integrator.config)
    return {'logits': core.predict()[1:5].tolist() if spec['ticks'] else None,
            'K_hash': k, 'Z_start_hash': z_hash, 'Z_end_hash': array_hash(core.Z),
            'rng_pre_hash': rng_pre, 'rng_post_hash': rng_hash(core), 'substeps': core.step_count - start}


def predict_packet(agent, query, arm, donor_k, spec):
    t0 = time.perf_counter_ns()
    pre = agent.snapshot()
    clone = copy.deepcopy(agent.core)
    setup_ns = time.perf_counter_ns() - t0
    start_steps, start_shadow = agent.core.step_count, agent.shadow_steps
    t1 = time.perf_counter_ns()
    main = agent.begin_query(query)
    main_ns = time.perf_counter_ns() - t1
    guard_start = time.perf_counter_ns()
    main_pending = agent.snapshot()
    guard_ns = time.perf_counter_ns() - guard_start
    t2 = time.perf_counter_ns()
    read = core_read(clone, agent.Z0, query, spec, agent.read_gate, donor_k)
    read_ns = time.perf_counter_ns() - t2
    t3 = time.perf_counter_ns()
    after = agent.snapshot()
    exact(main_pending, after)
    if main['Z_pre_hash'] != array_hash(pre['core_snapshot']['Z']):
        raise RuntimeError('Main Z changed before integration')
    assert main['rng_pre_hash'] == read['rng_pre_hash']
    guard_ns += time.perf_counter_ns() - t3
    logits = read['logits'] if spec['ticks'] else main['logits']
    packet = {'event_id': main['event_id'], 'query': query, 'arm': arm,
              'main': main, 'deployed_logits': logits, 'deployed_action': int(np.argmax(logits)) + 1,
              'sidecar_or_padding': read, 'pre_state_sha256': state_hash(pre),
              'main_pending_sha256': state_hash(main_pending), 'core_clones': 2,
              'main_snapshot_copies': 3,
              'substeps_main': agent.core.step_count - start_steps,
              'substeps_shadow': agent.shadow_steps - start_shadow,
              'substeps_sidecar': read['substeps'] if spec['ticks'] else 0,
              'substeps_padding': read['substeps'] if spec['padding'] else 0}
    costs = {'main_predict_ns': main_ns, 'sidecar_ns': read_ns if spec['ticks'] else 0,
             'padding_ns': read_ns if spec['padding'] else 0, 'snapshot_setup_ns': setup_ns, 'guard_ns': guard_ns}
    return packet, pre, costs


def proceed(agent, environment, journal, receipt, expected):
    packet = journal.committed(receipt)
    exact(packet['main'], agent.pending['prediction'])
    feedback, reward = environment.feedback(journal, receipt)
    reveal_ns = time.perf_counter_ns()
    update = agent.finish_query(feedback)
    exact({**packet['main'], **update}, expected)
    return {'packet': packet, 'feedback': {'target': feedback.target, 'reward': reward,
                                           'label_reveal_ns': reveal_ns}, 'update': update}


def run_life(plan, folder, seed, mapping, arm, wrong, canonical_dir=None, resume=None):
    validate_unit(plan, seed, mapping)
    folder.mkdir(parents=True, exist_ok=False)
    agent = load_agent(plan, 'B3', seed, mapping, plan['cursor_start'])
    cursor = plan['cursor_start']
    rows = archived_events(plan, seed, mapping)
    spec = ARMS[arm]
    initial = load_state(checkpoint_path(plan, 'B0', seed, mapping, 0))['agent']['core_snapshot']
    own_core = agent.snapshot()['core_snapshot']
    for state in (initial, wrong[2]):
        for field in ('B', 'W_readout', 'config', 'integrator_config', 'vortex_params', 'cell_params', 'substeps'):
            exact(state[field], own_core[field])
    records = []
    pending_receipt = None
    journal = ActionJournal(folder / 'actions_before_feedback.jsonl')
    if resume is not None:
        saved = load_state(resume)
        exact(saved['runner']['arm'], arm)
        exact(saved['runner']['unit'], [seed, mapping])
        exact(saved['runner']['plan_sha256'], digest(plan))
        agent.restore(saved['agent'])
        cursor = saved['runner']['cursor']
        if saved['runner']['receipt'] is not None:
            source = ActionJournal(resume.parent / 'actions_before_feedback.jsonl')
            packet = source.read_verified(saved['runner']['receipt'])
            exact(state_hash(agent.snapshot()), packet['main_pending_sha256'])
            pending_receipt = journal.publish(packet)
    env = FixedEnvironment(rows, cursor)
    checkpoint_io = []
    snapshot_copies = 1  # own_core snapshot for donor compatibility above
    restored_snapshots = int(resume is not None)

    def save(name, receipt=None):
        nonlocal snapshot_copies
        started = time.perf_counter_ns()
        agent.save_checkpoint(folder / name, {'cursor': env.cursor, 'receipt': receipt,
                              'arm': arm, 'unit': [seed, mapping], 'plan_sha256': digest(plan)})
        snapshot_copies += 1
        checkpoint_io.append({'file': name, 'ns': time.perf_counter_ns() - started,
                              'bytes': (folder / name).stat().st_size})

    if resume is None:
        save('checkpoint_048.npz')
    while env.cursor < plan['cursor_stop']:
        event = env.cursor
        if pending_receipt is not None:
            record = proceed(agent, env, journal, pending_receipt, rows[event])
            pending_receipt = None
        else:
            t0 = time.perf_counter_ns()
            matrix = initial['K'] if spec['donor'] == 'initial' else (
                wrong[1][event] if spec['donor'] == 'wrong' else None)
            packet, pre, costs = predict_packet(agent, env.observe(), arm, matrix, spec)
            if canonical_dir is not None:
                name = f'pre_{event + 1:03d}.npz'
                start = time.perf_counter_ns()
                save_state(canonical_dir / name, {'agent': pre, 'cursor': event})
                checkpoint_io.append({'file': 'pre_event/' + name, 'ns': time.perf_counter_ns() - start,
                                      'bytes': (canonical_dir / name).stat().st_size})
            before_commit = time.perf_counter_ns()
            receipt = journal.publish(packet)
            published = time.perf_counter_ns()
            costs.update(commit_io_ns=published - before_commit, audited_action_ns=published - t0)
            if spec['ticks']:
                costs['compute_action_ns'] = costs['main_predict_ns'] + costs['sidecar_ns']
            else:
                costs['compute_action_ns'] = costs['main_predict_ns']
            if resume is None and event == plan['resume_after']:
                save('pending_057.npz', receipt)
            record = proceed(agent, env, journal, receipt, rows[event])
            assert published < record['feedback']['label_reveal_ns']
            record['cost_ns'] = costs
            record['committed_ns'] = published
            record['receipt'] = receipt
            snapshot_copies += packet['main_snapshot_copies']
        records.append(record)
        if resume is None and env.cursor == plan['resume_after']:
            save('checkpoint_056.npz')
    exact(load_state(checkpoint_path(plan, 'B3', seed, mapping, 72))['agent'], agent.snapshot())
    snapshot_copies += 1
    save('checkpoint_072.npz')
    atomic_json(folder / 'events.json', records)
    result = {'arm': arm, 'events': len(records), 'identity_exact': True,
              'snapshot_copies': snapshot_copies, 'restored_snapshots': restored_snapshots,
              'checkpoint_io': checkpoint_io}
    atomic_json(folder / 'life.json', result)
    return result


def run_block(task):
    plan, output, seed, mapping = task
    validate_unit(plan, seed, mapping)
    folder = output / 'blocks' / f'{seed}_{mapping}'
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    cpu = time.process_time()
    wrong = wrong_trajectory(plan, seed, mapping)
    canonical_dir = folder / 'pre_event'
    canonical_dir.mkdir()
    for arm in plan['arm_order']:
        run_life(plan, folder / arm, seed, mapping, arm, wrong,
                 canonical_dir if arm == 'M-LIVE8' else None)
    row = {'unit': [seed, mapping], 'wrong_mapping': wrong[0],
           'wall_seconds': time.perf_counter() - started, 'cpu_seconds': time.process_time() - cpu}
    atomic_json(folder / 'COMPLETE.json', row)
    return row


def preflight(output, plan):
    verify(output, plan)
    checks = []
    for seed, mapping in plan['units']:
        validate_unit(plan, seed, mapping)
        agent = load_agent(plan, 'B3', seed, mapping, 48)
        environment = FixedEnvironment(archived_events(plan, seed, mapping), 48)
        correct = {}
        for arm, stage_a in (('S-Z0-64', 'sidecar_Z0'), ('S-LIVE64', 'sidecar_live64')):
            clone = copy.deepcopy(agent)
            packet, _, _ = predict_packet(clone, environment.observe(), arm, None, ARMS[arm])
            ref = read_json(Path(plan['stage_a']) / f'B3_{seed}_{mapping}' / 'continuity' / stage_a
                            / 'sidecar_pre_feedback.json')
            exact(packet['deployed_logits'], ref['horizons']['64']['logits'])
            correct[arm] = int(packet['deployed_action'] == plan['parent']['stage1_mappings'][mapping][packet['query'] - 1])
        checks.append({'unit': [seed, mapping], 'hits': correct, 'logits_bitwise_exact': True})
    hits = {arm: sum(x['hits'][arm] for x in checks) for arm in ('S-Z0-64', 'S-LIVE64')}
    exact(hits, {'S-Z0-64': 14, 'S-LIVE64': 4})
    atomic_json(output / 'B0_anchor.json', {'phase': 'PASS', 'counts': hits, 'checks': checks})


def resume_test(output, plan, seed, mapping):
    verify(output, plan)
    validate_unit(plan, seed, mapping)
    wrong = wrong_trajectory(plan, seed, mapping)
    suite = output / 'resume_checks'
    suite.mkdir(exist_ok=False)
    source = output / 'blocks' / f'{seed}_{mapping}'
    checks = []
    for arm in plan['arm_order']:
        for name, start in [('checkpoint_056.npz', 56), ('pending_057.npz', 56)]:
            folder = suite / arm / name.removesuffix('.npz')
            run_life(plan, folder, seed, mapping, arm, wrong, resume=source / arm / name)
            reference = read_json(source / arm / 'events.json')[start - plan['cursor_start']:]
            replay = read_json(folder / 'events.json')
            for a, b in zip(reference, replay, strict=True):
                for field in ('packet', 'update'):
                    exact(a[field], b[field])
                for field in ('target', 'reward'):
                    exact(a['feedback'][field], b['feedback'][field])
            exact(load_state(source / arm / 'checkpoint_072.npz'), load_state(folder / 'checkpoint_072.npz'))
            checks.append({'arm': arm, 'resume': name, 'verified_events': len(replay), 'exact': True})
    atomic_json(output / 'RESUME_CHECKS.json', {'unit': [seed, mapping], 'checks': checks, 'separate_process': True})


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('action', choices=('init', 'preflight', 'run', 'resume-test'))
    cli.add_argument('--output', required=True, type=Path)
    cli.add_argument('--history', type=Path)
    cli.add_argument('--stage-a', type=Path)
    cli.add_argument('--brief', type=Path)
    cli.add_argument('--workers', type=int, default=2)
    args = cli.parse_args()
    output = args.output.resolve()
    if args.action == 'init':
        initialize(args.history.resolve(), args.stage_a.resolve(), args.brief.resolve(), output)
        return
    plan = read_json(output / 'PLAN.json')
    verify(output, plan)
    if args.action == 'preflight':
        preflight(output, plan)
        return
    if args.action == 'resume-test':
        resume_test(output, plan, *plan['units'][0])
        return
    if not 1 <= args.workers <= plan['worker_limit']:
        cli.error('Use one or two workers')
    if read_json(output / 'B0_anchor.json')['phase'] != 'PASS' or not (output / 'B0_TESTS.xml').exists():
        raise RuntimeError('B0 tests and anchor must pass before B1')
    import xml.etree.ElementTree as ET
    test_suites = list(ET.parse(output / 'B0_TESTS.xml').getroot().iter('testsuite'))
    if not test_suites or any(int(x.attrib['failures']) + int(x.attrib['errors']) for x in test_suites):
        raise RuntimeError('B0 tests have failures')
    stage = output / 'STATUS.json'
    started = time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, row in enumerate(pool.map(run_block, [(plan, output, *u) for u in plan['units']]), 1):
                atomic_json(stage, {'phase': 'RUNNING', 'blocks': i, 'last': row['unit']})
                print(f"{i}/{len(plan['units'])} block {row['unit']}", flush=True)
    except Exception as exc:
        atomic_json(stage, {'phase': 'STOP_INVALID', 'error': str(exc), 'type': type(exc).__name__})
        raise
    atomic_json(stage, {'phase': 'COMPLETED', 'blocks': i, 'wall_seconds': time.perf_counter() - started})


if __name__ == '__main__':
    main()
