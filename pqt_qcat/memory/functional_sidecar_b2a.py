"""Action-dependent, fixed-rule development experiment after Functional Sidecar B1."""
from . import functional_sidecar as b1  # caps numerical worker threads before NumPy import

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from .access_state import exact, load_agent
from .checkpoint_io import load_state, save_state
from .codex_memory_location import atomic_json, digest, metrics, source_hashes
from .qnuf import ObservedEvidence, array_hash, rng_hash
from ..core.security import check_seed_safety

BASE = 'aaf51b9b886e9223cbe07b5b0871e8c12ab42ba5'
READERS = ('CARRY', 'LIVE64', 'Z0-64')
FACTORIAL = {f'{reader}_{credit}': (reader, credit, 'own')
             for reader in READERS for credit in ('main', 'aligned')}
CONTROLS = {'Z0-64_K0': ('Z0-64', 'main', 'initial'),
            'Z0-64_WRONG': ('Z0-64', 'main', 'wrong')}
TABLE = {'TABLE': ('TABLE', 'main', 'own')}
ARMS = {**FACTORIAL, **CONTROLS, **TABLE}


class RetryEnvironment:
    """Wrong actions repeat the query; correct actions advance by a fixed shared offset."""

    def __init__(self, mapping, offsets, query, cursor=0, streak=0):
        self._labels = mapping
        self._offsets = offsets
        self.query = query
        self.cursor = cursor
        self.streak = streak
        self._reaction = None

    def observe(self):
        if self._reaction is not None:
            raise RuntimeError('Release evidence before taking another observation')
        return self.query

    def snapshot(self):
        assert self._reaction is None
        return {'query': self.query, 'cursor': self.cursor, 'streak': self.streak}

    @staticmethod
    def transition(query, action, target, offset, streak):
        if action not in range(1, 5) or offset not in range(1, 6):
            raise ValueError('Action or common offset outside the environment alphabet')
        if action == target:
            return {'next_query': (query - 1 + offset) % 6 + 1, 'reward': 1.,
                    'correct': True, 'next_streak': 0}
        return {'next_query': query, 'reward': -.25 * (1 + min(streak, 3)),
                'correct': False, 'next_streak': streak + 1}

    def react(self, journal, receipt):
        packet = journal.committed(receipt)
        if packet['cursor'] != self.cursor or packet['query'] != self.observe():
            raise RuntimeError('Action does not match the current observation')
        target = self._labels[self.query - 1]
        decision = self.transition(self.query, packet['action'], target, self._offsets[self.cursor], self.streak)
        self._reaction = (packet, target, decision)
        journal.last_receipt = None
        return decision.copy()

    def reveal(self):
        if self._reaction is None:
            raise RuntimeError('Evidence requires an environment reaction to a committed action')
        packet, target, outcome = self._reaction
        self.query = outcome['next_query']
        self.streak = outcome['next_streak']
        self.cursor += 1
        self._reaction = None
        return ObservedEvidence(packet['main']['event_id'], target)


def input_files(plan):
    q, old = Path(plan['historical_source']), Path(plan['b1_source'])
    history = {'frozen_config.json'}
    for seed, mapping in plan['units']:
        for arm, cursor in (('B0', 0), ('B3', 48)):
            history.add(b1.bridge.checkpoint(q, arm, seed, mapping, cursor).relative_to(q).as_posix())
        history.add((b1.bridge.life_path(q, 'B3', seed, mapping)
                     / 'event_records.json').relative_to(q).as_posix())
    prior = {'PLAN.json', 'SOURCE.json', 'DELIVERY.json', 'results/stage_b1/negative_controls.json',
             'results/stage_b1/contrasts.json', 'results/stage_b1/continuity.json'}
    return ((q, 'sha256_manifest.json', history), (old, 'SHA256.json', prior))


def init(history, b1_source, output):
    import importlib.metadata
    import sys
    prior = b1.read_json(b1_source / 'DELIVERY.json')
    if prior['delivery_commit'] != BASE or prior['scientific'] != 'B1_DEPLOYED_SIGNAL':
        raise ValueError('Wrong or unaccepted B1 result')
    subprocess.run(['git', 'merge-base', '--is-ancestor', BASE, 'HEAD'], cwd=b1.bridge.ROOT, check=True)
    parent = b1.read_json(history / 'frozen_config.json')
    b1_plan = b1.read_json(b1_source / 'PLAN.json')
    exact(parent, b1_plan['parent'])
    # Fixed prefix of mapping IDs on each historical seed; chosen before B2A outcomes.
    units = [unit for unit in b1_plan['units'] if unit[1] < 4]
    assert len(units) == 12
    for seed, _ in units:
        check_seed_safety(seed)
    plan = {'base': BASE, 'historical_source': str(history), 'b1_source': str(b1_source),
            'parent': parent, 'units': units, 'steps': 24, 'cursor_start': 48, 'ticks': 64,
            'arms': {name: list(meta) for name, meta in ARMS.items()}, 'order': list(ARMS),
            'selection': 'three prior development seeds, first four existing mapping IDs, before B2A',
            'threshold_return': 2., 'main_retention_gain': .10, 'table_gate': .75,
            'bootstrap_seed': parent['bootstrap_seed'], 'bootstrap_replicates': 5000,
            'wrong_donor': '(mapping+1) mod 4, own Z0-64/main B2A continuation, same elapsed event age',
            'environment': 'reward +1 for correct, else -0.25*(1+min(prior consecutive misses,3)); '
                           'fail retries current query, success shifts by common nonzero offset; horizon 24',
            'thresholds': 'point gain >= threshold and paired lower CI > 0; superiority against both controls',
            'credit_aligned': 'deployed logits and matched eligibility from the same read branch',
            'saved_cut': 'pending event at age 60, action durable before reaction and label',
            'scope': 'B2A development only, single mapping'}
    plan['inputs'] = {}
    for root, manifest_name, files in input_files(plan):
        manifest = b1.read_json(root / manifest_name)
        plan['inputs'][str(root)] = {name: manifest[name] for name in sorted(files)}
    b1.verify_artifacts(plan)
    streams = {}
    for seed, mapping in units:
        rows = b1.archived_events(plan, seed, mapping)
        values = [1 + int.from_bytes(hashlib.sha256(
            f'LIFE-B2A-v1/{seed}/{mapping}/{i}'.encode()).digest()[:8], 'big') % 5
                  for i in range(plan['steps'])]
        streams[f'{seed}_{mapping}'] = {'start_query': rows[48]['query'], 'offsets': values}
    plan['streams'] = streams
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'PLAN.json', plan)
    atomic_json(output / 'SOURCE.json', {
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=b1.bridge.ROOT, text=True).strip(),
        'files': source_hashes(), 'plan_sha256': digest(plan), 'python': sys.version,
        'dependencies': {name: importlib.metadata.version(name) for name in ('numpy', 'torch', 'scipy', 'pytest')},
    })


def read_branch(core, z0, query, reader, donor, side_shadow, gate, *, stimulus=None):
    if reader == 'Z0-64':
        core.Z = z0.copy()
    if donor is not None:
        core.K = donor.copy()
    frozen = {name: array_hash(getattr(core, name)) for name in ('K', 'B', 'W_readout')}
    shadow = copy.deepcopy(core) if side_shadow else None
    z_start = core.Z.copy()
    rng_pre = rng_hash(core)
    if stimulus is None:
        stimulus = np.eye(8)[query]
    start_steps = core.step_count
    for _ in range(64):
        core.step(stimulus, write_gate=gate)
        if shadow is not None:
            shadow.step(None, write_gate=gate)
    assert frozen == {name: array_hash(getattr(core, name)) for name in frozen}
    if shadow is not None:
        assert rng_hash(core) == rng_hash(shadow)
        assert frozen['K'] == array_hash(shadow.K)
    logits = core.predict()[1:5]
    return {'logits': logits.tolist(), 'Z_start': z_start.tolist(), 'Z_after': core.Z.tolist(),
            'Z_no_input': shadow.Z.tolist() if shadow is not None else None,
            'eligibility': (np.tanh(core.Z) - np.tanh(shadow.Z)).tolist() if shadow is not None else None,
            'hashes': frozen, 'rng_pre': rng_pre, 'rng_post': rng_hash(core),
            'side_substeps': core.step_count - start_steps,
            'side_shadow_substeps': shadow.step_count - start_steps if shadow is not None else 0}


def forecast(agent, query, reader, credit, donor, table, *, stimulus=None):
    pre = agent.snapshot()
    clone = copy.deepcopy(agent.core) if reader in ('Z0-64', 'LIVE64') else None
    start_steps, start_shadow = agent.core.step_count, agent.shadow_steps
    main = agent.begin_query(query) if stimulus is None else agent.begin_query(query, stimulus=stimulus)
    pending = agent.snapshot()
    if clone is not None:
        external = read_branch(clone, agent.Z0, query, reader, donor,
                               credit == 'aligned', agent.read_gate, stimulus=stimulus)
        assert main['rng_pre_hash'] == external['rng_pre']
    else:
        external = None
    exact(pending, agent.snapshot())
    assert array_hash(pre['core_snapshot']['Z']) == main['Z_pre_hash']
    if reader == 'TABLE':
        logits = np.zeros(4)
        if str(query) in table:
            logits[table[str(query)] - 1] = 8.
        logits = logits.tolist()
    else:
        logits = external['logits'] if external is not None else main['logits']
    return {'query': query, 'main': main, 'external': external, 'logits': logits,
            'action': int(np.argmax(logits)) + 1, 'pre_sha256': b1.state_hash(pre),
            'pending_sha256': b1.state_hash(pending),
            'budget': {'main': agent.core.step_count - start_steps,
                       'main_shadow': agent.shadow_steps - start_shadow,
                       'side': external['side_substeps'] if external else 0,
                       'side_shadow': external['side_shadow_substeps'] if external else 0,
                       'snapshots': 3, 'clones': 1 + int(external is not None) + int(
                           external is not None and credit == 'aligned')}}


def learn(agent, packet, evidence, credit):
    """Bind logits and eligibility of the same branch, only after evidence."""
    exact(agent.pending['prediction'], packet['main'])
    label = 'main'
    if credit == 'aligned' and packet['external'] is not None:
        external = packet['external']
        logits = np.array(packet['logits'])
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        # Archive the historical main prediction separately. Here, after reveal,
        # substitute only the credit values in the disposable pending payload.
        true_main = agent.pending['prediction']
        agent.pending['prediction'] = {**true_main, 'logits': logits.tolist(),
                                       'probabilities': probabilities.tolist(), 'prediction': packet['action']}
        agent.pending['event'] = np.array(external['eligibility'])
        label = 'external'
    return label, agent.finish_query(evidence)


def probe(agent, targets):
    live, reset = [], []
    before = b1.state_hash(agent.snapshot())
    for state, output in ((False, live), (True, reset)):
        for q in range(1, 7):
            clone = copy.deepcopy(agent.core)
            if state:
                clone.Z = agent.Z0.copy()
            for _ in range(8):
                clone.step(np.eye(8)[q], write_gate=agent.read_gate)
            output.append(clone.predict()[1:5].tolist())
    assert before == b1.state_hash(agent.snapshot())
    return {'live8': metrics(live, targets), 'Z0_8': metrics(reset, targets), 'substeps': 12 * 16}


def donor_for_age(output, seed, mapping, age):
    return load_state(output / 'lives' / f'{seed}_{(mapping + 1) % 4}' / 'Z0-64_main'
                      / f'pre_{age:02d}.npz')['agent']['core_snapshot']['K']


def validate_unit(plan, seed, mapping):
    check_seed_safety(seed)
    if [seed, mapping] not in plan['units']:
        raise ValueError('Unit is outside frozen development panel')


def run_life(job):
    plan, root, output, seed, mapping, arm, operation = job
    validate_unit(plan, seed, mapping)
    reader, credit, donor_mode = ARMS[arm]
    name = f'{seed}_{mapping}'
    folder = output / 'lives' / name / arm
    folder.mkdir(parents=True, exist_ok=False)
    agent = load_agent(plan, 'B3', seed, mapping, plan['cursor_start'])
    targets = plan['parent']['stage1_mappings'][mapping]
    stream = plan['streams'][name]
    environment = RetryEnvironment(targets, stream['offsets'], stream['start_query'])
    table = {}
    initial = agent.snapshot()['core_snapshot']
    k0 = load_state(b1.checkpoint_path(plan, 'B0', seed, mapping, 0))['agent']['core_snapshot']['K']
    journal = b1.ActionJournal(folder / 'pre_feedback.jsonl')
    started = time.perf_counter()
    cpu = time.process_time()
    results, runtime = [], []
    if operation == 'resume':
        original_folder = root / 'lives' / name / arm
        checkpoint = load_state(original_folder / 'pending_12.npz')
        exact(checkpoint['plan_sha256'], digest(plan))
        agent.restore(checkpoint['agent'])
        table = checkpoint['table']
        environment = RetryEnvironment(targets, stream['offsets'], **checkpoint['environment'])
        original_journal = b1.ActionJournal(original_folder / 'pre_feedback.jsonl')
        pending = original_journal.read_verified(checkpoint['receipt'])
        assert b1.state_hash(agent.snapshot()) == pending['pending_sha256']
        receipt = journal.publish(pending)
    else:
        pending = None
    probe_before = probe(agent, targets) if operation == 'primary' else None
    while environment.cursor < plan['steps']:
        age = environment.cursor
        observe = environment.snapshot()
        start_ns = time.perf_counter_ns()
        if pending is None:
            query = environment.observe()
            if arm == 'Z0-64_main' and operation == 'primary':
                save_state(folder / f'pre_{age:02d}.npz', {'agent': agent.snapshot(), 'age': age})
            donor = k0 if donor_mode == 'initial' else (
                donor_for_age(root, seed, mapping, age) if donor_mode == 'wrong' else None)
            pending = forecast(agent, query, reader, credit, donor, table)
            pending.update(cursor=age, observation=observe)
            receipt = journal.publish(pending)
        publish_ns = time.perf_counter_ns()
        if age == 12 and operation == 'primary':
            save_state(folder / 'pending_12.npz', {'plan_sha256': digest(plan), 'agent': agent.snapshot(),
                       'environment': observe, 'table': table, 'receipt': receipt})
        reaction = environment.react(journal, receipt)
        reaction_ns = time.perf_counter_ns()
        evidence = environment.reveal()
        reveal_ns = time.perf_counter_ns()
        if reader == 'TABLE':
            table[str(pending['query'])] = evidence.target
        label, update = learn(agent, pending, evidence, credit)
        update_ns = time.perf_counter_ns()
        assert publish_ns <= reaction_ns <= reveal_ns <= update_ns
        target = evidence.target
        alternate = RetryEnvironment.transition(pending['query'], 1 + target % 4, target,
                                                stream['offsets'][age], observe['streak'])
        truthful = RetryEnvironment.transition(pending['query'], target, target,
                                               stream['offsets'][age], observe['streak'])
        assert alternate['next_query'] != truthful['next_query']
        results.append({'packet': pending, 'reaction': reaction, 'evidence_target': target,
                        'exogenous_offset': stream['offsets'][age], 'update_source': label, 'update': update,
                        'counterfactual_next_queries': [truthful['next_query'], alternate['next_query']],
                        'end_state': b1.state_hash(agent.snapshot())})
        runtime.append({'age': age, 'receipt': receipt, 'publish_ns': publish_ns, 'reaction_ns': reaction_ns,
                        'reveal_ns': reveal_ns, 'update_ns': update_ns, 'event_wall_ns': update_ns - start_ns})
        pending = None
    end = {'agent': agent.snapshot(), 'table': table, 'environment': environment.snapshot(),
           'plan_sha256': digest(plan)}
    save_state(folder / 'end.npz', end)
    atomic_json(folder / 'events.json', results)
    atomic_json(folder / 'runtime.json', runtime)
    result = {'unit': [seed, mapping], 'arm': arm, 'mode': operation, 'events': len(results),
              'wall_seconds': time.perf_counter() - started, 'cpu_seconds': time.process_time() - cpu}
    if operation == 'primary':
        result['retention_before'] = probe_before
        result['retention_after'] = probe(agent, targets)
        result['delta_K_norm'] = float(np.linalg.norm(agent.core.K - initial['K']))
        result['max_state_norm'] = agent.core.max_state_norm
    else:
        reference_folder = root / 'lives' / name / arm
        expected = b1.read_json(reference_folder / 'events.json')
        exact(results, expected if operation == 'replay' else expected[12:])
        exact(end, load_state(reference_folder / 'end.npz'))
        result['exact'] = True
    atomic_json(folder / 'COMPLETE.json', result)
    return result


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('action', choices=('init', 'factorial', 'controls', 'replay', 'resume'))
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--history', type=Path)
    cli.add_argument('--b1', type=Path)
    cli.add_argument('--workers', type=int, default=2)
    args = cli.parse_args()
    output = args.output.resolve()
    if args.action == 'init':
        init(args.history.resolve(), args.b1.resolve(), output)
        return
    if args.workers not in (1, 2):
        cli.error('Use one or two workers')
    plan = b1.read_json(output / 'PLAN.json')
    b1.verify(output, plan)
    if args.action == 'factorial':
        arms, units = list(FACTORIAL) + list(TABLE), plan['units']
    elif args.action == 'controls':
        arms, units = list(CONTROLS), plan['units']
    elif args.action == 'resume':
        arms, units = list(ARMS), plan['units'][:1]
    else:
        arms, units = list(ARMS), plan['units']
    primary = args.action in ('factorial', 'controls')
    destination = output if primary else output / args.action
    jobs = [(plan, output, destination, *unit, arm, 'primary' if primary else args.action)
            for unit in units for arm in arms]
    path = output / f'STATUS_{args.action}.json'
    start = time.perf_counter()
    atomic_json(path, {'phase': 'RUNNING', 'completed': 0, 'total': len(jobs)})
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for n, info in enumerate(pool.map(run_life, jobs), 1):
                atomic_json(path, {'phase': 'RUNNING', 'completed': n, 'total': len(jobs)})
                print(n, len(jobs), info['unit'], info['arm'], flush=True)
    except Exception as exc:
        atomic_json(path, {'phase': 'STOP_INVALID', 'type': type(exc).__name__, 'message': str(exc)})
        raise
    atomic_json(path, {'phase': 'COMPLETED', 'total': len(jobs), 'wall_seconds': time.perf_counter() - start})


if __name__ == '__main__':
    main()
