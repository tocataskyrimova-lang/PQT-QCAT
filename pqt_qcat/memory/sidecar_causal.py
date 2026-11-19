"""Causal credit and update-scale interventions on the frozen B2A development panel."""
from . import functional_sidecar_b2a as b2  # cap numerical threads before NumPy

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .access_state import exact, load_agent
from .checkpoint_io import load_state, save_state
from .codex_memory_location import atomic_json, digest, metrics, source_hashes
from .functional_sidecar import ActionJournal, checkpoint_path, read_json, state_hash
from .qnuf import ObservedEvidence, array_hash


BASE = '16c75c1f43ffc46b3f3277f8f45c8b5d047bfd74'
B2A_ZIP = '8366272bc8459f0bf9a4f7c9b934ebabdb88e3aafc7ec948ea83f1cbd3aa09c2'
SOURCES = {'MM': ('main', 'main'), 'SM': ('sidecar', 'main'),
           'MS': ('main', 'sidecar'), 'SS': ('sidecar', 'sidecar')}
SCALES = {'raw': None, 'low': .05, 'high': .50}
STAGES = {'raw': ('retry', 'raw', 'own'), 'low': ('retry', 'low', 'own'),
          'high': ('retry', 'high', 'own'), 'advance': ('advance', 'raw', 'own'),
          'initial': ('retry', 'raw', 'initial'), 'wrong': ('retry', 'raw', 'wrong')}
RESUME_AGE = 12


class AdvanceEnvironment(b2.RetryEnvironment):
    """Preserve the reward rule, with distinct nonzero shifts on success and error."""

    @staticmethod
    def transition(query, action, target, offset, streak):
        outcome = b2.RetryEnvironment.transition(query, action, target, offset, streak)
        if not outcome['correct']:
            outcome['next_query'] = (query - 1 + (offset % 5 + 1)) % 6 + 1
        return outcome


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def init(output, reference, archive, history, b1):
    old = read_json(reference / 'DELIVERY.json')
    assert old['delivery_commit'] == BASE and old['scientific'] == 'ACTION_DEPENDENT_DEPLOYED_SIGNAL'
    repo = b2.b1.bridge.ROOT
    subprocess.run(['git', 'merge-base', '--is-ancestor', BASE, 'HEAD'], cwd=repo, check=True)
    assert file_sha(archive) == B2A_ZIP
    manifest = reference / 'ARTIFACT_MANIFEST.sha256'
    entries = [line.split('  ', 1) for line in manifest.read_text(encoding='utf-8').splitlines()]
    for checksum, name in entries:
        if file_sha(reference / name) != checksum:
            raise ValueError(f'B2A artifact differs from its manifest: {name}')
    old_plan = read_json(reference / 'PLAN.json')
    assert len(old_plan['units']) == 12 and old_plan['steps'] == 24
    plan = {**old_plan, 'inputs': {}, 'b2a_reference': str(reference), 'base': BASE,
            'b2a_archive': str(archive), 'b2a_archive_sha256': B2A_ZIP, 'b2a_manifest_sha256': file_sha(manifest),
            'b2a_artifacts_verified': len(entries), 'credit_sources': SOURCES, 'scales': SCALES,
            'stages': STAGES, 'counterfactual_anchors': [['MM', 0], ['MM', 12], ['SS', 12]],
            'resume_age': RESUME_AGE, 'resume_unit': old_plan['units'][0],
            'accuracy_threshold': .05, 'return_threshold': 2., 'workers': 6,
            'scope': 'previous B2A development checkpoints only; one mapping per life',
            'norm_rule': 'normalize the clipped B3 candidate to the common nonzero norm; keep an exact zero at zero',
            'wrong_donor': 'own raw retry SS on the next mapping (mod 4), same seed and elapsed age',
            'advance_rule': 'on failure shift by offset modulo 5 plus 1; all other environment rules unchanged'}
    for key, local in [('historical_source', history), ('b1_source', b1)]:
        old_path = old_plan[key]
        plan[key] = str(local.resolve()) if local else old_path
        plan['inputs'][plan[key]] = old_plan['inputs'][old_path]
    b2.b1.verify_artifacts(plan)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'PLAN.json', plan)
    atomic_json(output / 'SOURCE.json', {
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        'files': source_hashes(), 'plan_sha256': digest(plan), 'python': sys.version,
        'dependencies': {p: importlib.metadata.version(p) for p in ('numpy', 'torch', 'scipy', 'scikit-learn', 'pytest')},
    })


def credit_values(packet, arm):
    logits_source, feature_source = SOURCES[arm]
    main, side = packet['main'], packet['external']
    logits = np.asarray(main['logits'] if logits_source == 'main' else packet['logits'])
    phi = (np.tanh(np.asarray(main['Z_post'])) - np.tanh(np.asarray(main['Z_noinput']))
           if feature_source == 'main' else np.asarray(side['eligibility']))
    return logits, phi


def apply_credit(agent, packet, evidence, arm, norm):
    """Apply the existing QNUF bookkeeping, substituting only chosen credit terms."""
    prediction = agent.pending['prediction']
    exact(prediction, packet['main'])
    logits, phi = credit_values(packet, arm)
    if SOURCES[arm][0] == 'sidecar':
        p = np.exp(logits - logits.max())
        p /= p.sum()
        agent.pending['prediction'] = {**prediction, 'logits': logits.tolist(), 'probabilities': p.tolist(),
                                       'prediction': packet['action']}
    if SOURCES[arm][1] == 'sidecar':
        agent.pending['event'] = phi.copy()
    exact(agent.pending['event'].tolist(), phi.tolist())
    before = agent.core.K.copy()
    update = agent.finish_query(evidence)
    raw_applied = update['applied_delta_norm']
    if norm is not None and update['candidate_norm_before_cap'] != 0.:
        clipped = np.clip(agent.eta_k * np.outer(agent.core.W_readout.T @ update['error_vector'], phi),
                          -agent.delta_core_max, agent.delta_core_max)
        clipped *= norm / update['candidate_norm_before_cap']
        agent.core.K[:] = before + clipped
        changed = array_hash(agent.core.K)
        update.update(applied_delta_norm=float(np.linalg.norm(agent.core.K - before)),
                      proposed_delta_norm=float(np.linalg.norm(clipped)), K_proposed_hash=changed, K_post_hash=changed)
        if not np.isclose(update['applied_delta_norm'], norm, atol=2e-14, rtol=0.):
            raise AssertionError('Applied update differs from the requested Frobenius norm')
        agent.last_decision = copy.deepcopy(update)
    return update, raw_applied


def unit_name(seed, mapping):
    return f'{seed}_{mapping}'


def lived(root, seed, mapping, stage, arm):
    return root / 'lives' / unit_name(seed, mapping) / stage / arm


def donor_matrices(root, plan, seed, mapping, kind):
    if kind == 'wrong':
        wrong = (mapping + 1) % 4
        return load_state(lived(root, seed, wrong, 'raw', 'SS') / 'donors.npz')['K']
    if kind == 'initial':
        k = load_state(checkpoint_path(plan, 'B0', seed, mapping, 0))['agent']['core_snapshot']['K']
        return np.broadcast_to(k, (plan['steps'], *k.shape))
    return [None] * plan['steps']


def run_life(job):
    root, plan, seed, mapping, stage, arm, resume = job
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    b2.validate_unit(plan, seed, mapping)
    environment_kind, scale, donor_kind = STAGES[stage]
    norm = SCALES[scale]
    base = lived(root, seed, mapping, stage, arm)
    folder = lived(root / 'resume', seed, mapping, stage, arm) if resume else base
    folder.mkdir(parents=True, exist_ok=False)
    agent = load_agent(plan, 'B3', seed, mapping, plan['cursor_start'])
    start_k = agent.core.K.copy()
    targets = plan['parent']['stage1_mappings'][mapping]
    stream = plan['streams'][unit_name(seed, mapping)]
    create_env = b2.RetryEnvironment if environment_kind == 'retry' else AdvanceEnvironment
    environment = create_env(targets, stream['offsets'], stream['start_query'])
    donors = donor_matrices(root, plan, seed, mapping, donor_kind)
    journal = ActionJournal(folder / 'pre_feedback.jsonl')
    pending = None
    if resume:
        cut = load_state(base / 'pending_12.npz')
        assert cut['plan_sha256'] == digest(plan) and cut['condition'] == [stage, arm]
        agent.restore(cut['agent'])
        environment = create_env(targets, stream['offsets'], **cut['environment'])
        pending = ActionJournal(base / 'pre_feedback.jsonl').read_verified(cut['receipt'])
        assert state_hash(agent.snapshot()) == pending['pending_sha256']
        receipt = journal.publish(pending)
    rows, timing, own_k = [], [], []
    while environment.cursor < plan['steps']:
        tick = time.perf_counter_ns()
        age = environment.cursor
        observation = environment.snapshot()
        if pending is None:
            if stage == 'raw' and arm == 'SS' and not resume:
                own_k.append(agent.core.K.copy())
            pending = b2.forecast(agent, environment.observe(), 'Z0-64', 'aligned', donors[age], {})
            pending.update(cursor=age, observation=observation)
            receipt = journal.publish(pending)
        published = time.perf_counter_ns()
        if not resume and (age == RESUME_AGE or (age == 0 and stage == 'raw' and arm == 'MM')):
            save_state(folder / f'pending_{age:02}.npz', {'agent': agent.snapshot(), 'receipt': receipt,
                       'environment': observation, 'condition': [stage, arm], 'plan_sha256': digest(plan)})
        reaction = environment.react(journal, receipt)
        reacted = time.perf_counter_ns()
        evidence = environment.reveal()
        revealed = time.perf_counter_ns()
        update, raw_applied = apply_credit(agent, pending, evidence, arm, norm)
        updated = time.perf_counter_ns()
        rows.append({'packet': pending, 'reaction': reaction, 'evidence_target': evidence.target,
                     'credit': list(SOURCES[arm]), 'update': update, 'raw_applied': raw_applied,
                     'norm_target': norm, 'end_state': state_hash(agent.snapshot())})
        timing.append({'receipt': receipt, 'publish_ns': published, 'reaction_ns': reacted,
                       'reveal_ns': revealed, 'update_ns': updated, 'event_wall_ns': updated - tick})
        pending = None
    end = {'agent': agent.snapshot(), 'environment': environment.snapshot(),
           'condition': [stage, arm], 'plan_sha256': digest(plan)}
    save_state(folder / 'end.npz', end)
    atomic_json(folder / 'events.json', rows)
    atomic_json(folder / 'runtime.json', timing)
    result = {'seed': seed, 'mapping': mapping, 'stage': stage, 'arm': arm, 'events': len(rows)}
    if resume:
        exact(rows, read_json(base / 'events.json')[RESUME_AGE:])
        assert state_hash(end) == state_hash(load_state(base / 'end.npz'))
        assert journal.path.read_bytes().splitlines(keepends=True) == (
            (base / 'pre_feedback.jsonl').read_bytes().splitlines(keepends=True)[RESUME_AGE:])
        result['exact'] = True
    else:
        if own_k:
            save_state(folder / 'donors.npz', {'K': np.stack(own_k)})
        result['retention'] = b2.probe(agent, targets)
        result['delta_K_norm'] = float(np.linalg.norm(agent.core.K - start_k))
    result.update(wall_seconds=time.perf_counter() - start_wall, cpu_seconds=time.process_time() - start_cpu)
    atomic_json(folder / 'COMPLETE.json', result)
    return result


def one_step(job):
    root, plan, seed, mapping = job
    b2.validate_unit(plan, seed, mapping)
    output = root / 'one_step' / unit_name(seed, mapping)
    output.mkdir(parents=True, exist_ok=False)
    agent = load_agent(plan, 'B3', seed, mapping, plan['cursor_start'])
    targets = plan['parent']['stage1_mappings'][mapping]
    stream = plan['streams'][unit_name(seed, mapping)]
    donors = {k: donor_matrices(root, plan, seed, mapping, k) for k in ('initial', 'wrong')}
    results = []
    for anchor, age in plan['counterfactual_anchors']:
        folder = lived(root, seed, mapping, 'raw', anchor)
        cut = load_state(folder / f'pending_{age:02}.npz')
        packet = ActionJournal(folder / 'pre_feedback.jsonl').read_verified(cut['receipt'])
        assert cut['environment']['cursor'] == age and cut['plan_sha256'] == digest(plan)
        agent.restore(cut['agent'])
        assert state_hash(agent.snapshot()) == packet['pending_sha256']
        q = packet['query']
        environment = b2.RetryEnvironment(targets, stream['offsets'], **cut['environment'])
        journal = ActionJournal(output / f'{anchor}_{age:02}_pre_feedback.jsonl')
        reaction = environment.react(journal, journal.publish(packet))
        evidence = environment.reveal()
        queries = list(dict.fromkeys((q, reaction['next_query'])))
        baseline = agent.core.K.copy()
        pre = state_hash(agent.snapshot())
        options = [('none', 'MM', None, None)]
        options.extend((scale, arm, norm, None) for scale, norm in SCALES.items() for arm in SOURCES)
        options.extend(('read-' + name, 'SS', None, values[age]) for name, values in donors.items())
        predictions, vectors = [], []
        for scale, arm, norm, donor in options:
            branch = copy.deepcopy(agent)
            if scale in SCALES:
                update, _ = apply_credit(branch, packet, evidence, arm, norm)
            else:
                update = None
            assert array_hash(branch.core.Z) == packet['main']['Z_post_hash']
            vector = branch.core.K - baseline
            vectors.append(vector)
            logits = [b2.read_branch(copy.deepcopy(branch.core), branch.Z0, query, 'Z0-64', donor,
                                    False, branch.read_gate)['logits'] for query in queries]
            predictions.append({'scale': scale, 'arm': arm, 'applied_delta_norm': float(np.linalg.norm(vector)),
                                'K_hash': array_hash(branch.core.K), 'donor_hash': array_hash(donor) if donor is not None else None,
                                'update': update, 'logits': logits, 'substeps': 128 * len(queries)})
        assert state_hash(agent.snapshot()) == pre
        save_state(output / f'{anchor}_{age:02}_deltas.npz', {'delta': np.stack(vectors), 'K': baseline})
        results.append({'anchor': anchor, 'age': age, 'pending_sha256': pre, 'issued_action': packet['action'],
                        'queries': queries, 'targets_for_scoring': [targets[i-1] for i in queries],
                        'reaction': reaction, 'predictions': predictions})
    result = {'unit': [seed, mapping], 'anchors': results}
    atomic_json(output / 'RESULTS.json', result)
    return {'unit': [seed, mapping], 'anchors': len(results)}


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('phase', choices=('init', 'primary', 'controls', 'resume', 'one-step'))
    cli.add_argument('--output', type=Path, required=True)
    for arg in ('reference', 'archive', 'history', 'b1'):
        cli.add_argument('--' + arg, type=Path)
    cli.add_argument('--workers', type=int, default=6)
    args = cli.parse_args()
    root = args.output.resolve()
    if args.phase == 'init':
        init(root, args.reference.resolve(), args.archive.resolve(), args.history, args.b1)
        return
    plan = read_json(root / 'PLAN.json')
    b2.b1.verify(root, plan)
    if not 1 <= args.workers <= plan['workers']:
        cli.error('Worker count outside frozen bounds')
    if args.phase == 'one-step':
        function, jobs = one_step, [(root, plan, *unit) for unit in plan['units']]
    else:
        function = run_life
        stages = tuple(STAGES) if args.phase == 'resume' else (
            ('raw', 'low', 'high', 'advance') if args.phase == 'primary' else ('initial', 'wrong'))
        units = [plan['resume_unit']] if args.phase == 'resume' else plan['units']
        jobs = [(root, plan, *unit, stage, arm, args.phase == 'resume') for unit in units for stage in stages
                for arm in (('SS',) if stage in ('initial', 'wrong') else SOURCES)]
    status = root / f'STATUS_{args.phase}.json'
    start = time.perf_counter()
    atomic_json(status, {'phase': 'RUNNING', 'completed': 0, 'total': len(jobs)})
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for completed, future in enumerate(as_completed([pool.submit(function, job) for job in jobs]), 1):
                result = future.result()
                atomic_json(status, {'phase': 'RUNNING', 'completed': completed, 'total': len(jobs)})
                print(completed, len(jobs), result, flush=True)
    except Exception as exc:
        atomic_json(status, {'phase': 'STOP_INVALID', 'error': type(exc).__name__, 'message': str(exc)})
        raise
    atomic_json(status, {'phase': 'COMPLETED', 'total': len(jobs), 'wall_seconds': time.perf_counter() - start})


if __name__ == '__main__':
    main()
