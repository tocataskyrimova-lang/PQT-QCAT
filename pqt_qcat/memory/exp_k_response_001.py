"""Bounded K-only finite-response experiment; see K_RESPONSE_001_PROTOCOL.md."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from .exp_sleep_need_001 import (
    compute_student_t_ci, generate_task_panel, query_core_prediction,
    safe_json_dumps, setup_fresh_core, wake_train_step,
)
from .m3_diagnostics import _get_quiescent_psi

SEEDS = (11, 12, 13, 42)
LAMBDAS = (0., .25, .5, .75, 1.)
EPSILONS = (.01, .001, .0001)
ROOT = Path(__file__).resolve().parents[2]


def fingerprint(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read_logits(core):
    return np.stack([query_core_prediction(core, token)[1] for token in range(1, 7)])


def score(logits, mapping):
    if not np.isfinite(logits).all():
        raise FloatingPointError('Non-finite logits')
    targets = np.array([mapping[t] - 1 for t in range(1, 7)])
    maximum = np.max(logits, axis=1)
    logsumexp = maximum + np.log(np.exp(logits - maximum[:, None]).sum(axis=1))
    nll = logsumexp - logits[np.arange(6), targets]
    margins = np.array([row[y] - max(np.delete(row, y)) for row, y in zip(logits, targets)])
    correct = np.argmax(logits, axis=1) == targets
    return correct, margins, nll


def run(output: Path):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    maps = generate_task_panel()[:6]
    protocol = ROOT / 'K_RESPONSE_001_PROTOCOL.md'
    (output / protocol.name).write_bytes(protocol.read_bytes())
    config = dict(seeds=SEEDS, maps=maps, lambdas=LAMBDAS, epsilons=EPSILONS,
                  prefix_epochs=25, old_epochs=20, new_epochs=20, query_ticks=8,
                  tail=False, sleep=False, purpose='exploratory K-only response')
    (output / 'config.json').write_text(safe_json_dumps(config), encoding='utf-8')
    source_paths = sorted((ROOT / 'pqt_qcat').rglob('*.py')) + [protocol]
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    (output / 'source_hashes.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    source_dir = output / 'source'
    for p in source_paths:
        dest = source_dir / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
    git_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    patch = subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=ROOT)
    (output / 'working_tree.patch').write_bytes(patch)
    environment = dict(python=platform.python_version(), numpy=np.__version__,
                       platform=platform.platform(), base_commit=git_head)
    (output / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    histories, curves, derivatives = [], [], []
    with (output / 'wake.jsonl').open('w', encoding='utf-8') as wake_log, \
         (output / 'queries.jsonl').open('w', encoding='utf-8') as query_log:
        for seed in SEEDS:
            for task_id, mapping in enumerate(maps):
                core = setup_fresh_core(seed)
                b, w = core.B.copy(), core.W_readout.copy()
                checkpoints = {'K_initial': core.K.copy(), 'Z_initial': core.Z.copy()}
                rng = np.random.default_rng(seed * 10000 + task_id * 100 + 42)
                schedules = ([rng.permutation([1, 2, 3]) for _ in range(25)],
                             [rng.permutation([1, 2, 3]) for _ in range(20)],
                             [rng.permutation([4, 5, 6]) for _ in range(20)])
                event_id = 0
                for group, epochs, checkpoint in zip(
                    ('prefix_old', 'day_old', 'day_new'), schedules, ('prefix', 'before', 'after')
                ):
                    for epoch, tokens in enumerate(epochs, 1):
                        for tok in tokens:
                            event_id += 1
                            pred, logits, delta = wake_train_step(core, int(tok), mapping[int(tok)], _get_quiescent_psi())
                            if not np.isfinite(core.K).all() or not np.isfinite(logits).all():
                                raise FloatingPointError('Non-finite training state')
                            wake_log.write(safe_json_dumps(dict(seed=seed, task_id=task_id,
                                event=event_id, group=group, epoch=epoch, token=int(tok),
                                target=mapping[int(tok)], prediction=pred, logits=logits,
                                delta_K_sha256=fingerprint(delta), K_sha256=fingerprint(core.K))) + '\n')
                    checkpoints['K_' + checkpoint] = core.K.copy()
                    checkpoints['Z_' + checkpoint] = core.Z.copy()
                assert event_id == 195
                assert np.array_equal(core.B, b) and np.array_equal(core.W_readout, w)
                before, after = checkpoints['K_before'], checkpoints['K_after']
                direction = after - before
                endpoint_error = float(np.max(np.abs(before + direction - after)))
                assert endpoint_error < 1e-13
                original = read_logits(core)
                core.K = before.copy()
                start_logits = read_logits(core)
                core.K = after.copy()
                assert np.array_equal(original, read_logits(core)), 'Rollback logits differ'
                assert np.array_equal(core.K, after)
                metadata = dict(seed=seed, task_id=task_id, mapping=mapping,
                    config=asdict(core.config), cell_params=asdict(core.cell_params),
                    vortex_params=asdict(core.vortex_params), rng_state=core.rng.bit_generator.state)
                np.savez_compressed(output / f'checkpoint_s{seed}_m{task_id}.npz',
                    **checkpoints, B=b, W_readout=w, metadata_json=np.array(safe_json_dumps(metadata)))
                for lam in LAMBDAS:
                    core.K = before + lam * direction
                    logits = read_logits(core)
                    if lam == 0:
                        assert np.array_equal(logits, start_logits)
                    if lam == 1:
                        assert np.allclose(logits, original, atol=1e-13, rtol=0)
                    correct, margins, nll = score(logits, mapping)
                    for i in range(6):
                        query_log.write(safe_json_dumps(dict(seed=seed, task_id=task_id,
                            kind='lambda', value=lam, token=i+1, target=mapping[i+1],
                            logits=logits[i], correct=bool(correct[i]),
                            margin=margins[i], nll=nll[i])) + '\n')
                    for group, idx in [('old', slice(0, 3)), ('new', slice(3, 6))]:
                        curves.append(dict(seed=seed, task_id=task_id, lam=lam, group=group,
                            accuracy=float(correct[idx].mean()), margin=float(margins[idx].mean()),
                            nll=float(nll[idx].mean())))
                estimates = []
                for eps in EPSILONS:
                    pair = []
                    for sign in (-1, 1):
                        core.K = before + sign * eps * direction
                        logits = read_logits(core)
                        pair.append(logits)
                        for i in range(6):
                            query_log.write(safe_json_dumps(dict(seed=seed, task_id=task_id,
                                kind='central_difference', value=sign*eps, token=i+1, logits=logits[i])) + '\n')
                    estimates.append((pair[1] - pair[0]) / (2 * eps))
                convergence = [float(np.linalg.norm(estimates[i] - estimates[i+1])) for i in (0, 1)]
                response = original - start_logits
                derivatives.append(dict(seed=seed, task_id=task_id,
                    convergence_absolute=convergence, derivative_norm=float(np.linalg.norm(estimates[-1])),
                    response_norm=float(np.linalg.norm(response)),
                    full_step_relative_linearization_error=float(np.linalg.norm(response-estimates[-1]) / max(np.linalg.norm(response), 1e-15))))
                histories.append(dict(seed=seed, task_id=task_id, events=event_id,
                    K_before_sha256=fingerprint(before), K_after_sha256=fingerprint(after),
                    endpoint_reconstruction_max_abs=endpoint_error, rollback_exact=True,
                    delta_K_norm=float(np.linalg.norm(direction))))
                print(f'seed={seed} map={task_id} complete', flush=True)
    (output / 'curves.json').write_text(safe_json_dumps(curves), encoding='utf-8')
    (output / 'derivatives.json').write_text(safe_json_dumps(derivatives), encoding='utf-8')
    (output / 'histories.json').write_text(safe_json_dumps(histories), encoding='utf-8')
    summary = {}
    for group in ('old', 'new'):
        summary[group] = {}
        for metric in ('accuracy', 'margin', 'nll'):
            changes, first, last = [], [], []
            for seed in SEEDS:
                values = [float(np.mean([r[metric] for r in curves
                    if r['seed'] == seed and r['group'] == group and r['lam'] == lam])) for lam in (0., 1.)]
                first.append(values[0]); last.append(values[1]); changes.append(values[1] - values[0])
            mean, sem, ci = compute_student_t_ci(changes)
            summary[group][metric] = dict(before=float(np.mean(first)), after=float(np.mean(last)),
                change=mean, seed_changes=changes, descriptive_ci95=ci)
    result = dict(status='COMPLETE', scope='exploratory, reused calibration seeds; no sleep',
        histories=len(histories), seeds=SEEDS, summary=summary,
        max_derivative_convergence_error=max(r['convergence_absolute'][-1] for r in derivatives),
        median_full_step_linearization_error=float(np.median([r['full_step_relative_linearization_error'] for r in derivatives])),
        wall_seconds=time.perf_counter() - started)
    (output / 'COMPLETE.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
