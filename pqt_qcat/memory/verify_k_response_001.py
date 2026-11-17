"""Independent artifact consistency checks and plot for K-RESPONSE-001."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from .exp_sleep_need_001 import setup_fresh_core, query_core_prediction


def verify(run: Path, baseline: Path | None = None):
    completed = json.loads((run / 'COMPLETE.json').read_text(encoding='utf-8'))
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    sources = json.loads((run / 'source_hashes.json').read_text(encoding='utf-8'))
    for name, expected in sources.items():
        assert hashlib.sha256((run / 'source' / name).read_bytes()).hexdigest() == expected
    queries = [json.loads(line) for line in (run / 'queries.jsonl').read_text(encoding='utf-8').splitlines()]
    wake = [json.loads(line) for line in (run / 'wake.jsonl').read_text(encoding='utf-8').splitlines()]
    curves = json.loads((run / 'curves.json').read_text(encoding='utf-8'))
    histories = json.loads((run / 'histories.json').read_text(encoding='utf-8'))
    assert len(wake) == 24 * 195 and len(queries) == 24 * (5 + 2 * 3) * 6
    assert len({(q['seed'], q['task_id'], q['kind'], q['value'], q['token']) for q in queries}) == len(queries)
    for row in curves:
        selected = [q for q in queries if q['kind'] == 'lambda' and q['value'] == row['lam']
                    and q['seed'] == row['seed'] and q['task_id'] == row['task_id']
                    and ((q['token'] <= 3) == (row['group'] == 'old'))]
        assert len(selected) == 3
        accuracies, margins, nlls = [], [], []
        for q in selected:
            logits = np.array(q['logits'])
            target = q['target'] - 1
            accuracies.append(int(np.argmax(logits) == target))
            margins.append(logits[target] - max(np.delete(logits, target)))
            nlls.append(float(np.logaddexp.reduce(logits) - logits[target]))
        for key, values in [('accuracy', accuracies), ('margin', margins), ('nll', nlls)]:
            assert np.isclose(row[key], np.mean(values), rtol=0, atol=1e-14)
    exact_queries = 0
    for seed in config['seeds']:
        for task_id in range(6):
            with np.load(run / f'checkpoint_s{seed}_m{task_id}.npz', allow_pickle=False) as checkpoint:
                core = setup_fresh_core(seed)
                assert np.array_equal(core.B, checkpoint['B'])
                assert np.array_equal(core.W_readout, checkpoint['W_readout'])
                for endpoint, lam in [('before', 0.), ('after', 1.)]:
                    core.K = checkpoint['K_' + endpoint].copy()
                    for token in range(1, 7):
                        q = next(q for q in queries if q['seed'] == seed and q['task_id'] == task_id
                                 and q['kind'] == 'lambda' and q['value'] == lam and q['token'] == token)
                        reconstructed = query_core_prediction(core, token)[1]
                        assert np.allclose(reconstructed, q['logits'], rtol=0, atol=1e-13)
                        exact_queries += int(np.array_equal(reconstructed, q['logits']))
    matched_baseline = 0
    if baseline:
        probes = [json.loads(line) for line in baseline.read_text(encoding='utf-8').splitlines()]
        for h in histories:
            for endpoint, stage in [('before', 'day_O_ep_20'), ('after', 'day_N_ep_20')]:
                p = next(p for p in probes if p['seed'] == h['seed'] and p['task_id'] == h['task_id']
                         and p['history'] == 'blocked' and p['stage'] == stage)
                assert h['K_' + endpoint + '_sha256'] == p['K_sha256']
                matched_baseline += 1
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    for axis, metric, title in zip(axes, ('accuracy', 'margin', 'nll'), ('Accuracy', 'Signed target margin', 'NLL')):
        for group, color in [('old', '#2458a6'), ('new', '#d67823')]:
            seed_curves = []
            for seed in config['seeds']:
                y = [np.mean([r[metric] for r in curves if r['seed'] == seed and r['group'] == group
                              and r['lam'] == lam]) for lam in config['lambdas']]
                seed_curves.append(y)
                axis.plot(config['lambdas'], y, color=color, alpha=.24, linewidth=1)
            axis.plot(config['lambdas'], np.mean(seed_curves, axis=0), 'o-', color=color, label=group)
        axis.set(title=title, xlabel='lambda: K before -> K after new learning')
        axis.grid(alpha=.16)
    axes[0].legend()
    fig.suptitle('K-RESPONSE-001 | 4 calibration seeds, 6 maps per seed | no sleep')
    fig.savefig(run / 'response_curves.png', dpi=160)
    plt.close(fig)
    checks = dict(status='PASS', source_hashes=len(sources), raw_queries=len(queries),
                  wake_events=len(wake), endpoint_queries_reconstructed=288,
                  endpoint_queries_bitwise_exact=exact_queries, original_pilot_K_hash_matches=matched_baseline)
    (run / 'artifact_verification.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print(json.dumps(checks, indent=2))
    return completed, checks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--baseline-probes', type=Path)
    args = parser.parse_args()
    verify(args.run, args.baseline_probes)
