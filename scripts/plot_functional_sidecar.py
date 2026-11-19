"""Plot the measured sequence and development accuracy of all six actions."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(output):
    scores = json.loads((output / 'results' / 'stage_b1' / 'contrasts.json').read_text(encoding='utf-8'))
    arms = list(scores['arms'])
    colors = {'M-LIVE8': '#344663', 'S-LIVE64': '#697b92', 'S-Z0-64': '#087e69',
              'S-Z0-8': '#379cc3', 'S-Z0-64-K0': '#c28a56', 'S-Z0-64-WRONG': '#a35b3c'}
    fig, (seq, overall) = plt.subplots(1, 2, figsize=(12.6, 4.8), layout='constrained',
                                      gridspec_kw={'width_ratios': [1.75, 1]})
    for arm in arms:
        data = scores['per_event_accuracy'][arm]
        seq.plot([int(x) for x in data], [100 * x for x in data.values()],
                 color=colors[arm], linewidth=1.9 if arm == 'S-Z0-64' else 1.2,
                 marker='o', markersize=3, label=arm)
    for boundary in (56.5, 64.5):
        seq.axvline(boundary, color='#d0d7e2', linestyle='--', linewidth=1)
    seq.set(title='Последовательные внешние ответы', xlabel='Событие',
            ylabel='Точность по 18 блокам, %', xticks=range(49, 73, 3), ylim=(0, 100))
    seq.legend(frameon=False, ncols=2, fontsize=8, loc='lower left')
    seq.grid(axis='y', alpha=.22)
    indices = np.arange(len(arms))
    for index, arm in enumerate(arms):
        row = scores['arms'][arm]['deployed_accuracy']
        mean = 100 * row['mean']
        bands = [100 * p for p in row['ci95']]
        overall.errorbar(mean, index, xerr=[[max(0, mean - bands[0])], [max(0, bands[1] - mean)]],
                         fmt='o', color=colors[arm], capsize=3)
    overall.set(title='Результат по событиям 49–72', xlabel='Точность, %', xlim=(0, 100))
    overall.set_yticks(indices, arms)
    overall.invert_yaxis()
    overall.grid(axis='x', alpha=.22)
    fig.suptitle('LIFE-ACCESS-STATE-001 · Functional Sidecar B1')
    fig.savefig(output / 'FUNCTIONAL_SIDECAR_B1.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    plot(parser.parse_args().output)
