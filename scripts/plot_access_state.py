"""Plot Stage A development accuracy and the cost of direct intervention."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(root):
    read = json.loads((root / 'READ_ANALYSIS.json').read_text())['results']
    selection = json.loads((root / 'FINAL_SELECTION.json').read_text())
    continuity = json.loads((root / 'CONTINUITY_ANALYSIS.json').read_text())['results']
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), layout='constrained',
                             gridspec_kw={'height_ratios': [1.7, 1]})
    for column, arm in enumerate(('B1', 'B3')):
        ax = axes[0, column]
        choices = list(read[arm])
        position = np.arange(len(choices))
        names = {'live': 'Живое Z', 'Z0': 'Полное Z0'}
        for offset, donor, label, color in [(-.25, 'own', 'Своя K', '#167b66'),
                                           (0., 'initial', 'Начальная K', '#697b92'),
                                           (.25, 'wrong', 'Чужая K', '#bf7548')]:
            values = [read[arm][name]['64']['donors'][donor]['accuracy']['mean'] for name in choices]
            ax.barh(position + offset, [100 * x if x is not None else np.nan for x in values],
                    height=.23, label=label, color=color)
        ax.set_yticks(position, [names.get(k, k) for k in choices])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel('Точность чтения на 64-м такте, %')
        ax.set_title(arm + ' · одинаковый бюджет чтения')
        ax.legend(frameon=False, fontsize=8, loc='lower right')
        ax.grid(axis='x', alpha=.2)
        ax = axes[1, column]
        group = continuity[arm]
        local = selection[arm]['selected']
        table = [(f'direct/{local}', f'Прямое {local}'), ('direct/Z0', 'Прямое Z0'),
                 ('sidecar/local', f'Sidecar {local}'), ('sidecar/Z0', 'Sidecar Z0')]
        for y, (key, label) in enumerate(table):
            value = group[key]['future_delta_historical']['accuracy']
            mean = 100 * value['mean']
            ci = value['ci95']
            error = [[max(0., mean - 100 * ci[0])], [max(0., 100 * ci[1] - mean)]] if ci else None
            ax.errorbar(mean, y, xerr=error, fmt='o', color='#167b66' if key.startswith('sidecar/') else '#26384b',
                        capsize=4)
        ax.set_yticks(np.arange(len(table)), [label for _, label in table])
        ax.invert_yaxis()
        ax.axvline(0., color='#697b92')
        ax.axvline(-5., color='#bf7548', linestyle='--', linewidth=1)
        ax.set_title('Основная ветвь после чтения')
        ax.set_xlabel('Изменение точности событий 50–61, п.п.')
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('LIFE-ACCESS-STATE-001 · Stage A')
    fig.savefig(root / 'ACCESS_STATE_STAGE_A.png', dpi=155)
    plt.close(fig)


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('output', type=Path)
    plot(cli.parse_args().output)
