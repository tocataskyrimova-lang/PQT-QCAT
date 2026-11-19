"""Plot the measured readout curves and equal-norm contrasts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(output):
    data = json.loads((output / 'ANALYSIS.json').read_text())
    horizons = (8, 16, 32, 64)
    display = {'initial': 'Начальная K', 'B1': 'Обученная K (свой mapping)',
               'B3': 'Обученная K (свой mapping)'}
    colors = {'initial': '#475569', 'own': '#146557', 'wrong': '#b2612c'}
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), layout='constrained')
    axes = axes.ravel()
    for arm, ax in zip(('B1', 'B3'), axes):
        for donor, label, color in [
            ('initial', display['initial'], colors['initial']),
            (arm, display[arm], colors['own']),
            (f'{arm}_wrong', 'Обученная K (другой mapping)', colors['wrong']),
        ]:
            curve = data['curves'][f'{arm}_live/{donor}']
            values = [100 * curve[str(h)]['accuracy']['mean'] for h in horizons]
            ax.plot(horizons, values, marker='o', color=color, label=label)
        ax.set_title(f'Чтение из живого Z · {arm}')
        ax.set_xlabel('Такт считывания')
        ax.set_ylabel('Точность, %')
        ax.set_xticks(horizons)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', color='#e2e8f0')
    axes[0].legend(fontsize=8, frameon=False, loc='upper left')
    ax = axes[2]
    for donor, label, color, style in [
        ('initial', 'Начальная K', '#475569', '-'),
        ('B1', 'B1, свой mapping', '#146557', '-'),
        ('B3', 'B3, свой mapping', '#2563eb', '-'),
        ('B1_wrong', 'B1, другой mapping', '#146557', '--'),
        ('B3_wrong', 'B3, другой mapping', '#2563eb', '--'),
    ]:
        curve = data['curves'][f'Z0/{donor}']
        ax.plot(horizons, [100 * curve[str(h)]['accuracy']['mean'] for h in horizons],
                marker='o', color=color, label=label, linestyle=style)
    ax.set_title('Чтение из диагностического Z0')
    ax.set_xlabel('Такт считывания')
    ax.set_ylabel('Точность, %')
    ax.set_xticks(horizons)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax.grid(axis='y', color='#e2e8f0')
    ax = axes[3]
    xs = np.arange(4)
    for index, name in enumerate(('target_0.08', 'target_0.32', 'full_0.08', 'full_0.32')):
        row = data['normalized'][name]['online']['accuracy']['event_minus_legacy']
        mean = 100 * row['mean']
        interval = row['ci95']
        yerr = [[mean - 100 * interval[0]], [100 * interval[1] - mean]] if interval is not None else None
        ax.errorbar(index, mean, yerr=yerr, color=colors['own'], marker='o', capsize=4)
    ax.set_title('Пластичность при равной норме')
    ax.set_ylabel('Event − legacy, online точность (п.п.)')
    ax.set_xticks(xs, ['target\n0,08', 'target\n0,32', 'full\n0,08', 'full\n0,32'])
    ax.axhline(0, color='#475569', linewidth=1)
    ax.grid(axis='y', color='#e2e8f0')
    figure.suptitle('LIFE-QNUF-001 / ACCESS-BRIDGE', fontsize=14)
    figure.savefig(output / 'ACCESS_BRIDGE_RESULTS.png', dpi=160)
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    plot(parser.parse_args().output)
