"""Plot the paired development returns and fixed-query CARRY probes."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    root = parser.parse_args().output
    result = json.loads((root / 'ANALYSIS.json').read_text(encoding='utf-8'))
    stats = result['stats']
    groups = ['CARRY', 'LIVE64', 'Z0-64']
    colors = ['#2d5682', '#b65320']
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4), layout='constrained')

    for ax, key, multiplier, label in [
        (left, 'reward_total', 1, 'Суммарная награда за 24 события'),
        (right, 'retention_live_acc', 100, 'Точность CARRY на 6 фиксированных запросах, %'),
    ]:
        for i, credit in enumerate(['main', 'aligned']):
            estimates = [stats[f'{g}_{credit}'][key] for g in groups]
            means = np.array([x['mean'] for x in estimates]) * multiplier
            intervals = np.array([x['ci95'] for x in estimates]).T * multiplier
            ax.errorbar(np.arange(3) + (i - .5) * .24, means,
                        yerr=np.abs(intervals - means), fmt='o', capsize=4,
                        color=colors[i], label=['По прогнозу CARRY', 'По выданному действию'][i])
        ax.set_xticks(range(3), groups)
        ax.set_ylabel(label)
        ax.grid(axis='y', alpha=.25)
        ax.set_axisbelow(True)
    for control, label, color, style in [
        ('TABLE', 'Таблица', '#447567', '--'),
        ('Z0-64_K0', 'Начальная K', '#555555', ':'),
        ('Z0-64_WRONG', 'Чужая K', '#858585', '-.'),
    ]:
        left.axhline(stats[control]['reward_total']['mean'], label=label, color=color, linestyle=style)
    left.legend(loc='best', fontsize=8)
    right.set_ylim(0, 100)
    fig.suptitle('Stage B2A · 12 development-блоков · 95% bootstrap-интервалы')
    fig.savefig(root / 'B2A_RESULTS.png', dpi=175)


if __name__ == '__main__':
    main()
