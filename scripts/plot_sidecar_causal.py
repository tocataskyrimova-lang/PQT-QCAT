"""Plot the four credit-source combinations under each frozen intervention."""
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
    data = json.loads((root / 'ANALYSIS.json').read_text(encoding='utf-8'))
    stats = data['standalone_arm_stats']
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), sharex=True, sharey=True, layout='constrained')
    for ax, (stage, name) in zip(axes.flat, [('raw', 'Исходный масштаб B3'), ('advance', 'Без повтора после ошибки'),
                                           ('low', 'Равная норма ΔK = 0,05'), ('high', 'Равная норма ΔK = 0,50')]):
        for arms, color, label in [(('MM', 'SM'), '#2d5682', 'Eligibility MAIN'),
                                    (('MS', 'SS'), '#b65320', 'Eligibility SIDECAR')]:
            rows = [stats[f'{stage}/{arm}']['accuracy'] for arm in arms]
            means = 100 * np.array([r['mean'] for r in rows])
            ci = 100 * np.array([r['ci95'] for r in rows]).T
            shift = -.035 if arms[0] == 'MM' else .035
            ax.errorbar(np.arange(2) + shift, means, yerr=np.abs(ci - means), fmt='o-', color=color,
                        capsize=4, lw=1.6, label=label)
        ax.set_title(name)
        ax.set_xticks([0, 1], ['Логиты MAIN', 'Логиты SIDECAR'])
        ax.set_ylim(-3, 103)
        ax.grid(axis='y', alpha=.25)
        ax.set_axisbelow(True)
    axes[0, 0].set_ylabel('Точность действий, %')
    axes[1, 0].set_ylabel('Точность действий, %')
    axes[0, 0].legend(fontsize=8, loc='best')
    fig.suptitle('LIFE-B2A-CAUSAL-001 · 12 development-блоков · 95% bootstrap-интервалы')
    fig.savefig(root / 'CAUSAL_RESULTS.png', dpi=175)


if __name__ == '__main__':
    main()
