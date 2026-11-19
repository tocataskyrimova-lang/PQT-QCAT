"""Visualize the fixed set of policies and controls on both environments."""
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
    arms = ('raw_MM', 'raw_SS', 'norm050_SM', 'norm050_SS', 'TABLE', 'initial_SS', 'wrong_SS')
    labels = ('raw MM', 'raw SS', 'norm 0,50 SM', 'norm 0,50 SS', 'Таблица', 'Начальная K', 'Чужая K')
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True, layout='constrained')
    for axis, key, multiplier, title in [(left, 'accuracy', 100, 'Точность действий, %'),
                                         (right, 'reward', 1, 'Награда за 24 события')]:
        for env, label, color, shift in [('retry', 'С повтором', '#657282', -.11),
                                         ('advance', 'Без повторов', '#a94e21', .11)]:
            for i, arm in enumerate(arms):
                value = data['stats'][f'{env}/{arm}'].get(key, {})
                if value.get('mean') is None:
                    continue
                lo, hi = value['ci95'] or [value['mean'], value['mean']]
                error = np.abs(np.array([[value['mean'] - lo], [hi - value['mean']]])) * multiplier
                axis.errorbar(value['mean'] * multiplier, i + shift, xerr=error, label=label if i == 0 else '',
                              color=color, fmt='o', capsize=3)
        axis.set_xlabel(title)
        axis.set_yticks(range(len(arms)), labels)
        axis.grid(axis='x', alpha=.25)
        axis.set_axisbelow(True)
    left.invert_yaxis()
    right.legend(loc='best', fontsize=8)
    fig.suptitle('LIFE-QNUF-CONFIRM-001 · независимая панель · 95% интервалы')
    fig.savefig(root / 'CONFIRM_RESULTS.png', dpi=175)


if __name__ == '__main__':
    main()
