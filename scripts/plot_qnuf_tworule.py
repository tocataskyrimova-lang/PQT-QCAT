"""Show accuracy for each conflicting rule under the two frozen starts."""
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
    data = json.loads((root/'ANALYSIS.json').read_text(encoding='utf-8'))
    values = {(r['encoder'],r['policy'],r['probe']): r for r in data['probe_means']}
    modes = [('separate','MM'),('separate','SS'),('opponent','MM'),('opponent','SS')]
    labels = [f'{mode}\nraw {arm}' for mode, arm in modes]
    fig, axes = plt.subplots(1,2,figsize=(11.4,4.4),sharey=True,layout='constrained')
    for axis, (probe, title) in zip(axes,(('z0','Из Z0, 64 такта'),('live','Из живого Z, 64 такта')),strict=True):
        for i, (rule,color) in enumerate((('A','#2d5682'),('B','#af5227'))):
            scores = [100*values.get((*mode,probe),{}).get(f'acc_{rule}',np.nan) for mode in modes]
            axis.bar(np.arange(len(modes))+(i-.5)*.32,scores,width=.32,color=color,label=f'Правило {rule}')
        axis.axhline(100*data['gate']['threshold'],color='#41464b',linestyle='--',label='Порог C0')
        axis.set_xticks(range(len(modes)),labels)
        axis.set_title(title)
        axis.set_ylim(0,105)
        axis.set_axisbelow(True)
        axis.grid(axis='y',alpha=.22)
    axes[0].set_ylabel('Точность на фиксированных запросах, %')
    axes[1].legend(loc='best',fontsize=8)
    fig.suptitle('LIFE-QNUF-TWORULE-001 · Stage C0 · 9 development-блоков')
    fig.savefig(root/'C0_RESULTS.png',dpi=175)


if __name__ == '__main__':
    main()
