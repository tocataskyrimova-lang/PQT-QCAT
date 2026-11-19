"""Plot matched single-rule controls and one-write loss changes."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    root = parser.parse_args().output
    data = json.loads((root/'ANALYSIS.json').read_text(encoding='utf-8'))
    settings = [('separate','MM'),('separate','SS'),('opponent','MM'),('opponent','SS')]
    lookup = {(r['mode'],r['arm'],r['training']):r for r in data['d0_means']}
    fig,axes = plt.subplots(1,3,figsize=(12.2,4.6),layout='constrained')
    for axis,ctx in zip(axes[:2],('A','B'),strict=True):
        for i,(trained,color,label) in enumerate(((ctx,'#32618a',f'{ctx}-only'),('joint','#a6532e','A+B'))):
            scores = [100*lookup[(*mode,trained)][f'acc_{ctx}'] for mode in settings]
            axis.bar(np.arange(4)+(i-.5)*.32,scores,width=.32,color=color,label=label)
        axis.axhline(75,color='#4e5156',linestyle='--',linewidth=1)
        axis.set_xticks(range(4),[f'{m}\n{a}' for m,a in settings])
        axis.set_title(f'Правило {ctx}, Z0-64')
        axis.set_ylim(0,105)
        axis.grid(axis='y',alpha=.22)
        axis.set_axisbelow(True)
        axis.legend(loc='lower left',fontsize=8)
    axes[0].set_ylabel('Точность, %')
    axis = axes[2]
    tags = [('raw','Исходный B3'),('token_given_context','Токен при cue'),
            ('context_given_token','Cue при токене')]
    for i,(feature,label) in enumerate(tags):
        r = next(item for item in data['update_effects'] if
                 (item['mode'],item['arm'],item['anchor'],item['feature'],item['write_context'],item['read_context'])==
                 ('separate','SS','A',feature,'B','A'))['before_correct_only']
        if r['mean'] is None:
            continue
        lo,hi = r['ci95'] or [r['mean'],r['mean']]
        error = np.abs(np.array([[r['mean']-lo],[hi-r['mean']]]))
        axis.errorbar(r['mean'],i,xerr=error,fmt='o',capsize=4,color='#a6532e')
    axis.axvline(0,color='#4e5156',linestyle='--',linewidth=1)
    axis.set_yticks(range(3),[label for _,label in tags])
    axis.set_title('Запись B на снимке A-only\nseparate raw SS')
    axis.set_xlabel('ΔNLL ранее правильных ответов A')
    axis.grid(axis='x',alpha=.22)
    fig.suptitle('LIFE-QNUF-CONTEXT-BIND-001 · development')
    fig.savefig(root/'CONTEXT_BIND_RESULTS.png',dpi=175)


if __name__=='__main__':
    main()
