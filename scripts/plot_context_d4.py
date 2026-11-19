"""Show paired NLL differences at the radius feasible for both directions."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    root = parser.parse_args().run
    data = json.loads((root/'ANALYSIS.json').read_text(encoding='utf-8'))['contrasts']
    fig,axes = plt.subplots(1,2,figsize=(10.6,4.2),layout='constrained')
    left = [('direction@min',k) for k in ('A_same','B_same','A_other','B_other')]
    right = [(key,'B_same') for key in ('RAW@min-NONE','CTX@min-NONE')]
    labels = [('A · записанный токен','B · записанный токен','A · остальные','B · остальные'),('RAW','CTX')]
    for axis,spec,names,title in zip(axes,(left,right),labels,('CTX минус RAW','Польза записи B против NONE'),strict=True):
        for y,(key,group) in enumerate(spec):
            value = data[key][group]['nll']
            if value['mean'] is None:
                continue
            mean = value['mean']
            lo,hi = value['ci95'] or (mean,mean)
            axis.errorbar(mean,y,xerr=np.abs(np.array([[mean-lo],[hi-mean]])),fmt='o',capsize=4,color='#346083')
        axis.set_yticks(range(len(names)),names)
        axis.set_ylim(-.55,len(names)-.45)
        axis.invert_yaxis()
        axis.axvline(0,color='#5d6064',linestyle='--',lw=1)
        axis.grid(axis='x',alpha=.25)
        axis.set_title(title)
        axis.set_xlabel('Разность NLL · 95% парный ДИ')
    fig.suptitle('D4 · одна запись B · общий меньший радиус каждой пары')
    fig.savefig(root/'D4_RESULTS.png',dpi=175)


if __name__=='__main__':
    main()
