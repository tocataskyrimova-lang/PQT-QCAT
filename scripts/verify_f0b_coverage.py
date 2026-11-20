"""Independently recompute F0B choices from the delivered F0 query table."""
import argparse
from collections import defaultdict
import csv
from itertools import combinations
import json
from pathlib import Path


def csv_rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def verify(run, original):
    plan = json.loads((run/'PLAN.json').read_text(encoding='utf-8'))
    order = plan['candidate_order']
    masks = [list(c) for size in (3,4,5,6) for c in combinations(range(1,7),size)]
    assert masks==plan['masks']
    events, banks, a_scores, b_scores = set(), {}, defaultdict(dict), {}
    for row in csv_rows(original/'F0_PER_QUERY.csv'):
        arm = row['policy']
        if arm not in order:
            continue
        key = tuple(int(row[field]) for field in ('seed','mapping','write_token'))
        events.add(key)
        correct = int(row['correct'])
        before = int(row['before_correct'])
        delta = float(row['delta_nll'])
        if row['context']=='A':
            a_scores[key,arm][int(row['read_token'])] = delta if correct>=before else float('inf')
        elif row['slice']=='B_current':
            b_scores[key,arm] = (delta<=-.10 and correct>=before, not before and correct==1 and delta<=-.10)
    decisions = json.loads((run/'DECISIONS_BEFORE_DIAGNOSTICS.json').read_text(encoding='utf-8'))
    for row in decisions:
        key = tuple(row[field] for field in ('seed','mapping','write_token'))
        mask = row['mask']
        allowed = {arm: b_scores[key,arm][0] and max(a_scores[key,arm][t] for t in mask)<=.01 for arm in order}
        selected = next((arm for arm in order if allowed[arm]),'NONE')
        assert allowed==row['admitted'] and selected==row['choice']
        slot = (key,tuple(mask))
        assert slot not in banks
        banks[slot] = selected
    assert len(events)==54 and len(banks)==54*42
    for row in csv_rows(run/'TRADEOFF.csv'):
        bound = float(row['bound'])
        available = gains = selected = 0
        for key in events:
            legal = [arm for arm in order if b_scores[key,arm][0] and max(a_scores[key,arm].values())<=bound]
            selected += bool(legal)
            available += any(b_scores[key,arm][1] for arm in legal)
            gains += bool(legal and b_scores[key,legal[0]][1])
        assert (available,gains,selected)==tuple(int(row[k]) for k in ('available_new_B','new_B','selected'))
    print(json.dumps({'exact_mask_choices':len(banks),'covered_historical_events':len(events),
                      'tradeoff_rows_verified':len(csv_rows(run/'TRADEOFF.csv'))},indent=2))


if __name__=='__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('run',type=Path)
    cli.add_argument('original',type=Path)
    args = cli.parse_args()
    verify(args.run,args.original)
