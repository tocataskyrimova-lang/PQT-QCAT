"""Resume a real QNUF life from disk in a fresh Python process."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pqt_qcat.memory.life_qnuf_001 import life_task
from pqt_qcat.memory.checkpoint_io import load_state


def compare(a, b):
    if isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            compare(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            compare(left, right)
    else:
        assert a == b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('replay', type=Path)
    args = parser.parse_args()
    plan = json.loads((args.root / 'frozen_config.json').read_text())
    seed = plan['stage1_seeds'][0]
    life = Path('stage1') / 'lives' / f'B3_{seed}_0'
    original = args.root / life
    target = args.replay / life
    target.mkdir(parents=True, exist_ok=False)
    shutil.copy2(original / 'checkpoint_024.npz', target / 'latest.npz')
    life_task((plan, str(args.replay), 'stage1', 'B3', seed, 0, None))
    compare(load_state(original / 'checkpoint_072.npz'), load_state(target / 'checkpoint_072.npz'))
    assert (original / 'event_records.json').read_bytes() == (target / 'event_records.json').read_bytes()
    print('Exact match for final checkpoint and all 72 event records after disk resume')


if __name__ == '__main__':
    main()
