"""Versioned, atomic NPZ checkpoints; all arrays are non-pickled numeric arrays."""
from pathlib import Path
import json
import os
import uuid
import numpy as np


def save_state(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}

    def pack(value):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError('Object arrays are not checkpoint data')
            key = f'array_{len(arrays)}'
            arrays[key] = value
            return {'__ndarray__': key}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {k: pack(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [pack(v) for v in value]
        return value

    metadata = json.dumps({'schema': 2, 'state': pack(state)}, allow_nan=False)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('wb') as f:
            np.savez_compressed(f, metadata=np.array(metadata), **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def load_state(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        if metadata.get('schema') != 2:
            raise ValueError('Unsupported checkpoint schema')

        def unpack(value):
            if isinstance(value, dict):
                if set(value) == {'__ndarray__'}:
                    return archive[value['__ndarray__']].copy()
                return {k: unpack(v) for k, v in value.items()}
            if isinstance(value, list):
                return [unpack(v) for v in value]
            return value

        return unpack(metadata['state'])
