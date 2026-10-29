"""Atomic trusted local training checkpoints; restore RNG at epoch boundaries."""
from pathlib import Path
import os
import random
import numpy as np
import torch


def atomic_save(value, path):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as handle:
        torch.save(value, handle); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return value


def rng_state(generator):
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [state[0], state[1].tolist(), state[2], state[3], state[4]],
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'shuffle': generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state['python'])
    name, values, pos, has_gauss, cached = state['numpy']
    np.random.set_state((name, np.asarray(values, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state['torch']); generator.set_state(state['shuffle'])
    if torch.cuda.is_available() and state['cuda']:
        if len(state['cuda']) != torch.cuda.device_count(): raise ValueError('CUDA device count changed since checkpoint')
        torch.cuda.set_rng_state_all(state['cuda'])
