"""Scalar conditional generator and portable checkpoint loading/saving."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import uuid

import torch
from torch import nn


# Keep the historical identifier so the supplied checkpoint loads unchanged.
SCHEMA = 'scalar_plain_mm500_input_compare_v1'
ARCHITECTURE = 'plain_mlp_2_96_96_1_gelu'
WEIGHT_BOUNDS = (0.5, 0.2, 1.0)
BIAS_BOUND = 0.05


class MMGenerator(nn.Module):
    def __init__(self, mean: float, std: float, mode: str, seed: int = 20261004,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise ValueError('Normalization requires a finite mean and positive finite std')
        if dtype not in (torch.float32, torch.float64):
            raise ValueError('MMGenerator supports float32 and float64')
        if mode not in ('standardized', 'raw'):
            raise ValueError('Mode must be standardized or raw')
        self.mode = mode
        self.register_buffer('x_mean', torch.tensor(mean if mode == 'standardized' else 0., dtype=dtype))
        self.register_buffer('x_std', torch.tensor(std if mode == 'standardized' else 1., dtype=dtype))
        if not bool(torch.isfinite(self.x_mean) & torch.isfinite(self.x_std)) or float(self.x_std) <= 0:
            raise ValueError('Normalization is not representable in the working dtype')
        self.initialization_seed = int(seed)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.net = nn.Sequential(
                nn.Linear(2, 96, dtype=dtype), nn.GELU(),
                nn.Linear(96, 96, dtype=dtype), nn.GELU(),
                nn.Linear(96, 1, dtype=dtype),
            )
            for layer, bound in zip((self.net[0], self.net[2], self.net[4]), WEIGHT_BOUNDS):
                nn.init.uniform_(layer.weight, -bound, bound)
                nn.init.uniform_(layer.bias, -BIAS_BOUND, BIAS_BOUND)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        state, noise = torch.broadcast_tensors(x, z)
        inputs = torch.stack(((state - self.x_mean) / self.x_std, noise), dim=-1)
        return self.net(inputs).squeeze(-1)


def cpu_state(module):
    """Clone saved values without changing parameters' or buffers' dtypes."""
    state = module if isinstance(module, Mapping) else module.state_dict()
    return {name: value.detach().cpu().contiguous().clone()
            for name, value in state.items()}


def state_hash(module):
    """Canonical SHA256 over names, dtypes, shapes, and saved tensor bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(cpu_state(module).items()):
        header = json.dumps({'name': name, 'dtype': str(tensor.dtype),
                             'shape': list(tensor.shape)},
                            sort_keys=True, separators=(',', ':')).encode('utf-8')
        raw = tensor.numpy().tobytes(order='C')
        digest.update(len(header).to_bytes(8, 'big'))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, 'big'))
        digest.update(raw)
    return digest.hexdigest()


def save_checkpoint(path, model, **metadata):
    """Atomically save the actual working model, including normalization bits.

    Source float64 normalization statistics may be passed separately in metadata;
    they cannot replace the values actually used by this model.
    """
    reserved = {'schema', 'architecture', 'model', 'normalization', 'dtype',
                'model_state_hash', 'mode'}
    if reserved.intersection(metadata):
        raise ValueError('Checkpoint metadata attempts to override structural fields')
    state = cpu_state(model)
    dtype = model.x_mean.dtype
    if dtype not in (torch.float32, torch.float64) or any(t.dtype != dtype for t in state.values()):
        raise ValueError('All model tensors must share float32 or float64 dtype')
    payload = dict(metadata)
    payload.update(
        schema=SCHEMA, architecture=ARCHITECTURE, model=state, mode=model.mode,
        normalization={'mean': float(state['x_mean']), 'std': float(state['x_std']),
                       'dtype': str(dtype).removeprefix('torch.')},
        dtype=str(dtype).removeprefix('torch.'), model_state_hash=state_hash(state),
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('schema') != SCHEMA or payload.get('architecture') != ARCHITECTURE:
        raise ValueError('Unrecognized joint-generator checkpoint schema or architecture')
    dtype = {'float32': torch.float32, 'float64': torch.float64}.get(payload.get('dtype'))
    if dtype is None:
        raise ValueError('Unsupported checkpoint dtype')
    state = payload['model']
    if any(t.dtype != dtype for t in state.values()):
        raise ValueError('Checkpoint tensor dtypes disagree with its declaration')
    if state_hash(state) != payload['model_state_hash']:
        raise ValueError('Checkpoint model tensor hash mismatch')
    norm = payload['normalization']
    if (norm['mean'] != float(state['x_mean']) or norm['std'] != float(state['x_std'])
            or norm['dtype'] != payload['dtype']):
        raise ValueError('Checkpoint normalization does not match saved working buffers')
    if payload.get('mode') not in ('standardized', 'raw'):
        raise ValueError('Missing or invalid checkpoint input mode')
    if payload['mode'] == 'raw' and (norm['mean'] != 0. or norm['std'] != 1.):
        raise ValueError('Raw-input checkpoint must have identity normalization buffers')
    model = MMGenerator(norm['mean'], norm['std'], mode=payload['mode'], dtype=dtype)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), payload
