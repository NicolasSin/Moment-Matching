"""Adjacent-pair RFF moments in original units, with Monte Carlo gradients.

All loss functions return the unscaled scientific loss.  Optional ``weight``
only multiplies an explicitly requested backward pass; existing gradients are
never cleared.  Fixed supplied noise is replayed without further RNG draws.
"""
from __future__ import annotations

import math

import numpy as np
import torch


class FeatureBank:
    """Independent x/y frequencies with variance gamma**-2 and float64 originals."""
    NAMES = ('wx', 'bx', 'wy', 'by')

    def __init__(self, wx, bx, wy, by, *, _canonical=None):
        tensors = (wx, bx, wy, by)
        if wx.ndim != 1 or wx.numel() == 0 or any(a.shape != wx.shape for a in tensors):
            raise ValueError('Feature banks must be nonempty equal-length vectors')
        if any(a.dtype != wx.dtype or a.device != wx.device for a in tensors):
            raise ValueError('Working feature tensors must share dtype and device')
        if wx.dtype not in (torch.float32, torch.float64):
            raise ValueError('FeatureBank supports float32 and float64')
        for name, value in zip(self.NAMES, tensors):
            setattr(self, name, value.detach())
        originals = tensors if _canonical is None else _canonical
        self._canonical = tuple(a.detach().cpu().double().contiguous().clone() for a in originals)

    @classmethod
    def draw(cls, size, gamma, seed_x, seed_y, device='cpu', dtype=torch.float64):
        if not isinstance(size, int) or size <= 0:
            raise ValueError('Feature count must be a positive integer')
        if not math.isfinite(gamma) or gamma <= 0:
            raise ValueError('RFF distance bandwidth must be positive and finite')
        if seed_x == seed_y:
            raise ValueError('Independent coordinate banks require distinct seeds')
        rng_x, rng_y = np.random.default_rng(seed_x), np.random.default_rng(seed_y)
        # NumPy's scale is the standard deviation: Var(w) = (1 / gamma)**2.
        arrays = (rng_x.normal(loc=0., scale=1 / gamma, size=size), rng_x.uniform(0., 2 * np.pi, size),
                  rng_y.normal(loc=0., scale=1 / gamma, size=size), rng_y.uniform(0., 2 * np.pi, size))
        originals = tuple(torch.from_numpy(a) for a in arrays)
        return cls(*(a.to(device=device, dtype=dtype) for a in originals), _canonical=originals)

    @property
    def size(self):
        return self.wx.numel()

    def to(self, device=None, dtype=None):
        device = self.wx.device if device is None else device
        dtype = self.wx.dtype if dtype is None else dtype
        return FeatureBank(*(a.to(device=device, dtype=dtype) for a in self._canonical),
                           _canonical=self._canonical)

    def state_dict(self):
        return {name: a.clone() for name, a in zip(self.NAMES, self._canonical)}

    @classmethod
    def from_state(cls, state, device='cpu', dtype=torch.float64):
        originals = tuple(state[name].detach().cpu().double() for name in cls.NAMES)
        return cls(*(a.to(device=device, dtype=dtype) for a in originals), _canonical=originals)

    def x_features(self, x):
        return torch.cos(x.unsqueeze(-1) * self.wx + self.bx)

    def y_features(self, y):
        return torch.cos(y.unsqueeze(-1) * self.wy + self.by)


def _validate_states(x, chunk):
    if x.ndim != 1 or x.numel() == 0:
        raise ValueError('States must be a nonempty one-dimensional tensor')
    if not isinstance(chunk, int) or chunk <= 0:
        raise ValueError('Chunk size must be a positive integer')


def _validate_noise(x, noise, minimum_columns=1):
    if noise.ndim != 2 or noise.shape[0] != len(x) or noise.shape[1] < minimum_columns:
        raise ValueError(f'Noise must have shape (len(x), R), R >= {minimum_columns}')
    if noise.device != x.device:
        raise ValueError('States and noise must share a device')


def _working_bank(bank, x):
    return bank if bank.wx.device == x.device and bank.wx.dtype == x.dtype else bank.to(x.device, x.dtype)


def data_moment(x, y, bank, chunk=2048):
    """Full pair cross matrix in original units, accumulated in float64."""
    _validate_states(x, chunk)
    if y.shape != x.shape or y.device != x.device or y.dtype != x.dtype:
        raise ValueError('Data-pair vectors must share shape, device, and dtype')
    working = _working_bank(bank, x)
    result = torch.zeros(bank.size, bank.size, device=x.device, dtype=torch.float64)
    with torch.no_grad():
        for start in range(0, len(x), chunk):
            a = working.x_features(x[start:start + chunk])
            b = working.y_features(y[start:start + chunk])
            result.add_((a.T @ b).double() / len(x))
    return result


def _contribution(model, x, noise, bank):
    generated = model(x[:, None], noise)
    expected = bank.y_features(generated).mean(dim=1)
    return bank.x_features(x).T @ expected


def model_moment(model, x, noise, bank, chunk=1024):
    """Fixed N-by-R noise, no graph, and float64 matrix accumulation."""
    _validate_states(x, chunk)
    _validate_noise(x, noise)
    working = _working_bank(bank, x)
    result = torch.zeros(bank.size, bank.size, device=x.device, dtype=torch.float64)
    with torch.no_grad():
        for start in range(0, len(x), chunk):
            contribution = _contribution(model, x[start:start + chunk],
                                         noise[start:start + chunk], working)
            result.add_(contribution.double() / len(x))
    return result


def moment_loss(model, x, noise, bank, target, chunk=1024, backward=False, weight=1.):
    """Mean of squared K-by-K errors; exact two-pass replay gradient.

    Feature evaluation and matrix products use the inputs' working dtype;
    accumulated moments, target differences, and the loss use float64.  This is
    the gradient of the supplied finite-MC objective, not an unbiased estimator
    of the square of the exact noise-integrated moments.
    """
    if target.shape != (bank.size, bank.size) or target.device != x.device:
        raise ValueError('Target must be a K-by-K moment matrix on the states device')
    if not math.isfinite(weight):
        raise ValueError('Gradient weight must be finite')
    current = model_moment(model, x, noise, bank, chunk=chunk)
    delta = current - target.detach().double()
    loss = float(delta.square().mean())
    if backward and weight != 0.:
        working = _working_bank(bank, x)
        sensitivity = (2. * float(weight) / (bank.size ** 2)) * delta.detach()
        for start in range(0, len(x), chunk):
            contribution = _contribution(model, x[start:start + chunk],
                                         noise[start:start + chunk], working)
            ((contribution.double() / len(x)) * sensitivity).sum().backward()
    return loss, current
