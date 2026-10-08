"""Float64 Gaussian quadrature for the unchanged raw-pair RFF moment loss.

The model and fixed feature bank are never modified. Optional positive input
weights integrate an empirical-input quadrature; acceptance against all original
observations and a refined latent rule belongs to the calling experiment.
"""
from __future__ import annotations

import math
from numbers import Integral

import numpy as np
from scipy.special import roots_hermitenorm
import torch


def gh_rule(order, device='cpu'):
    """Nodes and probability weights for Z~N(0,1), both torch.float64.

    Hermite-norm roots integrate exp(-z*z/2), so no sqrt(2) rescaling of nodes
    is needed. Normalizing the positive quadrature mass supplies the Gaussian
    density factor. Underflowed extreme-tail weights may be exactly zero.
    """
    if not isinstance(order, Integral) or isinstance(order, bool) or order < 1:
        raise ValueError('Gauss-Hermite order must be a positive integer')
    nodes, weights = roots_hermitenorm(int(order))
    nodes = np.asarray(nodes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (not np.isfinite(nodes).all() or not np.isfinite(weights).all()
            or np.any(weights < 0) or not float(weights.sum()) > 0):
        raise FloatingPointError('Invalid Gauss-Hermite rule')
    weights /= weights.sum()
    return (torch.tensor(nodes, dtype=torch.float64, device=device),
            torch.tensor(weights, dtype=torch.float64, device=device))


def _validate_rule(x, z_nodes, z_weights, chunk, x_weights):
    if x.ndim != 1 or x.numel() == 0 or x.dtype != torch.float64:
        raise ValueError('Input states must be a nonempty float64 vector')
    if not isinstance(chunk, Integral) or isinstance(chunk, bool) or chunk < 1:
        raise ValueError('Chunk size must be a positive integer')
    if (z_nodes.ndim != 1 or z_nodes.numel() == 0 or z_weights.shape != z_nodes.shape
            or z_nodes.dtype != torch.float64 or z_weights.dtype != torch.float64
            or z_nodes.device != x.device or z_weights.device != x.device):
        raise ValueError('Latent nodes/weights must be matching float64 vectors on the states device')
    if (not bool(torch.isfinite(z_nodes).all()) or not bool(torch.isfinite(z_weights).all())
            or not bool((z_weights >= 0).all())
            or not math.isclose(float(z_weights.sum()), 1., rel_tol=0., abs_tol=2e-12)):
        raise ValueError('Latent weights must be finite, nonnegative, and normalized to one')
    if x_weights is not None:
        if (x_weights.shape != x.shape or x_weights.dtype != torch.float64
                or x_weights.device != x.device):
            raise ValueError('Input weights must match the float64 states vector and device')
        if (not bool(torch.isfinite(x_weights).all()) or not bool((x_weights > 0).all())
                or not math.isclose(float(x_weights.sum()), 1., rel_tol=0., abs_tol=2e-12)):
            raise ValueError('Input weights must be finite, strictly positive, and normalized to one')


def _working_bank(bank, x):
    # FeatureBank.to restores the canonical float64 draws even if the caller's
    # working bank was rounded for an earlier float32 phase.
    return bank.to(device=x.device, dtype=torch.float64)


def _contribution(model, states, z_nodes, z_weights, bank, state_weights, total_states):
    generated = model(states[:, None], z_nodes[None, :])
    expected = (bank.y_features(generated) * z_weights[None, :, None]).sum(dim=1)
    if state_weights is None:
        return (bank.x_features(states).T @ expected) / total_states
    return bank.x_features(states).T @ (expected * state_weights[:, None])


def weighted_model_moment(model, x, z_nodes, bank, z_weights, chunk=128, x_weights=None):
    """Return the full K-by-K cross matrix with fixed latent and input weights.

    With no x_weights, each supplied observation receives mass 1/len(x).
    Explicit x_weights already sum to one and receive no additional 1/N factor.
    This function retains no autograd graph and never updates parameters.
    """
    _validate_rule(x, z_nodes, z_weights, chunk, x_weights)
    working = _working_bank(bank, x)
    current = torch.zeros(bank.size, bank.size, dtype=torch.float64, device=x.device)
    with torch.no_grad():
        for start in range(0, len(x), int(chunk)):
            end = start + int(chunk)
            local_weights = None if x_weights is None else x_weights[start:end]
            current.add_(_contribution(model, x[start:end], z_nodes, z_weights,
                                       working, local_weights, len(x)))
    return current


def weighted_moment_loss(model, x, z_nodes, bank, target, z_weights, chunk=128,
                         x_weights=None, backward=False, weight=1.):
    """Raw mean(delta**2) and matrix, with exact fixed-rule replay gradients.

    dL/dM = 2*delta/K**2. The second pass differentiates the identical weighted
    chunk matrices against that fixed sensitivity. Input and latent weights are
    both retained. ``weight`` scales only the accumulated backward contribution;
    the returned scientific loss is always unscaled. Existing parameter grads
    are never cleared. No model output, input state, or loss is clipped.
    """
    if (target.shape != (bank.size, bank.size) or target.dtype != torch.float64
            or target.device != x.device):
        raise ValueError('Target must be a float64 K-by-K matrix on the states device')
    if not math.isfinite(weight):
        raise ValueError('Gradient weight must be finite')
    current = weighted_model_moment(model, x, z_nodes, bank, z_weights,
                                    chunk=chunk, x_weights=x_weights)
    delta = current - target.detach()
    loss = float(delta.square().mean())
    if backward and weight != 0.:
        working = _working_bank(bank, x)
        sensitivity = delta.detach() * (2. * float(weight) / (bank.size ** 2))
        with torch.enable_grad():
            for start in range(0, len(x), int(chunk)):
                end = start + int(chunk)
                local_weights = None if x_weights is None else x_weights[start:end]
                contribution = _contribution(model, x[start:end], z_nodes, z_weights,
                                             working, local_weights, len(x))
                (contribution * sensitivity).sum().backward()
    return loss, current
