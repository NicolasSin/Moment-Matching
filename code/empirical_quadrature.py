"""Compress the scalar empirical measure into positive local Gaussian rules.

Each occupied interval contributes at most three nodes. The arithmetic order
matches the original experiment; only per-bin diagnostic bookkeeping is removed.
"""
from __future__ import annotations

import numpy as np


def _local_rule(points, counts):
    probability = counts.astype(np.float64) / float(counts.sum())
    low, high = float(points[0]), float(points[-1])
    center = low + (high - low) / 2
    scale = max(high - center, center - low)
    if len(points) <= 3 or scale == 0:
        return points.copy(), probability.copy()

    # Three Lanczos steps construct the local Jacobi matrix. Reorthogonalize
    # twice so closely spaced empirical atoms remain numerically stable.
    diagonal = (points - center) / scale
    vector = np.sqrt(probability)
    vector /= np.linalg.norm(vector)
    basis, alphas, betas = [], [], []
    previous = np.zeros_like(vector)
    previous_beta = 0.0
    for index in range(3):
        basis.append(vector.copy())
        transformed = diagonal * vector
        alpha = float(np.dot(vector, transformed))
        alphas.append(alpha)
        residual = transformed - alpha * vector - previous_beta * previous
        for _ in range(2):
            for direction in basis:
                residual -= np.dot(direction, residual) * direction
        if index == 2:
            break
        beta = float(np.linalg.norm(residual))
        if beta <= 32 * np.finfo(np.float64).eps:
            break
        betas.append(beta)
        previous, vector = vector, residual / beta
        previous_beta = beta

    jacobi = np.diag(np.asarray(alphas, dtype=np.float64))
    if betas:
        links = np.asarray(betas, dtype=np.float64)
        jacobi += np.diag(links, 1) + np.diag(links, -1)
    scaled_nodes, eigenvectors = np.linalg.eigh(jacobi)
    nodes = np.clip(center + scale * scaled_nodes, low, high)
    weights = np.square(eigenvectors[0, :])
    keep = weights > 0
    nodes, weights = nodes[keep], weights[keep]
    weights /= weights.sum()
    return nodes, weights


def build_empirical_quadrature(values, width):
    """Return nodes and probability weights using every supplied observation.

    Bins start at the observed minimum. Repeated values retain their empirical
    mass; there is no subsampling. Local polynomials through degree five are
    reproduced up to floating-point error, not arbitrary nonlinear functions.
    """
    observations = np.asarray(values, dtype=np.float64)
    if (observations.ndim != 1 or not len(observations)
            or not np.isfinite(observations).all()):
        raise ValueError("Expected a nonempty finite scalar sequence")
    if not np.isfinite(width) or width <= 0:
        raise ValueError("Quadrature width must be positive and finite")

    ordered = np.sort(observations)
    count = len(ordered)
    low, high = float(ordered[0]), float(ordered[-1])
    shifted = ordered - low
    if high == low:
        bin_starts = np.array([0, count], dtype=np.int64)
    else:
        relative_positions = shifted / float(width)
        if (not np.isfinite(relative_positions).all()
                or float(relative_positions[-1]) >= np.iinfo(np.int64).max):
            raise ValueError("Too many bins for the requested quadrature width")
        bin_numbers = np.floor(relative_positions).astype(np.int64)
        bin_starts = np.r_[0, np.flatnonzero(bin_numbers[1:] != bin_numbers[:-1]) + 1, count]

    node_parts, weight_parts = [], []
    for begin, end in zip(bin_starts[:-1], bin_starts[1:]):
        points, counts = np.unique(ordered[begin:end], return_counts=True)
        nodes, weights = _local_rule(points, counts)
        node_parts.append(nodes)
        weight_parts.append(weights * (float(end - begin) / count))
    nodes = np.concatenate(node_parts).astype(np.float64, copy=False)
    weights = np.concatenate(weight_parts).astype(np.float64, copy=False)
    weights /= float(weights.sum())
    if (not np.isfinite(nodes).all() or not np.isfinite(weights).all()
            or not np.all(weights > 0) or not np.all((nodes >= low) & (nodes <= high))):
        raise FloatingPointError("Invalid empirical quadrature")
    return nodes, weights
