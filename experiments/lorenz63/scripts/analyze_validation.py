"""Observed-input validation diagnostics in original state units.

Recreates the archived one-step metrics, conditional table and common-bin
histograms from the supplied model and validation sequence. No autonomous
trajectory is generated. The validation sequence was monitored during training.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import sys
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '2'
# Two threads match the original validation reductions.
import numpy as np
from scipy.stats import wasserstein_distance
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from model import load_checkpoint
R = 32
SEED = 2026100731
BANDWIDTH_SEED = 2026100740
RFF_SEEDS = (2026100741, 2026100742)

def basic(values):
    return dict(mean=float(np.mean(values)), std_population=float(np.std(values)))

def paired_histogram(x, data_y, model_y, x_edges, y_edges):
    n, repeats = (len(x), model_y.shape[1])
    data_counts = np.histogram2d(x, data_y, bins=[x_edges, y_edges])[0].astype(np.int64)
    model_counts = np.histogram2d(np.repeat(x, repeats), model_y.ravel(), bins=[x_edges, y_edges])[0].astype(
        np.int64,
    )
    if data_counts.sum() != n or model_counts.sum() != n * repeats:
        raise AssertionError('Histogram clipped observations')
    area = np.diff(x_edges)[:, None] * np.diff(y_edges)[None, :]
    dp, mp = (data_counts / n, model_counts / (n * repeats))
    dd, md = (dp / area, mp / area)
    return dict(
        data_counts=data_counts,
        model_counts=model_counts,
        data_probability=dp,
        model_probability=mp,
        data_density=dd,
        model_density=md,
        difference_density=md - dd,
        histogram_TV=float(0.5 * np.abs(mp - dp).sum()),
    )

def conditional_bins(x, y, samples):
    edges = np.linspace(x.min(), x.max(), 42)
    indices = np.minimum(np.searchsorted(edges, x, side='right') - 1, len(edges) - 2)
    rows = []
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = indices == i
        local_x = x[mask]
        dy = y[mask]
        my = samples[mask]
        count = int(mask.sum())
        row = dict(
            index=i,
            x_low=float(low),
            x_high=float(high),
            x_center=float((low + high) / 2),
            n_data=count,
            n_model=count * samples.shape[1],
            sparse=count < 200,
        )
        if count:
            dr = dy - local_x
            mr = my - local_x[:, None]
            row.update(
                W1_y=float(wasserstein_distance(dy, my.ravel())),
                W1_residual=float(wasserstein_distance(dr, mr.ravel())),
                data_y_mean=float(dy.mean()),
                model_y_mean=float(my.mean()),
                data_y_std=float(dy.std()),
                model_y_std=float(my.std()),
                data_residual_mean=float(dr.mean()),
                model_residual_mean=float(mr.mean()),
                data_residual_std=float(dr.std()),
                model_residual_std=float(mr.std()),
            )
            row['mean_bias_y'] = row['model_y_mean'] - row['data_y_mean']
            row['mean_bias_residual'] = row['model_residual_mean'] - row['data_residual_mean']
            for label, observed, generated in [('y', dy, my.ravel()), ('residual', dr, mr.ravel())]:
                (a, b) = (
                    min(float(observed.min()), float(generated.min())),
                    max(float(observed.max()), float(generated.max())),
                )
                if a == b:
                    a -= 0.5
                    b += 0.5
                bins = np.linspace(a, b, 101)
                width = np.diff(bins)
                dc = np.histogram(observed, bins)[0]
                mc = np.histogram(generated, bins)[0]
                assert dc.sum() == len(observed) and mc.sum() == len(generated)
                row[label + '_edges'] = bins.tolist()
                row['data_' + label + '_density'] = (dc / (len(observed) * width)).tolist()
                row['model_' + label + '_density'] = (mc / (len(generated) * width)).tolist()
        else:
            row.update(W1_y=None, W1_residual=None)
        rows.append(row)
    return dict(
        x_edges=edges.tolist(),
        sparse_threshold=200,
        interpretation='Distributions conditioned on a finite x bin; not exact pointwise conditional laws',
        bins=rows,
    )

def sliced_distances(x, y, samples, marginal_y_w1, residual_w1):
    repeated = np.repeat(x, samples.shape[1])
    flat = samples.ravel()
    angles = np.arange(24) * np.pi / 24
    rows = []
    for i, angle in enumerate(angles):
        direction = np.array([np.cos(angle), np.sin(angle)])
        if i == 0:
            direction = np.array([1.0, 0.0])
        if i == 12:
            direction = np.array([0.0, 1.0])
        if i == 18:
            direction = np.array([-1.0, 1.0]) / np.sqrt(2.0)
        if i == 12:
            distance = marginal_y_w1
        else:
            distance = float(
                wasserstein_distance(
                    direction[0] * x + direction[1] * y,
                    direction[0] * repeated + direction[1] * flat,
                ),
            )
        rows.append(dict(angle_degrees=float(angle * 180 / np.pi), direction=direction.tolist(), W1=distance))
    return dict(
        number_of_directions=len(rows),
        directions=rows,
        mean_W1=float(np.mean([row['W1'] for row in rows])),
        maximum_W1=float(max((row['W1'] for row in rows))),
        x_projection_W1=rows[0]['W1'],
        y_projection_W1=rows[12]['W1'],
        normalized_residual_projection_W1=rows[18]['W1'],
        raw_residual_W1=residual_w1,
        interpretation='Finite deterministic directions in raw (x,y) units; not full multivariate Wasserstein distance',
    )

def rff_bank_moments(x, y, samples, view, center, scale, omega, phase):
    dimensions = len(phase)
    data_sum = np.zeros(dimensions)
    a_sum = np.zeros(dimensions)
    b_sum = np.zeros(dimensions)
    for first in range(0, len(x), 512):
        xx = x[first:first + 512]
        yy = y[first:first + 512]
        mm = samples[first:first + 512]
        sx = (xx - center[0]) / scale[0]
        observed_y = yy if view == 'joint' else yy - xx
        generated_y = mm if view == 'joint' else mm - xx[:, None]
        sy = (observed_y - center[1]) / scale[1]
        sm = (generated_y - center[1]) / scale[1]
        observed = np.sqrt(2.0) * np.cos(sx[:, None] * omega[:, 0] + sy[:, None] * omega[:, 1] + phase)
        generated = np.sqrt(2.0) * np.cos(sx[:, None, None] * omega[:, 0] + sm[:, :, None] * omega[:, 1] + phase)
        data_sum += observed.sum(axis=0)
        midpoint = samples.shape[1] // 2
        a_sum += generated[:, :midpoint, :].sum(axis=(0, 1))
        b_sum += generated[:, midpoint:, :].sum(axis=(0, 1))
    data_mean = data_sum / len(x)
    first_mean = a_sum / (len(x) * (samples.shape[1] // 2))
    second_mean = b_sum / (len(x) * (samples.shape[1] // 2))
    model_mean = (first_mean + second_mean) / 2
    mmd2 = float(np.mean((model_mean - data_mean) ** 2))
    split = float(np.mean((first_mean - second_mean) ** 2))
    return (
        dict(
            MMD2=mmd2,
            MMD=float(np.sqrt(mmd2)),
            MC_half_split_MMD2=split,
            estimated_full_MC_variance_contribution=split / 4,
        ),
        dict(
            data_mean=data_mean,
            model_mean=model_mean,
            model_first_half_mean=first_mean,
            model_second_half_mean=second_mean,
            omega=omega,
            phase=phase,
        ),
    )

def independent_rff(x, y, samples):
    reports = {}
    for number, view in enumerate(['joint', 'residual']):
        observed = np.column_stack([x, y if view == 'joint' else y - x])
        center = observed.mean(axis=0)
        scale = observed.std(axis=0)
        if np.any(scale <= 0):
            raise ValueError('Degenerate diagnostic coordinate')
        standardized = (observed - center) / scale
        rng = np.random.default_rng(BANDWIDTH_SEED + number)
        first = rng.integers(len(x), size=200000)
        second = rng.integers(len(x) - 1, size=200000)
        second += second >= first
        gamma = float(np.median(np.linalg.norm(standardized[first] - standardized[second], axis=1)))
        banks = []
        for index, seed in enumerate(RFF_SEEDS):
            rng = np.random.default_rng(seed + 10 * number)
            omega = rng.normal(0.0, 1 / gamma, size=(256, 2))
            phase = rng.uniform(0, 2 * np.pi, size=256)
            result, _ = rff_bank_moments(x, y, samples, view, center, scale, omega, phase)
            banks.append(dict(seed=seed + 10 * number, **result))
        reports[view] = dict(
            features_per_bank=256,
            independent_banks=2,
            coordinate_mean=center.tolist(),
            coordinate_scale=scale.tolist(),
            bandwidth=gamma,
            bandwidth_method='Median Euclidean distance of 200000 sampled distinct-index pairs in data-standardized coordinates',
            bandwidth_seed=BANDWIDTH_SEED + number,
            kernel='Gaussian exp(-||standardized_coordinate_difference||^2/(2*bandwidth^2))',
            feature='sqrt(2) cos(omega dot standardized_coordinate + phase); MMD2 = mean squared mean-feature discrepancy',
            banks=banks,
            mean_MMD2=float(np.mean([b['MMD2'] for b in banks])),
            interpretation='Independent diagnostic features; not comparable numerically to original training MM loss. MC split measures only latent-sampling noise conditional on this fixed validation sequence.',
        )
    return reports

def analyze(checkpoint, sequence, output):
    """Return full diagnostics and save the three portable analysis artifacts."""
    model, _ = load_checkpoint(checkpoint, 'cpu')
    model.requires_grad_(False)
    if next(model.parameters()).dtype != torch.float64:
        raise ValueError('The recorded validation protocol requires a float64 model')
    x, y = (sequence[:-1], sequence[1:])
    n = len(x)
    noise = np.random.default_rng(SEED).standard_normal((n, R))
    samples = np.empty_like(noise)
    with torch.inference_mode():
        for begin in range(0, n, 1024):
            end = min(begin + 1024, n)
            samples[begin:end] = model(torch.from_numpy(x[begin:end, None]), torch.from_numpy(noise[begin:end])).numpy(

            )
    if not np.isfinite(samples).all():
        raise FloatingPointError('Nonfinite validation predictions')
    dr, mr = (y - x, samples - x[:, None])
    x_edges = np.linspace(x.min(), x.max(), 181)
    raw_edges = np.linspace(min(y.min(), samples.min()), max(y.max(), samples.max()), 181)
    delta_edges = np.linspace(min(dr.min(), mr.min()), max(dr.max(), mr.max()), 181)
    joint = paired_histogram(x, y, samples, x_edges, raw_edges)
    residual = paired_histogram(x, dr, mr, x_edges, delta_edges)
    np.savez_compressed(
        output / 'validation_histograms.npz',
        x_edges=x_edges,
        joint_y_edges=raw_edges,
        residual_y_edges=delta_edges,
        **{f'{name}_{key}': value for name, record in (('joint', joint), ('residual', residual)) for key, value in record.items() if isinstance(value, np.ndarray)},
    )
    marginal_y_w1 = float(wasserstein_distance(y, samples.ravel()))
    residual_w1 = float(wasserstein_distance(dr, mr.ravel()))
    conditional = conditional_bins(x, y, samples)
    directions = sliced_distances(x, y, samples, marginal_y_w1, residual_w1)
    rff = independent_rff(x, y, samples)
    diagnostics = dict(
        marginal_y_W1=marginal_y_w1,
        residual_W1=residual_w1,
        W1_y=marginal_y_w1,
        W1_residual=residual_w1,
        sliced_W1_mean=directions['mean_W1'],
        MC_split_W1_y=float(wasserstein_distance(samples[:, :R // 2].ravel(), samples[:, R // 2:].ravel())),
        MC_split_W1_residual=float(wasserstein_distance(mr[:, :R // 2].ravel(), mr[:, R // 2:].ravel())),
        MC_split_interpretation='Two independent sets of 16 latent samples at each identical input; model-side sampling variability only, not data uncertainty or confidence intervals.',
        data_y=basic(y),
        model_y=basic(samples),
        data_residual=basic(dr),
        model_residual=basic(mr),
        joint_histogram_TV=joint['histogram_TV'],
        residual_histogram_TV=residual['histogram_TV'],
        sliced_W1=directions,
        independent_RFF=rff,
        conditional_W1_y_data_weighted_mean=float(
            sum((r['n_data'] * r['W1_y'] for r in conditional['bins'] if r['n_data'])) / n,
        ),
        conditional_W1_residual_data_weighted_mean=float(
            sum((r['n_data'] * r['W1_residual'] for r in conditional['bins'] if r['n_data'])) / n,
        ),
    )
    report = dict(
        validation_pairs=n,
        samples_per_input=R,
        noise_seed=SEED,
        diagnostics=diagnostics,
        conditional_bins=conditional['bins'],
        limitations=[
            'Validation pairs are temporally dependent; no iid uncertainty or significance claim.',
            'Conditional bins mix nearby inputs and sparse bins have limited data support.',
            'Finite-direction sliced W1 and finite-feature MMD do not prove equality of full joint laws.',
            'Near-diagonal raw joint geometry can conceal transition errors; residual diagnostics are reported separately.',
            'Latent MC split measures model-side sampling variability conditional on this dataset only.',
            'This validation set was monitored during retraining; it is not a never-used test set.',
        ],
    )
    (output / 'one_step_metrics.json').write_text(
        json.dumps(report, indent=2, allow_nan=False) + '\n',
        encoding='utf-8',
        newline='\n',
    )
    conditional_fields = [
        'index',
        'x_low',
        'x_high',
        'x_center',
        'n_data',
        'n_model',
        'sparse',
        'W1_y',
        'W1_residual',
        'data_y_mean',
        'model_y_mean',
        'data_y_std',
        'model_y_std',
        'data_residual_mean',
        'model_residual_mean',
        'data_residual_std',
        'model_residual_std',
        'mean_bias_y',
        'mean_bias_residual',
    ]
    with (output / 'conditional_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=conditional_fields,
            extrasaction='ignore',
            lineterminator='\n',
        )
        writer.writeheader()
        writer.writerows(conditional['bins'])
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'models/k1500_gh4000.pt')
    parser.add_argument(
        '--validation-points',
        type=int,
        default=100000,
        help='Use a validation prefix; default: all 100000 observations',
    )
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/validation_analysis')
    args = parser.parse_args()
    sequence = np.load(ROOT / 'data/validation_x.npy', allow_pickle=False)
    if sequence.ndim != 1 or sequence.dtype != np.float64 or (not np.isfinite(sequence).all()):
        parser.error('Expected a finite float64 validation sequence')
    if not 3 <= args.validation_points <= len(sequence):
        parser.error('--validation-points must be between 3 and the sequence length')
    output = args.output.resolve()
    if any((output.is_relative_to(ROOT / name) for name in ('code', 'configs', 'data', 'docs', 'models', 'results', 'scripts'))):
        parser.error('Choose an output outside the supplied code, data, models and results')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    analyze(args.checkpoint, sequence[:args.validation_points], output)
if __name__ == '__main__':
    main()
