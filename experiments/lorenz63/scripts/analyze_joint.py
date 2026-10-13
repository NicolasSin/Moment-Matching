"""Independent autonomous trajectory and empirical adjacent-pair diagnostics."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import scipy
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance
import torch

from evaluate import SEEDS, load_checkpoint, rollout

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 2026101001


def input_label(path):
    resolved = path.resolve()
    return resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else f"external:{resolved.name}"


def load_sequence(path):
    values = np.asarray(np.load(path, allow_pickle=False, mmap_mode="r"), dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Expected a finite one-dimensional sequence with at least two states")
    return values


def joint_counts(sequence, edges):
    counts = np.histogram2d(sequence[:-1], sequence[1:], bins=(edges, edges))[0].astype(np.int64)
    if int(counts.sum()) != len(sequence) - 1:
        raise ValueError("Histogram edges did not include every adjacent pair")
    return counts


def histogram_metrics(p, q):
    midpoint = (p + q) / 2
    return dict(
        TV=float(0.5 * np.abs(p - q).sum()),
        Jensen_Shannon_divergence_nats=float(0.5 * (
            rel_entr(p, midpoint).sum() + rel_entr(q, midpoint).sum())),
        Hellinger_distance=float(np.sqrt(0.5 * np.square(np.sqrt(p) - np.sqrt(q)).sum())),
        cell_probability_RMSE=float(np.sqrt(np.mean((p - q) ** 2))),
    )


def compare_sequences(reference, trajectory, bin_counts=(120, 240, 480)):
    """Compare raw-coordinate states and pairs; normalize each sample by its own size."""
    low = min(float(reference.min()), float(trajectory.min()))
    high = max(float(reference.max()), float(trajectory.max()))
    padding = 0.5 if low == high else 0.0
    arrays, histogram_rows = {}, []
    for bins in bin_counts:
        edges = np.linspace(low - padding, high + padding, bins + 1)
        observed = joint_counts(reference, edges)
        generated = joint_counts(trajectory, edges)
        arrays.update({f"edges_{bins}": edges, f"data_counts_{bins}": observed,
                       f"K1500_counts_{bins}": generated})
        histogram_rows.append(dict(
            bins=bins, model="K1500", reference_pairs=int(observed.sum()),
            model_pairs=int(generated.sum()),
            **histogram_metrics(observed / observed.sum(), generated / generated.sum()),
        ))

    projections = []
    for index in range(24):
        angle = np.pi * index / 24
        u, v = float(np.cos(angle)), float(np.sin(angle))
        observed = u * reference[:-1] + v * reference[1:]
        generated = u * trajectory[:-1] + v * trajectory[1:]
        projections.append(dict(angle_degrees=index * 180 / 24,
                                W1=float(wasserstein_distance(observed, generated))))
    moments = {
        name: dict(count=len(values), mean=float(values.mean()),
                   std_population=float(values.std()), mass_positive=float(np.mean(values > 0)))
        for name, values in (("data", reference), ("K1500", trajectory))
    }
    report = dict(
        points=len(trajectory), pairs=len(trajectory) - 1,
        reference_points=len(reference), reference_pairs=len(reference) - 1,
        histogram_metrics=histogram_rows,
        unbinned_metrics=dict(
            state_W1=float(wasserstein_distance(reference, trajectory)),
            increment_W1=float(wasserstein_distance(np.diff(reference), np.diff(trajectory))),
            sliced_W1_24_directions=float(np.mean([row["W1"] for row in projections])),
            projection_W1=projections,
        ),
        moments=moments,
        histogram_rule=dict(
            source_range=[low, high], edge_range=[low - padding, high + padding],
            padding_per_side=padding, padding_rule="No padding; only a constant combined sequence uses 0.5 per side",
            bins=list(bin_counts), shared_coordinates=True, smoothing=False,
            orientation="counts[current_state_bin, next_state_bin]",
            endpoint_rule="Left-closed/right-open bins, except the last bin includes the right endpoint",
            normalization="Each sequence's counts divided by its own number of adjacent pairs; probability mass, not density",
        ),
        definitions=dict(
            W1="scipy.stats.wasserstein_distance: exact empirical one-dimensional CDF integration with uniform sample weights",
            sliced_W1_24_directions="Arithmetic mean of empirical W1 on unit projections at j*pi/24, j=0,...,23; not full two-dimensional W1",
            TV="0.5 * sum(abs(p-q))",
            Jensen_Shannon_divergence_nats="0.5*(KL(p||m)+KL(q||m)), m=(p+q)/2; natural logarithms",
            Hellinger_distance="sqrt(0.5 * sum((sqrt(p)-sqrt(q))**2))",
            cell_probability_RMSE="sqrt(mean((p-q)**2))",
        ),
        interpretation="Finite empirical comparison with the training reference. Serial dependence and one model trajectory do not justify iid confidence claims or establish equality of population laws.",
    )
    return arrays, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/joint",
                        help="New directory for trajectory, histograms, and metrics")
    parser.add_argument("--trajectory", type=Path,
                        help="Analyze an existing one-dimensional, already-retained .npy trajectory; no model run")
    parser.add_argument("--checkpoint", type=Path, help="Generated mode: default models/k1500_gh4000.pt")
    parser.add_argument("--seed", type=int, help=f"Generated mode: default {DEFAULT_SEED}")
    parser.add_argument("--steps", type=int, help="Generated mode: default 2100000")
    parser.add_argument("--burn-in", type=int, help="Generated mode: default 100000")
    args = parser.parse_args()
    if args.trajectory and any(value is not None for value in
                               (args.checkpoint, args.seed, args.steps, args.burn_in)):
        parser.error("--trajectory already contains retained states; do not combine it with generation options")
    steps = 2_100_000 if args.steps is None else args.steps
    burn = 100_000 if args.burn_in is None else args.burn_in
    seed = DEFAULT_SEED if args.seed is None else args.seed
    if not args.trajectory and (seed < 0 or burn < 0 or steps - burn < 2):
        parser.error("Generation requires a nonnegative seed and burn-in, and at least two retained steps")
    output = args.output.resolve()
    if any(output.is_relative_to(ROOT / name) for name in
           ("code", "configs", "data", "docs", "models", "results", "scripts")):
        parser.error("Choose an output outside the packaged code, data, models, and results")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    reference = load_sequence(ROOT / "data/train_x.npy")

    if args.trajectory:
        values = load_sequence(args.trajectory)
        protocol = dict(
            source="existing_retained_trajectory", source_file=input_label(args.trajectory),
            generated_in_this_run=False, seed=None, initial_index=None, initial_state=None,
            total_steps=None, burn_steps=None, checkpoint=None,
            trajectory_provenance="Not inferred from array contents; input is treated as already retained, with no further burn-in",
        )
    else:
        checkpoint = args.checkpoint or ROOT / "models/k1500_gh4000.pt"
        model, payload = load_checkpoint(checkpoint, "cpu")
        model.requires_grad_(False)
        if next(model.parameters()).dtype != torch.float64:
            raise ValueError("Joint rollout requires a float64 checkpoint")
        features = payload.get("config", {}).get("features_per_side")
        if features is not None and features != 1500:
            raise ValueError("The K1500 output schema requires a checkpoint with 1500 features per side")
        rng = np.random.default_rng(seed)
        index = int(rng.integers(len(reference)))
        initial = float(reference[index])
        noise = rng.standard_normal(steps)[:, None]
        full = rollout(model, np.asarray([initial]), noise, output)
        values = np.asarray(full[0, burn:])
        protocol = dict(
            source="autonomous_model_rollout", generated_in_this_run=True,
            checkpoint=input_label(checkpoint), adam_updates=payload.get("adam_updates"),
            lbfgs_updates=payload.get("gh_updates"), seed=seed,
            seed_distinct_from_marginal_rollouts=seed not in SEEDS,
            initial_index=index, initial_state=initial, initial_state_source="data/train_x.npy",
            initial_state_in_output=False, total_steps=steps, burn_steps=burn,
            randomness="NumPy default_rng(seed), then integers(len(train_x)), then standard_normal(total_steps)",
            device="cpu", dtype="float64", batch_size=1, block_steps=25_000,
            teacher_forced=False, clipping=False, restarts=False,
        )
    protocol.update(
        retained_steps=len(values), reference="data/train_x.npy", reference_states=len(reference),
        coordinates="Original scalar state units", analysis_dtype="float64", analysis_threads=1,
        file_counts={"joint_rollout_2m.npy": {"states": len(values), "adjacent_pairs": len(values) - 1},
                     "data/train_x.npy": {"states": len(reference), "adjacent_pairs": len(reference) - 1}},
    )
    np.save(output / "joint_rollout_2m.npy", values, allow_pickle=False)
    arrays, report = compare_sequences(reference, values)
    np.savez_compressed(output / "joint_histograms.npz", **arrays)
    report.update(
        protocol=protocol, seed=protocol["seed"], burn=protocol["burn_steps"],
        software=dict(python=sys.version.split()[0], numpy=np.__version__,
                      scipy=scipy.__version__, torch=str(torch.__version__)),
        elapsed_seconds=time.perf_counter() - started,
    )
    (output / "joint_metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n",
    )


if __name__ == "__main__":
    main()
