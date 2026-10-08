"""Evaluate the scalar conditional generator in original state units on CPU."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

# The original rollout used one thread for numerical libraries as well as PyTorch.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import scipy
from scipy.stats import wasserstein_distance
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from model import load_checkpoint

SEEDS = (20261304, 20261305, 20261306)
# Saved starts from the original 800,000-point historical trajectory.
STARTS = (-14.598013133703434, -9.732162349591096, 7.61336425587853)
INITIAL_INDICES = (95844, 538435, 773739)


class ScriptedRollout(nn.Module):
    """Original traced generator inside a scripted, sequential recurrence."""

    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        result = torch.empty_like(noise)
        alive = torch.isfinite(x)
        missing = torch.full_like(x, float("nan"))
        for step in range(noise.shape[0]):
            candidate = self.generator(x, noise[step])
            alive = alive & torch.isfinite(candidate)
            x = torch.where(alive, candidate, missing)
            result[step] = x
        return result


def make_streams(steps):
    rngs = [np.random.default_rng(seed) for seed in SEEDS]
    # Keep this draw even though starts are archived: it advances each RNG.
    indices = [int(rng.integers(800_000)) for rng in rngs]
    if tuple(indices) != INITIAL_INDICES:
        raise RuntimeError("NumPy random stream differs from the saved protocol")
    noise = np.stack([rng.standard_normal(steps) for rng in rngs], axis=1)
    return np.asarray(STARTS, dtype=np.float64), noise


def rollout(model, starts, noise, output):
    """Save [chain, time]; the initial state is not included in the output."""
    initial = torch.as_tensor(starts.copy(), dtype=torch.float64)
    latent = torch.as_tensor(noise, dtype=torch.float64)
    full = np.lib.format.open_memmap(
        output / "rollout_full.npy", mode="w+", dtype=np.float64,
        shape=(len(starts), len(noise)),
    )
    with torch.inference_mode():
        traced = torch.jit.trace(model, (initial, latent[0]), check_trace=True)
        scripted = torch.jit.script(ScriptedRollout(traced).eval())
        # Preserve the original warm-up before the main sequential loop.
        audit_steps = min(20, len(noise))
        state = initial.clone()
        expected = []
        for step in range(audit_steps):
            state = model(state, latent[step])
            expected.append(state.clone())
        np.testing.assert_allclose(
            scripted(initial, latent[:audit_steps]).numpy(),
            torch.stack(expected).numpy(), rtol=1e-12, atol=1e-12,
        )
        state = initial
        for begin in range(0, len(noise), 25_000):
            end = min(begin + 25_000, len(noise))
            chunk = scripted(state, latent[begin:end])
            if not bool(torch.isfinite(chunk).all()):
                raise FloatingPointError(f"Nonfinite rollout in steps {begin}:{end}")
            full[:, begin:end] = chunk.numpy().T
            state = chunk[-1]
            full.flush()
    return full


def statistics(values):
    return dict(
        count=len(values), mean=float(values.mean()),
        std_population=float(values.std()),
        mass_abs_x_lt_1=float(np.mean(np.abs(values) < 1)),
        minimum=float(values.min()), maximum=float(values.max()),
    )


def chain_metrics(values, train, seed):
    first, second = values[:len(values) // 2], values[len(values) // 2:]
    a, b = statistics(first), statistics(second)
    return dict(
        seed=seed, **statistics(values),
        W1_to_train_2m=float(wasserstein_distance(train, values)),
        first_second_half_W1=float(wasserstein_distance(first, second)),
        first_half=a, second_half=b,
        second_minus_first_mean=b["mean"] - a["mean"],
        second_minus_first_std=b["std_population"] - a["std_population"],
        second_minus_first_center_mass=b["mass_abs_x_lt_1"] - a["mass_abs_x_lt_1"],
    )


def validation_metrics(model, sequence, output):
    """Observed inputs, 32 independent latent draws per adjacent pair."""
    x, y = sequence[:-1], sequence[1:]
    noise = np.random.default_rng(2026100731).standard_normal((len(x), 32))
    samples = np.empty_like(noise)
    with torch.inference_mode():
        for begin in range(0, len(x), 1024):
            end = min(begin + 1024, len(x))
            samples[begin:end] = model(
                torch.from_numpy(x[begin:end, None]),
                torch.from_numpy(noise[begin:end]),
            ).numpy()
    if not np.isfinite(samples).all():
        raise FloatingPointError("Nonfinite validation predictions")
    residual = samples - x[:, None]
    saved, distances = {}, {}
    x_edges = np.linspace(x.min(), x.max(), 181)
    for name, observed, generated in (
        ("joint", y, samples), ("residual", y - x, residual),
    ):
        y_edges = np.linspace(
            min(observed.min(), generated.min()),
            max(observed.max(), generated.max()), 181,
        )
        data_counts = np.histogram2d(x, observed, bins=(x_edges, y_edges))[0]
        model_counts = np.histogram2d(
            np.repeat(x, 32), generated.ravel(), bins=(x_edges, y_edges),
        )[0]
        distances[name + "_histogram_TV"] = float(
            0.5 * np.abs(data_counts / len(x) - model_counts / samples.size).sum()
        )
        saved.update({name + "_y_edges": y_edges,
                      name + "_data_counts": data_counts.astype(np.int64),
                      name + "_model_counts": model_counts.astype(np.int64)})
    np.savez_compressed(output / "validation_histograms.npz", x_edges=x_edges, **saved)
    return dict(
        points=len(sequence), pairs=len(x), samples_per_input=32, noise_seed=2026100731,
        teacher_forced=True,
        marginal_y_W1=float(wasserstein_distance(y, samples.ravel())),
        residual_W1=float(wasserstein_distance(y - x, residual.ravel())),
        **distances,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1_100_000, help="Generated steps per chain")
    parser.add_argument("--burn-in", type=int, help="Default: 100000, or 10%% for short runs")
    parser.add_argument("--validation-points", type=int, default=100_000,
                        help="Validation prefix length; 0 skips this diagnostic")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/k1500_gh4000.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/evaluation")
    args = parser.parse_args()
    burn = min(100_000, args.steps // 10) if args.burn_in is None else args.burn_in
    if args.steps < 2 or burn < 0 or args.steps - burn < 2:
        parser.error("Need at least two retained steps and nonnegative burn-in")
    if args.validation_points not in (0,) and not 2 <= args.validation_points <= 100_000:
        parser.error("--validation-points must be 0 or between 2 and 100000")
    output = args.output.resolve()
    if output == ROOT / "results" or ROOT / "results" in output.parents:
        parser.error("Evaluation outputs must be outside the archived results directory")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("Output directory is not empty; choose a new --output directory")
    began = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model, _ = load_checkpoint(args.checkpoint, "cpu")
    model.requires_grad_(False)
    if next(model.parameters()).dtype != torch.float64:
        raise ValueError("This evaluation protocol requires a float64 checkpoint")
    train = np.load(ROOT / "data/train_x.npy", allow_pickle=False)
    starts, noise = make_streams(args.steps)
    full = rollout(model, starts, noise, output)
    kept = np.asarray(full[:, burn:])
    np.save(output / "rollout_samples.npy", kept, allow_pickle=False)
    records = [chain_metrics(row, train, seed) for row, seed in zip(kept, SEEDS)]
    keys = ("W1_to_train_2m", "mean", "std_population", "mass_abs_x_lt_1",
            "first_second_half_W1", "second_minus_first_mean",
            "second_minus_first_std", "second_minus_first_center_mass")
    report = dict(
        protocol=dict(total_steps=args.steps, burn_steps=burn, retained_steps=args.steps - burn,
                      seeds=list(SEEDS), initial_states=list(STARTS), initial_state_in_output=False,
                      device="cpu", dtype="float64", batch_size=3, block_steps=25_000),
        per_seed=records,
        averages={key: dict(mean=float(np.mean([r[key] for r in records])),
                           seed_std_population=float(np.std([r[key] for r in records])))
                  for key in keys},
    )
    if args.validation_points:
        validation = np.load(ROOT / "data/validation_x.npy", allow_pickle=False)
        report["validation"] = validation_metrics(model, validation[:args.validation_points], output)
    report["software"] = dict(python=sys.version.split()[0], numpy=np.__version__,
                              scipy=scipy.__version__, torch=torch.__version__)
    report["elapsed_seconds"] = time.perf_counter() - began
    (output / "rollout_metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(f"Mean W1 to training reference: {report['averages']['W1_to_train_2m']['mean']:.9g}")
    print(f"Saved evaluation to {output}")


if __name__ == "__main__":
    main()
