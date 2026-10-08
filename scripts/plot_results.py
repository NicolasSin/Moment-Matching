"""Redraw the saved model results and checkpoint ACFs; no model run is needed."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import irfft, next_fast_len, rfft

ROOT = Path(__file__).resolve().parents[1]
MODEL_COLOR, DATA_COLOR = "#008e83", "#477fae"


def save(fig, output, name):
    fig.savefig(output / (name + ".png"), dpi=180, facecolor="white", bbox_inches="tight")
    fig.savefig(output / (name + ".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def autocorrelation(values, max_lag=10_000):
    """Lag-adjusted ACF, centering each trajectory by its own sample mean.

    rho[k] = n/(n-k) * sum(x_centered[t] * x_centered[t+k]) / sum(x_centered**2).
    Zero padding prevents the end of the trajectory wrapping to its beginning.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Expected a finite one-dimensional trajectory")
    if not isinstance(max_lag, (int, np.integer)) or not 0 <= max_lag < len(values):
        raise ValueError("max_lag must be an integer between 0 and trajectory length minus 1")
    centered = values - values.mean()
    energy = float(np.dot(centered, centered))
    if not np.isfinite(energy) or energy <= 0:
        raise ValueError("Autocorrelation requires a finite positive sample variance")
    # Padding to n + max_lag is enough for the requested nonnegative lags.
    fft_length = next_fast_len(len(values) + max_lag, real=True)
    spectrum = rfft(centered, n=fft_length)
    products = irfft(spectrum * spectrum.conj(), n=fft_length)[:max_lag + 1]
    lags = np.arange(max_lag + 1)
    correlation = (products / energy) * (len(values) / (len(values) - lags))
    correlation[0] = 1.0
    return correlation


def autocorrelation_figure(results, output, max_lag=10_000):
    train = np.load(ROOT / "data/train_x.npy", allow_pickle=False)
    samples = np.load(results / "rollout_samples.npy", allow_pickle=False, mmap_mode="r")
    report = json.loads((results / "rollout_metrics.json").read_text(encoding="utf-8"))
    seeds = [record["seed"] for record in report["per_seed"]]
    if samples.ndim != 2 or samples.shape[0] != 3 or len(seeds) != 3:
        raise ValueError("Expected three saved rollout trajectories and their seed labels")
    max_lag = min(max_lag, len(train) - 1, samples.shape[1] - 1)
    reference = autocorrelation(train, max_lag)
    curves = np.stack([autocorrelation(row, max_lag) for row in samples])
    mean, spread = curves.mean(axis=0), curves.std(axis=0)
    lags = np.arange(max_lag + 1)
    times = lags * 0.01
    output.mkdir(parents=True, exist_ok=True)
    with (output / "autocorrelation.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["lag", "time", "reference", *[f"seed{seed}" for seed in seeds], "mean", "std"])
        writer.writerows(zip(lags, times, reference, *curves, mean, spread))

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.7), sharey=True)
    colors = ("#74b4a7", "#a2c9be", "#418e83")
    for ax, limit, title in zip(axes, (min(500, max_lag), max_lag), ("Short lags", "Long lags")):
        selection = slice(0, limit + 1)
        ax.plot(times[selection], reference[selection], color=DATA_COLOR, linewidth=1.8,
                label="Training reference (2M)")
        for seed, curve, color in zip(seeds, curves, colors):
            ax.plot(times[selection], curve[selection], color=color, linewidth=0.7, alpha=0.6,
                    label=f"Seed {seed}")
        ax.plot(times[selection], mean[selection], color=MODEL_COLOR, linewidth=1.8,
                label="Mean of three model ACFs")
        ax.axhline(0, color="#94a3b8", linewidth=0.7, zorder=0)
        ax.set(xlabel="Time lag (steps x 0.01)", ylabel="Autocorrelation",
               title=f"{title}: 0 to {limit:,} steps", xlim=(0, max(times[limit], 0.01)))
        ax.grid(alpha=0.15)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Autocorrelation: K1500 / GH4000", fontsize=15)
    fig.text(0.5, 0.015,
             f"{samples.shape[1]:,} retained states per seed; burn-in already removed. "
             "Each trajectory centered separately; lag-adjusted normalization.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    save(fig, output, "autocorrelation")


def checkpoint_autocorrelation_figures(path, output):
    """Redraw the original-training checkpoint ACF supplement from saved curves."""
    with np.load(path, allow_pickle=False) as stored:
        lags, times = stored["lag"], stored["time"]
        reference = stored["reference_acf"]
        steps, seeds = stored["steps"], stored["seeds"]
        curves = stored["model_acf"]
        selected = int(stored["selected_step"])
        excluded = stored["excluded_steps"]
    if curves.shape != (len(steps), len(seeds), len(lags)) or reference.shape != lags.shape:
        raise ValueError("Checkpoint ACF arrays have inconsistent dimensions")
    means, spreads = curves.mean(axis=1), curves.std(axis=1)
    colors = {4000: "#cf852d", selected: MODEL_COLOR, 24000: "#8962a6"}
    selected_steps = (4000, selected, 24000)
    output.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, mask, title in zip(axes, (lags <= 500, np.ones(len(lags), dtype=bool)),
                               ("Short lags", "Long lags")):
        ax.plot(times[mask], reference[mask], color="#20252b", linewidth=1.7,
                label="Training reference (2M)")
        for step in selected_steps:
            index = int(np.flatnonzero(steps == step)[0])
            mean, spread = means[index], spreads[index]
            ax.fill_between(times[mask], (mean - spread)[mask], (mean + spread)[mask],
                            color=colors[step], alpha=0.13, linewidth=0)
            ax.plot(times[mask], mean[mask], color=colors[step], linewidth=1.6,
                    label=f"GH {step:,}")
        ax.axhline(0, color="#94a3b8", linewidth=0.7, zorder=0)
        ax.set(xlabel="Time lag (steps x 0.01)", ylabel="Autocorrelation", title=title,
               xlim=(0, float(times[mask][-1])))
        ax.grid(alpha=0.15)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Original training checkpoints: temporal correlation", fontsize=15)
    fig.text(0.5, 0.015,
             "Three seeds, 2M retained states per seed. Lines: mean ACF; bands: seed population SD, not a confidence interval.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    save(fig, output, "autocorrelation_checkpoints")

    fig, axes = plt.subplots(4, 5, figsize=(16, 11), sharex=True, sharey=True)
    for index, ax in enumerate(axes.flat[:-1]):
        step = int(steps[index])
        color = colors.get(step, DATA_COLOR)
        ax.plot(times, reference, color="#20252b", linewidth=1.0,
                label="Training reference", zorder=3)
        for chain, curve in enumerate(curves[index]):
            ax.plot(times, curve, color=color, linewidth=0.55, alpha=0.35,
                    label="Individual seeds" if chain == 0 else None)
        ax.plot(times, means[index], color=color, linewidth=1.2, label="Model mean ACF", zorder=4)
        ax.axhline(0, color="#94a3b8", linewidth=0.6, zorder=0)
        ax.set_title(f"GH {step:,}" + (" (selected)" if step == selected else ""),
                     loc="left", fontsize=10)
        ax.set_xlim(0, float(times[-1]))
        ax.grid(alpha=0.12)
    note = axes.flat[-1]
    note.axis("off")
    excluded_text = "\n".join(f"GH {int(step):,}" for step in excluded)
    note.text(0.04, 0.92, f"Excluded checkpoints\n\n{excluded_text}\n\nNonfinite autonomous rollouts.\nNo ACF is shown.",
              transform=note.transAxes, va="top", fontsize=11, linespacing=1.5, color="#475569")
    for ax in axes[-1, :-1]:
        ax.set_xlabel("Time lag")
    for ax in axes[:, 0]:
        ax.set_ylabel("Autocorrelation")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.43, 0.005))
    fig.suptitle("Original training: all 19 finite checkpoints", fontsize=16)
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    save(fig, output, "autocorrelation_all_checkpoints")


def loss_figure(path, output):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for column, label, color in (
        ("train_loss", "Training MM loss", MODEL_COLOR),
        ("validation_loss", "Validation MM loss", DATA_COLOR),
    ):
        points = [(float(row["global_update"]), float(row[column])) for row in rows
                  if row[column] and np.isfinite(float(row[column])) and float(row[column]) > 0]
        if points:
            x, y = zip(*points)
            ax.semilogy(x, y, label=label, color=color, linewidth=1.5)
    ax.axvline(1000, color="#64748b", linewidth=1, linestyle="--", label="Adam to GH")
    ax.set(xlabel="Total updates (1,000 Adam + 4,000 GH)", ylabel="Moment-matching loss",
           title="K1500 / GH4000: recorded training and validation losses")
    ax.grid(alpha=0.15)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, output, "loss_history")


def rollout_figure(results, output):
    train = np.load(ROOT / "data/train_x.npy", allow_pickle=False)
    samples = np.load(results / "rollout_samples.npy", allow_pickle=False, mmap_mode="r")
    report = json.loads((results / "rollout_metrics.json").read_text(encoding="utf-8"))
    if samples.ndim != 2 or not np.isfinite(samples).all():
        raise ValueError("Expected finite rollout samples with shape [chain, time]")
    edges = np.linspace(min(train.min(), samples.min()), max(train.max(), samples.max()), 101)
    widths = np.diff(edges)
    reference = np.histogram(train, edges)[0] / (len(train) * widths)
    densities = np.stack([np.histogram(row, edges)[0] / (len(row) * widths) for row in samples])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        ax.stairs(reference, edges, fill=True, color=DATA_COLOR, alpha=0.25, label="Training data (2M)")
        density = densities.mean(axis=0) if i == 0 else densities[i - 1]
        ax.stairs(density, edges, color=MODEL_COLOR, linewidth=1.5, label="K1500 / GH4000")
        record = report["per_seed"][i - 1] if i else None
        w1 = record["W1_to_train_2m"] if record else report["averages"]["W1_to_train_2m"]["mean"]
        title = "Average across three seeds" if i == 0 else f"Seed {record['seed']}"
        ax.set_title(f"{title}  |  W1 = {w1:.4f}", loc="left", fontsize=10)
        ax.set(xlabel="State x (original units)", ylabel="Probability density")
        ax.grid(alpha=0.12)
        ax.legend(frameon=False, fontsize=8)
    protocol = report["protocol"]
    fig.suptitle("Autonomous rollout: K1500 / GH4000", fontsize=15)
    fig.text(0.5, 0.018,
             f"{protocol['total_steps']:,} steps per seed; first {protocol['burn_steps']:,} discarded. "
             "Shared bins; no clipping or restarts.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save(fig, output, "rollout_distribution")


def joint_figures(path, output, bins):
    with np.load(path, allow_pickle=False) as stored:
        edges = stored[f"edges_{bins}"]
        counts = [stored[f"{name}_counts_{bins}"] for name in ("data", "K1500")]
    if any(c.sum() != 1_999_999 for c in counts):
        raise ValueError("Archived 2M joint histograms must contain 1,999,999 adjacent pairs")
    probabilities = [c / c.sum() for c in counts]
    limit = max(p.max() for p in probabilities)
    titles = ("Observed training trajectory", "Autonomous K1500 / GH4000")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), constrained_layout=True)
    for ax, probability, title in zip(axes, probabilities, titles):
        mesh = ax.pcolormesh(edges, edges, probability.T, shading="flat", cmap="viridis", vmin=0, vmax=limit)
        ax.set(xlabel=r"$x_t$", ylabel=r"$x_{t+1}$", title=title, aspect="equal")
    fig.colorbar(mesh, ax=list(axes), label="Probability per bin", shrink=0.85)
    fig.suptitle(f"Adjacent-pair probabilities: 2,000,000 continuous points; {bins} x {bins} bins")
    save(fig, output, "joint_probability")

    centers = (edges[:-1] + edges[1:]) / 2
    xx, yy = np.meshgrid(centers, centers, indexing="ij")
    fig = plt.figure(figsize=(12, 5.4))
    for i, (probability, title) in enumerate(zip(probabilities, titles), 1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        ax.plot_surface(xx, yy, probability, cmap="viridis", vmin=0, vmax=limit,
                        rstride=1, cstride=1, linewidth=0, antialiased=False)
        ax.set(xlabel=r"$x_t$", ylabel=r"$x_{t+1}$", zlabel="Probability per bin",
               title=title, zlim=(0, limit * 1.03))
        ax.view_init(elev=28, azim=-62)
    fig.suptitle(f"Joint probability surfaces ({bins} x {bins} bins; no statistical smoothing)")
    fig.tight_layout()
    save(fig, output, "joint_probability_surface")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/figures")
    parser.add_argument("--joint-bins", type=int, choices=(120, 240, 480), default=120)
    parser.add_argument("--acf-max-lag", type=int, default=10_000,
                        help="Largest autocorrelation lag in steps (default: 10000)")
    args = parser.parse_args()
    if args.acf_max_lag < 0:
        parser.error("--acf-max-lag must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    rollout_figure(args.results, args.output)
    autocorrelation_figure(args.results, args.output, args.acf_max_lag)
    if (args.results / "loss_history.csv").exists():
        loss_figure(args.results / "loss_history.csv", args.output)
    if (args.results / "joint_histograms.npz").exists():
        joint_figures(args.results / "joint_histograms.npz", args.output, args.joint_bins)
    checkpoint_acf = args.results / "checkpoint_scan/autocorrelations.npz"
    if checkpoint_acf.exists():
        checkpoint_autocorrelation_figures(checkpoint_acf, args.output)
    print(f"Saved figures to {args.output.resolve()}")


if __name__ == "__main__":
    main()
