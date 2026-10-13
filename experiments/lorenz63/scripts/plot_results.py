"""Plot archived model diagnostics."""
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


def autocorrelation_zoom(times, reference, groups, output, name, title, note):
    """Plot short-lag ACF detail; groups contain (label, color, per-chain curves)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, limit in zip(axes, (0.1, 0.01)):
        mask = times <= limit + 1e-12
        last = np.flatnonzero(mask)[-1]
        detail = limit == 0.01
        label = "Training reference"
        if detail:
            label += f": {reference[last]:.9f}"
        ax.plot(times[mask], reference[mask], "o-", color="#20252b", lw=1.5,
                markersize=5, markerfacecolor="white", label=label, zorder=5)
        for (label, color, curves), marker in zip(groups, ("s", "^", "D")):
            mean, spread = curves.mean(axis=0), curves.std(axis=0)
            ax.fill_between(times[mask], (mean - spread)[mask], (mean + spread)[mask],
                            color=color, alpha=0.13, linewidth=0)
            if detail:
                label += f": {mean[last]:.9f}"
            ax.plot(times[mask], mean[mask], marker=marker, color=color, lw=1.2,
                    markersize=5, markerfacecolor="none", label=label)
        ax.set(xlabel="Time lag", ylabel="Autocorrelation",
               title="Short-lag detail: 0 to 0.1" if not detail else "One-step detail: 0 to 0.01",
               xlim=(-limit * 0.025, limit * 1.025))
        ax.set_xticks(times[mask])
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(alpha=0.15)
        ax.legend(frameon=False, fontsize=8,
                  title=f"ACF at time lag {times[last]:g}" if detail else None,
                  title_fontsize=9, loc="lower left")
    fig.suptitle(title, fontsize=14)
    fig.text(0.5, 0.045, note, ha="center", fontsize=9)
    fig.text(0.5, 0.012,
             "Sampling interval = 0.01. Markers are sampled lags; lines only connect them. "
             "No samples lie between 0 and 0.01.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.085, 1, 0.95))
    save(fig, output, name)


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
    fig.suptitle("Autocorrelation: K = 1500; 4,000 L-BFGS updates", fontsize=15)
    fig.text(0.5, 0.015,
             f"{samples.shape[1]:,} retained states per seed; burn-in already removed. "
             "Each trajectory centered separately; lag-adjusted normalization.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    save(fig, output, "autocorrelation")
    autocorrelation_zoom(times, reference, [("L-BFGS 4,000", MODEL_COLOR, curves)], output,
                         "autocorrelation_zoom", "Short-lag autocorrelation: K = 1500; 4,000 L-BFGS updates",
                         f"{samples.shape[1]:,} retained states per seed. Model: mean of three ACFs; "
                         "bands: seed population SD, not a confidence interval.")


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
    ax.axvline(1000, color="#64748b", linewidth=1, linestyle="--", label="Adam to L-BFGS")
    ax.set(xlabel="Total updates (1,000 Adam + 4,000 L-BFGS)", ylabel="Moment-matching loss",
           title="Training and validation losses (K = 1500)")
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
        ax.stairs(density, edges, color=MODEL_COLOR, linewidth=1.5, label="K = 1500; L-BFGS 4,000")
        record = report["per_seed"][i - 1] if i else None
        w1 = record["W1_to_train_2m"] if record else report["averages"]["W1_to_train_2m"]["mean"]
        title = "Average across three seeds" if i == 0 else f"Seed {record['seed']}"
        ax.set_title(f"{title}  |  W1 = {w1:.4f}", loc="left", fontsize=10)
        ax.set(xlabel="State x (original units)", ylabel="Probability density")
        ax.grid(alpha=0.12)
        ax.legend(frameon=False, fontsize=8)
    protocol = report["protocol"]
    fig.suptitle("Autonomous trajectories: K = 1500; 4,000 L-BFGS updates", fontsize=15)
    fig.text(0.5, 0.018,
             f"{protocol['total_steps']:,} steps per seed; first {protocol['burn_steps']:,} discarded. "
             "Shared bins; no clipping or restarts.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save(fig, output, "rollout_distribution")


def joint_figures(path, output, bins):
    with np.load(path, allow_pickle=False) as stored:
        edges = stored[f"edges_{bins}"]
        counts = [stored[f"{name}_counts_{bins}"] for name in ("data", "K1500")]
    probabilities = [c / c.sum() for c in counts]
    limit = max(p.max() for p in probabilities)
    metadata_path = path.with_name("joint_metrics.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    seed = metadata.get("seed", metadata.get("protocol", {}).get("seed"))
    label = "Autonomous model" + (f" (seed {seed})" if seed is not None else "")
    titles = ("Observed training trajectory", label)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), constrained_layout=True)
    for ax, probability, title in zip(axes, probabilities, titles):
        mesh = ax.pcolormesh(edges, edges, probability.T, shading="flat", cmap="viridis", vmin=0, vmax=limit)
        ax.set(xlabel=r"$x_t$", ylabel=r"$x_{t+1}$", title=title, aspect="equal")
    fig.colorbar(mesh, ax=list(axes), label="Probability per bin", shrink=0.85)
    fig.suptitle(f"Adjacent-pair probabilities: {int(counts[1].sum()):,} model pairs; {bins} x {bins} bins")
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



def validation_figure(path, output):
    """Observed-input joint and increment densities; arrays are stored as [x, y]."""
    with np.load(path, allow_pickle=False) as stored:
        histograms = dict(stored)
    report_path = path.with_name("one_step_metrics.json")
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        metrics = report["diagnostics"]
    else:
        report = json.loads(path.with_name("rollout_metrics.json").read_text(encoding="utf-8"))["validation"]
        metrics = report
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for row, view in enumerate(("joint", "residual")):
        x_edges, y_edges = histograms["x_edges"], histograms[view + "_y_edges"]
        area = np.diff(x_edges)[:, None] * np.diff(y_edges)[None, :]
        data = histograms[view + "_data_counts"]
        model = histograms[view + "_model_counts"]
        data_density, model_density = data / (data.sum() * area), model / (model.sum() * area)
        difference = model_density - data_density
        maximum, delta = max(data_density.max(), model_density.max()), np.max(np.abs(difference))
        for col, (values, title) in enumerate(((data_density, "Observed validation pairs"),
                (model_density, "L-BFGS 4,000: 32 latent draws per observed input"),
                (difference, "Model minus observed"))):
            ax = axes[row, col]
            artist = ax.pcolormesh(x_edges, y_edges, values.T, shading="auto",
                                  cmap="RdBu_r" if col == 2 else "magma",
                                  vmin=-delta if col == 2 else 0, vmax=delta if col == 2 else maximum)
            fig.colorbar(artist, ax=ax, label="Density difference" if col == 2 else "Probability density")
            ax.set(title=title, xlabel="Current x", ylabel="Next state y" if view == "joint" else "Increment y - x")
    pairs = int(histograms["joint_data_counts"].sum())
    generated = int(histograms["joint_model_counts"].sum())
    fig.suptitle("L-BFGS 4,000: validation-data joint distribution diagnostics", fontsize=17, fontweight="bold")
    fig.text(.07, .022,
             f"All {pairs:,} validation pairs; {generated:,} model pairs. Raw units, 180 x 180 common bins, no smoothing.\n"
             f"W1(y) = {metrics['marginal_y_W1']:.5g}; W1(y-x) = {metrics['residual_W1']:.5g}. "
             "Fixed observed inputs; no autonomous rollout.", fontsize=10)
    fig.tight_layout(rect=(0, .065, 1, .95))
    save(fig, output, "joint_residual_preview")

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
    if (args.results / "rollout_samples.npy").exists():
        rollout_figure(args.results, args.output)
        autocorrelation_figure(args.results, args.output, args.acf_max_lag)
    if (args.results / "loss_history.csv").exists():
        loss_figure(args.results / "loss_history.csv", args.output)
    if (args.results / "joint_histograms.npz").exists():
        joint_figures(args.results / "joint_histograms.npz", args.output, args.joint_bins)
    if (args.results / "validation_histograms.npz").exists():
        validation_figure(args.results / "validation_histograms.npz", args.output)
    print(f"Saved figures to {args.output.resolve()}")


if __name__ == "__main__":
    main()
