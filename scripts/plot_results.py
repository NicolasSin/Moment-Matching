"""Redraw the single-model figures from saved arrays; no model run is needed."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_COLOR, DATA_COLOR = "#008e83", "#477fae"


def save(fig, output, name):
    fig.savefig(output / (name + ".png"), dpi=180, facecolor="white", bbox_inches="tight")
    fig.savefig(output / (name + ".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


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
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    rollout_figure(args.results, args.output)
    if (args.results / "loss_history.csv").exists():
        loss_figure(args.results / "loss_history.csv", args.output)
    if (args.results / "joint_histograms.npz").exists():
        joint_figures(args.results / "joint_histograms.npz", args.output, args.joint_bins)
    print(f"Saved figures to {args.output.resolve()}")


if __name__ == "__main__":
    main()
