# Evaluation and figures

Run these commands from the repository root after installing its dependencies.
The supplied `results/` contains archived results. New evaluation runs default
to `outputs/evaluation/`, and new figures to `outputs/figures/`.

## Redraw the supplied results

```bash
python scripts/plot_results.py
```

This reads the saved loss table, three retained rollout trajectories, and the
2-million-point adjacent-pair histograms. It creates PNG and PDF figures for
the K1500 / GH4000 model only. It does not train or run the model. Use
`--output PATH` to choose a figure directory. `--joint-bins 120`, `240`, or `480`
selects one of the saved grids (default: 120).

The joint plots show **probability per bin**, counts divided by 1,999,999, not
probability density. Both panels use the same bin edges and colour range.
Connecting bin centres in a surface is a display operation, not statistical
smoothing. The archived edges are retained without re-binning.

## Quick CPU run

```bash
python scripts/evaluate.py --steps 1000 --burn-in 100 --validation-points 1000 --output outputs/quick
python scripts/plot_results.py --results outputs/quick --output outputs/quick_figures
```

This exercises the evaluation path. A short trajectory is not a reproduction
of the reported long-run distribution. Set `--validation-points 0` to skip
the separate observed-input transition diagnostic.

## Full archived rollout protocol

```bash
python scripts/evaluate.py
```

The default generates 1,100,000 steps for each of three seeds, discards the
first 100,000, and saves `rollout_samples.npy` with shape `(3, 1000000)`.
`rollout_full.npy` also retains the burn-in. The initial state is not an output
sample. Existing nonempty evaluation directories are not overwritten; use
`--output outputs/another_run` for a second run.

The frozen model is loaded from `models/k1500_gh4000.pt`. Input normalisation
comes from its saved buffers, including their original rounding. Outputs
already use original x units and receive no inverse scaling. Evaluation is
CPU float64 with one numerical-library and PyTorch compute thread. The three chains are batched,
using a traced model inside the original scripted sequential loop, in blocks
of 25,000 steps. There is no clipping or restart.

| Seed | Historical start index | Initial x |
| --- | ---: | ---: |
| 20261304 | 95844 | -14.598013133703434 |
| 20261305 | 538435 | -9.732162349591096 |
| 20261306 | 773739 | 7.61336425587853 |

Each independent NumPy `default_rng(seed)` first draws `integers(800000)`,
then its full `standard_normal(steps)` noise vector. The three start values
are archived constants, so the original 800,000-point historical dataset is
not required. Retaining the integer draw preserves the random stream. Batch
shape, precision, and numerical library versions can affect a long recurrent
trajectory; identical seeds alone do not guarantee identical paths on other
machines. The supplied arrays preserve the original results.

`rollout_metrics.json` reports the one-dimensional Wasserstein-1 distance to
all 2,000,000 training observations, mean, population standard deviation,
fraction with `|x| < 1`, and differences between retained temporal halves.
The averages describe variation across three seeds; they are not confidence
intervals and do not establish an optimal checkpoint.

## Validation transition diagnostic

The default also reads the 100,000-point `data/validation_x.npy` sequence.
For each adjacent pair it holds the observed input fixed and draws 32 latent
samples with NumPy seed 2026100731. It reports output and residual W1 and
180-by-180 histogram total variation, and saves `validation_histograms.npz`.
This is an observed-input transition diagnostic, distinct from the autonomous
2-million-point joint histograms supplied in `results/joint_histograms.npz`.
The validation set was monitored during training; it is not an untouched test
set. Serial dependence and finite samples limit statistical interpretation.

During packaging, the complete three 1,100,000-step trajectories reproduced
the archived arrays bit for bit (maximum absolute difference: zero). All
validation histogram edges and counts also matched exactly. The separate
historical validation run used two BLAS threads; W1 reductions can therefore
differ in their last floating-point digits under this script's single-thread
setting. The supplied results remain the original archived values.
