# Evaluation and figures

Evaluation separates autonomous model trajectories from transitions conditioned on observed inputs. Run commands from the experiment directory containing `README.md` and `scripts/`. In the manuscript repository, first run `cd experiments/lorenz63`. Archived results are in `results/`; new evaluations and figures default to `outputs/evaluation/` and `outputs/figures/`.

## Autonomous trajectories

```console
python scripts/evaluate.py
```

The default loads `models/k1500_gh4000.pt`, generates 1,100,000 steps for each of three seeds and discards the first 100,000. The initial state is not an output sample. `rollout_full.npy` includes burn-in; `rollout_samples.npy` has shape `(3, 1000000)` after burn-in. `rollout_metrics.json` records the protocol and metrics. A nonempty output directory is refused; `--output PATH` selects another directory.

Evaluation uses CPU float64, the saved normalization buffers and one numerical-library/PyTorch compute thread. Outputs are in original $x$ units. The three chains are batched through a traced model and a scripted sequential loop in blocks of 25,000 steps, without clipping or restarting.

| Seed | Archived start index | Initial $x$ |
| --- | ---: | ---: |
| 20261304 | 95844 | -14.598013133703434 |
| 20261305 | 538435 | -9.732162349591096 |
| 20261306 | 773739 | 7.61336425587853 |

Each independent NumPy `default_rng(seed)` draws `integers(800000)` before its `standard_normal(steps)` vector. Initial values are stored constants; the original 800,000-point sequence is not required. The retained integer draw fixes the noise-stream position. Identical seeds alone do not guarantee identical recurrent trajectories across numerical environments.

Metrics include one-dimensional Wasserstein-1 distance to the full 2,000,000-point training reference, mean, population standard deviation, the fraction with $|x|<1$ and comparisons of the retained temporal halves. Metrics are calculated per chain and then averaged; three-seed variation is descriptive, not a confidence interval.

## Autonomous joint distribution

```console
python scripts/analyze_joint.py --output outputs/joint
python scripts/plot_results.py --results outputs/joint --output outputs/joint_figures
```

The joint analysis uses the prespecified NumPy seed **2026101001**, distinct from the three marginal/ACF seeds. The generator first draws `integers(len(train_x))` to select its initial state from the bundled training array, then draws 2,100,000 standard normal innovations. CPU float64 inference uses one numerical-library/PyTorch thread, batch size one and blocks of 25,000 steps. Discarding the first 100,000 generated states leaves 2,000,000 states and 1,999,999 adjacent pairs. The initial state is not counted as an output. The selected training index is 1276103 and its state is 6.260278459735426.

This replaces the former seed-20261304 joint record, whose first million retained states coincided with marginal chain 0. The new seed was fixed before evaluating its results. The new trajectory uses a distinct model-noise stream; both analyses still use the same trained model and training reference, so this is not an independent held-out validation of training.

The script writes `joint_rollout_2m.npy`, `joint_histograms.npz` and `joint_metrics.json`. It also retains `rollout_full.npy`, including burn-in, in the new output directory. Metadata records the actual seed, initial state/index, retained counts, histogram edges and software versions. `--steps`, `--burn-in`, `--seed` and `--checkpoint` can select a different experiment; the archived results use the defaults above.

For each of 120, 240 and 480 bins per coordinate, both trajectories use equally spaced edges spanning the combined minimum and maximum of the training and generated states, without padding or smoothing. The final bin includes its right endpoint. All adjacent pairs are included, and each histogram is divided by its own number of pairs. Plots show probability mass per bin, not density. TV, Jensen–Shannon divergence in natural-log units, Hellinger distance and cell-probability RMSE are recorded, together with state/increment W1 and a 24-direction sliced W1. The latter is a finite projection diagnostic, not exact two-dimensional W1.

To recalculate statistics directly from the bundled trajectory, without generating it again:

```console
python scripts/analyze_joint.py --trajectory results/joint_rollout_2m.npy --output outputs/joint_reanalysis
```

Existing-trajectory mode treats the input as already retained; it does not infer a seed, initial state or burn-in from array contents. The original protocol remains recorded in `results/joint_metrics.json`. Numerical environment changes can change recurrent trajectories even with the same seed.

## Observed-input transition diagnostic

The default also reads all 100,000 points of `data/validation_x.npy`. Each of its 99,999 observed inputs is held fixed while 32 independent latent draws are generated using NumPy seed 2026100731. Diagnostics include next-state and residual Wasserstein-1 distances and 180-by-180 histogram total variation; `validation_histograms.npz` stores the histograms.

This diagnostic differs from autonomous rollout evaluation: generated outputs are not fed back as the next inputs. The validation sequence was monitored during training and is not an untouched test set. `--validation-points N` selects a prefix; `--validation-points 0` disables this diagnostic.

The complete observed-input analysis is also available as a standalone command:

```console
python scripts/analyze_validation.py --output outputs/validation
python scripts/plot_results.py --results outputs/validation --output outputs/validation_figures
```

This writes `one_step_metrics.json`, `conditional_metrics.csv` and `validation_histograms.npz`, including the original conditional-bin, sliced-W1 and independent-feature diagnostics. It uses the original two-thread numerical-library/PyTorch setting and the same fixed diagnostic seeds. `plot_results.py` draws `joint_residual_preview.png` and `.pdf` from these histograms. These figures condition on observed inputs and are distinct from the autonomous joint figures.

One-dimensional W1 always uses all supplied samples with uniform mass within each empirical distribution, in original state units. It is calculated by exact empirical CDF integration (`scipy.stats.wasserstein_distance`), up to floating-point rounding; no fixed quantile-grid approximation is used.

A short CPU execution exercises both paths without reproducing long-run statistics:

```console
python scripts/evaluate.py --steps 1000 --burn-in 100 --validation-points 1000 --output outputs/quick
python scripts/plot_results.py --results outputs/quick --output outputs/quick_figures
```

## Figures from saved results

```console
python scripts/plot_results.py
```

This reads saved arrays without training or generating new trajectories. `--results PATH` selects the result directory and `--output PATH` selects the figure directory. `--joint-bins 120`, `240` or `480` selects an archived joint grid; the default is 120.

PNG and PDF outputs are `loss_history`, `rollout_distribution`, `joint_probability`, `joint_probability_surface`, `autocorrelation`, `autocorrelation_zoom` and `joint_residual_preview`, when their input files are available. The autocorrelation calculation also writes `autocorrelation.csv` with lag, time lag, reference, individual seeds, mean and population standard deviation.

## Historical training records

```console
python scripts/export_training_records.py --output outputs/training_records
python scripts/plot_results.py --results outputs/training_records --output outputs/training_figures
```

The bundled `data/training_records.jsonl` contains the original scientific monitoring records through 1,000 Adam and 4,000 L-BFGS updates. Export reconstructs `loss_history.csv` and `training_endpoint.json`. It does not rerun training, recreate elapsed times, or recompute the historical quadrature audits. A fresh run of `train.py` produces the fixed-step model and summary described in [Training](TRAINING.md), rather than the original adaptive monitoring log.

All required analysis inputs are bundled. Historical source identifiers in [result_sources.json](result_sources.json) are provenance notes, not paths that another machine must have. Its `reproduction` entries identify executable commands and their repository-relative inputs.

## Autocorrelation convention

For each trajectory of length $n$, its own sample mean is subtracted and

$$
\rho(k)=\frac{n}{n-k}
\frac{\sum_{t=0}^{n-k-1}(x_t-\bar x)(x_{t+k}-\bar x)}
     {\sum_{t=0}^{n-1}(x_t-\bar x)^2},
\qquad 0\le k<n.
$$

This is lag-adjusted normalization, not a claim of an unbiased population estimator. Float64 FFTs are zero-padded to prevent circular wrap-around. Saved trajectories already exclude burn-in. Each chain is processed separately, with no concatenation across endpoints; reference and model retain their respective lengths.

With $\Delta t=0.01$, the main panels cover 500 steps (5 time units) and 10,000 steps (100 time units). `--acf-max-lag N` changes the main-model maximum lag, capped at trajectory length minus one. The zoom covers times 0–0.1 and 0–0.01, containing 11 and two sampled lags respectively. Markers identify these lags; connecting lines do not estimate additional observations.

The main figure shows the reference, three individual model curves and their arithmetic mean. Seed standard deviations and shaded bands describe variation across three trajectories; they are not confidence intervals.

Agreement of marginal or adjacent-pair distributions does not establish agreement of multistep dynamics or exact stationarity; the scalar observation process is not assumed to be exactly Markov.

Numerical comparisons with the archived evaluations are documented in [Numerical consistency and reproducibility](VERIFICATION.md).
