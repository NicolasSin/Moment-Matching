# Evaluation and figures

Evaluation separates autonomous model trajectories from transitions conditioned on observed inputs. Commands run from the repository root. Archived results are in `results/`; new evaluations and figures default to `outputs/evaluation/` and `outputs/figures/`.

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

The archived `results/joint_histograms.npz` uses a separate continuous trajectory with 2,000,000 retained states after 100,000 burn-in steps. Its 1,999,999 adjacent pairs are binned on common reference/model grids with 120, 240 or 480 bins per coordinate. Joint plots show probability per bin, with counts divided by 1,999,999; they do not show probability density. Surface connections between bin centres are a display operation. These archived grids are not re-binned.

## Observed-input transition diagnostic

The default also reads all 100,000 points of `data/validation_x.npy`. Each of its 99,999 observed inputs is held fixed while 32 independent latent draws are generated using NumPy seed 2026100731. Diagnostics include next-state and residual Wasserstein-1 distances and 180-by-180 histogram total variation; `validation_histograms.npz` stores the histograms.

This diagnostic differs from autonomous rollout evaluation: generated outputs are not fed back as the next inputs. The validation sequence was monitored during training and is not an untouched test set. `--validation-points N` selects a prefix; `--validation-points 0` disables this diagnostic.

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

PNG and PDF outputs are `loss_history`, `rollout_distribution`, `joint_probability`, `joint_probability_surface`, `autocorrelation` and `autocorrelation_zoom`, when their input files are available. The autocorrelation calculation also writes `autocorrelation.csv` with lag, time lag, reference, individual seeds, mean and population standard deviation.

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

## Checkpoint autocorrelation supplement

When `results/checkpoint_scan/autocorrelations.npz` is present, plotting also writes:

- `autocorrelation_checkpoints.png` and `.pdf`: checkpoints after 4,000, 23,000 and 24,000 L-BFGS updates.
- `autocorrelation_checkpoints_zoom.png` and `.pdf`: the same checkpoints at times 0–0.1 and 0–0.01.
- `autocorrelation_all_checkpoints.png` and `.pdf`: all 19 checkpoints with finite autonomous trajectories.

These saved curves use three trajectories of 2,000,000 retained states per checkpoint, distinct from the main model's three 1,000,000-state trajectories. Their stored lag range is independent of `--acf-max-lag`. Checkpoints after 9,000 and 11,000 updates are excluded because autonomous trajectories became nonfinite; no surviving-seed or truncated-chain substitute is used.

Selection of the 23,000-update checkpoint was based on marginal and adjacent-pair distributions. Autocorrelation is a separate temporal diagnostic and does not alter that ranking. Agreement of marginal or adjacent-pair distributions does not establish agreement of multistep dynamics or exact stationarity; the scalar observation process is not assumed to be exactly Markov.

Numerical comparisons with the archived evaluations are documented in [Numerical consistency and reproducibility](VERIFICATION.md).
