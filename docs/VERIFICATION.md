# Numerical consistency and reproducibility

The checks below establish numerical consistency within the stated settings and tested ranges. Source records for the supplied checkpoint, observations and numerical results are in [result_sources.json](result_sources.json).

## Bounded training comparisons

The current implementation was compared against the reference implementation on CPU and on the NVIDIA GeForce RTX 5070 Ti Laptop GPU. Each backend's paired comparisons had exact tensor equality and maximum absolute difference **0**; this does not assert equality between CPU and CUDA trajectories.

The comparisons covered:

- All eight initialized parameter/buffer tensors and forward predictions in float32 and float64.
- The complete $K=1500$ frequency and phase arrays for both coordinate feature banks.
- Monte Carlo loss and gradients with 23 features, 43 input states and seven latent samples per input.
- Three consecutive Adam updates using a 32-state fixture, followed by three L-BFGS updates using 12 input states and 12-point Gauss–Hermite quadrature.
- Model tensors, optimizer state and random-generator state through the short optimization sequence.
- Loading the supplied checkpoint, inference from it and checkpoint save/load round trips.

The separate empirical-input quadrature check reconstructed all **586 nodes and weights** from the complete training input array with exact equality to the reference calculation. Numerical-library thread limits were four.

A short end-to-end CPU execution of the current training entry point used the first 97 training observations, 23 features, seven latent samples, 12-point Gauss–Hermite quadrature and three updates in each stage. It produced a completed summary and a finite final model that loaded successfully. These bounded checks do not constitute a new full training run.

## Archived full evaluation

The recorded full evaluation of the supplied checkpoint generated all three 1,100,000-step CPU trajectories. Both the complete arrays and the three retained 1,000,000-state arrays matched the archived outputs exactly: maximum absolute difference **0**.

Per-seed rollout metrics and their averages matched with the reference one-thread numerical-library setting. The metrics were also checked by recomputation from the completed arrays.

All observed-input validation histogram edges and counts matched exactly. Predictions compared at three archived batches also matched exactly. Validation Wasserstein-1 differences were at most **$2.43\times10^{-17}$**; the reference validation calculation used two BLAS threads and the evaluator uses one.

The 100,000 supplied validation observations matched the source array exactly. Their burn-in and retained-index conventions are specified in the [data description](../data/README.md).

## Autocorrelation and figures

The FFT autocorrelation implementation was compared with direct summation using the same per-trajectory mean and lag-adjusted normalization:

| Comparison | Tested lags | Maximum absolute difference |
| --- | --- | ---: |
| Seven-point numerical fixture | Every lag, 0–6 | $2.23\times10^{-16}$ |
| Full training reference and three retained model trajectories | 0, 1, 10, 100, 1,000, 10,000 | $1.56\times10^{-15}$ |

The reported bounds round the measured maxima upward. The full-array check used the 2,000,000-point reference and all three 1,000,000-point model trajectories. Each trajectory was processed separately.

PNG and PDF figures were generated from the saved numerical results and visually inspected. The joint plots preserve the recorded bin edges and counts; autocorrelation zooms identify the actual sampled lags.

## Scope and environment

The training comparisons used Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1 and PyTorch 2.13.0+cu130, with four numerical-library threads. Autonomous evaluation used CPU float64 and one numerical-library/PyTorch compute thread.

Exact equality applies to the stated numerical comparisons in the tested environment. These checks do not establish equality of population distributions, statistical independence, exact scalar Markov dynamics or reproducibility on every hardware/software combination.
