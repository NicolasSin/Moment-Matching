# Verification of the simplified version

Prepared on 8 October 2026. The original research files were not edited.

## Preserved artifacts

The selected checkpoint, training and validation observations, the three retained rollout arrays, and the separate two-million-state rollout were copied byte for byte. Their source paths are recorded in `result_sources.json`.

The original checkpoint contains historical optimizer state and metadata, including an unused original machine path. The portable scripts read its saved model tensors; they do not depend on those historical metadata fields. The checkpoint remains unchanged.

## Numerical code comparisons

The original and simplified code passed 84 bounded CUDA comparisons with exact equality and maximum absolute difference zero. These covered initialization, feature draws, forward evaluation, Monte Carlo loss and gradients in float32 and float64, one Adam update and its random states, GH loss and gradients, one L-BFGS update, and inference from the archived GH4000 checkpoint.

The bounded fixtures used 23 features, 32 Monte Carlo input states with seven noise draws each, and a 12-state/12-node GH calculation. Separately, all 586 input quadrature nodes and weights reconstructed from the complete training input array matched the archive exactly. Matching the original numerical-library thread limits was necessary for this last comparison.

## Full evaluation of the saved checkpoint

The simplified evaluator completed all three 1,100,000-step CPU trajectories in approximately 117 seconds. The entire generated arrays and the retained three-by-1,000,000 arrays were exactly equal to the original outputs. Maximum absolute difference: **0**.

All per-seed rollout metrics and their reported averages matched after retaining the original one-thread numerical-library setting. This setting was applied after the full trajectory run and checked by recomputing metrics from the completed arrays; the trajectories were not rerun solely for that setting.

The full validation histogram counts and edges matched. Archived validation predictions were also compared at three batches and matched exactly. Validation W1 differed by at most **2.43e-17**, because the original validation calculation used two BLAS threads while the compact evaluator uses one. No additional dependency or process machinery was added solely to reproduce those last reduction bits.

## Complete retraining

The first simplified script completed all 1000 Adam and 4000 GH updates in 87.99 minutes on the original CUDA environment. All 8 saved parameter/buffer tensors are exactly equal to the selected historical checkpoint, with maximum absolute difference **0**. The terminal training loss is `1.0812081201689768e-08`.

This is a complete from-initialization reproduction in the tested environment, not merely a comparison of the supplied checkpoint. The newly saved checkpoint has different metadata and serialization, so whole-file identity is not required; tensor identity is the relevant check.

## Subsequent cleanup on 8 October 2026

The current version removes file/model fingerprint calculations, checks and
metadata fields, and fixes the model interface to standardized inputs. It also
spells out NumPy's frequency standard-deviation argument without changing the
sampled frequencies. The supplied checkpoint and numerical arrays are unchanged.

Separate CPU and CUDA comparisons against the first simplified version passed
with exact equality and maximum absolute difference **0**. They covered all eight
initial parameter/buffer tensors in float32 and float64, forward predictions,
all 1,500 frequencies/phases per coordinate, bounded Monte Carlo losses and
gradients, loading the original checkpoint, and checkpoint save/load round trips.
Three consecutive Adam updates followed by three GH/L-BFGS updates also matched
all model tensors, optimizer state and random streams. The bounded loss fixtures
used 23 features and seven noise samples per input; the GH fixture used 12 input
states and a 12-node Gaussian rule. The empirical-input quadrature source was
unchanged.

These comparisons ran on CPU and the original NVIDIA GeForce RTX 5070 Ti Laptop
GPU with four numerical-library threads in the original Python/NumPy/SciPy/PyTorch
environment. The checks live outside the shared training package. This cleanup
was not followed by another full 1000-Adam/4000-GH retrain; the full-retraining
record above applies to the first simplified version.

The current training entry point also completed a short end-to-end CPU run using
the first 97 supplied training observations, 23 features, seven noise samples,
12 Gaussian nodes and three updates in each stage. It wrote a final model and
completion summary, loaded successfully, and produced no stdout or stderr.

The validation array was compared directly to the original historical validation
array: all 100,000 values were equal, maximum absolute difference **0**. Its
generation record gives a 100,000-point burn-in and validation grid indices
900,000 through 999,999. Details are in [the data notes](../data/README.md).

The current training entry point and its five core modules have 702 physical
lines, down from 758 in the first simplified version.

## Scope

The first cleanup reduced the training entry point and its local numerical dependencies from 1,733 to 758 physical lines, including comments and blank lines. This comparison covers the original `train.py`, `common.py`, `model.py`, `objective.py`, `gh_objective.py`, `empirical_quadrature.py`, `validation_monitor.py` and `plot_progress.py`, versus the first simplified training entry point and its five core modules. It excludes data-generation, preparation, launching and evaluation scripts. Diagnostic/recovery functionality was removed, so this is not a claim of unchanged functionality in fewer lines.

PNG and PDF figures were redrawn from the saved numerical results and visually inspected. The plots keep the original loss records and joint-bin counts. These checks establish the stated implementation and archive consistency, not exact equality of population distributions or reproducibility on every hardware/software combination.
