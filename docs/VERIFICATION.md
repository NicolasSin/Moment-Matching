# Verification of the simplified version

Prepared on 8 October 2026. The original research files were not edited.

## Preserved artifacts

The selected checkpoint, training and validation observations, the three retained rollout arrays, and the separate two-million-state rollout were copied byte for byte. Their source paths and SHA256 checksums are recorded in `result_sources.json`.

Reference checkpoint SHA256:

`c5f28b103c53a0b494126a7e46a754adc1bd1f2aba6eaf8d1c8aa5c92a07422c`

Reference model tensor hash:

`bedf9b1bf335b14def0decdbb90ed5e766a94fc9a02dd3fa79d19457bc1530af`

The original checkpoint contains historical optimizer state and metadata, including an unused original machine path. The portable scripts do not use that path. The checkpoint was kept intact so its identity can be verified against the archive.

## Numerical code comparisons

The original and simplified code passed 84 bounded CUDA comparisons with exact equality and maximum absolute difference zero. These covered initialization, feature draws, forward evaluation, Monte Carlo loss and gradients in float32 and float64, one Adam update and its random states, GH loss and gradients, one L-BFGS update, and inference from the archived GH4000 checkpoint.

The bounded fixtures used 23 features, 32 Monte Carlo input states with seven noise draws each, and a 12-state/12-node GH calculation. Separately, all 586 input quadrature nodes and weights reconstructed from the complete training input array matched the archive exactly. Matching the original numerical-library thread limits was necessary for this last comparison.

## Full evaluation of the saved checkpoint

The simplified evaluator completed all three 1,100,000-step CPU trajectories in approximately 117 seconds. The entire generated arrays and the retained three-by-1,000,000 arrays were exactly equal to the original outputs. Maximum absolute difference: **0**.

The retained `.npy` file SHA256 is:

`cd0145d7764c1fe10d960d3f040b04c77664173091b4bd8a919cab336c43ca01`

All per-seed rollout metrics and their reported averages matched after retaining the original one-thread numerical-library setting. This setting was applied after the full trajectory run and checked by recomputing metrics from the completed arrays; the trajectories were not rerun solely for that setting.

The full validation histogram counts and edges matched. Archived validation predictions were also compared at three batches and matched exactly. Validation W1 differed by at most **2.43e-17**, because the original validation calculation used two BLAS threads while the compact evaluator uses one. No additional dependency or process machinery was added solely to reproduce those last reduction bits.

## Complete retraining

The simplified script completed all 1000 Adam and 4000 GH updates in 87.99 minutes on the original CUDA environment. All 8 saved parameter/buffer tensors are exactly equal to the selected historical checkpoint, with maximum absolute difference **0**. The final model tensor hash is the reference hash above. The terminal training loss is `1.0812081201689768e-08`.

This is a complete from-initialization reproduction in the tested environment, not merely a comparison of the supplied checkpoint. The newly saved checkpoint has different metadata and serialization, so whole-file identity is not required; tensor identity is the relevant check.

## Scope

The training entry point and its local numerical dependencies decreased from 1,733 to 758 physical lines, including comments and blank lines. This comparison covers the original `train.py`, `common.py`, `model.py`, `objective.py`, `gh_objective.py`, `empirical_quadrature.py`, `validation_monitor.py` and `plot_progress.py`, versus the simplified training entry point and its five core modules. It excludes data-generation, preparation, launching and evaluation scripts. Diagnostic/recovery functionality was removed, so this is not a claim of unchanged functionality in fewer lines.

PNG and PDF figures were redrawn from the saved numerical results and visually inspected. The plots keep the original loss records and joint-bin counts. These checks establish the stated implementation and archive consistency, not exact equality of population distributions or reproducibility on every hardware/software combination.
