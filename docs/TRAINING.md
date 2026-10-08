# Training the K=1500, GH4000 example

The supplied trained model is the original saved GH4000 checkpoint. Reusing it
does not require training. This script provides a readable version of the
optimization that produced that checkpoint: 1,000 Adam updates followed by
4,000 accepted L-BFGS updates. Here **GH4000 means 4,000 optimization updates**;
the Gaussian quadrature used by those updates has **128 nodes**.

## Run

Install the dependencies described in the README, then run from the repository
directory:

```console
python scripts/train.py --output outputs/retrain_01 --device cuda
```

The script automatically reads `configs/experiment.json` and
`data/train_x.npy`. It stays quiet while training. Choose a new output directory:
existing directories and destinations inside packaged code, data, models, or
results are refused. On success it writes only:

- `model_final.pt`: the last model, after the fixed update counts.
- `summary.json`: completed update counts, final training loss, and elapsed time.

If interrupted after model initialization, it saves `model_partial.pt` and an
incomplete summary. That partial model is not the GH4000 result. The simplified
script deliberately has no resume mechanism, validation selection, live plots,
progress messages, or intermediate checkpoint collection.

For the original CUDA 13.0 PyTorch build on a compatible Windows/Linux machine,
install PyTorch before the remaining pinned dependencies:

```console
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
```

This follows the [official PyTorch versioned installation instructions](https://pytorch.org/get-started/previous-versions/).
For CPU-only evaluation or another supported CUDA build, use the
[official installation selector](https://pytorch.org/get-started/locally/).

The default configuration has an eight-hour cooperative time limit. A numerical
operation already in progress may finish after the limit before the next check.
The script raises an error if the limit is reached before completion.

## What is preserved

The network is the original scalar conditional generator: two inputs, hidden
layers of width 96 with GELU activations, and one unrestricted output. Only the
state input is always standardized; there is no input-mode switch. Gaussian noise
has unit variance; random Fourier features are evaluated on original state coordinates.

The Gaussian frequencies have variance `gamma**-2`. NumPy's `normal` accepts a
standard deviation, so the code uses `scale=1/gamma`. The fixed value
`gamma=0.9954746006464024` was obtained as the median absolute pair distance of
the standardized training inputs, then used directly in the original-coordinate
features, following the historical experiment. See the README for this convention.

The supplied configuration retains the original initialization, independent
random seeds, 1,500 cosine features per coordinate, and full adjacent-pair data
targets. Adam draws 100,000 training inputs without replacement and 50 Gaussian
noise values per input on each update. Its float32 network calculations,
float64 moment accumulation, cosine learning-rate schedule, gradient clipping,
and two-pass gradient calculation are unchanged.

L-BFGS uses float64, the same strong-Wolfe line search and optimizer settings,
and the same positive empirical-input quadrature constructed from all
1,999,999 training input states. It uses 586 input nodes for the supplied data,
bin width equal to the archived float64 training standard deviation divided by
40, and a 128-node Gaussian rule. The reported scientific loss is the mean of
the squared entries of the full 1,500-by-1,500 moment discrepancy matrix; the
optimizer-only scale of one million does not change that reported loss.

Normalization buffers are initially rounded to float32 and later promoted to
float64 with the network. Replacing them with freshly computed float64
statistics would change the model. The loader preserves the archived values.

## What was removed

The historical run included diagnostic validation, adaptive diagnostic
quadrature, recovery machinery, file-access guards, preflight receipts, and
repeated plots and checkpoints. These are absent here. Its records show that
through GH4000 the **training** rule stayed at Gaussian order 128 and input
refinement level zero. Changes to the diagnostic rule did not change training
parameters or L-BFGS history. Validation noise used a separate unused random
stream. No validation or test observations are read by this training script.

`objective.py`, `gh_objective.py`, and the network/checkpoint implementation
retain the original numerical operations. `empirical_quadrature.py` removes
per-bin diagnostic bookkeeping while retaining the same arithmetic. The entry
point sets the six numerical-library thread limits to four before importing
those libraries, as the original launcher did; this matters even for the final
bits of the input quadrature.

## Resources and reproducibility

The original environment was Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1, and
PyTorch 2.13.0+cu130 on an NVIDIA GeForce RTX 5070 Ti Laptop GPU. The archived
update timings total approximately 33 minutes for Adam and 27 minutes for the
first 4,000 GH updates, excluding target construction and diagnostics. This is
a substantial training run, not a quick demonstration. Memory use is dominated
by feature evaluations and the full moment matrices, not the small network.

`--device cpu` is supported for portability but is much slower. CPU and CUDA
random-number streams differ, so a CPU retrain does not reproduce the archived
optimization trajectory. GPU model, library versions, precision settings, and
numerical-library thread counts can also affect exact reproducibility. The
supplied archived checkpoint is the reference for the shared results.

The first simplified implementation passed both the local numerical comparisons and a complete fresh 1000-Adam/4000-GH retrain in the original CUDA environment. Its final saved model tensors are exactly equal to the historical GH4000 checkpoint. The subsequent removal of file fingerprints and the input-mode switch has a separate comparison record. See [verification details](VERIFICATION.md) for the tested range and remaining portability limits.
