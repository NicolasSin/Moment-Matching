# Training protocol

The scalar conditional generator uses $K=1500$ random Fourier features per coordinate. The supplied model, `models/k1500_gh4000.pt`, corresponds to 1,000 Adam updates followed by 4,000 accepted L-BFGS updates.

## Model and moment objective

For scalar observations $x_n$ at intervals $\Delta t=0.01$, the model is

$$
\widehat X_{n+1}=G_\theta((\widehat X_n-m_x)/s_x,\varepsilon_n),
\qquad \varepsilon_n\overset{\mathrm{iid}}{\sim}\mathcal N(0,1).
$$

Here $G_\theta$ is the network; its implementation accepts the unstandardized state and performs the transformation internally. The network has dimensions $2\to96\to96\to1$, GELU hidden activations and a linear output in original $x$ units, with no residual connection. Only the state input is standardized. Layer weights are initialized uniformly on intervals with half-widths $0.5$, $0.2$ and $1.0$; all bias intervals have half-width $0.05$.

The source normalization values are $m_x=-0.06194128671694436$ and $s_x=7.924799010967371$. Model buffers are initially rounded to float32, then promoted with the network to float64 for L-BFGS. Checkpoint loading preserves those working values.

The coordinate feature banks are independent:

$$
\phi_i(x)=\cos(w_i x+b_i),\qquad
\psi_j(y)=\cos(v_j y+c_j),\qquad
w_i,v_j\sim\mathcal N(0,\gamma^{-2}),\quad b_i,c_j\sim U[0,2\pi).
$$

Features are evaluated in original coordinates. The fixed bandwidth $\gamma=0.9954746006464024$ is the median absolute distance over distinct-index pairs of standardized training inputs; this numerical value is used directly in the original-coordinate features. Frequency standard deviation is $1/\gamma$.

With $N=1{,}999{,}999$ adjacent pairs, the empirical target and noise-integrated model moment are

$$
T_{ij}=\frac1N\sum_{n=0}^{N-1}\phi_i(x_n)\psi_j(x_{n+1}),\qquad
M_{ij}(\theta)=\frac1N\sum_{n=0}^{N-1}\phi_i(x_n)
\mathbb E_\varepsilon[\psi_j(G_\theta((x_n-m_x)/s_x,\varepsilon))].
$$

The numerical model moment $\widehat M$ is estimated by Monte Carlo or quadrature, as specified below. The loss includes all $K^2$ entries:

$$
L(\theta)=\frac1{K^2}\|\widehat M(\theta)-T\|_F^2.
$$

The target uses every adjacent pair. The two-pass gradient calculation replays the same samples or quadrature nodes. A finite Monte Carlo squared loss is not an unbiased estimator of the squared, exactly integrated moment discrepancy.

## Optimization settings

| Setting | Adam | L-BFGS |
| --- | --- | --- |
| Updates | 1,000 | 4,000 accepted |
| Model and feature precision | float32 | float64 |
| Moment accumulation and loss | float64 | float64 |
| Input integration | 100,000 states drawn without replacement per update | 586 positive empirical-input quadrature nodes |
| Latent integration | 50 independent Gaussian draws per input | 128-point Gauss–Hermite quadrature |
| Input chunk size | 512 | 64 |

Adam uses a cosine learning-rate schedule from $10^{-3}$ to $10^{-5}$, $\beta=(0.9,0.999)$, $\epsilon=10^{-12}$, zero weight decay and gradient-norm clipping at 10. Data-target chunks contain 4,096 pairs.

The empirical-input quadrature is constructed from all $N$ input states, with bin width $s_x/40$ and at most three local nodes per occupied bin. Its weights sum to one. The latent rule in `code/gh_objective.py` integrates the standard normal distribution; its weights also sum to one.

L-BFGS uses learning rate 1, `max_iter=1`, `max_eval=25`, history size 100, zero gradient/change tolerances and a strong-Wolfe line search. A factor of $10^6$ scales only the optimizer's loss and gradient; recorded scientific losses are unscaled. Updates require finite loss, nonzero parameter movement and loss no greater than the preceding value plus $10^{-18}$. At most three optimizer-history restarts are allowed.

| Random stream | Seed |
| --- | ---: |
| Network initialization | 3888361926 |
| Input-coordinate features | 3149125487 |
| Output-coordinate features | 1524931226 |
| Adam input sampling | 2513425825 |
| Adam latent noise | 1447829901 |

## Execution

From the repository root, after installing the dependencies:

```console
python scripts/train.py --output outputs/retrain_01 --device cuda
```

Defaults are `configs/experiment.json` and `data/train_x.npy`. The output directory must be new. Successful completion writes `model_final.pt` and `summary.json`. An interrupted run after initialization writes `model_partial.pt` and an incomplete summary. The default cooperative time limit is eight hours.

The entry point uses the fixed update counts above and reads no validation or test observations. The supplied loss history includes validation monitoring from the recorded experiment; that monitored sequence is not an untouched test set.

## Numerical environment

The reference environment is Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1 and PyTorch 2.13.0+cu130 on an NVIDIA GeForce RTX 5070 Ti Laptop GPU. Training sets numerical-library thread limits to four before importing the libraries and disables TF32.

GPU training requires a CUDA-enabled PyTorch installation compatible with the
hardware. `requirements.txt` fixes the PyTorch version but does not specify the
CUDA wheel source. The recorded installation uses the CUDA 13.0 build; supported
installation options are listed in the [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

`--device cpu` is supported, but CPU and CUDA random streams differ. Hardware, library versions, precision and reduction order can affect the optimization trajectory. The supplied checkpoint and arrays define the recorded results; the tested numerical scope is described in [Numerical consistency and reproducibility](VERIFICATION.md).

## Historical monitoring and fixed-step replay

The supplied configuration replays the optimization path to the archived 4,000-update model. The original trainer supported adaptive monitoring, validation-based selection and quadrature audits; the current fixed-step entry point does not reproduce that monitoring toolchain. Through update 4,000, actual training used 128 noise nodes and 586 input nodes at input-refinement level zero. Monitoring, rather than training, progressed to 256 and 512 noise nodes. The stored best-validation observation at update 3,760 is distinct from the supplied update-4,000 weights.

`data/training_records.jsonl` preserves the scientific monitoring values needed to export the archived loss curve and endpoint with `scripts/export_training_records.py`. The historical training audit at update zero is not a numerical error bound for the final model; each audit retains its own recorded update and interpretation. The current training command writes its final model and summary, not a new copy of the historical monitoring records.
