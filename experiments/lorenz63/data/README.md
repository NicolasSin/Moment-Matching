# Scalar Lorenz-63 observations

The arrays contain the $x$ coordinate of the Lorenz-63 system

$$
\dot x=10(y-x),\qquad
\dot y=x(28-z)-y,\qquad
\dot z=xy-\frac83z.
$$

Both files are one-dimensional NumPy float64 arrays sampled at $\Delta t=0.01$. Training and transition diagnostics use adjacent pairs `(x[:-1], x[1:])`.

| File | Observations | Adjacent pairs |
| --- | ---: | ---: |
| `train_x.npy` | 2,000,000 | 1,999,999 |
| `validation_x.npy` | 100,000 | 99,999 |

## Training trajectory

The initial condition is $(x,y,z)=(2,3,4)$. Integration uses DOP853 with relative tolerance $10^{-13}$, absolute tolerance $10^{-15}$, maximum step $0.01$ and integration segments of length 100 time units.

After discarding 100,000 sampled observations, grid indices 100,000 through 2,099,999 are retained. These correspond to times 1000.00 through 20999.99. The supplied file contains only the retained observations.

## Validation trajectory

The separate validation source starts at $(1,1,1)$ and uses DOP853 with relative tolerance $10^{-12}$, absolute tolerance $10^{-14}$, maximum step $0.01$ and integration segments of length 100 time units.

That source sequence discards 100,000 observations, then assigns the next 800,000 to its original training partition. The following 100,000 observations form the supplied validation array: original grid indices 900,000 through 999,999, corresponding to times 9000.00 through 9999.99. Burn-in is already excluded.

The present 2,000,000-point training array comes from the $(2,3,4)$ trajectory, separately from the 800,000-point partition described above. The validation sequence was monitored during the recorded training experiment; it is not an untouched test set. No test set is included or used by the supplied scripts.

## Interpretation and numerical reproducibility

Different initial conditions do not establish statistical independence, and observations within each trajectory are serially dependent. The generator observes only $x$; it does not reconstruct the full three-dimensional Lorenz state. The scalar observation process is not assumed to be exactly Markov. Matching scalar marginals or adjacent-pair statistics therefore does not guarantee agreement of multistep dynamics.

The supplied arrays define the numerical data reference. Changes in solver version or integration segmentation can change a freshly integrated chaotic trajectory. Data-source records are listed in [result_sources.json](../docs/result_sources.json).

## Historical training monitoring

`training_records.jsonl` contains the scientific monitoring observations through 1,000 Adam and 4,000 L-BFGS updates from the original training run. Its header records the historical source identifier. `scripts/export_training_records.py` exports `loss_history.csv` and `training_endpoint.json` from these bundled records; the original local archive is not a runtime dependency. Export is distinct from running a new optimization or recomputing historical audits.
