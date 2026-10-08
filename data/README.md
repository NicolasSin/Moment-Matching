# Scalar observations

The supplied arrays are the exact observations used by the archived K1500 experiment. Both are one-dimensional NumPy float64 arrays. Training uses adjacent pairs `(x[:-1], x[1:])`.

| File | Points | Adjacent pairs |
| --- | ---: | ---: |
| train_x.npy | 2,000,000 | 1,999,999 |
| validation_x.npy | 100,000 | 99,999 |

The underlying system is the classical Lorenz-63 ODE with sigma=10, rho=28 and beta=8/3. Only its x coordinate is observed by the generator.

The training trajectory starts at (2,3,4), uses DOP853 with rtol=1e-13, atol=1e-15, max_step=0.01 and integration segments of length 100. Observations are sampled every 0.01 time units. After discarding 100,000 observations, indices 100,000 through 2,099,999 are retained, corresponding to times 1000 through 20999.99.

The validation array was inherited from a separate trajectory starting at (1,1,1), sampled every 0.01 time units using DOP853 with rtol=1e-12, atol=1e-14, max_step=0.01 and integration segments of length 100. This historical trajectory first discards 100,000 observations (1,000 time units), then reserves 800,000 observations for its historical training split. Validation is the following 100,000 observations: original grid indices 900,000 through 999,999, or times 9000.00 through 9999.99. The supplied validation array therefore already excludes the burn-in; do not discard its first 100,000 points again.

The present two-million-point training file comes from the separate (2,3,4) trajectory described above; it does not use that historical 800,000-point training split. Different initial conditions do not establish statistical independence. No test set is included or used. Data-source records are listed in [result_sources.json](../docs/result_sources.json).

Use the supplied files for reproducing this exact experiment. Freshly integrating a chaotic ODE with a different solver version or segmentation can change the observation array. The generator does not reconstruct the full three-dimensional Lorenz state, and the scalar observation process is not assumed to be exactly Markov.
