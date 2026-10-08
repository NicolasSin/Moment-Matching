# Scalar observations

The supplied arrays are the exact observations used by the archived K1500 experiment. Both are one-dimensional NumPy float64 arrays. Training uses adjacent pairs `(x[:-1], x[1:])`.

| File | Points | Adjacent pairs | SHA256 |
| --- | ---: | ---: | --- |
| train_x.npy | 2,000,000 | 1,999,999 | f79d22afa44a4db13cf9cd4c626cc5d6583fedd5cbe72726b0f2dcd5b40e0620 |
| validation_x.npy | 100,000 | 99,999 | 626c6a21c87b3688d76cacc298e8ef0dcd085d18784bd79d9d80637fa2c206bb |

The underlying system is the classical Lorenz-63 ODE with sigma=10, rho=28 and beta=8/3. Only its x coordinate is observed by the generator.

The training trajectory starts at (2,3,4), uses DOP853 with rtol=1e-13, atol=1e-15, max_step=0.01 and integration segments of length 100. Observations are sampled every 0.01 time units. After discarding 100,000 observations, indices 100,000 through 2,099,999 are retained, corresponding to times 1000 through 20999.99.

The validation array was inherited from a separate (1,1,1) trajectory. Different initial conditions do not establish statistical independence. No test set is included or used.

Use the supplied files for reproducing this exact experiment. Freshly integrating a chaotic ODE with a different solver version or segmentation can change the observation array. The generator does not reconstruct the full three-dimensional Lorenz state, and the scalar observation process is not assumed to be exactly Markov.
