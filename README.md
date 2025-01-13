# multiple-try-metropolis-experiments
Code accompaniment to reproduce the experiment presented in "A unifed framework for multiple-try Metropolis algorithms"



## Implementation Details

Four MTM samplers are implemented based on the four proposal distribution configurations outlined in the paper. The Julia scripts for these are located in the `functions/` folder under the following names:

| Source File             | Proposal Form                                      | Adaptation Scheme   |
| ------------------------|----------------------------------------------------|---------------------|
| `homogeneous_full.jl`   | $\mathbf y\sim\mathcal{MVN}(\mathbf x|\Sigma)$     | Adaptive Metropolis |
| `heterogeneous_full.jl` | $\mathbf y_m\sim\mathcal{MVN}(\mathbf x|\Sigma_m)$ | Adaptive Metropolis |
| `homogeneous_cw.jl`     | $y_i\sim\mathcal N(x_i,\sigma_i^2)$                | Adaptive Metropolis |
| `heterogeneous_cw.jl`   | $y_{i,m}\sim\mathcal N(x_i,\sigma_{i,m}^2)$        | Balanced Selection  |


- Multi-threading over the candidate proposals and weight evaluations is enabled. Additionally, multiple runs of the experiment are run in parallel using MPI.

- The terminating condition for the algorithm is the running time rather than the number of iterations.


## Experiment Design

Each simulation experiment uses a full factorial design. All experiments vary the proposal configuration, weight function, and number of candidates. The experiments for the banana distribution and Neal's funnel also vary the parameter that controls the non-Gaussianity of the distribution.

- Each experiment setting is run 50 times and results are saved into an experiment-specific directory, `experiments/results_<experiment name>`.



## Usage notes

The experiment scripts are located in the `experiments/` folder. To run 

`mpirun -np n julia <experiment>.jl`

- The weight updates can be performed in a multi-threaded regime. To use multiple threads, execute update the `JULIA_NUM_THREADS` variable in the Preferences.json file.

- Generating plots from the manuscript can be done by running the appropriate R file, `results/<experiment name>.R`
