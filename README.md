# multiple-try-metropolis-experiments
Code accompaniment to reproduce the experiment presented in "A unifed framework for multiple-try Metropolis algorithms"

## Experiment Design



## Usage notes

The experiment scripts are located in the `experiments/` folder. To run 

`mpirun -np n julia <experiment>.jl`

The weight updates can be performed in a multi-threaded regime. To use multiple threads, execute update the `JULIA_NUM_THREADS` variable in the Preferences.json file.
