# Project: A unified framework for multiple-try Metropolis algorithms
# Purpose: Experiment targeting a gaussian mixture distribution
# Author:  Renny Doig and Liangliang Wang
# Date:    January 12, 2025

# Preliminaries ----------------------------------------------------------------

using MPI, MPIPreferences, Distributions, Random, CSV, DataFrames, StatsBase, LinearAlgebra

# initialize the multiple processes and store process information
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
n_procs = MPI.Comm_size(comm)

# load in functions
include("../functions/utils.jl")
include("../functions/homogeneous_full.jl")
include("../functions/heterogeneous_full.jl")
include("../functions/homogeneous_cw.jl")
include("../functions/heterogeneous_cw.jl")


# Distributions ----------------------------------------------------------------

# log-posterior: taken from https://github.com/Julia-Tempering/Pigeons.jl/blob/main/examples/stan/funnel.stan
# sampled values = [y,x]
# fixed values = [dimension, scale]
# y ~ Normal(0,3)
# x ~ MvNormal(0, exp(y/scale)) - dimension from fixed values
function log_target(y, param::Vector)
	return sum(logpdf.(Cauchy(param[1], exp(param[2])), y))
end

dimension = 2
data = [0.9, 1.2, 1.21]

# initialization function
function init()
	return rand(Normal(0,1), 2)
end


# Simulation settings ----------------------------------------------------------

# total run-time for each run
t_max = 20.0

# number of runs for each setting
reps = 50

# weight functions
weight_strs = ["importance", "proportional", "jump distance", "locally balanced"]
# number of candidates
M_vec = [1, 10, 20, 30, 40, 50]

# manage distribution of simulations across the worker processes
runs_per_process = div(reps, n_procs) + (rank < rem(reps, n_procs) ? 1 : 0)
local_run_ids = collect((rank * (reps / n_procs) + 1):(rank * (reps / n_procs) + runs_per_process))
setting_cols = ["rep", "weight", "M", "method"]
result_cols = ["n_total", "n_actual", "x0", "y"]
local_results = DataFrame([Vector{Any}() for _ in 1:8], [setting_cols; result_cols])

for (i, w_str) in enumerate(weight_strs)
	if rank == 0
		println("Weight function: $w_str")
	end
		
	for (j, M) in enumerate(M_vec)
		if rank == 0
			println(" M: $M")
		end
		if rank == 0
			println("   Homogeneous-Full")
		end
		base_seed = 1 + (j-1)*4*reps + (i-1)*length(M_vec)*4*reps

		for id in local_run_ids
			results = homogeneous_full(t_max, M, data, log_target, init, w_str, Int(base_seed+id))
			iters = results.n_iters
			burnin = get_burnin(results.X, iters)
			n0 = Int(iters - burnin)
			means = mean.(eachcol(results.X[n0:end,:]))
			push!(local_results, [id; w_str; M; 1; iters; burnin; means])
			GC.gc()
		end

		if rank == 0
			println("   Heterogeneous-Full")
		end
		base_seed = 1 + reps + (j-1)*4*reps + (i-1)*length(M_vec)*4*reps

		for id in local_run_ids
			results = heterogeneous_full(t_max, M, data, log_target, init, w_str, Int(base_seed+id))
			iters = results.n_iters
			burnin = get_burnin(results.X, iters)
			n0 = Int(iters - burnin)
			means = mean.(eachcol(results.X[n0:end,:]))
			push!(local_results, [id; w_str; M; 2; iters; burnin; means])
			GC.gc()
		end

		if rank == 0
			println("   Homogeneous-CW")
		end
		base_seed = 1 + 2*reps + (j-1)*4*reps + (i-1)*length(M_vec)*4*reps

		for id in local_run_ids
			results = homogeneous_cw(t_max, M, data, log_target, init, w_str, Int(base_seed+id))
			iters = results.n_iters
			burnin = get_burnin(results.X, iters)
			n0 = Int(iters - burnin)
			means = mean.(eachcol(results.X[n0:end,:]))
			push!(local_results, [id; w_str; M; 3; iters; burnin; means])
			GC.gc()
		end

		# METHOD 4
		if M==1
			if rank == 0
				println("    -> skipping for M=1")
			end
		else
			if rank == 0
				println("   Heterogeneous-CW")
			end
			proposal_variance0 = [collect(range(-5,5, length=M)) for _ in 1:dimension]
			base_seed = 1 + (j-1)*4*reps + (i-1)*length(M_vec)*4*reps

			for id in local_run_ids
				results = heterogeneous_cw(t_max, M, data, log_target, proposal_variance0, init, w_str, Int(base_seed+id))
				iters = results.n_iters
				burnin = get_burnin(results.X, iters)
				n0 = Int(iters - burnin)
				means = mean.(eachcol(results.X[n0:end,:]))
				push!(local_results, [id; w_str; M; 4; iters; burnin; means])
				GC.gc()
			end
		end
	end
end
println("Saving results from processor $rank to CSV.")
CSV.write("results_lighthouse/proc$(rank).csv", local_results)
MPI.Barrier(comm)