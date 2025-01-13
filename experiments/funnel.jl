# Project: A unified framework for multiple-try Metropolis algorithms
# Purpose: Experiment using Neal's funnel as the target
# Author:  Renny Doig and Liangliang Wang
# Date:    January 12, 2025

# Preliminaries ----------------------------------------------------------------

using MPI, MPIPreferences, Distributions, Random, CSV, DataFrames, StatsBase

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
function log_target(fixed, sampled)
	dy = logpdf(Normal(0,3), sampled[1])
	dx = sum(logpdf.(Normal(0,exp(sampled[1]*10^fixed)), sampled[2:end]))
	return dy + dx
end

# initialization distribution
function init()
	return rand(Normal(0,10), 3)
end


# Simulation settings ----------------------------------------------------------

# total run-time for each run
t_max = 10.0

# number of runs for each setting
reps = 50

# weight functions
weight_strs = ["importance", "proportional", "jump distance", "locally balanced"]
# number of candidates
M_vec = [1, 5, 10, 15, 20, 25, 30]
inv_scales = [-0.5, -0.25, 0, 0.25, 0.5]

# manage distribution of simulations across the worker processes
runs_per_process = div(reps, n_procs) + (rank < rem(reps, n_procs) ? 1 : 0)
local_run_ids = collect((rank * (reps / n_procs) + 1):(rank * (reps / n_procs) + runs_per_process))
setting_cols = ["rep", "weight", "M", "iscale", "method"]
result_cols = ["n_total", "n_used", "mean", "variance", "mESS"]
local_results = DataFrame([Vector{Any}() for _ in 1:10], [setting_cols; result_cols])

for (i, w_str) in enumerate(weight_strs)
	if rank == 0
		println("Weight function: $w_str")
	end
		
	for (j, M) in enumerate(M_vec)
		if rank == 0
			println(" M: $M")
		end

		for (k, iscale) in enumerate(inv_scales)
			if rank == 0
				println("  Inverse-scale: $iscale")
			end

			if rank == 0
				println("   Homogeneous-Full")
			end
			base_seed = 1 + (j-1)*4*reps + (i-1)*length(M_vec)*4*reps + (k-1)*length(M_vec)*length(weight_strs)*4*reps

			for id in local_run_ids
				results = homogeneous_full(t_max, M, iscale, log_target, init, w_str, Int(base_seed+id))
				iters = results.n_iters
				burnin = get_burnin(results.X, iters)
				n0 = Int(iters - burnin)
				mESS = multiESS(results.X[n0:end,:])
				m = mean(results.X[n0:end,1])
				variance = var(results.X[n0:end,1])
				push!(local_results, [id, w_str, M, iscale, 1, iters, burnin, m, variance, mESS])
				GC.gc()
			end

			if rank == 0
				println("   Heterogeneous-Full")
			end
			base_seed = 1 + reps + (j-1)*4*reps + (i-1)*length(M_vec)*4*reps + (k-1)*length(M_vec)*length(weight_strs)*4*reps

			for id in local_run_ids
				results = heterogeneous_full(t_max, M, iscale, log_target, init, w_str, Int(base_seed+id))
				iters = results.n_iters
				burnin = get_burnin(results.X, iters)
				n0 = Int(iters - burnin)
				mESS = multiESS(results.X[n0:end,:])
				m = mean(results.X[n0:end,1])
				variance = var(results.X[n0:end,1])
				push!(local_results, [id, w_str, M, iscale, 2, iters, burnin, m, variance, mESS])
				GC.gc()
			end

			if rank == 0
				println("   Homogeneous-CW")
			end
			base_seed = 1 + 2*reps + (j-1)*4*reps + (i-1)*length(M_vec)*4*reps + (k-1)*length(M_vec)*length(weight_strs)*4*reps

			for id in local_run_ids
				results = homogeneous_cw(t_max, M, iscale, log_target, init, w_str, Int(base_seed+id))
				iters = results.n_iters
				burnin = get_burnin(results.X, iters)
				n0 = Int(iters - burnin)
				mESS = multiESS(results.X[n0:end,:])
				m = mean(results.X[n0:end,1])
				variance = var(results.X[n0:end,1])
				push!(local_results, [id, w_str, M, iscale, 3, iters, burnin, m, variance, mESS])
				GC.gc()
			end

			if M==1
				if rank == 0
					println("    -> skipping for M=1")
				end
			else
				if rank == 0
					println("   Heterogeneous-CW")
				end
				proposal_variance0 = [collect(range(-10,9, length=M)) for _ in 1:3]
				base_seed = 1 + (j-1)*4*reps + (i-1)*length(M_vec)*4*reps + (k-1)*length(M_vec)*length(weight_strs)*4*reps

				for id in local_run_ids
					results = heterogeneous_cw(t_max, M, iscale, log_target, proposal_variance0, init, w_str, Int(base_seed+id))
					iters = results.n_iters
					burnin = get_burnin(results.X, iters)
					n0 = Int(iters - burnin)
					mESS = multiESS(results.X[n0:end,:])
					m = mean(results.X[n0:end,1])
					variance = var(results.X[n0:end,1])
					push!(local_results, [id, w_str, M, iscale, 4, iters, burnin, m, variance, mESS])
					GC.gc()
				end
			end
		end
	end
end
println("Saving results from processor $rank to CSV.")
CSV.write("results_funnel/proc$(rank).csv", local_results)
MPI.Barrier(comm)