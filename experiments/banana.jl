# Project: A unified framework for multiple-try Metropolis algorithms
# Purpose: Experiment using banana target distribution
# Author:  Renny Doig and Liangliang Wang
# Date:    January 12, 2025


# Four algorithms:
# 1. Global MTM with single proposal distribution
# 2. Global MTM with multiple proposal distributions
# 3. Component-wise MTM with single proposal distribution
# 4. Component-wise MTM with multiple proposal distributions


# Target distribution: 5-dimensional banana distribution
# Iterations: 50,000
# Repetitions: 50
# No. candidates: 1, 5, 10, 15
# Weight functions: importance, proportional, jump distance, locally balanced

# Other implementation details:
# - Single-thread regime
# - Use distributed scheme across 10 cores for repetitions


# All use adaptation, implementing the respective schema (with some modifications, noted in Julia file)
# 1. Andrieu & Thoms (2008) Algorithm 4
# 2. Fontaine (2022) AM Algorithm
# 3. Andrieu & Thoms (2008) Algorithm 4
# 4. Yang (2018) Algorithm 1

# Four weight functions:
# "importance":         w = target / proposal           Liu (2001)
# "proportional":       w = target                      Liu (2001)
# "locally balanced":   w = sqrt(target)                Gagnon (2022)
# "jump distance":      w = target * (jump distance)^a  Yang (2018)

# Preliminaries ----------------------------------------------------------------

using MPI, MPIPreferences, DataFrames, CSV, Random, Distributions, LinearAlgebra, StatsBase

# initialize the multiple processes and store process information
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
n_procs = MPI.Comm_size(comm)

include("../functions/utils.jl")
include("../functions/homogeneous_full.jl")
include("../functions/heterogeneous_full.jl")
include("../functions/homogeneous_cw.jl")
include("../functions/heterogeneous_cw.jl")


# Target distribution ----------------------------------------------------------

# function to evaluate the log-posterior
function eval_log_target(y, theta)
	x1 = theta[1]
	x2 = theta[2]
	x3 = theta[3]
	x4 = theta[4]
	x5 = theta[5]
  B = 10.0^y

	return -x1^2/200 - 0.5*(x2+B*x1^2-100*B)^2 - 0.5*(x3^2+x4^2+x5^2)
end

# function to generate initial state
function init()
	return rand(Normal(), 5)
end


# Global simulation parameters -------------------------------------------------

# total run time for each simulation
t_max = 10.0

# number of repetitions of each simulation
reps = 50


# Run simulations --------------------------------------------------------------

weight_strs = ["importance", "proportional", "jump distance", "locally balanced"]
M_vec = [1, 5, 10, 15, 20]
B_vec = [-2, -1.75, -1.5, -1.25, -1, -0.75, -0.5]

runs_per_process = div(reps, n_procs) + (rank < rem(reps, n_procs) ? 1 : 0)
local_run_ids = collect((rank * (reps / n_procs) + 1):(rank * (reps / n_procs) + runs_per_process))
setting_cols = ["rep", "weight", "M", "method", "B"]
result_cols = ["n_total", "n_actual", "mESS"]
local_results = DataFrame([Vector{Any}() for _ in 1:8], [setting_cols; result_cols])

for (i,w_str) in enumerate(weight_strs)
	if rank ==0
  	println("Weight function: $w_str")
	end
  for (j,M) in enumerate(M_vec)
		if rank == 0
    	println(" M: $M")
		end

		for (k,B) in enumerate(B_vec)
			if rank == 0
				println("  B: $B")
			end

			## Method 1
			if rank == 0
				println("   Homogeneous-Full")
			end

			base_seed = 1 + (k-1)*4*reps + (j-1)*length(B_vec)*4*reps + (i-1)*length(B_vec)*length(M_vec)*4*reps

			for id in local_run_ids
				results = homogeneous_full(t_max, M, B, eval_log_target, init, w_str, Int(base_seed+id))
				iters = results.n_iters
				burnin = get_burnin(results.X, iters)
				n0 = Int(iters - burnin)
				mESS = multiESS(results.X[n0:end,:])
				push!(local_results, [id, w_str, M, 1, B, iters, burnin, mESS])
				GC.gc()
			end


			## Method 2
			if rank == 0
				println("   Heterogeneous-Full")
			end

			base_seed = 1 + reps + (j-1)*4*reps + (i-1)*length(M_vec)*4*reps

			for id in local_run_ids
				results = heterogeneous_full(t_max, M, B, eval_log_target, init, w_str, Int(base_seed+id))
				iters = results.n_iters
				burnin = get_burnin(results.X, iters)
				n0 = Int(iters - burnin)
				mESS = multiESS(results.X[n0:end,:])
				push!(local_results, [id, w_str, M, 2, B, iters, burnin, mESS])
				GC.gc()
			end


			## Method 3
			if rank == 0
				println("   Homogeneous-CW")
			end

			base_seed = 1 + 2*reps + (j-1)*4*reps + (i-1)*length(M_vec)*4*reps

			for id in local_run_ids
				results = homogeneous_cw(t_max, M, B, eval_log_target, init, w_str, Int(base_seed+id))
				iters = results.n_iters
				burnin = get_burnin(results.X, iters)
				n0 = Int(iters - burnin)
				mESS = multiESS(results.X[n0:end,:])
				push!(local_results, [id, w_str, M, 3, B, iters, burnin, mESS])
				GC.gc()
			end
			

			## Method 4
			if rank == 0
				println("   Heterogeneous-CW")
			end
			if M==1
				if rank == 0
					println("    -> skipping for M=1")
				end
			else
				base_seed = 1 + 3*reps + (j-1)*4*reps + (i-1)*length(M_vec)*4*reps

				proposal_variance0 = [collect(range(-10,9, length=M)) for _ in 1:5]

				for id in local_run_ids
					results = heterogeneous_cw(t_max, M, B, eval_log_target, proposal_variance0, init, w_str, Int(base_seed+id))
					iters = results.n_iters
					burnin = get_burnin(results.X, iters)
					n0 = Int(iters - burnin)
					mESS = multiESS(results.X[n0:end,:])
					push!(local_results, [id, w_str, M, 4, B, iters, burnin, mESS])
					GC.gc()
				end
			end
		end
  end
end

println("Saving results from processor $rank to CSV.")
CSV.write("results_banana/proc$(rank).csv", local_results)
MPI.Barrier(comm)
