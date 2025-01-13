# Project: A unified framework for multiple-try Metropolis algorithms
# Purpose: Experiment targeting posterior from the eight schools problem
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
function log_target(y, x::Vector)
	sample_mean = [28,8,-3,7,-1,1,18,12]
	sample_SD = [15.0,10,16,11,9,11,10,18]

	# hyperpriors on mu and tau
	lt = logpdf(Normal(0, 5), x[end-1])
	lt += logpdf(Cauchy(0, 5), exp(x[end])) - log(2) + x[end]
	# prior on theta
	lt += sum(logpdf.(Normal(x[end-1], exp(x[end])), x[1:(end-2)]))
	# likelihood of data
	lt += sum(logpdf.(Normal.(x[1:(end-2)], sample_SD), sample_mean))
	return lt
end

dimension = 10

# initialization function
function init()
	mu = rand(Normal(0,5))
	tau = log(abs(rand(Cauchy(0,5))))
	theta = rand(Normal(mu, exp(tau)), 8)
	return [theta; mu; tau]
end


# Simulation settings ----------------------------------------------------------

# total run-time for each run
t_max = 45.0

# number of runs for each setting
reps = 50

# weight functions
weight_strs = ["importance", "proportional", "jump distance", "locally balanced"]
# number of candidates
M_vec = [1, 5, 10, 15, 20]

# manage distribution of simulations across the worker processes
runs_per_process = div(reps, n_procs) + (rank < rem(reps, n_procs) ? 1 : 0)
local_run_ids = collect((rank * (reps / n_procs) + 1):(rank * (reps / n_procs) + runs_per_process))
setting_cols = ["rep", "weight", "M", "method"]
result_cols = ["n_total"; "n_actual"; ["theta$i" for i in 1:8]; "mu"; "tau"]
local_results = DataFrame([Vector{Any}() for _ in 1:16], [setting_cols; result_cols])

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
			results = homogeneous_full(t_max, M, missing, log_target, init, w_str, Int(base_seed+id))
			iters = results.n_iters
			burnin = get_burnin(results.X, iters)
			n0 = Int(iters - burnin)
			results.X[:,10] = exp.(results.X[:,10])
			means = mean.(eachcol(results.X[n0:end,:]))
			push!(local_results, [id; w_str; M; 1; iters; burnin; means])
			GC.gc()
		end

		# METHOD 2
		if rank == 0
			println("   Heterogeneous-Full")
		end
		base_seed = 1 + reps + (j-1)*4*reps + (i-1)*length(M_vec)*4*reps

		for id in local_run_ids
			results = heterogeneous_full(t_max, M, missing, log_target, init, w_str, Int(base_seed+id))
			iters = results.n_iters
			burnin = get_burnin(results.X, iters)
			n0 = Int(iters - burnin)
			results.X[:,10] = exp.(results.X[:,10])
			means = mean.(eachcol(results.X[n0:end,:]))
			push!(local_results, [id; w_str; M; 2; iters; burnin; means])
			GC.gc()
		end

		# METHOD 3
		if rank == 0
			println("   Homogeneous-CW")
		end
		base_seed = 1 + 2*reps + (j-1)*4*reps + (i-1)*length(M_vec)*4*reps

		for id in local_run_ids
			results = homogeneous_cw(t_max, M, missing, log_target, init, w_str, Int(base_seed+id))
			iters = results.n_iters
			burnin = get_burnin(results.X, iters)
			n0 = Int(iters - burnin)
			results.X[:,10] = exp.(results.X[:,10])
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
			proposal_variance0 = [collect(range(-10,9, length=M)) for _ in 1:dimension]
			base_seed = 1 + (j-1)*4*reps + (i-1)*length(M_vec)*4*reps

			for id in local_run_ids
				results = heterogeneous_cw(t_max, M, missing, log_target, proposal_variance0, init, w_str, Int(base_seed+id))
				iters = results.n_iters
				burnin = get_burnin(results.X, iters)
				n0 = Int(iters - burnin)
				results.X[:,10] = exp.(results.X[:,10])
				means = mean.(eachcol(results.X[n0:end,:]))
				push!(local_results, [id; w_str; M; 4; iters; burnin; means])
				GC.gc()
			end
		end
	end
end
println("Saving results from processor $rank to CSV.")
CSV.write("results_eight_schools/proc$(rank).csv", local_results)
MPI.Barrier(comm)
