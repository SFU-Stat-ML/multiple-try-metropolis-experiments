# Project: A unified framework for multiple-try Metropolis algorithms
# Purpose: Experiment targeting posterior from simulated Bayesian regression problem
# Author:  Renny Doig and Liangliang Wang
# Date:    January 12, 2025

# Target distribution: posterior distribution for Bayesian regression problem
# Iterations: 50,000
# Repetitions: 50
# No. candidates: 1, 5, 10, 15
# Weight functions: importance, proportional, jump distance, locally balanced
# No. datasets: 10 (seed taken as input to Julia script)

# Other implementation details:
# - Single-thread regime
# - Use distributed scheme across 10 cores for repetitions


# Target distribution (more details):
# - terms: intercept, 4 covariates
# - coefficients (including intercept): [1, 0.1, 5, -5, 10]
# - observation noise SD: 0.5
# - sample size: 1000


# Four algorithms:
# 1. Global MTM with single proposal distribution
# 3. Component-wise MTM with single proposal distribution
# 4. Component-wise MTM with multiple proposal distributions

# All use adaptation, implementing the respective schema (with some modifications, noted in Julia file)
# 1. Andrieu & Thoms (2008) Algorithm 4
# 2. Fontaine (2022) AM Algorithm
# 3. Andrieu & Thoms (2008) Algorithm 4
# 4. Yang (2018) Algorithm 1

# Four weight functions:
# "importance":         w = target / proposal           Liu (2001)
# "proportional":       w = target                      unsure of earliest reference
# "locally balanced":   w = sqrt(target)                Gagnon (2022)
# "jump distance":      w = target * (jump distance)^a  Yang (2018)

# Preliminaries ----------------------------------------------------------------

using MPI, MPIPreferences, DataFrames, CSV, Distributions, Random, LinearAlgebra, StatsBase

# initialize workers
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

# parse input
data_seed = parse(Int64, ARGS[1])


# Data generation --------------------------------------------------------------

# regression coefficients
beta = [1, 0.1, 5, -5, 10]

# noise standard deviation
sigma = 0.5

# sample size
n = 1000

# number of covariates
d = 4

# generate covariates
Random.seed!(data_seed)
X = hcat(ones(n), reshape(rand(Uniform(), n*d), n, d))

# generate random errors
epsilon = rand(Normal(0,sigma), n)

# generate observations
y = X * beta .+ epsilon

# collect covariates and response into a tuble
data = (X=X, y=y)


## Specify functions -----------------------------------------------------------

# prior specifications
# beta_j ~ Normal(0, 100) -- mean 0, sd 100
# sigma ~ IG(2.01, 1.01) -- mean 1, var 100

# posterior distribution
function eval_log_target(data, theta)
	beta = theta[1:end-1]
	logsigma = theta[end]

	# evaluate priors
	lp = sum(logpdf.(Normal(0,100), beta)) + logpdf(InverseGamma(2.01, 1.01), exp(logsigma)) + logsigma

	# evaluate likelihood
	lp += sum(logpdf.(Normal.(data.X * beta, exp(logsigma)), data.y))

	return lp
end

dim = 6

# initialize distribution (prior)
init() = [rand(Normal(0,100), 5); log(rand(InverseGamma(2.01,1.01)))]


# Global simulation parameters -------------------------------------------------

# max run time for each simulation
t_max = 90.0

# number of repetitions of each simulation
reps = 50


# Importance weight function ---------------------------------------------------

weight_strs = ["importance", "proportional", "jump distance", "locally balanced"]
M_vec = [1, 2, 4, 6, 8, 10]

# assign simulation IDs to each worker process
runs_per_process = div(reps, n_procs) + (rank < rem(reps, n_procs) ? 1 : 0)
local_run_ids = collect((rank * (reps / n_procs) + 1):(rank * (reps / n_procs) + runs_per_process))
setting_cols = ["rep", "weight", "M", "method"]
result_cols = ["n_total"; "n_actual"; ["beta$i" for i in 0:4]; "sigma"]
local_results = DataFrame([Vector{Any}() for _ in 1:12], [setting_cols; result_cols])

for (i,w_str) in enumerate(weight_strs)
	if rank == 0
		println("Weight function: $w_str")
	end
	for (j,M) in enumerate(M_vec)
		if rank == 0
			println(" M: $M")
		end
		
		## Method 1
		if rank == 0
			println("  Homogeneous-Full")
		end
		base_seed = 1 + (j-1)*4*reps + (i-1)*length(M_vec)*4*reps + (data_seed-1)*length(weight_strs)*length(M_vec)*4*reps

		for id in local_run_ids
			# run simulation
			results = homogeneous_full(t_max, M, data, eval_log_target, init, w_str, Int(base_seed+id))
			# extract total number of iterations
			iters = results.n_iters
			# transform log-sigma back to proper scale
			results.X[:,6] = exp.(results.X[:,6])
			# get burn-in based on the algorithm
			burnin = get_burnin(results.X, iters)
			n0 = Int.(iters - burnin)
			# compute posterior means, trimming off the first 25% of samples as burn-in
			means = mean.(eachcol(results.X[n0:end, :]))
			# add these results to our local results DataFrame
			push!(local_results, [id; w_str; M; 1; iters; burnin; means])
			GC.gc()
		end


		## Method 2
		if rank == 0
			println("  Heterogeneous-Full")
		end
		base_seed = 1 + reps + (j-1)*4*reps + (i-1)*length(M_vec)*4*reps + (data_seed-1)*length(weight_strs)*length(M_vec)*4*reps

		for id in local_run_ids
			results = heterogeneous_full(t_max, M, data, eval_log_target, init, w_str, Int(base_seed+id))
			# extract total number of iterations
			iters = results.n_iters
			# transform log-sigma back to proper scale
			results.X[:,6] = exp.(results.X[:,6])
			# get burn-in based on the algorithm
			burnin = get_burnin(results.X, iters)
			n0 = Int.(iters - burnin)
			# compute posterior means, trimming off the first 25% of samples as burn-in
			means = mean.(eachcol(results.X[n0:end, :]))
			# add these results to our local results DataFrame
			push!(local_results, [id; w_str; M; 2; iters; burnin; means])
			GC.gc()
		end


		## Method 3
		if rank == 0
			println("  Homogeneous-CW")
		end
		base_seed = 1 + 2*reps + (j-1)*4*reps + (i-1)*length(M_vec)*4*reps + (data_seed-1)*length(weight_strs)*length(M_vec)*4*reps

		for id in local_run_ids
			results = homogeneous_cw(t_max, M, data, eval_log_target, init, w_str, Int(base_seed+id))
			# extract total number of iterations
			iters = results.n_iters
			# transform log-sigma back to proper scale
			results.X[:,6] = exp.(results.X[:,6])
			# get burn-in based on the algorithm
			burnin = get_burnin(results.X, iters)
			n0 = Int.(iters - burnin)
			# compute posterior means, trimming off the first 25% of samples as burn-in
			means = mean.(eachcol(results.X[n0:end, :]))
			# add these results to our local results DataFrame
			push!(local_results, [id; w_str; M; 3; iters; burnin; means])
			GC.gc()
		end


		## Method 4
		if rank == 0
			println("  Heterogeneous-CW")
		end
		if M==1
			if rank == 0
				println("   -> skipping for M=1")
			end
		else
			base_seed = 1 + 3*reps + (j-1)*4*reps + (i-1)*length(M_vec)*4*reps + (data_seed-1)*length(weight_strs)*length(M_vec)*4*reps

      proposal_variance0 = [collect(range(-5,5, length=M)) for _ in 1:dim]

			for id in local_run_ids
				results = heterogeneous_cw(t_max, M, data, eval_log_target, proposal_variance0, init, w_str, Int(base_seed+id))
				# extract total number of iterations
				iters = results.n_iters
				# transform log-sigma back to proper scale
				results.X[:,6] = exp.(results.X[:,6])
				# get burn-in based on the algorithm
				burnin = get_burnin(results.X, iters)
				n0 = Int.(iters - burnin)
				# compute posterior means, trimming off the first 25% of samples as burn-in
				means = mean.(eachcol(results.X[n0:end, :]))
				# add these results to our local results DataFrame
				push!(local_results, [id; w_str; M; 4; iters; burnin; means])
				GC.gc()
			end
		end
	end
end

println("Saving results from processor $rank to CSV.")
CSV.write("results_bayesian_regression/seed$(ARGS[1])_proc$(rank).csv", local_results)
MPI.Barrier(comm)