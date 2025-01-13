# Project: A unified framework for multiple-try Metropolis algorithms
# Purpose: Experiment targeting a gaussian mixture distribution
# Author:  Renny Doig and Liangliang Wang
# Date:    January 12, 2025

# Target distribution: 5-component Gaussian mixture with a mode separation parameter of 3
# Iterations: 100,000
# Repetitions: 50
# No. candidates: 1, 5, 10, 15
# Weight functions: importance, proportional, jump distance, locally balanced
# Dimensions: 2, 4, 6, 8

# Other implementation details:
# - Single-thread regime
# - Use distributed scheme across 10 cores for repetitions

# Preliminaries ----------------------------------------------------------------

# load global libraries
using MPI, MPIPreferences, Random, DataFrames, CSV, LinearAlgebra, Distributions, StatsBase

# initialize worker processes
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


# Target distribution ----------------------------------------------------------

function initialize_target(dimension::Integer)
	# number of components
  K = 5
  # one-dimesional distance between modes
  dist = 3
  # component means
  m1 = zeros(dimension)
  m2 = repeat([dist],dimension)
  m3 = -m2
  m4 = [repeat([dist,-dist], Int(floor(dimension/2))); repeat([dist], mod(dimension,2))]
  m5 = -m4
  mu = [m1, m2, m3, m4, m5]
  # common covariance matrix
  Sigma = I(dimension)
  # component weights
  comp_weights = [1,2,4,2,1]
  tau = comp_weights / sum(comp_weights)

  # Categorical distribution for weights
  comp_dist = Categorical(tau)
  # Component distributions
  comps = [MvNormal(mu[k], Sigma) for k in 1:K]

  return MixtureModel(comps, comp_dist)
end

# function to evaluate the log-posterior
function eval_log_target(y, theta)
	which_target = (1:length(dims))[dims.==y][1]
	return logpdf(targets[which_target], theta)
end


# Global simulation parameters -------------------------------------------------

# total run time for each simulation
t_max = 30.0

# number of repetitions of each simulation
reps = 50


# Run simulations --------------------------------------------------------------

weight_strs = ["importance", "proportional", "jump distance", "locally balanced"]
M_vec = [1, 5, 10, 15, 20]
dims = [2, 4, 6, 8, 10]
targets = [initialize_target(d) for d in dims]

# manage distribution of simulations across the worker processes
runs_per_process = div(reps, n_procs) + (rank < rem(reps, n_procs) ? 1 : 0)
local_run_ids = collect((rank * (reps / n_procs) + 1):(rank * (reps / n_procs) + runs_per_process))
local_results = DataFrame(rep=[], weight=[], dimension=[], M=[], method=[], n_total=[], n_actual=[], KS=[])

# generate baseline samples for the KS distance
KS_baseline = [rand(targets[i], 50000)[1,:] for i in 1:length(targets)]

for (i, w_str) in enumerate(weight_strs)
  if rank ==0
    println("Weight function: $w_str")
  end
  for (j,dim) in enumerate(dims)
    if rank==0
      println(" dim: $dim")
    end

    for (k, M) in enumerate(M_vec)
      if rank == 0
			  println("  M: $M")
      end
      init() = rand(Normal(0, 20), dim)

      if rank == 0
        println("   Homogeneous-Full")
      end
			# set base seed value
			base_seed = 1 + (k-1)*4*reps + (j-1)*length(M_vec)*4*reps + (i-1)*length(dims)*length(M_vec)*4*reps

      for id in local_run_ids
			  results = homogeneous_full(t_max, M, dim, eval_log_target, init, w_str, Int(base_seed+id))
				iters = results.n_iters
				burn_in = ceil.([0.25, 0.5, 0.75]*iters)
        n0 = Int.(iters .- burn_in)
				D = [ksdist(KS_baseline[j], results.X[n0[l]:end,1]) for l in 1:3]
				best = findmin(D)
				push!(local_results, [id, w_str, dim, M, 1, iters, burn_in[best[2]], best[1]])
				GC.gc()
      end

      if rank == 0
			  println("   Heterogeneous-Full")
      end
			base_seed = 1 + reps + (k-1)*4*reps + (j-1)*length(M_vec)*4*reps + (i-1)*length(dims)*length(M_vec)*4*reps

      for id in local_run_ids
        results = heterogeneous_full(t_max, M, dim, eval_log_target, init, w_str, Int(base_seed+id))
				iters = results.n_iters
				burn_in = ceil.([0.25, 0.5, 0.75]*iters)
        n0 = Int.(iters .- burn_in)
				D = [ksdist(KS_baseline[j], results.X[n0[l]:end,1]) for l in 1:3]
				best = findmin(D)
				push!(local_results, [id, w_str, dim, M, 2, iters, burn_in[best[2]], best[1]])
				GC.gc()
      end

      if rank == 0
			  println("   Homogeneous-CW")
      end
			base_seed = 1 + 2*reps + (k-1)*4*reps + (j-1)*length(M_vec)*4*reps + (i-1)*length(dims)*length(M_vec)*4*reps

      for id in local_run_ids
        results = homogeneous_cw(t_max, M, dim, eval_log_target, init, w_str, Int(base_seed+id))
				iters = results.n_iters
				burn_in = ceil.([0.25, 0.5, 0.75]*iters)
				n0 = Int.(iters .- burn_in)
				D = [ksdist(KS_baseline[j], results.X[n0[l]:end,1]) for l in 1:3]
				best = findmin(D)
				push!(local_results, [id, w_str, dim, M, 3, iters, burn_in[best[2]], best[1]])
				GC.gc()
      end


      ## Method 4
      if rank ==0
			  println("   Heterogeneous-CW")
      end
			if M==1
        if rank == 0
				  println("    -> skipping for M=1")
        end
			else
				base_seed = 1 + 3*reps + (k-1)*4*reps + (j-1)*length(M_vec)*4*reps + (i-1)*length(dims)*length(M_vec)*4*reps
				
				proposal_variance0 = [collect(range(-10,10, length=M)) for _ in 1:dim]

        for id in local_run_ids
          results = heterogeneous_cw(t_max, M, dim, eval_log_target, proposal_variance0, init, w_str, Int(base_seed+id))
					iters = results.n_iters
					burn_in = ceil.([0.25, 0.5, 0.75]*iters)
					n0 = Int.(iters .- burn_in)
					D = [ksdist(KS_baseline[j], results.X[n0[l]:end,1]) for l in 1:3]
					best = findmin(D)
					push!(local_results, [id, w_str, dim, M, 4, iters, burn_in[best[2]], best[1]])
					GC.gc()
        end
			end
    end
  end
end
println("Saving results from processor $rank to CSV.")
CSV.write("results_gaussian_mixture/proc$(rank).csv", local_results)
MPI.Barrier(comm)