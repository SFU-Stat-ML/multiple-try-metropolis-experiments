# Project: A unified framework for multiple-try Metropolis algorithms
# Purpose: MTM with heterogeneous full proposals
# Author:  Renny Doig and Liangliang Wang
# Date:    January 11, 2025

# Implementation of the multiple-try Metropolis algorithm with different proposal distributions from Casarin et al. (2004)
# - Adaptive proposal covariance based on Algorithm 4 of Andrieu & Thoms (2008)
# - Learning rate for adaptation take to be gamma(m)=m^-0.6, following Gagnon et al. (2022)
function multi_mtm(runtime::Float64, M::Integer, data, target::Function, init::Function, weight::String, seed::Integer)
	Random.seed!(seed)

	## Adaptation settings =======================================================
	# scaling coefficient for adaptive covariance
	adapt_scale = 2.38^2
	# learning rate
	learn_rate(n) = n^-0.6
	# number of iterations prior to adaptation
	n0 = 100

	## Initialize variables ======================================================
	# initialize chain from initialization function
	X0 = init()
	# dimension of the state space
	d = length(X0)

	# initialize proposal covariance and mean
	prop_cov = [diagm(0.1*abs.(X0)) for _ in 1:M]
	prop_mean = [X0 for _ in 1:M]

	# runtime for a single evaluation of the target distribution
	t0 = @elapsed target(data, X0)
	# set initial size of the storage array to be based on the expected number of evaluations per iteration
	init_size = curr_size = Int(ceil(runtime / (t0*M)))

	# instantiate matrix of samples
	X = Matrix{Float64}(undef, init_size, d)
	X[1,:] .= X0

	# instantiate vectors for forward/backwards samples/weights
	y_star = Vector{Vector{Float64}}(undef, M)
	w_y = Vector{Float64}(undef, M)

	w(y, x, cov) = nothing
	# define log-weight function
	if weight == "proportional"
		w(y, x, cov) = target(data, y)
	elseif weight == "importance"
		w(y, x, cov) = target(data, y) - logpdf(MvNormal(x, adapt_scale/d*cov), y)
	elseif weight == "locally balanced"
		w(y, x, cov) = target(data, y) / 2
	elseif weight == "jump distance"
		w(y, x, cov) = target(data, y) + 2.9/2*log(sum((y .- x).^2))
	else
		error("Invalid weight option provided.")
	end

	## MCMC iterations ===========================================================
	n = 2
	start_time = time()
	while time() - start_time <= runtime
		# Generate candidates ------------------------------------------------------
		Threads.@threads for m in 1:M
			y_star[m] = rand(MvNormal(X[n-1,:], adapt_scale/d*prop_cov[m]))
			# compute forward weights
			w_y[m] = w(y_star[m], X[n-1,:], prop_cov[m])
		end

		# normalize the weights and sample a proposal from the candidates
		w_max = maximum(w_y)
		W = exp.(w_y .- w_max) / sum(exp.(w_y .- w_max))
		J = sample(pweights(W))

		# Generate reverse samples
		x_star = Vector{Vector{Float64}}(undef, M)
		w_x = Vector{Float64}(undef, M)

		Threads.@threads for m in 1:M
			x_star[m] = m==J ? X[n-1,:] : rand(MvNormal(y_star[J], adapt_scale/d*prop_cov[m]))
			# compute reverse weights
			w_x[m] = w(x_star[m], y_star[J], prop_cov[m]) 
		end

		# compute generalized Metropolis ratio
		ratio = logsum(w_y) - logsum(w_x)
		ratio += w_x[J] - w_y[J]
		ratio += target(data, y_star[J]) - target(data, X[n-1,:])
		ratio = min(1, exp(ratio))

		# accept/reject based on ratio
		if rand(Uniform()) < ratio
			X[n,:] = y_star[J]
		else
			X[n,:] = x_star[J]
		end

		## 3e. Adaptation
		# check if we have passed the initial period of no adaptation
		if n > n0
			Threads.@threads for m in 1:M
				# update proposal covariance
				prop_cov[m] += learn_rate(n) * ((X[n,:]-prop_mean[m]) * (X[n,:]-prop_mean[m])' - prop_cov[m])
				# update proposal mean
				prop_mean[m] += learn_rate(n) * (X[n,:] - prop_mean[m])

				# if the updated prop_cov is not positive definite (PD), compute PD matrix near it
				# - and for some reason, scaling can affect PD in Julia, so adjust the scaled matrix, then scale back
				if !isposdef(adapt_scale/d*prop_cov[m])
					prop_cov[m] = nearPD(adapt_scale/d*prop_cov[m]) ./ (adapt_scale/d)
				end

				# and finally, make sure it's Hermitian
				prop_cov[m] = Hermitian(prop_cov[m])
			end
		end

		# check if we need to increase the size of our storage matrices
		if n == curr_size
			# println("Expanding")
			temp = Matrix{Float64}(undef, curr_size, d)
			temp .= X
			X = Matrix{Float64}(undef, curr_size + init_size, d)
			X[1:curr_size,:] .= temp

			curr_size += init_size
		end
		n += 1
	end

	# trim off unused cells in the storage arrays
	X = X[1:(n-1),:]

	return (X=X, n_iters=n-1)
end
