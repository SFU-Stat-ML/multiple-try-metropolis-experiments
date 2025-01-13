# Project: A unified framework for multiple-try Metropolis algorithms
# Purpose: MTM with homogeneous component-wise proposals
# Author:  Renny Doig and Liangliang Wang
# Date:    January 11, 2025

function cw_mtm(runtime::Float64, M::Integer, data, target::Function, init::Function, weight::String, seed::Integer)
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

	# initialize proposal variances and means
	prop_var = 0.1*abs.(X0)
	prop_mean = X0

	# runtime for a single evaluation of the target distribution
	t0 = @elapsed target(data, X0)
	# set initial size of the storage array to be based on the expected number of evaluations per iteration
	init_size = curr_size = Int(floor(runtime / (t0*M*d)))

	# instantiate matrix of samples
	X = Matrix{Float64}(undef, init_size, d)
	X[1,:] = X0
	
	# instantiate vectors for forward/backwards weights
	w_y = Vector{Float64}(undef, M)
	w_x = Vector{Float64}(undef, M)

	# define log-weight function
	w(y, x, i) = nothing
	if weight == "proportional"
		w(y, x, i) = target(data, y)
	elseif weight == "importance"
		w(y, x, i) = target(data, y) - logpdf(Normal(x[i], sqrt(adapt_scale/d*prop_var[i])), y[i])
	elseif weight == "locally balanced"
		w(y, x, i) = target(data, y) / 2
	elseif weight == "jump distance"
		w(y, x, i) = target(data, y) + 2.9*log(abs(y[i] - x[i]))
	else
		error("Invalid weight option provided.")
	end

	## Begin iterations ==========================================================
	n = 2
	start_time = time()
	while time() - start_time <= runtime
		X[n,:] = X[n-1,:]

		for i in 1:d
			# Generate M candidate proposals and compute weights
			# reset forward weights to current state estimate
			y_star = [X[n,:] for _ in 1:M]

			# propose from each of the covariances and compute the weights
			fwd_prop = Normal(X[n-1,i], sqrt(adapt_scale/d*prop_var[i]))
			Threads.@threads for m in 1:M
				y_star[m][i] = rand(fwd_prop)
				w_y[m] = w(y_star[m], X[n-1,:], i)
			end

			## 3b. Sample a candidate proposal
			# normalize weights and sample based on them
			w_max = maximum(w_y)
			W = exp.(w_y .- w_max) / sum(exp.(w_y .- w_max))
			J = sample(1:M, pweights(W))

			## 3c. Generate M-1 reverse samples
			# reset backwards weights to current state estimates
			x_star = [X[n,:] for _ in 1:M]
			
			# generate reverse samples and compute weights for remaining M-1 samples
			bwd_prop = Normal(y_star[J][i], sqrt(adapt_scale/d*prop_var[i]))
			Threads.@threads for m in 1:M
				if m != J
					x_star[m][i] = rand(bwd_prop)
				end
				w_x[m] = w(x_star[m], y_star[J], i)
			end

			## 3d. Accept or reject the proposal
			# compute generalized Metropolis ratio
			ratio = logsum(w_y) - logsum(w_x)
			ratio += w_x[J] - w_y[J]
			ratio += target(data, y_star[J]) - target(data, X[n,:])
			ratio = min(1, exp(ratio))

			# accept/reject based on ratio
			if rand(Uniform()) < ratio
				X[n,i] = y_star[J][i]
			end

			## 3e. Adapt the proposal variance for component i
			if n > n0
				# update proposal covariance
				prop_var[i] += learn_rate(n) * ((X[n,i] - prop_mean[i])^2 - prop_var[i])
				# update proposal mean
				prop_mean[i] += learn_rate(n) * (X[n,i] - prop_mean[i])
			end
		end

		# check if we need to increase the size of our storage matrices
		if n == curr_size
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
