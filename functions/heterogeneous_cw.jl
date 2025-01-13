# Project: A unified framework for multiple-try Metropolis algorithms
# Purpose: MTM with heterogeneous component-wise proposals
# Author:  Renny Doig and Liangliang Wang
# Date:    January 11, 2025

function heterogeneous_cw(runtime::Float64, M::Integer, data, target::Function, proposal_variance, init::Function, weight::String, seed::Integer)
	Random.seed!(seed)

	## Adaptation settings =======================================================
	# lower bound on proposal variance (log2 scale)
	epsilon = -15
	# upper bound on proposal variance (log2 scale)
	L = 50
	# adaptation period
	beta = 100
	# initial adaptation probability
	P = 1
	#alpha = 2
	select_freq = zeros(d, M)
	adapt_period = 0
	

	## Initialize storage arrays =================================================
	# generate initial sample
	X0 = init()
	# dimension of the state space
	d = length(X0)
	# runtime for a single evaluation of the target distribution
	t0 = @elapsed target(data, X0)
	# set initial size of the storage array to be based on the expected number of evaluations per iteration
	init_size = curr_size = Int(ceil(runtime / (t0*M*d)))

	# instantiate matrix of samples
	X = Matrix{Float64}(undef, init_size, d)
	X[1,:] = X0
	
	# define log-weight function
	w(y, x, cov, i) = nothing
	if weight == "proportional"
		w(y, x, cov, i) = target(data, y)
	elseif weight == "importance"
		w(y, x, cov, i) = target(data, y) - logpdf(Normal(x[i], cov), y[i])
	elseif weight == "locally balanced"
		w(y, x, cov, i) = target(data, y) / 2
	elseif weight == "jump distance"
		w(y, x, cov, i) = target(data, y) + 2.9*log(abs(y[i] - x[i]))
	else
		error("Invalid weight option provided.")
	end


	## Begin sampling ============================================================
	n = 2
	start_time = time()
	while time() - start_time <= runtime
		X[n,:] = X[n-1,:]

		# check adaptation conditions ----------------------------------------------
		if mod(n-1, beta) == 0
			# adapt with probability P
			adapt = rand(Uniform()) <= P
			# update adaptation probability for next iteration
			a = (n-1)/beta
			P = max(0.99^(a-1), a^(-1/2))
		else
			adapt = false
		end

		if adapt
			# compute selection rates since last adaptation
			select_rate = select_freq / adapt_period

			# reset selection frequencies and adaptation period counter
			select_freq = zeros(d, M)
			adapt_period = 0
		end

		for i in 1:d
			# adapt proposal variances -----------------------------------------------
			if adapt
				# adjust upper proposal variance
				if select_rate[i,M] > 2/M
					temp = min(proposal_variance[i][M] + 1, L)
					proposal_variance[i] = collect(range(proposal_variance[i][1], temp, length=M))
				elseif (select_rate[i,M] < 1/(2*M)) && (proposal_variance[i][1]<(proposal_variance[i][M]-1))
					temp = max(proposal_variance[i][M] - 1, epsilon)
					proposal_variance[i] = collect(range(proposal_variance[i][1], temp, length=M))
				end

				# adjust lower proposal variance
				if select_rate[i,1] > 2 / M
					temp = max(proposal_variance[i][1] -1, epsilon)
					proposal_variance[i] = collect(range(temp, proposal_variance[i][M], length=M))
				elseif (select_rate[i,1]<1/(2*M)) && ((proposal_variance[i][1]+1)<proposal_variance[i][M])
					temp = min(proposal_variance[i][1] + 1, L)
					proposal_variance[i] = collect(range(temp, proposal_variance[i][M], length=M))
				end
			end

			# generate forward candidates --------------------------------------------
			y_star = [X[n,:] for _ in 1:M]
			w_y = Vector{Float64}(undef, M)

			# propose from each of the covariances and compute the weights
			Threads.@threads for m in 1:M
				y_star[m][i] = rand(Normal(X[n-1,i], 2^proposal_variance[i][m]))
				w_y[m] = w(y_star[m], X[n-1,:], 2^proposal_variance[i][m], i)
			end

			# normalize weights and sample a candidate 
			w_max = maximum(w_y)
			W = exp.(w_y .- w_max) / sum(exp.(w_y .- w_max))
			J = sample(1:M, pweights(W))

			# update selection frequency for adaption conditions
			select_freq[i,J] += 1

			# generate backwards samples ---------------------------------------------
			# reset backwards weights to current state estimates
			x_star = [X[n,:] for _ in 1:M]
			w_x = Vector{Float64}(undef, M)
			
			# generate reverse samples and compute weights for remaining M-1 samples
			Threads.@threads for m in 1:M
				if m != J
					x_star[m][i] = rand(Normal(y_star[J][i], 2^proposal_variance[i][m]))
				end
				w_x[m] = w(x_star[m], y_star[J], 2^proposal_variance[i][m], i)
			end

			# compute generalized Metropolis ratio
			ratio = logsum(w_y) - logsum(w_x)
			ratio += w_x[J] - w_y[J]
			ratio += target(data, y_star[J]) - target(data, X[n,:])
			ratio = min(1, exp(ratio))

			# accept/reject based on ratio
			if rand(Uniform()) < ratio
				X[n,i] = y_star[J][i]
			end
		end

		adapt_period += 1

		# increase storage matrix, as necessary 
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

	return (X=X, n_iters = n-1)
end
