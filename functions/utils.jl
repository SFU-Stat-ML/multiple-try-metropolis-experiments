# Project: A unified framework for multiple-try Metropolis algorithms
# Purpose: Utility functions for MTM experiments
# Author:  Renny Doig and Liangliang Wang
# Date:    January 11, 2025

# log-sum-exponential evaluation of log(sum(w))
function logsum(logw)
	logmax = maximum(logw)
	return log(sum(exp.(logw .- logmax))) + logmax
end

# get near positive definite matrix
function nearPD(X::Matrix)
	# compute nearest symmetric matrix to X
	Y = 0.5*(X + X')

	# compute a diagonal matrix of eigenvalues then take elementwise max(.,0)
	eigenvals = eigvals(Y)
	D = Diagonal(eigenvals)
	D = max.(D, 0)
	
	# compute the matrix of eigenvectors of Y
	Q = eigvecs(Y)

	# add small positive values to diagonal elements of QDQ' to make it PD
	epsilon = 1e-9

	return Q*D*Q' + epsilon*I
end

# copying algorithm from https://github.com/JuliaStats/HypothesisTests.jl/blob/1cbeccd8e31bc26f6d0f44301243b23eba558b05/src/kolmogorov_smirnov.jl#L150-L163
function ksdist(baseline::Vector, MCsample::Vector)
	n0, n1 = length(baseline), length(MCsample)
	# empirical increments
	d0, d1 = 1/n0, 1/n1

	dist = dist_pos = dist_neg = 0

	all_samples = [baseline; MCsample]
	sorted_ids = sortperm(all_samples)

	for i in 1:length(all_samples)
		if sorted_ids[i] > n0
			dist -= d1
		else
			dist += d0
		end

		if i==(n0+n1) || all_samples[sorted_ids[i]] != all_samples[sorted_ids[i+1]]
			if dist > dist_pos
				dist_pos = dist
			elseif dist < dist_neg
				dist_neg = dist
			end
		end
	end

	return max(dist_pos, -dist_neg)
end

# function to check convergence based on hand-wavy voodoo
function get_burnin(samples::Array, iters::Int, rel_window_size::Float64=0.05, n_consec_fails::Integer=2)
	# moving window size as a percentage of total no. iterations, to keep comp time from scaling too poorly with n
	# default value fixed to 5%
	window_size = Int(floor(iters*rel_window_size))

	# initialize values
	i = 2
	# m = mean.(eachcol(samples[end-window_size:end,:]))
	# s = std.(eachcol(samples[end-window_size:end,:]))
	L = quantile.(eachcol(samples[end-window_size:end,:]), 0.025)
	U = quantile.(eachcol(samples[end-window_size:end,:]), 0.975)
	flag = 0

	# perform the following iterations until either we've observed a certain number of consecutive intervals
	#  that fail to overlap or until we've covered the whole chain
	# in the case that the whole chain appears to have converged, remove the first window_size just to be safe
	while (flag < n_consec_fails) && ((i+1)*window_size < iters)
		m0 = mean.(eachcol(samples[end-i*window_size:end-(i-1)*window_size,:]))
		# s0 = std.(eachcol(samples[end-i*window_size:end-(i-1)*window_size,:]))
		# L0 = quantile.(eachcol(samples[end-i*window_size:end-(i-1)*window_size,:]), 0.025)
		# U0 = quantile.(eachcol(samples[end-i*window_size:end-(i-1)*window_size,:]), 0.975)
		# if U > L0 && L < U0 
		if m0 >= L && m0 <= U
			flag = 0
			# m = mean.(eachcol(samples[end-i*window_size:end,:]))
			# s = std.(eachcol(samples[end-i*window_size:end,:]))
			L = quantile.(eachcol(samples[end-i*window_size:end,:]), 0.025)
			U = quantile.(eachcol(samples[end-i*window_size:end,:]), 0.975)
		else
			flag += 1
		end
		# m = mean(samples[end-i*window_size:end,1])
		# println("Running interval: ($(L[1]), $(U[1])) ")
		# println("Running mean: $m")
		i += 1
	end
	return (i-3)*window_size
end

# following implementation from https://github.com/rsantet/multiESS/tree/master based on Vats et al.
function multiESS(X::Matrix)
	n = size(X)[1]
	p = size(X)[2]

	# take the number of batches to be the square-root of the number of rows
	B = Int(floor(sqrt(n)))
	# the size of each batch
	a = Int(floor(n/B))
	# number of offsets
	n_offset = 10
	offsets = unique(Int.(round.(range(0, n-a*B, n_offset))))

	# overall means
	grand_means = mean(X, dims=1)
	# determinant of the covariance matrix
	detLambda = det(cov(X))

	mESS = Vector{Float64}(undef, B)

	for b in 1:B
		# batch estimate of covariance
		Sigma = zeros(p, p)
		for i in offsets
			for j in 0:a-1
				# get batch of samples
				try
          global Y = X[(1+j*B+i):((j+1)*B+i),:]
        catch e
          println("DEBUG_MESSAGE: Lower index: $(1+j*B+i); Upper index: $((j+1)*B+i); Max: $n")
          rethrow()
        end
				# compute batch mean
				batch_mean = mean(Y, dims=1)
				# compute deviations between batch and overall mean
				dev = batch_mean .- grand_means
				# increment our estimate of Sigma
				Sigma .+= dev' * dev
			end
		end

		# scale Sigma by the appropriate amount
		Sigma = Sigma*B/(a-1)/n_offset

		mESS[b] = try Int(floor(n*(detLambda/det(Sigma))^(1/p))) 
      catch e 
      println("Warning: NaN in multiESS")
      Inf 
      end
	end

	return minimum(mESS)
end
