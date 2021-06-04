"""
BDA3 chapter 2 - Example. Estimating the probability of a female birth
"""

using Plots, StatsPlots, LaTeXStrings
using Distributions
using Random


Random.seed!(1994)
cd(@__DIR__)  # set wd to the script wd


### ==== 0. Data ==== ###
births = 978
fem_births = 437

function credible_interval(d::UnivariateDistribution; α=0.95)
    lb, ub = quantile(d, [1 - α/2, α/2])
    return lb, ub
end


function highest_posterior_density_interval(d::UnivariateDistribution; α=0.95, nx=1000)
	
    # discretize over the support
	lb = isfinite(minimum(support(d))) ? minimum(support(d)) : 1e-7
	ub = isfinite(maximum(support(d))) ? maximum(support(d)) : 1e7

	# histogram approximation of area under pdf
	x = range(lb, ub, length=nx)
	f = pdf(d, x)
	a = f * (x[2] - x[1])

	# sort all the histogram bins
	sp = sortperm(a)
	x = x[sp]
	a = a[sp]

	a_tot = 0.0        # running sum of area
	cred_x = Float64[] # inputs to pdf within credible interval
	
	i = length(a)
	while a_tot < α
		a_tot += a[i]
		push!(cred_x, x[i])
		i -= 1
	end

	return extrema(cred_x)
end

## example
posterior = Normal()
lb_ci, ub_ci = credible_interval(posterior) 
lb_hpd, ub_hpd = highest_posterior_density_interval(posterior)

plot(posterior)
vline!([lb_ci, ub_ci])
vline!([lb_hpd, ub_hpd])