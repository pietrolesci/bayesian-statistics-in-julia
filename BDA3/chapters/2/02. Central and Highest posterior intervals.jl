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


function central_credible_interval(d::UnivariateDistribution; α = 0.05, nx = 1000)
    """
    Compute central credible intervals
    """
    lb, ub = quantile(d, [α / 2, 1 - α / 2])
    return range(lb, ub, length = nx)
end


function highest_posterior_density_interval(d::UnivariateDistribution; α = 0.95, nx = 1000)
    """
 Compute highest posterior density intervals

 NOTE: does not work with infinite supports, yet
 """
    # discretize over the support
    lb = isfinite(minimum(support(d))) ? minimum(support(d)) : -1000
    ub = isfinite(maximum(support(d))) ? maximum(support(d)) : 1000

    # histogram approximation of area under pdf
    x = range(lb, ub, length = nx)
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

    return cred_x
end


# example
posterior = Normal()
cci = central_credible_interval(posterior)
hpd = highest_posterior_density_interval(posterior)


# plotting - first way
plot(B, label = "pdf with 95% region highlighted")
plot!(cci, pdf(B, cci), fill = (0, 0.9, :orange), alpha = 0, label = "")

plot(B, label = "pdf with 95% region highlighted")
plot!(hpd, pdf(B, hpd), fill = (0, 0.9, :orange), alpha = 0, label = "")


# plotting - second way
normal_distribution = Normal()
x_min = -0.5
x_max = 0.5

plot(normal_distribution)
plot!(normal_distribution, x_min, x_max, fill = true, linewidth = 0, label = false)
