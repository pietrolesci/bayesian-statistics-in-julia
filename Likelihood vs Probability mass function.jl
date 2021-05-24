"""
Likelihood function vs Probability mass function
"""

using Plots, StatsPlots, Distributions, LaTeXStrings, Random

Random.seed!(1994)


# plot of the probability mass function, note how this is discrete and is
# a function of y
begin
    bar(Binomial(10, 0.5), xticks=(0:10), label="", xlabel=L"y")
    annotate!(8.5, .23, L"p \; (y = y_{i} \; | \; \theta = 0.5, n = 10)")
    # savefig("plot.svg")
end


# plot the likelihood function, note how this is continuous and is
# a function of θ
begin
    θs = rand(Uniform(0, 1), 1000)
    likelihoods = [pdf(Binomial(10, p), 5) for p in θs];
    plot(θs, likelihoods, seriestype = :line, label="", xlabel=L"\theta", lw=2)
    annotate!(.85, .23, L"p \; (y = 5 \; | \; \theta_i, n = 10)")
    # savefig("plot.svg")
end
