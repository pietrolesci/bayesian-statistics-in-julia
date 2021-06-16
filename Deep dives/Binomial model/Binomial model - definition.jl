"""
Implementation of the Binomial model based on Bayesian Data Analysis
    - Chapter 2: the model definition
"""

using Plots, StatsPlots, Distributions, LaTeXStrings, Random
Random.seed!(1994)
cd(@__DIR__)  # set wd to the script wd

# cdf
begin
    bin = Binomial(10, 0.5)
    sup = support(bin)
    plot(
        sup,
        [cdf(bin, i) for i in sup],
        seriestype = :step,
        marker = :dot,
        legend = :topleft,
        label = L"p(y \leq i|\theta, n)",
        xlabel = L"y",
        ylabel = "cdf",
        lw = 2,
        legendfontsize = 12,
        xticks = sup,
    )
    savefig("plots/cdf.svg")
end

# pmf
begin
    bin = Binomial(10, 0.5)
    plot(
        bin,
        label = L"p(y = i|\theta, n)",
        xlabel = L"y",
        ylabel = "pmf",
        lw = 2,
        legendfontsize = 12,
        xticks = support(bin),
    )
    savefig("plots/pmf.svg")
end

# neg_logpmf
begin
    bin = Binomial(10, 0.5)
    sup = support(bin)
    plot(
        sup,
        [-loglikelihood(bin, i) for i in sup],
        marker = :dot,
        legend = :topright,
        ylabel = "Negative log-pmf",
        xlabel = L"y",
        label = L"-\log p(y = i|\theta, n)",
        lw = 2,
        legendfontsize = 12,
        xticks = sup,
    )
    savefig("plots/neg_logpmf.svg")
end

# likelihood
begin
    θs = range(0.0, stop = 1.0, length = 1000)
    plot(
        θs,
        [pdf(Binomial(10, θ), 5) for θ in θs],
        legend = :topright,
        seriestype = :line,
        ylabel = "Likelihood",
        xlabel = L"\theta",
        label = L"p(y = 5|\theta, n = 10)",
        lw = 2,
        legendfontsize = 12,
    )
    savefig("plots/likelihood.svg")
end

# neg_loglikelihood
begin
    θs = range(0.0, stop = 1.0, length = 1000)
    plot(
        θs,
        [-loglikelihood(Binomial(10, θ), 5) for θ in θs],
        legend = :top,
        seriestype = :line,
        ylabel = "Negative log-likelihood",
        xlabel = L"\theta",
        label = L"-\log p(y = 5|\theta, n = 10)",
        lw = 2,
        legendfontsize = 12,
    )
    savefig("plots/neg_loglikelihood.svg")
end

# likelihood for n
begin
    ns = 0:50
    plot(
        ns,
        [pdf(Binomial(n, 0.5), 5) for n in ns],
        marker = :dot,
        seriestype = :line,
        ylabel = "Likelihood",
        xlabel = L"n",
        label = L"p(y = 5|\theta = .5, n)",
        lw = 2,
        legendfontsize = 12,
    )
    savefig("plots/likelihood_n.svg")
end

begin
    ns = 0:50
    plot(
        ns,
        [-loglikelihood(Binomial(n, 0.5), 5) for n in ns],
        legend = :topleft,
        marker = :dot,
        seriestype = :line,
        ylabel = "Negative log-likelihood",
        xlabel = L"n",
        label = L"-\log p(y = 5|\theta = .5, n)",
        lw = 2,
        legendfontsize = 12,
    )
    savefig("plots/neg_loglikelihood_n.svg")
end

begin
    ns = 0:50
    plot(
        ns,
        [-loglikelihood(Binomial(n, 0.5), 5) for n in ns],
        legend = :topleft,
        marker = :dot,
        seriestype = :line,
        ylabel = "Negative log-likelihood",
        xlabel = L"n",
        label = L"-\log p(y = 5|\theta = .5, n)",
        lw = 2,
        legendfontsize = 12,
    )
    savefig("plots/neg_loglikelihood_n.svg")
end

# joint likelihood of n ad θ
begin
    ns = 1:50
    θs = range(0.0, stop = 1.0, length = 1000)
    A = zeros(length(ns), length(θs))
    for (i, n) in enumerate(ns), (j, θ) in enumerate(θs)
        A[i, j] = pdf(Binomial(n, θ), 5)
    end
    heatmap(
        A,
        c = :inferno,
        xlabel = L"\theta",
        ylabel = L"n",
        xformatter = x -> "$(x/1000)",
    )
    savefig("plots/joint_likelihood.png")
end

# joint negative log-likelihood of n ad θ
begin
    ns = 1:50
    θs = range(0.0, stop = 1.0, length = 1000)
    A = zeros(length(ns), length(θs))
    for (i, n) in enumerate(ns), (j, θ) in enumerate(θs)
        A[i, j] = -loglikelihood(Binomial(n, θ), 5)
    end
    heatmap(A, xlabel = L"\theta", ylabel = L"n", xformatter = x -> "$(x/length(θs))")
    savefig("plots/joint_neg_loglikelihood.png")
end
