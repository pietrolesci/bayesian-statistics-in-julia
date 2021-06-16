"""
BDA3 chapter 2 - Example. Estimating the probability of a female birth
"""

using Plots, StatsPlots, LaTeXStrings
using Distributions
using Random
using Optim: optimize, minimizer

Random.seed!(1994)
cd(@__DIR__)  # set wd to the script wd


### ==== 0. Data ==== ###
births = 978
fem_births = 437


### ==== 1. fit model by MLE ==== ###
begin
    # using Distribution.jl convenience method
    distributionsjl_mle = fit_mle(Binomial, births, [fem_births]).p
    
    # and computing is by "hand" as it has a closed form solution
    analytical_mle = fem_births / births

    # using optimization routines to minimize the negative loglikelihood
    loglik(θ) = -loglikelihood(Binomial(births, θ), fem_births)
    computational_mle = optimize(loglik, 0., 1.)
    optimized_mle = minimizer(computational_mle)

    println("Distributions.jl MLE: $(distributionsjl_mle)")
    println("Analytical MLE: $(analytical_mle)")
    println("Optimized MLE: $(optimized_mle)")
end


### ==== 2. fit model by Bayesian inference analytically ==== #
begin
    # define prior
    prior = Uniform(0, 1)

    # evaluate likelihood of data at a grid of θ values in its support
    θs = rand(prior, 1000)
    likelihood = [
        pdf(Binomial(births, θ), fem_births) 
        for θ in θs
    ]

    # analytical posterior
    posterior = Beta(fem_births + 1, births - fem_births + 1)

    # plots
    plot(prior, label="Prior", lw=2, legend=:topleft)
    plot!(posterior, label="Posterior", lw=2, legend=:topleft)
    plot!(twinx(), θs, likelihood, label="Likelihood", seriestype=:line)
    xlabel!(L"\theta")
    title!("Summary of the inference procedure")
    
    println("Distributions.jl MLE: $(distributionsjl_mle)")
    println("Analytical MLE: $(analytical_mle)")
    println("Optimized MLE: $(optimized_mle)")
    println("Bayesian estimate with uniform prior: $(mean(posterior))")
    println("Bayesian MAP estimate with uniform prior: $(mode(posterior))")
end
