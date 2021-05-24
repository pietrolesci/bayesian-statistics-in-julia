"""
Implementation of the Binomial model based on Bayesian Data Analysis
    - Chapter 2: the model definition and inference
    - Chapter 6: posterior predictive checks
"""

using Plots, StatsPlots, Distributions, LaTeXStrings
using Turing, Random, MCMCChains
using StatsBase: countmap
using Optim: optimize, minimizer
Random.seed!(1994)

# data
births = 978
fem_births = 437


### ==== 1. fit model by MLE ==== ###
begin
    # using Distribution.jl convenience method
    dist_pack_mle = fit_mle(Binomial, births, [fem_births])
    
    # and computing is by "hand" as it has a closed form solution
    analytical_mle = fem_births / births

    # using optimization routines to minimize the negative loglikelihood
    loglik(θ) = -loglikelihood(Binomial(births, θ), fem_births);
    computational_mle = optimize(loglik, 0., 1.);
    minimizer(computational_mle)
end


### ==== fit model by Bayesian inference ==== #
@model function binomial_model(n::Integer, y::Union{Integer, Missing}=missing)
    """
    The binomial model p(y|θ) assigns probability on the discrete set {0, ..., n} ⊆ ℕ.

    With a uniform prior, the marginal (or prior predictive distribution) p(y) assigns
    equal probability to all points in the set. In particular p(y) = 1 / (n + 1), ∀ i ∈ {0, ..., n}.

    Summary
    =======
    prior p(θ): distribution over θ
    posterior p(θ|y): distribution over θ, given the data y
    prior predictive p(y): distribution over y, averaging over θ weighted by its prior
    posterior predictive p(ỹ|y): distribution over new ỹ, averaging over θ weighted by its posterior
    """
    θ ~ Uniform(0, 1)
    y ~ Binomial(n, θ)
end

begin
    # using the NUTS sampler
    model = binomial_model(births, fem_births);
    posterior_chain = sample(model, NUTS(), 3_000);
    computational_estimate = mean(posterior_chain)
    
    # and the analytical posterior that can be computed in closed form
    analytical_posterior = Beta(fem_births + 1, births - fem_births + 1);  # eq. 2.3
    analytical_estimate = mean(analytical_posterior)

    # and check what would the model predict without seeing any data
    prior_chain = sample(model, Prior(), 3_000);
    analytical_prior_estimate = mean(prior_chain)
end

# plot results
begin
    # likelihood
    θs = rand(Uniform(0, 1), 1000);
    likelihoods = [pdf(Binomial(births, θ), fem_births) for θ in θs];
    plot(θs, likelihoods, seriestype=:line, label="Likelihood", xlabel=L"\theta", lw=2)

    # prior
    histogram!(prior_chain[:θ], normed=true, alpha=0.3, label="Prior")
    plot!(Uniform(0, 1), label="Analytical prior")

    # posterior
    histogram!(posterior_chain[:θ], normed=true, alpha=0.3, label="Posterior")
    plot!(analytical_posterior, label="Analytical posterior", lw=2)

    title!("Summary of the inference procedure")
end

# predictive checks: predictive distributions
begin
    # instantiate the "predictive model" (note: no new data)
    model_predict = binomial_model(births);
    
    # prior predictive
    prior_predictive = predict(model_predict, prior_chain)  # should be same as eq. 2.5
    analytical_prior_predictive = 1 / (births + 1)  # eq. 2.5
    analytical_prior_predictive_dist = BetaBinomial(births, 1, 1)
    
    # `posterior_predictive` is a probability about a rv ∈ {0, ..., n}!!!
    posterior_predictive = predict(model_predict, posterior_chain)
    analytical_posterior_predictive = BetaBinomial(births, fem_births + 1, births - fem_births + 1)

    # posterior predictive for single datapoint, i.e. probability that a new draw is success
    analytical_posterior_predictive_datapoint = (fem_births + 1) / (births + 2)  # eq. 2.6
end

# plot predictive distributions
begin
    # prior predictive
    histogram(prior_predictive[:y], normed=true, alpha=0.3, label="Prior predictive", xlabel=L"y")
    plot!(analytical_prior_predictive_dist, lw=2, label="Analytical prior predictive (BetaBinomial)")
    hline!([analytical_prior_predictive], lw=2, colour=:red, label="Analytical prior predictive")

    # posterior predictive
    histogram!(posterior_predictive[:y], normed=true, alpha=0.3, label="Posterior predictive")
    plot!(analytical_posterior_predictive, seriestype=:line, label="Analytical posterior predictive", lw=2)

    title!("Predictive checks")
end

# plot Bayesian p-value (or tail-area probability) - like Figure 6.4
begin    
    # posterior predictive
    histogram(posterior_predictive[:y], normed=true, alpha=0.5, label="Posterior predictive", xlabel=L"y")
    
    # compute Bayesian p-value
    vline!([fem_births], lw=2, colour=:red, label="True data")
    annotate!(fem_births - 50, 0.018, Plots.text("p-value=$(round(mean(posterior_predictive[:y] .> fem_births), digits=3))", 10))

    title!("Bayesian p-value (tail-area probability")
end
