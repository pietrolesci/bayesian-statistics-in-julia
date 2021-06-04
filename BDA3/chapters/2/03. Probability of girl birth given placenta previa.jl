"""
BDA3 chapter 2 - Example. 03. Probability of girl birth given placenta previa
"""

using Plots, StatsPlots, LaTeXStrings
using Distributions
using Random
using StatsFuns: logit, logistic
using DataFrames


Random.seed!(1994)
cd(@__DIR__)  # set wd to the script wd
round3(x) = round(x, digits=3)  # convenience function

### ==== 0. Data ==== ###
births = 980  # total number of births with placenta previa in Germany
fem_births = 437  # female births
pop_prop = 0.485  # upper bound on the expected population proportion of female births


### ==== 1. Analysis using a uniform prior distribution ==== #
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
end

# compute statistics on analytical posterior
begin
    # to compute 95% credible intervals
    α = 0.05
    a = quantile(posterior, α/2) |> round3
    b = quantile(posterior, 1 - α/2) |> round3
    
    println(
        """
        \nUniform prior:
            Posterior mean: $(mean(posterior) |> round3)
            Posterior std: $(std(posterior) |> round3)
            Posterior mode: $(mode(posterior) |> round3)
            Posterior median: $(median(posterior) |> round3)
            Central 95% credible interval: [$(a), $(b)]
        """
    )    

    plot(posterior, label="Exact posterior")
    xlims!((0.35, 0.55))
end

# compute statistics on the normal approximation
begin
    # normal approximation
    posterior_normal_approx = Normal(mean(posterior), std(posterior))

    # to compute 95% credible intervals
    a_normal_approx = quantile(posterior_normal_approx, α/2) |> round3
    b_normal_approx = quantile(posterior_normal_approx, 1 - α/2) |> round3

    println(
        """
        \nUniform prior - Normal approximation:
            Posterior mean: $(mean(posterior_normal_approx) |> round3)
            Posterior std: $(std(posterior_normal_approx) |> round3)
            Posterior mode: $(mode(posterior_normal_approx) |> round3)
            Posterior median: $(median(posterior_normal_approx) |> round3)
            Central 95% credible interval: [$(a_normal_approx), $(b_normal_approx)]
        """
    )

    plot!(posterior_normal_approx, label="Normal approx", seriestype=:scatter)
end

# compute statistics on the samples from the posterior
begin
    # normal approximation
    n_samples = 1000
    chain = rand(posterior, n_samples) |> sort
    
    # compute quantiles "by hand" sorting the samples and taking the 25th and 976th
    a_sampled = chain[Int(n_samples * 0.025)] |> round3
    b_sampled = chain[Int(n_samples * 0.975)] |> round3
    
    println(
        """
        \nUniform prior - sampled:
        Posterior mean: $(mean(chain) |> round3)
        Posterior std: $(std(chain) |> round3)
        Posterior mode: $(mode(chain) |> round3)
        Posterior median: $(median(chain) |> round3)
        Central 95% credible interval: [$(a_sampled), $(b_sampled)]
        """
    )

    # plot sampled posterior
    histogram!(chain, normed=true, alpha=0.3, label="Sampled posterior")
end

# compute statistics on the normal approximation to the samples from the posterior
begin
    # normal approximation
    posterior_normal_approx_sample = Normal(mean(chain), std(chain))

    # to compute 95% credible intervals
    a_normal_approx_sample = quantile(posterior_normal_approx_sample, α/2) |> round3
    b_normal_approx_sample = quantile(posterior_normal_approx_sample, 1 - α/2) |> round3

    println(
        """
        \nUniform prior - Normal approximation (sampled):
            Posterior mean: $(mean(posterior_normal_approx_sample) |> round3)
            Posterior std: $(std(posterior_normal_approx_sample) |> round3)
            Posterior mode: $(mode(posterior_normal_approx_sample) |> round3)
            Posterior median: $(median(posterior_normal_approx_sample) |> round3)
            Central 95% credible interval: [$(a_normal_approx_sample), $(b_normal_approx_sample)]
        """
    )

    plot!(posterior_normal_approx_sample, label="Normal approx using samples", seriestype=:scatter, marker=:star)
end

# compute statistics on the normal approximation to the transformed samples from the posterior
begin

    # trasform samples
    chain_logit = logit.(chain)

    # normal approximation
    posterior_normal_approx_sample_logit = Normal(mean(chain_logit), std(chain_logit))

    # to compute 95% credible intervals
    a_normal_approx_sample_logit = quantile(posterior_normal_approx_sample_logit, α/2) |> logistic |> round3
    b_normal_approx_sample_logit = quantile(posterior_normal_approx_sample_logit, 1 - α/2) |> logistic |> round3

    println(
        """
        \nUniform prior - Normal approximation (logit and sampled):
            Posterior mean: $(logistic(mean(posterior_normal_approx_sample_logit)) |> round3)
            Posterior std: $(logistic(std(posterior_normal_approx_sample_logit)) |> round3)
            Posterior mode: $(logistic(mode(posterior_normal_approx_sample_logit)) |> round3)
            Posterior median: $(logistic(median(posterior_normal_approx_sample_logit)) |> round3)
            Central 95% credible interval: [$(a_normal_approx_sample_logit), $(b_normal_approx_sample_logit)]
        """
    )

end


### ==== 2. Analysis using conjugate prior distributions ==== #
begin
    sample_sizes = [2, 2, 5, 10, 20, 100, 200]
    means = vcat([0.5], fill(pop_prop, length(sample_sizes) - 1))

    res = DataFrame(
        "Prior mean" => Real[], 
        "Prior sample size" => Real[],
        "Posterior median" => Real[],
        "Central 95% credible interval" => String[],
    )
    
    # look at blogpost for derivation of inverse mapping
    for (s, m) in zip(sample_sizes, means)
        a = s * m |> round3
        b = s - a |> round3
        
        posterior = Beta(fem_births + a, births - fem_births + b)
        lb = quantile(posterior, α/2) |> round3
        ub = quantile(posterior, 1 - α/2)|> round3

        push!(res, (m, s, round3(median(posterior)), "[$lb, $ub]"))
    end
    
    println(
        """
        \nConjugate prior:
        \t$(res)
        """
    )
end


### ==== 3. Analysis using nonconjugate prior distribution ==== #
begin
    # build the triangular prior
    θs = range(0.001, 1., step=0.001) |> collect
    prior = fill(1., length(θs))
    ascent = [(θ >= 0.385) && (θ <= 0.485) for θ in θs]
    descent = [(θ >= 0.485) && (θ <= 0.585) for θ in θs]
    peak = 11
    prior[ascent] = range(1, peak, length=sum(ascent)) |> collect
    prior[descent] = range(peak, 1, length=sum(descent)) |> collect
    prior /= sum(prior)  # normalize the prior

    # likelihood
    likelihood = [pdf(Binomial(births, θ), fem_births) for θ in θs]
    
    # posterior
    posterior = likelihood .* prior  # unnormalized
    # posterior = [pdf(Beta(fem_births, births - fem_births), θ) for θ in θs] .* prior  # unnormalized
    posterior /= sum(posterior)  # normalize
    
    # inverse cdf sampling
    # find the value smallest value θ at which the cumulative sum of the posterior densities is greater than r
    posterior_cdf = cumsum(posterior)
    n_samples = 100_000
    chain = θs[[sum(posterior_cdf .< r) + 1 for r in rand(n_samples)]]

    # statistics
    prior_mean = sum(prior .* θs)
    prior_std = (sum((θs .- prior_mean).^2) / length(θs))^(1/2)
    posterior_mean = sum(posterior .* θs)
    posterior_std = (sum((θs .- posterior_mean).^2) / length(θs))^(1/2)

    chain = sort(chain)
    a_sampled = chain[Int(n_samples * .025)] |> round3
    b_sampled = chain[Int(n_samples * .975)] |> round3

    println(
        """
        \nNonconjugate prior:
        \tPrior mean: $(prior_mean |> round3)
        \tPrior std: $(prior_std |> round3)
        \tPosterior mean: $(posterior_mean |> round3)
        \tPosterior std: $(posterior_std |> round3)
        \tSampled posterior mean: $(mean(chain) |> round3)
        \tSampled posterior std: $(std(chain) |> round3)
        \tSampled posterior median: $(median(chain) |> round3)
        \tCentral 95% credible interval: [$(a_sampled), $(b_sampled)]
        """
    )
        
    # plots
    plot(θs, prior, label="Prior", lw=2)
    plot!(θs, likelihood, label="Likelihood", lw=2)
    plot!(θs, posterior, label="Posterior", lw=2)
    histogram!(chain, normed=:probability, label="Samples from posterior", alpha=0.2)

end
