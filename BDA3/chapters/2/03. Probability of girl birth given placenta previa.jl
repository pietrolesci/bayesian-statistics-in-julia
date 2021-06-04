"""
BDA3 chapter 2 - Example. 03. Probability of girl birth given placenta previa
"""

using Plots, StatsPlots, LaTeXStrings
using Distributions
using Random
using StatsFuns: logit, logistic


Random.seed!(1994)
cd(@__DIR__)  # set wd to the script wd


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
    a = quantile(posterior, α/2)
    b = quantile(posterior, 1 - α/2)    
    
    println(
        """
        \nUniform prior:
            Posterior mean: $(mean(posterior))
            Posterior mean: $(std(posterior))
            Posterior mode: $(mode(posterior))
            Posterior median: $(median(posterior))
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
    a_normal_approx = quantile(posterior_normal_approx, α/2)
    b_normal_approx = quantile(posterior_normal_approx, 1 - α/2)

    println(
        """
        \nUniform prior - Normal approximation:
            Posterior mean: $(mean(posterior_normal_approx))
            Posterior mean: $(std(posterior_normal_approx))
            Posterior mode: $(mode(posterior_normal_approx))
            Posterior median: $(median(posterior_normal_approx))
            Central 95% credible interval: [$(a_normal_approx), $(b_normal_approx)]
        """
    )

    plot!(posterior_normal_approx, label="Normal approx", seriestype=:scatter)
end

# compute statistics on the samples from the posterior
begin
    # normal approximation
    chain = rand(posterior, 1000) |> sort
    
    # compute quantiles "by hand" sorting the samples and taking the 25th and 976th
    a_sampled = chain[25]
    b_sampled = chain[976]
    
    println(
        """
        \nUniform prior - sampled:
        Posterior mean: $(mean(chain))
        Posterior mean: $(std(chain))
        Posterior mode: $(mode(chain))
        Posterior median: $(median(chain))
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
    a_normal_approx_sample = quantile(posterior_normal_approx_sample, α/2)
    b_normal_approx_sample = quantile(posterior_normal_approx_sample, 1 - α/2)

    println(
        """
        \nUniform prior - Normal approximation (sampled):
            Posterior mean: $(mean(posterior_normal_approx_sample))
            Posterior mean: $(std(posterior_normal_approx_sample))
            Posterior mode: $(mode(posterior_normal_approx_sample))
            Posterior median: $(median(posterior_normal_approx_sample))
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
    a_normal_approx_sample_logit = quantile(posterior_normal_approx_sample_logit, α/2)
    b_normal_approx_sample_logit = quantile(posterior_normal_approx_sample_logit, 1 - α/2)

    println(
        """
        \nUniform prior - Normal approximation (logit and sampled):
            Posterior mean: $(logistic(mean(posterior_normal_approx_sample_logit)))
            Posterior mean: $(logistic(std(posterior_normal_approx_sample_logit)))
            Posterior mode: $(logistic(mode(posterior_normal_approx_sample_logit)))
            Posterior median: $(logistic(median(posterior_normal_approx_sample_logit)))
            Central 95% credible interval: [$(logistic(a_normal_approx_sample_logit)), $(logistic(b_normal_approx_sample_logit))]
        """
    )

end


### ==== 2. Analysis using conjugate prior distributions ==== #
begin
    
end
