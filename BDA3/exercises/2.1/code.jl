using Plots, StatsPlots, LaTeXStrings
using Distributions, Random


Random.seed!(1994)
cd(@__DIR__)  # set wd to the script wd


# define prior
prior = Beta(4, 4)

# define the unnormalized posterior
function unnormalized_posterior(θ)
    (1 - θ)^13 * θ^3
    +10 * (1 - θ)^12 * θ^4
    +45 * (1 - θ)^11 * θ^5
end

begin
    # create grid
    θs = range(0, 1, length = 1000)
    pointwise_likelihoods = [unnormalized_posterior(θ) for θ in θs]

    # plot unnormalized_posterior
    plot(
        θs,
        pointwise_likelihoods,
        seriestype = :line,
        label = "Unnormalized posterior",
        legend = :topleft,
    )
    plot!(twinx(), prior, color = "red", label = "Prior")
    xlabel!(L"\theta")
    savefig("plot.svg")
end

begin
    is = 1:200
    xs = cumsum([i * (1 - 0.4)^(i - 1) * 0.4 for i in is])
    plot(is, xs)
    hline!([1 / 0.4])
end
