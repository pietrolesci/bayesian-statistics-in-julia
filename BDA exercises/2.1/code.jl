using Plots, StatsPlots, Distributions, LaTeXStrings, Random
Random.seed!(1994)

# define prior
prior = Beta(4, 4)

# define the unnormalized posterior
function unnormalized_posterior(θ)
    (1 - θ)^13 * θ^3 
    + 10 * (1 - θ)^12 * θ^4 
    + 45 * (1 - θ)^11 * θ^5
end

begin
    # create grid
    xs = rand(1000);
    ys = [unnormalized_posterior(θ) for θ in xs];
    # plot
    plot(xs, ys, seriestype=:line, label="Unnormalized posterior")
    savefig("plot.svg")
end

