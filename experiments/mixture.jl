# univariate mixture model from richardson and green
# https://people.maths.bris.ac.uk/~mapjg/papers/RichardsonGreenRSSB.pdf

# modifications:
# - no upper bound to k (kmax) -- why did they do this??
# - only implement split/merge moves, not other moves..
# - instead of restricting the means to be ordered, add explicit swap moves
#   that permute the indices
# - no hyperpriors

# just test the split/merge move, from a reasonably chosen initial state, show
# that it accepts and works

using Gen
using PyPlot
import Random

include("dirichlet.jl")

@gen function model(n::Int)
    k ~ poisson(2)
    means = Vector{Float64}(undef, k)
    vars = Vector{Float64}(undef, k)
    for j in 1:k
        means[j] = ({(:mu, j)} ~ normal(0, 10))
        vars[j] = ({(:var, j)} ~ inv_gamma(1, 1))
    end
    probs ~ dirichlet(fill(2, k))
    for i in 1:n
        z = ({(:z, i)} ~ categorical(probs))
        {(:x, i)} ~ normal(means[z], vars[z])
    end
end

function get_n(trace)
    return get_args(trace)[1]
end

function get_means(trace)
    k = trace[:k]
    return [trace[(:mu, i)] for i in 1:k]
end

function get_vars(trace)
    k = trace[:k]
    return [trace[(:var, i)] for i in 1:k]
end

function get_xs(trace)
    n = get_n(trace)
    return [trace[(:x, i)] for i in 1:n]
end

function marginal_density(k, probs, means, vars, x)
    ls = zeros(k)
    for j in 1:k
        mu = means[j]
        var = vars[j]
        ls[j] = logpdf(normal, x, mu, var) + log(probs[j])
    end
    return exp(logsumexp(ls))
end

function get_densities_at(trace, xs)
    k = trace[:k]
    probs = trace[:probs]
    means = get_means(trace)
    vars = get_vars(trace)
    return [marginal_density(k, probs, means, vars, x) for x in xs]
end

# simulate data and plot a histogram..
function show_prior_data()
    Random.seed!(3)

    n = 1000
    trace = simulate(model, (n,))
    figure()
    xmin = -30.0
    xmax = 30.0

    # histogram
    #subplot(1, 2, 1)
    xs = get_xs(trace)
    (hist_data, ) = hist(xs, bins=collect(range(xmin, stop=xmax, length=50)))
    (ymin, ymax) = gca().get_ylim()

    # density plot
    #subplot(1, 2, 2)
    test_xs = collect(range(xmin, stop=xmax, length=1000))
    densities = get_densities_at(trace, test_xs)
    max_density = maximum(densities)
    scale = ymax / max_density
    plot(test_xs, densities * scale, color="orange")

    savefig("prior_sample.png")
end

show_prior_data()
