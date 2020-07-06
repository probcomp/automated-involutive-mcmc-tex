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
        {(:x, i)} ~ normal(means[z], sqrt(vars[z]))
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
        ls[j] = logpdf(normal, x, mu, sqrt(var)) + log(probs[j])
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

function render_trace(trace, xmin, xmax)
    # histogram
    xs = get_xs(trace)
    (hist_data, ) = hist(xs, bins=collect(range(xmin, stop=xmax, length=50)))
    (ymin, ymax) = gca().get_ylim()

    # density plot
    test_xs = collect(range(xmin, stop=xmax, length=1000))
    densities = get_densities_at(trace, test_xs)
    max_density = maximum(densities)
    scale = ymax / max_density
    plot(test_xs, densities * scale, color="orange")
end

# simulate data and plot a histogram..
function show_prior_data()
    Random.seed!(3)
    n = 1000
    trace = simulate(model, (n,))
    figure()
    xmin = -30.0
    xmax = 30.0
    render_trace(trace, xmin, xmax)
    savefig("prior_sample.png")
end

show_prior_data()

function generate_synthetic_two_mixture_data()
    Random.seed!(1)
    n = 1000
    constraints = choicemap()
    constraints[:k] = 2
    constraints[:probs] = [0.5, 0.5]
    constraints[(:mu, 1)] = -10.0
    constraints[(:mu, 2)] = 10.0
    constraints[(:var, 1)] = 40.0
    constraints[(:var, 2)] = 40.0
    trace, = generate(model, (n,), constraints)
    figure()
    xmin = -30.0
    xmax = 30.0
    render_trace(trace, xmin, xmax)
    savefig("synthetic_data.png")
end

generate_synthetic_two_mixture_data()

function merge_weight(w1, w2)
    u1 = w1 / w
    return (w1 + w2, u1)
end

function merge_mean_and_var(mu1, mu2, var1, var2, w1, w2, w)
    mu = (mu1 * w1 + mu2 * w2) / w
    var = (w1 * (mu1^2 + var1) + w2 * (mu2^2 + var2)) / w - mu^2
    C = (var1 * w1) / (var2 * w2)
    u3 = C / (1 + C)
    u2 = ((mu - mu1) / sqrt(var)) * sqrt(w1 / w2)
    return (mu, var, u2, u3)
end

function split_weights(w, u1)
    w1 = w * u1
    w2 = w * (1 - u1)
    return (w1, w2)
end

function split_means(mu, var, u2, w1, w2)
    mu1 = mu - u2 * sqrt(var) * sqrt(w2 / w1)
    mu2 = mu + u2 * sqrt(var) * sqrt(w1 / w2)
    return (mu1, mu2)
end

function split_vars(w, w1, w2, var, u2, u3)
    var1 = u3 * (1 - u2^2) * var * w / w1
    var2 = (1 - u3) * (1 - u2^2) * var * w / w2
    return (var1, var2)
end

@gen function split_merge_proposal(trace)
    # decide whether to split or merge
    k = trace[:k]
    if k > 0
        split ~ bernoulli(0.5)
    else
        split = true
    end
    if split
        # if split, pick random to split
        j = uniform_discrete(1, k)
        # then pick DoFs
        u1 ~ beta(2, 2)
        u2 ~ beta(2, 2)
        u3 ~ beta(1, 1)
        # then compute new split values
        w = trace[:probs][j] # TODO requires the change to allow for multivariate.. (merge from other branch)
        mu = trace[(:mu, j)]
        var = trace[(:var, j)]
        (w1, w2) = split_weights(w, u1)
        (mu1, mu2) = split_means(mu, var, u2, w1, w2)
        (var1, var2) = split_vars(w, w1, w2, var, u2, u3)
        # TODO write addrs (to j and k+1)
        # TODO write increase k by one
    else
        # if merge, then pick two to merge
        j1 = uniform_discrete(1, k-1)
        j2 = k
        # then compute merged values
        (w, u1) = merge_weight(w1, w2)
        (mu, var, u2, u3) = merge_mean_and_var(mu1, mu2, var1, var2, w1, w2, w)
        # TODO write addrs
        # TODO write decrease k by one
    end
end

@involution function split_merge_inv()
end

# TODO add permutation moves..
