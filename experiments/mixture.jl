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
include("mixture_of_normals.jl")

@dist poisson_plus_one(rate) = poisson(rate) + 1

@gen function model(n::Int)
    k ~ poisson_plus_one(1)
    means = [({(:mu, j)} ~ normal(0, 10)) for j in 1:k]
    vars = [({(:var, j)} ~ inv_gamma(1, 10)) for j in 1:k]
    weights ~ dirichlet([2.0 for j in 1:k])
    for i in 1:n
        {(:x, i)} ~ mixture_of_normals(weights, means, vars)
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

function marginal_density(k, weights, means, vars, x)
    ls = zeros(k)
    for j in 1:k
        mu = means[j]
        var = vars[j]
        ls[j] = logpdf(normal, x, mu, sqrt(var)) + log(weights[j])
    end
    return exp(logsumexp(ls))
end

function get_densities_at(trace, xs)
    k = trace[:k]
    weights = trace[:weights]
    means = get_means(trace)
    vars = get_vars(trace)
    return [marginal_density(k, weights, means, vars, x) for x in xs]
end

function render_trace(trace, xmin, xmax)
    # histogram
    xs = get_xs(trace)
    (hist_data, ) = hist(xs, bins=collect(range(xmin, stop=xmax, length=50)), color="gray")
    (ymin, ymax) = gca().get_ylim()

    # density plot
    test_xs = collect(range(xmin, stop=xmax, length=1000))
    densities = get_densities_at(trace, test_xs)
    max_density = maximum(densities)
    scale = ymax / max_density
    plot(test_xs, densities * scale, color="orange", linewidth=4, zorder=1)
    
    # individual component density plot
    #if trace[:k] > 1
        for j in 1:trace[:k]
            densities = [marginal_density(1, [1.0], [trace[(:mu, j)]], [trace[(:var, j)]], x) for x in test_xs]
            max_density = maximum(densities)
            scale = ymax / max_density
            plot(test_xs, densities * scale, color="red", linewidth=2, zorder=2)
        end
    #end
end

# simulate data and plot a histogram..
function show_prior_data()
    Random.seed!(3)
    n = 100
    trace = simulate(model, (n,))
    figure()
    xmin = -30.0
    xmax = 30.0
    render_trace(trace, xmin, xmax)
    savefig("prior_sample.png")
end

#show_prior_data()

function generate_synthetic_two_mixture_data()
    Random.seed!(1)
    n = 100
    constraints = choicemap()
    constraints[:k] = 2
    constraints[:weights] = [0.5, 0.5]
    constraints[(:mu, 1)] = -10.0
    constraints[(:mu, 2)] = 10.0
    constraints[(:var, 1)] = 50.0
    constraints[(:var, 2)] = 50.0
    trace, = generate(model, (n,), constraints)
    #figure()
    #xmin = -30.0
    #xmax = 30.0
    #render_trace(trace, xmin, xmax)
    #savefig("synthetic_data.png")
    return trace
end

#generate_synthetic_two_mixture_data()

function merge_weights(weights, j, k)
    w1 = weights[j]
    w2 = weights[k]
    w = w1 + w2
    u1 = w1 / w
    new_weights = [(i == j) ? w : weights[i] for i in 1:k-1]
    return (new_weights, u1)
end

function merge_mean_and_var(mu1, mu2, var1, var2, w1, w2, w)
    mu = (mu1 * w1 + mu2 * w2) / w
    var = (w1 * (mu1^2 + var1) + w2 * (mu2^2 + var2)) / w - mu^2
    C = (var1 * w1) / (var2 * w2)
    u3 = C / (1 + C)
    u2 = ((mu - mu1) / sqrt(var)) * sqrt(w1 / w2)
    return (mu, var, u2, u3)
end

function split_weights(weights, j, u1, k)
    w = weights[j]
    w1 = w * u1
    w2 = w * (1 - u1)
    new_weights = [(i == j) ? w1 : (i == k + 1) ? w2 : weights[i] for i in 1:k+1]
    @assert isapprox(sum(new_weights), 1.0)
    return new_weights
end

function split_means(mu, var, u2, w1, w2)
    mu1 = mu - u2 * sqrt(var) * sqrt(w2 / w1)
    mu2 = mu + u2 * sqrt(var) * sqrt(w1 / w2)
    return (mu1, mu2)
end

function split_vars(w1, w2, var, u2, u3)
    var1 = u3 * (1 - u2^2) * var * (w1 + w2) / w1
    var2 = (1 - u3) * (1 - u2^2) * var * (w1 + w2) / w2
    return (var1, var2)
end

@gen function split_merge_proposal(trace)
   k = trace[:k]
   split = (k == 1) ? true : ({:split} ~ bernoulli(0.5))
   if split
      # split; pick cluster to split and sample degrees of freedom
      cluster_to_split ~ uniform_discrete(1, k)
      u1 ~ beta(2, 2)
      u2 ~ beta(2, 2)
      u3 ~ beta(1, 1)
   else
      # merge; pick cluster to merge with last cluster
      cluster_to_merge ~ uniform_discrete(1, k-1)
   end
end

@bijection function split_merge_inv(_, _, _)
    k = @read_discrete_from_model(:k)
    split = (k == 1) ? true : @read_discrete_from_proposal(:split)
    if split

        cluster_to_split = @read_discrete_from_proposal(:cluster_to_split)
        u1 = @read_continuous_from_proposal(:u1)
        u2 = @read_continuous_from_proposal(:u2)
        u3 = @read_continuous_from_proposal(:u3)
        weights = @read_continuous_from_model(:weights)
        mu = @read_continuous_from_model((:mu, cluster_to_split))
        var = @read_continuous_from_model((:var, cluster_to_split))

        new_weights = split_weights(weights, cluster_to_split, u1, k)
        w1 = new_weights[cluster_to_split]
        w2 = new_weights[k+1]
        (mu1, mu2) = split_means(mu, var, u2, w1, w2)
        (var1, var2) = split_vars(w1, w2, var, u2, u3)

        @write_discrete_to_model(:k, k+1)
        @copy_proposal_to_proposal(:cluster_to_split, :cluster_to_merge)
        @write_discrete_to_proposal(:split, false)
        @write_continuous_to_model(:weights, new_weights)
        @write_continuous_to_model((:mu, cluster_to_split), mu1)
        @write_continuous_to_model((:mu, k+1), mu2)
        @write_continuous_to_model((:var, cluster_to_split), var1)
        @write_continuous_to_model((:var, k+1), var2)

    else

        cluster_to_merge = @read_discrete_from_proposal(:cluster_to_merge)
        mu1 = @read_continuous_from_model((:mu, cluster_to_merge))
        mu2 = @read_continuous_from_model((:mu, k))
        var1 = @read_continuous_from_model((:var, cluster_to_merge))
        var2 = @read_continuous_from_model((:var, k))
        weights = @read_continuous_from_model(:weights)
        w1 = weights[cluster_to_merge]
        w2 = weights[k]

        (new_weights, u1) = merge_weights(weights, cluster_to_merge, k)
        w = new_weights[cluster_to_merge]
        (mu, var, u2, u3) = merge_mean_and_var(mu1, mu2, var1, var2, w1, w2, w)
    
        @write_discrete_to_model(:k, k-1)
        @copy_proposal_to_proposal(:cluster_to_merge, :cluster_to_split)
        if k > 2
            @write_discrete_to_proposal(:split, true)
        end
        @write_continuous_to_model(:weights, new_weights)
        @write_continuous_to_model((:mu, cluster_to_merge), mu)
        @write_continuous_to_model((:var, cluster_to_merge), var)
        @write_continuous_to_proposal(:u1, u1)
        @write_continuous_to_proposal(:u2, u2)
        @write_continuous_to_proposal(:u3, u3)
    end
end

is_involution!(split_merge_inv)

function split_merge_move(trace)
    return mh(trace, split_merge_proposal, (), split_merge_inv; check=true)
end

# TODO add permutation moves.. (with the last cluster)

function test_split_merge_move()
    Random.seed!(1)
    trace = generate_synthetic_two_mixture_data()
    merged_trace = trace
    num_acc_merge = 0
    for rep in 1:1000
        new_trace, acc = split_merge_move(trace)
        if acc && new_trace[:k] == 1
            num_acc_merge += 1
            println("mu: $(new_trace[(:mu, 1)]), var: $(new_trace[(:var, 1)])")
            merged_trace = new_trace
        end
    end
    println("num_acc_merge: $num_acc_merge")
    @assert num_acc_merge > 0

    figure(figsize=(6, 2))
    xmin = -30.0
    xmax = 30.0
    subplot(1, 2, 1)
    render_trace(trace, xmin, xmax)
    gca().get_yaxis().set_visible(false)
    subplot(1, 2, 2)
    render_trace(merged_trace, xmin, xmax)
    gca().get_yaxis().set_visible(false)
    tight_layout()
    savefig("rjmcmc.png")
end

test_split_merge_move()
