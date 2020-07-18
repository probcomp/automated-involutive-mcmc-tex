using Gen
using PyPlot
using LinearAlgebra: Symmetric, I

function cov_fn(x, y)
    return 0.452 + exp(-(x-y)^2/0.132) * (x - 0.085) * (y - 0.085)
end

function make_cov_matrix(xs)
    n = length(xs)
    mat = zeros(n, n)
    for i in 1:n
        for j in 1:n
            mat[i, j] = cov_fn(xs[i], xs[j])
        end
        #mat[i, i] += 0.1
    end
    return mat
end

#xs = collect(range(-1, stop=1, length=30))
xs = collect(range(-1, stop=1, length=100))
println(xs)
cov = make_cov_matrix(xs)
println(cov)

eye(n) = Matrix{Float64}(I, n, n)

figure(figsize=(4, 2))
for i in 1:10
    ys = mvnormal(zeros(length(xs)), Symmetric(cov) .+ 0.001 * eye(length(xs)))
    plot(ys, color="black", alpha=0.5)
end
yticks([-2, 0, 2])
xlabel("x")
ylabel("y")
tight_layout()
savefig("gp_samples.png", dpi=300)

#####

# just rescale the mixture weights in a 1D finite mixture model
