using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "ModelA.jl"))
using .ModelA
using Random
using Statistics
using Distributions
using StatsPlots
using LinearAlgebra

# SMC sampler with adaptive tempering on theta. Outer cloud holds N_theta
# particles in (log_tau_R, log_tau_F, log_phi); inner marginal log-likelihood
# is the GeneralisedFilters bootstrap PF on the latent state. Tempering
# schedule chosen adaptively to keep ESS ratio near 0.5. Move step is
# pseudo-marginal MH with K iterations per tempering step.
#
# This is what `smc-jax/smc2/` calls "SMC²" but is more precisely an
# SMC sampler (Del Moral, Doucet, Jasra 2006) with adaptive tempering
# and a PF inner marginal.

const PRIOR_MEAN = [-4.0, -12.0, 2.3]
const PRIOR_SD   = [0.3, 0.5, 0.5]

theta_to_params(theta) = ModelAParams(theta[1], theta[2], theta[3])

function log_prior(theta::AbstractVector)
    s = 0.0
    for i in 1:3
        s += logpdf(Normal(PRIOR_MEAN[i], PRIOR_SD[i]), theta[i])
    end
    return s
end

function sample_prior(rng::AbstractRNG, N::Int)
    out = Matrix{Float64}(undef, N, 3)
    for i in 1:N, j in 1:3
        out[i, j] = PRIOR_MEAN[j] + PRIOR_SD[j] * randn(rng)
    end
    return out
end

ess_of(log_w) = exp(2 * logsumexp(log_w) - logsumexp(2 .* log_w))
logsumexp(x) = (m = maximum(x); m + log(sum(exp.(x .- m))))

# Find delta_beta so that ESS at (beta + delta_beta) is target.
function find_delta_beta(log_w::AbstractVector, ll::AbstractVector,
                          target_ess::Real, beta_max::Real)
    # Reweight by delta_beta * ll; find delta s.t. ESS = target.
    f(delta) = ess_of(log_w .+ delta .* ll) - target_ess
    # If even at delta = beta_max we have ESS > target, return beta_max.
    if f(beta_max) > 0
        return beta_max
    end
    # Bisection.
    lo, hi = 0.0, beta_max
    for _ in 1:60
        mid = 0.5 * (lo + hi)
        if f(mid) > 0
            lo = mid
        else
            hi = mid
        end
    end
    return 0.5 * (lo + hi)
end

function multinomial_resample(rng::AbstractRNG, log_w::AbstractVector)
    N = length(log_w)
    w = exp.(log_w .- logsumexp(log_w))
    idx = rand(rng, Categorical(w), N)
    return idx
end

function mh_move!(rng::AbstractRNG, cfg::ModelConfig, y::AbstractVector,
                  theta_mat::AbstractMatrix, ll_vec::AbstractVector,
                  beta::Real, proposal_cov::AbstractMatrix,
                  n_particles::Int, K::Int)
    N = size(theta_mat, 1)
    L_chol = cholesky(Symmetric(proposal_cov)).L
    accepts = 0
    for i in 1:N, _ in 1:K
        theta_cur = view(theta_mat, i, :)
        proposal = theta_cur .+ L_chol * randn(rng, 3)
        ll_p = pf_marginal_loglik(rng, cfg, theta_to_params(proposal), y; n_particles)
        log_alpha = beta * (ll_p - ll_vec[i]) +
                    log_prior(proposal) - log_prior(theta_cur)
        if log(rand(rng)) < log_alpha
            theta_mat[i, :] .= proposal
            ll_vec[i] = ll_p
            accepts += 1
        end
    end
    return accepts / (N * K)
end

function smc_sampler(rng::AbstractRNG, cfg::ModelConfig, y::AbstractVector;
                     n_theta::Int, n_particles::Int,
                     ess_target_ratio::Real = 0.5,
                     n_mh_per_step::Int = 2,
                     max_steps::Int = 50,
                     proposal_scale::Real = 0.4)
    theta_mat = sample_prior(rng, n_theta)
    ll_vec = Vector{Float64}(undef, n_theta)
    println("  computing initial PF marginal for $(n_theta) theta-particles...")
    for i in 1:n_theta
        ll_vec[i] = pf_marginal_loglik(rng, cfg, theta_to_params(view(theta_mat, i, :)),
                                       y; n_particles)
    end

    log_w = zeros(n_theta)
    beta = 0.0
    ess_target = ess_target_ratio * n_theta
    history = []

    for step in 1:max_steps
        delta = find_delta_beta(log_w, ll_vec, ess_target, 1.0 - beta)
        beta_new = beta + delta
        log_w .+= delta .* ll_vec
        push!(history, (step = step, beta = beta_new, delta = delta,
                        ess = ess_of(log_w)))
        println("  step $(step)  beta=$(round(beta_new; digits=4))  delta=$(round(delta; digits=4))  ESS=$(round(ess_of(log_w); digits=1))")

        # Resample if ESS below target.
        if ess_of(log_w) < ess_target
            idx = multinomial_resample(rng, log_w)
            theta_mat = theta_mat[idx, :]
            ll_vec = ll_vec[idx]
            log_w .= 0.0
        end

        if beta_new >= 0.9999
            beta = beta_new
            break
        end

        # Move step: K MH iterations per theta-particle.
        cov_now = cov(theta_mat; dims = 1) + 1e-8 * I
        prop_cov = (proposal_scale^2) * cov_now
        acc = mh_move!(rng, cfg, y, theta_mat, ll_vec, beta_new, prop_cov,
                       n_particles, n_mh_per_step)
        println("    MH accept = $(round(acc; digits=3))")

        beta = beta_new
    end

    # Final weighted sample.
    w = exp.(log_w .- logsumexp(log_w))
    return (theta = theta_mat, weights = w, ll = ll_vec, history = history)
end

function weighted_quantile(x::AbstractVector, w::AbstractVector, q::Real)
    idx = sortperm(x)
    xs = x[idx]; ws = w[idx]
    cw = cumsum(ws) ./ sum(ws)
    j = findfirst(>=(q), cw)
    return xs[j]
end

function main()
    cfg = default_config()
    T = 120
    ds = simulate(MersenneTwister(2), cfg, T;
                  log_tau_R = -4.0, log_tau_F = -12.0, log_phi = 2.5)

    n_theta = 128
    n_particles = 600
    println("running SMC sampler: T=$T n_theta=$n_theta n_particles=$n_particles")

    out = smc_sampler(MersenneTwister(11), cfg, ds.y;
                      n_theta, n_particles,
                      ess_target_ratio = 0.5,
                      n_mh_per_step = 2,
                      proposal_scale = 0.5)

    names_truth = [("log_tau_R", -4.0), ("log_tau_F", -12.0), ("log_phi", 2.5)]
    w = out.weights
    println("\nSMC posterior (weighted) summary:")
    for (i, (nm, tru)) in enumerate(names_truth)
        col = out.theta[:, i]
        m = sum(col .* w)
        v = sum(((col .- m) .^ 2) .* w)
        sd = sqrt(v)
        q05 = weighted_quantile(col, w, 0.05)
        q95 = weighted_quantile(col, w, 0.95)
        println("  $(nm) = $(round(m; digits=3)) +/- $(round(sd; digits=3))  [90% $(round(q05; digits=3)), $(round(q95; digits=3))]  (truth $(tru))")
    end

    plots = []
    for (i, (nm, tru)) in enumerate(names_truth)
        col = out.theta[:, i]
        h = histogram(col; weights = w, bins = 30, alpha = 0.75, color = 4,
                      label = "SMC posterior", title = nm,
                      legend = i == 1 ? :topright : false)
        vline!(h, [tru]; color = :black, linestyle = :dash, linewidth = 1.4,
               label = "truth")
        push!(plots, h)
    end

    betas = [h.beta for h in out.history]
    ess_trace = [h.ess for h in out.history]
    p_beta = plot(1:length(betas), betas; xlabel = "tempering step",
                  ylabel = "beta", color = 1, legend = false,
                  title = "Adaptive tempering schedule")
    p_ess  = plot(1:length(ess_trace), ess_trace; xlabel = "tempering step",
                  ylabel = "ESS post-reweight", color = 2, legend = false,
                  title = "ESS over tempering steps")
    hline!(p_ess, [0.5 * n_theta]; color = :black, linestyle = :dash,
           label = "target")

    plt = plot(p_beta, p_ess, plots[1], plots[2], plots[3];
               layout = @layout([a b; c d e]),
               size = (1200, 750),
               plot_title = "SMC sampler with adaptive tempering on Model A — PF marginal inner")
    out_png = joinpath(@__DIR__, "posterior.png")
    savefig(plt, out_png)
    println("saved: $(out_png)")
end

main()
