using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "ModelA.jl"))
using .ModelA
using Random
using Statistics
using Distributions
using StatsPlots

# PMMH: Metropolis-Hastings on theta = (log_tau_R, log_tau_F, log_phi) with a
# bootstrap-PF marginal log-likelihood from GeneralisedFilters supplying the
# acceptance ratio. Standard Andrieu-Doucet-Holenstein pseudo-marginal MCMC.

const PRIOR_MEAN = [-4.0, -12.0, 2.3]
const PRIOR_SD   = [0.3, 0.5, 0.5]

function log_prior(theta::AbstractVector)
    s = 0.0
    for i in 1:3
        s += logpdf(Normal(PRIOR_MEAN[i], PRIOR_SD[i]), theta[i])
    end
    return s
end

theta_to_params(theta) = ModelAParams(theta[1], theta[2], theta[3])

function pmmh(rng::AbstractRNG, cfg::ModelConfig, y::AbstractVector,
              theta0::AbstractVector;
              n_iter::Int, n_particles::Int,
              proposal_sd::AbstractVector,
              warmup::Int = 0)
    d = length(theta0)
    theta = copy(theta0)
    params = theta_to_params(theta)
    ll = pf_marginal_loglik(rng, cfg, params, y; n_particles)
    lp = log_prior(theta)

    chain = Matrix{Float64}(undef, n_iter, d)
    lls = Vector{Float64}(undef, n_iter)
    accepts = 0

    for it in 1:n_iter
        proposal = theta .+ proposal_sd .* randn(rng, d)
        params_p = theta_to_params(proposal)
        ll_p = pf_marginal_loglik(rng, cfg, params_p, y; n_particles)
        lp_p = log_prior(proposal)

        log_alpha = (ll_p + lp_p) - (ll + lp)
        if log(rand(rng)) < log_alpha
            theta = proposal
            ll = ll_p
            lp = lp_p
            accepts += 1
        end
        chain[it, :] .= theta
        lls[it] = ll
        if it % 100 == 0
            acc = accepts / it
            println("  iter $(it)/$(n_iter)  accept=$(round(acc; digits=3))  ll=$(round(ll; digits=2))")
        end
    end
    return (chain = chain, lls = lls, accept_rate = accepts / n_iter)
end

function main()
    cfg = default_config()
    T = 120
    ds = simulate(MersenneTwister(2), cfg, T;
                  log_tau_R = -4.0, log_tau_F = -12.0, log_phi = 2.5)

    n_iter = 1500
    n_particles = 1500
    proposal_sd = [0.10, 0.20, 0.07]
    theta0 = copy(PRIOR_MEAN)

    println("running PMMH: T=$T n_iter=$n_iter n_particles=$n_particles")
    out = pmmh(MersenneTwister(11), cfg, ds.y, theta0;
               n_iter, n_particles, proposal_sd)

    burn = n_iter ÷ 3
    post = out.chain[(burn+1):end, :]
    names_truth = [("log_tau_R", -4.0), ("log_tau_F", -12.0), ("log_phi", 2.5)]

    println("\nPMMH posterior means (post-burn=$burn, accept=$(round(out.accept_rate; digits=3))):")
    for (i, (nm, tru)) in enumerate(names_truth)
        m = mean(post[:, i]); s = std(post[:, i])
        println("  $(nm) = $(round(m; digits=3)) +/- $(round(s; digits=3))  (truth $(tru))")
    end

    plots = []
    for (i, (nm, tru)) in enumerate(names_truth)
        h = histogram(post[:, i]; bins = 30, alpha = 0.75, color = 2,
                      label = "PMMH posterior", title = nm,
                      legend = i == 1 ? :topright : false)
        vline!(h, [tru]; color = :black, linestyle = :dash, linewidth = 1.4,
               label = "truth")
        push!(plots, h)
    end
    p_trace = plot(1:size(out.chain, 1), out.chain[:, 1]; color = 1,
                   xlabel = "iter", ylabel = "log_tau_R",
                   title = "log_tau_R trace", legend = false)
    hline!(p_trace, [-4.0]; color = :black, linestyle = :dash)

    plt = plot(p_trace, plots[1], plots[2], plots[3];
               layout = @layout([a{0.5h}; b c d]),
               size = (1100, 750),
               plot_title = "PMMH on Model A — bootstrap PF marginal + MH on θ (T=$T, $(n_iter) iter)")
    out_png = joinpath(@__DIR__, "posterior.png")
    savefig(plt, out_png)
    println("saved: $(out_png)")
end

main()
