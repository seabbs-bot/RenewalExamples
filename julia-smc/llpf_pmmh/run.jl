using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "ModelA.jl"))
using .ModelA
using Random
using Statistics
using Distributions
using StatsPlots

# LLPF analogue of pmmh/. Same MH loop, same target posterior; the inner
# bootstrap PF is LowLevelParticleFilters.AdvancedParticleFilter instead of
# GeneralisedFilters.BF. theta extends to 4 dimensions because the LLPF
# initial distribution is fixed-form MvNormal (no per-particle latent
# initial-state draw), so log_I0 is sampled explicitly.

const PRIOR_MEAN = [-4.0, -12.0, 2.3, 2.3]
const PRIOR_SD   = [0.3, 0.5, 0.5, 1.0]
const PRIOR_NAMES = (:log_tau_R, :log_tau_F, :log_phi, :log_I0)

function log_prior(theta::AbstractVector)
    s = 0.0
    for i in 1:length(PRIOR_MEAN)
        s += logpdf(Normal(PRIOR_MEAN[i], PRIOR_SD[i]), theta[i])
    end
    return s
end

theta_to_params(theta) = ModelAParamsFull(theta[1], theta[2], theta[3], theta[4])

function pmmh(rng::AbstractRNG, cfg::ModelConfig, y::AbstractVector,
              theta0::AbstractVector;
              n_iter::Int, n_particles::Int,
              proposal_sd::AbstractVector)
    d = length(theta0)
    theta = copy(theta0)
    params = theta_to_params(theta)
    ll = llpf_marginal_loglik(cfg, params, y; n_particles)
    lp = log_prior(theta)

    chain = Matrix{Float64}(undef, n_iter, d)
    accepts = 0

    for it in 1:n_iter
        proposal = theta .+ proposal_sd .* randn(rng, d)
        params_p = theta_to_params(proposal)
        ll_p = llpf_marginal_loglik(cfg, params_p, y; n_particles)
        lp_p = log_prior(proposal)

        log_alpha = (ll_p + lp_p) - (ll + lp)
        if log(rand(rng)) < log_alpha
            theta = proposal; ll = ll_p; lp = lp_p
            accepts += 1
        end
        chain[it, :] .= theta
        if it % 100 == 0
            println("  iter $(it)/$(n_iter)  accept=$(round(accepts/it; digits=3))  ll=$(round(ll; digits=2))")
        end
    end
    return (chain = chain, accept_rate = accepts / n_iter)
end

function main()
    cfg = default_config()
    T = 120
    ds = ModelA.simulate(MersenneTwister(2), cfg, T;
                         log_tau_R = -4.0, log_tau_F = -12.0, log_phi = 2.5)

    n_iter = 1500
    n_particles = 1500
    proposal_sd = [0.10, 0.20, 0.07, 0.15]
    theta0 = copy(PRIOR_MEAN)

    println("running PMMH (LLPF inner): T=$T n_iter=$n_iter n_particles=$n_particles")
    out = pmmh(MersenneTwister(11), cfg, ds.y, theta0;
               n_iter, n_particles, proposal_sd)

    burn = n_iter ÷ 3
    post = out.chain[(burn+1):end, :]
    println("\nPMMH (LLPF) posterior means (post-burn=$burn, accept=$(round(out.accept_rate; digits=3))):")
    truths = [-4.0, -12.0, 2.5, ds.initial_state.log_I0]
    for i in 1:4
        m = mean(post[:, i]); s = std(post[:, i])
        println("  $(PRIOR_NAMES[i]) = $(round(m; digits=3)) +/- $(round(s; digits=3))  (truth $(round(truths[i]; digits=3)))")
    end

    plots = []
    main3 = [(1, "log_tau_R", -4.0), (2, "log_tau_F", -12.0), (3, "log_phi", 2.5)]
    for (i, nm, tru) in main3
        h = histogram(post[:, i]; bins = 30, alpha = 0.75, color = 2,
                      label = "PMMH (LLPF)", title = nm,
                      legend = i == 1 ? :topright : false)
        vline!(h, [tru]; color = :black, linestyle = :dash, linewidth = 1.4,
               label = "truth")
        push!(plots, h)
    end
    p_trace = plot(out.chain[:, 1]; color = 1,
                   xlabel = "iter", ylabel = "log_tau_R",
                   title = "log_tau_R trace", legend = false)
    hline!(p_trace, [-4.0]; color = :black, linestyle = :dash)

    plt = plot(p_trace, plots[1], plots[2], plots[3];
               layout = @layout([a{0.5h}; b c d]),
               size = (1100, 750),
               plot_title = "PMMH on Model A — LowLevelParticleFilters AdvancedPF inner (T=$T, $(n_iter) iter)")
    out_png = joinpath(@__DIR__, "posterior.png")
    savefig(plt, out_png)
    println("saved: $(out_png)")
end

main()
