using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "ModelA.jl"))
using .ModelA
using Random
using Statistics
using Distributions
using StatsPlots
using Turing
using LinearAlgebra: dot
using MCMCChains

# Variant: SMC for the latent path (via Turing's Particle Gibbs, which uses
# AdvancedPS bootstrap-PF primitives under the hood), NUTS for the static
# parameters. The canonical "off-the-shelf SMC + NUTS for statics" Julia
# pipeline that works with the renewal dynamics today.
#
# Why this path and not GenFilters v0.5 `ParticleGibbs(ConditionalSMC, NUTS)`:
# the GF v0.5 CSMC sampler calls `SSMProblems.distribution(dyn, step, state)`
# — a transition density. Model A's state-to-state map is deterministic in
# 14 of 18 state components (the I_buf shifts deterministically; only the
# four innovations are stochastic), so the transition density is singular on
# a 4-D manifold in 18-D state space. `src/ModelA.jl` documents the failed
# attempt and keeps the flat-vector wrappers around for bootstrap-filter use.
#
# The inner dynamics call here reuses `ModelA.step_state` so the renewal
# step is defined exactly once in the codebase, shared with `pmmh/run.jl`
# and `smc2/run.jl`.

@model function gibbs_modelA(y, cfg, ::Type{ET} = Float64) where {ET}
    log_tau_R ~ Normal(cfg.prior_log_tau_R_mean, cfg.prior_log_tau_R_sd)
    log_tau_F ~ Normal(cfg.prior_log_tau_F_mean, cfg.prior_log_tau_F_sd)
    log_phi   ~ Normal(cfg.prior_log_phi_mean,   cfg.prior_log_phi_sd)

    log_Rt0      ~ Normal(cfg.init_log_Rt_mean,      cfg.init_log_Rt_sd)
    log_sigma_R0 ~ Normal(cfg.init_log_sigma_R_mean, cfg.init_log_sigma_R_sd)
    log_F0       ~ Normal(cfg.init_log_F_mean,       cfg.init_log_F_sd)
    log_sigma_F0 ~ Normal(cfg.init_log_sigma_F_mean, cfg.init_log_sigma_F_sd)
    log_I0       ~ Normal(cfg.init_log_I0_mean,      cfg.init_log_I0_sd)

    T = length(y)
    L = buffer_len(cfg)
    g_pad = ModelA._pad_pmf(cfg.generation_interval, L)
    d_pad = ModelA._pad_pmf(cfg.delay_pmf, L)

    phi = exp(log_phi)
    state = (log_Rt = log_Rt0, log_sigma_R = log_sigma_R0,
             log_F = log_F0, log_sigma_F = log_sigma_F0,
             log_I0 = log_I0, I_buf = fill(exp(log_I0), L))

    eps_R = Vector{ET}(undef, T); eta_R = Vector{ET}(undef, T)
    eps_F = Vector{ET}(undef, T); eta_F = Vector{ET}(undef, T)

    for t in 1:T
        eps_R[t] ~ Normal()
        eta_R[t] ~ Normal()
        eps_F[t] ~ Normal()
        eta_F[t] ~ Normal()
        noise = (eps_R = eps_R[t], eta_R = eta_R[t],
                 eps_F = eps_F[t], eta_F = eta_F[t])
        state = step_state(state, log_tau_R, log_tau_F, noise, cfg, g_pad)
        mu_t = expected_observation(state.I_buf, cfg, d_pad)
        y[t] ~ negbin2(mu_t, phi)
    end
end

function main()
    cfg = default_config()
    T = 120
    ds = simulate(MersenneTwister(2), cfg, T;
                  log_tau_R = -4.0, log_tau_F = -12.0, log_phi = 2.5)

    n_particles = 200
    n_iter = 600
    n_adapts = 150
    println("running Gibbs(PG, NUTS): T=$T n_particles=$n_particles n_iter=$n_iter")

    rng = MersenneTwister(11)
    model = gibbs_modelA(ds.y, cfg)
    static_syms = (:log_tau_R, :log_tau_F, :log_phi,
                   :log_Rt0, :log_sigma_R0, :log_F0, :log_sigma_F0, :log_I0)
    latent_syms = (:eps_R, :eta_R, :eps_F, :eta_F)

    sampler = Gibbs(
        static_syms => NUTS(n_adapts, 0.8),
        latent_syms => PG(n_particles),
    )

    chn = sample(rng, model, sampler, n_iter;
                 progress = false, chain_type = MCMCChains.Chains)

    names_truth = [(:log_tau_R, -4.0), (:log_tau_F, -12.0), (:log_phi, 2.5)]
    println("\nGibbs(PG, NUTS) posterior means:")
    for (nm, tru) in names_truth
        m = mean(chn[nm]); s = std(chn[nm])
        println("  $(nm) = $(round(m; digits=3)) +/- $(round(s; digits=3))  (truth $(tru))")
    end

    plots = []
    for (i, (nm, tru)) in enumerate(names_truth)
        h = histogram(vec(chn[nm].data); bins = 30, alpha = 0.75, color = 5,
                      label = "Gibbs(PG, NUTS)", title = string(nm),
                      legend = i == 1 ? :topright : false)
        vline!(h, [tru]; color = :black, linestyle = :dash, linewidth = 1.4,
               label = "truth")
        push!(plots, h)
    end
    p_trace = plot(vec(chn[:log_tau_R].data); color = 1,
                   xlabel = "iter", ylabel = "log_tau_R",
                   title = "log_tau_R trace", legend = false)
    hline!(p_trace, [-4.0]; color = :black, linestyle = :dash)

    plt = plot(p_trace, plots[1], plots[2], plots[3];
               layout = @layout([a{0.5h}; b c d]),
               size = (1100, 750),
               plot_title = "Turing Gibbs(PG, NUTS) on Model A — SMC latents, NUTS statics")
    out_png = joinpath(@__DIR__, "posterior.png")
    savefig(plt, out_png)
    println("saved: $(out_png)")
end

main()
