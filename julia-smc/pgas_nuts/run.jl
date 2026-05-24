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
# parameters. This is the canonical "off-the-shelf SMC + NUTS for statics"
# Julia pipeline.
#
# The released GeneralisedFilters v0.4.2 does not yet expose its
# `ParticleGibbs(ConditionalSMC, NUTS)` sampler (that lives on the main
# branch slated for v0.5). Turing's built-in Gibbs(PG, NUTS) gives the
# same composition pattern today.

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
    tau_R = max(exp(log_tau_R), cfg.sigma_floor)
    tau_F = max(exp(log_tau_F), cfg.sigma_floor)

    log_Rt = log_Rt0
    log_sigma_R = log_sigma_R0
    log_F = log_F0
    log_sigma_F = log_sigma_F0
    I_buf = fill(exp(log_I0), L)

    eps_R = Vector{ET}(undef, T)
    eta_R = Vector{ET}(undef, T)
    eps_F = Vector{ET}(undef, T)
    eta_F = Vector{ET}(undef, T)

    for t in 1:T
        eps_R[t] ~ Normal()
        eta_R[t] ~ Normal()
        eps_F[t] ~ Normal()
        eta_F[t] ~ Normal()

        sigma_R = max(exp(log_sigma_R), cfg.sigma_floor)
        sigma_F = max(exp(log_sigma_F), cfg.sigma_floor)

        log_Rt      = log_Rt      + sigma_R * eps_R[t]
        log_F       = log_F       + sigma_F * eps_F[t]
        log_sigma_R = log_sigma_R + tau_R   * eta_R[t]
        log_sigma_F = log_sigma_F + tau_F   * eta_F[t]

        g_conv_I = dot(g_pad, I_buf)
        F_new = exp(log_F)
        log_Rt_clipped = clamp(log_Rt, -20.0, 20.0)
        exponent = clamp(log_Rt_clipped - F_new * g_conv_I, -20.0, 20.0)
        Rt_eff = exp(exponent)
        I_new = min(Rt_eff * g_conv_I, 1e15)
        I_buf = vcat(I_new, I_buf[1:end-1])

        mu_t = dot(d_pad, I_buf)
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
