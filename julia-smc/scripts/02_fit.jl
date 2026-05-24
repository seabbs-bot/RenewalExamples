using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

import ReverseDiff, MCMCChains
include(joinpath(@__DIR__, "..", "src", "ModelA.jl"))
using .ModelA
using Random
using Statistics
using StatsPlots
using Turing
using Turing: AutoReverseDiff, InitFromPrior

function _draw_namedtuple(chn::MCMCChains.Chains, T::Int, draw_idx::Int)
    eps_R = [chn[draw_idx, Symbol("eps_R[$t]"), 1] for t in 1:T]
    eta_R = [chn[draw_idx, Symbol("eta_R[$t]"), 1] for t in 1:T]
    eps_F = [chn[draw_idx, Symbol("eps_F[$t]"), 1] for t in 1:T]
    eta_F = [chn[draw_idx, Symbol("eta_F[$t]"), 1] for t in 1:T]
    return (
        log_tau_R = chn[draw_idx, :log_tau_R, 1],
        log_tau_F = chn[draw_idx, :log_tau_F, 1],
        log_phi = chn[draw_idx, :log_phi, 1],
        log_Rt0 = chn[draw_idx, :log_Rt0, 1],
        log_sigma_R0 = chn[draw_idx, :log_sigma_R0, 1],
        log_F0 = chn[draw_idx, :log_F0, 1],
        log_sigma_F0 = chn[draw_idx, :log_sigma_F0, 1],
        log_I0 = chn[draw_idx, :log_I0, 1],
        eps_R = eps_R, eta_R = eta_R,
        eps_F = eps_F, eta_F = eta_F,
    )
end

function main()
    cfg = default_config()
    T = 120
    ds = simulate(MersenneTwister(2), cfg, T;
                  log_tau_R = -4.0, log_tau_F = -12.0, log_phi = 2.5)

    n_warmup = 300
    n_samples = 300
    println("fitting NUTS: T=$T warmup=$n_warmup samples=$n_samples")

    mdl = model_a(ds.y, cfg)
    chn = sample(
        MersenneTwister(11), mdl,
        NUTS(n_warmup, 0.8; adtype = AutoReverseDiff(compile = false)),
        n_samples;
        progress = false,
        chain_type = MCMCChains.Chains,
        initial_params = InitFromPrior(),
    )

    println("\nposterior means (truth in parens):")
    for (name, truth) in [(:log_tau_R, -4.0), (:log_tau_F, -12.0),
                          (:log_phi, 2.5)]
        m = mean(chn[name])
        s = std(chn[name])
        println("  $name = $(round(m; digits=3)) +/- $(round(s; digits=3))",
                "  (truth $(truth))")
    end

    n_draw = size(chn, 1)
    traj_mat = Matrix{Float64}(undef, T, n_draw)
    for i in 1:n_draw
        d = _draw_namedtuple(chn, T, i)
        traj_mat[:, i] = model_a_logRt_trajectory(d, cfg, T)
    end

    q05 = [quantile(traj_mat[t, :], 0.05) for t in 1:T]
    q50 = [quantile(traj_mat[t, :], 0.50) for t in 1:T]
    q95 = [quantile(traj_mat[t, :], 0.95) for t in 1:T]
    in_band = sum((ds.log_Rt .>= q05) .& (ds.log_Rt .<= q95)) / T
    println("\nlog_Rt 90% credible-band coverage: ",
            round(in_band; digits = 3))

    t = 1:T
    p_traj = plot(t, q50; ribbon = (q50 .- q05, q95 .- q50),
                  color = 1, fillalpha = 0.2, label = "posterior 90%",
                  ylabel = "log Rt", xlabel = "day",
                  title = "log Rt(t) -- smoothed posterior band vs truth")
    plot!(p_traj, t, ds.log_Rt; color = :black, linewidth = 1.4,
          label = "truth")

    names_truth = [(:log_tau_R, -4.0), (:log_tau_F, -12.0), (:log_phi, 2.5)]
    hists = []
    for (i, (name, truth)) in enumerate(names_truth)
        h = histogram(vec(chn[name].data); bins = 30, color = 3, alpha = 0.75,
                      label = "posterior", title = string(name),
                      legend = i == 1 ? :topright : false)
        vline!(h, [truth]; color = :black, linestyle = :dash, linewidth = 1.2,
               label = "truth")
        push!(hists, h)
    end

    plt = plot(p_traj, hists[1], hists[2], hists[3];
               layout = @layout([a{0.55h}; b c d]),
               size = (1100, 800),
               plot_title = "Turing NUTS fit on Model A synthetic data (T=$T)")
    out = joinpath(@__DIR__, "..", "02_fit.png")
    savefig(plt, out)
    println("saved: $out")
end

main()
