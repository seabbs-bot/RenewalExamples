using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "ModelA.jl"))
using .ModelA
using Random
using StatsPlots

function main()
    cfg = default_config()
    T = 180
    ds = simulate(MersenneTwister(2), cfg, T;
                  log_tau_R = -4.0, log_tau_F = -12.0, log_phi = 2.5)

    println("T=$T log_Rt range: [",
            round(minimum(ds.log_Rt); digits=2), ", ",
            round(maximum(ds.log_Rt); digits=2), "]")
    println("infections mean=", round(sum(ds.infections)/T; digits=1),
            " peak=", round(maximum(ds.infections); digits=1))
    println("observed y mean=", round(sum(ds.y)/T; digits=1),
            " peak=", maximum(ds.y))

    t = 1:T
    p1 = plot(t, exp.(ds.log_Rt); ylabel = "Rt",
              title = "Effective reproduction number", color = 1, legend = false)
    hline!(p1, [1.0]; color = :black, linestyle = :dash, linewidth = 0.5)

    p2 = plot(t, exp.(ds.log_sigma_R); ylabel = "sigma_R(t)",
              title = "Innovation std of log Rt", color = 2, legend = false)

    p3 = plot(t, exp.(ds.log_F); ylabel = "F(t)", yscale = :log10,
              title = "Infection-feedback strength", color = 3, legend = false)

    p4 = plot(t, exp.(ds.log_sigma_F); ylabel = "sigma_F(t)",
              title = "Innovation std of log F", color = 4, legend = false)

    p5 = plot(t, ds.infections; ylabel = "I(t)", xlabel = "day",
              title = "Latent infections", color = 5, legend = false)

    p6 = plot(t, ds.mu_y; ylabel = "cases", xlabel = "day",
              title = "Observed cases", color = 6,
              label = "mu_y (delay-conv. infections)")
    scatter!(p6, t, ds.y; ms = 2, color = :black, alpha = 0.6, label = "y (NegBin)")

    plt = plot(p1, p2, p3, p4, p5, p6; layout = (3, 2), size = (1100, 750),
               plot_title = "Nested-RW renewal model -- one synthetic ground-truth trajectory")
    out = joinpath(@__DIR__, "..", "01_synthetic_tour.png")
    savefig(plt, out)
    println("saved: $out")
end

main()
