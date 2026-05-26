using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "ModelA.jl"))
using .ModelA
using Random
using Statistics
using Distributions
using StatsPlots
using LinearAlgebra
using LowLevelParticleFilters

const LLPF = LowLevelParticleFilters

# Variant: UKF marginal log-likelihood (LowLevelParticleFilters,
# augmented form for multiplicative noise) + Metropolis-Hastings on
# theta. Empirically demonstrates the Gaussian-filter blind spot on
# (log_tau_R, log_tau_F) documented in `smc-jax/README.md`: the UKF
# marginal moves with `log_phi` but barely with the tau parameters.
#
# NUTS was the original target ("NUTS for static params"). LLPF's UKF
# implementation has a `Float64(...)` cast in the Cholesky / SimpleMvNormal
# path that blocks all three Julia AD backends we tried:
#   - ForwardDiff:  MethodError: no method matching Float64(::Dual{...})
#   - ReverseDiff:  Converting TrackedReal to Float64 is not defined
#   - Mooncake:     AD has hit a :(jl_get_tls_world_age) ccall
# So the UKF marginal is not directly differentiable through Turing's
# usual paths. MH on theta avoids the AD requirement and demonstrates
# the same blind-spot finding.
#
# State layout matches build_ssm_flat: length 4 + L = 18,
# [log_Rt, log_sigma_R, log_F, log_sigma_F, I_buf...]. Measurement is
# Gaussian-moment-matched against NegBin: mean = mu_t,
# variance = mu_t + mu_t^2 / phi.

const PRIOR_MEAN  = [-4.0, -12.0, 2.3, 2.3]
const PRIOR_SD    = [0.3, 0.5, 0.5, 1.0]
const PRIOR_NAMES = (:log_tau_R, :log_tau_F, :log_phi, :log_I0)

theta_to_params(theta) = ModelAParamsFull(theta[1], theta[2], theta[3], theta[4])

function log_prior(theta::AbstractVector)
    s = 0.0
    for i in 1:length(PRIOR_MEAN)
        s += logpdf(Normal(PRIOR_MEAN[i], PRIOR_SD[i]), theta[i])
    end
    return s
end

function build_ukf(cfg::ModelConfig, params::ModelAParamsFull)
    L = buffer_len(cfg)
    g_pad = ModelA._pad_pmf(cfg.generation_interval, L)
    d_pad = ModelA._pad_pmf(cfg.delay_pmf, L)
    tau_R = max(exp(params.log_tau_R), cfg.sigma_floor)
    tau_F = max(exp(params.log_tau_F), cfg.sigma_floor)
    phi   = exp(params.log_phi)
    I0    = exp(params.log_I0)

    function dyn(x, u, p, t, w)
        sigma_R = max(exp(x[2]), cfg.sigma_floor)
        sigma_F = max(exp(x[4]), cfg.sigma_floor)
        log_Rt_new      = x[1] + sigma_R * w[1]
        log_sigma_R_new = x[2] + tau_R   * w[2]
        log_F_new       = x[3] + sigma_F * w[3]
        log_sigma_F_new = x[4] + tau_F   * w[4]
        g_conv_I = @views dot(g_pad, x[5:end])
        F_new = exp(log_F_new)
        log_Rt_clipped = clamp(log_Rt_new, -20.0, 20.0)
        exponent = clamp(log_Rt_clipped - F_new * g_conv_I, -20.0, 20.0)
        Rt_eff = exp(exponent)
        I_new = min(Rt_eff * g_conv_I, 1e15)
        out = similar(x, promote_type(eltype(x), eltype(w)))
        out[1] = log_Rt_new
        out[2] = log_sigma_R_new
        out[3] = log_F_new
        out[4] = log_sigma_F_new
        out[5] = I_new
        @views out[6:end] .= x[5:end-1]
        return out
    end

    function meas(x, u, p, t, e)
        I_buf = @view x[5:end]
        mu_t = dot(d_pad, I_buf)
        var_t = max(mu_t + mu_t^2 / phi, 1e-6)
        return [mu_t + sqrt(var_t) * e[1]]
    end

    R1 = Matrix{Float64}(LinearAlgebra.I, 4, 4)
    R2 = Matrix{Float64}(LinearAlgebra.I, 1, 1)

    means = vcat(
        [cfg.init_log_Rt_mean, cfg.init_log_sigma_R_mean,
         cfg.init_log_F_mean,  cfg.init_log_sigma_F_mean],
        fill(I0, L),
    )
    vars = vcat(
        [cfg.init_log_Rt_sd^2, cfg.init_log_sigma_R_sd^2,
         cfg.init_log_F_sd^2,  cfg.init_log_sigma_F_sd^2],
        fill(1e-8, L),
    )
    d0 = MvNormal(means, LinearAlgebra.Diagonal(vars))

    return LLPF.UnscentedKalmanFilter{false, false, true, true}(
        dyn, meas, R1, R2, d0; ny = 1, nu = 0)
end

function ukf_marginal_loglik(cfg::ModelConfig, params::ModelAParamsFull,
                             y::AbstractVector)
    ukf = build_ukf(cfg, params)
    T = length(y)
    u_dummy = [Float64[] for _ in 1:T]
    y_vec   = [[Float64(yt)] for yt in y]
    sol = LLPF.forward_trajectory(ukf, u_dummy, y_vec)
    return sol.ll
end

function ukf_mh(rng::AbstractRNG, cfg::ModelConfig, y::AbstractVector,
                theta0::AbstractVector;
                n_iter::Int, proposal_sd::AbstractVector)
    d = length(theta0)
    theta = copy(theta0)
    params = theta_to_params(theta)
    ll = ukf_marginal_loglik(cfg, params, y)
    lp = log_prior(theta)
    chain = Matrix{Float64}(undef, n_iter, d)
    accepts = 0
    for it in 1:n_iter
        proposal = theta .+ proposal_sd .* randn(rng, d)
        ll_p = ukf_marginal_loglik(cfg, theta_to_params(proposal), y)
        lp_p = log_prior(proposal)
        if log(rand(rng)) < (ll_p + lp_p) - (ll + lp)
            theta = proposal; ll = ll_p; lp = lp_p
            accepts += 1
        end
        chain[it, :] .= theta
        if it % 200 == 0
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

    println("smoke-check UKF marginal log-lik at truth:")
    truth_params = ModelAParamsFull(-4.0, -12.0, 2.5, ds.initial_state.log_I0)
    ll_truth = ukf_marginal_loglik(cfg, truth_params, ds.y)
    println("  ll(truth) = $(round(ll_truth; digits=2))")

    n_iter = 4000
    proposal_sd = [0.15, 0.30, 0.10, 0.20]
    theta0 = copy(PRIOR_MEAN)
    println("running UKF + MH on theta: T=$T n_iter=$n_iter (NUTS blocked by LLPF AD-trace cast — see header)")
    out = ukf_mh(MersenneTwister(11), cfg, ds.y, theta0;
                 n_iter, proposal_sd)

    burn = n_iter ÷ 3
    post = out.chain[(burn+1):end, :]
    truths = [-4.0, -12.0, 2.5, ds.initial_state.log_I0]
    prior_sds = PRIOR_SD
    println("\nUKF+MH posterior summary (post-burn=$burn, accept=$(round(out.accept_rate; digits=3))):")
    for i in 1:4
        m = mean(post[:, i]); s = std(post[:, i])
        ratio = s / prior_sds[i]
        flag = ratio > 0.7 ? "PRIOR-STUCK" : "identified"
        println("  $(PRIOR_NAMES[i]) = $(round(m; digits=3)) +/- $(round(s; digits=3))  " *
                "(truth $(round(truths[i]; digits=3)); post/prior sd = $(round(ratio; digits=2))) — $(flag)")
    end

    plots = []
    main3 = [(1, "log_tau_R", -4.0, PRIOR_MEAN[1], PRIOR_SD[1]),
             (2, "log_tau_F", -12.0, PRIOR_MEAN[2], PRIOR_SD[2]),
             (3, "log_phi",    2.5, PRIOR_MEAN[3], PRIOR_SD[3])]
    for (i, nm, tru, pm, ps) in main3
        h = histogram(post[:, i]; bins = 30, alpha = 0.65, color = 6,
                      normalize = :pdf, label = "UKF+MH posterior",
                      title = nm, legend = i == 1 ? :topright : false)
        xs = range(pm - 4 * ps, pm + 4 * ps, length = 200)
        plot!(h, xs, pdf.(Normal(pm, ps), xs);
              color = :red, linestyle = :dash, linewidth = 1.6, label = "prior")
        vline!(h, [tru]; color = :black, linestyle = :solid, linewidth = 1.4,
               label = "truth")
        push!(plots, h)
    end
    p_trace = plot(out.chain[:, 1]; color = 1,
                   xlabel = "iter", ylabel = "log_tau_R",
                   title = "log_tau_R trace (UKF+MH)", legend = false)
    hline!(p_trace, [-4.0]; color = :black, linestyle = :dash)

    plt = plot(p_trace, plots[1], plots[2], plots[3];
               layout = @layout([a{0.5h}; b c d]),
               size = (1100, 750),
               plot_title = "UKF + MH on Model A — blind spot on tau, log_phi identified")
    out_png = joinpath(@__DIR__, "posterior.png")
    savefig(plt, out_png)
    println("saved: $(out_png)")
end

main()
