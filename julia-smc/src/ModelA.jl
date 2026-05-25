module ModelA

using Distributions
using GeneralisedFilters
import LinearAlgebra
using LinearAlgebra: I, dot
using LogExpFunctions: logaddexp, logsumexp
using LowLevelParticleFilters: LowLevelParticleFilters
using Random
using SpecialFunctions: loggamma
using SSMProblems
using StatsFuns: normlogpdf

const LLPF = LowLevelParticleFilters

const GF = GeneralisedFilters

export ModelConfig, default_config, discretised_gamma,
       SyntheticDataset, simulate, step_state, sample_initial_state,
       expected_observation, negbin2, buffer_len,
       ModelAParams, build_ssm, pf_marginal_loglik,
       ModelAParamsFull, ModelAPriorFlat, ModelADynamicsFlat,
       ModelAObservationFlat, build_ssm_flat,
       llpf_dynamics_fn, llpf_measurement_likelihood_fn,
       llpf_initial_dist, llpf_dummy_measurement, llpf_marginal_loglik

Base.@kwdef struct ModelConfig
    generation_interval::Vector{Float64}
    delay_pmf::Vector{Float64}
    sigma_floor::Float64 = 1e-4
    init_log_Rt_mean::Float64 = 0.0
    init_log_Rt_sd::Float64 = 0.3
    init_log_sigma_R_mean::Float64 = -3.5
    init_log_sigma_R_sd::Float64 = 0.3
    init_log_F_mean::Float64 = -12.0
    init_log_F_sd::Float64 = 1.0
    init_log_sigma_F_mean::Float64 = -12.0
    init_log_sigma_F_sd::Float64 = 0.5
    init_log_I0_mean::Float64 = 2.3
    init_log_I0_sd::Float64 = 1.0
    prior_log_tau_R_mean::Float64 = -4.0
    prior_log_tau_R_sd::Float64 = 0.3
    prior_log_tau_F_mean::Float64 = -12.0
    prior_log_tau_F_sd::Float64 = 0.5
    prior_log_phi_mean::Float64 = 2.3
    prior_log_phi_sd::Float64 = 0.5
end

buffer_len(cfg::ModelConfig) = max(length(cfg.generation_interval),
                                   length(cfg.delay_pmf))

function discretised_gamma(shape::Real, scale::Real, support::AbstractVector{<:Integer})
    x = Float64.(support)
    log_dens = @. (shape - 1.0) * log(max(x, 1e-12)) - x / scale -
                  shape * log(scale) - loggamma(shape)
    dens = exp.(log_dens)
    dens[x .<= 0] .= 0.0
    return dens ./ sum(dens)
end

default_generation_interval(; max_lag::Int = 14) = discretised_gamma(3.0, 1.5, 1:max_lag)
default_delay_pmf(; max_lag::Int = 14) = discretised_gamma(2.5, 2.5, 1:max_lag)

function default_config()
    return ModelConfig(
        generation_interval = default_generation_interval(),
        delay_pmf = default_delay_pmf(),
    )
end

function _pad_pmf(pmf::AbstractVector, len::Int)
    out = zeros(eltype(pmf), len)
    out[1:length(pmf)] .= pmf
    return out
end

_floor_sigma(log_sigma, floor) = max(exp(log_sigma), floor)

function step_state(state, log_tau_R, log_tau_F, noise, cfg::ModelConfig,
                    g_pad::AbstractVector)
    sigma_R_old = _floor_sigma(state.log_sigma_R, cfg.sigma_floor)
    sigma_F_old = _floor_sigma(state.log_sigma_F, cfg.sigma_floor)
    tau_R = _floor_sigma(log_tau_R, cfg.sigma_floor)
    tau_F = _floor_sigma(log_tau_F, cfg.sigma_floor)

    log_Rt_new = state.log_Rt + sigma_R_old * noise.eps_R
    log_F_new = state.log_F + sigma_F_old * noise.eps_F
    log_sigma_R_new = state.log_sigma_R + tau_R * noise.eta_R
    log_sigma_F_new = state.log_sigma_F + tau_F * noise.eta_F

    g_conv_I = dot(g_pad, state.I_buf)
    F_new = exp(log_F_new)
    log_Rt_clipped = clamp(log_Rt_new, -20.0, 20.0)
    exponent = clamp(log_Rt_clipped - F_new * g_conv_I, -20.0, 20.0)
    Rt_eff = exp(exponent)
    I_new = min(Rt_eff * g_conv_I, 1e15)

    I_buf_new = vcat(I_new, state.I_buf[1:end-1])

    return (log_Rt = log_Rt_new,
            log_sigma_R = log_sigma_R_new,
            log_F = log_F_new,
            log_sigma_F = log_sigma_F_new,
            log_I0 = state.log_I0,
            I_buf = I_buf_new)
end

function expected_observation(I_buf::AbstractVector, cfg::ModelConfig,
                              d_pad::AbstractVector)
    return dot(d_pad, I_buf)
end

function negbin2(mu, phi)
    mu_safe = max(mu, 1e-10)
    p = phi / (mu_safe + phi)
    return NegativeBinomial(phi, p; check_args = false)
end

struct SyntheticDataset
    log_tau_R::Float64
    log_tau_F::Float64
    log_phi::Float64
    log_Rt::Vector{Float64}
    log_sigma_R::Vector{Float64}
    log_F::Vector{Float64}
    log_sigma_F::Vector{Float64}
    infections::Vector{Float64}
    mu_y::Vector{Float64}
    y::Vector{Int}
    initial_state::NamedTuple
end

function sample_initial_state(rng::AbstractRNG, cfg::ModelConfig)
    log_Rt = cfg.init_log_Rt_mean + cfg.init_log_Rt_sd * randn(rng)
    log_sigma_R = cfg.init_log_sigma_R_mean +
                  cfg.init_log_sigma_R_sd * randn(rng)
    log_F = cfg.init_log_F_mean + cfg.init_log_F_sd * randn(rng)
    log_sigma_F = cfg.init_log_sigma_F_mean +
                  cfg.init_log_sigma_F_sd * randn(rng)
    log_I0 = cfg.init_log_I0_mean + cfg.init_log_I0_sd * randn(rng)
    L = buffer_len(cfg)
    I_buf = fill(exp(log_I0), L)
    return (log_Rt = log_Rt, log_sigma_R = log_sigma_R,
            log_F = log_F, log_sigma_F = log_sigma_F,
            log_I0 = log_I0, I_buf = I_buf)
end

function simulate(rng::AbstractRNG, cfg::ModelConfig, T::Int;
                  log_tau_R::Real = -4.0,
                  log_tau_F::Real = -12.0,
                  log_phi::Real = 2.5,
                  initial_state = nothing)
    state = initial_state === nothing ? sample_initial_state(rng, cfg) :
                                         initial_state
    L = buffer_len(cfg)
    g_pad = _pad_pmf(cfg.generation_interval, L)
    d_pad = _pad_pmf(cfg.delay_pmf, L)

    log_Rt = Vector{Float64}(undef, T)
    log_sigma_R = Vector{Float64}(undef, T)
    log_F = Vector{Float64}(undef, T)
    log_sigma_F = Vector{Float64}(undef, T)
    infections = Vector{Float64}(undef, T)
    mu_y = Vector{Float64}(undef, T)
    y = Vector{Int}(undef, T)

    phi = exp(log_phi)

    for t in 1:T
        noise = (eps_R = randn(rng), eta_R = randn(rng),
                 eps_F = randn(rng), eta_F = randn(rng))
        state = step_state(state, log_tau_R, log_tau_F, noise, cfg, g_pad)
        log_Rt[t] = state.log_Rt
        log_sigma_R[t] = state.log_sigma_R
        log_F[t] = state.log_F
        log_sigma_F[t] = state.log_sigma_F
        infections[t] = state.I_buf[1]
        mu_y[t] = expected_observation(state.I_buf, cfg, d_pad)
        y[t] = rand(rng, negbin2(mu_y[t], phi))
    end

    return SyntheticDataset(log_tau_R, log_tau_F, log_phi,
                            log_Rt, log_sigma_R, log_F, log_sigma_F,
                            infections, mu_y, y, state)
end

# --- SSMProblems / GeneralisedFilters integration ---

struct ModelAParams
    log_tau_R::Float64
    log_tau_F::Float64
    log_phi::Float64
end

struct ModelAPrior <: StatePrior
    cfg::ModelConfig
end

# We define `simulate` directly rather than `distribution` because the initial
# state mixes continuous scalars with a deterministically-derived buffer.
function SSMProblems.simulate(rng::AbstractRNG, prior::ModelAPrior; kwargs...)
    return sample_initial_state(rng, prior.cfg)
end

struct ModelADynamics{P} <: LatentDynamics
    cfg::ModelConfig
    params::P
    g_pad::Vector{Float64}
end

ModelADynamics(cfg::ModelConfig, params::ModelAParams) =
    ModelADynamics{ModelAParams}(cfg, params, _pad_pmf(cfg.generation_interval, buffer_len(cfg)))

function SSMProblems.simulate(rng::AbstractRNG, dyn::ModelADynamics,
                              step::Integer, state; kwargs...)
    noise = (eps_R = randn(rng), eta_R = randn(rng),
             eps_F = randn(rng), eta_F = randn(rng))
    return step_state(state, dyn.params.log_tau_R, dyn.params.log_tau_F,
                      noise, dyn.cfg, dyn.g_pad)
end

struct ModelAObservation{P} <: ObservationProcess
    cfg::ModelConfig
    params::P
    d_pad::Vector{Float64}
end

ModelAObservation(cfg::ModelConfig, params::ModelAParams) =
    ModelAObservation{ModelAParams}(cfg, params, _pad_pmf(cfg.delay_pmf, buffer_len(cfg)))

function SSMProblems.distribution(obs::ModelAObservation, step::Integer, state;
                                  kwargs...)
    mu_t = expected_observation(state.I_buf, obs.cfg, obs.d_pad)
    return negbin2(mu_t, exp(obs.params.log_phi))
end

function build_ssm(cfg::ModelConfig, params::ModelAParams)
    return StateSpaceModel(
        ModelAPrior(cfg),
        ModelADynamics(cfg, params),
        ModelAObservation(cfg, params),
    )
end

function pf_marginal_loglik(rng::AbstractRNG, cfg::ModelConfig,
                            params::ModelAParams, y::AbstractVector;
                            n_particles::Int = 2000)
    model = build_ssm(cfg, params)
    _, ll = GF.filter(rng, model, GF.BF(n_particles), y)
    return ll
end

# --- Flat-vector SSM for GeneralisedFilters v0.5 PGAS (partial) ---
#
# `SSMTrajectory` and the v0.5 ConditionalSMC sampler expect a flat-vector
# state and a `Distribution`-valued initial prior. The existing NamedTuple
# state cannot express a `Distribution` directly because `I_buf` is
# deterministic in `log_I0`. We lift `log_I0` into a parameter rather than
# a state, treat `I_buf` as near-Dirac at `exp(log_I0)` in the initial
# prior (variance 1e-16 keeps PF maths well-defined while keeping the
# initial buffer ~exactly at `exp(log_I0)`), and represent the state as
# a length `4 + L` vector with layout
# `[log_Rt, log_sigma_R, log_F, log_sigma_F; I_buf...]`.
#
# This is enough for a bootstrap filter (only `simulate` needed). It is
# NOT enough for v0.5 `ConditionalSMC`, which also calls
# `SSMProblems.distribution(dyn, step, state)` — a transition density.
# Model A's transition is deterministic in 14 of 18 state components
# (`I_buf` shifts deterministically; only the four innovations are
# stochastic), so the transition density is singular on a 4-D manifold
# in 18-D state space. A workable `distribution` would be a singular
# Gaussian and the IS weights for CSMC are then ill-defined in the usual
# sense.
#
# The wrappers below are kept for completeness — they let you run a
# bootstrap filter via `build_ssm_flat`. They do NOT make PGAS via
# `ParticleGibbs(ConditionalSMC, NUTS)` work on Model A as-is.

struct ModelAParamsFull
    log_tau_R::Float64
    log_tau_F::Float64
    log_phi::Float64
    log_I0::Float64
end

struct ModelAPriorFlat <: StatePrior
    cfg::ModelConfig
    log_I0::Float64
end

function SSMProblems.distribution(prior::ModelAPriorFlat; kwargs...)
    cfg = prior.cfg
    L = buffer_len(cfg)
    I0 = exp(prior.log_I0)
    means = vcat(
        [cfg.init_log_Rt_mean, cfg.init_log_sigma_R_mean,
         cfg.init_log_F_mean,  cfg.init_log_sigma_F_mean],
        fill(I0, L),
    )
    vars = vcat(
        [cfg.init_log_Rt_sd^2, cfg.init_log_sigma_R_sd^2,
         cfg.init_log_F_sd^2,  cfg.init_log_sigma_F_sd^2],
        fill(1e-16, L),
    )
    return MvNormal(means, LinearAlgebra.Diagonal(vars))
end

struct ModelADynamicsFlat{P} <: LatentDynamics
    cfg::ModelConfig
    params::P
    g_pad::Vector{Float64}
end

ModelADynamicsFlat(cfg::ModelConfig, params::ModelAParamsFull) =
    ModelADynamicsFlat{ModelAParamsFull}(cfg, params,
                                         _pad_pmf(cfg.generation_interval,
                                                  buffer_len(cfg)))

function SSMProblems.simulate(rng::AbstractRNG, dyn::ModelADynamicsFlat,
                              step::Integer, state::AbstractVector; kwargs...)
    cfg = dyn.cfg
    L = buffer_len(cfg)
    log_Rt      = state[1]
    log_sigma_R = state[2]
    log_F       = state[3]
    log_sigma_F = state[4]
    I_buf       = @view state[5:end]

    sigma_R = _floor_sigma(log_sigma_R, cfg.sigma_floor)
    sigma_F = _floor_sigma(log_sigma_F, cfg.sigma_floor)
    tau_R   = _floor_sigma(dyn.params.log_tau_R, cfg.sigma_floor)
    tau_F   = _floor_sigma(dyn.params.log_tau_F, cfg.sigma_floor)

    eps_R = randn(rng); eta_R = randn(rng)
    eps_F = randn(rng); eta_F = randn(rng)

    log_Rt_new      = log_Rt      + sigma_R * eps_R
    log_F_new       = log_F       + sigma_F * eps_F
    log_sigma_R_new = log_sigma_R + tau_R   * eta_R
    log_sigma_F_new = log_sigma_F + tau_F   * eta_F

    g_conv_I = dot(dyn.g_pad, I_buf)
    F_new = exp(log_F_new)
    log_Rt_clipped = clamp(log_Rt_new, -20.0, 20.0)
    exponent = clamp(log_Rt_clipped - F_new * g_conv_I, -20.0, 20.0)
    Rt_eff = exp(exponent)
    I_new = min(Rt_eff * g_conv_I, 1e15)

    new = similar(state)
    new[1] = log_Rt_new
    new[2] = log_sigma_R_new
    new[3] = log_F_new
    new[4] = log_sigma_F_new
    new[5] = I_new
    @views new[6:end] .= I_buf[1:end-1]
    return new
end

struct ModelAObservationFlat{P} <: ObservationProcess
    cfg::ModelConfig
    params::P
    d_pad::Vector{Float64}
end

ModelAObservationFlat(cfg::ModelConfig, params::ModelAParamsFull) =
    ModelAObservationFlat{ModelAParamsFull}(cfg, params,
                                            _pad_pmf(cfg.delay_pmf,
                                                     buffer_len(cfg)))

function SSMProblems.distribution(obs::ModelAObservationFlat, step::Integer,
                                  state::AbstractVector; kwargs...)
    I_buf = @view state[5:end]
    mu_t = dot(obs.d_pad, I_buf)
    return negbin2(mu_t, exp(obs.params.log_phi))
end

function build_ssm_flat(cfg::ModelConfig, params::ModelAParamsFull)
    return StateSpaceModel(
        ModelAPriorFlat(cfg, params.log_I0),
        ModelADynamicsFlat(cfg, params),
        ModelAObservationFlat(cfg, params),
    )
end

# --- LowLevelParticleFilters wiring ---
#
# AdvancedParticleFilter handles non-additive (multiplicative) dynamics
# noise, which is what Model A's nested-RW structure needs. Reuses the
# same flat-Vector state as the GenFilters wrappers.

function llpf_dynamics_fn(cfg::ModelConfig, params::ModelAParamsFull)
    L = buffer_len(cfg)
    g_pad = _pad_pmf(cfg.generation_interval, L)
    tau_R = _floor_sigma(params.log_tau_R, cfg.sigma_floor)
    tau_F = _floor_sigma(params.log_tau_F, cfg.sigma_floor)
    function dyn(x, u, p, t, noise = false)
        sigma_R = _floor_sigma(x[2], cfg.sigma_floor)
        sigma_F = _floor_sigma(x[4], cfg.sigma_floor)
        eps_R = randn(); eta_R = randn()
        eps_F = randn(); eta_F = randn()
        log_Rt_new      = x[1] + sigma_R * eps_R
        log_sigma_R_new = x[2] + tau_R   * eta_R
        log_F_new       = x[3] + sigma_F * eps_F
        log_sigma_F_new = x[4] + tau_F   * eta_F
        g_conv_I = @views dot(g_pad, x[5:end])
        F_new = exp(log_F_new)
        log_Rt_clipped = clamp(log_Rt_new, -20.0, 20.0)
        exponent = clamp(log_Rt_clipped - F_new * g_conv_I, -20.0, 20.0)
        Rt_eff = exp(exponent)
        I_new = min(Rt_eff * g_conv_I, 1e15)
        out = similar(x)
        out[1] = log_Rt_new
        out[2] = log_sigma_R_new
        out[3] = log_F_new
        out[4] = log_sigma_F_new
        out[5] = I_new
        @views out[6:end] .= x[5:end-1]
        return out
    end
    return dyn
end

function llpf_measurement_likelihood_fn(cfg::ModelConfig,
                                        params::ModelAParamsFull)
    L = buffer_len(cfg)
    d_pad = _pad_pmf(cfg.delay_pmf, L)
    phi = exp(params.log_phi)
    function mll(x, u, y, p, t)
        I_buf = @view x[5:end]
        mu_t = dot(d_pad, I_buf)
        return logpdf(negbin2(mu_t, phi), y[1])
    end
    return mll
end

function llpf_initial_dist(cfg::ModelConfig, params::ModelAParamsFull)
    L = buffer_len(cfg)
    I0 = exp(params.log_I0)
    means = vcat(
        [cfg.init_log_Rt_mean, cfg.init_log_sigma_R_mean,
         cfg.init_log_F_mean,  cfg.init_log_sigma_F_mean],
        fill(I0, L),
    )
    vars = vcat(
        [cfg.init_log_Rt_sd^2, cfg.init_log_sigma_R_sd^2,
         cfg.init_log_F_sd^2,  cfg.init_log_sigma_F_sd^2],
        fill(1e-16, L),
    )
    return Distributions.MvNormal(means, LinearAlgebra.Diagonal(vars))
end

function llpf_dummy_measurement(x, u, p, t, noise = false)
    return [0.0]
end

function llpf_marginal_loglik(cfg::ModelConfig, params::ModelAParamsFull,
                              y::AbstractVector; n_particles::Int = 1500)
    dyn = llpf_dynamics_fn(cfg, params)
    mll = llpf_measurement_likelihood_fn(cfg, params)
    d0  = llpf_initial_dist(cfg, params)
    pf  = LLPF.AdvancedParticleFilter(n_particles, dyn, llpf_dummy_measurement,
                                       mll, nothing, d0)
    T = length(y)
    u_dummy = [Float64[] for _ in 1:T]
    y_vec   = [[Float64(yt)] for yt in y]
    sol = LLPF.forward_trajectory(pf, u_dummy, y_vec)
    return sol.ll
end

end # module
