# Julia SMC and particle-filter landscape vs `smc-jax`

This note maps the Julia SMC ecosystem onto what `smc-jax/` does for Model A.
Same model, same synthetic data, same target parameters.
The point is to see how close stock Julia tooling gets you and what is still missing.

## What is in `julia-smc/`

- `src/ModelA.jl` — a Julia port of the Model A spec from `smc-jax/`.
  Forward simulator that matches `smc-jax/src/smc_renewal/synthetic.py::simulate` step-for-step.
  Turing `@model` for the same generative process written non-centred over the 4 standard-normal innovation series (`eps_R`, `eta_R`, `eps_F`, `eta_F`) and the static parameters `(log_tau_R, log_tau_F, log_phi)`.
- `scripts/01_synthetic_demo.jl` — equivalent of `examples/01_synthetic_demo.py`.
- `scripts/02_fit.jl` — equivalent of `examples/02_pf_fit.py`, but using off-the-shelf NUTS instead of a Liu-West PF.

It sits in its own subdirectory with its own `Project.toml` so the env does not clash with the existing top-level Julia project that pins `Mooncake`/`Enzyme` for the GI examples.

## Julia inference packages relevant to this model

| Package | What it offers |
|---|---|
| `Turing.jl` | PPL with NUTS, HMC, MH, IS, Gibbs, plus particle-Gibbs samplers (`PG`, `SMC`, `PGAS`). The standard front door for Bayesian inference on a user-written generative model. |
| `AdvancedPS.jl` | Particle MCMC building blocks under Turing — bootstrap PF, particle-Gibbs, PG-AS, conditional SMC. Turing's `SMC`, `PG`, `PGAS` samplers dispatch through it. |
| `ParticleFilters.jl` | Standalone PF library oriented at POMDPs.jl (robotics / control). Bootstrap PF and basic SIR resampling. No PMCMC and no parameter cloud. |
| `SequentialMonteCarlo.jl` | Pure SMC sampler with adaptive tempering, lookahead, and dispatch to multiple resampling schemes. POMDP-flavoured rather than state-space-with-static-theta. |
| `LowLevelParticleFilters.jl` | Engineering-flavoured Kalman / UKF / particle filters with offline-smoothing utilities. Bootstrap PF; no Liu-West, no PMCMC. |
| `StateSpaceInference.jl`, `GeneralisedFilters.jl` | Smaller experimental SSM packages with Kalman / UKF / PF building blocks. Not at a maturity to substitute for the smc-jax PF on this model. |
| `Stheno.jl`, `GaussianProcesses.jl` | Not directly relevant but listed because the model has nested-RW (= GP-like) latent structure. |

## What was used and why

NUTS through Turing, with `AutoReverseDiff(compile=false)` for AD.

Reasoning:

- Model A as written is a continuous joint density over `(log_tau_R, log_tau_F, log_phi)` plus an initial-state vector plus 4 standard-normal innovation series of length T.
  Non-centred parameterisation makes the gradient well-defined everywhere and NUTS handles it directly.
- The point of the exercise is "off-the-shelf".
  NUTS is what Turing gives you for free for a continuous joint problem.
  No custom kernels, no Liu-West, no extension code.
- The particle samplers in Turing (`SMC`, `PG`, `PGAS`) target latent-state models with a small static-parameter Gibbs sweep, but the static parameters `(log_tau_R, log_tau_F, log_phi)` and the 4 long standard-normal innovation series get mixed through the same posterior.
  PG would only update the latent path inside SMC; static `tau` parameters need a separate kernel (MH/HMC) inside Gibbs.
  This stops being "off the shelf" the moment you write that wrapper.
  In practice the long latent path also degenerates PG on this scale (T ≈ 120, ~480 latent innovations), so the gain over NUTS is not obvious.
- `LowLevelParticleFilters.jl` and `ParticleFilters.jl` ship a bootstrap PF and resampling but assume a known dynamics object with static parameters — no built-in cloud over `(log_tau_R, log_tau_F, log_phi)` and no Liu-West jittering.

Result on the synthetic data with `T=120`, NUTS warmup 300 + 300 samples, `InitFromPrior()`:

- `log_tau_R` = -4.04 ± 0.31 (truth -4.0)
- `log_tau_F` = -12.00 ± 0.53 (truth -12.0)
- `log_phi`   =  2.51 ± 0.14 (truth +2.5)

All three within ~1.5% of truth, comparable to the `~5%` recovery the JAX Liu-West PF reports for the same parameters in `smc-jax/README.md`.
The smoothed log_Rt(t) posterior band tracks the ground-truth trajectory closely with ~90% empirical coverage.

Initialisation matters: `InitFromUniform()` (the Turing default) starts at log_F values that make the feedback term blow up the gradient on the first step and adapts the step size to ~1e-8.
`InitFromPrior()` starts in a well-conditioned region and adapts to a sensible step.

## What `smc-jax` does that no off-the-shelf Julia package gives you for free

These are the specific features in `smc-jax/pf/`, `smc-jax/smc2/`, `smc-jax/rolling_origin.py` that have no direct Julia counterpart you can pull in without writing the kernel yourself.

1. **Liu-West shrink-jitter on static parameters.**
   `smc-jax` carries `(log_tau_R, log_tau_F, log_phi)` as a per-particle parameter cloud and at each resample applies the Liu-West shrink (`mean + h * (theta_i - mean)`) plus a jitter of variance `(1 - h^2) * cov(theta)` to keep the marginal posterior moving without losing diversity.
   `AdvancedPS.jl` and `Turing.jl`'s `PG` / `SMC` / `PGAS` samplers do not provide this kernel.
   You can write it as a Gibbs step inside Turing, but at that point it is not off-the-shelf.

2. **Steyn-style fixed-lag resampling.**
   `smc-jax/pf/runner.py`'s `fixed_lag_L` argument restricts the resample permutation to the last `L` steps of state and parameter history, motivated by the observation that data at time `t` are only informative about latent states a few days back due to the reporting delay.
   None of the Julia PF packages implement this; they either resample the cloud only (current step) or assume offline smoothing.

3. **SMC² with a pluggable Gaussian inner filter (UKF, library EKF).**
   `smc-jax/smc2/` is an adaptive-tempered SMC² over `theta` whose inner marginal-likelihood is supplied by a UKF (`smc2.ukf`) or an EKF (`smc2.ekf_cuthbert`).
   Julia has Kalman / UKF (e.g. `LowLevelParticleFilters.jl`, `KalmanFilters.jl`) and you can write SMC² on top of `SequentialMonteCarlo.jl`, but the wiring is not provided by any single package.

4. **Guided / auxiliary-q proposals (Model E territory).**
   The Rao-Blackwellised auxiliary-q Wallenius proposal for the GDM cohort-partition observation in `smc-jax/pf/observation_gdm.py` is custom-written.
   No Julia package supplies it.
   Out of scope here because we are doing Model A only, but it is the biggest gap if the modelling moves to contact-tracing depletion / GDM observation.

5. **Sequential extension (`extend_liu_west`, `rolling_origin_forecast`).**
   `smc-jax` carries the trailing particle cloud across data arrivals and updates rather than refits.
   Turing has no first-class sequential-update API.
   `AdvancedPS.jl` exposes particle containers but there is no `extend` operator that handles the parameter cloud as well as the latent path.

6. **Trajectory smoothing with genealogy tracing in a typed result.**
   `pfjax.particle_smooth` gives backward-traced sample paths from the surviving genealogy.
   Julia PF packages give offline RTS / two-filter smoothers for Kalman variants but no genealogy-tracing trajectory smoother for a bootstrap PF with a parameter cloud.

In short: stock Turing / NUTS gets you the *posterior* on Model A on synthetic data to the same accuracy as the Liu-West PF in `smc-jax`.
What it does not give you is the *online* operation — the sequential update, the Liu-West cloud, fixed-lag resampling, SMC² over `theta`, and the guided proposals you need for Model E.
Those would each be a custom kernel inside a Turing Gibbs scheme or a custom loop on `AdvancedPS.jl` primitives.

## Why NUTS rather than PG on this model

PG / SMC in Turing handles models with a clean latent-state structure and a small static-parameter block updated by an outer Gibbs.
Model A here is naturally written with the latent path as one big block of innovations.
Particle Gibbs on T ≈ 120 steps with continuous latent state would degenerate via the standard ancestor-resampling path collapse, and the static-parameter updates would still need a non-PG kernel.
NUTS on the non-centred parameterisation sidesteps both problems: the innovations are unit normal a priori, the static parameters are continuous, and the gradient through `step_state` is well-defined.

Trying PG honestly here would mean writing the kernel that PG does not give you for free, which contradicts the brief.
The honest description is: Turing's `PG`/`SMC`/`PGAS` did not cover Model A as-written without that custom kernel, so we used NUTS, which did.

## Reproducing

```bash
cd julia-smc
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. scripts/01_synthetic_demo.jl
julia --project=. scripts/02_fit.jl
```

The fit script takes a few minutes on the default `T = 120` setting.
The synthetic demo is fast.
