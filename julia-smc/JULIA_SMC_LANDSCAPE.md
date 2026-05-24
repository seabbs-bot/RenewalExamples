# Julia SMC landscape vs `smc-jax`

What is in `julia-smc/` is three SMC-flavoured fits of Model A on the same synthetic dataset, built on the Julia SMC ecosystem (GeneralisedFilters.jl, SSMProblems.jl, AdvancedPS / Turing, a small hand-rolled SMC sampler).
The point is to see what stock Julia tooling gives you on Model A and where you still have to write code or work around structural limits.

An earlier round used Turing + NUTS over the joint posterior of (theta, latent innovations).
That works fine and is a well-known Julia approach, so it is dropped here in favour of three actual SMC variants.

## Three variants in this directory

| Variant | Outer (theta) | Inner (latent path) | Marginal for theta | Uses `src/ModelA.jl` SSMProblems wrappers? |
| --- | --- | --- | --- | --- |
| `pmmh/` | Metropolis-Hastings (manual loop, AdvancedMH-style) | Bootstrap PF, GeneralisedFilters.jl `BF` | PF marginal log-lik (noisy, unbiased) | **Yes** — via `build_ssm` → `StateSpaceModel(prior, dyn, obs)` → `BF` |
| `smc2/` | SMC sampler with adaptive tempering (hand-rolled, ~150 LOC) | Bootstrap PF, GeneralisedFilters.jl `BF` | same PF marginal, called per theta-particle | **Yes** — same path |
| `pgas_nuts/` | NUTS (Turing) | Particle Gibbs (Turing / AdvancedPS bootstrap PF) | Gibbs(PG, NUTS) alternates updates rather than computing an explicit marginal | **No** — Turing's `Gibbs(PG, NUTS)` requires latents declared inline in the `@model`. Calls `ModelA.step_state` for the renewal step so the dynamics live in one place, but does not feed through `build_ssm`. |

All three share `src/ModelA.jl`, which exposes the model in two forms.

- `step_state`, `simulate`, `sample_initial_state`, `negbin2`, etc. — the pure dynamics + forward simulator. Called by every variant.
- `ModelAParams`, `ModelAPrior`, `ModelADynamics`, `ModelAObservation`, `build_ssm`, `pf_marginal_loglik` — SSMProblems.jl wrappers used by PMMH and SMC².
- `ModelAParamsFull`, `ModelAPriorFlat`, `ModelADynamicsFlat`, `ModelAObservationFlat`, `build_ssm_flat` — flat-Vector versions added to attempt the GenFilters v0.5 `ParticleGibbs(ConditionalSMC, NUTS)` path; see the "GenFilters v0.5" section below for why this path stops short on Model A.

## Julia inference packages used or considered

| Package | What it offers | Used here? |
| --- | --- | --- |
| `Turing.jl` | PPL with NUTS, HMC, MH, IS, Gibbs, and particle-Gibbs (`PG`, `SMC`, `PGAS`) | Yes — `pgas_nuts` uses `Gibbs(PG, NUTS)` |
| `AdvancedPS.jl` | Particle MCMC primitives — bootstrap PF, particle-Gibbs, PG-AS, conditional SMC. Backs Turing's particle samplers | Indirectly, via Turing's `PG` |
| `GeneralisedFilters.jl` (v0.5, main) | Linear-Gaussian Kalman, bootstrap and auxiliary PFs, RBPF, ConditionalSMC, ParticleGibbs; Turing extension that hooks `x ~ SSMTrajectory(ssm, y)` | Yes — `BF` used as inner marginal for PMMH and SMC². v0.5 PGAS sampler installed and verified to load, but blocked on Model A by the issue below. |
| `SSMProblems.jl` | Common interface (StatePrior / LatentDynamics / ObservationProcess / StateSpaceModel) for SSM definitions | Yes — Model A wrapped as an SSM in `src/ModelA.jl` |
| `AdvancedMH.jl` | Metropolis-Hastings building blocks for AbstractMCMC | Conceptually — the manual PMMH loop in `pmmh/run.jl` is an MH walk written by hand. Could be swapped for `AdvancedMH.MetropolisHastings` with a LogDensityProblems wrapper. |
| `LowLevelParticleFilters.jl` | Standalone PF / KF / EKF / IteratedEKF / UKF / EnKF / IMM / RBPF / MUKF / AuxiliaryPF / AdvancedPF | Installed but not wired. UKF and EKF would hit the Gaussian-filter blind spot on `(log_tau_R, log_tau_F)` documented in `smc-jax/README.md`. AuxiliaryPF would give a smarter proposal than bootstrap for PMMH / SMC² inner; not pursued here since the bootstrap inner already recovers truth to within ~1-2%. |
| `SequentialMonteCarlo.jl` | Pure SMC sampler with adaptive tempering, several resamplers | Not used — wiring it to a PF marginal needs the same amount of glue as the ~150 LOC hand-rolled SMC sampler in `smc2/run.jl` |

## What the three variants recover

Same synthetic dataset throughout: `T = 120`, `MersenneTwister(2)`, truth `(log_tau_R = -4.0, log_tau_F = -12.0, log_phi = 2.5)`.

| Variant | log_tau_R (truth -4.0) | log_tau_F (truth -12.0) | log_phi (truth 2.5) | Notes |
| --- | --- | --- | --- | --- |
| `pmmh/` | -4.073 ± 0.301 | -12.042 ± 0.419 | 2.477 ± 0.143 | 1500 iter, 1500 PF particles, accept 0.589 |
| `smc2/` | -4.099 ± 0.304 | -12.117 ± 0.507 | 2.514 ± 0.148 | 128 theta-particles, 600 PF particles, 4 adaptive-tempering steps |
| `pgas_nuts/` | -4.145 ± 0.253 | -11.537 ± 0.369 | 2.456 ± 0.147 | 600 iter (150 NUTS adapts), 200 PG particles |

All three recover the parameters to within ~4% of truth on this synthetic data, comparable to what `smc-jax`'s Liu-West PF reports for the same parameters in its README.
`pgas_nuts` shows slightly larger bias on `log_tau_R` and `log_tau_F` (~3.5-4%), consistent with the known PG mixing penalty on long continuous latent paths.

## GenFilters v0.5 PGAS: why it stops short on Model A

The cleanest "off-the-shelf SMC + NUTS for statics" Julia path is the GeneralisedFilters v0.5 sampler:

```julia
@model function pgas_modelA(y, cfg)
    log_tau_R ~ Normal(...); log_tau_F ~ Normal(...); log_phi ~ Normal(...); log_I0 ~ Normal(...)
    ssm = build_ssm_flat(cfg, ModelAParamsFull(log_tau_R, log_tau_F, log_phi, log_I0))
    x ~ SSMTrajectory(ssm, y)
end
sampler = ParticleGibbs(ConditionalSMC(BF(n_particles)), NUTS(0.8))
```

GenFilters v0.5 ships this sampler.
The released v0.4.2 does not yet; we installed v0.5 from the main branch (pinning Turing 0.43 to match the GF Project.toml).

Trying this on Model A fails at the model-evaluation stage with
`MethodError: no method matching distribution(::ModelADynamicsFlat, ::Int64, ::Vector{Float64})`.
The v0.5 `ConditionalSMC` calls `SSMProblems.distribution(dyn, step, state)` to evaluate the transition density.
The transition in Model A is deterministic in 14 of 18 state components (`I_buf` shifts and gets one new entry; only the four innovation channels `eps_R, eta_R, eps_F, eta_F` are stochastic).
So the transition density is singular on a 4-D manifold in 18-D state space.
A workable `distribution` would be a singular Gaussian and the CSMC importance weights are then ill-defined in the usual sense.

This is a property of the renewal dynamics, not of the Julia tooling.
The same model in the `pgas_nuts/` variant works under Turing's `Gibbs(PG, NUTS)` because Turing's PG only needs the model evaluation — it does not call a separate transition density.

The trend-inflation PGAS example in the GeneralisedFilters repo works because the underlying model is fully Gaussian-linear (`LinearGaussianLatentDynamics`), where `distribution` is a clean MvNormal.

The flat-vector wrappers in `src/ModelA.jl` are kept for completeness — they are enough for a plain bootstrap filter via `build_ssm_flat`, but not for v0.5 `ConditionalSMC`.

## What `smc-jax` does that stock Julia does not give you for free

1. **Liu-West shrink-jitter on static parameters in a single forward pass.**
   `smc-jax`'s Liu-West PF carries `(log_tau_R, log_tau_F, log_phi)` as a per-particle parameter cloud and shrink-jitters them inside the same sweep that filters the latent path.
   AdvancedPS / GeneralisedFilters have no shrink-jitter kernel.
   The closest Julia analogue is the SMC sampler in `smc2/`, which carries a parameter cloud but does explicit MH moves between tempering steps.

2. **Steyn-style fixed-lag resampling.**
   `smc-jax/pf/runner.py`'s `fixed_lag_L` argument restricts the resample permutation to the last L steps of state and parameter history.
   No Julia PF package implements this.

3. **SMC² with a pluggable Gaussian inner filter (UKF or EKF).**
   `smc-jax/smc2/` lets the inner marginal-likelihood come from a UKF (`smc2.ukf`) or a library EKF (`smc2.ekf_cuthbert`).
   The Julia counterpart in `smc2/` here uses a PF inner instead — GeneralisedFilters has no UKF/EKF, and LowLevelParticleFilters' UKF/EKF would hit the same Gaussian-filter blind spot on `(log_tau_R, log_tau_F)` that `smc-jax/README.md` documents.

4. **Guided / auxiliary-q proposals (Model E territory).**
   The Rao-Blackwellised auxiliary-q Wallenius proposal for the GDM cohort-partition observation in `smc-jax/pf/observation_gdm.py` is custom-written.
   No Julia package supplies it.
   Out of scope here (we are doing Model A) but the biggest gap if the modelling moves to contact-tracing depletion + GDM.

5. **Sequential extension (`extend_liu_west`, `rolling_origin_forecast`).**
   `smc-jax` carries the trailing particle cloud across data arrivals and updates rather than refits.
   None of the Julia samplers here have a first-class `extend` operator that handles the parameter cloud and the latent path together.

6. **Trajectory smoothing with genealogy tracing returned in a typed result.**
   `pfjax.particle_smooth` gives backward-traced sample paths from the surviving genealogy.
   GeneralisedFilters has callbacks that capture the genealogy (`AncestorCallback`) and a `get_ancestry` helper that reconstructs paths after the fact — usable, but thinner than what `smc-jax` builds on.

## The Gaussian-filter blind spot is structural

`smc-jax/README.md` documents that SMC² + EKF / UKF is essentially blind to `(log_tau_R, log_tau_F)`.
The linearisation collapses the chain `tau → variance of log sigma → sigma via exp`, so the marginal log-lik moves with `log_phi` but barely with the tau parameters.

This is independent of language.
A Julia UKF inner SMC² (LowLevelParticleFilters UKF + a hand-rolled SMC over theta) would land on the same blind spot.
The fix is to keep the inner exact, which means a particle filter — bootstrap or guided.
That is why both `pmmh/` and `smc2/` here use a PF inner, and why no UKF-inner-with-NUTS-on-theta variant is provided.

## Reproducing

```bash
cd julia-smc
julia --project=. -e 'using Pkg; Pkg.instantiate()'

julia --project=. scripts/01_synthetic_demo.jl
julia --project=. pmmh/run.jl
julia --project=. smc2/run.jl
julia --project=. pgas_nuts/run.jl
```

PMMH takes ~10 min, the SMC sampler ~3 min (the adaptive schedule chose only 4 steps on this dataset), and Gibbs(PG, NUTS) the longest of the three.
The env pins Turing 0.43 to satisfy GeneralisedFilters main; both are listed in `Manifest.toml`.
