# Julia SMC landscape vs `smc-jax`

What is in `julia-smc/` is three SMC-flavoured fits of Model A on the same synthetic dataset, built using the Julia SMC ecosystem (GeneralisedFilters.jl, AdvancedPS / Turing, hand-rolled SMC sampler).
The point is to see what stock Julia tooling gives you and where you still have to write code if you want what `smc-jax` does.

The first round of this comparison used Turing + NUTS over the joint posterior of (theta, latent innovations).
That works fine and is already a known approach in Julia, so it is dropped here in favour of three actual SMC variants.

## Three variants in this directory

| Variant | Outer (theta) | Inner (latent path) | What gives you the marginal log-lik for theta? |
| --- | --- | --- | --- |
| `pmmh/` | Metropolis-Hastings (manual, AdvancedMH-style) | Bootstrap PF, GeneralisedFilters.jl `BF` | PF marginal log-lik (noisy, unbiased) |
| `smc2/` | SMC sampler with adaptive tempering (hand-rolled, ~150 LOC) | Bootstrap PF, GeneralisedFilters.jl `BF` | same PF marginal, called per theta-particle |
| `pgas_nuts/` | NUTS (Turing) | Particle Gibbs (Turing/AdvancedPS bootstrap PF) | Gibbs(PG, NUTS) alternates updates rather than computing an explicit marginal |

All three share `src/ModelA.jl`, which now exposes the model in two forms.
`step_state` and `simulate` remain the pure dynamics + forward simulator.
On top of that, `ModelAParams`, `ModelAPrior`, `ModelADynamics`, `ModelAObservation`, and `build_ssm` wrap the model in the `SSMProblems.jl` interface so the same dynamics feed straight into GeneralisedFilters.

## Julia inference packages used or considered

| Package | What it offers | Used here? |
| --- | --- | --- |
| `Turing.jl` | PPL with NUTS, HMC, MH, IS, Gibbs, and particle-Gibbs (`PG`, `SMC`, `PGAS`) | Yes — pgas_nuts variant uses Turing's `Gibbs(PG, NUTS)` |
| `AdvancedPS.jl` | Particle MCMC primitives — bootstrap PF, particle-Gibbs, PG-AS, conditional SMC. Backs Turing's particle samplers | Indirectly, via Turing's `PG` |
| `GeneralisedFilters.jl` | Kalman filter (linear-Gaussian only), bootstrap and auxiliary particle filters, RBPF; SSMProblems.jl interface | Yes — `BF` used as the inner marginal for both PMMH and the SMC sampler |
| `SSMProblems.jl` | Common interface (StatePrior / LatentDynamics / ObservationProcess / StateSpaceModel) for SSM definitions | Yes — Model A wrapped as an SSM in `src/ModelA.jl` |
| `AdvancedMH.jl` | Metropolis-Hastings building blocks for AbstractMCMC | Conceptually — the manual PMMH loop in `pmmh/run.jl` is an MH walk written by hand. Could be swapped for `AdvancedMH.MetropolisHastings` with a LogDensityProblems wrapper. |
| `LowLevelParticleFilters.jl` | Standalone PF / KF / EKF / UKF / RBPF library, more engineering-oriented | No — installed in the env but not used; UKF would have been an option for variant 1 (UKF + NUTS-on-theta) but blocked by the Gaussian-filter blind spot on `(log_tau_R, log_tau_F)` documented in `smc-jax/README.md` |
| `SequentialMonteCarlo.jl` | Pure SMC sampler with adaptive tempering, several resamplers | Not used — the SMC sampler in `smc2/run.jl` is hand-rolled at ~150 LOC because wiring this package to a PF marginal needs the same amount of glue |

## What the three variants recover

Same synthetic dataset throughout: `T = 120`, `MersenneTwister(2)`, truth `(log_tau_R = -4.0, log_tau_F = -12.0, log_phi = 2.5)`.

| Variant | log_tau_R (truth -4.0) | log_tau_F (truth -12.0) | log_phi (truth 2.5) | Notes |
| --- | --- | --- | --- | --- |
| `pmmh/` | -4.073 ± 0.301 | -12.042 ± 0.419 | 2.477 ± 0.143 | 1500 iter, 1500 PF particles, accept 0.589 |
| `smc2/` | -4.099 ± 0.304 | -12.117 ± 0.507 | 2.514 ± 0.148 | 128 theta-particles, 600 PF particles, 4 adaptive-tempering steps |
| `pgas_nuts/` | -3.998 ± 0.237 | -11.582 ± 0.374 | 2.466 ± 0.151 | 600 iter (150 NUTS adapts), 200 PF particles per PG sweep |

All three variants recover the parameters to within ~3% of truth on this synthetic data, comparable to what `smc-jax`'s Liu-West PF reports for the same parameters in its README.
`pgas_nuts` shows slightly larger bias on `log_tau_F` (~3.5%), consistent with the known PG mixing penalty on long continuous latent paths.

## What `smc-jax` does that stock Julia does not give you for free

The point of this comparison is to be specific about which features of `smc-jax/pf/`, `smc-jax/smc2/`, and `smc-jax/rolling_origin.py` have no direct stock-Julia counterpart, and so would need code if you wanted them.

1. **Liu-West shrink-jitter on static parameters in a single forward pass.**
   `smc-jax`'s Liu-West PF carries `(log_tau_R, log_tau_F, log_phi)` as a per-particle parameter cloud and shrink-jitters them inside the same sweep that filters the latent path.
   AdvancedPS / GeneralisedFilters have no shrink-jitter kernel.
   The closest Julia analogue is the SMC sampler in `smc2/`, which carries a parameter cloud but does explicit MH moves between tempering steps rather than continuous shrink-jitter.

2. **Steyn-style fixed-lag resampling.**
   `smc-jax/pf/runner.py`'s `fixed_lag_L` argument restricts the resample permutation to the last L steps of state and parameter history.
   No Julia PF package implements this; they resample the current step only or expose offline smoothing.

3. **SMC² with a pluggable Gaussian inner filter (UKF or EKF).**
   `smc-jax/smc2/` lets the inner marginal-likelihood come from a UKF (`smc2.ukf`) or a library EKF (`smc2.ekf_cuthbert`).
   The Julia counterpart in `smc2/` here uses a PF inner instead because GeneralisedFilters v0.4.2 has no UKF/EKF (only the linear-Gaussian KF and PF variants).
   LowLevelParticleFilters.jl has a UKF but pairing it with an SMC sampler is glue you write yourself.

4. **Guided / auxiliary-q proposals (Model E territory).**
   The Rao-Blackwellised auxiliary-q Wallenius proposal for the GDM cohort-partition observation in `smc-jax/pf/observation_gdm.py` is custom-written.
   No Julia package supplies it.
   Out of scope here (we are doing Model A) but the biggest gap if the modelling moves to contact-tracing depletion + GDM.

5. **Sequential extension (`extend_liu_west`, `rolling_origin_forecast`).**
   `smc-jax` carries the trailing particle cloud across data arrivals and updates rather than refits.
   None of the Julia samplers here have a first-class `extend` operator that handles the parameter cloud and the latent path together.
   You would write the outer loop yourself.

6. **Trajectory smoothing with genealogy tracing returned in a typed result.**
   `pfjax.particle_smooth` gives backward-traced sample paths from the surviving genealogy.
   GeneralisedFilters has callbacks that capture the genealogy (`AncestorCallback`) and a `get_ancestry` helper that lets you reconstruct paths after the fact — usable, but a thinner surface than what `smc-jax` builds on.

## The Gaussian-filter blind spot, and why it shapes variant choice

`smc-jax/README.md` documents that the SMC² + EKF / UKF path is essentially blind to `(log_tau_R, log_tau_F)`.
The Gaussian-filter linearisation collapses the chain `tau -> variance of log sigma -> sigma via exp`, so the marginal log-lik moves with `log_phi` but barely with the tau parameters.

That blind spot is **structural**, not specific to Python.
A Julia UKF inner SMC² (e.g. LowLevelParticleFilters UKF + a hand-rolled SMC over theta) would land on the same problem.
The fix is to keep the inner exact, which means a particle filter — bootstrap or guided.
That is why both `pmmh/` and `smc2/` here use a PF inner, and why a UKF-inner-with-NUTS-on-theta variant was not pursued.

## A note on the pgas_nuts variant

The cleanest off-the-shelf "SMC for latents + NUTS for statics" Julia path is the unreleased GeneralisedFilters v0.5 sampler:
```julia
sampler = ParticleGibbs(ConditionalSMC(BF(n_particles), AncestorSampling()), NUTS(0.8))
```
This combo is in main but not in any released version yet.
With the released v0.4.2 we use Turing's `Gibbs((statics) => NUTS, (latents) => PG)` instead, which is the same composition pattern via AdvancedPS-backed PG.
That requires indexed latent variables (`eps_R[t] ~ Normal()`) inside the model loop so each step gets its own VarName.

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
