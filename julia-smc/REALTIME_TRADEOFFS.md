# Real-time modelling: tradeoffs across the six variants

A short note on what each variant is good and bad for in an operational real-time epi setting.
Focus is on the three things that matter when this kind of model is on the rota for a weekly or daily refit: efficiency, reliability, extensibility.

The headline question this lets us answer: if you wanted to deploy one of these for weekly Rt nowcasting and forecasting, which would you pick and why?

## What we mean by each axis

- **Efficiency**: wall-clock per refit, and whether sequential update is possible (refit cost vs incremental update cost).
- **Reliability**: posterior calibration, diagnostics surface, sensitivity to tuning, behaviour when the data is unusual (low counts, regime shifts, missing days).
- **Extensibility**: cost of adding features the operational setting will need — different observation models, time-varying parameters, group structure, data revisions, custom proposals.

## Six variants at a glance

| Variant | Engine | Time-to-fit (T=120) | Sequential update | Identifies all params | Honest reliability rating |
| --- | --- | --- | --- | --- | --- |
| `pmmh/` | GF BF + manual MH | ~10 min | not provided | yes | acceptable for offline; long autocorr |
| `smc2/` | GF BF + hand-rolled tempered SMC | ~3 min | not provided | yes | best of the PF-inner group; clean tempering diagnostics |
| `pgas_nuts/` | Turing Gibbs(PG, NUTS) | ~10-15 min | not provided | yes (PG mixes slowly on long latent paths) | acceptable; small bias on tau under default PG |
| `llpf_pmmh/` | LLPF AdvancedPF + manual MH | ~10 min | not provided | yes | parity with `pmmh/` |
| `llpf_smc2/` | LLPF AdvancedPF + hand-rolled tempered SMC | ~3 min | not provided | yes | parity with `smc2/` |
| `llpf_ukf_nuts/` | LLPF UKF + MH | ~30 sec | natively (predict! / correct! exposed) | **no — tau prior-stuck** | unreliable for the nested-RW design; reliable for log_phi only |

(Numbers are wall-clock on a single thread on this machine. The 10-30s UKF figure is unique because the Gaussian filter does not Monte-Carlo over particles per step.)

## What real-time operation actually needs

A weekly Rt refit on respiratory case time series has six concrete asks.

1. **Refit faster than the data refresh window.** Weekly data ⇒ minutes-to-an-hour budget. Daily ⇒ a few minutes.
2. **Sequential update preferred.** If the model can update on a new week of data rather than fitting from scratch, the operational latency drops by an order of magnitude.
3. **Posterior calibration on the hyperparameters you actually report.** If `log_phi` and `Rt(t)` are the reported quantities and the volatility parameters are nuisance, the calibration constraint relaxes.
4. **Reproducibility of the fit across data updates.** Same data should give same posterior; small data changes should give small posterior changes.
5. **Diagnostics surface** — ESS, convergence statistics, divergences — visible to whoever runs the model.
6. **Cheap to extend** when modelling needs change (different obs model, hierarchical structure, partial reporting).

## How the six variants score on those asks

### Efficiency

- **`llpf_ukf_nuts/`** is the fastest by an order of magnitude (~30 sec) because the UKF tracks mean + covariance rather than a particle cloud. Sequential update is also native through LLPF's `predict!` / `correct!`. **But** the Gaussian inner breaks `(log_tau_R, log_tau_F)` identification, so this is the fastest option *only if you do not actually need those parameters*. The same logic that makes `smc-jax/smc2/ekf_cuthbert.py` essentially blind to the tau parameters applies here.
- **`smc2/` and `llpf_smc2/`** are the next fastest at ~3 min because the adaptive tempering schedule on this dataset converges in ~4 steps. No sequential update path.
- The MH-based variants (`pmmh/`, `llpf_pmmh/`) are slowest at ~10 min because MH on theta with a PF marginal mixes by random walk and there is no parallelism over theta-particles.
- `pgas_nuts/` sits in between but the wall clock varies more with `n_particles` and `n_iter` choices.

None of the Julia variants have a first-class `extend_liu_west`-style sequential extension for the parameter cloud.
`smc-jax/rolling_origin.py` is the operational gold standard here.

### Reliability

- **The PF-inner variants (pmmh, smc2, llpf_pmmh, llpf_smc2) are the only ones that identify the full parameter vector.** All four recover `(log_tau_R, log_tau_F, log_phi)` to within ~1.5% of truth.
- `pgas_nuts/` recovers within ~3.5% but has a known PG mixing penalty on continuous latent paths. With more PG particles or PGAS via GenFilters v0.5 (once `ParticleGibbs(ConditionalSMC, NUTS)` becomes available for non-Gaussian transitions) this should tighten.
- `llpf_ukf_nuts/` is **structurally unreliable** for tau, by design of the Gaussian filter. It is fine for `log_phi` and `Rt(t)` if those are all you report.
- Tuning sensitivity: PMMH needs a proposal sd that depends on the data scale; the SMC samplers self-adapt via the tempering schedule and need less tuning. PG needs `n_particles` large enough relative to T; this is the easiest to misconfigure.
- The PF marginal is noisy. PMMH and the SMC sampler both handle this via MH acceptance (the noise integrates out across iterations / theta-particles). NUTS on a noisy marginal does not work — which is why the UKF variant uses MH instead of NUTS.

### Extensibility

Where extensibility matters most is adding the next observation model (e.g. the GDM cohort partition + contact-tracing depletion in `smc-jax`'s Model E).

- **PMMH variants extend most easily.** The MH outer is agnostic to the inner; swap in any marginal log-likelihood and the chain is the same.
- **SMC sampler variants extend the same way** — the inner is a free choice.
- **PG variants are hardest to extend.** Turing's `Gibbs(PG, NUTS)` requires the latents declared inline in the `@model` body, so any change in the latent structure means rewriting the model. The shared `step_state` call in `pgas_nuts/` mitigates this somewhat.
- **GenFilters main has `ParticleGibbs(ConditionalSMC, NUTS)` that would be the cleanest off-the-shelf path** once it works on non-linear-Gaussian transitions. Today it requires a `distribution(dyn, step, state)` on the transition which Model A does not admit cleanly.

The Stoner et al GDM cohort partition (Model E) needs a guided proposal that is the same flavour as `smc-jax/pf/observation_gdm.py`.
None of the Julia samplers ship that proposal; you would write it.
**PMMH on a guided PF inner is the closest off-the-shelf path** — write the guided PF, plug it into a manual MH loop.

## Recommendations for real-time epi use

Concrete picks given the constraints above.

### Default for weekly Rt nowcasting on respiratory cases

**`smc2/` or `llpf_smc2/`** (they are interchangeable on this model).
Adaptive tempering self-tunes, ~3 min wall-clock, identifies all parameters, clean diagnostics from the ESS-over-tempering-steps trace.
Pair with hand-rolled sequential extension code if weekly refits become too expensive.

### When you only need `log_phi` and the `Rt(t)` trace

**`llpf_ukf_nuts/`** with the τ priors fixed at sensible values.
~30 sec refit, native sequential update through LLPF's `predict!` / `correct!`.
Document that the volatility hyperparameters are not inferred; treat them as tuning constants.

### When the observation model needs to change (e.g. cohort partition + delays)

**`pmmh/`** with a custom guided PF inner.
The MH outer is the cheapest place to drop in a new likelihood.
This is the path the team would take to port the Model E equivalent to Julia.

### When the team already runs Turing / Stan-flavoured workflows

**`pgas_nuts/`** for the `Gibbs(PG, NUTS)` familiarity.
Smaller bias-vs-variance gap is acceptable in many settings, and the model lives in a familiar `@model` block.

### What to actually do next operationally

1. Use `smc2/` or `llpf_smc2/` as the weekly refit baseline.
2. Write a small sequential-update wrapper on top of the SMC sampler — keep the theta-cloud, extend the tempering when new data arrives by reweighting against the new likelihood. Estimated half a day's code.
3. If contact-tracing / GDM observation enters the pipeline, port the smc-jax Model E guided proposal to PMMH on a custom Julia PF. **`smc-jax`'s guided proposal description in `Model12_further_details.md` is the spec.**
4. Skip the UKF path for any model where `(log_tau_R, log_tau_F)` matter.

## What the comparison does not tell us

- All numbers are on a single synthetic dataset with truth `(log_tau_R = -4.0, log_tau_F = -12.0, log_phi = 2.5)`. Real surveillance data may stress different parts of the model.
- We have not tested under regime shifts, missing days, or data revisions — these are the things that actually break Rt estimators in deployment. The `smc-jax/EXTENSION_nowcasting.md` design notes are the next step here.
- We have not benchmarked parallelisation. LLPF and GenFilters both support threaded particle propagation; the wall-clock numbers above are single-threaded.
- The `smc-jax` Liu-West PF is not represented in the Julia comparison because no Julia package ships a shrink-jitter parameter-cloud kernel today.
  An LLPF AdvancedPF augmented with shrink-jitter on theta would be the closest port.
