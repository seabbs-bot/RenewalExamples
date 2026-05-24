# How `smc-renewal` relates to `EpiNow2` and `epinowcast`

A side-by-side reading of the three packages, written so a reader who knows one can locate the equivalent piece in the others.

The intent is to make the design decisions visible, not to argue any of them is the right one.

## Top-level summary

| | `EpiNow2` | `epinowcast` | `smc-renewal` |
| --- | --- | --- | --- |
| Language / engine | R + Stan | R + Stan | Python + JAX |
| Inference | NUTS (full-pass each refit) | NUTS (full-pass each refit) | Particle filter (Liu-West, bootstrap or guided) **+** SMC² with Gaussian-filter inner |
| Sequential update | no (refit each week) | no (refit each week) | yes (`extend_liu_west`, `extend_smc2`) |
| Time-varying Rt prior | Gaussian process or random walk on `log Rt` | spline / hazard structure on cohort reporting; Rt is upstream | nested random walk on `log Rt` driven by `log σ_R(t)` (Models A/B/D) or integrated Brownian motion on `log Rt` via a velocity state (Models C/D/E) |
| Observation | delay convolution + NegBin | cohort hazard model on triangular reporting data (data-revision aware) | delay convolution + NegBin (Models A–D); GDM cohort partition (Model E) |
| Nowcasting / right-truncation | partial (truncation via reporting delay convolution) | first-class | not in the design today; closest analogue is Model E's GDM cohort structure |
| Generation interval | parametric prior, fit jointly with Rt | upstream of the package; usually fixed | parametric prior (Model A's `UnknownGI` machinery in the Julia side; smc-jax uses a fixed PMF) |

The dynamics layer of `smc-renewal` is similar to `EpiNow2`'s.
The observation layer of Model E is similar to `epinowcast`'s.
The inference machinery is the one place where `smc-renewal` is structurally different from both.

## Where each concept lives in code

### Time-varying `log Rt`

- `EpiNow2`: `inst/stan/functions/gaussian_process.stan` plus `functions/rt.stan`. The default is an approximated Hilbert-space Gaussian process on `log Rt(t)` with a Matérn kernel. Random walk on `log Rt` is also available as a switch.
- `epinowcast`: Rt is not the primary object of inference. The package models cumulative reporting given cases; `log_expected_latent_from_r.stan` is the connector to a `log Rt(t)` prior when present.
- `smc-renewal`: nested random walk on `log Rt` driven by `log σ_R(t)` which itself walks (Models A/B/D), or integrated BM `log Rt[t] = log Rt[t-1] + v_R[t-1]` with velocity state `v_R` (Models C/D/E). See `src/smc_renewal/synthetic.py::simulate` and `pf/runner.py::_step`.

### Renewal kernel

All three use the same renewal recursion in expectation.
`smc-renewal` adds an F-feedback term in Models A–D (`Rt_eff = Rt · exp(−F · g·I)`) and adds contact-tracing depletion in Model E (the buffer reads from `U_buf`, not from total cohort counts).

### Observation model

This is where the design splits.

- `EpiNow2` and `smc-renewal` Models A–D both use **delay convolution + NegBin**:
  `μ_y(t) = Σ_s d_s · I[t−s]`,  `y_t ~ NegBin(μ_y(t), φ)`.
  Treats `y_t` as conditionally independent given `μ_y(t)`. Fine for low-ascertainment surveillance noise; wrong for high-ascertainment regimes where cohort-budget coupling matters.
- `epinowcast` and `smc-renewal` Model E both use a **cohort-based** observation: each cohort of cases is partitioned across reporting stages.
  `epinowcast` parameterises the per-stage reporting probability via a discretised logit-hazard (`functions/discretised_logit_hazard.stan`) and supports group effects.
  `smc-renewal` Model E parameterises the per-stage Beta-Binomial means via probit-linear-in-stage (`b_0 + b_1·s`), with a shared Beta concentration `M = exp(log_M)`.
  These are different parameterisations of the same conceptual structure (per-cohort stick-breaking across reporting delays).
- `epinowcast` additionally exposes the **reporting triangle** directly so the model sees partial cohorts. `smc-renewal` Model E only sees the column-summed `y_t = Σ_s O_s`.

### Inference

- `EpiNow2`, `epinowcast`: full-pass NUTS at refit time. Costs O(T) per leapfrog step; typical operational fit takes minutes.
- `smc-renewal`: sequential filter. Cost per step is O(N_particles), filter runs once forward. Sequential extension via `extend_liu_west` reuses the existing parameter cloud and only steps forward through the new observations.
  This is the explicit design motivation in the `smc-jax/README.md`: an alternative to the team's current weekly NUTS refits of pyrenew models.

### Nowcasting / data revisions

- `epinowcast`: built around the (reference_date, report_date) snapshot structure. The Stan data block carries `latest_obs`, snapshot lookups, and `apply_missing_reference_effects` for revision handling. First-class.
- `EpiNow2`: handles right-truncation via the reporting delay convolution and an estimated truncation distribution (`estimate_truncation.stan`). Does not explicitly model revisions of already-reported cases.
- `smc-renewal`: no nowcasting in the current design. The closest analogue is the U-buffer in Model E, which carries cohort-budget state that *could* be exposed to make per-cohort partial-report predictions. See `EXTENSION_nowcasting.md` for the design path.

## How the three would behave on the same operational task

**Setup**: respiratory case time series, weekly refit, 4-week-ahead forecast, partial reporting for the most recent ~3 weeks.

| | `EpiNow2` | `epinowcast` | `smc-renewal` (Model A or B) |
| --- | --- | --- | --- |
| Time to refit each week | ~5–30 min | ~5–30 min | ~30–60 s (sequential update, no refit) |
| Handles right-truncated recent data | yes (truncation distribution) | yes (native triangle) | partial — depends on which model; Models A–D ignore the structure |
| Handles data revisions to already-reported cases | no | yes | no |
| Sharp posterior on volatility hyperparameter | usually, given GP length-scale prior | not directly inferred | yes for Model B (`log σ` directly); no for Model A (`log τ` is prior-stuck) |
| Median forecast extrapolates current trend | depends on prior | depends on prior | no for Models A/B (driftless RW); yes for Models C/D/E (integrated BM with velocity state) |

For weekly operational forecasting with no data revisions, Model B in `smc-renewal` plus an integrated-BM variant (Model C-style) is the closest functional substitute for an `EpiNow2` workflow, and gains the sequential-update property.

For nowcasting in the presence of data revisions, neither `smc-renewal` nor `EpiNow2` is competitive with `epinowcast` today.
Closing that gap is what `EXTENSION_nowcasting.md` discusses.

## What's good about `smc-renewal` that the other two don't have

- **Sequential update without refit**. Once the parameter cloud is fit, weekly updates are O(seconds), not O(minutes).
- **Joint posterior over `(θ, x_{0:T})`** in a single filter run, vs. fully separate machinery for state vs. parameters in NUTS.
- **Tail-aware posterior on hierarchical volatility** (Model B specifically). The particle cloud preserves correlation between the level state and the volatility parameter; NUTS on the same model usually marginalises this away.
- **A guided PF showcase (Model E)** for likelihoods where bootstrap PF dies, including the structural-failure complement showing that fit and mechanism are different things.

## What's missing relative to the other two

- **Data revisions**. `epinowcast` is the reference here; `smc-renewal` would need the U-buffer exposed as a 2-D state to match.
- **Multiple groups / hierarchical pooling**. Both R packages do this; `smc-renewal` is single-series-only.
- **An ergonomic plotting / posterior-summary surface**. R packages have a deep ecosystem; the JAX side here is research code with `_common.py` plotting helpers per example.
- **A wider user base, written-up methods paper, and CRAN-style stability**.
  `smc-renewal` is a demo / methods showcase, not a maintained package.

## Where this leaves the comparison

Think of the three as covering different points in a (inference method) × (observation model) grid.

|   | delay-conv + NegBin obs | cohort partition obs |
| --- | --- | --- |
| **NUTS** | EpiNow2 | epinowcast |
| **PF / SMC²** | smc-renewal Models A–D | smc-renewal Model E |

The bottom row is what `smc-renewal` exists to prototype.
The right column is where nowcasting and contact-tracing-style structural problems get represented honestly.
The bottom-right cell is where future operational deployments most likely want to be.
