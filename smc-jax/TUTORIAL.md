# Walkthrough tutorial — smc-renewal

A guided reading order for someone who knows particle filters and renewal models in general but wants to learn this codebase.
Pair this with an assistant that runs the scripts, opens the figures, and asks the checkpoint questions at each step.

The repo's per-example synopsis lives in `examples/README.md`.
This document layers on top of it with prerequisites, what each example is intended to teach, and a checkpoint to confirm the lesson landed before moving on.

## Setup checkpoint

Confirm both before starting.

```bash
uv sync
uv run pytest -q                                  # 4 passed expected
uv run python examples/01_synthetic_demo.py       # writes figures/01_synthetic_tour.png
```

If the first script writes a figure with four random walks plus latent infections plus observed cases, the env is good.

## Suggested path

The examples fall into four chunks.
Walk them in order on a first pass.

### Chunk 1 — Single-method intuition (01, 02)

Goal: be able to write down Model A's state equations from memory and explain what the filter posterior over `log Rt(t)` looks like.

| Step | Run | Lesson |
| --- | --- | --- |
| 01 | `01_synthetic_demo.py` | Read off the nested-RW structure from the panels. Two `σ` walks, two log-level walks, then `I(t)` via the renewal kernel, then NegBin cases. |
| 02 | `02_pf_fit.py` | Liu-West bootstrap PF on Model A. Smoother band tightens vs filter band. ESS decay over time. |

Checkpoint after 02:
- Why is the smoother band on `log Rt(t)` narrower than the filter band?
- What does the ESS-over-time panel tell you about how often the filter resampled?
- If you halved `h` in `run_liu_west`, which posterior would tighten and which would lose coverage?

### Chunk 2 — Two inference engines, same model (03, 05)

Goal: understand why SMC² and Liu-West PF disagree on `(log τ_R, log τ_F)` even on the same data.

| Step | Run | Lesson |
| --- | --- | --- |
| 03 | `03_smc2_vs_pf.py` | SMC² with the cuthbert EKF inner marginal-likelihood, head-to-head with the Liu-West PF. `log φ` agrees; the τ posteriors disagree. |
| 05 | `05_smc2_ekf_fit.py` | The diagnostics version of 03 with the prior overlaid on the τ posteriors. Lets you see the prior-collapse on `log τ_R, log τ_F`. |

Checkpoint after 05:
- The Gaussian-filter chain that lets `τ_R` enter the marginal likelihood is `τ_R → variance of log σ_R → σ_R via exp`. Where in that chain does the EKF/UKF linearisation lose the signal?
- What kind of observation would make the τ identifiable inside a Gaussian filter?

### Chunk 3 — Operational forecasting (04, 06, 07, 09, 10, 11)

Goal: understand the rolling-origin harness, how sequential updates (no refits) work, and how each Model A/B/C/D differs in forecast behaviour.

Run in this order; each comparison is best made by comparing figures pairwise.

| Step | Run | Compare with | Lesson |
| --- | --- | --- | --- |
| 04 | `04_rolling_origin_forecast.py` | — | Liu-West PF rolling origin on Model A, short T. Sequential extension via `extend_liu_west`. CRPS and interval coverage by horizon. |
| 06 | `06_smc2_ekf_rolling_origin.py` | 04 | Same harness, SMC²+EKF inner. EKF lacks the heavy tail Liu-West PF can produce; CRPS at long horizon is far smaller. |
| 07 | `07_multi_season.py` | — | Model A on multi-season data (T≈1440). Forced-seasonal `log Rt` plus slow `log F` drift. Fixed-lag resampling (`fixed_lag_L=21`) restores smoother coverage from ~0.77 to ~0.91. |
| 09 | `09_model_b_multi_season.py` | 07 | Model B = Liu-West on `log σ` directly, no τ. Identifies `log σ_R` and `log φ` sharply; forecasts are an order of magnitude tighter than Model A's. |
| 10 | `10_model_c_multi_season.py` | 07, 09 | Model C = integrated-BM trend. Velocity state `v_R` means medians extrapolate the slope, not freeze at the current level. Smoother coverage drops to ~0.51; genealogy-tracing degenerates when level + velocity both fit. |
| 11 | `11_model_d_discrete.py` | 10 | Model D = Model C + Poisson infections + immigration. Bootstrap PF handles a discrete latent process. |

Checkpoint after 11:
- Which model would you pick for weekly operational forecasting on respiratory cases, and why?
- When does the integrated-BM trend hurt rather than help?
- Why does immigration `μ` exist in Model D but not in C?

### Chunk 4 — Guided PF on a hard likelihood (12, 13, 14)

Goal: understand why bootstrap PF dies on the GDM observation, what an auxiliary-q Wallenius guided proposal does, and why a structurally wrong model can still fit the data.

For this chunk read `Model12_further_details.md` alongside; it is the narrative explainer for the whole block.

| Step | Run | Lesson |
| --- | --- | --- |
| 12 | `12_model_d_gdm.py` | Model E on 50-day contact-tracing outbreak. Guided PF achieves near-nominal coverage on `log Rt` and `I(t)`. |
| 13 | `13_model_c_on_gdm_data.py` | Model C on Model E's data. NegBin absorbs cohort noise so the fit is fine, but the inferred `log Rt` drifts from +0.5 to −1.5 when truth is constant. |
| 14 | `14_counterfactual_tracing_stops.py` | Counterfactual: tracing stops at day 25. Model E's 95% upper at day 39 is ~146× Model C's. |

Checkpoint after 14:
- Why is the bootstrap PF's per-step weight `w ∝ p(y_t | x_t)` essentially zero on the GDM observation?
- What does Rao-Blackwellisation of `q_s` buy you that fixed-`q` Wallenius alone could not?
- Pick one operational decision (e.g. relax contact tracing). Explain how Model E and Model C send opposite recommendations on the same data.

## Where the methods live

- Pure-JAX per-step transition, verified against pyrenew: `src/smc_renewal/transition.py`.
- Synthetic forward simulator: `src/smc_renewal/synthetic.py`.
- Liu-West bootstrap PF and variants:
  - Model A runner: `src/smc_renewal/pf/runner.py`.
  - Models B/C/D runners: `pf/runner_sigma.py`, `pf/runner_trend.py`, `pf/runner_discrete.py`.
  - Model E runner with guided proposal: `pf/runner_gdm.py`, `pf/model_gdm.py`.
  - Shared scan loop and Liu-West kernel: `pf/_runner_core.py`.
  - Sequential update entry point: `pf/update.py::extend_liu_west` (Model A only at present).
- SMC² over θ:
  - Adaptive tempering and resampling: `smc2/runner.py`.
  - UKF marginal log-likelihood: `smc2/ukf.py`.
  - EKF marginal log-likelihood via cuthbert: `smc2/ekf_cuthbert.py`.
- Rolling-origin harness: `src/smc_renewal/rolling_origin.py`.

## After the walkthrough

Open questions worth chasing once the path is internalised.

1. Why is the smoother coverage at ~0.51 on Model C (example 10)?
   What would a forward-filter-backward-sample replacement of `pfjax.particle_smooth` cost in code complexity?
2. Sequential update is Model-A only.
   The shared core in `pf/_runner_core.py` is variant-agnostic, so wrapping B/C/D should be a small change.
   Is there a use case that motivates it?
3. The Gaussian-filter blind spot on `(log τ_R, log τ_F)` is well-documented.
   Is a Rao-Blackwellised particle Gaussian (RBPF on the level, particle on the volatility) a path worth trying inside SMC²?
4. See `EXTENSION_time_varying_tau.md` for the design notes on letting `(τ_R, τ_F, log φ)` drift over time.
5. See `EXTENSION_nowcasting.md` for the design notes on adding the Stoner et al double-stick-break for nowcasting and data revisions.
6. See `COMPARISON_epinow_packages.md` for how this codebase relates to `EpiNow2` and `epinowcast` (similar dynamics, different inference, different observation model).
