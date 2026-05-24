# RenewalExamples

Samuel Brand's toy examples of epidemiological renewal models.
Owned by `SamuelBrand1`; cloned locally for read/explore.
Two sub-projects with independent toolchains:

## Layout

- `Project.toml`, `src/`, `scripts/` — Julia/Turing examples (renewal
  models, generation-interval inference, infection feedback). Uses
  `CensoredDistributions`, `Turing`, `Enzyme`, `Mooncake`, `ReverseDiff`,
  `StatsPlots`. Entry point: `scripts/readme-examples.jl` (rendered into
  the top-level `README.md`).
- `smc-jax/` — Python/JAX sub-project, **independent env**. Sequential
  Monte Carlo + particle-filter forecasting for a renewal model with
  nested random-walk volatility. Uses `jax`, `numpyro`, `blackjax`,
  `pyrenew`, `pfjax`, `cuthbert`. Managed by `uv`.

## Run

Julia side:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # done once
julia --project=. scripts/readme-examples.jl
```

Python (SMC) side:

```bash
cd smc-jax
uv sync                                                # done once
uv run pytest
uv run python examples/01_synthetic_demo.py
```

Both envs were instantiated 2026-05-24 and verified with a smoke test
(Julia `using RenewalExamples` + sample from `UnknownGI`; `pytest
tests/test_public_api.py` → 4 passed).

## Where the explanations are

- Top-level `README.md` — rendered from `scripts/readme-examples.jl`,
  covers the Julia/Turing GI + renewal examples.
- `smc-jax/README.md` — model spec for the SMC sub-project (state
  equations, renewal + feedback, observation, parameter list, variants
  A–E, status of what does/doesn't recover).
- `smc-jax/examples/README.md` — per-example synopsis (14 scripts:
  intuition → single-method → cross-method → forecasting → variant
  models). Read this first when picking what to run.
- `smc-jax/Model12_further_details.md` — long-form narrative explainer
  for Model E (contact-tracing + GDM + guided PF) and the Model C
  structural-failure complement. Effectively talk-ready prose for the
  guided-PF material.

## Walkthrough materials added 2026-05-24

- `smc-jax/TUTORIAL.md` — guided walkthrough of all 14 examples in four
  chunks, with prerequisites and checkpoint questions per chunk. Pair with
  an assistant that runs scripts and asks the checkpoints.
- `smc-jax/slides/12-14_guided_pf.md` — Marp slide deck adapting
  `Model12_further_details.md` for the guided-PF block (Models D, C, E).
- `smc-jax/COMPARISON_epinow_packages.md` — how this codebase relates to
  `EpiNow2` and `epinowcast`. Same dynamics layer as EpiNow2, same
  cohort-observation conceptual structure as epinowcast (different
  parameterisation), but PF/SMC² inference and sequential update instead
  of NUTS refit.
- `smc-jax/EXTENSION_time_varying_tau.md` — design notes for letting the
  static hyperparameters `(τ_R, τ_F, log φ)` drift over time. Three
  options (nested RW, regime indicator, NN with offline training).
- `smc-jax/EXTENSION_nowcasting.md` — design notes for adding the Stoner
  et al double-stick-break for nowcasting and data revisions, with the
  per-layer IS variance argument tied to Sam B's caveat.

## Model variants (smc-jax)

| Variant | Distinguishing feature | Examples |
|---|---|---|
| A | nested-RW volatility on `log Rt`, `log F` | 01–07 |
| B | direct σ (no nested RW; σ in Liu-West cloud) | 08, 09 |
| C | velocity / integrated-BM trend on `log Rt` | 10 |
| D | C + Poisson infections + immigration | 11 |
| E | contact-tracing depletion + GDM observation + guided PF | 12–14 |

Model A is what `smc2/` targets. Sequential-update path
(`extend_liu_west`, `rolling_origin_forecast`) is currently Model-A only.
