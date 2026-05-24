# Extension — time-varying `(τ_R, τ_F, log φ)`

Design note for letting the static hyperparameters drift over time.
Not implemented; this is the bridge from "what's there now" to "what it would take".

## Where these parameters sit today

In Model A (see `pf/runner.py` and `state.py::ParticleParams`) the parameter vector is `(log τ_R, log τ_F, log φ)`.
These are *static* across the whole time series and live in a 3-D Liu-West cloud, shrink-jittered at every step.

`τ_R, τ_F` enter the model via the second level of the nested random walk:

```
log σ_R[t]  =  log σ_R[t-1]  +  τ_R · η_R[t]
log Rt[t]   =  log Rt[t-1]   +  σ_R[t-1] · ε_R[t]
```

`log φ` is the NegBin dispersion of the observation.

The volatility `σ_R[t]` is already time-varying in the state.
What the proposed extension asks is to let the *rate at which volatility walks* itself vary over time, plus to let the observation dispersion vary too.

## Why this is hard in the current design

Three pinch-points.

1. **Identifiability**.
   Examples 05 and 08 already document that `(log τ_R, log τ_F)` are weakly identified through the Gaussian-filter chain (`τ → variance of log σ → σ via exp`).
   A static prior plus 1440 observations gives a posterior that mostly looks like the prior.
   Letting `τ_R` itself drift adds dimensions per timestep with no new likelihood term that constrains them.
   Expect a wide prior-dominated posterior unless you can either (a) tie the drift to an observable feature (school terms, NPIs, surveillance system changes) or (b) impose a strong smoothness prior with a learned-once amplitude.

2. **Liu-West cloud cost**.
   The Liu-West kernel shrink-jitters a static parameter at every step.
   Replacing the scalar `log τ_R` with `T` values blows up the parameter dimension from 3 to ~3000 (T=1000), which is no longer a Liu-West problem; it is a state-space smoothing problem.
   The natural redesign is to *move `log τ_R(t)` into the state vector* and walk it.

3. **SMC² / Gaussian-filter compatibility**.
   The cuthbert EKF in `smc2/ekf_cuthbert.py` expects a fixed-dimension parameter vector.
   Adding a third level of random walk to the state means the EKF needs derivatives of the new transition, and the Gaussian-filter blind spot from example 05 probably propagates one level deeper.
   Plan on the SMC² path being PF-only for the extension, at least at first.

## Three concrete options

Pick by how rich a model the data actually identifies.

### Option 1 — Third level of nested random walk on `log τ`

Smallest change.
Add `log τ_R[t]` to the state vector and walk it at a new static rate `κ_R`.

```
log τ_R[t]    =  log τ_R[t-1]    +  κ_R · ζ_R[t]
log σ_R[t]    =  log σ_R[t-1]    +  exp(log τ_R[t-1]) · η_R[t]
log Rt[t]     =  log Rt[t-1]     +  exp(log σ_R[t-1]) · ε_R[t]
```

The Liu-West cloud now carries `(log κ_R, log κ_F, log κ_φ, log φ_init)`.
The dynamics module in `pf/_runner_core.py` needs one extra walk per branch.
Same trick for `log φ` if you want a time-varying observation dispersion.

Cost: trivial code change. Mostly editing `state.py::ParticleParams`, `pf/runner.py::_step`, and the synthetic simulator.
Identifiability: probably the same problem one level deeper. You will likely need a tight prior on `κ_R` to keep the third level from drifting freely.

### Option 2 — Discrete regime change-points

If the actual reason `τ_R` should change over time is a *known* event (lockdown, new variant, surveillance system change), drop the random walk and use a regime indicator.

```
τ_R(t)  =  τ_R(regime[t]),         regime[t] = known piecewise-constant indicator
```

Each regime gets its own static `τ_R^{(k)}` in the Liu-West cloud.
Same for `log φ`.
Three or four regimes is fine.
This is the cheapest meaningful version if you know when the breaks are.

Cost: same Liu-West dimension as today, just K-times-wider.
Identifiability: each regime sees a contiguous block of data, so it can learn within-regime `τ`.

### Option 3 — Small NN parameterising `(τ_R(t), τ_F(t), log φ(t))` from covariates

Closest to the "time-varying NN" framing.
A small MLP (one hidden layer, ~16 units) maps `features[t] → (log τ_R(t), log τ_F(t), log φ(t))`.
Features are anything actually observed: day-of-week, week-of-year, school-term indicator, mobility, holiday flag, surveillance-system version.

The Liu-West cloud carries the NN weights instead of the scalars.
A 5-feature input × 16 hidden × 3 output is ~140 weights — possible to Liu-West, painful to identify.

Better path: **train the NN offline on historical data (or simulate-then-fit) and then freeze its weights when forecasting**.
The NN becomes part of the prior structure, not part of what's fit at runtime.
Inside the PF you call `nn(features[t])` to get the rates and the PF only fits the latent states.
This sidesteps the Liu-West cost and the identifiability problem in one move, at the cost of needing a separate training pipeline.

A nice intermediate: train the NN offline, then at runtime fit a *single global scaling factor* (`α` such that `τ_R(t) = α · nn(features[t])`) inside the Liu-West cloud.
One scalar parameter, gives the model freedom to dial the prior amplitude up or down at runtime.

Cost: needs an offline training pipeline (flax / equinox).
Identifiability: handled by the offline split.
Operational value: highest of the three, conditional on features that actually correlate with volatility regime shifts.

## Recommended order

1. Run example 05 again and stare at the τ posterior vs prior.
   If they overlap completely, the data is telling you it cannot identify `τ_R` even statically.
   Option 1 will not help; Option 2 with strong priors or Option 3 with offline training are the only paths.
2. Pick Option 2 first if you know the breakpoints. It is the cheapest test of "does letting `τ` change actually improve forecast calibration".
3. If Option 2 helps and you have features, try Option 3 with NN frozen at runtime.
4. Option 1 (third RW level) is only worth doing if you have a clear reason to expect smooth slow drift in `τ` itself, with no observable features to predict it from.

## Where the code touches would land

| Change | Files |
| --- | --- |
| `(τ_R, τ_F, log φ)` enter the state | `src/smc_renewal/state.py`, `src/smc_renewal/synthetic.py`, `pf/_runner_core.py`, `pf/runner.py` |
| Liu-West cloud shape | `pf/runner.py::run_liu_west`, `pf/update.py::extend_liu_west` |
| EKF/UKF marginal log-likelihood | `smc2/ekf_cuthbert.py`, `smc2/ukf.py` — likely skip for this extension |
| NN evaluation hook | new file `src/smc_renewal/dynamics_nn.py`, called from `_runner_core.py::_step` |
| Tests | `tests/test_transition.py`, `tests/test_pf_sequential.py`, plus a new identifiability test on synthetic data with a known time-varying `τ` |

None of this is large.
The interesting decisions are all on the prior / features side, not on the implementation side.
