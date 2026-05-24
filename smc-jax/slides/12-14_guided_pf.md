---
marp: true
theme: default
paginate: true
title: From bootstrap to guided particle filters
---

# From bootstrap to guided particle filters

Examples 12–14 of `smc-renewal`

Contact-tracing outbreak, GDM cohort partition, guided proposal, structural failure

Source for the long-form prose: `Model12_further_details.md`

---

## What this deck covers

1. **The model** (Model E): contact-tracing renewal with a Generalised Dirichlet Multinomial reporting delay.
2. **The non-bootstrap PF**: auxiliary-q Wallenius guided proposal with Rao-Blackwellisation.
3. **The structural-failure complement**: Model C on the same data.
4. Three takeaways for the talk.

Each numbered section maps directly to the section of the same number in `Model12_further_details.md`.

---

# Part I — The model

---

## 1. What we are representing

A short outbreak (50 days) where reporting *is removal*.

- **Transmission**: continues while there are still-infectious people in circulation.
- **Reporting delay**: shape (mean delay) plus over-dispersion (cohort-to-cohort variability).
- **Contact-tracing removal**: the moment a person is reported, they stop being part of the renewal pool.

The third mechanism is what makes this structurally different from Models A–D. It is a self-limiting feedback driven by the *observation process*, not by the transmission rate.

---

## 2. Latent dynamics

```
v_R[t]      =  v_R[t-1]   +  σ_vR · η_R[t]       velocity walk
log Rt[t]   =  log Rt[t-1] +  v_R[t-1]            integrated BM
g_conv      =  Σ_k g(k) · U_buf[k]                renewal sum
λ_t         =  exp(log Rt[t]) · g_conv            Poisson rate
N_t         ~  Poisson(λ_t)                       new infections today
```

Two things to notice.

- The renewal kernel reads from `U_buf` (still-circulating-and-not-yet-reported), not from "total infections per cohort". Under 100% ascertainment one buffer suffices.
- No F-feedback term, no immigration term. Bending comes from contact tracing on a 50-day timescale, not from susceptibility depletion.

---

## 3. Observation — Generalised Dirichlet Multinomial (Stoner et al)

Per-cohort stick-breaking across reporting stages via independent Beta-Binomials.

```
q_s   ~  Beta(α_s, β_s)
O_s   ~  Binomial(U_s, q_s)
Φ⁻¹(p_s)  =  b_0  +  b_1 · (s − 1)
α_s = p_s · M,  β_s = (1 − p_s) · M
```

Observed total is the simple sum `y_t = Σ_s O_s`, with no extra NegBin noise on top.

The "no extra noise" is what breaks the bootstrap PF later.

---

## 4. Buffer update

```
U_prop    =  [N_t,  U_buf_prev[0], …, U_buf_prev[L-2]]   new cohort enters
U_buf_new =  U_prop − O                                   remove reported
```

Min-delay is 1 day so no observations apply to slot 0 today.
This means `U_buf[:, :, 0]` over time *is* the latent infection time series — useful for plotting without a separate `I_buf`.

---

## 5. The model in one line

```
state    =  (log_Rt, v_R, log_I0, U_buf)
params   =  (log σ_vR, b_0, b_1, log_M)        4-D Liu-West cloud
```

Truth used in example 12:
`log_Rt = 0.7` (Rt ≈ 2), `v_R = 0` (constant Rt — bending from tracing, not slowing transmission), `log_I0 = 1.0`, `b_0 = −1.5, b_1 = 0.4` (mean delay ~3.5 days), `log_M = 3.5`.

Outbreak peaks around day 18 at ~25 cases/day, then declines as tracing drains the U-buffer.

---

# Part II — The non-bootstrap PF

---

## 6. PF in 30 seconds

State-space recursion:

```
x_t  ~  p(x_t | x_{t-1}, θ)         transition
y_t  ~  p(y_t | x_t, θ)             observation
```

Importance sampling per step:

```
x_t^(i)  ~  q(x_t | x_{t-1}^(i), y_t, θ)
w_t^(i)  ∝  w_{t-1}^(i) · p(x_t | x_{t-1}, θ) · p(y_t | x_t, θ)
                       ──────────────────────────────────────────
                              q(x_t | x_{t-1}, y_t, θ)
```

**Bootstrap PF**: `q = transition prior`. Transition cancels; weight is just the likelihood. Beautifully simple — and dead on arrival here.

---

## 7. Why bootstrap dies

Run the model forward from a particle.
You sample `N_t`, form `U_prop`, draw `O_s ~ BetaBin` per cohort, take the sum.

The probability that this sum *happens to equal* the observed `y_t` is effectively zero.

At ~30 cases/day with L ≈ 14 cohorts each contributing a BetaBin draw, the joint distribution over `Σ O_s` is too spread out for exact hits.

Every particle gets zero weight. ESS collapses to 1 immediately.

Same failure mode as a sharp likelihood. Bootstrap proposes without using `y_t`; the data demand an exact match.

---

## 8. Designing the proposal

Propose so `y_t` is matched exactly. Correct via IS.

```
q(x_t | x_{t-1}, y_t)  =  q_dyn(z_t | x_{t-1})  ·  q_part(O_t | z_t, y_t)
```

`q_part` must satisfy three properties.

1. **Hit the constraint** `Σ_s O_s = y_t` exactly.
2. **Respect per-cohort budgets** `0 ≤ O_s ≤ U_s`.
3. **Match the target's marginal variance** well enough that IS weights have bounded variance.

---

## 8 (cont.) — Three candidate proposals

| Proposal | Hits constraint | Caps | Marginal mean | Variance | IS weight |
| --- | --- | --- | --- | --- | --- |
| Multinomial(y_t, π_s) with `π_s ∝ U_s · p_s` | yes | **no** (escapes) | matches | similar | clean |
| Multivariate hypergeometric | yes | yes | **mismatched** | lighter | very clean |
| **Wallenius** (weights = `p_s`) | yes | yes | matches | lighter | needs multinomial-coef correction |

Wallenius is the structural skeleton.
It respects constraints, matches the mean per cohort, and the IS weight has a tractable closed form modulo the ordering→outcome-space multinomial-coefficient correction.

---

## 9. The IS pitfall — tail mismatch

Wallenius is lighter-tailed than the target.

The per-stage target marginal is BetaBin, with over-dispersion factor `1 + (U−1)/(M+1)` relative to Binomial.
Fixed-weight Wallenius is sub-Binomial because of without-replacement.

Per stage the target/proposal ratio can be several times higher.
Across L stages this compounds.

Some draws get log-weights tens of nats above the cloud average. One particle eats the whole posterior.

Classic IS failure: low-variance proposal against high-variance target → heavy-tailed weights → ESS collapses.

---

## 10. Rao-Blackwellisation — lift `q_s` into the proposal

The model says

```
q_s   ~  Beta(α_s, β_s)
O_s   ~  Binomial(U_s, q_s)
```

so `BetaBin(U_s, α_s, β_s) = ∫ Bin(U_s, q_s) Beta(q_s; α_s, β_s) dq_s`.

The over-dispersion comes entirely from `q_s` being random.

**Don't marginalise `q_s` out**. Sample it as part of the proposal; weight against the joint target `p(O, q | y_t)`.

---

## 10 (cont.) — Augmented target and weight

```
Target:
   p_aug(O, q | y_t)  ∝  ∏_s [ Bin(O_s; U_s, q_s) · Beta(q_s; α_s, β_s) ]  ·  𝟙[ΣO=y_t]

Proposal:
   q_s    ~  Beta(α_s, β_s)
   O      ~  Wallenius(U, weights = q_s, y_t)

Weight (Beta factors cancel):
   w(O, q)  =  ∏_s Bin(O_s; U_s, q_s)
             ────────────────────────────────────────────────────
             Wallenius(O; U, q_s, y_t) · multinomial-coef(y_t; O)
```

What remains is per-cohort Bin against Wallenius weighted by the same `q_s` on both sides.
Marginal means match; variances comparable; tails match; IS weights bounded.

The over-dispersion that was killing us is now produced *by the proposal itself*.

---

## 11. The algorithm end to end

```
sample η_R; advance log_Rt ← log_Rt + v_R; advance v_R ← v_R + σ_vR · η_R
g_conv = Σ_k g(k) · U_buf[k];   λ_t = exp(log_Rt) · g_conv
N_t ~ Poisson(λ_t)

α_s, β_s = gdm_beta_params(b_0, b_1, log_M)
q_s      ~ Beta(α_s, β_s)            s = 1, …, L−1   (q_0 ≡ 0)

U_prop = [N_t, U_buf_prev[0], …, U_buf_prev[L-2]]
if y_t > Σ U_prop:  log_w_inc = -∞   (infeasible particle)

# guided partition: sequential weighted without-replacement
for k = 1 … y_t:
    weights = U_curr · q_s;  π = weights / Σ weights
    s_drawn ~ Categorical(π); U_curr[s_drawn] -= 1
    log_q_ordered += log π[s_drawn]
O = count of cohorts drawn

log_target = Σ_s log Bin(O_s; U_s, q_s)
log_mc     = gammaln(y_t + 1) - Σ_s gammaln(O_s + 1)
log_w_inc  = log_target - log_mc - log_q_ordered
U_buf_new  = U_prop - O
```

Liu-West shrink-jitter + ESS-triggered multinomial resampling come from `pf/_runner_core.py` and are unchanged from the other variants. Only the per-step proposal + weight differ.
4-D Liu-West cloud: `(log σ_vR, b_0, b_1, log_M)`.

---

# Part III — Structural failure when the mechanism is missing

---

## 12. Numbers — Model E vs Model C on the same data

| Metric | Model E (contact tracing in model) | Model C (no contact tracing) |
| --- | --- | --- |
| log Rt 90% coverage | 0.98 (near nominal) | **0.36** (badly under nominal) |
| log Rt filter median | tracks truth (Rt ≈ 2.12, *constant*) | drifts from +0.5 to −1.5 |
| I(t) 90% coverage | 0.92 | 1.00 (over-wide) |
| log φ posterior | n/a | +3.1 (NegBin absorbing variance) |
| Min ESS | ~ 10 | ~ 590 (healthy) |

Model C fits the cases fine.
What it gets wrong is the inferred Rt trajectory.

---

## 13. Why

Truth `log Rt ≈ 0.75` (Rt ≈ 2.12) is pinned essentially constant.
The synthetic data was generated with very small `σ_vR` and rejected unless `log Rt > 0` throughout the first 25 days.

**Transmission never actually slows.**
Cases peak and decline because contact tracing depletes the still-circulating pool faster than new infections replenish it.

Model C does not have that mechanism, so to fit the decline it is *forced* to attribute the bending to a falling Rt.
Filter median walks from above 1 to well below 1 over 50 days.
NegBin `log φ` absorbs cohort-level noise; it cannot represent the depletion feedback.

The model has to put the bending somewhere. The only place it has is Rt.

---

## 14. Operational consequence

The two stories are **operationally opposite**.

- **Model E**: Rt is still ~2 but tracing outpaces transmission. *Action*: maintain tracing intensity; relaxing it would re-ignite the outbreak immediately.
- **Model C**: Rt has crashed below 1, transmission has slowed. *Action*: relax interventions, the epidemic is dying.

A model that fits the data fine and predicts the next 14 days reasonably can still recommend the wrong intervention.

The fit is not the test of the model. The *mechanism* is.

---

# Part IV — Three takeaways

---

## 1. Bootstrap PF is a special case, not the only PF

Bootstrap = `q = transition`.
Beautifully simple, universally taught, works when the likelihood is forgiving.

When the likelihood is an indicator (high ascertainment, exact accounting, contact tracing), bootstrap PF is dead.
You have to design a custom proposal that uses the observation.

---

## 2. A guided proposal is IS per step

Mechanics are mechanical.
Factor the joint, pick a proposal, write the weight.

The art is choosing a proposal that

- respects constraints,
- matches marginals,
- matches tails.

Skip any of these and the PF fails in a characteristic way.

- Constraint violations → wasted particles.
- Mean mismatch → high IS variance.
- Tail mismatch → ESS collapse.

**Auxiliary-variable Rao-Blackwellisation is the standard fix for tail mismatch.**
If your target is `p(O) = ∫ p(O | q) p(q) dq` and your proposal forces you to commit to a fixed `q`, lift `q` into the proposal instead.
The integral disappears; priors cancel; tails match.

---

## 3. A correct proposal fixes the variance problem; only a correct model fixes the interpretation problem

Example 13 demonstrates this directly.

A model lacking a structural mechanism can still fit the data well in marginal-variance terms while getting the dynamics operationally wrong.

NegBin overdispersion absorbs cohort-budget noise.
It cannot represent depletion feedback.

The cost of mis-specifying mechanism is a wrong recommendation for the public-health user, not a wrong fit to the data.

---

## See also

- `examples/12_model_d_gdm.py` — structural model.
- `examples/13_model_c_on_gdm_data.py` — structural-failure complement.
- `examples/14_counterfactual_tracing_stops.py` — counterfactual scenario.
- `examples/data/12_truth.npz` — locked synthetic dataset.
- Source: `src/smc_renewal/pf/model_gdm.py`, `pf/runner_gdm.py`, `observation_gdm.py`.
- Long-form prose: `Model12_further_details.md`.

---

# Backup slides

---

## Reading order around this block

Before: examples 01–11 (bootstrap PF, SMC², Liu-West placements, rolling-origin forecasts on Models A–D).
After: open question on what would replace genealogy-tracing smoothing on Model E for longer horizons.

The 14 examples form four chunks; see `TUTORIAL.md` for the walkthrough.
