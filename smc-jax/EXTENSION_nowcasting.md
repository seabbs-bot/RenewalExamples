# Extension — nowcasting via a double stick break, and adding data revisions

Two related things written up in one place.

1. What Sam B's "double stick break" hint refers to (Stoner et al style nowcasting).
2. What it would take to give `smc-renewal` the ability to nowcast and to handle data revisions.

Sam B's caveat on the guided PF is worth keeping in view while reading this.
He noted: *"the proposal is lower variance than the unconditional sampling which is usually not a good look for IS"*.
That is exactly the tail-mismatch failure mode that Section 9 of `Model12_further_details.md` names, and the auxiliary-q Rao-Blackwellisation in Section 10 is the standard fix.
If you push the same idea into nowcasting with two layers of stick break, the same risk and the same fix apply at each layer.

## Part I — What "double stick break" means

### Single stick break (what Model E already does)

For each cohort `r` and each delay stage `s ∈ {1, …, L−1}`:

```
q_s   ~  Beta(α_s, β_s)                       per-stage reporting fraction
O_{r,s}  ~  Binomial(U_{r,s}, q_s)            how many of cohort r report at age s
```

`U_{r,s}` is the remaining-unreported count from cohort `r` when it is at age `s`.
The Beta priors give Beta-Binomial marginals with `1 + (U−1)/(M+1)` over-dispersion factor over Binomial.

This single break models *when* a case from cohort `r` first appears in the data.
It does not model *what the value of that report is* after subsequent revisions.

### Double stick break (Stoner et al style)

Add a second layer that models the eventual value of a case that has appeared.

Layer 1 — delay stick break (as above).
Of the `N_r` infections in cohort `r`, how many appear in the data at each delay `s`.
Output: counts `M_{r,s}` for `s = 1, …, L−1`, with `Σ_s M_{r,s} ≤ N_r` (tail truncation drops the rest).

Layer 2 — revision stick break.
Each report has an initial value when it first appears, and that value may be revised over subsequent reporting periods.
For a report from cohort `r` that first appeared at delay `s`, its final-revised value is partitioned across revision ages `k = 0, …, K−1`:

```
p_{s,k}    ~  Beta(α_{s,k}, β_{s,k})                  per-(stage, revision) fraction
V_{r,s,k}  ~  Binomial(M_{r,s} − Σ_{j<k} V_{r,s,j}, p_{s,k})
```

The observed reporting triangle entry at (reference date `r`, report date `t`) is

```
Y_{r,t}  =  M_{r, t−r}  ×  Σ_{k ≤ snapshot_age(r, t)} V_{r, t−r, k}/M_{r, t−r}
```

— the data revisions cumulate over time.
The "double" refers to two cascaded per-cohort partitions, both stick-broken via Beta-Binomials.

Why this is the right structure for nowcasting:
- It lets the model produce a posterior over "what will the cohort `r` final count look like, given we have only seen the first few delay stages and the first few revisions".
- It separates the *delay* problem (when does a case appear at all) from the *revision* problem (what is its value when it has appeared).
- Under right-truncation (today's data is incomplete) the cohort budget `N_r` is what gets nowcast.
  Under data revisions the cohort budget is fixed but the value column shifts over time.

The Stoner et al GDM nowcasting paper packages this as a Generalised Dirichlet Multinomial regression on the triangular `Y_{r,t}` matrix.
The two stick breaks make the Dirichlet "generalised" in the sense that the per-stage / per-revision Beta means are not constrained to be the same.

### Why two breaks instead of one big multinomial

A single Dirichlet-Multinomial over all `(stage, revision)` cells would force the cells to share a single concentration parameter `M`, which conflates "delay shape uncertainty" with "revision shape uncertainty".

Two cascaded Beta-Binomial layers let each have its own concentration:

```
M_delay     =  exp(log_M_delay)
M_revision  =  exp(log_M_revision)
```

This is essential because in practice delay shape is *much* more concentrated than revision shape, or vice versa, depending on the surveillance system.

## Part II — What it would take to give `smc-renewal` nowcasting and data revisions

The current Model E (`pf/model_gdm.py`, `pf/runner_gdm.py`) does layer 1 only.
The U-buffer `U_buf[r, s]` carries per-cohort remaining-unreported counts and Sam B has already written the guided-PF + Rao-Blackwellisation for the single-layer case.
The extensions below are listed by increasing scope.

### Step 1 — Expose the U-buffer as a nowcast

Code change is minimal.

The U-buffer in Model E *already* knows, for each cohort still in flight, how many cases are unreported.
A nowcast for cohort `r` at the current time `t` is just

```
nowcast_r^{(t)}  =  Σ_s observed_so_far(r, s)  +  posterior over Σ_s remaining U_{r,s}
```

The "remaining U" piece is already in every particle.
What is missing is plotting code that reads it out and a particle-weighted summary across the cloud.

Cost: ~50 lines in a new `nowcast.py`, plus an example script.
Identifiability: handled — the data already constrains the U-buffer.
Limitation: this assumes no data revisions, only right-truncation.

### Step 2 — Add the revision stick break (Stoner et al double break)

Code change is larger.
State expands; the guided PF needs a second layer of partition; data input changes shape.

State:
```
U_buf[r, s]  unreported, remaining   (already exists)
V_buf[r, s, k]  reported with initial-delay s, currently at revision-age k
```

Observation:
```
Y_{r, t}  =  Σ_{s, k : r+s+k = t}  V_buf[r, s, k]    (cell of the reporting triangle)
```

Per-step transition:
- Sample new infections `N_t ~ Poisson(λ_t)` (as today).
- Layer 1 partition: draw `M_{t, s}` for `s = 1, …, L−1` using the existing guided-Wallenius + Rao-Blackwellised q proposal.
- Layer 2 partition: for each newly-appearing report at delay `s`, draw its initial revision value via a second guided-Wallenius + RB-q proposal, conditional on the cell-of-the-triangle constraint at time `t`.
- Update both buffers.

IS weight: product of layer-1 and layer-2 weights.
Each layer is an instance of the same Bin-against-Wallenius pattern, so the weight algebra is the same; you just write it twice.

**Sam B's caveat in this setting**: the second-layer Wallenius proposal will be even lighter-tailed than the layer-1 one, because the per-revision Beta concentration `M_revision` is typically much smaller (more revision over-dispersion).
That makes the RB-q lift on layer 2 essential, not optional.
Don't try to ship a layer-2 PF with fixed `q_{s,k}`; the tail mismatch will collapse ESS.

Code touches:
- `src/smc_renewal/observation_gdm.py` → split into `observation_delay.py` and `observation_revision.py`.
- `src/smc_renewal/pf/model_gdm.py` → state has both U and V buffers.
- `src/smc_renewal/pf/runner_gdm.py` → two-layer guided proposal in the scan.
- `src/smc_renewal/synthetic.py` → simulator outputs a triangular `Y[r, t]` matrix.
- New file `src/smc_renewal/nowcast.py` → posterior summaries on cohort budgets given partial data.
- New tests: `tests/test_double_stick.py`, `tests/test_revision_recovery.py`.
- New example: `examples/15_double_stick_nowcast.py`.

Cost: 1–2 weeks of focused work, mostly on the guided-PF correctness and a synthetic-data identifiability sweep.

### Step 3 — Apply this to Models A–D (or a "Model F = renewal NN with nowcasting")

You said you want this on the NN model too, or at least the ability to handle data revisions.

Two paths.

#### Path A — Retrofit the double break onto Models A–D

Currently Models A–D use `μ_y(t) = Σ_s d_s · I[t−s]`, a deterministic delay convolution, then NegBin on top.
You can't *add* a cohort partition on top of that — the model already collapsed the cohort structure into `μ_y`.
You would have to rewrite Models A–D to carry the cohort partition explicitly, which makes them look structurally like Model E.

That is fine and arguably the right end state: Models A–D's delay-conv + NegBin observation is the cheap approximation that becomes wrong when ascertainment is high or data is revised.
Once you have Model E's machinery, there is no methodological reason to keep the cheap approximation as a separate variant.
Keep it as a baseline only.

#### Path B — Stand up a new "Model F" that combines

- The dynamics layer of Models A or B (nested-RW volatility on `log Rt`, or Liu-West on `log σ` directly).
- The Model E observation layer extended with the revision stick break (Step 2 above).

This is the cleanest target if you also want the time-varying-`τ` extension (`EXTENSION_time_varying_tau.md`).
The state vector grows but the algebra is already in place from Steps 1–2 here and from the nested-RW machinery in `pf/runner.py`.

### Step 4 — A learned (NN) reporting model that replaces the parametric stick-break shape

Only meaningful if you have enough revisions data to fit a small NN that predicts the per-(stage, revision) Beta means from features (day-of-week, surveillance system version, weekend flag, holiday).

If yes, replace `Φ⁻¹(p_s) = b_0 + b_1·s` with `(p_s, p_{s,k}) = NN(features; weights)`.
Same advice as in `EXTENSION_time_varying_tau.md`: train the NN offline, freeze its weights at PF runtime, and let one or two global scaling parameters in the Liu-West cloud handle real-time adaptation.

This is where the "renewal NN with nowcasting" framing lands.
It is the right design only after Steps 1–3 are working, because the NN learns a shape that the parametric model is failing to capture, and you need the parametric model in place first to know what it fails at.

## Recommended order

1. Step 1: expose U-buffer as a single-layer nowcast. Days of work, immediate operational value. Demonstrates that the GDM cohort structure is already a nowcast machine; it just was not surfaced.
2. Step 2: add the revision stick break. The methods step. Verify on synthetic data with known revisions, compare nowcasts against the locked truth.
3. Compare nowcasts to `epinowcast` on a real dataset where revisions matter. This is the "are we competitive" test.
4. Step 4 (NN reporting model) only if Step 3 shows the parametric shape leaves substantial calibration error that features can explain.

## How this connects back to the talk

For a methods talk the right slide is *not* "we built nowcasting".
It is *"the guided-PF pattern from Model E generalises to a second layer at no extra design cost; the same Rao-Blackwellisation that saves the single-layer case is what saves the double layer; and the limiting factor is data quality on revisions, not the inference machinery"*.

That framing also makes Sam B's IS-variance concern part of the methods story rather than a footnote: the reason the design is principled is that you can predict the failure mode in advance and design the proposal to avoid it, layer by layer.
