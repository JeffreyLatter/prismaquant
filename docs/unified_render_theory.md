# Unified Render Theory — one loop, one variable

**Status:** theory + validation plan (2026-06-11). Nothing here changes a
production default until the validation ladder (§8) clears. Authored on
`claude/aura-improvements` per Robert's directive: *"Fundamentally, we should
be able to have a single uncomplicated render loop. I hesitate to tweak it
based on bespoke testing rather than theory. Take it as far as you can."*
Damp is folded in per 2026-06-11 (*"Real gptq doesn't include a damp sweep so
I think it's ok to defer [the damp study]. Maybe include damp in this task."*).

---

## 1. The thesis

Every per-Linear render decision we currently make by bespoke measurement —
the 5-candidate damp sweep, the experts-get-RTN-but-dense-gets-GPTQ split, the
act-order on/off toggle, the do-no-harm gates — is a special case of **one
statistical question**:

> *How much of the empirical Hessian's structure is evidence, and how much is
> sampling noise?*

GPTQ's error compensation is (exactly, not metaphorically) a sequence of
ordinary-least-squares regressions fitted on the calibration activations
(§2). A regression fitted on `n_eff` effective samples with `d_eff` effective
regressors generalizes iff its signal-to-noise ratio exceeds `d_eff/n_eff`,
and the optimal amount of compensation to apply is a closed-form shrinkage of
that regression (§3). The damp parameter λ *is* that shrinkage, applied
per-eigendirection (§4). So:

- **Dense Linears** (thousands of effective calibration rows, fast-decaying
  activation spectrum): evidence ratio ≫ 1 → shrinkage ≈ 0 → full GPTQ with
  small λ. This is what the damp sweep keeps discovering empirically.
- **Routed experts** (hundreds of routed rows, comparable `d`): evidence
  ratio ≲ 1 → shrinkage ≈ 1 → the law *derives* RTN. This is what the 35B
  six-arm study and the served finale discovered empirically
  (`moe_expert_gptq_vs_rtn` memory: RTN-static6 beat every GPTQ variant on
  served KL).

The unified loop computes the evidence ratio from quantities the GPTQ pass
already materializes, sets λ in closed form (no sweep, no fitted constants),
and lets compensation strength fall out continuously. RTN and GPTQ stop
being two recipes; they are the two ends of one law. The {6,4} scale
measurement (JSO) stays in-loop at every evidence level because its
hypothesis class is ~1 bit per 16-weight group — too small to overfit (§6.3).

This is the same epistemology AURA already ships on the allocation side
(per-row `predicted_dloss_stderr`, UCB charging `z·stderr`): *every empirical
estimate carries its evidence, and every decision discounts by it.* This doc
extends that spine from "which format" to "how to round."

---

## 2. GPTQ is regression: the exact identity

Per Linear: weight rows `w ∈ R^d` (`d = in_features`), calibration activation
matrix `X ∈ R^{n×d}` (n token rows), `H = XᵀX`. GPTQ quantizes columns
sequentially; after rounding column `j` with error `ε_j = w_j − q_j`, it
updates the not-yet-quantized columns `R = {j+1..d}`:

```
w_R ← w_R + ε_j · u_j,   u_j = −[H⁻¹]_{j,R} / [H⁻¹]_{jj}
```

The precision-matrix identity says `u_j = β̂_j`: **the OLS coefficients of
regressing activation channel `x_j` onto the surviving channels `x_R`,
fitted on the n calibration rows.** Equivalently, the in-sample residual
variance of that regression is `σ̂_j² = 1/(n·[H⁻¹]_jj)`.

So GPTQ's compensation replaces the lost output contribution `ε_j·x_j` with
`ε_j·β̂_jᵀx_R` — the best linear *prediction* of the deleted channel from the
channels that remain adjustable. Its entire benefit is the benefit of that
prediction **out of sample** (on deployment text, not calibration text).
Everything below is the standard statistics of when a fitted regression
predicts out of sample.

Two scale-free observables per column, free from the Cholesky factors GPTQ
already computes:

```
R²_j   = 1 − 1/(H_jj·[H⁻¹]_jj)          (in-sample explained fraction)
SNR_j  = H_jj·[H⁻¹]_jj − 1 = R²/(1−R²)  (in-sample signal-to-noise)
```

`H_jj·[H⁻¹]_jj ≥ 1` always, with equality iff channel `j` is uncorrelated
with the rest — i.e. iff compensation has nothing to work with.

---

## 3. The shrinkage law

### 3.1 Scalar form (clean derivation)

Apply the compensation scaled by `s ∈ [0,1]` (s=1 vanilla GPTQ, s=0 RTN).
Model: true regression `x_j = β*ᵀx_R + η`, residual variance `σ_j²`,
explainable signal power `S_j = β*ᵀΣ_R β*`, fitted `β̂ = β* + δ` with the
classic OLS sampling error `E[δᵀΣ_R δ] = σ_j²·d_R/n_eff`. Out-of-sample loss
per unit ε²:

```
L(s) = σ_j² + (1−s)²·S_j + s²·σ_j²·d_R/n_eff
       └ floor ┘  └ under-compensation ┘  └ overfitting tax ┘
```

Minimizing:

```
s*_j = S_j / (S_j + σ_j²·d_R/n_eff) = SNR_j / (SNR_j + d_R/n_eff)
```

With calibration→deployment distribution shift, the transferable signal is
`t·S_j` for a transfer coefficient `t ∈ [0,1]` (estimable — §5.3), giving

```
s*_j = t·SNR_j / (t·SNR_j + d_R/n_eff)        (THE LAW)
```

Sanity limits: `n_eff → ∞` ⇒ `s* → 1` (vanilla GPTQ is optimal with infinite
clean evidence). `n_eff/d_R → 0` ⇒ `s* → 0` (RTN). `t → 0` (calibration
unlike deployment) ⇒ RTN regardless of n. **GPTQ-vs-RTN is not a recipe
choice; it is the value of one ratio.** (The OLS form is for intuition; the
production-relevant regime is n < d and the operative dimension is
`d_eff(λ̃)`, not `d_R` — see the correction at the end of this section.)

**A fact that disciplines the whole derivation:** production builds H from at
most `max_act_rows` activation rows — default 256, 512 in the damp study —
against `d` = 2560–9728 (`_LinearActivationCollector`,
`production_weight_cache.py`). *Production dense GPTQ has always operated at
n < d*, and it demonstrably beats RTN there. So the OLS regime (and any hard
"no evidence at n ≤ d" cutoff) is the wrong frame: the correct frame is ridge
regression in the proportional/overparameterized regime, where the signal
lives in the top of the activation spectrum and a fit with effective
dimension `d_eff(λ̃) ≪ n_eff` still generalizes. The law survives intact with
`d_R → d_eff(λ̃)`; there is no special-case branch, because the fixed point
in §3.2 pushes λ̃* up until `d_eff(λ̃*) ≲ n_eff` — **the law automatically
caps the complexity it trusts at the evidence available.** RTN is just the
deep end of that continuum, never a separate recipe.

In-sample SNR remains optimism-inflated; at n < d the OLS adjusted-R² is
undefined and the SNR must be read from the *ridge* fit with dof
`d_eff(λ̃)` (the V0 implementation detail). Where a dof correction applies
at all, it must be the **uncentered** (no-intercept) form
`R̄² = 1 − (1−R²)·n/(n−dof)` — these are channel-on-channel regressions with
no intercept; the textbook centered formula has a degrees-of-freedom error
(caught by the Gemini referee pass, §10).

### 3.2 Eigen-adaptive form: damp IS the shrinkage

Scalar `s` shrinks all directions equally. Ridge does better: replacing `H`
with `H_λ̃ = H + λ̃I` shrinks the compensation along eigendirection `i` of
`Σ` by `μ_i/(μ_i + λ̃/n)` — most where the empirical eigenvalue is smallest,
i.e. exactly where `β̂` is noisiest. **GPTQ's damp is therefore not a
numerical-stability hack; it is per-eigendirection compensation shrinkage,
and it has an optimal value.** Under an isotropic signal prior
`β ~ N(0, (α²/d)·I)` (Dobriban & Wager 2018, predictive-optimal ridge,
valid at any d/n including d > n), `λ̃* = d·σ̄²/(n_eff·α²)` in Σ-units. The
prior variance relates to signal power by `S̄ = α²·μ̄_Σ` with
`μ̄_Σ = mean diag Σ`, and the code parameterizes damp relative to
`mean diag H = n·μ̄_Σ`, so the units cancel into something memorable:

```
damp* = (d / n_eff) · (1 / SNR̄)
      = evidence deficit ÷ signal-to-noise        (no fitted constants)
```

with `SNR̄ = S̄/σ̄²` aggregated per-Linear from the §2 observables. Since
`σ̂, Ŝ` are estimated under some λ, this is a fixed point — iterate twice
from damp=0.01 (monotone contracting in the relevant range).

Magnitude sanity against the 31-row log: d/n_eff ≈ 5–19 there, winners at
damp 0.001–0.01 ⇒ implied SNR̄ ≈ 500–19000 ⇒ per-column R² ≈ 0.998–0.9999.
LLM activation channels are exactly that mutually predictable, most of all
the attention-head outputs feeding o_proj — which is the role the log shows
wanting the smallest damp.

**V0 status (2026-06-11): the closed form as stated is NOT implementable
from in-sample quantities — measured, not speculated.** At production's
n < d, small-λ ridge interpolates the calibration rows: d_eff(λ) → n, the
dof correction degenerates, in-sample RSS ≈ 0 carries no information about
σ², and the fixed point diverges to damp* → ∞ on 28/31 real 4B Linears
(`v0_law_check_results.json`). The structural law (shrinkage ∝ evidence)
stands; the *estimator* of SNR must come from held-out rows, not in-sample
residuals. See §8 V0/V0b for what that audit then uncovered about the
production sweep itself.

Implementation note (load-bearing): any shrinkage must act **inside** the
loop — scale the OBS update vector, or equivalently use `H_λ̃*` — never by
blending the GPTQ and RTN dequantized outputs afterward. A blend of two
on-grid tensors is off-grid; the rounding must happen after the shrinkage so
the shipped bytes stay exactly representable.

### 3.3 What the law costs

Nothing material. `[H⁻¹]` diagonal: already materialized by GPTQ's Cholesky
inverse. `n_eff`: token-axis effective sample size via ~8 random projections
(§5.1), O(n·d·8). `d_eff(λ̃) = Σ_i μ_i/(μ_i + λ̃)` if wanted for diagnostics:
Hutchinson with the existing Cholesky factor, O(k·d²). One GPTQ pass instead
of five. The 5× damp sweep, the per-case expert recipe, and the binary gate
decision all collapse into arithmetic on quantities we already compute.

---

## 4. Why every previous damp scheme behaved the way it did

- **The 5-candidate sweep works** because it *measures* the bias-variance
  trade per Linear — it is a 5-point grid search for λ̃*. It is correct and
  5× too expensive, and its grid is log-coarse (factor-5 gaps).
- **The κ-target analytical damp failed (+100–161% KL, graveyard)** because
  conditioning is the wrong variable: κ measures whether `H` is *invertible*,
  not whether it is *trustworthy*. κ has no n-dependence — it cannot know
  that 512 routed rows are different from 32k dense rows. The 31-row
  damp-winner log falsifies it directly: winners at damp 0.005 occur at both
  κ=5.8e11 and κ=3.4e17 (§6.2).
- **Fixed damp=0.01 (vanilla GPTQ)** is a constant where the law says the
  optimum moves with `d·σ̄²/(n_eff·ᾱ²)`. On dense 4B Linears it is nearly
  right (median regret +2% in Hessian-weighted error) except on o_proj
  (+47–85%), whose attention-output input is the most cross-channel-
  predictable activation in the block — highest ᾱ², hence smallest λ̃*,
  hence most over-damped by a fixed constant (§6.2).

---

## 5. Observables

### 5.1 `n_eff` — effective calibration evidence

Calibration rows are autocorrelated within a sequence and, for routed
experts, weighted by routing probability. Both discounts are standard:

```
n_w    = (Σ_r w_r)² / Σ_r w_r²            (routing/Fisher row weights)
τ̂      = 1 + 2·Σ_k ρ̂_k                    (integrated autocorrelation time,
                                            averaged over ~8 random
                                            projections of x_t, per sequence)
n_eff  = n_w / τ̂
```

Since production caps dense H at 256–512 rows anyway (§3.1), raw row count
does *not* separate experts from dense — both are n < d. The separation the
law predicts comes from the discounts: an expert's routed rows are clustered
(top-k routing selects a narrow slice of token space, often few sequences →
large τ̂, peaky routing weights → small n_w), and its routing pattern shifts
between calibration and deployment (small t, §5.3 — exactly why the
cross-domain gate was load-bearing on the 35B experts while dense never
needed one). Prediction, measurable in V2: at matched raw rows, experts show
markedly lower n_eff·t than dense Linears.

### 5.2 `SNR_j`, `σ̂_j²`, `ᾱ²` — from the existing Cholesky

As in §2–3. Bias-correct via adjusted-R². Aggregate per-Linear (the Cholesky
is per-Linear; per-column λ̃ would require re-factorization per column — noted
as an open refinement in §9, not worth it until V1 says otherwise).

### 5.3 `t` — transfer, measured by the machinery we already built

The cross-domain do-no-harm gate (main-tree commit `0bd5d9c`) already renders
candidates and scores them on a held-out, *different-domain* corpus. Today it
returns a binary accept/reject. The theory wants its **ratio**:

```
t̂ = (out-of-domain gain of compensated render vs RTN)
    / (in-domain gain of compensated render vs RTN),  clipped to [0,1]
```

pooled per role (per-Linear t̂ is too noisy; per-role — qkv/o/gate-up/down/
expert — is the right granularity). The gate thereby demotes from decider to
**instrument**: it feeds `t̂` into the law, stays as a fail-safe assertion,
and in steady state should fire ~never because the law already withheld
compensation that wouldn't transfer.

---

## 6. Retrodictions (the theory must explain what we already measured)

### 6.1 The 35B expert arc — the headline retrodiction

The six-arm study found: full-stack GPTQ on packed experts (arm F, 16h) dead
last; batched fixed-damp GPTQ with cross-domain gate (arm E, 13min)
decisively better; and the served finale found plain RTN-static6 beating
every GPTQ variant (`moe_expert_gptq_vs_rtn`). Under the law: per-expert
evidence `n_eff·t` is a small fraction of dense (clustered routed rows +
routing shift, §5.1/5.3) → λ̃* = (d/n_eff)/(t·SNR̄) huge → compensation
shrunk to ≈ 0 → **RTN is the derived optimum, not a concession.** Arm E
beat arm F because batching + fixed damp + the gate *approximated* heavy
shrinkage; RTN beat arm E because the true optimum was deeper shrinkage
still. The ordering E > F and RTN ≥ E is predicted, not just accommodated —
but the quantitative premise (experts' n_eff·t ≪ dense at matched raw rows)
is a V2 measurement, not yet a number.

Corollary: the sweep and act-order being measured-negative on experts is
also predicted — with compensation shrunk to ~0, sweeping λ tunes a dead
knob against an in-sample objective (pure selection noise), and processing
order of a no-op is irrelevant.

### 6.2 The 31-row damp-winner log (Qwen3-4B dense, 2026-06-11)

Partial data from the deferred damp study
(`/home/rob/dq-runs/damp-collapse-4b/damp_winners.jsonl`, 31 Linears, full
5-point Hessian-weighted error curves each):

- Winners are **always ≤ 0.01** (13× at 0.001, 13× at 0.005, 5× at 0.01);
  damp 0.1 is never optimal and costs 1.3–10× — dense 4B Linears at 8×1024
  calibration sit deep in the high-evidence regime, as the law requires.
- **κ does not predict the winner** (κ spans 1e11–1e17 within a single
  winner bucket) — the graveyard's κ-target rejection, re-confirmed in-data.
- **o_proj `[2560,4096]` is the entire heavy tail**: always wins at 0.001,
  fixed-0.01 regret +47–85% while every other role's regret is ≤ 5%, and
  fixed-0.1 regret up to 10.4×. *(Held-out postscript, V0b: this entire
  structure is in-sample artifact — o_proj's input is the most predictable,
  so it shows the most apparent in-sample signal and overfits hardest; its
  held-out optimum is damp ≈ 1.0 like the rest of attention. Kept here as
  the cleanest demonstration of why in-sample winner logs cannot be trusted
  as ground truth.)*

### 6.3 Why JSO/{6,4} is safe at every evidence level

The in-loop scale choice selects 1-of-2 levels per 16-weight group against
the same empirical objective: hypothesis-class complexity ~1 bit per group,
n rows of evidence per group. Overfitting tax ~σ²·ln2/n per group —
negligible at any calibration size we use. That is why 4over6 measured
*neutral* on thin-evidence experts (bounded class ⇒ bounded harm) while GPTQ
compensation (a d-dimensional regression) measured *negative*. The law keeps
measured scales unconditionally and shrinks only the regression.

### 6.4 The honest tension: the +137.5% sweep-removal measurement

`damp_sweep_4b_essential` (2026-05): disabling the sweep (→ fixed 0.01) cost
+137.5% KL on Qwen3-4B. **Metric-tier correction (checked at source,
2026-06-11):** that number is *last-token hook KL* — the triage screen, tier
5 in the metric authority ladder — on n=2 trials; it was never confirmed on
served full-vocab KL. The same run's local activation-weighted MSE actually
*favored* sweep-off slightly, and its per-kind breakdown fingered
`self_attn.o_proj` as the only kind where the sweep helps locally (+5.45%) —
independently re-found by the 31-row log (o_proj is the entire fixed-damp
regret tail) and consistent with the law (highest cross-channel
predictability → optimum farthest from any fixed constant). So the "sweep is
essential" claim is screen-grade evidence of *something real centered on
o_proj*, not a gold-metric fact. The 31-row local regrets (median +2%, tail
+85% on o_proj) summing to far less than +137% under fp32 additivity is
consistent with the screen overstating it. **Still unresolved and still
blocks any production change** — V1 measures sweep vs fixed vs CV-selected
on the gold metric. Until then the sweep stays the default (defaults stay
backwards-compatible; everything here ships as opt-in levers).

---

## 7. The single uncomplicated loop

```
per Linear (dense or expert, identical code path):
  1. H = XᵀX  (routing/Fisher row weights as today)
  2. n_eff  (ESS: weights + autocorrelation, §5.1)         ~free
  3. Cholesky of H_λ (start damp=0.01); read [H_λ⁻¹]_jj     already computed
  4. damp* selection — in order of simplicity (all pending V1 served):
       a. simplest, V0c-validated locally: a SINGLE fixed damp in 0.3–1.0
          (no sweep, no selector; within ~2% of per-Linear optimal held-out;
          5× faster than today). One pass. This is the "uncomplicated loop".
       b. refinement (~2% more, sweep-cost): held-out CV — fit candidates on
          half the rows, score on the other half, refit winner on all rows;
       c. closed form (open): damp* = (d/n_eff)/(t̂·SNR̄) with a held-out
          SNR estimator — in-sample version diverges (§3.2 V0 status);
          leave-column-out GCV is the candidate
     (thin/shifted evidence ⇒ damp* large ⇒ compensation → 0: the render
      degrades continuously into RTN + measured {6,4} scales; this IS the
      expert path, derived not configured)
  5. ONE GPTQ pass with H_{λ̃*}, scales by in-loop {6,4} measurement (JSO),
     static_act_order as today (irrelevant when s*≈0, helpful when s*≈1)
  6. Gates run as instruments: record t̂ per role (§5.3), assert do-no-harm
```

What dies if validation clears: the 5-candidate damp sweep (5× hot-path
cost), the dense-vs-expert recipe fork (`--expert-render-mode` remains a
*performance* choice — batched vs per-expert execution — but the *recipe*
unifies), per-case act-order/sweep toggles for experts, binary gate
decisions. What stays untouched: the {6,4} measured scale rule, fused-
sibling/packed-expert format coherence, the one-cache architecture, every
serialization contract.

---

## 8. Validation ladder (graveyard bar explicit)

- **V0 — RUN (2026-06-11), and it failed forward.** Recomputed H for the 31
  logged 4B Linears and evaluated `damp* = (d/n_eff)/SNR̄`: the in-sample
  fixed point **diverges** (interpolation regime, §3.2 status note) —
  honest negative, the closed form needs a held-out SNR estimator. The audit
  then surfaced something bigger: **the production damp sweep's evaluator is
  itself in-sample** — `_gptq_obs_rounding_nvfp4_swept` fits each candidate's
  compensation on H and scores `tr(diff·H·diffᵀ)` on the *same* H from the
  same ≤512 rows. At n < d that systematically rewards overfit,
  under-damped compensation. The "measured winners" the sweep returns are
  biased, and the bias direction is exactly the §9.1 anti-conservative one.
  Measurement gap, not optimizer gap — the house diagnosis, found *in our
  own evaluator* by following the theory.
- **V0b/V0c — held-out basins, COMPLETE (2026-06-11):** 1024 reservoir rows
  per Linear; GPTQ-fit each damp in {0.0003…1.0}∪{3,10,100}∪{RTN} on a
  production-faithful 512 rows, score Hessian-weighted error on the held-out
  512. All 31 Linears, all basins interior. Results
  (`v0b_heldout_results.json` + `v0c_supplement_results.json`):
  - **31/31 in-sample winners are under-damped.** Held-out optima sit at
    damp 0.1–3.0 (attention → 1.0, gate/up → 0.1–0.3, down_proj → 3.0) —
    one to three *orders of magnitude* above the in-sample winners
    (0.0003–0.01) and above the production grid's center of mass.
  - **The in-sample sweep is worse than the constant it replaces:** geomean
    held-out regret vs per-Linear optimum — in-sample-sweep winner **+35.4%**,
    fixed-0.01 **+26.1%**. The 5× sweep pays compute to pick *worse* damps.
  - **RTN loses decisively on dense:** geomean **2.04×** (up to 14.3× on
    gate_proj). Compensation transfers at n < d — it just needs ~100× more
    shrinkage than the in-sample evaluator chooses. The law's dense-regime
    prediction (partial compensation ≫ both endpoints) is confirmed; the
    held-out objective is sane (it does not collapse to RTN).
  - **The basin top is wide and flat — one constant nearly suffices:** any
    single fixed damp in **0.3–1.0** gets within **+1.6–2.6% geomean
    (worst +7–12%)** of the per-Linear optimum. Per-Linear selection buys
    only ~2% beyond the right decade. This is the strongest possible
    endorsement of the "single uncomplicated loop": no sweep, no selector —
    one theory-located constant (Robert's "fix it to 0.1" instinct lands
    within 8% of optimal; 0.3 is the data's center).
  - The in-sample o_proj structure (§6.2's "regret tail") inverts held-out:
    o_proj's apparently-special tiny-damp preference was the *most overfit*
    pick (most predictable input ⇒ most apparent in-sample signal); its
    held-out optimum is 1.0 like the rest of attention. The May screen's
    per-kind o_proj finding was likewise in-sample.
  - Naive GCV-by-rows on the full design is degenerate (the target column
    sits in its own design; picks λ→0) — the closed form needs the
    leave-column-out variant, open (§9.7).
  Caveats: local held-out MSE (V1 served is the gate); 4B layers 0–4 at
  512-row fits; JSO off in these fits (scale-rule interaction untested).
- **Bridge caveat for V0b→V1:** held-out *output-MSE* is still a local
  proxy; the selection objective that matches the platform's currency is
  held-out **AURA cost** of the rendered candidate (the same Fisher-quadratic
  the allocator prices — `aura_cost.py` prices rendered dW directly). Local
  proxies have inverted against serving twice this spring (scale_sweep,
  grouped-KL); V1 is served-KL for exactly this reason.
- **V1 — RUN (2026-06-11), first calib draw: the basin direction TRANSFERS
  end-to-end.** Qwen3-4B, frozen 4.75-bpp allocation, four arms (in-sample
  sweep / fixed-0.01 / fixed-0.3 / fixed-1.0), deterministic renders,
  in-process vLLM measurement (all-position top-1024 KL vs shared BF16
  teacher on WikiText-train windows + WikiText-test PPL + max-chunk NLL).
  Results (`/home/rob/dq-runs/v1-damp-ab/metrics/`):
  | arm | KL mean | KL conf | KL p99 | PPL | max-chunk NLL |
  |---|---|---|---|---|---|
  | sweep | 0.5927 | 0.5284 | 5.79 | 27.145 | 3.7248 |
  | 0.01 | 0.5733 | 0.5063 | 5.80 | 27.237 | 3.7425 |
  | 0.3 | 0.5248 | 0.4479 | 5.04 | **26.757** | **3.6763** |
  | 1.0 | **0.5033** | **0.4267** | **4.79** | 26.892 | 3.6976 |
  Both heavy-damp arms beat the production sweep on EVERY readout
  (−11/−15% KL, −13/−17% p99, −1.4/−0.9% PPL, tail improved) despite the
  pre-registered headwind (allocation optimized under sweep rendering).
  C-vs-D is lane-split (KL prefers 1.0, PPL prefers 0.3) and within the
  predicted flat basin. **§6.4 RESOLVED: fixed-0.01 is KL-better than the
  sweep** — the +137.5% screen claim inverted on the gold lane; the sweep's
  5× cost buys a measurable negative. Builds: 11 min/arm fixed vs ~48 min
  sweep (4.4×). Caveats: single calib draw (seed-43 replication of A/C/D
  running), single model; promotion ladder still requires the second seed +
  a 27B confirmation. Mid-run instrument note: the legacy "full-vocab KL"
  lane scores only the 8 window-final contexts (teacher_shape [8, V]) —
  paired A-vs-B there was 0.008±0.29, useless; the all-position top-K mode
  was added (commit abc90a4) and is what the table reports.
- **V2 — regime audit, no new builds:** compute ν = n_eff/d and adjusted-R²
  for the 35B packed experts and the 27B dense Linears from existing caches;
  the law must place experts in the s*≈0 branch and dense in s*≈1. Pure
  retrodiction with numbers attached.
- **V3 — arm G (the unification with teeth):** render 35B experts with λ̃*
  instead of binary RTN-vs-GPTQ. Prediction: ≥ RTN (the law can only
  withhold harmful compensation or admit helpful partial compensation where
  t̂·SNR is mid-range). If arm G edges RTN on served KL, partial compensation
  is real and the binary fork was leaving quality on the table.
- **V4 — the discriminating new prediction:** on experts, grow calibration
  until n_eff crosses d (≈64× tokens at 128-expert/top-8): the law predicts
  GPTQ's served gain crosses zero from below at a *computable* calibration
  size. No existing recipe encodes an n-dependent crossover; κ-target and
  fixed-damp both predict no such crossover. This is the cleanest
  falsification test of the whole theory.

Promotion follows the standard ladder: research lever
(`PRISMAQUANT_DAMP_ANALYTICAL=law` or successor env) → candidate after
V0+V1 → default only after V2–V3 and a second model/shape.

---

## 9. Open math (known-unknowns, honestly)

1. **Sequential coupling:** the OBS errors ε_j depend on previous
   compensations, which are themselves functions of X — so ε_j is correlated
   with the design matrix and the §3.1 cross terms do not exactly vanish.
   Referee assessment (Gemini): the un-modeled correlation plausibly biases
   toward *under*-shrinkage (the law is anti-conservative at the margin),
   and the per-column analysis ignores the cumulative drift of later columns
   into higher-ε regions. V1's end-to-end arm is the safety net; if λ̃*
   consistently lands below the swept optimum in V0, this is why.
2. **Per-column vs per-Linear λ̃:** the law is naturally per-column; the
   Cholesky is per-Linear. V0 will show whether per-Linear aggregation loses
   anything (suspicion: o_proj vs rest is a *role* effect, so per-Linear
   suffices).
3. **τ̂ robustness:** integrated autocorrelation estimators are themselves
   noisy on short sequences, and n_eff enters the law linearly — a 2× error
   in τ̂ is a 2× error in damp*. Quantify on real calibration data whether
   the per-damp-error basin is wide enough (the 31-row curves suggest a
   factor-~3 basin around the optimum on most rows) to absorb it.
4. **Isotropic-prior assumption** behind λ̃* (Dobriban–Wager): activation
   signal is concentrated in the top eigendirections, not isotropic, so the
   scalar λ̃* over-shrinks the most informative compensation directions and
   under-shrinks the tail (referee point e). The sharper formulation is a
   diagonal shrinkage in the eigenbasis of H,
   `s_i = μ_i/(μ_i + σ²/(n·σ²_{β,i}))` with per-direction signal prior
   σ²_{β,i} — equivalently Ledoit–Wolf *nonlinear* shrinkage. That is the
   upgrade path if V0 shows the scalar law systematically off; it costs one
   eigendecomposition per Linear, still cheaper than the 5× sweep.
5. **Interaction ordering:** {6,4} scale choice is made under the λ̃*-damped
   compensation but the law's σ̂/Ŝ were estimated pre-rounding. Second-order;
   flag, don't fix preemptively.
6. **t̂ estimation noise** per role: needs the gate corpus sized so t̂'s
   stderr ≪ its distance from the s* decision boundary.
7. **Closed-form damp — adversarially settled (2026-06-11, workflow
   wf_80103582; scripts `scratch/ridge_claims_check.py` +
   `scratch/damp_referee_v0d.py`).** The algebra lens confirmed every
   identity exactly (β̂_j = −P[j,−j]/P[jj] for ridge by Schur complement —
   the λ on the (j,j) diagonal cancels in the ratio; held-out residual
   r_Bj = X_B P[:,j]/P[jj]; ‖r_Bj‖² = [P H_B P]_jj/P[jj]². Landmine for
   implementers: 1/P_jj ≠ RSS at λ>0 — exactly 1/P_jj = ‖r‖²+λ(1+‖β̂‖²)).
   The statistics lens **refuted the split-half D-W estimator as designed**:
   (a) the isotropic prior undershoots λ* by 4–9× on top-aligned activation
   signal — fails the ≤1.05 bar even with oracle inputs; (b) split-half
   noise estimates the half-fit's excess risk, not transferable σ² (SNR
   −36–59%), and the fixed point converges to the inflated optimum or
   diverges outright at low SNR (82→651 monotone); (c) the n_fit/n_build
   factor-2 "bug" was masking (a) by cancellation; (d) the shrinking-
   regressor-set slack (~×0.5 in damp) alone exceeds the validation budget.
   **The successor is strictly better and cheaper: per-column
   leave-column-out GCV on the full reservoir** — from one P per grid λ:
   exact RSS_j = [P H P]_jj/P_jj², exact dof_j = (d−1) − λ[(tr P − P_jj) −
   (‖P[:,j]‖²−P_jj²)/P_jj], GCV_j = (RSS_j/n)/(1−dof_j/n)², minimize Σ_j
   GCV_j over a 1-D λ grid (one eigh → O(d²) per grid point). No split, no
   prior, no fixed point; uniformly consistent for ridge predictive risk at
   arbitrary β in the proportional regime (Patil et al. 2021, Hastie et al.
   2022); matched the true grid argmin in all 6 referee worlds including
   both divergence corners. Remaining known gaps: the ~×0.5 shrinking-set
   correction, ε²-weighted column aggregation, near-iid rows. Real-data
   gate (V0d′): GCV-damp vs the 31 measured held-out GPTQ basins.
   Production note: the referee predicts fixed-0.3 ≈ ×1.02 geomean there —
   the closed form's value is regime portability (experts, other calib
   budgets), not beating the constant on dense.

---

## 10. Provenance

- Damp study deferred by Robert 2026-06-11 (*"Real gptq doesn't include a
  damp sweep"*); partial data (31 rows) preserved at
  `/home/rob/dq-runs/damp-collapse-4b/` with `DEFERRED.md`.
- Six-arm 35B study + served finale: `moe_expert_gptq_vs_rtn` memory;
  build artifacts under `/home/rob/dq-runs/aura-35b/`.
- κ-target analytical damp rejection: CLAUDE.md graveyard +
  `damp_analytical_rejected` memory; code path still present
  (`PRISMAQUANT_DAMP_ANALYTICAL`, `export_native_compressed.py`) as the
  natural insertion point for the law.
- AURA stderr/UCB epistemology: `prismaquant/aura_cost.py`,
  `allocator_candidates.cost_entry_predicted_dloss`,
  `PRISMAQUANT_COST_UCB_Z`.
- Gemini max-effort referee pass (2026-06-11):
  `/home/rob/dq-runs/damp-collapse-4b/gemini_sanity_review.txt`. Verdict:
  identities and D–W transcription correct; not double-counting; two
  substantive corrections folded in above (uncentered dof form in §3.1;
  ε–X sequential correlation → under-shrinkage risk in §9.1) plus the
  eigen-diagonal sharpening in §9.4.
