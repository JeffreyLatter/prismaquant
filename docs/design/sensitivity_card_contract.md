# The Sensitivity Card: probe once, price any format menu

**Status:** implemented, unit-tested, **not yet validated on a served artifact.**
The scalar tier is a byte-identical refactor of today's behaviour. The marginal
tier and AQUA-AURA are **research-tier** until a served A/B exists.
**Date:** 2026-08-14

---

## 1. The problem

Probing a model is the expensive, model-specific half of PrismaQuant. Choosing
formats is the cheap, *platform*-specific half. Today they are fused: `probe.pkl`
carries scalars that only become a cost next to a **rendered menu cache** built
for one particular format list, so:

- a new format menu means re-rendering the menu (and at CB-menu scale, the
  disk projection that motivated this work);
- a probe cannot be *shared*, because it is not sufficient on its own to price
  anything a downstream author might want;
- W4A4 and W4A8 are literally the same candidate, because the cost is
  weight-space only.

The goal Rob set: **probe a model once; the probe is shareable, small, and
requires only that the author specify their downstream format/platform.**

## 2. The key observation

`incremental_probe.py` already computes the per-element diagonal empirical
Fisher of each weight matrix:

```python
chunk_h = gy2_sq.t() @ x2_sq          # H[o,i] = sum_t g[t,o]^2 * x[t,i]^2
```

Storing `H` is what makes a full-detail probe unshippable — the unified-sweep
path documents it as *"47k x 17 MB = 800 GB CPU, doesn't fit"* and therefore
keeps only two scalars, `h_trace` and `h_w2_sum`.

But **every quantity a format-agnostic cost needs is a marginal of `H`**, and
each marginal is a pair of reductions that never forms the `[out, in]` matrix:

```
fisher_row[o] = sum_i H[o,i] = gy2_sq.t() @ x2_sq.sum(dim=1)     # [out]
fisher_col[i] = sum_o H[o,i] = gy2_sq.sum(dim=1) @ x2_sq         # [in]
```

Cost: `out + in` floats instead of `out * in`. Both reduction vectors are
*already materialized* by the existing `h_trace` line, so this is nearly free
and — critically — it is available even on the memory-bounded unified sweep that
cannot accumulate `h_full` at all.

Two further vectors are stored because they are **not** recoverable from the
weight-Fisher marginals:

```
act_sq_sum[i] = sum_t x[t,i]^2      # the imatrix / diag(X^T X)
g_sq_sum[o]   = sum_t g[t,o]^2      # the OUTPUT-space Fisher diagonal
```

`act_sq_sum` lets any format weight its weight error by the activation
distribution the layer actually sees. `g_sq_sum` is what turns an *output*
perturbation into a loss delta — the term AQUA-AURA is built on, and the one
thing `h_trace` structurally cannot supply.

**Free consistency check:** `sum(fisher_row) == sum(fisher_col) == h_trace_raw`.
`SensitivityUnit.validate()` enforces it, which catches the whole class of bugs
where one accumulator is normalized and another is not.

## 3. Measured sizes

| model | units | params | full `H` | **card** | ratio |
|---|---|---|---|---|---|
| Qwen3.6-27B dense | 505 | 26.0 B | 104.2 GB | **75.2 MB** | 1385x |
| Qwen3.6-35B-A3B | 391 | 34.1 B | 8.2 GB | **17.4 MB** | 469x |
| Qwen3-0.6B dense | 197 | 0.6 B | 2.4 GB | **7.4 MB** | 321x |
| MiniMax-M2.7 (per-expert rows) | 47,865 | 228.0 B | 0.9 TB | **2.3 GB** | 403x |

A 75 MB card for a 20 GB artifact is a 0.3% download. Scalar-only cards (built
from any existing probe, no re-probe) are ~1 MB even at 228 B params.

**Note on MoE:** card size tracks the number of probe *rows*, not parameters.
A probe that keeps rows per unpacked expert (MiniMax: 47,865 rows) produces a
much larger card than one that keeps packed-expert rows (35B: 391 rows). If
per-expert granularity is not needed downstream, packing before carding is the
lever. This is the one case where the card is not yet "small", and it is
flagged rather than hidden.

## 4. Why the storage projection collapses

The 18 TB figure is *consistent with* the cost of a **rendered menu cache**:
every (Linear, format) pair rendered and stored so its error can be measured.
(I did not locate the original derivation, so treat the attribution as inferred
— the measured card sizes in §3 answer the question either way.) The card
removes the need for such a cache entirely, because weight error is computed
**locally** from `W` plus the format's own quantizer:

```python
class FormatCostPlugin(Protocol):
    descriptor: FormatDescriptor
    def weight_error(self, unit, weight) -> np.ndarray: ...   # [out, in] squared error
```

A consumer quantizing a model already has its weights on disk. They do not need
our rendered bytes — they need our *sensitivity*, which is `O(out + in)`.
Rendering is then done **once, for the chosen assignment only**, at export.

This is also why the card is format-independent: adding a format is adding a
plugin, not re-probing and not re-rendering a menu.

## 5. The seam is unchanged

`allocator_solver.py` already defines the whole optimizer contract:

```python
@dataclass
class Candidate:
    fmt: str
    bits_per_param: float
    memory_bytes: int
    predicted_dloss: float
```

This work **feeds** that seam rather than replacing it. `CostComponents.to_predicted_dloss()`
produces the one float the multi-choice knapsack DP consumes, so an arbitrary
format menu becomes an arbitrary list of plugins and the solver is untouched.

Three fidelity tiers, selectable per run:

| tier | weight cost | notes |
|---|---|---|
| `SCALAR` | `0.5 * h_trace * weight_mse` | **exactly today's behaviour**; the fallback when a card has no vectors |
| `MARGINAL` | `0.5 * (row @ dW^2 @ col) / h_trace_raw` | rank-1 reconstruction of `H`; the scalar tier is its rank-0 collapse |
| `AQUA` | marginal + activation term | W4A4 and W4A8 stop being the same candidate |

The marginal form is **exact** when `H` is genuinely rank-1, which is the
sharpest available check on the quadratic form and its normalization; there is a
unit test asserting it to `rtol=1e-10`.

## 6. AQUA-AURA

`h_trace` is a **weight-space** curvature. An activation-quantization error is an
**input-side** perturbation `x -> x + dx` reaching the loss as `dy = W dx`.
Multiplying an input-side error by a weight-space sensitivity is a currency
error of exactly the kind `activation_fair_pricing.py` is a 120-line autopsy of
(84% rung-order violations when one family was priced on two bases), so this
module refuses to do it.

Under a diagonal model:

```
E||W dx||^2 = sum_j var(dx_j) * ||W[:,j]||^2
dLoss_a    ~= 0.5 * sum_o g_sq[o] * (W[o,:]^2 . var_dx)
```

— through `g_sq_sum`, never `h_trace`.

Design rules held:

- An unmeasured A-side returns **`None`, never `0.0`**, so a missing measurement
  can never read as a free one.
- The quantizer's `1/12` step variance is a property of a uniform grid, not a
  tuned constant.
- **No speed/quality scalarization constant.** Speed (`speed_index`) and quality
  (`predicted_dloss`) are returned as separate axes; choosing between them is a
  frontier selection, matching how the byte budget already selects a shipping
  point. Inventing a weighting constant would violate "no heuristics when an
  explicit exists".
- `activation_fair_pricing.py` is **left untouched**. Superseding it is a
  promotion decision on served evidence, not a drive-by refactor.

## 7. Design rules the card holds

**Structure travels; policy does not.** The card carries sibling *identity*
(q/k/v are siblings in one block), shapes, and source dtype — properties of the
checkpoint, true for every consumer. It carries **no** serving policy: that
fused siblings must share a format, and that packed experts need vLLM canonical
scheme names, are properties of the downstream runtime and are derived from the
profile the author names. Baking vLLM's packing into a shareable file makes it
wrong for llama.cpp, and wrong the day vLLM changes.

**Calibration is identity.** A CB codebook hashes its imatrix into its book key;
a card is the same kind of object and gets the same rule. `assert_compatible`
refuses cross-calibration merges and comparisons outright.

**Render basis is stamped, not assumed.** A shareable card is necessarily RTN:
compensated renders need per-Linear Hessians (~100 MB/Linear at 27B scale), which
are not shippable. This matters concretely — RTN-vs-compensated `dW` is
*immaterial at fp4 but ~+36% at fp8*, so a card priced on one basis mis-ranks
8-bit rungs on the other. Mismatched bases refuse to compare.

**Currency is explicit and fail-closed.** Only `DELTA_LOSS` may leave the module
toward the solver, because only loss is additive across units.

**Passthrough integrity.** BF16/FP8_SOURCE are legal only when the source dtype
already matches; the coster returns `None` rather than synthesizing them.
Formats are never rejected for looking risky — banning formats in the coster is
the post-allocator-rewrite antipattern.

**No pickle.** A shareable artifact must load without executing arbitrary
objects. The card is a single compressed `.npz` with a JSON header.

## 7b. FIRST REAL-MODEL RUN (2026-08-14): two bugs, and an lm_head blocker

The marginal emission was default-ON and had only ever been exercised by
synthetic unit tests. Its first run against a real model — Qwen3-0.6B, n=8,
T=512, `--emit-marginals`,
`/home/rob/dq-runs/aura-card-marginals-0p6b/` — found the following.

**Bug 1 (fixed): the probe crashed instantly.** `_compute_precompute_key()`
was called with `emit_marginals=...` but its signature never took the argument
(`TypeError`). The call site's reasoning was right — the resident marginals are
written *into* the cached stats, so a precompute cache built with the flag off
must not be reused with it on — so the fix adds the parameter and puts it in
the key rather than dropping it at the call site.

**Bug 2 (fixed): `worst_unit` could name the wrong Linear.** In
`tools/validate_probe_marginals.py`, the name list was filtered by key presence
and the value array by finiteness, so a single non-finite entry misaligned the
`argmax` index.

**The identity holds, and holds tightly.** 197/197 units carry marginals; zero
negative and zero non-finite entries across all vectors.

| check | median | p99 | max | over 1e-4 |
|---|---|---|---|---|
| `sum(row)` vs `sum(col)` | 2.3e-09 | 4.3e-08 | **6.1e-08** | **0 / 197** |
| `sum(row)` vs `h_trace_raw` | 2.3e-08 | 1.0e-07 | 1.0e-03 | 1 / 197 |
| `sum(col)` vs `h_trace_raw` | 2.3e-08 | 9.8e-08 | 1.0e-03 | 1 / 197 |

**So `rtol=1e-4` is empirically defensible** — it clears the real distribution
by roughly three orders of magnitude on 196 of 197 units.

**The blocker: `lm_head` fails `validate()`.**

```
ValueError: lm_head: sum(fisher_row)=1.80171e+08 does not match
h_trace_raw=1.80355e+08. The marginals and the trace must come from the
same accumulator.
```

The diagnosis is unambiguous and the assertion is doing its job:

| lm_head | value |
|---|---|
| `sum(fisher_row)` | 1.801713770e+08 |
| `sum(fisher_col)` | 1.801713709e+08 |
| row vs col | **3.4e-08** — the marginals agree with *each other* |
| `h_trace_raw` | 1.803550720e+08 |
| row vs trace | **1.0e-03** — both disagree with the trace |

The marginals are self-consistent; it is `h_trace_raw` that comes from a
different accumulator. That matches the probe log: lm_head is the only unit in
a resident-only shard (`"shard has only resident Linears (n=1); skipping
Phase-3 reverse sweep"`), so it never goes through the sweep that produces
every other unit's trace. A 1e-3 relative gap is bf16-accumulation territory
(bf16 eps 3.9e-3), consistent with the two paths differing in accumulation
precision or order.

**This is not fixed, and I did not weaken the assertion to make it pass.** A
correct assertion firing on a real inconsistency is a finding, not an obstacle
(principles 1 and 2). But as it stands **no card can be built from any probe
that includes lm_head**, which blocks the lane. The two candidate resolutions,
for Robert to pick:

1. **Make lm_head's trace use the marginal accumulator** — fixes the root
   cause, but touches the shipping probe's resident path.
2. **Exclude lm_head from the card as a non-allocatable unit** — principled
   rather than a fudge, because principle 12 already excludes `lm_head` from
   bpp accounting and it is never assigned a format. The card is a sensitivity
   card *for allocation*; carrying a unit that is never allocated is what
   created the exposure.

Option 2 is the smaller change and matches an existing project invariant;
option 1 is the one that makes the number right. **Neither is implemented.**

**Incidental measurement worth keeping:** the row-Fisher is *flat*, not
concentrated — the median unit needs **92.9%** of its output rows to hold 99%
of the Fisher mass (max 98.4%). Tempering expectations accordingly: the
MARGINAL tier's advantage over SCALAR will come from non-uniform *error*
placement, not from a few dominant output rows.

---

## 7c. Why the card cannot yet reproduce a production allocation (2026-08-14)

Chasing the end-to-end DP check turned up a structural gap that is worth more
than the check itself.

`price()` computes

```python
dw_sq      = plugin.weight_error(unit, weight)
weight_mse = float(np.mean(dw_sq))
```

— i.e. the weight error **always** comes from the plugin's own render. The only
concrete plugin, `RegistryFormatPlugin`, renders with the format's **RTN**
quantizer. Production `weight_mse` in a shipped `cost.pkl` comes from the
**GPTQ + JSO compensated** render. There is today **no plugin that serves a
measured, compensated `weight_mse`**, so the card path cannot reproduce a
production allocation — and it should not be expected to.

This is exactly why the zero-churn test in §8 is scoped the way it is: it feeds
the production `weight_mse` into `weight_dloss_scalar` *directly*, which
isolates the **pricing arithmetic** (proven bit-identical, 22,176 pairs) from
the **render basis** (not yet bridged). Both statements are true and neither
implies the other.

**Latent risk, and it is the project's classic one.** `render_basis` is carried
into `CostComponents` as provenance and is **never checked** against what the
plugin actually did. Nothing prevents a card stamped `COMPENSATED` from being
priced by an RTN plugin, or a menu mixing bases across formats. Principle 8
exists because the surrogate, the KL validation and the exported bytes must be
*one* rendering — an unenforced basis stamp is precisely the "rendering
confound" that reverted the JSO wall-off. Compare
[[factorization_currency_dependent]]: `production-render-score` is already
known to be unlicensed across a codebook-basis change.

**Two follow-ups, in order:**
1. **Enforce the basis.** Have the plugin *declare* the basis it renders in and
   have `price()` refuse a mismatch against the card's provenance, rather than
   stamping whatever it is told. Fail-closed, matching the house pattern.
2. **Add a `MeasuredCostPlugin`** that serves `weight_mse` (and ideally `dw_sq`)
   from an existing `cost.pkl` under `RenderBasis.COMPENSATED`. That is the
   object that makes an end-to-end DP re-solve meaningful; only then does
   "adopting the card changes no allocation" become testable.

---

## 8. What is proven, and what is not

**Proven on REAL production artifacts** (`tools/validate_card_zero_churn.py`,
2026-08-14) — the scalar tier prices bit-for-bit like the shipping allocator on
two completed Qwen3.6-27B runs:

| run | units | formats | pairs | bit-identical | worst rel. dev. |
|---|---:|---:|---:|---:|---:|
| `prod-27b-nvfp4cb-5p5` | 505 | 7 | 3,528 | **3,528** | 0.0 |
| `prod-27b-cb-20gb` | 505 | 37 | 18,648 | **18,648** | 0.0 |

**22,176 real (unit, format) pairs, zero deviation.**

Be precise about what that is and is not. What was measured is a **pricing
identity**: given the same `h_trace` and `weight_mse`, the card's SCALAR tier
and `allocator_solver.predicted_dloss` return bit-identical values. What a
"switching the pipeline changes nothing" claim *additionally* requires is
**candidate-construction identity** — that `candidates_from_card`'s
bits_per_param / memory_bytes / legality / fused-sibling promotion match
`allocator_candidates.py` closely enough that the DP reproduces a shipped
`layer_config.json`. **That end-to-end comparison has not been run.** It is the
next check, and it is cheap: re-solve a completed run from the card and diff the
assignment. (This says nothing about MARGINAL or AQUA, which are *supposed* to
differ; see below.)

**Proven (unit tests, 17/17):**
- the scalar tier reproduces `allocator_solver.predicted_dloss` *exactly*;
- the marginal tier is exact on a rank-1 Fisher (`rtol=1e-10`);
- the marginal tier distinguishes error placed on high- vs low-sensitivity
  channels where the scalar tier is provably blind;
- W4A4 prices strictly above W4A8 under `AQUA`, and **identically** under the
  weight-only model — the AQUA-AURA thesis in executable form;
- round-trip, pickle-free load, passthrough refusal, calibration/basis refusal.

**NOT proven:**
- No served A/B. The marginal tier and AQUA-AURA are **screening surrogates**
  until exact full-vocab vLLM KL-vs-BF16 + direct WikiText PPL says otherwise,
  per the standing rule that a screen is never sold as a result.
- The rank-1 reconstruction is an approximation whenever `H` is not rank-1.
  Its error is unquantified on real layers.
- `act_absmax` is a max over calibration tokens and will understate a true
  serving outlier.
- The `8 * sigma` Gaussian range fallback (used only when `act_absmax` is
  absent) is a surrogate, which is why `act_absmax` is preferred and captured.

- **The probe's marginal emission has never run on a real model.** It is
  default-ON but has only been exercised by synthetic fixtures — no real hooks,
  no real bf16 accumulation. `tools/validate_probe_marginals.py` exists to
  measure the identity `sum(fisher_row) == sum(fisher_col) == h_trace_raw` on a
  real probe and *report* the empirical tolerance rather than assume one. **Run
  it before any probe that matters.**

**Required before promotion:** (a) rank-agreement of marginal pricing against
measured `output_mse` on a small model, (b) ~~allocation churn vs a shipped
`cost.pkl`~~ — **done for SCALAR, exactly zero on 22,176 real pairs**; still
open for MARGINAL, (c) a served W4A4-vs-W4A8 A/B before AQUA-AURA is default-on,
(d) `validate_probe_marginals.py` green on a real probe.

## 9. Publishing (prismaquant.org)

A card is one `.npz` plus its fingerprint. What a publisher needs:

- `model_id`, `calib_hash`, `n_calib_samples`, `seq_len`, `probe_commit`,
  `render_basis` — all already in the header, and `fingerprint()` hashes the
  identity-bearing subset.
- Consumers refuse mismatched calibration and basis automatically, so a
  mis-shared card fails loudly rather than silently mis-ranking.

The author-facing contract is exactly Rob's requirement: **download a card, name
your format menu and platform, get an allocation** — with no probe, no menu
render, and no access to our calibration.

## 10. Files

| file | role |
|---|---|
| `prismaquant/sensitivity_card.py` | schema, invariants, `.npz` I/O, currency/basis/calibration refusals |
| `prismaquant/format_cost_protocol.py` | `FormatCostPlugin`, the three cost tiers, AQUA-AURA |
| `prismaquant/sensitivity_card_build.py` | build from `probe.pkl`, inspect, size; CLI |
| `tests/test_sensitivity_card.py` | 17 acceptance tests incl. the byte-identical gate |
