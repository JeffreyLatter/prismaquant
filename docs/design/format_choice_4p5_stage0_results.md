# Stage 0 results — free accuracy screen, native NVFP4 (4.5) vs FP8_CB_K36 (4.5)

**Status: screen. STOP-ONLY per `format_choice_4p5.md` §5 Stage 0 — this stage
can halt the programme and cannot promote anything.** Nothing here is a served
result, a KL result, or a default change. Run 2026-07-30 on branch
`claude/docs-consolidation` (HEAD moved `e6fa967` → `e241da4` during the run;
see "Provenance" below).

**Verdict up front: no STOP on either model.** K36 holds the act-weighted
majority on both the 27B and the 4B, under both codebook-fit conventions.
The programme is not halted. It is also not advanced — Stage 1 and Stage 2
remain exactly as specified.

---

## 1. What was run

`scripts/ab_nvfp4_vs_k36_dense.py` prices both formats through one code path
(`format_registry` RTN qdq → fp32 weight-MSE) at their exact 4.5 rungs. It was
Hy3-only (commit `c551e24`); this run generalised it:

- `--work-dir` / `--model-path` aliases; `--exclude`, `--col-weights`, `--out`,
  `--mem-fraction`, `--fmt-a/--fmt-b`, `--cb-imatrix-fit` added.
- Target inventory falls back to the **probe's own Linear inventory** when the
  work dir has no `exported_nvfp4_cb/quant_config.json` (neither the 27B nor the
  4B dir has one).
- **Namespace-robust source lookup** — the probe names Linears
  `model.layers.N.…`, the 27B checkpoint stores them under
  `model.language_model.layers.N.…`. Wrapper segments are collapsed; 496/496 and
  252/252 targets resolved, zero misses.
- **`h_trace` extraction fixed.** The original read `probe.get("h_trace", probe)`,
  but production probes store it at `probe['stats'][name]['h_trace']` — so the
  lookup always fell through to the `1.0` default. **Every `h_trace` value in the
  shipped Hy3 CSV (`prod-hy3-nvfp4cb-2p9/ab_nvfp4_vs_k36.csv`) is literally
  `1.0`**; the column was never populated. It is populated now.
- Act-column weighting is optional (a work dir without `cb_col_weights.pkl`
  still reports the unweighted and h_trace-weighted columns).
- Per-unit outlier diagnostics added to the CSV: `out_features`, `in_features`,
  `rowmax_over_rms`, `excess_kurtosis`, `maxabs_over_rms`, plus a `leaf` column.

Hy3 itself **cannot be re-run**: `/home/rob/dq-runs/hy3-prod/source` no longer
exists. Hy3 figures below are recomputed from its shipped CSV.

### Work dirs and provenance

| model | work dir | probe calib | act-column weights |
|---|---|---|---|
| Qwen3.6-27B (dense, 496 body Linears) | `/home/rob/dq-runs/prod-27b-nvfp4cb-5p5` | n=8 × 1024, `diverse-v1.jsonl` | `artifacts/cb_col_weights.pkl` (same run) |
| Qwen3-4B (252 Linears) | `/home/rob/dq-runs/phase-f-gptq-4b` | n=32 × 1024, `diverse-v1.jsonl` | **borrowed**: `nvfp4-cb-phase0/serve/fp8cb_k44_4b/col_weights_seed0.pkl` |

**No Qwen3-4B work dir on this box has an act cache or a `cb_col_weights.pkl`** —
`find` over `/home/rob/dq-runs` returns six such files (27B ×2, 35B, Hy3 ×2,
0.8B) and none for a 4B. Rather than build a probe (a production run needing the
box), the 4B act column uses the imatrix column-weight vector from the
2026-07-16 `fp8cb_k44_4b` CB build — same architecture (Qwen3-4B, hidden 2560,
36 layers, 252 keys, exact target match), **different calibration draw** from the
`h_trace` source. Treat 4B act-weighted numbers as cross-sourced.

Excluded from both inventories: `lm_head`, MTP sidecar, experts (neither model
has any) — the standard bpp-accounting convention (core principle 12).

Sources: `model.layers.*` weights read verbatim from the BF16 checkpoints
(`qwen36-27b-bf16`, `Qwen3-4B`). GPU-first, per-Linear transients only, peak
6.4 GB, `--mem-fraction` 0.20–0.25. The idle serve on :8000 was never touched.

---

## 2. The convention that decides the number

The screen has a free parameter nobody named before: **is the CB codebook fitted
with the imatrix or without it?** `FormatSpec.quantize_dequantize` for the CB
family is `f(w, col_weights=None)`, and the original screen called it with one
argument — an **unweighted** fit. Production does not: `harvest_cb_col_weights`
→ `export_nvfp4_cb --col-weights` / `build_production_cache --col-weights` means
**every shipped CB tensor is imatrix-fitted**.

So "unweighted fit, scored act-weighted" measures a render that never ships,
against an objective the render never optimised. Both diagonal cells below are
self-consistent; both off-diagonal cells are objective-mismatched.

### Qwen3.6-27B, 496 units — K36 win-rate / geomean(K36÷NVFP4) / Σ h·mse ratio

| CB codebook fit | scored **unweighted** | scored **act-column-weighted** |
|---|---|---|
| **unweighted** (as the Hy3 screen ran) | **93.8%** · 0.639 · **0.587** ✓matched | 62.9% · 0.961 · 1.395 ✗mismatched |
| **imatrix** (as production renders) | 79.8% · 0.766 · 0.865 ✗mismatched | **99.4%** · 0.391 · **0.405** ✓matched |

### Qwen3-4B, 252 units

| CB codebook fit | scored **unweighted** | scored **act-column-weighted** |
|---|---|---|
| **unweighted** | **100.0%** · 0.598 · **0.610** ✓matched | 89.3% · 0.813 · 4.466 ✗mismatched |
| **imatrix** | 94.4% · 0.709 · 0.938 ✗mismatched | **100.0%** · 0.521 · **0.472** ✓matched |

Each fit wins on its own objective and gives ground on the other — the expected
signature of a real fit, not of a bug. The **matched** cells are the ones that
describe a renderable artifact.

**A caution the matched cells do not remove:** the h_trace-weighted *aggregate*
(Σ h·mse, the quantity nearest the allocator's additive Fisher objective)
**inverts in both mismatched cells** — 1.395 on the 27B and 4.466 on the 4B —
and in both cases the inversion is carried by a handful of high-`h_trace` units
(27B: `layers.2.mlp.down_proj` alone is 33% of the aggregate deficit; 4B:
`layers.6.mlp.down_proj` alone is 274%). Unit-count majorities are robust to
this; Fisher-weighted aggregates are not. Any future screen that reports one
number should report the aggregate alongside the count.

---

## 3. Per-model results, matched conventions

Columns as §5 asks: unweighted, h_trace-weighted, act-weighted.

### Qwen3.6-27B — 496 units

| scoring | K36 wins | geomean K36÷NVFP4 | h_trace-weighted win | Σ h·mse K36÷NVFP4 |
|---|---|---|---|---|
| unweighted (unweighted fit) | 465 / 496 = **93.8%** | 0.639 | 99.9% | 0.587 |
| act-column-weighted (imatrix fit) | 493 / 496 = **99.4%** | 0.391 | 100.0% | 0.405 |

Per-unit ratio spread (act-weighted, imatrix fit): min 0.073 · p10 0.217 ·
median 0.451 · p90 0.540 · max 1.282.

### Qwen3-4B — 252 units

| scoring | K36 wins | geomean K36÷NVFP4 | h_trace-weighted win | Σ h·mse K36÷NVFP4 |
|---|---|---|---|---|
| unweighted (unweighted fit) | 252 / 252 = **100.0%** | 0.598 | 100.0% | 0.610 |
| act-column-weighted (imatrix fit) | 252 / 252 = **100.0%** | 0.521 | 100.0% | 0.472 |

Per-unit ratio spread (act-weighted, imatrix fit): min 0.094 · p10 0.438 ·
median 0.538 · p90 0.637 · max 0.883.

### Hy3-295B (recomputed from the shipped CSV, for reconciliation only)

503 units, unweighted fit throughout. Unweighted: 503/503 = 100%, geomean 0.590.
Act-weighted: **461/503 = 91.7% overall**; the recorded "87%" in
`format-speed-policy.md:9-14` is the `dense/attn` role alone (271/313 = 86.6%) —
the `shared` role is 190/190. Not a discrepancy, a scope label. Its `h_trace`
column is all `1.0` (the bug in §1), so no Hy3 h_trace-weighted number exists.

### Byte honesty

NVFP4's 4.5 bpw includes its scale plane exactly. FP8_CB_K36's 4.5 is index
stream only, plus per-output-channel fp32 scales at `32/in_features`: **+0.0018
to +0.0125 bpw** across these shapes (27B `in=5120` → +0.00625; 4B `in=2560` →
+0.0125; `down_proj` `in=17408` → +0.0018). CB is 0.04–0.28% richer at nominal
match. Small, one-directional, in CB's favour; noted so the margin is not read
as byte-exact.

---

## 4. Outlier-unit analysis — which units favour NVFP4, and what they share

### 27B, matched act-weighted (imatrix fit): **3 of 496**

All three are `linear_attn.in_proj_a` (layers 57, 58, 61). Shared properties vs
the 493 K36-favouring units:

| | NVFP4-favouring (n=3) | K36-favouring (n=493) |
|---|---|---|
| median `out_features` | **48** | 5120 |
| median excess kurtosis | **9.91** | 0.58 |
| median `rowmax/rms` | **12.54** | ~4.3 |
| median `h_trace` | **0.68** | 834 |
| worst ratio | 1.28× | — |
| share of Σ h·mse aggregate | **0.0%** | — |

The common property is **narrow, extremely heavy-tailed DeltaNet gate
projections**: 48 output rows, excess kurtosis ~10×, per-row max/rms ~12.
NVFP4's per-group-16 scale plane re-normalises locally around each outlier; a
product-VQ codebook fitted over the whole tensor cannot. Their Fisher weight is
~0.68 against a model median of 834, so they are irrelevant to the aggregate —
and under a joint menu the allocator would pick NVFP4 there on measured cost
alone, which is the argument for the joint menu, not for a carve-out.

### 27B, matched unweighted: **31 of 496**

Same family, wider: `linear_attn.in_proj_b` ×17, `in_proj_a` ×14 — again median
`out_features` 48, median kurtosis 5.70, median `h_trace` 10.5, worst ratio 1.2×,
0.0% aggregate share. **Zero MLP or self-attention units** appear.

### 4B: **0 of 252** under either matched convention

Qwen3-4B has no DeltaNet layers, hence no narrow high-kurtosis projections, hence
no NVFP4-favouring units at all.

### Reconciling the recorded "42 outlier-row units favor NVFP4" (Hy3)

Recomputed: exactly 42, and they are **`self_attn.k_proj` ×35, `q_proj` ×4,
`down_proj` ×1, `gate_proj` ×1, `v_proj` ×1**. That set comes from the
**mismatched** cell (unweighted fit scored act-weighted). On the 27B the same
mismatched cell produces 184 NVFP4-favouring units, which collapses to 3 once
the codebook is fitted the way production renders it. **`[UNVERIFIABLE — the Hy3
source is deleted]`** whether Hy3's 42 would collapse likewise; the mechanism is
identical and the direction is the same on both models measured here, so the "42
outlier-row units" figure should be treated as **fit-convention-dependent, not a
structural property of those Linears**, until someone re-runs it. This does not
change any served result and no doc has been rewritten on the strength of it.

---

## 5. Verdict against §5 Stage 0

> *"if K36 loses the act-weighted majority on either model, re-plan before
> spending a serve window; if it wins, it is a screen and nothing more."*

| model | act-weighted majority, matched (imatrix fit) | act-weighted majority, as-recorded (unweighted fit) | STOP? |
|---|---|---|---|
| Qwen3.6-27B | 99.4% (493/496) | 62.9% (312/496) | **no** |
| Qwen3-4B | 100.0% (252/252) | 89.3% (225/252) | **no** |

**No STOP on either model, under either convention.** The stop condition is not
met and the programme is not halted.

Nothing is promoted. This is a cost-model weight-error screen — the lowest rung
of the metric authority. It says nothing about served KL, PPL, ToolEvalBench,
prefill tax, or the K1–K4 criteria, all of which remain open exactly as
`format_choice_4p5.md` §4–§5 states. Three structural limits from §2.1 still
apply unchanged: **weight error only** (the A8-vs-A4 activation difference is
outside every number here), **RTN-vs-RTN** (production runs GPTQ+JSO on the
native arm), and **cost-model, not KL, not served**.

One thing to carry into Stage 1/2: the mismatched-cell aggregate inversions in
§2 mean a per-Linear cost run on the CB lane must render with the imatrix to be
comparable at all. That is core principle 8 (one render, cost == KL == shipped
bytes), and it is already how the production CB path behaves; the screen was the
odd one out.

---

## 6. Reproduce

```bash
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
# 27B — as-recorded convention, then production-faithful
$PY scripts/ab_nvfp4_vs_k36_dense.py \
  --work-dir /home/rob/dq-runs/prod-27b-nvfp4cb-5p5 \
  --model-path /home/rob/.cache/huggingface/qwen36-27b-bf16 \
  --exclude '.experts.,lm_head,mtp.' --mem-fraction 0.25 \
  --out /home/rob/dq-runs/prod-27b-nvfp4cb-5p5/ab_nvfp4_vs_k36_stage0.csv
$PY scripts/ab_nvfp4_vs_k36_dense.py ... --cb-imatrix-fit \
  --out .../ab_nvfp4_vs_k36_stage0_imatrixfit.csv

# 4B — act column weights borrowed from the phase-0 CB build (see §1)
CW=/home/rob/dq-runs/nvfp4-cb-phase0/serve/fp8cb_k44_4b/col_weights_seed0.pkl
$PY scripts/ab_nvfp4_vs_k36_dense.py \
  --work-dir /home/rob/dq-runs/phase-f-gptq-4b \
  --model-path /home/rob/.cache/huggingface/Qwen3-4B \
  --exclude '.experts.,lm_head,mtp.' --mem-fraction 0.20 --col-weights $CW \
  [--cb-imatrix-fit] --out /home/rob/dq-runs/phase-f-gptq-4b/ab_4b_act.csv
```

Wall: 27B 14 min (unweighted fit) / 62 min (imatrix fit); 4B ~3 / ~9 min.
Determinism check: the NVFP4 arm is **bit-identical across the two 27B runs**
(496/496 rows), so the imatrix flag moves only the CB arm.

Per-unit CSVs (13 columns + `leaf`):

- `/home/rob/dq-runs/prod-27b-nvfp4cb-5p5/ab_nvfp4_vs_k36_stage0.csv`
- `/home/rob/dq-runs/prod-27b-nvfp4cb-5p5/ab_nvfp4_vs_k36_stage0_imatrixfit.csv`
- `/home/rob/dq-runs/phase-f-gptq-4b/ab_nvfp4_vs_k36_stage0.csv` (no act column)
- `/home/rob/dq-runs/phase-f-gptq-4b/ab_4b_act.csv`
- `/home/rob/dq-runs/phase-f-gptq-4b/ab_4b_act_imatrixfit.csv`

### Provenance note

The run started at `e6fa967`; a **concurrent session on the same branch** moved
HEAD to `e241da4` mid-run and swept the modified
`scripts/ab_nvfp4_vs_k36_dense.py` into commit `3664a56` ("Strix Halo HIP CB
kernels…"). That commit was not made by this work and nothing here was committed
deliberately. The tool changes are therefore already on the branch; this results
file and the §5 status line are the only uncommitted additions.
