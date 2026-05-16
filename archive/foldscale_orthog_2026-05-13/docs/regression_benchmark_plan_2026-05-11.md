# Regression Benchmark Plan - 2026-05-11

Purpose: explain the 4B regressions from AWQ, HALO, and SmoothQuant on the
current production stack and define the benchmarks required before promoting
or deleting any of these opt-in levers.

## Observed Regressions

Baseline contract:

- Model: `/home/rob/.cache/huggingface/Qwen3-4B`
- Assignment: fixed `225 NVFP4`, `7 MXFP8`, `20 BF16`
- Stack: GPTQ + damp sweep + scale sweep + FourOverSix + PrismaClip
- Dataset: `/home/rob/dq-runs/calibration/diverse-v1.jsonl`
- KL validation: production-cache entries resident-prefetched, `232/232`
  entries, `6.50 GiB`, zero missing.

Results:

| Arm | KL | Relative |
|---|---:|---:|
| FourOverSix + PrismaClip | `0.056417932` | baseline |
| AWQ-v2 + full stack | `0.225788253` | `+300.2%` worse |
| HALO + full stack | `0.177460096` | `+214.5%` worse |
| SmoothQuant full-cache rerender | `0.167588803` | `+197.0%` worse |

The hot path was not CPU/NVMe-bound. The failure mode is numerical or
measurement-contract mismatch.

Follow-up isolation in
`/home/rob/dq-runs/qwen3-4b-smoothquant-targeted-20260511T113334Z` split the
SmoothQuant full-cache result:

- selected SmoothQuant layer 28 MXFP8 `q/k/v` only: `0.055455598`
  (`1.7%` better than baseline);
- three unrelated NVFP4 PrismaClip rerender files only: `0.163882268`;
- all six changed files, with or without SmoothQuant metadata:
  `0.167588803`.

So the SmoothQuant row above is a full-cache rerender regression, not a
conviction of the selected SmoothQuant fold itself.

## Working Hypotheses

### 1. Local output MSE is a poor gate for attention q/k/v

AWQ selected one input-layernorm reader group:

- AWQ: layer 29 `q_proj/k_proj/v_proj`, local rendered-MSE gain `3.50%`

That selected group is in the MXFP8 attention promotion region. A small
improvement in individual Q/K/V output MSE can still worsen attention logits:
Q and K errors are multiplied through the softmax score matrix, and V error is
weighted by the changed attention distribution. A per-Linear MSE gate does not
see that nonlinearity.

SmoothQuant's selected layer 28 `q/k/v` group did not regress in isolation;
it slightly improved KL. The full-cache SmoothQuant regression came from
unrelated PrismaClip NVFP4 rerender files whose local gains were below `0.2%`.

### 2. The fast AWQ/SmoothQuant search is not fully production-faithful

The selected scale is rendered once through the full stack, but the scale
search itself defaults to a cheaper objective:

- `PRISMAQUANT_AWQ_SEARCH_GPTQ=0`
- `PRISMAQUANT_AWQ_SEARCH_SCALE_SWEEP=0` in the latest AWQ run

That can choose a scale that wins in the cheap search and loses after GPTQ,
damp sweep, scale sweep, FourOverSix, and PrismaClip are recomputed.

For SmoothQuant, the immediate issue was not the selected scale. The isolated
scale was slightly positive. The broader lesson is that rerendering a full
cache can introduce unrelated PrismaClip differences unless those cache changes
are held fixed or explicitly attributed. A later 4B run showed that filtering
tiny local PrismaClip gains with a `0.002` floor removed collectively useful
clips, so nonzero floors are ablation knobs rather than production defaults.

### 3. The full stack may already remove the outlier problem

FourOverSix + PrismaClip improved 4B KL from `0.158064165` to `0.056417932`.
AWQ, SmoothQuant, and HALO all target outlier/coordinate conditioning. Once
PrismaClip and FourOverSix have already handled the block-scale bottleneck,
additional coordinate transforms can destroy useful structure without leaving
much residual outlier error to recover.

### 4. HALO may hurt NVFP4 block structure

HALO preserves the BF16 function, but it changes the basis before blockwise
NVFP4/MXFP8 rendering. Random rotations can spread a few structured outliers
across many 16-wide NVFP4 blocks, reducing the benefit of FourOverSix and
PrismaClip. HALO improved a weaker 4B tiny-slice recipe earlier, then regressed
against the stronger FourOverSix + PrismaClip stack, which supports an
interaction/saturation explanation rather than a blanket implementation bug.

### 5. Fixed assignments may be unfair to coordinate transforms

The current comparison reused the same fixed assignment. A transform can change
which Linears should be NVFP4, MXFP8, or BF16. PrismaQuant's production promise
requires an allocation-aware rerun before a final production decision.

## Benchmarks Before Promotion

Run these in order. Stop early if a method fails an earlier correctness gate.

### A. Exact-transform sanity

Goal: rule out fold or rotation implementation bugs.

- AWQ/SmoothQuant: apply the fold in BF16 with no quantization and compare
  logits to the original BF16 model.
- HALO: apply HALO in BF16 with no quantization and compare logits to the
  untied no-HALO BF16 model.
- Pass: max logit error stays within expected BF16 rounding tolerance and
  argmax agreement is effectively unchanged on the calibration slice.

### B. Single-group KL localization

Goal: prove whether the selected group itself causes the regression.

Use the baseline production cache, then override only:

- AWQ layer 29 `q_proj` alone, `k_proj` alone, `v_proj` alone, then all three.
- SmoothQuant layer 28 `q_proj` alone, `k_proj` alone, `v_proj` alone, then
  all three.
- HALO per-layer or layer-window cache overrides if available.

Measure end-KL with resident prefetch. If one q/k/v group accounts for most of
the regression, local MSE is the wrong gate for attention folds.

### C. Attention-specific diagnostics

Goal: bridge local MSE and end-KL.

For changed attention groups, measure on the same calibration batch:

- Q/K/V output MSE separately;
- attention-score MSE after RoPE;
- attention probability KL;
- attention-output MSE before `o_proj`;
- final end-KL.

Promotion requires the intermediate attention metrics to move in the same
direction as end-KL, or the local gate is not trustworthy.

### D. Full-render scale rerank

Goal: test whether the cheap AWQ/SmoothQuant search chose a bad scale.

For the top candidate groups only, evaluate every scale candidate through the
full production render:

- GPTQ;
- damp sweep;
- scale sweep;
- FourOverSix;
- PrismaClip when enabled.

Then run single-group KL for the top few fully rendered candidates. If the
best full-render candidate is identity or near-identity, leave the method off
for that group class.

### E. Component additivity ladder

Goal: identify destructive interactions.

For each method, fixed assignment, same calibration:

1. static NVFP4/MXFP8 only;
2. + GPTQ/damp sweep;
3. + scale sweep;
4. + FourOverSix;
5. + PrismaClip;
6. full stack.

This tells us whether the method is intrinsically bad or only conflicts with
a later lever that already solved the same error.

### F. Allocation-aware rerun

Goal: test the fair PrismaQuant production question.

For any method that survives A-E:

- rerun sensitivity/cost under that method;
- rerun allocator with the same bpp target;
- fill production cache with resident prefetch;
- validate KL against the same calibration contract.

The production decision is method A versus method A plus lever, with the
allocator re-derived for both arms.

### G. Held-out calibration check

Goal: distinguish signal from calibration overfit.

Search scales/clips on one split and validate KL on another split:

- `n=16`, `seqlen=1024` for fast triage;
- `n=64`, `seqlen=2048` for confirmation on 4B;
- 27B only after the 4B result is non-regressive.

## Default Policy

Until these benchmarks pass, keep AWQ, SmoothQuant, HALO, Fisher-weighted GPTQ,
and MXFP8 exponent shifts opt-in and off by default. The current production
candidate remains:

```text
GPTQ + damp sweep + scale sweep + FourOverSix + PrismaClip
```
