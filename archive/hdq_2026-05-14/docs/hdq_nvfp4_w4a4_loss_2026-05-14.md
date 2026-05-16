# HDQ + NVFP4 W4A4: Loss-Mismatch Diagnosis (2026-05-14)

## Trigger

User pushed back on the "NVFP4 rejects rotation" conclusion citing MXFP4 +
DuQuant success in the literature. Asked to consult Codex and find the
error in the analysis rather than resigning to failure.

## The bug

The HDQ joint-search solver optimizes a **weight-only** STE quantization
loss:

```
loss_w_only = || x @ (W - Q_w(W M^T) M)^T ||^2
```

with `x` being raw BF16 activations from forward hooks.

But the runtime for NVFP4 in PrismaQuant is **W4A4** (per
`export_native_compressed.py:4501`):

```python
NVFP4_SCHEME = {
    "weights":           {"num_bits": 4, "type": "float", "group_size": 16, ...},
    "input_activations": {"num_bits": 4, "type": "float", "group_size": 16,
                          "dynamic": "local", ...},
}
```

So at serve time both `W` and `X` are quantized. The runtime output is:

```
y_runtime = Q_a(x M^T) @ Q_w(W M^T)^T
```

The solver's W-only loss never sees `Q_a` and so cannot credit a
rotation whose primary benefit is reshaping per-G-block activation
distributions. DuQuant's mechanism (activation outlier spreading)
operates on the activation side; we were rejecting rotations on the
weight side without ever measuring what they did for activations.

## Evidence: W4A4 geodesic sweep

`tools/w4a4_geodesic_sweep.py` forward-only evaluates three loss
flavors (W-only, A-only, W4A4) at each `t in {0, 0.1, ..., 1.0}` along
`R(t) = orthogonalize((1-t) I + t R_init)` for three inits
(sylvester, svd_v, random) across 9 sampled clusters (3 per kind).
Calibration: 16 × 2048 = 32k tokens.

Result (Δw4a4% at best t vs t=0, sylvester init):

| cluster                              | t* | Δw_only% | Δa_only% | Δw4a4%  |
|--------------------------------------|----|----------|----------|---------|
| model.layers.0.mlp.residual          | 0.10 | −0.07    | **−1.89** | −1.00   |
| model.layers.1.mlp.residual          | 0.10 | −0.57    | −0.65    | −0.56   |
| model.layers.2.mlp.residual          | 0.00 | 0        | 0        | 0       |
| model.layers.0.mlp.down              | 0.00 | 0        | 0        | 0       |
| model.layers.1.mlp.down              | 0.00 | 0        | 0        | 0       |
| model.layers.2.mlp.down              | 0.00 | 0        | 0        | 0       |
| model.layers.3.attn.v_o              | 0.00 | 0        | 0        | 0       |
| model.layers.7.attn.v_o              | 0.20 | **−2.71** | +0.96   | **−1.13** |
| model.layers.11.attn.v_o             | 0.00 | 0        | 0        | 0       |

Key observations:

- **The signal is real but small.** Best cluster gets −1.13% W4A4 at
  Sylvester @ t=0.2. Most clusters have W4A4 minimum at t=0 — no rotation
  beats identity for them.
- **down_proj kind: zero benefit anywhere.** 3/3 clusters flat. SwiGLU
  outputs are already well-distributed.
- **residual kind: activation-driven gains.** layer 0 / layer 1 residual
  clusters benefit from Sylvester @ t=0.1 primarily via activation
  quantization improvement (A-only −1.89%, −0.65%), with W-only nearly
  flat.
- **v_o kind: mixed mechanism.** layer 7 v_o gets the largest W4A4 gain
  (−1.13%) — but here it's weight-driven (W-only −2.71%), activations
  slightly worse (+0.96%).

## Confirms Codex's diagnosis

Codex (consulted via `codex exec`) ranked likely causes:

1. **Objective mismatch (W-only loss for W4A4 deployment): definitely real**
2. NVFP4 G=16 rotation inversion: plausible, consistent with MR-GPTQ
   (ICLR 2026) which reports rotations can hurt NVFP4 with RTN
3. Bad basin from non-identity init: plausible (`R = exp(A_skew) @
   init_R` makes identity reachable but far from init's tangent point)
4. Tiny calibration set (1k tokens vs published 256k+)

The sweep confirms (1) is contributory but bounded — even with the
correct objective, the gain ceiling is roughly 1% W4A4 per cluster on
this model. Most clusters (5/9 sampled) gain nothing from rotation. The
"NVFP4 G=16 doesn't want rotation" hypothesis (2) holds for the
majority of clusters; the loss-mismatch (1) is the explanation for the
minority where rotation should but didn't appear to help.

## Fix

Added `_compute_cluster_loss_w4a4` and a `loss_kind` knob threaded
through:

- `prismaquant/hadamard_duquant.py:_compute_cluster_loss_w4a4`
- `prismaquant/hadamard_duquant.py:solve_cluster_rotation(loss_kind=...)`
- `prismaquant/joint_hadamard_format_search.py:run_joint_search(solver_loss=...)`
- `prismaquant/run_joint_hadamard_search.py --solver-loss {w_only,w4a4}`
- `prismaquant/run-pipeline.sh: HADAMARD_DUQUANT_SOLVER_LOSS env var`

W-only path is preserved as the default; opt in to W4A4 via env or CLI.
The joint loss costs ~2× compute per iter (extra `(N, out)` matmul to
compute `y_ref = x @ W^T`) but is the right objective for any
microscale-FP4-A4 deployment.

## What's still expected

Even with the joint loss, the upper bound per the geodesic sweep is
~1% W4A4 improvement on the ~40% of clusters where rotation helps at
all. This is far from DuQuant's published 5-15% wins, because:

1. **NVFP4 G=16 has per-block outlier isolation built in.** Each 16-element
   block already gets its own FP8 scale, so cross-channel outlier
   spread (DuQuant's mechanism) has less to gain.
2. **PrismaQuant runs Qwen3.5-0.8B**, much smaller than the 7B+ models
   DuQuant typically reports on. Outlier severity scales with model size.
3. **Calibration is tiny** (4 × 256 = 1k tokens vs published 256k+).
4. **MR-GPTQ (ICLR 2026)** explicitly reports rotations hurt NVFP4 with
   RTN; we're in the regime that paper warns about.

## Superseded theoretical expectation (Fisher-weighted KL prediction)

This section was computed before the corrected rotated-reference scorer.
It remains useful as historical context for why the work continued, but
the corrected 0.8B and 4B HDQ-only totals below supersede these exact
numbers.

Computed from the W4A4 sidecar Σ_cluster (no_rot vs rot fisher_mse),
treating cluster mass as the natural weight (the per-cluster loss is
already Fisher-weighted via row_weights, so Σ across clusters is the
aggregate Fisher-weighted output MSE):

| config                                    | Σ baseline fisher_mse | Σ with rotation | predicted Δ |
|-------------------------------------------|-----------------------|-----------------|-------------|
| 0.8B identity-init W4A4                   | 1.367e-01             | 1.347e-01       | **−1.45%**  |
| 0.8B multi-init (identity+sylvester_t0p3+t0p5+svd_v) | 1.367e-01 | 1.349e-01    | −1.30% (worse) |
| **4B identity-init W4A4**                 | 3.878e+00             | 3.798e+00       | **−2.06%**  |

By kind on 4B (the production target):

| kind      | Σ baseline | Σ with rot | Δ      | share of total error |
|-----------|------------|------------|--------|----------------------|
| down_proj | 4.82e−01   | 4.79e−01   | −0.56% | 12.4%                |
| residual  | 3.28e+00   | 3.21e+00   | **−2.28%** | **84.7%**         |
| v_o       | 1.13e−01   | 1.10e−01   | −2.07% | 2.9%                 |

Residual stream dominates the baseline error AND sees the biggest
proportional reduction — DuQuant's "residual outliers are the high-
leverage target" claim holds.

Mapping to model output KL: Fisher-weighted MSE ∝ KL to first order, so
**expected NVFP4-quantization KL reduction ≈ 2%** on 4B. If the NVFP4
baseline introduces 0.05 PPL over BF16, HDQ shaves ~0.001 PPL. Small
but real, concentrated on residual-stream Linears.

## Path C exploration — outcomes

User asked whether scaled-Sylvester guessing could be replaced with
data-inferred optima. Implemented and tested four alternatives:

1. **Multi-init basin search** (try {identity, sylvester_t0p3, sylvester_t0p5,
   svd_v} per cluster, pick W4A4 winner): aggregate Fisher-MSE on 0.8B
   = −1.30% vs identity-only −1.45%. **Worse by 0.15 percentage
   points.** The "winner" by in-solver STE loss doesn't always match
   the production-renderer score; multi-init's selection rule itself
   leaks a small amount of quality.
2. **Weight decay on A_skew**: implemented as opt-in
   (`solver_weight_decay`). Identity-init Adam produces median
   ||M−I||=0.21 on 4B (already small rotations); the data doesn't
   show overfit symptoms that warrant lessening. Kept as a knob.
3. **Givens-balance init** (constructive paired-Givens rotation
   computed from per-cluster GtG, EXPLICIT cost-reduction test per
   pair, no heuristic threshold): on outlier-heavy synthetic +
   balanced synthetic, Adam-from-identity outperforms Givens-balance
   Adam (+2.86% / +1.91% gain vs −0.00% / +2.15%). The constructive
   init pre-commits to a rotation structure that Adam can't refine
   away from cleanly.
4. **Channel permutation (cross-G)**: not implemented. Would need
   either a Permutation transform spec in compressed-tensors or a
   forward_pre_hook in vLLM serve path. Given the empirical ceiling
   (~2% Fisher-MSE on 4B), the kernel/runtime work isn't justified.

**Verdict**: Adam-from-identity under W4A4 STE loss IS the explicit
data-driven optimization. Every gradient step is data-derived; the
solver finds the local optimum of the actual W4A4 loss. Multi-init,
constructive inits, and Sylvester scaling were all things worth trying
empirically, but none beats the principled gradient-driven approach.
The "explicit" answer to "infer optimal R from data" is: the gradient
*is* the inference, and Adam follows it.

## Sylvester init under W4A4 loss — still regresses

Confirmed: the loss-mismatch and the bad-init-basin are independent
issues. Re-ran the 0.8B joint search with `--solver-init sylvester
--solver-loss w4a4` (artifact: `/dq-runs/qwen35-0p8b-hdq-syl-w4a4-20260514T064302Z`).

| init under W4A4 loss | down_proj median | residual median | v_o median       | rot beats no-rot |
|----------------------|------------------|------------------|------------------|-------------------|
| **identity**         | **−0.66%**       | **−1.33%**       | **−1.20%**       | **50/60**         |
| sylvester            | +0.67%           | +3.01%           | **+22.50%** (worst +76%) | 11/60     |

Conclusion: Sylvester init + STE-bounded Adam can't navigate back to the
better basin in 500 iters regardless of the loss kind. The W4A4 fix is
necessary but Adam from identity-init is *also* necessary. Combined they
give the win; either alone is insufficient.

Production-ready configuration: `--solver-init identity --solver-loss
w4a4` (the current defaults).

## Superseded 4B exploratory joint search

The first Qwen3-4B joint search at
`/dq-runs/qwen3-4b-hdq-w4a4-20260514T055532Z` was useful for scale
diagnosis, but its allocator/sidecar scores used the pre-fix rotated
reference path and the artifact also included non-HDQ production levers.
Retain it only as historical context. The corrected HDQ-only 4B run below
supersedes its rotation counts and loss totals.

The W4A4 loss fix is the correct, mandatory change for any
microscale-FP4 W4A4 deployment in PrismaQuant.

## Scale dependence — Qwen3-4B sweep changes the picture

Re-ran the geodesic sweep on Qwen3-4B (`/hfcache/Qwen3-4B`, model_type=qwen3,
36 layers, 2560 hidden, same E2M1 microscale assumption). Result at
`/dq-runs/qwen3-4b-hdq-sweep-20260514T055243Z/sweep.jsonl`:

| kind      | clusters | best Δw_only% | best Δa_only% | best Δw4a4%     |
|-----------|----------|---------------|---------------|-----------------|
| down_proj | 3        | **−18.2%**    | +34.2%        | **−7.89%** (sylvester t=1.0) |
| residual  | 3        | **−14.7%**    | −1.1%         | **−8.80%** (svd_v t=0.7) |
| v_o       | 3        | −0.8%         | −0.5%         | −0.80% (sylvester t=0.1) |

vs Qwen3.5-0.8B where best W4A4 reduction across all sampled clusters was
**−1.13%**. The peak gain is roughly **7×** larger on 4B.

Critical: for `down_proj.layer.2`, **the optimal rotation is full Sylvester
at t=1.0** — the rotation that the original Sylvester smoke saw as a
catastrophic regression under W-only loss (W-only Δ = +18.5% there). Now
that the loss correctly includes activation quantization, this same
rotation reduces W4A4 by 7.89%. The W-only loss simply could not see this
trade-off (W gets much better at the cost of A getting worse, with net
positive).

This is a clean two-step refutation of "NVFP4 G=16 rejects rotation":

1. **The 0.8B "rotation regresses" finding was a loss-measurement artifact** —
   the solver was optimizing W-only, but the runtime is W4A4. Fixing the
   loss makes more clusters rotation-eligible (50/60 vs the W-only's
   60/60-but-actually-some-regress).
2. **The 0.8B "small ceiling" finding was a model-size artifact** —
   outliers worth rotating away require enough channels and parameters
   to develop. At 4B scale, full Sylvester rotations recover DuQuant's
   original mechanism (W-only −18%) and the W4A4 metric correctly
   prefers them.

PrismaQuant's mainline targets are 4B / 27B / 27B-A3B, all firmly in the
"big enough for rotations" regime. The W4A4 fix is necessary and the
expected gain at production scale is **5-10× larger** than the 0.8B smoke
suggested.

## End-to-end smoke (W4A4-loss, identity init, Qwen3.5-0.8B)

Run: `/dq-runs/qwen35-0p8b-hdq-w4a4-20260514T054331Z` —
`HADAMARD_DUQUANT_SOLVER_LOSS=w4a4`, identity init, NSAMPLES=4 SEQLEN=256.

Per-cluster gain comparison (NOT directly comparable across loss kinds —
each delta is in its own loss units; what's comparable is the
rot-beats-norot count + sign distribution):

| metric            | W-only solver (matexp-nondet) | W4A4 solver (this run) |
|-------------------|------------------------------|------------------------|
| down_proj median  | −2.58% (W-only)              | −0.66% (W4A4)          |
| residual median   | −1.50%                       | −1.33%                 |
| v_o median        | −4.77%                       | −1.20%                 |
| rot beats no-rot  | **60/60**                    | **50/60**              |
| down_proj max     | −0.58% (always wins)         | **+0.00** (7 clusters flat) |
| residual max      | −0.16%                       | **+0.20** (2 clusters lose) |
| allocator picks   | (n/a from this run)          | **50/60 clusters chose rotation** |

Interpretation: the W4A4 solver gives a **more honest** picture. The 10
clusters it correctly identifies as not benefitting from rotation are
exactly the ones the allocator declines under the new metric — picks
align with cost cells (50/60 in both). Under W-only the previous
"60/60 universal benefit" was an artifact of optimizing the wrong
objective.

Direction of gains stays the same (rotation reduces error on most
clusters), but the magnitude is smaller. This matches the geodesic
sweep prediction of ~1% W4A4 ceiling per cluster. Still a real,
principled improvement: rotations now ship only where they net out
positive under the production objective.

## Corrected learned dense online scorer (2026-05-14)

Follow-up run:
`/dq-runs/qwen35-0p8b-hdq-learned-fresh-20260514T171018Z`.

Bug found during quantification: the rotated W4A4 sidecar scorer was
using rotated activations for the reference path as well as the runtime
path. That compared `xM^T @ W^T` against `Q(xM^T) @ Q(WM^T)^T`, which
incorrectly penalized rotations. The solver itself already used the
correct original reference `x @ W^T`; the sidecar/allocator score was
wrong. The scorer now keeps original activations for the reference and
rotated activations only for the runtime quantized path. Regression:
`tests/test_joint_hadamard_format_search.py::test_rotated_w4a4_score_uses_original_reference_activations`.

Corrected identity-init learned run (`artifacts_fixed_score`) on
Qwen3.5-0.8B, HDQ-only, `NVFP4,BF16`, target 4.5 bpp:

| metric | result |
|---|---:|
| discovered clusters | 60 |
| allocator-picked rotations | 58/60 |
| affected consumer Linears | 94 |
| online dense transform groups | 53 |
| folded/offline groups | 5 |
| runtime transform tensors | 89 |
| total cluster W4A4 score, no-rot | 0.271071 |
| total cluster W4A4 score, selected | 0.267544 |
| selected-score reduction | 1.30% |

Activation dynamic-range effect over the 94 selected consumer activation
tensors: median absolute max reduction 1.03%; median per-NVFP4-group p99
max reduction 2.73%; mean per-group p99 max reduction 3.78%. The largest
per-group p99 reductions were 10-17% on several MLP down/gate/up inputs.

Multi-start diagnostic (`artifacts_multistart_n4`) tried identity,
Sylvester, and four random orthogonal starts per cluster, then selected
by the same corrected W4A4 sidecar score. It did **not** improve this
0.8B smoke: all 60 winning solves still reported identity init, allocator
picked 54/60 rotations, and the target-4.5 predicted loss was slightly
worse than `artifacts_fixed_score`. For this calibration, random/Sylvester
basins do not justify their ~6x search cost.

Corrected HDQ-only export:
`/dq-runs/qwen35-0p8b-hdq-learned-fresh-20260514T171018Z/exported_fixed_score`.
Export log confirms GPTQ/scale-sweep/block-output-match are off and the
artifact contains 53 `random-matrix` transform groups plus 89 dense
runtime matrix tensors:
`/dq-runs/qwen35-0p8b-hdq-learned-fresh-20260514T171018Z/logs/export_fixed_score.log`.

## Corrected HDQ-only learned dense run on Qwen3-4B

Corrected Qwen3-4B run reused the existing probe/cost/activation capture
from `/dq-runs/qwen3-4b-hdq-w4a4-20260514T055532Z` and reran only HDQ:
`NVFP4,BF16`, W4A4 solver/score, identity init, learned dense online
rotations, no random/Sylvester multi-starts, no GPTQ/scale-sweep/four-over-six
production levers. Logs:

- Search:
  `/dq-runs/qwen3-4b-hdq-w4a4-20260514T055532Z/logs/hadamard_duquant_search_fixed_score_hdqonly.log`
- Allocator:
  `/dq-runs/qwen3-4b-hdq-w4a4-20260514T055532Z/logs/allocator_fixed_score_hdqonly.log`
- Corrected artifacts:
  `/dq-runs/qwen3-4b-hdq-w4a4-20260514T055532Z/artifacts_fixed_score_hdqonly`

Allocator result at target 4.5 bpp: all 144 fused clusters stay NVFP4
and 121/144 choose rotation. That covers 224 consumer Linears and 31
producer Linears, with 90 online dense transform groups and 31 folded
groups.

| metric | result |
|---|---:|
| discovered clusters | 144 |
| allocator-picked rotations | 121/144 |
| affected consumer Linears | 224 |
| affected producer Linears | 31 |
| online dense transform groups | 90 |
| folded/offline groups | 31 |
| total cluster W4A4 score, no-rot | 7.541589 |
| total cluster W4A4 score, selected | 7.370901 |
| selected-score reduction | 2.26% |
| picked-cluster score reduction | 2.44% |

By insertion kind:

| kind | picked clusters | affected consumer Linears | selected-score reduction |
|---|---:|---:|---:|
| down_proj | 21/36 | 21 | 1.42% |
| residual | 69/72 | 172 | 2.51% |
| v_o | 31/36 | 31 | 1.75% |

Activation dynamic-range effect over all 224 selected consumer activation
tensors:

| activation metric | mean reduction | median reduction | p10 | p90 |
|---|---:|---:|---:|---:|
| absolute max | 4.92% | 2.66% | -0.01% | 12.96% |
| scalar p99.9 | 2.02% | 0.89% | -0.15% | 6.63% |
| scalar p99 | -4.64% | -1.63% | -9.62% | 0.18% |
| per-G16 max p99.9 | 5.76% | 2.93% | -0.01% | 16.11% |
| per-G16 max p99 | 3.83% | 2.80% | -0.04% | 10.05% |
| per-G16 max mean | 1.25% | 0.71% | 0.01% | 2.64% |

The group-level numbers matter most for NVFP4 because each 16-value group
gets the shared microscale. The learned rotations reduce per-G16 high
tails on the median selected tensor and have almost no severe group-tail
regressions (worst `g_p99` regression was -0.42%).

By kind, the dynamic-range benefit is overwhelmingly residual-stream:

| kind | tensors | median per-G16 p99 reduction | median per-G16 mean reduction | median max reduction |
|---|---:|---:|---:|---:|
| down_proj | 21 | 0.03% | 0.00% | 0.01% |
| residual | 172 | 3.68% | 1.05% | 4.20% |
| v_o | 31 | 0.21% | 0.16% | 0.11% |

Conclusion: learned HDQ is doing real outlier-spreading work on 4B,
especially on residual fan-out clusters. It is not a replacement proof for
PrismaClip yet: scalar p99 often increases because rotation spreads mass,
while the per-G16 high tail that controls NVFP4 scale quality improves.
The deployment-relevant next gate is a corrected cache/export plus vLLM
KL/PPL or prompt smoke against the no-HDQ NVFP4 baseline using native
`hadamard` online transforms. Dense `random-matrix` online transforms are
research-only because vLLM serves them through a generic dense wrapper.

## Post-audit validation and fixes (2026-05-14)

The served-model gate did **not** validate the learned dense online HDQ
artifact. The internal W4A4 proxy improved, but the vLLM artifact regressed
against both BF16 and vanilla NVFP4:

| artifact | metric | result |
|---|---|---:|
| BF16 Qwen3-4B | WikiText-2 PPL, 4096 tokens, seqlen 512 | 20.3326 |
| vanilla NVFP4 | WikiText-2 PPL, 4096 tokens, seqlen 512 | 23.7852 |
| learned dense HDQ NVFP4 | WikiText-2 PPL, 4096 tokens, seqlen 512 | 31.5642 |
| vanilla NVFP4 vs BF16 | full-vocab KL, 8 train windows, seqlen 512 | 0.5322 |
| learned dense HDQ NVFP4 vs BF16 | full-vocab KL, 8 train windows, seqlen 512 | 1.1166 |

Logs/results:

- `/dq-runs/qwen3-4b-hdq-w4a4-20260514T055532Z/validation/ppl_bf16_wikitext_test_4096_s512.json`
- `/dq-runs/qwen3-4b-hdq-w4a4-20260514T055532Z/validation/ppl_vanilla_nvfp4_wikitext_test_4096_s512.json`
- `/dq-runs/qwen3-4b-hdq-w4a4-20260514T055532Z/validation/ppl_fixed_score_hdqonly_wikitext_test_4096_s512.json`
- `/dq-runs/qwen3-4b-hdq-w4a4-20260514T055532Z/validation/kl_vanilla_nvfp4_vs_bf16_n8_s512.json`
- `/dq-runs/qwen3-4b-hdq-w4a4-20260514T055532Z/validation/kl_fixed_score_hdqonly_vs_bf16_n8_s512.json`

The learned dense online transform path also fails the runtime bar: it can be
made to load with the local vLLM selector patch, but vLLM serves it through a
generic dense online transform wrapper rather than a performant hadacore-style
path. The exporter now treats online `random-matrix` HDQ as research-only and
requires `PRISMAQUANT_ALLOW_UNSUPPORTED_HDQ_ONLINE=1`; deployment exports
should use native `hadamard` online rotations or `folded_only`.

Retest after the one-line qutlass selector patch
(`random-matrix` routed through the generic transform wrapper) confirmed this
is not merely a load-path issue:

| artifact | metric | result |
|---|---|---:|
| learned dense HDQ NVFP4 (`exported_fixed_score_hdqonly_vllmfix`) | prompt smoke, 16 decode tokens | passed, 49.37s generate |
| learned dense HDQ NVFP4 (`exported_fixed_score_hdqonly_vllmfix`) | WikiText-2 PPL, first 1024 tokens, seqlen 512 | 32.0616 |
| learned dense HDQ NVFP4 (`exported_fixed_score_hdqonly_vllmfix`) vs BF16 | full-vocab KL, 8 train windows, seqlen 512 | 1.1166 |

Logs/results:

- `/dq-runs/qwen3-4b-hdq-w4a4-20260514T055532Z/logs/vllm_smoke_fixed_score_hdqonly_vllmfix.log`
- `/dq-runs/qwen3-4b-hdq-w4a4-20260514T055532Z/validation/ppl_fixed_score_hdqonly_vllmfix_wikitext_test_1024_s512.json`
- `/dq-runs/qwen3-4b-hdq-w4a4-20260514T055532Z/validation/kl_fixed_score_hdqonly_vllmfix_vs_bf16_n8_s512.json`

The dense runtime export stores `sqrt(G) * M^T`, matching vLLM's generic
transform wrapper, which applies `1 / sqrt(G)` internally. The remaining
failure is therefore rotation quality/generalization for the old learned
dense 1k-calibration artifact, not a simple transform-scale convention bug.

Implemented audit fixes:

- Activation capture now uses uniform priority-reservoir sampling instead of
  first-rows capture in both joint HDQ search and production cache fill.
- The HDQ STE NVFP4 group scale now rounds through FP8 E4M3 with STE, matching
  production NVFP4 scale quantization more closely.
- Export tests now cover non-symmetric dense matrices and the vLLM dense
  transform scaling convention explicitly.
- HDQ-enabled pipeline runs now include native online Hadamard clusters by
  default (`HADAMARD_DUQUANT_ROTATION_SCOPE=all`,
  `HADAMARD_DUQUANT_ONLINE_ROTATION_MODE=hadamard`).
- Production cache metadata now carries HDQ qname→cluster routing, and
  perturb/recache replay measures HDQ consumer activation ranges in the same
  runtime basis (`x @ H^T`) that vLLM quantizes online.
- Added `tools/measure_vllm_wikitext_ppl.py` for prompt-logprob PPL checks.
- Allocator-side HDQ cost ingestion now ignores non-finite sidecar scores for
  a cluster/format pair instead of poisoning the DP. In the 4B native-Hadamard
  rerun, five early `down_proj` NVFP4 scores were NaN; those cells correctly
  fell back to the original measured allocator cost.

## 2026-05-14 Qwen3-4B native-Hadamard HDQ rerun

Artifacts:

- Search/cache/export root:
  `/dq-runs/qwen3-4b-hdq-w4a4-20260514T055532Z/artifacts_hadamard_online`
- Export:
  `/dq-runs/qwen3-4b-hdq-w4a4-20260514T055532Z/exported_hadamard_online`
- vLLM image: `vllm-eugr-v020-hdq-h16:latest`

Build summary:

- Joint search used `rotation_scope=all`, `online_rotation_mode=hadamard`,
  group size 16, W4A4 solver/score, 4 calibration samples x 256 tokens.
- Allocator target `4.5` bpp selected all 144 serving groups as NVFP4.
- HDQ picks: 35/144 clusters chose `rot+NVFP4`; 109/144 chose
  `no_rot+NVFP4`.
- Export loaded 35 HDQ clusters; 3 runtime `transform_config` groups were
  emitted as native `type=hadamard`, `head_dim=16`. The other selected
  rotations were folded through adjacent producer/consumer weights.
- vLLM smoke loaded with `FlashInferCutlassNvFp4LinearKernel` and generated:
  `Paris. The capital of Spain is Madrid. The capital of Italy is Rome.`

Validation on the saved baseline contract:

| artifact | PPL, WikiText test 4096/s512 | mean NLL | KL vs BF16, n=8/s512 |
| --- | ---: | ---: | ---: |
| BF16 | 20.3326 | 3.01223 | - |
| vanilla NVFP4 | 23.7852 | 3.16906 | 0.53221 |
| native-Hadamard HDQ NVFP4 | 22.4271 | 3.11027 | 1.50505 |

Interpretation:

- PPL improved versus vanilla NVFP4 by 1.358 absolute / 5.71% relative.
- The model remains 10.30% worse than BF16 on this short PPL run.
- KL regressed versus vanilla NVFP4 and is dominated by an outlier sample
  (`kl_max=8.4539`). The next debugging target is why the KL contract is less
  stable than PPL for this small-calibration HDQ artifact.

Disposition of Claude's audit:

- Calibration size and held-out validation remain the highest-leverage
  unresolved issue. The current 1k-token search signal is not enough for a
  shipping claim; next run should use 32k+ train tokens and a disjoint held-out
  score.
- The STE/renderer proxy gap is reduced but not closed. E4M3 scale rounding is
  modeled; global input-scale interaction and production `four_over_six_mse`
  scale search still require either a closer surrogate or renderer-based
  candidate validation.
- Per-cluster greediness, fixed permutation, no weak-rotation interpolation,
  and candidate bootstrap stability remain research refinements. They should
  not block a native-Hadamard rerun, but they must be disclosed.
- HDQ x GPTQ interaction is unmeasured. The current HDQ-only result failed
  before GPTQ enters the picture; any production recipe must compare matched
  vanilla NVFP4+GPTQ/scale-sweep vs HDQ+NVFP4+GPTQ/scale-sweep end to end.
- Determinism and sidecar uncertainty remain unresolved for paper-grade
  artifacts. Sidecars still report point estimates, not confidence intervals.

## 2026-05-14 32k held-out gate rerun

Implemented the highest-leverage audit fixes for the HDQ search path:

- Joint search now captures separate train and held-out activation reservoirs.
- HDQ sidecar candidates carry train/validation Fisher-MSE and relative gains.
- Rotations are committed only when both train and validation relative gains
  clear the configured margin.
- `run_joint_hadamard_search.py` accepts `--n-validation-samples`,
  `--validation-seed`, `--cluster-kind-filter`,
  `--rotation-min-train-gain`, and `--rotation-min-validation-gain`.
- `tools/measure_vllm_full_kl.py` now separates `max_logprobs` from the real
  model vocabulary size; the n=32 Qwen3-4B teacher payload is strict
  full-vocab KL over 151,936 logits.

Search contract:

- Model: `/hfcache/Qwen3-4B`
- Train calibration: 128 x 256 = 32,768 tokens from
  `/dq-runs/calibration/diverse-v1.jsonl`
- Held-out validation: 32 x 256 = 8,192 tokens
- Activation reservoir: 1,024 rows per cluster
- Solver/scorer: W4A4, `solver_weight_decay=1e-4`
- Commit gate: train gain >= 0.1% and validation gain >= 0.1%
- Format menu: `NVFP4,BF16`; rotation-enabled format: `NVFP4`

Rotation acceptance:

| artifact | scope | accepted rotations | rejected rotations | export runtime transforms |
| --- | --- | ---: | ---: | ---: |
| `gated_online_down_32k` | online `down_proj` only | 0/36 | 36 | 0 |
| `gated_online_all_32k` | online all insertion kinds | 1/108 | 107 | 1 native Hadamard H16 |
| `gated_folded_32k` | folded-only `v_o` | 13/36 | 23 | 0 |

The single online-all survivor is `model.layers.0.mlp.residual`
(`train_gain=0.6158%`, `validation_gain=0.5705%`). The folded-only survivors
are all `v_o` rotations. All `down_proj` online rotations failed the held-out
gate.

Production recache ran for all three artifacts:

| artifact | activation max_abs after/before ratio | moved >5% |
| --- | --- | ---: |
| `gated_online_down_32k` | p50=1.000, p95=1.039, max=1.393 | 54/252 |
| `gated_online_all_32k` | p50=1.000, p95=1.040, max=1.318 | 53/252 |
| `gated_folded_32k` | p50=1.000, p95=1.069, max=2.188 | 51/252 |

vLLM validation used `vllm-eugr-v020-hdq-h16:latest` and loaded NVFP4 with
`FlashInferCutlassNvFp4LinearKernel`. Three concurrent vLLM KL instances at
`gpu_memory_utilization=0.22` overcommitted KV memory; two-at-a-time worked
for quantized artifacts, and failed runs were rerun sequentially.

Validation summary:

| artifact | PPL, WikiText test 4096/s512 | mean NLL | KL vs BF16, n=32/s512 | KL max |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 20.3326 | 3.01223 | - | - |
| vanilla NVFP4 | 23.7852 | 3.16906 | **0.41126** | **2.76764** |
| ungated native-Hadamard HDQ | **22.4271** | **3.11027** | 0.56312 | 8.45392 |
| gated online-down, no accepted rotations | 22.9040 | 3.13131 | 0.62615 | 6.26944 |
| gated online-all, 1 accepted online rotation | 22.8850 | 3.13048 | 0.44904 | 3.02905 |
| gated folded `v_o`, 13 accepted rotations | 22.6857 | 3.12173 | 0.61101 | 6.89624 |

Result:

- The PPL improvement is real: ungated native-Hadamard HDQ improves over saved
  vanilla NVFP4 by 1.358 PPL / 5.71% relative; the gated folded artifact still
  improves by 1.100 PPL / 4.62% relative.
- That PPL improvement does **not** yet justify making HDQ production-default.
  All HDQ variants measured here are worse than vanilla NVFP4 on n=32
  full-vocab KL. The best gated KL artifact (`gated_online_all_32k`) is close
  to vanilla but only accepts one online rotation and its PPL gain over the
  no-rotation 32k baseline is just 0.019.
- The held-out gate is doing useful work: most online Hadamard H16 rotations
  do not generalize even at 32k train tokens. This argues for keeping HDQ as a
  research/candidate arm until PPL and KL both move in the right direction
  under the same calibration contract.

Logs/results:

- `logs/search_gated_online_down_32k.log`
- `logs/search_gated_online_all_32k.log`
- `logs/search_gated_folded_32k.log`
- `logs/production_cache_gated_online_down_32k.log`
- `logs/production_cache_gated_online_all_32k.log`
- `logs/production_cache_gated_folded_32k.log`
- `logs/kl_teacher_bf16_n32_s512.log`
- `validation/kl_vanilla_nvfp4_vs_bf16_n32_s512.json`
- `validation/kl_hadamard_online_vs_bf16_n32_s512.json`
- `validation/kl_gated_online_down_32k_vs_bf16_n32_s512.json`
- `validation/kl_gated_online_all_32k_vs_bf16_n32_s512.json`
- `validation/kl_gated_folded_32k_vs_bf16_n32_s512.json`

## Open questions

- **2026-05-14 Qwen3.5-0.8B HDQ-only vLLM smoke.**
  Folded-only HDQ passed first. Run
  `/dq-runs/qwen35-0p8b-hdq-only-smoke-20260514T110139Z` used
  `HADAMARD_DUQUANT_ROTATION_SCOPE=folded_only`, W4A4 score/solver,
  `PRODUCTION_CACHE_LEVERS=none`, and explicit disables for GPTQ,
  scale-sweep, AWQ, clip solvers, Fisher-GPTQ/clip, HALO, and
  block-output-match. The pipeline completed, exported vanilla
  compressed-tensors with no runtime transform groups, and vLLM loaded
  it in both eager and graph mode using `FlashInferCutlassNvFp4LinearKernel`.
  Logs:
  `/dq-runs/qwen35-0p8b-hdq-only-smoke-20260514T110139Z/vllm_prompt_smoke_eager.log`
  and
  `/dq-runs/qwen35-0p8b-hdq-only-smoke-20260514T110139Z/vllm_prompt_smoke_graph.log`.

- **Online transform vLLM status.** Stock vLLM 0.20 rejects NVFP4 online
  `random-matrix`/H16 paths by routing them through the Qutlass wrapper
  assert before weight loading. The local research patch in
  `/home/rob/spark-vllm-docker/vllm_nvfp4_hadamard_h16.patch` excludes both
  `hadamard` and `random-matrix` transforms from that placeholder assert,
  and a disposable in-container application of that patch lets
  `exported_fixed_score_hdqonly_vllmfix` load. The learned dense path remains
  far too slow and inaccurate for deployment; native `hadamard` H16 remains
  the only plausible online vLLM-compatible path in this branch.

- Does end-to-end PPL/KL improve meaningfully vs the no-HDQ baseline?
  Per-cluster gains are modest but cumulative across 121 rotation-eligible
  clusters and 224 consumer Linears on Qwen3-4B.
- Would larger calibration (32k+ tokens) or a larger model materially
  change the picture? Worth retesting on Qwen3.5-4B once the small-model
  results are nailed down.
- Should W4A4 become the default `HADAMARD_DUQUANT_SOLVER_LOSS`? The
  rotations are more trustworthy under the production objective but the
  joint loss costs 2× compute per iter. For NVFP4 W4A4 deployment the
  trade-off is clearly worth it; for any future W4A16 (weight-only)
  format the W-only loss remains correct and faster.

## Related work

- **DuQuant++** (arXiv:2604.17789, Apr 2026): targets MXFP4 W4A4 with
  rotation block size aligned to MXFP4 G=32.
- **MR-GPTQ / "Bridging the Gap..."** (arXiv:2509.23202, ICLR 2026):
  studies NVFP4 vs MXFP4 directly; reports rotations hurt NVFP4 at
  small group size with RTN, GPTQ-style methods recover better.
- **NVIDIA Transformer Engine NVFP4 docs**: uses Random Hadamard
  Transform for NVFP4 *training*, not PTQ inference.
- **AMXFP4** (arXiv:2411.09909): argues microscaling FP4 activation
  outlier handling has different tradeoffs than integer rotation
  methods.
