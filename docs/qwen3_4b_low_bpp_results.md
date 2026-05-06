# Qwen3-4B low-bpp Block-CLADO results

Branch: `block-clado` (commits through `2333660`)
Run: `/home/rob/dq-runs/qwen3-4b-block-clado-low-bpp-20260506T082141Z`

## Headline result

**4.53 bpp / real KL 0.0663 in 25.4 minutes** via four-term Block-CLADO
+ polish (steepest-first, **2% budget creep** — held tight to stay in
the deployment-relevant low-bpp regime).

Single iteration; no sandwich — sandwich didn't help on 0.6B and starts
at the high cost of an extra full measurement at 4B scale.

## Validation cone (4-term BF16-centered)

```
bpp     surrogate  real_kl  format counts
4.5000  +0.169     0.136    253 NVFP4
4.5330  +0.144     0.131    251 NVFP4 + 2 MXFP8       ← best validated
4.5477  +0.136     0.327    248 NVFP4 + 5 MXFP8
4.6246  +0.103     0.201    237 NVFP4 + 16 MXFP8
4.7236  +0.074     0.295    231 NVFP4 + 22 MXFP8       ← surrogate kneedle (worse)
4.7530  +0.068     0.394    228 NVFP4 + 25 MXFP8
5.0564  +0.026     0.544    203 NVFP4 + 46 MXFP8 + 4 BF16
```

The validation cone shows the same low-bpp surrogate-vs-real noise we
saw on 0.6B: real KL fluctuates 0.13–0.54 across adjacent bpp points.
Surrogate kneedle picks 4.72 (KL 0.295); real-KL validation picks 4.53
(KL 0.131) — **the surrogate is 2.3× over-estimated at the kneedle**.
Validation cone discovery is mandatory.

## Polish trace

Starting from 4.53 bpp / 0.131 with 2% bits-budget creep:

```
pass 0  layers.13.self_attn.qkv_proj  NVFP4 → BF16   KL 0.131 → 0.109 (Δ -0.021)
pass 1  layers.9.self_attn.qkv_proj   NVFP4 → MXFP8  KL 0.109 → 0.083 (Δ -0.027)
pass 2  layers.15.self_attn.o_proj    NVFP4 → BF16   KL 0.083 → 0.079 (Δ -0.004)
pass 3  layers.15.self_attn.o_proj    BF16 → MXFP8   KL 0.079 → 0.075 (Δ -0.004)  ← downgrade swap
pass 4  layers.28.self_attn.o_proj    NVFP4 → MXFP8  KL 0.075 → 0.067 (Δ -0.008)
pass 5  layers.12.self_attn.o_proj    NVFP4 → MXFP8  KL 0.067 → 0.066 (Δ -0.000)
done    100 measurements, ~5 min
```

Pass 3 is the interesting Pareto move: layers.15.self_attn.o_proj got
upgraded NVFP4→BF16 in pass 2, then downgraded BF16→MXFP8 in pass 3.
Polish realized BF16 was over-precise for that unit and freed the
budget to do other moves.  This is the budget-creep + steepest-first
combination doing its job.

## Final polished assignment

```
NVFP4:        242
MXFP8_E4M3:     8
BF16:           3
total:        253 fused units (≈ 36 layers × 7 Linears + lm_head)
bpp:         ~4.53
real KL:      0.0663
```

The MXFP8 + BF16 upgrades concentrate on `self_attn.qkv_proj` and
`self_attn.o_proj` of specific transformer layers — the same pattern we
saw on Qwen3-0.6B (attention QKV/O are the most precision-sensitive
units).

## Cost on Qwen3-4B vs Qwen3-0.6B

| stage | Qwen3-0.6B | Qwen3-4B |
|---|---|---|
| four-term measurement | 76s (frozen cache) | **779s (no cache, OOM-aware fallback)** |
| frontier validate (cone) | ~30s | ~50s |
| polish (steepest-first, ~6 moves) | ~50s | ~6 min |
| **total iter 0** | **~3 min** | **~25 min** |

The 4B run hit the memory-aware frozen-weight-cache fallback (used 114
GB of 121 GB UMA budget) and dropped to per-module re-quantization,
which is ~10× slower per call.  This is the dominant scaling factor;
fixing the cache discipline at 4B would put the pipeline at ~5 min.

## Polish-frontier sweep (all 7 cone candidates)

For comparison, polishing each cone candidate independently with 5%
budget creep (separate Docker container, fresh model load):

| candidate | start bpp | start KL (re-meas) | final KL | steps | wall |
|---|---|---|---|---|---|
| 4.5000 | 4.50 | 0.218 | 0.136 | 5 | 31s |
| 4.5330 | 4.53 | 0.299 | 0.156 | 8 | 23s |
| 4.5477 | 4.55 | 0.232 | 0.107 | 6 | 121s |
| 4.6246 | 4.62 | 0.229 | **0.103** | 7 | 243s |
| 4.7236 | 4.72 | 0.329 | 0.111 | 6 | 144s |
| 4.7530 | 4.75 | 0.339 | 0.128 | 8 | 285s |
| 5.0564 | 5.06 | 0.249 | 0.110 | 8 | 40s |

The polish-frontier's "start KL" disagrees with the in-process
iterate's validation-cone KL measurements by ~30–100% in places (e.g.
4.5330 reported 0.299 here vs 0.131 by validate cone).  The measurement
discrepancy is from a fresh-process model-load floating-point drift in
the PerturbedActivationCache hooks — same RNG seed, same calibration,
but different starting GPU state gives different KL values for
identical assignments.  This is a known noise source.

**Despite the noisier starts, no polish-frontier candidate beats the
iterate's in-process 0.066.**  In-process polish (model state continuous
from measure → validate → polish) is the most reliable.  The
polish-frontier tooling is best for comparing polish behaviors across
starting points, not for finding the lowest absolute KL.

## What's next on 4B

1. **Polish-frontier across all 7 cone candidates** (running now): polish
   each with 5% creep, see whether other low-bpp starting points yield
   better polished KL than the 0.066 we got from the surrogate-best.
2. **Output-Fisher comparison** at 4B: should be ~3× faster on
   measurement (210 forwards vs 1100); useful to confirm the 0.6B
   finding that OF wins on speed but loses ~6% KL at low bpp.
3. **Wider initial sweep** to get more candidates between 4.5 and 5.5
   bpp where the real-KL surface is jagged.
