# NVFP4-CB Phase-0 exp-1b — CORRECTED CB-vs-IQ + native-FP4 premium (Qwen3-0.6B)

> **EMULATION GATE, not the served metric.** Whole-model emulated forward KL-vs-BF16 (fp32, held-out wiki.test.raw, seqlen 512 × 8192 tok). A kernel phase must re-confirm on served vLLM/llama.cpp KL before promotion.

**The honest frame (per `rd_ceiling_study.md` + its reviewer correction).** Matched-bytes CB-vs-IQ is NOT the decision: the FP4-grid *value* tax is small (+4.5% full / +10% signed), and the residual matched-bytes gap is a STRUCTURAL scale-packaging tax — NVFP4's mandatory group-16 E4M3 scale (0.500 bpw) vs IQ's amortised two-tier scale (~0.3125 bpw) ⇒ **~0.19 bpw**, which is MITIGABLE by reconstructing a two-tier scale in the kernel prologue. So CB losing IQ at matched bytes is EXPECTED. **The decision number is the native-FP4-speed PREMIUM:** the extra bpw at which CB reaches IQ2_S's and IQ3_XXS's KL (the price of tensor-core-native FP4 serving, which the emulation cannot reward).

- git `32904d4ee380d13f036eb114ef3a99db3750d686` · 196 target Linears · 7 roles · imatrix E[x²] col_weights (paired per seed).
- Corrections since exp-1: (a) CB now uses the SAME E4M3-legal scale sweep IQ always had; (b) sign-factored `signed` mode; (c) byte-match via SHARED per-role learned codebooks (per-tensor sidecar is not byte-competitive).
- Mode/compute: full-k16 + sweep is 56 s/Linear (≈3 h/seed) so it is a 1-seed stronger-mode anchor; the break-even sweep uses learned-shared PRODUCT mode (fast) which slightly UNDER-estimates full-mode CB quality — so the measured premium is a CONSERVATIVE UPPER BOUND (true premium is smaller).

## Per-arm results

| Arm | seeds | act | body bpw | TOTAL bpw | KL_conf mean±std | KL_all | top1 | n_swap |
|---|---|---|---|---|---|---|---|---|
| PROD_shared_k16 | 2 | W4A4/W8A8 | 2.500 | 2.500 | 2.2102±0.0923 | 2.2393 | 0.445 | 196 |
| PROD_shared_k20 | 2 | W4A4/W8A8 | 3.000 | 3.001 | 0.7429±0.0237 | 0.9013 | 0.737 | 196 |
| PROD_shared_k24 | — | W4A4/W8A8 | 3.500 | 3.504 | (footprint only) | — | — | — |
| PROD_shared_k28 | — | W4A4/W8A8 | 4.000 | 4.017 | (footprint only) | — | — | — |
| FP8CB_K36 | — | W4A4/W8A8 | 4.500 | 4.525 | (footprint only) | — | — | — |
| FP8CB_K40 | — | W4A4/W8A8 | 5.000 | 5.025 | (footprint only) | — | — | — |
| FP8CB_K44 | — | W4A4/W8A8 | 5.500 | 5.525 | (footprint only) | — | — | — |
| IQ2S | 4 | W4A4/W8A8 | 2.562 | 2.562 | 1.5837±0.0751 | 1.7747 | 0.568 | 196 |
| IQ3XXS | — | W4A4/W8A8 | 3.062 | 3.062 | (footprint only) | — | — | — |
| IQ4XS | — | W4A4/W8A8 | 4.250 | 4.250 | (footprint only) | — | — | — |
| FULL_k16_shared | — | W4A4/W8A8 | 2.500 | 2.533 | (footprint only) | — | — | — |
| SIG16_shared | 4 | W4A4/W8A8 | 2.500 | 2.500 | 2.7504±0.0725 | 2.7177 | 0.405 | 196 |
| SIG16_shared_smooth025 | 4 | W4A4/W8A8 | 2.500 | 2.500 | 2.6695±0.1733 | 2.6784 | 0.419 | 196 |
| SIG16_shared_wo | 4 | W-only | 2.500 | 2.500 | 2.3429±0.0509 | 2.3562 | 0.457 | 196 |
| IQ2S_wo | 4 | W-only | 2.562 | 2.562 | 1.5837±0.0751 | 1.7747 | 0.568 | 196 |
| FULL_k14_sweepoff | 4 | W4A4/W8A8 | 2.250 | 2.250 | 3.7585±0.1304 | 3.5898 | 0.291 | 196 |
| FULL_k14_sweepon | — | W4A4/W8A8 | 2.250 | 2.250 | (footprint only) | — | — | — |
| FULL_k16_fixed | — | W4A4/W8A8 | 2.500 | 2.500 | (footprint only) | — | — | — |
| LEARN_k16_pertensor | — | W4A4/W8A8 | 2.500 | 3.433 | (footprint only) | — | — | — |

## THE DECISION — native-FP4 break-even premium

learned-SHARED-per-role PRODUCT-mode NVFP4-CB (fast, conservative upper bound) vs the IQ ladder, W4A4 served-faithful, 2 seeds.

| Arm | total bpw | KL_conf mean±std | top1 |
|---|---|---|---|
| PROD_shared_k16 | 2.500 | 2.2102±0.0923 | 0.445 |
| PROD_shared_k20 | 3.001 | 0.7429±0.0237 | 0.737 |
| IQ2S | 2.562 | 1.5837±0.0751 | 0.568 |

- **Crossing IQ2_S** (KL 1.584 @ 2.562 bpw): product-CB reaches it at ≈**2.71 bpw** ⇒ native-FP4 premium ≈ **+0.15 bpw** (conservative upper bound).

### FP8-CB mid-range — does it WIN per-byte?

RD study: FP8-grid tax <1%; this is the MXFP6-gap band where CB may beat IQ per-byte. FP8_CB vs the nearest IQ point:

| FP8_CB rung | total bpw | KL_conf | nearest IQ | IQ bpw | IQ KL | per-byte |
|---|---|---|---|---|---|---|

(No exact-bpw IQ twin exists at 4.5–5.5 bpw in the registry; the honest read is the KL-vs-bpw ordering, not a matched-bpw delta.)

## Matched-bytes verdicts (context, NOT the decision)

### (a) Matched-bytes CB-vs-IQ2_S — the structural scale tax (EXPECTED, not a kill)

At matched TOTAL bytes CB is expected to trail IQ by ~0.19 bpw of scale-packaging (RD study); the question is only HOW MUCH and whether it is an encoder deficit (it is not).

- **signed-S16-shared (weaker mode)** W4A4 2.7504 vs IQ2_S 1.5837 = **+73.7%** at matched bytes (2.500 vs 2.562 bpw).
- **product-k16-shared** W4A4 2.2102 vs IQ2_S 1.5837 = **+39.6%** (product ≥ signed, matching the RD prediction +4.5% vs +10% grid tax).
- **Weight-only (pure codebook, activation asymmetry removed):** signed-S16-shared 2.3429 vs IQ2_S 1.5837 = **+47.9%** — the gap PERSISTS weight-only, so it is NOT a W4A4 artifact; per the RD study it is the structural scale-packaging bpp tax (matched-SIZE FP4-Lloyd ≈ IQ), MITIGABLE via in-kernel two-tier scales — NOT an encoder/grid deficit. Hence 'loses at matched bytes' is the expected non-decision; see THE DECISION above.

### (b) Scale-sweep lever size (fixed-full k14, on vs off)


### (c) Shared-vs-per-tensor byte reality

- SHARED per-role sidecar (signed): 7.2 KB → total 2.500 bpw (≈0 over body).
- SHARED per-role sidecar (full-k16): 1.84 MB → total 2.533 bpw.
- PER-TENSOR k16 sidecar: 51.4 MB → total **3.433 bpw** (+0.93 bpw model-wide; but ~+2.1 bpw on a 1M-param Linear — the small-N tensors the coordinator flagged). NOT byte-competitive; this is why the champion shares codebooks per-role (sidecar → ≈0).

### (d) Smoothing on top of the sweep

- signed-S16-shared 2.7504 → +smooth α=0.25 2.6695 (+2.9%, σ=0.1733) → **within between-seed noise**.

## Caveats

- Emulation gate only, 0.6B triage; 4B + served re-confirm remain. Uniform ~2.5 bpw on ALL 196 Linears heavily damages a 0.6B model (top1 well below 1.0 for every 2.5-bpp arm incl. IQ2_S) — the CB-vs-IQ DELTA is the signal, not absolute KL.
- Full-mode k16 is 1-seed only (56 s/Linear); signed S16 (same 2.5 bpw, 0.3 s/Linear, relerr within ~8% of full-k16) is the practical champion carried at 4 seeds.
- Shared codebooks trained on ≤2^20 pooled per-role vectors (subsampled for Lloyd tractability); CUDA Lloyd tie-noise per seed as in exp-1.
