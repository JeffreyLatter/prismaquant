# NVFP4-CB Phase-0 exp-1b — CORRECTED CB-vs-IQ (Qwen3-0.6B)

> **EMULATION GATE, not the served metric.** Whole-model emulated forward KL-vs-BF16 (fp32, held-out wiki.test.raw, seqlen 512 × 8192 tok). Corrects exp-1's rendering asymmetry: CB now uses the SAME E4M3-legal scale sweep the IQ arms always had, adds the sign-factored `signed` mode, and byte-matches via a SHARED per-role learned codebook. A kernel phase must re-confirm on served vLLM/llama.cpp KL before promotion.

- git `bf58068581abe7f0ee70be04a88eeb344a86fb8a` · 196 target Linears · 7 roles · imatrix E[x²] col_weights (paired per seed).
- Compute note: full-mode k16 + sweep costs ~56 s/Linear (≈3 h/seed) vs signed-S16 ~0.3 s/Linear — so signed S16 is the practical champion run at 4 seeds; the full-k16 arms are 1-seed CEILING references, and per-tensor-k16 is footprint-only.

## Per-arm results

| Arm | seeds | act | body bpw | TOTAL bpw | KL_conf mean±std | KL_all | top1 | n_swap |
|---|---|---|---|---|---|---|---|---|
| IQ2S | 4 | W4A4/W8A8 | 2.562 | 2.562 | 1.5837±0.0751 | 1.7747 | 0.568 | 196 |
| SIG16_shared | 4 | W4A4/W8A8 | 2.500 | 2.500 | 2.7504±0.0725 | 2.7177 | 0.405 | 196 |
| SIG16_shared_smooth025 | 4 | W4A4/W8A8 | 2.500 | 2.500 | 2.6695±0.1733 | 2.6784 | 0.419 | 196 |
| SIG16_shared_wo | 4 | W-only | 2.500 | 2.500 | 2.3429±0.0509 | 2.3562 | 0.457 | 196 |
| IQ2S_wo | 4 | W-only | 2.562 | 2.562 | 1.5837±0.0751 | 1.7747 | 0.568 | 196 |
| FULL_k14_sweepoff | — | W4A4/W8A8 | 2.250 | 2.250 | (footprint only) | — | — | — |
| FULL_k14_sweepon | — | W4A4/W8A8 | 2.250 | 2.250 | (footprint only) | — | — | — |
| IQ3XXS | — | W4A4/W8A8 | 3.062 | 3.062 | (footprint only) | — | — | — |
| FP8CB40_sweep | — | W4A4/W8A8 | 5.000 | 5.025 | (footprint only) | — | — | — |
| FULL_k16_fixed | — | W4A4/W8A8 | 2.500 | 2.500 | (footprint only) | — | — | — |
| FULL_k16_shared | — | W4A4/W8A8 | 2.500 | 2.533 | (footprint only) | — | — | — |
| LEARN_k16_pertensor | — | W4A4/W8A8 | 2.500 | 3.433 | (footprint only) | — | — | — |

## Decision-gate verdicts

### (a) Does corrected-CB close the exp-1 +66% IQ2_S gap?

- **W4A4 (served-faithful):** signed-S16-shared 2.7504 vs IQ2_S 1.5837 (+73.7%, σ=0.0751) at matched TOTAL bytes (2.500 vs 2.562 bpw) → **still LOSES IQ2_S by +73.7%** (exp-1 was +66%).
- **Weight-only (pure codebook-vs-codebook):** signed-S16-shared 2.3429 vs IQ2_S 1.5837 (+47.9%, σ=0.0751) → **LOSES by +47.9%** — **>15% on BOTH W4A4 and weight-only = KILL signal**.

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
