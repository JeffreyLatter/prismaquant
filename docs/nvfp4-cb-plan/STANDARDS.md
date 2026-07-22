# gridbook / NVFP4-CB — FINAL kernel & format standards

Dated 2026-07-21 (Robert: "make a definitive determination about final kernel
and format standards"). This page is the contract production runs build
against. Changes to it require a served A/B, not a preference.

## Format standard (what an artifact may contain)

**Production formats** — the complete set:

| Family | Rungs | Rate | Scale coding | Mode |
|---|---|---|---|---|
| NVFP4_CB (fp4 grid) | K12–K24, EVERY integer | 2.0–3.28 bpw (2.28–3.53 eff.) | two-tier v2 (E8M0 super + 4-bit sub codes, 0.28125 bpw) | product, ceil-first uneven splits |
| FP8_CB (e4m3 grid) | K28–K48, EVERY integer | 3.5–6.0 bpw | per-output-channel fp32 (~free) | product, ceil-first uneven splits |
| NVFP4 (vanilla) | — | 4.5 bpw | group-16 E4M3 | menu member; Blackwell-only serving |
| FP8_DYNAMIC | — | 8 bpw | per-channel | menu member |
| BF16 / FP8_SOURCE | — | 16 / ~8 | — | passthrough-only (source dtype) |

- Codeword layout: 32 k-bit codewords per 256-weight superblock, LSB-first;
  sub-index bit split is **ceil-first** (`_bit_split`), sub-0 at the LSBs.
  Encoder-anchored tests pin this; it is frozen.
- fp4 scale coding v1 (bare E4M3 plane) is legacy-compat only: readable,
  never produced by new exports.
- **Signed S-rungs (S13–S16): spec-legal, NOT production-standard.** Kernels
  exist (CUDA + Triton, GPU battery pending); they enter the production set
  only if the K-vs-S head-to-head shows a real win at matched bpp. They
  duplicate rate points K13–K16 already serve — value unproven, so they stay
  research. Full mode: spec-reserved, unimplemented.
- MTP sidecars: CB-quantized, rung by the canon throughput selector
  (`mtp_rung_selection.py`). Vision towers (VLMs): vanilla NVFP4.
- Standard production menu = both CB ladders (all integers) + NVFP4 +
  FP8_DYNAMIC + BF16 (+FP8_SOURCE where the source is fp8). Target hardware:
  Blackwell (GB10 sm_121 / RTX 5090 sm_120). Artifacts that happen to
  allocate zero vanilla-NVFP4 units remain Ada-servable as a bonus, never a
  constraint.

## Kernel standard (the serving surface)

| Regime | Standard path | Status |
|---|---|---|
| Decode M≤16, dense | CUDA GEMV: fp8 double-buffer; fp4-v2 round-2 schedule | DEFAULT |
| Decode M≤16, MoE | grouped CUDA GEMVs + deterministic combine | DEFAULT |
| Prefill dense fp8-CB | `cb_expand_fp8` (direct e4m3 bytes) → stock cutlass W8A8 | DEFAULT |
| Prefill dense fp4-CB | Triton v2 expand (bf16, composed scales) → cuBLAS | DEFAULT |
| Prefill MoE | per-expert loop (one host sync) | DEFAULT |
| Dispatch | M-branch-hoist opaque custom ops (layer registry) | DEFAULT |
| Mid-M 17–128 fp8-CB | CUTLASS fused decode-in-prologue | OPT-IN until served logprob A/B |
| Large-M fused | persistent-N §4b (CUTLASS restructure) | ROADMAP (GO recorded; 2–4 GPU days) |
| MoE prefill alts | stock-kernel (capture-safe) / batched | OPT-IN |
| w2 rowpack, damp sweep, … | measured negative | ARCHIVED behind env switches |

- Serving graph standard for ship configs: **mode-0 + FULL_DECODE_ONLY**.
  The compile lane (mode-3, any cudagraph flavor) is correctness-clean since
  the M-branch hoist and measured at decode parity (13.5–14.0 band on Hy3,
  all four configs) — supported, not default. Revisit only with a lever that
  moves the measured number.
- All kernels are k-parameterized over the full integer ladder (ceil-first
  splits); odd-k fp8 rows stage byte-granular until an odd rung is measured
  hot on a served artifact.
- Capability floor 8.0; JIT builds target the local arch.

## Cost standard

- CB rung costs: measured anchors + the **floor-anchored RD law**
  `D(k) = F + C·2^(−α·k)` per (Linear, family) — F is the measured
  infinite-k grid floor (fp8: per-channel-fp8 RTN render; fp4: two-tier
  E2M1 RTN render), (C, α) least-squares in log space over the anchors —
  **holdout-gated**: any tensor whose holdout error misses the bar falls
  back to full per-rung measurement. The gate is the contract; the law is
  only ever a proposal under it.
