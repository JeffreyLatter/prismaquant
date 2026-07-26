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
- **Signed S-rungs (S13–S16): CLOSED as research-only (measured,
  2026-07-22).** The K-vs-S head-to-head on Qwen3.5-0.8B (matched-rate menu,
  776 per-(Linear,k) direct cost comparisons): K wins 79%, median S penalty
  +0.5–2.2%, allocator placed 6 S-units vs 147 K-units (only linear-attn
  in_proj_a/b/qkv/z ever preferred S). Serving propriety PROVEN: the signed
  chain (encoder → export → vLLM load → decode) is bit-exact on the real
  artifact (max |serve − reconstruct| = 0) plus the 18-test GPU battery.
  S-rungs stay OFF production menus — correct but not worth menu space; the
  spec keeps them for exotic weight geometries. Full mode: spec-reserved,
  unimplemented.
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
| Prefill MoE fp8-CB | 'grouped_fused': tile-indexed grouped decode-in-prologue CUTLASS (one launch/stage, device descriptors, NO weight expansion; R1→stock fallback chain) | DEFAULT (promoted 2026-07-26: teacher-relative conf-KL gate passed on 35B, 4,285 vs stock 3,932 tok/s ±collapsed variance; serve with `--max-num-batched-tokens 16384`) |
| Prefill MoE fp8-CB alt | 'stock': CUDA-expand chunks → vLLM fused-MoE Triton kernel | fallback + PRISMAQUANT_CB_PREFILL=stock |
| Prefill MoE fp4-CB | per-expert loop (one host sync) | DEFAULT until stock's bf16-expand variant is measured |
| Dispatch | M-branch-hoist opaque custom ops (layer registry) | DEFAULT |
| Mid-M 17–128 fp8-CB | CUTLASS fused decode-in-prologue, fp32 EVT scale epilogue | DEFAULT (promoted 2026-07-26: 1.40× in-niche; conf-KL-vs-teacher gate 0.00305/99.83% ON vs 0.00324/99.88% OFF at forced-96 microbatching — preserved; 4-ulp cutlass_scaled_mm agreement) |
| Large-M fused dense | persistent-N self-contained TC kernel | MEASURED NEGATIVE (2026-07-26): parity-green but 2–5.7× slower than expand+fork at 27B shapes; the CUDA expander shrank the dense expand tax to ~10%, killing the original opportunity. Kernel kept quarantined (PRISMAQUANT_ENABLE_PTC=1) as the schedule reference. |
| Large-M fused MoE | persistent/grouped decode-in-mainloop | ROADMAP — the fat target (expand ≈ 35% of Laguna MoE layer time; f ≈ 50%) |
| MoE prefill alts | stock-kernel (capture-safe) / batched | OPT-IN |
| w2 rowpack, damp sweep, … | measured negative | ARCHIVED behind env switches |

- Decode contract v2 (scale-epilogue hoist): MEASURED NULL on served 27B
  (10.10 vs 10.13 tok/s, quality-neutral) — decode is bandwidth-bound at
  per-byte parity, nothing for the hoist to recover. Default stays v1;
  v2 remains supported via PRISMAQUANT_CB_DECODE_CONTRACT=v2.
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

- CB rung costs: measured anchors + the **split-aware floored RD law**
  `D(k) = F + C·R(k)` per (Linear, family) — F is the measured infinite-k
  grid floor (fp8: per-channel-fp8 RTN render; fp4: two-tier E2M1 RTN
  render), and `R(k) = Σᵢ 2^(−2·bᵢ/dᵢ)` is the EXACT rate factor over the
  rung's ceil-first sub-splits (bᵢ bits over dᵢ dims), so the k%n_sub
  sawtooth lives in the regressor instead of the residual; C by linear
  least squares over the anchors. Fit chain on rejection: split-aware →
  smooth floor law `F + C·2^(−α·k)` → legacy log-linear — each proposal
  **holdout-gated**: any tensor whose holdout error misses the bar falls
  back to full per-rung measurement. The gate is the contract; the laws
  are only ever proposals under it. (Measured on the 27B full-menu run:
  3.2% of tensors fall back.)
