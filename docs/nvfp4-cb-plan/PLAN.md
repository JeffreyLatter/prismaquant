# NVFP4-CB — Master Implementation Plan (synthesis)

**Status: DRAFT awaiting Robert's review. Nothing implemented.**
Drafted 2026-07-15 by three Opus 4.8 planning agents (sections below), reviewed
and corrected by Fable (orchestrator). Read the three sections for depth:

- `phase0-measurement.md` — the four gating experiments (emulation-only)
- `format-pipeline.md` — on-disk layout, FormatSpec/allocator/exporter integration
- `serving-kernel.md` — fused-expand kernel + vLLM plugin

## What NVFP4-CB is (one paragraph)

A vector-quantized codebook format whose codewords are 8-dim vectors of FP4
(E2M1) codes with NVFP4-identical group-16 E4M3 scales — a decoded tile is
bit-compatible NVFP4 and feeds the existing CUTLASS FP4 tensor-core path. A
k-bit index per 8 weights gives `k/8 + 0.5` bpw (k=12..24 → 2.0..3.5 in 0.125
steps; plain NVFP4 is the degenerate k=4·8 member). It composes IQ-class
sub-4-bit compression, NVINT's lossless-expansion envelope (without either of
NVINT's fatal serving modes), and AURA's measured allocation over a
near-continuous rate ladder. Storage layout: 256-weight superblocks
(32 k-bit indices + 16 E4M3 scales = 4k+16 bytes, integer for every k).

## The one constraint all three sections converged on (read this first)

**Flat-table codebooks top out at k≈13–14, for two independent reasons:**
encode-side, exhaustive weighted search is O(2^k) and infeasible past ~16k
codewords (phase0 §walls); serve-side, the LUT is 2^k×4 bytes against GB10's
**measured 100 KB smem/SM** (k=13 → 32 KB comfortable; k=14 → 64 KB marginal;
k≥15 impossible). So the flat-table variant only covers **2.0–2.25 bpw**.
Every rung above that requires a **structured/computed codebook** (small stored
generator + sign/permutation decomposition — exactly how the IQ family scales —
or QTIP-style computed codewords). Design consequence: the structured codebook
is the *default* carrier of the family; flat learned tables are a low-k special
case. This must be reflected in the format design before any kernel work.

## Decision points for Robert (the pause)

1. **Mixed container from day 1?** My recommendation: **yes** — the plugin
   delegates plain NVFP4/FP8 layers to vLLM's stock compressed-tensors schemes
   (serving-kernel §2), so mixing costs little, and a CB-only artifact would
   violate the "FP8 in every recipe" thesis. format-pipeline flags CB-only as
   the simpler Phase 1; I recommend overruling that.
2. **Learned-codebook scope.** Phase 0 measures fixed-lattice vs learned at
   k∈{12,13,14} only. The per-tensor sidecar penalty is a deterministic
   function of tensor size (crippling at 0.6B scale, ~negligible at 27B+ — see
   the review annotation in phase0 §walls), so the verdict must be reported as
   a curve over tensor size, not a single kill. Shared per-(model,role)
   codebooks are the fallback if learned wins on quality but loses on bytes.
3. **Phase-0 budget approval:** ~25–35 GPU-hours + ~600 LoC of emulation
   harness, ~1 week elapsed, zero kernel work. Its gates can kill the whole
   family before serious investment.
4. **Kill/advance gates as drafted** (phase0 §gates): family dies if CB loses
   IQ by >~15% KL at matched bytes on both models; entropy coding closed unless
   >0.25 bpw recoverable; fine ladder ships only if it beats coarse by more
   than between-seed noise; Fisher weighting promoted only if it beats imatrix
   beyond noise on both models.
5. **Weighting A/B doubles as a GGUF-lane lever** (exp 4): Fisher per-column
   weights are a llama.cpp-impossible upgrade that may close our known ~16%
   imatrix-arm gap — shippable win even if NVFP4-CB dies.

## Roadmap (strictly gated)

- **Phase 0** — emulation measurement ladder (phase0-measurement.md).
  Harnesses → exp1 (can kill family) → exp2/exp4/exp3. ~1 week.
- **Milestone A** — emulation-only integration: formats module, 13 FormatSpecs,
  allocator menu, cost path, KL-in-emulation on 0.6B/4B. No exporter, no
  kernel. (format-pipeline §8, ~650 LoC + tests.)
- **Milestone B** — byte packers + sibling exporter `export_nvfp4_cb.py` +
  pipeline gates + sidecar-aware footprint. Bit-exact pack==emulation pinned.
- **Milestone C** — serving (serving-kernel §4): (i) Triton dequant-GEMV
  correct serve (~4–6 d) → **first served gold-metric KL/PPL verdict**;
  (ii) decode perf parity + CUDA-graph capture (~3–5 d); (iii) CUTLASS/CuTe
  fused-expand FP4-MMA prefill (~15–25 d, the hard one, required for
  production-eligibility); (iv) MoE grouped variant (~8–12 d).
- Promotion follows the standard ladder; nothing is production-eligible until
  the artifact wins/preserves on exact served vLLM KL + PPL at matched bytes
  AND the prefill kernel clears the perf gate (no Triton masquerade — INV-2).

## Review corrections applied by Fable (2026-07-15)

1. **GB10 smem is 100 KB/SM (99 KB opt-in), measured on the box** — the
   serving draft assumed datacenter-Blackwell 228 KB; flat-LUT ceiling
   tightened to k≤13 (k=14 marginal). (serving-kernel §1a)
2. **`tcgen05` does not exist on sm_121** (it is sm_100a datacenter); the FP4
   path is the sm_120/121 block-scaled `mma` family — first kernel task is
   disassembling what CUTLASS emits for the working sm_121 NVFP4 GEMM.
   (serving-kernel §1a, §4iii)
3. **Learned-sidecar kill-gate de-fanged for scale**: the 2^k·32/N penalty
   shrinks ~50× from 0.6B-class to 27B-class tensors; exp-1 must gate per
   deployment scale via the analytic bytes-vs-N curve. (phase0 §walls)
4. **Family-coherence gate verified warn-by-default** (allocator.py:1371-1389);
   nothing blocks the ladder, per-family bucketing is hygiene not blocker.
   (format-pipeline open-Q 2)
5. Cross-section conflict noted and resolved in decision #1 (mixed container).

All other file:line citations in the three sections were spot-checked and are
accurate (`nvfp4_activation_qdq_served` fr.py:880, `_grid_fields`
gguf_iq_formats.py:229, `--target-disk-gb` allocator.py:1049, g_trace scalar
aura_cost.py:518/650, col_weights gguf_formats.py:389, FORMAT_SCHEME
export_native_compressed.py:6768, nvfp4_fused.py:250 bf16-MMA anti-pattern).

## Implementation log (orchestrator state — Fable updates this)

- 2026-07-15: Robert authorized implementation. Topology: Fable orchestrates +
  reviews diffs, Opus 4.8 agents implement; Opus fixes its own errors on
  Fable's instruction; Fable takes over only after Opus fails. FP8-grid
  codebook family (FP8_CB_K36/40/44/48, per-channel fp32 scales, product-VQ)
  added to scope by Robert.
- Branch: claude/nvfp4-cb. Disk cleaned 11GB→108GB free (deleted re-downloadable
  Hy3 bench shards; published on HF).
- **Wave 1 (running):** Agent-1 nvfp4_cb_formats.py + registry/layer_config/
  cache-mechanism + tests (grid-generic FP4+FP8 VQ, product default, full≤k14,
  fixed Gaussian-kmeans lattice → data/nvfp4_cb_lattices.pt); Agent-2
  emu_forward_kl.py + nvfp4_cb_footprint.py + index_entropy.py + tests
  (two-pass buffered-logits KL, per-format served-faithful act emulation);
  Agent-3 fisher_col_weights.py + additive aura_cost per-column harvest
  (default-off, bit-identical regression pinned) + tests.
- Wave 2 (queued): review diffs → fix cycles → run exp-1 (fixed vs learned vs
  IQ, 0.6B then 4B) → exp-2/4/3 per phase0-measurement.md gates.
- 2026-07-15 (Robert): exp-1 gains a SMOOTHING SUB-ARM — joint SmoothQuant-style
  α grid × k∈{12,14} on 0.6B, reusing the existing joint (α,format) machinery
  (May landing; keep α conservative — its recorded cascade bug at aggressive α
  stands). Requirements: (a) col_weights recomputed in lockstep under the
  smoothed distribution (E[x'^2]=E[x^2]/s^2, analytic); (b) gate on whole-model
  emulated KL, never the per-Linear screen; (c) measure jointly with
  fixed-vs-learned (smoothing and learned codebooks are partial substitutes —
  sequential measurement would misattribute the win); (d) expect the optimal α
  direction to be rate-dependent (helps A4 side at ≥4bpw; may invert toward
  AWQ-direction at 2bpw). Rotation (fused activation-side Hadamard) noted as
  roadmap follow-on now that CB layers ship on our kernel anyway — OUT of
  current scope.
- Wave-1 review status: Agent-3 fisher_col_weights (7d63e2e) ACCEPTED no fixes
  (clean additive aura_cost harvest, sum==h_trace pinned, compositions correct).
  Agent-2 harnesses (6b5fca0) — 3 fixes requested (act-hook fail-fast,
  unmatched-qname gate, FP8_CB footprint family/entry-bytes), fix cycle running.
  Agent-1 formats core still implementing. Fable fixed the 3 pre-existing doc
  tests (README rewrite fallout + 5 unindexed flags) directly.
