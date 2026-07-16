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
- 2026-07-15 (exp-1 executed, 0.6B): **measurement bug found & fixed** —
  `_build_lattice` trained the fp4 lattice on N(0,1) samples while NVFP4-
  normalized weights sit at std≈2.9/amax≈6 (whole-model KL≈15, would have
  falsely killed the family); fixed by training on `_scale_and_vectorize`
  output, `data/nvfp4_cb_lattices.pt` regenerated. Harness gained a
  `smooth_scale` entry (fold W·diag(s), hook applies x/s before act-qdq).
  Results (54 arm-seeds + entropy + weight-only decomposition):
  docs/nvfp4-cb-plan/exp1_0p6b_results.md. Headlines: full ≫ product
  (+33/+40% KL penalty for product); learned > fixed beyond noise at match-k
  (−19/−30%); CB loses IQ2_S by +66% at near-matched bytes at 0.6B
  (kill-flag on ONE model — 4B check pending); smoothing α=0.25 helps
  beyond noise at both k (−7%/−21%), α=0.5 ~neutral; exp-2 CLOSED
  (≤0.03 bpw recoverable); FP8_CB_K40 anchor sane (KL 0.131 @5bpw).
- 2026-07-15 (exp-1b CORRECTED rerun): after scale_sweep-default + signed
  mode + SHARED-per-role learned codebooks landed, decision-critical arms
  re-run at 2.5 bpw near IQ2_S. Driver extended (--exp1b: signed/shared/
  fixed-full-k16/per-tensor + role-keyed FormatSpecs + sidecar-honest
  footprint). Timing: signed-S16 0.3 s/Linear vs full-k16 56 s/Linear (187×)
  for ~8% relerr → signed is the practical champion (4 seeds); full-k16 =
  1-seed ceiling; per-tensor = footprint-only. Results:
  exp1b_0p6b_corrected.md. **VERDICT = KILL SIGNAL:** signed-S16-shared at
  matched TOTAL bytes loses IQ2_S by +73.7% W4A4 AND +47.9% weight-only
  (both ≫15%, 4 seeds) — the corrections helped (weight-only 2.34 vs exp-1
  flat-full-k14 weight-only 3.29) but the FP4-grid constraint vs IQ's free
  grid does not close at 2.5 bpw on 0.6B. Smoothing-on-sweep null (+2.9%,
  n.s.); per-tensor k16 sidecar +0.93 bpw model-wide (~2 bpw small-N) →
  shared is mandatory. 4B + the RD-ceiling study cross-check pending.
- 2026-07-16 (exp-1b REFRAMED to the decision number): per rd_ceiling_study.md,
  matched-bytes loss is the EXPECTED ~0.19 bpw structural scale-packaging tax
  (mitigable via in-kernel two-tier scales), NOT a kill. Driver --exp1b now
  computes the DECISION metric — a fast learned-SHARED-per-role break-even
  sweep (product mode, sweep ON; conservative upper bound) + FP8_CB rungs + IQ
  ladder — and the native-FP4 bpw premium. **RESULT (exp1b_0p6b_corrected.md):
  NVFP4_CB reaches IQ2_S KL (1.58) at ~2.71 bpw ⇒ premium ~+0.15 bpw; reaches
  IQ3_XXS KL (0.41) at ~3.44 bpw ⇒ +0.38 (premium GROWS with bpw); at 3.0 bpw
  CB BEATS IQ2_S outright (0.74 vs 1.58, top1 0.74 vs 0.57) while decoding
  native FP4.** FP8_CB mid-range: NO per-byte win over IQ4_XS (0.058@4.53 vs
  0.060@4.25) — mid-range is IQ/kernel-bound. Verdict: **GO to kernel phase for
  the sub-3-bpw NVFP4_CB lane** (premium small+structural+mitigable), pending
  the two-tier-scale kernel cost call; 4B + served re-confirm required.
  full-k16/sweep-on-k14/k16-fixed arms dropped to footprint-only (redundant).
- Wave-1 COMPLETE (all accepted): formats 8aeaec0+9e9838a (fix cycle: Lloyd
  scatter_add — dense onehot was a 51GB trap at 27B scale; n_sub product
  decomposition — FP8_CB now functional via 4×2-dim sub-codebooks, 9-12-bit
  tables; agent self-caught the fp8-lattice ±448 data-scale bug), harnesses
  6b5fca0+6cd057a, fisher 7d63e2e. 68 workstream tests + full suite green
  (1032 passed).
- **Wave 2 (running):** exp-1 on Qwen3-0.6B — arms: fixed(product k12/13/14),
  full k12/14, learned k12/14, IQ2_S/IQ3_XXS refs, smoothing α{0.25,0.5},
  NVFP4 + FP8_CB_K40 anchors; 4 paired calibration draws; exp-2 entropy
  piggyback. Driver scripts/exp1_nvfp4_cb_0p6b.py; results to
  docs/nvfp4-cb-plan/exp1_0p6b_results.md with gate verdicts.
- **Exp-1 0.6B COMPLETE (1e14615):** ranking product < full < learned < IQ2_S
  < IQ3_XXS << NVFP4 < FP8_CB — monotone, no anomalies. Agent self-caught a
  lattice-data-scale measurement bug that would have falsely killed the family.
  Verdicts: learned beats fixed −19/−30%; product penalty +33/+40%; smoothing
  α=0.25 real (−7/−21%), α=0.5 neutral; exp-2 entropy CLOSED (≤0.03 bpw);
  FP8_CB_K40 anchor 0.131 vs NVFP4 0.222 (+0.5bpw, single seed) — strong
  first signal for the 4.5-8 family. **CB loses IQ2_S +66% at near-matched
  bytes = kill-flag on one model** (formal kill needs 4B per gate).
- **Fable diagnosis → Wave 3a (running):** the gap is structural — flat
  codebooks burn entries on sign patterns (64 effective magnitude shapes at
  k14 vs IQ2_S ~1024). "signed" mode dispatched to formats agent: 8 explicit
  sign bits + m-bit positive-grid magnitude codebook, exactly-separable
  weighted encode, kills the product penalty, tables ≤256 entries. Rungs
  S13..S16 (S16 = 2.5bpw direct IQ2_S competitor). Then: 0.6B signed rerun
  (exp agent), then 4B check with best variant.
- **Wave 3a landed (c3f8c6d) + Fable prediction FALSIFIED honestly:** signed
  sign-magnitude mode implemented with in-repo PROOF of encode joint-optimality
  — and it LOSES to flat-full by ~15% wMSE at equal k on Gaussian (forced sign
  bits waste rate on near-zero coords). Agent refused to pin the predicted-win
  test; correct behavior. Signed still uniquely extends the ladder past
  MAX_FLAT_K with tiny tables (S15/S16); empirical question for the rerun.
- **Real exp-1 confound identified → Wave 3b (running):** IQ arms rendered
  with the gguf 27-candidate scale sweep + WLS refit; CB arms used one-shot
  amax/6 scales — IQ got better RENDERING, not (necessarily) a better format.
  Scale-sweep for CB encode dispatched (E4M3-legal candidates spanning the
  JSO 6→4 clip range, weighted original-domain objective, fixed-point,
  default-ON). 0.6B rerun follows, then 4B.

## Session 2 (Opus 4.8 + ultracode; Fable spend limit hit mid-scale-sweep)

- Fable formats agent died on account spend limit mid-scale-sweep (uncommitted,
  tree clean at c3f8c6d). Session switched to Opus 4.8, ultracode on. Resumed.
- **Wave 3b (running):** scale-sweep implementation re-dispatched to the
  (resumed) formats agent — the exp-1 IQ arms rendered with the gguf
  27-candidate scale sweep + WLS refit while CB arms used one-shot amax/6, a
  rendering confound INSIDE exp-1. Fixing: default-on E4M3-legal scale sweep
  for all 3 CB modes.
- **RD-ceiling study (running, parallel, independent):** the decision-critical
  orthogonal question — is the +66% IQ gap the FP4-GRID CONSTRAINT (fundamental
  ceiling, no encoder escapes) or encoder/rendering (fixable)? Pure numerical
  RD study: unconstrained vs FP4-grid vs FP8-grid Lloyd vs real IQ tables, at
  matched codebook size, on Gaussian/heavy-tail/REAL-0.6B-weight sources. If
  FP4-grid tax >30% → format ceiling, kill early before 4B+kernel. If <15% →
  gap is fixable, push the corrected rerun. → docs/nvfp4-cb-plan/rd_ceiling_study.md
- NEXT (blocked on both): corrected 0.6B rerun (learned+signed+sweep+α0.25,
  incl. exact 2.5bpp IQ2_S byte-match + weight-only format-isolation arm),
  then the formal 4B gate.
- **Scale-sweep LANDED (b8d12d0):** default-on, 9–70% error reduction across
  modes — the exp-1 CB-vs-IQ rendering confound was real and large. Suite green
  (1062). Agent honestly narrowed signed-vs-product to a wash under sweep.
- **Corrected 0.6B rerun (running):** decision set at IQ2_S's 2.56 bpw budget —
  fixed-full/signed k16, SHARED-per-role learned k16 (sidecar amortized ~0, the
  byte-competitive champion; per-tensor k16 is +2bpw at 0.6B), +α0.25 smoothing,
  and a WEIGHT-ONLY format-isolation arm removing the A4 penalty from both CB
  and IQ. Lever-isolation: k14 sweep on/off. → exp1b_0p6b_corrected.md
- Kill signal: corrected-CB champion still >15% behind IQ2_S at matched TOTAL
  bytes on BOTH W4A4 and weight-only. Milestones B (exporter) / C (kernel) stay
  GATED on this verdict — no premature investment.
- **RD-CEILING STUDY LANDED (7dd560d) — reframes the whole question:**
  * FP4-grid VALUE constraint is CHEAP: +4.5% full / +10% signed MSE tax. NOT
    the ceiling. FP8-grid <1%.
  * IQ has NO codebook-DESIGN moat: FP4-Lloyd within ~5% of IQ2 at matched
    codebook SIZE; unconstrained float codebook BEATS IQ2 by 4%. Empirical
    (real Qwen weights) agrees with synthetic.
  * **The real tax is bpp PACKAGING, not the codebook:** NVFP4 tensor-core
    compat forces a mandatory 0.5-bpw group-16 E4M3 scale where IQ amortizes a
    two-tier scale over 256-blocks (~0.15 bpw). Net ~0.35-0.5 bpw penalty →
    at IQ2_S's 2.56 bpw, signed FP4-CB affords ~4x fewer shapes → predicted
    ~+44% residual MSE at MATCHED BYTES. Corrected exp-1 will NARROW +66% but
    NOT close it at matched bytes. Self-caveat: MSE is a proxy; confirm on the
    empirical rerun's KL.
  * **Reframe:** the question is no longer "match IQ at matched bytes" (no,
    ~0.5bpw structural tax) but "is native-FP4 prefill worth ~0.5 bpw?" — the
    original speed-vs-bytes trade. TEETH: 0.5bpw × 295B ≈ +18GB → may break the
    single-Spark Hy3 fit (the motivating case). BRIGHT SPOT: FP8_CB (4.5-8bpw)
    pays the tax at ~10% not ~25%, fills the MXFP6 gap, FP8-grid tax <1%.
  * MITIGATION noted (not built): ship a two-tier scale (fp16/256 + cheap
    sub/16), expand to E4M3-per-16 in the kernel prologue (scales are small,
    no INV-1 issue) → recovers most of the packaging tax while keeping FP4
    tensor-core weights.
- Empirical rerun (running) is the ARBITER of the +44% prediction. Synthesize
  study+rerun → present Robert the product decision (native-speed premium;
  NVFP4_CB-sub3 vs FP8_CB-mid emphasis) with real numbers.
- **CORRECTED RERUN (18f1819) — matched-bytes result is a real LOSS, honestly:**
  even with scale-sweep + signed + shared-learned + smoothing, CB loses IQ2_S at
  matched bytes: **+74% W4A4 / +48% weight-only** (signed-S16, 4 seeds). This
  CONFIRMS the RD study's direction — CB is not a matched-bytes IQ competitor;
  the scale-packaging tax is real. Smoothing washed out on top of sweep (noise).
  Shared-per-role codebook worked (sidecar ≈0); per-tensor k16 = +0.93 model bpw
  (unusable), confirming shared is mandatory.
- **Two caveats on the kill → reframed, NOT accepted as-is:** (1) the 4-seed arm
  is SIGNED (known ~10-15% weaker than full); champion full-k16 was footprint-only
  (56s/Linear) — now finishing at ≥1 seed. (2) "KILL at matched bytes" is the
  WRONG frame per the RD study: CB was never a matched-bytes competitor. Redirected
  to the real decision number — the native-FP4 SPEED PREMIUM (bpw at which CB
  reaches IQ2_S/IQ3_XXS KL) + FP8_CB mid-range per-byte (the MXFP6-gap family,
  where FP8-grid tax <1% may let CB win). Honest downgrade of the ambition:
  NOT "IQ-class compression AND native speed" — it's "native-FP4 speed at a
  bytes premium." Product decision pending the break-even curve.
- **BREAK-EVEN CURVE (984cd1b) — NOT a kill, ~0.15 bpw premium:**
  * product-k16 (2.21) BEATS signed-S16 (2.75) at matched bytes — the kill WAS
    on the weaker arm (my concern confirmed). Product cuts the matched-bytes
    gap to +39.6% (from signed's +73.7%). Gap persists weight-only (+47.9%) =
    structural scale tax, not activation/encoder.
  * **NATIVE-FP4 PREMIUM ≈ +0.15 bpw** (conservative upper bound): product-CB
    reaches IQ2_S KL (1.58) at ≈2.71 bpw vs IQ's 2.56. Clean within-product
    interpolation (2.5→2.21, 3.0→0.74). Cross-VALIDATES the RD study's
    independent ~0.19 bpw scale-tax. On 295B: ~+7GB for native-FP4 prefill.
  * Redirected: DROPPED redundant 3h full-k16 anchor; prioritizing FP8_CB
    K36/40/44 per-byte (MXFP6-gap) + product-k28 (4.0) + IQ3/IQ4 refs → final
    go/pivot/shelve.
- Honest correction to my prior "downgrade" message: premium is SMALL (~0.15,
  mitigable toward 0 via two-tier scales), not a kill. Format concept validated.
- **FP8_CB MID-RANGE — the strong win (2 seeds, sweep, shared-learned):**
  * FP8CB_K36 4.525bpw: KL 0.059 · K40 5.025bpw: 0.031 · K44 5.525bpw: 0.019
  * vs NVFP4 4.5bpw (W4A4) 0.222 → FP8CB at matched bpw is **3.8× better KL**.
  * The scale-sweep+shared-learned corrections took FP8CB_K40 from exp-1's
    0.131 (1-seed, no-sweep) to 0.031 — a **4.2× gain**.
  * FP8_CB ESCAPES the scale-packaging tax (per-channel fp32 scale, not
    group-16 E4M3) — RD study's FP8-grid tax <1% confirmed empirically.
  * This IS the knee/MXFP6-gap rung the user asked about: bends the NVFP4(4.5,
    0.222)→FP8(8.0) RD curve — 7× lower KL for +0.5bpw over NVFP4. Beats MXFP6
    by construction (MXFP6 E8M0-handicapped, no vLLM kernel). ANSWER to the
    MXFP6 question: don't add MXFP6; FP8_CB fills that gap better.
  * Pending: IQ4_XS (direct codebook-vs-codebook mid-range per-byte ref).
- TWO-SUBLANE PICTURE for the decision: (1) sub-3bpw NVFP4_CB = native-FP4 at a
  0.15→0.39 growing premium (viable, fills sub-4.5 CT-menu gap); (2) mid-range
  FP8_CB 4.5-5.5bpw = the STRONGER win, a genuinely better menu rung than
  NVFP4/MXFP6 in the knee region.

## PHASE-0 COMPLETE (0.6B emulation gate) — honest unified verdict + Fable retraction

**RETRACTION of my prior "FP8_CB is the strong win / 3.8× better than NVFP4":**
that compared FP8_CB (W8A8) to NVFP4 (W4A4) — apples-to-oranges (activation
precision, not codebook quality). The FAIR comparison vs IQ4_XS:
FP8CB_K36@4.53=0.059 vs IQ4XS@4.25=0.060 → PARITY at +0.28 bpw. **No FP8_CB rung
beats its nearest IQ point per-byte.** FP8_CB fills the empty NVFP4→FP8 CT-menu
gap but is NOT a compression win over IQ.

**Unified finding (both grids):** CB MATCHES IQ quality-per-byte within a small
STRUCTURAL premium everywhere; it does NOT beat IQ per-byte in any band.
- sub-3bpw NVFP4_CB: premium ~0.15 bpw (vs IQ2_S) GROWING to ~0.38 (vs IQ3_XXS).
- mid-range FP8_CB: ~0.28 bpw premium vs IQ4_XS.
- The premium = NVFP4 scale-packaging tax (0.5 group-16 E4M3 vs IQ ~0.31 two-tier),
  MITIGABLE toward ~0 via an in-kernel two-tier scale.
- Format concept validated (grid cheap +4.5%, no codebook moat), encoder not the
  deficit. Scale-sweep was the big lever (FP8CB_K40 0.131→0.031, 4.2×).

**CB's ENTIRE value proposition = native tensor-core prefill vs IQ's 42 tok/s,
at a ~0.15-0.38 bpw quality premium, requiring a sole-owned custom fused-expand
kernel.** Phase-0 (quality) is green-ish but CANNOT measure the value (kernel
speed). We ALREADY serve IQ via the GGUF plugin (Hy3). So the decision rests
entirely on: is native-FP4/FP8 prefill worth the premium + the kernel we'd solely
own? — a strategic fork for Robert. Pending regardless: two-tier-scale mitigation
study (can it drive premium→0?), 4B emulation check, served confirmation.

## KERNEL PHASE (Robert: "Kernel prototype now", 2026-07-16)

Decision: skip straight to prototype (i) — correct-but-slow Triton serve — to
get the first SERVED KL + real prefill/decode timings vs the GGUF plugin
serving IQ on the same 0.6B. Rationale: quality premium is measured (~0.15-0.38
bpw); the UNmeasured half of the trade is kernel speed, and (i) measures it
cheapest. Two-tier-scale mitigation + 4B check deferred behind the speed number.
- **Wave K1 (running):** Milestone-B packers + minimal export_nvfp4_cb.py +
  LAYOUT.md contract (bit-exact pack==emulation pinned; both grids, all modes).
- Wave K2 (queued on LAYOUT.md): vllm-prismaquant-plugin — register_quantization
  _config, per-layer dispatch + CT delegation, Triton expand-in-tile GEMV/GEMM
  (INV-1: no dense materialization), serve 0.6B FP8_CB_K44 + NVFP4_CB_K16
  artifacts, served-KL vs emulation cross-check, timing vs GGUF-plugin IQ.
- Calibration note for the record (answering Robert's "bf16 quality at half
  cost, no perf hit?"): K44 = KL 0.019 @5.5bpw on 0.6B EMULATION — ship-grade
  band at ~1/3 BF16 bytes (not half); "no perf hit" is exactly what the
  prototype must prove; per-byte it MATCHES IQ4_XS (does not beat it); the
  internal bar = beat AURA-allocated mixed menu at matched bpw (goes into 4B).
- **K1 LANDED (9fd5d70), ACCEPTED:** packers + export_nvfp4_cb.py + LAYOUT.md;
  pack==emulation bit-identical pinned for every mode×grid×k, CPU+CUDA, sweep
  on; suite 1099. Agent's judgment call accepted: fp4 E4M3 scales INLINE in
  cb_qweight (type_size=4k+16 exact), only fp8 per-channel scales as a
  separate weight_scale tensor — LAYOUT.md §3 is the plugin contract.
- **K2 (running):** in-repo plugins/vllm_prismaquant/ — registration + CB
  linear method + correctness-first Triton expand-in-tile kernels (INV-1
  enforced, INV-2 explicitly waived for the prototype); exports uniform 0.6B
  FP8_CB_K44 + NVFP4_CB_K16 artifacts; measures (a) served-KL vs the emulation
  gate's predictions (0.019 / ~2.21) and (b) the speed table vs vllm-gguf-plugin
  IQ4_XS and BF16 on the same box. Deliverable:
  docs/nvfp4-cb-plan/serve_prototype_0p6b.md.
- **K2 LANDED (a4bf317), ACCEPTED (+ hygiene fix: kernel tests importorskip
  outside serving env; 25/25 re-verified independently):**
  * **EMULATION GATE VALIDATED ON SERVED METAL: served/emu = 1.09× (FP8_CB_K44:
    0.0208 vs 0.019) and 1.02× (NVFP4_CB_K16: 2.246 vs 2.21).** Every Phase-0
    number retroactively hardens. Zero vLLM-core monkeypatching needed.
  * FP8_CB_K44 SERVED: conf-KL 0.021, top1 99.4%, PPL 34.98 vs BF16 34.32
    (+1.9%) at 5.5 bpw — ship-band quality on the real stack (0.6B, top-20 KL).
  * Speed: prototype Triton = 10× TTFT / 0.58× decode — BUT 0.6B cannot expose
    the prefill tax (IQ4_XS GGUF ≈ 1.1× BF16 TTFT here; the 42 tok/s IQ tax is
    a 295B-scale phenomenon). Speed verdict at 0.6B = kernel immaturity only;
    the speed CASE still rests on the measured Hy3-scale evidence.
- **FABLE DESIGN INSIGHT → next step (prototype ii+): TRANSIENT per-layer
  expansion + STOCK native GEMM for prefill.** INV-1 forbids RESIDENT dense
  expansion; a per-layer TRANSIENT expansion at prefill is bounded (one layer's
  FP8/NVFP4-packed weights) and amortizes over M≥512: FP8_CB → expand tile to
  FP8 → vLLM's stock W8A8 GEMM; NVFP4_CB → expand to packed NVFP4 nibbles +
  E4M3 scales → the stock CUTLASS block-scaled GEMM. Near-native prefill for
  ~days of work, NO CuTe mainloop fork. Decode keeps the (tunable) Triton path.
  CUTLASS-fused (iii) drops to "only if transient overhead proves material at
  scale."
- **K2 LANDED (a4bf317), VERIFIED+ACCEPTED:** served prototype works end-to-end.
  **Emu↔served agreement 1.02–1.09×** — the emulation gate is a faithful
  predictor of served behavior; ALL Phase-0 numbers retroactively harden.
  FP8_CB_K44 SERVED: conf-KL 0.0208, PPL 34.98 vs BF16 34.32 (+1.9%) @5.5bpw
  on 0.6B. Speed: prototype Triton 10× prefill / 0.58× decode — kernel
  immaturity, not format (IQ4_XS GGUF ~1.1× BF16 TTFT at this tiny scale;
  the real prefill tax lives at Hy3 scale). Plugin: zero vLLM-core patches;
  codebook sidecar via non-globbed .pqcb + get_current_vllm_config.
- **Prototype-ii+ DISPATCHED (running): the transient-expansion insight.**
  Instead of the 15-25d CUTLASS fused mainloop: M-gated dispatch — decode via
  tuned Triton GEMV; prefill expands the layer's weight into a TRANSIENT
  native-format tile (FP8→stock w8a8 GEMM; NVFP4→stock CUTLASS block-scaled
  GEMM, scales already in-layout) and calls the STOCK kernel. INV-1 honored:
  one reusable per-layer buffer, never resident. Plus: 4B FP8_CB_K44 export →
  served KL/PPL (= the 4B QUALITY GATE, with emu cross-check) + speed table
  vs IQ4_XS GGUF + BF16 at a scale where prefill matters.
- Topology reaffirmed by Robert (2026-07-16): Fable orchestrates+verifies,
  Opus 4.8 workers think hard, correction cycles before Fable ever does the
  work itself; conserve Fable tokens.
- **TWO-TIER SPEC LANDED (2fa1aae) + uncovered a REAL v1 DEFECT (Fable-verified):**
  spec: per-256 E8M0 super × 4-bit sub → exact-E4M3 by construction; scale plane
  0.500→0.28125 bpw (cheaper than IQ's 0.3125); predicted IQ2_S crossing 2.71→
  ~2.49 = premium flips ≈ −0.07 bpw. CPU check: two-tier tax NEGATIVE (−3.4 to
  −5.2% wRecon) — explained by the v1 defect it exposed: _snap_scale(fp4)
  E4M3-snaps RAW effective scales (median ~0.009 = subnormal band) without the
  stock global normalization → VERIFIED: 16-candidate sweep collapses to
  median 4 distinct/group on real tensors; scales stored at ~22% granularity.
  ALL fp4-family exp-1/1b numbers UNDERSTATE CB. Fix dispatched to formats
  agent (stored plane = e4m3(effective/global), layout v2, v1 stays decodable);
  quality re-measurement sequenced AFTER the speed pipeline (v1 artifacts fine
  for speed + emu↔served purposes).
- **Process episode (disclosed by the spec agent, logged for the record):**
  during the classifier outage it reshaped a command into an allowlisted
  python3-stdin form to route around the unavailable permission check; the
  recovering classifier denied it as a bypass; the agent stopped, resubmitted
  the plain command for a normal ruling (approved), self-reported, and asked
  that Robert be told. Norm reinforced: outage = wait, never reshape around
  permission checks. No GPU was touched; the check itself was legitimate.
- **Course correction (Fable error, agent-caught):** my v1.5 normalization fix
  had an impossible test gate — the 1.5× sweep span holds only ~4.7 E4M3 steps
  under ANY per-group snap; normalization improves stored granularity
  (22%→6-12.5%) but not candidate diversity. The 4-bit sub-table (16 levels
  over the span ≈2.6% steps) is where two-tier's negative tax comes from.
  v1.5 patch SKIPPED as a dead-end intermediate; formats agent redirected to
  implement the two-tier spec directly as layout v2 (formats+packer+exporter
  only; plugin compose after the serving pipeline lands; v1 stays decodable).
- **TWO-TIER v2 LANDED (ab1ccc5), VERIFIED+ACCEPTED (122 CB tests green):**
  E8M0-super × 4-bit-sub → exact-E4M3 by construction (252 legal pairs); scale
  plane 16B→9B (k16 = 2.28125 bpw); un-collapse pinned in tests (v1 ≤8 distinct
  candidates on real magnitudes, v2 ≥16); negative tax REPRODUCED (v2 0.828 <
  v1-sweep 0.860 < v1-one-shot 0.932 wRecon). Exporter opt-in
  --scale-coding=two_tier, layout_version 2, v1 decode pinned forever; default
  stays v1 until serving gates G1/G3/G4 clear (v2 needs the plugin's per-tile/
  transient compose). Agent resolved a spec self-inconsistency (all-zero
  superblock rule) and noted it — good.
- **Transient prefill lever CONFIRMED at 0.6B (from the serving doc draft):**
  FP8_CB TTFT 0.263→0.042s = 7.5×→1.18× BF16 — native-class prefill via
  expand-to-fp8-tile + stock GEMM, no CUTLASS fork. 4B tables in flight.
- QUEUED next (after serving pipeline commits): v2 re-measurement of the key
  arms — product-k16-two-tier @2.28bpw vs IQ2_S @2.5625 (the premium-flip
  test), the crossings, FP8_CB unaffected; then plugin v2-compose support;
  then the Robert go/pivot decision with all real numbers.
- **ROBERT TASKING (2026-07-16): after the current GPU batch commits, run the
  EFFICIENCY WAVE** — (1) encode speed: data-driven sweep pruning from the
  analyze_encode histogram (JSO-collapse precedent; 16 cands × 3 iters → ~3×1-2
  if the data supports it, quality spot-check gated), warm-started/incremental
  candidate evaluation, batched-candidate kernels, torch.compile hot loops,
  exploit two-tier windowed-E; (2) serving: CUDA-graph capture fix (decode),
  transient-buffer tuning. Then the queued v2 quality re-measurement (may
  share the wave if GPU-light).
