# PrismaQuant — context for /ultraplan session 2026-05-09

## TL;DR

We just resolved the HALO smoke on Qwen3.5-0.8B. We found two HALO bugs (latent
since the original module, surfaced by the 2026-05-09 d5626e2 enable-on-qwen3.5-dense
commit), fixed them on branch `fix/halo-qwen35-rmsnorm-fold` (commits 5898583
+ 22c6c65, not pushed), verified end-to-end, and made HALO opt-in / EXPERIMENTAL
in the CLI help. Now we want to plan the next phase of PrismaQuant additions.

The strategic question is which improvements actually move the bpp/KL Pareto on
the production stack and are worth the engineering investment, and which are
research code paths that don't ship.

## Repository state

- Branch: `fix/halo-qwen35-rmsnorm-fold` (local, not pushed). 2 commits ahead of
  `0a92c3d docs: add claude handover` on `main`.
- Shipped 27B artifact: `dq-runs/qwen36-27b-step00-production-cache-vllmlegal-export-main-20260509T073125Z/exported`.
  KL=0.031 / 5.16 bpp / Qwen3.6-27B / vLLM-load verified. **Does not use HALO.**
- Allocator stack today: per-Linear knapsack/Pareto + GPTQ (OBS rounding) +
  closed-form scale_sweep + activation_clip + sibling-globals unification +
  block-output-match refinement (NVFP4_block_match) + mixed-format allocation.
- Pre-existing cross-layer machinery in repo: PrismaSCOUT (L3 redesign),
  PrismaCLADE / block-CLADO, output-Fisher refinement, polish (iterate-perturbed-
  allocation), KL sensitivity probe, dense cone, QUBO refinements. Most
  exists; few have demonstrated reproducible non-regressive wins on the
  production stack at modern scale.

## Today's HALO smoke result (so it's grounded)

Setup: untied lm_head Qwen3.5-0.8B (HALO requires untied; cloned embed →
lm_head shard, 485 MiB, hardlinked the rest). Layer config: 195 entries
(150 NVFP4 body + 45 BF16 covering lm_head, linear_attn.in_proj_a/b, MTP).
RTN-only (no production cache, no GPTQ, no scale_sweep — pure infra smoke).

Outcome before fix:
- Both no-HALO and HALO exports completed. Both vLLM-loaded in eager+graph.
- No-HALO: coherent generation (' Paris, and the capital of the United
  States is Washington, D. C.').
- HALO: newline spam (`'\n\n\n…'`, max_tokens of newlines).

Diagnosis (via BF16-only HALO sanity test): 0% argmax agreement, |logit
diff| max 31.1. Math/code bug, not RTN compounding.

Root cause: TWO bugs.

1. **Gamma-fold bug.** `prismaquant/halo.py:fold_gamma_into_linears` assumed
   Llama-style RMSNorm (`weight * normalize(x)`). Qwen3_5RMSNorm computes
   `(1 + weight) * normalize(x)` (centered residual, weight init zeros).
   The fold absorbed `weight` instead of `(1 + weight)`, then reset
   `norm.weight = 1.0`, leaving the post-fold norm computing `(1+1) = 2x`.
   Mismultiplied by `(2*weight) / (1 + weight)` per fold site — for typical
   pretrained weights ≈ 5–10×. Compounded across 49 fold sites per Qwen3.5/3.6
   dense model.

2. **`default_block_specs` missing linear_attn.** The in-memory full-model
   spec builder didn't detect the linear_attention layers' linear_attn block;
   only `block_specs_for_layer` (the streaming-export per-layer path) did.
   So in BF16 sanity tests, 30 specs found instead of 48 — 18 linear_attn
   blocks missing means residual stream rotated but linear_attn weights
   unrotated → garbage from those layers regardless of the gamma fix.

Fix (commit 5898583):
- Added `_OFFSET_RESIDUAL_NORM_CLASSES = {"Qwen3_5RMSNorm"}` whitelist plus
  explicit `offset_residual: bool | None` override on `fold_gamma_into_linears`.
  When offset-residual: gamma = `1 + weight`, post-fold reset = `weight = 0`.
- Refactored `default_block_specs` to delegate to `block_specs_for_layer` per
  layer so both paths emit identical specs (48 not 30).
- 3 new tests: offset-residual fold preservation, class-name auto-detect,
  Llama-style regression guard. 16/16 halo + 81/81 halo+export pass.

Verification post-fix:
- BF16 sanity: 48 specs, |diff| max 0.375 on logits magnitude ~17, **96.9%
  argmax agreement**. (The 3.1% disagreement is BF16 cancellation noise
  through 24 layers + Hadamard + 248k vocab — expected at BF16 precision.)
- Quantized HALO eager: `' the capital of the United States.\nA. True\n…'`
- Quantized HALO graph: `' the capital of France.\n- The capital of France
  is the capital of France'`
- Coherent English in both modes. Quality at this 0.8B / RTN-only scale is
  similar to no-HALO; HALO's intended quality win shows up at lower bpp /
  with the production cache + GPTQ + scale_sweep stack — not here.

Documentation update (commit 22c6c65): `--halo-mode` help text now leads with
"EXPERIMENTAL, opt-in; default OFF" + notes that the literature ~0.20-0.30
PPL gain is from RTN baselines, not from the full PrismaQuant production
stack. Runtime warning prints when HALO is engaged.

## The strategic conversation we had

User asked the meta question: cross-layer interaction machinery (PrismaSCOUT,
CLADO, output-Fisher, polish-of-many) was a huge investment, but the per-
Linear allocator generates great results. Is cross-layer interaction simply
not as big a deal as one might expect?

My read on the empirical evidence:

- Per-Linear knapsack + GPTQ + scale_sweep + activation_clip is doing 90%+ of
  the work on Qwen3.6-27B. KL=0.031 at 5.16 bpp is competitive.
- Polish-of-1 reliably adds ~5% KL improvement on top (the shipped 27B step
  00 polish move). Polish-of-many is noise-dominated at n=8 calibration:
  between-seed variance ~40% of mean, train→val sign flip on most moves
  (May 7 audit `AUDIT_OVERFITTING_2026-05-07.md`).
- L3 / CLADO / output-Fisher haven't demonstrated reproducible non-
  regressive wins on the production stack.
- May 7 KL sensitivity probe on Qwen3-0.6B got -44% KL with **per-Linear
  sensitivity → multi-choice knapsack**. No cross-layer modeling.
- PrismaSCOUT v5's claimed -34% on Qwen3-4B at 4.5 bpp is unreplicated end-
  to-end (blocked on GB10 OOM, never made it to a shipped recipe).

Why per-Linear seems to work so well, plausibly:

1. NVFP4 group_size=16 already does local outlier diffusion.
2. GPTQ is implicitly cross-layer-aware: minimizes output MSE given the
   activation distribution at this Linear, which is a function of every
   upstream layer.
3. Modern LLMs are massively over-parameterized; quant errors damp through
   downstream redundancy.
4. Cross-layer signal is small relative to calibration noise. If per-Linear
   SNR is ~5 and cross-layer interaction SNR is ~0.5 at n=8, no clever DP
   recovers signal from below the noise floor.

Conclusion: the cross-layer infrastructure isn't wasted (it's how you proved
the null result, which is publication-grade), but extending it as a quality
lever is not where the next gains live. **Per-Linear improvements are
where to invest.**

## User-imposed format constraint

vLLM-supported, weight/activation rate-matched. Final menu:

- NVFP4 (W4A4)
- MXFP8_E4M3 (W8A8 microscaling)
- MXFP8_E5M2 (W8A8 microscaling, larger dynamic range)
- FP8_E4M3 (W8A8 per-tensor or per-channel)
- FP8_E5M2 (W8A8 alternative)
- BF16 (W16A16 passthrough)

**Excluded**: NVINT2/3 (no W4A4 path), MXFP4 (4-bit activations not viable),
MXFP6 (no vLLM kernel), OWQ/SqueezeLLM-style mixed-precision per-weight
(needs custom kernels). No new kernels period.

The interesting per-Linear decisions in this menu:
- **NVFP4 vs MXFP8 vs FP8 vs BF16** — bpp tradeoff (allocator-level)
- **E4M3 vs E5M2** within MXFP8 and within FP8 — precision/range tradeoff
  (per-Linear, currently undifferentiated by the probe)
- **MXFP8 vs FP8** at the same nominal bpp — group structure tradeoff

The current KL sensitivity probe doesn't differentiate E4M3/E5M2 within a
format family. It should.

## Calibration data discussion

Today's default is presumably WikiText-2 or similar. Literature drift
suggests:

- Wikitext-2 is monoculture encyclopedic English; under-exercises code/math/
  multilingual outlier patterns.
- Modern good defaults: RedPajama-V2, FineWeb-Edu (curated, reproducible,
  diverse).
- C4/OpenWebText: diverse but noisy, license issues.
- The Pile: gated.

My suggested calibration mix (4-bucket diversity-weighted, fixed dataset
hashes for reproducibility):
- 40% prose (FineWeb-Edu)
- 20% code (StarCoder data subset)
- 20% math (proof-pile or GSM8K-train, or Lean/Coq subsets)
- 20% multilingual (FLORES or mC4 sample)
- n=256 sequences × 4096 tokens (vs n=128 × 2048 baseline)

WikiText-2 stays as a comparability column in any results table — not as
the working calibration set.

Single most testable claim: **diversity matters more than n_samples beyond
~64.** Worth measuring once on Qwen3.6-27B (KL probe on monoculture wikitext
vs diverse mix at same n) and looking at how many promotions diverge.

## Activation calibration discussion

Today's pipeline:
- `--activation-cache-dir` — per-Linear input cache during a probe forward
- `compute_nvfp4_input_global_scale(acts)` — max_abs/6.0 type scaling
- `PRISMAQUANT_ACT_CLIP_QUANTILE` — p99.9 default quantile clip
- Sibling-globals unification across fused groups (qkv, gate_up, in_proj_qkvz)

Improvements I'd consider (per-Linear, vLLM-compatible, no new kernels):

1. **Per-Linear quantile sweep.** Today's clip quantile is one value
   model-wide. Different Linears want different clips (early-layer attention
   wants higher, late-layer down_proj wants lower). Sweep on calibration,
   pick per-Linear, store in cache. Plausible 1-2% improvement.

2. **Asymmetric quantization for SiLU-output Linears.** `gate_proj` /
   `up_proj` outputs are post-SiLU positive-skewed. Symmetric NVFP4 wastes
   half the codebook. vLLM's `nv_fp4_with_static_gs` supports asymmetric
   variant — confirm it's wired and benchmarked.

3. **Production-faithful activation re-cache** (Codex's Step F applied to
   activations). Today's activation cache is built when most Linears are
   BF16 — the activation distribution at Linear L when upstream Linears
   are at NVFP4 is *different*. Re-cache after staging the production
   assignment, then re-fit `input_global_scale` per Linear from that state.
   **This is the single most underexploited correction in the pipeline.**
   Plausible 5-10% if the May 7 audit finding ("most calibration work is
   fishing in noise") points to upstream context shift as the dominant
   bottleneck.

4. **Length-distribution-aware calibration.** Most activation work today
   is at one sequence length. State-space / linear_attn / RoPE-position
   behaviors differ at long context. n=64 at 4096 + n=64 at 16384 may
   catch outliers that n=128 at 2048 misses.

5. **Per-token (dynamic) activation scaling for outlier-heavy Linears.**
   Already supported by NVFP4 path (`act_dynamic: true`). Audit that it's
   actually engaged at runtime for the Linears that need it.

## Numerical methods landscape (ParoQuant + alternatives)

User flagged ParoQuant as a candidate replacement for HALO. After
evaluation:

**ParoQuant** (Liang et al., ICLR 2026, arXiv:2511.10645):
- Pairwise rotations (Givens-style) instead of full Hadamard
- Trained per layer (not random) — adds calibration-time optimization step
- Group-aligned with NVFP4's group_size=16 — the big differentiator over
  HALO/QuaRot which scramble all channels uniformly
- Same vLLM compatibility model: absorb rotation into adjacent Linear weights
- Same prerequisites as HALO: untied lm_head, norm-style awareness (inherits
  our gamma-fold fix), profile-specific topology

**Worth considering alongside ParoQuant**:
- **SmoothQuant** — per-channel activation→weight migration (s such that
  Y = (X/s) @ (sW)). vLLM-compatible (absorb into upstream norm gamma).
  Untested on NVFP4+group=16, but simpler than rotation.
- **FlatQuant** (2024) — learned per-Linear affine optimized for "flatness"
  rather than incoherence. Beats QuaRot on Llama at W4A4.
- **OmniQuant** — jointly learnable clip + weight transform via SGD.
  Replaces scale_sweep, doesn't compose.
- **Per-Linear GPTQ damping sweep** — tune λ per Linear instead of global
  scalar. Simple, low-risk, ~1% improvement.
- **Fisher-weighted GPTQ loss** — weight by Fisher info instead of uniform
  output MSE. PrismaQuant has Fisher infrastructure already.

**Skip**:
- AWQ (codebase comment: +230% PPL on NVFP4 group=16, designed for W4A16
  per-channel only)
- OWQ / SqueezeLLM (per-weight mixed precision needs custom kernels)
- QuIP# / LDLQ (target sub-2-bit; not differentiated at 4-bit)
- AffineQuant (subset of FlatQuant)
- AdaRound true-SGD (closed-form approximation = our scale_sweep; diminishing
  returns)

## Per-Linear damping vs ParoQuant interaction (user asked)

Damping doesn't compete with ParoQuant directly:
- Damping = regularization on weight-update size to prevent overfitting
  calibration noise. Operates on GPTQ inner loop.
- ParoQuant = reshapes input distribution so groups absorb outliers.
- Different mechanisms, different problems. Optimal λ shifts post-rotation
  but is just per-Linear scalar config — re-sweep is cheap.

Quantile sweep DOES compete with ParoQuant:
- Both handle outliers. Aggressive activation clipping reduces what
  rotation has to redistribute.
- The two methods compete for the same outlier-handling budget.

Compose-friendly with rotation (always run regardless):
- Production-faithful re-cache
- Format menu E4M3/E5M2 differentiation
- Calibration diversity
- scale_sweep (already in)

Methodology rule: anytime you compare A vs A+rotation, both arms must use
the same per-Linear sweeps re-run on their respective configurations.
Otherwise comparing well-tuned-no-rot vs default-tuned-rot understates
rotation's value (or vice versa).

## My recommended priority ordering (the actual plan question)

1. **HALO measurement on 27B production stack.** One overnight run. Re-
   export Qwen3.6-27B with `--halo-mode random` on top of the existing seed
   candidate + production cache + GPTQ + scale_sweep, validate vs the
   shipped artifact's KL=0.031. **Decides whether rotation pays at all on
   this stack** — gates everything downstream.
2. **Production-faithful activation re-cache** (Step F applied to
   activations). Composes with everything, including ParoQuant. Plausibly
   the largest non-rotation single win.
3. **Format menu E4M3/E5M2 differentiation.** Cheap, local, no new methods.
   Pure allocator change. Plausibly 1-2% KL.
4. **Per-Linear GPTQ damping sweep.** Cheap, local, low-risk, ~1%.
5. **Calibration diversity** (4-bucket mix, fixed dataset hashes).
6. **THEN if HALO measurement is positive**: ParoQuant evaluation. Pick
   ParoQuant over FlatQuant if the HALO-on-27B result shows clustered gains
   (some Linears benefit a lot, most don't) — that's the ParoQuant signal.
   Pick FlatQuant if uniform-ish gains.
7. **Per-Linear activation quantile sweep.** Defer until rotation question
   is decided to avoid attribution issues.
8. **Fisher-weighted GPTQ loss.** Heavier; save for after simpler wins
   validate.

What we DO NOT pursue:
- Sub-NVFP4 formats (no W4A4 path on weight, no kernels)
- AWQ
- Custom-kernel methods (OWQ, SqueezeLLM, etc.)
- Polish-of-many extensions (noise-dominated)
- Cross-layer DP refinement (CLADO/output-Fisher) as a quality lever — keep
  as research code path

## Files to know

- Branch state: `/home/rob/prismaquant/` on `fix/halo-qwen35-rmsnorm-fold`,
  2 commits ahead of main, not pushed.
- Smoke artifacts: `/home/rob/dq-runs/qwen35-0p8b-halo-smoke-20260509T132625Z/`
  - `SUMMARY.md` — full smoke writeup including fix verification
  - `exports/{nohalo,halo-random,halo-random-fixed}/` — the three artifact variants
  - `halo_bf16_sanity.py` — BF16-only HALO test
  - logs in `logs/`
- Shipped 27B: `/home/rob/dq-runs/qwen36-27b-step00-production-cache-vllmlegal-export-main-20260509T073125Z/`
- Current handover: `/home/rob/prismaquant/docs/claude_handover_2026-05-09.md`
- Memory notes referenced:
  - `paroquant_candidate.md` — user flagged 2026-05-09
  - `halo_qwen35_norm_bug.md` — bug + fix record
  - `polish_overfit_2026_05_07.md` — n=8 polish overfitting audit
  - `prismaclade_l3_non_additivity.md` — L3 DP regression
  - `prismaquant_traction_context.md` — 60k HF downloads, paper-grade work,
    bias toward provably-non-regressive methods
  - `feedback_cuda_graphs_everywhere.md` — fixed-shape paths capture into
    CUDA graphs by default with env-gated fallback

## What I want from the plan

Given the priority ordering above, generate a concrete plan for:

1. **The HALO-on-27B measurement run** (#1). Specify: seed candidate to use,
   export invocation, validation methodology (KL on what calibration set,
   how many samples), pass/fail thresholds, and what the decision tree is
   based on the outcome (HALO ships behind a flag / HALO drops / pursue
   ParoQuant directly).

2. **The production-faithful activation re-cache** (#2). Specify: where in
   the pipeline this slots, what existing code paths it modifies (probably
   `production_weight_cache.py` + `build_rtn_cache.py` + the activation
   indexing path), how to validate it (smoke on Qwen3-0.6B → 4B → 27B), and
   whether it's a default-on or opt-in change.

3. **The format menu E4M3/E5M2 differentiation** (#3). Specify: probe
   changes to `kl_sensitivity_probe.py`, allocator changes to consume the
   richer table, exporter changes to handle E4M3/E5M2 selection (most likely
   already supported via `data_type` field — just needs the probe to emit
   the right entries).

4. **Damping sweep + calibration diversity** (#4-5). Specify: what gets
   parametrized, where the sweep lives, what the diverse calibration mix
   is concretely (HF dataset names, hashes, mixing logic).

5. **Sequencing**: what runs first, what blocks what, what can run in
   parallel, what's the critical path to a Qwen3.6-27B re-export with all
   wins composed.

The output should be implementable — concrete file paths, commit boundaries,
test plan, validation gates. Not a research roadmap; a delivery plan with
optional research forks.

If the plan thinks the priority ordering above is wrong, push back with
reasoning and an alternative.
