# PrismaQuant Numerical-Methods Audit — 2026-07-02

Scope: line-by-line mathematical/numerical correctness of the quantization and
numerical-methods core, on branch `claude/fix-issues-4-6` @ e233818 (working
tree). Seven parallel domain audits (format math, GPTQ/JSO render algebra,
export bit-packing/codecs, Fisher/probe estimators, cost models, allocator/
selection algorithms, KL/eval metrics, streaming/cache dtype fidelity), each
deriving the math independently and running CUDA micro-repros against
reference oracles (brute-force codebooks, a hand-written sequential GPTQ,
exhaustive toy knapsacks, analytic Fisher/KL, compressed-tensors and the vLLM
serving image as byte oracles). Third-party libraries were not audited; only
our conformance to their documented semantics.

Dedupe baseline: all numerical findings from docs/codebase_audit_2026-05-10.md,
docs/audit_findings_2026-05-22.md, scratch/review-2026-06-09/REVIEW.md, and the
codex 2026-06-11 closure batch were mapped to HEAD first (all closed with
commits/tests except the residuals listed in §5). Nothing below re-reports a
closed finding.

Confidence grades: **CONFIRMED** = repro executed (agent) and/or verified
line-by-line in the main session; **DERIVED** = math argument from code
reading; **SUSPECTED** = plausible, unverified trigger. The two criticals and
the top majors were re-verified line-by-line in the main session, including
reading `generate_gparam` from the installed compressed-tensors.

Repro scripts referenced below live in the session scratchpad (transient);
findings are self-contained.

---

## 1. CRITICAL

### C1. NVFP4 `input_global_scale` is `6/amax` — 448× below the compressed-tensors/vLLM convention; serve-time activation block scales lose the FP8 range
- **Where:** `export_native_compressed.py:864-879` (`compute_nvfp4_input_global_scale`),
  `:1130` (`_production_cache_scales`), `:1511-1525`
  (`_packed_expert_input_global_scale`), plus `DEFAULT_INPUT_GLOBAL_SCALE=1.0`.
- **Ground truth:** compressed-tensors `generate_gparam` (verified in the venv:
  `global_scale = scale_data.max * quant_data.max / max_val_pos` = **448·6/amax**;
  docstring: "attempts to use the entire FP8 dtype range"). vLLM's
  `CompressedTensorsW4A4Fp4` loads the on-disk value, and the CUTLASS/reference
  activation quant computes each 16-block's **FP8-stored** scale as
  `fp8(block_amax/6 · G)` with `G` = the loaded value, compensating via alpha.
- **Issue:** the dequant identity is invariant to `G` — its *only* function is
  placing serve-time activation block scales in the representable FP8 range.
  Ours yields `sf ∈ (0, 1]` instead of `(0, 448]`: any activation block with
  amax >64× below the calibration tensor amax gets a subnormal FP8 scale
  (progressive mantissa loss); >~1024× below rounds the scale to 0 and vLLM
  zeroes the entire block. Calibration amax is a max over the whole calibration
  set, so serving-time ratios are at least this skewed. The function's own
  docstring says it returns `max|a|/6` while the code returns `6/max|a|`.
- **Evidence:** found independently by two auditors. Kernel-faithful repro
  through vLLM's own `ref_nvfp4_quant`: with 100× outlier channels (typical LLM
  activation structure), 97% of block scales subnormal, activation-quant relMSE
  +5.2% vs the conventional G; at 1024× block dynamic range, blocks are zeroed
  outright. Real shipped artifact inspected
  (`/home/rob/dq-runs/aura-35b-frontier-20260701/exported`): `input_global_scale`
  ≈ 0.2697 / 0.6358 — exactly 6/amax; convention would be ~121/285.
- **Blast radius:** every shipped W4A4 NVFP4 artifact (dense + packed experts +
  MTP), including the flagship. Invisible to the cost model (registry activation
  emulation uses exact fp32 group scales with no static global and no FP8 snap —
  a measurement gap per principle #1) and invisible to served-KL candidate
  selection (all candidates share the same G). Same latent family as M19
  (−6.6% served KL when fixed).
- **Fix:** one line at each of the three sites (`FP8_E4M3_MAX * _FP4_E2M1_MAX / max_abs`)
  + the default; note `tests/test_prismaquant_export_native_compressed.py:2998-3021`
  and `:2160` currently pin the wrong convention. The fused-sibling `min()` join
  (:3725) stays correct under either convention.
- **Honest accounting:** the serving-metric impact is **unmeasured**. The +5.2%
  is activation-quant MSE on vLLM's reference emulation, not a served KL/PPL
  A/B. Per house rules, do not claim a recovery until a paired served A/B at
  matched bpp lands (M19 precedent applies). Confidence: **CONFIRMED**
  (convention + shipped bytes + mechanism); served magnitude **DERIVED**.

### C2. block_output_match scale recovery explodes (±1e12–1e33) when the reference weight's max-|·| element is negative; export multiplies shipped weights by it
- **Where:** `block_output_match.py:114-121` and `:172-179` (the `getter` in
  both block specs), consumed at `export_native_compressed.py:5704-5708`.
- **Issue:** `s = flat_cur[idx] / flat_ref[idx].clamp_min(1e-12)` with
  `idx = ref.abs().argmax()`. The max-magnitude element is negative ~half the
  time; `clamp_min` replaces the negative denominator with 1e-12, so
  `s ≈ cur[idx]·1e12`. Export then applies `_w_dq *= s` for any `|s−1| ≥ 1e-8`.
  The `refine_block_scales` revert path installs the same garbage on the live
  block mid-search, so subsequent candidate MSEs are meaningless too.
- **Evidence:** integration-shaped repro: rendered |w|max 0.196 → shipped
  |w|max **7.1e21** (gate_proj), 0.143 → **4.2e32** (down_proj), while the
  refine-reported MSE looks healthy. Verified line-by-line in the main session.
- **Blast radius:** `PRISMAQUANT_BLOCK_OUTPUT_MATCH` is **default ON**
  (enc:5505). Production runs are spared only because the production-cache hit
  path `continue`s before the block-linear branch (verified enc:5464-5476) —
  with `PRODUCTION_CACHE=1` every assigned Linear takes the cache path. Any
  no-production-cache export (research exports, ad-hoc `_quantize_2d` lanes)
  destroys ~half its q/k/v/o/gate/up/down NVFP4 Linears.
  `tests/test_block_output_match.py` only exercises synthetic getters.
- **Fix:** divide by the signed value (`s = flat_cur[idx] / flat_ref[idx]`;
  `|ref[idx]|` is the tensor max by construction, never ~0).
- Confidence: **CONFIRMED**.

---

## 2. MAJOR

### M1. Batched NVFP4 export ships **all-zero weights** for any Linear with no cached activations
- **Where:** `export_batched_gptq.py:84-125` (`_build_H_stack`: empty acts →
  `H=I`, `dead_mask[e]=all-True`; comment says it plainly: "Caller will see all
  columns 'dead' and weights zero out"), consumed at
  `export_native_compressed.py:4424/4552-4582`.
- **Issue:** dead-column zeroing is sound per-column on calibration data;
  zeroing an entire Linear because calibration never exercised it destroys it
  for serving traffic. The per-Linear path ships RTN in the same situation; the
  batched do-no-harm gate explicitly skips zero-act Linears, so nothing rescues
  it. Contradicts the module's "bitwise-equivalent to the per-Linear functions"
  contract. The packed-expert *production* path has an explicit empty-expert→RTN
  rescue (production_weight_cache.py:3408-3419); the export path lacks it.
- **Trigger:** batched export default-ON + any NVFP4 Linear absent from the
  activation cache (never-routed per-expert nn.Linear on DSv4/MiniMax layouts,
  shape-mismatched capture, cache miss falling through `_pack_production_cached_2d`).
  Repro: batched output max|w| = 0.0 vs source 3.77; experts *with* acts match
  the per-Linear path bitwise. Confidence: **CONFIRMED**.

### M2. The M19 match-render-scale fix does not cover packed-expert re-pack or the fused joint-global pre-pass
- **Where:** `export_native_compressed.py:5889-5893`, `:6149-6153` (bare
  `_quantize_2d` → env scale rule, default static_6), `:5321`
  (`_compute_layer_joint_nvfp4`); dense path correctly wraps
  `_temporary_export_nvfp4_scale_rule(_export_match_render_scale_rule(cache))` at :1408.
- **Issue:** a joint_mse/four_over_six-rendered expert dequant re-derived under
  static_6 cannot recover its codes — the exact pre-M19 dense bug (−6.6% served
  KL class), reintroduced for the tensors that are 91% of MoE params. Measured:
  rule-mismatched re-derive flips 28,031/65,536 packed bytes (43%) with 6.3%
  RMS dequant divergence; rule-matched batched-default flow is self-consistent
  **by coincidence** (both sides read the same env default).
- **Triggers:** (a) `render_mode="per_expert"` (full JSO stack records
  joint_mse; export env says static_6), (b) any re-export session where
  `PRISMAQUANT_NVFP4_SCALE_RULE` differs from the cache-build env — the exact
  "re-export the flagship" scenario M19 was fixed for. Found independently by
  two auditors.
- **Fix:** wrap both expert call sites like the dense path and record the
  expert render's rule (and `PRISMAQUANT_NVFP4_JOINT_SCALE_LEVELS`, which the
  lever record also drops) in the cache. Note: **joint_mse re-derive is
  mathematically exact for any grid-valued group** (verified by enumeration
  over all possible group max codes), while static_6 re-derive is inexact for
  groups whose max code is 4 or 2 — so re-deriving under joint_mse strictly
  dominates, even for static_6 renders (measured 3.3e-4 vs 1.5e-4 drift).
  Confidence: **CONFIRMED** math/magnitude; trigger reachability DERIVED.

### M3. Packed-MoE expert h_trace is sum-then-square — the exact 5–50× estimator bug the nn.Linear path documents and fixed
- **Where:** `sensitivity_probe.py:484-495` (`_GradNormCapture.backward`
  squares the token-summed packed-weight gradient), consumed via
  `incremental_probe.py:2196-2201/2339-2344`; contrast the correct per-token
  path and its own comment at `incremental_probe.py:1315-1330` ("sum-then-squared
  … inflates by the cross-token gradient covariance — 5-50×").
- **Issue:** `‖Σ_t∇_t‖² = Σ_t‖∇_t‖² + cross-terms`; after `/T_total`
  normalization the coherent component grows ~linearly with calibration token
  count, so packed h_trace does not converge as calibration grows while dense
  h_trace does — a layer-non-uniform, uncontrolled inflation of expert rows in
  the same knapsack. Repro: 1.6× on iid tokens, 3.7× at T=512 correlated.
  Also propagates into `h_trace_per_expert` (channel accumulator squares the
  same summed grad).
- **Blast radius:** every packed-MoE model under `COST_MODE=production-render-score`
  (L1 h_trace × output_mse). Partially superseded where the M4 empirical-expert
  hybrid supplies expert rows (COST_MODE=aura). Verified line-by-line in the
  main session. Confidence: **CONFIRMED**.

### M4. MoE routing-probability normalization is applied in three mutually inconsistent conventions across expert-cost paths
- **Where:** unpacked per-expert Linears: `sensitivity_probe.py:1582-1589`
  divides h_trace by routed-token count (already ≈÷ token-fraction) **and
  again** by `route_prob` (also baked into h-detail `h_diag`/`g2_per_token`,
  :1630-1645); packed experts: token-count only ("routing baked into the
  signal", :1658-1663); `expert_empirical_cost._unit_kl`: true per-token KL, no
  division. The production incremental probe applies the single implicit
  division (correct); the `run_probe_pass` backend double-divides, so the two
  "interchangeable" probe backends disagree by ~1/p_e (20–100×) on expert
  h_trace.
- **Issue:** all three price the same physical quantity (mean-per-token end-KL
  contribution). Within one knapsack, unpacked-expert rows are overweighted
  ~1/p (≈16× at 8-of-128 routing) relative to dense and packed rows → bits
  over-spent on sparse experts on DSv4/MiniMax-layout runs. The ÷route_prob is
  *documented as intended* (CLAUDE.md §3), so this is an internal-consistency
  and objective-bias finding, not a typo. Found from both the probe side and
  the consumer side independently. Confidence: **CONFIRMED** (code), impact
  DERIVED.

### M5. AURA `--hook-harvest` + `--probe-microbatch` silently zeroes the cost of any Linear absent from the final micro-batch's autograd graph
- **Where:** `aura_cost.py:521-581` (`_harvest_gate` + `_make_hook`).
- **Issue:** the post-accumulate-grad hook harvests only during the final
  micro-batch's backward; a data-dependent-routed Linear that fired only in
  earlier micro-batches accumulates a real gradient that is never harvested and
  is then discarded (`weight.grad = None`). Result: `predicted_dloss = 0.0`
  for every format → the allocator sees it as free. Repro: expert routed only
  in micro-batch 0 → cost 0.0 with 0 probe samples (legacy monolithic: 1.77e-4).
  `aura_additivity_gate` degrades gracefully, so nothing fails loudly.
- **Status:** dormant at current pipeline defaults (`AURA_COST_NSAMPLES=8` =
  microbatch → single batch) but `run-pipeline.sh:766-785` hardcodes both flags
  and the flag's help text targets exactly the 32×1024 production configuration
  that arms it. Confidence: **CONFIRMED**.

### M6. The legacy default cost `½·h_trace·output_mse` carries activation energy twice, and its default score field excludes activation-quant error
- **Where:** `allocator_candidates.py:266-275` + `production_render_cost.py:142-171`
  (default `--score-field output_mse`); producers `production_weight_cache.py:1186-1190`,
  `render_score.py:84-98`.
- **Issue (two related defects in the same objective):**
  (a) h_trace is the *weight-space* Fisher trace (contains `E‖x‖²`); output_mse
  contains `E‖x‖²` again. The dimensionally consistent collapses are
  `½·h_trace·weight_mse` (verified exact in repro) or `½·(Σ‖g‖²/T)·output_mse`.
  The mixed product biases per-Linear costs by `∝ in_features·x_rms²`
  (measured 258× cross-Linear mis-ranking in a toy); per-format
  `calibrated_gains` cannot absorb a per-Linear factor.
  (b) The consumed `raw_render_score` is `‖X·ΔW‖²/(rows·out)` — weight-delta
  only. NVFP4 is W4A4 and FP8 is W8A8; omitting activation error understates
  NVFP4 damage on outlier-heavy Linears in exactly the NVFP4-vs-FP8 rung
  choice. The act-aware `score` field exists in the same record and is not
  consumed; the in-code comment mis-states the difference as row-weighting.
- **Status:** this is the *documented* legacy objective ("the original
  prismaquant cost objective"), and AURA's −38% served-KL win over exactly this
  cost is corroborating evidence the bias is expensive. Flagged for the ledger,
  and because `COST_MODE=production-render-score` remains the run-pipeline.sh
  default for non-MoE runs. Related: the per-row *fallback* on render error
  mixes `h_trace·weight_mse` rows into an `h_trace·output_mse` population —
  failed-measurement rows look ~E‖x‖²× cheaper (adverse selection).
  Confidence: **CONFIRMED** (math + repro); design-choice status noted.

### M7. `_end_kl` teacher/student KL-scope mismatch under `PRISMAQUANT_FULL_SEQUENCE_KL=1`
- **Where:** `validation_harness.py:685-696` + `kl_measurement.py:5390-5395/5507-5513`.
- **Issue:** `_end_kl` hard-codes a last-token teacher (`logits[:, -1:, :]`)
  but calls `measure_assignment_kl` without `kl_scope`, which resolves from the
  env; a [1,1,V] teacher silently broadcasts against [1,T,V] student log-probs,
  producing `mean_t KL(p_last ‖ q_t)` — not a KL of anything. Repro: correct
  0.00433 → reported 4.266 with no error. No shape assertion exists in the path.
- **Fix:** pass `kl_scope="last_token"` explicitly (or assert
  `teacher.shape == student.shape`). Confidence: **CONFIRMED**.

### M8. Perturbed-X activation capture keeps the *first* `input_rows` rows, not a uniform sample
- **Where:** `perturbed_x_cache.py:649-662` (`PerturbedActivationCache._capture`).
- **Issue:** `need = input_rows − rows_got` fills from the earliest batches and
  skips the rest (`SharedRowSubsampler` only subsamples within one call) — the
  exact early-token bias `activation_sampling.py`'s priority reservoir was
  written to eliminate, and which the other two collectors use. With defaults
  (NSAMPLES=32, input_rows≤1024) the entire perturbed-X second moment — the L2
  fixed-point's re-measured output-MSE distribution — is estimated from
  calibration document #1 only. Repro: rows kept per batch {64, 0, 0, 0}.
  `max_abs` (tracked over all batches), `validate_assignments_kl`
  (capture_inputs=False), and `production_recache` (input_rows=0) are
  unaffected. Any fix must preserve the fused-sibling shared-row-index property.
  Confidence: **CONFIRMED** (behavior); L2-quality impact DERIVED.

### M9. h-detail files from the two probes are in different units (×n_tokens apart) and the consumer normalizes neither
- **Where:** `incremental_probe.py:2473-2488` writes `{"H": Σ_t …}` raw;
  `sensitivity_probe.py:1624-1653` writes `{"h_diag": Σ_t …/(tokens·route_prob)}`;
  `measure_quant_cost.py:180-186` returns whichever key exists verbatim into
  `predicted_dloss = 0.5·(h_full·err²).sum()`.
- **Issue:** full-Fisher predicted_dloss differs by ~n_tokens (10⁴–10⁵)
  depending on which probe produced the directory. Currently masked only
  because cost-source precedence prefers measured output_mse; any row exposing
  `predicted_dloss` without output_mse alongside per-token rows gets pinned to
  max precision. Confidence: **CONFIRMED** (code paths read end-to-end); latent.

### M10 (latent). Lane-replay KL mispairs teacher microbatches when `calib_microbatch_size > 1` meets the replay cache
- **Where:** `kl_measurement.py:3389-3401` + `:3617-3646`.
- **Issue:** refs regrouped to [mb,L,V] but the replay branch pairs teacher
  group i against student row i via broadcast and divides by N — simulated 2×
  error (0.881 → 0.396). Unreachable today (both call sites pin mb=1); nothing
  asserts `teacher.size(0) == student.size(1)`. Confidence: **CONFIRMED**
  (simulation); reachability DERIVED.

---

## 3. MINOR (selected; all verified as stated)

1. **Kneedle log-floor cliff — present in BOTH kneedle implementations.** A
   single zero/negative measured value maps to a floor 6 decades below the
   smallest positive point, compressing the real curve and flipping the knee to
   the curve start / the worst point. `allocator.py:153-161` (+
   `refine_knee_golden`'s `max(dloss,1e-300)`) — diagnostic only at HEAD; but
   `select_validated_frontier.py:84-117` is on the **ship path**: a
   near-passthrough high-bpp frontier point measuring KL 0.0 (or −1e-9 fp32
   round-off; realistic on FP8-native sources) flips the shipped knee to the
   lowest-bpp candidate. CONFIRMED (both repro'd independently).
2. **LOO "stable" flag is near-vacuous** under default `kl_noise_floor=0`
   (`select_validated_frontier.py:354-393`): dropping the knee point always
   shifts the pick, so `stable=False` on essentially every real frontier; with
   non-default `--unstable-policy` the kneedle is silently never used. Also the
   LOO recomputes the knee on a frozen envelope (dominated points can't
   re-enter), understating instability. CONFIRMED / DERIVED.
3. **Bin quantization gives free upgrades** (`allocator_solver.py:237`): units
   with avg-bit delta < 0.5·bit_precision are charged 0 bins; one-directional,
   invisible to the tightening loop (stall exit returns an assignment violating
   its own overshoot tolerance — repro'd). Benign at current model shapes;
   bites at ≥10³ sub-100k-param quantizable tensors. CONFIRMED.
4. **mse_promotion**: inf/NaN measured MSE coerced to 0.0 → the catastrophic
   group sorts dead last (priority inversion, repro'd); and for non-BF16
   targets the score uses current-format MSE, not the (current−target) delta —
   benefit overstated, ranking can invert. CONFIRMED.
5. **`expert_empirical_cost._unit_kl` measures NVFP4 experts under a
   chunk-shared global scale** while export ships per-expert globals — the
   measurement depends on the `--expert-chunk` knob (≤0.1% aggregate, ±1.1%
   per-expert at 4× magnitude spread). CONFIRMED.
6. **vLLM full-KL tool** (`tools/measure_vllm_full_kl.py`): teacher top-K
   logprobs stored fp16 while student is fp32 (asymmetric rounding; mostly
   cancels in paired A/Bs, biases absolute published confident-KL); pad entries
   `(-1, −inf)` produce `0·(−inf)=NaN` poisoning the whole run's mean (rare at
   top_k=1024); student-floor substitution can clamp the tail bucket and inject
   `pt·(log pt+27.6)` spikes at exactly the divergent positions. CONFIRMED /
   DERIVED.
7. **FP8 block-dequant robustness** (`layer_streaming.py:269/359`): a
   transposed `(in_blocks,out_blocks)` scale_inv reshapes silently (numel-
   compatible) and mis-scales every block — one shape assert closes it; block
   size hardcoded 128×128 (`weight_block_size` checked for presence, value
   unread); an *unmapped* float8 tensor silently casts raw codes → bf16 (the
   historical ±448-range bug; guard relies on the suffix scan finding every
   scale). CONFIRMED mechanism / SUSPECTED reachability.
8. **Compiled-vs-eager RTN is not bit-identical at exact codebook midpoints**
   (`format_registry.py:433-494`): 0.036% of bf16 elements flip ties under
   Inductor fusion (MSE-identical, but breaks compiled-vs-eager bit
   reproducibility of screens; the docstring's "5e-7 max diff" claim is wrong
   at bf16 inputs). Also midpoint tie conventions differ pairwise across
   registry (sign-asymmetric: negative ties round away from zero → net negative
   bias), export (`_round_to_codebook`, toward zero), and the CUTLASS kernel
   (RNE). CONFIRMED.
9. **Cache-fill progressive gate scores on unclipped activations while GPTQ
   optimizes under 0.999-quantile-clipped X** (`production_weight_cache.py:1483-1487`)
   — the export-side gate is clip-consistent; the cache-fill gate re-creates
   the exact mismatch the shared-matrix comment (enc:723-728) warns about.
   Accept-vs-RTN decisions near ties biased by outlier rows. DERIVED.
10. **Batched damp-sweep evaluator uses diag-H approximation** vs per-Linear
    full quadratic form (enc:4499 vs :2040) — different winners possible;
    matters only for historical-artifact reproduction with
    `PRISMAQUANT_GPTQ_DAMP_SWEEP=1`. DERIVED. Related provenance inversion:
    `run-pipeline.sh:447` records the sweep flag default as `1` in the stage
    manifest while the code default is OFF — the manifest claims render math
    that didn't run. CONFIRMED.
11. **Cross-layer pairs study uses an unpaired stderr on shared windows**
    (`cross_layer_residual.py:309-317`): window-difficulty variance is
    common-mode and would cancel in the paired test; the printed stderr
    overstates the true one, biasing the "3/1180 pairs significant" additivity
    null **toward the null**. The per-window vectors needed for the paired test
    are already stored. Paper-relevant (additivity narrative). DERIVED.
12. **`_wikitext_ppl` stride machinery is dead code** (`validation_harness.py:387-425`):
    windows are non-overlapping, the mask slice is empty, and the remainder
    window is overweighted by L/(L−1) — chunked PPL, not strided; identical for
    every artifact so A/Bs stay internally consistent. DERIVED.
13. **AURA per-row stderr uses population (1/K) variance** while the additivity
    gate uses sample (1/(K−1)) — ~1.6% understatement at K=32, feeds the
    opt-in UCB charge. DERIVED.
14. **footprint.py under-counts exactly 8 bytes/Linear on NVFP4** (fp32 global
    sidecars) and 4·E bytes per packed-expert tensor — consistent with the
    recorded "−0.00%" match; everything else exact vs actual emission
    (FP8/MX/FP8_SOURCE incl. odd shapes). CONFIRMED.
15. **Kernel loose ends** (`kernels/nvfp4_fused.py`): `bucketize` in
    `_indices_from_signed_e2m1_values` maps ε-above-code to the NEXT code
    (full-step error) — contract says "already-rounded" but bf16 round-trips
    violate it; no production caller today. Activation-quant tie rounding is
    sign-asymmetric (toward −∞) and uses exact fp32 scales with no fp8 snap/
    global — a known contributor to resident-W4A4 emulation infidelity.
    CONFIRMED / DERIVED.
16. **GPTQ dead-column branch is unreachable** (enc:1789-1798 and the fp8/mxfp4
    variants; export_batched_gptq.py:116-123): `dead = diag(H) ≤ 0` is checked
    AFTER damping, so it never fires; dead channels are quantized normally
    (output-neutral on calibration, arguably safer) — inert code masquerading
    as the standard GPTQ safety net. In the batched builder this is what makes
    M1's all-dead path the *only* live use of the mask. DERIVED.
17. **JSO's two halves optimize different objectives** (unweighted weight-MSE
    global grid vs activation-weighted per-group levels), and the in-loop JSO
    selector scores under the *snapped* scale while the final packer defaults
    to un-snapped scoring (the m-M21 research lever) — the same {6,4} grid is
    optimized under two objectives in one render. INFO-grade given the measured
    no-op, recorded for coherence. DERIVED.
18. **Cholesky-failure fallback discards the JSO-optimized tensor global**
    (enc:1870-1879; single attempt → RTN under ambient rule, no jitter
    escalation). Rare (damped H is PD); still score-gated in the cache path.
    DERIVED.
19. **bf16-storage re-derive residuals quantified** (the accepted M19-residual
    contract): same-rule NVFP4 round-trip is code/scale bit-exact but the
    re-derived fp32 global drifts ~2⁻⁹ (1.5e-3 rel dequant); FP8 per-row scales
    drift ~1.4e-3 the same way; GPTQ renders add 375–651/65,536 code flips even
    rule-matched. If bit-faithful export ever matters, persist rendered scales
    (or fp32 dequant). CONFIRMED.
20. **Registry `_end_kl`-adjacent probe softmaxes** (`incremental_probe.py:2749`,
    `sensitivity_probe.py:1992/2210`) run log_softmax on bf16 logits where
    sibling sites cast `.float()` first — Fisher-side only (~0.4% rel), probe
    seed noise dominates. INFO.

---

## 4. Verified correct (coverage highlights — the negative space)

- **GPTQ column-update algebra is bitwise-exact** against an independently
  written sequential Frantar reference (block 128 and block 1); damp→∞ → RTN
  exactly; identity-H → RTN exactly; Cholesky conventions correct.
- **The classic act_order × group-16 scale bug is absent**: scales frozen in
  the original group layout, jointly permuted with W and H, output verified on
  the original group grid; batched GPTQ bitwise-equal to per-Linear for experts
  with activations.
- **Knapsack DP matched brute force on 300/300 random instances** (multi-choice
  recurrence, tie-breaks, float64 table, finite sentinel); bit/bpp arithmetic
  is a single funnel (`memory_bytes_for_shape`) so DP weights and achieved-bits
  cannot diverge; bit-attribution conserves to 0 ulp; prior m-M7 no-op guard
  confirmed live.
- **KL gold lane is clean**: Σp·(log p − log q) with `.float()` before
  log_softmax on both arms at every site; direction consistent everywhere
  (teacher always P); confident filters teacher-only; paired lanes share one
  forward; Spearman matches scipy; p99 is a real percentile (m-M41 fix
  verified); MTP acceptance formula correct.
- **AURA adjoint validated end-to-end** on CUDA toys: probe covariance
  = diag(p)−ppᵀ exactly, predictions match true forward KL within MC error,
  micro-batch global-count normalization correct, fp32/fp64 accumulation
  throughout, M4 hybrid unit-consistent with `_unit_kl`, M9/M10 closure genuine.
- **Dense-path Fisher units chain is right**: h_trace = per-token-mean Fisher
  diag summed over elements; `½·h_trace·weight_mse` reproduces the exact
  second-order term (~1%); no double division on the incremental (default)
  backend; shard merge double-count-safe; zero-token experts → 0, never NaN.
- **NVFP4 bit layout conforms**: pack bytes decode identically via
  compressed-tensors' `unpack_fp4_from_uint8`; nibble order matches the vLLM
  kernel; E8M0 scale bytes byte-equal to `generate_mx_scales`; MXFP8 codes
  byte-equal to the reference; FP8 dynamic delegates to compressed-tensors'
  own qparams (the oracle); FP8_SOURCE verbatim-copy semantics match DeepSeek's;
  Triton kernel weight path bit-exact vs the export codec.
- **FP8 block dequant (streaming)**: block indexing incl. tails verified
  against an fp32 elementwise reference; bf16 multiply costs <1% MSE vs the
  fp8 step (measured); cache store/load round-trips bit-exact; WeightSession
  stage/revert bit-exact incl. interleaved LIFO; all lazy-cache consumption
  sites do explicit `.to(device, dtype)`.
- **fp8 casts**: torch `float8_e4m3fn` is non-saturating (≥~464 → NaN) but
  every live call site clamps first — recorded as a footgun for future code.

---

## 5. Prior-audit closure status (full map produced this audit)

All numerical findings from 2026-05-10 / 2026-05-22 / 2026-06-09 / codex
2026-06-11 verified CLOSED at HEAD with commits/tests, except these known-open
residuals (unchanged, by recorded decision or pending): hooks-KL never
quantizes the packed down_proj A4 input; m-M2's `n≥20` in-sample gate fallback
+ unused `row_weights_list`; M18's activation-side emulation infidelity
(weights-only alignment by design — now compounded by C1); M19's bf16-store
re-derive contract (quantified in §3.19); M27's 8-vs-32-sample cost cache;
`practical_knee` 0.5% tolerance vs frontier noise; m-M33 KV-shared Fisher gap
(fail-fast guard only); M35 BOS handling opt-in. The `PRISMAQUANT_FISHER_CAP_MULTIPLIER`
robust-clip lever remains unlanded (research WIN, still absent from the tree).

---

## 6a. Fix status addendum (2026-07-02, same day)

All findings below §6 were FIXED the same day (full suite 839 passed / 3
pre-existing skips), each with a pinning test. Specifics that matter
operationally:

- **C1**: fixed at all three sites via `_nvfp4_input_global_scale_from_max_abs`
  (default-ON; `PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE=0` reproduces legacy
  bytes), oracle-tested against `generate_gparam`. **Changes the bytes of every
  future NVFP4 export.** Serving-metric impact still unmeasured — a paired
  served A/B at matched bpp is required before claiming any KL recovery or
  re-shipping (M19 discipline).
- **C2/M1**: fixed; no-prod-cache exports and empty-activation Linears now ship
  RTN-grade bytes instead of garbage/zeros.
- **M2**: packed-expert re-pack + joint pre-passes now honor the cache-recorded
  scale rule under the existing M19 flag. Residual: `PRISMAQUANT_NVFP4_JOINT_SCALE_LEVELS`
  is still not recorded in cache levers (render-side change, open).
- **M3**: faithful per-token packed-Fisher capture shipped (F.linear
  interception; ~1e-7 vs brute-force autograd Fisher; escaping compute patterns
  fail fast, `PRISMAQUANT_ALLOW_SUMSQ_PACKED_FISHER=1` restores legacy).
  **Packed-MoE h_trace values change** under production-render-score — probe/
  cost artifacts for packed-MoE models must be regenerated before reuse.
- **M9**: h-detail blobs now unit-stamped; **legacy raw-"H" h-detail dirs are
  refused by design** and must be regenerated.
- **M4** fixed as backend agreement only (single documented ÷routing-prob
  convention retained); the deeper convention question (per-routed-token vs
  per-model-token pricing of unpacked experts vs packed/empirical rows) remains
  a recorded design decision, not silently changed.
- **M10** got the correct microbatch regrouping (not just a guard): mb=1
  bit-equivalent, mb>1 now exact.
- Everything in §3 fixed as specified except: §3.8 tie-convention alignment
  (would change all shipped bytes; A/B territory), §3.17/§3.19 (recorded
  design contracts), §3.20 registry-emulation static-global gap (design
  decision, interacts with C1), and the M6 default-objective change (below).

**Deliberately HELD (need a measured decision, not a silent flip):**
1. **M6** — the legacy `½·h_trace·output_mse` double-count: changing the
   default cost objective re-ranks every non-AURA allocation; promotion-ladder
   A/B required. (AURA, the current default for new work, is unaffected.)
2. §3.8 RNE tie alignment across RTN paths (byte-changing, measure first).
3. Registry NVFP4 activation emulation modeling the static global + FP8 snap
   (M18-residual; the C1 fix shrinks the gap this causes, re-evaluate after
   the C1 served A/B).

## 6. Recommended sequencing (original — superseded by §6a)

1. **C2** (one-line sign fix + a real-getter test) and **M1** (RTN fallback for
   empty-act Linears in the batched builder) — small, unambiguous, no serving
   re-validation needed to be strictly better.
2. **C1**: fix the three sites + DEFAULT, fix the two tests pinning the wrong
   convention, then a paired served A/B at matched bpp on 4B (and the 35B
   frontier artifact) before any claim — M19 discipline. Also close the
   measurement gap (registry activation emulation should model the static
   global + FP8 snap) so the platform *sees* this cost class per principle #1.
3. **M2**: wrap the two packed-expert call sites with match-render-scale (and
   consider joint_mse as the universal re-derive rule); record scale rule +
   joint levels in expert cache levers.
4. **M7/M8/M9** and the §3 robustness asserts (scale_inv shape, unmapped-fp8,
   teacher/student shape) — cheap fail-fast/correctness patches.
5. **M3/M4/M6** are cost-model debt on the non-AURA paths: decide whether
   production-render-score remains a supported default for dense models; if
   yes, fix the packed sum-then-square and pick ONE routing normalization
   convention; if no, gate it as research like the other archived cost modes.
6. **§3.11** (paired stderr) before the additivity null is cited again in the
   paper.
