# MoE in the CB lane — route-flip cost design + plugin gap (DESIGN ONLY)

Target: Qwen3.6-35B-A3B (packed 3-D expert stacks, `experts.gate_up_proj` /
`experts.down_proj` per layer). Status: encode/export/allocator are MoE-ready
and tested (this wave); the expert COST design below is specced, not
implemented; the serving plugin has a precisely-scoped gap (§4).

## 1. What already holds (implemented + tested this wave)

- `nvfp4_cb_fields` encodes 3-D stacks with per-expert `col_weights`
  `(E, 1, in)` (gguf `_qw_blocks` semantics); shared-per-role codebooks pool
  across experts AND layers (`_role_of` → `gate_up_proj` / `down_proj`).
- `export_nvfp4_cb` writes stacked `cb_qweight` uint8 `(E, out, bytes)` and
  fp8 `weight_scale` `(E, out)` (LAYOUT.md §3, stacked experts); per-expert
  round-trip is pinned bit-exact vs emulation
  (`test_exporter_packed_experts_roundtrip`).
- Serving-unit uniformity: `promote_serving_units` is format-agnostic — CB
  rungs promote to ONE rung per layer-unit by `format_rank`, mixing across
  layers only (`test_moe_expert_uniformity_across_cb_rungs`); `%256` shapes
  are excluded by the group-divisibility legality gate and fall back
  (`test_cb_legality_rejects_odd_shapes_falls_back`).

## 2. The route-flip problem applies to COST_MODE=local

House record (`aura_expert_routeflip_floor_confirmed`, Step A 2026-06-29):
SMOOTH per-Linear costs are route-flip-blind on routed experts — under
faithful dW, cost-vs-KL Spearman drops 0.45→0.35 and predicted NVFP4/FP8
ratios run 2–49× vs measured 1.1–1.5×. The CB lane's `COST_MODE=local`
weighted-recon cost is exactly such a smooth cost: it scores each expert
stack's reconstruction error and cannot see that quantization noise FLIPS
ROUTES (a tiny logit shift in the router re-ranks experts; the damage is a
discrete routing event, not a smooth output perturbation). Conclusion carried
over unchanged: **expert-stack costs must be MEASURED (empirical unit-KL),
not modeled — for CB exactly as for AURA.**

## 3. Minimal integration (design; ~90–130 LoC, next wave)

The M4 machinery (`prismaquant/expert_empirical_cost.py`) is already
format-NAME-driven and registry-backed, so CB rungs slot in with two small
changes rather than a new tool:

1. **Invoke under the CB lane.** `run-pipeline.sh` `EXPORT_CONTAINER=nvfp4_cb`
   (COST_MODE=local) gains an opt-in step (`EXPERT_EMPIRICAL=1`, default ON
   for MoE profiles): run `expert_empirical_cost --formats "<expert menu>"
   --merge-base <local cost pkl> --output <merged pkl>`. Per MoE layer it
   measures end-to-end mean-token KL(BF16 ‖ unit-quantized) with everything
   else at source precision, splits the unit cost across member tensors by
   n_params, and unions the rows into the local payload — the allocator DP
   reads the merged `predicted_dloss` rows unchanged. (~15 LoC lane gate.)
2. **Weighted render inside the measurement.** `expert_empirical_cost`
   currently renders candidates with the registry qdq UNWEIGHTED (fine for
   the scalar RTN formats it was built for). CB's deliberate render is the
   imatrix-weighted VQ search — measuring an unweighted render while the
   exporter ships weighted bytes is the rendering-confound class. Fix: pass
   the unit's per-expert `col_weights` to the qdq when the closure accepts
   them (the `emu_forward_kl._qdq_accepts_col_weights` probe, ~15 LoC).
3. **Which formats get empirical treatment.** The expert menu = the CB rungs
   offered to experts (+ BF16 passthrough; FP8_CB rungs included — FP8 stays
   in the menu per the standing 2026-06-29 decision, the DP and real-KL
   frontier judge it). Non-expert Linears keep the local weighted-recon cost;
   only `packed_expert_format_group` members get empirical rows.
4. **Cost of the measurement + the RD-law lever.** 35B ≈ 48 layers × 2 stacks
   × |menu| unit-KL forwards. With the fine 13-rung ladder that is ~1250
   measurements — the RD-law helper (`predict_cb_ladder_costs`, validated
   ±3% on weighted-recon at 0.6B) can anchor 3 rungs per unit + holdout and
   predict the rest (~3.5× fewer measurements, ~40 LoC inside
   expert_empirical_cost, opt-in `PRISMAQUANT_CB_LADDER_INTERP=1`).
   **Open: the law is validated on weighted-recon, NOT on unit-KL** — the
   anchors+holdout structure gates it per unit, falling back to full
   measurement where the holdout misses.
5. **Encode tier.** The registry closures resolve `PRISMAQUANT_CB_ENCODE_TIER`
   per call, so the empirical pass inherits the tier for free; `fast` is the
   right tier for a 96-unit × menu screen (encode_tiers.md).

## 4. Plugin gap (for the serving agent — precise contract)

`plugins/vllm_prismaquant/` serves DENSE Linears only (`linear.py`; no
FusedMoE method, no `experts` handling anywhere in the package). A CB MoE
artifact exports today but cannot serve. The missing method must:

- dispatch when a FusedMoE module's prefix matches a CB config group
  (targets are the packed names, e.g. `model.layers.N.mlp.experts.gate_up_proj`);
- load `cb_qweight` uint8 `(E, out, bytes)` (+ fp8 `weight_scale` `(E, out)`)
  into its w13/w2 expert buffers, decoding per expert with the existing codec
  (each `cb_qweight[e]` is exactly the dense §1 layout);
- split the fused `gate_up_proj` stack into vLLM's w1/w3 halves using the
  canonical `gate_proj/up_proj/down_proj` scheme names in config_groups (the
  compressed-tensors packed-MoE lesson — vLLM canonical names even when
  on-disk leaves differ);
- keep experts uniform per layer (guaranteed at export by union-find; the
  method may assert it) and stay CUDA-graph-safe / INV-1-clean (no resident
  expansion beyond the transient pattern).

## 5. Open questions

1. Unit-KL additivity across 96 units at 13-rung granularity — the AURA
   hybrid validated at 8-format menus; the fine ladder's smaller per-rung
   deltas may sit inside unit-KL noise (uncertainty-is-decision-level note
   suggests it washes at ship targets; verify on the 4B-MoE first).
2. RD-law on unit-KL (vs validated weighted-recon) — holdout-gated per unit.
3. Imatrix collection for expert stacks: per-expert `(E, 1, in)` needs
   routed-token activation capture per expert (the zero-expert calibration
   lesson from Ornith — experts with no routed calibration tokens need the
   `fill_packed_experts_from_source` guard analog).
4. Whether the 35B run offers experts the FULL 13-rung ladder or a coarse
   {K12, K16, K20, S16} subset first (measurement budget vs allocator
   granularity — the DP collapse note says fine spacing is partly cosmetic
   for small Linears, but expert stacks are the LARGEST units, where fine
   rungs pay).
