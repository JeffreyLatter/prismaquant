# Speed-aware format policy: fp8-CB at ~4.5 bpw vs native NVFP4

2026-07-27, per Robert's directive ("seriously consider where it's truly
appropriate to take the speed hit for the purposes of accuracy"). Every
number below is measured; sources cited inline.

## 1. The measured substrate

**Accuracy at matched 4.5 bpw** (dense-tier A/B, c551e24, cost-model-level
RTN-vs-RTN fp32): FP8CB_K36 beats vanilla NVFP4 on 503/503 units unweighted
(geomean −40% error), 87% act-weighted. 42 outlier-row units favor NVFP4 —
the joint menu already lets the allocator pick NVFP4 there on accuracy
alone. Part of CB's edge is structural: fp8 rungs run A8 activations where
native NVFP4 runs A4.

**Speed at matched bpw** (served, 2026-07-23..27):
- **Decode: neutral.** CB decode is at per-byte parity (measured twice:
  27B vs AURA; Laguna vs poolside NVFP4 — tok/s ratio == byte ratio). At
  MATCHED bpw both formats read the same bytes → no decode tax at all.
- **Prefill: the entire tax, magnitude regime-dependent.**
  - Dense, current kernels: ~10% (post CUDA-expander + promoted mid-M
    fused; 27B lane).
  - MoE large-expert, current kernels: ~40% vs native grouped CUTLASS
    (Laguna 2,063 vs 3,603 tok/s under `auto`).
  - MoE small-expert: near parity (35B `auto` 4,405 vs its own native-path
    ceiling).
- Serving-cost raw material now exists per layer: the `auto` tuner logs
  every candidate's measured ms per layer — a free per-format,
  per-shape serving-cost table accumulating in every serve.

## 2. The decision structure

At matched storage, choosing fp8-CB over native NVFP4 buys accuracy
(−40% cost-model error + A8 activations) and costs **prefill throughput
only**. So "where is the hit appropriate" reduces to workload shape:

- **Decode-dominated serving** (chat, long generations, agents that think
  more than they read): the tax is ~0. CB is strictly better. No policy
  change needed.
- **Prefill-heavy serving** (RAG, long-document, batch scoring): the tax
  is real (10–40% today). Two honest mitigations already exist: the card
  guidance pattern (the 27B/Laguna cards route prefill-heavy users
  explicitly), and the allocator's joint menu (NVFP4 already wins
  placement on the outlier units).
- The tax is also **shrinking under the kernel campaign** (12×→1.75× on
  the worst lane in one week); a policy over-fit to today's tax would be
  stale by the next kernel round.

## 3. Policy

1. **Default stays accuracy-first** (status quo). Rationale: decode
   neutrality means most served tokens see no tax; the accuracy edge is
   large and measured; the tax is a moving target.
2. **Implement the opt-in serving-cost axis in the allocator** (the
   principled lever, no heuristics): a per-(format, shape-regime) cost
   table measured on the target box — seedable directly from `auto`-tuner
   logs — and an objective `quality + λ·serving_ms` with λ=0 default.
   A user who declares a prefill-heavy profile sets λ>0 and the knapsack
   trades the outlier-adjacent CB units to NVFP4 first (exactly the 42
   units where the accuracy gap is smallest). This is measurement-driven
   end to end: costs measured, tradeoff explicit, no format bans.
3. **Re-run this deliberation after the persistent-schedule build** — if
   MoE prefill reaches ~1.2× of native, the λ knob becomes nearly moot
   and CB's accuracy edge decides everything except hard-A4 latency
   targets.

## 4. What is NOT proposed

No format bans, no bpw-band carve-outs, no "MoE always native" rules —
the 35B/Laguna divergence under identical formats shows regime, not
format, drives the tax, and the allocator + auto-tuner pair already
resolves regime per layer by measurement.
