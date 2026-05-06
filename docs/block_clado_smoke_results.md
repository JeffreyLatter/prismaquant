# Block-CLADO smoke results (Qwen3-0.6B)

Date: 2026-05-06
Branch: `block-clado` (commit `5ea1391`)
Run: `/home/rob/dq-runs/qwen3-0p6b-block-clado-20260506T013751Z`

## Pipeline

1. Measure unary `Ω_ii(unit, fmt) = KL(teacher ‖ unit←fmt only)` for every fused-sibling group × format pair.
2. Measure intra-block pair `Ω_ij(unit_a, unit_b, fmt_a, fmt_b)` via the four-term identity, restricted to within-block edges only.
3. Lagrangian λ-sweep on the cost payload to recover the surrogate Pareto frontier.
4. Real-KL validation of every positive-cost frontier point.

Cost payload measurement: 226 unary + 1,512 pair = 1,738 forward passes, **75.9 s** total on Qwen3-0.6B with 2 × 128 calibration tokens.

## Frontier comparison

| bpp ceiling | Block-CLADO best | Unary-only best | Δ |
|---|---|---|---|
| ≤ 4.6 | **0.124** @ 4.57 bpp | 0.202 @ 4.57 bpp | **39% better** |
| ≤ 5.0 | **0.124** @ 4.57 bpp | 0.202 @ 4.57 bpp | **39% better** |
| ≤ 5.5 | **0.124** @ 4.57 bpp | 0.178 @ 5.19 bpp | **30% better** |
| ≤ 6.0 | **0.124** @ 4.57 bpp | 0.175 @ 5.91 bpp | **29% better** |
| ≤ 7.0 | **0.124** @ 4.57 bpp | 0.175 @ 5.91 bpp | **29% better** |
| ≤ 9.0 | **0.124** @ 4.57 bpp | 0.155 @ 7.87 bpp | **20% better** |
| ≤ 13.0 | 0.124 @ 4.57 bpp | **0.073** @ 10.15 bpp | unary-only 41% better |

**Block-CLADO with pair terms produces a strictly better Pareto frontier in the entire deployment range (4.5–9 bpp).** Above ~10 bpp the unary-only frontier extends into a near-BF16 regime that Block-CLADO's frontier didn't sample (we capped λ-sweep at the kneedle's natural range).

The single best validated assignment from Block-CLADO is **4.5726 bpp / real KL 0.1238**, with format mix `NVFP4=191, MXFP8_E4M3=6`. The pair-term measurements identify *which* 6 layers to upgrade to MXFP8; unary-only at the same bpp picks a different 7-layer upgrade set and lands at real KL 0.202 — a 39% relative regression from the same compression budget.

## Surrogate quality

| Method | n | Spearman ρ (surrogate vs real_kl) |
|---|---|---|
| Block-CLADO | 14 | 0.231 |
| Unary-only  | 9  | 0.633 |

**Counterintuitive but informative**: Unary-only's surrogate ranks individual frontier points more reliably than Block-CLADO's, despite producing a worse frontier overall. Interpretation: Block-CLADO's pair-cancellation correctly compresses harder for the same predicted cost, but the cancellation also amplifies the per-point real-KL noise. The surrogate gets the *macroscopic frontier shape* right but is unreliable at point-level discrimination.

This means Block-CLADO must be paired with real-KL validation across the frontier. The surrogate-kneedle picked 4.86 bpp / real KL 0.217; the best validated point is 4.57 bpp / real KL 0.124 — strictly better on both axes. Trusting the surrogate kneedle alone leaves a 43% relative real-KL improvement on the table.

## Surrogate cost going negative above ~5.8 bpp

`cost_total` becomes negative at high bpp because the cumulative pair-cancellation `Σ Ω_ij` exceeds the unary `Σ Ω_ii` once enough layers are at high precision. This is mathematically consistent (Ω_ij can be negative when quantization errors partially cancel between adjacent layers — a real second-order phenomenon, not a bug), but it's outside the second-order Taylor trust region. The kneedle CLI restricts kneedle picking to the positive-cost region for this reason.

## Action items

1. **The math works.** Block-CLADO's frontier strictly dominates additive surrogates in the deployment range on a real LLM.
2. **Surrogate kneedle is unreliable.** Real-KL validation across the frontier is mandatory; pick the lowest validated point, not the surrogate elbow.
3. **Trust region matters.** Surrogate goes negative above the linearisation regime; don't trust it there.
4. **Worth scaling.** Wall time is ~76 s on 0.6B with 2 × 128 calibration tokens. At 27B-class scale, projected ~2–3 hours per anchor — comparable to the existing adjoint sketch but with cleaner mathematical guarantees.

Open questions before scaling to 4B/27B:

- **Activation quantization**. The current measurement uses weight-only quantization; for NVFP4/MXFP8 deployment, including activation quant would give more deployable cost estimates.
- **Sandwich recalibration**. Re-measuring `Ω_ii`, `Ω_ij` centred at the proposed assignment (rather than at BF16) would tighten the trust region. Empirically, this is the round-02 deliberation idea that was never built.
- **Calibration breadth**. 2 × 128 tokens of wikitext is thin. The TC-42 regression on the 27B kneedle pipeline was a calibration-distribution issue, not a surrogate issue. Worth running on a more representative calibration mix.
