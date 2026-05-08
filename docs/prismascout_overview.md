# PrismaSCOUT Overview

PrismaSCOUT is the current selection layer in PrismaQuant. It uses cheap
surrogates to propose candidate assignments, then selects using measured
end-to-end KL rather than trusting a purely additive or low-rank objective.

That design is deliberate. The mathematically cleaner surrogate allocators were
useful proposal engines, but their final selections underperformed real
measurement on the 27B runs. The current architecture keeps the useful math where
it is cheap and falsifiable, and gives the final decision to empirical behavior.

## Current Shape

1. Probe and cost stages build per-Linear statistics and cheap format costs.
2. `iterate_perturbed_allocation` generates candidate knees and frontier points.
3. Candidate assignments are evaluated on calibration/held-out KL.
4. `polish_from_assignment` optionally runs production-faithful local search near
   the chosen assignment.
5. Export writes the compressed-tensors artifact and validation checks the served
   model behavior.

The important contract is simple: surrogate scores can rank and prune, but a
candidate does not become the shipping choice unless a real KL gate supports it.

## Status Of Reported Numbers

The public Qwen3.6-27B PrismaSCOUT artifact reports `0.0151` held-out KL at
`5.31` bpp against the earlier `5.50` bpp baseline.

The newer production-faithful polish result, `0.0054` at `5.39` bpp, is a
polish-time signal on the matched `2x128` calibration split. It is not yet a
replacement held-out claim; the larger `8x512` re-measurement is still the gate
for promoting that number.

## Legacy Material

Older Block-CLADO, dense-cone, adjoint-L3, and handover notes were archived on
2026-05-06 under `archive/legacy_frontiers_2026-05-06/`. They are useful for
history and artifact replay, but they should not be read as the current pipeline
description.
