# Model coverage ledgers

**Status: normative for new-architecture intake and for every export, once the
executable gates land (campaign R5). Adopted 2026-08-21 after the `wo_a`
finding.**

## The failure class this exists to kill

On DeepSeek-V4, `attn.wo_a` — 17.9% of all decode read traffic — was never an
allocator decision. The probe skips `DeepseekV4GroupedLinear`, so no cost cell
was ever created, the tensor shipped as 8-bit passthrough by omission, and
every coverage statistic still read as complete. Nothing was wrong with any
individual stage. The defect is structural: **the pipeline's decision universe
is defined by the pipeline's own enumeration.** When the probe enumerates the
units, and cost, allocation, bpp, and "coverage" are all computed over that
same enumeration, an omission is invisible by construction — numerator and
denominator come from the same code path.

This is a class, not an event. Instances already in the institutional record:

- AQUA priced 5.5% of an MoE's mass because packed experts were outside the
  A-side's enumeration (fixed in `d61bddf`).
- `units_on_fallback_route = 0` published as clean while no unit had a declared
  lane — a zero over an empty denominator.
- The completeness gate read fused-group claims from a dense-only source, so
  routed claims were invisible to it.
- The byte-budget audit caught overshoot but not undershoot — a one-sided
  check on a two-sided invariant.
- Silent-zero cost lookups ranked broken arms first.
- The vLLM post-load sweep matched by `isinstance`, so a non-subclass was
  skipped silently.
- 73.7% of a built body rode an `Sm80` fallback that its own `selection.json`
  recorded and nothing consumed.
- The test-suite sibling: `importorskip` turned our own import errors into
  green all-skip runs, twice.

One principle unifies the fixes: **absence must be loud.** A coverage claim is
valid only when its denominator comes from an authority outside the code that
makes the claim.

## The external authorities

For a quantized artifact there are exactly two:

1. **The checkpoint itself.** The safetensors header enumerates every tensor
   and every byte that ships. Nothing the pipeline forgot is missing from it.
2. **One traced forward.** Every matmul the model executes, with operand
   shapes and bytes, from instrumenting a real forward pass — not from the
   profile's list of what it expects to find.

Every ledger below takes its total from one of these two, never from the
probe, the profile, or the cost table.

## The four ledgers

Each ledger partitions a conserved quantity into **decided** (a priced,
gate-covered allocator decision) and **named-fixed** (an explicit exclusion
with a reason string and an owner). The two must sum to the external total.
Residue — anything in neither bucket — fails the export closed.

| # | Quantity | External total from | Catches |
|---|---|---|---|
| L1 | Bytes on disk | Checkpoint header | Unclassified tensors; budget undershoot and overshoot (two-sided) |
| L2 | Read bytes per decode token | Checkpoint header × read probability (dense p=1, routed p=topk/E) | `wo_a`-class omissions: a held-fixed 1.44 GB/token appears on the shipcard on day one, labeled, instead of nowhere |
| L3 | Executed matmuls → serving routes | Traced forward + the runtime's own published contract (principle 14) | Fallback kernels, lane-ineligible layers, decode-vs-batch regime splits |
| L4 | Cost-model mass | L1/L2 totals | "Priced 5.5% of the model" — the unpriced remainder is itemized, not averaged away |

Named-fixed entries are first-class output, not an escape hatch: each carries a
reason (`probe skips grouped operands`, `runtime cannot serve a CB lm_head`,
`MTP sidecar, spec-decode off`) and lands in the shipcard and on the model
card. The honest state becomes visible instead of silent. An exclusion without
a reason string is residue.

## When the grid runs

**New-architecture intake (profile plugin time).** Trace one forward. Diff the
executed op set against the profile's classification. Disposition every
operand: decide, pin with a reason, or exclude with a reason. The
teacher-forced forward-fidelity gate is this step's quality sibling; the
ledger diff is its coverage sibling. Budget about a day per new architecture.

**Every export.** The L1–L4 closure checks run as gates and fail closed on
residue, exactly as the route-status doctrine already fails closed on an
unbacked route.

**Every card.** The ledger summary table is the card's honesty section.
Principle 12 already requires bpp plus the route histogram; this systematizes
what sits next to them.

## Why a checklist alone is not the answer

This repo already knows that prose rots — the prime directive, and "currency
is not truth." A checklist that a person consults is one refactor away from
describing a pipeline that no longer exists. The durable form is executable:
gates whose denominators are derived from the artifact at run time, plus the
stamped per-artifact table. This document records the frame and the reasons;
the gates enforce it. If the gates and this document disagree, the gates win
and this document is stale.

## Honest limits

The ledgers catch omissions of quantities you know to conserve. They do not
catch a quantity nobody has conceived of yet — KV-cache traffic is excluded
from L2 and labeled as such, and the next `wo_a` may live in a dimension no
ledger tracks. The outside-report channel (a user with a profiler) remains
part of the system, and it worked this time. The ledgers' job is to make sure
the same lesson is never paid for twice.
