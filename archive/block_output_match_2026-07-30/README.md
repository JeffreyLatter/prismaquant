# Block-output match (quality lever #12) — archived 2026-07-30

**Kill order.** Architecture re-vet 2026-07-30, proposal **R25** (Lens 5 P5),
accepted by Robert as part of the wave-2 mandate: *"implement ALL proposals."*
Closes standing debt **D16** as **unreachable**, not *unmeasured*. See
`docs/audits/architecture_re-vet_2026-07-30.md` §R25 and
`docs/ARCHITECTURE.md` §11 / §12 D16.

## What it was

After the per-Linear quantization pass, each decoder layer's attention block
(q/k/v/o) and MLP block (gate/up/down) ran a greedy coordinate-descent search
over per-Linear multiplicative gains `{0.95, 1.0, 1.05}`, scoring candidates by
the **block's** output MSE against an FP32 reference forward — the idea being to
capture inter-Linear composition (q_proj error amplified through the attention
dot product) that per-Linear MSE cannot see. It packed lazily: eligible NVFP4
Linears were computed with `_quantize_2d(..., compute_only=True)`, deferred into
`_BLOCK_COMPUTE_PENDING`, refined, then frozen by `_finalize_compute_only`.
Default **ON** (`PRISMAQUANT_BLOCK_OUTPUT_MATCH`, `"1"`).

## Why it went — three independent findings, none of which an A/B would surface

**1. It never ran.** In the per-Linear emit loop the production-cache pack fires
**first** and `continue`s, the batched-NVFP4 path fires **second**, and the
block-match branch was **third**. With `PRODUCTION_CACHE=1` — the shipping
default — every dense NVFP4 Linear is emitted from the cache and never reached
the branch. Empirically confirmed on two real production exports:
`grep -c "block-output-match"` → **0** in both
`/home/rob/dq-runs/ornith-35b-aura-20260703/logs/export.log` and
`prod-27b-nvfp4cb-5p5/logs/export.log`; the ornith histogram reads
`NVFP4_PRODUCTION_CACHE`, never `NVFP4_block_match`.

**2. Had it run, it would have re-introduced M19.** `_finalize_compute_only`
called `quantize_dequantize_nvfp4(w, group_size=16, global_real_override=...)`
**outside** any `_temporary_export_nvfp4_scale_rule` context — unlike the cache
path (`_pack_production_cached_2d`) and the joint-global pre-pass, both of which
wrap in `_export_match_render_scale_rule`. So it would have re-derived per-group
scales under the default env rule and discarded the render's `joint_mse`
scales: exactly the −6.6% KL defect M19 fixed everywhere else.

**3. Its mechanism is subsumed and its failure mode was silent.** A per-tensor
gain re-search run *after* JSO already solved for the scale by
activation-weighted MSE inside the GPTQ loop — the same argument that retired
the clip solvers and `scale_sweep` (*"clipping is just another way of asking
what the right scale is, and JSO already answers it"*). The whole block was
wrapped in `except Exception → print WARN`, so any failure was an invisible
no-op. Its docstring's "~0.05–0.10 PPL gain" was a **pre-JSO expectation, never
a measurement**.

It also carried a live numerical hazard: numerical-audit C2 (2026-07-02) found
the scale recovery exploding to ±1e12–1e33 when the reference weight's
max-|·| element is negative.

## Durable lesson

**Dead code is not "unmeasured", and the fix for an unmeasured default-ON lever
is not always an A/B.** D16 sat in the debt register asking for a gold-lane A/B
on one 27B-class artifact — an export cycle to measure a `continue`. Reading
the *dispatch order* answered the question for free. Before funding a
measurement, check that the code under test executes on the recipe you ship.

Second lesson: a lever whose entire body is inside
`except Exception: print(WARN)` cannot be trusted to have done anything, ever.
A silent no-op and a working optimization are indistinguishable from the
artifact.

## Contents

- `prismaquant/block_output_match.py` — the block spec builders
  (`make_attention_block_spec`, `make_mlp_block_spec`), `block_output_mse`, and
  the greedy `refine_block_scales` coordinate descent.
- `tests/test_block_output_match.py` — the synthetic-getter tests (they never
  exercised a real block, per numerical-audit C2).
- `export_branches.py` — the three verbatim sites removed from
  `prismaquant/export_native_compressed.py` (per-layer FP32 snapshot capture,
  the per-Linear deferred-pack branch, the post-loop refine+finalize block)
  plus `_finalize_compute_only`. A record, not importable.

## Live-tree state after the wall

- `export_native_compressed.py`: the three branches and `_finalize_compute_only`
  are gone. `_quantize_2d(..., compute_only=True)` **remains** — it is a
  no-pack introspection hook the export tests use for lever threading, and it
  now has no production consumer (documented at the call site).
- **Hard refusal, not a silent behaviour change:** `main()` calls
  `_refuse_archived_block_output_match()`, which `SystemExit`s when
  `PRISMAQUANT_BLOCK_OUTPUT_MATCH` is set to anything truthy. `=0` and unset
  both pass (they already asked for what now always happens).
- `_render_lever_provenance()` now records the key as
  `"archived_2026-07-30"` instead of echoing the env var. **This moves the
  export-cache fingerprint once** — intentional: any in-flight export cache was
  built by a binary that still carried the branch, and re-rendering is the
  honest outcome. No production export was in flight when this landed.
- `docs/design/runtime_flags.md`'s row is rewritten as archived.

## To resurrect

Restore `block_output_match.py`, paste the three sites and
`_finalize_compute_only` back from `export_branches.py`, delete
`_refuse_archived_block_output_match`, restore the env echo in
`_render_lever_provenance`, and move the test file back. **Then fix M19 first**:
wrap `_finalize_compute_only`'s `quantize_dequantize_nvfp4` call in
`_export_match_render_scale_rule`, or it will silently discard the render's
`joint_mse` scales. And give it a reachable dispatch position, or it will do
nothing again.
