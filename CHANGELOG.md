# Changelog

## 0.5.0 — 2026-08-01

This release establishes the production boundary between PrismaQuant and
Gridbook. PrismaQuant owns quantization, allocation, serialized-byte accounting,
and artifact export; Gridbook alone owns serving code, kernels, runtime flags,
tests, packaging, and releases.

### One runtime, one producer contract

- Deleted the complete vendored Gridbook runtime, CUDA/HIP sources, runtime
  tests, and source-sync machinery (38,216 lines removed).
- Added one immutable Gridbook commit pin, PEP 610 provenance checks, a packaged
  consumer contract, and tiny real-artifact compatibility tests.
- Consolidated producer-owned CB layout and export metadata in
  `cb_layout.py` and `cb_export_config.py`, shared by resident and streaming
  exporters.
- Moved the sole Gridbook pin and resolver into packaged assets under
  `prismaquant/gridbook_runtime/`. This is an intentional 0.x interface change:
  external scripts that sourced `scripts/lib/gridbook_runtime.sh` must source
  `prismaquant/gridbook_runtime/gridbook_runtime.sh` instead.

### Exact accounting and constrained selection

- Unified serialized-payload accounting across candidate construction,
  allocation, reporting, and exporter assertions, including FP4-CB layout-v2
  scale planes, FP8 per-row scales, and shared codebook sidecars.
- Replaced blended latency scoring with quality minimization under exact whole-
  artifact bytes, phase-specific serving SLOs, memory, backend, shape, TP, and
  serving-unit constraints.
- Excluded signed S13-S16 rungs from production menus while retaining research
  export and decoder compatibility.
- Fixed partial LFM packed-expert CB export layouts.

### Packaging correctness

- Wheels and sdists now include the canonical IQ grids and NVFP4/FP8-CB lattice
  tables. Earlier distributions omitted them: IQ failed at first use and CB
  could silently regenerate expensive lattices.
- The shipcard CLI is now installed as
  `python -m prismaquant.shipcard_cli`; the packaged pipeline no longer points
  at a checkout-only `tools/shipcard.py`.
- Distribution and clean installed-wheel gates now exercise every model,
  serving and lane spec, both tensor-table assets, the exact Gridbook pin and
  resolver, the pipeline, and the shipcard CLI.

### Fused NVFP4 safety decision

Gridbook's installed-wheel CUDA operator gate passed, but the teacher-backed
LFM2.5 A/B rejected promotion (exact full-vocabulary KL 0.247178, delta NLL
+0.054964, perplexity +5.65%). Dense and grouped fused-NVFP4 paths therefore
remain explicit opt-ins and default off in Gridbook 0.4.1.

## 0.4.1 — 2026-07-30

Tied-embedding models could not be quantized at all. Found by running the
pipeline on a real checkpoint rather than by reading code.

### A tied `lm_head` is structurally non-quantizable

On `google/gemma-4-31b-it` the cost stage cleared all 60 body layers, skipped the
vision-tower shards, then died on the `lm_head` shard with
`NotImplementedError: Cannot copy out of meta tensor`. Cause: the config declares
`tie_word_embeddings: True` and the checkpoint ships **no `lm_head` tensor at
all** — only `model.language_model.embed_tokens.weight` — so `lm_head.weight` is
a tied alias that nothing materialized. `tie_word_embeddings` appeared nowhere in
the streaming or cost path; there was no weight-tying support. Every
tied-embedding model hit this, which is most of the Gemma family; it went
unnoticed because every shipped artifact (Qwen3.6-27B, Qwen3.5-35B-A3B, Hy3) is
untied.

The head is now **materialized** (phase-2's CE backward runs through it, so meta
is never acceptable) via transformers' own `get_output_embeddings()` /
`get_input_embeddings()` accessors, so no embedding path is hardcoded and the
VL-prefixed name resolves like the plain one. Detection is from the config
declaration plus the index's absence of a head tensor — never a name guess. A
meta head with **no** declared tie now raises immediately instead of surfacing
thousands of lines later.

And a tied head is **excluded from probe, cost and the DP**, rather than
measured. Tying means one `Parameter`: quantizing the head would quantize the
embedding, and the surrogate cannot see that cost — probe and cost measure only
the head's output MSE, while the identical perturbation enters every token
embedding and thus layer 0's input for the whole forward, which no surrogate and
not even the L2 perturbed-X fixed point observes. There is also nothing to
re-encode: a tied source has no `lm_head.weight` bytes, so the footprint would
either fail to resolve the name or subtract the embedding from the floor while it
still ships verbatim. The codebase had already reached this conclusion in one
place — `aura_cost.py` hard-raises on a tied head with the same argument — so
this makes automatic what was an operator instruction, and extends it to the
L1/L2 path AURA does not cover. The exclusion deliberately ignores
`--allow-pinned lm_head`, because the tie is a property of the checkpoint rather
than of the serving profile.

Also removed: an ad-hoc repair in the probe that hardcoded three embedding names
inside a `try/except Exception` that only warned.

### Measured end to end

With this fix, Gemma4-31B completes **probe → cost → allocate → export** for the
first time. The probe was already passing (411 rows, all nonzero `h_trace`, 60
layers); cost now completes with zero errors; the allocator hits
`achieved_bits=6.000` with a genuinely heterogeneous 244 NVFP4 / 119 FP8 / 27
BF16 assignment; and the export writes a 27.18 GB compressed-tensors artifact
whose `config_groups` carry 4-bit `tensor_group` and 8-bit `channel` schemes,
with `tie_word_embeddings` preserved and **no `lm_head` tensor** — the embedding
ships once, so the tie is not silently materialized into duplicated bytes.

That run used a deliberately tiny calibration (2 samples, seqlen 512) to reach
failures fast. **It is an enablement result, not a quality claim** — the artifact
has not been served and no KL/PPL has been measured.

## 0.4.0 — 2026-07-30

Closes #29 and lands the KV-cotangent path, which removes the default-off guard
that was blocking KV-sharing architectures. Minor rather than patch because the
Fisher measurement for KV-sharing models changes (it was wrong), and because an
export that previously succeeded by silently demoting FP8_SOURCE now behaves
differently. Shipped allocations are unchanged (35B: 0 of 500).

### Fisher: the KV-cotangent path (part of #9, closes MINOR-M33)

Gemma4-style architectures share K/V across layers: a "storing" layer computes
K/V and later "sharing" layers consume them. Phase-3 forwards each layer in
isolation and handed the consumer a **detached** K/V, so its backward stopped
at that boundary and the storing layer's `k_proj`/`v_proj` Fisher never saw any
consumer's contribution — an under-count on precisely the layers that feed other
layers. That is why `num_kv_shared_layers > 0` was blocked behind
`PRISMAQUANT_ALLOW_KV_SHARED_FISHER`.

Consumers are now handed grad-enabled leaf clones; their `.grad` is the cotangent
each contributes, accumulated per storing layer and used to seed that layer's
backward alongside its own output cotangent. Phase-3 sweeps in reverse and
`kv_shared_layer_index` is derived from layers strictly below the sharing point,
so every consumer is harvested before its producer is forwarded — one pass, no
disk state. Both facts are pinned against the installed modeling source.

**Verified by exact equivalence, not plausibility:** on an fp64 synthetic model,
`h_trace` through the isolated protocol is bit-identical to a single end-to-end
autograd backward (relative error 0.00e+00), while the pre-fix protocol
under-counts `k_proj` by 85.1% and `v_proj` by 38.5%.

Three things the equivalence surfaced that the design did not predict:

- **The under-count was never confined to k/v_proj.** Phase-3 chains each layer's
  input gradient downward, so the producer's truncated input gradient was
  inherited by every layer *below* it — all of `layers.0.*` moves without the fix.
- **The Fisher hook must fire exactly once.** These hooks pop their saved forward
  input, so a backward hook firing once per root would silently drop half the
  Fisher; both roots go through one `torch.autograd.backward` so autograd
  accumulates at the shared node first. Pinned by counting hook invocations.
- **A borrowed leaf must never be seeded as a root.** In reverse order a consumer
  is handed a container keyed identically to the entry the previous consumer just
  filled; seeding it would inject one consumer's cotangent into another's harvest.

The guard is inverted rather than deleted: it now fires only when the cotangent
path is unavailable (`PRISMAQUANT_KV_COTANGENT=0`), and
`PRISMAQUANT_ALLOW_KV_SHARED_FISHER=1` still reproduces a pre-fix probe. Models
without KV sharing are bit-for-bit unaffected with the accumulator on or off.

**Honest limit:** no real `num_kv_shared_layers > 0` checkpoint has been probed.
The percentages above are a correctness demonstration on a toy; the real-model
magnitude is unmeasured. Three conditions would still make such a probe unsafe
and are documented in code: non-differentiable shared state (no cotangent
exists), cotangent left unclaimed at sweep end, and an architecture whose
consumer sits below its producer — the last two surface as diagnostics rather
than wrong-but-silent numbers.

### Export: passthrough source integrity (#29)

The runtime coercion never passed `source_kind`, so passthrough-integrity judged
**every** `FP8_SOURCE` Linear illegal and rewrote it to BF16. The bytes were fine
(the config overlay restored it, materialization copied verbatim) but every
FP8-source artifact's `runtime_coercions` was full of demotions that never
happened — making a real coercion invisible — and it forced a passthrough
exemption in 0.3.1's serving-group escalation.

The source dtype now comes from `_scan_source_dtype_manifest`, the same
recipe-keyed map `allocator.main` feeds `build_candidates` to gate passthrough
candidates, so the exporter judges legality against exactly the vocabulary the
gate that admitted the allocation used. It is scanned lazily, so a BF16-source
export does no extra header IO. Bogus rows went from 4-of-4 to 0 on a synthetic
fp8 checkpoint, and the exemption is gone, so a genuine passthrough mismatch
inside a serving unit now escalates like any other illegality.

No bespoke raise was added, on a measurement: a genuinely non-fp8 `FP8_SOURCE`
assignment is repaired by the coercion rather than the overlay, so it already
ships as BF16 today and a hard raise would be the only change turning a
succeeding export into a failure. The 0.3.1 policy decides instead — refuse
inside a serving unit (naming the legal rungs and the byte cost), coerce alone
when dense, now with a true `delta_bytes` and a passthrough-specific banner.

One required side-fix: with `FP8_SOURCE` surviving the guard it reached
`_production_cache_expected_keys` for the first time, and since its emit branch
returns before the packer, that check would have demanded a render entry nothing
reads — newly failing a valid FP8-source export. All passthrough formats are now
skipped there, not just BF16.

### Streaming: text-only skeletons for vision-language wrapper configs

`CALIBRATION_MODALITY=text-only` decided *what to calibrate on*, but it also
silently decided *how the skeleton is built*: the multimodal path instantiates
via the declared architecture specifically to bypass `AutoModelForCausalLM`'s
text-only downgrade, while the text-only path passed the top-level config
straight to `AutoModelForCausalLM` — which fails on any model whose wrapper
config is not in that mapping (reported on MiniMax-M3 in #12, where the error's
own accepted-class list contains only the model's *text* config).

The text-only path now falls back to the text sub-config **class** rebuilt from
the staged top-level keys. That distinction matters: `stage_text_only` pops the
nested `text_config` and lifts its keys up, so the sub-config object still
hanging off the wrapper is default-constructed and reading dimensions from it
would silently build a wrong-sized skeleton. Detection asks the same two
questions `from_config` asks itself (remote-code `auto_map`, then membership in
the mapping the call consults), so it is config-only and contains no model-type
or class name; a config the auto class can resolve returns the identical object
and takes the original path unchanged.

**This makes no architecture supported.** MiniMax-M3 still needs a
model-structure profile and a serving profile, and the mechanism was validated
against two unrelated real wrapper families since no M3 checkpoint exists here.
The next wall for any VL checkpoint is tensor-name matching (a text skeleton
expects `model.layers.*` where a VL checkpoint often ships
`model.language_model.layers.*`), which is per-architecture profile work.

## 0.3.1 — 2026-07-30

Closes #28: a serving-atomic group could end up with a quantized + BF16 mix
inside it, reported only by the fused-coherence gate at the very end of export.
Fixed at both the cause and the safety net. Allocations on the shipped
Qwen3.6-27B and Qwen3.5-35B-A3B are unchanged (0 of 614 and 0 of 500).

### Cause: promotion now picks a format the whole unit can run

`_promote_group_components` took the highest-**rank** format assigned to any
member of a serving-atomic component and wrote it to all of them, with no check
that the format was legal for the rest — it only received `assignment`,
`format_rank` and `groups`, so it had no way to know. Members of one unit do not
share a shape (gate_up vs down differ on the reduce dim; an odd
`moe_intermediate_size` makes one projection's group/scale-block divisibility
fail while the other's passes), so the promoted format could be illegal for a
subset.

Promotion now takes per-row legal-format sets (`legal_formats_from_candidates`)
derived from the candidate lists, which already encode source-passthrough
integrity, serving-profile rules, group/scale-block divisibility and kernel
shape rules. It picks the cheapest legal-for-all format at or above the max
rank — preserving promotion's non-degrading contract, which
`solve_with_promotion`'s tightening loop is built around — and only downgrades
to the highest legal-for-all when nothing above is common. In the illegal case
every member is written unconditionally, since a member on an equal-rank but
different format would otherwise survive and leave the unit mixed. No common
legal format raises, naming every member with its legal set and the three
upstream causes.

The argument is optional: omit it and the legacy max-rank path runs verbatim, so
callers that cannot supply legality (auxiliary MTP/visual pins, hand-built
assignments) keep today's behaviour rather than acquiring a new failure.

Two paths were genuinely reachable and are now covered: the un-aggregated
(`--no-packed-aggregation` / `--no-fused-aggregation`) solve path, where
promotion is the only coherence mechanism — there the pre-fix symptom was
actually an aborted run, since `compute_achieved` refuses to price an
unpriceable member — and the Pareto seed-JSON promotion, which is **not** priced
by `compute_achieved` and so could let an illegal member format escape silently.
The aggregated path already intersected member candidate sets and needed nothing;
that is now pinned by a test rather than assumed.

### Safety net: export coercion is group-aware

The per-Linear shape/policy coercion (deliberately preserved in 0.3.0) could
rewrite a single member of a unit to BF16. It now resolves whole serving-atomic
components, unioning overlapping units — on the split per-expert representation
a Linear can be both a fused sibling and a packed-expert member — using the same
profile accessors the fused-coherence gate uses, never by parsing names.

The resolution is deliberately asymmetric. If some emittable quantized format is
legal for every member, export **raises** and names it: coercing would ship the
whole unit at 16 bpp (for a packed-expert unit, `num_experts ×` the per-Linear
cost), the dimension that made one member illegal is model-wide so it recurs in
every layer, and a re-solve lands the unit on that legal format for free.
Export must not substitute a format itself — the format the allocator picked is
the one the production weight cache holds a deliberate render for, so a
substitute is a cache miss at best and an RTN render at worst. Only when no
quantized format is legal for every member is BF16 the sole representable
answer; then the whole unit is coerced, loudly, with every member and the byte
delta recorded in `runtime_coercions` and the BF16 audit.

Since the cause is fixed upstream, this path should be unreachable in normal
operation, and the report is written to make any firing look like the upstream
regression it would be. One case it must keep catching regardless: rank-1 legacy
probe stats carry no shape, so `check_stats_format_applicability` admits a
shape-illegal format legitimately and the exporter is the only gate.

## 0.3.0 — 2026-07-30

Closes the three open issues that were ours (#27, #19, #9 item 1). Minor rather
than patch because an export that previously "succeeded" can now raise, and
because a vendored-modelling override that cannot take effect now stops the run
instead of silently continuing on the wrong code.

### Export refuses what it cannot emit (#27)

`export_native_compressed` now declares `EXPORTABLE_FORMATS`, derived from
`FORMAT_SCHEME` plus the container passthrough rather than hand-listed, and the
vLLM serving lane reads its menu from that one place. A format with no
compressed-tensors emit path is a **hard error** naming the Linear, the format
and the resolved profile — it used to be silently rewritten to BF16 with only a
`print`, so a Linear allocated at ~4.25 bpp would ship at 16, blowing the byte
budget and leaving the artifact's real bpp disagreeing with its own
`layer_config.json`.

The *legitimate* coercion is unchanged: a format the exporter can emit but which
is shape-illegal or profile-denied still falls back to BF16 and is still audited
into `mixed_native_manifest.json`. Two facts corrected by reading the exporter:
`FP8_SOURCE` **is** emittable (verbatim-copy path, no packer branch) and
`FP8_E5M2` is **not** (packer branch, no scheme entry) — so the set cannot be
derived from the packer branches. Menu unchanged for every lane.

Consequence worth knowing: allocating under the `research` profile and then
exporting compressed-tensors now fails loudly instead of shipping ~16 bpp.

### Vendored modelling overrides verify or die (#19)

`register_qwen3()` returned cleanly, set its "registered" flag, and on
transformers ≥ 5.13.0 did nothing — after which a probe ran **upstream** Qwen3
modelling code, on the architecture family behind most shipped artifacts, with
no exception anywhere. Root cause is upstream: `_LazyAutoMapping.register`
returns early whenever the config key's `__module__` starts with
`transformers.`, so no override of a natively-supported `model_type` can land
through that call.

- The override now genuinely applies, via public API only: a PrismaQuant-owned
  subclass of the native config (same `__name__`, non-`transformers.`
  `__module__`, picklable) registered through `AutoConfig.register`, which
  applies no such filter. No transformers internals are patched, and the
  fallback engages only when the direct route is verified dead.
- Every registration is **verified** by resolving it config-only, and a failure
  raises with the transformers version, the resolved class, the upstream
  file/function and the remedy. The "registered" flag is set only after
  verification, so a failure stays retryable rather than caching as done.
- `register_deepseek_v4` had a second silent no-op of its own and was resolving
  correctly only by module-path hijack; it now gets the same verification plus a
  guard against a foreign module occupying its path.
- `detect_profile` no longer loses that verdict: it consults the recorded
  override failures and refuses to hand back a profile whose vendored path is
  known dead. The surrounding `except Exception: pass` is correct for keeping
  detection alive, but it cannot be allowed to re-hide a silent no-op.
- The version boundary is now measured, not guessed: healthy through 5.12.1,
  broken from 5.13.0. The old `xfail` threshold of 5.7 was six minor versions
  pessimistic, and the `xfail` is gone — the suite goes red on the wrong
  modelling path.

### Gemma4 KV-sharing pass state (#9, item 1)

The per-forward-pass state hook had already landed; what was missing is that
`_save_precompute_cache` never persisted it and the load path omitted the field
entirely. Since the precompute cache is the normal path for a sharded or
resumed probe, the first checkpoint with `num_kv_shared_layers > 0` would
capture the shared K/V in phase 1, silently drop it on save, and `KeyError`
inside attention for every sharing layer in phase 3 — after hours of phase-1
work, untested in either direction. Fixed both ways, an old cache now hits a
loud error rather than a `KeyError` deep in attention, and a sharing layer with
no captured source K/V raises naming the layer and the remedy instead of
handing back an empty dict. The merge of pass state into per-layer kwargs is now
one shallow-by-design function that raises on key collision instead of silently
overriding.

Item 2 of that issue is unchanged and still needs a GPU run. Two caveats worth
carrying: `google/gemma-4-31b-it` has `num_kv_shared_layers = 0`, so the sharing
path is covered only synthetically until a genuinely KV-sharing checkpoint is
probed; and KV-sharing probes remain default-off because phase-3's isolated
forward detaches the borrowed K/V, under-counting the storing layers'
`k_proj`/`v_proj` Fisher — a cost-model gap, not a flag.

### Also

`F8_E8M0` reporting, the verified DSv4 source layout, and the routed-only
expert-declaration scope all shipped in 0.2.1 and are unchanged here.

## 0.2.1 — 2026-07-30

Corrects one thing that shipped in 0.2.0 on an unverified assumption, settled by
pulling the real `deepseek-ai/DeepSeek-V4-Flash` config and safetensors headers
(a few hundred KB — no weights) plus the authors' `inference/convert.py`.

- **Declared-MXFP4 expert scope is routed-only again.** 0.2.0 widened it to
  `shared_experts.*` on the reasoning that `expert_dtype` describes all of a
  layer's experts. The headers refute that: routed-expert weights are `I8`
  nibble-packs (2304/2304 sampled) while shared-expert weights are `F8_E4M3`
  block-FP8 (9/9), and the authors' converter gates its fp4 path on
  `"experts" in name and dtype == torch.int8`. With the widening in place a real
  DSv4 load would have pushed block-FP8 into the nibble decode and hard-failed
  the packed-grid assertion. No other model is affected — the widening only ever
  applied to a checkpoint declaring `expert_dtype: fp4`.
- `F8_E8M0` added to the safetensors dtype table (it fell to the unknown-dtype
  default of 2 bytes in `dominant_source_bytes_per_param`; span-based accounting
  was already exact).
- The verified DSv4 source layout is recorded in
  `model_profiles/specs/deepseek_v4.json`, including the accounting trap that
  its scales are 1-byte E8M0 planes named `.scale` rather than fp32
  `.weight_scale_inv`, so DSv4 byte accounting must use the per-tensor manifest.

Everything else confirmed as the code already assumed: the `expert_dtype` key
and value, `scale_fmt: ue8m0`, the E8M0 exponent bias, the packed grid, and — the
question that previously could only be guessed — the nibble order and E2M1 table,
which match the authors' reference decode value-for-value.

## 0.2.0 — 2026-07-30

First published release. 0.1.0 existed only as a version string in
`pyproject.toml` and was never uploaded anywhere.

Everything below is allocator/solver/footprint/profile logic. **No shipped
artifact is affected:** re-solving the shipped Qwen3.6-27B and Qwen3.5-35B-A3B
probe/cost pairs at `TARGET_BITS` produces the same assignments as before (0 of
614 and 0 of 500 changed), and the byte-budget floors are byte-identical on both
real checkpoints (27B 6.012 GB, 35B 4.661 GB).

### Allocator

- **Fisher `h_trace` is normalized by the global calibration token count**, for
  every row. Per-routed-token normalization inflated a rarely-routed
  per-expert-`nn.Linear` row by `global/routed` (typically `n_experts/top_k`),
  i.e. inverted importance weighting. Tokens never routed to an expert
  contribute a genuine zero that belongs in the mean-Δloss average. Dense and
  packed-3D probes are numerically unchanged; existing probes are corrected at
  load time from their stored raw accumulators, and a probe carrying raw
  accumulators without token metadata now hard-fails
  (`--allow-legacy-fisher-norm` restores the old warning path). This reverses
  the convention audit M4 documented; the reversal is recorded in `CLAUDE.md`
  §3 and `docs/prismaquant_design.md` §2.2.
- **Packed serving groups are first-class DP units.** A packed-MoE serving
  group is atomic at serve time but the DP priced upgrades per row while
  promotion charged the whole group — a systematic mispricing that starved
  cheap dense rows. `aggregate_packed_serving_groups` collapses each group into
  one multi-choice item, so the DP and the serving constraint price identical
  moves and post-DP promotion is a validated no-op. `--no-packed-aggregation`
  restores per-row pricing.
- **Solver termination is feasible-only.** A rung either returns an iterate
  satisfying `achieved <= target + tolerance` or reports INFEASIBLE; deep
  undershoot is recovered by bisection rather than shipped. Among feasible
  iterates the solver keeps **minimum predicted Δloss** (ties to denser), which
  is its actual objective — denser is not monotonically better. `--target-bits`
  runs that previously emitted an over-budget config now exit, with the format
  floor, what the floor solve promotes to, and the closest achieved bits in the
  message.
- **Byte-budget "fit the card" selection ships minimum predicted Δloss among
  the rungs that fit** (ties to the larger footprint), matching the solver's
  objective. Filling the card is a proxy that can select a denser artifact with
  worse predicted loss than a sparser one that also fits. `selection.json` is
  self-describing (schema `…byte_budget_selection.v2`): objective, feasibility
  test, the tightened search ceiling, whether bisection ran and why not, the
  full ratchet trace, and the max-bytes pick for comparison.
- **Bit-exact re-encode pricing is gated on an identity activation path.** A
  measured `weight_mse == 0.0` proves `W' == W`, but for W·A· formats the
  measured `output_mse` is real activation-side error, so pricing such an entry
  at zero Δloss handed the DP an unbeatable global minimum for a W4A4
  assignment. The short-circuit now requires
  `FormatSpec.act_quant_changes_input` to be false — a dtype-level declaration,
  pinned registry-wide. Relatedly, a W·A· candidate whose activation cost was
  never measured and prices at exactly 0.0 is excluded from the menu with a
  counted, logged reason instead of winning every budget.
- **Fused-sibling and packed-group UCB hedges aggregate in quadrature**
  (`z·√Σ(stderr·gain)²`) instead of linearly, which over-hedged an N-member
  group by up to √N. Byte-for-byte identical at the default `COST_UCB_Z=0`.
- **`--packed-role-split` hard-errors unless the resolved serving profile
  declares `supports_per_role_expert_schemes`** (GGUF only). It could otherwise
  emit gate_up=NVFP4 with down=FP8 in one MoE layer — a checkpoint vLLM cannot
  load. Role grouping now comes from the model profile rather than a projection
  table inside the allocator.
- A fused group whose members have disjoint format menus, and an assignment
  row whose format has no candidate to price it, are now hard errors naming the
  group/row instead of silently vanishing from the DP or scoring as free.

### Footprint

- **Source bytes are priced from an exact per-tensor safetensors-span
  manifest.** The regime-wide accounting charged every re-encoded Linear at the
  FP8_SOURCE layout as soon as any fp8 dtype appeared, which on a mixed source
  removed more bytes than the checkpoint holds and drove the non-quantizable
  floor negative — letting an artifact twice the budget "fit". A negative floor
  is now always a hard error.
- Tensors whose live-name mapper declines them (MTP sidecars, visual towers)
  keep their source bytes: the mapper answers "is this in the live graph", not
  "does this have bytes on disk". Without this, every `--target-disk-gb` run on
  an MTP-carrying model failed.
- Two re-encoded names resolving to the **same** source span is rejected
  structurally (the manifest carries per-entry span provenance), not by
  docstring convention. Charging both a per-expert name and its packed parent
  would subtract the expert mass twice.
- One accounting path: the byte-budget selector calls
  `footprint.assignment_artifact_bytes` rather than reimplementing the identity,
  and `source_manifest` is a required keyword so the legacy regime
  approximation cannot be reached by omission.

### Serving profiles

- **A profile's format menu is bounded by its lane's exporter**, read from the
  exporter's own declaration (`export_native_compressed:FORMAT_SCHEME`,
  `gguf_formats:GGUF_BLOCK_BYTES`) rather than a duplicated list. Weight-only
  A16 rungs were legal for dense Linears on the vLLM lane while the exporter
  cannot emit them. No production format was narrowed; GGUF is unchanged.
  `research` declares `emulation_only` instead of a lane, deliberately, so
  unserved rungs stay measurable.

### Probe / streaming (DeepSeek-V4-Flash enablement)

- MXFP4-packed routed experts dequant on a dedicated vectorized path, triggered
  by the checkpoint's `expert_dtype` declaration rather than a tensor-shape
  heuristic, with the packed grid and the E8M0 scale-plane dtype as assertions.
  Shared experts are covered by the same declaration. E8M0 `0xFF` decodes to
  NaN. Bit-exactness is pinned against an independent scalar reference.
- Nibble-packed `I8`/`U8` expert tensors are sized at 2 logical elements per
  disk byte by both pre-load cache estimators; sizing them verbatim under-counts
  the resident tensor 4× and makes prefetch silently refuse layers.
- Compressed-sparse-attention layer types and the rope-axis `layer_types` dict
  are handled; phase-1 activations stream to host per layer
  (`PRISMAQUANT_PROBE_BATCHED_ACT_TRANSFER=1` restores the batched transfer).
- Per-expert cost rows resolve, and `PRISMAQUANT_EXPERT_COST_SAMPLE` works on
  the default `COST_MODE=production-render-score` path.
- h-detail blobs record their normalization denominator and a stale directory is
  refused rather than mixed with differently-normalized scalars.

### Packaging / CI

- CI runs the test suite on every push and pull request (Python 3.11 and 3.12,
  CPU torch) plus an import-surface job.
- Tag-driven release pipeline (`docs/RELEASING.md`): builds, asserts the tag
  matches the built version, asserts the runtime JSON specs and
  `run-pipeline.sh` are packaged in both wheel and sdist, verifies a
  non-editable install resolves those specs from site-packages, then publishes
  to PyPI via Trusted Publishing (no API token) and creates the GitHub Release.
- `prismaquant.__version__` is resolved from installed metadata, so
  `pyproject.toml` stays the single source of truth.
- Three tests that drive repo-root `tools/` scripts skip cleanly when run
  against an installed package instead of failing collection.
