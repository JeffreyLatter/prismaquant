# Native-parity and production format-selection policy

This is the normative policy for balancing Gridbook quality against native
execution performance. It supersedes the earlier `quality + lambda*time`
proposal, the blanket decode-neutrality claim, and the retracted 503/503
screen. Historical experiments remain useful evidence, but they do not
override these acceptance rules.

## 1. Optimize quality under hard deployment constraints

For an assignment `a`, production selection is:

```text
minimize    predicted_quality_loss(a)

subject to  exact_whole_artifact_bytes(a) <= B
            p95_TTFT(a, workload)          <= SLO_prefill
            p95_ITL(a, workload)           <= SLO_decode_itl
            p05_TPS(a, workload)           >= SLO_decode_tps
            resident + KV + peak_scratch   <= device_budget
            backend, shape, TP, fallback and serving-unit coupling are legal
```

Latency is not blended into the objective. There is no `lambda`, no single
phase-weighted `serve_ms`, and no default workload mix hidden in the allocator.
Prefill and decode are separate constraints because a format can move them in
opposite directions. Operators choose explicit SLOs; the allocator minimizes
quality loss within them.

Per-layer or per-operator timing tables may generate candidate assignments.
They are not final evidence. A selected assignment still requires same-session
end-to-end timing plus served KL/PPL/tasks, because routing, scheduler batching,
graphs, scratch pressure, and fallback can change the global rank.

When reporting a relative speed tax, the reference is the fastest **globally
feasible assignment under the same whole-artifact byte budget**, memory limits,
legality rules, and serving-unit coupling. Summing each unit's independently
fastest format is invalid: that combination can exceed the byte budget,
especially below 4.5 bpp. Define the fastest feasible reference independently
for prefill and decode; do not blend the phases.

The constrained Pareto solver described here is not yet implemented. Until it
is, timing tables are proposal data and final selection is an external release
gate, not an allocator capability claim.

## 2. Bytes are a serialized-artifact constraint

Candidate construction, allocation, reports, and exporter assertions use the
versioned `CBSerializationContext` payload accountant. The final authority is
the exact exported artifact: model shards and metadata plus every served
sidecar, with shared sidecars charged once by serialized identity. Target bpp
and index-body bytes are diagnostics, not the acceptance gate.

The relevant serialized contracts are:

- production NVFP4-CB K12..K24 uses two-tier layout v2: `4k + 9` bytes per
  256-weight superblock, about 1.78125..3.28125 bpw before shared sidecars;
- FP8-CB uses a `4k`-byte index body per superblock **plus** one FP32
  `weight_scale` per output row; and
- product-VQ FP16 subtable sidecars are serialized once per codebook
  reference/format identity. A lattice reference does not imply a free
  sidecar.

Layout version, scale coding, activation contract, render identity, sidecar
identity, and serving-unit identity are part of a candidate. A format name
alone is not a byte or execution identity.

## 3. Benchmark the execution contract

Every result records and pins:

- format and rung;
- serialized layout version and scale coding;
- activation quantization (`W4A4`, `W8A8`, or the observed fallback contract);
- concrete kernel/backend and fallback state;
- model, tokenizer, Gridbook, vLLM/runtime, GPU/driver, and image commits;
- tensor-parallel size and scheduler/graph configuration; and
- exact whole-artifact bytes and budget.

Native NVFP4 is W4A4 while FP8-CB K36 is W8A8. A pure endpoint comparison
therefore measures the complete format-plus-activation execution contract, not
weight encoding alone. In delegated NVFP4 MoE, vLLM `auto` can select Marlin,
drop activation scales, and execute W4A16. Such a run is not a W4A4 baseline;
the server backend trace must prove the declared contract.

Release evidence covers the workload rather than one convenient shape:

- a prompt-length distribution and concurrency ladder;
- chunked prefill on and off;
- plain M=1 decode;
- batched and speculative decode at the shipped M, with acceptance recorded;
- MoE routed-token histograms and expert imbalance; and
- whole grouped-MoE operators, not isolated tensor or summed per-expert times.

Plain low-M decode cannot be extrapolated into tensor-core, batched, or
speculative regimes. Final timing uses streaming TTFT/ITL/TPS percentiles and
same-session arm ordering; offline whole-request latency is directional only.

## 4. Same-rate comparisons and production capability

At 4.5 bpp, compare native NVFP4 against FP8-CB K36 only after exact
whole-artifact accounting. Below native NVFP4's 4.5-bpp floor, “same average
rate” is an assignment-level comparison: every NVFP4 promotion must be funded
by lower CB rungs elsewhere. Evaluate the byte-neutral bundle's net quality
loss and phase-specific latency gain; never compare an isolated promoted layer
against an unfunded baseline.

NVFP4-CB v2 is the capacity backbone in the approximately
1.78125..3.28125-bpw band, but it does not yet have teacher-backed same-rate
quality validation. Keep its fused native-FP4 prefill paths explicit opt-ins
until served KL/PPL and routing/shape gates pass; the path changes activation
scales from the fp32-emulated bucket to native ue4m3 factors.

Packed expert stacks currently deny stock NVFP4/FP8 in the Gridbook producer
profile. The native-versus-CB mixed frontier is therefore feasible only for
dense and shared units. Packed experts require a lane-level native/CB A/B and
native expert delegation before the allocator may claim that frontier.

Signed S13..S16 remain registered, exportable, decodable, and shape-compatible
for legacy and explicitly research-scoped use. They are excluded from the
production `nvfp4_cb` allow-list after losing 609/776 (78.48%, conventionally
rounded to 79%) matched weight-MSE comparisons. Every product rung remains in
the production menu: NVFP4-CB K12..K24 and FP8-CB K28..K48.

## 5. Evidence boundaries

The current record supports only the following statements:

- Strong BF16-teacher-backed wins are FP8-CB at Qwen3.6-27B/5.5 and
  Ornith-35B/4.75. They do not establish low-bit FP4-CB quality.
- Exact-4.5 Stage 0 strongly favors production-faithful K36 weight error:
  493/496 units at 27B and 252/252 at 4B. This is a stop-only surrogate screen,
  not served KL/PPL and not a promotion.
- Hy3-295B/2.9 has fit, serve, and TEB evidence but no BF16-teacher quality
  claim. Zero selected NVFP4 units are circular evidence because that allocator
  optimized accuracy only.
- Published “native” reference artifacts are generally mixed NVFP4/FP8, not
  pure NVFP4. Name their actual assignment and activation/backend contract.
- The rapid 0.6B endpoint pair is approximate performance evidence only:
  native is 870,290,032 bytes; FP8-CB model plus sidecar is 871,628,664 bytes,
  a 1,338,632-byte (+0.154%) excess that misses the <=0.1% formal target. The
  arms are also W4A4 versus W8A8.

Raw standalone kernel timing is never served evidence. A result advances only
after the exact production dispatch, quantization, fallback, and end-to-end
workload contract is measured.

## 6. Approved W4A16 support backlog

W4A16 is an approved Gridbook support addition, not merely an external
comparison. Gridbook will own exact symmetric packing, serialized scale and
metadata accounting, serving-profile declaration, loader/delegation, and
validation. The first execution implementation should reuse upstream vLLM's
`RDNAHybridW4A16` backend; it must not create a duplicate custom W4A16 kernel.

The initial experiment covers BF16, TP1, symmetric/no-`g_idx` W4A16 at group
sizes 128/64/32 (nominal 4.125/4.25/4.5 bpw), with exact serialized bytes
including scales and metadata. Report two explicitly different views:

1. served W4A16-A16 versus production CB-A8; and
2. weight-isolated W4A16 versus an explicitly named CB-A16 contract.

This work is paused until suitable validation hardware is available. It is
unimplemented and unvalidated; `INT4_W4A16_g128` therefore remains research
only and denied by every production Gridbook scope (dense, shared, and packed).
No production profile, exporter, loader, or performance claim may imply support
before the full integration and served-quality gates pass.

## 7. Implementation order

1. Use unified serialized byte accounting and candidate execution identity.
2. Keep fused FP4 opt-in until the served quality/routing gate passes.
3. Rebuild exact-byte 0.6B/4B/27B endpoints and optimized menus over the full
   workload matrix.
4. Check whether per-layer timing tables predict end-to-end ranks.
5. Implement the constrained Pareto allocator above.
6. Resume the Gridbook-owned W4A16 packing/delegation feature when validation
   hardware and the upstream backend are available.
