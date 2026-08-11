# Routed-MoE learned-codebook runtime specification

## Status and scope

This is a design specification for a future Gridbook release. It does not
authorize PrismaQuant to vendor or import Gridbook, and it does not change the
runtime pin. PrismaQuant/Gridbook compatibility continues to cross the
repository boundary only through the immutable release pin and packaged runtime
contract (`AGENTS.md:25-43`,
`prismaquant/gridbook_runtime/gridbook_runtime_pin.json:1-6`). Until a release
implementing this ABI is pinned and served validation passes, PrismaQuant must
refuse learned-codebook refs on routed-MoE stacks.

The change is not required for dense fused Linears. Gridbook 0.8.2 already
resolves each dense shard role independently, interns blocks by the exact ref
tuple, constructs a per-output-row LUT offset, and asserts that the offset
covers every packed row
(`/home/rob/gb-release-prep/gridbook/linear.py:374-445`). That supports distinct
gate/up books inside dense `gate_up_proj` and distinct q/k/v books inside dense
`qkv_proj` without a runtime ABI change. Its no-offset fused FP8 mid-M path is
explicitly ineligible when more than one LUT block exists, and the exact
offset-aware expansion path serves those rows instead
(`/home/rob/gb-release-prep/gridbook/linear.py:492-497`,
`/home/rob/gb-release-prep/gridbook/linear.py:1205-1245`).

The missing behavior is specific to routed experts. The uniform MoE resolver
compares format fields but not `codebook_ref`, returns `schemes[0]`, and the
loader materializes only that scheme's one LUT
(`/home/rob/gb-release-prep/gridbook/config.py:1339-1374`,
`/home/rob/gb-release-prep/gridbook/moe.py:458-505`). Both FP8-CB `w13` and `w2`
then receive the same LUT
(`/home/rob/gb-release-prep/gridbook/moe.py:1915-1946`). The mixed-expert path
owns separate `w13` and `w2` runtime groups, but a `w13` group remains one fused
gate+up scheme and one LUT
(`/home/rob/gb-release-prep/gridbook/moe_mixed.py:351-417`,
`/home/rob/gb-release-prep/gridbook/moe_mixed.py:688-710`).

## 1. Independent fail-closed safety patch

Before adding multi-ref execution, Gridbook should independently add
`codebook_ref` to the signature compared by `_moe_scheme_for_prefix`. The
current signature contains only `grid`, `mode`, `k`, `n_sub`, `type_size`, and
`activation_contract`, then returns the first scheme
(`/home/rob/gb-release-prep/gridbook/config.py:1364-1374`). A normalized ref is
the exact string or ordered tuple of subtable names; ref order is semantic
because product-codebook lists are subtable ordered
(`prismaquant/cb_export_config.py:643-667`).

With the current one-LUT ABI, any differing ref must raise at model load with
the conflicting targets and refs in the error. It must never be downgraded to a
warning and must never select `schemes[0]`. This safety patch is useful on its
own: it turns a currently possible wrong-table decode into an early refusal.
The later multi-ref implementation will replace this single-scheme resolution
with a structured role map rather than weakening the comparison.

## 2. TP-safe wire ABI

### 2.1 Scheme field

A multi-ref MoE CB scheme needs a new, explicitly versioned member:

```json
{
  "codebook_ref_by_role": {
    "gate": ["cb_codebook.<cell>.FP8_CB_K28.sub0", "...sub1", "...sub2", "...sub3"],
    "up": ["cb_codebook.<cell>.FP8_CB_K28.sub0", "...sub1", "...sub2", "...sub3"]
  }
}
```

For `w13`, the mapping must contain exactly `gate` and `up`; for `w2`, it must
contain exactly `down`. Each value is an ordered physical subtable-ref tuple,
not a new logical alias. `codebook_ref_by_role` is mutually exclusive with the
legacy singular `codebook_ref` for multi-ref execution. A list cannot be
overloaded to mean roles because the existing `codebook_ref` list already means
the product codebook's ordered subtables
(`prismaquant/cb_export_config.py:643-667`).

The wire must not persist global `row_start` or `row_count` values. Gridbook
creates TP-local expert buffers with `w13` output width `2 * inter` and `w2`
output width `hidden`, where `inter` is already the partition-local intermediate
size (`/home/rob/gb-release-prep/gridbook/moe.py:353-396`). The loader derives
the local gate/up boundary from those loaded dimensions. This mirrors the dense
loader's use of authoritative local widths rather than assuming a global
checkpoint split (`/home/rob/gb-release-prep/gridbook/linear.py:381-401`).

For architectures whose fused expert roles are not the canonical equal-width
gate/up pair, the model profile must provide the ordered role decomposition and
TP-local widths. The runtime must refuse a mapping it cannot reconcile to the
loaded row count.

### 2.2 Per-expert-format declaration

The current split-stack schema supports only version 1, models a group as
`family`, `format_wire_id`, `expert_ids`, and `tensor_prefix`, and strictly
accepts exactly three serialized entry keys
(`/home/rob/gb-release-prep/gridbook/per_expert_format.py:14-48`,
`/home/rob/gb-release-prep/gridbook/per_expert_format.py:73-108`,
`/home/rob/gb-release-prep/gridbook/per_expert_format.py:149-157`). Version 2
should add `codebook_ref_by_role` to each CB group entry. The same mapping must
appear in the config-group scheme resolved for `tensor_prefix`; the parser must
require exact equality rather than choosing either copy. The duplication binds
the routing declaration and decode scheme to the same physical bytes and makes
drift fail closed.

`ExpertFormatGroup` should retain the normalized immutable role-to-ref mapping.
For each CB entry, the CPU-only parser must validate before device work:

- the exact role set for its family (`gate,up` for `w13`; `down` for `w2`);
- the expected subtable count and order for the declared grid/rung;
- non-empty, distinct physical names within each role's tuple;
- exact agreement with the matching config-group scheme;
- presence of every ref in the externally verified sidecar provenance; and
- the existing layer, expert partition, tensor-prefix, grid, and rung checks.

The current parser already resolves the matching scheme and validates its grid
and rung against the group wire id, which is the seam to extend
(`/home/rob/gb-release-prep/gridbook/per_expert_format.py:217-245`). Version 1
remains readable only with its legacy one-ref semantics. Unknown versions must
continue to fail rather than guessing
(`/home/rob/gb-release-prep/gridbook/per_expert_format.py:101-108`).

Uniform MoE layers do not carry `per_expert_format_groups`, so the scheme-level
mapping is authoritative there. Both uniform and split paths should call one
shared role-map validator.

## 3. Loader representation

Port the dense block-interning mechanism rather than inventing a parallel LUT
cache. At `process_weights_after_loading`, the MoE method should:

1. Resolve every role's exact ref tuple and obtain its already hash-verified
   tensors through `quant_config.get_codebooks()`. That function is the one
   memoized sidecar-loading choke point for dense, MoE, and top-level loaders
   (`/home/rob/gb-release-prep/gridbook/config.py:836-866`).
2. Intern by exact ref tuple, not tensor equality. Differently named refs remain
   distinct even if their current FP16 values are equal, matching the dense
   contract (`/home/rob/gb-release-prep/gridbook/linear.py:405-435`).
3. Concatenate each unique table set into one flat LUT and build TP-local
   `int32` vectors `_cb_row_offset_w13` and `_cb_row_offset_w2`. The first
   `inter` entries of `w13` select gate, the second `inter` select up, and all
   `hidden` entries of `w2` select down. A book pooled across all experts needs
   an offset by output row, not an `[E,N]` table; the same vector applies to
   every expert.
4. Assert dtype, contiguity, device, exact row coverage, legal LUT bounds, and
   role boundaries before any route can be selected. The dense path's
   `cb_row_offset.numel() == rows` assertion is the minimum precedent
   (`/home/rob/gb-release-prep/gridbook/linear.py:435-445`).

The mixed method must perform the same construction independently for every CB
sub-stack it creates. Its current loop resolves one scheme and materializes one
flat LUT per family/format group
(`/home/rob/gb-release-prep/gridbook/moe_mixed.py:368-429`); the new role map and
offset vectors belong on that group's lane object.

## 4. Decode launcher ABI

The FP8 grouped decode op should become conceptually:

```text
cb_moe_gemv_fp8(
    xq, qw_stack, cb_flat_fp8, cb_row_offset, scale,
    pair_expert, pair_xrow, k_bits, n_sub, type_size)
```

The Python custom op, fake implementation, extension binding, and CUDA launcher
must all accept the offset vector. The existing MoE op has no such operand
(`/home/rob/gb-release-prep/gridbook/ops.py:441-457`), whereas the dense op
already passes one (`/home/rob/gb-release-prep/gridbook/ops.py:52-64`).

The CUDA kernel indexes an expert stack by `(expert, output_row)` but currently
uses the unoffset LUT directly
(`/home/rob/gb-release-prep/gridbook/csrc/cb_gemv.cu:1001-1073`). Port the dense
operation exactly: load `cb_base = cb_row_offset[n]` and add it before each
subtable lookup. The existing dense kernel demonstrates that address calculation
and validates an `int32` vector covering every output row
(`/home/rob/gb-release-prep/gridbook/csrc/cb_gemv.cu:295-317`,
`/home/rob/gb-release-prep/gridbook/csrc/cb_gemv.cu:640-662`). The MoE launcher
must additionally check that the offset is contiguous, on the same CUDA device,
has `Nout` entries, and cannot address beyond the supplied LUT.

`moe.py` must pass `_cb_row_offset_w13` and `_cb_row_offset_w2` to the two stages
independently; today both stages pass the same LUT without an offset
(`/home/rob/gb-release-prep/gridbook/moe.py:1915-1946`). The mixed decode path
must do the same with the lane-local vectors
(`/home/rob/gb-release-prep/gridbook/moe_mixed.py:688-710`).

The first runtime release may scope multi-ref support to FP8-CB product books.
Any FP4 or unsupported-mode scheme carrying `codebook_ref_by_role` must fail at
load until its own grouped kernels implement the same ABI.

## 5. Prefill and alternate serving routes

Supporting only decode GEMV is insufficient because a loadable artifact may
select a different prefill route.

The exact BF16 quality bridge already expands FP8-CB through an op that accepts
row offsets, but the MoE helper currently supplies a cached all-zero vector
(`/home/rob/gb-release-prep/gridbook/moe.py:1064-1099`). For an expert slice of
size `n_e`, it must repeat the appropriate TP-local family vector in expert-major
order to cover the flattened `n_e * Nout` rows. The mixed prefill path calls this
same helper and inherits the correction
(`/home/rob/gb-release-prep/gridbook/moe_mixed.py:712-738`).

The optimized FP8 grouped prefill currently obtains one LUT and passes it to
both `w13` and `w2` launches
(`/home/rob/gb-release-prep/gridbook/moe.py:1695-1704`,
`/home/rob/gb-release-prep/gridbook/moe.py:1769-1792`). It must either accept
the corresponding per-row offset in each launch or be load-time ineligible for
a multi-ref layer. Until ported and validated, selection must fall back to the
corrected quality bridge with explicit route telemetry; silently invoking the
old one-LUT kernel is forbidden.

The same rule applies to every alternative fused, persistent, expansion, or
decode route: it must consume the role offsets or be deterministically gated
off for multi-ref schemes. For example, persistent-B prefill currently accepts
one LUT and no row-offset tensor
(`/home/rob/gb-release-prep/gridbook/ops.py:273-315`,
`/home/rob/gb-release-prep/gridbook/moe.py:1289-1307`). Even when a route is
FP4-only and therefore outside the first FP8-CBL implementation, its selector
must not accidentally claim a future multi-ref scheme.

All offset tensors are materialized once at load and remain device resident.
Steady-state dispatch must not read refs, environments, or device values on the
host; this preserves the existing GPU-bound/capturable contract.

## 6. Minimum served validation gate

A Gridbook release may declare routed learned books production-capable only
after all of the following pass:

1. **CPU parser/refusal tests.** Version-1 compatibility, version-2 happy paths,
   and failures for missing/extra roles, malformed subtable tuples, declaration
   versus scheme mismatch, unknown refs, wrong family/rung, and unsupported
   grids. Every malformed artifact fails before device work.
2. **Loader-offset tests.** Use gate/up/down refs whose tensor values are
   deliberately identical and prove that name-distinct refs remain distinct,
   that exact duplicates are interned, and that TP-local `w13`/`w2` offsets
   cover every row with the expected block bases. This protects the deliberate
   reference-identity behavior already present in dense loading
   (`/home/rob/gb-release-prep/gridbook/linear.py:405-443`).
3. **CUDA bit-exactness.** Compare grouped decode and every eligible prefill
   route against the codec/emulation result for distinct gate/up/down books at
   representative certified rungs, including K28, K38, and K43; cover odd and
   even subtable splits, multiple experts, zero-token experts, and skewed
   routing. Equality is bit-exact at the declared reconstruction boundary, not
   merely a tolerance claim.
4. **CUDA graph replay.** Capture and replay decode and prefill with distinct
   role refs and prove that graph execution retains the correct offsets.
5. **Artifact negatives.** On unforked vLLM, boot a complete artifact and prove
   missing, extra, stale, and digest-mismatched refs refuse. Gridbook already
   requires the external digest name set to cover the sidecar exactly; the new
   role map must preserve that gate
   (`/home/rob/gb-release-prep/gridbook/cb_digest.py:72-114`).
6. **End-to-end serving.** Generate through both decode and prefill, record route
   telemetry proving the offset-capable kernels engaged, and compare output to
   the producer's exact learned-book emulation.
7. **Performance.** Served speed on representative routed shapes is at least at
   parity with the lattice container displaced. Any correctness fallback used
   by a multi-ref artifact must itself satisfy the production performance gate.

After those gates, Gridbook must package the producer profile in its immutable
`runtime_contract.json`, publish a release, and PrismaQuant must update its
runtime pin. Producer refusal can be removed only for the exact declared
contract; dense learned-codebook production remains independent of this future
work (`AGENTS.md:25-43`).
