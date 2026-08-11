# Routed-MoE learned-codebook boundary

## Status

The producer and consumer implementations exist, but the serving path is not
released or qualified. Gridbook commit
`032e8158acc01b13c125db4c5463267f21b1debf` prepares version 0.8.3 and declares
`abi_features.routed_moe_per_role_codebook_lut = 1` in its packaged
`runtime_contract.json`. It is not tagged or published, PrismaQuant's pin has
`version_is_release: false`, and no device execution has been performed.

Nothing becomes default-on. PrismaQuant still defaults
`CB_CODEBOOK_SOURCE_SCOPE=none`; Gridbook still requires
`PRISMAQUANT_CB_MOE_PER_ROLE_LUT=1`; and serving-lane metadata gives the
unreleased 0.8.3 pin no fused-rung credit. Compatibility crosses the repository
boundary only through the immutable pin and packaged contract. PrismaQuant does
not import or vendor Gridbook.

## Consumer ABI

The wire uses ordinary config groups, not a new role-map field. A learned
routed layer has up to three groups whose exact targets end in:

- `experts.gate_proj`, with gate's ordered `codebook_ref` tuple;
- `experts.up_proj`, with up's ordered `codebook_ref` tuple; and
- `experts.down_proj`, with down's ordered `codebook_ref` tuple.

For a split per-expert-format subgroup, each logical target retains the
physical declaration's discriminator, for example
`experts.gate_proj.format_group_fp8_cb_k28` and
`experts.up_proj.format_group_fp8_cb_k28`. The immutable bundle cell remains
the unsuffixed `(layer, projection, rung)` identity shared by that subgroup.

The checkpoint payload remains physically fused as
`experts.gate_up_proj.cb_qweight` followed by its row-scale tensor, plus the
ordinary `experts.down_proj` payload. Split stacks insert the same
`format_group_*` discriminator after those physical parent names. Gate rows
precede up rows. The config must not also declare a fused `gate_up_proj` ref,
and a gate/up pair must be complete.

Gridbook commit `49733a5` first made its legacy uniform MoE resolver compare
`codebook_ref`; without the opt-in, mismatched refs raise at model load instead
of decoding all roles against the first scheme. Commit `776c45d` adds the
opt-in execution ABI. It ports the dense loader's exact-reference block
interning and per-output-row LUT offsets to routed experts:

- distinct reference tuples remain distinct even when their current values are
  byte-equal;
- exact duplicate tuples are interned;
- `w13` receives one offset per `[gate rows; up rows]` output row, shared by all
  experts rather than repeated on the expert axis; and
- `w2` independently resolves down's book.

Offset-aware native decode operations consume those resident vectors. A
multi-book prefill uses the exact BF16 bridge; one-LUT fused or persistent
routes are ineligible. The flag is process-stable: changing it after first use
raises. Missing refs, mixed formats/activation contracts, invalid role
coverage, or a route that cannot consume offsets fail before generation.

## Producer ABI

PrismaQuant represents one learned book as a logical
`(layer, projection, rung)` bundle cell. The fused physical tensor is split by
rows for encoding only: gate and up are encoded independently with their own
book and per-expert imatrix rows, then their packed qweight and FP8 row-scale
planes are concatenated back into physical order. Down is encoded independently
the same way. The streaming per-expert route applies this independently to each
rung subgroup while preserving the declaration's ascending expert-id order.
Resident and streaming exporters emit the same logical role/config contract.

Bundle inputs bind both levels of identity:

- the logical role records the complete rank-3 `(experts, out, in)` source and
  stacked `(experts, 1, in)` imatrix digests; and
- aliases record each per-expert source/imatrix digest used by the empirical
  cost and cache paths.

Every alias resolves the role's same physical ref tuple. This preserves the
per-Linear measurement vocabulary without training or duplicating a book per
expert. Footprint accounting charges gate and up sidecars independently and
deduplicates exact complete ref sets, even though the weight payload is fused.

Routed learned production is limited to FP8 product-codebook K28–K33. This is
not another allocator rate implementation: the allocator's common
candidate-versus-source byte gate is authoritative. DSv4 routed experts have
MXFP4 source rate 4.25 bpp, so K33 (4.140625 bpp) is legal and K34 (4.265625
bpp) is not. The producer range check is a fail-closed bank/ABI boundary for
the only expert cells that may reach export.

The bank was measured with one-shot `cbl_poolb`, not LDLQ. Both exporters
therefore refuse a routed learned role when the active LDLQ scope includes
FP8; the DSv4 production spelling is `PRISMAQUANT_CB_LDLQ_SCOPE=nvfp4` (or
`none`). This keeps the expert encode plane identical to the burn rather than
silently applying a different post-fit assignment at export.

## Immutable burn-book selection

Routed cells never call `learn_pool`. They are supplied from an explicit
operator-selected burn-shard manifest. Each entry names one accepted shard and
its `(layer, projection, rung)`; the manifest also names the content-addressed
book root. The loader verifies, in order:

1. burn-cell, pass-tag, content-key, layer/projection/rung, expert-population,
   source, and imatrix identities;
2. `adopted_encoder=cbl_poolb`, one-shot/no-LDLQ measurement semantics, and
   trainer parameters `row_sample=64`, `row_seed=4321`, `cap=2_000_000`,
   `iters=4`, `seed=0`, fixed-lattice initialization, and `cand0_v1`
   normalization;
3. the content-addressed path/key and bank metadata;
4. every FP16 subtable's name, shape, dtype, finite values, file stability,
   historical pooled digest, and exact payload digest; and
5. equality between the burn identities and the current role tensors before
   those exact FP16 values enter the bundle.

There is deliberately no directory scan, nearest-rung choice, retraining,
or lattice fallback. Missing, duplicate, unreadable, stale, or mismatched
entries stop the build. The selection manifest participates in the pipeline's
stage-settings hash, so replacing it cannot silently reuse an older bundle.
Each copied bundle cell also persists the selection digest/path, burn
content/pass identity, content-addressed book path/key/file digest, pooled and
per-subtable digests, role/rung, and source/imatrix digests. Bundle reload
validates that bank-origin schema and ties it back to the cell's input and
table identities; legacy and dense cells omit the field unchanged.

The DSv4 run directory currently needs operator reconciliation before a real
bundle can be built. A read-only strict audit resolves 287 valid K28–K33 book
paths covering 284 distinct `(layer, projection, rung)` cells and 285 distinct
FP16 payloads. The reported 236 is exactly the path count through layer 37;
completed layers 38–42 add 51 more, so using 236 would silently discard the
campaign tail. Final layer-shard semantics imply 264 needed logical cells, but
no persisted file chooses pass precedence or the export assignment; three
cells have multiple valid paths and one duplicate cell has different payloads.
Of the strict 287 paths, 76 files are root-owned mode `0600` (141 of all 638
files in the complete bank have that ownership). Producer code must not guess
the intended selection or bypass those permissions.

## Version gate

`prismaquant.gridbook_runtime_pin` strictly parses the sole packaged pin. A
final numeric version `>=0.8.3` carries this ABI; `version_is_release` is
separate because an immutable preparation commit may implement the ABI before
the tag exists. The current exact 0.8.3 pin therefore permits CPU production
and export tests while still receiving no released-runtime performance credit.

The routed refusal remains active for:

- every final numeric version below 0.8.3;
- prerelease, local, malformed, missing, or structurally invalid versions/pins;
- any routed name, explicit routed flag, or rank-3 learned source under such a
  pin.

Dense learned Linears do not consult this gate because their offset ABI was
already available. Required compatibility CI installs Gridbook from the exact
VCS commit and asserts that the version-derived capability agrees with the
packaged `abi_features` marker.

## Ship gate

Before tagging or enabling the runtime flag in a shipping image, all of the
following must pass on unforked vLLM in the known-good container:

1. Load a complete DSv4 artifact with deliberately different gate, up, and
   down books at K28 and K33 (include odd K29), with exact sidecar digest and
   fail-closed missing/extra/stale/mismatched-ref negatives.
2. Prove decode and every selected prefill route bit-exact against independent
   producer emulation across multiple experts, zero-token experts, and skewed
   routing; record route telemetry and prove no one-LUT route engaged.
3. Capture and repeatedly replay CUDA graphs for both decode and prefill. Put
   at least two layers with different LUT allocations in one graph, interleave
   batches/routing patterns, and introduce allocator churn between replays.
   Compare every replay to eager output. This specifically detects a captured
   stale LUT or row-offset pointer.
4. Exercise TP-local row boundaries and verify offset lengths/bases for w13 and
   w2 after load, including equal-valued but name-distinct refs and exact-ref
   interning.
5. Run generation/KL/PPL on the same calibration and assignment contract used
   for the artifact, with no fallback or missing-load telemetry.
6. Measure prefill, decode, memory, and served tokens/s on representative DSv4
   shapes. Performance must be at least parity with the lattice container the
   learned artifact displaces.

Only after those gates should the owner tag/publish Gridbook, flip the pin's
release truth, and consider enabling the runtime opt-in in a served recipe.
