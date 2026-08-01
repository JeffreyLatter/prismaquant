# Stage 1 (0.6B): pure NVFP4 vs pure FP8-CB_K36 at 4.5 bpw

**Status: approximate endpoint evidence only; not the formal same-rate
decider.** The completed native artifact is **870,290,032 B**. The FP8-CB model
plus `cb_codebooks.pqcb` is **871,628,664 B**, or **1,338,632 B (+0.154%)**
larger. That misses the protocol's <=0.1% exact-byte tolerance. The pair is
useful for performance iteration, but any native-parity decision must be rerun
under an exact whole-artifact byte constraint.

**Historical execution note (2026-08-01):** this run predated removal of the
vendored Gridbook tree. Its old copy/editable-install commands are intentionally
not reusable. Current reproduction must resolve the exact external runtime via
`scripts/lib/gridbook_runtime_pin.json` and `gridbook_runtime.sh`; a result that
cannot attest that commit is invalid.

Executes the endpoint half of `docs/design/format_choice_4p5.md` §5 Stage 1 /
§5 Stage 2 ("Endpoints first"): two *pure* single-format builds of the same
model at the same nominal 4.5 rung, so there is no assignment-choice confound;
the measured artifacts are not exact-byte matched. Neither endpoint is
shippable — a single-rung menu is
the sanctioned isolation pattern only (`[MEMORY: FP8 in every recipe]`).

Run commit `550710b` (branch `claude/docs-consolidation`; the criteria doc's
`d6dbf58` is an ancestor). Box: one GB10 / DGX Spark, **with the idle serve on
:8000 left running throughout** — every step below is either CPU/meta-device or
a small single-process GPU job kept inside a ~35 GB envelope.

## 1. Tooling: `tools/make_uniform_assignment.py`

New in this session. Given a model path + ONE format + the detected profile it
emits an allocator-compatible `layer_config.json` assigning that format to
every quantizable unit, with the standard exceptions. It reimplements no
legality: the quantizable set and the fused/packed unit decomposition come from
`ModelProfile.build_model_graph`, per-(shape, format, profile) legality and
passthrough integrity from `allocator_candidates.check_format_applicability`,
source dtypes from `allocator_candidates._scan_source_dtype_manifest`,
incomplete fused groups from `decision_units.incomplete_fused_group_members`,
the tied-head test from `tied_embeddings.lm_head_is_tied_alias`, and the
serving-profile resolution from `serving_profiles.resolve_target_profile`.

Design points worth stating:

- **Units, not Linears, are the unit of decision.** A legality verdict may
  never split a fused-sibling or packed-expert group, so an illegal member
  demotes the WHOLE unit to `--fallback-format` (default `BF16`, itself
  legality-checked so a non-bf16 source is never handed a synthesized BF16 —
  core principle 11); if the fallback is illegal too the unit is omitted.
- **Exceptions land where the allocator lands them.** Profile-pinned names, a
  tied LM head and the present members of an incomplete fused group are
  *omitted* from the assignment, which is exactly what makes export keep them
  as BF16 passthrough + ignore-list entries (`allocator.py`
  `incomplete_fused_group_dp_exclusions` / `tied_lm_head_dp_exclusions` drop
  excluded names from the DP, so they never reach `layer_config`).
- **Output is re-gated independently.** `assert_assignment_legal` re-runs the
  allocator's legality over the FINAL assignment and asserts one-format-per
  serving unit; `layer_config_payload` validates against
  `schemas.validate_layer_config_payload` and round-trips through
  `layer_config.canonicalize_assignment`.
- Model access is a **meta-device skeleton** (`init_empty_weights` +
  `AutoModelForCausalLM.from_config`, the `export_native_compressed` idiom) —
  zero weight bytes read, so it scales to any model size.

Tests: `tests/test_make_uniform_assignment.py`, 11 cases on a **synthetic
architecture + synthetic profile** (never registered, no vLLM class, no
structure spec) — coverage, pinned-head exclusion, incomplete-group exclusion,
tied-head exclusion, fused-unit atomicity, BF16-fallback refusal on an fp8
source, no-fallback omission, payload round-trip, and three negative cases on
the legality gate.

```
PYTHONPATH=. /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python \
  -m pytest tests/test_make_uniform_assignment.py -q     # 11 passed
```

### Cross-validation against the allocator

On Qwen3-0.6B the tool's uniform NVFP4 assignment is **bit-identical** to what
`prismaquant.allocator` emits from a real probe + cost under a single-rung
`FORMATS=NVFP4` menu at `TARGET_BITS=4.5` (196/196 entries, same formats):

```
[compare] allocator entries: 196  uniform-tool entries: 196   IDENTICAL: True
```

That is why the builds below could be driven end-to-end through
`run-pipeline.sh` unmodified: the pipeline's own allocator stage reproduces the
uniform assignment exactly, so no stage had to be bypassed and no artifact was
hand-edited. (`run-pipeline.sh` has no skip-if-exists on `layer_config.json` —
the allocator stage always runs — so a single-rung menu is the only way to
drive a uniform endpoint through the stock orchestrator.)

## 2. Setup

| | arm N | arm C |
|---|---|---|
| container | stock `compressed-tensors` | `nvfp4_cb` (gridbook) |
| format (every unit) | `NVFP4` | `FP8_CB_K36` |
| nominal rate | 4.5 bpw (4b + fp8/16 scale) | 4.5 bpw index stream |
| serving profile | `vllm_packed_moe` | `nvfp4_cb` |
| activation contract | native NVFP4 W4A4 | FP8-CB W8A8 (A8) |
| render | production cache: `gptq,static_act_order,joint_scale_opt` | imatrix-weighted VQ, `lattice` codebooks, scale sweep on, `two_tier` scale coding (inert on an fp8-only menu) |
| work dir | `/home/rob/dq-runs/fc45-0p6b-nvfp4` | `/home/rob/dq-runs/fc45-0p6b-fp8cb` |

Common: `MODEL_PATH=/home/rob/models/Qwen3-0.6B` (196 quantizable Linears, 112
serving units, 440,401,920 quantizable params; `lm_head` is tied + profile-pinned
and is BF16 passthrough in both arms), `NSAMPLES=32 SEQLEN=1024` (pipeline
defaults), `DATASET=/home/rob/dq-runs/calibration/diverse-v1.jsonl` (pipeline
default), `TARGET_BITS=4.5 PARETO_TARGETS=4.5`, `COST_MODE=local`.

`COST_MODE` cannot influence either allocation — a one-rung menu leaves the DP
no choice — so `local` was used on both arms purely to keep the cost stage
cheap. **The renders are deliberately NOT matched**: each arm gets its own
lane's production render, which is the comparison the criteria doc describes
("pure-CB pays the known ~5.4 h `balanced` encode wall, pure-NVFP4 is a fast
native-path export"). This is a full execution-contract endpoint comparison,
not a weight-encoding comparison: format, render, container/backend, and
activation quantization all change together. In particular native is W4A4
while FP8-CB is A8.

## 3. Builds — exact commands

```bash
# --- the two uniform assignments (CPU / meta-device only) -------------------
cd /home/rob/prismaquant
PYTHONPATH=. TMPDIR=/home/rob/dq-runs/tmp \
/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python -m tools.make_uniform_assignment \
  --model /home/rob/models/Qwen3-0.6B --format NVFP4 \
  --out    /home/rob/dq-runs/fc45-0p6b-nvfp4/artifacts/uniform_layer_config.json \
  --report /home/rob/dq-runs/fc45-0p6b-nvfp4/artifacts/uniform_assignment_report.json

PYTHONPATH=. TMPDIR=/home/rob/dq-runs/tmp \
/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python -m tools.make_uniform_assignment \
  --model /home/rob/models/Qwen3-0.6B --format FP8_CB_K36 --target-profile nvfp4_cb \
  --out    /home/rob/dq-runs/fc45-0p6b-fp8cb/artifacts/uniform_layer_config.json \
  --report /home/rob/dq-runs/fc45-0p6b-fp8cb/artifacts/uniform_assignment_report.json
```

Both report `units=112 assigned=196 omitted=0 achieved_bits=4.5000`.

```bash
# --- arm N: native container, uniform NVFP4 --------------------------------
export PATH=/home/rob/dq-runs/venvs/prismaquant-cu130/bin:$PATH
export TMPDIR=/home/rob/dq-runs/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL_PATH=/home/rob/models/Qwen3-0.6B \
WORK_DIR=/home/rob/dq-runs/fc45-0p6b-nvfp4 \
FORMATS=NVFP4 TARGET_BITS=4.5 PARETO_TARGETS=4.5 \
NSAMPLES=32 SEQLEN=1024 COST_MODE=local DEVICE=cuda EXPORT_DEVICE=cuda \
  bash prismaquant/run-pipeline.sh

# --- arm C: CB container, uniform FP8_CB_K36 -------------------------------
MODEL_PATH=/home/rob/models/Qwen3-0.6B \
WORK_DIR=/home/rob/dq-runs/fc45-0p6b-fp8cb \
EXPORT_CONTAINER=nvfp4_cb TARGET_PROFILE=nvfp4_cb \
PRODUCTION_CACHE=0 PRODUCTION_RECACHE=0 \
FORMATS=FP8_CB_K36 TARGET_BITS=4.5 PARETO_TARGETS=4.5 \
NSAMPLES=32 SEQLEN=1024 COST_MODE=local \
CB_CODEBOOK_SOURCE=lattice CB_SCALE_SWEEP=1 CB_SCALE_CODING=two_tier \
CB_LADDER_INTERP=0 DEVICE=cuda EXPORT_DEVICE=cuda \
  bash prismaquant/run-pipeline.sh
```

(The launchers are checked in at `<work_dir>/run_armN.sh` / `run_armC.sh`.)

## 4. Bytes

| arm | measured payload | bytes |
|---|---|---:|
| native NVFP4 | model artifact | 870,290,032 |
| FP8-CB K36 | model artifact + `cb_codebooks.pqcb` | 871,628,664 |

FP8-CB is 1,338,632 B (+0.154%) larger. Since the preregistered exact-rate
tolerance is <=0.1%, neither timing nor quality result from this pair may be
reported as the formal same-rate winner.

## 5. KL-vs-BF16 and PPL

RESULTS PENDING.

## 6. Verdict

No formal winner can be declared from this pair: it misses exact-byte matching
and compares W4A4 with A8. Preserve any measured timing as approximate
full-contract evidence and rerun exact-byte endpoints before promotion.

## 7. Ladder economics — two claims verified in code

### (a) `PRISMAQUANT_EXPORT_REUSE_PRIOR` — does a changed `layer_config` re-encode only the changed units?

> Historical audit, quarantined 2026-07-31. Current HEAD rejects both unbound
> delta reuse and streaming resume and requires a fresh output directory. The
> mechanism below describes the superseded implementation; none of its timing
> or safety claims are release evidence until reuse binds the exact source,
> imatrix, codebook content, exporter ABI, and every copied tensor.

**Yes — with four caveats that materially narrow the ladder's "partial
re-export" economics.** All line numbers in
`prismaquant/export_nvfp4_cb_streaming.py` unless noted.

The mechanism is a per-target eligibility gate, `_cb_reuse_reason` (`:518-546`):
a CB target is byte-copy eligible **iff** the prior artifact assigns it the same
`format` (`:528`), the same scheme signature `{grid, mode, k, n_sub, type_size,
codebook_ref, scale_coding}` (`:530-538`), a byte-identical codebook
(`:539-540`), and already holds every planned output tensor at exactly the
planned dtype+shape (`:541-545`). Eligible targets are written with
`copy_src=prior.raw_slice(name)` (`:994-1002`), which the writer streams as raw
bytes straight from the prior shard with no encode and no torch round-trip
(`:305-329`). Ineligible targets fall to the normal encode branch and are
counted by reason (`:1021-1027`: `format_changed`, `scheme_changed`,
`not_in_prior`, `codebook_mismatch`, `tensor_missing`, `dtype_shape_mismatch`).
Stock-CT rungs (NVFP4 / FP8_DYNAMIC) have their own gate — prior must hold every
planned output at the exact dtype+shape (`:1064-1074`) — which is sound because
RTN is deterministic and NVFP4's only cross-tensor input, the fused-group shared
global, is pinned identical by the union-find coherence invariant (`:1055-1063`).
A mandatory pre-write verification re-encodes `--reuse-verify` (default 3,
`PRISMAQUANT_EXPORT_REUSE_VERIFY`) deterministically-sampled copy targets and
aborts on any byte mismatch (`:559-602`).

So a CB→NVFP4 substitution on unit *u* re-encodes *u* (its CB group entry is
gone ⇒ `not_in_prior`; on the stock side its `weight_packed` is absent ⇒
`stock_not_in_prior`) and byte-copies everything else. **That is the claim, and
it holds.** The caveats:

1. **It exists only in the STREAMING exporter.** `prismaquant/export_nvfp4_cb.py`
   contains the string "reuse" zero times, and so does
   `prismaquant/export_native_compressed.py`. `run-pipeline.sh` selects the
   streaming module only when the SOURCE tree is ≥ `EXPORT_STREAMING_THRESHOLD_GB`
   (default **80 GB**) under `EXPORT_STREAMING=auto` (`run-pipeline.sh:1908-1926`).
   A 27B bf16 source is ~54 GB ⇒ non-streaming ⇒ **no reuse at all** unless the
   ladder explicitly sets `EXPORT_STREAMING=1`.
2. **Setting `EXPORT_REUSE_PRIOR` on a sub-80 GB source hard-fails.**
   `run-pipeline.sh:1953-1959` appends `--reuse-prior` to `CB_EXPORT_ARGS`
   regardless of which module was selected, and `export_nvfp4_cb.py`'s argparse
   (`:819-840`) has no such flag ⇒ `unrecognized arguments: --reuse-prior`. The
   ladder must set `EXPORT_STREAMING=1` *and* `EXPORT_REUSE_PRIOR` together.
3. **Learned codebooks defeat it.** `group_cb_ok` is computed once per
   `(codebook_ref, format)` by byte-comparing this run's serialized codebook
   against the prior sidecar, and the code says it plainly: *"A group whose
   codebook differs makes every target on it re-encode"* (`:897-910`). With
   `CB_CODEBOOK_SOURCE=learned`, moving units out of a role changes that role's
   training set ⇒ new codebook ⇒ the **whole role** re-encodes. With `lattice`
   (the pipeline default and what the 27B/35B/Hy3 artifacts shipped) codebooks
   are deterministic and this never fires.
4. **The safety net is 3 samples and a warning.** A prior built from a different
   calibration only produces a WARNING, not an abort (`:571-576`) — the copied
   bytes then rest on the default 3-target verification sample.

**Net for the ladder:** with `EXPORT_STREAMING=1`, `EXPORT_REUSE_PRIOR=<base>`
and `lattice` codebooks, each interior rung really is O(flipped units) of encode
plus an O(artifact) byte copy. Nothing equivalent exists on the native side, so
the native endpoint and any native-container variant are always full re-exports.

### (b) Mixed-container delegation — do substituted stock-NVFP4 groups reach vLLM's native path?

**Yes, and it is verified in-container.** External Gridbook `gridbook/config.py`
splits `config_groups` on the presence of a `"scheme"` key — CB groups have it,
stock compressed-tensors groups do not (`:161-172`) — and builds a **real
`CompressedTensorsConfig`** over the stock groups, re-keyed to
`quant_method="compressed-tensors"` with every CB target added to CT's ignore
list and a valid `mixed-precision` top-level format (`:196-214`). At dispatch,
a Linear that resolves no CB scheme and is not explicitly ignored is handed
straight to `self.ct_config.get_quant_method(layer, _canonical_prefix(prefix))`
(`:350-355`); embeddings (`:357-362`) and stock FusedMoE stacks (`:366-374`)
delegate the same way, and `apply_vllm_mapper` keeps the delegated config's
namespace in lockstep (`:408-425`). The exporter is the other half: stock rungs
are written with the exact CT scheme vocabulary and **no** `"scheme"` key —
"the presence of a `scheme` key is the CB-vs-stock dispatch marker"
(`export_nvfp4_cb.py:706-735`).

Verified by running Gridbook's `tests/test_delegation.py` inside
`vllm-node:latest` (vLLM `0.23.1rc1.dev764`) — 5/5 pass, including
`test_ct_owns_stock_and_ignores_cb`, which resolves a stock prefix through
vLLM's own `find_matched_target`. Pushing one step further, asking the delegated
config to build the scheme for the stock NVFP4 group reaches vLLM's **NVFP4
kernel selector** and fails only for lack of a CUDA driver in that CPU-only
container:

```
ct_config: CompressedTensorsConfig | CT ignore includes CB target: True
CT target_scheme_map keys: ['re:.*mlp.gate_proj$', 're:.*mlp.up_proj$', 're:.*self_attn.*_proj$']
    re:.*mlp.gate_proj$ -> ValueError Failed to find a kernel that can implement the NVFP4 linear layer
```

i.e. the substituted group is being priced by vLLM's stock NVFP4 kernel
registry, not silently dropped to an unquantized or Triton fallback. **The speed
benefit is architecturally real**: the ladder can serve every rung — endpoints
included — through one gridbook container.

**Not verified here:** that on a live Blackwell GPU the selector lands on the
CUTLASS block-scaled kernel rather than a slower NVFP4 path, and the resulting
tok/s. A GPU container could not be launched — the box's GPU-exclusivity guard
refuses a second `--gpus` container while the serve on :8000 holds the pool:

```
GPU-exclusivity guard: serving container [pq_laguna 0.0.0.0:8000->8000/tcp] is running and
holds most of the 128 GB unified pool. ... Stop the serve first
```

That check belongs to the same box window as the ladder's timing readouts.

## 8. Command provenance

See §3 for the builds and §5 for the measurements. The original one-offs copied
the then-vendored runtime into mutable snapshots; those commands were removed
when the two-tree design was retired. Git history preserves them for forensic
reconstruction. A current rerun must use the immutable external pin helper and
record Gridbook's resolved commit in the serve fingerprint.

## 9. Blocked / not done

RESULTS PENDING.
