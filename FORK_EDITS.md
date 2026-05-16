# Fork Edits

Changes made to the upstream PrismaQuant codebase during the Qwen3.6-35B-A3B pipeline run,
with rationale for each.

---

## 1. Packed MoE expert coverage false-positives

**Files changed:**
- `prismaquant/build_production_cache.py`
- `prismaquant/production_weight_cache.py`
- `prismaquant/export_native_compressed.py`

### Background

Qwen3.5/3.6 MoE models contain packed expert tensors (`mlp.experts.gate_up_proj`,
`mlp.experts.down_proj`). These are 3-D weight tensors whose modules are not `nn.Linear`
instances. The production weight cache is built over `nn.Linear` modules only (`eligible_qnames`).
The allocator's `render_assignment` dict, however, contains entries for *all* quantizable
parameters including these packed expert tensors — 190 extra entries on Qwen3.6-35B-A3B.

The export pipeline handles packed expert tensors directly via `_quantize_2d`, bypassing
the production cache entirely. The cache coverage checks in three places did not account
for this, treating all 190 expert entries as cache misses and aborting with false-positive
failures.

### Changes

**`build_production_cache.py` (coverage check, ~line 355)**

Before checking cache coverage, filter `render_assignment` down to `eligible_set = set(qnames)`
(the `nn.Linear` qnames actually rendered into the cache). Print a count of skipped expert
entries so the distinction is visible in the log.

**`production_weight_cache.py` (metadata + prefetch, ~lines 3381 and 3391)**

1. Store `eligible_qnames` (sorted list of the `nn.Linear` qnames rendered) in the cache
   metadata blob so downstream consumers can read which qnames the cache covers without
   having to re-derive it.

2. Filter the `prefetch_assignment` call to `eligible_qnames` so the prefetch pass does not
   emit "WARNING: production cache missing N assignment entries" for expert qnames it was
   never asked to cache.

**`export_native_compressed.py` (startup coverage check, ~line 5895)**

Filter `_assignment_for_cache` to the eligible set before calling
`_production_cache_expected_keys`. The eligible set is read from `cache.metadata["eligible_qnames"]`
(written by the change above). For caches built before that metadata key was added, the code
falls back to inferring the eligible set from the cache's own `weights` keys. If neither is
available it falls through to the original unfiltered check for safety.

---

## 2. fla backward pass on GB10 Grace Blackwell (SM 12.1): three issues

**Files changed:**
- `/tmp/fla-src/fla/utils.py` (editable fla install, not in this repo)
- `/tmp/fla-src/fla/ops/common/backends/tilelang/__init__.py` (editable fla install)
- `prismaquant/streaming_model.py`

### Background

Three independent bugs caused `Triton Error [CUDA]: misaligned address` (surfaced
asynchronously as a sticky CUDA context error) in SSM backward passes on the GB10 Grace
Blackwell chip (SM 12.1).

**Issue 1 — `IS_NVIDIA_BLACKWELL` detection: `== 10` is correct but SM 12.x needs its own allocator**

`fla/utils.py` computes:
```python
IS_NVIDIA_BLACKWELL = (IS_NVIDIA and torch.cuda.get_device_capability()[0] == 10)
```
The GB10 chip reports SM `(12, 1)`. The `== 10` check is intentionally exact — fla's
Blackwell-specific kernel paths (tmem, alternative layouts) target SM 10.0 and cause
genuine CUDA misaligned address on SM 12.x. `IS_NVIDIA_BLACKWELL` must remain `False`
for SM 12.x. However, fla's scratch allocator registration (in the `elif IS_NVIDIA_BLACKWELL`
block) also never fires for SM 12.x. In Triton 3.6.0, `NullAllocator` raises
`RuntimeError: Kernel requires a runtime memory allocation` for any autotuned kernel that
requests global scratch.

The existing block also uses `triton.set_allocator`, which stores the allocator in a Python
`ContextVar`. PyTorch's C++ autograd worker threads receive an empty `contextvars` context
and see `NullAllocator` regardless of what the main thread registered.

**Issue 2 — TileLang backend activated on non-Hopper hardware**

`fla/ops/common/backends/tilelang/__init__.py` defines a `TileLangBackend` with
`default_enable = True`, so it is attempted on every GPU. It was added solely to work around
Triton ≥ 3.4.0 regressions on Hopper (SM 9.x, fla issue #640). On SM 12.1 the TileLang
`chunk_bwd_dqkwg_tilelang` kernel fails with `CUDA_ERROR_MISALIGNED_ADDRESS`. Because CUDA
kernel launches are asynchronous, this error propagates silently and only surfaces when the
next synchronous CUDA operation (e.g., `cuModuleLoadDataEx` inside Triton's `load_binary`)
is called — making it appear to come from an unrelated kernel.

The Triton implementation of `chunk_bwd_dqkwg` works correctly on SM 12.1 and was confirmed
with a minimal isolated reproduction (`CUDA_LAUNCH_BLOCKING=1`).

**Issue 3 — CUDA mask causing cascading import bugs**

`_mask_cuda_queries_during_meta_init` patches `torch.cuda.is_available → False` during
`init_empty_weights()` to prevent slow NVML probes. This caused:
- `@lru_cache` poisoning in `transformers.utils.import_utils.is_torch_cuda_available()` and
  `is_causal_conv1d_available()`, permanently caching `False` → SSM fast-path disabled
- fla imported under the mask with CPU-only constants (`device_torch_lib = torch.cpu`) →
  `AttributeError: module 'torch.cpu' has no attribute 'device'` at runtime

### Fix

**`fla/utils.py` — one change**

New `elif IS_NVIDIA and not IS_NVIDIA_BLACKWELL and ... >= 10` branch added after the
existing `elif IS_NVIDIA_BLACKWELL` block. Registers only the scratch allocator for SM 12.x
without activating any Blackwell-specific code paths.

Both the new `_SM12PlusAllocator` and the existing `_BlackwellAllocator` patch
`triton.runtime._allocation._allocator` as a module attribute directly instead of using
`triton.set_allocator` (ContextVar). Triton's `driver.py` reads `_allocator` as a dynamic
attribute at each call site, so the replacement is visible to all threads including C++
autograd workers. Both allocators maintain a persistent growing CUDA `uint8` buffer to avoid
per-call lifetime hazards.

**`fla/ops/common/backends/tilelang/__init__.py` — one change**

`chunk_bwd_dqkwg_verifier` gains a hardware guard as its first check:
```python
if not (IS_NVIDIA_HOPPER and TRITON_ABOVE_3_4_0):
    return False, "TileLang backend only needed on Hopper+Triton≥3.4; ..."
```
On non-Hopper hardware (including SM 12.1) the verifier rejects immediately, the dispatch
falls through to the Triton implementation, and `chunk_bwd_dqkwg` succeeds. On Hopper with
Triton ≥ 3.4.0, TileLang is still selected as intended.

The verifier also gained two additional guards that prevent silent misuse on Hopper itself:
`g.dtype != float32` (TileLang's kernel requires float32 for the gate tensor) and the
existing `h.dtype != q.dtype` check.

**`streaming_model.py` + `test-pipeline.sh`** — disable the CUDA mask

`PRISMAQUANT_MASK_CUDA_DURING_META_INIT` default changed from `"1"` to `"0"` in the
function, and `export PRISMAQUANT_MASK_CUDA_DURING_META_INIT=0` set in `test-pipeline.sh`.
The mask body is preserved for systems that genuinely need it — re-enable with
`PRISMAQUANT_MASK_CUDA_DURING_META_INIT=1`.

---

## 3. Phase-1 OOM kill on Qwen3.6-27B (dense): activation triple-copy + autoscale gap

**Files changed:**
- `prismaquant/incremental_probe.py`
- `prismaquant/autoscale.py`

### Background

On Qwen3.6-27B (64 layers, hidden=5120, N=32, T=1024, BF16) the pipeline was killed by
the kernel OOM killer during or immediately after the phase-1 forward pass on the GB10
Grace Blackwell (121 GB UMA). Two independent bugs combined to cause this.

**Bug 1 — triple copy of phase-1 activations**

The v22 Fix E1 batch device→host transfer at the end of phase-1
(`incremental_probe.py`, ~line 1172):

```python
stacked = torch.stack(device_acts, dim=0).cpu()
activations_cpu = [stacked[i].clone() for i in range(stacked.size(0))]
del device_acts, stacked
```

creates three simultaneous full copies of all 65 layer activation tensors:

| Allocation | Size |
|---|---|
| `device_acts` list (accumulated during loop) | 20.3 GB |
| `stacked` temp (torch.stack result, before .cpu()) | 20.3 GB |
| `stacked` host copy (from .cpu()) | 20.3 GB |
| `activations_cpu` (from .clone() × 65) | 20.3 GB |
| **Peak** | **60.9 GB** |

Combined with the layer cache holding all 64 layers (51.7 GB) and Python/CUDA overhead
(~8 GB), peak UMA usage was **120.6 GB** — only 0.4 GB margin on 121 GB. Any CUDA
context variation pushes it over.

**Bug 2 — autoscale headroom doesn't account for phase-1**

`pick_cache_headroom_gb` computed headroom as `shard_working + safety` (37.9 GB for
27B with LPS=4). This gave a cache budget of ~76 GB. For a 51.7 GB model the cache
fills entirely, leaving no room for the phase-1 activation stash.

### Changes

**`incremental_probe.py` (~line 1172)**

Replace `torch.stack(...).cpu()` + `[stacked[i].clone() ...]` with a per-element
`.cpu()` list comprehension. Peak activation memory drops from 60.9 GB to 40.6 GB
(device_acts + host copy being built simultaneously, no extra stack or clone):

```python
activations_cpu: list[torch.Tensor] = [t.cpu() for t in device_acts]
del device_acts
```

**`autoscale.py` — `pick_cache_headroom_gb`**

Headroom now accounts for two non-overlapping phases and takes the binding constraint
(max of both):

- **Phase-1**: `2 × (n_layers+1) × N × T × hidden × dtype_bytes` + fixed + safety
  (factor-2 = device_acts still alive while host copy is building)
- **Shard phase**: `shard_working + fixed + safety`

For 27B this changes cache headroom from 37.9 GB to 75.6 GB (binding: phase-1), giving
a cache budget of ~42 GB. The model partially streams (42 of 51.7 GB fits), which is
the intended streaming behavior. Combined peak: 97.8 GB — 23 GB margin on 121 GB UMA.

---

## 4. test-pipeline.sh vs prismaquant/run-pipeline.sh

`test-pipeline.sh` in the repo root is a local test harness derived from the canonical
`prismaquant/run-pipeline.sh`. Differences:

### Config defaults

| Variable | `run-pipeline.sh` | `test-pipeline.sh` |
|---|---|---|
| `MODEL_PATH` | commented-out usage example | hardcoded Qwen3.6-35B-A3B snapshot path |
| `WORK_DIR` | commented-out usage example | `./dq-runs/Qwen36-35B-A3B-Prism` |
| `VISUAL_FORMAT` | `BF16` | `NVFP4` |
| `DATASET` | `/home/rob/dq-runs/calibration/diverse-v1.jsonl` | `./dq-runs/calibration/diverse-v1.jsonl` (relative) |

### Removed: CUDA guard checks

`run-pipeline.sh` contains an early guard that exits 2 if `DEVICE`/`EXPORT_DEVICE` are not
`cuda*` and a Python snippet that checks `torch.cuda.is_available()`. These are removed from
`test-pipeline.sh` as they are redundant with the pipeline's own runtime checks and add
startup latency.

### Removed: `PRISMAFISHERCLIP` and `AWQ` levers

`run-pipeline.sh` includes env-var handling for `PRISMAFISHERCLIP`, `PRISMAFISHERCLIP_MODE`,
and `AWQ` with corresponding `LEVER_CACHE_TAG` and export-arg wiring. These are not used in
the current test configuration and are omitted from `test-pipeline.sh`.

### Added: timing and compression summary

`test-pipeline.sh` records `start_time_seconds` at the top and prints a summary block at the
end:

```
----------------------------------------
END TIME:   2026-05-12 14:37:02
ORIGINAL:   69.50 GiB (safetensors)
COMPRESSED: 18.32 GiB (safetensors)  →  3.79x compression  (saves 51.18 GiB)
TOTAL TIME: 04:12:07
========================================
```

Two helpers implement this:
- `_sum_safetensors_bytes DIR` — sums byte sizes of all `*.safetensors` files under a tree
- `_fmt_bytes N` — formats a byte count as TiB / GiB / MiB / B

The original model size is snapshotted immediately after `mkdir -p` (before any pipeline
stage runs). The exported size is measured after the export stage completes.

### Added: interactive allocator prompts

After the allocator runs, `test-pipeline.sh` parses its log for two interactive checkpoints
that let the operator adjust parameters without restarting from scratch:

**Prompt 1 — knee-point / target-bits**

If the allocator suggests a knee point that differs from `TARGET_BITS`, a box is displayed
showing both values and offering `[Y]es / [N]o / [I]nput` to accept the suggestion, keep the
current value, or enter a custom bpp. If the user accepts or enters a new value, the
allocator is re-run immediately with the updated `TARGET_BITS` before export begins.

**Prompt 2 — format-leg usage**

If any requested format received zero layers from the allocator, a box is displayed showing
per-format layer counts and offering `[Y]es / [D]rop / [I]nput` to proceed as-is, drop
zero-layer legs and re-run the allocator, or enter a custom format list. This prevents
spending export time on format legs that the allocator elected against.

A shared `_run_allocator()` helper re-runs the allocator with the current
`TARGET_BITS`/`FORMATS`/`VISUAL_SENSITIVITY` values and is called from both prompts.

---

## 5. Qwen3-Coder-Next profile

**Files changed:**
- `prismaquant/model_profiles/qwen3_next.py` (new)
- `prismaquant/model_profiles/base.py`
- `prismaquant/model_profiles/registry.py`
- `prismaquant/model_profiles/__init__.py`
- `prismaquant/streaming_model.py`
- `prismaquant/export_native_compressed.py`

### Background

`Qwen/Qwen3-Coder-Next` (`model_type: "qwen3_next"`) is a hybrid
Gated Delta Networks + sparse MoE architecture. It has two layer types:

- **DeltaNet linear-attention** (`linear_attention`): layers 0,1,2,4,5,6,...
  `model.layers.X.linear_attn.{in_proj_qkvz, in_proj_ba, out_proj}`
- **Standard GQA full-attention** (`full_attention`): every 4th layer (3,7,11,...)
  `model.layers.X.self_attn.{q|k|v|o}_proj`

All 48 layers include a shared MoE block with 512 experts.

**Key mismatch — checkpoint vs. live HF model:**

The checkpoint stores per-expert 2D tensors:
```
model.layers.X.mlp.experts.N.{gate|up|down}_proj.weight  (512 experts × 48 layers = 73,728 tensors)
```

But the HF model (`Qwen3NextExperts`) holds 3D packed `nn.Parameter` directly:
```
model.layers.X.mlp.experts.gate_up_proj  [512, 1024, 2048]
model.layers.X.mlp.experts.down_proj     [512, 2048, 512]
```

This mismatch caused `_fast_install` in the streaming loader to fail: the install
resolver (built from `named_parameters()` on the meta skeleton) only knows the packed
keys, but the per-expert checkpoint keys are not found in the resolver, causing a
fallback to `set_module_tensor_to_device` which then fails with `AttributeError`
trying to navigate to `experts.0` as a submodule.

### Changes

**`model_profiles/base.py` — new `pack_checkpoint_expert_tensors` default**

Added a no-op `pack_checkpoint_expert_tensors(layer_prefix, tensors) -> dict`
method to `ModelProfile`. Called by the streaming loader after
`_read_layer_to_device`; architectures that pack on the fly override it.

**`model_profiles/qwen3_next.py` — new profile**

`Qwen3NextProfile` implements:
- `matches()`: catches `model_type="qwen3_next"` and `Qwen3Next*` architectures.
- `fused_sibling_group()`: uses vLLM's `packed_modules_mapping` with a fallback
  dict that covers `qkv_proj`, `gate_up_proj` (shared expert), and the DeltaNet
  identity singletons `in_proj_qkvz` / `in_proj_ba`.
- `packed_expert_param_names()`: returns `frozenset()` — the checkpoint has
  per-expert 2D tensors, not 3D packed. The validate check inspects checkpoint
  keys, so returning empty avoids a false-negative. The export path detects 3D
  params on the live model via `_is_packed_experts_module` independently.
- `pack_checkpoint_expert_tensors()`: packs 512×{gate|up}_proj → `gate_up_proj`
  [E, gate+up, in] and 512×down_proj → `down_proj` [E, in, out].
- `per_expert_moe_regex()`: catch-all regex for compressed-tensors scheme dispatch.
- `split_packed_experts_for_format()`: always True (vLLM expects per-expert layout).
- `has_mtp()`, `per_expert_mtp_regex()`, `source_passthrough_prefixes()`: all no-op.

**`streaming_model.py` — profile-driven packing in hot paths**

`StreamingContext` gains a `profile` field. `_build_streaming_context` calls
`detect_profile(model_path)` and stores it on the context. Both `_prefetch_worker`
and `ensure_loaded` call `profile.pack_checkpoint_expert_tensors(prefix, tensors)`
immediately after `_read_layer_to_device` so cached tensors always use the packed
keys the install resolver expects.

**`model_profiles/registry.py` + `__init__.py`** — register `Qwen3NextProfile`
before the Qwen3.5 family (distinct `model_type` avoids any overlap).

**`export_native_compressed.py` — `_allocator_target_profile_for_audit`**

Added `"qwen3_next"` to the set that maps to `"vllm_qwen3_5_packed_moe"`, so the
BF16 audit knows to use the MoE-aware allocator profile for format coverage checks.

---

## 6. Streaming production cache for models too large for CUDA UMA

**Files changed:**
- `prismaquant/build_production_cache.py`

### Background

Qwen3-Coder-Next (79.67B params) is approximately 159 GB in BF16, larger than the GB10
Grace-Blackwell's 121.63 GB CUDA-visible UMA. The standard
`AutoModelForCausalLM.from_pretrained(device_map="cuda")` path triggers transformers'
`caching_allocator_warmup` which attempts to pre-allocate
`min(model_bytes, cuda_total - 1.2 GB) = 120.43 GiB` in a single `torch.empty(fp16)`,
immediately OOM-ing with 118.94 GiB free.

Even without the warmup, a 159 GB model cannot be fully resident in 121 GB.

### Solution: streaming production cache

`_fill_production_cache_streaming` replaces the full-model load for models that exceed
`cuda_total - 4 GB`. It reuses the existing streaming infrastructure
(`_build_streaming_context`, `_call_layer`, `LayerCache`) already used by
the Fisher probe:

1. **Build skeleton**: same `init_empty_weights()` meta-skeleton approach as incremental_probe.
   Only the always-resident head modules (embed_tokens, norm, lm_head, rotary) go on GPU.
   Peak GPU: ~0.5 GB.

2. **Build qnames from skeleton**: `iter_quantizable_tensors(skeleton)` yields the same
   `nn.Linear` modules as on the live model (meta tensors have correct shapes). The
   `qname_to_module` dict stores direct module references that remain valid through
   `install(L)` / `unload(L)` cycles (module identity is stable, only `.weight` attribute changes).

3. **Per-layer streaming loop** (`for L in range(num_layers)`):
   - `ctx.install(L)`: swap layer L weights from disk → GPU (~3–4 GB for Qwen3Next)
   - Run all N calibration samples through layer L: `_call_layer(layers[L], hidden[i], ...)`
     — activation hooks installed on the skeleton fire and collect per-qname tensors.
   - Compute NVFP4 joint globals via `_compute_nvfp4_joint_global(skeleton, layer_assignment)`:
     works because layer L's modules have GPU weights while installed.
   - Compute per-qname `activation_max_abs` with sibling unification.
   - Render production weights while layer is installed: `render_production_weight(mod.weight.data, ...)`.
   - `ctx.unload(L)`: weights freed back to meta.
   - Free this layer's activation tensors from the collector.

4. **Hidden-state propagation**: `hidden_states[i]` accumulates the running output of all
   completed layers for sample i. Between layers, the total hidden-state buffer is
   `N × T × H × 2 bytes = 32 × 1024 × 2048 × 2 ≈ 134 MB` — trivial.

5. **Resume support**: checks `cache_dir/<qname>__<fmt>.pt` files before rendering
   (same as `fill_production_weight_cache`). Writes `activation_max_abs.json` sidecar after
   completion.

Peak memory:
- Standard path: 159 GB (OOM on 121 GB system)
- Streaming path: ~4 GB (one layer) + 134 MB (hidden states) + ~30 MB (activations) ≈ 4.2 GB

### Limitations

- HALO not supported (requires full model for Hadamard rotation).
- Recache pass (`--recache-layer-config`) not supported.
- AWQ, SmoothQuant, `fisher_gptq`, `fisher_clip` not supported.
- Default pipeline (`gptq,scale_sweep` on NVFP4) is fully supported.

### Detection logic

`main()` calls `_estimate_model_bytes_from_index(staged)` to read `total_size` from
`model.safetensors.index.json`. If `model_bytes > cuda_total - 4 GB`, streaming is
used automatically. No CLI flag required — the routing is transparent.
