# gridbook (in-tree copy)

Out-of-tree vLLM quantization plugin for the **NVFP4-CB / FP8-CB**
product-codebook formats. Weights ship as packed k-bit codeword indices over a
tiny flat codebook whose entries lie exactly on a hardware grid (E2M1 / E4M3),
so a decoded tile *is* a native tile and feeds the stock Blackwell tensor-core
GEMM. Zero vLLM-core patches.

This file is the **developer** view for people working inside `prismaquant`.
The user-facing docs (INSTALL / TROUBLESHOOTING / BENCHMARKS / SPEC / cards)
live in the separate public repo `github.com/RobTand/gridbook`, whose package
tree mirrors this one; the rest of the drift is doc-path strings.

**Versions.** PyPI serves **0.1.1**, and the standalone `/home/rob/gridbook`
repo is the **release source** — releases are cut there, never from this tree.
This in-tree copy is the **development head**: `__version__ = "0.2.0.dev0"`
(`gridbook/__init__.py:19`), a dev suffix that says truthfully "unreleased,
post-0.1.1" — it is ahead of the released package in kernel work (R6 smem LUT,
single-storage dense weights) and must never be read as a published version.

**Syncing the two trees.** `scripts/sync_gridbook.py` is the one-way path:
`plugins/gridbook/gridbook/` → `<release>/gridbook/` and `plugins/gridbook/tests/`
→ `<release>/tests/`, mirrored (deletions included), from **committed** content
only — in-flight kernel authoring in the working tree is excluded by
construction, since Robert's rule is that kernels join the release project
*when they're ready*. It never touches the release repo's distribution
scaffolding (`LICENSE`, `pyproject.toml`, `MANIFEST.in`, `Dockerfile`, `docs/`,
`.github/`), and it never commits, tags or publishes. Run
`python3 scripts/sync_gridbook.py --check` for a drift report (exit 1 on drift)
or without `--check` to apply; `tests/test_gridbook_sync.py` is the gate that
keeps the two trees from silently diverging again, and it skips — never passes —
when `GRIDBOOK_REPO` (default `/home/rob/gridbook`) is not present. After a sync,
commit in the release repo by hand; `__version__` moves with the sync, so both
trees always report the same version.

Format and kernel contracts: `docs/lanes/nvfp4-cb/STANDARDS.md` (authoritative),
byte layout `docs/lanes/nvfp4-cb/LAYOUT.md`.

## Install

| Route | Command |
|---|---|
| Published | `pip install gridbook` (PyPI **0.1.1**, released tokenlessly by the tag pipeline 2026-07-28 from `/home/rob/gridbook`) |
| In-tree, editable | `pip install -e plugins/gridbook --no-deps` |
| Serve container | the serve scripts copy the tree in and `pip install -e` it (`scripts/serve_laguna_smoke.sh:54-55`) |

The CUDA sources live **inside** the package (`gridbook/csrc`, shipped via
`[tool.setuptools.package-data]` in `pyproject.toml`) and JIT-build on first
model load; a repo-root-relative `csrc` broke every non-editable install before
`bf1ada0`. Without `nvcc` the plugin still serves correctly through Triton
fallbacks — a correctness path, not a speed path. `tp=1` only.

## Registration and dispatch

`register()` (`gridbook/plugin.py:129-151`) registers `PrismaQuantConfig` under
registry key **`gridbook`**, plus the legacy alias **`prismaquant`** that older
shipped artifacts carry in their `config.json`. Both dispatch to the same config
(`config.py:223`, `:242-250`). Registration is via the
`vllm.general_plugins` entry point (`pyproject.toml`).

`PrismaQuantConfig.get_quant_method` (`config.py:326-375`) resolves every module
in one place — **mixed containers are implemented, not stubbed**:

| Module | Condition | Method |
|---|---|---|
| `LinearBase` | prefix (or its fused shards) hits a config group carrying `"scheme"` | `PrismaQuantCBLinearMethod` (`config.py:339-346`) |
| `LinearBase` | prefix matches `ignore` | `UnquantizedLinearMethod` — BF16 (`:348-349`) |
| `LinearBase` | otherwise, a stock-CT group exists | delegated to a real `CompressedTensorsConfig` — NVFP4 W4A4 / FP8_DYNAMIC (`:350-354`, built at `:196-214`) |
| `VocabParallelEmbedding` | CT group else default | `:357-362` |
| `RoutedExperts` | expert targets under this prefix carry a CB scheme | `PrismaQuantCBMoEMethod` (`:366-371`) |
| `RoutedExperts` | otherwise | stock-CT MoE (`:372-373`) |

A group **without** a `"scheme"` key is stock compressed-tensors vocabulary; the
delegated CT config gets our CB targets added to *its* ignore list so the two can
never both claim a module (`config.py:161-171`, `:208`). Covered by
`tests/test_delegation.py` (mixed split, CB-precedence over the substring ignore
test, uniform-CB ⇒ no CT config, HunYuan shared-MLP collapsed-prefix aliases).

Namespace resolution (checkpoint vs canonical vs vLLM wrapper-class vintages,
plus `apply_vllm_mapper`'s fourth) has exactly one owner: `_candidate_bases` /
`shard_target_keys` (`config.py:72-89`, `:267-307`). A second hand-rolled copy is
what caused issue #1; do not add another.

## MoE serving

Supported and the main use — 3 of the 4 proven artifacts are MoE.
`PrismaQuantCBMoEMethod` (`gridbook/moe.py`) registers stacked expert buffers
(`w13_cb_qweight` `(E, 2·inter, bytes)`, `w2_cb_qweight` `(E, hidden, bytes)`,
plus fp8 per-channel `weight_scale`) matching the exporter's stacked layout
byte-for-byte, so loading is a `copy_` with no split and no transpose. One
uniform CB rung per layer (union-find at export; asserted here). Decode runs the
grouped CUDA GEMVs (one launch per projection over all routed *(token, expert)*
pairs); prefill runs the measured `auto` selector. fp4 experts require two-tier
v2 scale coding — fp4-v1 raises (`moe.py:112-117`).

### Per-arch loader wiring — the #1 support trap

Some MoE archs map experts at the **top-level** `load_weights` via a per-expert
`expert_params_mapping` and `continue` past `mlp.experts`, so they never call the
per-layer `FusedMoE.load_weights` our instance hook wraps. Our stacked-CB tensors
match neither their stacked nor their per-expert mapping. For those archs
`plugin.py` installs a thin top-level wrapper
(`install_toplevel_cb_expert_loader`, `moe_toplevel_loader.py`) that fills the
registered fused params and delegates everything else unchanged.

The opt-in is **data**: one vLLM *module path* per arch in
`_CB_TOPLEVEL_MODULE_PATHS` (`plugin.py:91-118`), fed to
`_install_on_module_classes` (`:38-63`), which discovers the entrypoint classes
each module *defines*. Module paths rather than class imports because class names
drift across vLLM versions (Qwen3.5 alone has ForCausalLM / MoeForCausalLM /
(Moe)ForConditionalGeneration), a missing module degrades to a no-op, and
over-installing is harmless. Adding an arch is **one line** — promote the tuple
to a JSON sidecar only if third parties need to extend it without patching.

Wired today:

| Arch | Module path |
|---|---|
| HunYuan V3 | `vllm.model_executor.models.hy_v3` |
| HunYuan V3 MTP drafter | `…models.hy_v3_mtp` |
| poolside Laguna S/XS 2.x | `…models.laguna` |
| Qwen3.5-MoE (+ MTP, + VL wrapper classes) | `…models.qwen3_5`, `…models.qwen3_5_mtp` |
| **DSv4-class** | **commented candidate** — `…models.deepseek_v4`, uncomment once the module exists in the target build (`:114-117`) |

**An unwired arch used to fail silently.** The engine booted, no warning was
emitted (vLLM does not warn on never-matched checkpoint tensors), the FusedMoE
params kept their initialization memory, and generation was garbage ("D D D…").
Cost on Laguna: one wasted boot plus an hour of dispatch theory when the answer
was one registry line. It is now a **hard serve-time failure**: `create_weights`
stamps `_pq_cb_filled = False` on `w13/w2_cb_qweight`, both fill paths (the
per-layer instance hook in `moe.py`, `moe_toplevel_loader.load_weights`) stamp
`True`, and `process_weights_after_loading` raises — naming the model class and
the module path to add — if a registered, non-empty stack was never filled
(`cb_fill_guard.py`). No env bypass; scoped to the params the local rank
registered, so an EP/PP-absent or zero-expert shard is skipped. A
`--load-format dummy` boot is not a supported CB path and will trip it.

The wrap itself is inert for non-CB checkpoints (it only fires on
`…experts.<proj>.cb_qweight`), so over-installing is harmless. Tests:
`tests/test_toplevel_loader.py` (routing, deferral, shared-MLP fuse, MTP spec-layer
rename, HF-mapper prefixes, idempotence) and `tests/test_cb_fill_guard.py`
(loader-not-installed ⇒ raises; installed ⇒ passes).

Bringing up a new arch also needs the pipeline-side profile + structure spec; see
the `gridbook-new-MoE-arch` checklist in auto-memory.

## Formats served

| Family | Rungs | Body rate | Scale coding |
|---|---|---|---|
| `NVFP4_CB_K{k}` | K12–K24, every integer | `k/8 + 0.5` bpw | two-tier v2 (E8M0 super + 4-bit sub, 0.28125 bpw); v1 E4M3 plane read-only |
| `NVFP4_CB_S{k}` | S13–S16 | `k/8 + 0.5` | research-only, menu-excluded; serving chain bit-exact |
| `FP8_CB_K{k}` | K28–K48, every integer | `k/8` | per-output-channel fp32, separate tensor |

Ladder pinned in `prismaquant/format_registry.py:943,947` and
`prismaquant/layer_config.py:34-39`. **Rung splitting is all-integer ceil-first**
(`_ceil_first_split`, `gridbook/expand.py:276-286` = the encoder's `_bit_split`,
`prismaquant/nvfp4_cb_formats.py:168-172`): sub-widths `base+1` for the first
`k mod n_sub` subs, sub-0 at the LSBs. Even splits are the special case, not the
scope. Signed mode is `n_sub == 1` (`expand.py:306`). Hard constraint
`in_features % 256 == 0`; anything else ships BF16.

## Kernel defaults (one line each)

| Regime | Default path | Code |
|---|---|---|
| Dense decode M ≤ 8 | CUDA GEMV — fp8 (double-buffered) or fp4-v2 | `linear.py:360-377`, `:46,53` |
| Dense decode M 9–16 | Triton `cb_gemm` | `linear.py:359,383` |
| Dense prefill fp8, M 17–128 | **CUTLASS fused decode-in-prologue, ON by default** (`PRISMAQUANT_CB_FUSED_MIDM`, default `"1"`); rungs k ∈ {28,32,36,40,44,48} | `linear.py:437-448` |
| Dense prefill fp8, M > 128 | `cb_expand_fp8` → stock `cutlass_scaled_mm` (transient `[N,K]` e4m3, freed per forward) | `linear.py:478-490` |
| Dense prefill fp4-v2 | transient bf16 expand → `F.linear` | `linear.py:391-406` |
| MoE decode ≤ 16 tokens | grouped CUDA GEMVs + deterministic combine | `moe.py:292-294` |
| MoE prefill fp8 | **`auto`** — per-layer cuda-event measurement over `stock` + `grouped_fused` at each compiled TileM, cached, deterministic on the tuning call | `moe.py:404-407`, `moe_autotune.py:110-157` |
| MoE prefill fp4 | `loop` (per-expert, one host sync) until the stock bf16-expand variant is measured at scale | `moe.py:404-405` |
| Codebook LUT residency | **R6**: staged smem-resident per (TileM, KBits) where headroom allows; k48 stages a half table, TileM=256/k32 stays global; `static_assert`ed per instantiation | `csrc/cutlass_fork/sm120_cb_fused_mma.hpp:141-176`, commit `1ede688` |
| fp8 dense GEMV schedule | double-buffer (bit-identical, +3–6%); `=legacy` for single-buffer | `csrc/cb_gemv.cu:470` |
| fp4-v2 dense GEMV schedule | single-buffer; `=db` is the opt-in that measured the loss | `csrc/cb_gemv.cu:759` |
| grouped w2 schedule | round-2 warp schedule (+50% on the Hy3 w2 shape, served-KL-validated); `legacy` / `rowpack` for bisection | `csrc/cb_gemv.cu:1418-1440` |
| Dispatch | M-branch hoisted into opaque custom ops so `torch.compile` never bakes the prefill path into the decode graph | `linear.py:340-352`, `moe.py:277-281`, `ops.py` |
| Decode contract | v1 (v2 measured null on the served 27B) | `csrc/cb_gemv.cu:198`, `:471` |

Not defaults: `grouped_fused` (won +9% on the 35B, **regressed** Laguna-class
1,503 vs 1,821 tok/s — promotion reverted under the two-model ladder rule,
`moe.py:371-377`); `l2_pipeline` (**diagnostic-only** — wedged the live serve
three times, `moe.py:399-403`); persistent-N dense TC (measured negative, kept
quarantined behind `PRISMAQUANT_ENABLE_PTC=1`, `linear.py:450-476`,
`cuda_ext.py:165`); `batched` MoE prefill (opt-in, crashed at 1.4k-token scale
on thin post-KV slack).

## Serve-time env

All switches are read host-side in launchers, so they are CUDA-graph-capture-safe.
The serve scripts pass them through explicitly (`scripts/serve_laguna_smoke.sh:43-52`).

| Variable | Default | Effect |
|---|---|---|
| `PRISMAQUANT_CB_PREFILL` | `auto` (fp8) / `loop` (fp4) | `stock` · `batched` · `grouped_fused[_r1]` · `l2_pipeline` · `loop` |
| `PRISMAQUANT_CB_PREFILL_AUTO_FORCE` | unset | pin an `auto` winner without timing (bisection) |
| `PRISMAQUANT_CB_AUTOTUNE_MIN_M` | 1024 | token floor below which `auto` runs stock and caches nothing |
| `PRISMAQUANT_CB_FUSED_MIDM` | `1` | `0` disables the mid-M fused prefill |
| `PRISMAQUANT_CB_DECODE` | `cuda` | anything else forces the Triton decode path |
| `PRISMAQUANT_CB_CUDA_M_MAX` / `PRISMAQUANT_PREFILL_M_THRESHOLD` | 8 / 16 | the two dense M gates |
| `PRISMAQUANT_CB_DISPATCH` | op | `inline` restores in-graph branching |
| `PRISMAQUANT_CB_FP8_SCHED` / `_FP4V2_SCHED` / `_W2_SCHED` / `_W2_WARPS` / `_W2_ROWS` | see table above | kernel schedule bisection |
| `PRISMAQUANT_CB_DECODE_CONTRACT` | `v1` | `v2` = scale-epilogue hoist (measured null) |
| `PRISMAQUANT_CB_PREFILL_DENSE` / `PRISMAQUANT_ENABLE_PTC` | off | quarantined persistent-TC dense prefill |
| `PRISMAQUANT_PRELOAD_FUSED` | off | force-build the fused ext so both arms of a served A/B carry identical extension residency (session-arithmetic drift) |
| `PRISMAQUANT_DEBUG_PREFIXES` | off | print every module's CB / no-scheme dispatch decision |

`PRISMAQUANT_CB_EXT_DIR` relocates the JIT build dir. Full flag list:
`grep -rn PRISMAQUANT_ gridbook/`.

## Invariants

- **INV-1 (held everywhere).** The resident weight is the packed index stream +
  the flat codebook + scales. Dense `[N,K]` weights exist only as per-layer or
  per-expert-chunk transients, freed each forward — never model-wide (the NVINT2
  OOM trap). The mid-M fused and grouped-fused paths decode inside the kernel and
  materialize nothing.
- **INV-2 (native MMA).** Held by the CUTLASS fused prefill and by
  expand-to-e4m3 → `cutlass_scaled_mm`; waived on the Triton decode-GEMM and the
  fp4-v2 bf16 expand (Triton cannot emit the sm_121 block-scaled FP4 MMA).
- **Numerics.** Identical weight rounding to the reference decode
  (`w = bf16_rn(codebook · scale)`), activation QDQ bit-exact to `codec.py`, fp32
  accumulation. CUDA differs from the reference only by summation
  **reassociation**, held to ≤1 bf16 output ULP plus a norm backstop
  (`_assert_triton_close`, `tests/test_cuda_gemv.py`). Any reassociating schedule
  change is gated by a served logprob/KL A/B before it becomes a default.

## Tests

`tests/` runs on CPU where it can and skips GPU-only cases. `test_cb_kernels.py`
(Triton path on real exported 0.6B tensors) · `test_cuda_gemv.py` (dense +
grouped-MoE fp8/fp4-v2, QDQ bit-exactness, expander) · `test_fused_prefill.py`,
`test_persistent_prefill.py`, `test_persistent_tc.py` (prefill kernels) ·
`test_moe_stacked.py`, `test_moe_batched_prefill.py`, `test_moe_stock_prefill.py`,
`test_moe_grouped_fused.py`, `test_moe_l2_pipeline.py` (MoE paths) ·
`test_delegation.py`, `test_target_namespace_compat.py`, `test_toplevel_loader.py`
(dispatch/loading) · `test_two_tier_v2.py`, `test_transient_fp8.py`,
`test_weight_residency.py`.

GPU test batteries run with the serve **stopped** — a `docker run --gpus`
alongside a live `pq_*` serve OOM-killed the box (a PreToolUse hook now denies it).
