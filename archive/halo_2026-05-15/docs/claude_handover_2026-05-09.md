# Claude Handover: PrismaQuant / PrismaScout

Date: 2026-05-09

This handover is for continuing the PrismaQuant cleanup/refactor work from the current `origin/main`.

## Repository State

- Canonical branch: `main`
- Canonical remote: `origin/main`
- Current HEAD: `d5626e2 feat: support halo on qwen3.5 dense`
- Local branch cleanup is complete: only `main` remains locally.
- Remote branch cleanup is complete: only `origin/main` remains, plus `origin/HEAD`.
- Linked worktree cleanup is complete: only `/home/rob/prismaquant` remains.
- Scratch archive from deleted dirty worktrees:
  `/home/rob/prismaquant-branch-cleanup-20260509T170704Z`

Recent important commits:

- `d5626e2 feat: support halo on qwen3.5 dense`
- `3a04e0d fix: enforce runtime format legality in optimizers`
- `e3139fb fix: coerce unsupported mxfp8 export shapes`
- `9713ceb feat: export from production weight cache`
- `6bb7ad3 fix: prefetch validation assignment deltas`

## User Goals And Constraints

Primary goal: make PrismaQuant/PrismaScout shippable by generating high-quality mixed-format quantizations for arbitrary BF16 LLMs. We should maximize speed and minimize error using both format allocation and numerical cleanup.

Current supported production formats are intentionally narrow:

- `NVFP4`
- `MXFP8`
- `BF16`

Memory and scheduling constraints:

- This machine is UMA with 128 GB shared RAM. Keeping some data on CPU and other data on GPU is not a meaningful memory-tier strategy here.
- When memory pressure is a concern, build LRU caches with prefetch.
- Do not disable prefetch as a workaround. If prefetch is broken or too aggressive, fix or bound it.
- Using more memory is fine when it improves performance or reliability. Do not spend memory for no benefit.
- Avoid long-running brute-force paths unless they are clearly producing value.

Operational preference:

- User wants strategic work, not shortcut-driven shipping.
- We should ship only when the artifact and method are defensible.
- The next phase is a large cleanup/refactor, starting from the cleaned `origin/main`.

## Current Production 27B Artifact

Current 27B artifact:

`/home/rob/dq-runs/qwen36-27b-step00-production-cache-vllmlegal-export-main-20260509T073125Z/exported`

Source model:

`/home/rob/.cache/huggingface/qwen36-27b-bf16`

Selected candidate:

`/home/rob/dq-runs/qwen36-27b-center-polish-validation-main-20260509T063717Z/selected_step_00/layer_config_mtp_bf16.json`

Selection rationale:

- The seed candidate was the low-bpp pareto kneedle.
- Only the first n=4 polish move survived n=64 validation.
- Later n=4 polish moves worsened KL and were not shipped.

Validation numbers:

- Seed last-token KL: `0.03281387012110004`
- Step 00 last-token KL: `0.031157573805702388`
- Last-token improvement: `0.001656296315397654`
- Seed full-sequence KL at n=8: `0.03145481`
- Step 00 full-sequence KL at n=8: `0.03019302`
- Full-sequence improvement: `0.0012617958709597588`

Bitrate:

- Validator convention: `5.1643 bpp`
- Expanded export convention counting BF16 MTP tensors: `5.3410 bpp`

Final format breakdown from manifest:

- `linear/NVFP4_PRODUCTION_CACHE`: 398
- `linear/MXFP8_PRODUCTION_CACHE`: 14
- `linear/BF16`: 84
- `layer_passthrough/BF16`: 352
- `mtp_linear/BF16`: 8
- `mtp_passthrough/BF16`: 7
- `head_passthrough/BF16`: 3

Runtime legality:

- 10 small MXFP8 linears were coerced to BF16 because vLLM/FlashInfer MXFP8 requires larger supported shapes.
- The coercions are persisted in `mixed_native_manifest.json`.

Runtime validation:

- Docker image used: `vllm-fresh-b12x-fla:latest`
- Eager vLLM smoke passed.
- CUDA graph smoke passed with `enforce_eager=False`.
- vLLM used FlashAttention 2, FlashInfer NVFP4, FlashInfer MXFP8, and CUDA graph capture.
- Generated output was sane for prompt `The capital of France is`.

## Current Algorithmic Stack

The current proto-production path is:

1. Generate candidate frontier from the known-good pareto/knapsack allocator.
2. Measure candidate KL after production numerical cleanup, not raw RTN.
3. Use production weight cache where possible so measured weights and exported weights match.
4. Polish around good candidates, but accept only moves that survive real validation.
5. Export directly from production cache.
6. Validate the exported compressed-tensors artifact in vLLM.

Production numerical cleanup:

- GPTQ OBS-style rounding for NVFP4.
- Closed-form scale sweep for NVFP4, analogous to AutoRound's iterative scale/rounding search but enumerative and per-group.
- Do-no-harm gate against RTN.
- Calibrated activation input scales.
- Fused-sibling scale unification.

Important distinction:

- Allocation chooses formats.
- Production cache renders each assigned format with the numerical tricks that export will use.
- KL validation should be performed on production-rendered candidates when the candidate is intended for production export.

## HALO / Rotation State

Recent HALO changes in `d5626e2`:

- Non-power-of-two hidden sizes now work through structured block-Hadamard rotations.
- Qwen3.6 hidden size 5120 decomposes as `[4096, 1024]`.
- Qwen3.5/3.6 dense profile is now allowed by the HALO exporter guard.
- Qwen3.5/3.6 `linear_attn` projections are now included:
  - `linear_attn.in_proj_qkv`
  - `linear_attn.in_proj_z`
  - `linear_attn.in_proj_b`
  - `linear_attn.in_proj_a`
  - `linear_attn.out_proj`
- HALO metadata is persisted in `mixed_native_manifest.json` and `halo_rotation.pt`.

Important caveat:

- MTP is no longer a blanket HALO blocker, but MTP acceptance still needs separate validation under speculative decoding.
- Main target logits without speculative decoding are covered.
- MTP has its own final norm feeding the shared `lm_head`, so speculative acceptance must be checked if HALO is enabled for a model with MTP.

Relevant files:

- `prismaquant/halo.py`
- `prismaquant/export_native_compressed.py`
- `tests/test_halo.py`

## Qwen3.5-0.8B Smoke Target

Downloaded small proxy model:

`/home/rob/.cache/huggingface/qwen35-0p8b-bf16`

Sanity check:

- Size on disk: about `1.7G`
- Architecture: `Qwen3_5ForConditionalGeneration`
- Model type: `qwen3_5`
- PrismaQuant profile: `qwen3_5_dense`
- MTP: `True`
- Hidden size: `1024`
- Layers: `24`
- Layer pattern includes `linear_attention` and `full_attention`.

This is the recommended smoke-test proxy for Qwen3.6-specific infrastructure because no official tiny Qwen3.6 release appears to exist. Qwen3.5-0.8B shares the Qwen3.5/Qwen3.6 dense GDN/linear-attention conventions closely enough for infrastructure smoke tests.

## Tests Run Recently

After HALO/Qwen3.5 changes:

```bash
python3 -m py_compile prismaquant/halo.py prismaquant/export_native_compressed.py tests/test_halo.py
python3 -m pytest tests/test_halo.py tests/test_prismaquant_export_native_compressed.py::TestProductionCacheExportPath tests/test_prismaquant_export_native_compressed.py::TestRuntimeLegalAssignment -q
```

Result:

- `16 passed`

Earlier production-cache/export tests also passed before the latest HALO commit.

## Known Risks

1. The 27B artifact is validated and vLLM-loadable, but the HALO changes were not part of that artifact.
2. HALO for Qwen3.5/3.6 is now infrastructure-ready but needs an actual 0.8B smoke run before being trusted.
3. MTP plus HALO requires spec-decode acceptance validation, not just target-logit validation.
4. Full-sequence KL is more expensive and can become memory heavy. Use bounded caches and prefetch.
5. Production cache is load-bearing. Manifest/fingerprint integrity matters.
6. Some old experimental code remains in the repo and should be audited during cleanup rather than blindly deleted.

## Refactor Starting Recommendations

Start from high-impact, low-risk consolidation:

1. Centralize format legality.
   - The allocator, probe, optimizer, and exporter should all call the same legality API.
   - MXFP8 shape coercion should not be duplicated.

2. Centralize production rendering.
   - There should be one canonical path for "assignment -> production-rendered weight".
   - Export and KL validation should share this path.

3. Split orchestration from math.
   - Keep quantization math, candidate generation, validation, and export orchestration in separate modules.

4. Normalize artifact schemas.
   - Current schema names and aliases have drifted.
   - Consider a clean v2 manifest for KL probes/frontiers/polish results.

5. Preserve streaming and prefetch as first-class primitives.
   - Do not reintroduce all-in-memory paths as the default.
   - Big-model viability depends on bounded LRU and layer/block prefetch.

6. Clean old branches/experiments from code only after mapping references.
   - Branch refs and worktrees have been cleaned.
   - The codebase still contains historical pathways such as CLADO variants, adjoint L3, dense cone, QUBO-like refinements, etc.
   - Delete only after confirming they are not used by the current production stack or useful as tests/baselines.

## Suggested Next Smoke

Use Qwen3.5-0.8B to test HALO infrastructure before touching 27B:

1. Run a minimal no-HALO baseline export/validation.
2. Run `--halo-mode random --halo-seed 0`.
3. Compare:
   - export success
   - vLLM eager load
   - vLLM CUDA graph load if feasible
   - target-logit KL
   - MTP spec-decode acceptance if serving with speculative decoding
4. If HALO helps on the proxy, test it on a small validation slice of 27B before committing it to production artifacts.

## Practical Notes

- Disk is tight: about `52G` free after cleanup and Qwen3.5-0.8B download.
- The prior `/home/rob/prismaquant-kl-repro-guardrails` venv/worktree was deleted during cleanup. Use the main environment unless a new venv is needed.
- If a large run gets CPU-heavy with low GPU utilization, inspect cache/prefetch behavior first.
- If a job risks OOM, prefer streaming and bounded LRU over disabling useful cache behavior.
