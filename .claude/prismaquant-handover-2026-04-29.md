# PrismaQuant HALO / Quality-Wins Handover

Date: 2026-04-29

This repo is `/home/rob/prismaquant`. The active HALO implementation is in the sibling worktree `/home/rob/prismaquant-quality-wins`. Do not assume both worktrees contain the same code.

## Current State

- Main repo has launcher and DSv4 changes, but no `prismaquant/halo.py`.
- HALO code and tests live in `/home/rob/prismaquant-quality-wins/prismaquant/halo.py` and `/home/rob/prismaquant-quality-wins/tests/test_halo.py`.
- `pytest -q tests/test_halo.py` passes in the quality-wins worktree, but it only covers a tiny standard-transformer topology.
- The Qwen smoke launcher in the main repo maps `+halo` and `full` to `halo_mode=off`; only `solo_halo` currently enables `random`.
- The observed `full` run was not a HALO run. It was export with GPTQ plus scale-sweep and block-output-match flags.
- There are uncommitted edits in both worktrees. Treat them as user work; do not revert them.

## HALO Risks To Handle First

1. Tied embeddings: Qwen3.5-4B has `tie_word_embeddings=true` and baseline export does not emit `lm_head.weight`. HALO needs separate embedding and LM-head transforms after final norm gamma folding. Either explicitly materialize/untie `lm_head` and update config semantics, or reject HALO on tied-head models.

2. Topology coverage: current HALO only handles standard `self_attn.q/k/v/o` and dense `mlp.gate/up/down`. Qwen3.5 has many `linear_attn.*` layers. DSv4 uses `wq_a/wq_b/wkv/wo_a/wo_b`, hyper-connections, packed routed experts, shared experts, and skipped compressor paths. Do not enable HALO broadly until profile-specific coverage exists.

3. Dense QR fallback: `random_hadamard(d)` falls back to dense QR for non-power-of-two hidden sizes. Qwen3.5-4B hidden size is 2560, so this is already seconds and scales badly. Prefer a structured padded/block Hadamard path or explicitly restrict supported dims.

4. Export cache: per-layer export cache is not keyed by HALO mode/seed/topology. A cache hit bypasses layer load, HALO rotation, and quantization. Add cache fingerprinting or disable layer-cache reuse when HALO is active.

5. Launcher order: only fix the `+halo`/`full` mapping after the above guards are in place.

## Recent Cheap Numerical Wins

The quality-wins work focused on low-calibration numerical hygiene:

- Activation cache FP32 via `PRISMAQUANT_ACT_CACHE_FP32=1`.
- GPTQ damping sweep via `PRISMAQUANT_GPTQ_DAMP_SWEEP=1`.
- Activation clipping via `PRISMAQUANT_ACT_CLIP_QUANTILE=0.999`.
- Calibrated `input_global_scale` from activation cache.
- Fused-sibling unification for input and weight global scales.
- GPTQ plus per-group scale-sweep auto-enabled when activation cache is supplied.
- AWQ defaulted off because it worsened NVFP4 group quantization in bakeoffs.
- Batched NVFP4 GPTQ/scale-sweep path for same-shape linears.
- Experimental block-output match via `PRISMAQUANT_BLOCK_OUTPUT_MATCH=1`.

## Active POC Branch

Branch `feat/expert-balanced-calibration-poc` adds an additive two-step
proof of concept:

1. `prismaquant/expert_calibration_survey.py` runs a cheap router-only
   survey over candidate JSONL rows and emits per-sample `hits` shaped
   like `{router_qname: {expert_id: mass}}`.
2. `prismaquant/expert_calibration_select.py` greedily selects rows that
   improve fractional coverage of underrepresented router/expert pairs,
   with optional domain minimums.

Tests live in `tests/test_expert_calibration_survey.py` and
`tests/test_expert_calibration_select.py`. The POC is not wired into
`multi_chunk_probe.py` or any launcher yet.

Example two-step flow:

```bash
python3 -m prismaquant.expert_calibration_survey \
  --model /path/to/model \
  --dataset /path/to/candidates.jsonl \
  --output /tmp/router_survey.jsonl \
  --summary /tmp/router_survey.summary.json \
  --device cuda --dtype bf16 --max-length 2048

python3 -m prismaquant.expert_calibration_select \
  --survey /tmp/router_survey.jsonl \
  --budget 256 \
  --output /tmp/calibration_selected.jsonl \
  --summary /tmp/calibration_selected.summary.json
```

This is meant to address the current REAP weakness: if the calibration
mix rarely activates a niche expert, the saliency estimate for that
expert is low-confidence even when the allocator math is correct. The
next useful step is to run the survey on Qwen4B or another small MoE and
compare REAP prune choices from random calibration rows versus selected
rows.

## Mistral Medium 3.5 / vLLM Docker State

There is a separate worktree for Mistral Medium 3.5 support:

- Worktree: `/home/rob/prismaquant-mistral-medium`
- Branch: `feat/mistral-medium-35-support`
- Source model: `/models/Mistral-Medium-3.5-128B`
- Downloaded HF-format files only: `model-00001-of-00003.safetensors`
  through `model-00003-of-00003.safetensors`, tokenizer files, config,
  model index, model card, license, and auxiliary metadata.

The source checkpoint is dense text plus multimodal shell, not MoE:

- Outer `config.json` has `model_type=mistral3` and
  `architectures=["Mistral3ForConditionalGeneration"]`.
- Inner `text_config` has `model_type=ministral3` and 88 decoder layers.
- The model card states this is a dense 128B model with 256k context,
  multimodal image+text input, function calling, and per-request reasoning
  effort.
- The source weights are FP8 with static activation quantization metadata.
  The profile strips source `quantization_config` during text-only staging
  and drops `.weight_scale_inv` / `.activation_scale` from the BF16 probe
  skeleton; revisit `FP8_SOURCE` export if preserving source FP8 serving
  artifacts is needed.
- The model card confirms EAGLE is a separate draft model:
  `mistralai/Mistral-Medium-3.5-128B-EAGLE`. vLLM serving wires it with
  `--speculative_config`, not with keys in the base checkpoint config.
- The model card recommends vLLM nightly with `mistral_common >= 1.11.1`
  and `transformers >= 5.4.0`. Relevant vLLM flags include
  `--tool-call-parser mistral`, `--enable-auto-tool-choice`,
  `--reasoning-parser mistral`, `--max_num_batched_tokens 16384`, and
  `--max_num_seqs 128`.

Current branch-local changes:

- Added `prismaquant/model_profiles/mistral3.py`.
- Registered `Mistral3Profile` in `prismaquant/model_profiles`.
- Added `tests/test_mistral3_profile.py`.
- Focused tests passed:
  `python3 -m pytest -q tests/test_mistral3_profile.py tests/test_prismaquant_visual_phase2.py tests/test_dsv4_layer_streaming_rename.py`.
- `python3 -m prismaquant.model_profiles.validate --model /models/Mistral-Medium-3.5-128B`
  passes 6/7 checks on this host. The only failure is local vLLM registry
  resolution for `Mistral3ForConditionalGeneration`; the model card says
  this requires vLLM nightly, so re-run inside the refreshed
  `spark-vllm-docker` environment before changing the architecture name.

`/home/rob/spark-vllm-docker` was resynced from Eugr upstream. The old
snapshot was moved to `/home/rob/spark-vllm-docker.pre-v020-20260429-140449`.
The new checkout is clean at `87cb9f6e1e9dc2c702f4dbc904cf68c95deb7709`
on `main`, matching `origin/main`. The current GitHub
`prebuilt-vllm-current` release asset is
`vllm-0.20.1rc1.dev55+g3f1a4bb63.d20260429.cu132-cp312-cp312-linux_aarch64.whl`.

## More Low-Calibration PPL Ideas

These are the next pragmatic candidates for lowering perplexity without a new expensive calibration regime:

1. Per-linear "do no harm" gate: for each NVFP4 linear with cached activations, compare RTN, GPTQ, and GPTQ plus scale-sweep output MSE on the existing activation cache. Emit the best candidate. This catches cases where GPTQ or scale-sweep locally worsens a layer.

2. Activation-scale grid: current `input_global_scale` is max-based, optionally after clipping. For each fused sibling group, try a small multiplier grid around the calibrated scale, for example `0.80,0.90,1.00,1.10,1.25`, and pick the lowest cached-output MSE. This is cheap and directly targets A4 range error.

3. Targeted local reconstruct: wire `local_reconstruct.py` into the export path for the top critical units only. Use existing activation cache and optional Fisher detail to choose rowwise/groupwise weight clipping and small GPTQ-lite refinements, then write a sidecar consumed by export.

4. Fisher-weighted GPTQ for top units: if probe Fisher detail exists, use `X^T diag(g2) X` instead of plain `X^T X` for the most sensitive NVFP4 linears. Keep the plain path as default for the rest.

5. Precision denylist audit, not a blanket override: the allocator is dynamic, so do not hardcode BF16 just because a tensor "sounds sensitive." First report tensors whose cost/sensitivity coverage is missing, whose runtime compressed-tensors path is unsupported, or whose byte savings are negligible relative to observed PPL risk. Candidates to audit include routers, attention sinks, q/k/kv norms, `linear_attn.norm`, hyper-connection tensors, compressors/indexers, final norm, embeddings, and heads. Only make a hard denylist for unmeasured, unsupported, or empirically catastrophic cases.

6. Boundary-layer soft prior: first/last layers can be under-modeled by local proxy metrics, but this should not replace allocation. Start with a report or soft penalty/tie-break toward MXFP8/BF16 for first layer, last layer, and final attention outputs only when the size cost is small. Disable it if smoke validation shows the allocator already handles those layers cleanly.

7. Targeted block-output match: keep `PRISMAQUANT_BLOCK_OUTPUT_MATCH` but make it top-K/block-scoped and profile-aware. Include Qwen `linear_attn` and DSv4 blocks before using it for broad comparisons.

## Suggested Fix Order

1. Add HALO fail-fast guards for unsupported tied embeddings and profiles.
2. Add export-cache fingerprinting for quality-affecting flags.
3. Fix launcher `+halo`/`full` only after the guards exist.
4. Implement topology-specific HALO only one profile at a time.
5. Add one cheap PPL idea at a time, validating with the Qwen4B smoke ladder before DSv4.
