# HALO 27B Attempt, 2026-05-09

## Summary

The first Qwen3.6-27B HALO seed-0 export attempt was invalidated before KL
measurement. The export used:

- source model: `/home/rob/dq-runs/qwen36-27b-untied-bf16`
- layer config: `/home/rob/dq-runs/qwen36-27b-center-polish-validation-main-20260509T063717Z/selected_step_00/layer_config_mtp_bf16.json`
- production cache: `/home/rob/dq-runs/qwen3p6-27b-kl-probe-triad-n64-production-20260508T032958Z-directpy/production_weight_cache_nvfp4_mxfp8.pkl`
- HALO: `--halo-mode random --halo-seed 0`
- runtime: `vllm-fresh-b12x-fla:latest`, CUDA, `NVIDIA GB10`

The exporter completed and vLLM loaded the artifact, but the greedy generation
smoke was incoherent:

```text
prompt: 'The capital of France is'
HALO output: '世yka faj最新更新 addTargetoniittenkestonetจรuristicoyaessakestoyaAILS'
```

The shipped no-HALO artifact loaded on the same vLLM path and produced the
expected coherent output:

```text
prompt: 'The capital of France is'
no-HALO output: ' Paris.\nThe capital of France is Paris.\nThe capital of France is'
```

The HALO checkpoint was deleted and the run directory was marked invalid:

`/home/rob/dq-runs/qwen36-27b-halo-step1-gpu-20260509T205742Z/HALO_EXPORT_INVALID.txt`

## Root Cause

`--production-weight-cache` replays already-rendered compressed weights in the
original no-HALO residual basis. HALO must rotate weights before NVFP4/MXFP8
rendering. Combining the two rotated norms and BF16 passthrough tensors while
leaving production-cached quantized Linears in the original basis, corrupting
the residual stream.

The exporter now rejects `--halo-mode random` with `--production-weight-cache`.

## Correct Next Measurement

HALO must be measured by rendering a HALO-specific production cache, or by
exporting from a HALO-rotated model with activation-aware passes recomputed on
that rotated basis. Reusing the shipped no-HALO production cache is not a valid
shared-control shortcut.

Until that path exists, the 27B HALO outcome is **not measured**. The observed
smoke failure should be treated as an invalid export path, not as evidence that
HALO itself regresses.
