# PrismaQuant Package Notes

This package README intentionally stays short. The user-facing overview lives in
the repository root `README.md`; the system map (stages, contracts, containers)
lives in `docs/ARCHITECTURE.md`.

`run-pipeline.sh` — the actual stage orchestrator — lives **in this package
directory**, not the repo root: `prismaquant/run-pipeline.sh`. `pipeline.py` is a
declarative contract layer, not the executor.

## CLI entrypoints (`python -m prismaquant.<module>`)

| Stage | Modules |
|---|---|
| Probe | `incremental_probe`, `kl_sensitivity_probe` |
| Cost | `incremental_measure_quant_cost`, `production_render_cost`, `aura_cost` (`COST_MODE=aura`), `expert_empirical_cost` (MoE hybrid), `aura_additivity_gate` (trust-region check) |
| Allocate | `allocator` |
| Cache | `build_production_cache`, `production_recache` |
| Select | `validate_assignments_kl`, `select_validated_frontier` |
| Export | `export_native_compressed` (compressed-tensors), `export_gguf` / `export_gguf_direct` (GGUF), `export_nvfp4_cb` / `export_nvfp4_cb_streaming` (codebook, served by `plugins/gridbook`) |
| Validate | `validation_harness`, `validate_native_export`, `validate_quantized_model` |

## Library modules with no CLI (imported by the stages above)

- `footprint` — exact per-tensor byte accounting; the byte-budget ("fit the
  card") selection target.
- `saturation_select` — saturation-point bit-rate selection.
- `gguf_formats`, `gguf_iq_formats`, `gguf_gptq` — GGUF k-quant / IQ quantizers
  and the GPTQ-under-frozen-scales lever.
- `nvfp4_cb_formats`, `nvfp4_cb_footprint` — product-codebook format specs and
  their bpw accounting.

## Archive

Legacy additive, interaction, Block-CLADO, dense-cone, adjoint, and PrismaSCOUT
iteration/polish tools are archived for artifact replay and comparison under the
dated `archive/` walls.
