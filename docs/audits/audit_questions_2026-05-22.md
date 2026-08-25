# PrismaQuant Audit Questions

Date: 2026-05-22

These are not filed as findings because they need a product or serving-support
decision, not just a local numerical fix.

## 1. Registry Entries Without Served Export Paths

`MXFP6_E3M2`, `MXFP6_E2M3`, `INT8_W8A16`, and `INT4_W4A16_g128` remain in
`format_registry.REGISTRY`, but the native exporter has no served-metadata
reconciliation path for them. The new registry test marks them as explicit
gaps so new formats cannot silently join that bucket.

Question: should these stay in the same central registry as production-exported
formats, or should the registry expose an explicit `exportable_to_vllm` flag so
allocator menus cannot accidentally include research-only entries?

## 2. MXFP8_E5M2 Serving Status

PrismaQuant can pack and test `MXFP8_E5M2`, but the vLLM MXFP8 scheme documents
E4M3 weights. E5M2 should remain research-only unless we have a vLLM load/generate
smoke proving the actual kernel accepts and correctly serves E5M2 weights.

Question: do we want to keep E5M2 in the registry as a research probe, or remove
it from any menu that claims vLLM production support?

## 3. Cost Anchor Completeness

The checked-in cost anchor includes prior 5.5 bpp grouped no-FP8 metrics, the
fresh FP8-menu 5.5 metrics, and the shipped 5.5 exact vLLM comparison. The prior
no-FP8 reproduction doc does not record the full calibration dataset/sequence
contract for the PPL run.

Question: should we rerun one canonical small model and one 27B budget under a
fully specified calibration contract, then replace the partial historical anchor
with that canonical result?

## 4. FP8_SOURCE Variants

The current FP8_SOURCE path is pinned for legacy `.weight_scale_inv` source
checkpoints and copies those bytes as compressed-tensors `weight_scale`.
DeepSeek-style source checkpoints may use a different scale sibling convention.

Question: should FP8_SOURCE be split into source-format variants, or should model
profiles provide a required source-scale convention before FP8_SOURCE is allowed?

## 5. Fisher-Weighted Render Paths Are Archived but Still Partially Present

Runtime flags document Fisher-weighted GPTQ/output-MSE allocation as archived.
Some helper parameters still accept `fisher_row_weights`, and the batched NVFP4
export path does not thread row weights through its activation-weighted gates.

Question: if Fisher-weighted GPTQ is revived, should batched NVFP4 export be
disabled until it is row-weight equivalent to the per-Linear path, or should the
batched path be upgraded first?
