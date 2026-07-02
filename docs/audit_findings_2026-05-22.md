# PrismaQuant Quantization-Correctness Audit

Date: 2026-05-22

Audit policy: every boundary-crossing transformation needs a pinning test:
renderer to stored metadata to served dequant, cost surrogate to held-out
quality, and calibration metric to held-out metric. Findings below are limited
to numerical disagreement with a consumer/reference implementation or semantic
duplication that must remain mathematically identical.

## Finding 1: MXFP4 E8M0 Scale Encoding Disagreed With compressed-tensors

1. File: `prismaquant/export_native_compressed.py:2100`

2. Ground truth:
   `compressed_tensors/quantization/utils/mxfp_utils.py:26` defines the MX
   element offset as `floor(log2(type_max))`, and
   `compressed_tensors/quantization/utils/mxfp_utils.py:110` generates MX
   scales by rounding block amax to a power of two, subtracting that offset,
   and adding E8M0 bias 127.

3. Type: correctness bug.

4. Issue:
   The previous MXFP4 packer used `ceil(log2(max_abs / 6.0))`. That matches
   some values, but it does not match compressed-tensors' `generate_mx_scales`
   for MXFP4 blocks near the power-of-two rounding boundary. The new
   compressed-tensors reconciliation test initially failed on MXFP4 while
   MXFP8 passed.

5. Proposed fix:
   Patch-sized and applied: add shared `_mx_rounded_amax_power2` /
   `_mx_base_exponent_from_amax`, then route both `_mxfp8_base_exponent` and
   `_mxfp4_grouped_codec` through it. MXFP4 dense and packed export now share
   `_mxfp4_grouped_codec`.

6. Test that would have caught it:
   `tests/test_prismaquant_export_native_compressed.py:623`
   `test_mx_e8m0_scale_encoding_matches_compressed_tensors_reference`
   compares MXFP4 and MXFP8 stored E8M0 scale bytes directly against
   compressed-tensors `generate_mx_scales`.

## Finding 2: MXFP8 Activation Scoring Used Weight-Scale Semantics

1. File: `prismaquant/format_registry.py:578`

2. Ground truth:
   vLLM's MXFP8 runtime activation quantizer in
   `vllm/model_executor/layers/quantization/utils/mxfp8_utils.py:52` requires
   32-value blocks, computes `floor(log2(amax)) + 127`, divides by that
   descale, then casts to E4M3. vLLM's MXFP8 scheme documents W8A8 dynamic
   activation quantization at
   `vllm/.../compressed_tensors_w8a8_mxfp8.py:31`.

3. Type: cost-surrogate-vs-deployment correctness bug.

4. Issue:
   `MXFP8_E4M3.activation_quantize_dequantize` used the registry's generic
   MX weight RTN path, which scales weights by the element-format offset.
   vLLM activation quantization does not use that weight observer convention.
   Random probes showed small but nonzero dequant differences, so local
   render scores were not exactly pricing the served activation path.

5. Proposed fix:
   Patch-sized and applied: add `_mxfp8_e4m3_activation_vllm_rtn` and register
   it as the activation quantizer for `MXFP8_E4M3`.

6. Test that would have caught it:
   `tests/test_format_registry.py:44`
   `test_mxfp8_activation_quantizer_matches_vllm_runtime_reference`
   reconstructs the vLLM torch reference formula and compares it exactly to
   the registry activation quantizer.

## Finding 3: Registry Formats Lacked a Served-Metadata Reconciliation Gate

1. File: `tests/test_prismaquant_export_native_compressed.py:411`

2. Ground truth:
   vLLM NVFP4 allocates `weight_packed`, FP8 `weight_scale`, and FP32
   `weight_global_scale` at
   `vllm/.../compressed_tensors_w4a4_nvfp4.py:59`, `:69`, and `:62`, then
   inverts CT global-scale divisors at `:105`.
   vLLM MXFP8 stores float8 weights plus uint8 E8M0 scales with group size 32
   at `vllm/.../compressed_tensors_w8a8_mxfp8.py:31`.
   vLLM FP8 channel strategy preserves `weight` and `weight_scale` at
   `vllm/.../fp8_utils.py:1064`, and FP8 dequant multiplies by the scale in
   `vllm/.../w8a8_utils.py:45`.
   compressed-tensors block FP8 dequant multiplies FP8 weight by
   `weight_scale_inv` at
   `compressed_tensors/.../fp8block_dequantizer.py:146`.

3. Type: missing-test gap.

4. Issue:
   The code had an NVFP4-specific check, but no single registry-driven test
   forced every `format_registry.REGISTRY` entry to be either reconciled with
   served metadata math or marked as an explicit non-exported research gap.
   That allowed the NVFP4 scale mismatch class to exist in multiple call
   sites, and it would also have missed MXFP4 scale-byte drift.

5. Proposed fix:
   Patch-sized and applied: add a registry-driven test with two responsibilities:
   it asserts every registered format is reconciled or explicitly gapped, and
   for exportable formats it compares renderer dequant to served-load dequant.
   Explicit current gaps are `MXFP6_E3M2`, `MXFP6_E2M3`, `INT8_W8A16`, and
   `INT4_W4A16_g128`.

6. Test that would have caught it:
   `tests/test_prismaquant_export_native_compressed.py:426`
   `test_registry_served_metadata_reconciliation_covers_all_registered_formats`
   and `tests/test_prismaquant_export_native_compressed.py:447`
   `test_registry_render_dequant_matches_served_metadata`.

## Finding 4: Cost-Surrogate Calibration Had No Checked-In Anchor

1. File: `tests/test_cost_surrogate_calibration_anchors.py:22`

2. Ground truth:
   The allocator formula is correct in code:
   `prismaquant/allocator_solver.py:34` computes
   `0.5 * h_trace * weight_mse`, `allocator_candidates.py:229` routes
   `output_mse` and `weight_mse` through that helper, and
   `allocator_solver.py:267` sums candidate losses for an assignment.
   Empirical deployment anchors are recorded in
   `docs/grouped_kl_allocator_results_2026-05-20.md:74`,
   `:94`, `:139`, `:153`, `:210`, and `:221`.

3. Type: missing-test gap.

4. Issue:
   We had documented 27B predicted dloss, held-out PPL, and vLLM exact KL/PPL
   anchors, but no test made those anchors discoverable or opt-in comparable
   to fresh runs. This is the class of drift that let a cost surrogate look
   locally plausible while deployment metrics moved the other way.

5. Proposed fix:
   Patch-sized and applied: add
   `tests/fixtures/cost_surrogate_calibration_anchors.json` with the 5.5 bpp
   grouped no-FP8, grouped FP8-menu, and shipped-vLLM anchors, plus an opt-in
   metrics comparator gated by `PRISMAQUANT_COST_ANCHOR_RESULTS`.

6. Test that would have caught it:
   `tests/test_cost_surrogate_calibration_anchors.py:22`
   validates that anchors include predicted-dloss/PPL and KL coverage.
   `tests/test_cost_surrogate_calibration_anchors.py:52` compares fresh metrics
   against the checked-in anchors when a run-result JSON is supplied.

## Finding 5: Format Math Was Duplicated Across Render Paths

1. File: `prismaquant/export_native_compressed.py:326`,
   `prismaquant/export_native_compressed.py:349`,
   `prismaquant/export_native_compressed.py:2057`,
   `prismaquant/export_native_compressed.py:2089`,
   and `prismaquant/export_batched_gptq.py:269`

2. Ground truth:
   NVFP4 served dequant is codebook value times FP8 group scale times
   inverted CT global scale per vLLM
   `compressed_tensors_w4a4_nvfp4.py:105`.
   MXFP8 served dequant is FP8 value times `2 ** (scale - 127)` per vLLM
   `mxfp8_utils.py:111`.
   FP8 served dequant is FP8 value times weight scale per vLLM
   `w8a8_utils.py:45`.

3. Type: duplication risk.

4. Issue:
   NVFP4, FP8, MXFP8, and MXFP4 each had pack/dequant math spread across RTN,
   GPTQ, scale sweep, packed expert, and batched paths. The NVFP4 bug showed
   that these paths can silently disagree while each local unit test still
   passes.

5. Proposed fix:
   Patch-sized and applied for the active exporter paths:
   `_nvfp4_quantize_grouped_codec`, `_nvfp4_quantize_dequantize_with_eff_scale`,
   `_fp8_codec`, `_fp8_dynamic_codec`, `_mxfp8_grouped_codec`, and
   `_mxfp4_grouped_codec` are now the single primitives that call sites use.

6. Test that would have caught it:
   The registry reconciliation test at
   `tests/test_prismaquant_export_native_compressed.py:447` catches renderer
   versus served-metadata drift; existing path-specific tests exercise dense,
   packed, GPTQ, and production-cache entry points.

## Non-Findings Checked

- Allocator arithmetic matches the documented Taylor approximation:
  `predicted_dloss = 0.5 * h_trace * mse`, and assignment loss is a sum of
  candidate losses.
- FP8_SOURCE scale convention is multiplier, not divisor, in both streaming
  dequant and export passthrough. compressed-tensors' FP8 block dequant uses
  `weight_fp8 * weight_scale_inv`, matching PrismaQuant's rename to
  `weight_scale`.
