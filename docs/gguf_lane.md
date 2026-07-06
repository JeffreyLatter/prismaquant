# GGUF lane — llama.cpp / vLLM-GGUF serving

*Added 2026-07-06 (branch `claude/gguf-lane-a`). Status: enabled end-to-end;
GPTQ-into-k-quant rounder and MoE expert stacking are open work. Trust the
code and the measured tables over this prose (prime directive applies).*

## What it is

A second export container. The allocator chooses per-Linear among the GGUF
k-quants — **Q2_K 2.625 / Q3_K 3.4375 / Q4_K 4.5 / Q5_K 5.5 / Q6_K 6.5625 /
Q8_0 8.5 bpw** (fixed bpw: all scales live inside the superblocks) plus BF16
passthrough — and the artifact is a single `.gguf` that llama.cpp serves
natively and vLLM serves via its GGUF path (in-tree ≤0.19, the official
`vllm-gguf-plugin` on current vLLM). No custom kernels anywhere; the 2–3 bpw
regime this unlocks has no NVIDIA-native alternative (NVFP4 is the floor of
the compressed-tensors stack).

## Subsystem map

| Concern | File |
|---|---|
| Formats: field quantizers, emulation QDQ, byte packers | `prismaquant/gguf_formats.py` |
| Registry entries (family `"gguf"`) | `format_registry.py` (GGUF block at the end) |
| Serving constraints (menu + `%256` shape rules) | `serving_profile_specs/gguf.json` |
| Exporter (skeleton requantizer) | `prismaquant/export_gguf.py` |
| Batched cost path (`family == "gguf"` branch + imatrix) | `measure_quant_cost.py` |
| Pipeline stage (`EXPORT_CONTAINER=gguf`) | `run-pipeline.sh` |
| Tests (bit-exactness, profiles, batched==unbatched) | `tests/test_gguf_formats.py` |

## Design invariants

1. **One math path.** Each format has a single field quantizer whose output
   feeds *both* the registry emulation (`quantize_dequantize`, what cost
   measurement scores) and the export byte packer. `gguf-py`'s
   `dequantize(pack(w))` is pinned **bit-identical** to the emulation in
   tests — measured cost and shipped bytes cannot diverge.
2. **Reference-parity scale selection.** The quantizers port llama.cpp's
   `make_qkx2_quants` / `make_qx_quants` (weighted grid + weighted-LS refit,
   sign-aware symmetric search), vectorized in torch, GPU-first. Verified at
   parity: their preset mix re-rendered by our packers measures within ~2.5%
   KL of their own artifact.
3. **imatrix in lockstep.** `PRISMAQUANT_GGUF_IMATRIX` (default **on**)
   applies activation weighting (per-column mean squared activation, llama.cpp
   composition `qw·sqrt(sigma2+x²)`) in the batched *cost* path, and the
   pipeline passes `--imatrix-from-act-cache` to the exporter under the same
   flag — same calibration corpus, same rendering, both sides.
4. **Fail fast, never coerce.** The exporter hard-errors on assignments
   containing non-GGUF formats (allocate with `--target-profile gguf`), and
   on assignment entries that match no skeleton tensor.
5. **Container correctness is delegated.** The exporter requantizes a
   *skeleton* produced by llama.cpp's own `convert_hf_to_gguf.py --outtype
   bf16` — their converter owns metadata/tokenizer/arch/naming; we own only
   tensor bytes. Provenance (git commit, assignment sha256, per-tensor format
   map) is baked into `prismaquant.*` KV metadata.

## Running it

```bash
EXPORT_CONTAINER=gguf TARGET_PROFILE=gguf \
FORMATS=Q2_K,Q3_K,Q4_K,Q5_K,Q6_K,Q8_0,BF16 \
TARGET_BITS=2.95 PRODUCTION_CACHE=0 \
  ./run-pipeline.sh
```

Cost-objective note: use the M6 objective (`h_trace × weight_mse`) for this
lane — `h_trace × output_mse` allocation *lost to llama.cpp's hand heuristic*
at matched size (KLD 3.96 vs 2.73 on the 0.6B screen) while `weight_mse` beat
it (2.33). Embedding/head policy (`GGUF_TOKEN_EMBEDDING_FORMAT`,
`GGUF_OUTPUT_FORMAT`) matters for size-matched comparisons: llama.cpp presets
quantize `token_embd`/`output` (their Q2_K preset uses Q2_K/Q6_K).

Evaluation on the llama.cpp serving metric:

```bash
# once: save base logits from the bf16 skeleton
llama-perplexity -m skeleton.gguf -f wiki.test.raw \
  --kl-divergence-base base_logits.bin --chunks 64 -ngl 99
# per artifact
llama-perplexity -m exported.gguf --kl-divergence-base base_logits.bin \
  --kl-divergence --chunks 64 -ngl 99
```

## Measured status (Qwen3-0.6B, all arms 347 MB, 64-chunk KL-vs-BF16)

| arm | mean KLD | top-1 |
|---|---|---|
| llama.cpp Q2_K preset | 2.728 | 32.1% |
| their mix, our packers (render parity check) | 2.796 | 31.8% |
| **ours: M6 allocation, no imatrix** | **2.327** | **35.0%** |
| llama.cpp preset + imatrix (same corpus) | 0.913 | 55.6% |
| ours: M6 allocation + imatrix, fully consistent | 1.061 | 53.5% |

Measured allocation beats the hand heuristic by −14.7% KL at matched bytes
(and independently rediscovers its v/o/down-get-more shape). The imatrix arm
is currently lost by +16%: llama.cpp's imatrix-mode `quantize_row_*_impl`
paths carry extra refinement beyond the reference path ported here. The
planned answer is the GPTQ-into-k-quant rounder (full-Hessian error
propagation strictly subsumes diagonal imatrix), not chasing their refinement
passes.

## Known limitations / open work

- **MoE expert stacking**: the name map handles dense models; stacked
  `ffn_*_exps` tensors (one GGUF tensor per layer/projection, one format for
  all experts) are not yet wired into cost/export.
- **GPTQ-into-k-quant rounder**: freeze fp16 super-scales per 256-superblock,
  JSO-style quantized sub-scale grid inside the GPTQ loop; render-mechanism
  registration (`_format_supports_render_mechanism`) still returns none for
  gguf formats — they render via the registry-RTN fallback.
- **IQ formats** (IQ2_XXS 2.06 … IQ3_S 3.44): registry/menu candidates below
  Q2_K, research-only until the slower serving path (MMVQ/Triton, no CUDA
  MMQ) passes a perf gate.
- **Selection**: `SELECTION_MODE=surrogate` only; the validated-frontier
  real-KL selection has not been wired to a llama.cpp evaluator yet — and the
  ~3 bpw cliff is exactly where measured selection should pay.
- **Gold metric**: the tables above are the llama.cpp KL harness (the serving
  metric *for this lane's runtime*); vLLM-GGUF serving of the same artifacts
  was smoke-verified on the 0.19.2 venv but not KL-measured there.
