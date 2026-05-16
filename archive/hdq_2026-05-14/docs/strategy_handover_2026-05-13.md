# Handover — Quantization Strategy Conversation (2026-05-13)

This doc captures a long discussion about quantization mechanisms, vLLM integration, and a candidate path for adding DuQuant-style outlier handling on top of the existing NVFP4 path on SM121. It exists so the next Claude session can pick up planning without re-deriving the analysis.

## Project context

Rob is on DGX Spark (SM121 / GB10), serving NVFP4 quantized models via vLLM with custom CUTLASS/Marlin kernels. Qwen3.5-122B NVFP4 is shipped (31.6 tok/s). Active work is on lm_head / MTP MoE NVFP4 optimization. Consult `~/.claude/projects/-home-rob/memory/MEMORY.md` for full background.

The conversation in question explored adding a stronger outlier-handling layer to the existing NVFP4 stack — specifically whether DuQuant / DuQuant++ style rotations are worth the integration effort, or whether a simpler Hadamard-based approach captures most of the benefit.

## What we settled (do NOT re-derive)

1. **Quantization axes** — Rob wanted the orthogonal-bases enumeration. Format × granularity × symmetry × what-quantized × calibration × rounding × assignment × equivalent-transforms × outlier-handling × mixed-precision × sparsity. Microscaling formats fix granularity, symmetry, and partly subsume outlier handling. Equivalent transforms (rotations, SmoothQuant, AWQ) live on the same axis and should not stack.

2. **vLLM support map** — verified current state: NVFP4/MXFP4 via compressed-tensors+modelopt, FP8 widely, GPTQ/AWQ/Marlin, llm-compressor pipeline for quantization. **SpinQuant + QuaRot are integrated** via `SpinQuantModifier` in llm-compressor → compressed-tensors `transforms` config → vLLM with HadaCore kernels (vLLM v0.11+). Online R3/R4 supported. **No Givens rotation kernel in vLLM today.**

3. **Static vs dynamic activation quant** — covered. Modern LLM standard is dynamic per-token activation + static per-block weight. NVFP4 activation path uses dynamic per-token amax → per-block FP8 scale derivation.

4. **The down_proj rotation algebraically cannot be offline-folded.** Element-wise SwiGLU multiplication does not distribute over rotation — `(a ⊙ b) @ R ≠ (a @ R) ⊙ (b @ R)`. So R3-style rotations (residual stream and V→O can be folded; down_proj cannot if you want the activation quant benefit). This is fundamental, not a tooling limitation. The only escapes are: skip down_proj rotation (lose accuracy), keep down_proj at W4A16 (lose throughput), or fuse the runtime rotation into a SwiGLU+rotation+quant kernel (still online, but cheap).

5. **DuQuant++ code exists but is research-grade.** `github.com/Hsu1023/DuQuant-v2` — MXFP4 only (block=32), Llama-3 only, pure PyTorch fake-quant (no real low-precision matmul kernels), no compressed-tensors export. Released 2026-04-21, 4 commits. The repo demonstrates the algorithm but is not a serving codebase.

6. **ParoQuant kernel is reusable.** `github.com/z-lab/paroquant`, MIT licensed, has a clean Givens rotation CUDA kernel (`paroquant/kernels/cuda/rotation.{cu,cuh}`) templated on `{float, half, bfloat16}` × KROT × GROUP_SIZE. Currently compiles GROUP_SIZE=128 only (TODO comments mention 64). Trivial to add GROUP_SIZE=16 instantiation. Kernel input is generic block-Givens `(idx_ij, theta)` — DuQuant rotations could feed it, BUT DuQuant's full rotation needs more rounds than ParoQuant's KROT=8 (would need to compile larger KROT). For NVFP4 block=16, dense block-matmul might be simpler than Givens decomposition.

7. **vLLM plugin pattern works for this.** ParoQuant integrates via vLLM `general_plugins` entry point — no fork. ParoQuant's `paroquant/inference/backends/vllm/plugin.py` (12.8KB) is the template to crib for the plugin shell.

8. **Hadamard substitution analysis.** Replacing Givens with block-Hadamard at block=16 (NVFP4 microscale) gives up data-aware outlier targeting (~0.5-1 PPL accuracy cost) but uses existing HadaCore kernels and existing llm-compressor SpinQuantModifier — zero new kernel work. DuQuant's other contributions survive Hadamard substitution: insertion-point identification (down_proj), zigzag permutation, and the double-rotation-with-permutation structure. "Hadamard-DuQuant" recipe captures ~30% of DuQuant's edge over QuaRot at zero kernel cost.

## Recommended path (current consensus before restart)

In order of escalation:

1. **Hadamard-DuQuant first** (default): block-Hadamard at block=16 via existing llm-compressor SpinQuant pipeline, plus a calibrated zigzag permutation (data-aware, no kernel needed — fold into adjacent layer column ordering), plus a second block-Hadamard pass for double-rotation effect. Zero new CUDA. Two HadaCore launches per FFN instead of one. Permutation calibration is CPU-cheap.

2. **If accuracy gap remains after QAD/finetuning**: escalate to true Givens-DuQuant by cribbing ParoQuant kernel as a starting skeleton, adding GROUP_SIZE=16 instantiation, extending KROT for DuQuant's denser rotation, plus checkpoint converter from DuQuant-v2 format → compressed-tensors with custom transform spec. Plugin shell follows ParoQuant's pattern.

3. **Don't pursue offline-only rotation** at down_proj — algebraically blocked. Don't pursue stacking with AWQ/SmoothQuant — same axis as rotations, fights them.

Quantitative expectations (rough, must measure on actual model):
- BF16: 0 PPL gap (reference)
- W4A4 RTN: +5-10 PPL
- QuaRot (block-Hadamard only): +0.5-1 PPL
- Hadamard-DuQuant (proposed): +0.3-0.7 PPL
- Full DuQuant++ (Givens): +0.1-0.4 PPL

## Open questions for next session

These are the ones the new conversation should pin down with Rob before starting implementation:

1. **Target model** — Qwen3.5-122B (already shipped in NVFP4) for accuracy parity, or a smaller Llama-3 for faster iteration first?
2. **Accuracy bar** — what PPL/eval gap vs BF16 is acceptable? This determines whether Hadamard-DuQuant is sufficient or whether escalation to Givens is required.
3. **Does compressed-tensors `transforms` config support permutation as a first-class transform?** If yes, Hadamard-DuQuant is purely a configuration change. If no, need to fold permutation into adjacent linear layer columns at checkpoint creation time. Worth checking `compressed-tensors` source before planning.
4. **Calibration data source** — what corpus is being used for activation scale calibration today on the Qwen3.5-122B path? Same data needed for permutation calibration.
5. **Where does prismaquant fit?** This conversation didn't establish what code lives in this repo vs spark-vllm-docker vs upstream contributions. The new session should clarify whether prismaquant is the home for the DuQuant-style work, or just the planning/docs hub.

## Reference materials

### Papers
- DuQuant (NeurIPS 2024 Oral): https://arxiv.org/abs/2406.01721
- DuQuant++ (microscaling): https://arxiv.org/html/2604.17789v2
- ParoQuant (ICLR 2026): https://arxiv.org/abs/2511.10645
- SpinQuant: https://arxiv.org/abs/2405.16406
- NVIDIA NVFP4-QAD report (2026-03-05): https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf

### Repos to crib from
- `github.com/Hsu1023/DuQuant-v2` — DuQuant++ algorithm reference (research grade)
- `github.com/z-lab/paroquant` — MIT, reusable Givens kernel + vLLM plugin pattern
- `github.com/vllm-project/llm-compressor` — SpinQuantModifier, transforms config
- `github.com/neuralmagic/compressed-tensors` — TransformScheme, Hadamard support
- vLLM PR #16443 (Online Rotations / HadaCore integration): https://github.com/vllm-project/vllm/pull/16443

### Specific files in ParoQuant kernel (for reuse)
- `paroquant/kernels/cuda/rotation.cuh` — dtype traits (float / half / bfloat16) — directly reusable
- `paroquant/kernels/cuda/rotation.cu` — Givens kernel + dispatch — adapt KROT and GROUP_SIZE
- `paroquant/kernels/cuda/pybind.cpp` — torch op binding pattern
- `paroquant/inference/backends/vllm/plugin.py` — vLLM general_plugins entry point template
- `pyproject.toml` — entry point declaration pattern

### Specific files in DuQuant-v2 (algorithm reference, not for reuse)
- `quantize/quantizer.py` — Givens rotation construction + per-token MXFP4 simulation
- `quantize/duquant.py` — offline pipeline (LET fold, LWC fine-tune, GPTQ)
- `models/transformation.py` — `_inplace` (offline weight folding) vs `_temporary` (online) logic
- `quantize/fp4_ops.py` — fake-quant only (NOT a reusable serving path)
- `get_rot.py` — rotation matrix generation script

## Don't re-explore

The next session should NOT:
- Re-derive the orthogonal axes of quantization (covered exhaustively)
- Re-establish that vLLM lacks a Givens kernel (verified)
- Re-evaluate whether OmniQuant fits microscaling (confirmed: technically yes, practically displaced)
- Re-litigate AWQ/SmoothQuant + DuQuant compatibility (incompatible, same axis)
- Re-walk the per-layer dataflow of W4A4 transformers (covered)
- Re-explore why down_proj rotation can't be offline (algebraic block, covered)

## Suggested opening for the next session

Read this doc, then ask Rob the five open questions above (model, accuracy bar, compressed-tensors permutation support, calibration data, scope of prismaquant). Based on answers, either:
- Build an implementation plan for Hadamard-DuQuant via existing llm-compressor pipeline (most likely path)
- Or build an implementation plan for the Givens-DuQuant kernel + plugin if Rob wants to skip ahead
- Or scope out an accuracy comparison harness to decide between the two empirically before committing

Rob will direct from there.
