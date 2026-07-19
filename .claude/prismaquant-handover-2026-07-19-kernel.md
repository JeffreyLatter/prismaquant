# Handover — CB serving kernels: decode DONE at native parity; fused prefill next

**For:** the next session on `claude/nvfp4-cb`. **Date:** 2026-07-19 (session
of 07-18 evening). **Prior handover:** `prismaquant-handover-2026-07-18-kernel.md`
(still the orientation doc; this one records what its mission plan produced).
**Prime directive applies:** verify against code/tests/served numbers.

## 0. TL;DR — where the mission stands

Mission was: make CB serving match native. **Decode is DONE — served at/above
native.** Prefill moved 1.622 → 1.075 s (AURA 0.746); the remaining gap needs
the fused CUTLASS kernel, whose baseline-parity gate PASSED this session.

| 27B served (`--enforce-eager`) | TTFT(1400) | decode tok/s |
|---|---|---|
| AURA-5.5 (native target) | **0.746 s** | 10.26 |
| CB before this session (Triton proto-i) | 1.622 s | 4.20 |
| **CB now (commits c5741ad..)** | **1.075 s** | **10.27–10.30** |

KL preserved (see §3 — read it, there's a subtle measurement finding).

## 1. What landed (all committed on `claude/nvfp4-cb`, tested, served)

1. **fp8-direct transient expand** (`c5741ad`): expand writes e4m3 bytes
   directly (byte-gather LUT), no bf16 tile + cast. Bit-exact pinned.
2. **CUDA decode GEMV + fused act-QDQ** (`ee6c46e` + tuning commit):
   `csrc/cb_gemv.cu`. One block/row, warps stride superblocks, coalesced u64
   staging (packed stream read once, `__ldcs`), 32 codewords ↔ 32 lanes,
   E4M3-byte codebook gathers (16 KB @k48), WARPS∈{4,8} by n_sb (K=5120 has
   20 superblocks — 4 warps divide exactly), fp32 FMA + block reduce.
   3.46× over Triton at M=1 (250–355 GB/s). Dispatch: CUDA M≤8
   (`PRISMAQUANT_CB_CUDA_M_MAX`), Triton 9–16, expand+GEMM above.
   QDQ is ONE kernel, bit-exact to `codec.fp8_dynamic_act_qdq` — two real
   1-ULP traps fixed: `__nv_cvt_float_to_fp8` double-rounds via f16 (ported
   c10's single-rounded conversion) and torch tensor/scalar division is a
   reciprocal MULTIPLY (matched). 39 tests in `tests/test_cuda_gemv.py`.
3. **CUDA transient expander** (`cb_expand_fp8`): 2× the Triton expander
   (123–132 GB/s wr); prefill serial cost of the expand halved.
4. **JIT infra**: `vllm_prismaquant/cuda_ext.py` builds in-container (~30 s,
   `PRISMAQUANT_CB_EXT_DIR` to persist; warm-built at weight load so the
   first request doesn't pay it). Loud Triton fallback.
   `PRISMAQUANT_CB_DECODE=triton` reverts everything CUDA.
5. **CUTLASS baseline-parity gate PASSED** (`csrc/sm120_fp8_gemm.cu`):
   sm120 CollectiveBuilder fp8 GEMM (e4m3/e4m3→bf16, 128×128×128, vendored
   CUTLASS 4.3.4 headers `include` + `tools/util/include`) at 0.91–0.99× of
   vLLM `cutlass_scaled_mm` on 27B M=1400 shapes. We can own the mainloop.
6. **Fork-without-change gate PASSED** (`db2c331`):
   `csrc/cutlass_fork/sm120_cb_mma_tma.hpp` = the sm120 collective verbatim
   under a new policy tag `MainloopSm120CbTmaWarpSpecialized`;
   `csrc/cb_fused_gemm.cu` rebinds the builder's resolved collective onto it
   (variadic policy-swap metafunction) and runs it through `GemmUniversal` —
   **bit-identical to the builder version, speed-equal (0.93–1.00×)**. The
   kernel layer accepts the new policy because its enable_if keys on
   `DispatchPolicy::Schedule`. Start the decode-injection INSIDE this fork:
   the `load()` method's B-operand TMA + the smem staging between pipeline
   arrive and the consumer's `SmemCopyAtomB` reads.

## 2. The one open speed item — fused FP8_CB prefill (the 0.33 s)

Remaining prefill gap = transient traffic (write 24 GB + re-read 24 GB vs
AURA's single resident read). Facts for the fork:
- Fork target: `sm120_mma_tma.hpp` (copy at
  `csrc/cutlass_fork/sm120_mma_tma_orig.hpp`), policy
  `MainloopSm120TmaWarpSpecialized`, driven by the sm90-style cooperative WS
  kernel. `NumProducerThreadEvents = 1` (single producer thread issues TMA) —
  the DECODE therefore cannot live in the producer; plan it at the consumer
  stage-head or add a decode warp role.
- **Layout gift:** a 128- or 256-wide K-tile of a row maps to a CONTIGUOUS
  packed-byte slice (codewords are LSB-first in order: codewords 0..15 = bytes
  [0, 2k)). So the packed B-tile is a plain 2-D TMA load; decode expands it
  in smem to the standard SmemLayoutB tile, MMA/epilogue untouched.
- LUT in smem: e4m3 bytes, ≤16 KB for k≤48 (fp8 rungs) — fits alongside
  stages.
- REJECTED en route: N-chunked expand+GEMM overlap (0.46×, and NOT bit-exact —
  `cutlass_scaled_mm` reconfigures on narrow N). If an interim is wanted
  before the fused kernel: cross-layer expand PREFETCH (side stream, learn
  forward order on first pass) hides the 2.7 ms/layer-set expand behind the
  GEMM without touching GEMM shapes; ~100 LoC, stateful — design note only.

## 3. KL measurement finding (read before comparing any dumps)

The 27B dump lands in exactly TWO bit-reproducible states: conf-KL 0.01134 /
ALL 0.0134 / PPL 9.166, or 0.01328 / 0.0142 / 9.189. The flip is **whether
the CUDA extension .so is resident in the serving process during the dump**
(warm-build/pre-dump traffic loads it; `PRISMAQUANT_CB_DECODE=triton` or a
pre-GEMV dump doesn't). It is NOT the CB kernels' arithmetic: both prefill
paths are bit-identical offline (pinned by tests + an offline repro of the
served Linear). Mechanism: extension load shifts allocator addresses →
alignment-sensitive kernel selection elsewhere in the model → global
reassociation drift (~97% of positions move a ULP-ish). This is the concrete
mechanism behind CLAUDE.md's "KL drifts across sessions". Consequences:
- Compare arms only with IDENTICAL extension residency.
- The artifact's conf-KL has ±17% evaluation sensitivity; quote the range.
- Either reading beats AURA by −45%+ conf-KL / −56%+ ALL-KL; PPL gap to BF16
  2–3× smaller. The quality verdict is unaffected.

## 4. How to reproduce the numbers

- Serve: `serve_one.sh <name> /dqruns/prod-27b-nvfp4cb-5p5/exported_nvfp4_cb
  --gpu-memory-utilization 0.85` (installs the plugin from /repo live).
- Measure: `measure.py dump m <tok> out.json` THEN `speed m <tok>` (first
  speed rep absorbs Triton-JIT warmup — read reps 2–3; the CUDA ext itself is
  warm-built at load now).
- Baselines in `/home/rob/dq-runs/prod-27b-nvfp4cb-5p5/ab/`:
  `{bf16,aura,ours*,ours_v3*}.json|_speed.txt`. `ours_v3` = this session's
  final state.
- Kernel tests (container):
  `PYTHONPATH=/repo:/repo/plugins/vllm_prismaquant python3 -m pytest
  /repo/plugins/vllm_prismaquant/tests/test_cuda_gemv.py` (needs
  `/artifacts` = nvfp4-cb-phase0/serve mount and nvcc). Bench:
  `tests/bench_cuda_gemv.py`. Profile: scratchpad scripts (not committed).

## 4b. Second half of the session (encode track + fused verdict)

- **Fused prefill verdict** (commits 80e6414/8936a01/108d6b6): decode-in-
  prologue collective is DONE and bit-exact; **0.22× at M=1400 (structural:
  per-M-tile re-decode)**, wins only M∈(16,128] (1.04–1.45×); chunked overlap
  also dead (GEMM already memory-bound at M=1400). Large-M parity = the
  persistent-N/decode-once schedule (task; cutlass-kernel-notes has the full
  verdict + wiring spec for the mid-M band). TTFT stays 1.075 s by default.
- **Encode track** (75eb95d/fbd666d/924ce74): local-stage expert-measure skip
  under the hybrid; expert KL-sampling implemented then **REFUTED by
  measurement** — and the refutation surfaced the real finding: **the bf16
  expert unit-KL at CB fidelity is a perturbation floor** (S=1 of 256 ≈
  full-stack KL ≈ 0.016–0.019; memory `cb_expert_unitkl_floor`). Recommended
  35B-resume recipe (pending Robert): `CB_EXPERT_EMPIRICAL=0` +
  `PRISMAQUANT_EXPERT_COST_SAMPLE=16` (MSE sampling is unbiased) + ladder →
  cost stage ~14 h → ~1 h. Three latent 35B blockers fixed en route:
  zero-expert fill missing from the empirical CLI (+ loud all-zero guard),
  fill no-op against staged dirs (pass the ORIGINAL model dir), missing
  down_proj imatrix (replay synthesis, persisted for exporter lockstep).
  Ladder generalized (FP8 rungs, fitted slope, pays at 4 rungs);
  `PRISMAQUANT_EXPERT_CALIB_BATCH` for the forward wall.

## 5. Still open beyond the fused kernel (unchanged from prior handover)

- Encode-time speed (the 35B cost stage; JSO-style candidate-collapse prune).
- 35B MoE serve (CB MoE path never served; decode GEMV needs a grouped/MoE
  variant — `_decode_expert` in moe.py is still per-expert transient bf16).
- fp4-CB (NVFP4_CB) CUDA paths: GEMV + expander are fp8-product-only; fp4
  v1/v2 stay on Triton. Hy3/DSv4 class needs the fp4 fused kernel
  (`sm120_blockscaled_mma_tma.hpp`) — same layout gift applies.
- The GEMV M∈(8,16] regime loses to Triton (x re-read per row-block); a
  rows-per-block variant would close it if multi-stream decode matters.
