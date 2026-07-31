# MXFP4-grid codebooks: feasibility (measured)

2026-07-30. Commissioned by Robert: *"explore the feasibility of using MXFP4 for
low-bit CB formats that would be deployable cross-platform."* Sibling of
`mxfp6_cb_feasibility.md` (which was decided on a subset/dominance argument with
no GPU work). **This one needed measurement**, and got it: MXFP4 is *not* a
subset case — it shares NVFP4's element grid exactly and differs only in the
scale plane, which in the fp4 CB family *is* a storage dial.

Driver scripts (scratch, not shipping): `scratch/mxfp4_cb/`. Raw results:
`/home/rob/dq-runs/mxfp4-cb-study/`. No shipping module was modified.

---

## Verdict: NO-GO for the quality claim; the cross-platform claim is FALSE as stated

Three findings, in descending order of how much they matter.

1. **The cross-platform premise does not survive contact with the hardware.**
   Native MXFP4 matrix silicon exists on exactly two vendors: NVIDIA Blackwell
   (sm_100/120/**121** — including this box) and AMD CDNA4 `gfx950`
   (MI350X/MI355X). It does **not** exist on RDNA4, CDNA3 (MI300X), any Intel
   part shipping today, Apple, or CPU — and most pointedly **not on
   `gfx1151` / Strix Halo, which has no FP4 *and no FP8* matrix instruction at
   all** (its WMMA set is f16/bf16/iu8/iu4). So on the one non-NVIDIA target
   named in the commission, an MXFP4-grid codebook decodes to bf16 exactly like
   an NVFP4-grid one, and the grid choice buys nothing. §4.
2. **The quality price is large and it is paid in the scale plane, not the
   codeword.** MXFP4 and NVFP4 use the *same* E2M1 element grid
   (`format_registry.py:669` vs `:696` — both `weight_element_dtype="fp4_e2m1"`).
   The entire delta is group 16 → 32 and an E4M3 → **E8M0 (power-of-two-only)**
   block scale. Measured on Qwen3-0.6B, imatrix-weighted: see §5. The
   E8M0 penalty at the 4-bit grid is roughly an **order of magnitude larger**
   than the 8-bit MXFP8 analogue measured in the same harness — the intuition
   behind de-menuing MXFP8 is not merely confirmed at 4 bits, it is amplified.
3. **And the grid constraint is currently unpaid-for on *both* grids.** No
   shipped gridbook path feeds an FP4 tensor core. Dense fp4-CB prefill is
   "Triton v2 expand (**bf16**, composed scales) → cuBLAS"
   (`STANDARDS.md:49`; `linear.py:392-403`, comment: *"bf16 MMA — INV-2 waived;
   the FP4-MMA CUTLASS prefill is prototype iii"*); the decode GEMV gathers a
   **bf16** codebook (`codec.py:42`). The only path that reaches native tensor
   cores in the whole CB lane is FP8_CB → e4m3 → `cutlass_scaled_mm`. §6.

**So: NO-GO.** The conditions that would flip it are named in §8, and one of
them (a CDNA4 HIP lane) is a real future, not a hypothetical.

---

## 1. What actually differs (and what does not)

| | NVFP4-CB (shipped) | MXFP4-CB (proposed) |
|---|---|---|
| element grid | E2M1 `{0,±.5,±1,±1.5,±2,±3,±4,±6}` | **identical** |
| codeword | d=8 vector of grid values, k-bit index | **identical** |
| superblock | 256 weights = 32 codewords = 4k index bytes | **identical** |
| scale group | 16 | 32 |
| scale dtype | E4M3 (v1 bare / v2 two-tier composed) | **E8M0, powers of two only** |
| per-tensor global scale | none in the CB path (`export_nvfp4_cb.py` has no `weight_global_scale`) | none (OCP MX has no global) |

This is why the MXFP6 verdict's argument does not transfer. There, the MXFP6
grids were *proper subsets* of e4m3, so an MXFP6-grid codebook was a handicapped
FP8-CB codebook — dominance, nothing to measure. Here the codeword alphabet is
bit-for-bit the same and the only difference is a scale plane that is genuinely
*cheaper* (§2–3) and genuinely *coarser* (§5). That is a tradeoff, and tradeoffs
get measured.

---

## 2. The rung ladder, exact

Bytes per 256-weight superblock. Index plane is `4k` in every arm; only the
scale plane moves. All integer, all byte-aligned.

| scale plane | B/256 | scale bpw | type_size |
|---|---|---|---|
| NVFP4-CB **v1** — 16 × bare E4M3 | 16 | 0.50000 | `4k+16` |
| NVFP4-CB **v2** — E8M0 super + 16 × 4-bit sub (SHIPPED) | 9 | 0.28125 | `4k+9` |
| MXFP4-CB — 8 × bare E8M0 (OCP wire form) | 8 | 0.25000 | `4k+8` |
| MXFP4-CB — E8M0 super + 8 × 4-bit exponent delta | 5 | 0.15625 | `4k+5` |
| MXFP4-CB — E8M0 super + 8 × **3-bit** delta (§3: covers 100%) | 4 | 0.12500 | `4k+4` |
| MXFP4-CB — 8 × E8M0 + 1 selector bit/group (§5 mitigation) | 9 | 0.28125 | `4k+9` |

**Does the freed scale budget buy index bits at matched bpw?** Against the
shipped v2 coding, bare MX frees only `9−8 = 1 B/256 = 0.03125 bpw`. One index
bit costs 32 bits/superblock = **0.125 bpw**, so bare MX frees exactly **¼ of an
index bit** — not a rung. What it does buy exactly is **one bit per 32-weight
group** (8 bits = 1 B/superblock), which is the selector-bit mitigation priced
in §5.

With the delta coding of §3 the answer changes: 3-bit delta frees
`0.28125 − 0.125 = 0.15625 bpw = 1.25 index bits`; the 4-bit-delta variant frees
1 index bit exactly, so **`MXFP4-CB(4-bit delta) @ k+1` is byte-identical to
`NVFP4-CB v2 @ k`** (`4(k+1)+5 = 4k+9`). That is the honest matched-bytes
comparison and is the one §5 reports.

> Doc hygiene: `STANDARDS.md:13` and `ARCHITECTURE.md §9.2` label the shipped
> ladder "2.0–3.28 bpw". Under v2 coding `K12..K24` is **1.78125–3.28125**; the
> "2.0" low end is v1 accounting (`k/8+0.5`). Harmless, but the two ends of that
> range come from different eras.

---

## 3. Two-tier scale coding does not port — a different code does

The v2 trick stores `scale_g = T[c_g] × 2^(E−127)` and requires the composed
value to be **exactly representable in the wire scale dtype**
(`two-tier-scale-spec.md §1.2`, legality mask). E4M3 has a 3-bit mantissa, so
`T` carries real information (~6–12% relative granularity) and the sub code
earns its 4 bits.

**E8M0 has zero mantissa bits.** For `T[c] × 2^(E−127)` to be E8M0-exact, every
`T[c]` must itself be a power of two — at which point the sub code is just the
low bits of the exponent and carries nothing the super does not. *The second
tier degenerates.* Derived, not guessed: it follows from the legality contract.

What survives is not "two-tier" but **exponent-range compression** (DPCM on the
exponent): one super exponent per superblock plus a small per-group delta. That
is a different code, and its width is a data question. Measured over **all
1,720,320** superblocks of Qwen3-0.6B (`scale_plane_entropy.py`), spread of the
8 no-clip E8M0 exponents within a superblock:

| spread (octaves) | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| superblocks | 396,159 | 1,186,639 | 131,650 | 5,451 | 409 | 12 |

Coverage: a **2-bit** delta (4-octave window) covers **99.98%**; a **3-bit**
delta covers **100.0%** — and 100.0% in every role separately (q/k/v/o/gate/up/
down). So the MX scale plane compresses from 8 B to **4 B per superblock with
zero distortion**, i.e. 0.25 → **0.125 bpw**.

This is a real structural advantage and it comes from three independent places:
half as many scales (g32), no mantissa to store, and tight within-superblock
exponent clustering. NVFP4-CB v2 cannot match it — its sub code must carry
mantissa granularity, which is the entire reason E4M3 beats E8M0, and the spec
already measured 4-bit subs as the optimum (`two-tier-scale-spec.md §1.3/§3`).

**Caveat, stated plainly:** the histogram is of the *no-clip ceiling* exponent
`ceil(log2(amax/6))`. The encoder may pick up to 3 octaves lower (clipping
tradeoff), which widens the realized spread. 3-bit (8-octave) absorbs that with
margin; 2-bit does not, and would need re-measuring against chosen exponents.

---

## 4. The serving matrix

The distinction that decides everything: **[NATIVE]** = a matrix instruction
consumes FP4 operands and hardware applies the E8M0 block scale;
**[UPCONVERT]** = FP4 expanded to bf16/fp16 (or requantized to int8) and a
higher-precision GEMM runs.

### Silicon

| Silicon | Native FP4 matrix? | Evidence |
|---|---|---|
| NVIDIA Blackwell sm_100/101/103 (B200/GB200) | **YES** (ue8m0@32 *and* ue4m3@16) | `tcgen05.mma.kind::mxf4` / `.kind::mxf4nvf4` |
| **NVIDIA sm_120a / sm_121a (RTX 50, GB10 / DGX Spark)** | **YES** | PTX ISA 9.3 Table 39 + release-history Table 63: `kind::mxf4`, `kind::mxf4nvf4` @ PTX 8.7, `sm_120a, sm_121a`. `mma.sync…kind::mxf4.block_scale.scale_vec::2X.m16n8k64…ue8m0` |
| NVIDIA Hopper sm_90 / Ada sm_89 | NO (FP8 floor) | — |
| **AMD CDNA4 `gfx950` (MI350X/MI355X)** | **YES** (E8M0 @32) | CDNA4 ISA §7.2 "Block Scaled Matrices": block 32, scale "Format is E8M0"; `V_MFMA_SCALE_F32_{16X16X128,32X32X64}_F8F6F4`, operand selector `4=E2M1` |
| AMD CDNA3 `gfx942` (MI300X) | NO | CDNA4 whitepaper Table 1 lists MI300X Matrix MXFP4 = "NA" |
| AMD RDNA4 `gfx1200/1201` (RX 9070) | **NO** | FP8 WMMA only (`V_WMMA_F32_16X16X16_FP8_FP8`); no fp4/e2m1 in the RDNA4 ISA |
| **AMD RDNA3.5 `gfx1151` (Strix Halo)** | **NO — and no FP8 matrix either** | LLVM `FeatureISAVersion11_5_1` → `FeatureWMMA256bInsts` = f16/bf16/(tied variants)/**iu8/iu4 only**; no `FeatureWMMA128bInsts`, no `FeatureFP8ConversionInsts`. AMD's own `matrix_calculator.py --architecture gfx1151` returns exactly those |
| AMD `gfx1250` (next gen) | YES + an NVFP4-shaped 16-bit-exponent variant | `wmma_scale{,16}_f32_*_f8f6f4` in LLVM; no public ISA doc (UNCONFIRMED mapping) |
| Intel Xe2/Xe3 XMX, Gaudi 3 | NO | XMX dtypes fp16/bf16/int8/int4/int2 + BF8/HF8 |
| Intel Crescent Island | announced (MXFP4 XMX), sampling H2-2026, GA 2027 | — |
| Apple GPU / AMX / ANE, x86 AMX, ARM SME2 | NO | — |
| Strix Halo XDNA2 NPU | NO — BFP16 (shared exponent per 8), **not OCP MX** | — |

### Runtimes

| Runtime | Hardware | MXFP4 execution |
|---|---|---|
| llama.cpp CUDA | **sm_120/121 (incl. GB10)** | **[NATIVE] W4A4** — `ggml-cuda/mma.cuh:1138` emits the mxf4 block-scaled MMA; PR #17906 merged 2025-12-24. Needs `-DCMAKE_CUDA_ARCHITECTURES=121a-real` |
| llama.cpp CUDA | sm_100 (B200) | falls to int8 MMQ (gate is `CC ≥ 1200 && < 1300`) |
| llama.cpp CUDA/CPU | Turing–Hopper, AVX2/512, NEON | int8 MMQ / SIMD dot vs Q8_0 (no fp16 materialization) |
| llama.cpp Metal / Vulkan | Apple / any | **[UPCONVERT]** dequant → half / f16 coopmat |
| llama.cpp HIP/ROCm | all AMD | **[UPCONVERT]** — zero hits for `mfma_scale`/`wmma_scale`/`f8f6f4` in the repo |
| **vLLM** | sm_100 family only | **[NATIVE]** FlashInfer TRTLLM / CUTLASS / DeepGEMM — gate is literally `is_device_capability_family(100)` |
| **vLLM** | **sm_120 / sm_121 (Spark)** | **[UPCONVERT] Marlin W4A16.** `(cap//10)==(100//10)` fails for 12.1. vllm#30135; **vllm#37030 (open): the SM80-targeted Marlin MXFP4 kernel emits a wrong first Harmony token on SM121 — "No SM121-specific Marlin kernels exist."** vllm#31089 (Triton MXFP4 on SM120) closed unmerged 2026-03-16, slower than Marlin |
| vLLM | Hopper / Ampere / Ada | **[UPCONVERT]** Triton `mxfp4_to_bf16_triton` + `tl.dot`, or Marlin |
| vLLM-ROCm | `gfx950` / `gfx1250` | **[NATIVE]** AITER CK W4A4/W4A8 (`on_gfx950() or on_gfx1250()`) |
| vLLM-ROCm | `gfx942`, `gfx120x`, **`gfx1151`** | **[UPCONVERT]** — `supports_mx()` excludes them |
| TensorRT-LLM | Blackwell | **[NATIVE]**; DGX Spark gpt-oss MXFP4 beta in 1.2 |
| oneDNN / OpenVINO / MLX | Intel / Apple | **[UPCONVERT]** — weight decompression by construction |

`llama.cpp`'s ggml type is `GGML_TYPE_MXFP4 = 39`, `block_mxfp4 { uint8_t e;
uint8_t qs[16]; }`, `QK_MXFP4 = 32` — exactly OCP MX. It landed with gpt-oss
(PR #15091, merged 2025-08-05). A separate `GGML_TYPE_NVFP4 = 40` was added
~Mar–Apr 2026.

### Is the portability difference real?

Yes — but it is a difference in **reach, not rate**. On any part that has
*either* format, both run on the same unit at the same rate. On Blackwell they
are two `.kind` variants of one instruction — `mma.sync…block_scale` on
sm_120a/121a, `tcgen05.mma` on sm_100 — differing only in `.kind` and
`.scale_vec_size`; a B200 microbenchmark (arXiv:2512.02189 Table IV) shows both
lowering to a single `OMMA` SASS family. On CDNA4 both are operand-format
selections of the same `F8F6F4` unit. MXFP4 is an OCP
standard (MX v1.0, Sept 2023; MX Alliance = AMD, Arm, Intel, Meta, Microsoft,
NVIDIA, Qualcomm); NVFP4 is NVIDIA-defined and NVIDIA-only in silicon today.
But "MXFP4 runs everywhere" means **[UPCONVERT] everywhere except two vendors'
parts** — and on Strix Halo specifically, upconvert is the *only* option for any
4-bit weight format, MX or not.

---

## 5. Measurement

**Metric authority: this is a tier-5 screen.** Imatrix-weighted weight-MSE, not
a served metric, not KL. It is decisive here only because the effect sizes are
large and the comparison is exactly paired (same weights, same codebook
machinery, same imatrix, arms differing in one structural axis).

Setup: Qwen3-0.6B; imatrix `E[x²]` per input column from 32 × 1024 tokens of
`diverse-v1`; product-VQ d=8, n_sub=2, ceil-first bit split; **each arm learns
its own codebook at its own post-normalization data scale** (the 2026-07-15
lattice-scale lesson — a codebook trained at the wrong normalization collapses
reconstruction and would falsely kill a format).

Two fairness notes, both pointing *against* the conclusion this doc reaches, so
the conclusion survives them:

- The MX arms run the **exhaustive** `max`-equivalent sweep. Their pow2 candidate
  window `{ceil−3 … ceil+1}` is exhaustive over the useful pow2 lattice — the
  smaller candidate count reflects a poorer *lattice*, not a poorer search.
- `nv16_v2` runs the **production** `balanced` encode tier
  (`nvfp4_cb_formats._ENCODE_TIER_DEFAULT`), i.e. what actually ships, which is a
  slight handicap versus the exhaustive tier the MX arms get. The unhandicapped
  `nv16_v1` arm (16-candidate exhaustive, v1 scale coding) bounds that handicap.

### 5a. Same-harness RTN anchors (196 Linears, whole model, no codebook)

The commission's framing was "the 8-bit analogue was +13.8%; the 4-bit number is
the load-bearing unknown." Both, measured here in one harness
(`rtn_anchor.py`, shipping `format_registry` callables, imatrix-weighted):

| pair | aggregate | median/Linear | min | max |
|---|---|---|---|---|
| **8-bit** `MXFP8_E4M3` vs `FP8_E4M3` | **+1.69%** | +1.71% | +0.95% | +20.4% |
| **4-bit** `MXFP4` vs `NVFP4` | **+42.5%** | +42.2% | +39.3% | +64.9% |

The 4-bit E8M0 penalty is **~25× the 8-bit one**. The mechanism is
straightforward: at 8 bits the element format (E4M3, 4 exponent + 3 mantissa
bits) is itself scale-robust across a binade, so a power-of-two scale error
costs almost nothing; at 4 bits E2M1 offers 7 magnitudes spanning 12× with one
mantissa bit, and a ≤2× scale error strands or clips a large fraction of them
with no mantissa depth to absorb it.

*Honest scoping of the +13.8%:* that number is **output**-MSE over 410 Gemma
Linears (`ARCHITECTURE.md:2071`), a different metric on a different model. My
+1.69% is weight-MSE on Qwen3-0.6B and is **not** a reproduction of it — it is
the same-harness companion to the 4-bit number, which is the only comparison
that matters here. Both confound group size with scale dtype; §5c isolates them.

### 5b. CB ladder — matched rung and matched bytes

<!--CB-TABLES-->

### 5c. Decomposition

<!--CB-DECOMP-->

---

## 6. The interaction the commission asked to be stated explicitly

**No shipped fp4-CB path executes FP4 on a tensor core — on either grid.**

| path | what it actually runs |
|---|---|
| dense decode M≤8 | CUDA GEMV `cb_gemv_fp4_v2`, **bf16** codebook gather, act-QDQ'd x cast to **bf16** (`linear.py:367`, `codec.py:42`) |
| dense M 9–16 | Triton `cb_gemm`, same |
| dense prefill M>16 | `expand_fp4_v2_to_weight` → `torch.empty(..., dtype=torch.bfloat16)` → `F.linear` (cuBLAS **bf16**) (`expand.py:295`, `linear.py:399-402`) |
| MoE fp4-CB prefill | per-expert loop over the same bf16 expand (`STANDARDS.md:52`) |
| *only* native-TC path in the lane | **FP8_CB** → `cb_expand_fp8` → `cutlass_scaled_mm` W8A8 |

The code says so in its own words: *"bf16 MMA — INV-2 waived; the FP4-MMA
CUTLASS prefill is prototype iii"* (`linear.py:392-398`). Prototype (iii) is
scoped at ~1000–2000 LoC + a CUTLASS ramp, 15–25 days, "real risk it needs a
mainloop fork" (`serving-kernel.md`).

Three consequences:

1. **On Strix Halo the grid is irrelevant to serving.** `gfx1151` has no FP4 and
   no FP8 matrix path, so a HIP CB-decode kernel must decode to f16/bf16 no
   matter what grid the codebook lives on. The MXFP4 grid would be pure encode-
   side cost there. If the HIP lane is the motivation, this is a NO-GO on its
   own.
2. **Today the fp4 grid constraint is unpaid-for on NVFP4 too.** `rd_ceiling_
   study.md` prices it at **+4.5%** weighted MSE (full mode) / **+10.0%**
   (signed) versus an unconstrained codebook. Since every fp4-CB consumer takes
   bf16, an *ungridded* codebook — same bytes, same kernels, same scale plane —
   would be strictly better today on every platform. That is a smaller change
   than adding an MX family (drop the `_snap_to_grid` in the codebook learner)
   and it is the obvious thing to measure next. **The counter-argument is real
   and strategic:** the grid is what keeps prototype (iii) reachable; dropping it
   forecloses native-FP4 prefill permanently. Do not act on this without Robert.
3. **`LAYOUT.md:32-33` is stale.** It still says a decoded fp4 tile "feeds the
   existing CUTLASS FP4 path unchanged." `STANDARDS.md:49` (newer, and the
   contract page) says bf16 → cuBLAS. The code agrees with STANDARDS.

Also worth recording: `STANDARDS.md:38-39` already notes that CB artifacts with
no vanilla-NVFP4 units "remain Ada-servable as a bonus." That is the same fact
from the other side — **the CB lane is already arch-portable within CUDA
precisely because it decodes rather than executing FP4.** The portability barrier
is the gridbook decoder (CUDA/sm_120+ `csrc`, ~5 `.cu` + a CUTLASS fork), not the
grid. A CB artifact is not a wire format any runtime reads; it needs the plugin.
"Servable wherever MXFP4 GEMMs exist" would be true only *after* someone ported
the decoder to that platform **and** wrote a decode-into-native-MXFP4-tile
mainloop there — i.e. prototype (iii), on a less mature stack.

---

## 7. What an implementation would touch

| area | work |
|---|---|
| `nvfp4_cb_formats.py` | `FP4_GROUP` is a module constant with 35 references; `_fp4_group_scale`, `_per_element_scale`, `_group_amax`, `_group_reduce`, `_snap_scale`, `_candidate_scales`, `_type_size` all key on `grid == "fp4"` ⇒ 16. Needs a `(group, scale_lattice)` parameter threaded through, plus a **new** exponent-delta scale coder replacing `_sweep_encode_two_tier` (§3). ~250–400 LoC + tests |
| `format_registry.py` | `MXFP4_CB_K*` ladder specs (~25 LoC, mirroring `_make_nvfp4_cb_spec` `:913-...`) |
| `export_nvfp4_cb.py` (+ `_streaming`) | new `layout_version`, scale-plane packer, `quant_config` scheme, family tag. ~150–250 LoC across 865 lines |
| `nvfp4_cb_footprint.py` | byte accounting for the new `type_size` |
| gridbook `codec.py` / `linear.py` / `moe.py` / `expand.py` | an E8M0-delta expander parallel to `build_compose_table`, plus decode paths; `is_fp4`/`is_v2` branching is already 178 call sites of grid-conditional code |
| gridbook `csrc` | `cb_gemv.cu` fp4-v2 schedule + any fused path would need an MX variant; a *native* MXFP4 GEMM is prototype (iii) all over again |
| `serving_profile_specs/nvfp4_cb.json`, `layer_config.py:58-62` | rung names, allow-lists |

**What it would NOT solve:** it does not port the decoder to any non-CUDA
platform (the actual barrier); it does not make a CB artifact readable by any
stock runtime; it does not reach a tensor core on Strix Halo, RDNA4, CDNA3,
Intel, Apple or CPU, because none of them have an FP4 matrix path; it does not
change speed on Blackwell (same OMMA datapath); and it does not touch the CB
lane's actual quality lever, which per §9.2 is fp8 rungs running A8 where fp4
rungs run A4.

---

## 8. What would change this answer

1. **A CDNA4 (`gfx950`) HIP lane.** MI350X/MI355X is the one non-NVIDIA part with
   native MXFP4, at the same rate as MXFP6 (both 16384 FLOPs/clk/CU). If gridbook
   ever targets CDNA4, an MX-grid rung is the only fp4 CB rung that could reach
   its matrix core — a genuine GO case *on that hardware*. Note this is the same
   trigger `mxfp6_cb_feasibility.md` recorded, and on CDNA4 the MXFP6 rung is the
   more interesting one (fp6 at fp4 rate = free precision).
2. **Prototype (iii) landing on Blackwell**, at which point the grid stops being
   decorative and the NVFP4-vs-MXFP4 choice becomes a real 16-vs-32/E4M3-vs-E8M0
   accuracy question — which §5 has already answered against MX.
3. **A measured refutation of §5 at the very bottom of the ladder** (k≤14). The
   penalty is rung-dependent; see §5b for where it actually lands.
4. Intel Crescent Island shipping (GA 2027) would add a third native vendor.

**Graveyard lesson (for ARCHITECTURE §11), if this is filed there:** MXFP4 and
NVFP4 share an element grid, so an MX-grid codebook is not a handicapped
codebook (unlike MXFP6) — it is a *cheaper scale plane bought with a
power-of-two-only scale*. At 8 bits that trade is nearly free; at 4 bits the
E2M1 alphabet has no mantissa depth to absorb a ≤2× scale error and the trade
inverts hard. And in a lane that decodes to bf16, neither grid is buying
anything at all.

---

## 9. Provenance

- Model `/home/rob/models/Qwen3-0.6B`; calibration
  `/home/rob/dq-runs/calibration/diverse-v1.jsonl`; imatrix 32 × 1024 tokens.
- Repo `claude/docs-consolidation` @ `d6dbf58`. venv
  `/home/rob/dq-runs/venvs/prismaquant-cu130`. GPU-resident throughout; peak
  well under the 30 GB budget; the `:8000` serve was untouched.
- Scripts: `scratch/mxfp4_cb/{gridlab,measure_mse,scale_plane_entropy,rtn_anchor,report}.py`.
  Results JSON: `/home/rob/dq-runs/mxfp4-cb-study/`.
- **Not measured:** served KL, emulated whole-model KL, any MoE weights, any
  model above 0.6B. This doc's quality claims are weight-MSE screens and are
  labelled as such; nothing here is promotion-grade evidence under §2.3.
