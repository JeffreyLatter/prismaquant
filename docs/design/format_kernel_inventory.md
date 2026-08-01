# Format / kernel inventory — what the hardware actually does, and what to build

> **Frozen inventory snapshot.** Access to the sole gfx1151 machine was lost on
> 2026-07-31 and all Strix/ROCm implementation work was canceled. Gridbook's
> unqualified HIP prototype was removed. Strix rankings and build directions
> below are historical evidence only, not an active backlog; the Blackwell
> measurements remain useful context.

2026-07-30. Commissioned by Robert: *"take an inventory of the available formats
and build kernels to support them. We can allow the allocator to decide what
formats to use where."* This document is the **inventory and the ranked kernel-gap
list**; it authors no kernels.

**Rule of the document: every rate below was measured, or is cited to a specific
measurement.** Anything not measured is labelled `[UNMEASURED]` with the reason.
Two boxes are in scope:

| box | chip | mem | achieved copy BW |
|---|---|---|---|
| **GB10 / DGX Spark** ("sparky") | Blackwell `sm_121` | 128 GB unified | ~273 GB/s (cited: `docs/lanes/nvfp4-cb/cutlass-kernel-notes.md:13`) |
| **Strix Halo** (`192.168.1.200`) | RDNA3.5 `gfx1151`, 20 CU @ ≤2.9 GHz | 58 GB unified | **210.4 GB/s measured** |

---

## 0. Two corrections to the prior record, both measured

**(a) The "23 GB/s bf16 GEMV" on Strix was a clock artifact — it is dead.** The
GPU idles at sclk 1214 MHz and takes tens of seconds of sustained load to reach
~2.6 GHz. Under a 45 s pre-burn plus a 2000-iteration warmup, bf16 GEMV measures
**201–233 GB/s against a 210 GB/s copy ceiling** — i.e. *at* bandwidth, not 9×
below it. The 9× gap does not exist.

```
gemv bf16 [K×N], sustained, K-major-contiguous weight:
  2048×2048  150.9 GB/s     4096×4096  201.5 GB/s
  4096×11008 223.1 GB/s     8192×8192  204.9 GB/s
```
*(Layout is worth up to 2×: the same 4096×11008 GEMV run against an N-major
weight with a lazy `.t()` collapses to 108.6 GB/s. A Strix decode kernel must own
its weight layout.)* Every Strix number in this document was taken after clock
settling; each table records the observed sclk.

**(b) On GB10, int8 measuring slower than fp8 is *probably* kernel maturity after
all — but it does not matter, because int8's ceiling is fp8 parity.** Disassembly
(§1.1.1) shows int8 and fp8 issue the **same k=32 MMA shape** on `sm_121`
(`IMMA.16832` vs `QMMA.16832`), so the measured 0.81× gap is plausibly closable by
a better kernel — and closing it buys a **tie**, not a win, while still opening the
activation-outlier problem. The parent's suggestion to hunt a better cuBLASLt/CUTLASS
int8 path is therefore answered *statically*: the hunt has no prize. See
`DO-NOT-BUILD-1`.

---

## 1. The compute-path matrix, measured

### 1.1 GB10 (`sm_121`)

`torch 2.11.0+cu130`, venv `/home/rob/dq-runs/venvs/prismaquant-cu130`. N=4096 cubed.

| compute dtype | available? | measured 4096³ | vs bf16 | how it is reached |
|---|---|---|---|---|
| **bf16** | yes | **22.5 TFLOP/s** [P] | 1.00× | `torch.mm` → cuBLAS |
| **f16** | yes | `[UNMEASURED]` — not on the shipping path (all PQ renders are bf16) | — | `torch.mm` → cuBLAS |
| **fp8_e4m3** | yes | **48.0 TFLOP/s** [P] | **2.13×** | `torch._scaled_mm`; in serving, vLLM `cutlass_scaled_mm` (W8A8) |
| **fp8_e5m2** | dtype yes, GEMM no | `[UNMEASURED-BY-DESIGN]` | — | valid **kv-cache** dtype only; `FP8_E5M2` is registry-present, research-only, and has **no compressed-tensors scheme** (`export_native_compressed.py:7305-7308`) |
| **int8** | yes | **38.7 TOP/s** [P] | 1.72× | `torch._int_mm` → cuBLASLt. SASS `IMMA.16832.S8.S8.SAT` — **same k=32 MMA shape as fp8** (§1.1.1) |
| **int4** | **EMULATED, not native** | n/a — bounded **below int8** | ≤0.86× | `mma.m8n8k32.s4` *compiles* for `sm_121a`, but ptxas lowers it to **2× `IMMA.16816.S8.S8`** (§1.1.1). Emulation on the int8 datapath ⇒ ~½ the int8 rate. |
| **nvfp4 / e2m1 block-scaled** | **yes, native** | see §1.3 | ~2× fp8 (now ISA-confirmed) | `cutlass_scaled_fp4_mm` in vLLM (what every shipped AURA/NVFP4 artifact serves on). SASS **`OMMA.SF.16864.F32.E2M1.E2M1.E8`** — k=**64** |
| **mxfp4 (e2m1 + e8m0/g32)** | yes, native | same datapath as nvfp4 | — | `mma…kind::mxf4.block_scale`; registry `MXFP4`, rarely chosen |
| **fp6 (e2m3/e3m2)** | rides the 8-bit datapath | **no separate rate** | = fp8 | CUTLASS `kind::mxf8f6f4`, operands byte-padded. Established NO-GO — `docs/design/mxfp6_cb_feasibility.md` |

`[P]` = measured by the parent agent (established, cited, not re-run).

#### 1.1.1 What the ISA actually does — SASS-level probe (compile-only, zero GPU load)

The rate table above leaves two questions the box was too busy to answer by
benchmark. Both are answerable *statically*, by assembling each candidate `mma`
through **ptxas** (`nvcc -arch=sm_121a -cubin`) and disassembling the result — this
runs no kernel and consumed no GPU. Note `-ptx` alone is **useless** for this: it
passes inline asm through verbatim without validating it against the target ISA.

| PTX instruction requested | accepted for `sm_121a`? | **SASS actually emitted** | reading |
|---|---|---|---|
| `mma.m8n8k32…s32.s4.s4.s32` | yes | **2 × `IMMA.16816.S8.S8`** | **int4 is EMULATED** — silently lowered onto two int8 MMAs |
| `mma.m16n8k32…s32.s8.s8.s32` | yes | `IMMA.16832.S8.S8.SAT` | native int8, **k=32** |
| `mma.m16n8k32…f32.e4m3.e4m3.f32` | yes | `QMMA.16832.F32.E4M3.E4M3` | native fp8, **k=32** |
| `mma.m16n8k64.kind::mxf4.block_scale…e2m1…ue8m0` | yes | **`OMMA.SF.16864.F32.E2M1.E2M1.E8`** | native block-scaled fp4, **k=64** |
| `mma.m16n8k32.kind::f8f6f4…e2m1.e2m1` | yes | `QMMA.16832.F32.E2M1.E2M1` | fp4 *on the fp8 datapath* — k=32, i.e. **no fp4 speedup** |

Three conclusions, each load-bearing later:

1. **Blackwell has no int4 tensor core.** `s4` compiling is a trap: ptxas emits two
   `IMMA.16816` (2 × 2048 MACs of capacity to do one 2048-MAC `m8n8k32`), so int4
   runs at *best* half the int8 rate. This upgrades `DO-NOT-BUILD-2` from an
   assertion to a disassembly.
2. **The 2:1 fp4:fp8 ratio is confirmed at the ISA level, not just from CUTLASS
   marketing.** fp8 and fp4 issue from the same MMA family, but fp4's native form
   is `OMMA…16864` (**k=64**) against fp8's `QMMA…16832` (**k=32**) — double the K
   per instruction. This substantially firms up §1.3's ~96 TFLOP/s projection.
3. **There are two fp4 paths and only one is fast.** `kind::f8f6f4` accepts e2m1 but
   emits a **k=32 QMMA** — fp4 operands at the fp8 rate. Any fp4 kernel must target
   `kind::mxf4`/`mxf4nvf4` (OMMA, k=64) or it will do the work of fp4 at the speed
   of fp8. Relevant directly to gap #1: the gridbook fused mainloop today is on the
   f8f6f4 path (`sm120_cb_fused_mma.hpp:214-216` static-asserts `float_e4m3_t`).
4. **int8's ceiling on Blackwell is fp8 *parity*, never a win** — identical k=32 MMA
   shape. Measured 38.7 vs 48.0 (0.81×) is therefore plausibly closable by a better
   kernel, but the best case is a tie. See `DO-NOT-BUILD-1`.

### 1.2 gfx1151 (Strix Halo) — all measured here

`torch 2.9.1`, `torch.version.hip = 7.1.52802`. Fedora quirk: `hipcc` needs
`-L/usr/lib64 -lamdhip64`. Probes in `scratch/fmt_inventory/`.

**Library GEMM, `torch.mm` / `torch._int_mm` at 4096³, clocks settled (sclk ~1.95 GHz):**

| compute dtype | available? | measured | vs bf16 | how it is reached |
|---|---|---|---|---|
| **bf16** | yes | **23.8 TFLOP/s** (23.71 / 23.70 / 23.89 over 3 interleaved passes) | 1.00× | `torch.mm` → hipBLASLt |
| **f16** | yes | **26.4 TFLOP/s** | 1.11× | `torch.mm` → hipBLASLt |
| **f32** | yes | 2.42 TFLOP/s | 0.10× | (reference only) |
| **int8** | yes | **19.7–20.0 TOP/s** std layout · **27.9 TOP/s** with B column-major | 0.83× / **1.17×** | `torch._int_mm`. Layout swing is 1.4× — hipBLASLt int8 is immature. |
| **fp8_e4m3 / e5m2 / e4m3fnuz** | **ABSENT IN SILICON** | n/a | — | `torch._scaled_mm` raises *"only supported on CUDA ≥9.0/8.9 or ROCm MI300+"*; **and** the hardware has no fp8 at all — see the three-way proof below |
| **int4** | **PRESENT, verified** | see §1.2.1 | **2.06–2.86×** | raw `__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32`. **No library exposes it** — not hipBLASLt, not rocBLAS, not torch. |
| **nvfp4 / mxfp4 / fp6** | absent | n/a | — | no fp4/fp6 WMMA on RDNA3.5 |

**fp8 absence on gfx1151, proven three independent ways** (this closes the question):
1. WMMA compile-probe: every `..._fp8_...` / `..._bf8_...` WMMA builtin ABSENT [P].
2. VALU dot compile-probe: `__builtin_amdgcn_dot4_f32_fp8_fp8` and `..._bf8_bf8`
   fail with *"needs target feature **dot11-insts**"* — not enabled for gfx1151.
3. Even the **conversion** instruction is missing:
   `__builtin_amdgcn_cvt_f32_fp8` fails with *"needs target feature
   **fp8-conversion-insts**"*.

There is no fp8 on this chip in any form. Any fp8 path on Strix is software
unpacking to bf16, i.e. **slower than bf16** — never a speed lever.

Present integer paths, numerically verified on device (A=1,B=2, 16×16×16 ⇒ 32):

```
wmma_i32_16x16x16_iu4_w32   run=no error  c[0]=32   PRESENT + CORRECT
wmma_i32_16x16x16_iu8_w32   run=no error  c[0]=32   PRESENT + CORRECT
__builtin_amdgcn_sudot4  (VALU int8 dot4)  PRESENT (=8)
__builtin_amdgcn_sudot8  (VALU int4 dot8)  PRESENT (=8)
```
(`dot1-insts` / CDNA-style `sdot4`/`sdot8` are absent; the RDNA `sudot*` forms are
the correct spelling.)

#### 1.2.1 The gfx1151 silicon ceiling — WMMA instruction-rate probe

Library numbers conflate silicon with kernel maturity. Two probes separate them:
**register-resident** (pure issue rate, the ALU ceiling) and **LDS-fed** (operands
re-read from LDS every op — a real GEMM mainloop's fragment traffic). 160 blocks ×
256 threads, 40-iteration pre-burn.

| dtype | register-resident peak | vs bf16 | **LDS-fed** | **vs bf16** |
|---|---|---|---|---|
| f16 | 47.4 TFLOP/s | 1.00× | 26.8 TFLOP/s | 0.97× |
| **bf16** | **47.4 TFLOP/s** | 1.00× | **27.5 TFLOP/s** | 1.00× |
| **iu8** | 50.0 TOP/s | **1.06×** | **42.9 TOP/s** | **1.56×** |
| **iu4** | 97.4 TOP/s | **2.06×** | **78.7 TOP/s** | **2.86×** |

*(repeat-run drift 1.4%; bf16 re-measured last in both probes to rule out clock drift.)*

Two things fall out, and they invert the naive reading:

- **At the ALU, int8 is worth nothing on RDNA3.5 (1.06×).** iu8 WMMA issues at
  essentially the f16/bf16 rate. Only **iu4 doubles** (2.06×). This is a silicon
  fact, not a library gap — it is why §4 ranks an int8-only Strix kernel low.
- **Under realistic operand traffic the narrow types gain, because they move 4×
  fewer fragment bytes.** bf16 is LDS-bandwidth-limited to 58% of its own ALU peak
  (27.5 of 47.4); iu8 rises to 1.56× and **iu4 to 2.86×**. The LDS-fed column is
  the honest ceiling for a real GEMM.

**Validating the LDS-fed column as a predictor:** hipBLASLt's measured bf16 GEMM
(23.8 TFLOP/s) is **87% of the LDS-fed bf16 ceiling** (27.5). So the LDS-fed
number predicts achievable GEMM rate well. Applying the same 87%:

| projected achievable gfx1151 GEMM | rate | vs bf16 | status |
|---|---|---|---|
| bf16 | 23.8 TOP/s | 1.00× | **measured** (hipBLASLt) |
| int8 | ~37 TOP/s | ~1.56× | `[PROJECTED]` — torch's best today is 27.9, so ~33% is left on the table by the library |
| **int4** | **~68 TOP/s** | **~2.86×** | `[PROJECTED]` — **no kernel exists at any rate** |

`[PROJECTED]` = LDS-fed ceiling × the 0.87 efficiency hipBLASLt demonstrates for
bf16. This is the single load-bearing extrapolation in the document, and §4 names
the cheap experiment that would confirm or kill it.

### 1.3 Blackwell block-scaled fp4 — what the existing kernels establish

Per the brief, cited rather than re-benchmarked:

- **A block-scaled fp4 GEMM is reachable and is already the shipping path.** Every
  served AURA/NVFP4 artifact runs vLLM's `cutlass_scaled_fp4_mm` on Blackwell.
- **CUTLASS rates the fp4-only kinds (`mxf4`, `mxf4nvf4`) at "4× Hopper FP8" vs
  "2×" for `mxf8f6f4`** — the architectural 2:1 fp4:fp8 ratio
  (`docs/design/mxfp6_cb_feasibility.md:28-33`). **§1.1.1 independently confirms
  this at the ISA level on this exact chip**: native fp4 is `OMMA.SF.16864` (k=64)
  against fp8's `QMMA.16832` (k=32) — double the K per instruction. Against the
  measured fp8 48.0 TFLOP/s that implies **~96 TFLOP/s fp4** on GB10.
  `[UNMEASURED on this box]` — a bare 4096³ fp4 GEMM number was not obtainable in
  this session's window (the box ran a production build throughout; see §1.4). The
  projection now rests on a disassembled shape ratio rather than a vendor claim,
  which is why gap #1 is ranked first despite the missing benchmark.
- **A kernel must target the right fp4 kind.** `kind::f8f6f4` also accepts e2m1 but
  emits a k=32 `QMMA` — fp4 operands at the **fp8** rate, no speedup. Only
  `kind::mxf4`/`mxf4nvf4` reaches the k=64 `OMMA`.
- The gridbook fork's fp8 GEMM is at **0.91–0.99× of vLLM's `cutlass_scaled_mm`**
  (`cutlass-kernel-notes.md:33-44`) — i.e. the fork is at parity and the fork's
  rates can be read as the native rates.
- `torch 2.11` exposes `torch.float4_e2m1fn_x2` and `torch.float8_e8m0fnu`, so a
  block-scaled fp4 `torch._scaled_mm` path is *plausibly* reachable outside vLLM.
  `[UNMEASURED]` — probe written (`scratch/fmt_inventory/gb10_gemm.py`), not run.

### 1.4 What could not be measured, and why

- **GB10 fp4 / int8-layout / copy-BW re-measurements.** The box ran a 0.6B
  production build (`run-pipeline.sh` + `build_production_cache`, GPU 96%) plus an
  idle `:8000` vLLM serve for the entire session. Per the brief, no heavy GPU work
  was started. The probe is written and **queued behind a PID-based wait** on the
  two build processes (`scratch/fmt_inventory/wait_then_bench.sh` — PID polling,
  deliberately not `pgrep`, which self-matches and has deadlocked this repo before);
  it is a ~2-minute job that will run itself in the next free window.
  **Mitigation, and why the gap is not blocking:** the one question the fp4
  benchmark was needed for — *is fp4 ~2× fp8 on `sm_121`?* — was answered instead
  by static disassembly (§1.1.1: k=64 `OMMA` vs k=32 `QMMA`), which required no GPU
  and is arguably better evidence for a *ratio* than a single benchmark would be.
  What remains missing is the absolute TFLOP/s, which changes no ranking in §4.
- **A real int4 WMMA *GEMM* on gfx1151.** No library implements it, and authoring
  kernels is out of scope here. The instruction-rate + LDS-fed probes are the
  proxy, with the 0.87 efficiency factor stated explicitly above.
- **f16 on GB10** — irrelevant to the shipping path (all PQ renders are bf16).

---

## 2. CB-grid viability

**The anchor** (established, `docs/design/mxfp6_cb_feasibility.md`, do not
re-derive): *in a codebook format the stored weight stream is the k-bit **index**
stream. The element grid appears only in the codebook's **values**. Changing the
grid changes the codebook and the decode target — **it does not change the
artifact's byte count**.* Consequently a grid change is only ever a
**quality × compute-rate** question, never a storage question, and the decision
rule is:

> A grid G is worth a rung iff, relative to the incumbent grid E at the same k,
> either (i) G is **not** a subset of E — it can encode something E cannot — or
> (ii) G reaches a **faster compute path** than E on the target platform.
> If G ⊆ E *and* G computes no faster, G is **strictly dominated** (the MXFP6 result).

The two subset relations that decide everything below, verified numerically:

```
int8 → e4m3 :  176 of 255 values NOT exactly representable  ⇒ int8 ⊄ e4m3
int4 → e4m3 :  all 16 values exact                          ⇒ int4 ⊂ e4m3
bf16        :  strict SUPERSET of e4m3
```

### 2.1 The table

| grid | ladder today | stored (per 8-wt vector) | decode target | GB10 verdict | gfx1151 verdict |
|---|---|---|---|---|---|
| **fp8 e4m3** | **`FP8_CB_K28..K48`** (21 rungs, bpw = k/8, 3.500–6.000) | k-bit index + per-channel fp32 scale + codebook sidecar | e4m3 byte → W8A8 tensor core | **INCUMBENT, optimal** — the only CB grid that reaches a native tensor core | **weak** — no fp8 silicon, so it must decode to bf16; the e4m3 restriction then buys nothing |
| **fp4 e2m1** | **`NVFP4_CB_K12..K24`** (2.000–3.500), **`NVFP4_CB_S13..S16`** (2.125–2.500) | k-bit index + 16 e4m3 group-scale bytes / superblock (`+0.5` bpw) | e2m1 → *(no fp4 MMA path exists in gridbook — §3)* | ladder exists; **the kernel does not** | absent in silicon |
| **bf16** | **none** | identical k-bit index stream | bf16 → native bf16 MMA | **pointless** (bf16 MMA is 0.47× fp8; strictly slower for a superset gain) | **STRICTLY DOMINATES `FP8_CB`** — see §2.2 |
| **int8** | **none** | identical k-bit index stream | int8 → `IMMA` / iu8 WMMA | **no upside at any effort**: measured 38.7 TOP/s vs fp8 48.0, and the **same k=32 MMA shape** (§1.1.1) caps its best case at fp8 *parity* — bought with the activation-outlier problem | genuine but **weak** tradeoff: 1.56× bf16, and grid ⊄ e4m3 (176/255) so quality is *not* automatically worse |
| **int4** | **none** | identical k-bit index stream | int4 → iu4 WMMA | **STRICTLY DOMINATED**: int4 ⊂ e4m3 *and* int4 is **emulated as 2× `IMMA`** (§1.1.1) ⇒ worse grid *and* ≤0.86× bf16. Textbook MXFP6 case. | **genuine tradeoff**: 2.86× bf16 (the only real speed lever on the chip) bought with a strict grid handicap |
| **mxfp4** | — | — | e2m1 + e8m0 scale | **not a new grid** — same e2m1 elements as `NVFP4_CB`, only the scale plane differs | absent |
| **mxfp6 e2m3/e3m2** | — | — | — | **NO-GO, settled** (`mxfp6_cb_feasibility.md`) | n/a |

### 2.2 The one genuinely new result: on gfx1151, `BF16_CB` dominates `FP8_CB`

This is the MXFP6 argument run in reverse, and it is a **GO** by the same logic
that made MXFP6 a **NO-GO**.

`FP8_CB` earns its e4m3 grid on GB10 because e4m3 is the *compute* grid — the
decoded values feed a W8A8 tensor core directly. On gfx1151 there is **no fp8
compute** (§1.2, proven three ways), so an `FP8_CB` weight must be decoded
e4m3 → bf16 and multiplied on the bf16 MMA. At that point the e4m3 restriction is
pure loss: bf16 ⊃ e4m3, so a bf16-valued codebook

- stores **byte-for-byte the same** artifact (the index stream is unchanged),
- computes at **exactly the same rate** (both end at the bf16 MMA),
- and can represent every `FP8_CB` codebook plus strictly more.

Equal bytes, equal speed, superset grid ⇒ **strict dominance**. The VQ learner can
only do better. Nothing needs measuring to establish the direction; only the
*magnitude* of the quality gain is open.

**The one real constraint to size before building:** the codebook LUT doubles,
1 B → 2 B per entry. The CUDA k48 LUT is 32 KiB (`sm120_cb_fused_mma.hpp:167-172`);
a bf16 k48 LUT is **64 KiB**, which is the entire gfx1151 LDS allocation per
workgroup. So a `BF16_CB` ladder on Strix is comfortable at low-to-mid k and hits
an LDS wall near k48 — the rung ladder should be sized against LDS, not copied
from CUDA. `[UNMEASURED]` — exact per-rung LDS budget for a gfx1151 tile is for
whoever authors the kernel.

### 2.3 Cost of a new grid, if one is built

Sized from the existing in-repo estimate for exactly this exercise
(`docs/design/mxfp4_cb_feasibility.md:303-313`) and confirmed against the current
tree: **~250–400 LoC in `nvfp4_cb_formats.py`, ~25 in `format_registry.py`,
~150–250 in `export_nvfp4_cb.py`**, plus the gridbook serving side.

The grid-conditional choke points in `nvfp4_cb_formats.py` (each currently a hard
`if grid == "fp4" / elif "fp8" / else raise`): `_snap_to_grid` (:185-205),
`_product_n_sub` (:161-165), `_build_lattice` (:307-347) + the cached
`data/nvfp4_cb_lattices.pt`, `_scale_and_vectorize` (:449-464), `_per_element_scale`
(:442-446), `_group_amax`/`_group_reduce` (:524-536), `_snap_scale` (:538-543),
`_candidate_scales` (:546-557), `_type_size`/`nvfp4_cb_effective_bits`
(:1532-1556), `nvfp4_cb_assemble_bytes`/`nvfp4_cb_unpack` (:1674-1784),
`nvfp4_cb_reconstruct` (:1404-1405), and two inline `FP4_GROUP if grid=="fp4" else
in_f` hardcodes (:868, :936, :970).

Structural parameters are grid-independent and would not move: `VEC_DIM = 8`,
`SUPERBLOCK = 256`, `MAX_FLAT_K = 14`, and `_bit_split(k, n_sub)`.

**Serving-side breakages a non-e4m3 grid causes** (these are the real cost, not the
encoder): `linear.py:259-260` builds `_cb_flat_fp8 = cb_flat.to(torch.float8_e4m3fn)`
— lossless *only* because CB values are on the e4m3 grid; the whole prefill route
is expand → `float8_e4m3fn` tile → `cutlass_scaled_mm` (`expand.py:312-352`,
`linear.py:487-490`); and the CUDA GEMV hard-checks `cb_flat.scalar_type() == kUInt8`
(`csrc/cb_gemv.cu:497-498`). Only the Triton decode-GEMM (`kernels.py:34-178`) is
grid-agnostic — so **a new-grid rung with no new kernel ships on the slow
prototype path by construction.** Which is precisely why this document's
deliverable is a kernel list.

---

## 3. What exists vs what is missing — the (grid × platform × phase) cell map

### 3.1 CUDA / GB10 — the gridbook surface

Dispatch thresholds, all in external Gridbook `gridbook/linear.py`: `PREFILL_M_THRESHOLD
= 16` (:46), `CUDA_GEMV_M_MAX = 8` (:53), mid-M fused window `16 < M ≤ 128` (:437),
persistent-TC `M > 128` (:456).

| grid | decode (M ≤ 8) | low-mid (8 < M ≤ 16) | mid-M (16 < M ≤ 128) | prefill (M > 128) |
|---|---|---|---|---|
| **FP8_CB** | ✅ CUDA `cb_gemv_fp8` — `csrc/cb_gemv.cu`, fused act-QDQ, E4M3-byte LUT, smem-resident (R6). 250–355 GB/s effective | ✅ Triton `cb_gemm` (bf16 MMA) | ✅ **CUTLASS decode-in-prologue fused**, default ON, `csrc/cb_fused_gemm.cu` — **but only k ∈ {28,32,36,40,44,48}** | ✅ CUDA expand → `float8_e4m3fn` transient → vLLM `cutlass_scaled_mm` (W8A8). Opt-in persistent-N TC (`cb_persistent_tc.cu`) is quarantined behind `PRISMAQUANT_ENABLE_PTC=1` |
| **NVFP4_CB v1** (the *registered/shipping* accounting) | ❌ | ❌ | ❌ | ❌ — **Triton `cb_decode_linear` at EVERY M**, forced by `linear.py:359`. `kernels.py:15-20` self-labels it *"INV-2 WAIVED … will fail the perf gate by construction and must not be promoted."* |
| **NVFP4_CB v2** (two-tier, opt-in) | ✅ CUDA `cb_gemv_fp4_v2` | ✅ Triton | ⚠️ Triton expand → **bf16** tile → `F.linear` (cuBLAS bf16) | ⚠️ same — bf16, not fp4 |

**The headline CUDA gap, stated precisely: no CB path anywhere reaches an FP4
tensor core.** `expand_cb_to_value` explicitly refuses fp4 (`expand.py:373-379`:
*"a transient FP4 tile would still need the Blackwell FP4-MMA (prototype iii)"*),
and the fused mid-M mainloop static-asserts `ElementB == float_e4m3_t`
(`sm120_cb_fused_mma.hpp:214-216`). The only CB grid that touches a native tensor
core is fp8.

**Secondary gap:** the 15 non-step-4 `FP8_CB` rungs (K29, K30, K31, K33, K34, …)
are registered and allocator-legal but have **no fused mid-M kernel** — they
silently fall through to expand+cutlass in the `16 < M ≤ 128` window. The
allocator can pick a rung whose serving path is quietly worse than its neighbour's.

**Native (non-CB) on Blackwell, provided by vLLM/CUTLASS, no PQ kernel needed:**
NVFP4 W4A4 (`cutlass_scaled_fp4_mm`), FP8 W8A8 (`cutlass_scaled_mm`), MXFP4,
MXFP8, BF16. This is why the native lane ships on stock vLLM.

### 3.2 ROCm / gfx1151 — the surface is empty

| grid | decode | prefill |
|---|---|---|
| bf16 / f16 | ✅ hipBLASLt (23.8 / 26.4 TFLOP/s) | ✅ hipBLASLt |
| int8 | ⚠️ `torch._int_mm` requires M > 16 — **no int8 GEMV path at all** | ⚠️ 19.7–27.9 TOP/s, layout-fragile |
| int4 | ❌ nothing | ❌ nothing (WMMA exists; no library, no kernel) |
| fp8 / fp4 / fp6 | ❌ absent in silicon | ❌ absent in silicon |
| **any CB grid** | ❌ | ❌ |

A ROCm CB prototype was being wired when this inventory was written. That work
was canceled on 2026-07-31 after access to the sole gfx1151 machine was lost,
before serving and packaging gates could be completed. Its sources and dispatch
hook were removed from canonical Gridbook; there is no supported ROCm CB lane.
The following two observations are retained only as historical experiment notes:

- **It is landing on the fp8 grid, and §2.2 argues the grid should be bf16 on this
  chip.** That is gap #2, and it is a format decision, not a kernel decision — worth
  settling before the kernel surface hardens around e4m3 the way the CUDA side did.
- **Its decode target is 210 GB/s, and that is reachable** — sustained bf16 GEMV
  already measures 201–233 GB/s (§0a), so a CB GEMV at native-bandwidth parity is
  the correct acceptance bar, mirroring CUDA's 250–355 GB/s on a 273 GB/s part.

As of this inventory **no CB format serves on gfx1151 by any measured path**, and
no int4 kernel exists on either box.

---

## 4. RANKED KERNEL GAP LIST

Ranked by value-per-effort. Each entry: what it enables · the measured delta that
justifies it · effort · the accuracy question it opens · portability.

### BUILD FIRST

---

**#1 — Blackwell FP4-MMA CB prefill ("prototype iii"): the fp4 tensor-core path**
*(CUDA / GB10 only · effort **L** · no new accuracy question)*

- **Enables:** the entire `NVFP4_CB_K12..K24` / `S13..S16` ladder to serve at a
  competitive rate. Today it is Triton-only at every M and self-labelled
  unpromotable (§3.1). This is the difference between a **17-rung** low-bit ladder
  (13 flat K12–K24 + 4 signed S13–S16, spanning 2.000–3.500 bpw) that exists on
  paper and one that ships.
- **Delta:** fp8 measured **48.0 TFLOP/s**; native fp4 is `OMMA.SF.16864` (k=64)
  against fp8's `QMMA.16832` (k=32) — **2× the K per instruction, confirmed by
  disassembly on this exact chip** (§1.1.1) ⇒ **~96 TFLOP/s**
  `[UNMEASURED on this box — §1.3]`. Against the *current* fp4-CB path (Triton
  bf16 decode-GEMM at every M) the delta is far larger than 2×: the incumbent is
  not a tensor-core path at all.
- **Hard design constraint, from the disassembly:** the kernel must target
  `kind::mxf4` / `mxf4nvf4` (→ k=64 `OMMA`). The `kind::f8f6f4` path *also* accepts
  e2m1 operands but emits a k=32 `QMMA` — fp4 data at the **fp8 rate**, i.e. all of
  the work and none of the win. The existing fused mainloop is on exactly that
  f8f6f4 path today (`sm120_cb_fused_mma.hpp:214-216`), so this is a live footgun,
  not a hypothetical.
- **Accuracy question: none new.** Bit-exactness against the existing decode is
  the gate, exactly as for the fp8 fused kernel.
- **Why first:** it is the only gap that is *pure upside* — no grid change, no new
  rung, no quality tradeoff, an existing registered 17-rung ladder going unused,
  and the largest rate delta on the primary serving box. Everything else on this
  list trades something.
- **Cheap precondition (do this before committing to L):** run the queued
  `scratch/fmt_inventory/gb10_gemm.py` to put a measured number on Blackwell fp4.
  If fp4 does *not* measure ~2× fp8 on `sm_121`, #1 drops below #2. The ISA
  evidence makes that unlikely, which is why #1 is ranked first without it.

---

**HISTORICAL #2 — CANCELED: gfx1151 `BF16_CB` grid + decode GEMV and bf16-MMA prefill**
*(ROCm / gfx1151 · effort **M** · quality strictly ≥ `FP8_CB`)*

- **Enables:** CB artifacts to serve on Strix Halo **at all** (today: zero paths,
  §3.2), on the grid §2.2 proves is the correct one for that chip.
- **Delta:** decode is bandwidth-bound and the ceiling is **measured at 210 GB/s**
  with sustained GEMV already reaching **201–233 GB/s** — so a competent CB GEMV
  should land at native-bandwidth parity, mirroring the CUDA result (250–355 GB/s
  on a 273 GB/s part). Prefill lands on hipBLASLt bf16 at **23.8 TFLOP/s
  measured** — which, note, is *faster than GB10's own bf16* (22.5).
- **Accuracy question: none — it is a strict improvement.** bf16 ⊃ e4m3, so the
  codebook can only get better at identical stored bytes (§2.2). The single
  engineering constraint is the 2 B LUT hitting the 64 KiB LDS wall near k48.
- **Portability:** the *format* is portable (a bf16-grid rung is meaningful
  anywhere); the kernel is gfx1151-specific.
- **Why second:** highest value-per-effort of anything on Strix, and the only item
  here whose accuracy question is answered "strictly better" before any
  measurement.

---

**HISTORICAL #3 — CANCELED: gfx1151 int4-WMMA GEMM (W4A4), fed by an int4-grid CB or native INT4**
*(ROCm / gfx1151 · effort **L** · **opens a serious accuracy question**)*

- **Enables:** the only path to GB10-class prefill throughput on Strix. Strix has
  no fp8, so its prefill ceiling is bf16's 23.8 TFLOP/s — **half** of GB10's fp8
  48.0. int4 WMMA is the one lever that closes that gap.
- **Delta (the best-measured number in this document): iu4 WMMA is 2.06× bf16
  register-resident and 2.86× bf16 LDS-fed** — projecting to **~68 TOP/s vs 23.8**
  at the 0.87 efficiency hipBLASLt demonstrates for bf16 (§1.2.1).
- **Accuracy question — flagged, not hand-waved, and it is the real one.**
  `iu4` WMMA requires **both** operands int4, i.e. **W4A4 integer activations**.
  Integer A4 is far more outlier-hostile than fp4: e2m1's exponent gives it dynamic
  range within a 16-element group that a 16-level uniform integer grid does not
  have. This is the exact failure mode that pushed the field from int8 to fp8, and
  it is worse at 4 bits. Two mitigations exist and both cost some of the 2.86×:
  per-group int32 accumulation with a float scale applied per K-group (standard,
  cheap), and a rotation/smoothing transform on the activation side (a
  `(format, transform)` question, and note `ReSpinQuant`-class layer-wise rotations
  are already graveyarded for needing a serve-time adapter).
  **Additionally**, int4 ⊂ e4m3 (verified) — so an int4-grid *codebook* is a strict
  grid handicap versus `FP8_CB` at the same k. On gfx1151 that is a legitimate
  tradeoff (it buys 2.86× where fp8 buys nothing); on GB10 it would be strict
  domination, which is why this item is single-platform.
- **De-risking experiment before committing (cheap, hours not days):** author one
  int4 WMMA tile and measure whether it reaches the projected ~0.87 of the LDS-fed
  78.7. If a realistic tile lands near bf16 instead, the whole item dies and the
  `[PROJECTED]` row in §1.2.1 is falsified. **Do this before the W4A4 accuracy work.**

---

### BUILD IF CHEAP

**#4 — Fill the `FP8_CB` mid-M fused rung holes (K29–K31, K33–K35, …)**
*(CUDA · effort **S** · no accuracy question · single-platform)*

The mid-M CUTLASS fused kernel is templated on k ∈ {28,32,36,40,44,48} only; the
other 15 registered rungs fall through to expand+cutlass in the `16 < M ≤ 128`
window (§3.1). The allocator can already select them. Delta is bounded by the
fused kernel's in-niche win — **1.04× / 1.26× / 1.45× at M = 32 / 64 / 128**
(`cutlass-kernel-notes.md`) — so this is a small but genuinely free win: pure
template instantiation against an existing, bit-exact kernel. **Alternative worth
pricing first: if the ladder does not need 0.125-bpw granularity, de-registering
the unbacked rungs is a 5-line change that removes the same footgun for zero
kernel work.**

**#5 — gfx1151 int8 GEMM/GEMV cleanup**
*(ROCm · effort **M** · int8-activation outlier question · single-platform)*

Two measured defects: `torch._int_mm` swings **19.7 → 27.9 TOP/s** purely on B
layout, and it **refuses M ≤ 16 entirely** (`self.size(0) needs to be greater than
16`) — so Strix has no int8 decode path at all. The LDS-fed ceiling says ~37 TOP/s
is available (§1.2.1), i.e. ~33% is being left on the table. **But the prize is
capped at 1.56× bf16**, and int8 activations carry the classic outlier problem for
that modest return. Worth doing only as a by-product of #3 (an int4 kernel's
infrastructure covers int8), not as its own project.

---

### DO NOT BUILD

**DO-NOT-BUILD-1 — an int8-grid CB rung, or any int8 serving path, on GB10.**
Measured: int8 **38.7 TOP/s** vs fp8 **48.0 TFLOP/s** (0.81×). More decisive than
the measurement, though, is the disassembly: int8 and fp8 issue the **identical
k=32 MMA shape** on `sm_121` (`IMMA.16832.S8.S8.SAT` vs `QMMA.16832.F32.E4M3.E4M3`,
§1.1.1). **int8's best case is a tie with fp8**, bought at the price of the
activation-outlier problem that pushed the whole field from int8 to fp8. This also
**settles the parent's open question** — a better cuBLASLt/CUTLASS int8 path might
well recover the 24%, but the ISA proves there is nothing beyond parity on the far
side, so the probe is not worth a GPU window. No upside exists at any effort.

**DO-NOT-BUILD-2 — an int4-grid CB rung on GB10 / CUDA.** Two independent
disqualifications. (i) int4 ⊂ e4m3 (verified: all 16 values round-trip exactly), so
the grid is a strict handicap versus the incumbent `FP8_CB` at the same k. (ii)
Blackwell has **no int4 tensor core**: `mma.m8n8k32.s4` compiles for `sm_121a` — a
trap — but ptxas lowers it to **two `IMMA.16816.S8.S8`**, so it runs at ≤½ the int8
rate, itself ≤ fp8. Worse grid *and* slower path: the textbook MXFP6
strict-domination case. Zero. *(Note this is the exact inverse of gap #3, which is
the same grid on the other box — the difference is entirely that gfx1151's iu4 WMMA
is native and 2.86×, where Blackwell's is emulation.)*

**DO-NOT-BUILD-3 — any fp8 path on gfx1151.** Proven absent three independent ways
(§1.2): no WMMA builtin, no `dot11-insts` VALU dot, not even
`fp8-conversion-insts`. Software fp8 unpacks to bf16 and is therefore *slower than
just using bf16*. Corollary: **`FP8_CB` should not be the CB grid shipped to
Strix** — that is what #2 exists to fix.

**DO-NOT-BUILD-4 — MXFP6-grid CB.** Settled NO-GO,
`docs/design/mxfp6_cb_feasibility.md`. Not re-derived. (Its one revisit condition —
CDNA4/MI355X, where MXFP6 runs at 2× fp8 — is *not* satisfied by gfx1151, which is
RDNA3.5 and has no fp6 at all.)

**DO-NOT-BUILD-5 — a bf16-grid CB rung on GB10.** The mirror of #2: on Blackwell
fp8 *is* a compute grid at 2.13× bf16, so widening the codebook to bf16 would
forfeit that 2.13× to buy a superset the fp8 grid barely loses anything by
excluding. `FP8_CB` is correct on CUDA and stays.

---

## 5. Allocator integration sketch — what a new (grid, platform) needs

*(one paragraph per item; no implementation.)*

**Registry `FormatSpec`.** Each rung is registered via `register_format` in
`format_registry.py`. The CB ladders use a deliberate accounting trick worth
preserving: `weight_bits=0`, `group_size=256` (the superblock), and the entire
k-bit index stream declared in `scale_bits`, so `effective_bits` returns
`scale_bits/256` — `32k` for `FP8_CB` (⇒ exactly `k/8`) and `32k+128` for
`NVFP4_CB` (⇒ `k/8 + 0.5`, the 16 e4m3 group-scale bytes). A new grid adds one
`_make_<grid>_cb_spec` factory plus a register loop, ~25 LoC. Note the module-level
docstring at `:908-910` describing the fp8 accounting is **stale** — the inline
comment at `:955-960` is the correct one; a new grid should follow the code, not
that docstring. `FormatSpec` carries **no** served-vs-research field; the only
hardware hint is `min_capability_sm`, which is a CUDA compute-capability integer
and **has no ROCm meaning** — a genuinely multi-platform registry needs a target
key that is not an `sm` number, and that is a prerequisite for any gfx1151 rung
rather than an afterthought.

**Serving-profile allow-list.** Allocator-selectability is gated by
`serving_profile_specs/*.json` through `allocator_candidates.check_format_applicability`.
`allow_formats` is a **hard whitelist** — non-empty and absent ⇒ denied — and for
the CB lane the rung name must be added in **two** hand-maintained arrays in
`nvfp4_cb.json`: `format_rules[0].allow_formats` and `shape_rules[0].formats` (the
latter is what applies `in_features_multiple_of: 256`). A new platform wants its
own profile (`extends: nvfp4_cb`) rather than more rows in the CUDA one, because
`export_lane` and the kernel-shape constraints differ per platform; note
`emulation_only` is deliberately **not** inherited through `extends`, so a new
profile must name its export lane or declare itself emulation-only — fail-closed
by design.

**Export lane.** Beyond the profile, the name must appear in
`export_lane.emittable_formats()`, which for the CB lane resolves to
`export_nvfp4_cb:EXPORTABLE_FORMATS` = `frozenset(_NVFP4_CB_FORMAT_NAMES) |
{NVFP4, FP8_E4M3, FP8_SOURCE, BF16}` — so the name must *also* be in
`layer_config.py:61-65`, which hand-duplicates the k ranges from `format_registry`.
This is a structural bound applied **after** the profile rules: no allow-list can
widen past what the exporter can actually emit. Three hand-maintained copies of the
same rung set (`format_registry` loop, `layer_config` set, two `nvfp4_cb.json`
arrays) is the standing footgun; a new grid is the natural moment to derive them
from one source.

**Cost path.** Grids are not commensurable through the existing surrogate for
free: the allocator's per-(Linear, format) cost comes from the render-score /
AURA path, which measures rendered `dW` — so a new grid needs a working
`quantize_dequantize` closure (`make_nvfp4_cb_qdq`) and nothing more *in
principle*, because cost is measured, not modelled. The real constraint is core
principle 1: the cost must be measured on the **same rendering** that ships, so a
new grid must land in `ProductionWeightCache` before its costs mean anything. For
grids whose value is *platform-specific speed* rather than quality (#3), the
accuracy axis alone will rank them **below** `FP8_CB` at equal k by construction —
which is correct and is exactly the blindness §4(d) of
`docs/design/format_choice_4p5.md` exists to fix.

**Per-platform serving-time entry for the §4(d) budget.** `format_choice_4p5.md`
§4(d) proposes a second explicit budget, `serve_ms(assignment) ≤ (1+T)·serve_floor`,
where `serve_ms = Σ_phases w_phase · Σ_units ms(format_u, shape_u, phase)` with
phases `{prefill, decode@shipped-M}`. This inventory supplies the missing
**per-platform** dimension: that table must be keyed `(format, shape-regime, phase,
**platform**)`, because the same format has categorically different entries on the
two boxes — `FP8_CB` prefill is a 48.0 TFLOP/s tensor-core path on GB10 and a
23.8 TFLOP/s bf16 path on gfx1151, and `NVFP4` has **no entry at all** on gfx1151.
The measurements in §1 are the seed rows for the platform axis of that table; the
`auto` tuner's per-layer ms logs remain the per-shape source. §4(d) is explicitly
**PROPOSED, not implemented**, and this document does not change that status — it
only records that a single-platform ms table would be wrong the moment a gfx1151
artifact exists. *(That file is Robert's to edit; nothing here modifies it.)*

---

## Appendix — reproducing these numbers

Probe scripts, all in `scratch/fmt_inventory/` (none touch shipping modules):

| file | what it measures | where it runs |
|---|---|---|
| `strix_gemm.py` | first pass — **exhibits the clock artifact** (sclk 1214→2647 mid-run) | Strix |
| `strix_gemm2.py` | 45 s pre-burn, then interleaved bf16/f16/int8 passes + sustained GEMV + layout variants | Strix |
| `wmma_full.hip` | numerical verification of `iu4`/`iu8` WMMA (expect 32) | Strix |
| `wmma_rate.hip` | register-resident WMMA instruction-rate ceiling per dtype | Strix |
| `wmma_lds.hip` | **LDS-fed** WMMA rate — the realistic GEMM-mainloop ceiling | Strix |
| `dot_probe.hip` | fp8/bf8/int VALU dot + fp8 conversion availability | Strix |
| `ptx2.cu`, `ptx3.cu` | **ptxas/SASS probe** of `s4` / `s8` / `e4m3` / `e2m1` MMA on `sm_121a` — the §1.1.1 table. Compile-only, runs no kernel | GB10 (no GPU needed) |
| `gb10_gemm.py` | bf16/fp8/int8-layouts/**block-scaled fp4**/copy BW | GB10 — **written, queued, not yet run** (box busy all session) |

SASS probe line (this is the one that matters — `-ptx` does **not** validate inline
asm, only `-cubin` runs ptxas):
```
/usr/local/cuda-13.0/bin/nvcc -arch=sm_121a -DP_<X> -cubin ptx2.cu -o p.cubin \
  && /usr/local/cuda-13.0/bin/cuobjdump -sass p.cubin | grep -oE '(IMMA|HMMA|OMMA|QMMA)[.A-Z0-9_]*'
```
(`nvcc` lives at `/usr/local/cuda-13.0/bin/nvcc`; `/usr/local/cuda/bin/nvcc` does
**not** exist on this box and silently produces empty results if used.)

Build line on Fedora ROCm: `hipcc -w -O3 -x hip --offload-arch=gfx1151 X.hip -o X
-L/usr/lib64 -lamdhip64` (the `-L/usr/lib64 -lamdhip64` is required or you get
`undefined symbol: __hipUnregisterFatBinary`).

**Clock discipline for any future Strix measurement:** the GPU idles at
sclk 1214 MHz / 38 W and needs tens of seconds of sustained load to reach
~2.6 GHz / 66 W. Pre-burn before measuring, and re-measure the baseline dtype
*last* to bound clock drift (done throughout: bf16 repeated 3× within 0.8%).
