# gridbook HIP kernels — RDNA 3.5 / gfx1151 (Strix Halo)

ROCm implementation of the CB (codebook) serving path: a decode GEMV for the
decode regime and a decode-in-prologue bf16 **WMMA** GEMM for prefill, plus the
transient expander and the fused activation QDQ.  The CUDA lane's equivalents
live in `../csrc/cb_gemv.cu`; the on-disk format both consume is
`docs/lanes/nvfp4-cb/LAYOUT.md` and `docs/lanes/nvfp4-cb/two-tier-scale-spec.md`.

---

## Status — 2026-07-30

Authored **and validated on real hardware**: an AMD Ryzen AI MAX+ 395 (Radeon
8060S, `gfx1151`), Fedora 44, ROCm 7.1.1, torch 2.9.1 (`torch.version.hip =
7.1.52802`).

| | state | evidence |
|---|---|---|
| compiles | **YES** | `hipcc --offload-arch=gfx1151 -O3`, zero warnings-as-errors, first-attempt clean for the kernels |
| runs | **YES** | standalone `cb_hip_selftest` and the torch extension both execute |
| numerically correct | **YES** | 34/34 pytest cases + 44/44 standalone cases, gated at **1 bf16 output ULP** against an fp64 torch/CPU reference |
| benchmarked | **YES** | numbers below, `hipEvent`-timed, 50 (GEMV) / 20 (GEMM) iterations after 5 warmups |
| served end-to-end under vLLM | **NO** | not attempted — no vLLM-ROCm on the box, and the plugin's dispatch integration is authored but never exercised inside a live serve |
| accuracy-validated (KL / PPL) | **NO** | no served artifact, so no serving-metric claim of any kind is made here |

**No performance claim is made against the CUDA lane.** Different silicon,
different memory system; the numbers below stand on their own.

### Measurement methodology, and a retraction

**gfx1151 idles at ~1.2 GHz and needs tens of seconds of continuous load to
reach its ~2.9 GHz boost.**  A conventional "5 warmup iterations" is single-digit
milliseconds of load here, so it measures the *idle* clock.  The first revision
of this file reported numbers taken that way, plus a "~28% of nominal bandwidth"
framing derived from them.  **Those numbers are retracted**; everything in the
performance section below was re-taken after `sustain_clock()` drives the device
for 45 s and the achieved clock is printed next to each result.  Nothing on this
box should be benchmarked without it.

A second correction, and a more important one: an earlier framing implied this
lane had large bandwidth headroom to reclaim from stock kernels.  It does not.
Measured properly, **stock bf16 GEMV reaches 201–233 GB/s against a ~210 GB/s
copy ceiling** — the machine is already saturated by ordinary kernels, and there
is no multiple to be won by out-coding hipBLAS.

**So what is the decode GEMV actually for?**  The index stream, not the
arithmetic.  A CB rung reads **k/8 bits per weight** — 4.5 bpw at K36, 5.5 at
K44 — where a bf16 weight reads 16.  On a bandwidth-saturated machine serving a
memory-bound decode step, that ratio *is* the win, and it is a property of the
format rather than of the kernel.  The kernel's job is to spend that saved
bandwidth without giving it back to decode overhead; the honest way to read the
GEMV numbers below is "how much of the format's bandwidth advantage survives
the decode", not "how much faster than hipBLAS".

---

## The load-bearing fact: WMMA on RDNA 3.5

Measured, not read: `wmma_probe.hip` resolves `__has_builtin` **in the device
compilation pass** (in the host pass every `__builtin_amdgcn_*` reports absent —
a naive file-scope `#if` therefore reports "no WMMA" on a device that has it,
which is exactly what the first bring-up run did) and reports what the compiler
accepts for `--offload-arch=gfx1151`:

| builtin | gfx1151 |
|---|---|
| `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32` | **PRESENT** |
| `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` | **PRESENT** |
| `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` | **PRESENT** |
| `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32` | **ABSENT** |
| `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` | **ABSENT** |
| `__builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32` | **ABSENT** |

**There is no fp8 matrix instruction on RDNA 3.5** — it arrives with gfx12
(RDNA4), where the builtins carry the `_gfx12` suffix and the `wmma-128b-insts`
feature.  So for this lane **fp8 is a storage format**: FP8_CB codewords are
decoded to bf16 fragments and multiplied on the bf16 WMMA unit with f32
accumulate.  That is not a compromise — an e4m3 value has 3 mantissa bits and
bf16 has 7, so the decode is *lossless*, and the arithmetic is strictly more
accurate than an fp8-MMA path would be.

### Fragment layout (established on device, not from documentation)

`wmma_probe.hip` fixes a convention for A and B, then searches for the D mapping
that makes the product come out right.  A GEMM only ever needs a *self-consistent*
layout — a fixed permutation applied to both the A load and the D store cancels —
so this is sufficient and is what the GEMM is written against:

```
A (16x16): lane l holds row (l % 16), register j holds k = j
B (16x16): lane l holds col (l % 16), register j holds k = j
           (lanes 16..31 duplicate lanes 0..15 — required by the wave32 encoding)
D (16x16, 8 f32/lane): register i of lane l holds D[2*i + l/16][l % 16]
```

The two rejected candidates (`D[8*(l/16)+i][l%16]`, `D[i+8*(l/16)][l%16]`) miss
by O(100) on the same data, so the identification is unambiguous.

**Why this matters architecturally:** a B fragment is 16 *consecutive* k values
of *one* output channel — and a CB codeword covers exactly 8 weights.  So **one
lane fills its whole B fragment by decoding exactly two codewords**, with no
cross-lane traffic.  The GEMM therefore needs **no LDS tile for A or B**, no
bank-conflict padding, and no barrier in the K loop.  Only the codebook LUT is
in LDS, and it is read-only after a single barrier at workgroup entry.

### WMMA arithmetic fidelity (measured, and it changes the test design)

The unit is **not** exactly-rounded IEEE.  With operands that are exact small
integers — where a plain `fmaf` chain on the same device is exact to the last
bit — the WMMA result is off by ~0.5 f32 ULP:

| operands | WMMA max abs err | relative | same product via `fmaf` |
|---|---|---|---|
| A = I, B = int[-7,8] | 4.77e-07 | 5.96e-08 | **0** |
| A = 1, B = int[-7,8] | 1.91e-06 | 5.61e-08 | **0** |
| A, B ~ N(0,1) | 2.38e-07 | 4.65e-08 | **0** |

101 of 256 accumulator slots came back non-integer on integer inputs.  This is
~one f32 rounding — negligible for serving, and far inside a bf16 output step —
but it means **bit-equality with a CPU reference is not an available gate**, so
every parity test here is a 1-bf16-ULP relative gate plus a norm backstop, the
same discipline as `tests/test_cuda_gemv.py::_assert_triton_close`.

### Device properties the launchers assert

`warpSize = 32` · `sharedMemPerBlock = 65536 B` · `multiProcessorCount = 20`
(WGPs; the part has 40 CUs) · `maxThreadsPerBlock = 1024` · 58 GiB unified.
Wave32 is not a preference: 32 codewords per 256-weight superblock is what makes
the lane↔codeword mapping exact, and the host launcher refuses a non-32 wave
rather than miscompute.

---

## The LUT dtype contract, and the LDS budget it implies

gfx1151 has **no fp8 anything**, so an FP8_CB codeword has to be materialised as
bf16 for WMMA no matter what the sidecar stored.  That makes the codebook's grid
a free choice on this platform: a **bf16-grid** codebook is the same bytes, the
same kernel cost, and a strict superset of the e4m3 grid — so it can only be
better quality.  The likely artifact design is one index stream plus a per-grid
codebook (a codebook is ~0.02% of artifact bytes, so carrying both is nearly
free), with Blackwell reading the e4m3 table and Strix the bf16 one.

These kernels are therefore **agnostic to the sidecar dtype**, with one rule:

> The LDS LUT is **always materialised as bf16**, and any e4m3 → bf16
> conversion happens **exactly once, at LUT-fill time**.  The hot loop gathers
> bf16 out of LDS and never learns what the sidecar held.

Mechanically: `stage_lut_bf16` (`cb_decode_hip.h`) is the single conversion
point; `gather_fp8_bits<SRC>` is instantiated `CB_SRC_BF16` on the LDS path and
`CB_SRC_E4M3` only on the global-gather fallback.  The WMMA prologue consumes
those bits directly — they *are* the operand format — so on the staged path the
B-fragment decode carries **zero** conversion ALU.  This is the ALU term the
CUDA lane's R6 work removed on Blackwell, and it stays removed here.

Correctness of the contract is pinned two ways, both green: the standalone
harness runs every rung with a bf16 sidecar built from the e4m3 one, and
`test_bf16_grid_sidecar_is_identical` asserts the two agree **bit-exactly**
(`torch.equal` on the raw bf16 bits) rather than merely closely — which is
available precisely because e4m3 → bf16 is exact.

The cost of the contract is LDS footprint: **a materialised LUT is 2 bytes per
codebook element**, twice the on-disk e4m3 size.  That is the budget the rung
ladder has to be designed against, against 64 KiB per workgroup.

**FP8_CB (n_sub = 4, sub_dim = 2) — bf16 LUT bytes:**

| rung | LUT | rung | LUT | rung | LUT |
|---|---|---|---|---|---|
| K28 | 2 KiB | K35 | 7 KiB | K42 | 24 KiB |
| K29 | 2.5 KiB | K36 | **8 KiB** | K43 | 28 KiB |
| K30 | 3 KiB | K37 | 10 KiB | K44 | **32 KiB** |
| K31 | 3.5 KiB | K38 | 12 KiB | K45 | 40 KiB |
| K32 | 4 KiB | K39 | 14 KiB | K46 | 48 KiB |
| K33 | 5 KiB | K40 | **16 KiB** | K47 | 56 KiB |
| K34 | 6 KiB | K41 | 20 KiB | K48 | **64 KiB** |

**NVFP4_CB (n_sub = 2, sub_dim = 4) — bf16 LUT bytes:**

| rung | LUT | rung | LUT | rung | LUT |
|---|---|---|---|---|---|
| K12 | 1 KiB | K17 | 6 KiB | K22 | 32 KiB |
| K13 | 1.5 KiB | K18 | 8 KiB | K23 | 48 KiB |
| K14 | 2 KiB | K19 | 12 KiB | K24 | **64 KiB** |
| K15 | 3 KiB | K20 | **16 KiB** | S13–S16 | 0.5–4 KiB |
| K16 | 4 KiB | K21 | 24 KiB | | |

### The top rungs fit here, for a structural reason worth stating

The usual objection to a bf16 LUT is that K48 and K24 need the *entire* 64 KiB
and therefore cannot coexist with the GEMM's operand tiles.  **That objection
does not apply to this GEMM**, and it is not luck: because the wave32 fragment
layout lets each lane fill its whole B fragment from two codewords, A and B are
**register-resident and never touch LDS at all** (see above).  The LUT is the
only LDS consumer in either kernel, so the full 64 KiB is available to it.

Verified rather than argued: K48 (64 KiB) and the whole fp4 ladder launch and
pass parity on the staged path.  So *feasibility* is not the constraint at the
top rungs — **occupancy** is, and that is a measurement, not a budget question.
See the policy below: the top rungs are served by the global-gather arm because
it is measurably faster there, not because the LUT would not fit.  Neither
LUT splitting nor a packed LUT with per-gather conversion is implemented, and
neither is needed for the rungs that matter on a 58 GB box.

### The LDS-vs-global policy is measured, and it is shape-dependent

Staging is an occupancy trade, not a free win.  At N = K = 4096, M = 1:

| rung | LUT | LDS-LUT | global | verdict |
|---|---|---|---|---|
| K36 | 4 KiB | 0.147 ms | 0.172 ms | LDS **+17%** |
| K44 | 16 KiB | 0.171 ms | 0.184 ms | LDS **+8%** |
| K45 | 20 KiB | 0.269 ms | 0.173 ms | global **+36%** |
| K46 | 24 KiB | 0.281 ms | 0.185 ms | global **+34%** |
| K47 | 28 KiB | 0.280 ms | 0.185 ms | global **+34%** |
| K48 | 32 KiB | 0.270 ms | 0.178 ms | global **+34%** |

The cliff is exactly between 16 and 20 KiB, so the shipped default is *stage if
it costs ≤ 16 KiB* (`kLdsLutMaxBytes`, `cb_gemv_hip.hip`), overridable with
`PRISMAQUANT_CB_HIP_LUT=lds|global` for an A/B.

**Honest limit on that default:** it was calibrated at N = K = 4096.  At
N = 16384, K = 8192 (a 92 MB packed tensor, larger than the 32 MB MALL) the
global variant wins even at K44 — 1.285 ms vs 1.594 ms, **global +24%**.  So the
16 KiB threshold is a *shape-local* measurement, and a shape-aware retune is
open work, flagged rather than papered over.

---

## Measured performance

`cb_hip_selftest --bench`, gfx1151, `hipEvent` timing.  "GB/s (packed)" is the
packed weight stream divided by kernel time — the quantity the decode regime is
supposed to be bound by.

**K44, N = 5120, K = 4096 (14.4 MB packed — MALL-resident, so this is a
cache-rate, not a DRAM rate):**

| M | GEMV LDS | GEMV global | GEMM |
|---|---|---|---|
| 1 | **0.171 ms** (84.6 GB/s) | 0.184 ms | — |
| 4 | 0.242 ms | 0.255 ms | — |
| 8 | 0.343 ms | 0.364 ms | — |
| 16 | 0.560 ms | 0.569 ms | — |
| 32 | — | — | 0.514 ms (2.61 TFLOP/s) |
| 64 | — | — | 0.576 ms (4.66 TFLOP/s) |
| 128 | — | — | 0.668 ms (8.03 TFLOP/s) |
| 256 | — | — | 1.131 ms (9.49 TFLOP/s) |
| 512 | — | — | **2.31 ms (9.30 TFLOP/s)** |

**K44, N = 16384, K = 8192 (92 MB packed — exceeds the 32 MB MALL, so this is a
DRAM-resident rate):** GEMV M=1 global **1.285 ms = 71.8 GB/s**.

Two readings worth stating plainly:

* **The WMMA prefill GEMM is the win it was built to be.** At M = 512 it runs
  2.31 ms where looping the GEMV would take ~32 × 0.56 = 17.9 ms — **~7.8×**.
  That is the whole reason the kernel exists.
* **Both kernels are well short of the machine.** 71.8 GB/s against ~256 GB/s of
  LPDDR5X is ~28% of nominal; 9.3–10.1 TFLOP/s (the range across two runs)
  against a spec-sheet ~59 TFLOP/s bf16 peak (40 CU × 512 FLOP/clk × 2.9 GHz —
  a derivation, not a measurement here) is ~16–17%.  These are first-cut
  kernels, and the gap is not mysterious; see "deferred" below.

---

## What is validated, and what is only authored

**Validated on device** — 34 pytest cases (`tests/test_hip_decode_parity.py`)
and 44 standalone cases (`cb_hip_selftest`), all at a 1-bf16-ULP gate:

* FP8_CB decode GEMV across **K28, K29, K32, K33, K36, K40, K44, K45, K47, K48**
  — deliberately including the odd rungs, whose ceil-first sub-split gives
  unequal sub-tables and is where an "even k" assumption would break;
* M ∈ {1, 2, 3, 4, 8, 16} — every register-tile boundary, including M values
  that fall between tiles and must be predicated off rather than read OOB;
* both LUT variants (LDS-staged and global), byte-for-byte the same answer;
* the fused-module case (3 roles, distinct codebooks, `cb_row_offset` spanning
  blocks) — the case where a workgroup's staged LUT is valid for only some of
  its rows;
* the transient expander, **byte-exact** against an independent reference
  decode, at K28/36/44/47/48;
* the WMMA prefill GEMM at K36/K44/K48, M ∈ {17, 32, 64, 96, 200}, including
  ragged M (17) and ragged N (100) edge tiles;
* decode contract v2 (scale in the epilogue) as well as the default v1;
* **NVFP4_CB two-tier (layout v2)** GEMV at K12–K24, exercising the E8M0 super +
  4-bit sub compose (`compose[E*16 + c]`), cross-checked against
  `codec.build_compose_table` for exact equality of the table itself.

**Authored but NOT validated:**

* the **signed S-rung** path (`n_sub == 1`) — compiled and reachable, no test;
* `linear_hip.maybe_apply` dispatch inside a live vLLM serve — the extension is
  exercised directly from python, never through the plugin;
* MoE / grouped-expert kernels — **not written at all**.  The CUDA lane's
  `cb_moe_gemv_*` have no HIP counterpart, so an MoE artifact falls through to
  Triton on ROCm;
* the fp4 prefill GEMM — the fp4 decode-to-bf16 prologue is the same shape as
  the fp8 one, but it is not written and there is no measured demand.

---

## Bring-up sequence (day one on a fresh ROCm box)

```bash
# 0. toolchain: hipcc, rocm-hip-devel, rocminfo (+ a ROCm torch for step 3)
rocminfo | grep -m1 gfx                      # expect gfx1151

# 1. capability probe + kernel parity, standalone (no torch, no vLLM)
plugins/gridbook/gridbook/csrc_hip/compile_check.sh
#    -> "probe PASS" then "ALL PASS (0 failures)"

# 2. benchmarks
plugins/gridbook/gridbook/csrc_hip/compile_check.sh --bench

# 3. the torch extension + the pytest gate
python3 -m pip install --user ninja pytest   # torch's JIT needs ninja
PYTHONPATH=plugins/gridbook python3 -m pytest \
    plugins/gridbook/tests/test_hip_decode_parity.py -v
```

Step 1 is the real gate: if it fails, nothing above it can be trusted.  Step 3
additionally proves the packaging/JIT path (`gridbook.hip_ext`), which is where
every environment-specific problem below showed up.

### Landmines, all hit for real during this bring-up

1. **`hipcc` needs an explicit runtime link on Fedora.**
   `hipcc --offload-arch=gfx1151 x.hip -o x` fails with
   `ld.lld: error: undefined symbol: __hipUnregisterFatBinary`.
   Add `-L/usr/lib64 -lamdhip64`.  `compile_check.sh` detects and applies this.

2. **`__has_builtin` for `__builtin_amdgcn_*` must be evaluated in the device
   pass.** In the host pass it is always false, so a file-scope `#if` silently
   compiles the kernel out and reports "no WMMA on this device".

3. **torch's ROCm JIT compiles for every arch the wheel supports** unless
   `PYTORCH_ROCM_ARCH` is set.  Passing `--offload-arch` in `extra_cuda_cflags`
   does **not** suppress it: `_get_rocm_arch_flags` scans the *cxx* flag list,
   which is built before `extra_cuda_cflags` is appended.  `hip_ext.py` sets the
   env var.  Symptom: a build that fails on `gfx1010`.

4. **`#include_next` breaks when torch puts `/usr/include` on `-isystem`.**
   Every source dies on `fatal error: 'math.h' file not found` from
   `/usr/include/c++/16/cmath:55` — while the identical source compiles fine
   under bare `hipcc`.  clang de-duplicates include dirs keeping the first, so
   `/usr/include` is hoisted out of its natural late position and
   `#include_next` has nothing left to search.  `hip_ext.py` appends an
   `-idirafter` shim directory of symlinks to `/usr/include/*.h`, on **both** the
   host and device flag lists (the host TU fails identically on `stdlib.h`).

5. **Do not build the torch binding with hipcc.**  `__HIPCC__` makes torch's
   `headeronly/util/complex.h` include `<thrust/complex.h>`; rocThrust is a
   separate package Fedora 44 does not ship, so the binding fails to compile on
   a box where the *kernels* compile perfectly.  `cb_hip_torch.cpp` is a `.cpp`
   for exactly this reason, and it contains no device code.

6. **Use the masquerading c10 HIP API.**  A ROCm torch reports tensors as
   `DeviceType::CUDA`; `c10::hip::HIPGuard` rejects them outright
   (`HIPGuardImpl initialized with non-HIP DeviceType: cuda`), and — worse —
   `c10::hip::getCurrentHIPStream()` reads a *different* stream slot than the
   one torch's own kernels use, which would be a silent ordering bug rather than
   an error.  Use `at::hip::getCurrentHIPStreamMasqueradingAsCUDA()` and the
   device-type-agnostic `c10::OptionalDeviceGuard`.

7. **`.hip` sources are still run through hipify.**  `load()` passes every
   source to `hipify_python` regardless of extension, and hipify's kernel-launch
   rewriter is textual and mis-parses template arguments containing commas.
   These sources therefore use `hipLaunchKernelGGL(HIP_KERNEL_NAME(...))`, and
   `hip_ext.py` stages a copy of the sources into the build directory so hipify
   can never write back into an installed package.

---

## Deferred performance work (measurable levers, none guessed)

Listed with the measurement that motivates each; none is implemented, and none
should be until it can be A/B'd on this box.

1. **B-decode redundancy in the GEMM (biggest lever).**  Lanes 16–31 duplicate
   lanes 0–15 by ISA requirement, and the two waves sharing a `wave_n` decode
   the same B fragment — so B is decoded **4×**.  The evidence: at
   N = 16384, K = 8192, M = 32 the GEMM moves the packed stream at 10.8 GB/s
   where the GEMV does the same work at 71.8 GB/s.  Fix: decode B once into LDS
   per workgroup and broadcast, trading the current zero-LDS-traffic design for
   one staged tile — which is only worth it because the decode, not the load, is
   the cost.
2. **Shape-aware LUT policy.**  The 16 KiB threshold inverts at large N (above).
   The honest fix is a two-variable policy (LUT bytes × workgroup count),
   seeded from a sweep, not a second constant.
3. **K-loop double buffering in the GEMV.**  There is none; the CUDA lane
   measured this as a win in its own (different) shape and it is untested here.
4. **`int8` / `int4` WMMA — measured, and explicitly NOT started.**
   `iu8` and `iu4` WMMA are present on this device, and the throughput has been
   measured elsewhere on this box: **iu8 is 1.56x bf16 when LDS-fed but only
   1.06x register-resident**, and **iu4 is 2.86x**.  The iu8 shape of that
   result is the informative part — its gain is halved LDS *traffic*, not
   faster math, so it would buy little in a GEMM whose operands are already
   register-resident, which is exactly what this one is.  Both also require
   integer activations of matching width, and an accuracy gate on that is
   running.  **No int8 or int4 variant is to be started until that gate
   reports.**  (An earlier draft of this file speculated "~2x matrix
   throughput" for iu8 from vendor generalities; that was wrong in magnitude
   and wrong about the mechanism, and is retracted here rather than quietly
   edited.)
5. **MoE grouped kernels.**  Absent entirely; an MoE artifact on ROCm falls back
   to Triton today.

## Files

| file | role |
|---|---|
| `cb_decode_hip.h` | the format: bit extraction, ceil-first sub-split, codebook gathers, two-tier compose, e4m3/bf16 conversions.  Shared by both kernels so they cannot drift. |
| `cb_gemv_hip.hip` | decode GEMV (fp8 + fp4-v2), fused fp8 activation QDQ, transient expander, host launchers |
| `cb_gemm_hip.hip` | bf16 WMMA decode-in-prologue prefill GEMM |
| `cb_hip_torch.cpp` | torch bindings (host-only `.cpp` — see landmine 5) |
| `cb_hip_selftest.hip` | standalone parity + benchmark harness, torch-free |
| `wmma_probe.hip` | capability + fragment-layout + arithmetic-fidelity probe |
| `compile_check.sh` | day-one bring-up driver |
