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
| numerically correct | **YES** | 42/42 standalone + 38/38 pytest cases, gated at **1 bf16 output ULP** against an fp64 torch/CPU reference |
| benchmarked | **YES** | numbers below, `hipEvent`-timed **at sustained clock** (see the methodology note — the first attempt measured the idle clock and was thrown away) |
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

### The LDS-vs-global policy: measured, and it is an occupancy trade

At N = K = 4096, M = 1, all timings at sustained clock, with the bf16 LUT:

| rung | LUT (bf16) | LDS-LUT | global | verdict |
|---|---|---|---|---|
| K36 | 8 KiB | **0.103 ms** | 0.179 ms | LDS **+74%** |
| K40 | 16 KiB | **0.114 ms** | 0.178 ms | LDS **+56%** |
| K42 | 24 KiB | 0.214 ms | **0.180 ms** | global **+19%** |
| K44 | 32 KiB | 0.230 ms | **0.192 ms** | global **+20%** |

The cliff sits between **16 and 24 KiB**, so the shipped default is *stage if it
costs ≤ 16 KiB* (`kLdsLutMaxBytes`), overridable with
`PRISMAQUANT_CB_HIP_LUT=lds|global`.  Note this threshold is stated in **LDS
bytes**, not in rungs, so it survives a change of LUT dtype — the earlier
byte-LUT measurement put the same cliff at the same *byte* figure but at a
different *rung* (K44), which is exactly the confusion the byte framing avoids.

**Honest limit:** calibrated at N = K = 4096.  At N = 16384, K = 8192 the global
arm wins even at K44, so a shape-aware policy (LUT bytes × workgroup count) is
open work, flagged rather than papered over.

---

## Measured performance

All numbers at sustained clock (2280–2666 MHz, printed per shape), `hipEvent`
timing, 20 warmup + 50 (GEMV) / 20 (GEMM) timed iterations.

### The GEMV, against the baseline that actually matters

Since stock bf16 GEMV already saturates this machine, the meaningful baseline is
**a perfectly bandwidth-bound bf16 GEMV of the same logical matrix**: `N*K*2`
bytes at the ~210 GB/s copy ceiling.  The ratio below is what the *format* buys
after the decode gives some of it back — that is the number this lane lives or
dies on, and it is reported for the best arm at each rung.

| rung | bpw | LUT | best arm | time | vs bf16-BW-bound floor |
|---|---|---|---|---|---|
| K36 | 4.50 | 8 KiB | LDS | 0.103 ms | **1.55×** |
| K40 | 5.00 | 16 KiB | LDS | 0.114 ms | **1.40×** |
| K42 | 5.25 | 24 KiB | global | 0.180 ms | 0.89× |
| K45 | 5.62 | 40 KiB | global | 0.182 ms | 0.88× |
| K46 | 5.75 | 48 KiB | global | 0.197 ms | 0.81× |
| K47 | 5.88 | 56 KiB | global | 0.199 ms | 0.80× |
| K48 | 6.00 | 64 KiB | global | 0.184 ms | 0.87× |

(N = K = 4096 throughout, so the rungs are directly comparable.  Two other
shapes, for scale: K44 at 5120×4096 → 0.192 ms = 1.04×; K44 at 16384×8192,
a 92 MB tensor well past the 32 MB MALL → 1.396 ms = 0.92×.)

**Read this plainly: the format's byte advantage survives only while the LUT is
LDS-resident.** At K36–K40 the decode GEMV genuinely beats what a bf16 GEMV
could do on this machine *at its bandwidth ceiling* — 1.40–1.55×. From K42 up it
does not: the ratio collapses to a flat **0.80–0.89×** and, tellingly, stays
flat while the byte count keeps rising.  Flat-in-bytes means the top rungs are
**decode-bound, not bandwidth-bound** — the bytes saved are being spent on the
gather, and the cliff coincides exactly with the LUT leaving LDS.

The practical consequence for Strix is concrete and worth acting on:
**prefer the K≤40 rungs**, which is also where a 58 GB box wants to be. It also
sharpens the open work — the top-rung fix is a decode-cost problem (a partial
LDS LUT, say), not a bandwidth problem, so anything aimed at load efficiency
there will miss.

### The WMMA prefill GEMM

| rung | shape | M | time | throughput |
|---|---|---|---|---|
| K44 | 5120×4096 | 128 | 0.452 ms | **11.9 TFLOP/s** |
| K44 | 5120×4096 | 512 | 1.908 ms | 11.3 TFLOP/s |
| K36 | 4096×4096 | 128 | 0.354 ms | 12.1 TFLOP/s |

Against a spec-derived peak of ~54 TFLOP/s (40 CU × 512 FLOP/clk × 2.65 GHz —
a *derivation*, not measured here) that is ~22%. The GEMM is still the right
call at prefill — at M=512 it beats looping the GEMV by ~7× — but it is
untuned, and the reason is known and stated below rather than guessed at.

**Large shapes are worse, and that is the diagnostic:** at N = 16384, K = 8192
the GEMM falls to 5–6 TFLOP/s while the GEMV holds its rate. The GEMM decodes B
**4×** redundantly (lanes 16–31 duplicate lanes 0–15 by ISA requirement, and the
two waves sharing a `wave_n` decode the same fragment), which is invisible when
the codebook is L0-resident and dominant when it is not.

## What is validated, and what is only authored

**Validated on device** — 34 pytest cases (`tests/test_hip_decode_parity.py`)
and 44 standalone cases (`cb_hip_selftest`), all at a 1-bf16-ULP gate:

* FP8_CB decode GEMV across **K28, K29, K32, K33, K36, K40, K44, K45, K47, K48**
  — deliberately including the odd rungs, whose ceil-first sub-split gives
  unequal sub-tables and is where an "even k" assumption would break;
* M ∈ {1, 2, 3, 4, 8, 16} — every register-tile boundary, including M values
  that fall between tiles and must be predicated off rather than read OOB;
* both LUT variants (LDS-staged and global), byte-for-byte the same answer;
* **both codebook grid sources** — an e4m3-byte sidecar and a bf16 sidecar
  holding the same values give **bit-identical** output at K28/36/44/48, on
  both LUT arms (the LUT dtype contract);
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

1. **Get the top rungs back above the bf16 baseline (biggest lever).**  K42+
   currently sits at 0.89–1.04× a bandwidth-bound bf16 GEMV, i.e. the format's
   byte advantage is being spent on decode.  The cause is identified — the LUT
   stops being LDS-resident at the occupancy cliff — so the candidates are a
   *partial* LDS LUT (stage the two hottest sub-tables, gather the rest from
   global), or a nibble-packed fp4 LUT for the NVFP4 ladder, or simply accepting
   that Strix prefers K≤40.  All three are measurable; none is implemented,
   because the third may well be the right answer and costs nothing.
2. **B-decode redundancy in the GEMM.**  B is decoded **4×** (ISA lane
   duplication plus the two waves sharing a `wave_n`).  Evidence: at
   N = 16384, K = 8192 the GEMM drops to 5–6 TFLOP/s while the GEMV holds rate.
   Fix: decode B once into LDS per workgroup and broadcast — trading the
   zero-LDS-traffic design for one staged tile, which is only worth it because
   decode, not load, is the cost.  Note this competes with lever 1 for the same
   64 KiB.
3. **Shape-aware LUT policy.**  The 16 KiB threshold inverts at large N.  The
   honest fix is a two-variable policy (LUT bytes × workgroup count) seeded from
   a sweep, not a second hand-picked constant.
4. **K-loop double buffering in the GEMV.**  There is none; the CUDA lane
   measured this as a win in its own (different) shape and it is untested here.
5. **`int8` / `int4` WMMA — measured, and explicitly NOT started.**
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
6. **MoE grouped kernels.**  Absent entirely; an MoE artifact on ROCm falls back
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
