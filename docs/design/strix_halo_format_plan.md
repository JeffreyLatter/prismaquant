# Strix Halo: which native formats, which CB formats

**Status: plan, proposed 2026-07-30.** Implements nothing. Every hardware claim
below was measured on the actual gfx1151 box this session (compile-probe or
on-device run); anything unmeasured is labelled. Commissioned by Robert: *"come
up with a plan about what native formats to support on Strix Halo and what CB
formats you want to implement there."*

## 1. The three facts the plan rests on

**(a) The gfx1151 matrix-op inventory** — by compiling each builtin against the
real target (`--offload-arch=gfx1151`), not from documentation:

| WMMA builtin | gfx1151 |
|---|---|
| `wmma_f32_16x16x16_bf16_w32` | **PRESENT** — and numerically verified on device (16×16×16, A=1, B=2 → 32.0) |
| `wmma_f32_16x16x16_f16_w32` | PRESENT |
| `wmma_i32_16x16x16_iu8_w32` | PRESENT |
| `wmma_i32_16x16x16_iu4_w32` | PRESENT |
| `wmma_f32_16x16x16_fp8_*_w32` | **ABSENT** |
| `wmma_f32_16x16x32_f4_w32` | **ABSENT** |
| `wmma_scale_f32_16x16x128_f8f6f4` (block-scaled) | **ABSENT** |

RDNA 3.5 is a **bf16/f16/int** matrix machine. There is no fp8 and no fp4
tensor-core path, and no block-scaled MMA of the kind Blackwell's NVFP4 uses.

**(b) Every fp8 and fp4 codeword is exact in bf16** (verified numerically: 254
finite e4m3 values, 250 e5m2, all 16 e2m1 — every one round-trips bf16
losslessly). Therefore decoding an FP8-CB or NVFP4-CB codebook to bf16 is
**bit-lossless with respect to the codebook**. Strix serves the artifacts we
already ship, at identical quality, with no re-encode and no format fork.

**(c) Decode is bandwidth-bound; prefill is compute-bound.** Measured on the
box: 208 GB/s achieved copy bandwidth, 58 GB RAM. At batch-1 decode, arithmetic
intensity is ~1 and speed ≈ model bytes ÷ bandwidth, so the compute dtype is
irrelevant there; a 30B-class model is ~60 GB at bf16 (**does not fit**) and
~17 GB at 4.5 bpw. Prefill is where the compute dtype bites.

## 2. Native formats (compressed-tensors lane, vanilla vLLM-ROCm)

The shipped native menu does not survive the trip. `NVFP4` needs fp4 +
block-scaled MMA (both absent); `FP8_E4M3`/`FP8_DYNAMIC` need fp8 MMA (absent).
Either could only run by dequantising to bf16 in software — which is what the CB
lane already does, but better (finer rungs, fitted codebook).

| Native format | gfx1151 path | Verdict |
|---|---|---|
| `NVFP4` (W4A4) | none — no fp4, no block-scaled MMA | **not viable** |
| `FP8_E4M3` / `FP8_DYNAMIC` | none — no fp8 MMA | **not viable** |
| `INT8` W8A8 | `iu8` WMMA | **viable — the only compressed native format with real silicon** |
| `INT4` weight-only (W4A16) | `iu4` WMMA | viable; needs an activation story |
| `BF16` | native | passthrough baseline |

**Plan for the native lane: support INT8 W8A8, and nothing else new.** It is the
one native format the hardware accelerates, and it earns its place by being
*fast where CB is slow* — `iu8` WMMA should beat a bf16 decode in prefill (rate
UNMEASURED on this part; first benchmark). Its costs are equally clear: 8 bpw
against CB's 4.5 (≈1.8× the bytes ⇒ proportionally slower decode by (c)), and
int8 activations reopen the outlier problem that pushed the field to fp8 — a
real accuracy question, not a footnote. INT4 weight-only is a later question,
gated on whether anything needs it that the fp4-CB ladder does not already serve
better.

## 3. CB formats

**Plan: implement no new CB formats. Implement the decode kernels.**

By fact (b), the shipped ladders — `NVFP4_CB_K12..K24` (2.0–3.28 bpw) and
`FP8_CB_K28..K48` (3.5–6.0 bpw) — already serve on Strix losslessly through a
bf16 decode. The grid was only ever there to make the decoded tile bit-standard
for fp8/fp4 tensor cores; where those do not exist, the constraint costs nothing
and buys nothing, and the artifact is unchanged.

**A bf16-grid ("unconstrained") CB for Strix is DO-NOT-BUILD.** It would buy the
grid-constraint cost back — measured at **+0.2–0.7% weighted MSE** in
`rd_ceiling_study.md` — at the price of forking the artifact per platform,
doubling encode, and breaking the one-artifact property that makes (b) valuable.
The trade is not close.

**An int8-grid CB is DO-NOT-BUILD.** Measured on the GB10: int8 38.7 TOP/s vs
fp8 48.0 TFLOP/s — int8 is *slower* than fp8 on Blackwell, so an int8-grid
artifact would regress the primary platform to help the secondary one. Same
argument as MXFP6-CB (`mxfp6_cb_feasibility.md`): in a codebook the grid is a
kernel-side decode target, not a storage dial, so a platform-specific grid is
never the right lever — a platform-specific *kernel* is.

### The kernels to build, in order

| # | Kernel | Why | Effort |
|---|---|---|---|
| **K1** | CB decode **GEMV** → bf16, LDS-resident codebook LUT | The decode-phase workhorse and the whole product on this box: bandwidth-bound, so this is where 4.5 bpw becomes tokens/s. Stock ROCm bf16 GEMV measured **23 GB/s against 208 GB/s available** — if that gap survives sustained-clock re-measurement, a purpose-written kernel has ~9× of headroom. | M |
| **K2** | CB **decode-in-prologue GEMM** → bf16 WMMA | Prefill. Carries the lane's original thesis onto this box: tensor-core prefill instead of GGUF-IQ's dequant-bound path. | L |
| **K3** | *(conditional)* int8 decode target for K2 | Only if `iu8` benchmarks materially above bf16 WMMA **and** the activation-outlier accuracy question is settled on a served A/B. Strix-local optimisation, never an artifact change. | M |

K1 and K2 are already being authored against the real toolchain. K3 stays a
measured option, not a commitment.

## 4. How the allocator decides (Robert's framing, made concrete)

Both lanes end up in one menu and the allocator picks per Linear, as it always
has — but the menu is now **platform-conditional**, because availability and
rate differ per box. Two mechanisms, both already designed:

1. **Availability** — `serving_profile_specs/` gains a Strix profile whose
   allow-list is `{FP8_CB_*, NVFP4_CB_*, INT8, BF16}` and whose deny-list
   carries the *reason* (`NVFP4: no fp4 tensor-core path on gfx1151`). Denial
   rules are institutional memory; that is why the registry keeps formats it
   cannot serve everywhere.
2. **Speed** — the serving-time budget of `format_choice_4p5.md` §4(d):
   `serve_ms ≤ (1+T)·serve_floor` over `{prefill, decode@shipped-M}`, fed by a
   per-(format, shape-regime, **box**) timing table. This is what lets the
   allocator trade INT8's prefill rate against CB's decode bytes *by
   measurement* instead of by operator judgment — and Strix is the first box
   that makes the table genuinely two-valued, which is R21's own stated trigger
   for the machinery to earn its keep.

The honest consequence: **on Strix the interesting frontier is CB-vs-INT8, not
CB-vs-NVFP4** — NVFP4 is not on the menu there at all. The 4.5-bit question
(`format_choice_4p5.md`) is a Blackwell question; its Strix analogue is
"fp8-CB@k versus INT8 W8A8", a different experiment with the same machinery.

## 5. What would change this plan

- `iu8`/`iu4` WMMA measuring *below* bf16 on gfx1151 (kills K3 and weakens the
  INT8 native case).
- The 23 GB/s stock-GEMV figure proving to be a low-power-clock artifact rather
  than a real gap (shrinks K1's headroom, though not its necessity).
- vLLM-ROCm shipping a competent NVFP4 or fp8 dequant path for gfx1151 (would
  make the native lane viable-but-slow rather than not-viable).
- A future AMD part with fp8/fp4 matrix ops (RDNA4/CDNA4 already have them) —
  at which point Strix's kernels become the legacy path and the Blackwell
  design ports directly.
