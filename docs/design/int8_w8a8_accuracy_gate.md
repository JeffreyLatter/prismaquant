# INT8 vs FP8 at W8A8 — the accuracy gate for the Strix int8 kernel direction

2026-07-30. **Screen, not a result.** Every number is a *whole-model emulated*
forward KL-vs-BF16 on Qwen3-0.6B / 4B — metric rung 3–5, nothing served, no
promotion language. Emulated activation quant is known to *under-count* the
served penalty (`aura_4b_render_discrepancy`: ~3× missed at 4B), so the
int8-activation penalty below is a **lower bound**.

**Question.** `gfx1151` has no fp8 tensor-core path; int8 WMMA measures 1.56×
bf16 LDS-fed (42.9 vs 27.5 TFLOP/s, established elsewhere). WMMA needs *both*
operands int8, so int8 prefill forces int8 **activations** — the problem that
drove the field to fp8. Does a uniform int8 grid hold at W8A8?

## Method

2×2 separating the weight grid from the activation grid. RTN only, no GPTQ, so
the render lever cannot confound it. Held-out `wiki.test.raw`, full-vocab fp32
KL, teacher-confident lane at p>0.5 — the repo's own conventions, reused verbatim
from `prismaquant/emu_forward_kl.py`. Block Linears only, `lm_head` excluded;
0.6B runs **two disjoint 8192-token windows** (single-draw 8×512 is known-noisy),
4B runs 4096. The int8 grid is a scratch construct, deliberately **not** a
`FormatSpec`. Its granularity matches `FP8_E4M3` exactly and is *proved*, not
asserted (`check_granularity.py`): both arms use a symmetric
**per-output-channel** weight scale and a dynamic **per-token** activation
scale, verified by scale-equivariance to <1e-7.

## The 2×2 — KL vs BF16 (`w1`/`w2` = the two disjoint 0.6B windows)

| arm | weights | acts | conf-KL w1 / w2 | conf-KL 4B | vs A (0.6B / 4B) | all-KL w1 / w2 | all-KL 4B |
|---|---|---|---|---|---|---|---|
| **A** | fp8 e4m3 | fp8 e4m3 | 0.01254 / 0.01143 | 0.01042 | 1.00× | 0.02121 / 0.02046 | 0.01558 |
| **B** | **int8** | fp8 e4m3 | **0.00828 / 0.00851** | **0.00681** | **0.66–0.74× / 0.65×** | 0.01494 / 0.01467 | 0.01022 |
| **C** | fp8 e4m3 | **int8** | 0.02475 / 0.02015 | 0.04225 | 1.76–1.97× / **4.05×** | 0.04266 / 0.03701 | 0.05989 |
| **D** | **int8** | **int8** | 0.02011 / 0.01695 | 0.03659 | 1.48–1.60× / **3.51×** | 0.03457 / 0.03038 | 0.05448 |
| refW | fp8 e4m3 | bf16 (A16) | 0.00715 / 0.00627 | 0.00572 | 0.55–0.57× / 0.55× | 0.01246 / 0.01146 | 0.00889 |
| refW | **int8** | bf16 (A16) | 0.00275 / 0.00256 | 0.00188 | 0.22× / 0.18× | 0.00570 / 0.00572 | 0.00322 |

PPL over BF16 (w1/w2/4B): A +1.45/+1.55/+0.19%, B +1.48/+1.09/+0.35%, C
+3.07/+3.16/+3.19%, D +2.48/+1.95/+1.48%. 4B top-1: A .9916 B .9945 C .9735 D .9782.
**D/A: 1.6× at 0.6B → 3.5× at 4B, while B/A is flat at 0.65–0.74×** — the
weight-grid verdict is scale-stable; the activation penalty worsens with scale.

## Which grid carries the damage

**Not the weight grid — there int8 wins.** Mean per-Linear relative weight MSE
`||W−Ŵ||²/||W||²`: 0.6B fp8 **6.98e-04** vs int8 **8.89e-05** (7.9× lower); 4B
7.02e-04 vs 1.07e-04 (6.5× lower); int8 better in *every* role (ratio 0.111–0.167).
e4m3 spends 4 of 8 bits on an exponent the per-channel scale already removed.

**All the damage is the activation grid**: **+87%/+305%** conf-KL swapping
activations alone (C vs A, 0.6B/4B), **+120%/+437%** (D vs B). **Hypothesis HELD**.

## Where in the block, and why — *one group* of inputs to int8, rest fp8 (w1)

| arm | int8 activations on | conf-KL | Δ vs B | share of D's penalty |
|---|---|---|---|---|
| E | q,k,v,gate,up (5/7, RMSNorm-fed) | 0.01506 | +0.00678 | 57% |
| F | o_proj + down_proj (2/7) | 0.01215 | +0.00387 | 33% |
| G | **down_proj alone (1/7)** | 0.01327 | **+0.00499** | **42%** |
| D | all | 0.02011 | +0.01183 | 100% |

`G > F` is not noise — int8 on `o_proj` *reduces* KL vs `down_proj` alone:

| input to (act_stats.py) | amax/rms | chan-max/median | relMSE fp8 | relMSE int8 | int8/fp8 |
|---|---|---|---|---|---|
| down_proj (SwiGLU product) | 19.1 | **225.9** | 5.84e-04 | 2.22e-03 | **3.8×** |
| gate/up (post-attn norm) | 15.6 | 14.5 | 5.19e-04 | 1.37e-03 | 2.6× |
| q/k/v (input norm) | 13.8 | 14.7 | 5.61e-04 | 1.05e-03 | 1.9× |
| o_proj (attn output) | 9.1 | 6.9 | 6.71e-04 | 4.69e-04 | **0.7×** |

A per-token quantizer spends its scale on the token's largest magnitude, so a
uniform grid's resolution falls linearly in `amax/rms`; a log-spaced grid holds
constant *relative* precision. Purely an outlier effect, and concentrated.

## Verdict

**Naive int8 W8A8 does not hold — 1.5× the conf-KL of fp8 W8A8 at 0.6B and 3.5×
at 4B, PPL regressing in step.** Sharper than pass/fail, though:

1. The **weight** grid favours int8 decisively and scale-stably (6.5–7.9× lower
   weight MSE, 0.18–0.22× weight-only KL). Any int8-weight path beats fp8.
2. The **activation** grid is the whole problem, and arm **B is the ceiling**: at
   fp8-parity activation error int8 W8A8 lands at **0.65–0.74× fp8 W8A8's KL —
   better than the fp8 path, not merely equal.**
3. So the gate is not "int8 vs fp8" but "can activation outliers be handled", and
   it must hold at 27B+, where this trend says it is harder.

## Caveat: the weight-grid result does NOT transfer to codebooks

int8's 6.5–7.9× weight-MSE win is an **RTN** result and must not be read as
"an int8-grid codebook would beat an fp8-grid one by 7.9×". Under RTN every
weight snaps to a grid point, so the error *is* the grid spacing and the grid's
shape dominates. In a codebook the codewords are **fitted**: the error is
dominated by how well a codeword represents its cluster, and grid-snapping is a
second-order correction on top. That is why `rd_ceiling_study.md` measures the
fp8 grid constraint at only **+0.2–0.7%** weighted MSE for codebooks — three
orders of magnitude milder than the RTN gap. Both numbers are correct for their
own question. Any int8-grid CB proposal must be argued from the codebook figure,
not this one.

## Consequence for the Strix kernel plan

**int8 (a fortiori int4) WMMA on `gfx1151` is GATED behind an
activation-quantization solution.** W8A8-int8 as-is buys 1.56× prefill for
1.5–3.5× KL — not a trade that clears this repo's metric authority, and the
emulation caveat says the served penalty is likely worse. Do not build the int8
GEMM as a drop-in for fp8. Candidates, with only this screen's evidence:

- **Per-channel activation rescaling (SmoothQuant/AWQ-style migration)** — the
  only candidate with positive evidence. The damage is *channel-persistent*
  (chan-max/median 226 on `down_proj`, 14–15 elsewhere), exactly the structure
  such a rescale migrates into the weights, and the weight grid has ~7× headroom
  to absorb it. Every affected input is reachable with **no runtime op**: q/k/v
  fold into `input_layernorm`, gate/up into `post_attention_layernorm`,
  `down_proj` into `up_proj`'s rows (the SwiGLU product is elementwise), `o_proj`
  into `v_proj`'s. **Not measured — this locates the target, not the fix.**
- **Finer activation granularity** — *not* supported as the primary lever: both
  arms are already per-token, and the residual is intra-token channel spread that
  per-token scaling cannot see. Per-group-along-K fights the WMMA operand layout.
- **Activations in bf16** (W8A16) — the *quality* optimum (0.18–0.22× fp8 W8A8)
  but it **forfeits WMMA entirely**, the whole point. Decode fallback, not prefill.

**Gate order:** measure a smoothed int8-A arm against arm B on this harness, at
4B not 0.6B (0.6B understates the penalty ~2×). If it closes most of the
0.00681→0.03659 gap, int8 kernel work is justified and lands better-than-fp8
quality; if not, cancel int8 prefill rather than ship 3.5× KL.

**Artifacts** (`scratch/int8-w8a8-gate/`): `int8_grid.py`, `check_granularity.py`
(fairness proof), `run_gate.py` (`--arm-set main|localize`, `--low-mem`),
`act_stats.py`, and `gate_{0p6b_w1,0p6b_w2,0p6b_localize,4b_w1}.json` +
`act_stats_0p6b.json`, each with provenance (commit, dataset sha256, per-Linear
tables). Peak GPU 3.4/14.4 GB. No `prismaquant/` module modified.
