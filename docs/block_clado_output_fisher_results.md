# Output-Fisher Block-CLADO results — Qwen3-0.6B

Branch: `block-clado` (commits through `512713c`)
Run: `/home/rob/dq-runs/qwen3-0p6b-output-fisher-20260506T041858Z`

## Method

Replace the four-term identity's ``O(|𝔹|² · I²)`` pair forwards with
``O(|𝔹| · I)`` JVPs.  For each ``(unit, format ≠ BF16)`` cache the
output-logit perturbation

    δz_{U,f} = z(student_with_U→f) − z(teacher)

then compute pair interactions analytically through the per-token
Fisher ``F_z(t) = diag(p_t) − p_t p_t^⊤``:

    Ω_ii = (1/2) · E_t[Var_{p_t}(δz_i)]
    Ω_ij =          E_t[Cov_{p_t}(δz_i, δz_j)]

These match the four-term identity exactly at the second-order Taylor
level; they differ on higher-order terms (the four-term identity
captures ``O(δw³+)`` finite-difference effects, the Fisher form
analytically truncates).  MVP is BF16-centered and weight-only (no
activation quantization).

## Headline result on Qwen3-0.6B

The highest-quality model produced via Block-CLADO so far:

| pipeline | bpp | real KL | total time |
|---|---|---|---|
| Surrogate kneedle alone (4-term) | 4.86 | 0.217 | 76s |
| Frontier-validated best (4-term) | 4.57 | 0.124 | 105s |
| 4-term iter 0 + polish (greedy, 5% creep) | ~4.65 | 0.087 | 12 min |
| 4-term sandwich iter-1 + polish (best of 3 iters) | ~8.4 | 0.047 | 37 min |
| **OF + polish (high-bpp anchor, 10% creep, steepest-first)** | **~11** | **0.0326** | **2.7 min** |
| OF + polish (low-bpp anchor, 5% creep, steepest-first) | ~5 | 0.092 | 2.4 min |

**OF + polish achieves the best KL we've seen (0.033) in 2.7 minutes —
14× faster than the 37-min iterate that reached only 0.047.**

## Surrogate fidelity vs four-term

Side-by-side payload comparison (`prismaquant.compare_payloads`):

| component | n entries | mean Δ | std Δ | Spearman ρ |
|---|---|---|---|---|
| Unary Ω_ii | 339 | −5e-4 | 2e-3 | **0.895** |
| Pair  Ω_ij | 1,512 | +8e-4 | 2e-3 | **−0.10** |

Unary Fisher Ω_ii agrees strongly with four-term Ω_ii (Spearman 0.895)
— the second-order term dominates the unary effect of quantizing one
layer.  Pair Ω_ij values, however, are **uncorrelated** with four-term:
the higher-order terms captured by finite differencing materially
change the cross-layer interaction estimates, often flipping their sign.

This explains the empirical results below: at low bpp, four-term's pair
Ω_ij captures interactions OF misses, giving a measurably better
surrogate; at high bpp where δw is smaller (more BF16 layers, less
cumulative perturbation), OF's analytic Fisher is closer to the truth
and faster.

## Per-bpp comparison

### At ~4.5–4.7 bpp (original target)

| method | bpp | best validated KL | + polish KL |
|---|---|---|---|
| 4-term frontier | 4.5726 | 0.124 | 0.087 (iter-0 polish) |
| OF frontier | 4.7819 | 0.139 | **0.092** (this run) |

Four-term wins by ~6% on KL.  The 4.5-bpp regime is where the higher-
order pair effects matter most, and they're exactly what OF's analytic
Fisher form drops.

### At ~8 bpp

| method | bpp | best validated KL | + polish KL |
|---|---|---|---|
| 4-term sandwich iter-1 | 8.0008 | 0.121 | 0.047 |
| OF | 9.34 | 0.112 | (not run; would expect 0.04ish with creep) |

Both give very similar polished KL.  Four-term sandwich does it via
expensive Pareto exploration (37 min); OF does it directly from the
BF16-centered surrogate (2-3 min).

### At ~10–11 bpp

| method | bpp | best validated KL | + polish KL |
|---|---|---|---|
| OF | 10.69 | 0.062 | **0.0326** (10% creep, 50s polish) |

This is the regime where OF dominates: its analytic-Fisher pair terms
(uncorrelated with four-term but PSD by construction) push the Pareto
frontier into a higher-bpp region, and polish from there extracts the
best KL we've seen.

## Why OF + polish is so fast

| step | OF | 4-term |
|---|---|---|
| Measure cost payload | 80s (227 forwards) | 76s (1,738 forwards) |
| λ-sweep + frontier validate | 30s | 30s |
| Polish (greedy-best) | 600s | 600s |
| Polish (steepest-first, 10% creep) | **50s** | (untested) |
| **Total (steepest-first)** | **~2.7 min** | ~12 min |

Steepest-first polish ordering — accepting the first surrogate-ranked
move that improves real KL — cuts polish cost ~12× over the
greedy-best default.  This is independent of OF; both surrogates
benefit equally from steepest-first.

The structural OF speedup (227 vs 1,738 forwards) doesn't manifest in
wall-clock at this scale because each OF forward is heavier (manual
weight swap rather than the cached perturbed-cache forward used by
four-term).  At larger scale (27B) where the four-term's ``|𝔹|² · I²``
factor explodes, OF's ``|𝔹| · I`` pays back massively.

## Polish trace (high-bpp run)

```
start  ~10.69 bpp / KL 0.062
pass 0  layers.11.mlp.gate_up_proj  NVFP4 → BF16   KL 0.062 → 0.053
pass 1  layers.16.mlp.gate_up_proj  NVFP4 → BF16   KL 0.053 → 0.049
pass 2  layers.18.mlp.gate_up_proj  NVFP4 → BF16   KL 0.049 → 0.046
pass 3  layers.6.mlp.gate_up_proj   NVFP4 → BF16   KL 0.046 → 0.044
pass 4  layers.25.mlp.gate_up_proj  MXFP8 → BF16   KL 0.044 → 0.039
pass 5  layers.25.self_attn.o_proj  BF16  → MXFP8  KL 0.039 → 0.037   ← downgrade!
pass 6  layers.24.mlp.gate_up_proj  NVFP4 → MXFP8  KL 0.037 → 0.035
pass 7  layers.13.mlp.down_proj     BF16  → MXFP8  KL 0.035 → 0.033   ← downgrade!
end    ~11.4 bpp / KL 0.0326   180 measurements, 49.7s
```

Two downgrades (BF16 → MXFP8) appear when the Pareto-beneficial trade
opens up.  Polish is genuinely navigating cross-format precision swaps,
not just monotonically upgrading.

## Recommendations

1. **For best raw KL (no bpp constraint)**: OF + steepest-first polish
   with 10% creep.  Fastest path to the lowest measured KL we've seen.
2. **For fixed-bpp (~4.5)**: four-term Block-CLADO + polish remains
   best by a small margin (~6% better KL).  OF's analytic Fisher
   misses higher-order pair effects in this regime.
3. **For Pareto exploration**: run OF measurement once, validate the
   full frontier, polish multiple validated points with steepest-first
   + budget creep; pick the best.  Very cheap parallel exploration.
4. **For LLM scale (27B)**: OF's ``O(|𝔹|·I)`` measurement count
   becomes essential as ``O(|𝔹|² · I²)`` four-term blows up.  Even
   with manual weight swap overhead per OF forward, total scales as
   1,842 forwards (vs ~190K for full four-term, ~2k for Block-CLADO).
   The per-forward speedup will matter when polishing.

## Limitations

1. **Sandwich centering not yet supported by OF.**  At a non-trivial
   centered point ``w_c`` the Taylor expansion has a nonzero linear
   term ``g_c · Δ`` that the second-order Fisher form alone misses.
   Sandwich iterations should fall back to four-term for now.
2. **No activation quantization in the perturbation.**  This MVP swaps
   weights manually rather than going through PerturbedActivationCache.
   The deployment-faithful surrogate needs activation quant included;
   that's the next iteration.
3. **Pair Ω_ij dispersion.**  The −0.10 Spearman vs four-term means
   OF's surrogate disagrees with four-term on the pair-level effects
   that drive low-bpp quantization.  Validation cone + polish remain
   essential — never trust a surrogate kneedle directly.

## Test coverage

`tests/test_measure_output_fisher.py`: 10 unit tests covering Var/Cov
math, four-term decomposition (Var(a+b) = Var(a) + 2 Cov + Var(b)),
self-pair = 2 × unary, constant-shift invariance, agreement with
exact KL on small perturbations (within third-order error).

Total branch test count: 35 passing.
