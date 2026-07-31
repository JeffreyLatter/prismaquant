# Choosing between native NVFP4 and FP8-CB at a 4.5-bit budget

**Status: proposed criteria + measurement plan. Changes no default, bans no
format, implements nothing.** Drafted 2026-07-30 against HEAD `601639d`.
Every claim is cited to code/measurement, or labelled `[EXTRAPOLATION]` /
`[MEMORY]` (auto-memory, not re-verified this session).

## 1. The question, and why 4.5 specifically

| | rate | compute | serving |
|---|---|---|---|
| `NVFP4` (vanilla) | 4 w-bits + 8-bit E4M3 per 16 = **4.5 bpp exactly** (`format_registry.py:668-677`) | W4A4, per-group-16 dynamic A4 | stock vLLM, CUTLASS block-scaled, **zero maintained serving code** |
| `FP8_CB_K36` | 36-bit index / 8-weight vector = **4.5 bpw** index stream (`format_registry.py:954-983`) | product-VQ codebook on the **E4M3** grid; A8 per-token | out-of-tree `gridbook` plugin (ARCHITECTURE §9.2) |

**4.5 is the only budget where both families have an exact rung.** FP8_CB is
registered at *every* integer k from 28 to 48 (`format_registry.py:984`) = 3.5–6.0
bpw in 0.125 steps, so K36's neighbours are **K35 = 4.375** and **K37 = 4.625** —
K36 is the exact-4.5 rung, no interpolation needed on the CB side. **Verified: at
4.5 the CB side is FP8-CB, not NVFP4-CB** — the fp4-grid ladder is K12–K24 =
2.0–3.5 bpw (v1) and **1.78–3.28125 bpw under the shipped two-tier v2**
(`format_registry.py:980`; `two-tier-scale-spec.md` §2.1), so it cannot reach 4.5.

**Byte asymmetry, so a "matched 4.5" A/B is honest.** NVFP4's 4.5 *includes* its
scale plane; FP8_CB_K36's 4.5 is index stream only, plus per-output-channel fp32
scales at `32/in_features` bpw (`nvfp4_cb_footprint.py:18-19,43`; +0.00625 at
`in=5120`) and a codebook sidecar if codebooks are learned rather than lattice
(`:41`; zero for the lattice codebooks the 27B/35B runs shipped). Match on
**`footprint`/`nvfp4_cb_footprint` bytes, not labels** — as the 27B A/B did
(16.713 vs 16.707 GB body, Δ 0.04%, `prod_27b_results.md:32-37`).

### The gap, named first — **nothing has ever been served at 4.5 on the CB side**

| artifact | bpp | what it establishes |
|---|---|---|
| Qwen3.6-27B | **5.5** | matched-byte quality A/B vs native AURA (`prod_27b_results.md`) |
| Ornith-1.0-35B MoE | **4.75** | matched-byte quality A/B vs native AURA (`prod_35b_results.md`) |
| Hy3-295B | 2.9 | serving/perf/TEB only — "**No quality claims** — a 295B cannot be KL-validated on this box" (ARCHITECTURE §9.2) |
| Laguna-S-2.1 | 6.0 | prefill kernel campaign only |

Both matched-byte *quality* decisions are at **≥ 4.75**. 4.5 sits below every
served CB quality datum, and is simultaneously the lowest rung of the native
lane's default Pareto sweep (`PARETO_TARGETS=4.5,…`, ARCHITECTURE §3.3). That is
why the question is open rather than settled by the shipped evidence.

## 2. Measured at 4.5 vs extrapolated to 4.5

### 2.1 Measured at exactly 4.5

**The 503/503 sweep is a genuine K36-vs-native-4.5 pairing** — not an
interpolation between rungs; that was the first thing checked.
`scripts/ab_nvfp4_vs_k36_dense.py` (commit `c551e24`) prices both formats through
one code path at their exact 4.5 rungs:

> FP8CB_K36 beats vanilla NVFP4 on **503/503 units unweighted (geomean −40%
> error), 87% act-weighted. 42 outlier-row units favor NVFP4**
> — `format-speed-policy.md:9-14`

Three structural limits, stated before it is leaned on:
(1) **Weight-error only** — `cost()` quantizes `W` and nothing else
(`ab_nvfp4_vs_k36_dense.py:91-103`), so the A8-vs-A4 activation difference (the
mechanism the policy itself names as part of CB's edge) **is not inside the −40%**;
it is an additional, separately-unmeasured advantage at 4.5.
(2) **One model, one tier** — Hy3-295B dense/attention/shared targets, experts and
`layers.80` excluded (`:70-73`); no 27B, no 4B, no MoE expert unit at 4.5.
(3) **Cost-model, not KL, not served** — a screen under the metric authority
(ARCHITECTURE §2.3); the commit says so itself.

Measured at *other* budgets:

- **Served, matched bytes:** 27B @5.5 — conf-KL −45…−53%, ALL-KL −56/−58%, PPL gap
  to BF16 3× smaller (`prod_27b_results.md:41-55,124-134`); 35B MoE @4.75 — conf-KL
  −53%, ALL-KL −43%, PPL gap −30% (`prod_35b_results.md:26-40`).
- **Revealed allocator preference — three joint solves, zero vanilla NVFP4
  chosen:** 27B @5.5, 386/386 body Linears on CB (`prod_27b_results.md:13-24`);
  35B @4.75, all-CB body (`prod_35b_results.md:20-22`); Hy3 @2.9 on an 11-format
  joint menu → "36 dense/shared Linears → vanilla FP8_DYNAMIC; **0 → vanilla
  NVFP4** (offered and never chosen)" (`prod_hy3_results.md:277-282`). **None at
  4.5.**
- **Speed at matched bpw:** decode **per-byte neutral** (measured twice); the
  entire tax is prefill — dense ~10%, MoE 0–40%, *regime*- not format-dependent
  (`format-speed-policy.md:16-29`). The "12% decode / 30% prefill" framing in
  `serving-tax-elimination.md:1-5` is **superseded** (pre-dates the CUDA expander
  and mid-M fused promotion) — cite the 07-27 policy / ARCHITECTURE §9.2.
- **CB's deployment cost:** out-of-tree plugin, per-arch top-level expert-loader
  wiring (`_CB_TOPLEVEL_MODULE_PATHS`), **single-GPU only, no TP handling in
  `gb/*.py`** (ARCHITECTURE §9.2); an unwired arch used to serve uninitialised
  memory as coherent-looking garbage, now a hard serve-time failure
  `[MEMORY: gridbook new-MoE-arch checklist]`.

### 2.2 EXTRAPOLATION to 4.5

- `[EXTRAPOLATION]` that the weight-MSE edge maps to **served KL** at 4.5. It did
  at 5.5 and 4.75 on two models; different rung, different mix, unmeasured.
- `[EXTRAPOLATION]` that it holds when the budget forces the **low** rungs. Every
  served CB solve sat high on the ladder (27B: K36–K48; 35B: K28–K48 at 4.75); a
  4.5 body average pushes mass toward K28–K36, where each 8-dim vector is coded by
  four 9-bit sub-tables (`nvfp4_cb_formats.py:161-172`) — fewer shapes per vector
  than anything yet shipped as a body average.
- `[EXTRAPOLATION]` that the prefill tax at 4.5 equals the tax at 5.5/6.0. Prefill
  is expand-bound and expand traffic scales with k, so 4.5 is *plausibly* cheaper.
  Plausible ≠ measured.
- **Not extrapolable at all:** NVFP4-CB at 4.5 (ladder tops at 3.28125) and any
  295B-class quality verdict (unmeasurable on this box).

## 3. Mechanism — what each side should win, and why

**fp8-CB should win quality at 4.5.**

1. **Storage rate and compute precision are independent dials** (ARCHITECTURE
   §9.2, verbatim). `FP8_CB_K36` stores 4.5 bpw and computes in fp8 with per-token
   A8 (`format_registry.py:963-966`); `NVFP4` computes W4A4 with per-group-16 A4
   (`:670-676`). At matched storage the activation path is the one axis where the
   two differ **for free** — activations are not stored, so A8 costs zero of the 4.5.
2. **Fitted codebook vs fixed grid.** NVFP4 snaps every weight to 16 E2M1 levels;
   its only fitted parameter is the per-16 scale. FP8-CB fits an imatrix-weighted
   product-VQ codebook per Linear, and the grid constraint itself is nearly free on
   the fp8 grid: **+0.2–0.7%** weighted MSE vs unconstrained across all four sources
   (`rd_ceiling_study.md` Part 1, column C). Almost all of the fitting advantage
   survives being forced onto E4M3.
3. **Menu granularity at the knee.** The CB ladder has a rung every 0.125 bpw from
   3.5 to 6.0; the native menu around 4.5 is `{4.5, 8.0, 16}` — a 3.5-bpw hole
   above 4.5. `[MEMORY: AURA RD frontier convex]` a sharp knee needs a *menu rung*.

**Native should win prefill and deployment.**

1. **No expand.** CUTLASS consumes NVFP4's stored bytes directly; CB prefill runs
   `cb_expand_fp8` → stock `cutlass_scaled_mm` (STANDARDS kernel table), and the
   transient's write-then-read traffic is explicitly the residual gap: "the
   remaining 0.33 s vs AURA is the transient's write+read traffic"
   (`prod_27b_results.md:116-118`).
2. `[EXTRAPOLATION — architectural, not isolated on this box]` at large M, W4A4
   tensor-core throughput is nominally above W8A8 on Blackwell, so the A8 that buys
   quality also costs prefill FLOPs. The measured tax bundles this with the expand.
3. **Deployment surface.** Native = stock vLLM, no plugin, no per-arch wiring, TP
   available. CB = out-of-tree package, per-arch registry, single-GPU, advisory
   lane gates (`prismaquant/lane_specs/nvfp4_cb.json`).
4. **Capability points both ways.** `NVFP4` needs `min_capability_sm=100`; gridbook's
   JIT floor is **capability 8.0** (STANDARDS), and "artifacts that happen to
   allocate zero vanilla-NVFP4 units remain Ada-servable as a bonus, never a
   constraint" (`STANDARDS.md:38-40`).

## 4. PROPOSED CRITERIA

### (a) Quality at 4.5 is per-Linear, and belongs to the allocator

Inside the CB container both families are already in **one menu**: the serving
profile allow-lists `NVFP4`, `FP8_DYNAMIC`, `FP8_SOURCE`, `BF16` next to 38 CB
rungs (`prismaquant/serving_profile_specs/nvfp4_cb.json`), whose own description
calls it "**deliberately a MIXED container (PLAN.md decision #1, 'FP8 in every
recipe')**". Measured cost decides per Linear; under
`SELECTION_MODE=validated-surrogate`, real held-out KL decides the frontier point.

**So this document names no layers.** "Attention `o_proj` gets NVFP4 at 4.5" is a
hardcoded format ban with better manners — vetoed by core principle 1 and by the
standing policy (*"No format bans, no bpw-band carve-outs, no 'MoE always native'
rules"*, `format-speed-policy.md:66-72`). The 42 outlier-row units favouring NVFP4
are **the argument for the joint menu, not for a hand-picked list**: the allocator
picks NVFP4 there on accuracy alone (`:12-14`). The criteria's job is to say
**which menu to offer**.

### (b) The artifact-level rule, keyed on what the allocator cannot see

**K1 — Serving-environment constraint (hard, binary, decides alone).**
- Stock vLLM only / TP or multi-GPU required / architecture not CB-declared
  (`supported_lanes`, `_CB_TOPLEVEL_MODULE_PATHS`) ⇒ **native container, NVFP4
  menu.** Not a quality judgment: CB cannot serve there.
- **Pre-Blackwell target** (sm < 100) ⇒ **CB container, fp8-CB ladder with no
  vanilla-NVFP4 rungs offered** — NVFP4 has no kernel there, gridbook's floor is
  8.0. A constraint, never a quality claim.

**K2 — Prefill weight of the deployment** (reached only when K1 leaves both open).
- **Decode-dominated** (chat, agents, long generations): decode is per-byte neutral
  at matched bytes, tax ≈ 0 ⇒ **CB menu**, accuracy-first, status quo.
- **Prefill-heavy** (RAG, long-document, batch scoring): tax is real and
  regime-dependent (dense ~10%, MoE 0–40%) ⇒ **native menu, or CB with the
  deployment declared on the card** — today's honest mitigations are card routing
  guidance plus the joint menu (`format-speed-policy.md:36-47`).
- K2 is an **operator declaration about the deployment**, expressed as *which menu
  is offered* — the same class of input as `TARGET_DISK_GB`, not a heuristic inside
  the objective, and it must never become one.

**K3 — The R21 sink graduates K2 from judgment to a table.** The λ term
(`quality + λ·serving_ms`, λ=0 default) is **specified, not implemented** — no λ in
`allocator_solver.py` (ARCHITECTURE §9.2) — and R21 defers it until two boxes'
tables disagree in ranking. The raw material already accrues: the `auto` tuner logs
measured ms per candidate per layer, "a free per-format, per-shape serving-cost
table accumulating in every serve" (`format-speed-policy.md:26-29`). Until that
table exists and disagrees, K2 stays an operator key.

**K4 — tie-breaker, a cost not a quality signal.** CB costs encode wall (uniform-FP8
27B ≈ 5.4 h at `balanced`, `encode_tiers.md:31,66-75`) and its lane gates are
advisory. On a genuine K1–K3 tie, prefer the cheaper lane to re-gate.

### (c) Tie to "FP8 in every recipe"

`[MEMORY: FP8 in EVERY recipe — "that's the whole point of pq"; 2-rung menus only
for cost-model A/B isolation, never shippable.]` Consequence: **the 4.5 question is
never "NVFP4 or CB" as a uniform format choice** — both answers are *menus*, both
containing FP8:

| decision | menu offered |
|---|---|
| native | `NVFP4, FP8_DYNAMIC, BF16` (+ `FP8_SOURCE` where the source is fp8) |
| CB | `FP8_CB_K28..K48, NVFP4_CB_K12..K24/S13..S16, NVFP4, FP8_DYNAMIC, BF16` (STANDARDS "standard production menu") |

A "pure NVFP4 at 4.5" artifact is not a candidate in either branch; a CB artifact
that allocates zero NVFP4 units is an *outcome of measurement*, not a container
property.

## 5. MEASUREMENT PLAN — validating 4.5 (acceptance fixed before each stage runs)

### Stage 0 — free, no GPU window

Run `scripts/ab_nvfp4_vs_k36_dense.py` against the **27B and 4B** work dirs (today
it is Hy3-only), reporting the `h_trace`-weighted column the CSV already carries.
**Acceptance: this stage can only *stop* the programme, never promote it** — if K36
loses the act-weighted majority on either model, re-plan before spending a serve
window; if it wins, it is a screen and nothing more.

### Stage 1 — small-scale-first screens (0.6B + 4B)

Two matched-byte 4.5 allocations per model, native menu vs CB menu,
`--calib-repeats ≥ 4`, held-out split disjoint from cost generation
(`VALIDATED_FRONTIER_SKIP_CALIB=$NSAMPLES`, on by default).

**The emulation asymmetry sets the acceptance rule.**
`[MEMORY: aura-4b-render-discrepancy]` the resident HF W4A4 emulation
**under-counts native NVFP4's served penalty ~3× at 4B**, so an emulated screen is
**biased in favour of native**: emu showing **CB winning** is conservative evidence
(the served margin should be larger) and proceeds to Stage 2, while emu showing
**NVFP4 winning or a tie** is **uninformative** and must be re-measured served at 4B
before it influences anything.

**4B served arms** then decide whether Stage 2 is worth the box. Residency matching
is mandatory (§7.4: ±17% conf-KL keyed purely on extension residency) and has a
mechanism — the native arm sets `PRISMAQUANT_PRELOAD_FUSED=1`
(`plugins/gridbook/gridbook/plugin.py:134`), which exists for exactly this.
`tools/kl_ab.py` refuses a delta across mismatched `serve_fingerprint`s (§7.4 R15);
unmatched arms yield a **range against the ±20% band**, not a delta.

### Stage 2 — the decider: two matched-byte 4.5 artifacts, 27B-class

- **Arm N** — native container, `FORMATS=NVFP4,FP8_DYNAMIC,BF16`, `COST_MODE=aura`
  (default), `TARGET_BITS=4.5`.
- **Arm C** — CB container (`EXPORT_CONTAINER=nvfp4_cb`), STANDARDS production menu,
  lane-default cost mode (`local`), `TARGET_BITS=4.5`.
- **Matched on measured body bytes ≤0.1%** via `footprint`/`nvfp4_cb_footprint`
  (27B precedent: 0.04%). Nominal bpw labels are not a match (§1).
- **Model:** Qwen3.6-27B — the only model with both a served CB datum (5.5) and a
  shipped native twin on the same harness, so 4.5 is readable against 5.5. If dense
  and MoE answers diverge (expected: the prefill tax is regime-driven), the 35B MoE
  rerun is the **follow-up, not a substitute**.
- **Readouts per arm** (fresh `--enforce-eager` container, same BF16 dump): exact
  vLLM top-20 KL-vs-BF16 (conf + ALL) · WikiText PPL + mean NLL · ToolEvalBench
  (`--no-think --hardmode --parallel 1`) · TTFT(1400) and prefill tok/s · decode
  tok/s · shipcard + `serve_manifest.json`.

| verdict | condition |
|---|---|
| **CB-FP8 becomes the 4.5 default** | arm C beats arm N on **both** conf-KL and ALL-KL by **more than the ±20% residency band**, **and** direct PPL does not regress (authority #2 can veto a KL win), **and** TEB is within the established ±2–3 single-seed churn band or better, **and** decode tok/s ≥ arm N − 5% |
| **NVFP4 becomes the 4.5 default** | arm C's KL win **fails to clear the ±20% band** (a null — then the prefill tax and the plugin/TP/per-arch cost buy nothing), **or** arm C regresses PPL or TEB at any KL |
| **Mixed** | arm C wins KL outside the band **but** its prefill deficit on this model class exceeds ~20%; re-solve arm C on the joint menu and check whether the allocator now spends NVFP4 units. All three prior joint solves chose **zero** vanilla NVFP4 — a mixed outcome at 4.5 would itself be the new information |

Any arm lacking provenance (git commit, calibration hash, assignment hash, cache
hit/miss counts, `serve_fingerprint`) is quarantined, not compared (§7.4).

### Stage 3 — only if Stage 2 is ambiguous

Dump the `auto` tuner's per-layer measured ms from both arms into the per-(format,
shape-regime) table the λ term *would* consume — **not to implement λ**, but to size
whether such a table reorders anything, making R21's own trigger ("two boxes' tables
disagree in ranking") testable rather than aspirational.

### Box constraints, stated honestly

- **One Spark.** 121 GiB unified; at drafting time 69 used / 51 available with the
  **idle serve on :8000 holding roughly half the pool**. A 27B arm needs the pool
  (artifact + serve at util 0.80–0.85 + BF16 dump): stop the idle serve and get
  **Robert's box clearance first**
  `[MEMORY: no over-parallel during production — a production run gets the box]`.
- **Disk:** 224 GB free of 1.8 TB = 12.4% — above the ≥10% floor, but two 27B caches
  (~90 GB each) do not fit concurrently: build → measure → delete → build, `df -h
  /home/rob` before each launch.
- **Serve discipline:** `--host 0.0.0.0 --port 8000` + `-p 8000:8000`, slack gate and
  watchdog, util 0.80–0.85, never 0.94/0.95.

## 6. Explicit non-goals

1. **No λ in the objective.** The serving-cost term stays specified-not-implemented;
   ruling **R21** stands. Stage 3 only *reads* the timing data R21 named.
2. **No AURA-on-CB conclusions.** The CB lane can reach `COST_OBJECTIVE=aura-adjoint`
   since Milestone C, but it is opt-in, non-default, and "the A/B that would justify
   a CB default has not been run" (ARCHITECTURE §4.7). Arm C uses the lane default;
   comparing cost objectives is a different experiment.
3. **No Strix Halo numbers** — nothing measured there. **No NVFP4-CB claims at 4.5**
   — the fp4 ladder tops out at 3.28125 bpw. **No GGUF comparison** — separate lane,
   separate serving stack.
4. **No format bans, no bpw-band carve-outs**, including no "below X bpp use native"
   rule. If the allocator picks something that serves badly, the measurement is
   wrong, not the optimizer.
