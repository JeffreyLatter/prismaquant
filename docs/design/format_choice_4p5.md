# Choosing between native NVFP4 and FP8-CB at a 4.5-bit budget

> **Historical design record.** The experiment plan and Stage-0 results remain
> useful, but `docs/lanes/nvfp4-cb/format-speed-policy.md` now owns the
> production decision. In particular, ignore this draft's 503/503 screen,
> blended `serve_ms`/independent-minimum floor, and any blanket decode-neutrality
> reading; the normative policy uses exact serialized bytes and separate hard
> TTFT/ITL/TPS constraints.

**Status: superseded policy draft; retained for experiment provenance.** Drafted 2026-07-30 against HEAD `601639d`.
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

**The 4.5 labels do not make the artifacts byte-identical.** NVFP4's 4.5
includes its group-scale plane; FP8_CB_K36's 4.5 is the index stream, plus
`32/in_features` scale bpw and a serialized product-codebook sidecar even for a
deterministic lattice reference. Match on one versioned whole-artifact byte
accountant and retain exact exported bytes as the final assertion. The
versioned `CBSerializationContext` serialized-payload API is authoritative;
exact exported inventory is the final whole-artifact assertion.

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

The original 503/503 Hy3 screen is retracted: its `h_trace` column was constant
and it fit unweighted codebooks before act-weighted scoring. The corrected
production-faithful Stage 0 favors K36 on **493/496** 27B units and **252/252**
4B units. This remains weight-error-only, excludes the W8A8-vs-W4A4 activation
effect, and can only stop the program; it is not served KL/PPL or a promotion.
See `format_choice_4p5_stage0_results.md`.

Measured at *other* budgets:

- **Served, matched bytes:** 27B @5.5 — conf-KL −45…−53%, ALL-KL −56/−58%, PPL gap
  to BF16 3× smaller (`prod_27b_results.md:41-55,124-134`); 35B MoE @4.75 — conf-KL
  −53%, ALL-KL −43%, PPL gap −30% (`prod_35b_results.md:26-40`).
- **Zero vanilla-NVFP4 selections are not speed evidence.** The 27B/35B/Hy3
  allocators optimized accuracy only, so preferring the more accurate format is
  circular with respect to a quality-versus-latency decision.
- **Speed at matched bpw:** decode **per-byte neutral at plain low-M decode**
  (measured twice — but never with a draft active; §4(d) scopes this claim: in
  the shipped spec-decode/batched regime the fp8:fp4 tensor-core ratio predicts
  ~2:1 against CB, unmeasured); prefill tax dense ~10%, MoE 0–40%, *regime*- not
  format-dependent (`format-speed-policy.md:16-29`). The "12% decode / 30% prefill" framing in
  `serving-tax-elimination.md:1-5` is **superseded** (pre-dates the CUDA expander
  and mid-M fused promotion) — cite the 07-27 policy / ARCHITECTURE §9.2.
- **CB's deployment cost:** an external Gridbook runtime with architecture and
  parallelism capabilities declared by its packaged consumer contract at the
  exact PrismaQuant pin (ARCHITECTURE §9.2). An unwired architecture used to
  serve uninitialised memory as coherent-looking garbage; the runtime now
  fails closed and cross-repository CI compares producer eligibility with that
  contract.

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
profile allow-lists `NVFP4`, `FP8_DYNAMIC`, `FP8_SOURCE`, `BF16` next to the
product CB
rungs (`prismaquant/serving_profile_specs/nvfp4_cb.json`), whose own description
calls it "**deliberately a MIXED container (PLAN.md decision #1, 'FP8 in every
recipe')**". Measured cost decides per Linear; under
`SELECTION_MODE=validated-surrogate`, real held-out KL decides the frontier point.

**So this document names no layers.** A promotion below 4.5 bpw is an
assignment-level byte-neutral bundle: lower CB rungs elsewhere must fund the
NVFP4 unit, and the bundle's net quality and phase-specific latency decide it.

### (b) The artifact-level rule, keyed on what the allocator cannot see

**K1 — Serving-environment constraint (hard, binary, decides alone).**
- Stock vLLM only / TP or multi-GPU required / architecture not CB-declared
  (producer `supported_lanes` versus the pinned Gridbook consumer contract) ⇒
  **native container, NVFP4 menu.** Not a quality judgment: CB cannot serve
  there.
- **Pre-Blackwell target** (sm < 100) ⇒ **CB container, fp8-CB ladder with no
  vanilla-NVFP4 rungs offered** — NVFP4 has no kernel there, gridbook's floor is
  8.0. A constraint, never a quality claim.

**K2 — Phase-specific deployment SLOs.** Declare p95 TTFT and p95 ITL / p05 TPS
limits over a representative workload. Plain M=1 decode and prefill are separate
constraints; neither substitutes for batched/speculative decode at shipped M.

**K3 — Timing tables propose, end-to-end evidence decides.** The `auto` tuner
logs candidate timings, but no `lambda` time objective is permitted. Validate a
candidate's same-session whole-serve ranks and served quality before selection.

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
| CB | `FP8_CB_K28..K48, NVFP4_CB_K12..K24, NVFP4, FP8_DYNAMIC, BF16` (signed S-rungs remain research/legacy compatible) |

A "pure NVFP4 at 4.5" artifact is not a candidate in either branch; a CB artifact
that allocates zero NVFP4 units is an *outcome of measurement*, not a container
property.

### (d) The systematic balance — constrained quality optimization

The normative form is the hard-constraint system in
`format-speed-policy.md`: exact serialized bytes; separate p95 TTFT and p95 ITL
/ p05 TPS SLOs; resident+KV+peak-scratch fit; and execution/shape/TP/coupling
legality. The objective is predicted quality loss only.

For a relative tax, compare against the fastest globally feasible assignment
under the same byte budget. The old sum of independently fastest unit formats
can be byte-infeasible and is retired. Per-layer timing additivity must first
predict end-to-end ranks; the final frontier is remeasured with served
KL/PPL/tasks and end-to-end streaming timing.

## 5. MEASUREMENT PLAN — validating 4.5 (acceptance fixed before each stage runs)

### Stage 0 — free, no GPU window

Run `scripts/ab_nvfp4_vs_k36_dense.py` against the **27B and 4B** work dirs (today
it is Hy3-only), reporting the `h_trace`-weighted column the CSV already carries.
**Acceptance: this stage can only *stop* the programme, never promote it** — if K36
loses the act-weighted majority on either model, re-plan before spending a serve
window; if it wins, it is a screen and nothing more.

**RUN 2026-07-30 — no STOP; results in `format_choice_4p5_stage0_results.md`.**
27B (496 units) and 4B (252 units) both hold the act-weighted majority: 99.4% and
100.0% under the production-faithful imatrix codebook fit (Σ h·mse ratios 0.405 /
0.472), 62.9% and 89.3% under the unweighted fit the Hy3 screen used. The stop
condition is not met; **nothing is promoted** and Stages 1–2 are unchanged. Two
caveats recorded there: the `h_trace` column of the shipped Hy3 CSV is all `1.0`
(tool bug, now fixed), and the Σ h·mse aggregate inverts in the fit/scoring
mismatched cells on both models — so a CB cost run must render with the imatrix
to be comparable at all.

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
(external Gridbook `gridbook/plugin.py`), which exists for exactly this.
`tools/kl_ab.py` refuses a delta across mismatched `serve_fingerprint`s (§7.4 R15);
unmatched arms yield a **range against the ±20% band**, not a delta.

### Stage 2 — the decider: the substitution ladder (revised 2026-07-30)

*Robert's correction: two independently-solved artifacts are "the same target
bit rate, yes, but not apples to apples" — different allocations put different
bits in different places, confounding format with allocation. The controlled
form: take the CB artifact and replace specific Linears with NVFP4, measuring
the impact of the substitution directly.*

**Endpoints first (Robert, same day: "or better yet, a pure nvfp4 build vs a
pure fp8 cb build").** Pure `FP8_CB_K36` and pure `NVFP4` isolate the complete
W8A8-versus-W4A4 execution contracts with zero allocation confound, but they are
formal same-rate endpoints only after exact whole-artifact accounting places
both under the same byte budget. Pure single-format menus are the sanctioned isolation pattern
(`[MEMORY: FP8 in every recipe]` — "2-rung menus only for cost-model A/B
isolation, never shippable"; neither endpoint ships). Neither endpoint needs
the allocator: a uniform assignment over the profile's quantizable set
(BF16-pinned and incomplete-fused exceptions as in any build); the existing 27B
work dir's act cache reuses for the imatrix harvest; pure-CB pays the known
~5.4 h `balanced` encode wall (`encode_tiers.md:31`), pure-NVFP4 is a fast
native-path export.

**Interior rungs (Robert's substitution design) only if the endpoint gap
justifies mapping the exchange curve.** Per-unit timing may rank proposals, but
each rung is a globally byte-feasible bundle: lower CB rungs must fund any
NVFP4 promotion and the final solve minimizes quality loss under the phase
SLOs. Substitute the top ~{25, 50, 75}% feasible bundles and report exact
whole-artifact bytes. Serve every rung — endpoints
included — through the SAME gridbook container (mixed-container delegation
routes stock-NVFP4 units onto the native CUTLASS path; verified in-container
before the ladder runs) against the same BF16 teacher, residency-matched.

**Readouts per rung:** ΔKL (conf + ALL) vs base · decode tok/s plain, with the
draft active, and at max-num-seqs {1,4} · prefill tok/s · bytes. The deliverable
is the **measured KL-vs-serving-time exchange curve within one allocation** —
apples to apples by construction, and the validation data the §4(d) DP's
solutions must track before that machinery is trusted.

**Variant cost:** with export-reuse (only flipped units re-encode), each variant
is a partial re-export + one serve window, not a rebuild. The 4.5-bpp question
inherits this protocol on a 4.5 CB base when a build window exists; the 5.5 base
answers the exchange-rate question first because it is already on disk.

*(The prior two-artifact design is retained below as the eventual cross-check —
an allocation-level comparison answers "which MENU at 4.5", the ladder answers
"what does each substitution buy" — but the ladder now leads.)*

### Stage 2-alt — the menu-level cross-check: two matched-byte 4.5 artifacts, 27B-class

- **Arm N** — native container, `FORMATS=NVFP4,FP8_DYNAMIC,BF16`, `COST_MODE=aura`
  (default), `TARGET_BITS=4.5`.
- **Arm C** — CB container (`EXPORT_CONTAINER=nvfp4_cb`), STANDARDS production menu,
  lane-default cost mode (`local`), `TARGET_BITS=4.5`.
- **Matched under the declared serialized-payload budget** using the versioned
  serialization context, then checked against the exporter's exact shipped-file
  inventory. Nominal bpw labels alone are not a match (§1). Report both payload
  and whole-directory totals so container
  metadata cannot disappear into rounding.
- **Model:** Qwen3.6-27B — the only model with both a served CB datum (5.5) and a
  shipped native twin on the same harness, so 4.5 is readable against 5.5. If dense
  and MoE answers diverge (expected: the prefill tax is regime-driven), the 35B MoE
  rerun is the **follow-up, not a substitute**.
- **Readouts per arm** (fresh `--enforce-eager` container, same BF16 dump): exact
  vLLM top-20 KL-vs-BF16 (conf + ALL) · WikiText PPL + mean NLL · ToolEvalBench
  (`--no-think --hardmode --parallel 1`) · TTFT(1400) and prefill tok/s ·
  **decode tok/s BOTH plain (M=1) and with the arm's draft active + at
  max-num-seqs ∈ {1, 4} — the decode-tax measurement the record lacks; draft
  configs matched in draft length across arms** · shipcard + `serve_manifest.json`.

| verdict | condition |
|---|---|
| **CB-FP8 becomes the 4.5 default** | arm C beats arm N on **both** conf-KL and ALL-KL by **more than the ±20% residency band**, **and** direct PPL does not regress (authority #2 can veto a KL win), **and** TEB is within the established ±2–3 single-seed churn band or better, **and** decode tok/s ≥ arm N − 5% **in the shipped regime (draft active, max-num-seqs 4) — not only plain M=1**; a CB arm that wins KL but decodes near Robert's predicted 2:1 in the shipped regime routes to the Mixed verdict, not to CB-default |
| **NVFP4 becomes the 4.5 default** | arm C's KL win **fails to clear the ±20% band** (a null — then the prefill tax and the plugin/TP/per-arch cost buy nothing), **or** arm C regresses PPL or TEB at any KL |
| **Mixed** | arm C wins KL outside the band **but** its prefill deficit on this model class exceeds ~20%; re-solve arm C on the joint menu and check whether the allocator now spends NVFP4 units. All three prior joint solves chose **zero** vanilla NVFP4 — a mixed outcome at 4.5 would itself be the new information |

Any arm lacking provenance (git commit, calibration hash, assignment hash, cache
hit/miss counts, `serve_fingerprint`) is quarantined, not compared (§7.4).

### Stage 3 — the timing table (runs with Stage 2's window, not only on ambiguity)

Dump the `auto` tuner's per-layer measured ms from both arms into the per-(format,
shape-regime) table — this is the table §4(d)'s prefill-tax budget consumes, so it
is harvested in the same serve window regardless of Stage 2's verdict, plus the
additivity check (Σ per-layer ms vs measured end-to-end prefill on both arms).
**Not λ** — R21's ruling stands; §4(d) is a constraint, not an objective term.

### Box constraints, stated honestly

- **One Spark.** A 27B arm needs exclusive access to the unified-memory pool
  (artifact + serve at util 0.80–0.85 + BF16 dump). No existing service is
  required to remain online; get box clearance before a production run.
- **Disk:** 224 GB free of 1.8 TB = 12.4% — above the ≥10% floor, but two 27B caches
  (~90 GB each) do not fit concurrently: build → measure → delete → build, `df -h
  /home/rob` before each launch.
- **Serve discipline:** `--host 0.0.0.0 --port 8000` + `-p 8000:8000`, slack gate and
  watchdog, util 0.80–0.85, never 0.94/0.95.

## 6. Explicit non-goals

1. **No λ in the objective.** Timing enters only through separate hard
   TTFT/ITL/TPS constraints; Stage 3 may use the R21 table to propose candidates.
2. **No AURA-on-CB conclusions.** The CB lane can reach `COST_OBJECTIVE=aura-adjoint`
   since Milestone C, but it is opt-in, non-default, and "the A/B that would justify
   a CB default has not been run" (ARCHITECTURE §4.7). Arm C uses the lane default;
   comparing cost objectives is a different experiment.
3. **No Strix Halo numbers** — nothing measured there. **No NVFP4-CB claims at 4.5**
   — the fp4 ladder tops out at 3.28125 bpw. **No GGUF comparison** — separate lane,
   separate serving stack.
4. **No heuristic format bans or bpw carve-outs.** Capability/evidence policy is
   still binding: packed experts cannot select unimplemented native delegation,
   and signed S-rungs remain research-only.
