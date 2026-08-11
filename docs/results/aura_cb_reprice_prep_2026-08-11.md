# AURA-on-CB re-price preparation — 2026-08-11

Status: **PREPARATION ONLY; NO GPU WORK RUN.** Robert Tand authorized “Aura on
CB for moe and dense” on 2026-08-11. This note records the CPU-only inventory,
the source-rate-derived measurement domain, the launch contract, and the
fail-closed blockers found on branch `perf/ldlq-atom-compile`.

The prepared entrypoints are:

- `tools/aura_cb_reprice_preflight.py` — CUDA-hidden inventory and capability
  audit.
- `tools/run_aura_cb_reprice.sh` — fail-closed cost-only launch specification.
  It is **not executable on the current branch** because the required worker
  and resume interfaces do not exist. A future successful launch
  re-runs preflight, opens `/home/rob/dq-runs/gpu.lock`, executes `flock -x 9`,
  re-runs preflight while holding the lock, and only then starts a GPU
  container. It never makes an accelerator-idleness query.

## Outcome

The intended generic mapping is present:

```text
COST_MODE=aura
  -> COST_RENDER=cached-menu
  -> COST_OBJECTIVE=aura-adjoint
  -> aura_cost --require-production-cache
```

For a packed MoE, the generic pipeline intends to omit expert tensors from
smooth AURA and merge empirical serving-unit KL. For a dense model, the omitted
set is empty, so the result is straight AURA.

However, **the current branch cannot safely launch the requested DSv4 run, and
there is no actual Qwen3.8-27B checkpoint/profile to launch.** Consequently the
requested ready-to-run/per-unit-resumable deliverable is not yet achievable
from this checkout. The prepared launcher records the exact command contract
but fails before acquiring the GPU or creating a container. This is a
correctness refusal, not a scheduling delay.

## DSv4 inputs that exist

The newest complete calibration capture is `prod-cal-0p7`, not the smaller
`prod-cal-0p6-v2` cost-extension tree.

| Input | Exact state | Reusable for this campaign? |
|---|---|---|
| Source | `/home/rob/dq-runs/dsv4-flash-0731/source`; revision `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`; 48 model shards; 166,886,535,336 file bytes; index payload 166,878,536,440 bytes / 72,317 tensors | Yes, through the existing streaming source loader. The expanded non-scale tensors are 608,360,836,432 bytes, so a resident BF16 load is forbidden on 128 GB UMA. |
| Source index | `source/model.safetensors.index.json`; 5,602,871 bytes; SHA-256 `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b` | Yes |
| Source config | `source/config.json`; 1,888 bytes; SHA-256 `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023` | Yes |
| Corpus | `/home/rob/dq-runs/calibration/diverse-v1.jsonl`; 4,597,898 bytes; SHA-256 `e09a138a4903c4af66a3bf2f9367185f3432224391f1dfe8c94ccc29d99315ba` | Yes; retain the 16×512 / 8,192-token comparison contract where applicable |
| Sensitivity probe | `prod-cal-0p7/artifacts/probe.pkl`; 7,383,774 bytes; SHA-256 `a0fdbb62c075fdc2d3fa3518e22ef87226aa4f619989d2e8162a2e3f9eda0535`; 33,325 rows; 16×512; streaming-layerwise | Partly. It is CE empirical-Fisher (`h_trace`/`h_w2`), **not** a KL-Fisher adjoint. It supplies inventory/routing metadata but cannot be passed to `aura_cost` (which has no `--probe`). |
| Activation cache | `prod-cal-0p7/act`; 33,274 `.pt` files; 25,292,991,482 bytes | Yes for a future streamed render. The 51 absent projections are 17 never-routed experts × three projections. |
| CB imatrix | `prod-cal-0p7/artifacts/cb_col_weights.pkl`; 466,388,371 bytes; SHA-256 `df045bde786f7d092e501bfa856984243106a13f05594f4a11fe30270fb09379` | Yes. It covers all 33,325 rows. Its JSON sidecar records 51 neutral fills with `unrouted_expert_neutral_prior:layer_routed_mean`. |
| Incremental precompute | `prod-cal-0p7/work/work/precomputed.pt`; 12,082,888,305 bytes | No as an adjoint. It contains boundary activations and a CE tail gradient, not the KL-Fisher probes AURA needs. |
| Local cost | `prod-cal-0p7/artifacts/cost.pkl`; 63,497,938 bytes; SHA-256 `08db119fe4e57da4c457106bea498ad1c4a1a4b5d6370777455beb9e29f79a9a`; 33,325 rows | Baseline/backfill only. It is local weight/output-MSE, with no AURA `predicted_dloss` and no empirical expert unit-KL. Its 43 banked layer shards are complete. |
| A-FAST cost | `cost-ldlq/burn-afast/cost_merged.pkl`; 109,058,144 bytes; weight-MSE selection; K28–K38 | No. It is the currency being replaced. Its bucket books are research inputs, not a strict `ProductionWeightCache`, and its manifest still says `eligible_le_k44`. |
| GPU mutex | `/home/rob/dq-runs/gpu.lock`; empty regular file, mode 0664 | Yes; this is the only admission mechanism used by the new driver |

No persisted AURA/adjoint output, strict production cached-menu, empirical
expert unit-KL result, or production learned `.pqcb` bundle was found under the
DSv4 run root.

## What must be built

1. A DSv4-safe KL-adjoint implementation for the 301 nonexpert units. It must
   stream the source/model state and durably checkpoint each unit; the existing
   CE probe cannot substitute for it.
2. A strict, identity-bound cached-menu render consumed with
   `aura_cost --require-production-cache`. It must use the existing
   `ProductionWeightCache` abstraction, the pinned 0p7 imatrix, and one exact
   learned/lattice producer context. It must resume verified CB pair shards
   rather than rejecting them.
3. Empirical routed serving-unit KL for all expert rows, with durable per-unit
   checkpoints and the same imatrix/bundle identity. Its output must map back
   to the allocator-visible per-expert rows without baseline-cost backfill.
4. A union-preserving hybrid merge that accepts a source-class format plan:
   expert FP8-CB K28–K33 and nonexpert FP8-CB K28–K48.
5. The immutable learned bundle and routed-book selection produced through the
   now-landed learned-bundle/MoE wiring. The new driver consumes these as
   `CB_CODEBOOK_BUNDLE` and `CB_ROUTED_MOE_BOOK_SELECTION`; it does not rebuild
   or duplicate that work.
6. An external, commit-bound execution receipt after interruption/resume, identity-drift,
   split-menu, hybrid-keyspace, and calibration-contract integration tests
   pass. Source-string capability diagnostics cannot authorize a launch.

## Source-rate-derived menu

This re-price follows the FP8-CB family used by
`tools/dsv4_ldlq_cost_campaign.py`; no demand heuristic determines its upper
bound.

- Routed experts: 43 layers × 256 experts × 3 projections = **33,024 units**.
  Their MXFP4 source is 4.25 bpp, so the legal FP8-CB menu is **K28–K33**.
  K33 is 4.140625 bpp and K34 is 4.265625 bpp for `in_features=2048`; K34 is
  byte-exactly illegal. For the separate gate/up orientation
  (`in_features=4096`), K33/K34 are 4.1328125/4.2578125 and the boundary is the
  same.
- Nonexperts: 43 × 7 = **301 units**. Their FP8 source admits **K28–K48**.
- Total FP8 encode cells: `33,024×6 + 301×21 = 204,465`.
- Learned placement is K28–K46; K47/K48 stay lattice. Thus 203,863 of those
  cells are CBL and 602 nonexpert cells are lattice.

If “CB menu” is broadened to the registry's entire product-CB set, NVFP4-CB
K12–K24 is also below both source rates. That adds `33,325×13 = 433,225` cells,
for 637,690 total product-CB cells. The prepared driver intentionally scopes
this re-price to the FP8-CB campaign identified by the request; adding the
NVFP4 family is a separate explicit scope expansion, not a demand truncation.

The landed learned-bundle work replaces the old scalar K44 ceiling with a
per-rung policy: K44/K45/K46 enabled from measured GO rows, K47/K48 disabled.
This preparation did not touch those files.

## Derived encode-time estimate

The probe inventory has 22,016 gate/up weights shaped 2048×4096 and 11,008
down weights shaped 4096×2048. Let `t_2048x4096(k)` and `t_4096x2048(k)` be
their measured per-encode seconds at rung `k`, after the active optimization
worker lands. Let `t_q(k)` be the measured time for nonexpert unit `q`. The
exact FP8-CB encode-only time is:

```text
T_encode = 22,016 × Σ(k=28..33) t_2048x4096(k)
         + 11,008 × Σ(k=28..33) t_4096x2048(k)
         + Σ(q in 301 nonexperts) Σ(k=28..48) t_q(k)
```

With the two supplied production-shape measurements only:

```text
measured-orientation part
  = 22,016 × (0.072 + t_2048x4096(29) + t_2048x4096(30)
                  + t_2048x4096(31) + t_2048x4096(32) + 0.134)
  = 4,535.296 + 22,016 × Σ(k=29..32) t_2048x4096(k) seconds
```

If, and only if, the optimized per-rung timings are confirmed monotone and
remain between those endpoints, that 22,016-unit orientation contributes
9,510.912–17,700.864 seconds = **2.642–4.917 GPU-hours**. There is no supplied
timing for the 11,008 reverse-orientation experts or the 301 nonexperts, so
neither may be folded into this bound. AURA forward/backward probes, routed
empirical-KL forwards, cache I/O, and bundle work are also separate terms. A
single-point total would therefore be a guess and is not reported.

## Code contradictions and blockers

1. `COST_MODE=aura` is not merely reachable on this branch: it is the global
   shell default and is pinned as such by tests. The CB warning at
   `run-pipeline.sh:399` simultaneously calls its objective opt-in/not-default.
   The warning and default contradict each other; the new driver sets the mode
   explicitly.
2. The generic MoE hybrid is not currently a DSv4 hybrid. DSv4 vendoring
   unfolds experts into per-expert `nn.Linear` modules. AURA's packed detector
   skips `nn.Linear`, while `expert_empirical_cost` discovers profile-declared
   3-D packed parameters. The omitted count becomes zero and the empirical leg
   is skipped.
3. Even on a recognized packed MoE, the AURA `[2d]` invocation omits
   `--col-weights`; the local CB hybrid passes it. That breaks the one-imatrix
   identity required for learned cost/cache/export.
4. One global `FORMATS` value is passed to cache/AURA/expert stages. K28–K33
   globally truncates nonexperts; K28–K48 globally measures illegal expert
   rows. The current merge also replaces the payload-wide format list with the
   empirical invocation's list.
5. `build_production_cache --streaming` rejects `render-scope=format-menu`.
   The nonstreaming cache builder, `aura_cost`, and
   `expert_empirical_cost` all load a whole model. DSv4 expands past 524 GiB
   before embeddings/activations, so none is a safe fallback.
6. CB pair files are written atomically, but current code explicitly rejects
   resuming any pre-existing CB shard. Cache manifests, AURA costs, and expert
   empirical costs are published only at full completion, contrary to the
   requested per-unit checkpoint contract.
7. Current cached-menu values are stored as BF16 rendered weights. The 301
   DSv4 nonexperts alone require about 165.8 GiB for the 21 FP8 rungs. A dense
   27B body would require roughly a tebibyte; the target filesystem exposed
   305,686,548,480 free bytes at inventory time. Exact tensor-header census,
   serialization overhead, and a free-space reserve must pass, or the cache
   needs an identity-preserving consume/evict representation inside
   `ProductionWeightCache`.
8. No Qwen3.8/Qwen38 checkpoint exists in the scanned local model/run roots.
   The dense producer profile recognizes Qwen3.5/3.6 metadata (and the
   original Qwen3 family), not a true `qwen3_8`/`Qwen3_8*` tuple. A real config
   must either resolve to an existing profile by unchanged metadata or receive
   explicit profile/serving-contract onboarding.
9. A nonempty cache/cost file is not proof of completion. The launcher no
   longer skips such files; the missing `--resume` interfaces must verify the
   model, source menu, bundle, imatrix, calibration, code revision, and every
   completed unit before reuse.

## Preflight gates before launch

All of the following must be green in one CPU-only preflight. The current
checker fully inventories the pinned DSv4 inputs and diagnoses the known code
gaps, but it deliberately retains a final BLOCK until
`AURA_CB_LAUNCH_RECEIPT` names a reviewed, HEAD-bound JSON attestation outside
the repository (and outside `/tmp` and `/home/rob/prismaquant`). Keeping the
receipt external avoids a self-referential commit hash. Bundle/selection
existence alone is not identity proof.

1. Exact source config/index identity, every indexed shard present and
   nonempty, 33,325-unit inventory, and the pinned corpus/probe/imatrix
   identities.
2. Byte-exact source census: expert K34 eliminated while K33 remains; dense
   FP8 K48 remains. Thread `019fee38` landed that shared gate as commit
   `c96eefc` while this preparation was in progress; its source-bpp tests must
   pass. This preparation consumes the gate and does not duplicate it.
3. CBL policy/bundle validation: learned K≤46, lattice K47/K48; immutable
   `.pqcb` and (for DSv4) routed selection present with matching digests.
4. Streamed DSv4 format-menu render, streamed/checkpointed KL-adjoint,
   streamed/checkpointed expert unit-KL, split-format plan, and union-preserving
   hybrid merge tests.
5. Identity-bound CB shard resume after deliberate interruption. No stale or
   unverifiable pair is admitted.
6. One explicit calibration contract across render, AURA, and empirical KL:
   `diverse-v1`, 16 samples × 512 tokens, seed 42; 32 AURA probe directions.
7. Dense-only: actual checkpoint is complete, config is dense, producer
   profile is not `DefaultProfile`, source is natively FP8 by header census,
   bundle/imatrix share the calibration identity, and exact cache payload plus
   shard overhead fits the target filesystem.
8. Known-good `gridbook:test` image present; work/scratch paths outside
   `/tmp` and `/home/rob/prismaquant`; `/home/rob/dq-runs/gpu.lock` present.

The launch re-runs the same preflight while holding the mutex. It does not
inspect accelerator state before locking.

The external receipt schema is
`prismaquant.aura_cb_reprice_launch_receipt.v1`. It must name the clean
40-hex `git rev-parse HEAD`, include the intended target in `targets`, record
the executed test selectors in a nonempty `tests` array, and set every
target-required capability listed in
`tools/aura_cb_reprice_preflight.py::_implementation_receipt_check` to JSON
`true`. It is an auditable review attestation, not a substitute for those
tests.

## Commands

CPU-only inventory (works now and reports blockers):

```bash
cd /home/rob/pq-cbl-export
tools/run_aura_cb_reprice.sh inventory dsv4
```

DSv4 launch gate (expected to stop before the mutex/container on this branch):

```bash
CB_CODEBOOK_BUNDLE=/absolute/immutable-dsv4.pqcb \
CB_ROUTED_MOE_BOOK_SELECTION=/absolute/routed-selection.json \
AURA_CB_LAUNCH_RECEIPT=/home/rob/dq-runs/aura-cb-launch-receipt.json \
  tools/run_aura_cb_reprice.sh preflight dsv4

# Run only after that returns zero:
CB_CODEBOOK_BUNDLE=/absolute/immutable-dsv4.pqcb \
CB_ROUTED_MOE_BOOK_SELECTION=/absolute/routed-selection.json \
AURA_CB_LAUNCH_RECEIPT=/home/rob/dq-runs/aura-cb-launch-receipt.json \
  tools/run_aura_cb_reprice.sh launch dsv4
```

Dense Qwen3.8-27B template (requires the actual FP8 checkpoint, its imatrix,
and immutable bundle):

```bash
MODEL_PATH=/absolute/Qwen3.8-27B-FP8 \
WORK_DIR=/home/rob/dq-runs/qwen38-27b-aura-cb \
CB_COL_WEIGHTS=/absolute/qwen38-cb-col-weights.pkl \
CB_CODEBOOK_BUNDLE=/absolute/qwen38-cbl.pqcb \
AURA_CB_LAUNCH_RECEIPT=/home/rob/dq-runs/aura-cb-launch-receipt.json \
  tools/run_aura_cb_reprice.sh preflight dense

# Run only after that returns zero:
MODEL_PATH=/absolute/Qwen3.8-27B-FP8 \
WORK_DIR=/home/rob/dq-runs/qwen38-27b-aura-cb \
CB_COL_WEIGHTS=/absolute/qwen38-cb-col-weights.pkl \
CB_CODEBOOK_BUNDLE=/absolute/qwen38-cbl.pqcb \
AURA_CB_LAUNCH_RECEIPT=/home/rob/dq-runs/aura-cb-launch-receipt.json \
  tools/run_aura_cb_reprice.sh launch dense
```

The dense launch contract is cost-only: strict cached-menu followed by straight
AURA. It does not call `run-pipeline.sh`, allocate, export, validate, or serve.
It too is expected to stop until the checkpoint, profile, identity-bound resume
interfaces, capacity proof, and commit-bound receipt exist.

## Provenance

- Repository: `/home/rob/pq-cbl-export`
- Branch inspected: `perf/ldlq-atom-compile`
- HEAD at final CPU validation: `f9c4dab` (`feat(cb): wire learned books through
  routed MoE`); source-rate gate commit `c96eefc` is its direct parent
- Date: 2026-08-11 (America/New_York)
- Authorization: Robert Tand, “Aura on CB for moe and dense.”
- GPU/CUDA work performed for this preparation: none
- Accelerator-state polling performed: none
- External writes performed: none; new files are isolated to this worktree
