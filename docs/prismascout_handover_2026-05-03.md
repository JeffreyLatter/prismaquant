# PrismaSCOUT Handover - 2026-05-03

This note is for the next Codex instance picking up PrismaSCOUT/SMRF work.

## Repository State

- Repo: `/home/rob/prismaquant`
- Branch: `main`
- Latest pushed commit: `8b509c0 Add validated surrogate kneedle search`
- Working tree after that commit only had pre-existing untracked local items:
  `.claude/`, `CLAUDE.md`, `propagated-cost`, `references/`, `scratch/dsv4_smoke_test.py`, `tools/`.
- No relevant Docker quantization jobs were running after the final validation.

## User Goal

The user wants PrismaSCOUT to find the best bang-for-the-buck quantization point, not just satisfy a fixed bit budget. They are comfortable with the actual bpp landing above or below a nominal target if the Pareto/knee point is better. The core thesis is that bits should be spent where they preserve downstream distribution fidelity, including inter-layer and intra-layer interactions.

Practical near-term goal:

1. Produce a better/smaller Qwen3.6-27B dense artifact than the shipped PrismaQuant v1 5.5-bit model.
2. Benchmark new artifact against shipped Hugging Face artifact.
3. Extend the method to MoE, then repeat on Qwen3.6-35B MoE.

## Important Results So Far

### Shipped 27B Baseline

Shipped model KL measurement:

- File: `/home/rob/dq-runs/qwen3p6-27b-rerun/kl_shipped_5p5_20260503T012223Z/shipped_5p5_kl.json`
- KL mean vs BF16: `0.04749751463532448`
- Calibration: `8` samples, seqlen `512`

### Strong New 27B Point

Old overnight SMRF run:

- Run dir: `/home/rob/dq-runs/qwen3p6-27b-smrf-overnight/27b-smrf-global-ship3-20260503T061645Z`
- Final assignment: `out/final_assignment_bpp_5.25.json`
- Final layer config: `out/final_layer_config_bpp_5.25.json`
- Achieved bpp: `5.3117184348527635`
- Format counts: `476 NVFP4`, `1 MXFP8_E4M3`, `137 BF16`
- Tiny validation KL: `0.015127555539947934`
- Old log had L2 KL `0.04136`, L3 KL `0.01513`, delta `-0.02623`

This is currently the best known 27B point. It needs final held-out validation using the same calibration as the shipped baseline, because the SMRF/L3 sanity runs used tiny calibration.

## New Feature Added

Implemented validated surrogate kneedle search in:

- `prismaquant/iterate_perturbed_allocation.py`
- Tests in `tests/test_iterate_perturbed_allocation.py`

New flags:

- `--knee-surrogate`
- `--knee-surrogate-points`
- `--knee-surrogate-bpp-step`
- `--knee-surrogate-neighbors`
- `--knee-surrogate-validate`
- `--knee-surrogate-validation-candidates`
- `--knee-include-assignment`

Behavior:

- CPU-only phase loads one cached L3 cost table and solves many target bpp points to produce a surrogate frontier.
- Validation phase loads the model and scores selected frontier candidates with real KL.
- The surrogate loss is treated only as candidate-generation signal.
- Explicitly included assignments are always validated.
- Generated validation candidates are spread across the frontier, capped by `--knee-surrogate-validation-candidates`, and the surrogate knee is forced into the validation set.
- `measure_assignment_kl(..., use_frozen_weight_cache=False)` is used in this mode to avoid the 27B OOM path from building a whole-model frozen assignment cache.

Tests run:

```bash
python3 -m pytest tests/test_iterate_perturbed_allocation.py -k 'knee or surrogate or multi_budget'
```

Result:

```text
13 passed, 43 deselected, 1 warning
```

## Validated-Kneedle Sanity Run

Run dir:

```text
/home/rob/dq-runs/qwen3p6-27b-smrf-overnight/validated-knee-ship3-20260503T105059
```

Output files:

- `out/surrogate_frontier.json`
- `out/validated_frontier.json`
- `out/validated_summary.json`
- `out/validated_knee_assignment.json`
- `out/final_layer_config_validated_knee.json`

Validation result:

```text
Validated knee: bpp=5.3117 KL=0.0151276 source=included_assignment
```

Candidate table from `validated_frontier.json`:

```text
5.1656 KL=0.053241 source=surrogate_frontier loss=8.556918737711385
5.3117 KL=0.015128 source=included_assignment loss=None
5.3371 KL=0.018352 source=surrogate_frontier loss=8.105775340693071
5.5090 KL=0.046646 source=surrogate_frontier loss=7.676078539574519
5.6853 KL=0.031769 source=surrogate_frontier loss=7.293711761245504
5.8336 KL=0.033356 source=surrogate_frontier loss=6.992718226509169
5.8569 KL=0.055692 source=surrogate_frontier loss=6.941715176450089
5.8801 KL=0.071730 source=surrogate_frontier loss=6.8992981754709035
6.0564 KL=0.051443 source=surrogate_frontier loss=6.587634813273326
6.2280 KL=0.075630 source=surrogate_frontier loss=6.307263671653345
6.4043 KL=0.052859 source=surrogate_frontier loss=6.0499221126083285
6.5759 KL=0.025567 source=surrogate_frontier loss=5.821134232217446
```

Interpretation:

- The CPU surrogate knee alone was bad: it picked around `5.8569` bpp and real KL was `0.055692`.
- Real validation corrected this and selected the included `5.3117` bpp assignment.
- This means validated-kneedle is the right direction, but current surrogate candidate generation is not reliable enough by itself.
- The sanity run used tiny calibration (`2 x 128`), so do not claim final model quality from it.

## Known Caveats

- The strong `5.3117` bpp point needs held-out KL at least comparable to shipped baseline (`8 x 512`), preferably larger if memory/time permit.
- Assignment count vs stats count has looked inconsistent in quick scripts before; use `_format_histogram`/repo helpers for bpp rather than ad hoc counting.
- Pairwise interaction refine was previously expensive and sometimes harmful. Do not assume it is accretive without measured validation.
- The Lagrangian/QUBO/B&B ideas were discussed but not implemented in the final committed patch. Current committed improvement is validated surrogate kneedle, not a full new optimizer.
- CUDA graph/custom kernel paths were previously suspected of hurting L3. With the algorithmic direction now more validation-candidate oriented, they may become useful again for repeated KL measurement, but only after careful profiling.

## Recommended Next Steps

1. Run held-out KL for the selected new 27B assignment against BF16 with the same calibration as the shipped baseline.

   Use:

   ```text
   /home/rob/dq-runs/qwen3p6-27b-smrf-overnight/validated-knee-ship3-20260503T105059/out/final_layer_config_validated_knee.json
   ```

   Compare directly to shipped KL `0.04749751463532448`.

2. If held-out KL confirms the sanity result, materialize/export the new 27B artifact and run smoke tests.

3. Run `tool-eval-bench` after smoke testing. User referenced:

   ```bash
   uv tool install git+https://github.com/SeraphimSerapis/tool-eval-bench.git
   ```

4. Improve candidate generation. Good options:

   - Add Lagrangian/lambda sweep candidate generation, but require real KL validation before selection.
   - Add branch-and-bound or exact DP only where the subproblem is genuinely small.
   - Treat SMRF as the main path; preserve time-bounded validation.
   - Avoid dense all-pairs unless profiling shows it pays.

5. Extend to MoE.

   User preference:

   - For Qwen3.6-27B dense: exclude MXFP4.
   - For 35B MoE: allow MXFP4 only for experts.
   - Reason discussed: MXFP4 activation support/performance in vLLM may be a constraint; be careful before mixing it into dense non-expert layers.

6. Revisit CUDA graphs/kernels only after the validated-kneedle flow is stable.

   Likely useful area:

   - repeated real-KL candidate validation with stable shapes.

   Risky area:

   - exploratory swap/interactions where pointer/format churn can thrash graph caches.

## Useful Commands

Inspect final validated result:

```bash
RUN=/home/rob/dq-runs/qwen3p6-27b-smrf-overnight/validated-knee-ship3-20260503T105059
cat "$RUN/out/validated_summary.json"
python3 - <<'PY'
import json
from pathlib import Path
run=Path('/home/rob/dq-runs/qwen3p6-27b-smrf-overnight/validated-knee-ship3-20260503T105059')
p=json.load(open(run/'out/validated_frontier.json'))
print(p['knee'])
for row in p['validated_candidates']:
    print(row['achieved_bpp'], row['validation_kl'], row['source'], row.get('surrogate_loss'))
PY
```

Check repo status:

```bash
git status --short --branch
```

Check GPU/job state:

```bash
docker ps --format '{{.Names}} {{.Status}}' | rg 'pq27b|qwen|smrf|validated' || true
nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.current.graphics,utilization.gpu --format=csv,noheader,nounits
```

## Immediate Advice To Future Codex

Do not spend hours sweeping fixed bpp points unless the user explicitly asks. The user wants the Pareto/knee point. Use surrogate/SMRF data to propose a small number of candidates, then use real KL to choose. The current best evidence says `5.3117` bpp is much stronger than both the shipped 5.5-bit baseline and the raw surrogate knee, but it still needs full held-out validation before export.
