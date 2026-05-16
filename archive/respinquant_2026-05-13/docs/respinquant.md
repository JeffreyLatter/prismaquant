# ReSpinQuant Investigation

Date: 2026-05-13

## Status

ReSpinQuant is not enabled in production. PrismaQuant now includes a
compatibility scout in `prismaquant.respinquant`,
`tools/respinquant_scout.py`, and an optional vLLM residual-adapter plugin
documented in `docs/residual_adapter_plugin.md`. The scout applies no
transform to exported artifacts.

After the 2026-05-13 paper-fidelity audit, the current random-Givens
residual-basis artifact is explicitly classified as a runtime-substrate smoke.
It is not paper-faithful ReSpinQuant. See
`docs/respinquant_paper_audit_2026-05-13.md`.

## Why

ReSpinQuant's key idea is layer-wise residual-basis rotation. That is more
expressive than HALO's single global residual basis, but a transformer residual
path contains identity additions. If a layer changes from basis `R_in` to
`R_out`, the residual branch needs the transition:

```text
T = R_out R_in^T
```

Without a runtime transition operator, that identity residual branch is no
longer an identity in the exported graph. This is the kernel boundary:

- **Kernel-free:** identity basis, or one shared global basis across the whole
  connected residual stream. This collapses to HALO/QuaRot-style global
  rotation.
- **Not vanilla-vLLM-safe:** true layer-wise ReSpinQuant with independent
  per-layer bases. It needs runtime residual-basis adapters, even if the paper
  makes those adapters low-rank and low-overhead.

PrismaQuant's default deployment contract is vanilla vLLM plus
compressed-tensors formats. Runtime adapters are allowed only as explicit
plugin artifacts, so we can benchmark "with adapter" and "without adapter"
variants without making the adapter a production default.

## Implementation Boundary

`prismaquant.respinquant` can:

- describe layer-wise residual-basis plans;
- identify within-layer and between-layer residual-basis transitions;
- fail fast when a plan would require runtime adapters;
- write JSON scout reports for model configs.

`run-pipeline.sh` supports:

```bash
RESPIN_MODE=scout ./prismaquant/run-pipeline.sh
```

This writes:

```text
<WORK_DIR>/artifacts/respinquant_scout.json
```

The scout applies no transform.

The optional plugin path supports hardlinked artifact variants:

```bash
python3 tools/create_residual_adapter_variant.py \
  --model-dir /path/to/exported-model \
  --output /path/to/exported-model-residual-adapter \
  --overwrite
```

Rank-0 variants are identity smokes. Rank > 0 variants need trained adapter
tensors before they are a meaningful ReSpinQuant benchmark.

`tools/create_respin_equivalent_variant.py` can materialize a research
residual-basis artifact and can write the paper's SVD/polar residual-transition
form with:

```bash
python3 tools/create_respin_equivalent_variant.py \
  --model-dir /path/to/bf16-model \
  --output /path/to/residual-basis-smoke \
  --rank 32 \
  --transition-mode paper-svd \
  --overwrite
```

That command still uses an untrained random disjoint-Givens basis unless a
future trainer supplies learned rotations. It validates tensorization and
runtime plumbing, not ReSpinQuant quality.

The first trained-rotation entry point is:

```bash
python3 tools/train_respinquant_rotations.py \
  --model /path/to/bf16-model \
  --output-dir /path/to/respin-rotation-run \
  --dataset wikitext-2 \
  --n-samples 32 \
  --seqlen 512 \
  --steps 100 \
  --lr 15
```

This command is GPU-only. It learns Hadamard-initialized dense rotations with a
Cayley optimizer, cosine LR, CE calibration loss, and activation fake
quantization. The first supported training topology is
`single_boundary_basis`, so the checkpoint is a research input for the next
export pass, not a production-ready paper-faithful ReSpinQuant artifact.

Materialize a trained checkpoint into a residual-adapter artifact with:

```bash
python3 tools/create_respin_equivalent_variant.py \
  --model-dir /path/to/bf16-model \
  --output /path/to/trained-residual-basis-artifact \
  --rotation-checkpoint /path/to/respin-rotation-run/respin_rotations.pt \
  --rank 32 \
  --transition-mode paper-svd \
  --overwrite
```

With `--rotation-checkpoint`, the writer rotates layer weights with the trained
per-layer bases, rotates the embedding table when the first layer basis is
non-identity, and writes SVD/polar low-rank transitions between consecutive
bases and back to identity after the final layer.

The first CUDA smoke is recorded in
`docs/respinquant_training_smoke_2026-05-13.md`.

## Qwen3.5-0.8B Scout

Command:

```bash
tools/respinquant_scout.py \
  --model /home/rob/.cache/huggingface/qwen35-0p8b-bf16 \
  --mode layerwise \
  --enable-layers all \
  --allow-runtime-adapter \
  --output /home/rob/dq-runs/respinquant-qwen35-0p8b-scout-20260513T133345Z.json
```

Result:

```text
kernel_free=False
requires_runtime_adapter=True
equivalent=runtime_residual_adapter
layers=24
transitions=24
```

This confirms the production boundary on the small model: true layer-wise
ReSpinQuant is not a no-kernel compressed-tensors export for Qwen3.5-0.8B.
Testing it as an exported vLLM artifact would be misleading unless vLLM gains
an explicit residual-adapter implementation.

## Pareto Recalibration Rule

Any rotation that actually changes model weights or activation coordinates
invalidates the previous probe/cost/allocator curve. Rotation changes the
per-Linear error distribution and can change which Linears should stay NVFP4,
move to FP8/MXFP8, or remain BF16.

Therefore a valid rotation arm must run:

```text
rotation applied to source/checkpoint
-> fresh probe on the rotated model
-> fresh cost table on the rotated model
-> fresh allocator Pareto curve
-> production cache/render using that assignment
-> KL + vLLM validation
```

Reusing a no-rotation layer config for a rotation arm is only a smoke test,
not a quality decision.

## References

- ReSpinQuant: https://arxiv.org/abs/2604.11080
- HALO/QuaRot basis for kernel-free global rotation:
  https://arxiv.org/abs/2404.00456
