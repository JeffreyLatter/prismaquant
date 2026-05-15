# ReSpinQuant Rotation Training Smoke — 2026-05-13

## Scope

This smoke validates the GPU-only trained-rotation path:

- Hadamard-initialized dense rotations.
- Cayley optimizer.
- CE loss with activation fake quantization.
- CUDA-only guard.
- Rotation checkpoint and metadata writing.
- Trained checkpoint materialization through SVD/polar residual adapters.
- vLLM residual-adapter plugin load.
- Local render attribution versus the existing `.8B` progressive baseline.

This is not full paper-faithful ReSpinQuant. The trainer topology is
`single_boundary_basis`, not the paper's separate MHSA, FFN, and intermediate
attention rotations.

## CUDA Environment

Host Python was CPU-only, so all hot-path work ran in:

```text
vllm-fresh-b12x-fla:latest
```

The container reported CUDA available on `NVIDIA GB10`. `accelerate` was
installed inside the disposable container so
`from_pretrained(..., device_map={"": "cuda"})` could place the model directly
on GPU. The trainer refuses CPU-staged fallback when `accelerate` is missing.

## All-Layer Training Smoke

Run directory:

```text
/home/rob/dq-runs/qwen35-0p8b-respin-trained-alllayers-lr1p5-smoke-20260513T162509Z
```

Command shape:

```bash
python3 tools/train_respinquant_rotations.py \
  --model /home/rob/dq-runs/qwen35-0p8b-untied-bf16 \
  --output-dir /home/rob/dq-runs/qwen35-0p8b-respin-trained-alllayers-lr1p5-smoke-20260513T162509Z \
  --dataset /home/rob/dq-runs/qwen35-0p8b-respin-trained-alllayers-lr1p5-smoke-20260513T162509Z/calib.txt \
  --n-samples 4 \
  --seqlen 128 \
  --steps 12 \
  --batch-size 1 \
  --layers all \
  --lr 1.5 \
  --log-every 2
```

Result:

```text
loss_initial=5.437130928039551
loss_final=4.720870494842529
rotation_count=24
orthogonality_max_abs≈2.3e-5
checkpoint=respin_rotations.pt
```

## Materialization

The checkpoint was materialized with rank-32 SVD/polar residual adapters:

```bash
python3 tools/create_respin_equivalent_variant.py \
  --model-dir /home/rob/dq-runs/qwen35-0p8b-untied-bf16 \
  --output /home/rob/dq-runs/qwen35-0p8b-respin-trained-alllayers-lr1p5-smoke-20260513T162509Z/artifact-r32 \
  --rotation-checkpoint /home/rob/dq-runs/qwen35-0p8b-respin-trained-alllayers-lr1p5-smoke-20260513T162509Z/respin_rotations.pt \
  --rank 32 \
  --transition-mode paper-svd \
  --overwrite
```

Materialization completed and correctly absorbed the final basis into the
untied `lm_head`:

```text
absorbs_final_basis_into_lm_head=true
closure_max_abs_error=0.034311845898628235
transition_mean_relative_fro_error=0.7934087440371513
transition_min_sv_energy_retained=0.14543311297893524
```

The high transition error means this simplified trainer still does not produce
the paper's low-rank/near-identity residual transitions. That is the main
negative finding.

## vLLM Smoke

The artifact loaded through the standalone residual-adapter plugin:

```bash
VLLM_PLUGINS=respinquant_residual_adapter \
python3 tools/vllm_prompt_smoke.py \
  --model /home/rob/dq-runs/qwen35-0p8b-respin-trained-alllayers-lr1p5-smoke-20260513T162509Z/artifact-r32 \
  --prompt "The capital of France is" \
  --max-new-tokens 8
```

Output:

```text
":\nA. Paris\nB."
```

The smoke validates plugin loading and generation, but the output and slow
decode do not support promotion.

## Local Attribution

Attribution command shape:

```bash
python3 tools/respin_render_attribution.py \
  --run-dir /home/rob/dq-runs/qwen35-0p8b-progressive-gates-v2-20260512T224854Z \
  --model /home/rob/dq-runs/qwen35-0p8b-untied-bf16 \
  --respin-model /home/rob/dq-runs/qwen35-0p8b-respin-trained-alllayers-lr1p5-smoke-20260513T162509Z/artifact-r32 \
  --rank 32 \
  --rotation-checkpoint /home/rob/dq-runs/qwen35-0p8b-respin-trained-alllayers-lr1p5-smoke-20260513T162509Z/respin_rotations.pt
```

Results:

```text
baseline_static_score=0.3626752692083799
trained_respin_static_score=0.4079142683616889
respin_reduction_vs_base=-0.045238999153309034
progressive_final_score=0.020980543352683595
```

For comparison, the no-rotation replay from the same baseline ended at:

```text
no_rotation_progressive_final_score=0.02007699597698948
```

## Conclusion

The trained rotation substrate works mechanically, but this implementation does
not buy quality yet. It worsens static NVFP4 local score and ends slightly worse
after FourOverSix + Fisher-GPTQ + scale sweep than the no-rotation path.

The likely issue is topology/objective mismatch: this `single_boundary_basis`
trainer lets adjacent learned bases drift in directions that rank-32 SVD/polar
adapters cannot recover. A next attempt would need either the full paper
MHSA/FFN/intermediate topology or an explicit transition/initialization
regularizer. Until then, ReSpin remains research-only and off the production
path.

The materialized 2.2 GB artifact was deleted after the smoke. The 97 MB
rotation checkpoint and attribution outputs were kept.
