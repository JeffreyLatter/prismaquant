# ReSpinQuant Residual-Adapter vLLM Plugin

The ReSpinQuant runtime plugin now ships as a separate package:

```text
https://github.com/RobTand/respinquant-vllm-plugin
```

PrismaQuant writes the artifact metadata and adapter tensors, but it no longer
ships the vLLM runtime plugin. Install the external package when loading
artifacts that advertise `PrismaResidualAdapterForCausalLM`.

The plugin is generic: artifacts keep their original architecture in
`prisma_residual_adapters.json`, while `config.json` advertises
`PrismaResidualAdapterForCausalLM` so vLLM instantiates the wrapper.

The wrapper delegates to the base vLLM model and attaches forward hooks at
manifest module paths. A rank-`r` adapter applies:

```text
x -> x + (x @ U) @ V
```

This uses ordinary PyTorch matmul kernels and no custom CUDA kernel. Rank `0`
is identity and is intended for vLLM load smoke tests.

For paper-style ReSpinQuant residual transitions, PrismaQuant stores
`U=Q` and `V=(R_sub - I)Q^T`, where `Q` is the top singular subspace of
`T - I` and `R_sub` is the polar-orthogonalized projection of `T` into that
subspace. The plugin is only the runtime representation; a valid quality
benchmark still requires trained layer-wise rotations.

`tools/train_respinquant_rotations.py` trains rotation checkpoints, but its
initial topology is `single_boundary_basis`. Do not treat those checkpoints as
full paper-faithful ReSpinQuant until the runtime adapter manifest can represent
the internal MHSA and FFN residual-basis transitions required by the paper.

## Create an Optional Variant

```bash
python3 tools/create_residual_adapter_variant.py \
  --model-dir /path/to/base-or-exported-model \
  --output /path/to/model-with-residual-adapter \
  --overwrite
```

That command hardlinks the source artifact and writes only:

- patched `config.json`
- `prisma_residual_adapters.json`
- optional `prisma-residual-adapters.safetensors` for rank > 0 adapters

For actual trained adapters, pass insertion sites and rank:

```bash
python3 tools/create_residual_adapter_variant.py \
  --model-dir /path/to/base-or-exported-model \
  --output /path/to/model-with-residual-adapter \
  --site model.layers.0 \
  --site model.layers.1 \
  --rank 16 \
  --overwrite
```

The current creator initializes rank > 0 tensors to zero. A ReSpin trainer or
export pass should overwrite those tensors with learned residual-basis
transitions before benchmarking.

## vLLM Loading

Install the standalone package so vLLM sees the `vllm.general_plugins` entry
point:

```bash
pip install git+https://github.com/RobTand/respinquant-vllm-plugin.git
VLLM_PLUGINS=respinquant_residual_adapter python3 -m prismaquant.validate_native_export \
  --model /path/to/model-with-residual-adapter \
  --prompt "The capital of France is" \
  --max-new-tokens 16
```

The plugin package also registers the legacy entry point name
`prismaquant_residual_adapter` for already-written local launch scripts.

vLLM may log the resolved architecture as the base model architecture. That is
intentional: the plugin delegates capability inspection to the base model so
multimodal, M-RoPE, hybrid, and pipeline-parallel metadata remain correct,
while model construction still resolves to the residual-adapter wrapper.

This is a research/runtime-adapter path, not a production default. It may be
used to compare artifacts with and without residual adapters, and any rotation
that changes the residual basis still requires a fresh Pareto probe because it
changes the activation and error geometry.

## Smoke Results

On 2026-05-13, the rank-0 identity variant
`/home/rob/dq-runs/qwen35-0p8b-residual-adapter-identity-20260513T000000Z`
loaded in `vllm-fresh-b12x-fla:latest` with
`VLLM_PLUGINS=prismaquant_residual_adapter` and generated from:

```text
The capital of France is
```

The smoke validated plugin registration, base-architecture capability
delegation, checkpoint weight accounting, Qwen3.5 multimodal/M-RoPE delegation,
and generation on GPU. The artifact has no active adapters; it is a wrapper
integration smoke, not a ReSpin quality result.

The active-adapter smoke used:

```text
/home/rob/dq-runs/qwen35-0p8b-residual-adapter-rank16-alllayers-20260513T141354Z
```

It attached rank-16 zero-initialized adapters to all 24 Qwen3.5-0.8B decoder
layers:

```text
language_model.model.layers.0 ... language_model.model.layers.23
```

Results against `/home/rob/.cache/huggingface/qwen35-0p8b-bf16` on the prompt
`The capital of France is`, `max_tokens=4`, `temperature=0.0`:

| Arm | Mode | Init seconds | Generate seconds | Text |
|---|---:|---:|---:|---|
| Base BF16 | eager | 74.641 | 46.391 | ` Paris.\nThe` |
| Rank-16 residual adapters | eager | 76.104 | 46.486 | ` Paris.\nThe` |
| Rank-16 residual adapters | graph | 126.114 | 18.658 | ` Paris.\nThe` |

The active smoke confirms that vLLM loads adapter tensors from the second
safetensors shard, initializes the wrapper-visible parameters, captures CUDA
graphs, and generates with the adapter hooks present. Because the tensors are
zero-initialized, this is still an integration/performance smoke, not a quality
measurement of trained residual-basis transitions.

During this test, the variant creator was fixed to copy mutable metadata files
(`config.json`, `model.safetensors.index.json`) instead of hardlinking them.
That prevents patched variant metadata from mutating the source artifact.

On 2026-05-13, a nonzero ReSpin-style transition smoke was created at:

```text
/home/rob/dq-runs/qwen35-0p8b-respin-givens-r16-a0p05-20260513T145022Z
```

Settings:

```text
rank=16
initializer=respin-givens
angle=0.05 radians
seed=0
sites=all 24 decoder layers
```

The initializer wrote eight disjoint Givens residual rotations per layer. The
adapter shard contained 48 tensors with 384 nonzero `U` entries and 768
nonzero `V` entries; `max(abs(V)) = 0.050048828125` in BF16.

vLLM graph-mode inference on GPU:

| Arm | Mode | Init seconds | Generate seconds | Text |
|---|---:|---:|---:|---|
| Rank-16 ReSpin-Givens adapters | graph | 126.654 | 17.627 | ` Paris.\nThe capital of France is Paris.\nThe capital of France is` |

This validates nonzero adapter tensor loading, CUDA graph capture, and
generation with active residual transitions. It is still a runtime substrate
smoke: the base model weights were not rotated into matching per-layer bases,
so this is not a ReSpinQuant quality result.

On 2026-05-13, the plugin was extracted to
`/home/rob/respinquant-vllm-plugin` as the separately shipping repository
`RobTand/respinquant-vllm-plugin`. The standalone package was installed in the
vLLM container and loaded the corrected ReSpin-equivalent `.8B` artifact:

```text
/home/rob/dq-runs/qwen35-0p8b-respin-equivalent-gemmafix-r16-a0p05-20260513T151343Z
```

with `VLLM_PLUGINS=respinquant_residual_adapter`, CUDA graph mode, and output:

```text
" Paris.\nThe"
```

That artifact was a random-Givens residual-basis smoke, not a paper-faithful
ReSpinQuant run. The paper-fidelity audit is recorded in
`docs/respinquant_paper_audit_2026-05-13.md`.
