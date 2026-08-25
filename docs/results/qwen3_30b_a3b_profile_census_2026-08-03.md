# Qwen3-30B-A3B profile census — 2026-08-03

**Status: PROFILE-ONLY / CPU-VERIFIED.** This is a checkpoint census and
producer-profile onboarding record, not a quantization-quality or serving
claim. No CUDA probe, allocation, export, or served measurement was performed.

## Selection

**Selected: `Qwen/Qwen3-30B-A3B` at revision
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.** It is a text-only BF16 routed
MoE using `model_type=qwen3_moe` and
`architectures=["Qwen3MoeForCausalLM"]`. Its 61.067 GB of safetensors fits the
requested 60–65 GB source window and leaves about 69 GB in a 130 GB serve
budget for the quantized artifact and runtime state.

The model search applied both external-runtime and installed-engine gates
before size:

| Candidate | Safetensors | Architecture | Result |
|---|---:|---|---|
| `Qwen/Qwen3-30B-A3B` | 61.067 GB | `Qwen3MoeForCausalLM` | **selected** |
| Qwen3 Coder / Instruct-2507 30B-A3B | 61.067 GB | `Qwen3MoeForCausalLM` | qualifies but is not smaller; specialized derivative |
| `Qwen/Qwen3-30B-A3B-Base` | 61.067 GB | `Qwen3MoeForCausalLM` | qualifies but is not smaller |
| `Qwen/Qwen3.5-35B-A3B` | 71.904 GB | `Qwen3_5MoeForConditionalGeneration` | true routed MoE, but fails the 65 GB ceiling |
| `Qwen/Qwen3.5-27B` | 55.563 GB | `Qwen3_5ForConditionalGeneration` / dense `qwen3_5_text` | fails the routed-MoE requirement |
| `Qwen/Qwen3-235B-A22B` | 470.192 GB | `Qwen3MoeForCausalLM` | fails the size ceiling |

The official Qwen catalogue search found no Qwen3.5 routed MoE smaller than
35B-A3B. The plain 30B-A3B was chosen over equal-sized derivatives so the
smoke exercises the base family contract rather than a coder or dated
instruction specialization.

## Gridbook and vLLM gates

Gridbook `origin/master` is
`9011a19228ddb96b8a49e11a20ac75c99c83998e`. Its tracked
`gridbook/runtime_contract.json` says, verbatim (numbered output from
`git show origin/master:gridbook/runtime_contract.json | nl -ba`):

```text
149  "producer_profiles": {
150    "supported_ids": [
151      "deepseek_v4",
152      "hy_v3",
153      "laguna",
154      "qwen3",
155      "qwen3_5",
156      "qwen3_5_dense"
157    ],
158    "top_level_loader_modules": [
159      "vllm.model_executor.models.hy_v3",
160      "vllm.model_executor.models.hy_v3_mtp",
161      "vllm.model_executor.models.laguna",
162      "vllm.model_executor.models.qwen3_5",
163      "vllm.model_executor.models.qwen3_5_mtp",
164      "vllm.model_executor.models.lfm2_moe",
165      "vllm.models.deepseek_v4.nvidia.model"
166    ]
```

The contract schema does not carry a separate architecture-name array. It
gates producer ID `qwen3`; the absence of a Qwen3 top-level module is expected
because this vLLM class delegates expert loading to each layer's generic
`FusedMoE` path. PrismaQuant therefore uses producer profile ID **`qwen3`**, not
the HF/vLLM module spelling `qwen3_moe`.

The image has vLLM 0.24.0. Its installed registry proves the checkpoint's
exact architecture class is present:

```text
200  "Qwen3ForCausalLM": ("qwen3", "Qwen3ForCausalLM"),
201  "Qwen3MoeForCausalLM": ("qwen3_moe", "Qwen3MoeForCausalLM"),
```

The model config selects the second line. This is a three-way match: producer
ID `qwen3` is in Gridbook, `Qwen3MoeForCausalLM` is in the downloaded config,
and that exact class is in this image's vLLM registry.

## Download

Downloaded with `huggingface_hub.snapshot_download` to
`/w/models/Qwen3-30B-A3B`, pinned to the revision above and allowing only
safetensors plus tokenizer/config files. No `.bin`, `.pt`, `.pth`, or `.gguf`
weight file is present.

| Item | Value |
|---|---:|
| safetensors shards | 16 |
| safetensors files on disk | 61,066,575,648 bytes |
| tensor data (`model.safetensors.index.json`) | 61,064,245,248 bytes |
| complete local snapshot excluding HF download cache | 61,084,157,680 bytes |
| tensors / parameters | 18,867 / 30,532,122,624 |
| index keys found in headers | 18,867 / 18,867 |

`/models/` is root-anchored in `.gitignore`; the checkpoint is not part of the
commit.

## Full safetensors census

The table is derived from the index plus all 16 shard headers. `N` denotes a
layer index and `E` a routed-expert index; bytes are tensor-data bytes, not
safetensors header overhead. Every tensor is BF16 and there are no scale or
quantization sidecars.

| Unit class / checkpoint pattern | Count | Shape | Dtype | Data GB |
|---|---:|---:|---:|---:|
| routed expert `layers.N.mlp.experts.E.gate_proj.weight` | 6,144 | `[768, 2048]` | BF16 | 19.327353 |
| routed expert `layers.N.mlp.experts.E.up_proj.weight` | 6,144 | `[768, 2048]` | BF16 | 19.327353 |
| routed expert `layers.N.mlp.experts.E.down_proj.weight` | 6,144 | `[2048, 768]` | BF16 | 19.327353 |
| routed router `layers.N.mlp.gate.weight` | 48 | `[128, 2048]` | BF16 | 0.025166 |
| attention `q_proj.weight` | 48 | `[4096, 2048]` | BF16 | 0.805306 |
| attention `k_proj.weight` | 48 | `[512, 2048]` | BF16 | 0.100663 |
| attention `v_proj.weight` | 48 | `[512, 2048]` | BF16 | 0.100663 |
| attention `o_proj.weight` | 48 | `[2048, 4096]` | BF16 | 0.805306 |
| input + post-attention RMSNorm | 96 | `[2048]` | BF16 | 0.000393 |
| attention q/k RMSNorm | 96 | `[128]` | BF16 | 0.000025 |
| final RMSNorm `model.norm.weight` | 1 | `[2048]` | BF16 | 0.000004 |
| token embedding + untied `lm_head.weight` | 2 | `[151936, 2048]` | BF16 | 1.244660 |
| **total** | **18,867** | — | **BF16** | **61.064245** |

Architecture facts verified from the real config and all headers:

- `L=48`, `E=128`, top-k 8; every layer is sparse
  (`decoder_sparse_step=1`, `mlp_only_layers=[]`).
- There is **no shared expert**: no config
  `shared_expert_intermediate_size`, no `shared_expert` tensor in any header,
  and the meta-instantiated MoE block has no `shared_expert` attribute.
- The checkpoint does **not** fuse routed gate/up weights: it stores 18,432
  separate 2-D expert projections. Transformers packs these into
  `gate_up_proj [128,1536,2048]` and
  `down_proj [128,2048,768]`.
- Attention is GQA: 32 query heads and four KV heads, head dimension 128, with
  q/k RMSNorm and no projection bias.

The quantizable body represented by the producer recipe is 336 units: 96
packed expert tensors (w13 + w2), 48 routers, and 192 attention projections,
totalling 29,909,581,824 quantizable parameters. Norms, embeddings, and
`lm_head` are outside that allocator denominator.

## Producer-profile decisions

The unified `Qwen3Profile` plus `specs/qwen3.json` owns both original dense
Qwen3 and routed Qwen3 under producer ID `qwen3`:

- the checkpoint/live/recipe namespace is otherwise identity
  (`model.layers.*`);
- per-expert checkpoint leaves stack gate then up along the output axis, then
  stack experts on axis 0 to form `gate_up_proj`; down stacks directly;
- export reverses that packed representation to per-expert
  `gate_proj` / `up_proj` / `down_proj`, including BF16;
- q/k/v are one fused serving group; dense gate/up is declared for the dense
  family member and is inert on the all-MoE selected model;
- all routed gate/up/down projections in a layer share one serving format;
- only `lm_head` is profile-pinned. The routed router is a quantizable vLLM
  `ReplicatedLinear` and remains in the empirical per-Linear menu;
- per-expert BF16 header kinds fold onto each packed recipe parent. BF16 is the
  only legal source passthrough; `FP8_SOURCE` and `MXFP4_SOURCE` are masked,
  while FP8-CB, standard FP8, and MXFP8 remain re-encode formats.

The obsolete split producer profile `qwen3_moe` is not registered or shipped.

## Probe capture points

The real config instantiated successfully on the meta device as
`transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeForCausalLM` with
30,532,122,624 meta parameters. Verified importable capture points are:

- `model.layers.N.mlp`: `Qwen3MoeSparseMoeBlock`, parent input and router;
- `model.layers.N.mlp.gate`: `Qwen3MoeTopKRouter [128,2048]`;
- `model.layers.N.mlp.experts`: `Qwen3MoeExperts`, packed w13 input and routed
  intermediate replay for w2;
- `model.layers.N.self_attn.{q,k,v,o}_proj`: ordinary `nn.Linear` modules.

## What measurement needs next

The first GPU phase should run a one-layer probe/cost smoke and assert 336 body
recipe units, 96 packed-expert units, zero unknown source kinds, parent
activation capture for every w13, and routed-intermediate replay coverage for
every w2. The production run must use the existing resident-prefetch/cache
paths and one fixed calibration contract for BF16 teacher, FP8-CB rungs, BF16
fallback, FP8_E4M3, and MXFP8_E4M3.

Promotion requires exact quantizable-parameter bpp, same-calibration KL,
resident-cache counts/misses, and served Gridbook runtime at least at parity
with the displaced container. The BF16 and quantized artifacts must co-reside
within the 130 GB serve budget. This census makes no quality or speed claim.

## CPU verification commands

```bash
python -m prismaquant.model_profiles.validate \
  --model /w/models/Qwen3-30B-A3B
pytest -q tests/test_qwen3_profile.py tests/test_spec_match_profile.py
pytest -q tests/test_model_profile_conformance.py -m 'not integration and not slow'
```
