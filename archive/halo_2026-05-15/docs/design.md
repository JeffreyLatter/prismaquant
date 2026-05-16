# HALO/SpinQuant rotation preprocessor for prismaquant — design

Branch: `feat/quality-wins-batch1`
Task: #4 / #39
Status: design draft (implementation pending)
Date: 2026-04-29

## Goal

Add a pre-quantization rotation pass that diffuses weight outliers
across channels. NVFP4 / MXFP8 quantization on the rotated weights
produces lower reconstruction error because the per-channel max norms
become more uniform after rotation. No new vLLM kernel required: the
rotation matrix is **folded into adjacent Linear weights and norm
parameters** so the runtime sees an unmodified compressed-tensors
artifact.

Expected gain: ~0.20–0.30 PPL on Llama-class models at the same bpp.
For DSv4 at 2.6 bpp, roughly Δloss 3.38 → 2.5–3.0.

## Background — why rotations help

For a Linear `y = W·x`:

- Per-channel weight outliers force NVFP4's per-group FP8 scale up,
  which spreads quantization error over the bulk of the weights.
- A random Hadamard rotation `R` (orthogonal, `R·R^T = I`) applied via
  `W' = W·R^T`, `x' = R·x` produces a mathematically identical output
  (`y = W'·x'`) but with `W'` having more uniform per-channel max norms.
- After rotation, NVFP4 quantization captures the bulk of the
  distribution accurately because no single channel dominates the scale.

QuIP# proved that random Hadamard rotations are sufficient (no learning
needed) for incoherence; SpinQuant extended this to *learned* rotations
via Cayley parameterization for a small additional gain. HALO
generalizes to *any* downstream quantization format — NVFP4, MXFP8,
GPTQ — applying rotation as a preprocessor before format-specific
quantization.

## Rotation absorption (no kernel change)

For a standard transformer block:

```
x → RMSNorm(γ) → q_proj / k_proj / v_proj → attn → out_proj → +residual
                 ↓                                ↓
              rotated input                    rotated output
```

To insert rotation `R` between RMSNorm and the q/k/v projections:

```
γ stays as 1-D vector (RMSNorm is per-channel scaling, not a matrix).
q_proj.weight ← q_proj.weight @ R^T      # absorb rotation into weight
k_proj.weight ← k_proj.weight @ R^T      # same
v_proj.weight ← v_proj.weight @ R^T      # same
```

At inference: `q_proj(LN(x)) = (W·R^T)·LN(x) = W·(R^T·LN(x))` — runtime
behavior unchanged for pre-rotation Linears, but the *quantization
target* (`W·R^T`) has uniform per-channel statistics.

For the **residual stream**, rotation must be consistent: either rotate
the entire residual stream once and stay rotated throughout the model,
or apply the rotation locally per attention/MLP block and absorb the
inverse on the way out.

### Local-rotation (per-block) approach — simpler

Each attention block:
1. Apply `R_attn` between input_layernorm and q/k/v projections.
2. Absorb `R_attn^T` into output_projection: `out_proj.weight ← R_attn @ out_proj.weight`.
3. Output projection now produces output in the unrotated frame; residual add proceeds normally.

Each MLP/MoE block:
1. Apply `R_mlp` between post_attention_layernorm and gate/up projections.
2. Absorb `R_mlp^T` into down_projection: `down_proj.weight ← R_mlp @ down_proj.weight`.
3. Same residual integrity.

Each block has its own independent rotation. No global residual-stream
rotation. Simpler to implement and reason about.

### Global-rotation approach (advanced, defer)

Rotate the entire residual stream once at the embedding output. Every
norm + linear in the network is rotated. Better outlier handling
because rotation persists through residual sums. More complex —
requires modifying every Linear in the model, including output_proj
and down_proj. Defer to v2.

## Rotation choice

| Variant | Rotation source | Quality | Cost |
|---|---|---|---|
| Random Hadamard | Fixed seed | Baseline | ~free |
| Per-Linear SVD | Eigendecomp of `H = X^T X` | +0.05 PPL | ~minutes/Linear |
| SpinQuant (Cayley) | Learned via SGD on cal data | +0.10 PPL | ~hours total |

Recommendation: **start with random Hadamard** (QuIP#-style). It's
free, deterministic from a seed, and recovers ~80% of the achievable
gain. Add learned variants later if validation shows headroom.

## DSv4 specifics

DSv4 has unusual norm topology:
- Standard `input_layernorm`, `post_attention_layernorm` on each block
- Per-attention norms: `q_norm`, `kv_norm`, `compressor.norm`
- Per-expert norms inside MoE blocks
- Compressor-specific projections (wq_a, wq_b, wkv, wo_a, wo_b)
- Hash-cluster heads (hc_*)

The per-block local-rotation approach handles the standard norm sites
cleanly. The compressor and hash-cluster paths need separate handling:
each compressor path (compressor.wq → compressor.norm → compressor.wkv)
is its own mini-block and gets its own rotation.

## Implementation plan

### Phase 1: scaffolding (week 1)

1. New module `prismaquant/halo.py`:
   - `compute_random_hadamard(d: int, seed: int) -> torch.Tensor`
   - `compute_learned_rotation(W: torch.Tensor, H: torch.Tensor) -> torch.Tensor` (SVD-based)
   - `apply_halo_rotation(block: nn.Module, R: torch.Tensor, block_kind: str) -> None`

2. Model-profile API extension:
   - `profile.halo_blocks(model) -> list[BlockSpec]`
   - Each `BlockSpec` declares: name, pre-rotation Linears (qkv/gate/up), post-rotation Linears (out_proj/down_proj), rotation_dim
   - Default implementation handles standard transformer blocks
   - DSv4-specific override handles compressor, indexer, etc.

3. CLI: `--halo-mode {off, random, learned}`, `--halo-seed N`.

### Phase 2: integration (week 2)

1. Hook into export pipeline before per-Linear NVFP4 quantization:
   - For each block in `profile.halo_blocks(model)`:
     - Compute rotation matrix (random or learned)
     - Apply to pre-rotation Linears: `W ← W @ R^T`
     - Apply to post-rotation Linears: `W ← R @ W`
   - Continue with existing NVFP4/MXFP8/etc quantization

2. Validation harness:
   - Pre/post-rotation max-channel-magnitude logging per Linear
   - Rotation invariance test: verify `(W·R^T) @ (R·x) == W @ x` numerically

### Phase 3: validation (week 2-3)

1. Qwen 4B with HALO at 4 bpp NVFP4: measure PPL improvement over
   no-HALO baseline.
2. DSv4 with HALO + cheap-batch wins: measure Δloss improvement.

## Open design questions

1. **Per-Linear vs per-block rotation.** Per-Linear gives more
   flexibility but breaks residual integrity unless paired carefully.
   Per-block keeps residual structure intact (recommended).

2. **Rotation dimension.** For attention, rotation dim = head_dim ×
   num_heads = hidden_size. For MoE/MLP, rotation dim = hidden_size
   (input to gate/up). DSv4 compressor has different dims per stage.

3. **Norm parameter handling.** RMSNorm γ is per-channel; can it
   absorb a dense rotation? **No** — γ is 1-D, rotation is 2-D.
   Solution: rotation goes into Linear weights, NOT into norm
   parameters. Norm γ is unchanged.

4. **Interaction with retired transform work.** Predecessor folds should not
   be composed into HALO runs. HALO rotates Linear weights only; keep it as a
   separate recipe arm until a new transform path is validated.

5. **MoE expert rotations.** Each expert is a separate Linear set.
   They can either share a layer-level rotation (uniform across experts
   per layer) or have per-expert rotations. Per-expert is more flexible
   but per-layer is what the kernel can absorb cleanly. Start with
   per-layer-uniform rotation across experts.

## Cost / value reminder

- Engineering: ~2 weeks (scaffolding + integration + validation)
- Calibration cost: ~minutes per layer (random Hadamard) to ~hours
  per layer (learned)
- Quality gain: ~0.20–0.30 PPL on Llama-class
- Kernel work: **none** (rotation absorbed into Linear weights)
- Stacks with: GPTQ damp sweep (#46), activation clipping (#42),
  norm FP32 (#41), activation cache FP32 (#43)
