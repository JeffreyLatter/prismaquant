"""HALO/QuaRot-style rotation preprocessor for prismaquant.

Applies an orthogonal rotation `R` to the residual stream that diffuses
weight outliers across channels, producing more uniform per-channel
weight statistics. Subsequent NVFP4/MXFP8 quantization on the rotated
weights has lower reconstruction error.

The rotation is absorbed into existing Linear weights and the embedding
table — vLLM serves the resulting artifact unchanged. No runtime
Hadamard transform is needed because the rotation cancels along its
path through the network: residual stream is rotated, every Linear
that reads from it is right-rotated (W ← W @ R^T), every Linear that
writes to it is left-rotated (W ← R @ W), and the cancellation
preserves model semantics exactly.

Critical prerequisite: RMSNorm gamma must be folded into the downstream
Linear before rotation. Otherwise diag(gamma) doesn't commute with R
and `LN(R·x) ≠ R·LN(x)`. We fold gamma first, then rotate. After folding
the norm gamma vector is set to 1.

This module is architecture-agnostic. Block topology is declared by the
ModelProfile (see `halo_block_specs`); the default profile handles
standard transformer blocks (input_layernorm + q/k/v/o_proj, post_attn_norm
+ gate/up/down_proj). MoE models reuse the same machinery — experts'
gate/up/down projections are treated as block-input/output linears with
the relevant per-expert tensors passed through.

References:
- QuaRot (Ashkboos et al. 2024): residual-stream rotation via Hadamard
  with norm-gamma absorption.
- HALO (Ashkboos et al. 2025): generalizes to NVFP4/MXFP8/etc.
- QuIP# (Tseng et al. 2024): proves random Hadamard is sufficient for
  incoherence; learning the rotation gives only ~0.05 PPL more.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Rotation generation
# ---------------------------------------------------------------------------

def hadamard_matrix(d: int, device: torch.device | str = "cpu",
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Sylvester-construction Hadamard matrix for power-of-2 dimensions.

    Sylvester construction is deterministic and gives exactly ±1/sqrt(d)
    entries — preserves the Hadamard property `H @ H.T = I` exactly in
    floating point.
    """
    if d <= 0:
        raise ValueError(f"Hadamard dimension must be positive; got {d}")
    if d & (d - 1) == 0:  # d is a power of 2
        h = torch.tensor([[1.0]], device=device, dtype=dtype)
        size = 1
        while size < d:
            h = torch.cat([
                torch.cat([h, h], dim=1),
                torch.cat([h, -h], dim=1),
            ], dim=0)
            size *= 2
        return h / math.sqrt(d)
    raise ValueError(
        f"HALO random Hadamard requires a power-of-2 dimension; got {d}. "
        "Non-power-of-2 dense QR fallback is intentionally disabled for "
        "streaming export.")


def random_hadamard(d: int, seed: int = 0,
                    device: torch.device | str = "cpu",
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Randomized Hadamard rotation: Sylvester Hadamard composed with a
    random sign diagonal. Equivalent in incoherence properties to the
    Walsh-Hadamard + random sign trick used by QuIP#/QuaRot.
    """
    H = hadamard_matrix(d, device=device, dtype=dtype)
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    sign = torch.randint(0, 2, (d,), generator=g, device=device,
                         dtype=torch.int32) * 2 - 1
    sign = sign.to(dtype=dtype)
    # H @ diag(sign) — sign-shuffled rows of H. Still orthogonal.
    return H * sign[None, :]


# ---------------------------------------------------------------------------
# Block topology declaration (architecture-agnostic via ModelProfile)
# ---------------------------------------------------------------------------

@dataclass
class HaloBlockSpec:
    """Declares one rotation site in the model.

    The rotation `R` of dimension `dim` flows through this block:
      - `norm_qname` is the RMSNorm whose gamma must be folded first.
      - Each `input_linears` reads from the rotated residual: weight is
        right-rotated (W ← W @ R^T).
      - Each `output_linears` writes to the rotated residual: weight is
        left-rotated (W ← R @ W).

    For MoE blocks, `input_linears` and `output_linears` should list the
    fused-experts tensor names; the rotation is applied uniformly across
    all experts (per-expert rotation would require a kernel change).
    """
    name: str                          # human-readable label (e.g., "layer0.attn")
    dim: int                           # rotation dimension (typically hidden_size)
    norm_qname: str                    # RMSNorm to fold gamma from
    input_linears: list[str] = field(default_factory=list)   # right-rotated
    output_linears: list[str] = field(default_factory=list)  # left-rotated


@dataclass
class HaloModelSpec:
    """Top-level rotation spec for an entire model.

    `embed_qname`: embedding table to right-rotate (W ← W @ R^T) so
    residual stream starts in the rotated frame.

    `lm_head_qname`: LM head to right-rotate (W ← W @ R^T) so the final
    logits compute correctly on the rotated final-residual.

    `final_norm_qname`: RMSNorm immediately before lm_head; gamma is
    folded into lm_head weights.

    `block_specs`: list of per-block rotation sites.

    `dim`: the residual stream dimension (must match across all blocks).
    """
    dim: int
    embed_qname: str
    lm_head_qname: str
    final_norm_qname: str | None
    block_specs: list[HaloBlockSpec] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Gamma folding
# ---------------------------------------------------------------------------

def _get_module_by_qname(model: nn.Module, qname: str) -> nn.Module:
    """Resolve a dotted qname to a submodule."""
    parts = qname.split(".")
    cur = model
    for p in parts:
        if not p:
            continue
        if p.isdigit():
            cur = cur[int(p)]
        else:
            cur = getattr(cur, p)
    return cur


def fold_gamma_into_linears(model: nn.Module, norm_qname: str,
                            downstream_linears: list[str]) -> None:
    """Fold an RMSNorm's gamma into one or more downstream Linear weights
    by per-input-channel scaling. After this:
      - The Linear's effective behavior on `gamma * normalize(x)` is
        unchanged from its prior behavior on the same input.
      - The norm's gamma is reset to 1.0 so subsequent rotation is valid.

    For shared norms feeding multiple Linears (the standard q/k/v fan-out
    after input_layernorm), each Linear absorbs the full gamma.
    """
    norm = _get_module_by_qname(model, norm_qname)
    if not hasattr(norm, "weight"):
        return
    gamma = norm.weight.detach().to(torch.float32)
    for linear_qname in downstream_linears:
        lin = _get_module_by_qname(model, linear_qname)
        if not hasattr(lin, "weight"):
            continue
        W = lin.weight.detach().to(torch.float32)
        # W has shape [out, in]; gamma is per-input-channel scaling.
        if W.shape[1] != gamma.shape[0]:
            raise ValueError(
                f"gamma shape {gamma.shape} doesn't match Linear input dim "
                f"{W.shape[1]} for {linear_qname}")
        W = W * gamma[None, :]
        lin.weight.data = W.to(lin.weight.dtype)
    # Reset gamma to 1 so future operations don't double-apply.
    with torch.no_grad():
        norm.weight.fill_(1.0)


# ---------------------------------------------------------------------------
# Rotation application
# ---------------------------------------------------------------------------

def _right_rotate_linear(model: nn.Module, qname: str,
                         R: torch.Tensor) -> None:
    """Rotate a Linear's input direction: `W ← W @ R^T`.

    Used for Linears that read from the rotated residual (q/k/v_proj,
    gate_proj, up_proj, lm_head).
    """
    lin = _get_module_by_qname(model, qname)
    if not hasattr(lin, "weight"):
        return
    W = lin.weight.detach().to(torch.float32)
    R32 = R.to(W.device, dtype=torch.float32)
    if W.shape[1] != R32.shape[0]:
        raise ValueError(
            f"Linear {qname} input dim {W.shape[1]} doesn't match "
            f"rotation dim {R32.shape[0]}")
    W = W @ R32.t()
    lin.weight.data = W.to(lin.weight.dtype)


def _left_rotate_linear(model: nn.Module, qname: str,
                        R: torch.Tensor) -> None:
    """Rotate a Linear's output direction: `W ← R @ W`.

    Used for Linears that write to the rotated residual (out_proj,
    down_proj).
    """
    lin = _get_module_by_qname(model, qname)
    if not hasattr(lin, "weight"):
        return
    W = lin.weight.detach().to(torch.float32)
    R32 = R.to(W.device, dtype=torch.float32)
    if W.shape[0] != R32.shape[0]:
        raise ValueError(
            f"Linear {qname} output dim {W.shape[0]} doesn't match "
            f"rotation dim {R32.shape[0]}")
    W = R32 @ W
    lin.weight.data = W.to(lin.weight.dtype)


def _right_rotate_embedding(model: nn.Module, qname: str,
                            R: torch.Tensor) -> None:
    """Rotate embedding's output direction: `W ← W @ R^T`.

    Embedding shape is [vocab, hidden]; rotating the hidden dim
    (right-mul by R^T) makes embedding output start in the rotated frame.
    """
    embed = _get_module_by_qname(model, qname)
    if not hasattr(embed, "weight"):
        return
    W = embed.weight.detach().to(torch.float32)
    R32 = R.to(W.device, dtype=torch.float32)
    if W.shape[1] != R32.shape[0]:
        raise ValueError(
            f"Embedding {qname} hidden dim {W.shape[1]} doesn't match "
            f"rotation dim {R32.shape[0]}")
    W = W @ R32.t()
    embed.weight.data = W.to(embed.weight.dtype)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def apply_halo_rotation(model: nn.Module, spec: HaloModelSpec, *,
                        seed: int = 0,
                        verbose: bool = False) -> torch.Tensor:
    """Apply a global HALO/QuaRot rotation to an entire model in-place.

    Steps (must run in this order):
      1. Fold all RMSNorm gammas into their downstream Linears
         (block-level norms + final norm).
      2. Generate a random Hadamard rotation `R` of dimension `spec.dim`.
      3. Right-rotate the embedding output: `embed.W ← embed.W @ R^T`.
      4. Per block:
         - Right-rotate input_linears (q/k/v/gate/up): `W ← W @ R^T`
         - Left-rotate output_linears (o_proj/down_proj): `W ← R @ W`
      5. Right-rotate lm_head: `lm_head.W ← lm_head.W @ R^T`.

    After this, the model produces identical outputs (up to floating-
    point numerical equivalence) but its weights have more uniform
    per-channel statistics, suitable for downstream low-bit quantization.

    Returns the rotation matrix `R` (saved alongside the artifact for
    forensic reproducibility).
    """
    R = random_hadamard(spec.dim, seed=seed, dtype=torch.float32)

    # 1. Fold gammas first (block norms + final norm).
    for bspec in spec.block_specs:
        downstream = list(bspec.input_linears)
        fold_gamma_into_linears(model, bspec.norm_qname, downstream)
        if verbose:
            print(f"[halo] folded gamma {bspec.norm_qname} → "
                  f"{len(downstream)} linears")
    if spec.final_norm_qname is not None and spec.lm_head_qname:
        fold_gamma_into_linears(
            model, spec.final_norm_qname, [spec.lm_head_qname])
        if verbose:
            print(f"[halo] folded final gamma {spec.final_norm_qname} → "
                  f"{spec.lm_head_qname}")

    # 2-3. Embedding right-rotation.
    _right_rotate_embedding(model, spec.embed_qname, R)
    if verbose:
        print(f"[halo] rotated embedding {spec.embed_qname}")

    # 4. Per-block rotations.
    n_in = 0
    n_out = 0
    for bspec in spec.block_specs:
        for q in bspec.input_linears:
            _right_rotate_linear(model, q, R)
            n_in += 1
        for q in bspec.output_linears:
            _left_rotate_linear(model, q, R)
            n_out += 1
    if verbose:
        print(f"[halo] rotated {n_in} input linears, {n_out} output linears")

    # 5. lm_head rotation.
    if spec.lm_head_qname:
        _right_rotate_linear(model, spec.lm_head_qname, R)
        if verbose:
            print(f"[halo] rotated lm_head {spec.lm_head_qname}")

    return R


# ---------------------------------------------------------------------------
# Default block-spec builder for standard transformer architectures
# ---------------------------------------------------------------------------

def block_specs_for_layer(layer_mod: nn.Module, layer_qname: str,
                          dim: int) -> list[HaloBlockSpec]:
    """Layer-scoped variant of `default_block_specs`. Used by the
    streaming export integration which materializes one layer at a
    time. Returns 0–2 HaloBlockSpec entries for this layer (one for
    attention, optionally one for MLP/MoE) using qnames RELATIVE to
    `layer_qname` (e.g., `f"{layer_qname}.self_attn.q_proj"`).

    Each spec's qnames are resolvable via _get_module_by_qname against
    the FULL model, but the rotation applier can also operate against
    a single layer module by stripping the layer prefix.
    """
    blocks: list[HaloBlockSpec] = []

    def _has(qname: str) -> bool:
        try:
            _get_module_by_qname(layer_mod, qname)
            return True
        except (AttributeError, IndexError, KeyError):
            return False

    # Attention block.
    attn_in: list[str] = []
    for proj in ("self_attn.q_proj", "self_attn.k_proj",
                 "self_attn.v_proj"):
        if _has(proj):
            attn_in.append(f"{layer_qname}.{proj}")
    attn_out: list[str] = []
    for proj in ("self_attn.o_proj", "self_attn.out_proj"):
        if _has(proj):
            attn_out.append(f"{layer_qname}.{proj}")
            break
    if attn_in and attn_out and _has("input_layernorm"):
        blocks.append(HaloBlockSpec(
            name=f"{layer_qname}.attn",
            dim=dim,
            norm_qname=f"{layer_qname}.input_layernorm",
            input_linears=attn_in,
            output_linears=attn_out,
        ))

    # MLP / MoE block. Standard MLP has gate/up/down. MoE has experts
    # under various names — here we only handle dense MLPs; per-expert
    # rotation (uniform across experts in a layer) is a separate path.
    mlp_in: list[str] = []
    for proj in ("mlp.gate_proj", "mlp.up_proj"):
        if _has(proj):
            mlp_in.append(f"{layer_qname}.{proj}")
    if _has("mlp.down_proj") and mlp_in and _has("post_attention_layernorm"):
        blocks.append(HaloBlockSpec(
            name=f"{layer_qname}.mlp",
            dim=dim,
            norm_qname=f"{layer_qname}.post_attention_layernorm",
            input_linears=mlp_in,
            output_linears=[f"{layer_qname}.mlp.down_proj"],
        ))

    return blocks


def apply_halo_to_layer(model: nn.Module, layer_mod: nn.Module,
                        layer_qname: str, R: torch.Tensor, *,
                        strict: bool = True) -> int:
    """Apply HALO rotation to a single layer's modules in-place.

    Folds input_layernorm and post_attention_layernorm gammas into
    their downstream Linears, then applies right/left rotations to
    the q/k/v/o (attention) and gate/up/down (MLP) projections.

    `model` is the full model (so qnames resolve); `layer_mod` is just
    the body for module resolution; `layer_qname` is the dotted prefix
    (e.g. "model.layers.5"). Returns the count of Linears rotated.
    """
    dim = R.shape[0]
    specs = block_specs_for_layer(layer_mod, layer_qname, dim)
    if not specs:
        if strict:
            raise RuntimeError(
                f"[halo] {layer_qname}: no supported standard "
                "attention/MLP rotation sites. HALO currently supports "
                "standard dense transformer layers only.")
        return 0
    n = 0
    for bspec in specs:
        try:
            # 1. Fold gamma first.
            fold_gamma_into_linears(model, bspec.norm_qname, bspec.input_linears)
            # 2. Right-rotate input linears, left-rotate output linears.
            for q in bspec.input_linears:
                _right_rotate_linear(model, q, R)
                n += 1
            for q in bspec.output_linears:
                _left_rotate_linear(model, q, R)
                n += 1
        except (AttributeError, IndexError, KeyError, ValueError) as exc:
            if strict:
                raise RuntimeError(
                    f"[halo] failed rotating {bspec.name}: {exc}") from exc
    return n


def apply_halo_to_head(model: nn.Module, R: torch.Tensor, *,
                       embed_qname: str = "model.embed_tokens",
                       lm_head_qname: str = "lm_head",
                       final_norm_qname: str | None = "model.norm",
                       strict: bool = True,
                       ) -> int:
    """Apply HALO rotation to embedding + final_norm + lm_head.

    Run ONCE at the start of streaming materialization for the head
    tensors. Folds final_norm gamma into lm_head, right-rotates
    embedding output direction, right-rotates lm_head input direction.

    Returns the count of head tensors rotated.
    """
    n = 0
    try:
        embed = _get_module_by_qname(model, embed_qname)
        head = _get_module_by_qname(model, lm_head_qname)
        tied = (
            hasattr(embed, "weight")
            and hasattr(head, "weight")
            and (
                embed.weight is head.weight
                or embed.weight.data_ptr() == head.weight.data_ptr()
            )
        )
        if tied:
            raise ValueError(
                f"{embed_qname}.weight and {lm_head_qname}.weight are tied")
    except (AttributeError, IndexError, KeyError, ValueError) as exc:
        if strict:
            raise RuntimeError(
                f"[halo] unsupported head topology for embed={embed_qname} "
                f"lm_head={lm_head_qname}: {exc}") from exc
        return n

    if final_norm_qname:
        try:
            fold_gamma_into_linears(model, final_norm_qname, [lm_head_qname])
        except (AttributeError, IndexError, KeyError, ValueError) as exc:
            if strict:
                raise RuntimeError(
                    f"[halo] failed folding final norm {final_norm_qname} "
                    f"into {lm_head_qname}: {exc}") from exc
    try:
        _right_rotate_embedding(model, embed_qname, R)
        n += 1
    except (AttributeError, IndexError, KeyError, ValueError) as exc:
        if strict:
            raise RuntimeError(
                f"[halo] failed rotating embedding {embed_qname}: {exc}") from exc
    try:
        _right_rotate_linear(model, lm_head_qname, R)
        n += 1
    except (AttributeError, IndexError, KeyError, ValueError) as exc:
        if strict:
            raise RuntimeError(
                f"[halo] failed rotating lm_head {lm_head_qname}: {exc}") from exc
    return n


def default_block_specs(model: nn.Module, *,
                        body_layer_prefix: str = "model.layers",
                        embed_qname: str = "model.embed_tokens",
                        lm_head_qname: str = "lm_head",
                        final_norm_qname: str = "model.norm",
                        ) -> HaloModelSpec:
    """Build a HaloModelSpec by walking a standard transformer model.

    Identifies blocks by enumerating `model.layers.*` and locating the
    canonical projections (q_proj, k_proj, v_proj, o_proj/out_proj,
    gate_proj, up_proj, down_proj) plus their norms.

    Works out of the box for Llama-style and Qwen-style architectures.
    For MoE models with experts, `o_proj` rotation is per-block (one
    rotation across all experts in the block); `gate_proj`/`up_proj` is
    handled at the per-expert tensor names if the model exposes them
    fused (e.g., `experts.gate_up_proj`) or per-expert.

    Models with non-standard topology (DSv4 compressors, indexers) need
    a profile-specific override that emits HaloBlockSpec entries for
    each non-standard rotation site.
    """
    # Discover hidden_size by inspecting the embedding.
    embed = _get_module_by_qname(model, embed_qname)
    if not hasattr(embed, "weight"):
        raise ValueError(f"embedding {embed_qname} not found or has no weight")
    hidden = embed.weight.shape[1]

    # Enumerate body layers by attempting attribute access on the layer list.
    blocks: list[HaloBlockSpec] = []
    parts = body_layer_prefix.split(".")
    layer_list = model
    for p in parts:
        if not p:
            continue
        layer_list = getattr(layer_list, p)
    n_layers = len(layer_list)

    for i in range(n_layers):
        layer_qname = f"{body_layer_prefix}.{i}"

        # Attention block.
        attn_in = []
        for proj in ("self_attn.q_proj", "self_attn.k_proj",
                     "self_attn.v_proj"):
            try:
                _get_module_by_qname(model, f"{layer_qname}.{proj}")
                attn_in.append(f"{layer_qname}.{proj}")
            except AttributeError:
                pass
        attn_out = []
        for proj in ("self_attn.o_proj", "self_attn.out_proj"):
            try:
                _get_module_by_qname(model, f"{layer_qname}.{proj}")
                attn_out.append(f"{layer_qname}.{proj}")
                break
            except AttributeError:
                continue
        if attn_in and attn_out:
            blocks.append(HaloBlockSpec(
                name=f"layer{i}.attn",
                dim=hidden,
                norm_qname=f"{layer_qname}.input_layernorm",
                input_linears=attn_in,
                output_linears=attn_out,
            ))

        # MLP / MoE block.
        mlp_in = []
        for proj in ("mlp.gate_proj", "mlp.up_proj"):
            try:
                _get_module_by_qname(model, f"{layer_qname}.{proj}")
                mlp_in.append(f"{layer_qname}.{proj}")
            except AttributeError:
                pass
        mlp_out = []
        try:
            _get_module_by_qname(model, f"{layer_qname}.mlp.down_proj")
            mlp_out.append(f"{layer_qname}.mlp.down_proj")
        except AttributeError:
            pass
        if mlp_in and mlp_out:
            blocks.append(HaloBlockSpec(
                name=f"layer{i}.mlp",
                dim=hidden,
                norm_qname=f"{layer_qname}.post_attention_layernorm",
                input_linears=mlp_in,
                output_linears=mlp_out,
            ))

    return HaloModelSpec(
        dim=hidden,
        embed_qname=embed_qname,
        lm_head_qname=lm_head_qname,
        final_norm_qname=final_norm_qname,
        block_specs=blocks,
    )
