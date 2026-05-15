#!/usr/bin/env python3
"""Create a BF16 residual-basis adapter artifact.

This is a research smoke tool, not a production exporter. It builds a model
variant whose transformer layers run in alternating residual bases and whose
inter-layer basis changes are represented by the optional PrismaQuant vLLM
residual-adapter plugin.

The construction is intentionally conservative for runtime-substrate
validation:

* the first and final residual bases are identity, so embeddings, final norm,
  and lm_head are unchanged;
* layer pairs use a low-rank disjoint-Givens transition followed by its
  inverse, so the full model returns to the original residual basis;
* RMSNorm gamma is folded into the linears that consume each norm and then the
  norm weight is set to one, which keeps the rotated residual basis coherent.

This does not implement the full ReSpinQuant paper by itself. The paper trains
full layer-wise rotations from Hadamard initialization with the Cayley
optimizer, then compresses residual-basis transitions using the SVD/polar
subspace approximation. This tool can write that paper-style transition
approximation for a supplied dense transition, but its built-in basis source is
still an untrained random disjoint-Givens smoke.

The expensive transforms are matrix multiplies on CUDA. File IO remains an
offline artifact-materialization step.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismaquant.residual_adapter import MANIFEST_FILENAME, ResidualAdapterManifest
from prismaquant.respinquant_core import (
    paper_subspace_residual_transition as _paper_subspace_residual_transition,
    residual_transition_from_bases,
)
from tools.create_residual_adapter_variant import (
    ADAPTER_TENSOR_FILE,
    create_variant,
)


LANGUAGE_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
INPUT_NORM = "input_layernorm.weight"
POST_NORM = "post_attention_layernorm.weight"
ATTN_INPUT_WEIGHTS = {
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "linear_attn.in_proj_a.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
}
ATTN_OUTPUT_WEIGHTS = {
    "self_attn.o_proj.weight",
    "linear_attn.out_proj.weight",
}
MLP_INPUT_WEIGHTS = {
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
}
MLP_OUTPUT_WEIGHTS = {
    "mlp.down_proj.weight",
}


@dataclass(frozen=True)
class GivensPlan:
    pairs: tuple[tuple[int, int, float], ...]


@dataclass(frozen=True)
class TransitionTensors:
    u: torch.Tensor
    v: torch.Tensor
    matrix: torch.Tensor
    approximation: dict[str, float | int | str] | None = None


def _text_config(config: Mapping[str, object]) -> Mapping[str, object]:
    for key in ("text_config", "language_model_config", "llm_config"):
        value = config.get(key)
        if isinstance(value, Mapping):
            return value
    return config


def _hidden_size(config: Mapping[str, object]) -> int:
    for key in ("hidden_size", "d_model", "n_embd"):
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return _hidden_size(_text_config(config))


def _num_layers(config: Mapping[str, object]) -> int:
    text_config = _text_config(config)
    for key in ("num_hidden_layers", "num_layers"):
        value = text_config.get(key)
        if isinstance(value, int) and value >= 0:
            return value
    raise ValueError("could not infer num_hidden_layers from config")


def _adapter_sites(n_layers: int) -> list[str]:
    return [f"language_model.model.layers.{idx}" for idx in range(n_layers)]


def _sample_givens_plan(hidden_size: int,
                        rank: int,
                        *,
                        generator: torch.Generator) -> GivensPlan:
    pair_count = min(int(rank) // 2, int(hidden_size) // 2)
    if pair_count <= 0:
        return GivensPlan(())
    coords = torch.randperm(hidden_size, generator=generator)[:2 * pair_count]
    pairs: list[tuple[int, int, float]] = []
    for pair_idx in range(pair_count):
        a = int(coords[2 * pair_idx])
        b = int(coords[2 * pair_idx + 1])
        sign = -1.0 if float(torch.rand((), generator=generator)) < 0.5 else 1.0
        pairs.append((a, b, sign))
    return GivensPlan(tuple(pairs))


def _givens_transition(hidden_size: int,
                       rank: int,
                       plan: GivensPlan,
                       *,
                       angle: float,
                       device: torch.device) -> TransitionTensors:
    u = torch.zeros(hidden_size, rank, dtype=torch.float32)
    v = torch.zeros(rank, hidden_size, dtype=torch.float32)
    for pair_idx, (a, b, sign) in enumerate(plan.pairs):
        col_a = 2 * pair_idx
        col_b = col_a + 1
        if col_b >= rank:
            break
        theta = float(sign) * float(angle)
        c = math.cos(theta)
        s = math.sin(theta)
        u[a, col_a] = 1.0
        u[b, col_b] = 1.0
        v[col_a, a] = c - 1.0
        v[col_a, b] = -s
        v[col_b, a] = s
        v[col_b, b] = c - 1.0
    eye = torch.eye(hidden_size, device=device, dtype=torch.float32)
    matrix = eye + u.to(device=device) @ v.to(device=device)
    return TransitionTensors(
        u=u,
        v=v,
        matrix=matrix,
        approximation={
            "mode": "exact_disjoint_givens",
            "rank": int(rank),
            "pairs": int(len(plan.pairs)),
        },
    )


def paper_subspace_residual_transition(
    transition: torch.Tensor,
    rank: int,
    *,
    device: torch.device,
) -> TransitionTensors:
    """Approximate a dense residual transition using ReSpinQuant Eq. 5-11.

    ReSpinQuant forms ``Q`` from the top singular vectors of ``T - I``,
    projects ``T`` into that subspace, polar-orthogonalizes the projected
    matrix, and applies only ``Q (R_sub - I) Q^T`` at runtime.  The vLLM
    plugin's low-rank adapter stores this as ``U=Q`` and
    ``V=(R_sub - I) Q^T`` for row-vector execution.
    """

    approx = _paper_subspace_residual_transition(
        transition,
        rank,
        device=device,
    )
    return TransitionTensors(
        u=approx.u,
        v=approx.v,
        matrix=approx.matrix,
        approximation=approx.metadata,
    )


def build_alternating_transitions(hidden_size: int,
                                  n_layers: int,
                                  rank: int,
                                  *,
                                  angle: float,
                                  seed: int,
                                  device: torch.device,
                                  transition_mode: str = "exact-givens",
                                  ) -> tuple[list[TransitionTensors], list[torch.Tensor]]:
    """Return transitions and input bases for an identity-closing plan."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    transitions: list[TransitionTensors] = []
    idx = 0
    while idx < n_layers:
        if idx == n_layers - 1:
            empty = GivensPlan(())
            transitions.append(_givens_transition(
                hidden_size, rank, empty, angle=0.0, device=device,
            ))
            break
        plan = _sample_givens_plan(hidden_size, rank, generator=generator)
        forward = _givens_transition(
            hidden_size, rank, plan, angle=angle, device=device,
        )
        inverse = _givens_transition(
            hidden_size, rank, plan, angle=-float(angle), device=device,
        )
        if transition_mode == "paper-svd":
            forward = paper_subspace_residual_transition(
                forward.matrix, rank, device=device,
            )
            inverse = paper_subspace_residual_transition(
                inverse.matrix, rank, device=device,
            )
        elif transition_mode != "exact-givens":
            raise ValueError(f"unsupported transition mode: {transition_mode}")
        transitions.append(forward)
        transitions.append(inverse)
        idx += 2

    eye = torch.eye(hidden_size, device=device, dtype=torch.float32)
    bases: list[torch.Tensor] = [eye]
    for transition in transitions:
        bases.append(bases[-1] @ transition.matrix)
    return transitions, bases[:-1]


def build_transitions_from_basis_checkpoint(
    checkpoint_path: str | Path,
    hidden_size: int,
    n_layers: int,
    rank: int,
    *,
    device: torch.device,
    transition_mode: str = "paper-svd",
    final_transition: str = "to-identity",
) -> tuple[list[TransitionTensors], list[torch.Tensor], dict[str, object]]:
    """Build per-layer bases and adapters from a trained rotation checkpoint."""

    if final_transition not in {"to-identity", "identity"}:
        raise ValueError(f"unsupported final transition: {final_transition}")
    raw = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(raw, dict) and "rotations" in raw and isinstance(raw["rotations"], dict):
        raw = raw["rotations"]
    if not isinstance(raw, dict):
        raise ValueError(f"rotation checkpoint must be a dict, got {type(raw).__name__}")
    eye = torch.eye(hidden_size, device=device, dtype=torch.float32)
    bases: list[torch.Tensor] = []
    used: list[str] = []
    missing: list[int] = []
    for idx in range(n_layers):
        key = f"layer.{idx}"
        tensor = raw.get(key)
        if tensor is None:
            bases.append(eye)
            missing.append(idx)
            continue
        basis = tensor.to(device=device, dtype=torch.float32)
        if basis.shape != (hidden_size, hidden_size):
            raise ValueError(
                f"{key} has shape {tuple(basis.shape)}, expected "
                f"{(hidden_size, hidden_size)}"
            )
        bases.append(basis)
        used.append(key)

    transitions: list[TransitionTensors] = []
    for idx, basis in enumerate(bases):
        if idx + 1 < n_layers:
            next_basis = bases[idx + 1]
        elif final_transition == "identity":
            next_basis = basis
        else:
            next_basis = eye
        dense = residual_transition_from_bases(
            basis,
            next_basis,
            convention="row",
        )
        if transition_mode == "paper-svd":
            transitions.append(paper_subspace_residual_transition(
                dense,
                rank,
                device=device,
            ))
        elif transition_mode == "exact-givens":
            # Kept for CLI compatibility. A trained dense transition cannot be
            # represented exactly at low rank; use a full-rank U/V only when
            # explicitly requested by setting rank >= hidden_size.
            if rank < hidden_size:
                raise ValueError(
                    "trained dense transitions need --transition-mode paper-svd "
                    "or --rank >= hidden_size for exact representation"
                )
            transitions.append(TransitionTensors(
                u=torch.eye(hidden_size, dtype=torch.float32),
                v=(dense - eye).detach().cpu(),
                matrix=dense,
                approximation={
                    "mode": "exact_dense_full_rank",
                    "rank": int(hidden_size),
                },
            ))
        else:
            raise ValueError(f"unsupported transition mode: {transition_mode}")

    meta = {
        "rotation_checkpoint": str(checkpoint_path),
        "basis_source": "trained_checkpoint",
        "used_rotation_count": len(used),
        "used_rotation_keys": used,
        "missing_identity_layers": missing,
        "final_transition": final_transition,
    }
    return transitions, bases, meta


def _read_index(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return {}
    data = json.loads(index_path.read_text())
    weight_map = data.get("weight_map", {})
    if not isinstance(weight_map, dict):
        return {}
    return {str(key): str(value) for key, value in weight_map.items()}


def _read_tensor(model_dir: Path,
                 key: str,
                 *,
                 weight_map: Mapping[str, str]) -> torch.Tensor:
    filename = weight_map.get(key)
    if filename is None:
        candidates = sorted(model_dir.glob("*.safetensors"))
    else:
        candidates = [model_dir / filename]
    for path in candidates:
        if path.name == ADAPTER_TENSOR_FILE:
            continue
        with safe_open(path, framework="pt", device="cpu") as handle:
            if key in handle.keys():
                return handle.get_tensor(key)
    raise KeyError(f"tensor not found in {model_dir}: {key}")


def _try_read_tensor(model_dir: Path,
                     key: str,
                     *,
                     weight_map: Mapping[str, str]) -> torch.Tensor | None:
    try:
        return _read_tensor(model_dir, key, weight_map=weight_map)
    except KeyError:
        return None


def _layer_gammas(model_dir: Path,
                  n_layers: int,
                  *,
                  weight_map: Mapping[str, str]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    input_gammas: list[torch.Tensor] = []
    post_gammas: list[torch.Tensor] = []
    for idx in range(n_layers):
        prefix = f"model.language_model.layers.{idx}"
        input_gammas.append(_read_tensor(
            model_dir,
            f"{prefix}.{INPUT_NORM}",
            weight_map=weight_map,
        ))
        post_gammas.append(_read_tensor(
            model_dir,
            f"{prefix}.{POST_NORM}",
            weight_map=weight_map,
        ))
    return input_gammas, post_gammas


def _norm_style(config: Mapping[str, object], requested: str = "auto") -> str:
    requested = requested.strip().lower()
    if requested != "auto":
        if requested not in {"standard", "gemma"}:
            raise ValueError(f"unsupported norm style: {requested}")
        return requested
    text_config = _text_config(config)
    model_type = str(text_config.get("model_type") or config.get("model_type") or "")
    architectures = config.get("architectures") or ()
    if isinstance(architectures, str):
        architectures = (architectures,)
    arch_text = " ".join(str(item) for item in architectures)
    if "qwen3_5" in model_type or "Qwen3_5" in arch_text:
        return "gemma"
    return "standard"


def _effective_norm_gamma(raw: torch.Tensor, norm_style: str) -> torch.Tensor:
    if norm_style == "gemma":
        return raw + 1.0
    return raw


def _identity_norm_weight(raw: torch.Tensor, norm_style: str) -> torch.Tensor:
    if norm_style == "gemma":
        return torch.zeros_like(raw)
    return torch.ones_like(raw)


def _fold_norm_and_rotate_input(weight: torch.Tensor,
                                gamma: torch.Tensor,
                                basis: torch.Tensor,
                                *,
                                device: torch.device,
                                norm_style: str = "standard") -> torch.Tensor:
    original_dtype = weight.dtype
    w = weight.to(device=device, dtype=torch.float32)
    g = _effective_norm_gamma(gamma, norm_style).to(device=device, dtype=torch.float32)
    out = (w * g.unsqueeze(0)) @ basis
    return out.to(dtype=original_dtype, device="cpu")


def _rotate_output_projection(weight: torch.Tensor,
                              basis: torch.Tensor,
                              *,
                              device: torch.device) -> torch.Tensor:
    original_dtype = weight.dtype
    w = weight.to(device=device, dtype=torch.float32)
    out = basis.transpose(0, 1) @ w
    return out.to(dtype=original_dtype, device="cpu")


def _transform_tensor(key: str,
                      tensor: torch.Tensor,
                      *,
                      bases: list[torch.Tensor],
                      input_gammas: list[torch.Tensor],
                      post_gammas: list[torch.Tensor],
                      final_gamma: torch.Tensor | None,
                      absorb_final_basis: bool,
                      device: torch.device,
                      norm_style: str) -> tuple[torch.Tensor, str | None]:
    if (
        key.endswith(".embed_tokens.weight")
        or key.endswith(".wte.weight")
        or key == "embed_tokens.weight"
    ):
        if tensor.ndim == 2 and bases and tensor.shape[1] == bases[0].shape[0]:
            original_dtype = tensor.dtype
            w = tensor.to(device=device, dtype=torch.float32)
            out = w @ bases[0].to(device=device, dtype=torch.float32)
            return out.to(dtype=original_dtype, device="cpu"), "rotate_embedding"
        return tensor, None
    if absorb_final_basis and key == "model.language_model.norm.weight":
        return _identity_norm_weight(tensor, norm_style), "fold_final_norm"
    if absorb_final_basis and key == "lm_head.weight" and final_gamma is not None:
        return _fold_norm_and_rotate_input(
            tensor,
            final_gamma,
            bases[-1],
            device=device,
            norm_style=norm_style,
        ), "rotate_lm_head"
    match = LANGUAGE_LAYER_RE.match(key)
    if match is None:
        return tensor, None
    layer_idx = int(match.group(1))
    suffix = match.group(2)
    if layer_idx >= len(bases):
        return tensor, None
    basis = bases[layer_idx]
    if suffix == INPUT_NORM:
        return _identity_norm_weight(tensor, norm_style), "fold_input_norm"
    if suffix == POST_NORM:
        return _identity_norm_weight(tensor, norm_style), "fold_post_norm"
    if suffix in ATTN_INPUT_WEIGHTS:
        return _fold_norm_and_rotate_input(
            tensor, input_gammas[layer_idx], basis,
            device=device,
            norm_style=norm_style,
        ), "rotate_attention_input"
    if suffix in MLP_INPUT_WEIGHTS:
        return _fold_norm_and_rotate_input(
            tensor, post_gammas[layer_idx], basis,
            device=device,
            norm_style=norm_style,
        ), "rotate_mlp_input"
    if suffix in ATTN_OUTPUT_WEIGHTS:
        return _rotate_output_projection(tensor, basis, device=device), "rotate_attention_output"
    if suffix in MLP_OUTPUT_WEIGHTS:
        return _rotate_output_projection(tensor, basis, device=device), "rotate_mlp_output"
    return tensor, None


def _rewrite_model_safetensors(model_dir: Path,
                               *,
                               bases: list[torch.Tensor],
                               input_gammas: list[torch.Tensor],
                               post_gammas: list[torch.Tensor],
                               final_gamma: torch.Tensor | None,
                               absorb_final_basis: bool,
                               device: torch.device,
                               norm_style: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for shard in sorted(model_dir.glob("*.safetensors")):
        if shard.name == ADAPTER_TENSOR_FILE:
            continue
        tensors: dict[str, torch.Tensor] = {}
        changed = False
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                transformed, action = _transform_tensor(
                    key,
                    tensor,
                    bases=bases,
                    input_gammas=input_gammas,
                    post_gammas=post_gammas,
                    final_gamma=final_gamma,
                    absorb_final_basis=absorb_final_basis,
                    device=device,
                    norm_style=norm_style,
                )
                tensors[key] = transformed
                if action is not None:
                    changed = True
                    counts[action] = counts.get(action, 0) + 1
        if changed:
            tmp = shard.with_name(f"{shard.name}.tmp")
            save_file(tensors, str(tmp))
            os.replace(tmp, shard)
    return counts


def _write_transition_tensors(model_dir: Path,
                              manifest: ResidualAdapterManifest,
                              transitions: list[TransitionTensors],
                              *,
                              dtype: torch.dtype) -> dict[str, int]:
    tensors: dict[str, torch.Tensor] = {}
    nonzero_u = 0
    nonzero_v = 0
    active = [spec for spec in manifest.adapters if spec.enabled and spec.rank > 0]
    if len(active) != len(transitions):
        raise ValueError(
            f"adapter count ({len(active)}) does not match transition count "
            f"({len(transitions)})"
        )
    for spec, transition in zip(active, transitions):
        if spec.u_name is None or spec.v_name is None:
            raise ValueError(f"rank-{spec.rank} adapter missing tensor names: {spec}")
        u = transition.u.to(dtype=dtype)
        v = transition.v.to(dtype=dtype)
        tensors[spec.u_name] = u
        tensors[spec.v_name] = v
        nonzero_u += int(torch.count_nonzero(u).item())
        nonzero_v += int(torch.count_nonzero(v).item())
    save_file(tensors, str(model_dir / ADAPTER_TENSOR_FILE))
    return {"adapter_tensors": len(tensors), "nonzero_u": nonzero_u, "nonzero_v": nonzero_v}


def create_respin_equivalent_variant(model_dir: str | Path,
                                     output: str | Path,
                                     *,
                                     rank: int = 16,
                                     angle: float = 0.05,
                                     seed: int = 0,
                                     device: str = "cuda",
                                     norm_style: str = "auto",
                                     transition_mode: str = "exact-givens",
                                     rotation_checkpoint: str | Path | None = None,
                                     overwrite: bool = False) -> dict[str, object]:
    if rank < 0:
        raise ValueError("--rank must be >= 0")
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise RuntimeError("ReSpin equivalent materialization is GPU-only; pass --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing CPU materialization")
    if rotation_checkpoint is not None and transition_mode == "exact-givens":
        transition_mode = "paper-svd"

    src = Path(model_dir)
    dst = Path(output)
    config = json.loads((src / "config.json").read_text())
    hidden_size = _hidden_size(config)
    n_layers = _num_layers(config)
    resolved_norm_style = _norm_style(config, norm_style)

    summary = create_variant(
        src,
        dst,
        module_paths=_adapter_sites(n_layers),
        rank=rank,
        dtype="bfloat16",
        initializer="zero",
        overwrite=overwrite,
    )
    patched_config_path = dst / "config.json"
    patched_config = json.loads(patched_config_path.read_text())
    basis_source = "random_disjoint_givens_untrained"
    if rotation_checkpoint is not None:
        basis_source = "trained_checkpoint"
    fidelity_note = (
        "Not full ReSpinQuant: rotations are untrained random disjoint "
        "Givens bases, not Hadamard-initialized Cayley-optimized full "
        "layer-wise rotations. Use this only to validate residual-adapter "
        "runtime plumbing unless a trained basis is supplied."
    )
    if rotation_checkpoint is not None:
        fidelity_note = (
            "Not full paper ReSpinQuant unless the checkpoint topology "
            "contains the MHSA, FFN, and intermediate-attention rotations "
            "required by the paper. The current PrismaQuant trainer writes "
            "single_boundary_basis checkpoints."
        )
    patched_config["prisma_respin_equivalent"] = {
        "angle": float(angle),
        "basis_source": basis_source,
        "basis_schedule": (
            "trained_checkpoint_per_layer"
            if rotation_checkpoint is not None else
            "alternating_pair_inverse"
        ),
        "description": (
            "Research-only BF16 residual-basis smoke: layer weights are "
            "rotated offline and low-rank residual transitions are applied "
            "by the PrismaQuant vLLM residual-adapter plugin."
        ),
        "hidden_size": hidden_size,
        "num_layers": n_layers,
        "norm_style": resolved_norm_style,
        "paper_faithful": False,
        "paper_fidelity_note": fidelity_note,
        "rank": int(rank),
        "rotation_checkpoint": str(rotation_checkpoint) if rotation_checkpoint is not None else None,
        "seed": int(seed),
        "transition_mode": transition_mode,
        "version": 1,
    }
    patched_config_path.write_text(json.dumps(patched_config, indent=2, sort_keys=True) + "\n")

    checkpoint_meta: dict[str, object] = {}
    if rotation_checkpoint is not None:
        transitions, bases, checkpoint_meta = build_transitions_from_basis_checkpoint(
            rotation_checkpoint,
            hidden_size,
            n_layers,
            rank,
            device=torch_device,
            transition_mode=transition_mode,
        )
    else:
        transitions, bases = build_alternating_transitions(
            hidden_size,
            n_layers,
            rank,
            angle=float(angle),
            seed=int(seed),
            device=torch_device,
            transition_mode=transition_mode,
        )
    weight_map = _read_index(src)
    has_lm_head = _try_read_tensor(src, "lm_head.weight", weight_map=weight_map) is not None
    absorb_final_basis = bool(rotation_checkpoint is not None and has_lm_head)
    if rotation_checkpoint is not None and not has_lm_head:
        raise RuntimeError(
            "trained ReSpin basis materialization requires an untied lm_head "
            "so the final residual basis can be absorbed exactly. Stage an "
            "untied source checkpoint first."
        )
    input_gammas, post_gammas = _layer_gammas(src, n_layers, weight_map=weight_map)
    final_gamma = (
        _try_read_tensor(src, "model.language_model.norm.weight", weight_map=weight_map)
        if absorb_final_basis else
        None
    )
    if absorb_final_basis and final_gamma is None:
        raise RuntimeError("could not find final norm weight for lm_head basis absorb")
    if rotation_checkpoint is not None:
        # Rebuild transitions with the final basis absorbed into lm_head instead
        # of approximating a dense last-basis -> identity adapter.
        transitions, bases, checkpoint_meta = build_transitions_from_basis_checkpoint(
            rotation_checkpoint,
            hidden_size,
            n_layers,
            rank,
            device=torch_device,
            transition_mode=transition_mode,
            final_transition="identity",
        )
    patched_config["prisma_respin_equivalent"][
        "absorbs_final_basis_into_lm_head"
    ] = bool(absorb_final_basis)
    patched_config_path.write_text(json.dumps(patched_config, indent=2, sort_keys=True) + "\n")
    eye = torch.eye(hidden_size, device=torch_device, dtype=torch.float32)
    if transitions:
        chain = bases[0].clone()
        for transition in transitions:
            chain = chain @ transition.matrix
        target = bases[-1] if absorb_final_basis else eye
        closure_error = float((chain - target).abs().max().item())
    else:
        closure_error = 0.0
    transform_counts = _rewrite_model_safetensors(
        dst,
        bases=bases,
        input_gammas=input_gammas,
        post_gammas=post_gammas,
        final_gamma=final_gamma,
        absorb_final_basis=absorb_final_basis,
        device=torch_device,
        norm_style=resolved_norm_style,
    )
    manifest = ResidualAdapterManifest.load(dst / MANIFEST_FILENAME)
    adapter_counts = _write_transition_tensors(
        dst,
        manifest,
        transitions,
        dtype=torch.bfloat16,
    )
    approximation_modes: dict[str, int] = {}
    relative_errors: list[float] = []
    retained_energies: list[float] = []
    for transition in transitions:
        meta = transition.approximation or {}
        mode = str(meta.get("mode", "unknown"))
        approximation_modes[mode] = approximation_modes.get(mode, 0) + 1
        if "relative_fro_error" in meta:
            relative_errors.append(float(meta["relative_fro_error"]))
        if "sv_energy_retained" in meta:
            retained_energies.append(float(meta["sv_energy_retained"]))

    out = dict(summary)
    out.update({
        "angle": float(angle),
        "basis_source": basis_source,
        "basis_schedule": (
            "trained_checkpoint_per_layer"
            if rotation_checkpoint is not None else
            "alternating_pair_inverse"
        ),
        "closure_max_abs_error": closure_error,
        "hidden_size": hidden_size,
        "num_layers": n_layers,
        "norm_style": resolved_norm_style,
        "paper_faithful": False,
        "rank": int(rank),
        "absorbs_final_basis_into_lm_head": bool(absorb_final_basis),
        "rotation_checkpoint": str(rotation_checkpoint) if rotation_checkpoint is not None else None,
        "seed": int(seed),
        "transition_approximation_modes": approximation_modes,
        "transition_max_relative_fro_error": (
            max(relative_errors) if relative_errors else None
        ),
        "transition_mean_relative_fro_error": (
            sum(relative_errors) / len(relative_errors) if relative_errors else None
        ),
        "transition_min_sv_energy_retained": (
            min(retained_energies) if retained_energies else None
        ),
        "transition_mode": transition_mode,
        "transform_counts": transform_counts,
        "adapter_counts": adapter_counts,
    })
    out.update(checkpoint_meta)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True,
                        help="Source Hugging Face/vLLM model artifact.")
    parser.add_argument("--output", required=True,
                        help="Output artifact directory to create.")
    parser.add_argument("--rank", type=int, default=16,
                        help="Low-rank transition rank; even values are preferred.")
    parser.add_argument("--angle", type=float, default=0.05,
                        help="Disjoint-Givens angle in radians.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Deterministic transition seed.")
    parser.add_argument("--device", default="cuda",
                        help="Materialization device. Only CUDA is accepted.")
    parser.add_argument("--norm-style", default="auto",
                        choices=("auto", "standard", "gemma"),
                        help="RMSNorm checkpoint convention. Qwen3.5 uses gemma.")
    parser.add_argument("--transition-mode", default="exact-givens",
                        choices=("exact-givens", "paper-svd"),
                        help=(
                            "Residual-adapter tensorization. exact-givens writes "
                            "the random smoke transition exactly; paper-svd writes "
                            "the ReSpinQuant Eq. 5-11 SVD/polar low-rank "
                            "approximation of that transition."
                        ))
    parser.add_argument("--rotation-checkpoint", default=None,
                        help=(
                            "Optional trained rotation checkpoint from "
                            "tools/train_respinquant_rotations.py. When set, "
                            "per-layer bases come from the checkpoint and "
                            "--transition-mode paper-svd is recommended."
                        ))
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace the output directory if it already exists.")
    args = parser.parse_args(argv)

    summary = create_respin_equivalent_variant(
        args.model_dir,
        args.output,
        rank=args.rank,
        angle=args.angle,
        seed=args.seed,
        device=args.device,
        norm_style=args.norm_style,
        transition_mode=args.transition_mode,
        rotation_checkpoint=args.rotation_checkpoint,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
