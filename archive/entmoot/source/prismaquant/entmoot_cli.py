"""Command-line tools for Entmoot v1.

The CLI intentionally separates collection from planning:

* `collect-jsonl` runs a calibration stream through the model and writes a
  bounded activation-sketch artifact.
* `plan-from-collector` turns that artifact into an exportable
  `entmoot_router_id_v1` merge manifest.
* `summarize-collector` / `summarize-manifest` provide quick sanity checks
  before any expensive export.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from prismaquant.entmoot import (
    build_router_id_merge_plan,
    merge_plan_manifest,
    select_rrqr_anchors_from_gram,
)
from prismaquant.entmoot_collector import (
    EntmootActivationCollector,
    LayerSketchBuffer,
    load_collector_state,
)
from prismaquant.entmoot_router_diag import choose_router_strategy
from prismaquant.expert_calibration_survey import iter_jsonl_rows, text_from_row
from prismaquant.observers.expert_saliency import saliency_from_packed_moe
from prismaquant.sensitivity_probe import (
    read_top_k,
    resolve_execution_device,
    stage_text_only,
)


_DEFAULT_SYNTHESIS_ALPHA_GRID = (
    0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.40, 0.50, 0.60, 0.70,
    0.80, 0.90, 0.95,
)
_DEFAULT_DECOUPLED_SYNTHESIS_GRID = (
    0.05, 0.15, 0.30, 0.45,
    0.55, 0.70, 0.85, 0.95,
)


def collect_jsonl(
    model,
    tokenizer,
    dataset: str | Path,
    output: str | Path,
    *,
    device: torch.device,
    max_length: int = 2048,
    limit: int | None = None,
    max_samples_per_expert: int = 256,
    router_regex: str | None = None,
    seed: int = 0,
    forward_mode: str = "backbone",
    stop_after_layer: int | None = None,
    progress_every: int = 1,
) -> dict[str, Any]:
    """Collect Entmoot activation sketches from a JSONL calibration file."""

    packed_blocks = saliency_from_packed_moe(model)
    if router_regex:
        import re
        rx = re.compile(router_regex)
        packed_blocks = [
            b for b in packed_blocks
            if rx.search(str(b.get("router_qname", "")))
        ]
    if not packed_blocks:
        raise ValueError("no packed MoE blocks discovered for Entmoot collection")

    collector = EntmootActivationCollector(
        model,
        packed_moe_blocks=packed_blocks,
        max_samples_per_expert=max_samples_per_expert,
        seed=seed,
    )
    restore_layers = _truncate_decoder_layers(model, stop_after_layer)
    n_seen = 0
    n_used = 0
    try:
        for _line_no, row in iter_jsonl_rows(dataset, limit=limit):
            n_seen += 1
            text = text_from_row(row, tokenizer)
            if not text:
                continue
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            kwargs = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in encoded.items()
            }
            with torch.inference_mode():
                _run_collection_forward(model, kwargs, forward_mode=forward_mode)
            n_used += 1
            if progress_every > 0 and n_used % progress_every == 0:
                print(
                    f"entmoot collect: rows_used={n_used} rows_seen={n_seen}",
                    file=sys.stderr,
                    flush=True,
                )
        collector.save(output)
        summary = {
            "format": "entmoot_collect_jsonl_v1",
            "dataset": str(dataset),
            "output": str(output),
            "rows_seen": n_seen,
            "rows_used": n_used,
            "max_length": int(max_length),
            "max_samples_per_expert": int(max_samples_per_expert),
            "forward_mode": forward_mode,
            "stop_after_layer": stop_after_layer,
            "routers": sorted(collector.layers),
            "layers": collector.summaries(),
        }
        return summary
    finally:
        collector.remove_hooks()
        restore_layers()


def _run_collection_forward(
    model: nn.Module,
    kwargs: Mapping[str, Any],
    *,
    forward_mode: str,
) -> None:
    """Run one calibration forward without computing logits when possible."""

    if forward_mode == "causal-lm":
        target = model
    elif forward_mode == "backbone":
        target = _backbone_module(model)
    else:
        raise ValueError(f"unknown forward mode {forward_mode!r}")

    call_kwargs = dict(kwargs)
    call_kwargs.setdefault("use_cache", False)
    call_kwargs.setdefault("output_attentions", False)
    call_kwargs.setdefault("output_hidden_states", False)
    target(**call_kwargs)


def _backbone_module(model: nn.Module) -> nn.Module:
    """Return the decoder backbone for common HF CausalLM wrappers."""

    for attr in ("model", "language_model", "transformer"):
        child = getattr(model, attr, None)
        if isinstance(child, nn.Module) and child is not model:
            if hasattr(child, "layers") or hasattr(child, "h"):
                return child
            nested = getattr(child, "model", None)
            if isinstance(nested, nn.Module) and (
                hasattr(nested, "layers") or hasattr(nested, "h")
            ):
                return nested
    return model


def _truncate_decoder_layers(
    model: nn.Module,
    stop_after_layer: int | None,
):
    """Temporarily keep decoder layers through an inclusive layer index."""

    if stop_after_layer is None:
        return lambda: None
    if stop_after_layer < 0:
        raise ValueError("stop_after_layer must be >= 0")

    owner, attr, layers = _decoder_layers(model)
    keep = int(stop_after_layer) + 1
    if keep >= len(layers):
        return lambda: None
    original = layers
    replacement = nn.ModuleList(list(layers[:keep]))
    setattr(owner, attr, replacement)

    def restore() -> None:
        setattr(owner, attr, original)

    return restore


def _decoder_layers(model: nn.Module) -> tuple[nn.Module, str, nn.ModuleList]:
    for root in (model, _backbone_module(model)):
        for attr in ("layers", "h", "blocks"):
            layers = getattr(root, attr, None)
            if isinstance(layers, nn.ModuleList):
                return root, attr, layers
    raise ValueError("could not find decoder layer ModuleList")


def plan_from_collector(
    collector_path: str | Path,
    output: str | Path,
    *,
    target_experts: int | None = None,
    keep_ratio: float | None = None,
    routers: Sequence[str] | None = None,
    activation_accept_threshold: float = 0.05,
    activation_tentative_threshold: float = 0.10,
    min_routed_mass: float = 0.0,
    min_samples: int = 128,
    router_strategy: str = "anchor",
    normalize_features: bool = False,
    activation_fit_model: str | Path | None = None,
    activation_fit_device: str = "cuda",
    activation_fit_dtype: str = "bf16",
    activation_fit_top_anchors: int = 4,
    activation_fit_mode: str = "anchor",
    synthesis_alpha_grid: Sequence[float] | None = None,
    synthesis_beta_grid: Sequence[float] | None = None,
    synthesis_refine_steps: int = 0,
    synthesis_refine_radius: float = 0.12,
    synthesis_refine_shrink: float = 0.5,
    synthesis_refine_threshold: float | None = None,
) -> dict[str, Any]:
    """Build an Entmoot merge manifest from a saved collector artifact."""

    layers = load_collector_state(collector_path)
    router_filter = set(routers or [])
    plans = []
    for router, layer in sorted(layers.items()):
        if router_filter and router not in router_filter:
            continue
        ids, features = layer.output_feature_matrix(normalize=normalize_features)
        routed_mass = layer.routed_mass()
        sample_counts = {s.expert_id: s.samples for s in layer.stats()}
        k = _target_for_layer(
            layer,
            target_experts=target_experts,
            keep_ratio=keep_ratio,
        )
        anchor_residuals = None
        candidate_anchor_ids = None
        synthesis_weights = None
        tensor_synthesis_weights = None
        if activation_fit_model is not None:
            if activation_fit_mode == "anchor":
                anchor_residuals, candidate_anchor_ids = _activation_anchor_residuals(
                    layer,
                    ids,
                    features,
                    target_experts=k,
                    model_path=activation_fit_model,
                    device=activation_fit_device,
                    dtype=_dtype_from_name(activation_fit_dtype),
                    top_anchors=activation_fit_top_anchors,
                    min_samples=min_samples,
                )
            elif activation_fit_mode == "synthesis_pair":
                (
                    anchor_residuals,
                    candidate_anchor_ids,
                    synthesis_weights,
                ) = _activation_synthesis_pair_residuals(
                    layer,
                    ids,
                    features,
                    target_experts=k,
                    model_path=activation_fit_model,
                    device=activation_fit_device,
                    dtype=_dtype_from_name(activation_fit_dtype),
                    top_anchors=activation_fit_top_anchors,
                    min_samples=min_samples,
                    alpha_grid=synthesis_alpha_grid or _DEFAULT_SYNTHESIS_ALPHA_GRID,
                )
            elif activation_fit_mode == "synthesis_pair_decoupled":
                grid = synthesis_alpha_grid or _DEFAULT_DECOUPLED_SYNTHESIS_GRID
                (
                    anchor_residuals,
                    candidate_anchor_ids,
                    synthesis_weights,
                    tensor_synthesis_weights,
                ) = _activation_synthesis_pair_decoupled_residuals(
                    layer,
                    ids,
                    features,
                    target_experts=k,
                    model_path=activation_fit_model,
                    device=activation_fit_device,
                    dtype=_dtype_from_name(activation_fit_dtype),
                    top_anchors=activation_fit_top_anchors,
                    min_samples=min_samples,
                    alpha_grid=grid,
                    beta_grid=synthesis_beta_grid or grid,
                    refine_steps=synthesis_refine_steps,
                    refine_radius=synthesis_refine_radius,
                    refine_shrink=synthesis_refine_shrink,
                    refine_threshold=synthesis_refine_threshold,
                )
            else:
                raise ValueError(f"unknown activation_fit_mode {activation_fit_mode!r}")
        plan = build_router_id_merge_plan(
            features,
            routed_mass,
            expert_ids=ids,
            target_experts=k,
            router_qname=router,
            routed_mass=routed_mass,
            sample_counts=sample_counts,
            activation_accept_threshold=activation_accept_threshold,
            activation_tentative_threshold=activation_tentative_threshold,
            min_routed_mass=min_routed_mass,
            min_samples=min_samples,
            router_strategy=router_strategy,
            anchor_residuals=anchor_residuals,
            candidate_anchor_ids=candidate_anchor_ids,
            synthesis_weights=synthesis_weights,
            tensor_synthesis_weights=tensor_synthesis_weights,
        )
        plans.append(plan)

    manifest = merge_plan_manifest(plans)
    Path(output).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "format": "entmoot_plan_from_collector_v1",
        "collector": str(collector_path),
        "output": str(output),
        "routers": sorted(manifest),
        "n_layers": len(manifest),
        "n_experts_orig": sum(int(e["num_experts_orig"]) for e in manifest.values()),
        "n_experts_kept": sum(int(e["num_experts_kept"]) for e in manifest.values()),
        "activation_fit_model": None if activation_fit_model is None else str(activation_fit_model),
        "activation_fit_mode": activation_fit_mode,
    }


def _activation_anchor_residuals(
    layer: LayerSketchBuffer,
    expert_ids: Sequence[int],
    features: torch.Tensor,
    *,
    target_experts: int,
    model_path: str | Path,
    device: str,
    dtype: torch.dtype,
    top_anchors: int,
    min_samples: int,
) -> tuple[dict[tuple[int, int], float], dict[int, list[int]]]:
    """Measure anchor substitution on the same hidden rows routed to each expert."""

    ids = [int(e) for e in expert_ids]
    G = features.to(torch.float64) @ features.to(torch.float64).T
    anchors = select_rrqr_anchors_from_gram(G, ids, target_experts=target_experts)
    anchor_set = set(anchors)
    id_to_idx = {eid: i for i, eid in enumerate(ids)}
    if int(top_anchors) <= 0:
        top_n = len(anchors)
    else:
        top_n = max(1, min(int(top_anchors), len(anchors)))
    candidates_by_eid: dict[int, list[int]] = {}
    for eid in ids:
        if eid in anchor_set:
            continue
        j = id_to_idx[eid]
        scored = []
        for anchor in anchors:
            a = id_to_idx[anchor]
            norm = float(G[j, j].clamp_min(1e-12))
            rel = max(float(G[j, j] + G[a, a] - 2.0 * G[j, a]), 0.0) / norm
            scored.append((rel, anchor))
        candidates_by_eid[eid] = [a for _rel, a in sorted(scored)[:top_n]]

    gate_up, down = _load_packed_expert_tensors(model_path, layer.router_qname)
    act_name = _hidden_act_from_config(model_path)
    run_device = torch.device(device)
    if run_device.type == "cuda" and not torch.cuda.is_available():
        run_device = torch.device("cpu")
    compute_dtype = dtype if run_device.type == "cuda" else torch.float32
    gate_up = gate_up.to(device=run_device, dtype=compute_dtype)
    down = down.to(device=run_device, dtype=compute_dtype)

    residuals: dict[tuple[int, int], float] = {}
    measured = 0
    with torch.inference_mode():
        for eid in ids:
            if eid in anchor_set:
                continue
            hidden, target, route_weight = layer.experts[eid].stacked()
            if target.numel() == 0 or int(target.shape[0]) < int(min_samples):
                continue
            candidates = candidates_by_eid.get(eid, [])
            if not candidates:
                continue
            idx = torch.as_tensor(candidates, dtype=torch.long, device=run_device)
            h = hidden.to(device=run_device, dtype=compute_dtype)
            y = target.to(device=run_device, dtype=torch.float32)
            w = route_weight.to(device=run_device, dtype=torch.float32).clamp_min(0.0)
            if float(w.sum().item()) <= 0.0:
                w = torch.ones_like(w)
            pred = _packed_expert_forward_candidates(
                h,
                gate_up.index_select(0, idx),
                down.index_select(0, idx),
                act_name=act_name,
            ).to(torch.float32)
            err = (pred - y.unsqueeze(0)).pow(2).sum(dim=-1)
            denom = y.pow(2).sum(dim=-1).clamp_min(1e-12)
            rel = (err * w.unsqueeze(0)).sum(dim=1) / (denom * w).sum().clamp_min(1e-12)
            for n, anchor in enumerate(candidates):
                residuals[(int(eid), int(anchor))] = float(rel[n].item())
            measured += 1
            if measured % 32 == 0:
                print(
                    f"entmoot activation-fit: {layer.router_qname} measured={measured}",
                    file=sys.stderr,
                    flush=True,
                )

    if run_device.type == "cuda":
        torch.cuda.empty_cache()
    print(
        f"entmoot activation-fit: {layer.router_qname} pairs={len(residuals)}",
        file=sys.stderr,
        flush=True,
    )
    return residuals, candidates_by_eid


def _activation_synthesis_pair_residuals(
    layer: LayerSketchBuffer,
    expert_ids: Sequence[int],
    features: torch.Tensor,
    *,
    target_experts: int,
    model_path: str | Path,
    device: str,
    dtype: torch.dtype,
    top_anchors: int,
    min_samples: int,
    alpha_grid: Sequence[float],
) -> tuple[
    dict[tuple[int, int], float],
    dict[int, list[int]],
    dict[tuple[int, int], dict[int, float]],
]:
    """Fit pairwise stock expert rows by blending packed expert weights.

    A candidate new expert row is:

        W_new = (1 - alpha) * W_anchor + alpha * W_dropped

    measured on the calibration hidden rows for both the anchor and dropped
    expert.  This preserves the stock packed-expert checkpoint format; the
    exporter already materializes these linear tensor blends.
    """

    ids = [int(e) for e in expert_ids]
    G = features.to(torch.float64) @ features.to(torch.float64).T
    anchors = select_rrqr_anchors_from_gram(G, ids, target_experts=target_experts)
    anchor_set = set(anchors)
    id_to_idx = {eid: i for i, eid in enumerate(ids)}
    alphas = tuple(float(a) for a in alpha_grid if 0.0 <= float(a) <= 1.0)
    if not alphas:
        raise ValueError("synthesis alpha grid must contain values in [0, 1]")
    if int(top_anchors) <= 0:
        top_n = len(anchors)
    else:
        top_n = max(1, min(int(top_anchors), len(anchors)))

    candidates_by_eid: dict[int, list[int]] = {}
    for eid in ids:
        if eid in anchor_set:
            continue
        j = id_to_idx[eid]
        scored = []
        for anchor in anchors:
            a = id_to_idx[anchor]
            norm = float(G[j, j].clamp_min(1e-12))
            rel = max(float(G[j, j] + G[a, a] - 2.0 * G[j, a]), 0.0) / norm
            scored.append((rel, anchor))
        candidates_by_eid[eid] = [a for _rel, a in sorted(scored)[:top_n]]

    gate_up, down = _load_packed_expert_tensors(model_path, layer.router_qname)
    act_name = _hidden_act_from_config(model_path)
    run_device = torch.device(device)
    if run_device.type == "cuda" and not torch.cuda.is_available():
        run_device = torch.device("cpu")
    compute_dtype = dtype if run_device.type == "cuda" else torch.float32
    gate_up = gate_up.to(device=run_device, dtype=compute_dtype)
    down = down.to(device=run_device, dtype=compute_dtype)

    residuals: dict[tuple[int, int], float] = {}
    synthesis_weights: dict[tuple[int, int], dict[int, float]] = {}
    measured = 0
    with torch.inference_mode():
        for eid in ids:
            if eid in anchor_set:
                continue
            hidden_j, target_j, weight_j = layer.experts[eid].stacked()
            if target_j.numel() == 0 or int(target_j.shape[0]) < int(min_samples):
                continue
            for anchor in candidates_by_eid.get(eid, []):
                hidden_a, target_a, weight_a = layer.experts[anchor].stacked()
                if target_a.numel() == 0 or int(target_a.shape[0]) < int(min_samples):
                    continue
                rel, alpha = _measure_synthesized_pair(
                    hidden_a,
                    target_a,
                    weight_a,
                    hidden_j,
                    target_j,
                    weight_j,
                    gate_up_anchor=gate_up[int(anchor)],
                    down_anchor=down[int(anchor)],
                    gate_up_drop=gate_up[int(eid)],
                    down_drop=down[int(eid)],
                    alphas=alphas,
                    act_name=act_name,
                    device=run_device,
                    dtype=compute_dtype,
                )
                residuals[(int(eid), int(anchor))] = rel
                synthesis_weights[(int(eid), int(anchor))] = {
                    int(anchor): 1.0 - alpha,
                    int(eid): alpha,
                }
            measured += 1
            if measured % 16 == 0:
                print(
                    f"entmoot synthesis-fit: {layer.router_qname} measured={measured}",
                    file=sys.stderr,
                    flush=True,
                )

    if run_device.type == "cuda":
        torch.cuda.empty_cache()
    print(
        f"entmoot synthesis-fit: {layer.router_qname} pairs={len(residuals)}",
        file=sys.stderr,
        flush=True,
    )
    return residuals, candidates_by_eid, synthesis_weights


def _activation_synthesis_pair_decoupled_residuals(
    layer: LayerSketchBuffer,
    expert_ids: Sequence[int],
    features: torch.Tensor,
    *,
    target_experts: int,
    model_path: str | Path,
    device: str,
    dtype: torch.dtype,
    top_anchors: int,
    min_samples: int,
    alpha_grid: Sequence[float],
    beta_grid: Sequence[float],
    refine_steps: int = 0,
    refine_radius: float = 0.12,
    refine_shrink: float = 0.5,
    refine_threshold: float | None = None,
) -> tuple[
    dict[tuple[int, int], float],
    dict[int, list[int]],
    dict[tuple[int, int], dict[int, float]],
    dict[tuple[int, int], dict[str, dict[int, float]]],
]:
    """Fit pairwise stock expert rows with separate gate_up/down blends."""

    ids = [int(e) for e in expert_ids]
    G = features.to(torch.float64) @ features.to(torch.float64).T
    anchors = select_rrqr_anchors_from_gram(G, ids, target_experts=target_experts)
    anchor_set = set(anchors)
    id_to_idx = {eid: i for i, eid in enumerate(ids)}
    alphas = tuple(float(a) for a in alpha_grid if 0.0 <= float(a) <= 1.0)
    betas = tuple(float(b) for b in beta_grid if 0.0 <= float(b) <= 1.0)
    if not alphas or not betas:
        raise ValueError("decoupled synthesis grids must contain values in [0, 1]")
    if int(top_anchors) <= 0:
        top_n = len(anchors)
    else:
        top_n = max(1, min(int(top_anchors), len(anchors)))

    candidates_by_eid: dict[int, list[int]] = {}
    for eid in ids:
        if eid in anchor_set:
            continue
        j = id_to_idx[eid]
        scored = []
        for anchor in anchors:
            a = id_to_idx[anchor]
            norm = float(G[j, j].clamp_min(1e-12))
            rel = max(float(G[j, j] + G[a, a] - 2.0 * G[j, a]), 0.0) / norm
            scored.append((rel, anchor))
        candidates_by_eid[eid] = [a for _rel, a in sorted(scored)[:top_n]]

    gate_up, down = _load_packed_expert_tensors(model_path, layer.router_qname)
    act_name = _hidden_act_from_config(model_path)
    run_device = torch.device(device)
    if run_device.type == "cuda" and not torch.cuda.is_available():
        run_device = torch.device("cpu")
    compute_dtype = dtype if run_device.type == "cuda" else torch.float32
    gate_up = gate_up.to(device=run_device, dtype=compute_dtype)
    down = down.to(device=run_device, dtype=compute_dtype)

    residuals: dict[tuple[int, int], float] = {}
    synthesis_weights: dict[tuple[int, int], dict[int, float]] = {}
    tensor_weights: dict[tuple[int, int], dict[str, dict[int, float]]] = {}
    measured = 0
    with torch.inference_mode():
        for eid in ids:
            if eid in anchor_set:
                continue
            hidden_j, target_j, weight_j = layer.experts[eid].stacked()
            if target_j.numel() == 0 or int(target_j.shape[0]) < int(min_samples):
                continue
            for anchor in candidates_by_eid.get(eid, []):
                hidden_a, target_a, weight_a = layer.experts[anchor].stacked()
                if target_a.numel() == 0 or int(target_a.shape[0]) < int(min_samples):
                    continue
                rel, alpha, beta = _measure_synthesized_pair_decoupled(
                    hidden_a,
                    target_a,
                    weight_a,
                    hidden_j,
                    target_j,
                    weight_j,
                    gate_up_anchor=gate_up[int(anchor)],
                    down_anchor=down[int(anchor)],
                    gate_up_drop=gate_up[int(eid)],
                    down_drop=down[int(eid)],
                    alphas=alphas,
                    betas=betas,
                    refine_steps=refine_steps,
                    refine_radius=refine_radius,
                    refine_shrink=refine_shrink,
                    refine_threshold=refine_threshold,
                    act_name=act_name,
                    device=run_device,
                    dtype=compute_dtype,
                )
                residuals[(int(eid), int(anchor))] = rel
                avg = 0.5 * (alpha + beta)
                synthesis_weights[(int(eid), int(anchor))] = {
                    int(anchor): 1.0 - avg,
                    int(eid): avg,
                }
                tensor_weights[(int(eid), int(anchor))] = {
                    "gate_up_proj": {int(anchor): 1.0 - alpha, int(eid): alpha},
                    "down_proj": {int(anchor): 1.0 - beta, int(eid): beta},
                }
            measured += 1
            if measured % 8 == 0:
                print(
                    f"entmoot synthesis-fit-2d: {layer.router_qname} measured={measured}",
                    file=sys.stderr,
                    flush=True,
                )

    if run_device.type == "cuda":
        torch.cuda.empty_cache()
    print(
        f"entmoot synthesis-fit-2d: {layer.router_qname} pairs={len(residuals)}",
        file=sys.stderr,
        flush=True,
    )
    return residuals, candidates_by_eid, synthesis_weights, tensor_weights


def _measure_synthesized_pair(
    hidden_anchor: torch.Tensor,
    target_anchor: torch.Tensor,
    weight_anchor: torch.Tensor,
    hidden_drop: torch.Tensor,
    target_drop: torch.Tensor,
    weight_drop: torch.Tensor,
    *,
    gate_up_anchor: torch.Tensor,
    down_anchor: torch.Tensor,
    gate_up_drop: torch.Tensor,
    down_drop: torch.Tensor,
    alphas: Sequence[float],
    act_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, float]:
    h_anchor = hidden_anchor.to(device=device, dtype=dtype)
    y_anchor = target_anchor.to(device=device, dtype=torch.float32)
    w_anchor = weight_anchor.to(device=device, dtype=torch.float32).clamp_min(0.0)
    h_drop = hidden_drop.to(device=device, dtype=dtype)
    y_drop = target_drop.to(device=device, dtype=torch.float32)
    w_drop = weight_drop.to(device=device, dtype=torch.float32).clamp_min(0.0)
    if float(w_anchor.sum().item()) <= 0.0:
        w_anchor = torch.ones_like(w_anchor)
    if float(w_drop.sum().item()) <= 0.0:
        w_drop = torch.ones_like(w_drop)

    alpha_t = torch.tensor(list(alphas), device=device, dtype=torch.float32)
    shape = (len(alpha_t),) + (1,) * gate_up_anchor.ndim
    alpha_w = alpha_t.reshape(shape)
    gate_up = (
        (1.0 - alpha_w) * gate_up_anchor.float().unsqueeze(0)
        + alpha_w * gate_up_drop.float().unsqueeze(0)
    ).to(dtype=dtype)
    down = (
        (1.0 - alpha_w) * down_anchor.float().unsqueeze(0)
        + alpha_w * down_drop.float().unsqueeze(0)
    ).to(dtype=dtype)

    pred_anchor = _packed_expert_forward_candidates(
        h_anchor, gate_up, down, act_name=act_name,
    ).to(torch.float32)
    pred_drop = _packed_expert_forward_candidates(
        h_drop, gate_up, down, act_name=act_name,
    ).to(torch.float32)
    err_anchor = (pred_anchor - y_anchor.unsqueeze(0)).pow(2).sum(dim=-1)
    err_drop = (pred_drop - y_drop.unsqueeze(0)).pow(2).sum(dim=-1)
    denom_anchor = y_anchor.pow(2).sum(dim=-1).clamp_min(1e-12)
    denom_drop = y_drop.pow(2).sum(dim=-1).clamp_min(1e-12)
    num = (err_anchor * w_anchor.unsqueeze(0)).sum(dim=1)
    num = num + (err_drop * w_drop.unsqueeze(0)).sum(dim=1)
    den = (denom_anchor * w_anchor).sum() + (denom_drop * w_drop).sum()
    rel = num / den.clamp_min(1e-12)
    best = int(rel.argmin().item())
    return float(rel[best].item()), float(alpha_t[best].item())


def _measure_synthesized_pair_decoupled(
    hidden_anchor: torch.Tensor,
    target_anchor: torch.Tensor,
    weight_anchor: torch.Tensor,
    hidden_drop: torch.Tensor,
    target_drop: torch.Tensor,
    weight_drop: torch.Tensor,
    *,
    gate_up_anchor: torch.Tensor,
    down_anchor: torch.Tensor,
    gate_up_drop: torch.Tensor,
    down_drop: torch.Tensor,
    alphas: Sequence[float],
    betas: Sequence[float],
    refine_steps: int = 0,
    refine_radius: float = 0.12,
    refine_shrink: float = 0.5,
    refine_threshold: float | None = None,
    act_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, float, float]:
    h_anchor = hidden_anchor.to(device=device, dtype=dtype)
    y_anchor = target_anchor.to(device=device, dtype=torch.float32)
    w_anchor = weight_anchor.to(device=device, dtype=torch.float32).clamp_min(0.0)
    h_drop = hidden_drop.to(device=device, dtype=dtype)
    y_drop = target_drop.to(device=device, dtype=torch.float32)
    w_drop = weight_drop.to(device=device, dtype=torch.float32).clamp_min(0.0)
    if float(w_anchor.sum().item()) <= 0.0:
        w_anchor = torch.ones_like(w_anchor)
    if float(w_drop.sum().item()) <= 0.0:
        w_drop = torch.ones_like(w_drop)

    denom_anchor = y_anchor.pow(2).sum(dim=-1).clamp_min(1e-12)
    denom_drop = y_drop.pow(2).sum(dim=-1).clamp_min(1e-12)
    den = (denom_anchor * w_anchor).sum() + (denom_drop * w_drop).sum()
    den = den.clamp_min(1e-12)

    def evaluate(pairs: Sequence[tuple[float, float]]) -> tuple[float, float, float]:
        alpha_t = torch.tensor(
            [p[0] for p in pairs], device=device, dtype=torch.float32,
        )
        beta_t = torch.tensor(
            [p[1] for p in pairs], device=device, dtype=torch.float32,
        )
        shape = (len(pairs),) + (1,) * gate_up_anchor.ndim
        alpha_w = alpha_t.reshape(shape)
        beta_w = beta_t.reshape(shape)
        gate_up = (
            (1.0 - alpha_w) * gate_up_anchor.float().unsqueeze(0)
            + alpha_w * gate_up_drop.float().unsqueeze(0)
        ).to(dtype=dtype)
        down = (
            (1.0 - beta_w) * down_anchor.float().unsqueeze(0)
            + beta_w * down_drop.float().unsqueeze(0)
        ).to(dtype=dtype)

        pred_anchor = _packed_expert_forward_candidates(
            h_anchor, gate_up, down, act_name=act_name,
        ).to(torch.float32)
        pred_drop = _packed_expert_forward_candidates(
            h_drop, gate_up, down, act_name=act_name,
        ).to(torch.float32)
        err_anchor = (pred_anchor - y_anchor.unsqueeze(0)).pow(2).sum(dim=-1)
        err_drop = (pred_drop - y_drop.unsqueeze(0)).pow(2).sum(dim=-1)
        num = (err_anchor * w_anchor.unsqueeze(0)).sum(dim=1)
        num = num + (err_drop * w_drop.unsqueeze(0)).sum(dim=1)
        rel = num / den
        best_idx = int(rel.argmin().item())
        return (
            float(rel[best_idx].item()),
            float(alpha_t[best_idx].item()),
            float(beta_t[best_idx].item()),
        )

    coarse_pairs = [(float(a), float(b)) for a in alphas for b in betas]
    best_rel, best_alpha, best_beta = evaluate(coarse_pairs)
    if refine_threshold is not None and best_rel > float(refine_threshold):
        return best_rel, best_alpha, best_beta
    radius = max(float(refine_radius), 0.0)
    shrink = min(max(float(refine_shrink), 0.05), 0.95)
    for _step in range(max(0, int(refine_steps))):
        if radius <= 0.0:
            break
        local_alphas = _local_unit_grid(best_alpha, radius)
        local_betas = _local_unit_grid(best_beta, radius)
        rel, alpha, beta = evaluate([
            (a, b) for a in local_alphas for b in local_betas
        ])
        if rel < best_rel:
            best_rel, best_alpha, best_beta = rel, alpha, beta
        radius *= shrink
    return best_rel, best_alpha, best_beta


def _local_unit_grid(center: float, radius: float) -> tuple[float, ...]:
    values = [
        center - radius,
        center - 0.5 * radius,
        center,
        center + 0.5 * radius,
        center + radius,
    ]
    return tuple(sorted({
        min(1.0, max(0.0, round(float(v), 8))) for v in values
    }))


def _packed_expert_forward_candidates(
    hidden: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    *,
    act_name: str,
) -> torch.Tensor:
    gate_up_out = torch.einsum("nd,cid->cni", hidden, gate_up)
    gate, up = gate_up_out.chunk(2, dim=-1)
    inter = _apply_activation(gate, act_name) * up
    return torch.einsum("cni,cdi->cnd", inter, down)


def _apply_activation(x: torch.Tensor, act_name: str) -> torch.Tensor:
    if act_name in {"silu", "swish"}:
        return F.silu(x)
    if act_name == "gelu":
        return F.gelu(x)
    if act_name == "relu":
        return F.relu(x)
    raise ValueError(f"unsupported activation for Entmoot fit: {act_name!r}")


def _load_packed_expert_tensors(
    model_path: str | Path,
    router_qname: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    gate_up = _load_model_tensor(
        model_path,
        _packed_expert_weight_candidates(router_qname, "gate_up_proj"),
    )
    down = _load_model_tensor(
        model_path,
        _packed_expert_weight_candidates(router_qname, "down_proj"),
    )
    return gate_up, down


def _load_model_tensor(model_path: str | Path, candidates: Sequence[str]) -> torch.Tensor:
    from safetensors.torch import safe_open

    src = Path(model_path)
    idx_path = src / "model.safetensors.index.json"
    if idx_path.exists():
        with idx_path.open() as f:
            weight_map = json.load(f)["weight_map"]
        for key in candidates:
            rel = weight_map.get(key)
            if rel is None:
                continue
            with safe_open(str(src / rel), framework="pt", device="cpu") as sf:
                return sf.get_tensor(key)
    for f in sorted(src.glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            keys = set(sf.keys())
            for key in candidates:
                if key in keys:
                    return sf.get_tensor(key)
    raise KeyError(f"could not find any tensor candidate: {list(candidates)}")


def _packed_expert_weight_candidates(router_qname: str, leaf: str) -> list[str]:
    parent = router_qname[:-len(".gate")] if router_qname.endswith(".gate") else router_qname
    if parent.endswith(".router"):
        parent = parent[:-len(".router")]
    base = parent + ".experts." + leaf
    names = [base]
    if base.startswith("model.layers."):
        names.append("model.language_model." + base[len("model."):])
    if base.startswith("language_model.model.layers."):
        names.append("model.language_model." + base[len("language_model.model."):])
    return list(dict.fromkeys(names))


def _hidden_act_from_config(model_path: str | Path) -> str:
    cfg_path = Path(model_path) / "config.json"
    with cfg_path.open() as f:
        cfg = json.load(f)
    text = cfg.get("text_config") or {}
    for source in (text, cfg):
        value = source.get("hidden_act")
        if isinstance(value, str) and value:
            return value
    return "silu"


def summarize_collector(path: str | Path) -> dict[str, Any]:
    layers = load_collector_state(path)
    return {
        "format": "entmoot_collector_summary_v1",
        "path": str(path),
        "n_layers": len(layers),
        "layers": {
            router: _layer_summary(layer)
            for router, layer in sorted(layers.items())
        },
    }


def summarize_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    layers = {}
    for router, entry in sorted(payload.items()):
        decisions = entry.get("expert_decisions", [])
        n_accept = sum(1 for d in decisions if d.get("decision") == "accept")
        n_reject = sum(1 for d in decisions if d.get("decision") == "reject")
        layers[router] = {
            "method": entry.get("method"),
            "router_strategy": entry.get("router_strategy"),
            "num_experts_orig": int(entry["num_experts_orig"]),
            "num_experts_kept": int(entry["num_experts_kept"]),
            "accepted_merges": n_accept,
            "rejected_identity": n_reject,
        }
    return {
        "format": "entmoot_manifest_summary_v1",
        "path": str(path),
        "n_layers": len(layers),
        "layers": layers,
    }


def choose_router_strategies(
    collector_path: str | Path,
    manifest_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    *,
    top_k: int | None = None,
    routers: Sequence[str] | None = None,
    max_hidden_rows: int = 4096,
    kl_cap: float = 0.05,
    top1_floor: float = 0.95,
    topk_floor: float = 0.90,
) -> dict[str, Any]:
    """Resolve `router_strategy=auto` by measuring stock router rows."""

    layers = load_collector_state(collector_path)
    manifest = json.loads(Path(manifest_path).read_text())
    router_filter = set(routers or [])
    k = int(top_k) if top_k is not None else _read_top_k_from_config(model_path)
    summaries = {}
    for router, entry in sorted(manifest.items()):
        if router_filter and router not in router_filter:
            continue
        layer = layers.get(router)
        if layer is None:
            continue
        hidden = _hidden_rows_for_layer(layer, max_rows=max_hidden_rows)
        if hidden.numel() == 0:
            continue
        weight = _load_router_weight(model_path, router)
        choice = choose_router_strategy(
            hidden,
            weight,
            entry,
            top_k=k,
            kl_cap=kl_cap,
            top1_floor=top1_floor,
            topk_floor=topk_floor,
        )
        entry["router_strategy"] = choice.selected_strategy
        diag = dict(entry.get("diagnostics") or {})
        diag["router_strategy_choice"] = choice.to_dict()
        entry["diagnostics"] = diag
        summaries[router] = choice.to_dict()

    Path(output_path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "format": "entmoot_choose_router_strategies_v1",
        "collector": str(collector_path),
        "manifest": str(manifest_path),
        "model": str(model_path),
        "output": str(output_path),
        "top_k": k,
        "routers": summaries,
    }


def _layer_summary(layer: LayerSketchBuffer) -> dict[str, Any]:
    stats = layer.stats()
    sampled = [s.samples for s in stats]
    seen = [s.seen for s in stats]
    return {
        "num_experts": layer.num_experts,
        "total_tokens": layer.total_tokens,
        "experts_with_samples": sum(1 for x in sampled if x > 0),
        "min_samples": min(sampled) if sampled else 0,
        "median_samples": _median(sampled),
        "max_samples": max(sampled) if sampled else 0,
        "min_seen": min(seen) if seen else 0,
        "median_seen": _median(seen),
        "max_seen": max(seen) if seen else 0,
    }


def _hidden_rows_for_layer(layer: LayerSketchBuffer, *, max_rows: int) -> torch.Tensor:
    rows = []
    for eid in range(layer.num_experts):
        hidden, _output, _weight = layer.experts[eid].stacked()
        if hidden.numel() > 0:
            rows.append(hidden)
    if not rows:
        return torch.empty(0, 0, dtype=torch.float32)
    X = torch.cat(rows, dim=0)
    if X.shape[0] > int(max_rows):
        X = X[: int(max_rows)]
    return X


def _read_top_k_from_config(model_path: str | Path) -> int:
    cfg_path = Path(model_path) / "config.json"
    with cfg_path.open() as f:
        cfg = json.load(f)
    text = cfg.get("text_config") or {}
    for source in (text, cfg):
        for key in ("num_experts_per_tok", "moe_top_k", "num_active_experts"):
            value = source.get(key)
            if isinstance(value, int) and value > 0:
                return value
    return 2


def _load_router_weight(model_path: str | Path, router_qname: str) -> torch.Tensor:
    from safetensors.torch import safe_open

    src = Path(model_path)
    candidates = _router_weight_candidates(router_qname)
    idx_path = src / "model.safetensors.index.json"
    if idx_path.exists():
        with idx_path.open() as f:
            weight_map = json.load(f)["weight_map"]
        for key in candidates:
            rel = weight_map.get(key)
            if rel is None:
                continue
            with safe_open(str(src / rel), framework="pt", device="cpu") as sf:
                return sf.get_tensor(key)
    for f in sorted(src.glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            keys = set(sf.keys())
            for key in candidates:
                if key in keys:
                    return sf.get_tensor(key)
    raise KeyError(f"could not find router weight for {router_qname}")


def _router_weight_candidates(router_qname: str) -> list[str]:
    base = router_qname[:-len(".weight")] if router_qname.endswith(".weight") else router_qname
    names = [base + ".weight"]
    if base.startswith("model.layers."):
        names.append("model.language_model." + base[len("model."):] + ".weight")
    if base.startswith("language_model.model."):
        names.append("model.language_model." + base[len("language_model.model."):] + ".weight")
    if base.startswith("model.language_model."):
        names.append(base + ".weight")
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(names))


def _target_for_layer(
    layer: LayerSketchBuffer,
    *,
    target_experts: int | None,
    keep_ratio: float | None,
) -> int:
    if target_experts is not None:
        k = int(target_experts)
    elif keep_ratio is not None:
        if keep_ratio <= 0.0 or keep_ratio > 1.0:
            raise ValueError("keep_ratio must be in (0, 1]")
        k = max(1, int(round(layer.num_experts * keep_ratio)))
    else:
        raise ValueError("one of target_experts or keep_ratio is required")
    return max(1, min(k, layer.num_experts))


def _median(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    xs = sorted(int(v) for v in values)
    n = len(xs)
    mid = n // 2
    if n % 2:
        return float(xs[mid])
    return 0.5 * float(xs[mid - 1] + xs[mid])


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unknown dtype {name!r}")


def _parse_float_list(text: str) -> list[float]:
    values = []
    for part in str(text).split(","):
        item = part.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("expected at least one float")
    return values


def _print_or_write(payload: Mapping[str, Any], path: str | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path:
        Path(path).write_text(text)
    else:
        print(text, end="")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Entmoot MoE merge tools")
    sub = parser.add_subparsers(dest="cmd", required=True)

    collect = sub.add_parser("collect-jsonl", help="collect activation sketches")
    collect.add_argument("--model", required=True)
    collect.add_argument("--dataset", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--summary", default=None)
    collect.add_argument("--device", default="cuda")
    collect.add_argument("--device-map", default=None)
    collect.add_argument("--offload-folder", default=None)
    collect.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    collect.add_argument("--max-length", type=int, default=2048)
    collect.add_argument("--limit", type=int, default=None)
    collect.add_argument("--max-samples-per-expert", type=int, default=256)
    collect.add_argument("--router-regex", default=None)
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument(
        "--forward-mode",
        choices=["backbone", "causal-lm"],
        default="backbone",
        help="use decoder backbone by default to avoid final logits",
    )
    collect.add_argument(
        "--stop-after-layer",
        type=int,
        default=None,
        help="temporarily run decoder layers only through this inclusive index",
    )
    collect.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="write collection progress every N used rows; 0 disables progress",
    )

    plan = sub.add_parser("plan-from-collector", help="build merge manifest")
    plan.add_argument("--collector", required=True)
    plan.add_argument("--output", required=True)
    group = plan.add_mutually_exclusive_group(required=True)
    group.add_argument("--target-experts", type=int)
    group.add_argument("--keep-ratio", type=float)
    plan.add_argument("--router", action="append", default=None)
    plan.add_argument("--activation-accept-threshold", type=float, default=0.05)
    plan.add_argument("--activation-tentative-threshold", type=float, default=0.10)
    plan.add_argument("--min-routed-mass", type=float, default=0.0)
    plan.add_argument("--min-samples", type=int, default=128)
    plan.add_argument(
        "--router-strategy",
        choices=["anchor", "weighted_average"],
        default="anchor",
    )
    plan.add_argument("--normalize-features", action="store_true")
    plan.add_argument(
        "--activation-fit-model",
        default=None,
        help="source model path for same-input expert/anchor residual measurement",
    )
    plan.add_argument(
        "--activation-fit-mode",
        choices=["anchor", "synthesis_pair", "synthesis_pair_decoupled"],
        default="anchor",
        help="measure hard-anchor substitution or pairwise synthesized stock experts",
    )
    plan.add_argument("--activation-fit-device", default="cuda")
    plan.add_argument(
        "--activation-fit-dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
    )
    plan.add_argument("--activation-fit-top-anchors", type=int, default=4)
    plan.add_argument(
        "--synthesis-alpha-grid",
        default=None,
        help="comma-separated alpha values for synthesis_pair mode",
    )
    plan.add_argument(
        "--synthesis-beta-grid",
        default=None,
        help="comma-separated beta values for synthesis_pair_decoupled mode",
    )
    plan.add_argument(
        "--synthesis-refine-steps",
        type=int,
        default=0,
        help="local alpha/beta refinement steps after the decoupled coarse grid",
    )
    plan.add_argument(
        "--synthesis-refine-radius",
        type=float,
        default=0.12,
        help="initial local refinement radius in alpha/beta space",
    )
    plan.add_argument(
        "--synthesis-refine-shrink",
        type=float,
        default=0.5,
        help="radius multiplier after each local refinement step",
    )
    plan.add_argument(
        "--synthesis-refine-threshold",
        type=float,
        default=None,
        help="only refine coarse decoupled candidates at or below this residual",
    )
    plan.add_argument("--summary", default=None)

    sc = sub.add_parser("summarize-collector", help="summarize collector artifact")
    sc.add_argument("--collector", required=True)
    sc.add_argument("--summary", default=None)

    sm = sub.add_parser("summarize-manifest", help="summarize merge manifest")
    sm.add_argument("--manifest", required=True)
    sm.add_argument("--summary", default=None)

    cr = sub.add_parser(
        "choose-router-strategy",
        help="measure anchor vs weighted_average router rows and update manifest",
    )
    cr.add_argument("--collector", required=True)
    cr.add_argument("--manifest", required=True)
    cr.add_argument("--model", required=True)
    cr.add_argument("--output", required=True)
    cr.add_argument("--top-k", type=int, default=None)
    cr.add_argument("--router", action="append", default=None)
    cr.add_argument("--max-hidden-rows", type=int, default=4096)
    cr.add_argument("--kl-cap", type=float, default=0.05)
    cr.add_argument("--top1-floor", type=float, default=0.95)
    cr.add_argument("--topk-floor", type=float, default=0.90)
    cr.add_argument("--summary", default=None)

    args = parser.parse_args(argv)

    if args.cmd == "collect-jsonl":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = _dtype_from_name(args.dtype)
        staged = stage_text_only(args.model)
        tokenizer = AutoTokenizer.from_pretrained(staged, trust_remote_code=True)
        load_device_map = args.device_map if args.device_map is not None else args.device
        kwargs = {
            "torch_dtype": dtype,
            "device_map": load_device_map,
            "low_cpu_mem_usage": False,
            "trust_remote_code": True,
        }
        if args.offload_folder:
            Path(args.offload_folder).mkdir(parents=True, exist_ok=True)
            kwargs["offload_folder"] = args.offload_folder
            kwargs["offload_buffers"] = True
            kwargs.pop("low_cpu_mem_usage", None)
        model = AutoModelForCausalLM.from_pretrained(staged, **kwargs)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        device = resolve_execution_device(model, args.device)
        # Force config access early; this catches staged dense/non-MoE mistakes.
        read_top_k(model, default=2)
        summary = collect_jsonl(
            model,
            tokenizer,
            args.dataset,
            args.output,
            device=device,
            max_length=args.max_length,
            limit=args.limit,
            max_samples_per_expert=args.max_samples_per_expert,
            router_regex=args.router_regex,
            seed=args.seed,
            forward_mode=args.forward_mode,
            stop_after_layer=args.stop_after_layer,
            progress_every=args.progress_every,
        )
        _print_or_write(summary, args.summary)
        return 0

    if args.cmd == "plan-from-collector":
        summary = plan_from_collector(
            args.collector,
            args.output,
            target_experts=args.target_experts,
            keep_ratio=args.keep_ratio,
            routers=args.router,
            activation_accept_threshold=args.activation_accept_threshold,
            activation_tentative_threshold=args.activation_tentative_threshold,
            min_routed_mass=args.min_routed_mass,
            min_samples=args.min_samples,
            router_strategy=args.router_strategy,
            normalize_features=args.normalize_features,
            activation_fit_model=args.activation_fit_model,
            activation_fit_device=args.activation_fit_device,
            activation_fit_dtype=args.activation_fit_dtype,
            activation_fit_top_anchors=args.activation_fit_top_anchors,
            activation_fit_mode=args.activation_fit_mode,
            synthesis_alpha_grid=(
                None if args.synthesis_alpha_grid is None
                else _parse_float_list(args.synthesis_alpha_grid)
            ),
            synthesis_beta_grid=(
                None if args.synthesis_beta_grid is None
                else _parse_float_list(args.synthesis_beta_grid)
            ),
            synthesis_refine_steps=args.synthesis_refine_steps,
            synthesis_refine_radius=args.synthesis_refine_radius,
            synthesis_refine_shrink=args.synthesis_refine_shrink,
            synthesis_refine_threshold=args.synthesis_refine_threshold,
        )
        _print_or_write(summary, args.summary)
        return 0

    if args.cmd == "summarize-collector":
        _print_or_write(summarize_collector(args.collector), args.summary)
        return 0

    if args.cmd == "summarize-manifest":
        _print_or_write(summarize_manifest(args.manifest), args.summary)
        return 0

    if args.cmd == "choose-router-strategy":
        summary = choose_router_strategies(
            args.collector,
            args.manifest,
            args.model,
            args.output,
            top_k=args.top_k,
            routers=args.router,
            max_hidden_rows=args.max_hidden_rows,
            kl_cap=args.kl_cap,
            top1_floor=args.top1_floor,
            topk_floor=args.topk_floor,
        )
        _print_or_write(summary, args.summary)
        return 0

    raise AssertionError(f"unhandled subcommand {args.cmd!r}")


if __name__ == "__main__":
    raise SystemExit(main())
