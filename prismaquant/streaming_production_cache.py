"""Streaming per-layer production weight-cache fill for very large models.

Renders GPTQ/JSO production weights for a checkpoint too large to load whole
(e.g. Tencent Hy3, 295B, 192 experts/layer). Mirrors the shard-by-shard
architecture of ``incremental_measure_quant_cost``: one ``StreamingContext``
with the head (embed + rotary + lm_head) resident and every decoder layer
offloaded to disk / on meta. Each layer is installed on demand, its assigned
non-BF16 dense Linears and packed experts are rendered from the probe's
activation cache, then the layer is unloaded. Only one decoder layer is
resident at a time, so peak memory is head + one layer + working set.

The rendered ``ProductionWeightCache`` is byte-identical in SEMANTICS to the
resident ``fill_production_weight_cache`` path: same ``(qname, fmt)`` keys, the
same ``activation_max_abs`` / ``packed_expert_coverage`` / ``levers`` metadata,
and the same coverage contract — export / recache consume it unchanged.

Unlike the resident path there is no whole-model forward pass here: dense
activation rows come from the probe's per-Linear activation cache (keyed by the
canonical Linear name) and each packed-experts module's input snapshot comes
from the same cache (keyed by the experts-module name). Routing is recomputed
offline from that snapshot + the resident gate weight, exactly as the resident
packed render does.
"""
from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

# Must be set before the cuda allocator initializes — matches the streaming
# cost path so the caching allocator doesn't hoard freed blocks on the GB10
# unified-memory pool.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn

from prismaquant import format_registry as fr
from prismaquant.build_rtn_cache import iter_quantizable_tensors
from prismaquant.measure_quant_cost import (
    ActivationIndex,
    canonical_linear_name,
)
from prismaquant.production_weight_cache import (
    ProductionWeightCache,
    _FisherRowWeightCache,
    _fused_sibling_leaf_mapping_from_profile,
    _render_base_format,
    _render_score_record,
    _render_score_record_key,
    _resolve_production_render_levers,
    _resolve_render_mechanism_plan,
    _store_rendered_weight_entry,
    fill_packed_expert_cache_entries,
    render_production_weight,
)
from prismaquant.streaming_model import _build_streaming_context


def _canon_fmt(fmt: str) -> str:
    return fr.canonical_format_name(str(fmt).strip().upper())


def _layer_index_of(qname: str, layers_prefix: str) -> int | None:
    """Return the decoder-layer index a qname lives under, or None (head)."""
    if not qname.startswith(layers_prefix):
        return None
    rest = qname[len(layers_prefix):]
    head = rest.split(".", 1)[0]
    return int(head) if head.isdigit() else None


def _nonbf16_assignment(render_assignment: Mapping[str, str]) -> dict[str, str]:
    """Canonicalize and drop BF16 entries (dense keys are live qnames)."""
    out: dict[str, str] = {}
    for qname, fmt in render_assignment.items():
        fmt_canon = _canon_fmt(fmt)
        if fmt_canon == "BF16":
            continue
        out[str(qname)] = fmt_canon
    return out


def _eligible_dense_qname_modules(
    model: nn.Module, profile, skip_tokens: Sequence[str],
) -> dict[str, nn.Module]:
    """Map live qname -> module for every eligible dense Linear (resident
    selection semantics: same ``iter_quantizable_tensors`` + pinned-skip
    filter that ``build_production_cache`` applies)."""
    out: dict[str, nn.Module] = {}
    skip = set(skip_tokens)
    for full_name, mod, attr in iter_quantizable_tensors(model, profile):
        if attr != "weight" or not isinstance(mod, nn.Linear):
            continue
        qname = full_name[:-7] if full_name.endswith(".weight") else full_name
        if any(s in qname.split(".") for s in skip):
            continue
        out[qname] = mod
    return out


def _render_dense_layer(
    model: nn.Module,
    layer_dense_modules: Mapping[str, nn.Module],
    *,
    assignment_nonbf16: Mapping[str, str],
    act_index: ActivationIndex,
    cache: ProductionWeightCache,
    levers: Mapping[str, object],
    cache_dir_path: Path | None,
    profile,
    device: torch.device,
    fisher_rows: _FisherRowWeightCache | None,
    render_score_records: dict[str, dict[str, object]],
    progress: bool,
) -> int:
    """Render this layer's assigned non-BF16 dense Linears from the act cache.

    Reproduces ``fill_production_weight_cache``'s per-Linear render exactly:
    joint fused-sibling NVFP4 globals (siblings are resident in the same layer),
    fused-group-unified calibrated max_abs, then
    ``render_production_weight`` per (qname, fmt) stored via the shared
    ``_store_rendered_weight_entry``.
    """
    from prismaquant.decision_units import fused_group_key
    from prismaquant.export_native_compressed import _compute_nvfp4_joint_global

    render_formats_by_qname: dict[str, str] = {}
    for qname in layer_dense_modules:
        fmt = assignment_nonbf16.get(qname)
        if fmt is not None:
            render_formats_by_qname[qname] = fmt
    qname_to_module = {
        q: layer_dense_modules[q] for q in render_formats_by_qname
    }
    if not qname_to_module:
        return 0

    render_base_fmts = {
        _render_base_format(f) for f in render_formats_by_qname.values()
    }
    needs_nvfp4 = "NVFP4" in render_base_fmts

    # Joint fused-sibling NVFP4 global (max across q/k/v or gate/up). Restricted
    # to this layer's non-BF16 dense qnames; siblings are co-resident. The
    # synthetic all-NVFP4 assignment matches the resident derivation.
    joint_globals: dict[str, torch.Tensor] = {}
    if needs_nvfp4:
        joint_globals = _compute_nvfp4_joint_global(
            model,
            {q: "NVFP4" for q in qname_to_module},
            profile=profile,
        )

    # Per-Linear calibrated max_abs, unified (max) across fused sibling groups
    # — reproduces the resident block; the value drives only the export
    # activation scale (metadata), never the rendered weight.
    if needs_nvfp4:
        per_qname_max_abs: dict[str, float] = {}
        for qname in qname_to_module:
            canonical = canonical_linear_name(qname, profile)
            if canonical not in act_index:
                continue
            X, _ = act_index.load_with_row_indices(canonical)
            mx = float(X.abs().max().item())
            if mx > 0:
                per_qname_max_abs[qname] = mx
        groups: dict[str, list[str]] = defaultdict(list)
        for qname in per_qname_max_abs:
            gk = (
                fused_group_key(profile, qname)
                if profile is not None else qname
            )
            groups[gk].append(qname)
        for members in groups.values():
            shared = max(per_qname_max_abs[m] for m in members)
            for m in members:
                cache.activation_max_abs[m] = shared

    rendered = 0
    for qname, mod in qname_to_module.items():
        weight = mod.weight.data
        canonical = canonical_linear_name(qname, profile)
        if canonical not in act_index:
            raise RuntimeError(
                "[stream-prod-cache] no cached activations for dense Linear "
                f"{qname} (canonical={canonical}); the probe activation cache "
                "must cover every non-BF16 assignment entry — streaming render "
                "cannot fabricate the GPTQ Hessian."
            )
        X_cpu, _ = act_index.load_with_row_indices(canonical)
        # Activation-residency landmine: act-cache tensors are CPU-resident;
        # move to the compute device + fp32 explicitly or GPTQ silently runs
        # on CPU (and diverges from the resident render dtype).
        X = X_cpu.to(device=device, dtype=torch.float32)
        activations = {qname: X}
        joint = joint_globals.get(qname)
        max_abs = cache.activation_max_abs.get(qname)
        # Export input_global_scale convention (6.0 / max_abs); metadata-only,
        # does not affect the rendered weight — kept identical to the resident
        # render loop.
        export_scale = (
            (6.0 / max_abs) if (max_abs is not None and max_abs > 0) else None
        )
        row_weights = (
            fisher_rows.get(qname)
            if (fisher_rows is not None
                and bool(levers.get("fisher_gptq", False)))
            else None
        )
        fmt = render_formats_by_qname[qname]
        render_fmt = _render_base_format(fmt)
        gate_trace: list[dict[str, object]] = []
        w_dq = render_production_weight(
            weight, render_fmt,
            qname=qname,
            activations=activations,
            levers=levers,
            joint_global_real=joint,
            input_global_scale=export_scale,
            fisher_row_weights=row_weights,
            gate_trace=gate_trace,
        )
        render_score_records[_render_score_record_key(qname, fmt)] = (
            _render_score_record(
                qname=qname,
                fmt=fmt,
                render_format=render_fmt,
                reference_weight=weight,
                rendered_weight=w_dq,
                activations=X,
                activation_max_abs=max_abs,
            )
        )
        _store_rendered_weight_entry(
            weights=cache.weights,
            cache_dir_path=cache_dir_path,
            qname=qname,
            fmt=fmt,
            tensor=w_dq,
            weight_dtype=weight.dtype,
        )
        rendered += 1
        del w_dq, X
        activations.clear()
    if progress and rendered:
        print(f"[stream-prod-cache] rendered {rendered} dense Linears",
              flush=True)
    return rendered


def _render_packed_layer(
    model: nn.Module,
    layer_experts_qnames: Sequence[str],
    *,
    act_index: ActivationIndex,
    cache: ProductionWeightCache,
    render_assignment: Mapping[str, str],
    levers: Mapping[str, object],
    cache_dir_path: Path | None,
    profile,
    module_token_budget: int,
    max_rows_per_expert: int,
    render_mode: str,
    progress: bool,
) -> dict:
    """Render this layer's packed experts via the shared packed-expert path,
    feeding each experts-module's input snapshot from the probe act cache."""
    module_acts: dict[str, torch.Tensor] = {}
    for experts_qname in layer_experts_qnames:
        if experts_qname not in act_index:
            continue
        module_acts[experts_qname] = act_index.load(experts_qname)
    if not module_acts:
        return {}
    return fill_packed_expert_cache_entries(
        cache, model, None,
        render_assignment=render_assignment,
        levers=levers,
        profile=profile,
        module_token_budget=module_token_budget,
        max_rows_per_expert=max_rows_per_expert,
        cache_dir=cache_dir_path,
        render_mode=render_mode,
        module_acts_override=module_acts,
        progress=progress,
    )


def _experts_qnames_by_layer(
    model: nn.Module, profile, layers_prefix: str, num_layers: int,
) -> dict[int | None, list[str]]:
    """Group packed-experts module names by decoder-layer index (structure
    only — safe while layers are on meta)."""
    from prismaquant.sensitivity_probe import _is_packed_experts_module

    out: dict[int | None, list[str]] = defaultdict(list)
    for name, mod in model.named_modules():
        if not _is_packed_experts_module(mod, profile):
            continue
        out[_layer_index_of(name, layers_prefix)].append(name)
    return out


def run_streaming_render(
    model: nn.Module,
    *,
    layers_prefix: str,
    num_layers: int,
    render_assignment: Mapping[str, str],
    act_index: ActivationIndex,
    formats: Sequence[str],
    levers: Mapping[str, object],
    cache_dir_path: Path | None,
    profile,
    skip_tokens: Sequence[str],
    device: torch.device,
    expert_render_mode: str = "batched",
    expert_module_token_budget: int = 32768,
    max_rows_per_expert: int = 2048,
    h_detail_dir: str | Path | None = None,
    install=None,
    unload=None,
    set_priority=None,
    progress: bool = True,
) -> ProductionWeightCache:
    """Layer-by-layer render loop over a (streaming or resident) model.

    ``install``/``unload``/``set_priority`` are the ``StreamingContext`` hooks
    for a real streamed checkpoint. When ``None`` (an already-resident model,
    used by the tests) the loop just renders each layer in place — the render
    math is identical, only weight residency differs.
    """
    levers = _resolve_production_render_levers(levers)
    mechanism_plan = _resolve_render_mechanism_plan(levers)
    assignment_nonbf16 = _nonbf16_assignment(render_assignment)

    cache = ProductionWeightCache(
        weights={},
        levers=dict(levers),
        activation_max_abs={},
        failed={},
        cache_dir=str(cache_dir_path) if cache_dir_path is not None else None,
        metadata={},
    )

    fused_mapping = (
        _fused_sibling_leaf_mapping_from_profile(profile)
        if profile is not None else {}
    )
    fisher_rows = (
        _FisherRowWeightCache(h_detail_dir, fused_mapping or None)
        if (bool(levers.get("fisher_gptq", False)) and h_detail_dir)
        else None
    )

    dense_modules = _eligible_dense_qname_modules(model, profile, skip_tokens)
    per_layer_dense: dict[int | None, dict[str, nn.Module]] = defaultdict(dict)
    for qname, mod in dense_modules.items():
        per_layer_dense[_layer_index_of(qname, layers_prefix)][qname] = mod
    per_layer_experts = _experts_qnames_by_layer(
        model, profile, layers_prefix, num_layers,
    )

    render_score_records: dict[str, dict[str, object]] = {}
    coverage: dict[str, dict[str, object]] = {}

    def _process_layer(L: int | None) -> None:
        dense = per_layer_dense.get(L, {})
        experts = per_layer_experts.get(L, [])
        if not dense and not experts:
            return
        did_install = False
        if L is not None and install is not None:
            if set_priority is not None:
                set_priority({L})
            install(L)
            did_install = True
        try:
            _render_dense_layer(
                model, dense,
                assignment_nonbf16=assignment_nonbf16,
                act_index=act_index,
                cache=cache,
                levers=levers,
                cache_dir_path=cache_dir_path,
                profile=profile,
                device=device,
                fisher_rows=fisher_rows,
                render_score_records=render_score_records,
                progress=progress,
            )
            if experts:
                cov = _render_packed_layer(
                    model, experts,
                    act_index=act_index,
                    cache=cache,
                    render_assignment=render_assignment,
                    levers=levers,
                    cache_dir_path=cache_dir_path,
                    profile=profile,
                    module_token_budget=expert_module_token_budget,
                    max_rows_per_expert=max_rows_per_expert,
                    render_mode=expert_render_mode,
                    progress=progress,
                )
                coverage.update(cov)
        finally:
            if did_install:
                unload(L)
                if set_priority is not None:
                    set_priority(set())
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    for L in range(num_layers):
        _process_layer(L)
    # Head / root-level Linears (rare — lm_head is normally pinned-skipped) are
    # resident throughout; render them last with no install.
    _process_layer(None)

    if coverage:
        cache.metadata["packed_expert_coverage"] = coverage
    requested_formats = tuple(
        dict.fromkeys(_canon_fmt(f) for f in formats if str(f).strip())
    )
    cache.metadata.update({
        "render_scope": "assignment",
        "requested_formats": list(requested_formats),
        "requested_entries": int(len(cache.weights)),
        "streaming": True,
        "render_mechanism_order": [
            {
                "name": spec.name,
                "operation": spec.operation,
                "scope": spec.scope,
                "gate_metric": spec.gate_metric,
            }
            for spec in mechanism_plan.ordered
        ],
        "render_scores": {
            "schema": "prismaquant.production_render_scores.v1",
            "entries": int(len(render_score_records)),
            "records": dict(sorted(render_score_records.items())),
        },
    })
    if cache.failed:
        cache.metadata["render_failures"] = {
            f"{q}|{fmt}": str(err)
            for (q, fmt), err in sorted(cache.failed.items())
        }
    return cache


def _priority_setter(ctx):
    """Set the resident-layer priority AND re-arm the memory-pressure floor
    before each install — mirrors the streaming cost path so a UMA-pressured
    box pre-evicts instead of OOMing on the layer read."""
    def _set(layers: set) -> None:
        ctx.layer_cache.set_priority_layers(layers)
        ctx.configure_runtime_pressure_floor()
    return _set


def fill_production_weight_cache_streaming(
    model_path: str,
    *,
    render_assignment: Mapping[str, str],
    activation_cache_dir: str | Path,
    formats: Sequence[str],
    levers: Mapping[str, object] | None,
    cache_dir: str | Path,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    skip_tokens: Sequence[str] | None = None,
    expert_render_mode: str = "batched",
    expert_module_token_budget: int = 32768,
    max_rows_per_expert: int = 2048,
    h_detail_dir: str | Path | None = None,
    offload_folder: str | Path | None = None,
    progress: bool = True,
) -> ProductionWeightCache:
    """Build a production δw cache one decoder layer at a time.

    No whole-model ``from_pretrained``: a ``StreamingContext`` keeps only the
    head resident and streams each decoder layer's weights on demand. Requires
    ``cache_dir`` (disk streaming — the pickle is a manifest) and a probe
    activation cache produced with the same calibration.
    """
    device = torch.device(device)
    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    if offload_folder is None:
        offload_folder = cache_dir_path / "streaming_offload"

    ctx = _build_streaming_context(
        model_path,
        device=device,
        dtype=dtype,
        offload_folder=str(offload_folder),
        log_prefix="[stream-prod-cache]",
    )
    try:
        model = ctx.model
        try:
            from prismaquant.model_profiles import profile_from_model
            profile = profile_from_model(model)
        except Exception:
            profile = None
        if skip_tokens is None:
            skip_tokens = (
                list(profile.pinned_names())
                if profile is not None
                and hasattr(profile, "pinned_names")
                else []
            )
        act_index = ActivationIndex(Path(activation_cache_dir), [])
        cache = run_streaming_render(
            model,
            layers_prefix=ctx.layers_prefix,
            num_layers=ctx.num_layers,
            render_assignment=render_assignment,
            act_index=act_index,
            formats=formats,
            levers=levers,
            cache_dir_path=cache_dir_path,
            profile=profile,
            skip_tokens=skip_tokens,
            device=device,
            expert_render_mode=expert_render_mode,
            expert_module_token_budget=expert_module_token_budget,
            max_rows_per_expert=max_rows_per_expert,
            h_detail_dir=h_detail_dir,
            install=ctx.install,
            unload=ctx.unload,
            set_priority=_priority_setter(ctx),
            progress=progress,
        )
    finally:
        ctx.shutdown()
    return cache
