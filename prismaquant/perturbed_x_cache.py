"""Build activation caches under a perturbed allocation.

The regular probe cache captures BF16 model inputs. Perturbed-X iterations need
the same cache shape after upstream layers have already run with the current
allocation's weight and activation quantization. This module installs one
forward_pre_hook per quantized module: it snapshots the original input first,
then returns the activation-quantized input for the actual forward. Weights are
RTN-quantized just for that module call and restored in the forward hook.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from prismaquant import format_registry as fr
from prismaquant.build_rtn_cache import iter_quantizable_tensors
from prismaquant.memory_management import (
    enforce_gpu_memory_budget,
    env_int,
    register_budget_evictor,
)

_FNAME_SUB = re.compile(r"[^A-Za-z0-9_-]")
_SHARED_FROZEN_WEIGHT_FORMAT_CACHE: OrderedDict[
    tuple[str, str, int, str, str],
    torch.Tensor,
] = OrderedDict()


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Clamping inputs to the calibrated max(|activations|) before per-group
# RTN matches the export's act-clip behavior.  Without this, dynamic
# per-group RTN sets scales from the raw input's local max, so outliers
# dominate and any pre-scaling is mathematically a no-op
# (Q(x/s)*s == Q(x) under purely dynamic Q — codex round-3 caught this).


def _activation_max_abs_lookup(
    activation_max_abs: dict,
    param_name: str | None,
) -> float | None:
    """Resolve ``param_name`` against ``activation_max_abs`` with the same
    alias-fallbacks as ``ProductionWeightCache.get`` so cache hits and
    activation-clip lookups stay consistent."""
    if param_name is None or not activation_max_abs:
        return None
    candidates = [param_name]
    if param_name.endswith(".weight"):
        candidates.append(param_name[:-len(".weight")])
    if param_name.startswith("model.language_model."):
        candidates.append("model." + param_name[len("model.language_model."):])
    for cand in candidates:
        v = activation_max_abs.get(cand)
        if v is not None:
            return v
    return None


def _maybe_clip_activations(
    x: "torch.Tensor",
    activation_max_abs: dict,
    param_name: str | None,
) -> "torch.Tensor":
    """Clamp activations to ±max_abs when a calibrated value is known.

    ``activation_max_abs`` is the dict from
    ``ProductionWeightCache.activation_max_abs`` (calibrated max(|x|)
    per fused-sibling group).  Returns ``x`` unchanged when:

      * no entry is registered for ``param_name`` (or its aliases),
      * the registered value is non-positive, or
      * ``PRISMAQUANT_PROD_ACT_SCALES`` is explicitly disabled.
    """
    max_abs = _activation_max_abs_lookup(activation_max_abs, param_name)
    if max_abs is None or max_abs <= 0:
        return x
    if not _env_truthy("PRISMAQUANT_PROD_ACT_SCALES", default=True):
        return x
    return x.clamp(-float(max_abs), float(max_abs))


def activation_cache_filename(name: str) -> str:
    return _FNAME_SUB.sub("__", name) + ".pt"


def _tensor_hash_update(h: "hashlib._Hash", tensor: torch.Tensor) -> None:
    t = tensor.detach().to("cpu").contiguous()
    h.update(str(tuple(t.shape)).encode())
    h.update(str(t.dtype).encode())
    h.update(t.view(torch.uint8).numpy().tobytes())


def calibration_data_hash(calibration_data) -> str:
    """Stable content hash used to seed shared row subsampling."""
    h = hashlib.blake2b(digest_size=16)
    if isinstance(calibration_data, torch.Tensor):
        _tensor_hash_update(h, calibration_data)
        return h.hexdigest()
    if isinstance(calibration_data, Mapping):
        for key in sorted(calibration_data):
            h.update(str(key).encode())
            value = calibration_data[key]
            if isinstance(value, torch.Tensor):
                _tensor_hash_update(h, value)
            else:
                h.update(repr(value).encode())
        return h.hexdigest()
    for sample in calibration_data:
        if isinstance(sample, torch.Tensor):
            _tensor_hash_update(h, sample)
        elif isinstance(sample, Mapping):
            for key in sorted(sample):
                h.update(str(key).encode())
                value = sample[key]
                if isinstance(value, torch.Tensor):
                    _tensor_hash_update(h, value)
                else:
                    h.update(repr(value).encode())
        else:
            h.update(repr(sample).encode())
    return h.hexdigest()


def _seed_from(cal_hash: str, group_key: str) -> int:
    digest = hashlib.blake2b(
        f"{cal_hash}:{group_key}".encode(),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def fused_subsample_group(name: str, profile=None) -> str:
    """Return the deterministic row-subsample group for a recipe name."""
    if profile is not None:
        try:
            group = profile.fused_sibling_group(name)
            if group is not None:
                return str(group)
        except Exception:
            pass
    bare = name[:-7] if name.endswith(".weight") else name
    parent, _, leaf = bare.rpartition(".")
    if leaf in {"q_proj", "k_proj", "v_proj"}:
        return f"{parent.rsplit('.', 1)[0]}.qkv"
    if leaf in {"gate_proj", "up_proj"}:
        return f"{parent.rsplit('.', 1)[0]}.gate_up"
    if leaf in {"in_proj_qkv", "in_proj_z"}:
        return f"{parent.rsplit('.', 1)[0]}.in_proj_qkvz"
    if leaf in {"in_proj_a", "in_proj_b"}:
        return f"{parent.rsplit('.', 1)[0]}.in_proj_ab"
    return bare


class SharedRowSubsampler:
    def __init__(self, input_rows: int, cal_hash: str, profile=None):
        self.input_rows = int(input_rows)
        self.cal_hash = cal_hash
        self.profile = profile
        self._indices: dict[tuple[str, int, int], torch.Tensor] = {}

    def select(self, name: str, flat: torch.Tensor, need: int) -> torch.Tensor:
        if need <= 0 or flat.size(0) <= need:
            return flat
        group = fused_subsample_group(name, self.profile)
        key = (group, int(flat.size(0)), int(need))
        idx = self._indices.get(key)
        if idx is None:
            g = torch.Generator(device="cpu")
            g.manual_seed(_seed_from(self.cal_hash, group))
            idx = torch.randperm(flat.size(0), generator=g)[:need]
            self._indices[key] = idx
        return flat.index_select(0, idx.to(flat.device))


@dataclass
class _ParamPlan:
    name: str
    attr: str
    spec: fr.FormatSpec


@dataclass
class _ModulePlan:
    module: nn.Module
    params: list[_ParamPlan] = field(default_factory=list)
    active_originals: list[tuple[torch.nn.Parameter, torch.Tensor]] = field(
        default_factory=list
    )
    act_spec: fr.FormatSpec | None = None
    act_conflict: bool = False

    @property
    def cache_names(self) -> list[str]:
        return [p.name for p in self.params]


def build_quantizable_map(model: nn.Module) -> dict[str, tuple[nn.Module, str]]:
    """Map recipe/probe names to live module parameters."""
    out: dict[str, tuple[nn.Module, str]] = {}
    for full_name, mod, attr in iter_quantizable_tensors(model):
        names = {full_name}
        if full_name.endswith(".weight"):
            names.add(full_name[:-7])
        for name in list(names):
            if name.startswith("model."):
                suffix = name[len("model."):]
                names.add(f"model.language_model.{suffix}")
        for name in names:
            out[name] = (mod, attr)
    return out


def _build_module_plans(
    model: nn.Module,
    assignment: Mapping[str, str],
) -> tuple[list[_ModulePlan], list[str], list[dict]]:
    quant_map = build_quantizable_map(model)
    by_module: dict[int, _ModulePlan] = {}
    missing: list[str] = []
    for name, fmt in assignment.items():
        target = quant_map.get(name)
        if target is None:
            missing.append(name)
            continue
        mod, attr = target
        spec = fr.get_format(fmt)
        plan = by_module.setdefault(id(mod), _ModulePlan(module=mod))
        plan.params.append(_ParamPlan(name=name, attr=attr, spec=spec))

    skipped: list[dict] = []
    for plan in by_module.values():
        low_act = {
            p.spec.name: p.spec
            for p in plan.params
            if p.spec.act_bits is not None and p.spec.act_bits < 16
        }
        if len(low_act) == 1:
            plan.act_spec = next(iter(low_act.values()))
        elif len(low_act) > 1:
            plan.act_conflict = True
            skipped.append(
                {
                    "module": type(plan.module).__name__,
                    "weights": sorted(plan.cache_names),
                    "formats": sorted(low_act),
                }
            )
    return list(by_module.values()), missing, skipped


def _first_tensor_location(args, kwargs):
    if args:
        for idx, value in enumerate(args):
            if isinstance(value, torch.Tensor):
                return "args", idx, value
    if kwargs:
        for key in ("hidden_states", "inputs_embeds", "input"):
            value = kwargs.get(key)
            if isinstance(value, torch.Tensor):
                return "kwargs", key, value
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                return "kwargs", key, value
    return None, None, None


def _replace_tensor_input(args, kwargs, where, key, value):
    if where == "args":
        args_list = list(args)
        args_list[int(key)] = value
        return tuple(args_list), kwargs
    if where == "kwargs":
        kwargs = dict(kwargs or {})
        kwargs[key] = value
        return args, kwargs
    return args, kwargs


class PerturbedActivationCache:
    def __init__(
        self,
        model: nn.Module,
        assignment: Mapping[str, str],
        cache_dir: str | Path,
        *,
        input_rows: int = 256,
        cal_hash: str,
        profile=None,
        production_weight_cache=None,
        include_activation_quant: bool = True,
    ):
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.input_rows = int(input_rows)
        self.include_activation_quant = bool(include_activation_quant)
        self.subsampler = SharedRowSubsampler(input_rows, cal_hash, profile)
        self.plans, self.missing, self.skipped = _build_module_plans(
            model, assignment
        )
        self._production_weight_cache = production_weight_cache
        # MED-3: per-Linear calibrated max(|activations|), unified across
        # fused-sibling groups.  Used by the activation-quant hook to
        # clamp activations to ±max_abs before per-group RTN, matching
        # the export's act-clip behavior.  See production_weight_cache.py
        # for the convention note (we store max_abs directly; the export's
        # vLLM-facing metadata convention is 6.0 / max_abs).
        if production_weight_cache is not None and (
            production_weight_cache.activation_max_abs
            or production_weight_cache.activation_scales
        ):
            src = (
                production_weight_cache.activation_max_abs
                or production_weight_cache.activation_scales
            )
            self._activation_scales: dict[str, float] = dict(src)
        else:
            self._activation_scales = {}
        self._snaps: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._rows_got: dict[str, int] = defaultdict(int)
        self.max_abs: dict[str, float] = {}
        self._handles = []
        self._frozen_weight_cache: OrderedDict[
            tuple[int, str], torch.Tensor
        ] | None = None
        self._frozen_weight_format_cache: OrderedDict[
            tuple[str, str, int, str, str], torch.Tensor
        ] = (
            _SHARED_FROZEN_WEIGHT_FORMAT_CACHE
            if _env_truthy("PRISMAQUANT_SHARED_WEIGHT_FORMAT_CACHE")
            else OrderedDict()
        )
        self._fused_forward_originals: list[tuple[nn.Module, object]] = []
        self._fused_nvfp4_weight_cache: OrderedDict[
            tuple[str, str, str, int],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = OrderedDict()
        self._materialized_frozen_weight_depth = 0
        self._frozen_weight_cache_evictions = 0
        self._frozen_weight_cache_eviction_reported = False
        register_budget_evictor(self)

    @property
    def installed(self) -> bool:
        return bool(self._handles)

    def install(self) -> None:
        for plan in self.plans:
            if self._try_install_nvfp4_fused_forward(plan):
                continue
            self._handles.append(
                plan.module.register_forward_pre_hook(
                    self._make_pre_hook(plan),
                    with_kwargs=True,
                )
            )
            self._handles.append(
                plan.module.register_forward_hook(
                    self._make_post_hook(plan),
                    with_kwargs=True,
                )
            )

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for module, original_forward in reversed(self._fused_forward_originals):
            module.forward = original_forward
        self._fused_forward_originals.clear()
        for plan in self.plans:
            self._restore_plan(plan)

    def _find_param_plan(self, name: str) -> tuple[_ModulePlan, _ParamPlan]:
        for plan in self.plans:
            for param_plan in plan.params:
                if param_plan.name == name:
                    return plan, param_plan
        raise KeyError(f"no quantized parameter named {name!r}")

    def _quantized_weight_for(
        self,
        plan: _ModulePlan,
        param_plan: _ParamPlan,
        spec: fr.FormatSpec,
    ) -> torch.Tensor | None:
        param = getattr(plan.module, param_plan.attr)
        if not isinstance(param, torch.nn.Parameter) or param.is_meta:
            return None
        fmt = fr.canonical_format_name(spec.name)
        # Include production-cache identity in the key so a SHARED
        # frozen_weight_format_cache that's seen multiple instances
        # (with/without production cache, or different production
        # caches) doesn't return stale entries.  Same-instance reuse
        # across polish trials still hits because id() is stable.
        prod_id = (
            id(self._production_weight_cache)
            if self._production_weight_cache is not None else 0
        )
        cache_key = (
            param_plan.name,
            fmt,
            int(param.data_ptr()),
            str(param.device),
            str(param.dtype),
            prod_id,
        )
        q = self._frozen_weight_format_cache.get(cache_key)
        if q is None:
            enforce_gpu_memory_budget(
                [self],
                device=param.device if param.device.type == "cuda" else None,
                reason="frozen weight cache fill",
            )
            production = (
                self._production_weight_cache.get(param_plan.name, fmt)
                if self._production_weight_cache is not None
                else None
            )
            if production is not None:
                q = production.to(
                    device=param.device,
                    dtype=param.dtype,
                ).contiguous()
            else:
                if (
                    self._production_weight_cache is not None
                    and fmt != "BF16"
                    and _env_truthy("PRISMAQUANT_STRICT_PRODUCTION_CACHE")
                ):
                    raise RuntimeError(
                        f"production_weight_cache miss for "
                        f"({param_plan.name!r}, {fmt!r}); set "
                        f"PRISMAQUANT_STRICT_PRODUCTION_CACHE=0 to fall back "
                        f"to RTN, or rebuild the cache to cover this Linear."
                    )
                original = param.data.detach().clone()
                q = spec.quantize_dequantize(original).to(
                    device=param.device,
                    dtype=param.dtype,
                ).contiguous()
            # cache_key now includes production-cache identity, so we
            # can safely populate the shared cache regardless of
            # production-active state.  Different production caches
            # (or no cache) get distinct keys; no cross-contamination.
            if self._frozen_weight_cache_max_entries() > 0:
                self._frozen_weight_format_cache[cache_key] = q
                self._evict_frozen_weight_format_cache_to_limit()
            enforce_gpu_memory_budget(
                [self],
                device=param.device if param.device.type == "cuda" else None,
                reason="frozen weight cache fill",
            )
        else:
            self._frozen_weight_format_cache.move_to_end(cache_key)
        return q

    def build_frozen_weight_cache(self) -> dict[tuple[int, str], torch.Tensor]:
        cache: OrderedDict[tuple[int, str], torch.Tensor] = OrderedDict()
        for plan in self.plans:
            seen_attrs: set[str] = set()
            for param_plan in plan.params:
                if param_plan.attr in seen_attrs:
                    continue
                seen_attrs.add(param_plan.attr)
                q = self._quantized_weight_for(plan, param_plan, param_plan.spec)
                if q is None:
                    continue
                cache[(id(plan.module), param_plan.attr)] = q
        self._frozen_weight_cache = cache
        return cache

    @contextmanager
    def frozen_weight_cache(self) -> Iterator["PerturbedActivationCache"]:
        previous = self._frozen_weight_cache
        self.build_frozen_weight_cache()
        try:
            yield self
        finally:
            self._frozen_weight_cache = previous
            self._emit_frozen_weight_cache_evictions()

    @contextmanager
    def materialized_frozen_weights(self) -> Iterator["PerturbedActivationCache"]:
        """Apply the active frozen weights to modules for whole-forward reuse."""
        if self._frozen_weight_cache is None:
            raise RuntimeError("frozen weight cache is not active")
        if self._materialized_frozen_weight_depth > 0:
            self._materialized_frozen_weight_depth += 1
            try:
                yield self
            finally:
                self._materialized_frozen_weight_depth -= 1
            return

        originals: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
        seen_keys: set[tuple[int, str]] = set()
        self._materialized_frozen_weight_depth = 1
        try:
            for plan in self.plans:
                for param_plan in plan.params:
                    cache_key = (id(plan.module), param_plan.attr)
                    if cache_key in seen_keys:
                        continue
                    seen_keys.add(cache_key)
                    param = getattr(plan.module, param_plan.attr)
                    if not isinstance(param, torch.nn.Parameter) or param.is_meta:
                        continue
                    q = self._frozen_weight_cache.get(cache_key)
                    if q is None:
                        continue
                    self._frozen_weight_cache.move_to_end(cache_key)
                    originals.append((param, param.data.detach().clone()))
                    param.data.copy_(q.to(device=param.device, dtype=param.dtype))
            yield self
        finally:
            for param, original in reversed(originals):
                param.data.copy_(original.to(device=param.device, dtype=param.dtype))
            self._materialized_frozen_weight_depth = 0

    def set_frozen_weight_format(self, name: str, fmt: str) -> None:
        if self._frozen_weight_cache is None:
            raise RuntimeError("frozen weight cache is not active")
        plan, param_plan = self._find_param_plan(name)
        spec = fr.get_format(fmt)
        q = self._quantized_weight_for(plan, param_plan, spec)
        if q is None:
            return
        self._frozen_weight_cache[(id(plan.module), param_plan.attr)] = q
        self._frozen_weight_cache.move_to_end((id(plan.module), param_plan.attr))
        param_plan.spec = spec

    @contextmanager
    def temporary_frozen_weight_format(
        self,
        name: str,
        fmt: str,
    ) -> Iterator["PerturbedActivationCache"]:
        with self.override({name: fmt}):
            yield self

    @contextmanager
    def override(
        self,
        assignment_delta: Mapping[str, str],
    ) -> Iterator["PerturbedActivationCache"]:
        if self._frozen_weight_cache is None:
            raise RuntimeError("frozen weight cache is not active")
        previous: list[
            tuple[tuple[int, str], torch.Tensor | None, _ParamPlan, fr.FormatSpec]
        ] = []
        for name, fmt in assignment_delta.items():
            plan, param_plan = self._find_param_plan(name)
            cache_key = (id(plan.module), param_plan.attr)
            previous.append(
                (
                    cache_key,
                    self._frozen_weight_cache.get(cache_key),
                    param_plan,
                    param_plan.spec,
                )
            )
            self.set_frozen_weight_format(name, fmt)
        try:
            yield self
        finally:
            for cache_key, previous_q, param_plan, previous_spec in reversed(previous):
                if previous_q is None:
                    self._frozen_weight_cache.pop(cache_key, None)
                else:
                    self._frozen_weight_cache[cache_key] = previous_q
                    self._frozen_weight_cache.move_to_end(cache_key)
                param_plan.spec = previous_spec

    def _capture(self, plan: _ModulePlan, x: torch.Tensor) -> None:
        flat = x.detach().reshape(-1, x.size(-1))
        mx = float(flat.abs().max().item())
        for name in plan.cache_names:
            if mx > self.max_abs.get(name, 0.0):
                self.max_abs[name] = mx
            need = self.input_rows - self._rows_got[name]
            if need <= 0:
                continue
            selected = self.subsampler.select(name, flat, need)
            self._snaps[name].append(selected.to("cpu"))
            self._rows_got[name] += int(selected.size(0))

    def _apply_weight_quant(self, plan: _ModulePlan) -> None:
        plan.active_originals.clear()
        if self._materialized_frozen_weight_depth > 0:
            return
        if _env_truthy("PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT", default=False):
            # Caller (e.g. WeightSession) has installed the desired weights
            # directly on model.params; we just observe + activation-
            # quantize, no clone/restore.  Saves ~50 MB clone per module
            # on the hot path and lets polish on big models avoid the
            # cumulative-clone OOM.
            return
        seen_attrs: set[str] = set()
        for param_plan in plan.params:
            if param_plan.attr in seen_attrs:
                continue
            seen_attrs.add(param_plan.attr)
            param = getattr(plan.module, param_plan.attr)
            if not isinstance(param, torch.nn.Parameter) or param.is_meta:
                continue
            original = param.data.detach().clone()
            q = None
            if self._frozen_weight_cache is not None:
                cache_key = (id(plan.module), param_plan.attr)
                q = self._frozen_weight_cache.get(cache_key)
                if q is not None:
                    self._frozen_weight_cache.move_to_end(cache_key)
            if q is None and _env_truthy("PRISMAQUANT_SHARED_WEIGHT_FORMAT_CACHE"):
                q = self._quantized_weight_for(plan, param_plan, param_plan.spec)
            if q is None and self._production_weight_cache is not None:
                fmt_canon = fr.canonical_format_name(param_plan.spec.name)
                production = self._production_weight_cache.get(
                    param_plan.name, fmt_canon,
                )
                if production is not None:
                    q = production.to(
                        device=param.device, dtype=param.dtype,
                    ).contiguous()
                elif (
                    fmt_canon != "BF16"
                    and _env_truthy("PRISMAQUANT_STRICT_PRODUCTION_CACHE")
                ):
                    raise RuntimeError(
                        f"production_weight_cache miss for "
                        f"({param_plan.name!r}, {fmt_canon!r}); set "
                        f"PRISMAQUANT_STRICT_PRODUCTION_CACHE=0 to allow "
                        f"RTN fallback."
                    )
            if q is None:
                q = param_plan.spec.quantize_dequantize(original)
            if q is None:
                continue
            param.data.copy_(q.to(device=param.device, dtype=param.dtype))
            plan.active_originals.append((param, original))

    def _active_activation_spec(self, plan: _ModulePlan) -> fr.FormatSpec | None:
        if not self.include_activation_quant:
            return None
        low_act = {
            p.spec.name: p.spec
            for p in plan.params
            if p.spec.act_bits is not None and p.spec.act_bits < 16
        }
        if len(low_act) == 1:
            return next(iter(low_act.values()))
        return None

    def _nvfp4_fused_param_plan(self, plan: _ModulePlan) -> _ParamPlan | None:
        if not _env_truthy("PRISMAQUANT_FUSED_KERNEL_NVFP4"):
            return None
        # When a production cache is active, the fused fast path's
        # `nvfp4_pack_weight` re-computes per-group scales locally and
        # ignores the cache's joint NVFP4 sibling globals — so the
        # packed FP4 codes diverge from what the export would produce.
        # Refuse to use the fast path in that mode unless the user
        # explicitly opts in via PRISMAQUANT_FUSED_KERNEL_OVER_PROD_CACHE.
        if (
            self._production_weight_cache is not None
            and not _env_truthy("PRISMAQUANT_FUSED_KERNEL_OVER_PROD_CACHE")
        ):
            return None
        if not isinstance(plan.module, nn.Linear) or len(plan.params) != 1:
            return None
        param_plan = plan.params[0]
        if param_plan.attr != "weight":
            return None
        if fr.canonical_format_name(param_plan.spec.name) != "NVFP4":
            return None
        act_spec = self._active_activation_spec(plan)
        if act_spec is None or fr.canonical_format_name(act_spec.name) != "NVFP4":
            return None
        return param_plan

    def _try_install_nvfp4_fused_forward(self, plan: _ModulePlan) -> bool:
        param_plan = self._nvfp4_fused_param_plan(plan)
        if param_plan is None:
            return False
        try:
            from prismaquant.kernels.nvfp4_fused import nvfp4_fused_aw_matmul  # noqa: F401
        except Exception:
            return False

        module = plan.module
        original_forward = module.forward

        def _forward(x, *args, **kwargs):
            if args or kwargs or not isinstance(x, torch.Tensor):
                return original_forward(x, *args, **kwargs)
            return self._nvfp4_fused_linear_forward(plan, param_plan, x)

        module.forward = _forward
        self._fused_forward_originals.append((module, original_forward))
        return True

    def _weight_for_reference_forward(
        self,
        plan: _ModulePlan,
        param_plan: _ParamPlan,
    ) -> torch.Tensor:
        param = getattr(plan.module, param_plan.attr)
        if not isinstance(param, torch.nn.Parameter) or param.is_meta:
            return param
        q = None
        if self._frozen_weight_cache is not None:
            cache_key = (id(plan.module), param_plan.attr)
            q = self._frozen_weight_cache.get(cache_key)
            if q is not None:
                self._frozen_weight_cache.move_to_end(cache_key)
        if q is None:
            q = self._quantized_weight_for(plan, param_plan, param_plan.spec)
        if q is None:
            return param
        return q.to(device=param.device, dtype=param.dtype)

    def _reference_linear_forward(
        self,
        plan: _ModulePlan,
        param_plan: _ParamPlan,
        x: torch.Tensor,
    ) -> torch.Tensor:
        act_spec = self._active_activation_spec(plan)
        if act_spec is not None:
            # MED-3: act-clip the input to the calibrated max_abs before
            # per-group RTN.  The dynamic per-group quantizer in
            # `act_spec.activation_quantize_dequantize` would otherwise
            # set its scales from the input's per-group max — outliers
            # then dominate.  Production export does the same clipping
            # as `_resolve_act_clip_quantile`, so this matches what the
            # shipped artifact sees at runtime.  `Q(x/s)*s == Q(x)` for
            # purely dynamic Q, so the previous "pre-scale + post-multiply"
            # formulation was a no-op (codex round-3).
            x = _maybe_clip_activations(
                x, self._activation_scales, param_plan.name,
            )
            x = act_spec.activation_quantize_dequantize(x)
        weight = self._weight_for_reference_forward(plan, param_plan)
        return F.linear(x, weight, plan.module.bias)

    def _packed_nvfp4_weight_for(
        self,
        plan: _ModulePlan,
        param_plan: _ParamPlan,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        param = getattr(plan.module, param_plan.attr)
        if not isinstance(param, torch.nn.Parameter) or param.is_meta:
            raise RuntimeError("cannot pack a missing or meta Linear weight")
        cache_key = (
            param_plan.name,
            str(param.device),
            str(param.dtype),
            int(param.data_ptr()),
        )
        packed = self._fused_nvfp4_weight_cache.get(cache_key)
        if packed is None:
            from prismaquant.kernels.nvfp4_fused import nvfp4_pack_weight

            enforce_gpu_memory_budget(
                [self],
                device=param.device if param.device.type == "cuda" else None,
                reason="NVFP4 packed weight cache fill",
            )
            # HIGH: prefer the production cache's GPTQ + scale_sweep
            # weight when present.  Without this, the fused NVFP4 fast
            # path packs the raw BF16 param and bypasses the entire
            # production cache (silently runs RTN-equivalent weights
            # through the kernel).  Strict mode raises on miss so the
            # fast path matches the slow-path miss semantics.
            source = param.detach()
            if self._production_weight_cache is not None:
                w_dq = self._production_weight_cache.get(
                    param_plan.name, "NVFP4",
                )
                if w_dq is not None:
                    source = w_dq.to(
                        device=param.device, dtype=param.dtype,
                    ).contiguous()
                elif _env_truthy("PRISMAQUANT_STRICT_PRODUCTION_CACHE"):
                    raise RuntimeError(
                        f"production_weight_cache miss for "
                        f"({param_plan.name!r}, 'NVFP4') on the fused "
                        f"NVFP4 fast path; set "
                        f"PRISMAQUANT_STRICT_PRODUCTION_CACHE=0 to allow "
                        f"raw-weight fallback or rebuild the cache."
                    )
            packed = nvfp4_pack_weight(source)
            self._fused_nvfp4_weight_cache[cache_key] = packed
            self._fused_nvfp4_weight_cache.move_to_end(cache_key)
            enforce_gpu_memory_budget(
                [self],
                device=param.device if param.device.type == "cuda" else None,
                reason="NVFP4 packed weight cache fill",
            )
        else:
            self._fused_nvfp4_weight_cache.move_to_end(cache_key)
        return packed

    def _frozen_weight_cache_max_entries(self) -> int:
        return env_int("PRISMAQUANT_FROZEN_WEIGHT_CACHE_MAX_ENTRIES", 400)

    def _evict_frozen_weight_format_cache_to_limit(self) -> None:
        max_entries = self._frozen_weight_cache_max_entries()
        if max_entries <= 0:
            evicted = len(self._frozen_weight_format_cache)
            self._frozen_weight_format_cache.clear()
            self._frozen_weight_cache_evictions += evicted
            return
        while len(self._frozen_weight_format_cache) > max_entries:
            self._frozen_weight_format_cache.popitem(last=False)
            self._frozen_weight_cache_evictions += 1

    def evict_oldest_for_memory_budget(self) -> bool:
        if self._frozen_weight_format_cache:
            self._frozen_weight_format_cache.popitem(last=False)
            self._frozen_weight_cache_evictions += 1
            return True
        if self._fused_nvfp4_weight_cache:
            self._fused_nvfp4_weight_cache.popitem(last=False)
            return True
        if self._frozen_weight_cache:
            self._frozen_weight_cache.popitem(last=False)
            self._frozen_weight_cache_evictions += 1
            return True
        return False

    def _emit_frozen_weight_cache_evictions(self) -> None:
        if (
            self._frozen_weight_cache_evictions <= 0
            or self._frozen_weight_cache_eviction_reported
        ):
            return
        self._frozen_weight_cache_eviction_reported = True
        print(
            "[frozen-weight-cache] evicted "
            f"{self._frozen_weight_cache_evictions} entries "
            f"(max_entries={self._frozen_weight_cache_max_entries()})",
            file=sys.stderr,
            flush=True,
        )

    def _nvfp4_fused_linear_forward(
        self,
        plan: _ModulePlan,
        param_plan: _ParamPlan,
        x: torch.Tensor,
    ) -> torch.Tensor:
        self._capture(plan, x)
        act_spec = self._active_activation_spec(plan)
        fused_active = (
            fr.canonical_format_name(param_plan.spec.name) == "NVFP4"
            and act_spec is not None
            and fr.canonical_format_name(act_spec.name) == "NVFP4"
            and x.is_cuda
            and x.shape[-1] % 16 == 0
        )
        if not fused_active:
            return self._reference_linear_forward(plan, param_plan, x)

        from prismaquant.kernels.nvfp4_fused import nvfp4_fused_aw_matmul

        w_packed, w_scales, w_global_scale = self._packed_nvfp4_weight_for(
            plan, param_plan
        )
        flat_x = x.reshape(-1, x.shape[-1])
        # MED-3: act-clip the activation to the calibrated max_abs before
        # the fused kernel's internal per-group RTN.  Same rationale as
        # ``_reference_linear_forward``: pre-scale + post-multiply
        # cancels under dynamic per-group RTN, but clipping forces
        # outliers to the calibrated range so per-group scales are
        # bounded — matching production's act-clip semantics.
        flat_x = _maybe_clip_activations(
            flat_x, self._activation_scales, param_plan.name,
        )
        out = nvfp4_fused_aw_matmul(flat_x, w_packed, w_scales, w_global_scale)
        out = out.reshape(*x.shape[:-1], plan.module.out_features)
        if plan.module.bias is not None:
            out = out + plan.module.bias.to(device=out.device, dtype=out.dtype)
        return out

    def _restore_plan(self, plan: _ModulePlan) -> None:
        if _env_truthy("PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT", default=False):
            # Mirror of _apply_weight_quant's bypass — WeightSession
            # owns weight transitions, so there's nothing to restore.
            return
        for param, original in reversed(plan.active_originals):
            param.data.copy_(original.to(device=param.device, dtype=param.dtype))
        plan.active_originals.clear()

    def _make_pre_hook(self, plan: _ModulePlan):
        def _pre_hook(_module, args, kwargs):
            where, key, x = _first_tensor_location(args, kwargs)
            if isinstance(x, torch.Tensor):
                self._capture(plan, x)
                act_spec = self._active_activation_spec(plan)
                if act_spec is not None:
                    # MED-3: act-clip to the calibrated max_abs before the
                    # quantizer, so outliers don't dominate per-group
                    # scales.  See ``_maybe_clip_activations`` for the
                    # math; pre-scale + post-multiply was a no-op (codex
                    # round-3 caught Q(x/s)*s == Q(x)).
                    member_name = next(
                        (p.name for p in plan.params if p.attr == "weight"),
                        None,
                    )
                    x_in = _maybe_clip_activations(
                        x, self._activation_scales, member_name,
                    )
                    qx = act_spec.activation_quantize_dequantize(x_in)
                    args, kwargs = _replace_tensor_input(args, kwargs, where, key, qx)
            self._apply_weight_quant(plan)
            return args, kwargs

        return _pre_hook

    def _make_post_hook(self, plan: _ModulePlan):
        def _post_hook(_module, _args, _kwargs, output):
            self._restore_plan(plan)
            return output

        return _post_hook

    def finalize(self) -> dict:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        for name, snaps in self._snaps.items():
            if not snaps:
                continue
            x = torch.cat(snaps, dim=0)[:self.input_rows]
            x = x.to(torch.bfloat16).contiguous()
            torch.save(
                {"inputs": x, "name": name, "source": "perturbed_x"},
                self.cache_dir / activation_cache_filename(name),
            )
            written.append(name)
        return {
            "cache_dir": str(self.cache_dir),
            "written": sorted(written),
            "missing": sorted(self.missing),
            "skipped_activation_quant": self.skipped,
        }


def _model_device(model: nn.Module) -> torch.device:
    for p in model.parameters():
        if not p.is_meta:
            return p.device
    return torch.device("cpu")


def _to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return value


def iter_calibration_forwards(
    calibration_data,
    device: torch.device,
    *,
    microbatch_size: int = 1,
):
    """Yield (args, kwargs) for one forward pass per calibration microbatch.

    ``microbatch_size`` controls how many calibration rows are stacked into
    each forward — default 1 preserves the historical one-sample-at-a-time
    behaviour every existing caller relies on.  Callers that want to amortize
    Python and kernel launch overhead can request a larger microbatch; the
    yielded batch dim becomes ``min(microbatch_size, remaining_rows)``.
    """
    if isinstance(calibration_data, torch.Tensor):
        n = int(calibration_data.size(0))
        m = max(1, int(microbatch_size))
        for i in range(0, n, m):
            yield (calibration_data[i:i + m].to(device),), {}
        return
    if isinstance(calibration_data, Mapping):
        yield (), {k: _to_device(v, device) for k, v in calibration_data.items()}
        return
    for sample in calibration_data:
        if isinstance(sample, torch.Tensor):
            yield (sample.to(device),), {}
        elif isinstance(sample, Mapping):
            yield (), {k: _to_device(v, device) for k, v in sample.items()}
        elif isinstance(sample, tuple):
            yield tuple(_to_device(v, device) for v in sample), {}
        else:
            yield (sample,), {}


@torch.no_grad()
def capture_perturbed_activation_cache(
    model: nn.Module,
    assignment: Mapping[str, str],
    calibration_data,
    cache_dir: str | Path,
    *,
    input_rows: int = 256,
    profile=None,
    cal_hash: str | None = None,
) -> dict:
    """Run calibration forwards and write an ActivationIndex-compatible cache."""
    cal_hash = cal_hash or calibration_data_hash(calibration_data)
    builder = PerturbedActivationCache(
        model,
        assignment,
        cache_dir,
        input_rows=input_rows,
        cal_hash=cal_hash,
        profile=profile,
    )
    device = _model_device(model)
    builder.install()
    try:
        # PRISMAQUANT_L2_CUDA_GRAPHS is intentionally not applied here.
        # These forwards must execute Python hooks on every batch to snapshot
        # perturbed-X activations; CUDA graph replay would skip those hooks and
        # silently under-fill the activation cache.
        for args, kwargs in iter_calibration_forwards(calibration_data, device):
            model(*args, **kwargs)
    finally:
        builder.remove()
    manifest = builder.finalize()
    manifest["calibration_hash"] = cal_hash
    manifest["input_rows"] = int(input_rows)
    with open(Path(cache_dir) / "perturbed_x_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def stage_text_only_under_work_root(model_path: str, work_root: str | Path) -> str:
    """Text-only staging equivalent to sensitivity_probe, but never under /tmp."""
    src = Path(model_path)
    cfg_path = src / "config.json"
    if not cfg_path.exists():
        return str(src)
    with open(cfg_path) as f:
        cfg = json.load(f)
    try:
        from .model_profiles import detect_profile
        profile = detect_profile(str(src))
    except Exception:
        profile = None
    strip_keys = (
        list(profile.stage_text_only_strip_keys())
        if profile is not None
        else [
            "vision_config",
            "audio_config",
            "speech_config",
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
        ]
    )
    needs_num_experts_alias = (
        "num_local_experts" in cfg and "num_experts" not in cfg
    )
    if (
        not any(k in cfg for k in ("vision_config", "text_config", "audio_config", "speech_config"))
        and not any(k in cfg for k in strip_keys)
        and not needs_num_experts_alias
    ):
        return str(src)

    promote_inner_mt = (
        profile.stage_text_only_promote_inner_model_type()
        if profile is not None else False
    )
    for key in strip_keys:
        cfg.pop(key, None)
    if "num_local_experts" in cfg and "num_experts" not in cfg:
        cfg["num_experts"] = cfg["num_local_experts"]
    if "text_config" in cfg:
        text_cfg = cfg.pop("text_config")
        for key, value in text_cfg.items():
            if key == "model_type":
                if promote_inner_mt:
                    cfg[key] = value
                continue
            cfg[key] = value
    archs = cfg.get("architectures", [])
    if archs:
        cfg["architectures"] = [
            arch.replace("ForConditionalGeneration", "ForCausalLM")
            for arch in archs
        ]

    root = Path(work_root)
    root.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix="prismaquant_stage_", dir=str(root)))
    skip = {
        "config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "processor_config.json",
    }
    for p in src.iterdir():
        if p.name in skip:
            continue
        (staged / p.name).symlink_to(p.resolve())
    with open(staged / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    return str(staged)


def load_text_model_under_work_root(
    model_path: str,
    *,
    device: str,
    dtype: torch.dtype,
    work_root: str | Path,
    device_map: str | None = None,
) -> nn.Module:
    from transformers import AutoModelForCausalLM

    staged = stage_text_only_under_work_root(model_path, work_root)
    load_device_map = device_map if device_map is not None else device
    load_kwargs = {
        "torch_dtype": dtype,
        "device_map": load_device_map,
        "low_cpu_mem_usage": False,
        "trust_remote_code": True,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
    except ValueError as exc:
        if "requires `accelerate`" not in str(exc) and "requires accelerate" not in str(exc):
            raise
        load_kwargs.pop("device_map", None)
        model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
        model.to(torch.device(device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
