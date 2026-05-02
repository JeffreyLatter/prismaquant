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
import re
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping

import torch
import torch.nn as nn

from prismaquant import format_registry as fr
from prismaquant.build_rtn_cache import iter_quantizable_tensors


_FNAME_SUB = re.compile(r"[^A-Za-z0-9_-]")


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
    ):
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.input_rows = int(input_rows)
        self.subsampler = SharedRowSubsampler(input_rows, cal_hash, profile)
        self.plans, self.missing, self.skipped = _build_module_plans(
            model, assignment
        )
        self._snaps: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._rows_got: dict[str, int] = defaultdict(int)
        self._handles = []
        self._frozen_weight_cache: dict[tuple[int, str], torch.Tensor] | None = None
        self._frozen_weight_format_cache: dict[
            tuple[str, str], torch.Tensor
        ] = {}

    @property
    def installed(self) -> bool:
        return bool(self._handles)

    def install(self) -> None:
        for plan in self.plans:
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
        cache_key = (param_plan.name, fmt)
        q = self._frozen_weight_format_cache.get(cache_key)
        if q is None:
            original = param.data.detach().clone()
            q = spec.quantize_dequantize(original).to(
                device=param.device,
                dtype=param.dtype,
            ).contiguous()
            self._frozen_weight_format_cache[cache_key] = q
        return q

    def build_frozen_weight_cache(self) -> dict[tuple[int, str], torch.Tensor]:
        cache: dict[tuple[int, str], torch.Tensor] = {}
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

    def set_frozen_weight_format(self, name: str, fmt: str) -> None:
        if self._frozen_weight_cache is None:
            raise RuntimeError("frozen weight cache is not active")
        plan, param_plan = self._find_param_plan(name)
        spec = fr.get_format(fmt)
        q = self._quantized_weight_for(plan, param_plan, spec)
        if q is None:
            return
        self._frozen_weight_cache[(id(plan.module), param_plan.attr)] = q
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
                param_plan.spec = previous_spec

    def _capture(self, plan: _ModulePlan, x: torch.Tensor) -> None:
        flat = x.detach().reshape(-1, x.size(-1))
        for name in plan.cache_names:
            need = self.input_rows - self._rows_got[name]
            if need <= 0:
                continue
            selected = self.subsampler.select(name, flat, need)
            self._snaps[name].append(selected.to("cpu"))
            self._rows_got[name] += int(selected.size(0))

    def _apply_weight_quant(self, plan: _ModulePlan) -> None:
        plan.active_originals.clear()
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
                q = self._frozen_weight_cache.get((id(plan.module), param_plan.attr))
            if q is None:
                q = param_plan.spec.quantize_dequantize(original)
            param.data.copy_(q.to(device=param.device, dtype=param.dtype))
            plan.active_originals.append((param, original))

    def _active_activation_spec(self, plan: _ModulePlan) -> fr.FormatSpec | None:
        low_act = {
            p.spec.name: p.spec
            for p in plan.params
            if p.spec.act_bits is not None and p.spec.act_bits < 16
        }
        if len(low_act) == 1:
            return next(iter(low_act.values()))
        return None

    def _restore_plan(self, plan: _ModulePlan) -> None:
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
                    qx = act_spec.activation_quantize_dequantize(x)
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


def iter_calibration_forwards(calibration_data, device: torch.device):
    if isinstance(calibration_data, torch.Tensor):
        for i in range(calibration_data.size(0)):
            yield (calibration_data[i:i + 1].to(device),), {}
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
    model = AutoModelForCausalLM.from_pretrained(
        staged,
        torch_dtype=dtype,
        device_map=load_device_map,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
