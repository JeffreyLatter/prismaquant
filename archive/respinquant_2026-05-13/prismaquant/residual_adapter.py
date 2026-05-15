"""Generic residual-adapter manifest and math.

This is the deployment substrate for ReSpinQuant-style runtime adapters. It is
intentionally architecture-neutral: PrismaQuant records module paths and
low-rank tensors, while the vLLM plugin decides how to attach them to whatever
base model architecture the artifact declares.
"""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn


PLUGIN_ARCHITECTURE = "PrismaResidualAdapterForCausalLM"
MANIFEST_FILENAME = "prisma_residual_adapters.json"
CONFIG_KEY = "prisma_residual_adapter"
_SAFE_RE = re.compile(r"[^0-9A-Za-z_]+")


def safe_adapter_id(module_path: str) -> str:
    text = _SAFE_RE.sub("_", module_path.strip("."))
    text = text.strip("_")
    return text or "adapter"


@dataclass(frozen=True)
class ResidualAdapterSpec:
    """One residual adapter insertion point."""

    module_path: str
    adapter_id: str
    rank: int = 0
    mode: str = "residual_pair"
    u_name: str | None = None
    v_name: str | None = None
    enabled: bool = True

    @staticmethod
    def from_module_path(module_path: str, *, rank: int = 0,
                         adapter_id: str | None = None,
                         mode: str = "residual_pair") -> "ResidualAdapterSpec":
        adapter_id = adapter_id or safe_adapter_id(module_path)
        u_name = None
        v_name = None
        if rank > 0:
            u_name = f"prisma_residual_adapters.{adapter_id}.u"
            v_name = f"prisma_residual_adapters.{adapter_id}.v"
        return ResidualAdapterSpec(
            module_path=module_path,
            adapter_id=adapter_id,
            rank=int(rank),
            mode=mode,
            u_name=u_name,
            v_name=v_name,
        )

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "ResidualAdapterSpec":
        module_path = str(data["module_path"])
        adapter_id = str(data.get("adapter_id") or safe_adapter_id(module_path))
        rank = int(data.get("rank", 0))
        u_name = data.get("u_name")
        v_name = data.get("v_name")
        if rank > 0:
            u_name = u_name or f"prisma_residual_adapters.{adapter_id}.u"
            v_name = v_name or f"prisma_residual_adapters.{adapter_id}.v"
        return ResidualAdapterSpec(
            module_path=module_path,
            adapter_id=adapter_id,
            rank=rank,
            mode=str(data.get("mode", "residual_pair")),
            u_name=u_name,
            v_name=v_name,
            enabled=bool(data.get("enabled", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResidualAdapterManifest:
    version: int
    base_architectures: tuple[str, ...]
    adapters: tuple[ResidualAdapterSpec, ...]
    format: str = "low_rank_residual_delta"

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "ResidualAdapterManifest":
        return ResidualAdapterManifest(
            version=int(data.get("version", 1)),
            base_architectures=tuple(str(x) for x in data.get("base_architectures", ())),
            adapters=tuple(
                ResidualAdapterSpec.from_dict(item)
                for item in data.get("adapters", ())
            ),
            format=str(data.get("format", "low_rank_residual_delta")),
        )

    @staticmethod
    def load(path: str | Path) -> "ResidualAdapterManifest":
        return ResidualAdapterManifest.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "format": self.format,
            "base_architectures": list(self.base_architectures),
            "adapters": [adapter.to_dict() for adapter in self.adapters],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def write(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json(indent=2) + "\n")
        return out


class LowRankResidualAdapter(nn.Module):
    """Residual adapter using existing GEMM kernels: ``x + (x @ u) @ v``."""

    def __init__(self, hidden_size: int, rank: int, *,
                 dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        if self.rank <= 0:
            self.register_parameter("u", None)
            self.register_parameter("v", None)
            return
        dtype = dtype or torch.float32
        self.u = nn.Parameter(torch.zeros(self.hidden_size, self.rank, dtype=dtype))
        self.v = nn.Parameter(torch.zeros(self.rank, self.hidden_size, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rank <= 0:
            return x
        u = self.u.to(device=x.device, dtype=x.dtype)
        v = self.v.to(device=x.device, dtype=x.dtype)
        return x + (x @ u) @ v


def apply_adapter_to_output(adapter: nn.Module, output: Any, *,
                            mode: str = "residual_pair") -> Any:
    """Apply an adapter to common decoder-layer output shapes.

    vLLM pre-norm decoder layers often return ``(hidden_states, residual)``
    and defer the residual add into the next RMSNorm. A residual-stream adapter
    must transform both tensors so ``T(a + b) = T(a) + T(b)``.
    """

    if isinstance(output, torch.Tensor):
        return adapter(output)
    if isinstance(output, tuple):
        values = list(output)
        if mode == "hidden_only":
            if values and isinstance(values[0], torch.Tensor):
                values[0] = adapter(values[0])
        else:
            for idx, value in enumerate(values):
                if isinstance(value, torch.Tensor):
                    values[idx] = adapter(value)
        return tuple(values)
    if isinstance(output, list):
        values = list(output)
        for idx, value in enumerate(values):
            if isinstance(value, torch.Tensor):
                values[idx] = adapter(value)
        return values
    if isinstance(output, dict):
        values = dict(output)
        for key in ("hidden_states", "residual"):
            value = values.get(key)
            if isinstance(value, torch.Tensor):
                values[key] = adapter(value)
        return values
    return output


def infer_hidden_size_from_config(config: Mapping[str, Any]) -> int:
    for key in ("hidden_size", "d_model", "n_embd"):
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    for key in ("text_config", "language_model_config", "llm_config"):
        nested = config.get(key)
        if isinstance(nested, Mapping):
            try:
                return infer_hidden_size_from_config(nested)
            except ValueError:
                pass
    raise ValueError("could not infer hidden size from config")


def read_base_architectures(config: Mapping[str, Any]) -> tuple[str, ...]:
    arch = config.get("architectures") or ()
    if isinstance(arch, str):
        return (arch,)
    return tuple(str(x) for x in arch)


def patch_config_for_residual_adapter(config: Mapping[str, Any],
                                      manifest: ResidualAdapterManifest,
                                      *,
                                      manifest_file: str = MANIFEST_FILENAME
                                      ) -> dict[str, Any]:
    out = json.loads(json.dumps(dict(config)))
    out["architectures"] = [PLUGIN_ARCHITECTURE]
    out[CONFIG_KEY] = {
        "base_architectures": list(manifest.base_architectures),
        "manifest_file": manifest_file,
        "plugin_architecture": PLUGIN_ARCHITECTURE,
        "version": int(manifest.version),
    }
    return out


def hardlink_or_copy_tree(src: str | Path, dst: str | Path,
                          *,
                          overwrite: bool = False,
                          copy_filenames: Iterable[str] = ()) -> None:
    src_p = Path(src)
    dst_p = Path(dst)
    copy_names = set(copy_filenames)
    if dst_p.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {dst_p}")
        shutil.rmtree(dst_p)
    dst_p.mkdir(parents=True)
    for item in src_p.iterdir():
        target = dst_p / item.name
        if item.is_dir():
            hardlink_or_copy_tree(item, target, copy_filenames=copy_names)
            continue
        if item.name in copy_names:
            shutil.copy2(item, target)
            continue
        try:
            target.hardlink_to(item)
        except OSError:
            shutil.copy2(item, target)


def make_identity_manifest(config: Mapping[str, Any],
                           module_paths: Iterable[str] = (),
                           *,
                           rank: int = 0) -> ResidualAdapterManifest:
    base_arch = read_base_architectures(config)
    return ResidualAdapterManifest(
        version=1,
        base_architectures=base_arch,
        adapters=tuple(
            ResidualAdapterSpec.from_module_path(path, rank=rank)
            for path in module_paths
        ),
    )
