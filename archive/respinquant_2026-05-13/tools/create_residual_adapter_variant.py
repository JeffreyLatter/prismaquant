#!/usr/bin/env python3
"""Create an optional vLLM residual-adapter model variant.

The output directory hardlinks the source artifact and patches only metadata
plus optional adapter tensors. This keeps "with plugin" and "without plugin"
variants cheap to spin without duplicating model weights.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismaquant.residual_adapter import (
    CONFIG_KEY,
    MANIFEST_FILENAME,
    ResidualAdapterManifest,
    hardlink_or_copy_tree,
    infer_hidden_size_from_config,
    make_identity_manifest,
    patch_config_for_residual_adapter,
)


ADAPTER_TENSOR_FILE = "prisma-residual-adapters.safetensors"


def _parse_dtype(name: str) -> torch.dtype:
    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported adapter dtype: {name}")


def _respin_givens_tensors(hidden_size: int,
                           rank: int,
                           *,
                           dtype: torch.dtype,
                           generator: torch.Generator,
                           angle: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Build an untrained low-rank smoke transition from disjoint Givens rotations."""

    u = torch.zeros(hidden_size, rank, dtype=torch.float32)
    v = torch.zeros(rank, hidden_size, dtype=torch.float32)
    pair_count = min(rank // 2, hidden_size // 2)
    if pair_count <= 0 or angle == 0.0:
        return u.to(dtype=dtype), v.to(dtype=dtype)

    coords = torch.randperm(hidden_size, generator=generator)[:2 * pair_count]
    for pair_idx in range(pair_count):
        a = int(coords[2 * pair_idx])
        b = int(coords[2 * pair_idx + 1])
        col_a = 2 * pair_idx
        col_b = col_a + 1
        sign = -1.0 if float(torch.rand((), generator=generator)) < 0.5 else 1.0
        theta = sign * float(angle)
        c = math.cos(theta)
        s = math.sin(theta)

        u[a, col_a] = 1.0
        u[b, col_b] = 1.0
        v[col_a, a] = c - 1.0
        v[col_a, b] = -s
        v[col_b, a] = s
        v[col_b, b] = c - 1.0

    return u.to(dtype=dtype), v.to(dtype=dtype)


def _module_paths(site_args: Iterable[str], sites_arg: str | None) -> list[str]:
    paths: list[str] = []
    for item in site_args:
        if item:
            paths.append(item.strip())
    if sites_arg:
        paths.extend(part.strip() for part in sites_arg.split(",") if part.strip())
    seen: set[str] = set()
    out: list[str] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def _write_adapter_tensors(output: Path,
                           manifest: ResidualAdapterManifest,
                           hidden_size: int,
                           dtype: torch.dtype,
                           *,
                           initializer: str = "zero",
                           angle: float = 0.0,
                           seed: int = 0) -> tuple[str | None, int]:
    tensors: dict[str, torch.Tensor] = {}
    total_size = 0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    for spec in manifest.adapters:
        if not spec.enabled or spec.rank <= 0:
            continue
        if not spec.u_name or not spec.v_name:
            raise ValueError(f"rank-{spec.rank} adapter missing tensor names: {spec}")
        if initializer == "zero":
            u = torch.zeros(hidden_size, spec.rank, dtype=dtype)
            v = torch.zeros(spec.rank, hidden_size, dtype=dtype)
        elif initializer == "respin-givens":
            u, v = _respin_givens_tensors(
                hidden_size,
                spec.rank,
                dtype=dtype,
                generator=generator,
                angle=angle,
            )
        else:
            raise ValueError(f"unsupported adapter initializer: {initializer}")
        tensors[spec.u_name] = u
        tensors[spec.v_name] = v
        total_size += u.numel() * u.element_size()
        total_size += v.numel() * v.element_size()
    if not tensors:
        return None, 0
    tensor_file = output / ADAPTER_TENSOR_FILE
    save_file(tensors, str(tensor_file))
    return ADAPTER_TENSOR_FILE, total_size


def _update_safetensors_index(output: Path,
                              manifest: ResidualAdapterManifest,
                              tensor_file: str | None,
                              tensor_bytes: int) -> None:
    if tensor_file is None:
        return
    index_path = output / "model.safetensors.index.json"
    if not index_path.is_file():
        return
    data = json.loads(index_path.read_text())
    weight_map = data.setdefault("weight_map", {})
    for spec in manifest.adapters:
        if spec.rank <= 0:
            continue
        if spec.u_name:
            weight_map[spec.u_name] = tensor_file
        if spec.v_name:
            weight_map[spec.v_name] = tensor_file
    metadata = data.setdefault("metadata", {})
    old_total = metadata.get("total_size")
    if isinstance(old_total, int):
        metadata["total_size"] = old_total + tensor_bytes
    index_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def create_variant(model_dir: str | Path,
                   output: str | Path,
                   *,
                   module_paths: Iterable[str] = (),
                   rank: int = 0,
                   dtype: str = "bfloat16",
                   initializer: str = "zero",
                   angle: float = 0.0,
                   seed: int = 0,
                   overwrite: bool = False) -> dict[str, object]:
    src = Path(model_dir)
    dst = Path(output)
    config_path = src / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing config.json in {src}")
    config = json.loads(config_path.read_text())
    rank = int(rank)
    if rank < 0:
        raise ValueError("--rank must be >= 0")

    manifest = make_identity_manifest(config, module_paths, rank=rank)
    hidden_size = infer_hidden_size_from_config(config)

    hardlink_or_copy_tree(
        src,
        dst,
        overwrite=overwrite,
        copy_filenames={"config.json", "model.safetensors.index.json"},
    )
    manifest.write(dst / MANIFEST_FILENAME)
    patched = patch_config_for_residual_adapter(
        config,
        manifest,
        manifest_file=MANIFEST_FILENAME,
    )
    (dst / "config.json").write_text(json.dumps(patched, indent=2, sort_keys=True) + "\n")

    tensor_file, tensor_bytes = _write_adapter_tensors(
        dst,
        manifest,
        hidden_size,
        _parse_dtype(dtype),
        initializer=initializer,
        angle=float(angle),
        seed=int(seed),
    )
    _update_safetensors_index(dst, manifest, tensor_file, tensor_bytes)

    return {
        "source": str(src),
        "output": str(dst),
        "architecture": patched["architectures"][0],
        "base_architectures": list(manifest.base_architectures),
        "adapter_count": len(manifest.adapters),
        "rank": rank,
        "initializer": initializer,
        "angle": float(angle),
        "seed": int(seed),
        "adapter_tensor_file": tensor_file,
        "config_key": CONFIG_KEY,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True,
                        help="Source Hugging Face/vLLM model artifact.")
    parser.add_argument("--output", required=True,
                        help="Output artifact directory to create.")
    parser.add_argument("--site", action="append", default=[],
                        help="Module path to wrap. May be repeated.")
    parser.add_argument("--sites", default=None,
                        help="Comma-separated module paths to wrap.")
    parser.add_argument("--rank", type=int, default=0,
                        help="Adapter rank. Rank 0 creates an identity smoke artifact.")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=("bfloat16", "bf16", "float16", "fp16", "float32", "fp32"),
                        help="Adapter tensor dtype for rank > 0.")
    parser.add_argument("--initializer", default="zero",
                        choices=("zero", "respin-givens"),
                        help=(
                            "Adapter tensor initializer. respin-givens is an "
                            "untrained runtime smoke, not paper-faithful "
                            "ReSpinQuant."
                        ))
    parser.add_argument("--angle", type=float, default=0.0,
                        help="Givens angle in radians for --initializer respin-givens.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Deterministic initializer seed.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace the output directory if it already exists.")
    args = parser.parse_args(argv)

    summary = create_variant(
        args.model_dir,
        args.output,
        module_paths=_module_paths(args.site, args.sites),
        rank=args.rank,
        dtype=args.dtype,
        initializer=args.initializer,
        angle=args.angle,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
