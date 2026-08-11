"""Build the immutable learned-codebook bundle before any CB render stage.

This is orchestration, not a second model cache: source values are decoded by
the streaming export reader, one dense Linear at a time, and the certified
trainer retains only the tiny canonical-FP16 books.  Cost/cache/KL/export then
open this exact ``.pqcb`` through ``CB_CODEBOOK_BUNDLE``.
"""
from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Mapping, Sequence

import torch

from . import format_registry as fr
from .cb_learned_bundle import (
    CBL_RUNG_POLICY,
    train_and_save_bundle_streaming,
)
from .export_nvfp4_cb import _try_resolve_skeleton
from .export_nvfp4_cb_streaming import _LazySkeleton
from .model_profiles import detect_profile
from .nvfp4_cb_footprint import is_cb_format


_ROUTED_MOE_QNAME = re.compile(r"(?:^|[.])experts(?:[.]|$)")


def _canonical_cb_formats(formats: Sequence[str]) -> tuple[str, ...]:
    result = tuple(sorted({
        fr.get_format(str(name).strip()).name
        for name in formats
        if str(name).strip() and is_cb_format(str(name).strip())
    }))
    if not result:
        raise ValueError("learned bundle build was given no CB formats")
    return result


def build_bundle_from_model(
    *,
    model_dir: str | Path,
    col_weights: Mapping[str, torch.Tensor],
    formats: Sequence[str],
    output: str | Path,
    device: str | torch.device,
) -> object:
    """Stream dense source weights and publish one value-bearing bundle."""

    canonical_formats = _canonical_cb_formats(formats)
    learned_formats = tuple(
        name for name in canonical_formats
        if name.startswith("FP8_CB_")
        and CBL_RUNG_POLICY.get(
            int(name.rsplit("K", 1)[1]), {}
        ).get("enabled") is True
    )
    if not learned_formats:
        raise ValueError(
            "CB_CODEBOOK_SOURCE_SCOPE enables FP8 learned books, but the "
            "requested format menu contains no FP8_CB rung"
        )
    normalized_col = {
        str(name): torch.as_tensor(value)
        for name, value in col_weights.items()
    }
    routed = {
        name for name in normalized_col if _ROUTED_MOE_QNAME.search(name)
    }
    dense_qnames = tuple(sorted(set(normalized_col) - routed))
    if not dense_qnames:
        raise ValueError("learned bundle build found no dense Linear qnames")

    source = _LazySkeleton(model_dir)
    try:
        profile = detect_profile(str(model_dir))
    except Exception:
        profile = None
    resolved: dict[str, str] = {}
    for qname in dense_qnames:
        key = _try_resolve_skeleton(qname, source, profile)
        if key is None:
            raise KeyError(
                f"{qname}: no source weight for learned bundle training"
            )
        shape = source.logical_shape(key)
        if len(shape) != 2:
            raise ValueError(
                f"{qname}: dense learned bundle source must be rank 2, got "
                f"{shape}; routed stacks require a future Gridbook LUT-offset "
                "contract"
            )
        resolved[qname] = key

    target_device = torch.device(device)

    def provide_weight(qname: str) -> torch.Tensor:
        return source.dequant_weight(resolved[qname]).to(target_device)

    return train_and_save_bundle_streaming(
        output,
        qnames=dense_qnames,
        weight_provider=provide_weight,
        col_weights=normalized_col,
        formats=canonical_formats,
        learned_formats=learned_formats,
        routed_moe_qnames=routed,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--col-weights", required=True)
    parser.add_argument("--formats", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    from .gpu_guard import require_cuda_hot_path

    require_cuda_hot_path("build_cb_learned_bundle")
    with open(args.col_weights, "rb") as handle:
        raw_col_weights = pickle.load(handle)
    if not isinstance(raw_col_weights, Mapping):
        raise ValueError("--col-weights must contain a qname -> tensor mapping")
    bundle = build_bundle_from_model(
        model_dir=args.model_dir,
        col_weights=raw_col_weights,
        formats=[item for item in args.formats.split(",") if item.strip()],
        output=args.output,
        device=args.device,
    )
    print(
        f"[cbl-bundle] wrote {bundle.path}: "
        f"{len(bundle.sidecar_tensors)} canonical FP16 tables, "
        f"sha256={bundle.bundle_content_sha256}",
        flush=True,
    )


if __name__ == "__main__":
    main()
