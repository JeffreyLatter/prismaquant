#!/usr/bin/env python3
"""Run the CPU-only tier-3 within-Linear row-dispersion pilot.

Examples::

    PYTHONPATH=. python3 scripts/tier3_row_dispersion_pilot.py \
      --weights synthetic:64x128 --acts synthetic:256x128 \
      --quantized cheap=synthetic:64x128:0.10 \
                  expensive=synthetic:64x128:0.03 \
      --bytes-per-row cheap=64 --bytes-per-row expensive=128 \
      --out /path/to/output

``LABEL=PATH`` sources accept a single-tensor ``.pt`` or ``.safetensors``
file.  Append ``::tensor_key`` to select from a multi-tensor file.  Synthetic
weight/activation specs are ``synthetic:ROWSxCOLS``.  A synthetic
reconstruction adds deterministic shared noise to W and may append its noise
scale, as in ``synthetic:ROWSxCOLS:0.05``.
"""
from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

from prismaquant.row_dispersion import (
    assert_error_decomposition,
    per_row_error,
    split_prize_curve,
    tail_metrics,
)

DEFAULT_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20)
_SYNTHETIC = re.compile(
    r"^synthetic:(?P<rows>\d+)[xX](?P<cols>\d+)"
    r"(?::(?P<noise>\d+(?:\.\d+)?(?:[eE][-+]?\d+)?))?$"
)
_LABEL = re.compile(r"^[A-Za-z0-9_.-]+$")


def _parse_synthetic(spec: str) -> tuple[int, int, float | None] | None:
    match = _SYNTHETIC.fullmatch(spec)
    if match is None:
        return None
    rows, cols = int(match.group("rows")), int(match.group("cols"))
    if rows <= 0 or cols <= 0:
        raise ValueError("synthetic dimensions must be positive")
    noise = match.group("noise")
    return rows, cols, None if noise is None else float(noise)


def _single_tensor(payload: Any, *, source: str, key: str | None) -> torch.Tensor:
    if isinstance(payload, torch.Tensor):
        if key is not None:
            raise ValueError(f"{source} is a bare tensor; ::{key} is invalid")
        return payload
    if isinstance(payload, dict):
        tensors = {str(name): value for name, value in payload.items()
                   if isinstance(value, torch.Tensor)}
        if key is not None:
            if key not in tensors:
                raise KeyError(f"tensor {key!r} not found in {source}")
            return tensors[key]
        if len(tensors) == 1:
            return next(iter(tensors.values()))
        raise ValueError(
            f"{source} contains {len(tensors)} tensors; select one with ::KEY"
        )
    raise TypeError(f"{source} did not contain a tensor")


def _load_tensor(source: str) -> torch.Tensor:
    path_text, separator, key = source.rpartition("::")
    if not separator:
        path_text, key = source, None
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth", ".pkl"}:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    elif suffix == ".safetensors":
        from safetensors.torch import load_file

        payload = load_file(str(path), device="cpu")
    else:
        raise ValueError(
            f"unsupported tensor file {path}; expected .pt or .safetensors"
        )
    return _single_tensor(payload, source=str(path), key=key).detach().cpu()


def _load_base(source: str, *, seed: int) -> torch.Tensor:
    synthetic = _parse_synthetic(source)
    if synthetic is None:
        return _load_tensor(source)
    rows, cols, noise = synthetic
    if noise is not None:
        raise ValueError("noise scale is valid only for synthetic reconstructions")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(rows, cols, generator=generator, dtype=torch.float64)


def _load_reconstruction(source: str, W: torch.Tensor) -> torch.Tensor:
    synthetic = _parse_synthetic(source)
    if synthetic is None:
        return _load_tensor(source)
    rows, cols, noise = synthetic
    if (rows, cols) != tuple(W.shape):
        raise ValueError(
            f"synthetic reconstruction shape {(rows, cols)} != W {tuple(W.shape)}"
        )
    scale = 0.05 if noise is None else noise
    if scale < 0:
        raise ValueError("synthetic reconstruction noise must be non-negative")
    # Every synthetic format uses the same perturbation direction.  Changing
    # scale therefore gives a row-wise ordered pair suitable for smoke tests.
    generator = torch.Generator(device="cpu").manual_seed(2)
    perturbation = torch.randn(
        W.shape, generator=generator, dtype=W.dtype, device="cpu"
    )
    return W + scale * perturbation


def _labeled_source(value: str) -> tuple[str, str]:
    label, separator, source = value.partition("=")
    if not separator or not label or not source:
        raise argparse.ArgumentTypeError(
            f"expected LABEL=SOURCE for --quantized, got {value!r}"
        )
    if not _LABEL.fullmatch(label):
        raise argparse.ArgumentTypeError(
            f"format label {label!r} must match {_LABEL.pattern}"
        )
    return label, source


def _label_float(value: str) -> tuple[str, float]:
    label, separator, raw = value.partition("=")
    if not separator or not _LABEL.fullmatch(label):
        raise argparse.ArgumentTypeError(
            f"expected LABEL=FLOAT for --bytes-per-row, got {value!r}"
        )
    try:
        number = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not np.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("bytes per row must be finite and non-negative")
    return label, number


def _fractions(value: str) -> list[float]:
    try:
        fractions = [float(item) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not fractions or any(not 0.0 <= fraction <= 1.0 for fraction in fractions):
        raise argparse.ArgumentTypeError("fractions must be a comma-separated list in [0, 1]")
    return fractions


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="W tensor path or synthetic:NxK")
    parser.add_argument("--acts", required=True, help="X tensor path or synthetic:SxK")
    parser.add_argument(
        "--quantized", required=True, nargs="+", type=_labeled_source,
        metavar="LABEL=SOURCE",
        help="one or more reconstructed weights, each with a format label",
    )
    parser.add_argument(
        "--bytes-per-row", action="append", default=[], type=_label_float,
        metavar="LABEL=BYTES",
        help=("serialized format cost per output row; defaults to the dense "
              "reconstruction tensor's row bytes"),
    )
    parser.add_argument(
        "--fractions", type=_fractions,
        default=list(DEFAULT_FRACTIONS),
        help="comma-separated routed-row fractions (default: 0.01,0.02,0.05,0.1,0.2)",
    )
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    return parser


def _print_table(formats: dict[str, dict[str, Any]]) -> None:
    header = (
        f"{'format':<20} {'mean':>12} {'p99':>12} {'p99/p50':>12} "
        f"{'CV':>10} {'Gini':>10} {'B/row':>12}"
    )
    print(header)
    print("-" * len(header))
    for label, record in formats.items():
        metrics = record["tail_metrics"]
        print(
            f"{label:<20} {metrics['mean']:12.5g} {metrics['p99']:12.5g} "
            f"{metrics['ratio_p99_over_p50']:12.5g} "
            f"{metrics['coefficient_of_variation']:10.4f} "
            f"{metrics['gini']:10.4f} {record['bytes_per_row']:12.3f}"
        )


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats with JSON ``null`` recursively."""
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    W = _load_base(args.weights, seed=0)
    X = _load_base(args.acts, seed=1)
    if W.ndim != 2 or X.ndim != 2 or X.shape[1] != W.shape[1]:
        raise ValueError(
            f"expected W [N,K] and X [S,K], got {tuple(W.shape)} and {tuple(X.shape)}"
        )

    byte_overrides = dict(args.bytes_per_row)
    unknown_overrides = set(byte_overrides) - {label for label, _ in args.quantized}
    if unknown_overrides:
        raise ValueError(
            "--bytes-per-row names absent from --quantized: "
            + ", ".join(sorted(unknown_overrides))
        )

    args.out.mkdir(parents=True, exist_ok=True)
    errors: dict[str, torch.Tensor] = {}
    format_records: dict[str, dict[str, Any]] = {}
    for label, source in args.quantized:
        if label in errors:
            raise ValueError(f"duplicate format label {label!r}")
        W_hat = _load_reconstruction(source, W)
        if W_hat.shape != W.shape:
            raise ValueError(
                f"{label} reconstruction shape {tuple(W_hat.shape)} != W {tuple(W.shape)}"
            )
        W_hat = W_hat.to(dtype=W.dtype, device="cpu")
        error = per_row_error(X, W, W_hat)
        assert_error_decomposition(X, W, W_hat, error)
        error_file = f"{label}_per_row_error.npy"
        np.save(args.out / error_file, error.numpy())
        bytes_per_row = byte_overrides.get(
            label, float(W_hat[0].numel() * W_hat.element_size())
        )
        errors[label] = error
        format_records[label] = {
            "source": source,
            "error_file": error_file,
            "total_error": float(error.sum()),
            "bytes_per_row": bytes_per_row,
            "bytes_per_row_source": (
                "cli_override" if label in byte_overrides else "dense_reconstruction"
            ),
            "tail_metrics": tail_metrics(error),
        }

    ordered_labels = sorted(errors, key=lambda label: (format_records[label]["bytes_per_row"], label))
    prize_curves = []
    for cheap_label, expensive_label in combinations(ordered_labels, 2):
        prize_curves.append({
            "cheap_format": cheap_label,
            "expensive_format": expensive_label,
            "curve": split_prize_curve(
                errors[cheap_label],
                errors[expensive_label],
                format_records[cheap_label]["bytes_per_row"],
                format_records[expensive_label]["bytes_per_row"],
                args.fractions,
            ),
        })

    summary = {
        "schema": "prismaquant.row_dispersion.v1",
        "inputs": {
            "weights": args.weights,
            "weights_shape": list(W.shape),
            "acts": args.acts,
            "acts_shape": list(X.shape),
        },
        "fractions": args.fractions,
        "formats": format_records,
        "prize_curves": prize_curves,
    }
    (args.out / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    _print_table(format_records)
    if prize_curves:
        violations = sum(
            len(item["curve"]["violations"]) for item in prize_curves
        )
        print(f"\nprize curves: {len(prize_curves)}; row-wise violations: {violations}")
    print(f"wrote {args.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
