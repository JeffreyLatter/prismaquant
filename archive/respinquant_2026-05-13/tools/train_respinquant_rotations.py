#!/usr/bin/env python3
"""Train ReSpinQuant-style layer rotations on calibration CE loss.

This is the GPU-only training half of the ReSpin path. It learns dense
Hadamard-initialized orthogonal rotations with a Cayley optimizer while
fake-quantizing layer-boundary activations using a straight-through estimator.

The current topology is ``single_boundary_basis``: one learned residual basis
per decoder-layer input boundary. That is a trained layer-wise rotation
checkpoint, but not the full paper topology with separate MHSA/FFN bases and
the intermediate attention rotation. Export code must keep treating the result
as research until the exact residual edges are representable by the runtime
adapter path.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismaquant.gpu_guard import require_cuda_hot_path
from prismaquant.calibration_data import load_wikitext_calibration_windowed
from prismaquant.respinquant_core import (
    CayleySGD,
    TrainableRotation,
    fake_quantize_activation,
    rotation_metadata,
    rotation_state_dict,
)
from prismaquant.sensitivity_probe import load_calibration


def _dtype_from_name(name: str) -> torch.dtype:
    norm = name.strip().lower().replace("-", "_")
    if norm in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if norm in {"fp16", "float16", "half"}:
        return torch.float16
    if norm in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _nested_get(obj: Any, path: str):
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _find_decoder_layers(model) -> tuple[str, torch.nn.ModuleList]:
    candidates = (
        "model.layers",
        "language_model.model.layers",
        "model.language_model.layers",
        "transformer.h",
        "gpt_neox.layers",
    )
    for path in candidates:
        value = _nested_get(model, path)
        if isinstance(value, torch.nn.ModuleList):
            return path, value
    raise RuntimeError(
        "could not find decoder layer ModuleList; add this architecture to "
        "tools/train_respinquant_rotations.py"
    )


def _parse_layers(spec: str, n_layers: int) -> list[int]:
    raw = spec.strip().lower()
    if raw in {"all", "*"}:
        return list(range(n_layers))
    if raw in {"", "none"}:
        return []
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return sorted(idx for idx in out if 0 <= idx < n_layers)


def _replace_hidden(args, kwargs, hidden):
    if "hidden_states" in kwargs:
        new_kwargs = dict(kwargs)
        new_kwargs["hidden_states"] = hidden
        return args, new_kwargs
    if not args:
        return args, kwargs
    new_args = list(args)
    new_args[0] = hidden
    return tuple(new_args), kwargs


def _install_boundary_fake_quant_hooks(
    layers: torch.nn.ModuleList,
    rotations: dict[str, TrainableRotation],
    *,
    bits: int,
    symmetric: bool,
) -> list[Any]:
    handles: list[Any] = []

    def make_hook(name: str):
        rotation = rotations[name]

        def hook(_module, args, kwargs):
            hidden = kwargs.get("hidden_states") if "hidden_states" in kwargs else (
                args[0] if args else None
            )
            if not isinstance(hidden, torch.Tensor):
                return args, kwargs
            rotated = rotation(hidden)
            quantized = fake_quantize_activation(
                rotated,
                bits=bits,
                symmetric=symmetric,
            )
            restored = rotation(quantized, transpose=True)
            return _replace_hidden(args, kwargs, restored)

        return hook

    for name, rotation in rotations.items():
        del rotation
        idx = int(name.rsplit(".", 1)[-1])
        handles.append(layers[idx].register_forward_pre_hook(
            make_hook(name),
            with_kwargs=True,
        ))
    return handles


def train_rotations(args: argparse.Namespace) -> dict[str, Any]:
    device = require_cuda_hot_path("respinquant rotation training", args.device)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        import accelerate  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "respinquant rotation training requires accelerate so the model "
            "can be placed directly on CUDA with device_map. Install "
            "`accelerate` in the CUDA environment; refusing a CPU-staged "
            "fallback."
        ) from exc

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=True,
    )
    if args.dataset.strip().lower() in {"wikitext-2", "wikitext2", "wiki2"}:
        calib = load_wikitext_calibration_windowed(
            tokenizer,
            n_samples=args.n_samples,
            seqlen=args.seqlen,
            seed=int(args.seed),
        )
    else:
        calib = load_calibration(
            tokenizer,
            args.dataset,
            n_samples=args.n_samples,
            seqlen=args.seqlen,
        )
    if not isinstance(calib, torch.Tensor):
        raise RuntimeError("rotation trainer currently expects tensor calibration ids")
    calib = calib.to(device=device, non_blocking=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=_dtype_from_name(args.dtype),
        trust_remote_code=True,
        device_map={"": str(device)},
    )
    model.eval()
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    for param in model.parameters():
        param.requires_grad_(False)

    layer_path, layers = _find_decoder_layers(model)
    layer_ids = _parse_layers(args.layers, len(layers))
    if not layer_ids:
        raise RuntimeError("no layers selected for rotation training")
    hidden = int(getattr(model.config, "hidden_size", 0) or getattr(model.config, "d_model", 0))
    if hidden <= 0:
        text_cfg = getattr(model.config, "text_config", None)
        hidden = int(getattr(text_cfg, "hidden_size", 0) if text_cfg is not None else 0)
    if hidden <= 0:
        raise RuntimeError("could not infer hidden size from model config")

    rotations = {
        f"layer.{idx}": TrainableRotation(
            hidden,
            # ReSpinQuant relies on relative transitions staying near identity:
            # all layer rotations start from the same Hadamard basis, then
            # Cayley optimization lets them drift as needed.
            seed=int(args.seed),
            device=device,
        )
        for idx in layer_ids
    }
    trainable = [rot.weight for rot in rotations.values()]
    optimizer = CayleySGD(
        trainable,
        lr=float(args.lr),
        momentum=float(args.momentum),
        grad_clip=args.grad_clip,
    )
    handles = _install_boundary_fake_quant_hooks(
        layers,
        rotations,
        bits=int(args.activation_bits),
        symmetric=bool(args.symmetric_activation),
    )

    losses: list[float] = []
    try:
        for step in range(int(args.steps)):
            lr = float(args.lr) * 0.5 * (
                1.0 + math.cos(math.pi * float(step) / max(float(args.steps), 1.0))
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            start = (step * int(args.batch_size)) % calib.shape[0]
            end = start + int(args.batch_size)
            if end <= calib.shape[0]:
                input_ids = calib[start:end]
            else:
                input_ids = torch.cat([calib[start:], calib[:end - calib.shape[0]]], dim=0)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=_dtype_from_name(args.dtype)):
                out = model(input_ids=input_ids, labels=input_ids)
                loss = out.loss
            loss.backward()
            optimizer.step()
            value = float(loss.detach().float().item())
            losses.append(value)
            if (step + 1) % max(1, int(args.log_every)) == 0 or step == 0:
                print(
                    f"[respin-train] step={step + 1}/{args.steps} "
                    f"loss={value:.6f} lr={lr:.6g}",
                    flush=True,
                )
    finally:
        for handle in handles:
            handle.remove()

    rotation_path = out_dir / "respin_rotations.pt"
    torch.save(rotation_state_dict(rotations), rotation_path)
    metadata = rotation_metadata(
        rotations,
        extra={
            "schema": "prismaquant.respinquant.rotation_checkpoint.v1",
            "paper_faithful": False,
            "paper_fidelity_note": (
                "Trained Hadamard-initialized Cayley rotations with CE and "
                "activation fake quantization, but topology is "
                "single_boundary_basis rather than the full paper MHSA/FFN/"
                "intermediate-attention topology."
            ),
            "method_family": "ReSpinQuant/SpinQuant rotation training",
            "topology": "single_boundary_basis",
            "model": str(args.model),
            "dataset": str(args.dataset),
            "n_samples": int(args.n_samples),
            "seqlen": int(args.seqlen),
            "steps": int(args.steps),
            "lr": float(args.lr),
            "activation_bits": int(args.activation_bits),
            "activation_quantization": (
                "symmetric" if args.symmetric_activation else "asymmetric"
            ),
            "batch_size": int(args.batch_size),
            "layer_path": layer_path,
            "layers": layer_ids,
            "loss_initial": losses[0] if losses else None,
            "loss_final": losses[-1] if losses else None,
            "rotation_path": str(rotation_path),
        },
    )
    metadata_path = out_dir / "respin_rotation_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"[respin-train] wrote {rotation_path}", flush=True)
    print(f"[respin-train] wrote {metadata_path}", flush=True)
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="Source Hugging Face model path.")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for rotation checkpoint and metadata.")
    parser.add_argument("--dataset", default="wikitext-2",
                        help="Calibration source accepted by sensitivity_probe.load_calibration.")
    parser.add_argument("--n-samples", type=int, default=32,
                        help="Calibration windows to keep resident on GPU.")
    parser.add_argument("--seqlen", type=int, default=512,
                        help="Calibration sequence length.")
    parser.add_argument("--steps", type=int, default=100,
                        help="Cayley optimization steps. Paper default is 100.")
    parser.add_argument("--lr", type=float, default=15.0,
                        help="Initial cosine-scheduled learning rate. Paper uses 15.")
    parser.add_argument("--momentum", type=float, default=0.0,
                        help="Optional Cayley momentum.")
    parser.add_argument("--grad-clip", type=float, default=None,
                        help="Optional gradient norm clip.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Calibration microbatch size.")
    parser.add_argument("--activation-bits", type=int, default=4,
                        help="Activation fake-quant bits used during training.")
    parser.add_argument("--symmetric-activation", action="store_true",
                        help="Use symmetric activation fake quantization instead of asymmetric.")
    parser.add_argument("--layers", default="all",
                        help="Layer set to train, e.g. all, 0-3, or 0,5,9.")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=("bfloat16", "bf16", "float16", "fp16", "float32", "fp32"),
                        help="Model/autocast dtype.")
    parser.add_argument("--device", default="cuda",
                        help="Training device. CUDA is required.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Hadamard sign seed.")
    parser.add_argument("--log-every", type=int, default=1,
                        help="Progress logging interval.")
    args = parser.parse_args(argv)

    summary = train_rotations(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
