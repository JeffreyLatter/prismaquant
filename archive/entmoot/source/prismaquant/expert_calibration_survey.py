"""Cheap router survey for expert-balanced calibration selection.

The expensive REAP/Fisher passes need a good calibration mix. This module
adds a lightweight pre-pass that runs candidate samples through the model
with router hooks only, then emits one JSONL row per sample containing
per-router expert hit mass. The output feeds
``prismaquant.expert_calibration_select``.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from prismaquant.adaptive_sampling import _GLOBAL_DOMAIN, infer_chunk_domain
from prismaquant.observers import saliency_from_packed_moe
from prismaquant.sensitivity_probe import (
    discover_moe_structure,
    read_top_k,
    resolve_execution_device,
    stage_text_only,
)


class RouterSurveyTracker:
    """Forward-hook tracker that records per-sample router expert mass."""

    def __init__(
        self,
        model: nn.Module,
        routers: Sequence[str],
        *,
        top_k: int,
        softmax_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.top_k = int(top_k)
        self.softmax_dtype = softmax_dtype
        self._handles: list[Any] = []
        self._num_experts_by_router: dict[str, int] = {}
        self._hits: dict[str, defaultdict[int, float]] = {}
        self._tokens: defaultdict[str, int] = defaultdict(int)

        for router_qname in routers:
            try:
                module = model.get_submodule(router_qname)
            except AttributeError:
                continue
            num_experts = _infer_num_experts(module)
            if num_experts is None or num_experts <= 0:
                continue
            self._num_experts_by_router[router_qname] = num_experts
            self._hits[router_qname] = defaultdict(float)
            self._handles.append(
                module.register_forward_hook(self._make_hook(router_qname))
            )

    @property
    def routers(self) -> list[str]:
        return sorted(self._num_experts_by_router)

    def reset(self) -> None:
        for router_qname in self._hits:
            self._hits[router_qname].clear()
        self._tokens.clear()

    def hits(
        self,
        *,
        normalize_by_tokens: bool = True,
        min_mass: float = 0.0,
    ) -> dict[str, dict[int, float]]:
        out: dict[str, dict[int, float]] = {}
        for router_qname, per_expert in self._hits.items():
            denom = 1.0
            if normalize_by_tokens:
                denom = float(max(1, self._tokens.get(router_qname, 0)))
            clean = {
                int(eid): value
                for eid, mass in per_expert.items()
                if (value := float(mass) / denom) > min_mass
            }
            if clean:
                out[router_qname] = dict(sorted(clean.items()))
        return out

    def token_counts(self) -> dict[str, int]:
        return {
            router_qname: int(self._tokens.get(router_qname, 0))
            for router_qname in self._num_experts_by_router
        }

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, router_qname: str):
        def hook(_module, _inp, out):
            scores = out if isinstance(out, torch.Tensor) else out[0]
            if not isinstance(scores, torch.Tensor) or scores.ndim < 2:
                return
            flat = scores.detach().reshape(-1, scores.size(-1))
            if flat.numel() == 0:
                return
            k = min(self.top_k, int(flat.size(-1)))
            if k <= 0:
                return
            topk_v, topk_i = flat.to(self.softmax_dtype).topk(k, dim=-1)
            probs = F.softmax(topk_v, dim=-1)
            weighted = torch.bincount(
                topk_i.reshape(-1),
                weights=probs.reshape(-1).to(torch.float64),
                minlength=int(flat.size(-1)),
            )
            self._tokens[router_qname] += int(flat.size(0))
            target = self._hits[router_qname]
            nz = torch.nonzero(weighted > 0, as_tuple=False).reshape(-1)
            for idx in nz.tolist():
                target[int(idx)] += float(weighted[int(idx)].item())

        return hook


def discover_router_qnames(model: nn.Module) -> list[str]:
    """Return MoE router qnames supported by the existing probe discovery."""

    routers = {router for router, _eid in discover_moe_structure(model).values()}
    for entry in saliency_from_packed_moe(model):
        router = entry.get("router_qname")
        if isinstance(router, str) and router:
            routers.add(router)
    return sorted(routers)


def iter_jsonl_rows(path: str | Path, *, limit: int | None = None):
    source = Path(path)
    yielded = 0
    with source.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            if limit is not None and yielded >= limit:
                break
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"{source}:{line_no} is not a JSON object")
            yielded += 1
            yield line_no, dict(row)


def survey_jsonl(
    model: nn.Module,
    tokenizer,
    input_path: str | Path,
    output_path: str | Path,
    *,
    device: torch.device,
    routers: Sequence[str] | None = None,
    top_k: int | None = None,
    max_length: int = 2048,
    limit: int | None = None,
    normalize_by_tokens: bool = True,
    min_mass: float = 0.0,
    keep_empty: bool = False,
) -> dict[str, Any]:
    """Run a router-only survey over a JSONL calibration candidate file."""

    router_qnames = (
        list(routers) if routers is not None else discover_router_qnames(model)
    )
    if not router_qnames:
        raise ValueError("no MoE routers discovered for survey")
    tracker = RouterSurveyTracker(
        model,
        router_qnames,
        top_k=top_k or read_top_k(model, default=2),
    )

    n_seen = 0
    n_written = 0
    n_empty = 0
    out_path = Path(output_path)
    try:
        with out_path.open("w", encoding="utf-8") as out_f:
            for line_no, row in iter_jsonl_rows(input_path, limit=limit):
                n_seen += 1
                text = text_from_row(row, tokenizer)
                if not text:
                    n_empty += 1
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
                tracker.reset()
                with torch.inference_mode():
                    model(**kwargs)
                hits = tracker.hits(
                    normalize_by_tokens=normalize_by_tokens,
                    min_mass=min_mass,
                )
                if not hits and not keep_empty:
                    n_empty += 1
                    continue

                out_row = survey_row_from_hits(
                    row,
                    hits,
                    source_path=str(input_path),
                    line_no=line_no,
                    router_tokens=tracker.token_counts(),
                )
                out_f.write(json.dumps(out_row, sort_keys=True) + "\n")
                n_written += 1
    finally:
        tracker.remove_hooks()

    return {
        "input": str(input_path),
        "output": str(output_path),
        "rows_seen": n_seen,
        "rows_written": n_written,
        "rows_without_hits": n_empty,
        "routers": tracker.routers,
        "top_k": int(top_k or read_top_k(model, default=2)),
        "normalize_by_tokens": bool(normalize_by_tokens),
    }


def survey_row_from_hits(
    row: Mapping[str, Any],
    hits: Mapping[str, Mapping[int, float]],
    *,
    source_path: str,
    line_no: int,
    router_tokens: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    out = dict(row)
    out.setdefault("id", _row_id(row, source_path=source_path, line_no=line_no))
    out.setdefault("domain", _row_domain(row, source_path=source_path))
    out["source"] = source_path
    out["source_line"] = int(line_no)
    out["hits"] = {
        router: {
            str(eid): float(mass)
            for eid, mass in sorted(per_expert.items(), key=lambda kv: int(kv[0]))
            if math.isfinite(float(mass)) and float(mass) > 0.0
        }
        for router, per_expert in sorted(hits.items())
    }
    if router_tokens is not None:
        out["router_tokens"] = {
            router: int(count) for router, count in sorted(router_tokens.items())
        }
    return out


def text_from_row(row: Mapping[str, Any], tokenizer) -> str | None:
    text = row.get("text")
    if isinstance(text, str) and text.strip():
        return text

    messages = row.get("messages") or row.get("conversations")
    if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
        try:
            rendered = tokenizer.apply_chat_template(messages, tokenize=False)
            if isinstance(rendered, str) and rendered.strip():
                return rendered
        except Exception:
            pass
        parts: list[str] = []
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            content = message.get("content") or message.get("value")
            if isinstance(content, str) and content.strip():
                parts.append(content)
        if parts:
            return "\n\n".join(parts)

    for value in row.values():
        if isinstance(value, str) and value.strip():
            return value
    return None


def _infer_num_experts(module: nn.Module) -> int | None:
    if isinstance(module, nn.Linear):
        return int(module.out_features)
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.ndim >= 1:
        return int(weight.shape[0])
    return None


def _row_id(row: Mapping[str, Any], *, source_path: str, line_no: int) -> str:
    for key in ("sample_id", "id", "uid"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return f"{source_path}:{line_no}"


def _row_domain(row: Mapping[str, Any], *, source_path: str) -> str:
    domain = row.get("domain")
    if isinstance(domain, str) and domain:
        return domain
    inferred = infer_chunk_domain(source_path)
    return inferred if inferred else _GLOBAL_DOMAIN


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unknown dtype {name!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit per-sample MoE router hits for calibration selection.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True, help="candidate JSONL path")
    parser.add_argument("--output", required=True, help="survey JSONL path")
    parser.add_argument("--summary", help="optional summary JSON path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--offload-folder", default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-mass", type=float, default=0.0)
    parser.add_argument(
        "--raw-mass",
        action="store_true",
        help="write summed router mass instead of per-token normalized mass",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="write rows even when no router hits were recorded",
    )
    args = parser.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = _dtype_from_name(args.dtype)
    staged = stage_text_only(args.model)
    tokenizer = AutoTokenizer.from_pretrained(staged, trust_remote_code=True)
    load_device_map = args.device_map if args.device_map is not None else args.device
    from_pretrained_kwargs = {
        "torch_dtype": dtype,
        "device_map": load_device_map,
        "low_cpu_mem_usage": False,
        "trust_remote_code": True,
    }
    if args.offload_folder:
        Path(args.offload_folder).mkdir(parents=True, exist_ok=True)
        from_pretrained_kwargs["offload_folder"] = args.offload_folder
        from_pretrained_kwargs["offload_buffers"] = True
        from_pretrained_kwargs.pop("low_cpu_mem_usage", None)
    model = AutoModelForCausalLM.from_pretrained(staged, **from_pretrained_kwargs)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    device = resolve_execution_device(model, args.device)

    summary = survey_jsonl(
        model,
        tokenizer,
        args.dataset,
        args.output,
        device=device,
        top_k=args.top_k,
        max_length=args.max_length,
        limit=args.limit,
        normalize_by_tokens=not args.raw_mass,
        min_mass=args.min_mass,
        keep_empty=args.keep_empty,
    )
    if args.summary:
        Path(args.summary).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
