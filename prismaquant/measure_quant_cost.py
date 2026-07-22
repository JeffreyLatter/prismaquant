#!/usr/bin/env python3
"""measure_quant_cost.py — per-(Linear, format) RTN quantization error.

Two execution modes:

  1. BATCHED GPU (default when --device cuda): groups Linears by
     (in_features, out_features) signature, stacks each group into a
     single 3D tensor, and runs ONE torch.bmm per (group, format).
     For a 35B MoE model this reduces ~31 000 tiny kernel launches
     down to ~360, which is the difference between 42-hour GPU runtime
     and ~1-3 minute runtime on unified-memory systems.

  2. UNBATCHED CPU (default when --device cpu): streams weights one at
     a time via the live model, processes sequentially. Simpler, slower
     per-item but avoids any memory-packing cost — fine for systems
     where the GPU path is slower than CPU (e.g. GB10 when operating
     on many sub-millisecond matmuls through unified memory).

Output format is identical between modes: a dict keyed by Linear name,
each entry mapping format name to {weight_mse, output_mse,
rel_output_mse}. When h-detail is supplied, entries may also include
`predicted_dloss` and `fisher_output_mse`.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import re
import signal
import threading
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import format_registry as fr


def _packed_expert_parent_for_projection(profile, projection_name: str) -> str | None:
    if profile is None:
        try:
            from .model_profiles import DefaultProfile
            profile = DefaultProfile()
        except Exception:
            profile = None
    if profile is not None:
        try:
            return profile.packed_expert_parent_for_projection(projection_name)
        except Exception:
            pass
    return None


def canonical_linear_name(name: str, profile=None) -> str:
    """Map live module names onto the probe's canonical naming.

    Qwen3.5/3.6 MoE can unfuse into per-expert:
      experts.<eid>.gate_proj / up_proj / down_proj
    while the probe/cost pipeline historically keys those as:
      experts.gate_up_proj.<eid> / experts.down_proj.<eid>
    """
    m = re.match(r"^(.+\.experts)\.(\d+)\.([^.]+)$", name)
    if not m:
        return name
    prefix, expert_id, proj = m.groups()
    parent = _packed_expert_parent_for_projection(profile, proj)
    if parent is None:
        return name
    return f"{prefix}.{parent}.{expert_id}"


def _accumulate_result(bucket: dict, name: str, fmt: str,
                       weight_mse: float, output_mse: float,
                       rel_output_mse: float,
                       predicted_dloss: float | None = None,
                       fisher_output_mse: float | None = None,
                       output_mse_measured: bool = True):
    per_name = bucket.setdefault(name, {})
    acc = per_name.setdefault(fmt, {
        "_count": 0,
        "_weight_mse_sum": 0.0,
        "_output_mse_sum": 0.0,
        "_rel_output_mse_sum": 0.0,
        "_predicted_dloss_sum": 0.0,
        "_predicted_dloss_count": 0,
        "_fisher_output_mse_sum": 0.0,
        "_fisher_output_mse_count": 0,
        "_output_mse_measured": True,
    })
    acc["_count"] += 1
    acc["_weight_mse_sum"] += weight_mse
    acc["_output_mse_sum"] += output_mse
    acc["_rel_output_mse_sum"] += rel_output_mse
    if predicted_dloss is not None:
        acc["_predicted_dloss_sum"] += predicted_dloss
        acc["_predicted_dloss_count"] += 1
    if fisher_output_mse is not None:
        acc["_fisher_output_mse_sum"] += fisher_output_mse
        acc["_fisher_output_mse_count"] += 1
    acc["_output_mse_measured"] = bool(
        acc["_output_mse_measured"] and output_mse_measured
    )


def _finalize_results(bucket: dict[str, dict]) -> dict[str, dict]:
    out = {}
    for name, per_name in bucket.items():
        out[name] = {}
        for fmt, acc in per_name.items():
            if "error" in acc:
                out[name][fmt] = acc
                continue
            n = max(int(acc.pop("_count", 1)), 1)
            dloss_n = int(acc.pop("_predicted_dloss_count", 0) or 0)
            dloss_sum = acc.pop("_predicted_dloss_sum", 0.0)
            fisher_n = int(acc.pop("_fisher_output_mse_count", 0) or 0)
            fisher_sum = acc.pop("_fisher_output_mse_sum", 0.0)
            output_mse_measured = bool(acc.pop("_output_mse_measured", True))
            entry = {
                "weight_mse": acc.pop("_weight_mse_sum") / n,
                "output_mse": acc.pop("_output_mse_sum") / n,
                "rel_output_mse": acc.pop("_rel_output_mse_sum") / n,
            }
            if not output_mse_measured:
                entry["output_mse_measured"] = False
            if dloss_n > 0:
                # Full per-weight Δloss from the H-detail path. The
                # allocator prefers this scalar over the scalar-proxy
                # fallback when it's present.
                entry["predicted_dloss"] = dloss_sum / dloss_n
            if fisher_n > 0:
                # Fisher row-weighted output reconstruction objective:
                # local output MSE with rows weighted by end-loss
                # gradient² from the probe's h-detail.
                entry["fisher_output_mse"] = fisher_sum / fisher_n
            out[name][fmt] = entry
    return out


class HDetailIndex:
    """Disk-backed Fisher H-diagonal cache — the per-weight equivalent
    of `ActivationIndex`.

    Points at a directory where `sensitivity_probe.FisherAccumulator`
    dumped per-Linear `[out, in]` tensors (and per-packed-expert
    `[E, M]` tensors). `load(name)` returns the H diagonal tensor for
    that Linear on demand."""

    _FNAME_SUB = re.compile(r"[^A-Za-z0-9_-]")

    def __init__(self, detail_dir: "Path", candidate_names):
        self.detail_dir = detail_dir
        self._paths: dict[str, Path] = {}
        for name in candidate_names:
            fname = self._FNAME_SUB.sub("__", name) + ".pt"
            fp = detail_dir / fname
            if fp.is_file():
                self._paths[name] = fp

    def __contains__(self, name: str) -> bool:
        return name in self._paths

    def __len__(self) -> int:
        return len(self._paths)

    def load(self, name: str) -> torch.Tensor:
        blob = torch.load(self._paths[name], map_location="cpu",
                          weights_only=False)
        return self.h_diag_from_blob(blob)

    def load_blob(self, name: str) -> dict:
        """Return the full saved dict (for callers that want g2_per_token,
        kind, version, etc., not just h_diag)."""
        return torch.load(self._paths[name], map_location="cpu",
                          weights_only=False)

    @staticmethod
    def h_diag_from_blob(blob: dict) -> torch.Tensor:
        """Return the Fisher diagonal from an h-detail blob, in PER-TOKEN
        units.

        Both writers (`sensitivity_probe.FisherAccumulator.finalize` and
        `incremental_probe`) now normalize by token count and stamp
        ``units: "per_token"`` via `sensitivity_probe.h_detail_blob`
        (audit M9). Legacy blobs without the marker:

          - ``h_diag``: accepted as per-token — that writer always
            divided by token count. Caveat: pre-v3 sensitivity blobs for
            UNPACKED expert Linears additionally divided by route_prob
            (audit M4) and are indistinguishable from the blob alone;
            regenerate such h-detail dirs if expert rows matter.
          - raw ``H``: the old incremental writer's token-SUMMED
            accumulator — ~n_tokens× hot for this consumer. Refuse
            rather than silently mis-scale predicted_dloss; regenerate
            the h-detail directory with the current probe.
        """
        units = blob.get("units")
        if units is not None and units != "per_token":
            raise ValueError(
                f"h-detail blob {blob.get('name')!r} has unknown units "
                f"{units!r}; this consumer requires 'per_token'.")
        if "h_diag" in blob:
            return blob["h_diag"]
        if "H" in blob:
            if units == "per_token":
                return blob["H"]
            raise ValueError(
                f"h-detail blob {blob.get('name')!r} carries a raw "
                "token-summed 'H' tensor with no units marker (legacy "
                "incremental_probe writer). Its scale is ~n_tokens× off "
                "for predicted_dloss; regenerate the h-detail directory "
                "with the current probe instead of consuming it.")
        raise KeyError("h-detail blob has neither 'h_diag' nor 'H'")


def _normalize_fisher_output_mse_row_weights(
    row_weights: torch.Tensor | None,
    row_indices: torch.Tensor | None,
    n_rows: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Return non-negative per-row Fisher weights normalized to mean 1."""
    if row_weights is None or row_indices is None or n_rows <= 0:
        return None
    try:
        source = row_weights.detach().reshape(-1).to(device=device, dtype=torch.float32)
        idx = row_indices.detach().reshape(-1).to(device=device, dtype=torch.long)
    except Exception:
        return None
    if idx.numel() < n_rows or source.numel() <= 0:
        return None
    idx = idx[:n_rows]
    if int(idx.min().item()) < 0 or int(idx.max().item()) >= int(source.numel()):
        return None
    rw = source.index_select(0, idx)
    rw = torch.nan_to_num(rw, nan=0.0, posinf=0.0, neginf=0.0)
    rw = rw.clamp_min(0.0)
    mean = rw.mean()
    if not torch.isfinite(mean) or float(mean.item()) <= 0.0:
        return None
    rw = rw / mean.clamp_min(1e-12)
    try:
        clip = float(os.environ.get(
            "PRISMAQUANT_FISHER_OUTPUT_MSE_ROW_WEIGHT_CLIP",
            os.environ.get("PRISMAQUANT_FISHER_GPTQ_ROW_WEIGHT_CLIP", "64"),
        ))
    except Exception:
        clip = 64.0
    if clip > 0.0:
        rw = rw.clamp_max(float(clip))
        mean2 = rw.mean()
        if torch.isfinite(mean2) and float(mean2.item()) > 0.0:
            rw = rw / mean2.clamp_min(1e-12)
    return rw


# ---------------------------------------------------------------------------
# Activation cache — lazy path index
# ---------------------------------------------------------------------------
class ActivationIndex:
    """Disk-backed activation cache.

    Building the index walks the cache dir once to map Linear name → path,
    but no tensor data is read until `load()` is called for a specific name.
    This keeps resident memory small even when the cache is 20 GB on disk.

    The probe writes files as `re.sub(r"[^A-Za-z0-9_-]", "__", name) + ".pt"`,
    so we apply the same forward transform to each candidate name from the
    probe stats and check for the file. This avoids ambiguity if a name ever
    contained characters that collapse under the substitution.
    """

    _FNAME_SUB = re.compile(r"[^A-Za-z0-9_-]")

    def __init__(self, cache_dir: Path, candidate_names):
        self.cache_dir = cache_dir
        self._paths: dict[str, Path] = {}
        self._aliases: dict[str, str] = {}

        def _add_name(name: str):
            fname = self._FNAME_SUB.sub("__", name) + ".pt"
            fp = self.cache_dir / fname
            if fp.is_file():
                self._paths[name] = fp

        if isinstance(candidate_names, dict):
            for name, meta in candidate_names.items():
                _add_name(str(name))
                if isinstance(meta, dict):
                    experts_qname = meta.get("_packed_experts_module")
                    if isinstance(experts_qname, str) and experts_qname:
                        _add_name(experts_qname)
                        if experts_qname in self._paths:
                            self._aliases[str(name)] = experts_qname
        else:
            for name in candidate_names:
                _add_name(str(name))

    def _path_for_name(self, name: str) -> Path | None:
        if name in self._paths:
            return self._paths[name]
        alias = self._aliases.get(name)
        if alias is not None and alias in self._paths:
            return self._paths[alias]
        fname = self._FNAME_SUB.sub("__", name) + ".pt"
        fp = self.cache_dir / fname
        if fp.is_file():
            self._paths[name] = fp
            return fp
        return None

    def __contains__(self, name: str) -> bool:
        return self._path_for_name(name) is not None

    def __len__(self) -> int:
        return len(self._paths)

    def load(self, name: str) -> torch.Tensor:
        blob = self.load_blob(name)
        return blob["inputs"]

    def load_blob(self, name: str) -> dict:
        fp = self._path_for_name(name)
        if fp is None:
            raise KeyError(name)
        return torch.load(fp, map_location="cpu", weights_only=False)

    def load_with_row_indices(self, name: str) -> tuple[torch.Tensor, torch.Tensor | None]:
        blob = self.load_blob(name)
        row_indices = blob.get("row_indices") if isinstance(blob, dict) else None
        if not isinstance(row_indices, torch.Tensor):
            row_indices = None
        return blob["inputs"], row_indices

    def names(self):
        return self._paths.keys()


# ---------------------------------------------------------------------------
# Memory-pressure watchdog
# ---------------------------------------------------------------------------
def _read_meminfo() -> dict[str, int]:
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, v = line.partition(":")
            parts = v.strip().split()
            if parts:
                info[k] = int(parts[0]) * 1024  # kB → bytes
    return info


def start_mem_watchdog(swap_grow_limit_mb: int = 256,
                       min_mem_available_mb: int = 1024,
                       interval_s: float = 2.0):
    """Background thread that aborts the process if memory pressure rises.

    Triggers a hard abort when either:
      - swap used grows by more than `swap_grow_limit_mb` vs. the baseline
        captured at watchdog start
      - MemAvailable drops below `min_mem_available_mb`

    The abort uses `os._exit(3)` after printing a diagnostic to stderr,
    bypassing any Python-level cleanup that could itself allocate memory.
    """
    baseline = _read_meminfo()
    swap_baseline = baseline.get("SwapTotal", 0) - baseline.get("SwapFree", 0)

    def loop():
        while True:
            try:
                info = _read_meminfo()
                swap_used = info.get("SwapTotal", 0) - info.get("SwapFree", 0)
                mem_avail = info.get("MemAvailable", 0)
                swap_grow_mb = (swap_used - swap_baseline) / (1024 * 1024)
                mem_avail_mb = mem_avail / (1024 * 1024)
                if swap_grow_mb > swap_grow_limit_mb:
                    print(f"\n[watchdog] ABORT: swap grew {swap_grow_mb:.0f} MB "
                          f"(limit {swap_grow_limit_mb} MB). "
                          f"MemAvailable={mem_avail_mb:.0f} MB",
                          flush=True)
                    os._exit(3)
                if mem_avail_mb < min_mem_available_mb:
                    print(f"\n[watchdog] ABORT: MemAvailable={mem_avail_mb:.0f} MB "
                          f"< floor {min_mem_available_mb} MB. "
                          f"swap_grow={swap_grow_mb:.0f} MB",
                          flush=True)
                    os._exit(3)
            except Exception:
                pass
            time.sleep(interval_s)

    t = threading.Thread(target=loop, name="mem-watchdog", daemon=True)
    t.start()
    print(f"[watchdog] armed: swap_grow_limit={swap_grow_limit_mb}MB "
          f"min_mem_avail={min_mem_available_mb}MB interval={interval_s}s  "
          f"baseline swap_used={swap_baseline//(1024*1024)}MB "
          f"mem_avail={(baseline.get('MemAvailable', 0))//(1024*1024)}MB",
          flush=True)
    return t


# ---------------------------------------------------------------------------
# Unbatched CPU path (legacy, robust)
# ---------------------------------------------------------------------------
def _stage_text_only(model_path: str) -> str:
    from .sensitivity_probe import stage_text_only
    return stage_text_only(model_path)


def _load_live_model(model_path: str, device: str, dtype: torch.dtype,
                     device_map: str | None = None) -> nn.Module:
    from transformers import AutoModelForCausalLM

    staged = _stage_text_only(model_path)
    load_device_map = device_map if device_map is not None else device
    model = AutoModelForCausalLM.from_pretrained(
        staged, torch_dtype=dtype, device_map=load_device_map,
        low_cpu_mem_usage=False, trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def measure_unbatched(model: nn.Module, act_cache: "ActivationIndex",
                     target_names: set[str], specs: list[fr.FormatSpec],
                     device: str, dtype: torch.dtype,
                     h_detail: "HDetailIndex | None" = None,
                     profile=None) -> dict:
    """One-Linear-at-a-time measurement. Simple, safe, slow on small ops
    running through unified memory but robust when batching isn't an option.

    When `h_detail` is provided, also emits full per-weight Δloss
    `0.5 · <H_full, (W - W_hat)²>` and a Fisher row-weighted
    `fisher_output_mse` from `g2_per_token` when those tensors are present.
    """
    accum: dict[str, dict[str, dict]] = {}
    processed = 0
    tstart = time.time()
    target_list = list(target_names)
    n_total = len(target_list)

    for name, mod in model.named_modules():
        canonical_name = canonical_linear_name(name, profile)
        if not isinstance(mod, nn.Linear) or canonical_name not in target_names:
            continue
        if canonical_name not in act_cache:
            continue
        W = mod.weight.detach()
        X_cpu, row_indices = act_cache.load_with_row_indices(canonical_name)
        X = X_cpu.to(W.dtype).to(W.device)
        y_ref = X @ W.T
        ref_energy = float(y_ref.float().pow(2).mean().item())
        # Per-weight H diagonal and per-token g² weights if available.
        # Shape of H matches W; g² aligns with rows of X.
        h_full = None
        gq_rows = None
        if h_detail is not None and canonical_name in h_detail:
            blob = h_detail.load_blob(canonical_name)
            try:
                h_full = HDetailIndex.h_diag_from_blob(blob).to(W.device).float()
                if h_full.shape != W.shape:
                    h_full = None  # shape mismatch → fall back to scalar only
            except Exception:
                h_full = None
            gq_rows = _normalize_fisher_output_mse_row_weights(
                blob.get("g2_per_token") if isinstance(blob, dict) else None,
                row_indices,
                int(X.shape[0]),
                W.device,
            )

        gguf_qw = None
        if any(_cost_render_uses_imatrix(s) for s in specs):
            # Shared activation imatrix (per-input-column mean-sq act). Same
            # op/data as export_gguf.build_imatrix_from_act_cache AND
            # export_nvfp4_cb's --col-weights: full fp32 rows, mean over dim 0.
            gguf_qw = X_cpu.float().pow(2).mean(dim=0).to(W.device)

        for spec in specs:
            try:
                if (spec.family == "gguf" and gguf_qw is not None
                        and _gguf_imatrix_enabled()):
                    from prismaquant.gguf_formats import (
                        gguf_quantize_dequantize,
                    )

                    W_hat = gguf_quantize_dequantize(
                        W.clone(), spec.name, col_weights=gguf_qw,
                    )
                elif spec.family in _CB_COST_FAMILIES and gguf_qw is not None:
                    # Lockstep: the CB exporter ships imatrix-weighted VQ bytes,
                    # so the cost render applies the same col_weights.
                    W_hat = spec.quantize_dequantize(
                        W.clone(), col_weights=gguf_qw,
                    )
                else:
                    W_hat = spec.quantize_dequantize(W.clone())
                X_hat = spec.activation_quantize_dequantize(X.clone())
                err = (W - W_hat).float()
                weight_mse = float(err.pow(2).mean().item())
                y_q = X_hat @ W_hat.T
                y_err_sq = (y_ref - y_q).float().pow(2)
                output_mse = float(y_err_sq.mean().item())
                fisher_output_mse = None
                if gq_rows is not None:
                    fisher_output_mse = float(
                        (y_err_sq * gq_rows.unsqueeze(1)).mean().item()
                    )
                predicted_dloss = None
                if h_full is not None:
                    predicted_dloss = float(0.5 * (h_full * err.pow(2)).sum().item())
                _accumulate_result(
                    accum,
                    canonical_name,
                    spec.name,
                    weight_mse,
                    output_mse,
                    output_mse / max(ref_energy, 1e-12),
                    predicted_dloss=predicted_dloss,
                    fisher_output_mse=fisher_output_mse,
                )
            except Exception as e:
                accum.setdefault(canonical_name, {})[spec.name] = {"error": str(e)}
        processed += 1
        if processed % 128 == 0:
            elapsed = time.time() - tstart
            eta = elapsed / processed * (n_total - processed)
            print(f"[cost] {processed}/{n_total} eta={eta:.0f}s", flush=True)
    return _finalize_results(accum)


# ---------------------------------------------------------------------------
# Batched GPU path (fast)
# ---------------------------------------------------------------------------
def _group_by_shape(model: nn.Module, target_names: set[str], profile=None
                    ) -> dict[tuple[int, int], list[tuple[str, nn.Linear]]]:
    """Group target Linears by (in_features, out_features).

    Expert projections within an MoE share shape exactly — across layers
    too for uniform MoE models — so this groups, say, all 10 240
    gate_proj experts across 40 layers into one bucket for 35B Qwen3.6.
    """
    groups: dict[tuple[int, int], list[tuple[str, nn.Linear]]] = {}
    for name, mod in model.named_modules():
        canonical_name = canonical_linear_name(name, profile)
        if not isinstance(mod, nn.Linear) or canonical_name not in target_names:
            continue
        key = (mod.in_features, mod.out_features)
        groups.setdefault(key, []).append((canonical_name, mod))
    return groups


def _chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _enumerate_packed_experts(model: nn.Module, target_names: set[str],
                              profile=None,
                              ) -> list[tuple[str, nn.Parameter, str, nn.Module]]:
    """Find every 3D nn.Parameter that lives directly under a module
    named like an MoE experts container. Uses the same class-name +
    param-name filters as sensitivity_probe._is_packed_experts_module
    so we never accidentally treat e.g. a Conv1d weight as a packed
    expert tensor.

    Returns [(canonical_name, packed_param, module_qname, module), ...]
    where canonical_name is `<module_qname>.<param_name>` to match the
    probe's stat keys. Only entries appearing in `target_names` are
    returned.
    """
    from .sensitivity_probe import _is_packed_experts_module, _packed_experts_param_names
    out = []
    for qname, mod in model.named_modules():
        if not _is_packed_experts_module(mod, profile):
            continue
        for pn in _packed_experts_param_names(mod, profile):
            p = getattr(mod, pn)
            full = f"{qname}.{pn}" if qname else pn
            if full in target_names:
                out.append((full, p, qname, mod))
    return out


def _packed_experts_parent_module(model: nn.Module, experts_qname: str) -> nn.Module | None:
    if not experts_qname:
        return None
    if experts_qname.endswith(".experts"):
        parent_qname = experts_qname[: -len(".experts")]
    elif ".experts." in experts_qname:
        parent_qname = experts_qname.rsplit(".experts.", 1)[0]
    elif "." in experts_qname:
        parent_qname = experts_qname.rsplit(".", 1)[0]
    else:
        return None
    try:
        return model.get_submodule(parent_qname)
    except AttributeError:
        return None


def _packed_experts_router(parent_module: nn.Module | None) -> nn.Module | None:
    if parent_module is None:
        return None
    for attr in ("gate", "router"):
        router = getattr(parent_module, attr, None)
        if isinstance(router, nn.Module):
            return router
    return None


def _packed_router_topk(
    router: nn.Module,
    hidden_states: torch.Tensor,
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (top_k_index, top_k_weights) for a Qwen/DeepSeek-style router.

    Some routers (HYV3TopKRouter) take the parent MoE block's
    e_score_correction_bias buffer as a required positional — callers pass
    it from ``parent_mod`` so routing matches the model's real forward."""
    import inspect
    try:
        fwd_params = inspect.signature(router.forward).parameters
    except (TypeError, ValueError):
        fwd_params = {}
    if "e_score_correction_bias" in fwd_params:
        if e_score_correction_bias is None:
            raise ValueError(
                f"{type(router).__name__}.forward requires "
                "e_score_correction_bias; the parent module must supply it")
        out = router(
            hidden_states,
            e_score_correction_bias.to(hidden_states.device),
        )
    else:
        out = router(hidden_states)
    if isinstance(out, (tuple, list)):
        if len(out) >= 3:
            second = out[1]
            third = out[2]
            if not isinstance(second, torch.Tensor) or not isinstance(third, torch.Tensor):
                raise ValueError("router tuple entries 1 and 2 must be tensors")
            if second.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                indices = second
                weights = third
            elif third.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                weights = second
                indices = third
            else:
                raise ValueError("router tuple does not expose integer top-k indices")
            if weights.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                raise ValueError("router top-k weights must be floating point")
            return indices.to(device=hidden_states.device), weights.to(
                device=hidden_states.device,
            )
        if len(out) >= 2:
            first, second = out[0], out[1]
            if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
                if second.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                    return second.to(device=hidden_states.device), first.to(
                        device=hidden_states.device,
                    )
                if first.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                    return first.to(device=hidden_states.device), second.to(
                        device=hidden_states.device,
                    )
                raise ValueError("router tuple does not expose integer top-k indices")
    if not isinstance(out, torch.Tensor):
        raise ValueError(f"unsupported router output type {type(out)!r}")
    top_k = int(getattr(router, "top_k", 0) or 0)
    if top_k <= 0:
        raise ValueError("router returned logits only and exposes no top_k")
    probs = torch.softmax(out.float(), dim=-1)
    weights, indices = torch.topk(probs, top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return indices.to(device=hidden_states.device), weights.to(device=hidden_states.device)


def _packed_experts_forward_with_weights(
    experts_mod: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    input_quantize=None,
    intermediate_quantize=None,
) -> torch.Tensor:
    """Replay the packed-experts forward with explicit packed weights.

    This mirrors the Qwen3.5/DeepSeek-style fused expert module: one
    packed gate/up tensor, one packed down tensor, and router-provided
    token-to-expert assignments. Quantization hooks are placed at the
    same Linear boundaries as the dense measurement path.
    """
    num_experts = int(getattr(experts_mod, "num_experts", gate_up_weight.size(0)))
    act_fn = getattr(experts_mod, "act_fn", None)
    if act_fn is None:
        raise ValueError("packed experts module exposes no act_fn")
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index.to(torch.long), num_classes=num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx_t in expert_hit:
        expert_idx = int(expert_idx_t[0].item())
        if expert_idx >= num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        if input_quantize is not None:
            current_state = input_quantize(current_state)
        gate_up = F.linear(current_state, gate_up_weight[expert_idx])
        apply_gate = getattr(experts_mod, "_apply_gate", None)
        if callable(apply_gate):
            current_hidden_states = apply_gate(gate_up)
        else:
            gate, up = gate_up.chunk(2, dim=-1)
            current_hidden_states = act_fn(gate) * up
        if intermediate_quantize is not None:
            current_hidden_states = intermediate_quantize(current_hidden_states)
        current_hidden_states = F.linear(
            current_hidden_states,
            down_weight[expert_idx],
        )
        current_hidden_states = (
            current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        )
        final_hidden_states.index_add_(
            0,
            token_idx,
            current_hidden_states.to(final_hidden_states.dtype),
        )
    return final_hidden_states


def derive_per_expert_activations(
    experts_mod: nn.Module,
    X: torch.Tensor,
    parent_mod: nn.Module | None,
    *,
    capture_down: bool = True,
    max_rows_per_expert: int | None = None,
    subsample_seed: int = 1234,
) -> dict:
    """Single source of truth for per-expert GPTQ activations.

    Routes the module-level expert input ``X`` ([*, hidden]) through the MoE
    block's own router and collects, per expert ``e``, exactly the tensors the
    per-expert GPTQ Hessian needs — identical to the routed forward in
    ``_packed_experts_forward_with_weights``, but COLLECTING activations instead
    of producing the output. Shared by the cost path and the export render path
    so the routing + SwiGLU derivation lives in ONE place (no duplication).

    Returns a dict of length-E lists:
      - ``gate_up``: each [n_e, hidden]  — the routed input to ``gate_up_proj``.
      - ``down``:    each [n_e, inter]   — the post-SwiGLU input to ``down_proj``
                     (``_apply_gate(gate_up)`` when present, else ``act_fn(gate)*up``).
                     Empty list when ``capture_down=False``.
      - ``gate_weights``: each [n_e]     — the router weight per routed token.
      - ``row_counts``: list[int]        — routed tokens/expert BEFORE subsample
                     (use for the fail-on-insufficient-routed-rows gate).

    Subsampling (``max_rows_per_expert``) is deterministic (fixed ``subsample_seed``)
    so the render is reproducible. ``None`` keeps every routed row. Raises (never
    silently degrades) if the router / act_fn cannot be resolved.
    """
    gate_up_w = getattr(experts_mod, "gate_up_proj", None)
    if gate_up_w is None:
        raise ValueError("packed experts module lacks gate_up_proj")
    num_experts = int(getattr(experts_mod, "num_experts", gate_up_w.size(0)))
    act_fn = getattr(experts_mod, "act_fn", None)
    apply_gate = getattr(experts_mod, "_apply_gate", None)
    router = _packed_experts_router(parent_mod)
    if router is None:
        raise ValueError("no router found for packed-experts module")
    Xf = X.reshape(-1, X.size(-1))
    dev, dt, hidden = Xf.device, Xf.dtype, Xf.size(-1)
    inter = gate_up_w.size(1) // 2
    with torch.no_grad():
        route_fn = getattr(parent_mod, "route_tokens_to_experts", None)
        if callable(route_fn):
            top_k_index, top_k_weights = route_fn(router(Xf))
        else:
            top_k_index, top_k_weights = _packed_router_topk(
                router, Xf,
                e_score_correction_bias=getattr(
                    parent_mod, "e_score_correction_bias", None),
            )
        expert_mask = F.one_hot(
            top_k_index.to(torch.long), num_classes=num_experts).permute(2, 1, 0)
    gate_up_list: list[torch.Tensor] = []
    down_list: list[torch.Tensor] = []
    gw_list: list[torch.Tensor] = []
    counts: list[int] = []
    for e in range(num_experts):
        top_k_pos, token_idx = torch.where(expert_mask[e])
        n = int(token_idx.numel())
        counts.append(n)
        if n == 0:
            gate_up_list.append(torch.empty(0, hidden, device=dev, dtype=dt))
            gw_list.append(torch.empty(0, device=dev, dtype=dt))
            if capture_down:
                down_list.append(torch.empty(0, inter, device=dev, dtype=dt))
            continue
        if max_rows_per_expert is not None and n > max_rows_per_expert:
            gen = torch.Generator(device=dev).manual_seed(subsample_seed + e)
            keep = torch.randperm(n, device=dev, generator=gen)[:max_rows_per_expert]
            token_idx, top_k_pos = token_idx[keep], top_k_pos[keep]
        Xe = Xf[token_idx]
        gate_up_list.append(Xe)
        gw_list.append(top_k_weights[token_idx, top_k_pos])
        if capture_down:
            with torch.no_grad():
                gate_up = F.linear(Xe, gate_up_w[e])
                if callable(apply_gate):
                    di = apply_gate(gate_up)
                elif act_fn is not None:
                    g, u = gate_up.chunk(2, dim=-1)
                    di = act_fn(g) * u
                else:
                    raise ValueError(
                        "packed experts module exposes neither _apply_gate nor act_fn")
            down_list.append(di)
    return {"gate_up": gate_up_list, "down": down_list,
            "gate_weights": gw_list, "row_counts": counts}


def _packed_expert_activation_quantizer(spec: fr.FormatSpec):
    def _quantize(x: torch.Tensor) -> torch.Tensor:
        return spec.activation_quantize_dequantize(x.clone())
    return _quantize


def _measure_packed_experts(
    model: nn.Module,
    target_names: set[str],
    specs: list[fr.FormatSpec],
    device: str,
    dtype: torch.dtype,
    accum: dict,
    act_cache: "ActivationIndex | None" = None,
    h_detail: "HDetailIndex | None" = None,
    profile=None,
) -> None:
    """Measure per-format cost for each packed-expert tensor.

    The 3D `[num_experts, out, in]` packed tensor reuses the existing
    batched codebook RTN path with N = num_experts. When the probe has
    cached inputs for the experts module, this path now replays the routed
    packed-MoE forward and records the same output_mse objective that
    dense Linears use. If the router or activation cache is unavailable,
    the entry is marked unmeasured so the allocator falls back explicitly.

    When `h_detail` is provided, we also emit a per-weight Δloss based
    on the packed H diagonal stored with per-expert per-output-channel
    resolution (`[E, M]`). That resolution is coarser than the Linear
    path's `[out, in]` — full `[E, M, N]` for 35B packed experts would
    need 160+ GB — but it still captures the expert × channel structure
    that the scalar trace loses.
    """
    dev = torch.device(device)
    entries = _enumerate_packed_experts(model, target_names, profile)
    if not entries:
        return
    if os.environ.get("PRISMAQUANT_SKIP_PACKED_EXPERT_COST", "0") == "1":
        # CB M4-hybrid: the expert_empirical_cost stage REPLACES every
        # packed-expert row wholesale (merge_cost_payloads
        # replace_experts=True pops them), so measuring them here — the
        # single most expensive part of the local cost stage (full-stack
        # imatrix-weighted CB encodes per rung) — is discarded work. The
        # pipeline sets this env only when that replacement is guaranteed
        # to run; if the empirical stage then fails, the pipeline dies
        # there, before the allocator ever sees the row-less payload.
        print(f"[cost] SKIPPING {len(entries)} packed-expert tensors "
              f"(PRISMAQUANT_SKIP_PACKED_EXPERT_COST=1: the CB empirical "
              f"expert stage replaces these rows)", flush=True)
        return
    measured = 0
    fallback = 0
    for full_name, packed_param, experts_qname, experts_mod in entries:
        w = packed_param.detach().to(device=dev, dtype=dtype)
        param_name = full_name.rsplit(".", 1)[-1]

        X = None
        top_k_index = None
        top_k_weights = None
        y_ref = None
        ref_energy = None
        gate_up = None
        down = None
        can_measure_output = False
        if act_cache is not None and experts_qname in act_cache:
            parent_mod = _packed_experts_parent_module(model, experts_qname)
            router = _packed_experts_router(parent_mod)
            try:
                X_cpu = act_cache.load(experts_qname)
                X = X_cpu.to(device=dev, dtype=dtype).reshape(-1, X_cpu.size(-1))
                if router is None:
                    raise ValueError(f"no router found for {experts_qname}")
                gate_up_param = getattr(experts_mod, "gate_up_proj", None)
                down_param = getattr(experts_mod, "down_proj", None)
                if gate_up_param is None or down_param is None:
                    raise ValueError("packed experts module lacks gate_up/down params")
                gate_up = gate_up_param.detach().to(device=dev, dtype=dtype)
                down = down_param.detach().to(device=dev, dtype=dtype)
                with torch.no_grad():
                    # Prefer the MoE block's own routing when it exposes a
                    # `route_tokens_to_experts` method — that is faithful to
                    # whatever selection the architecture actually uses (e.g.
                    # sigmoid scores + expert_bias top-k + norm + routed
                    # scaling, with `top_k` living on the block rather than the
                    # bare gate Linear) instead of the generic softmax top-k.
                    # Falls back to the generic extraction for routers that
                    # don't expose it (Qwen/DeepSeek-style bare gate Linears).
                    _route_fn = getattr(parent_mod, "route_tokens_to_experts", None)
                    if callable(_route_fn):
                        top_k_index, top_k_weights = _route_fn(router(X))
                    else:
                        top_k_index, top_k_weights = _packed_router_topk(
                            router, X,
                            e_score_correction_bias=getattr(
                                parent_mod, "e_score_correction_bias", None),
                        )
                    y_ref = _packed_experts_forward_with_weights(
                        experts_mod,
                        X,
                        top_k_index,
                        top_k_weights,
                        gate_up,
                        down,
                    )
                ref_energy = float(y_ref.float().pow(2).mean().item())
                can_measure_output = True
            except Exception as e:
                print(f"[cost] WARN: packed output_mse unavailable for "
                      f"{full_name}: {e}", flush=True)
                X = None
                top_k_index = None
                top_k_weights = None
                y_ref = None
                ref_energy = None
                gate_up = None
                down = None

        # Per-(expert, out-channel) Fisher row-sum h_em, shape [E, M]
        # (= Σ_n grad² over in-features; see the Δloss derivation below).
        # Weighted elementwise against per-row mean err² to yield one
        # Δloss scalar per (layer, format).
        h_em = None
        if h_detail is not None and full_name in h_detail:
            h = h_detail.load(full_name).to(dev).float()
            if h.shape == (w.size(0), w.size(1)):
                h_em = h
        packed_gguf_qw = None
        if any(_cost_render_uses_imatrix(s) for s in specs):
            # Pooled imatrix from the experts-module input snapshot: exact
            # source for gate_up_proj (its input IS the module input); the
            # shape guard leaves down_proj unweighted (its input is the
            # per-expert intermediate, not cached — the v2 replay pass will
            # weight it). MUST stay op-identical to the exporter's builder
            # (full rows, fp32, mean over dim 0) — lockstep contract.
            try:
                _x = act_cache.load(experts_qname)
                qw_vec = _x.float().pow(2).mean(dim=0)
                if qw_vec.numel() == w.shape[-1]:
                    packed_gguf_qw = qw_vec.reshape(1, 1, -1).to(dev)
            except Exception:
                packed_gguf_qw = None

        # Optional stratified expert subsample for gguf-family COST entries:
        # the allocator prices the whole stack as one unit (mean over
        # experts), so sampling S of E experts estimates that mean at E/S
        # less quantize work (IQ exhaustive-grid on 3.5G-elem stacks is
        # ~25 min/layer at full E — 2026-07-11). Export always quantizes
        # every expert exactly; this only affects the DP's cost estimates.
        sample_n = int(os.environ.get(
            "PRISMAQUANT_EXPERT_COST_SAMPLE",
            os.environ.get("PRISMAQUANT_GGUF_EXPERT_COST_SAMPLE", "0"),
        ) or 0)
        for spec in specs:
            try:
                # Family-agnostic: NVFP4/FP8 registry quantize on a full
                # 2.4G-elem stack swap-kills a UMA box just like the gguf
                # search did (2026-07-11 CT cost abort at layer 1).
                use_sample = (
                    sample_n > 0
                    and w.ndim >= 3 and int(w.shape[0]) > sample_n
                )
                if use_sample:
                    s_idx = torch.linspace(
                        0, w.shape[0] - 1, sample_n, device=w.device,
                    ).round().long().unique()
                    w_in = w[s_idx]
                else:
                    w_in = w
                cw_use = packed_gguf_qw
                if _cost_render_uses_imatrix(spec):
                    ext_cw = _cb_col_weights_lookup(full_name)
                    if ext_cw is not None:
                        cw_use = ext_cw.to(device=w.device,
                                           dtype=torch.float32)
                        if (use_sample and cw_use.ndim >= 3
                                and cw_use.shape[0] == w.shape[0]):
                            cw_use = cw_use[s_idx]
                w_hat = _batched_quantize(
                    spec, w_in,
                    col_weights=(
                        cw_use
                        if _cost_render_uses_imatrix(spec) else None
                    ),
                )
                err = (w_in - w_hat).float()
                weight_mse = float(err.pow(2).mean().item())
                dloss_val = None
                if h_em is not None:
                    # h_em[e,m] = Σ_n g[e,m,n]² — the per-row SUM over
                    # in-features of the per-weight gradient² (see
                    # sensitivity_probe channel_accumulator). The exact
                    # per-weight Fisher-OBS loss is 0.5·Σ_{e,m,n} g²·err²
                    # (a sum-of-products), matching the dense path at
                    # line 464 / 1120. With only the row-summed h_em and
                    # per-weight err², the mean-field estimate (g² ≈ const
                    # across n within a row) is
                    #   0.5·Σ_{e,m} h_em · mean_n(err²),
                    # which is on the SAME scale as the dense 0.5·Σ g²·err².
                    # Do NOT multiply by N here: h_em is already summed over
                    # n, so an extra ×N would make this a product-of-sums
                    # (Σ_n g²)(Σ_n err²) and inflate packed-expert Δloss ~N×
                    # relative to dense Linears, over-promoting experts in
                    # the allocator (N = in-features ≈ 1.5k–4k).
                    per_ch_mse = err.pow(2).mean(dim=-1)   # [E or S, M]
                    if use_sample:
                        # Unbiased estimate of the full-stack sum from the
                        # stratified sample.
                        scale = float(w.shape[0]) / float(w_in.shape[0])
                        dloss_val = float(
                            0.5 * (h_em[s_idx] * per_ch_mse).sum().item()
                            * scale
                        )
                    else:
                        dloss_val = float(
                            0.5 * (h_em * per_ch_mse).sum().item()
                        )
                    del per_ch_mse
                output_mse = 0.0
                rel_mse = 0.0
                output_mse_measured = False
                if can_measure_output and not use_sample:
                    act_quant = _packed_expert_activation_quantizer(spec)
                    with torch.no_grad():
                        if param_name == "gate_up_proj":
                            y_q = _packed_experts_forward_with_weights(
                                experts_mod,
                                X,
                                top_k_index,
                                top_k_weights,
                                w_hat,
                                down,
                                input_quantize=act_quant,
                            )
                        elif param_name == "down_proj":
                            y_q = _packed_experts_forward_with_weights(
                                experts_mod,
                                X,
                                top_k_index,
                                top_k_weights,
                                gate_up,
                                w_hat,
                                intermediate_quantize=act_quant,
                            )
                        else:
                            y_q = None
                    if y_q is not None:
                        y_err_sq = (y_ref - y_q).float().pow(2)
                        output_mse = float(y_err_sq.mean().item())
                        rel_mse = output_mse / max(float(ref_energy), 1e-12)
                        output_mse_measured = True
                        measured += 1
                        del y_q, y_err_sq
                if not output_mse_measured:
                    fallback += 1
                _accumulate_result(accum, full_name, spec.name,
                                   weight_mse, output_mse, rel_mse,
                                   predicted_dloss=dloss_val,
                                   output_mse_measured=output_mse_measured)
                del w_hat, err
            except Exception as e:
                accum.setdefault(full_name, {})[spec.name] = {"error": str(e)}
        del w
        if X is not None:
            del X
        if top_k_index is not None:
            del top_k_index
        if top_k_weights is not None:
            del top_k_weights
        if y_ref is not None:
            del y_ref
        if gate_up is not None:
            del gate_up
        if down is not None:
            del down
        if h_em is not None:
            del h_em
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    if measured or fallback:
        print(f"[cost] packed experts output_mse measured={measured} "
              f"fallback={fallback}", flush=True)


def _batched_codebook_rtn(stacked_w: torch.Tensor, codebook: torch.Tensor,
                          group_size: int, mx_scale: bool = False
                          ) -> torch.Tensor:
    """Apply the same bucketize-based FP-codebook RTN used by format_registry,
    but on a stacked `(N, out, in)` tensor in one call. No allocation per
    inner Linear.

    `mx_scale=True` snaps the per-group scale to the nearest power of two
    (E8M0), matching the OCP MX serving path.
    """
    N, out_f, in_f = stacked_w.shape
    w2 = stacked_w.reshape(N, -1, in_f).float()
    if group_size > 0 and group_size < in_f:
        w2 = w2.reshape(N, -1, in_f // group_size, group_size)
    else:
        w2 = w2.unsqueeze(2)

    cb = codebook.to(device=w2.device, dtype=torch.float32).contiguous()
    cmax = cb.abs().max()
    max_abs = w2.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = max_abs / cmax
    if mx_scale:
        scale = fr._snap_scale_e8m0(scale, element_max=cmax)
    x = (w2 / scale).contiguous()

    # Bucketize returns int64 by default; cast to int32 to halve the index
    # tensor footprint. A 256-entry FP8 codebook fits comfortably in int32.
    idx = torch.bucketize(x, cb).to(torch.int32)
    idx_lo = (idx - 1).clamp_min(0)
    idx_hi = idx.clamp_max(cb.numel() - 1)
    del idx
    lo = cb[idx_lo]
    hi = cb[idx_hi]
    del idx_lo, idx_hi
    choose_hi = (hi - x).abs() < (x - lo).abs()
    q = torch.where(choose_hi, hi, lo)
    del lo, hi, choose_hi
    w_rec = q * scale
    del q
    return w_rec.reshape(N, out_f, in_f).to(stacked_w.dtype)


def _batched_int_rtn(stacked_w: torch.Tensor, bits: int, group_size: int,
                     symmetric: bool = True) -> torch.Tensor:
    N, out_f, in_f = stacked_w.shape
    w2 = stacked_w.reshape(N, -1, in_f).float()
    if group_size > 0 and group_size < in_f:
        w2 = w2.reshape(N, -1, in_f // group_size, group_size)
    else:
        w2 = w2.unsqueeze(2)
    max_abs = w2.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    if symmetric:
        levels = (1 << (bits - 1)) - 1
        scale = max_abs / levels
        q = torch.round(w2 / scale).clamp(-levels - 1, levels)
        w_rec = q * scale
    else:
        levels = (1 << bits) - 1
        w_min = w2.amin(dim=-1, keepdim=True)
        w_max = w2.amax(dim=-1, keepdim=True)
        scale = (w_max - w_min) / levels
        zp = torch.round(-w_min / scale.clamp_min(1e-8))
        q = torch.round(w2 / scale.clamp_min(1e-8) + zp).clamp(0, levels)
        w_rec = (q - zp) * scale
    return w_rec.reshape(N, out_f, in_f).to(stacked_w.dtype)


# Map FormatSpec to a batched RTN function.  We could have FormatSpec carry
# its own batched op, but hardcoding the two families (codebook vs integer)
# keeps the registry simple.  New formats just declare which family they're
# in via their weight_element_dtype.
_CODEBOOK_NAMES = {
    "fp4_e2m1": "_e2m1",
    "fp6_e3m2": "_e3m2",
    "fp6_e2m3": "_e2m3",
    "fp8_e4m3": "_e4m3",
    "fp8_e5m2": "_e5m2",
}


def _gguf_imatrix_enabled() -> bool:
    """PRISMAQUANT_GGUF_IMATRIX: activation-weighted (imatrix) scale
    selection for GGUF k-quant cost measurement. Default ON — the GGUF
    exporter ships imatrix-weighted bytes (--imatrix-from-act-cache), so
    the cost the allocator optimizes must be measured on the same render
    or the A/B has a rendering confound. Set =0 only together with an
    unweighted export."""
    # Parse MUST stay in lockstep with run-pipeline.sh's shell parse:
    # set-but-empty means default (on); 0/false/no/off in any case = off.
    value = os.environ.get("PRISMAQUANT_GGUF_IMATRIX", "1").strip().lower() or "1"
    return value not in {"0", "false", "no", "off"}


# VQ codebook families whose exporter (export_nvfp4_cb) ships imatrix-weighted
# bytes UNCONDITIONALLY: --col-weights is required, there is no unweighted CB
# export. So their measured COST render must ALWAYS apply the imatrix to stay in
# lockstep (cost == shipped-bytes weighting; the one-cache/no-confound rule).
# Unlike gguf there is deliberately NO toggle — a toggle could only desync the
# cost from an export that is always weighted.
_CB_COST_FAMILIES = ("nvfp4_cb", "fp8_cb")


def _cost_render_uses_imatrix(spec: fr.FormatSpec) -> bool:
    """Whether this spec's COST render is activation-imatrix-weighted, matching
    the family's exporter. gguf tracks the PRISMAQUANT_GGUF_IMATRIX toggle (its
    export is optionally --imatrix-from-act-cache); CB families are always
    weighted (their export always is)."""
    if spec.family == "gguf":
        return _gguf_imatrix_enabled()
    return spec.family in _CB_COST_FAMILIES


def _item_col_weights(
    col_weights: torch.Tensor | None, i: int, n: int
) -> torch.Tensor | None:
    """Per-input-column imatrix vector for stacked item ``i`` as 1-D
    ``(in_features,)`` (the registry qdq broadcasts it to ``(out, in)``). A
    per-item stack ``(N, ..., in)`` is indexed; a single pooled/broadcast
    vector ``(1, 1, in)`` or ``(in,)`` is shared across all items."""
    if col_weights is None:
        return None
    cw = col_weights
    if cw.ndim >= 1 and cw.shape[0] == n:
        cw = cw[i]
    return cw.reshape(-1)


def _batched_quantize(
    spec: fr.FormatSpec,
    stacked_w: torch.Tensor,
    col_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    elt = spec.weight_element_dtype
    if spec.family == "nv":
        # NVFP4 registry weights are export-codec-aligned (one rendering
        # everywhere); the registry fn reshapes (-1, in) internally, so it
        # is natively batch-shaped. Using the local codebook replica here
        # would re-introduce the resident-vs-export scale mismatch the
        # alignment removed (cost values must match the unbatched path,
        # which calls spec.quantize_dequantize).
        # Per-slice: the export codec derives a per-TENSOR global scale,
        # and each stacked Linear must get its own (matching unbatched).
        return torch.stack([
            spec.quantize_dequantize(stacked_w[i].clone())
            for i in range(stacked_w.shape[0])
        ])
    if spec.family == "gguf":
        # GGUF k-quants: the registry qdq reshapes (..., in) internally and
        # every scale is local to a 256-superblock (no per-tensor state), so
        # the stacked (N, out, in) tensor quantizes in one call and matches
        # the unbatched path bit-for-bit. col_weights (per-item imatrix
        # vectors, broadcastable to stacked_w) bias scale selection exactly
        # as the exporter's --imatrix-from-act-cache does.
        # Big stacks (192-expert MoE layers ~2.4G elements) are sliced along
        # dim 0 — exact by superblock locality — or the search's fp32
        # temporaries (~20x element count) blow the unified-memory budget.
        from prismaquant.gguf_formats import (
            gguf_quantize_dequantize,
            gguf_slice_max_elems,
        )

        max_elems = gguf_slice_max_elems(spec.name)
        if stacked_w.ndim >= 2 and stacked_w.numel() > max_elems:
            step = max(1, max_elems // max(stacked_w[0].numel(), 1))
            outs = []
            for i in range(0, stacked_w.shape[0], step):
                cw = col_weights
                if cw is not None and cw.ndim >= 1 and cw.shape[0] == stacked_w.shape[0]:
                    cw = cw[i:i + step]
                outs.append(gguf_quantize_dequantize(
                    stacked_w[i:i + step], spec.name, col_weights=cw,
                ))
            return torch.cat(outs, dim=0)
        return gguf_quantize_dequantize(
            stacked_w, spec.name, col_weights=col_weights,
        )
    if spec.family in _CB_COST_FAMILIES:
        # VQ codebook families: render per-slice through the SAME registry qdq
        # closure the unbatched path and the exporter use (fixed lattice + the
        # default-on scale sweep), so the measured cost is the render export
        # ships. col_weights (per-input-column imatrix) bias the weighted VQ
        # search exactly as export_nvfp4_cb's --col-weights does. Per-slice
        # (like the nv family) keeps each stacked Linear independent and matches
        # the unbatched path bit-for-bit — no cross-slice coupling exists (fp4
        # scales are per-group-16, fp8 scales per-output-channel; lattice fixed).
        n = stacked_w.shape[0]
        return torch.stack([
            spec.quantize_dequantize(
                stacked_w[i].clone(),
                col_weights=_item_col_weights(col_weights, i, n),
            )
            for i in range(n)
        ])
    if col_weights is not None:
        raise ValueError(
            f"col_weights is only supported for gguf-family and CB codebook "
            f"formats, got {spec.name}"
        )
    if elt in _CODEBOOK_NAMES:
        # Reuse the registry's codebook tables. MX-family formats need
        # E8M0 scale snapping to match the OCP MX serving path; NV/FP
        # families use real-valued (FP8 / FP32) scales unchanged.
        cb = fr._CODEBOOKS[elt]
        mx_scale = spec.family == "mx"
        return _batched_codebook_rtn(stacked_w, cb, spec.group_size,
                                     mx_scale=mx_scale)
    elif elt.startswith("int"):
        return _batched_int_rtn(stacked_w, spec.weight_bits, spec.group_size)
    elif elt == "bfloat16":
        return stacked_w.clone()
    else:
        raise ValueError(f"Unknown weight_element_dtype {elt!r} for "
                         f"batched RTN")


# Holdout gate tolerance for the dense-path ladder (matches the expert
# stage's --ladder-holdout-tol default).
_CB_LADDER_TOL = float(os.environ.get("PRISMAQUANT_CB_LADDER_TOL", "0.10"))

_CB_CW_CACHE: dict | None = None


def _cb_col_weights_lookup(name: str):
    """Shared CB col-weights pickle (PRISMAQUANT_CB_COL_WEIGHTS, set by the
    pipeline when local packed-expert costs must match the exporter's
    weighting — incl. the synthesized per-expert down_proj replay entries the
    inline module-input pool can never provide)."""
    global _CB_CW_CACHE
    path = os.environ.get("PRISMAQUANT_CB_COL_WEIGHTS")
    if not path:
        return None
    if _CB_CW_CACHE is None:
        with open(path, "rb") as fh:
            _CB_CW_CACHE = {k: torch.as_tensor(v)
                            for k, v in pickle.load(fh).items()}
        print(f"[cost] packed-expert col-weights from {path} "
              f"({len(_CB_CW_CACHE)} entries)", flush=True)
    return _CB_CW_CACHE.get(name)


def _cb_ladder_plan(specs: list[fr.FormatSpec]):
    """Dense-path RD-ladder plan (PRISMAQUANT_CB_LADDER_INTERP=1, default
    OFF): per-(family,mode) CB rung ladders from the shared splitter. Returns
    ``(ladders, predicted_names)`` or None. Anchors+holdout are measured
    normally; predicted rungs are fitted per TENSOR and holdout-gated, with a
    measured fallback — so a tensor that defies the law never receives an
    interpolated cost (encode_tiers.md §B/§C)."""
    if os.environ.get("PRISMAQUANT_CB_LADDER_INTERP", "0") != "1":
        return None
    # Lazy import: expert_empirical_cost lazily imports helpers from this
    # module, so a top-level import would be a cycle.
    from prismaquant.expert_empirical_cost import _cb_ladder_split
    split = _cb_ladder_split([s.name for s in specs])
    if not split:
        return None
    predicted_names = {f for (_, _, _, pred) in split for f in pred}
    print(f"[cost] CB ladder interp ON: predicting {sorted(predicted_names)} "
          f"per tensor from anchors (holdout-gated)", flush=True)
    return split, predicted_names


def _ladder_rate_factor(fmt_name: str, k: int) -> float:
    """Exact per-sub rate factor R(k) = sum_i 2^(-2*b_i/d_i) under the
    ceil-first bit split — the theory-faithful rate variable for the CB
    ladder. The smooth 2^(-alpha*k) law treats k as evenly divisible, but at
    k % n_sub != 0 the ceil-first split gives some sub-tables one bit more:
    the true error carries a +4-6% SAWTOOTH by split phase that a smooth law
    cannot represent — and the (28, 38, 48) anchors + k=39 holdout sit on
    DIFFERENT phases, which is exactly where the 8-12% holdout rejections
    came from (2026-07-21 27B cost run). R is exact per phase and reduces to
    the old law at even splits (R(4m) = 4*2^(-m) for fp8)."""
    n_sub = 4 if fmt_name.startswith("FP8_CB") else 2
    sub_dim = 8 // n_sub
    base, extra = divmod(int(k), n_sub)
    r = 0.0
    for i in range(n_sub):
        b = base + (1 if i < extra else 0)
        r += 2.0 ** (-2.0 * b / sub_dim)
    return r


def _ladder_metric_fit(kmap, anchors, fmt_values, target_fmt):
    """Fit ONE metric on the anchors and predict target_fmt.

    Chain (holdout gates every variant, so each is a proposal only):
      1. Split-aware FLOORED LINEAR law D = F + C*R(k), with R the exact
         ceil-first per-sub rate factor — plain linear least squares in
         (1, R); kills both the high-k floor miss AND the k%n_sub sawtooth.
      2. The smooth floor law D = F + C*2^(-b*k) (exact 3-anchor solve).
      3. Log-linear LS (the original law).
    Returns None if any anchor value is unusable."""
    try:
        xs = [float(kmap[f]) for f in anchors]
        vs = [float(fmt_values[f]) for f in anchors]
    except (KeyError, TypeError):
        return None
    if len(anchors) >= 2:
        # -- 1. split-aware floored linear LS ------------------------------
        rs = [_ladder_rate_factor(f, kmap[f]) for f in anchors]
        n = float(len(rs))
        mr = sum(rs) / n
        mv = sum(vs) / n
        den = sum((r - mr) ** 2 for r in rs)
        if den > 0.0:
            C = sum((r - mr) * (v - mv) for r, v in zip(rs, vs)) / den
            F = mv - C * mr
            if C > 0.0 and F >= 0.0:
                r_t = _ladder_rate_factor(target_fmt, kmap[target_fmt])
                return float(F + C * r_t)
            if C > 0.0 and F < 0.0:
                # floor clamped to 0: refit C through the origin
                den0 = sum(r * r for r in rs)
                if den0 > 0.0:
                    C0 = sum(r * v for r, v in zip(rs, vs)) / den0
                    if C0 > 0.0:
                        r_t = _ladder_rate_factor(target_fmt,
                                                  kmap[target_fmt])
                        return float(C0 * r_t)
    if len(anchors) == 3:
        # -- 2. smooth floor law (exact solve) -----------------------------
        from prismaquant.expert_empirical_cost import _fit_floor_law
        fl = _fit_floor_law(xs, vs)
        if fl is not None:
            F, C, b = fl
            return float(F + C * 2.0 ** (-b * kmap[target_fmt]))
    ys = [math.log2(max(v, 1e-20)) for v in vs]
    n = float(len(xs))
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    b = (-sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
         if denom > 0 else 0.25)
    a = my + b * mx
    return float(2.0 ** (a - b * kmap[target_fmt]))


def _chunk_metric(accum, name, fmt, key, count_key="_count"):
    row = accum.get(name, {}).get(fmt)
    if not row or "_count" not in row or row["_count"] <= 0:
        return None
    if key == "predicted_dloss":
        if row.get("_predicted_dloss_count", 0) <= 0:
            return None
        return row["_predicted_dloss_sum"] / row["_predicted_dloss_count"]
    if key == "fisher_output_mse":
        if row.get("_fisher_output_mse_count", 0) <= 0:
            return None
        return row["_fisher_output_mse_sum"] / row["_fisher_output_mse_count"]
    return row.get(f"_{key}_sum", 0.0) / row["_count"]


def measure_batched_gpu(model: nn.Module, act_cache: "ActivationIndex",
                       target_names: set[str], specs: list[fr.FormatSpec],
                       device: str, dtype: torch.dtype,
                       chunk_size: int = 256,
                       h_detail: "HDetailIndex | None" = None,
                       profile=None) -> dict:
    """Batched GPU measurement.

    Groups Linears by shape, then within each group processes `chunk_size`
    Linears at a time. Each chunk does one stacked quantize-and-bmm per
    format. This converts the 31k-kernel-launch pathological case into a
    few hundred well-sized kernel launches.

    `chunk_size` trades latency for VRAM. For Qwen3.6-35B MoE experts
    (shape 2048×512) at BF16, one chunk of 256 = 256 MB weights; 256×256
    rows × 2048 = 128 MB activations; 3 formats × (W, Ŵ, Y_ref, Y_q) peak
    ~2 GB. Safe at chunk_size=256 on any GPU with 4+ GB free.

    When `h_detail` is provided, also emits full per-weight Δloss
    `0.5 · <H_full, (W - W_hat)²>` and a Fisher row-weighted
    `fisher_output_mse` from `g2_per_token` when those tensors are present.
    """
    dev = torch.device(device)
    groups = _group_by_shape(model, target_names, profile)
    total_linears = sum(len(v) for v in groups.values())
    print(f"[cost] batched: {len(groups)} shape groups, "
          f"{total_linears} Linears total", flush=True)
    ladder_plan = _cb_ladder_plan(specs)

    accum: dict[str, dict[str, dict]] = {}
    processed = 0
    tstart = time.time()

    # v24: async activation prefetch. The previous synchronous path
    # spent ~30-40% of the cost step's wall in the per-Linear file
    # reads at chunk-start (e.g. chunk_size=256 × ~5 ms/file = ~1.3 s
    # of disk I/O blocking the GPU after every chunk). Overlap by
    # loading chunk N+1's activations on a small thread pool while
    # chunk N's measurements run on the GPU. Default off — opt-in via
    # PRISMAQUANT_COST_PREFETCH_ACT=1 — until we've validated the win
    # at production scale.
    import os as _os
    from concurrent.futures import ThreadPoolExecutor as _Pool
    # v26: default ON. PRISMAQUANT_COST_PREFETCH_ACT=0 reverts to the
    # synchronous per-chunk activation read path.
    _raw_prefetch = _os.environ.get("PRISMAQUANT_COST_PREFETCH_ACT")
    _prefetch_enabled = (
        True if _raw_prefetch is None
        else _raw_prefetch not in ("0", "", "false", "False", "FALSE", "no", "NO")
    )
    _prefetch_pool = _Pool(max_workers=2) if _prefetch_enabled else None

    def _load_chunk_acts(_names):
        return [act_cache.load_with_row_indices(n) for n in _names]

    for (in_f, out_f), entries in groups.items():
        entries_with_acts = [(n, m) for n, m in entries if n in act_cache]
        if not entries_with_acts:
            continue

        chunks_list = list(_chunked(entries_with_acts, chunk_size))
        # Kick off the first chunk's load so the loop can pull from the
        # future immediately on entry. Subsequent iterations submit the
        # next chunk's load before processing the current one.
        next_acts_fut = (
            _prefetch_pool.submit(
                _load_chunk_acts, [n for n, _ in chunks_list[0]])
            if _prefetch_enabled and chunks_list
            else None
        )

        for chunk_i, chunk in enumerate(chunks_list):
            names = [n for n, _ in chunk]
            N = len(chunk)
            # Lazy load activations for this chunk only. With prefetch
            # enabled, the future is already in flight from the prior
            # iteration (or kicked off above for the first chunk).
            if _prefetch_enabled:
                act_items_cpu = next_acts_fut.result()
                # Submit the NEXT chunk's load before we touch the GPU
                # so the disk reads overlap with the upcoming bmm.
                if chunk_i + 1 < len(chunks_list):
                    nxt_names = [n for n, _ in chunks_list[chunk_i + 1]]
                    next_acts_fut = _prefetch_pool.submit(
                        _load_chunk_acts, nxt_names)
                else:
                    next_acts_fut = None
            else:
                act_items_cpu = [act_cache.load_with_row_indices(n) for n in names]
            acts_cpu = [item[0] for item in act_items_cpu]
            row_indices_cpu = [item[1] for item in act_items_cpu]
            chunk_min_rows = min(a.size(0) for a in acts_cpu)
            # Stack weights
            W = torch.stack([m.weight.detach().to(device=dev, dtype=dtype)
                             for _, m in chunk], dim=0)   # (N, out, in)
            # Stack activations (truncated to chunk_min_rows)
            X = torch.stack(
                [a[:chunk_min_rows].to(device=dev, dtype=dtype)
                 for a in acts_cpu], dim=0)               # (N, rows, in)
            row_indices_cpu = [
                idx[:chunk_min_rows] if isinstance(idx, torch.Tensor) else None
                for idx in row_indices_cpu
            ]
            gguf_qw = None
            if any(_cost_render_uses_imatrix(s) for s in specs):
                # Per-item imatrix, computed with the IDENTICAL op on the
                # IDENTICAL data as export_gguf.build_imatrix_from_act_cache
                # (FULL fp32 CPU act rows, mean over dim 0) — NOT from the
                # chunk-truncated compute-dtype X. The k-quant scale search
                # is a discrete grid: a numerically different importance
                # vector can flip (sc, m, q) choices, and then the measured
                # cost would not describe the shipped bytes. Must run
                # before acts_cpu is freed below.
                gguf_qw = torch.stack([
                    a.float().pow(2).mean(dim=0) for a in acts_cpu
                ]).unsqueeze(1).to(dev)  # (N, 1, in)
            del acts_cpu, act_items_cpu
            # Reference output (per-item BMM): shape (N, rows, out)
            y_ref = torch.bmm(X, W.transpose(1, 2))
            ref_energy = y_ref.float().pow(2).mean(dim=(1, 2))   # (N,)

            # Per-item H full tensor stacked across the chunk, for the
            # per-weight Δloss computation. Missing items get None.
            h_stacked = None
            gq_stacked = None
            if h_detail is not None:
                h_items = []
                gq_items = []
                all_have_h = True
                all_have_gq = True
                for idx_nm, nm in enumerate(names):
                    if nm in h_detail:
                        blob = h_detail.load_blob(nm)
                        try:
                            h = HDetailIndex.h_diag_from_blob(blob)
                        except Exception:
                            h = None
                        if h is not None and h.shape == (W.size(1), W.size(2)):
                            h_items.append(h.to(dev).float())
                        else:
                            all_have_h = False
                        gq = (
                            blob.get("g2_per_token")
                            if isinstance(blob, dict)
                            else None
                        )
                        gq_rows = _normalize_fisher_output_mse_row_weights(
                            gq,
                            row_indices_cpu[idx_nm],
                            chunk_min_rows,
                            dev,
                        )
                        if gq_rows is not None:
                            gq_items.append(gq_rows)
                        else:
                            all_have_gq = False
                    else:
                        all_have_h = False
                        all_have_gq = False
                if all_have_h and len(h_items) == N:
                    h_stacked = torch.stack(h_items, dim=0)   # (N, out, in)
                if all_have_gq and len(gq_items) == N:
                    gq_stacked = torch.stack(gq_items, dim=0)  # (N, rows)
                del h_items, gq_items

            def _measure_spec_into_accum(spec, idx=None):
                """One spec's batched measure for the whole chunk (idx=None)
                or an index-subset (the ladder's per-tensor measured
                fallback). Identical math either way."""
                sub_names = (names if idx is None
                             else [names[i] for i in idx])
                try:
                    Ws = W if idx is None else W[idx]
                    Xs = X if idx is None else X[idx]
                    y_refs = y_ref if idx is None else y_ref[idx]
                    ref_e = ref_energy if idx is None else ref_energy[idx]
                    gqw = gguf_qw if (gguf_qw is None or idx is None) \
                        else gguf_qw[idx]
                    hs = h_stacked if (h_stacked is None or idx is None) \
                        else h_stacked[idx]
                    gqs = gq_stacked if (gq_stacked is None or idx is None) \
                        else gq_stacked[idx]
                    n_sub = Ws.size(0)
                    W_hat = _batched_quantize(
                        spec, Ws,
                        col_weights=(
                            gqw if _cost_render_uses_imatrix(spec) else None
                        ),
                    )
                    X_hat = spec.activation_quantize_dequantize(Xs.clone())
                    err = (Ws - W_hat).float()
                    weight_mse = err.pow(2).mean(dim=(1, 2))  # (n_sub,)
                    y_q = torch.bmm(X_hat, W_hat.transpose(1, 2))
                    y_err_sq = (y_refs - y_q).float().pow(2)
                    output_mse = y_err_sq.mean(dim=(1, 2))  # (n_sub,)
                    rel_mse = output_mse / ref_e.clamp_min(1e-12)
                    fisher_output_mse = None
                    if gqs is not None:
                        fisher_output_mse = (
                            y_err_sq * gqs.unsqueeze(2)
                        ).mean(dim=(1, 2))
                    # Per-item predicted Δloss from full per-weight
                    # Fisher. shape (n_sub,).
                    dloss_per = None
                    if hs is not None:
                        dloss_per = 0.5 * (hs * err.pow(2)).sum(dim=(1, 2))
                    # Move all scalar metrics back to the host in one shot.
                    # Calling `.item()` per Linear forces a CUDA sync for
                    # each row and turns the batched path back into
                    # serialized work.
                    metric_cols = [weight_mse, output_mse, rel_mse]
                    if fisher_output_mse is not None:
                        metric_cols.append(fisher_output_mse)
                    metrics = torch.stack(
                        metric_cols, dim=1).detach().cpu().tolist()
                    if dloss_per is not None:
                        dloss_values = dloss_per.detach().cpu().tolist()
                    else:
                        dloss_values = [None] * n_sub

                    # Unpack per-item into results dict after the single sync.
                    for name, row, dloss_val in zip(sub_names, metrics,
                                                    dloss_values):
                        w_mse, out_mse, rel = row[:3]
                        fisher_val = row[3] if len(row) > 3 else None
                        _accumulate_result(
                            accum,
                            name,
                            spec.name,
                            float(w_mse),
                            float(out_mse),
                            float(rel),
                            predicted_dloss=(
                                float(dloss_val)
                                if dloss_val is not None
                                else None
                            ),
                            fisher_output_mse=(
                                float(fisher_val)
                                if fisher_val is not None
                                else None
                            ),
                        )
                    del W_hat, X_hat, err, y_q, y_err_sq
                    del weight_mse, output_mse, rel_mse
                    del metrics, dloss_values
                except Exception as e:
                    for name in sub_names:
                        accum.setdefault(name, {})[spec.name] = {
                            "error": str(e)}

            measured_specs = (
                specs if ladder_plan is None
                else [s for s in specs if s.name not in ladder_plan[1]])
            for spec in measured_specs:
                _measure_spec_into_accum(spec)

            if ladder_plan is not None:
                # Per-tensor fit + holdout gate; fill accepted predictions,
                # batch-measure the rejects (exact same math via the closure).
                specs_by_name = {s.name: s for s in specs}
                for kmap, anchors, holdout, predicted in ladder_plan[0]:
                    if not all(f in specs_by_name for f in
                               list(anchors) + [holdout] + list(predicted)):
                        continue
                    primary = ("predicted_dloss" if h_stacked is not None
                               else "output_mse")
                    fail_idx = []
                    for i, name in enumerate(names):
                        vals = {f: _chunk_metric(accum, name, f, primary)
                                for f in list(anchors) + [holdout]}
                        if any(v is None or v <= 0 for v in vals.values()):
                            fail_idx.append(i)
                            continue
                        pred_h = _ladder_metric_fit(
                            kmap, anchors, vals, holdout)
                        rel_err = (abs(pred_h - vals[holdout])
                                   / max(vals[holdout], 1e-20))
                        if pred_h is None or rel_err > _CB_LADDER_TOL:
                            fail_idx.append(i)
                            continue
                        for fmt in predicted:
                            fills = {}
                            for key in ("weight_mse", "output_mse",
                                        "rel_output_mse", "predicted_dloss",
                                        "fisher_output_mse"):
                                mvals = {f: _chunk_metric(accum, name, f,
                                                          key)
                                         for f in anchors}
                                if any(v is None or v <= 0
                                       for v in mvals.values()):
                                    fills[key] = None
                                else:
                                    fills[key] = _ladder_metric_fit(
                                        kmap, anchors, mvals, fmt)
                            _accumulate_result(
                                accum, name, fmt,
                                float(fills["weight_mse"] or 0.0),
                                float(fills["output_mse"] or 0.0),
                                float(fills["rel_output_mse"] or 0.0),
                                predicted_dloss=fills["predicted_dloss"],
                                fisher_output_mse=fills["fisher_output_mse"],
                                output_mse_measured=False,
                            )
                    if fail_idx:
                        print(f"[cost] ladder holdout rejected "
                              f"{len(fail_idx)}/{N} tensors in chunk — "
                              f"measuring {sorted(predicted)} for them",
                              flush=True)
                        for fmt in predicted:
                            _measure_spec_into_accum(
                                specs_by_name[fmt], idx=fail_idx)

            del W, X, y_ref, ref_energy
            if h_stacked is not None:
                del h_stacked
            if gq_stacked is not None:
                del gq_stacked
            processed += N
            if processed % (chunk_size * 4) == 0 or processed == total_linears:
                elapsed = time.time() - tstart
                eta = elapsed / processed * (total_linears - processed)
                print(f"[cost] {processed}/{total_linears} "
                      f"eta={eta:.0f}s  ({N} per chunk × {len(specs)} formats)",
                      flush=True)
    if _prefetch_pool is not None:
        _prefetch_pool.shutdown(wait=False)
    return _finalize_results(accum)


def prepare_cost_context(probe_path: str,
                         activation_cache_dir: str,
                         formats_csv: str,
                         skip_missing_activations: bool):
    with open(probe_path, "rb") as f:
        probe = pickle.load(f)
    stats = probe["stats"]
    print(f"[cost] loaded probe stats for {len(stats)} Linears")

    # Refuse pre-fix packed-expert probes: before the 2026-07-02 M3 fix the
    # packed h_trace was sum-then-square (5-50x cross-token-covariance
    # inflation, calibration-length dependent). Version-stamped pickles are
    # required whenever packed-expert entries are present, so a stale
    # probe.pkl reused via skip-if-exists cannot silently drive allocations.
    has_packed = any(
        isinstance(m, dict) and m.get("_packed_experts_module")
        for m in stats.values()
    )
    estimator = (probe.get("meta") or {}).get("packed_fisher_estimator")
    if has_packed and estimator != "per_token_v2":
        if os.environ.get(
                "PRISMAQUANT_ALLOW_SUMSQ_PACKED_FISHER", "0") != "1":
            raise SystemExit(
                f"probe {probe_path} contains packed-expert entries but no "
                f"packed_fisher_estimator=per_token_v2 marker (found: "
                f"{estimator!r}) — it predates the 2026-07-02 per-token "
                "packed-Fisher fix and its expert h_trace values are "
                "sum-then-square inflated (5-50x). Regenerate the probe, or "
                "set PRISMAQUANT_ALLOW_SUMSQ_PACKED_FISHER=1 to accept the "
                "biased legacy estimator.")
        print("[cost] WARNING: accepting pre-fix sum-then-square packed "
              "Fisher probe (PRISMAQUANT_ALLOW_SUMSQ_PACKED_FISHER=1)")

    cache = Path(activation_cache_dir)
    if not cache.exists():
        raise SystemExit(f"activation cache {cache} does not exist")

    if formats_csv:
        fmt_names = [s.strip() for s in formats_csv.split(",") if s.strip()]
    else:
        fmt_names = [s.name for s in fr.list_formats()]
    specs = [fr.get_format(n) for n in fmt_names]
    print(f"[cost] measuring {len(specs)} formats: {[s.name for s in specs]}")

    act_cache = ActivationIndex(cache, stats)
    print(f"[cost] activation cache (lazy index): "
          f"{len(act_cache)} Linears mapped", flush=True)
    if len(act_cache) == 0 and stats:
        # Fail LOUD, not empty: an unpopulated act dir (e.g. probe.pkl copied
        # from a prior run without its act cache — the probe stage is what
        # writes activations) would otherwise produce a 0-row cost.pkl and a
        # blanket-INFEASIBLE allocator (27B 2026-07-21). Measurement gap ->
        # error, never silent garbage.
        raise SystemExit(
            f"activation cache {cache} maps 0 Linears while {len(stats)} "
            "targets expect activations. The act cache is written by the "
            "PROBE stage: if probe.pkl was reused from another run, copy its "
            "act/ directory too, or delete probe.pkl so the probe re-runs "
            "and repopulates activations.")

    target_names = set(stats.keys())
    def _has_activation_for_target(name: str) -> bool:
        if name in act_cache:
            return True
        meta = stats.get(name)
        if isinstance(meta, dict):
            experts_qname = meta.get("_packed_experts_module")
            if isinstance(experts_qname, str) and experts_qname in act_cache:
                return True
        return False

    missing_act = [n for n in target_names if not _has_activation_for_target(n)]
    if missing_act and not skip_missing_activations:
        raise SystemExit(f"{len(missing_act)} Linears missing activation; "
                         f"pass --skip-missing-activations to proceed.")

    return probe, stats, act_cache, target_names, missing_act, fmt_names, specs


def run_cost_pass(model: nn.Module,
                  act_cache: "ActivationIndex",
                  target_names: set[str],
                  missing_act: list[str],
                  specs: list[fr.FormatSpec],
                  model_name: str,
                  probe_path: str,
                  device: str,
                  dtype: torch.dtype,
                  mode: str,
                  chunk_size: int,
                  output_path: str,
                  h_detail_dir: str | None = None):
    chosen_mode = mode
    if chosen_mode == "auto":
        chosen_mode = "batched" if device.startswith("cuda") else "unbatched"
    print(f"[cost] mode: {chosen_mode}")
    try:
        from .model_profiles import profile_from_model
        model_profile = profile_from_model(model)
    except Exception:
        model_profile = None

    h_detail: "HDetailIndex | None" = None
    if h_detail_dir:
        detail_path = Path(h_detail_dir)
        if detail_path.exists():
            h_detail = HDetailIndex(detail_path, target_names)
            print(f"[cost] h-detail cache: {len(h_detail)} / {len(target_names)} "
                  "Linears have h-detail → using per-weight Δloss and "
                  "Fisher row-weighted output MSE when available",
                  flush=True)
        else:
            print(f"[cost] WARN: h-detail dir {detail_path} not found; "
                  "falling back to scalar proxy", flush=True)

    if chosen_mode == "batched":
        results = measure_batched_gpu(model, act_cache, target_names, specs,
                                      device, dtype,
                                      chunk_size=chunk_size,
                                      h_detail=h_detail,
                                      profile=model_profile)
    else:
        results = measure_unbatched(model, act_cache, target_names, specs,
                                    device, dtype,
                                    h_detail=h_detail,
                                    profile=model_profile)

    # Packed-expert tensors aren't visible to the nn.Linear-based path.
    # Measure them separately. Both paths share the same accumulator
    # format so finalization is uniform.
    packed_accum: dict[str, dict] = {}
    _measure_packed_experts(model, target_names, specs, device, dtype,
                            packed_accum, act_cache=act_cache,
                            h_detail=h_detail,
                            profile=model_profile)
    if packed_accum:
        results.update(_finalize_results(packed_accum))
        n_packed = len(packed_accum)
        print(f"[cost] measured {n_packed} packed-expert tensors", flush=True)

    missing_from_results = [n for n in target_names if n not in results]
    if missing_from_results:
        print(f"[cost] WARNING: {len(missing_from_results)} Linears had no "
              f"measurement output (cache miss or skipped)")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "costs": results,
            "formats": [s.name for s in specs],
            "meta": {
                "model": model_name,
                "probe": probe_path,
                "n_linears": len(results),
                "missing_activations": missing_act,
                "mode": chosen_mode,
                "h_detail_dir": str(Path(h_detail_dir)) if h_detail_dir else None,
            },
        }, f)
    print(f"[cost] wrote {out_path} ({len(results)} Linears)")
    return results
