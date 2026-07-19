"""Packed-expert imatrix synthesis — CHECKPOINT-based, no model load.

The activation cache holds only each experts MODULE's input, so the harvested
imatrix (``export_gguf.build_imatrix_from_act_cache``) can never contain:

* ``<qn>.gate_up_proj`` — the harvest emits the MODULE name, while the CB
  cost/export look up the packed-param name;
* ``<qn>.down_proj``   — its input is the PER-EXPERT intermediate, which is
  never cached anywhere.

Both the exporter (hard-fails, "no silent RTN") and the local packed-expert
cost (which would otherwise render down_proj unweighted while the export
ships weighted bytes — the rendering-confound class) need these entries, from
ONE shared source. This module synthesizes them by replaying the routed
forward directly from the CHECKPOINT tensors (router weight + per-expert
gate/up) on the cached module inputs: route -> per-expert gate/up ->
activation -> intermediate, mean-square pooled per expert. The
model-loaded twin of this replay lives in
``expert_empirical_cost.ensure_unit_col_weights``; keep semantics in
lockstep.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from safetensors import safe_open


def _weight_map(model_path: Path) -> dict[str, str]:
    idx = model_path / "model.safetensors.index.json"
    if idx.exists():
        return json.loads(idx.read_text())["weight_map"]
    st = model_path / "model.safetensors"
    if st.exists():
        with safe_open(str(st), framework="pt") as f:
            return {k: "model.safetensors" for k in f.keys()}
    raise FileNotFoundError(f"no safetensors index under {model_path}")


def _load_tensors(model_path: Path, weight_map: dict[str, str],
                  keys: list[str], dtype=torch.float32) -> dict[str, torch.Tensor]:
    by_shard: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        by_shard[weight_map[k]].append(k)
    out: dict[str, torch.Tensor] = {}
    for shard, ks in by_shard.items():
        with safe_open(str(model_path / shard), framework="pt") as f:
            for k in ks:
                out[k] = f.get_tensor(k).to(dtype)
    return out


def _load_act_entry(p: Path) -> tuple[str, torch.Tensor | None]:
    """(module name, input rows) from one act-cache blob — the same schema
    ``export_gguf.build_imatrix_from_act_cache`` consumes."""
    blob = torch.load(p, map_location="cpu", weights_only=False)
    inputs = blob.get("inputs") if isinstance(blob, dict) else None
    name = (blob.get("name") if isinstance(blob, dict) else None) or (
        p.stem.replace("__", "."))
    if inputs is None or inputs.ndim != 2:
        return name, None
    return name, inputs.float()


@torch.no_grad()
def synthesize_packed_expert_col_weights(
    model_path: str | Path,
    act_dir: str | Path,
    col_weights: dict,
    profile=None,
    *,
    max_rows: int = 4096,
    device: str | None = None,
) -> list[str]:
    """Fill missing ``<experts_qn>.gate_up_proj`` / ``.down_proj`` imatrix
    entries in ``col_weights`` IN PLACE from the checkpoint + act cache.

    Returns the names added. Loud failure over silent omission: an experts
    module whose router/per-expert tensors can't be resolved raises (the
    exporter would hard-fail later anyway — better here, with the cause).
    """
    model_path = Path(model_path)
    act_dir = Path(act_dir)
    if profile is None:
        from prismaquant.model_profiles import detect_profile_with_warning
        profile = detect_profile_with_warning(
            str(model_path), entrypoint="moe-imatrix")
    wm = _weight_map(model_path)
    cfg = json.loads((model_path / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    top_k = int(tc.get("num_experts_per_tok", 8))
    norm_topk = bool(tc.get("norm_topk_prob", True))
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    added: list[str] = []
    for p in sorted(act_dir.glob("*.pt")):
        qn, X = _load_act_entry(p)
        src = profile.source_tensor_name(qn)
        if f"{src}.0.gate_proj.weight" not in wm:
            continue                    # not a per-expert experts module
        gu_name, dn_name = f"{qn}.gate_up_proj", f"{qn}.down_proj"
        if gu_name in col_weights and dn_name in col_weights:
            continue
        if X is None:
            raise ValueError(f"{qn}: activation cache entry unreadable — "
                             f"cannot synthesize the packed-expert imatrix")
        X = X[:max_rows].to(dev)

        if gu_name not in col_weights:
            col_weights[gu_name] = (
                X.pow(2).mean(dim=0).reshape(1, 1, -1).cpu())
            added.append(gu_name)
        if dn_name not in col_weights:
            # Router: <parent>.gate.weight in SOURCE naming.
            src_parent = src.rsplit(".", 1)[0]
            gate_key = f"{src_parent}.gate.weight"
            if gate_key not in wm:
                raise ValueError(
                    f"{qn}: router weight {gate_key!r} not in checkpoint — "
                    f"cannot replay routing for the down_proj imatrix")
            E = 0
            while f"{src}.{E}.gate_proj.weight" in wm:
                E += 1
            if E == 0:
                raise ValueError(f"{qn}: no per-expert gate_proj tensors")
            keys = [gate_key]
            for e in range(E):
                keys += [f"{src}.{e}.gate_proj.weight",
                         f"{src}.{e}.up_proj.weight"]
            t = _load_tensors(model_path, wm, keys)
            Wg = t[gate_key].to(dev)
            logits = X @ Wg.t()
            bias_key = f"{src_parent}.gate.e_score_correction_bias"
            if bias_key in wm:
                logits = logits + _load_tensors(
                    model_path, wm, [bias_key])[bias_key].to(dev)
            scores = torch.softmax(logits, dim=-1)
            topv, topi = torch.topk(scores, top_k, dim=-1)
            if norm_topk:
                topv = topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            inter = int(t[f"{src}.0.gate_proj.weight"].shape[0])
            out = torch.zeros(E, inter, dtype=torch.float32, device=dev)
            hit = torch.zeros(E, dtype=torch.bool)
            for e in range(E):
                tok = (topi == e).any(dim=-1).nonzero(as_tuple=True)[0]
                if tok.numel() == 0:
                    continue
                g = X[tok] @ t[f"{src}.{e}.gate_proj.weight"].to(dev).t()
                u = X[tok] @ t[f"{src}.{e}.up_proj.weight"].to(dev).t()
                out[e] = (F.silu(g) * u).pow(2).mean(dim=0)
                hit[e] = True
            if bool(hit.any()) and not bool(hit.all()):
                out[~hit] = out[hit].mean(dim=0)
            elif not bool(hit.any()):
                out[:] = 1.0
            col_weights[dn_name] = out.reshape(E, 1, inter).cpu()
            added.append(dn_name)
            del t
    return added
