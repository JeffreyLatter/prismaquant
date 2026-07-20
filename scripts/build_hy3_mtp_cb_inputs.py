#!/usr/bin/env python3
"""Build the MTP-sidecar layer_config + uniform col-weights for the Hy3 CB draft.

The Hy3 body ships CB-quantised (``prod-hy3-nvfp4cb-2p9``) but its bf16 MTP
draft (``model.layers.80.*``, 7.5 GB) OOMs next to ~102 GiB of weights on the
128 GB Spark. Robert's rule is "always include MTP when it's available", so the
MTP module is CB-quantised too. A draft's weights can NEVER change outputs
(spec-decode is exact via rejection sampling) — they only move the acceptance
rate — so a hand-chosen **modal-rung** policy is acceptable and is documented as
such here rather than allocated end-to-end.

Policy (mirrors the body's dominant rung per role; evidence printed at build):

  * routed experts (``experts.gate_up_proj`` / ``experts.down_proj``)
        -> the body's **modal fp4 expert rung** (computed from its layer_config)
  * shared expert + attention (``shared_experts.{gate,up,down}_proj``,
    ``self_attn.{q,k,v,o}_proj``)  ->  **FP8_CB_K32** (the modal fp8 rung for
    those roles; the sensitive-tail default)

Everything the MTP layer_config does NOT name (``enorm``/``hnorm``/``eh_proj``/
``final_layernorm``, the block's norms, ``q_norm``/``k_norm``, the router gate,
``expert_bias``) stays bf16/f32 — carried through unquantised, exactly like the
body-layer conventions.

Outputs (into ``--out-dir``): ``mtp_layer_config.json`` and
``mtp_col_weights.pkl``, both keyed by the recipe qnames the streaming exporter
expects (``…mlp.experts.gate_up_proj``, ``…mlp.shared_experts.gate_proj``,
``…self_attn.q_proj``). col-weights are **uniform** (all-ones per input column):
no imatrix — the draft-quality caveat above.

CPU-only; reads safetensors metadata (shapes), never tensor data. No GPU.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open

from prismaquant.model_profiles import detect_profile

# Recipe-qname roles under model.layers.{L}. and their target CB family.
# (data_type, cb_k) — cb_k for experts is filled from the computed modal rung.
_ATTN_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj")
_SHARED_LEAVES = ("gate_proj", "up_proj", "down_proj")
_EXPERT_LEAVES = ("gate_up_proj", "down_proj")


def _shard_map(model_dir: Path) -> dict[str, str]:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        return json.loads(index.read_text())["weight_map"]
    single = model_dir / "model.safetensors"
    if not single.exists():
        raise FileNotFoundError(
            f"no model.safetensors[.index.json] under {model_dir}")
    with safe_open(single, framework="pt", device="cpu") as f:
        return {k: "model.safetensors" for k in f.keys()}


def _shape(model_dir: Path, shard_map: dict[str, str], name: str):
    if name not in shard_map:
        raise KeyError(f"{name}: not in source checkpoint")
    with safe_open(model_dir / shard_map[name], framework="pt",
                   device="cpu") as f:
        return tuple(f.get_slice(name).get_shape())


def _canonical_entries(body_lc: dict) -> dict[tuple[str, int], dict]:
    """{(data_type, cb_k): a representative entry dict} from the body config —
    cloned verbatim so the MTP scheme fields (act_bits, act_data_type,
    group_size, ...) are byte-identical to the body's for that rung."""
    out: dict[tuple[str, int], dict] = {}
    for v in body_lc.values():
        dt, k = v.get("data_type"), v.get("cb_k")
        if dt in ("nvfp4_cb", "fp8_cb") and k is not None:
            out.setdefault((dt, int(k)), dict(v))
    return out


def _modal_expert_fp4_k(body_lc: dict) -> tuple[int, Counter]:
    """The modal fp4 cb_k over the body's routed-expert entries (both
    gate_up_proj and down_proj), with the full distribution for the record."""
    dist: Counter = Counter()
    for q, v in body_lc.items():
        if ".mlp.experts." in q and q.rsplit(".", 1)[1] in _EXPERT_LEAVES \
                and v.get("data_type") == "nvfp4_cb":
            dist[int(v["cb_k"])] += 1
    if not dist:
        raise ValueError("no nvfp4_cb routed-expert entries in the body "
                         "layer_config — cannot pick a modal fp4 expert rung")
    return dist.most_common(1)[0][0], dist


def build(source_dir: Path, body_layer_config: Path, out_dir: Path,
          mtp_layer: int, expert_fp4_k: int | None,
          shared_attn_fp8_k: int) -> dict:
    body_lc = json.loads(Path(body_layer_config).read_text())
    canon = _canonical_entries(body_lc)
    modal_k, dist = _modal_expert_fp4_k(body_lc)
    if expert_fp4_k is None:
        expert_fp4_k = modal_k

    profile = detect_profile(str(source_dir))
    if profile is None:
        raise RuntimeError(f"no model profile detected for {source_dir}")
    shard_map = _shard_map(Path(source_dir))
    L = mtp_layer
    base = f"model.layers.{L}"

    # (recipe qname, (data_type, cb_k), source-key-for-in_features)
    targets: list[tuple[str, tuple[str, int], str]] = []
    for leaf in _ATTN_LEAVES:
        q = f"{base}.self_attn.{leaf}"
        targets.append((q, ("fp8_cb", shared_attn_fp8_k),
                        profile.source_tensor_name(q) + ".weight"))
    for leaf in _SHARED_LEAVES:
        q = f"{base}.mlp.shared_experts.{leaf}"
        targets.append((q, ("fp8_cb", shared_attn_fp8_k),
                        profile.source_tensor_name(q) + ".weight"))
    # routed experts: fused gate_up (in from expert 0's gate_proj) + down.
    targets.append((f"{base}.mlp.experts.gate_up_proj",
                    ("nvfp4_cb", expert_fp4_k),
                    f"{base}.mlp.experts.0.gate_proj.weight"))
    targets.append((f"{base}.mlp.experts.down_proj",
                    ("nvfp4_cb", expert_fp4_k),
                    f"{base}.mlp.experts.0.down_proj.weight"))

    layer_config: dict[str, dict] = {}
    col_weights: dict[str, torch.Tensor] = {}
    for qname, key, src in targets:
        if key not in canon:
            raise ValueError(
                f"{qname}: body layer_config has no {key[0]} cb_k={key[1]} "
                "entry to clone the scheme from — pick an existing rung")
        layer_config[qname] = dict(canon[key])
        in_features = int(_shape(Path(source_dir), shard_map, src)[1])
        if in_features % 256 != 0:
            raise ValueError(
                f"{qname}: in_features={in_features} not a multiple of 256 "
                "(CB superblock) — cannot CB-quantise")
        col_weights[qname] = torch.ones(in_features, dtype=torch.float32)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lc_path = out_dir / "mtp_layer_config.json"
    cw_path = out_dir / "mtp_col_weights.pkl"
    lc_path.write_text(json.dumps(layer_config, indent=2, sort_keys=True))
    with open(cw_path, "wb") as fh:
        pickle.dump({k: v for k, v in col_weights.items()}, fh)

    print(f"[mtp-inputs] modal fp4 expert rung: NVFP4_CB_K{modal_k}  "
          f"(fp4 dist over experts: "
          f"{dict(sorted(dist.items()))})")
    if expert_fp4_k != modal_k:
        print(f"[mtp-inputs] OVERRIDE expert fp4 rung -> NVFP4_CB_K{expert_fp4_k}")
    print(f"[mtp-inputs] shared+attention rung: FP8_CB_K{shared_attn_fp8_k}")
    print(f"[mtp-inputs] {len(layer_config)} CB targets for layer {L}:")
    for q in sorted(layer_config):
        e = layer_config[q]
        fmt = (f"NVFP4_CB_K{e['cb_k']}" if e["data_type"] == "nvfp4_cb"
               else f"FP8_CB_K{e['cb_k']}")
        print(f"    {q:52s} {fmt:14s} in={col_weights[q].numel()}")
    print(f"[mtp-inputs] wrote {lc_path}")
    print(f"[mtp-inputs] wrote {cw_path}")
    return {"layer_config": str(lc_path), "col_weights": str(cw_path),
            "expert_fp4_k": expert_fp4_k, "shared_attn_fp8_k": shared_attn_fp8_k,
            "n_targets": len(layer_config)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-dir", required=True,
                    help="HF bf16 source (has model.layers.{L}.* MTP tensors)")
    ap.add_argument("--body-layer-config", required=True,
                    help="the body artifact's layer_config.json (rung source)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mtp-layer", type=int, default=80,
                    help="the MTP sidecar layer index (num_hidden_layers)")
    ap.add_argument("--expert-fp4-k", type=int, default=None,
                    help="override the routed-expert fp4 rung (default: modal)")
    ap.add_argument("--shared-attn-fp8-k", type=int, default=32,
                    help="the shared-expert + attention fp8 rung (default 32)")
    args = ap.parse_args(argv)
    build(Path(args.source_dir), Path(args.body_layer_config),
          Path(args.out_dir), args.mtp_layer, args.expert_fp4_k,
          args.shared_attn_fp8_k)


if __name__ == "__main__":
    main()
