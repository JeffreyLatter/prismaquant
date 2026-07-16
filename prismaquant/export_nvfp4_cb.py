"""Materialize a PrismaQuant recipe as an NVFP4-CB / FP8-CB checkpoint.

Sibling of :mod:`prismaquant.export_gguf` — the same skeleton-requantize
strategy, but the container is safetensors + a custom compressed-tensors-**style**
``quant_config.json`` whose scheme vocabulary (``nvfp4_cb`` / ``fp8_cb``) only
the out-of-tree vLLM plugin understands (docs/nvfp4-cb-plan/serving-kernel.md
§2). It is explicitly **not** stock compressed-tensors (whose schemes cannot
express codebooks) — do not route a CB assignment through
:mod:`prismaquant.export_native_compressed`; that exporter hard-fails on CB.

Pipeline: read the bf16 HF skeleton (config.json + *.safetensors), VQ-pack each
target Linear with the **same** weighted closure the cost measured
(:func:`prismaquant.nvfp4_cb_formats.nvfp4_cb_pack`), copy every non-target
tensor verbatim (bf16 passthrough), and emit:

  * ``<name>.cb_qweight``  uint8 (rows, bytes_per_row) — the §1 superblock byte
    stream (index bits + fp4 group-16 E4M3 scale plane; fp8 index bits only);
  * ``<name>.weight_scale`` fp32 (out_features,) — fp8 families only (fp8 has no
    on-disk scale plane; the plane is per-output-channel);
  * ``cb_codebook.<ref>.<fmt>[.sub{i}]`` fp16 — the resolved codebook, shipped
    **once** per (ref, format): ``ref = "lattice"`` for the fixed lattice,
    ``ref = "<role>"`` for a shared per-(role) learned codebook;
  * ``config.json`` (verbatim + a ``quantization_config`` pointer) and
    ``quant_config.json`` (the custom scheme + provenance).

Bit-layout + tensor-naming + config-schema contract: docs/nvfp4-cb-plan/LAYOUT.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from prismaquant import nvfp4_cb_formats as cb
from prismaquant.layer_config import (
    _NVFP4_CB_FORMAT_NAMES,
    load_assignment,
)

# Codebook entry bytes for the sidecar-honest footprint (footprint.py owns the
# real accounting; recorded here in provenance for cross-checking).
_CB_FAMILY_ENTRY_BYTES = {"fp4": 4, "fp8": 8}


def _git_commit() -> str:
    from prismaquant.aura_cost import _git_commit as _aura_git_commit

    return _aura_git_commit() or "unknown"


def _parse_cb_format(fmt: str) -> tuple[str, str, int] | None:
    """``NVFP4_CB_K{k}`` -> (fp4, product, k); ``NVFP4_CB_S{k}`` -> (fp4,
    signed, k); ``FP8_CB_K{k}`` -> (fp8, product, k). None for non-CB."""
    up = str(fmt).strip().upper()
    if up not in _NVFP4_CB_FORMAT_NAMES:
        return None
    if up.startswith("NVFP4_CB_S"):
        return "fp4", "signed", int(up[len("NVFP4_CB_S"):])
    if up.startswith("NVFP4_CB_K"):
        return "fp4", "product", int(up[len("NVFP4_CB_K"):])
    if up.startswith("FP8_CB_K"):
        return "fp8", "product", int(up[len("FP8_CB_K"):])
    return None


def _role_of(qname: str) -> str:
    """Shared-codebook grouping key — the Linear's projection role (last qname
    component), e.g. ``model.layers.3.mlp.gate_proj`` -> ``gate_proj``."""
    return qname.split(".")[-1]


def _load_skeleton(model_dir: Path) -> dict[str, torch.Tensor]:
    """Load every tensor from a HF safetensors dir (single file or sharded)."""
    index = model_dir / "model.safetensors.index.json"
    tensors: dict[str, torch.Tensor] = {}
    if index.exists():
        shards = sorted({
            v for v in json.loads(index.read_text())["weight_map"].values()
        })
        for shard in shards:
            tensors.update(load_file(str(model_dir / shard)))
        return tensors
    single = model_dir / "model.safetensors"
    if not single.exists():
        raise FileNotFoundError(
            f"no model.safetensors[.index.json] under {model_dir}")
    return load_file(str(single))


def _vecs_and_wq(w: torch.Tensor, cw: torch.Tensor | None, grid: str):
    """One-shot scaled 8-dim vectors + per-vector weights for one Linear (the
    same scaling the encoder feeds the VQ search) — mirrors the exp1b driver's
    shared-codebook pooling."""
    w2d = w.reshape(-1, w.shape[-1]).to(torch.float32)
    vectors, _, _ = cb._scale_and_vectorize(w2d, grid)
    wq = None
    if cw is not None:
        cw2d = torch.broadcast_to(cw.to(w2d.device, torch.float32),
                                  w2d.shape).contiguous()
        wq = cb._col_weight_vectors(cw2d)
    return vectors, wq


def _train_shared_codebook(weights, cws, *, grid, mode, k, seed, iters,
                           train_cap):
    """One learned codebook over a role's pooled scaled vectors (the exp1b
    shared-per-role logic): signed -> positive magnitude table; product ->
    n_sub grid-snapped sub-tables; full -> one (2^k, 8) table."""
    vlist, wlist = [], []
    for w, cw in zip(weights, cws):
        v, wq = _vecs_and_wq(w, cw, grid)
        vlist.append(v)
        wlist.append(wq if wq is not None else torch.ones_like(v))
    vec = torch.cat(vlist, 0)
    wq = torch.cat(wlist, 0)
    if vec.shape[0] > train_cap:
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(vec.shape[0], generator=g)[:train_cap].to(
            vec.device)
        vec, wq = vec[idx], wq[idx]
    if mode == "signed":
        return cb.learn_codebook(vec.abs(), k - cb.VEC_DIM, grid=grid,
                                 col_weights=wq, positive=True, iters=iters,
                                 seed=seed).cpu()
    if mode == "product":
        n_sub = cb._product_n_sub(grid)
        sub_dim = cb.VEC_DIM // n_sub
        bits = cb._bit_split(k, n_sub)
        subs = []
        for i, b in enumerate(bits):
            xs = vec[:, i * sub_dim:(i + 1) * sub_dim]
            ws = wq[:, i * sub_dim:(i + 1) * sub_dim]
            init_i = cb.fixed_lattice(b, grid, sub_dim).to(vec.device)
            subs.append(cb.learn_codebook(xs, b, grid=grid, col_weights=ws,
                                          init=init_i, iters=iters,
                                          seed=seed).cpu())
        return tuple(subs)
    return cb.learn_codebook(vec, k, grid=grid, col_weights=wq, iters=iters,
                             seed=seed).cpu()


def _codebook_tensors(ref: str, fmt: str, codebook) -> dict[str, torch.Tensor]:
    """Serialize a codebook (single table or product sub-table tuple) to fp16
    safetensors tensors under ``cb_codebook.<ref>.<fmt>[.sub{i}]`` (grid values
    are exact in fp16 for both the E2M1 and E4M3 grids)."""
    base = f"cb_codebook.{ref}.{fmt}"
    if isinstance(codebook, (tuple, list)):
        return {f"{base}.sub{i}": t.to(torch.float16).cpu().contiguous()
                for i, t in enumerate(codebook)}
    return {base: codebook.to(torch.float16).cpu().contiguous()}


def export_nvfp4_cb(
    model_dir: str | Path,
    layer_config_path: str | Path,
    out_dir: str | Path,
    col_weights: dict[str, torch.Tensor],
    *,
    shared_codebook_spec: dict | None = None,
    device: str | None = None,
    scale_sweep: bool = True,
) -> dict[str, int]:
    """Export a CB checkpoint. See module docstring / LAYOUT.md for the layout.

    ``col_weights`` maps each CB-target qname to its per-input-column importance
    (imatrix / Fisher). ``shared_codebook_spec`` (or None) selects the codebook
    source:

      * ``None`` / ``{"source": "lattice"}`` — the deterministic fixed lattice
        (no per-tensor sidecar), shipped once per format;
      * ``{"source": "learned", "train": True, "iters", "seed", "train_cap"}`` —
        a shared per-(role) learned codebook trained here on pooled vectors;
      * ``{"source": "learned", "codebooks": {role: cb_obj}}`` — use provided
        per-role codebooks (a missing role for a target hard-fails).
    """
    model_dir = Path(model_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = shared_codebook_spec or {}
    source = str(spec.get("source", "lattice")).lower()
    if source not in ("lattice", "learned"):
        raise ValueError(f"shared_codebook_spec source must be lattice/learned,"
                         f" got {source!r}")

    assignment = load_assignment(layer_config_path)
    skeleton = _load_skeleton(model_dir)

    # --- Coverage gate: classify + validate every assigned format. ---
    cb_targets: dict[str, tuple[str, str, int]] = {}   # qname -> (grid,mode,k)
    illegal = []
    for qname, fmt in assignment.items():
        if fmt == "BF16":
            continue
        parsed = _parse_cb_format(fmt)
        if parsed is None:
            illegal.append((qname, fmt))
            continue
        cb_targets[qname] = parsed
    if illegal:
        raise ValueError(
            f"assignment contains formats the NVFP4-CB container cannot carry: "
            f"{sorted({f for _, f in illegal})} — allocate with "
            f"--target-profile nvfp4_cb (BF16 passthrough is the only non-CB "
            f"format allowed)")

    for qname, (grid, mode, k) in cb_targets.items():
        wname = qname + ".weight"
        if wname not in skeleton:
            raise ValueError(
                f"{qname}: assigned {grid}/{mode} k{k} but no weight tensor "
                f"'{wname}' in the skeleton")
        in_f = int(skeleton[wname].shape[-1])
        if in_f % cb.SUPERBLOCK != 0:
            raise ValueError(
                f"{qname}: in_features={in_f} is not a multiple of "
                f"{cb.SUPERBLOCK}; fall back to a coarser legal rung or BF16 "
                f"(no block-32 CB rung in Phase 0)")
        if qname not in col_weights:
            raise ValueError(
                f"{qname}: CB target has no col_weights entry — exporting "
                f"unweighted bytes would silently diverge from the "
                f"imatrix-weighted cost measurement (no silent RTN)")
        cwn = col_weights[qname].numel()
        if cwn != in_f:
            raise ValueError(
                f"{qname}: col_weights has {cwn} columns but the weight has "
                f"{in_f} — the imatrix does not describe this checkpoint")

    # --- Resolve/train codebooks, grouped by (ref, format). ---
    provided = spec.get("codebooks", {}) if source == "learned" else {}
    train = bool(spec.get("train", False))
    iters = int(spec.get("iters", 4))
    seed = int(spec.get("seed", 0))
    train_cap = int(spec.get("train_cap", 1 << 20))

    # (ref, fmt) -> codebook object; ref = "lattice" or role.
    codebooks: dict[tuple[str, str], object] = {}
    # qname -> (ref, fmt, codebook, source_kind)
    target_cb: dict[str, tuple[str, str, object, str]] = {}
    by_group: dict[tuple[str, str], list[str]] = {}
    for qname, (grid, mode, k) in cb_targets.items():
        fmt = assignment[qname]
        ref = _role_of(qname) if source == "learned" else "lattice"
        by_group.setdefault((ref, fmt), []).append(qname)

    for (ref, fmt), qnames in by_group.items():
        grid, mode, k = cb_targets[qnames[0]]
        if source == "lattice":
            codebooks[(ref, fmt)] = cb._resolve_codebook(
                k, grid, mode, None, torch.device(device))
            kind = "lattice"
        else:
            role = ref
            if train:
                weights = [skeleton[q + ".weight"].to(device) for q in qnames]
                cws = [col_weights[q].to(device) for q in qnames]
                codebooks[(ref, fmt)] = _train_shared_codebook(
                    weights, cws, grid=grid, mode=mode, k=k, seed=seed,
                    iters=iters, train_cap=train_cap)
            elif role in provided:
                codebooks[(ref, fmt)] = provided[role]
            else:
                raise ValueError(
                    f"role {role!r} ({fmt}): codebook_source=learned but no "
                    f"codebook supplied and train=False — missing learned "
                    f"sidecar for {len(qnames)} tensor(s)")
            kind = "learned"
        for q in qnames:
            target_cb[q] = (ref, fmt, codebooks[(ref, fmt)], kind)

    # --- Pack targets; copy everything else verbatim. ---
    out_tensors: dict[str, torch.Tensor] = {}
    cb_tensor_blobs: dict[str, torch.Tensor] = {}
    counts: Counter[str] = Counter()
    ignore: list[str] = []
    packed_qnames = set(cb_targets)

    for name, tensor in skeleton.items():
        qname = name[:-len(".weight")] if name.endswith(".weight") else None
        if qname in packed_qnames:
            grid, mode, k = cb_targets[qname]
            ref, fmt, codebook, _ = target_cb[qname]
            cbook = _to_device(codebook, device)
            w = tensor.to(device)
            packed, fields = cb.nvfp4_cb_pack(
                w, k, grid=grid, mode=mode,
                col_weights=col_weights[qname].to(device),
                codebook=cbook, scale_sweep=scale_sweep)
            out_tensors[qname + ".cb_qweight"] = packed.to(torch.uint8).cpu(
            ).contiguous()
            if grid == "fp8":
                out_tensors[qname + ".weight_scale"] = fields["scales"].reshape(
                    -1).to(torch.float32).cpu().contiguous()
            counts[fmt] += 1
        else:
            out_tensors[name] = tensor.contiguous()
            if qname is not None and tensor.dim() >= 2:
                ignore.append(qname)
            counts["copied"] += 1

    # --- Codebook tensors, shipped once per (ref, fmt). ---
    for (ref, fmt), codebook in codebooks.items():
        for tname, blob in _codebook_tensors(ref, fmt, codebook).items():
            cb_tensor_blobs[tname] = blob
    out_tensors.update(cb_tensor_blobs)

    # --- Provenance hashes. ---
    assignment_sha = hashlib.sha256(json.dumps(
        dict(sorted(assignment.items())), separators=(",", ":")).encode(),
    ).hexdigest()
    ih = hashlib.sha256()
    for q in sorted(col_weights):
        ih.update(q.encode())
        ih.update(col_weights[q].to(torch.float32).cpu().numpy().tobytes())
    imatrix_sha = ih.hexdigest()
    codebook_sha = {
        tname: hashlib.sha256(
            blob.to(torch.float16).cpu().numpy().tobytes()).hexdigest()
        for tname, blob in cb_tensor_blobs.items()
    }

    # --- Custom quant config (config_groups keyed by scheme signature). ---
    config_groups: dict[str, dict] = {}
    for gi, ((ref, fmt), qnames) in enumerate(sorted(by_group.items())):
        grid, mode, k = cb_targets[qnames[0]]
        codebook = codebooks[(ref, fmt)]
        n_sub = len(codebook) if isinstance(codebook, (tuple, list)) else 1
        base = f"cb_codebook.{ref}.{fmt}"
        codebook_ref = ([f"{base}.sub{i}" for i in range(n_sub)]
                        if n_sub > 1 else base)
        config_groups[f"group_{gi}"] = {
            "targets": sorted(qnames),
            "format": fmt,
            "scheme": {
                "grid": grid,
                "mode": mode,
                "k": k,
                "superblock": cb.SUPERBLOCK,
                "group_size": cb.FP4_GROUP if grid == "fp4" else 0,
                "vec_dim": cb.VEC_DIM,
                "n_sub": n_sub,
                "type_size": cb.nvfp4_cb_type_size(k, grid),
                "act_bits": 4 if grid == "fp4" else 8,
                "codebook_source": (
                    "lattice" if ref == "lattice" else "learned"),
                "codebook_ref": codebook_ref,
                "codebook_group": None if ref == "lattice" else ref,
            },
        }
    quant_config = {
        "quant_method": "prismaquant",
        "format": "nvfp4_cb",
        "config_groups": config_groups,
        "ignore": sorted(set(ignore)),
        "provenance": {
            "git_commit": _git_commit(),
            "assignment_sha256": assignment_sha,
            "imatrix_sha256": imatrix_sha,
            "codebook_sha256": codebook_sha,
            "codebook_source": source,
            "scale_sweep": bool(scale_sweep),
            "cb_targets": len(cb_targets),
            "tensor_formats": {q: assignment[q] for q in sorted(cb_targets)},
        },
    }

    # --- Write safetensors + configs. ---
    save_file(out_tensors, str(out_dir / "model.safetensors"),
              metadata={"format": "pt", "quant_method": "prismaquant"})
    (out_dir / "quant_config.json").write_text(
        json.dumps(quant_config, indent=2, sort_keys=True))
    src_config = model_dir / "config.json"
    config = json.loads(src_config.read_text()) if src_config.exists() else {}
    config["quantization_config"] = {
        "quant_method": "prismaquant",
        "format": "nvfp4_cb",
        "config_file": "quant_config.json",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    # Copy tokenizer / generation sidecars verbatim (best effort).
    for aux in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model",
                "special_tokens_map.json", "generation_config.json",
                "vocab.json", "merges.txt"):
        p = model_dir / aux
        if p.exists():
            (out_dir / aux).write_bytes(p.read_bytes())
    return dict(counts)


def _to_device(codebook, device):
    if isinstance(codebook, (tuple, list)):
        return tuple(t.to(device) for t in codebook)
    return codebook.to(device)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True,
                    help="HF model dir (config.json + *.safetensors, bf16)")
    ap.add_argument("--layer-config", required=True,
                    help="assignment JSON (qname -> CB format)")
    ap.add_argument("--out", required=True, help="output checkpoint dir")
    ap.add_argument("--col-weights", required=True,
                    help="pickle: {qname: per-column importance tensor}")
    ap.add_argument("--codebook-source", default="lattice",
                    choices=["lattice", "learned"],
                    help="fixed lattice (no sidecar) or shared per-role "
                    "learned codebooks trained at export time")
    ap.add_argument("--codebook-iters", type=int, default=4)
    ap.add_argument("--codebook-seed", type=int, default=0)
    ap.add_argument("--no-scale-sweep", action="store_true",
                    help="one-shot amax/grid-max scale (A/B only; default is "
                    "the joint scale sweep, IQ-rendering parity)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    with open(args.col_weights, "rb") as fh:
        col_weights = pickle.load(fh)
    col_weights = {k: torch.as_tensor(v) for k, v in col_weights.items()}
    spec = {"source": args.codebook_source}
    if args.codebook_source == "learned":
        spec.update(train=True, iters=args.codebook_iters,
                    seed=args.codebook_seed)
    counts = export_nvfp4_cb(
        args.model_dir, args.layer_config, args.out, col_weights,
        shared_codebook_spec=spec, device=args.device,
        scale_sweep=not args.no_scale_sweep,
    )
    size = sum(p.stat().st_size for p in Path(args.out).glob("*")) / 1e9
    print(f"wrote {args.out} ({size:.3f} GB)")
    for fmt, n in sorted(counts.items()):
        print(f"  {fmt}: {n}")


if __name__ == "__main__":
    main()
