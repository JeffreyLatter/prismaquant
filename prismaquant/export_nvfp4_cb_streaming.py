"""Streaming NVFP4-CB / FP8-CB exporter for 200-300B-class models.

Sibling of :mod:`prismaquant.export_nvfp4_cb` (the in-memory exporter) built
for models whose full weights do not fit resident: Hy3 (~557 GB, bf16) and
DSv4-Flash (~295 GB, fp8-native). The in-memory exporter's ``_load_skeleton``
materialises EVERY shard into one dict and accumulates EVERY output tensor
before writing (dsv4_readiness.md gap 2) — a full-model materialisation twice
over. This exporter streams instead:

  * lazy shard index (``_LazySkeleton``): one source tensor resident at a time
    (the ``export_gguf_direct._ShardIndex`` pattern), with fp8-block
    dequant-on-read for native-fp8 sources (``layer_streaming``);
  * per-expert -> stacked bridging: MoE experts stored per-expert on disk
    (Hy3: ``…experts.{i}.{gate,up,down}_proj``) are packed one expert at a
    time and the SMALL packed byte-rows stacked — never all experts resident;
  * two-pass streaming safetensors write (``_StreamWriter``): sizes are
    computed analytically in pass 1 (CB type_size / source metadata, no data
    load), the header is written, then each tensor is produced+written+freed
    in pass 2. Peak residency ~= one source tensor + the codebooks.

The PACKED BYTES are identical to the in-memory exporter (both call
``cb.nvfp4_cb_pack``; CB scales are per-expert/per-row/per-group, so packing
one expert alone equals packing it inside the stack) — pinned byte-for-byte
in tests/test_nvfp4_cb_streaming.py. Container/config/sidecar mirror
export_nvfp4_cb + LAYOUT.md exactly.

Scope (this milestone): bf16 source + fp8-source READ + CB families + BF16
passthrough + FP8_SOURCE passthrough. Stock-CT mixed rungs (NVFP4 /
FP8_DYNAMIC quantised in-container) are NOT streamed yet — they hard-fail with
a pointer to the in-memory exporter (their codec output sizes need a pack to
know; a bounded extension). See the module TODOs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import struct
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open

from prismaquant import nvfp4_cb_formats as cb
from prismaquant.layer_config import load_assignment
from prismaquant.model_profiles import detect_profile
from prismaquant.export_nvfp4_cb import (
    _canonical_qname,
    _codebook_tensors,
    _export_base_name,
    _git_commit,
    _parse_cb_format,
    _role_of,
    _to_device,
    _try_resolve_skeleton,
)

# safetensors dtype string codes for the tensors we emit.
_ST_DTYPE = {
    torch.uint8: "U8", torch.float32: "F32", torch.float16: "F16",
    torch.bfloat16: "BF16", torch.float8_e4m3fn: "F8_E4M3",
    torch.int64: "I64", torch.int32: "I32", torch.int8: "I8",
    torch.bool: "BOOL",
}
_EXPERT_RE = re.compile(
    r"^(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


# ---------------------------------------------------------------------------
# Lazy weight source (one tensor resident at a time; fp8-block dequant-on-read)
# ---------------------------------------------------------------------------

class _LazySkeleton:
    """Shard-indexed lazy safetensors reader. ``__contains__`` matches the
    dict-skeleton contract the export_nvfp4_cb name resolvers expect, but
    tensor data is only touched on ``load``/``dequant_weight`` and shape/dtype
    come from the safetensors metadata (no data load)."""

    def __init__(self, model_dir: str | Path):
        self.dir = Path(model_dir)
        index = self.dir / "model.safetensors.index.json"
        if index.exists():
            self.weight_map = json.loads(index.read_text())["weight_map"]
        else:
            single = self.dir / "model.safetensors"
            if not single.exists():
                raise FileNotFoundError(
                    f"no model.safetensors[.index.json] under {self.dir}")
            with safe_open(single, framework="pt", device="cpu") as f:
                self.weight_map = {k: "model.safetensors" for k in f.keys()}
        self._open: dict[str, object] = {}
        # fp8-block weight_scale_inv sibling map (native-fp8 sources), keyed by
        # the base name (so `dequant_weight` can apply it on read).
        self._block_size: tuple[int, int] | None = None

    def __contains__(self, name: str) -> bool:
        return name in self.weight_map

    def keys(self):
        return self.weight_map.keys()

    def _handle(self, name: str):
        shard = self.weight_map[name]
        if shard not in self._open:
            self._open[shard] = safe_open(
                self.dir / shard, framework="pt", device="cpu")
        return self._open[shard]

    def get_shape(self, name: str) -> tuple[int, ...]:
        return tuple(self._handle(name).get_slice(name).get_shape())

    def get_dtype(self, name: str) -> torch.dtype:
        t = self._handle(name).get_slice(name)[0:0]
        return t.dtype

    def load(self, name: str) -> torch.Tensor:
        return self._handle(name).get_tensor(name)

    def _fp8_block(self) -> tuple[int, int]:
        if self._block_size is None:
            cfg = self.dir / "config.json"
            bs = (128, 128)
            if cfg.exists():
                qc = json.loads(cfg.read_text()).get(
                    "quantization_config", {}) or {}
                wbs = qc.get("weight_block_size")
                if isinstance(wbs, (list, tuple)) and len(wbs) == 2:
                    bs = (int(wbs[0]), int(wbs[1]))
            self._block_size = bs
        return self._block_size

    def dequant_weight(self, weight_key: str) -> torch.Tensor:
        """Return the weight as fp32 for encoding. bf16/fp16 sources cast
        straight through; native-fp8 sources apply the ``weight_scale_inv``
        block scale (``layer_streaming._dequant_fp8_block_weight``) — the
        DSv4/MiniMax fp8-block ingestion path."""
        w = self.load(weight_key)
        if w.dtype == torch.float8_e4m3fn:
            base = weight_key[:-len(".weight")]
            sname = base + ".weight_scale_inv"
            if sname in self.weight_map:
                from prismaquant.layer_streaming import (
                    _dequant_fp8_block_weight,
                )
                scale = self.load(sname).float()
                deq = _dequant_fp8_block_weight(
                    w, scale, block=self._fp8_block(), name=base)
                return deq.float()
            # per-tensor-scale fp8 (rare) — fall through to raw cast.
        return w.to(torch.float32)


# ---------------------------------------------------------------------------
# Streaming safetensors writer (analytic sizes -> header -> streamed data)
# ---------------------------------------------------------------------------

def _raw_bytes(t: torch.Tensor) -> bytes:
    return t.detach().contiguous().flatten().view(torch.uint8).numpy().tobytes()


def _nbytes(dtype: torch.dtype, shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n * torch.empty((), dtype=dtype).element_size()


class _StreamWriter:
    """Two-pass safetensors writer. ``add`` records (name, dtype, shape) and a
    zero-arg ``producer`` that yields the tensor at write time; ``write`` lays
    out contiguous offsets, writes the header, then streams every producer's
    bytes in order — one output tensor resident at a time."""

    def __init__(self):
        self._entries: list[tuple[str, torch.dtype, tuple, object]] = []

    def add(self, name, dtype, shape, producer):
        self._entries.append((name, dtype, tuple(int(d) for d in shape),
                              producer))

    def names(self) -> list[str]:
        return [e[0] for e in self._entries]

    def write(self, path: Path) -> None:
        header: dict[str, dict] = {}
        off = 0
        for name, dtype, shape, _ in self._entries:
            nb = _nbytes(dtype, shape)
            header[name] = {"dtype": _ST_DTYPE[dtype], "shape": list(shape),
                            "data_offsets": [off, off + nb]}
            off += nb
        header["__metadata__"] = {"format": "pt", "quant_method": "prismaquant"}
        hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(hjson)))
            f.write(hjson)
            for name, dtype, shape, producer in self._entries:
                t = producer()
                if t.dtype != dtype or tuple(t.shape) != shape:
                    raise AssertionError(
                        f"{name}: produced {t.dtype}{tuple(t.shape)} != "
                        f"declared {dtype}{shape}")
                b = _raw_bytes(t)
                if len(b) != _nbytes(dtype, shape):
                    raise AssertionError(f"{name}: byte count mismatch")
                f.write(b)
                del t, b


# ---------------------------------------------------------------------------
# Per-expert -> stacked plan
# ---------------------------------------------------------------------------

def _plan_expert_stacks(skeleton: _LazySkeleton) -> dict[str, dict]:
    """Group per-expert on-disk tensors into stacked-output plans keyed by the
    LIVE packed qname (``…experts.gate_up_proj`` = fused gate+up, or
    ``…experts.down_proj``). Returns {live_packed_base: {proj: {i: base}}}."""
    experts: dict[str, dict[str, dict[int, str]]] = {}
    for name in skeleton.keys():
        m = _EXPERT_RE.match(name)
        if not m:
            continue
        prefix, idx, proj = m.group(1), int(m.group(2)), m.group(3)
        base = name[:-len(".weight")]
        experts.setdefault(prefix, {}).setdefault(proj, {})[idx] = base
    return experts


def _stacked_source_weight(skeleton, prefix, packed_proj, members) -> \
        torch.Tensor:
    """Materialise the full stacked source weight (E, out, in) for a packed
    expert group — used only where a stack must be resident (codebook
    training sampling); the packer streams per expert."""
    return torch.stack([_expert_weight(skeleton, prefix, packed_proj,
                                       members, e)
                        for e in range(_n_experts(members))])


def _n_experts(members: dict[str, dict[int, str]]) -> int:
    proj = next(iter(members))
    ids = members[proj]
    n = len(ids)
    if sorted(ids) != list(range(n)):
        raise ValueError(f"non-contiguous expert ids for {proj}: {sorted(ids)}")
    return n


def _expert_weight(skeleton, prefix, packed_proj, members, e) -> torch.Tensor:
    """One expert's fp32 weight for the packed projection: gate_up_proj fuses
    cat([gate_e, up_e], dim=0); down_proj is the single down_e."""
    if packed_proj == "gate_up_proj":
        g = skeleton.dequant_weight(members["gate_proj"][e] + ".weight")
        u = skeleton.dequant_weight(members["up_proj"][e] + ".weight")
        return torch.cat([g, u], dim=0)
    proj = packed_proj if packed_proj in members else "down_proj"
    return skeleton.dequant_weight(members[proj][e] + ".weight")


# ---------------------------------------------------------------------------
# Streaming export
# ---------------------------------------------------------------------------

def export_nvfp4_cb_streaming(
    model_dir: str | Path,
    layer_config_path: str | Path,
    out_dir: str | Path,
    col_weights: dict[str, torch.Tensor],
    *,
    shared_codebook_spec: dict | None = None,
    device: str | None = None,
    scale_sweep: bool = True,
    scale_coding: str = cb.SCALE_CODING_V1,
) -> dict[str, int]:
    """Streaming counterpart of :func:`export_nvfp4_cb.export_nvfp4_cb`. Same
    signature + container; peak residency ~= one source tensor + codebooks.
    See the module docstring for the scope of this milestone."""
    model_dir = Path(model_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if scale_coding not in (cb.SCALE_CODING_V1, cb.SCALE_CODING_TWO_TIER):
        raise ValueError(f"unknown scale_coding {scale_coding!r}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = shared_codebook_spec or {}
    source = str(spec.get("source", "lattice")).lower()
    if source not in ("lattice", "learned"):
        raise ValueError(f"shared_codebook_spec source must be lattice/learned")

    assignment = load_assignment(layer_config_path)
    skeleton = _LazySkeleton(model_dir)
    try:
        profile = detect_profile(str(model_dir))
    except Exception:
        profile = None
    expert_groups = _plan_expert_stacks(skeleton)

    # --- Classify every assignment target (CB / FP8_SOURCE / stock-CT). ---
    cb_targets: dict[str, tuple[str, str, int]] = {}
    source_targets: list[str] = []
    stock_illegal = []
    for qname, fmt in assignment.items():
        if fmt == "BF16":
            continue
        parsed = _parse_cb_format(fmt)
        if parsed is not None:
            cb_targets[qname] = parsed
            continue
        from prismaquant.format_registry import canonical_format_name
        if canonical_format_name(fmt) == "FP8_SOURCE":
            source_targets.append(qname)
            continue
        stock_illegal.append((qname, fmt))
    if stock_illegal:
        raise ValueError(
            "streaming CB export supports CB families + FP8_SOURCE + BF16 "
            "only; stock-CT rungs "
            f"{sorted({f for _, f in stock_illegal})} need the in-memory "
            "export_nvfp4_cb (their codec output sizes need a pack to size "
            "the streaming header — bounded TODO).")

    def _resolve_target(qname, suffix=".weight"):
        """Locate a target's source: a stacked skeleton tensor, an expert
        group (per-expert on disk), or a resolved skeleton key. Returns
        (kind, handle)."""
        key = _try_resolve_skeleton(qname, skeleton, profile, suffix)
        if key is not None:
            return "tensor", key
        # packed-expert group keyed by the live packed name.
        m = re.match(r"^(.*\.experts)\.(gate_up_proj|down_proj|gate_proj|"
                     r"up_proj)$", qname)
        if m:
            prefix, packed_proj = m.group(1), m.group(2)
            grp = expert_groups.get(prefix)
            if grp is not None:
                return "experts", (prefix, packed_proj, grp)
        return None, None

    def _target_shape(qname):
        kind, h = _resolve_target(qname)
        if kind == "tensor":
            return tuple(skeleton.get_shape(h))
        if kind == "experts":
            prefix, packed_proj, grp = h
            n = _n_experts(grp)
            if packed_proj == "gate_up_proj":
                g = skeleton.get_shape(grp["gate_proj"][0] + ".weight")
                u = skeleton.get_shape(grp["up_proj"][0] + ".weight")
                return (n, int(g[0]) + int(u[0]), int(g[1]))
            proj = packed_proj if packed_proj in grp else "down_proj"
            s = skeleton.get_shape(grp[proj][0] + ".weight")
            return (n, int(s[0]), int(s[1]))
        raise KeyError(f"{qname}: no streaming source (tensor or expert group)")

    # --- Coverage gate (lazy: shapes only). ---
    for qname, (grid, mode, k) in cb_targets.items():
        shape = _target_shape(qname)
        in_f = int(shape[-1])
        if in_f % cb.SUPERBLOCK != 0:
            raise ValueError(
                f"{qname}: in_features={in_f} not a multiple of "
                f"{cb.SUPERBLOCK}")
        if qname not in col_weights:
            raise ValueError(
                f"{qname}: CB target has no col_weights entry (no silent RTN)")
        cwn = col_weights[qname].numel()
        n_exp = int(shape[0]) if len(shape) == 3 else 1
        if cwn not in (in_f, n_exp * in_f):
            raise ValueError(
                f"{qname}: col_weights has {cwn} elements but the weight "
                f"wants {in_f} or {n_exp}x{in_f}")

    # --- Resolve/train codebooks (bounded pooling for learned). ---
    provided = spec.get("codebooks", {}) if source == "learned" else {}
    train = bool(spec.get("train", False))
    iters = int(spec.get("iters", 4))
    seed = int(spec.get("seed", 0))
    train_cap = int(spec.get("train_cap", 1 << 20))
    codebooks: dict[tuple[str, str], object] = {}
    target_cb: dict[str, tuple] = {}
    by_group: dict[tuple[str, str], list[str]] = {}
    for qname in cb_targets:
        fmt = assignment[qname]
        ref = _role_of(qname) if source == "learned" else "lattice"
        by_group.setdefault((ref, fmt), []).append(qname)
    for (ref, fmt), qnames in by_group.items():
        grid, mode, k = cb_targets[qnames[0]]
        if source == "lattice":
            codebooks[(ref, fmt)] = cb._resolve_codebook(
                k, grid, mode, None, torch.device(device))
            kind = "lattice"
        elif train:
            codebooks[(ref, fmt)] = _train_shared_codebook_streaming(
                skeleton, profile, expert_groups, _resolve_target,
                qnames, col_weights, grid=grid, mode=mode, k=k, seed=seed,
                iters=iters, train_cap=train_cap, device=device)
            kind = "learned"
        elif ref in provided:
            codebooks[(ref, fmt)] = provided[ref]
            kind = "learned"
        else:
            raise ValueError(
                f"role {ref!r} ({fmt}): learned but no codebook + train=False")
        for q in qnames:
            target_cb[q] = (ref, fmt, codebooks[(ref, fmt)], kind)

    # --- Build the streaming plan + config in one metadata pass. ---
    writer = _StreamWriter()
    counts: Counter[str] = Counter()
    ignore: list[str] = []
    cb_targets_set = set(cb_targets)
    source_set = set(source_targets)
    emitted_bases: set[str] = set()   # checkpoint tensor keys we consume

    # CB + FP8_SOURCE targets (keyed by canonical/recipe qname).
    for qname in list(cb_targets) + list(source_targets):
        export_base = _export_base_name(qname, profile)
        kind, h = _resolve_target(qname)
        if qname in source_set:
            wkey = _try_resolve_skeleton(qname, skeleton, profile)
            skey = _try_resolve_skeleton(qname, skeleton, profile,
                                         ".weight_scale_inv")
            if wkey is None or skey is None or \
                    skeleton.get_dtype(wkey) != torch.float8_e4m3fn:
                raise ValueError(
                    f"{qname}: FP8_SOURCE but source is not native fp8")
            emitted_bases.add(wkey)
            emitted_bases.add(skey)
            wsh = skeleton.get_shape(wkey)
            ssh = skeleton.get_shape(skey)
            writer.add(export_base + ".weight", torch.float8_e4m3fn, wsh,
                       (lambda k=wkey: skeleton.load(k).contiguous()))
            writer.add(
                export_base + ".weight_scale", torch.float32, ssh,
                (lambda k=skey: skeleton.load(k).to(torch.float32)
                 .contiguous()))
            counts["FP8_SOURCE"] += 1
            continue
        grid, mode, k = cb_targets[qname]
        ref, fmt, codebook, _ = target_cb[qname]
        shape = _target_shape(qname)
        rows = 1
        for d in shape[:-1]:
            rows *= int(d)
        n_sb = int(shape[-1]) // cb.SUPERBLOCK
        coding = scale_coding if grid == "fp4" else cb.SCALE_CODING_V1
        ts = cb.nvfp4_cb_type_size(k, grid, coding)
        if kind == "tensor":
            emitted_bases.add(h)
        packed_shape = ((shape[0], shape[1], n_sb * ts) if len(shape) == 3
                        else (rows, n_sb * ts))
        state: dict = {}

        def _pack(qname=qname, h=(kind, h), grid=grid, mode=mode, k=k,
                  codebook=codebook, coding=coding, shape=shape,
                  packed_shape=packed_shape, state=state):
            packed, scale = _stream_pack_target(
                skeleton, profile, h, qname, grid, mode, k, codebook,
                col_weights[qname], scale_sweep, coding, shape, device)
            state["scale"] = scale
            return packed.reshape(packed_shape)

        writer.add(export_base + ".cb_qweight", torch.uint8, packed_shape,
                   _pack)
        counts[fmt] += 1
        if grid == "fp8":
            scale_shape = tuple(int(d) for d in shape[:-1])

            def _scale(state=state, scale_shape=scale_shape):
                return state["scale"].reshape(scale_shape).to(
                    torch.float32).contiguous()
            writer.add(export_base + ".weight_scale", torch.float32,
                       scale_shape, _scale)

    # Passthrough: every remaining checkpoint tensor verbatim (BF16/norms/etc).
    # Per-expert tensors consumed by a stacked CB target are NOT passthrough.
    # NOTE: expert groups are keyed by the on-disk (checkpoint) prefix; for a
    # non-nested source (Hy3) that equals the assignment's live packed qname.
    # A nested-infix MoE source (DSv4 `model.language_model.…`) needs the group
    # keyed by the canonical prefix — a bounded follow-up (see module scope).
    consumed_expert_bases = set()
    for prefix, projs in expert_groups.items():
        packed_names = {f"{prefix}.gate_up_proj", f"{prefix}.down_proj",
                        f"{prefix}.gate_proj", f"{prefix}.up_proj"}
        if packed_names & cb_targets_set:
            for proj, ids in projs.items():
                for e, base in ids.items():
                    consumed_expert_bases.add(base + ".weight")
    for name in skeleton.keys():
        if name in emitted_bases or name in consumed_expert_bases:
            continue
        if name.endswith(".weight_scale_inv"):
            continue   # consumed with its fp8 weight, or an unused sidecar
        ckpt_qname = (name[:-len(".weight")] if name.endswith(".weight")
                      else None)
        canon = _canonical_qname(ckpt_qname, profile) if ckpt_qname else None
        if canon in cb_targets_set or canon in source_set:
            continue
        shape = skeleton.get_shape(name)
        dtype = skeleton.get_dtype(name)
        writer.add(name, dtype, shape, (lambda k=name: skeleton.load(k)
                                        .contiguous()))
        if ckpt_qname is not None and len(shape) >= 2:
            ignore.append(ckpt_qname)
        counts["copied"] += 1

    # --- Codebook sidecar (small; in-memory) + config + write. ---
    cb_tensor_blobs: dict[str, torch.Tensor] = {}
    for (ref, fmt), codebook in codebooks.items():
        for tname, blob in _codebook_tensors(ref, fmt, codebook).items():
            cb_tensor_blobs[tname] = blob
    codebook_file = "cb_codebooks.pqcb" if cb_tensor_blobs else None
    quant_config = _build_config(
        assignment, cb_targets, source_targets, by_group, codebooks,
        col_weights, cb_tensor_blobs, ignore, codebook_file, scale_coding,
        source, profile)

    print(f"[export-cb-stream] streaming {len(writer.names())} tensors ...",
          flush=True)
    from safetensors.torch import save_file
    writer.write(out_dir / "model.safetensors")
    if cb_tensor_blobs:
        save_file(cb_tensor_blobs, str(out_dir / codebook_file),
                  metadata={"format": "pt"})
    (out_dir / "quant_config.json").write_text(
        json.dumps(quant_config, indent=2, sort_keys=True))
    src_config = model_dir / "config.json"
    config = json.loads(src_config.read_text()) if src_config.exists() else {}
    config["quantization_config"] = {
        "quant_method": "prismaquant", "format": "nvfp4_cb",
        "config_file": "quant_config.json"}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    for aux in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model",
                "special_tokens_map.json", "generation_config.json",
                "vocab.json", "merges.txt", "chat_template.jinja",
                "chat_template.json", "preprocessor_config.json",
                "video_preprocessor_config.json", "processor_config.json"):
        p = model_dir / aux
        if p.exists():
            (out_dir / aux).write_bytes(p.read_bytes())
    return dict(counts)


def _stream_pack_target(skeleton, profile, resolved, qname, grid, mode, k,
                        codebook, cw, scale_sweep, coding, shape, device):
    """Pack ONE target, streaming experts. Returns (packed uint8 (rows,bytes)
    or (E,out,bytes), scale-plane fp32 or None). Per-expert scales make
    per-expert packing byte-identical to whole-stack packing."""
    kind, h = resolved
    cbook = _to_device(codebook, device)
    if kind == "tensor":
        w = skeleton.dequant_weight(h).to(device)
        packed, fields = cb.nvfp4_cb_pack(
            w, k, grid=grid, mode=mode, col_weights=cw.to(device),
            codebook=cbook, scale_sweep=scale_sweep, scale_coding=coding)
        if w.dim() == 3:
            packed = packed.reshape(w.shape[0], w.shape[1], -1)
        scale = (fields["scales"].reshape(*w.shape[:-1]).cpu()
                 if grid == "fp8" else None)
        return packed.to(torch.uint8).cpu().contiguous(), scale
    # Experts: build ONE layer's stack (fp4 derives a single per-tensor global
    # over the whole stack, so per-expert packing would diverge — the stack is
    # the byte-identity working set) and pack it whole, exactly as the
    # in-memory exporter packs a pre-stacked 3-D tensor. Peak = one MoE layer's
    # experts, not the model.
    prefix, packed_proj, grp = h
    n = _n_experts(grp)
    w = torch.stack([_expert_weight(skeleton, prefix, packed_proj, grp, e)
                     for e in range(n)]).to(device)
    packed, fields = cb.nvfp4_cb_pack(
        w, k, grid=grid, mode=mode, col_weights=cw.to(device),
        codebook=cbook, scale_sweep=scale_sweep, scale_coding=coding)
    packed = packed.reshape(w.shape[0], w.shape[1], -1)
    scale = (fields["scales"].reshape(*w.shape[:-1]).cpu()
             if grid == "fp8" else None)
    return packed.to(torch.uint8).cpu().contiguous(), scale


def _train_shared_codebook_streaming(skeleton, profile, expert_groups,
                                     resolve_target, qnames, col_weights, *,
                                     grid, mode, k, seed, iters, train_cap,
                                     device):
    """Bounded-pool learned codebook: sample scaled vectors from each target's
    source (streamed) up to ``train_cap`` total, then train — never all
    role weights resident. For a small role (< train_cap) the pooled set
    equals the in-memory exporter's, so the codebook is identical."""
    from prismaquant.export_nvfp4_cb import _train_shared_codebook
    weights, cws = [], []
    for q in qnames:
        kind, h = resolve_target(q)
        if kind == "tensor":
            weights.append(skeleton.dequant_weight(h).to(device))
        else:
            prefix, packed_proj, grp = h
            weights.append(_stacked_source_weight(
                skeleton, prefix, packed_proj, grp).to(device))
        cws.append(col_weights[q].to(device))
    return _train_shared_codebook(
        weights, cws, grid=grid, mode=mode, k=k, seed=seed, iters=iters,
        train_cap=train_cap)


def _build_config(assignment, cb_targets, source_targets, by_group, codebooks,
                  col_weights, cb_tensor_blobs, ignore, codebook_file,
                  scale_coding, source, profile):
    """quant_config.json — mirrors export_nvfp4_cb's config emitter exactly
    (config_groups keyed by scheme signature + provenance)."""
    assignment_sha = hashlib.sha256(json.dumps(
        dict(sorted(assignment.items())), separators=(",", ":")).encode()
    ).hexdigest()
    ih = hashlib.sha256()
    for q in sorted(col_weights):
        ih.update(q.encode())
        ih.update(col_weights[q].to(torch.float32).cpu().numpy().tobytes())
    imatrix_sha = ih.hexdigest()
    codebook_sha = {
        t: hashlib.sha256(b.to(torch.float16).cpu().numpy().tobytes())
        .hexdigest() for t, b in cb_tensor_blobs.items()}
    config_groups: dict[str, dict] = {}
    for gi, ((ref, fmt), qnames) in enumerate(sorted(by_group.items())):
        grid, mode, k = cb_targets[qnames[0]]
        codebook = codebooks[(ref, fmt)]
        n_sub = len(codebook) if isinstance(codebook, (tuple, list)) else 1
        base = f"cb_codebook.{ref}.{fmt}"
        codebook_ref = ([f"{base}.sub{i}" for i in range(n_sub)]
                        if n_sub > 1 else base)
        coding = scale_coding if grid == "fp4" else cb.SCALE_CODING_V1
        targets = sorted(_export_base_name(q, profile) for q in qnames)
        scheme = {
            "grid": grid, "mode": mode, "k": k, "superblock": cb.SUPERBLOCK,
            "group_size": cb.FP4_GROUP if grid == "fp4" else 0,
            "vec_dim": cb.VEC_DIM, "n_sub": n_sub,
            "type_size": cb.nvfp4_cb_type_size(k, grid, coding),
            "act_bits": 4 if grid == "fp4" else 8,
            "codebook_source": "lattice" if ref == "lattice" else "learned",
            "codebook_ref": codebook_ref,
            "codebook_group": None if ref == "lattice" else ref,
        }
        if coding == cb.SCALE_CODING_TWO_TIER:
            table, _, _ = cb._two_tier_tables("cpu")
            scheme["scale_coding"] = {
                "kind": "two_tier", "sub_bits": 4,
                "super_bias": cb.TWO_TIER_SUPER_BIAS,
                "table": [float(t) for t in table.tolist()]}
        config_groups[f"group_{gi}"] = {
            "targets": targets, "format": fmt, "scheme": scheme}
    quant_config = {
        "quant_method": "prismaquant", "format": "nvfp4_cb",
        "config_groups": config_groups, "ignore": sorted(set(ignore)),
        **({"codebook_file": codebook_file} if codebook_file else {}),
        "provenance": {
            "git_commit": _git_commit(), "assignment_sha256": assignment_sha,
            "imatrix_sha256": imatrix_sha, "codebook_sha256": codebook_sha,
            "codebook_source": source, "scale_coding": scale_coding,
            "streaming": True, "cb_targets": len(cb_targets),
            "fp8_source_targets": len(source_targets),
        },
    }
    if scale_coding == cb.SCALE_CODING_TWO_TIER:
        quant_config["layout_version"] = 2
    return quant_config


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Streaming NVFP4-CB exporter")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--layer-config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--col-weights", required=True,
                    help="pickle {qname: per-column importance}")
    ap.add_argument("--codebook-source", default="lattice",
                    choices=["lattice", "learned"])
    ap.add_argument("--codebook-iters", type=int, default=4)
    ap.add_argument("--codebook-seed", type=int, default=0)
    ap.add_argument("--no-scale-sweep", action="store_true")
    ap.add_argument("--scale-coding", default=cb.SCALE_CODING_V1,
                    choices=[cb.SCALE_CODING_V1, cb.SCALE_CODING_TWO_TIER])
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    with open(args.col_weights, "rb") as fh:
        col_weights = {k: torch.as_tensor(v) for k, v in pickle.load(fh).items()}
    spec = {"source": args.codebook_source}
    if args.codebook_source == "learned":
        spec.update(train=True, iters=args.codebook_iters,
                    seed=args.codebook_seed)
    counts = export_nvfp4_cb_streaming(
        args.model_dir, args.layer_config, args.out, col_weights,
        shared_codebook_spec=spec, device=args.device,
        scale_sweep=not args.no_scale_sweep, scale_coding=args.scale_coding)
    size = sum(p.stat().st_size for p in Path(args.out).glob("*")) / 1e9
    print(f"wrote {args.out} ({size:.3f} GB)")
    for fmt, n in sorted(counts.items()):
        print(f"  {fmt}: {n}")


if __name__ == "__main__":
    main()
