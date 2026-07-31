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

Scope: bf16 source + fp8-source READ + CB families + BF16 passthrough +
FP8_SOURCE passthrough + stock-CT **DENSE** rungs (vanilla NVFP4 / FP8_DYNAMIC
quantised in-container). Stock rungs are packed RTN via the authoritative
``export_native_compressed`` codecs (byte-identical to the in-memory
export_nvfp4_cb and to those packers called directly; no GPTQ/act-order in this
lane — the CB cost stage measures stock rungs RTN-grade). Their on-disk sizes
are ANALYTIC so the streaming header needs no pack:

  * NVFP4  -> ``weight_packed`` uint8 [N, K/2] + ``weight_scale`` fp8_e4m3
    [N, K/16] + ``weight_global_scale`` fp32 [1] + ``input_global_scale`` fp32 [1]
  * FP8_DYNAMIC -> ``weight`` fp8_e4m3 [N, K] + ``weight_scale`` fp32 [N, 1]

Stock rungs on MoE **expert stacks** are NOT streamed: the CB container's stock
config emits a packed-name regex that vLLM's MoE dispatch cannot match to its
per-expert probes, and the CT codec is 2-D — an expert stack assigned a stock
format hard-fails with a pointer to constrain the allocator (put experts on a
CB rung / FP8_SOURCE / BF16; the dense tier is where vanilla formats win). The
config_groups for stock rungs use the EXACT compressed-tensors vocabulary (no
``"scheme"`` key) under the vLLM-internal target name so the plugin delegates
them to CompressedTensorsConfig.
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
# Inverse (safetensors dtype string -> torch dtype), used by the DELTA-EXPORT
# reuse path to read a prior artifact's per-tensor dtype from its header.
_ST_DTYPE_INV = {v: k for k, v in _ST_DTYPE.items()}
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

    # Bound concurrently-open shard mmaps. Unbounded handles grew total_vm
    # to ~1TB on the 233-shard Hy3 source and the box global-OOMed with the
    # exporter's CPU RSS ~0 and torch CUDA alloc 0 — the consumer was
    # driver-side pinning tied to live source mappings (2026-07-19).
    _MAX_OPEN_SHARDS = 4

    def _handle(self, name: str):
        shard = self.weight_map[name]
        if shard not in self._open:
            while len(self._open) >= self._MAX_OPEN_SHARDS:
                old = next(iter(self._open))
                del self._open[old]
            self._open[shard] = safe_open(
                self.dir / shard, framework="pt", device="cpu")
        else:
            self._open[shard] = self._open.pop(shard)   # LRU refresh
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


def _stock_output_specs(fmt: str, shape) -> list[tuple[str, torch.dtype, tuple]]:
    """Analytic on-disk ``(suffix, dtype, out_shape)`` list for a DENSE stock
    target whose source weight is ``shape`` (out=N, in=K). Mirrors
    ``export_native_compressed._quantize_2d`` output EXACTLY (verified against
    the packers) so the streaming header is sized without a pack:

      * NVFP4 (W4A4): ``weight_packed`` uint8 [N, K/2], ``weight_scale``
        fp8_e4m3 [N, K/16], ``weight_global_scale`` fp32 [1],
        ``input_global_scale`` fp32 [1].
      * FP8_E4M3 (W8A8 per-channel): ``weight`` fp8_e4m3 [N, K],
        ``weight_scale`` fp32 [N, 1].
    """
    n, k = int(shape[-2]), int(shape[-1])
    if fmt == "NVFP4":
        return [
            ("weight_packed", torch.uint8, (n, k // 2)),
            ("weight_scale", torch.float8_e4m3fn, (n, k // 16)),
            ("weight_global_scale", torch.float32, (1,)),
            ("input_global_scale", torch.float32, (1,)),
        ]
    if fmt == "FP8_E4M3":
        return [
            ("weight", torch.float8_e4m3fn, (n, k)),
            ("weight_scale", torch.float32, (n, 1)),
        ]
    raise ValueError(f"no stock streaming spec for {fmt!r}")


class _StreamWriter:
    """Two-pass safetensors writer. ``add`` records (name, dtype, shape) and a
    zero-arg ``producer`` that yields the tensor at write time; ``write`` lays
    out contiguous offsets, writes the header, then streams every producer's
    bytes in order — one output tensor resident at a time."""

    def __init__(self):
        self._entries: list[tuple[str, torch.dtype, tuple, object, object]] = []

    def add(self, name, dtype, shape, producer, copy_src=None):
        """Record an output tensor. ``producer`` yields it at write time; when
        ``copy_src=(path, file_offset, nbytes)`` is given (DELTA-EXPORT reuse)
        those raw bytes are streamed straight from a prior artifact's shard file
        instead — ``producer`` is then unused (may be None)."""
        self._entries.append((name, dtype, tuple(int(d) for d in shape),
                              producer, copy_src))

    def names(self) -> list[str]:
        return [e[0] for e in self._entries]

    def write(self, path: Path) -> None:
        header: dict[str, dict] = {}
        off = 0
        for name, dtype, shape, _, _ in self._entries:
            nb = _nbytes(dtype, shape)
            header[name] = {"dtype": _ST_DTYPE[dtype], "shape": list(shape),
                            "data_offsets": [off, off + nb]}
            off += nb
        header["__metadata__"] = {"format": "pt", "quant_method": "gridbook"}
        hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
        data0 = 8 + len(hjson)

        # RESUME: offsets are analytic and producers deterministic, so a
        # partial file identifies exactly which entries are already complete.
        # Only resume a file whose header matches this plan bit-for-bit
        # (same assignment/codebooks); otherwise start over.
        skip = 0
        if path.exists():
            size = path.stat().st_size
            ok = False
            if size >= data0:
                with open(path, "rb") as f:
                    (hlen,) = struct.unpack("<Q", f.read(8))
                    ok = hlen == len(hjson) and f.read(hlen) == hjson
            if ok:
                while skip < len(self._entries):
                    name = self._entries[skip][0]
                    if data0 + header[name]["data_offsets"][1] > size:
                        break
                    skip += 1
                # Sibling producers share state (fp8 weight_scale reads the
                # scale its cb_qweight pack produced) — back the boundary up
                # to the start of the export-base group. Re-produced entries
                # rewrite identical bytes (producers are deterministic).
                base = lambda i: self._entries[i][0].rsplit(".", 1)[0]
                while 0 < skip < len(self._entries) and \
                        base(skip) == base(skip - 1):
                    skip -= 1
                print(f"[export-cb-stream] resuming {path.name}: "
                      f"{skip}/{len(self._entries)} entries already written",
                      flush=True)
            else:
                path.unlink()

        cuda = torch.cuda.is_available()
        with open(path, "r+b" if skip else "wb") as f:
            if skip:
                first = self._entries[skip][0] if skip < len(self._entries) \
                    else None
                f.truncate(data0 + (header[first]["data_offsets"][0]
                                    if first else off))
                f.seek(0, 2)
            else:
                f.write(struct.pack("<Q", len(hjson)))
                f.write(hjson)
            for i, (name, dtype, shape, producer, copy_src) in enumerate(
                    self._entries[skip:], start=skip):
                if copy_src is not None:
                    # DELTA-EXPORT: stream the tensor's raw bytes straight from
                    # a prior artifact's shard at the recorded offset (no torch
                    # round-trip; dtype/shape/layout are pinned identical by the
                    # eligibility gate). Chunked so peak residency stays tiny.
                    src_path, foff, nb = copy_src
                    if nb != _nbytes(dtype, shape):
                        raise AssertionError(
                            f"{name}: copy_src {nb}B != declared "
                            f"{_nbytes(dtype, shape)}B")
                    with open(src_path, "rb") as sf:
                        sf.seek(foff)
                        remaining = nb
                        while remaining:
                            chunk = sf.read(min(remaining, 1 << 24))
                            if not chunk:
                                raise AssertionError(
                                    f"{name}: prior artifact truncated at "
                                    f"offset {foff} (needed {nb}B)")
                            f.write(chunk)
                            remaining -= len(chunk)
                    if i % 50 == 0 or nb > (1 << 30):
                        print(f"[export-cb-stream] {i + 1}/"
                              f"{len(self._entries)} {name} copied "
                              f"{nb / 2**30:.2f}G from prior", flush=True)
                    continue
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
                if cuda:
                    # Unified-memory hygiene: differently-shaped 10GB-class
                    # pack transients must not accumulate as cached segments
                    # (Hy3 2026-07-19 global OOM 2GB into the write).
                    torch.cuda.empty_cache()
                    if i % 20 == 0 or _nbytes(dtype, shape) > (1 << 30):
                        print(f"[export-cb-stream] {i + 1}/"
                              f"{len(self._entries)} {name} "
                              f"cuda alloc "
                              f"{torch.cuda.memory_allocated() / 2**30:.1f}G "
                              f"reserved "
                              f"{torch.cuda.memory_reserved() / 2**30:.1f}G",
                              flush=True)


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
# DELTA-EXPORT reuse: read a PRIOR artifact + decide byte-copy eligibility
# ---------------------------------------------------------------------------

class _PriorArtifact:
    """Read-only view of a PRIOR CB export for DELTA-EXPORT reuse.

    Exposes, for any tensor by name: presence, dtype, shape, and the exact
    ``(shard_path, file_offset, nbytes)`` byte slice (sharded via index.json or
    single-file). Also parses ``quant_config.json`` into the per-export-base CB
    ``(format, scheme)`` and loads the codebook sidecar — everything the
    eligibility gate needs to prove a re-encode would reproduce these bytes."""

    def __init__(self, prior_dir: str | Path):
        self.dir = Path(prior_dir)
        index = self.dir / "model.safetensors.index.json"
        self._single: Path | None = None
        if index.exists():
            self.weight_map = json.loads(index.read_text())["weight_map"]
        else:
            single = self.dir / "model.safetensors"
            if not single.exists():
                raise FileNotFoundError(
                    "reuse-prior: no model.safetensors[.index.json] under "
                    f"{self.dir}")
            self.weight_map = None
            self._single = single
        self._shard_hdr: dict[str, tuple[dict, int]] = {}
        qc_path = self.dir / "quant_config.json"
        if not qc_path.exists():
            raise FileNotFoundError(
                f"reuse-prior: no quant_config.json under {self.dir}")
        qc = json.loads(qc_path.read_text())
        # CB targets carry a "scheme"; stock/FP8_SOURCE groups do not.
        self.cb_by_base: dict[str, tuple[str, dict]] = {}
        self.stock_fmt_by_target: dict[str, str] = {}
        for g in qc.get("config_groups", {}).values():
            fmt = g.get("format")
            if "scheme" in g:
                for t in g.get("targets", []):
                    self.cb_by_base[t] = (fmt, g["scheme"])
            else:
                for t in g.get("targets", []):
                    self.stock_fmt_by_target[t] = fmt
        self.provenance = qc.get("provenance", {}) or {}
        self.scale_coding = qc.get("provenance", {}).get(
            "scale_coding") or "v1"
        self.codebooks: dict[str, torch.Tensor] = {}
        cbf = qc.get("codebook_file")
        if cbf and (self.dir / cbf).exists():
            from safetensors.torch import load_file as _lf
            self.codebooks = _lf(str(self.dir / cbf))

    def _shard_of(self, name: str) -> Path:
        if self._single is not None:
            return self._single
        return self.dir / self.weight_map[name]

    def _hdr(self, shard: Path) -> tuple[dict, int]:
        key = str(shard)
        if key not in self._shard_hdr:
            with open(shard, "rb") as f:
                (hlen,) = struct.unpack("<Q", f.read(8))
                hdr = json.loads(f.read(hlen))
            self._shard_hdr[key] = (hdr, 8 + hlen)
        return self._shard_hdr[key]

    def has(self, name: str) -> bool:
        if self.weight_map is not None:
            return name in self.weight_map
        hdr, _ = self._hdr(self._single)
        return name in hdr

    def _meta(self, name: str):
        shard = self._shard_of(name)
        hdr, data0 = self._hdr(shard)
        return hdr[name], data0, shard

    def dtype(self, name: str):
        meta, _, _ = self._meta(name)
        return _ST_DTYPE_INV.get(meta["dtype"])

    def shape(self, name: str) -> tuple[int, ...]:
        meta, _, _ = self._meta(name)
        return tuple(int(d) for d in meta["shape"])

    def raw_slice(self, name: str) -> tuple[Path, int, int]:
        """(shard_path, absolute file offset, nbytes) for a raw byte copy."""
        meta, data0, shard = self._meta(name)
        lo, hi = meta["data_offsets"]
        return shard, data0 + int(lo), int(hi) - int(lo)

    def read_bytes(self, name: str) -> bytes:
        shard, foff, nb = self.raw_slice(name)
        with open(shard, "rb") as f:
            f.seek(foff)
            return f.read(nb)

    def codebook_tensor(self, name: str) -> torch.Tensor | None:
        return self.codebooks.get(name)

    def matches_dtype_shape(self, name, dtype, shape) -> bool:
        return (self.has(name) and self.dtype(name) == dtype
                and self.shape(name) == tuple(int(d) for d in shape))


def _prior_scale_coding_norm(scheme: dict) -> str:
    sc = scheme.get("scale_coding")
    if isinstance(sc, dict) and sc.get("kind") == "two_tier":
        return "two_tier"
    return "v1"


def _cb_reuse_reason(prior: _PriorArtifact, export_base: str, fmt: str,
                     cur_subset: dict, expected_outputs, group_cb_ok: bool):
    """Return None if this CB target is byte-copy eligible from ``prior``, else
    a short reason string. Eligible iff the prior assigns the SAME format +
    scheme signature, its codebook is byte-identical, and every planned output
    tensor already exists in the prior at EXACTLY the planned dtype+shape."""
    entry = prior.cb_by_base.get(export_base)
    if entry is None:
        return "not_in_prior"
    pfmt, pscheme = entry
    if pfmt != fmt:
        return "format_changed"
    pscheme_norm = {
        "grid": pscheme.get("grid"), "mode": pscheme.get("mode"),
        "k": pscheme.get("k"), "n_sub": pscheme.get("n_sub"),
        "type_size": pscheme.get("type_size"),
        "codebook_ref": pscheme.get("codebook_ref"),
        "scale_coding": _prior_scale_coding_norm(pscheme),
    }
    if pscheme_norm != cur_subset:
        return "scheme_changed"
    if not group_cb_ok:
        return "codebook_mismatch"
    for name, dtype, shape in expected_outputs:
        if not prior.has(name):
            return "tensor_missing"
        if not prior.matches_dtype_shape(name, dtype, shape):
            return "dtype_shape_mismatch"
    return None


def _current_imatrix_sha(col_weights: dict[str, torch.Tensor]) -> str:
    """The imatrix hash exactly as ``_build_config`` computes it — used to
    diagnose whether the reuse prior shares this run's calibration."""
    ih = hashlib.sha256()
    for q in sorted(col_weights):
        ih.update(q.encode())
        ih.update(col_weights[q].to(torch.float32).cpu().numpy().tobytes())
    return ih.hexdigest()


def _reuse_verify_and_report(prior, reuse, reuse_verify, reuse_prior,
                             col_weights, scale_coding, counts):
    """MANDATORY reuse safety gate (runs BEFORE any bytes are written): fresh
    re-encode ``reuse_verify`` random copy-eligible CB targets and byte-compare
    against what would be copied from the prior; ANY mismatch means the
    determinism contract broke and aborts the export. Also logs the copied/
    encoded/ineligible summary and folds ``reuse_*`` counters into ``counts``."""
    import random

    cur_sha = _current_imatrix_sha(col_weights)
    prior_sha = prior.provenance.get("imatrix_sha256")
    imatrix_match = (prior_sha is not None and prior_sha == cur_sha)
    if prior_sha is not None and not imatrix_match:
        print("[export-cb-stream] WARNING reuse-prior imatrix_sha256 differs "
              f"(prior {prior_sha[:12]} vs current {cur_sha[:12]}) — encoding "
              "inputs may have changed; copied bytes rest on the verification "
              "sample below. Double-check --reuse-prior points at the SAME "
              "source+calibration.", flush=True)

    pool = reuse["verify_pool"]
    n = min(int(reuse_verify), len(pool))
    if n > 0:
        # Deterministic sample (reproducibility gate): seed from the stable set
        # of eligible bases so a resumed run verifies the same targets.
        key = "|".join(sorted(c["base"] for c in pool))
        rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16],
                                16))
        for cand in rng.sample(pool, n):
            fresh = cand["fresh"]()
            for name, dtype, shape in cand["specs"]:
                fb = _raw_bytes(fresh[name])
                pb = prior.read_bytes(name)
                if fb != pb:
                    raise RuntimeError(
                        "[export-cb-stream] REUSE VERIFICATION FAILED: fresh "
                        f"re-encode of {name} does NOT byte-match the prior "
                        f"artifact ({len(fb)}B vs {len(pb)}B copied). The "
                        "determinism/RESUME contract is broken for this "
                        "(source, imatrix, codebook, scheme) — refusing to "
                        "ship reused bytes. Re-run WITHOUT --reuse-prior, or "
                        "point it at the artifact this allocation derives from.")
            reuse["verified"] += 1
        print(f"[export-cb-stream] reuse verify OK: {reuse['verified']} "
              f"sampled copy target(s) byte-match the prior", flush=True)

    print(f"[export-cb-stream] reuse-prior {reuse_prior}: "
          f"copied {reuse['copied']} / encoded {reuse['encoded']} targets; "
          f"imatrix {'MATCH' if imatrix_match else 'differ/absent'}; "
          f"scale_coding prior={prior.scale_coding} current={scale_coding}",
          flush=True)
    if reuse["reasons"]:
        print("[export-cb-stream] reuse re-encode reasons: "
              f"{dict(sorted(reuse['reasons'].items()))}", flush=True)

    counts["reuse_copied"] = reuse["copied"]
    counts["reuse_encoded"] = reuse["encoded"]
    counts["reuse_verified"] = reuse["verified"]
    for reason, c in reuse["reasons"].items():
        counts[f"reuse_ineligible_{reason}"] = c


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
    subset_prefixes: list[str] | None = None,
    reuse_prior: str | Path | None = None,
    reuse_verify: int = 3,
) -> dict[str, int]:
    """Streaming counterpart of :func:`export_nvfp4_cb.export_nvfp4_cb`. Same
    signature + container; peak residency ~= one source tensor + codebooks.
    See the module docstring for the scope of this milestone.

    ``subset_prefixes`` (opt-in) scopes the export to a subset of the model:
    the passthrough copies ONLY checkpoint tensors whose name starts with one of
    the prefixes, and every allocation target must resolve to an export base
    within them (else the allocation and the declared subset disagree — fail
    fast). Default ``None`` = whole-model passthrough, byte-identical to before.
    Used to export just the MTP sidecar (``model.layers.80.``) without dragging
    the ~550 GB body through as bf16 passthrough.

    ``reuse_prior`` (opt-in DELTA-EXPORT; default ``None`` == byte-identical to
    today) points at a PRIOR artifact dir. On a re-allocation of the same source
    most CB/stock targets keep their exact ``(format, scheme, codebook)``, so
    their re-encode would reproduce byte-identical tensors (the producers are
    deterministic — the RESUME contract). Such targets are byte-copied straight
    from the prior's shard file(s) instead of re-encoded; ineligible/changed
    targets encode fresh, silently. ``reuse_verify`` (default 3, env
    ``PRISMAQUANT_EXPORT_REUSE_VERIFY``) freshly re-encodes N random copy-eligible
    CB targets and byte-compares them against the copied bytes — any mismatch
    means the determinism contract broke and ABORTS the export (nothing is
    written). FP8_SOURCE + BF16 passthrough deliberately stay on the source path
    (verbatim/raw already, and a cross-artifact copy has no IO win but adds a
    wrong-prior footgun the CB-only verification would not catch)."""
    model_dir = Path(model_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if scale_coding not in (cb.SCALE_CODING_V1, cb.SCALE_CODING_TWO_TIER):
        raise ValueError(f"unknown scale_coding {scale_coding!r}")
    subset_prefixes = list(subset_prefixes) if subset_prefixes else None
    prior = _PriorArtifact(reuse_prior) if reuse_prior else None
    reuse = {"copied": 0, "encoded": 0, "verified": 0,
             "reasons": Counter(), "verify_pool": []}
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

    # --- Stock-CT codecs (mixed container: the plugin delegates non-"scheme"
    # groups to vLLM's CompressedTensors path). REUSE the authoritative
    # export_native_compressed packers — never reimplement packing; RTN only
    # (no GPTQ/act-order), matching how the CB cost stage measures stock rungs. ---
    from prismaquant.format_registry import canonical_format_name
    from prismaquant.export_native_compressed import (
        _quantize_2d as _ct_quantize_2d,
        compute_nvfp4_global_real as _ct_nvfp4_global_real,
    )
    _STOCK_CT_FORMATS = ("NVFP4", "FP8_E4M3")   # FP8_DYNAMIC canonicalizes here

    # --- Classify every target (CB / FP8_SOURCE / stock-CT dense / BF16). ---
    cb_targets: dict[str, tuple[str, str, int]] = {}
    source_targets: list[str] = []
    stock_targets: dict[str, str] = {}          # qname -> "NVFP4" | "FP8_E4M3"
    illegal = []
    for qname, fmt in assignment.items():
        if fmt == "BF16":
            continue
        parsed = _parse_cb_format(fmt)
        if parsed is not None:
            cb_targets[qname] = parsed
            continue
        canon = canonical_format_name(fmt)
        if canon == "FP8_SOURCE":
            source_targets.append(qname)
            continue
        if canon in _STOCK_CT_FORMATS:
            stock_targets[qname] = canon
            continue
        illegal.append((qname, fmt))
    if illegal:
        raise ValueError(
            "streaming CB export carries CB families + stock NVFP4/FP8_DYNAMIC "
            "(CT-delegated) + FP8_SOURCE + BF16 only; unsupported rung(s) "
            f"{sorted({f for _, f in illegal})} — assign a legal format or use "
            "the in-memory export_nvfp4_cb.")

    # Subset gate: every quantised target's export base must live under a
    # declared prefix, else the allocation reaches outside the subset the caller
    # asked to export (a mistake worth failing on, not silently over/under
    # covering). Passthrough is filtered by the same prefixes below.
    if subset_prefixes is not None:
        outside = sorted(
            q for q in list(cb_targets) + list(source_targets)
            + list(stock_targets)
            if not any(_export_base_name(q, profile).startswith(p)
                       for p in subset_prefixes))
        if outside:
            raise ValueError(
                f"--subset-prefix {subset_prefixes}: {len(outside)} allocation "
                f"target(s) resolve outside the subset, e.g. {outside[:5]} — "
                "the layer_config and the declared subset disagree")

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
            if grp is None and profile is not None:
                # Nested-prefix checkpoints (Qwen3.5-VLM: recipe
                # `model.layers.*` vs on-disk `model.language_model.layers.*`)
                # — the groups are keyed by CHECKPOINT prefixes; map the
                # recipe prefix through the profile (same resolution the
                # dense `_try_resolve_skeleton` path uses).
                try:
                    grp = expert_groups.get(
                        profile.source_tensor_name(prefix))
                except Exception:
                    grp = None
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

    # --- Stock-CT coverage + expert-stack gate. Stock rungs stream for DENSE
    # (2-D) Linears only; a MoE expert stack assigned a stock format has no
    # safe streaming pack here (the CB container's stock config emits a
    # packed-name regex vLLM's MoE dispatch cannot match to its per-expert
    # probes, and the CT codec is 2-D), so fail fast pointing at the fix. ---
    stock_expert: list[str] = []
    for qname in stock_targets:
        kind, _h = _resolve_target(qname)
        if kind is None:
            raise KeyError(
                f"{qname}: assigned {stock_targets[qname]} but no streaming "
                "source (tried the .weight key + the profile-mapped checkpoint "
                "name)")
        shape = _target_shape(qname)
        if kind == "experts" or len(shape) == 3:
            stock_expert.append(qname)
            continue
        if stock_targets[qname] == "NVFP4" and int(shape[-1]) % 16 != 0:
            raise ValueError(
                f"{qname}: stock NVFP4 needs in_features % 16 == 0 (group 16), "
                f"got in_features={int(shape[-1])}")
    if stock_expert:
        raise ValueError(
            "streaming CB export carries stock NVFP4/FP8_DYNAMIC on DENSE "
            "Linears only; these MoE expert-stack target(s) were assigned a "
            f"stock format: {sorted(stock_expert)[:5]}"
            f"{' ...' if len(stock_expert) > 5 else ''} "
            f"({len(stock_expert)} total). Assign expert stacks a CB rung "
            "(nvfp4_cb / fp8_cb), FP8_SOURCE, or BF16 — or use the in-memory "
            "export_nvfp4_cb on a model small enough to materialise. The dense "
            "tier is where vanilla NVFP4/FP8_DYNAMIC won the A/B; constrain the "
            "allocator to keep experts on CB/passthrough.")

    # --- Stock NVFP4 fused-sibling coherence (mirrors export_nvfp4_cb /
    # export_native_compressed): q/k/v and gate/up landing on NVFP4 MUST share
    # ONE weight_global_scale or vLLM's fused loader sees inconsistent
    # per-tensor globals. Take the max over each fused group's natural
    # global_real (streamed — one weight resident at a time) and override every
    # sibling's pack. Singleton groups get their own global, exactly like the
    # in-memory exporter (so the streamed bytes are byte-identical). ---
    _nvfp4_shared_global: dict[str, torch.Tensor] = {}
    _nvfp4_groups: dict[str, list[str]] = {}
    for _q, _f in stock_targets.items():
        if _f != "NVFP4":
            continue
        _gk = (profile.fused_sibling_group(_q)
               if profile is not None else None) or _q
        _nvfp4_groups.setdefault(_gk, []).append(_q)
    for _members in _nvfp4_groups.values():
        _grs = []
        for _m in _members:
            _k, _h = _resolve_target(_m)
            _w = skeleton.dequant_weight(_h).to(device)
            _grs.append(_ct_nvfp4_global_real(_w, 16).reshape(()))
            del _w
        _shared = torch.stack(_grs).max()
        for _m in _members:
            _nvfp4_shared_global[_m] = _shared

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

    # DELTA-EXPORT: a CB group's codebook is a byte-copy input, so compare this
    # run's serialized codebook against the prior sidecar ONCE per (ref, fmt).
    # A group whose codebook differs makes every target on it re-encode.
    group_cb_ok: dict[tuple[str, str], bool] = {}
    if prior is not None:
        for (ref, fmt), codebook in codebooks.items():
            cur_t = _codebook_tensors(ref, fmt, codebook)
            ok = True
            for tname, t in cur_t.items():
                pt = prior.codebook_tensor(tname)
                if pt is None or tuple(pt.shape) != tuple(t.shape) \
                        or not torch.equal(pt, t):
                    ok = False
                    break
            group_cb_ok[(ref, fmt)] = ok

    # --- Build the streaming plan + config in one metadata pass. ---
    writer = _StreamWriter()
    counts: Counter[str] = Counter()
    ignore: list[str] = []
    cb_targets_set = set(cb_targets)
    source_set = set(source_targets)
    stock_set = set(stock_targets)
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

        qw_name = export_base + ".cb_qweight"
        scale_shape = tuple(int(d) for d in shape[:-1])
        scale_name = export_base + ".weight_scale"
        # Planned output tensors (name, dtype, shape) — the eligibility gate.
        expected = [(qw_name, torch.uint8, packed_shape)]
        if grid == "fp8":
            expected.append((scale_name, torch.float32, scale_shape))

        # DELTA-EXPORT eligibility: same format + scheme signature + byte-equal
        # codebook + every planned output already present in the prior at the
        # planned dtype+shape => byte-copy instead of re-encode.
        reason = "disabled"
        if prior is not None:
            n_sub = (len(codebook) if isinstance(codebook, (tuple, list))
                     else 1)
            base_ref = f"cb_codebook.{ref}.{fmt}"
            cb_ref = ([f"{base_ref}.sub{i}" for i in range(n_sub)]
                      if n_sub > 1 else base_ref)
            cur_subset = {
                "grid": grid, "mode": mode, "k": k, "n_sub": n_sub,
                "type_size": ts, "codebook_ref": cb_ref,
                "scale_coding": ("two_tier"
                                 if coding == cb.SCALE_CODING_TWO_TIER
                                 else "v1"),
            }
            reason = _cb_reuse_reason(
                prior, export_base, fmt, cur_subset, expected,
                group_cb_ok.get((ref, fmt), False))

        if prior is not None and reason is None:
            for name, dtype, _sh in expected:
                writer.add(name, dtype, prior.shape(name), None,
                           copy_src=prior.raw_slice(name))
            reuse["copied"] += 1

            def _fresh(qname=qname, h=(kind, h), grid=grid, mode=mode, k=k,
                       codebook=codebook, coding=coding, shape=shape,
                       packed_shape=packed_shape, scale_shape=scale_shape,
                       qw_name=qw_name, scale_name=scale_name):
                packed, scale = _stream_pack_target(
                    skeleton, profile, h, qname, grid, mode, k, codebook,
                    col_weights[qname], scale_sweep, coding, shape, device)
                out = {qw_name: packed.reshape(packed_shape)}
                if scale is not None:
                    out[scale_name] = scale.reshape(scale_shape).to(
                        torch.float32).contiguous()
                return out
            reuse["verify_pool"].append(
                {"base": export_base, "specs": expected, "fresh": _fresh})
        else:
            writer.add(qw_name, torch.uint8, packed_shape, _pack)
            if grid == "fp8":
                def _scale(state=state, scale_shape=scale_shape):
                    return state["scale"].reshape(scale_shape).to(
                        torch.float32).contiguous()
                writer.add(scale_name, torch.float32, scale_shape, _scale)
            if prior is not None:
                reuse["encoded"] += 1
                reuse["reasons"][reason] += 1
        counts[fmt] += 1

    # Stock-CT DENSE targets: analytic on-disk tensors packed RTN via the
    # export_native_compressed codec (byte-identical to export_nvfp4_cb). ONE
    # producer packs the weight once and caches every suffix tensor; the writer
    # streams them one at a time. RTN is deterministic, so RESUME re-runs the
    # group from its base boundary and rewrites identical bytes.
    for qname in sorted(stock_targets):
        canon_fmt = stock_targets[qname]
        export_base = _export_base_name(qname, profile)
        kind, h = _resolve_target(qname)              # dense: kind == "tensor"
        shape = _target_shape(qname)
        override = (_nvfp4_shared_global.get(qname)
                    if canon_fmt == "NVFP4" else None)
        emitted_bases.add(h)
        state: dict = {}

        def _render(h=h, canon_fmt=canon_fmt, override=override, state=state):
            if "out" not in state:
                w = skeleton.dequant_weight(h).to(device)
                packed = _ct_quantize_2d(
                    w, canon_fmt, nvfp4_global_real_override=override)
                state["out"] = {s: t.cpu().contiguous()
                                for s, t in packed.items()}
                del w
            return state["out"]

        specs = _stock_output_specs(canon_fmt, shape)
        expected = [(export_base + "." + s, d, o) for s, d, o in specs]
        # DELTA-EXPORT: RTN stock rungs are deterministic from the (unchanged)
        # source weight. FP8_E4M3 is per-channel (no cross-tensor coupling);
        # NVFP4's only cross-tensor input is the fused-group shared global,
        # which the union-find coherence invariant pins identical whenever this
        # target is on NVFP4 in both allocations (q/k/v, gate/up move as a unit,
        # weights unchanged). So the prior having every planned output at the
        # exact dtype+shape is a sound copy gate.
        stock_ok = prior is not None and all(
            prior.matches_dtype_shape(n, d, o) for n, d, o in expected)
        if stock_ok:
            for name, dtype, _sh in expected:
                writer.add(name, dtype, prior.shape(name), None,
                           copy_src=prior.raw_slice(name))
            reuse["copied"] += 1
        else:
            for (name, dtype, out_shape), (suffix, _d, _o) in zip(
                    expected, specs):
                def _prod(suffix=suffix, _render=_render):
                    return _render()[suffix]
                writer.add(name, dtype, out_shape, _prod)
            if prior is not None:
                reuse["encoded"] += 1
                reuse["reasons"][
                    "stock_not_in_prior" if not prior.has(expected[0][0])
                    else "stock_dtype_shape_mismatch"] += 1
        counts[canon_fmt] += 1

    # Passthrough: every remaining checkpoint tensor verbatim (BF16/norms/etc).
    # Per-expert tensors consumed by a stacked CB target are NOT passthrough.
    # Expert groups are keyed by the on-disk (checkpoint) prefix; a nested
    # source (Qwen3.5-VLM `model.language_model.*`, DSv4) needs the CANONICAL
    # prefix for the membership test against the recipe-named CB targets —
    # without it every per-expert bf16 source ships verbatim NEXT TO its
    # packed CB stack (35B first-contact: 31511 copied tensors, 82 GB
    # artifact at a 4.75 bpp target).
    consumed_expert_bases = set()
    for prefix, projs in expert_groups.items():
        canon_prefix = _canonical_qname(prefix, profile) or prefix
        packed_names = set()
        for p in {prefix, canon_prefix}:
            packed_names |= {f"{p}.gate_up_proj", f"{p}.down_proj",
                             f"{p}.gate_proj", f"{p}.up_proj"}
        if packed_names & cb_targets_set:
            for proj, ids in projs.items():
                for e, base in ids.items():
                    consumed_expert_bases.add(base + ".weight")
    for name in skeleton.keys():
        if subset_prefixes is not None and \
                not any(name.startswith(p) for p in subset_prefixes):
            continue   # outside the declared subset (e.g. non-MTP body layers)
        if name in emitted_bases or name in consumed_expert_bases:
            continue
        if name.endswith(".weight_scale_inv"):
            continue   # consumed with its fp8 weight, or an unused sidecar
        ckpt_qname = (name[:-len(".weight")] if name.endswith(".weight")
                      else None)
        canon = _canonical_qname(ckpt_qname, profile) if ckpt_qname else None
        if canon in cb_targets_set or canon in source_set \
                or canon in stock_set:
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
        assignment, cb_targets, source_targets, stock_targets, by_group,
        codebooks, col_weights, cb_tensor_blobs, ignore, codebook_file,
        scale_coding, source, profile)

    # DELTA-EXPORT: verify sampled copies + log the summary BEFORE writing (an
    # abort here leaves no partial artifact). No-op when reuse is disabled.
    if prior is not None:
        _reuse_verify_and_report(
            prior, reuse, reuse_verify, reuse_prior, col_weights,
            scale_coding, counts)

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
        "quant_method": "gridbook", "format": "nvfp4_cb",
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


def _build_config(assignment, cb_targets, source_targets, stock_targets,
                  by_group, codebooks, col_weights, cb_tensor_blobs, ignore,
                  codebook_file, scale_coding, source, profile):
    """quant_config.json — mirrors export_nvfp4_cb's config emitter exactly
    (CB config_groups keyed by scheme signature + stock-CT / FP8_SOURCE groups
    in the exact compressed-tensors vocabulary + provenance)."""
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
    # Stock-CT + FP8_SOURCE config_groups: EXACT compressed-tensors vocabulary
    # (weights/input_activations/format at the group top, NO "scheme" key) so
    # the plugin hands them to CompressedTensorsConfig under delegation (the
    # "scheme" key is the CB-vs-stock dispatch marker, LAYOUT.md §4). Targets
    # carry the vLLM-INTERNAL name (to_vllm_internal_name): the delegated CT
    # path matches vLLM's module tree, NOT the checkpoint tensor names, so a
    # hy_v3 shared_mlp Linear (params under .shared_mlp.*, dispatch prefix
    # collapsed to .mlp.*) is matched only via the vLLM name (28b6862 /
    # export_native_compressed.build_quantization_config). CB groups instead
    # keep the checkpoint name and are runtime-aliased inside the plugin.
    from copy import deepcopy as _deepcopy
    from prismaquant.export_native_compressed import (
        _explicit_regex as _ct_explicit_regex,
        NVFP4_SCHEME as _NVFP4_SCHEME,
        FP8_E4M3_SCHEME as _FP8_E4M3_SCHEME,
        FP8_SOURCE_SCHEME as _FP8_SOURCE_SCHEME,
    )
    _STOCK_CT_SCHEMES = {"NVFP4": _NVFP4_SCHEME, "FP8_E4M3": _FP8_E4M3_SCHEME}

    def _vllm_target(q):
        return profile.to_vllm_internal_name(q) if profile is not None else q

    _stock_by_fmt: dict[str, list[str]] = {}
    for _q, _f in stock_targets.items():
        _stock_by_fmt.setdefault(_f, []).append(_q)
    for _f, _qnames in sorted(_stock_by_fmt.items()):
        _group = _deepcopy(_STOCK_CT_SCHEMES[_f])
        _group["targets"] = sorted(
            _ct_explicit_regex(_vllm_target(q)) for q in _qnames)
        config_groups[f"group_{len(config_groups)}"] = _group
    if source_targets:
        _src_group = _deepcopy(_FP8_SOURCE_SCHEME)
        _src_group["targets"] = sorted(
            _ct_explicit_regex(_vllm_target(q)) for q in source_targets)
        config_groups[f"group_{len(config_groups)}"] = _src_group
    quant_config = {
        "quant_method": "gridbook", "format": "nvfp4_cb",
        "config_groups": config_groups, "ignore": sorted(set(ignore)),
        **({"codebook_file": codebook_file} if codebook_file else {}),
        "provenance": {
            "git_commit": _git_commit(), "assignment_sha256": assignment_sha,
            "imatrix_sha256": imatrix_sha, "codebook_sha256": codebook_sha,
            "codebook_source": source, "scale_coding": scale_coding,
            "streaming": True, "cb_targets": len(cb_targets),
            "stock_ct_targets": len(stock_targets),
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
    ap.add_argument("--subset-prefix", action="append", default=None,
                    metavar="PREFIX",
                    help="opt-in: export ONLY tensors under this checkpoint "
                         "prefix (repeatable), e.g. 'model.layers.80.' for the "
                         "MTP sidecar; every allocation target must fall within "
                         "it. Default: whole-model passthrough.")
    ap.add_argument("--reuse-prior", default=None, metavar="DIR",
                    help="opt-in DELTA-EXPORT: byte-copy CB/stock targets whose "
                         "(format, scheme, codebook) are unchanged from this "
                         "PRIOR artifact dir instead of re-encoding; the delta "
                         "encodes fresh. Env PRISMAQUANT_EXPORT_REUSE_PRIOR is "
                         "the fallback. Default: encode everything.")
    ap.add_argument("--reuse-verify", type=int, default=None, metavar="N",
                    help="reuse safety: fresh re-encode N random copy-eligible "
                         "CB targets and byte-check them (default 3, env "
                         "PRISMAQUANT_EXPORT_REUSE_VERIFY); a mismatch aborts.")
    args = ap.parse_args(argv)
    from prismaquant.gpu_guard import require_cuda_hot_path
    require_cuda_hot_path("export_nvfp4_cb_streaming")
    import os
    reuse_prior = args.reuse_prior or os.environ.get(
        "PRISMAQUANT_EXPORT_REUSE_PRIOR") or None
    reuse_verify = (args.reuse_verify if args.reuse_verify is not None
                    else int(os.environ.get(
                        "PRISMAQUANT_EXPORT_REUSE_VERIFY", "3")))
    if torch.cuda.is_available():
        # Box-safety net on the unified pool: a runaway allocation must raise
        # a clean torch OOM (with the offending tensor in the traceback), not
        # drive the whole box to a kernel global OOM (3x on 2026-07-19).
        torch.cuda.set_per_process_memory_fraction(0.75)
    with open(args.col_weights, "rb") as fh:
        col_weights = {k: torch.as_tensor(v) for k, v in pickle.load(fh).items()}
    spec = {"source": args.codebook_source}
    if args.codebook_source == "learned":
        spec.update(train=True, iters=args.codebook_iters,
                    seed=args.codebook_seed)
    counts = export_nvfp4_cb_streaming(
        args.model_dir, args.layer_config, args.out, col_weights,
        shared_codebook_spec=spec, device=args.device,
        scale_sweep=not args.no_scale_sweep, scale_coding=args.scale_coding,
        subset_prefixes=args.subset_prefix, reuse_prior=reuse_prior,
        reuse_verify=reuse_verify)
    size = sum(p.stat().st_size for p in Path(args.out).glob("*")) / 1e9
    print(f"wrote {args.out} ({size:.3f} GB)")
    for fmt, n in sorted(counts.items()):
        print(f"  {fmt}: {n}")


if __name__ == "__main__":
    main()
