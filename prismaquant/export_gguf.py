"""Materialize a PrismaQuant recipe as a GGUF checkpoint (llama.cpp lane).

Two-step container strategy: llama.cpp's own ``convert_hf_to_gguf.py
--outtype bf16`` produces the *skeleton* — a full-precision GGUF whose
metadata, tokenizer embedding, tensor naming, and expert stacking are
guaranteed llama.cpp-correct for every architecture the converter knows.
This exporter then rewrites the skeleton: it copies all key/value metadata
verbatim and requantizes each weight tensor to the format the allocator
assigned, using the packers in :mod:`prismaquant.gguf_formats` (whose math
is bit-identical to the registry emulation the cost measurement used).

The result serves in llama.cpp natively and in vLLM via the GGUF path
(in-tree <= 0.19, vllm-gguf-plugin on current vLLM).

Usage:
    python convert_hf_to_gguf.py <hf_model_dir> --outtype bf16 \
        --outfile skeleton.gguf
    python -m prismaquant.export_gguf \
        --skeleton skeleton.gguf \
        --layer-config WORK_DIR/artifacts/layer_config.json \
        --out model-prismaquant.gguf

Provenance (git commit, assignment hash, per-tensor format map) is baked
into the output KV metadata under the ``prismaquant.*`` namespace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import gguf
import numpy as np
import torch
from gguf import GGMLQuantizationType as QT

from prismaquant.gguf_formats import GGUF_BLOCK_BYTES, gguf_pack
from prismaquant.layer_config import load_assignment

# Skeleton tensor types we are willing to treat as a full-precision source.
_SOURCE_TYPES = {QT.F32, QT.F16, QT.BF16}

# GGUF field names that the writer emits itself; never copied from the
# skeleton reader.
_VIRTUAL_KEYS = {
    "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",
    "general.alignment",
    # Rewritten below: the skeleton's file_type (BF16) would mislabel the
    # mixed-precision output.
    "general.file_type",
}


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _reader_tensor_to_torch(tensor: "gguf.ReaderTensor") -> torch.Tensor:
    """Skeleton tensor -> float32 torch tensor with torch-order shape."""
    shape = tuple(int(d) for d in reversed(tensor.shape.tolist()))
    if tensor.tensor_type == QT.BF16:
        raw = np.ascontiguousarray(tensor.data).view(np.uint16).reshape(shape)
        t = torch.from_numpy(raw.astype(np.uint16)).to(torch.uint16)
        return t.view(torch.bfloat16).to(torch.float32)
    data = gguf.quants.dequantize(
        np.ascontiguousarray(tensor.data), tensor.tensor_type
    )
    return torch.from_numpy(np.ascontiguousarray(data)).reshape(shape).to(
        torch.float32
    )


def _copy_metadata(reader: "gguf.GGUFReader", writer: "gguf.GGUFWriter") -> str:
    """Copy every KV field from skeleton to output. Returns the arch."""
    arch = None
    for field in reader.fields.values():
        if field.name in _VIRTUAL_KEYS:
            continue
        if field.name == "general.architecture":
            arch = str(field.contents())
            continue  # the writer wrote it at construction time
        value = field.contents()
        vtype = field.types[0]
        if vtype == gguf.GGUFValueType.ARRAY:
            writer.add_key_value(field.name, value, vtype,
                                 sub_type=field.types[-1])
        else:
            writer.add_key_value(field.name, value, vtype)
    if arch is None:
        raise ValueError("skeleton GGUF has no general.architecture")
    return arch


def _map_assignment_to_gguf(
    arch_name: str, n_layers: int, assignment: dict[str, str],
) -> tuple[dict[str, str], set[str]]:
    """HF-qname assignment -> {gguf tensor name: format}.

    Uses gguf-py's per-arch tensor name map (the same table
    convert_hf_to_gguf used to write the skeleton), mapping forward from
    the assignment's HF module qnames. Returns the gguf-name map and the
    set of assignment entries that did not map (a naming bug upstream).
    """
    arch = None
    for key, value in gguf.MODEL_ARCH_NAMES.items():
        if value == arch_name:
            arch = key
            break
    if arch is None:
        raise ValueError(f"unknown GGUF architecture: {arch_name}")
    name_map = gguf.get_tensor_name_map(arch, n_layers)
    gguf_formats: dict[str, str] = {}
    unmatched: set[str] = set()
    for hf_qname, fmt in assignment.items():
        gguf_name = name_map.get_name(hf_qname)
        if gguf_name is None:
            unmatched.add(hf_qname)
        else:
            gguf_formats[gguf_name + ".weight"] = fmt
    return gguf_formats, unmatched


def export_gguf(
    skeleton_path: str | Path,
    layer_config_path: str | Path,
    out_path: str | Path,
    default_format: str | None = None,
) -> dict[str, int]:
    assignment = load_assignment(layer_config_path)
    reader = gguf.GGUFReader(str(skeleton_path))

    arch_field = reader.fields["general.architecture"]
    arch_name = str(arch_field.contents())
    n_layers = int(reader.fields[f"{arch_name}.block_count"].contents())

    writer = gguf.GGUFWriter(str(out_path), arch_name)
    _copy_metadata(reader, writer)

    gguf_fmt_map, unmatched_assignment = _map_assignment_to_gguf(
        arch_name, n_layers, assignment
    )
    if unmatched_assignment:
        # Fail fast: an assignment entry that maps to no gguf tensor name is
        # a naming bug that would otherwise silently ship the wrong bytes.
        raise ValueError(
            f"{len(unmatched_assignment)} assignment entries have no GGUF "
            f"name mapping, e.g. {sorted(unmatched_assignment)[:8]}"
        )
    seen_gguf_names: set[str] = set()

    counts: Counter[str] = Counter()
    tensor_formats: dict[str, str] = {}

    for tensor in reader.tensors:
        fmt = gguf_fmt_map.get(tensor.name)
        if fmt is not None:
            seen_gguf_names.add(tensor.name)
        wants_quant = fmt is not None and fmt in GGUF_BLOCK_BYTES
        if fmt is None and default_format is not None:
            # Opt-in fallback for 2-D weights the allocator did not cover.
            if tensor.name.endswith(".weight") and len(tensor.shape) >= 2:
                fmt = default_format
                wants_quant = fmt in GGUF_BLOCK_BYTES

        if (
            wants_quant
            and tensor.tensor_type in _SOURCE_TYPES
            and len(tensor.shape) >= 2
            and int(tensor.shape[0]) % GGUF_BLOCK_BYTES[fmt][0] == 0
            # GGUF shape order is reversed: shape[0] is the input dim.
        ):
            w = _reader_tensor_to_torch(tensor)
            packed = gguf_pack(w, fmt)
            # No raw_shape: for quantized dtypes gguf-py derives the logical
            # shape from the packed byte shape (quant_shape_from_byte_shape).
            writer.add_tensor(tensor.name, packed, raw_dtype=getattr(QT, fmt))
            counts[fmt] += 1
            tensor_formats[tensor.name] = fmt
        else:
            if wants_quant:
                counts[f"passthrough({fmt}->src)"] += 1
            # Verbatim copy: reader data is already in the writer's expected
            # layout for every tensor type (typed+logical for f32/f16,
            # byte-shaped for bf16/quantized).
            data = np.ascontiguousarray(tensor.data)
            writer.add_tensor(tensor.name, data, raw_dtype=tensor.tensor_type)
            counts[tensor.tensor_type.name] += 1
            tensor_formats[tensor.name] = tensor.tensor_type.name

    missing = set(gguf_fmt_map) - seen_gguf_names
    if missing:
        raise ValueError(
            f"{len(missing)} assignment entries matched no skeleton "
            f"tensor, e.g. {sorted(missing)[:8]}"
        )

    digest = hashlib.sha256(
        json.dumps(dict(sorted(assignment.items())),
                   separators=(",", ":")).encode()
    ).hexdigest()
    writer.add_file_type(gguf.LlamaFileType.GUESSED)
    writer.add_key_value("prismaquant.git_commit", _git_commit(),
                         gguf.GGUFValueType.STRING)
    writer.add_key_value("prismaquant.assignment_sha256", digest,
                         gguf.GGUFValueType.STRING)
    writer.add_key_value("prismaquant.tensor_formats",
                         json.dumps(tensor_formats, sort_keys=True),
                         gguf.GGUFValueType.STRING)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    return dict(counts)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skeleton", required=True,
                    help="bf16/f16 GGUF produced by convert_hf_to_gguf.py")
    ap.add_argument("--layer-config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--default-format", default=None,
        help="optional GGUF format for 2-D weights absent from the "
        "assignment (e.g. Q6_K); default: keep skeleton precision",
    )
    args = ap.parse_args(argv)
    counts = export_gguf(
        args.skeleton, args.layer_config, args.out,
        default_format=args.default_format,
    )
    size = Path(args.out).stat().st_size / 1e9
    print(f"wrote {args.out} ({size:.2f} GB)")
    for fmt, n in sorted(counts.items()):
        print(f"  {fmt}: {n}")


if __name__ == "__main__":
    main()
