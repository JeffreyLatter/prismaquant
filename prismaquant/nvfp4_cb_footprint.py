"""Authoritative serialized-payload accounting for Gridbook CB formats.

The producer writes two kinds of payload:

* one per-Linear packed tensor (plus an FP32 row-scale tensor for FP8-CB); and
* FP16 codebook subtables shared once per ``(codebook_ref, format)``.

This module describes those tensors, not an abstract nominal rate.  In
particular, production FP4-CB uses layout-v2 ``4k + 9`` byte superblocks,
FP8-CB carries ``4 * output_rows`` scale bytes, and neither CB family has an
NVFP4-style global-scale scalar.  Product and signed codebook sizes are derived
from the exact subtable shapes emitted by :mod:`prismaquant.export_nvfp4_cb`.

``cb_footprint`` is retained as the backwards-compatible Phase-0 entry point.
New producer code should use :func:`cb_tensor_payload_breakdown` and
:func:`cb_assignment_payload_breakdown` directly.  Every returned payload is
versioned so persisted reports remain interpretable after future layouts land.
These are tensor-data bytes, not safetensors container or export-directory
bytes; :func:`finalize_cb_export_artifact_inventory` measures and persists that
separate post-export scope.
"""

from __future__ import annotations

import json
import math
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import format_registry as fr

CB_SERIALIZED_PAYLOAD_SCHEMA = "prismaquant.cb_serialized_payload.v1"
CB_EXPORT_ARTIFACT_INVENTORY_SCHEMA = (
    "prismaquant.cb_export_artifact_inventory.v1"
)
PRODUCTION_FP4_SCALE_CODING = "two_tier"
LEGACY_FP4_SCALE_CODING = "v1"
_SCALE_CODINGS = {PRODUCTION_FP4_SCALE_CODING, LEGACY_FP4_SCALE_CODING}
_LAYOUT_FOR_SCALE_CODING = {
    LEGACY_FP4_SCALE_CODING: 1,
    PRODUCTION_FP4_SCALE_CODING: 2,
}
_FP16_BYTES = 2
_FP32_BYTES = 4
_SUPERBLOCK = 256
_VEC_DIM = 8

_CB_NAME_RE = re.compile(r"^(NVFP4|FP8)_CB_([KS])(\d+)$")


@dataclass(frozen=True)
class CBSerializationContext:
    """Artifact-wide producer choices needed for exact CB byte pricing.

    ``codebook_source`` is the producer's sharing policy: ``lattice`` shares
    one table set per format; ``learned`` shares one per projection role.  A
    caller with an already-materialized config can pass exact physical tensor
    names through ``codebook_refs`` (qname -> string/list).  Omitting the
    context is an error on exact producer paths: otherwise an old caller would
    silently fall back to legacy-v1 bytes.
    """

    scale_coding: str
    codebook_source: str
    layout_version: int | None = None
    codebook_refs: Mapping[str, str | Sequence[str]] | None = None

    def __post_init__(self) -> None:
        coding = str(self.scale_coding).strip().lower()
        source = str(self.codebook_source).strip().lower()
        if coding not in _SCALE_CODINGS:
            raise ValueError(
                f"unknown CB scale_coding {self.scale_coding!r}; expected "
                f"{sorted(_SCALE_CODINGS)}"
            )
        if source not in {"lattice", "learned"}:
            raise ValueError(
                f"unknown CB codebook_source {self.codebook_source!r}; "
                "expected 'lattice' or 'learned'"
            )
        expected_layout = _LAYOUT_FOR_SCALE_CODING[coding]
        layout = expected_layout if self.layout_version is None else int(
            self.layout_version
        )
        if layout != expected_layout:
            raise ValueError(
                f"scale_coding={coding!r} requires layout_version="
                f"{expected_layout}, got {layout}"
            )
        object.__setattr__(self, "scale_coding", coding)
        object.__setattr__(self, "codebook_source", source)
        object.__setattr__(self, "layout_version", layout)
        if self.codebook_refs is not None:
            normalized_refs: dict[str, str | tuple[str, ...]] = {}
            for qname, raw_refs in self.codebook_refs.items():
                normalized_refs[str(qname)] = (
                    str(raw_refs)
                    if isinstance(raw_refs, str)
                    else tuple(str(item) for item in raw_refs)
                )
            object.__setattr__(self, "codebook_refs", normalized_refs)

    @classmethod
    def production(
        cls,
        *,
        codebook_source: str = "lattice",
        codebook_refs: Mapping[str, str | Sequence[str]] | None = None,
    ) -> CBSerializationContext:
        return cls(
            scale_coding=PRODUCTION_FP4_SCALE_CODING,
            layout_version=2,
            codebook_source=codebook_source,
            codebook_refs=codebook_refs,
        )

    @classmethod
    def legacy_v1(
        cls,
        *,
        codebook_source: str = "lattice",
        codebook_refs: Mapping[str, str | Sequence[str]] | None = None,
    ) -> CBSerializationContext:
        """Explicit legacy writer context; old artifacts remain readable."""
        return cls(
            scale_coding=LEGACY_FP4_SCALE_CODING,
            layout_version=1,
            codebook_source=codebook_source,
            codebook_refs=codebook_refs,
        )


def cb_serialization_context_stamp(context: CBSerializationContext) -> dict:
    """Small identity stamp suitable for an allocator recipe's metadata."""
    if context is None:
        raise ValueError("CB serialization context stamp requires a context")
    return {
        "schema": CB_SERIALIZED_PAYLOAD_SCHEMA,
        "scale_coding": context.scale_coding,
        "layout_version": context.layout_version,
        "codebook_source": context.codebook_source,
    }


def validate_cb_serialization_context_stamp(
    stamp: Mapping[str, object] | None,
    context: CBSerializationContext,
    *,
    where: str,
) -> None:
    """Fail when export choices drift from an allocator-stamped recipe.

    Hand-written and historical recipes have no stamp and remain loadable.
    Once a producer stamps the identity, however, ignoring a mismatch would
    make allocation bytes describe a different artifact than the exporter.
    """
    if stamp is None:
        return
    if not isinstance(stamp, Mapping):
        raise TypeError(f"{where}: CB serialized-payload stamp is not an object")
    expected = cb_serialization_context_stamp(context)
    observed = {key: stamp.get(key) for key in expected}
    if observed != expected:
        raise ValueError(
            f"{where}: CB serialization context differs from allocator "
            f"recipe: recipe={observed}, exporter={expected}"
        )


def _cb_info(format_name: str) -> tuple[str, str, int] | None:
    """Return ``(grid, mode, k)`` for a registered CB format."""
    canonical = str(format_name).strip().upper()
    match = _CB_NAME_RE.match(canonical)
    if match is None:
        return None
    try:
        registered = fr.get_format(canonical)
    except KeyError:
        return None
    if str(registered.name).strip().upper() != canonical:
        return None
    family, rung, raw_k = match.groups()
    if family == "FP8" and rung != "K":
        return None
    grid = "fp4" if family == "NVFP4" else "fp8"
    mode = "signed" if rung == "S" else "product"
    return grid, mode, int(raw_k)


def is_cb_format(format_name: str) -> bool:
    return _cb_info(format_name) is not None


def _bit_split(k: int, n_sub: int) -> tuple[int, ...]:
    base, extra = divmod(int(k), int(n_sub))
    return tuple(base + (1 if index < extra else 0) for index in range(n_sub))


def codebook_subtable_shapes(format_name: str) -> tuple[tuple[int, int], ...]:
    """Exact FP16 subtable shapes emitted for one CB format."""
    info = _cb_info(format_name)
    if info is None:
        raise ValueError(f"{format_name!r} is not a CB format")
    grid, mode, k = info
    if mode == "signed":
        magnitude_bits = k - _VEC_DIM
        if magnitude_bits < 1:
            raise ValueError(f"signed CB format requires k > {_VEC_DIM}, got {k}")
        return ((1 << magnitude_bits, _VEC_DIM),)
    n_sub = 2 if grid == "fp4" else 4
    sub_dim = _VEC_DIM // n_sub
    return tuple((1 << bits, sub_dim) for bits in _bit_split(k, n_sub))


def codebook_sidecar_payload_bytes(format_name: str) -> int:
    """FP16 tensor payload bytes for one codebook table set."""
    return sum(rows * cols * _FP16_BYTES
               for rows, cols in codebook_subtable_shapes(format_name))


def _default_logical_ref(qname: str, source: str) -> str:
    return "lattice" if source == "lattice" else str(qname).rsplit(".", 1)[-1]


def _physical_codebook_refs(
    qname: str,
    format_name: str,
    context: CBSerializationContext,
) -> tuple[str, ...]:
    expected_count = len(codebook_subtable_shapes(format_name))
    supplied = None
    if context.codebook_refs is not None:
        supplied = context.codebook_refs.get(qname)
    if supplied is not None:
        refs = (supplied,) if isinstance(supplied, str) else tuple(
            str(item) for item in supplied
        )
        if len(refs) != expected_count:
            raise ValueError(
                f"{qname}: {format_name} needs {expected_count} codebook "
                f"subtable ref(s), got {len(refs)}"
            )
        return tuple(str(item) for item in refs)

    logical = _default_logical_ref(qname, context.codebook_source)
    base = f"cb_codebook.{logical}.{str(format_name).strip().upper()}"
    if expected_count == 1:
        return (base,)
    return tuple(f"{base}.sub{index}" for index in range(expected_count))


def _sidecar_identity(
    qname: str,
    format_name: str,
    context: CBSerializationContext,
) -> dict:
    canonical = str(format_name).strip().upper()
    refs = _physical_codebook_refs(qname, canonical, context)
    shapes = codebook_subtable_shapes(canonical)
    return {
        "format": canonical,
        "codebook_source": context.codebook_source,
        "codebook_ref": list(refs),
        "dtype": "float16",
        "subtable_shapes": [list(shape) for shape in shapes],
        "payload_bytes": codebook_sidecar_payload_bytes(canonical),
    }


def _identity_key(identity: Mapping) -> str:
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def cb_tensor_payload_breakdown(
    format_name: str,
    shape: tuple[int, ...] | Sequence[int],
    *,
    qname: str,
    context: CBSerializationContext,
) -> dict:
    """Versioned byte breakdown for one serialized CB Linear.

    ``tensor_payload_bytes`` excludes the shared codebook sidecar; the returned
    ``sidecar_identity`` is what assignment-level accounting deduplicates.
    """
    if context is None:
        raise ValueError(
            "exact CB byte pricing requires CBSerializationContext; refusing "
            "to assume legacy-v1 bytes"
        )
    canonical = str(format_name).strip().upper()
    info = _cb_info(canonical)
    if info is None:
        raise ValueError(f"{format_name!r} is not a CB format")
    dims = tuple(int(dim) for dim in shape)
    if len(dims) < 2 or any(dim <= 0 for dim in dims):
        raise ValueError(
            f"{qname}: exact CB bytes need a positive rank>=2 Linear shape, "
            f"got {dims}"
        )
    in_features = dims[-1]
    if in_features % _SUPERBLOCK:
        raise ValueError(
            f"{qname}: CB in_features={in_features} is not divisible by "
            f"{_SUPERBLOCK}"
        )
    output_rows = int(math.prod(dims[:-1]))
    n_superblocks = in_features // _SUPERBLOCK
    grid, mode, k = info
    index_bytes = output_rows * n_superblocks * (4 * k)
    fp4_scale_bytes = 0
    if grid == "fp4":
        scale_bytes_per_superblock = (
            9 if context.scale_coding == PRODUCTION_FP4_SCALE_CODING else 16
        )
        fp4_scale_bytes = output_rows * n_superblocks * scale_bytes_per_superblock
    fp8_row_scale_bytes = _FP32_BYTES * output_rows if grid == "fp8" else 0
    packed_weight_bytes = index_bytes + fp4_scale_bytes
    tensor_payload_bytes = packed_weight_bytes + fp8_row_scale_bytes
    sidecar = _sidecar_identity(qname, canonical, context)
    identity = {
        "schema": CB_SERIALIZED_PAYLOAD_SCHEMA,
        "format": canonical,
        "grid": grid,
        "mode": mode,
        "k": k,
        "artifact_scale_coding": context.scale_coding,
        "layout_version": context.layout_version,
        "tensor_scale_coding": context.scale_coding if grid == "fp4" else "none",
        "type_size": (4 * k + (9 if context.scale_coding ==
                               PRODUCTION_FP4_SCALE_CODING else 16))
        if grid == "fp4" else 4 * k,
        "sidecar": sidecar,
    }
    return {
        "schema": CB_SERIALIZED_PAYLOAD_SCHEMA,
        "identity": identity,
        "identity_key": _identity_key(identity),
        "qname": str(qname),
        "format": canonical,
        "shape": list(dims),
        "params": int(math.prod(dims)),
        "output_rows": output_rows,
        "superblocks_per_row": n_superblocks,
        "index_bytes": int(index_bytes),
        "fp4_scale_bytes": int(fp4_scale_bytes),
        "fp8_row_scale_bytes": int(fp8_row_scale_bytes),
        "global_scale_bytes": 0,
        "packed_weight_bytes": int(packed_weight_bytes),
        "tensor_payload_bytes": int(tensor_payload_bytes),
        "sidecar_identity": sidecar,
        "sidecar_identity_key": _identity_key(sidecar),
        "sidecar_payload_bytes": int(sidecar["payload_bytes"]),
    }


def cb_assignment_payload_breakdown(
    assignment: Mapping[str, str],
    shapes: Mapping[str, tuple[int, ...] | Sequence[int]],
    *,
    context: CBSerializationContext,
) -> dict:
    """Exact CB payload bytes for an assignment, deduplicating sidecars."""
    if context is None:
        raise ValueError(
            "exact CB assignment pricing requires CBSerializationContext; "
            "refusing to assume legacy-v1 bytes"
        )
    per_tensor: dict[str, dict] = {}
    sidecars: dict[str, dict] = {}
    totals = {
        "index_bytes": 0,
        "fp4_scale_bytes": 0,
        "fp8_row_scale_bytes": 0,
        "global_scale_bytes": 0,
        "tensor_payload_bytes": 0,
    }
    for qname, format_name in assignment.items():
        if not is_cb_format(format_name):
            continue
        if qname not in shapes:
            raise KeyError(f"CB byte accounting has no shape for {qname!r}")
        item = cb_tensor_payload_breakdown(
            format_name, shapes[qname], qname=qname, context=context
        )
        per_tensor[qname] = item
        for key in totals:
            totals[key] += int(item[key])
        sidecar_key = item["sidecar_identity_key"]
        previous = sidecars.get(sidecar_key)
        if previous is None:
            sidecars[sidecar_key] = item["sidecar_identity"]
        elif previous != item["sidecar_identity"]:
            raise ValueError(
                f"conflicting CB sidecar identity for {qname}: {previous} vs "
                f"{item['sidecar_identity']}"
            )
    sidecar_bytes = sum(int(item["payload_bytes"]) for item in sidecars.values())
    total_bytes = totals["tensor_payload_bytes"] + sidecar_bytes
    return {
        "schema": CB_SERIALIZED_PAYLOAD_SCHEMA,
        "context": {
            "scale_coding": context.scale_coding,
            "layout_version": context.layout_version,
            "codebook_source": context.codebook_source,
        },
        **{key: int(value) for key, value in totals.items()},
        "codebook_sidecar_bytes": int(sidecar_bytes),
        "total_bytes": int(total_bytes),
        "per_tensor": per_tensor,
        "sidecars": list(sidecars.values()),
    }


def cb_payload_summary(breakdown: Mapping[str, object]) -> dict:
    """Compact persisted form of an assignment payload breakdown.

    Export provenance needs the version/layout and shared-sidecar identities,
    but duplicating every per-tensor record in ``quant_config.json`` would be
    needlessly large.  Keep the independently checkable totals and physical
    sidecar identities; config groups already map tensors to those refs.
    """
    byte_keys = (
        "index_bytes",
        "fp4_scale_bytes",
        "fp8_row_scale_bytes",
        "global_scale_bytes",
        "tensor_payload_bytes",
        "codebook_sidecar_bytes",
        "total_bytes",
    )
    per_tensor = breakdown.get("per_tensor", {})
    if not isinstance(per_tensor, Mapping):
        raise TypeError("CB payload breakdown has invalid per_tensor data")
    sidecars = breakdown.get("sidecars", [])
    if not isinstance(sidecars, list):
        raise TypeError("CB payload breakdown has invalid sidecars data")
    return {
        "schema": breakdown.get("schema", CB_SERIALIZED_PAYLOAD_SCHEMA),
        "context": breakdown.get("context"),
        **{key: int(breakdown.get(key, 0)) for key in byte_keys},
        "n_tensors": len(per_tensor),
        "sidecars": sidecars,
    }


def _safetensors_data_spans(path: Path) -> dict[str, int]:
    """Read exact tensor data-span bytes from a safetensors container.

    The serialized-payload API deliberately prices tensor data spans.  A real
    file is larger by its eight-byte prefix plus JSON header, so exporters use
    this parser for the separate final artifact inventory instead of treating
    payload bytes as filesystem bytes.
    """
    size = path.stat().st_size
    if size < 8:
        raise AssertionError(f"{path}: truncated safetensors prefix ({size}B)")
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        (header_length,) = struct.unpack("<Q", raw_length)
        if header_length > size - 8:
            raise AssertionError(
                f"{path}: safetensors header length {header_length} exceeds "
                f"the {size}B container"
            )
        raw_header = handle.read(header_length)
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AssertionError(f"{path}: invalid safetensors JSON header") from exc
    if not isinstance(header, Mapping):
        raise TypeError(f"{path}: safetensors header is not an object")

    spans: dict[str, int] = {}
    ranges: list[tuple[int, int, str]] = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, Mapping):
            raise TypeError(f"{path}: tensor {name!r} header is not an object")
        offsets = entry.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise AssertionError(f"{path}: tensor {name!r} has invalid offsets")
        start, end = (int(offsets[0]), int(offsets[1]))
        if start < 0 or end < start:
            raise AssertionError(
                f"{path}: tensor {name!r} has invalid span [{start}, {end})"
            )
        spans[str(name)] = end - start
        ranges.append((start, end, str(name)))

    previous_end = 0
    for start, end, name in sorted(ranges):
        if start < previous_end:
            raise AssertionError(
                f"{path}: tensor {name!r} overlaps a preceding data span"
            )
        previous_end = end
    data_start = 8 + int(header_length)
    if data_start + previous_end != size:
        raise AssertionError(
            f"{path}: header plus tensor extent is {data_start + previous_end}B "
            f"but container is {size}B"
        )
    return spans


def cb_export_artifact_inventory(
    out_dir: str | Path,
    *,
    serialized_payload: Mapping[str, object],
    cb_tensor_names: Sequence[str],
    codebook_file: str | None,
) -> dict:
    """Inventory an already-written CB export and assert both byte scopes.

    ``serialized_payload`` is the analytic CB tensor-data contract.  The
    returned ``export_directory_bytes`` is a different, measured quantity: all
    regular files below ``out_dir``, including safetensors headers, the
    codebook container header, JSON configs, tokenizer files, and any other
    copied sidecars.  Keeping both fields prevents the allocator's payload
    budget from being misreported as an exact filesystem size.
    """
    root = Path(out_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"CB export directory does not exist: {root}")
    files = {
        path.relative_to(root).as_posix(): int(path.stat().st_size)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    if not files:
        raise AssertionError(f"{root}: CB export produced no files")

    container_spans: dict[str, dict[str, int]] = {}
    for relative in files:
        path = root / relative
        if path.suffix == ".safetensors" or (
            codebook_file is not None and relative == codebook_file
        ):
            container_spans[relative] = _safetensors_data_spans(path)

    expected_names = {str(name) for name in cb_tensor_names}
    found_names: dict[str, tuple[str, int]] = {}
    for relative, spans in container_spans.items():
        if codebook_file is not None and relative == codebook_file:
            continue
        for name, span in spans.items():
            if name not in expected_names:
                continue
            if name in found_names:
                raise AssertionError(
                    f"{root}: CB tensor {name!r} appears in both "
                    f"{found_names[name][0]!r} and {relative!r}"
                )
            found_names[name] = (relative, int(span))
    if set(found_names) != expected_names:
        raise AssertionError(
            f"{root}: final CB tensor inventory differs from the export plan: "
            f"missing={sorted(expected_names - set(found_names))}, "
            f"extra={sorted(set(found_names) - expected_names)}"
        )
    cb_tensor_bytes = sum(span for _relative, span in found_names.values())

    codebook_spans = (
        container_spans.get(codebook_file, {}) if codebook_file is not None else {}
    )
    cb_codebook_bytes = sum(codebook_spans.values())
    expected_tensor_bytes = int(serialized_payload.get("tensor_payload_bytes", 0))
    expected_codebook_bytes = int(
        serialized_payload.get("codebook_sidecar_bytes", 0)
    )
    if cb_tensor_bytes != expected_tensor_bytes:
        raise AssertionError(
            f"{root}: final CB tensor data spans are {cb_tensor_bytes}B, "
            f"accounting expected {expected_tensor_bytes}B"
        )
    if cb_codebook_bytes != expected_codebook_bytes:
        raise AssertionError(
            f"{root}: final codebook data spans are {cb_codebook_bytes}B, "
            f"accounting expected {expected_codebook_bytes}B"
        )

    container_bytes = sum(files[name] for name in container_spans)
    tensor_data_bytes = sum(
        sum(spans.values()) for spans in container_spans.values()
    )
    directory_bytes = sum(files.values())
    cb_payload_bytes = cb_tensor_bytes + cb_codebook_bytes
    expected_total = int(serialized_payload.get("total_bytes", 0))
    if cb_payload_bytes != expected_total:
        raise AssertionError(
            f"{root}: final CB payload is {cb_payload_bytes}B, accounting "
            f"expected {expected_total}B"
        )
    return {
        "schema": CB_EXPORT_ARTIFACT_INVENTORY_SCHEMA,
        "scope": "all_regular_files_recursive",
        "file_bytes": files,
        "export_directory_bytes": int(directory_bytes),
        "safetensors_container_bytes": int(container_bytes),
        "safetensors_tensor_data_bytes": int(tensor_data_bytes),
        "safetensors_container_overhead_bytes": int(
            container_bytes - tensor_data_bytes
        ),
        "non_safetensors_file_bytes": int(directory_bytes - container_bytes),
        "cb_serialized_payload_bytes": int(cb_payload_bytes),
        "cb_tensor_payload_bytes": int(cb_tensor_bytes),
        "cb_codebook_sidecar_bytes": int(cb_codebook_bytes),
    }


def finalize_cb_export_artifact_inventory(
    out_dir: str | Path,
    quant_config: dict,
    *,
    serialized_payload: Mapping[str, object],
    cb_tensor_names: Sequence[str],
    codebook_file: str | None,
) -> dict:
    """Write ``quant_config.json`` with a self-consistent final inventory.

    The inventory includes ``quant_config.json`` itself.  Its byte length can
    change when the measured totals are embedded, so write/measure iterations
    continue until the embedded inventory equals the bytes on disk.  The
    representation contains sizes rather than a self-hash and converges in a
    handful of iterations; failure to converge is a hard exporter error.
    """
    root = Path(out_dir)
    provenance = quant_config.setdefault("provenance", {})
    if not isinstance(provenance, dict):
        raise TypeError("quant_config provenance must be an object")
    provenance.setdefault(
        "artifact_inventory",
        {
            "schema": CB_EXPORT_ARTIFACT_INVENTORY_SCHEMA,
            "scope": "pending_final_write",
        },
    )
    config_path = root / "quant_config.json"
    for _attempt in range(16):
        config_path.write_text(json.dumps(quant_config, indent=2, sort_keys=True))
        inventory = cb_export_artifact_inventory(
            root,
            serialized_payload=serialized_payload,
            cb_tensor_names=cb_tensor_names,
            codebook_file=codebook_file,
        )
        if provenance.get("artifact_inventory") == inventory:
            return inventory
        provenance["artifact_inventory"] = inventory
    raise AssertionError(
        f"{root}: quant_config artifact inventory did not reach a byte-size "
        "fixed point after 16 writes"
    )


def validate_cb_sidecar_tensors(
    breakdown: Mapping[str, object],
    tensors: Mapping[str, object],
    *,
    where: str,
) -> int:
    """Assert that materialized sidecars match a payload identity exactly.

    Kept torch-free by using the small tensor protocol shared by torch tensors:
    ``dtype``, ``shape``, ``numel()``, and ``element_size()``. Returns the
    actual tensor payload bytes after validating refs, FP16 dtype, shapes, and
    the aggregate byte total.
    """
    raw_sidecars = breakdown.get("sidecars", [])
    if not isinstance(raw_sidecars, list):
        raise TypeError(f"{where}: CB payload sidecars are not a list")
    expected: dict[str, tuple[int, ...]] = {}
    for sidecar in raw_sidecars:
        if not isinstance(sidecar, Mapping):
            raise TypeError(f"{where}: CB sidecar identity is not an object")
        refs = sidecar.get("codebook_ref", [])
        shapes = sidecar.get("subtable_shapes", [])
        if not isinstance(refs, list) or not isinstance(shapes, list):
            raise TypeError(f"{where}: CB sidecar refs/shapes are not lists")
        if len(refs) != len(shapes):
            raise AssertionError(
                f"{where}: CB sidecar has {len(refs)} refs but "
                f"{len(shapes)} shapes"
            )
        for ref_name, ref_shape in zip(refs, shapes):
            name = str(ref_name)
            shape_tuple = tuple(int(dim) for dim in ref_shape)
            previous = expected.setdefault(name, shape_tuple)
            if previous != shape_tuple:
                raise AssertionError(
                    f"{where}: {name} has conflicting expected shapes "
                    f"{previous} and {shape_tuple}"
                )
    if set(tensors) != set(expected):
        raise AssertionError(
            f"{where}: emitted CB sidecars do not match accounting identity: "
            f"expected={sorted(expected)}, actual={sorted(tensors)}"
        )
    actual_bytes = 0
    for name, expected_shape in expected.items():
        tensor = tensors[name]
        dtype_name = str(getattr(tensor, "dtype", "")).removeprefix("torch.")
        shape = tuple(int(dim) for dim in getattr(tensor, "shape", ()))
        if dtype_name != "float16" or shape != expected_shape:
            raise AssertionError(
                f"{where}: {name} emitted {dtype_name}{shape}, accounting "
                f"identity requires float16{expected_shape}"
            )
        actual_bytes += int(tensor.numel()) * int(tensor.element_size())
    expected_bytes = int(breakdown.get("codebook_sidecar_bytes", 0))
    if actual_bytes != expected_bytes:
        raise AssertionError(
            f"{where}: emitted CB sidecars are {actual_bytes}B, accounting "
            f"expected {expected_bytes}B"
        )
    return actual_bytes


def _legacy_context_from_sources(
    assignment: Mapping[str, str],
    codebook_sources: Mapping[str, object] | None,
    *,
    scale_coding: str,
) -> CBSerializationContext:
    sources = codebook_sources or {}
    kinds: set[str] = set()
    refs: dict[str, str | Sequence[str]] = {}
    for qname, format_name in assignment.items():
        if not is_cb_format(format_name):
            continue
        raw = sources.get(qname)
        kind = "lattice"
        logical_ref = None
        if isinstance(raw, str):
            candidate = raw.strip().lower()
            if candidate in {"lattice", "learned"}:
                kind = candidate
        elif isinstance(raw, Mapping):
            if "learned" in raw:
                kind = "learned"
            elif "lattice" in raw:
                kind = "lattice"
            else:
                candidate = str(raw.get("kind", raw.get("source", "lattice"))).lower()
                if candidate in {"lattice", "learned"}:
                    kind = candidate
            logical_ref = raw.get("group") or raw.get("shared_group") or raw.get(
                "codebook_group"
            )
        kinds.add(kind)
        if logical_ref:
            count = len(codebook_subtable_shapes(format_name))
            base = f"cb_codebook.{logical_ref}.{str(format_name).strip().upper()}"
            refs[qname] = base if count == 1 else [
                f"{base}.sub{index}" for index in range(count)
            ]
    if len(kinds) > 1:
        raise ValueError(
            "cb_footprint compatibility wrapper cannot represent mixed lattice/"
            "learned sources in one artifact context; use "
            "cb_assignment_payload_breakdown with exact codebook refs"
        )
    source = next(iter(kinds), "lattice")
    return CBSerializationContext(
        scale_coding=scale_coding,
        codebook_source=source,
        codebook_refs=refs or None,
    )


def cb_footprint(
    assignment: Mapping[str, str],
    shapes: Mapping[str, tuple[int, ...]],
    *,
    codebook_sources: Mapping[str, object] | None = None,
    scale_coding: str = PRODUCTION_FP4_SCALE_CODING,
    context: CBSerializationContext | None = None,
) -> dict:
    """Backwards-compatible mixed-assignment footprint wrapper.

    Unlike the obsolete Phase-0 formula, lattice tables are real FP16
    sidecars and are charged, FP4 has no global scalar, and production defaults
    to layout-v2.  Pass ``context=CBSerializationContext.legacy_v1(...)`` to
    reproduce an older artifact explicitly.
    """
    ctx = context or _legacy_context_from_sources(
        assignment, codebook_sources, scale_coding=scale_coding
    )
    cb_assignment = {
        qname: fmt for qname, fmt in assignment.items() if is_cb_format(fmt)
    }
    cb_breakdown = cb_assignment_payload_breakdown(
        cb_assignment, shapes, context=ctx
    )
    non_cb_bytes = 0
    n_params = 0
    per_tensor: dict[str, dict] = {}
    sidecar_first_owner: set[str] = set()
    for qname, format_name in assignment.items():
        if qname not in shapes:
            raise KeyError(f"cb_footprint: no shape for {qname!r}")
        shape = tuple(int(dim) for dim in shapes[qname])
        params = int(math.prod(shape)) if shape else 1
        n_params += params
        if is_cb_format(format_name):
            item = cb_breakdown["per_tensor"][qname]
            sidecar_key = item["sidecar_identity_key"]
            charged = 0
            if sidecar_key not in sidecar_first_owner:
                sidecar_first_owner.add(sidecar_key)
                charged = int(item["sidecar_payload_bytes"])
            grid, _mode, k = _cb_info(format_name)  # type: ignore[misc]
            per_tensor[qname] = {
                "format": str(format_name),
                "k": k,
                "cb_family": "nvfp4" if grid == "fp4" else "fp8",
                "params": params,
                "body_bytes": int(item["packed_weight_bytes"]),
                "global_scale_bytes": 0,
                "channel_scale_bytes": int(item["fp8_row_scale_bytes"]),
                "sidecar_bytes": charged,
                "codebook_source": ctx.codebook_source,
                "body_bpw": 8.0 * int(item["packed_weight_bytes"]) / max(params, 1),
                "serialization_identity": item["identity"],
            }
        else:
            spec = fr.get_format(str(format_name))
            body = int(spec.memory_bytes_for_shape(shape))
            non_cb_bytes += body
            per_tensor[qname] = {
                "format": str(format_name),
                "k": None,
                "cb_family": None,
                "params": params,
                "body_bytes": body,
                "global_scale_bytes": 0,
                "channel_scale_bytes": 0,
                "sidecar_bytes": 0,
                "codebook_source": "none",
                "body_bpw": 8.0 * body / max(params, 1),
            }
    body_bytes = int(cb_breakdown["index_bytes"] +
                     cb_breakdown["fp4_scale_bytes"] + non_cb_bytes)
    channel_scale_bytes = int(cb_breakdown["fp8_row_scale_bytes"])
    sidecar_bytes = int(cb_breakdown["codebook_sidecar_bytes"])
    total_bytes = body_bytes + channel_scale_bytes + sidecar_bytes
    return {
        "schema": CB_SERIALIZED_PAYLOAD_SCHEMA,
        "serialization_context": cb_breakdown["context"],
        "total_bytes": total_bytes,
        "body_bytes": body_bytes,
        "sidecar_bytes": sidecar_bytes,
        "codebook_sidecar_bytes": sidecar_bytes,
        "global_scale_bytes": 0,
        "channel_scale_bytes": channel_scale_bytes,
        "fp8_row_scale_bytes": channel_scale_bytes,
        "index_bytes": int(cb_breakdown["index_bytes"]),
        "fp4_scale_bytes": int(cb_breakdown["fp4_scale_bytes"]),
        "n_params": int(n_params),
        "body_bpw": 8.0 * body_bytes / max(n_params, 1),
        "total_bpw": 8.0 * total_bytes / max(n_params, 1),
        "per_tensor": per_tensor,
        "sidecars": cb_breakdown["sidecars"],
    }
