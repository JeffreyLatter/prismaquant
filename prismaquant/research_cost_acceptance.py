"""Explicit acceptance boundary for study-grade assembled cost tables.

Production CB cost/cache provenance remains fail-closed.  This module owns the
one sanctioned exception: a user-acknowledged, content-inventoried assembly of
complete per-layer study segments over a production base table.
"""
from __future__ import annotations

import copy
import hashlib
import pickle
import re
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path


RESEARCH_COST_PROVENANCE = (
    "research_assembled_segments_user_accepted_2026-08-03"
)
RESEARCH_COST_MANIFEST_SCHEMA = "prismaquant.research_cost_manifest.v1"
DEFAULT_SEGMENT_FORMATS = tuple(
    f"NVFP4_CB_K{k}" for k in range(12, 19)
)
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_SCALARS = ("predicted_dloss", "output_mse", "rel_output_mse", "weight_mse")


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, dict) or not isinstance(payload.get("costs"), dict):
        raise ValueError(f"{path}: expected a cost payload with a costs mapping")
    return payload


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _float_bits(value: object) -> bytes | None:
    try:
        return struct.pack(">d", float(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _overlap_check(base: Mapping, segments: Mapping, formats: Sequence[str]) -> dict:
    result: dict[str, dict] = {}
    for fmt in formats:
        compared = full_equal = 0
        scalar_equal = {key: 0 for key in _SCALARS}
        scalar_compared = {key: 0 for key in _SCALARS}
        segment_sources: dict[str, int] = {}
        for name, seg_row in segments.items():
            base_entry = (base.get(name) or {}).get(fmt)
            seg_entry = (seg_row or {}).get(fmt)
            if not isinstance(base_entry, Mapping) or not isinstance(seg_entry, Mapping):
                continue
            compared += 1
            full_equal += int(dict(base_entry) == dict(seg_entry))
            source = str(seg_entry.get("cost_source", "measured_unstamped"))
            segment_sources[source] = segment_sources.get(source, 0) + 1
            for key in _SCALARS:
                if key not in base_entry or key not in seg_entry:
                    continue
                scalar_compared[key] += 1
                scalar_equal[key] += int(
                    _float_bits(base_entry[key]) == _float_bits(seg_entry[key])
                )
        result[fmt] = {
            "entries_compared": compared,
            "full_entry_bit_equal": full_equal,
            "full_entry_bit_different": compared - full_equal,
            "shared_scalar_bit_equal": scalar_equal,
            "shared_scalar_compared": scalar_compared,
            "all_full_entries_bit_equal": bool(compared and full_equal == compared),
            "segment_cost_sources": segment_sources,
        }
    return result


def assemble_research_cost_table(
    base_path: str | Path,
    segments_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    expected_layers: int = 43,
    expected_rows_per_layer: int = 775,
    segment_formats: Sequence[str] = DEFAULT_SEGMENT_FORMATS,
) -> tuple[dict, dict]:
    """Assemble verified layer segments over ``base_path`` and stamp acceptance.

    The base wins every overlapping cell.  This is deliberately stronger than
    inspecting ``cost_source``: a production measurement is never displaced by
    study data, including a study entry whose provenance field is incomplete.
    """
    base_path = Path(base_path).resolve()
    segments_dir = Path(segments_dir).resolve()
    base = _load_pickle(base_path)
    layer_files = sorted(segments_dir.glob("layer_*.pkl"))
    expected_ids = list(range(expected_layers))
    found_ids: list[int] = []
    segment_costs: dict[str, dict] = {}
    sources: list[dict] = []

    for path in layer_files:
        match = re.fullmatch(r"layer_(\d+)\.pkl", path.name)
        if match is None:
            continue
        layer = int(match.group(1))
        found_ids.append(layer)
        payload = _load_pickle(path)
        costs = payload["costs"]
        if len(costs) != expected_rows_per_layer:
            raise ValueError(
                f"{path}: expected {expected_rows_per_layer} rows, got {len(costs)}"
            )
        actual_layers = {
            int(m.group(1))
            for name in costs
            if (m := _LAYER_RE.search(str(name))) is not None
        }
        unkeyed = [name for name in costs if _LAYER_RE.search(str(name)) is None]
        if actual_layers != {layer} or unkeyed:
            raise ValueError(
                f"{path}: filename layer {layer} does not match row keying "
                f"(layers={sorted(actual_layers)}, unkeyed={unkeyed[:3]})"
            )
        missing_formats = sorted(
            fmt for fmt in segment_formats
            if any(fmt not in row for row in costs.values())
        )
        if missing_formats:
            raise ValueError(f"{path}: missing format columns {missing_formats}")
        duplicate = sorted(set(segment_costs).intersection(costs))
        if duplicate:
            raise ValueError(f"{path}: duplicate cost rows, e.g. {duplicate[:3]}")
        segment_costs.update(copy.deepcopy(costs))
        sources.append({
            "layer": layer,
            "path": str(path),
            "sha256": _sha256(path),
            "row_count": len(costs),
            "row_layer_keying_verified": True,
        })

    if found_ids != expected_ids:
        raise ValueError(
            f"{segments_dir}: expected layer ids {expected_ids}, got {found_ids}"
        )
    expected_total = expected_layers * expected_rows_per_layer
    if len(segment_costs) != expected_total:
        raise ValueError(
            f"assembled segments: expected {expected_total} unique rows, "
            f"got {len(segment_costs)}"
        )
    base_costs = base["costs"]
    if set(segment_costs) != set(base_costs):
        missing = sorted(set(base_costs) - set(segment_costs))
        extra = sorted(set(segment_costs) - set(base_costs))
        raise ValueError(
            "segment/base row sets differ: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )

    overlap = _overlap_check(
        base_costs, segment_costs, ("NVFP4_CB_K14", "NVFP4_CB_K15")
    )
    merged_costs = copy.deepcopy(segment_costs)
    overlap_cells = 0
    for name, base_row in base_costs.items():
        for fmt, entry in base_row.items():
            overlap_cells += int(fmt in merged_costs[name])
            merged_costs[name][fmt] = copy.deepcopy(entry)

    manifest = {
        "schema": RESEARCH_COST_MANIFEST_SCHEMA,
        "cost_provenance": RESEARCH_COST_PROVENANCE,
        "acceptance": "explicit_user_decision_for_learning_experiment",
        "base": {
            "path": str(base_path),
            "sha256": _sha256(base_path),
            "row_count": len(base_costs),
        },
        "segments_directory": str(segments_dir),
        "layers": sources,
        "layer_count": expected_layers,
        "rows_per_layer": expected_rows_per_layer,
        "assembled_row_count": len(merged_costs),
        "segment_formats": list(segment_formats),
        "formats": sorted(set(base.get("formats", ())) | set(segment_formats)),
        "precedence": "production_v2_base_wins_every_overlapping_cell",
        "base_overlap_cells_kept": overlap_cells,
        "k14_k15_cross_run_bit_equality": overlap,
    }
    provenance = copy.deepcopy(base.get("provenance") or {})
    provenance["cost_provenance"] = RESEARCH_COST_PROVENANCE
    provenance["research_cost_manifest"] = manifest
    assembled = {
        "costs": merged_costs,
        "formats": manifest["formats"],
        "provenance": provenance,
        "meta": {
            **copy.deepcopy(base.get("meta") or {}),
            "research_assembly": {
                "row_count": len(merged_costs),
                "layer_count": expected_layers,
                "rows_per_layer": expected_rows_per_layer,
            },
        },
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as fh:
            pickle.dump(assembled, fh)
    return assembled, manifest


def accepted_cost_provenance(payload: Mapping) -> dict | None:
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        return None
    if provenance.get("cost_provenance") != RESEARCH_COST_PROVENANCE:
        return None
    manifest = provenance.get("research_cost_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("research-stamped cost table has no assembly manifest")
    if manifest.get("schema") != RESEARCH_COST_MANIFEST_SCHEMA:
        raise ValueError("research-stamped cost table has an unknown manifest schema")
    if int(manifest.get("assembled_row_count", -1)) != len(payload.get("costs", {})):
        raise ValueError("research cost manifest row count does not match the table")
    return copy.deepcopy(dict(manifest))


def propagated_cost_provenance(manifest: Mapping | None) -> dict:
    """JSON-safe fragment used by both selection and layer-config emission."""
    if manifest is None:
        return {}
    return {"cost_provenance": copy.deepcopy(dict(manifest))}


def enforce_research_export_acknowledgement(
    layer_config_payload: Mapping,
    *,
    acknowledged: bool,
    where: str,
) -> dict | None:
    meta = layer_config_payload.get("__prismaquant__")
    stamp = meta.get("cost_provenance") if isinstance(meta, Mapping) else None
    if stamp is None:
        return None
    if not isinstance(stamp, Mapping) or stamp.get("cost_provenance") != RESEARCH_COST_PROVENANCE:
        raise ValueError(f"{where}: malformed or unknown research cost provenance")
    if not acknowledged:
        raise ValueError(
            f"{where}: refusing to export a research-stamped cost selection; "
            "pass --allow-research-cost-selection only after separately "
            "acknowledging that study-grade assembled costs are not production provenance"
        )
    return copy.deepcopy(dict(stamp))
