"""Fail-closed pre-export gate for the fixed DSv4-Flash W8A16 release.

This module does not launch an exporter and never writes an artifact.  It
turns the reviewed readmission publication and every immutable exporter input
into one machine-readable handoff receipt immediately before the GPU job is
started.  The release driver may consume stdout only after this function
returns successfully.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

from prismaquant.allocator_candidates import (
    ROUTE_GRIDBOOK_FP8_SOURCE_W8A16,
    ROUTE_PENDING_PASSTHROUGH_FORMATS,
    SOURCE_PASSTHROUGH_CONTRACTS,
)
from prismaquant.anchored_cost import AURA_CURRENCY
from prismaquant.cb_anchored_cost import (
    CB_ANCHORED_COST_SCHEMA,
    CB_ARTIFACT_PUBLISH_SCHEMA,
)
from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from prismaquant.dsv4_aura_cb_reprice import (
    DSV4_BUDGET_BYTES,
    DSV4_TOTAL_UNITS,
    DSV4_W8A16_APPROVED_ASSIGNMENT_SHA256,
    DSV4_W8A16_APPROVED_CB_COL_WEIGHTS_SHA256,
    DSV4_W8A16_APPROVED_LAYER_CONFIG_SHA256,
    DSV4_W8A16_APPROVED_SELECTION,
    DSV4_W8A16_APPROVED_SELECTION_SHA256,
    DSV4_W8A16_READMISSION_SCHEMA,
)
from prismaquant.format_registry import get_format
from prismaquant.gridbook_runtime_pin import (
    GRIDBOOK_RUNTIME_CONTRACT_SCHEMA,
    load_gridbook_runtime_pin,
    require_exact_gridbook_runtime_release,
    supports_source_fp8_block128_w8a16,
)
from prismaquant.layer_config import load_assignment
from prismaquant.nvfp4_cb_footprint import (
    assignment_serialization_sha256,
    cb_serialization_metadata_from_assignment_payload,
)


DSV4_W8A16_EXPORT_HANDOFF_SCHEMA = (
    "prismaquant.dsv4_w8a16.export_handoff.v1"
)
_PUBLISH_MANIFEST = ".anchored_publish.json"
_PUBLISHED_FILES = frozenset({
    "layer_config.json",
    "selection.json",
    "pareto.knees.json",
    "cb_col_weights.pkl",
})
_FROZEN_EXPORT_CODE_SHA256 = {
    "prismaquant/export_nvfp4_cb_streaming.py": (
        "ba80d731556f8e2cfc2f6284a3e5f43e8c7e3e913e9f1572fc34ea907dfb81eb"
    ),
    "prismaquant/cb_export_config.py": (
        "41ef8609736d419e12e86810d2dd005f8e09ade2ad138bf5d5c4f123087e3acb"
    ),
    "prismaquant/nvfp4_cb_formats.py": (
        "bd7cfa7a8930bf2652a86f2a87649d51bf2af497cd350890e9246c7ff01bce9f"
    ),
}


class W8A16ExportHandoffError(RuntimeError):
    """The exact reviewed DSv4 W8A16 export handoff is not intact."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _real_file(path: Path, *, where: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise W8A16ExportHandoffError(f"{where} is not a regular file: {path}")
    return path


def _json_object(path: Path, *, where: str) -> dict[str, Any]:
    _real_file(path, where=where)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise W8A16ExportHandoffError(f"{where} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise W8A16ExportHandoffError(f"{where} is not a JSON object: {path}")
    return value


def _verify_publication(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    if root.is_symlink() or not root.is_dir():
        raise W8A16ExportHandoffError(
            f"readmission publication is not a real directory: {root}"
        )
    observed_names = {path.name for path in root.iterdir()}
    expected_names = _PUBLISHED_FILES | {_PUBLISH_MANIFEST}
    if observed_names != expected_names:
        raise W8A16ExportHandoffError(
            "readmission publication file set differs: "
            f"missing={sorted(expected_names - observed_names)}, "
            f"extra={sorted(observed_names - expected_names)}"
        )
    manifest = _json_object(
        root / _PUBLISH_MANIFEST, where="readmission publication manifest"
    )
    identity = manifest.get("identity")
    outputs = manifest.get("outputs")
    if not isinstance(identity, Mapping) or not isinstance(outputs, Mapping):
        raise W8A16ExportHandoffError(
            "readmission publication lacks identity/output mappings"
        )
    try:
        identity_sha256 = canonical_json_sha256(
            identity, where="DSv4 W8A16 export publication identity"
        )
    except (TypeError, ValueError) as exc:
        raise W8A16ExportHandoffError(
            "readmission publication identity is non-canonical"
        ) from exc
    if (
        manifest.get("schema") != CB_ARTIFACT_PUBLISH_SCHEMA
        or manifest.get("complete") is not True
        or manifest.get("identity_sha256") != identity_sha256
        or identity.get("schema") != CB_ARTIFACT_PUBLISH_SCHEMA
        or outputs != identity.get("outputs")
        or set(map(str, outputs)) != _PUBLISHED_FILES
    ):
        raise W8A16ExportHandoffError(
            "readmission publication is incomplete, unbound, or has the "
            "wrong output set"
        )
    observed: dict[str, str] = {}
    for name in sorted(_PUBLISHED_FILES):
        descriptor = outputs.get(name)
        path = _real_file(root / name, where=f"published {name}")
        if not isinstance(descriptor, Mapping):
            raise W8A16ExportHandoffError(
                f"published {name} has no checksum descriptor"
            )
        digest = _sha256(path)
        actual = {"size_bytes": path.stat().st_size, "sha256": digest}
        if descriptor != actual:
            raise W8A16ExportHandoffError(
                f"published {name} differs from its atomic manifest"
            )
        observed[name] = digest
    return manifest, observed


def _selection_contract(selection: Mapping[str, object]) -> dict[str, object]:
    whole = selection.get("whole_artifact_budget")
    if not isinstance(whole, Mapping):
        raise W8A16ExportHandoffError(
            "readmitted selection lacks whole-artifact accounting"
        )
    observed = {
        "budget_bytes": selection.get("budget_bytes"),
        "chosen_achieved_bits": selection.get("chosen_achieved_bits"),
        "predicted_dloss": selection.get("predicted_dloss"),
        "selection_tensor_payload_bytes": whole.get(
            "selection_tensor_payload_bytes"
        ),
        "selection_whole_artifact_upper_bound_bytes": whole.get(
            "selection_whole_artifact_upper_bound_bytes"
        ),
    }
    if observed != DSV4_W8A16_APPROVED_SELECTION:
        raise W8A16ExportHandoffError(
            f"readmitted selection metrics differ from approval: {observed}"
        )
    return observed


def _verify_frozen_export_code(repo_root: Path) -> dict[str, str]:
    if repo_root.is_symlink() or not repo_root.is_dir():
        raise W8A16ExportHandoffError(
            f"PrismaQuant root is not a real directory: {repo_root}"
        )
    observed: dict[str, str] = {}
    for relative, expected in _FROZEN_EXPORT_CODE_SHA256.items():
        path = _real_file(
            repo_root / relative, where=f"frozen exporter source {relative}"
        )
        digest = _sha256(path)
        if digest != expected:
            raise W8A16ExportHandoffError(
                f"frozen exporter source changed: {relative}; "
                f"observed={digest}, expected={expected}"
            )
        observed[relative] = digest
    return observed


def _verify_runtime_contract() -> dict[str, object]:
    try:
        pin = load_gridbook_runtime_pin()
        require_exact_gridbook_runtime_release(pin)
    except Exception as exc:
        raise W8A16ExportHandoffError(
            f"Gridbook release pin is unresolved: {exc}"
        ) from exc
    contract = SOURCE_PASSTHROUGH_CONTRACTS["FP8_BLOCK_UE8M0_SOURCE"]
    if (
        pin.version != "0.8.5"
        or pin.version_is_release is not True
        or pin.runtime_contract_schema != GRIDBOOK_RUNTIME_CONTRACT_SCHEMA
        or not supports_source_fp8_block128_w8a16(pin)
        or contract.serving_route != ROUTE_GRIDBOOK_FP8_SOURCE_W8A16
        or not contract.route_backed
        or "FP8_BLOCK_UE8M0_SOURCE" in ROUTE_PENDING_PASSTHROUGH_FORMATS
    ):
        raise W8A16ExportHandoffError(
            "FP8 block W8A16 is not backed by the exact released Gridbook "
            "runtime contract"
        )
    block = get_format("FP8_BLOCK_UE8M0_SOURCE")
    direct = get_format("MXFP8_UE8M0_G32")
    if (
        block.act_quant_changes_input
        or not direct.act_quant_changes_input
        or direct.act_bits != 8
    ):
        raise W8A16ExportHandoffError(
            "source W8A16 and direct group-32 W8A8 contracts have collapsed"
        )
    return {
        "schema": pin.schema,
        "repository": pin.repository,
        "commit": pin.commit,
        "version": pin.version,
        "version_is_release": pin.version_is_release,
        "runtime_contract_schema": pin.runtime_contract_schema,
        "required_abi_features": dict(pin.required_abi_features),
        "serving_route": contract.serving_route,
    }


def _verify_bundle(
    bundle_path: Path, layer_payload: Mapping[str, object]
) -> dict[str, object]:
    _real_file(bundle_path, where="immutable codebook bundle")
    context_stamp, _tensor_stamps = (
        cb_serialization_metadata_from_assignment_payload(layer_payload)
    )
    if not isinstance(context_stamp, Mapping):
        raise W8A16ExportHandoffError(
            "readmitted assignment lacks a CB serialization stamp"
        )
    try:
        from prismaquant.cb_learned_bundle import load_bundle

        bundle = load_bundle(bundle_path)
    except Exception as exc:
        raise W8A16ExportHandoffError(
            f"immutable codebook bundle is invalid: {bundle_path}"
        ) from exc
    if (
        context_stamp.get("codebook_content_sha256")
        != bundle.codebook_content_digests
        or context_stamp.get("codebook_source_by_format")
        != bundle.codebook_source_by_format
    ):
        raise W8A16ExportHandoffError(
            "codebook bundle bytes/source map differ from the assignment stamp"
        )
    return {
        "path": str(bundle_path.resolve(strict=True)),
        "file_sha256": _sha256(bundle_path),
        "bundle_content_sha256": bundle.bundle_content_sha256,
        "codebook_count": len(bundle.codebook_content_digests),
    }


def verify_dsv4_w8a16_export_handoff(
    *,
    publication_dir: str | Path,
    approved_raw_publication_dir: str | Path,
    source_model_dir: str | Path,
    source_identity_path: str | Path,
    codebook_bundle_path: str | Path,
    output_path: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    """Verify the fixed release handoff without mutating any input or output."""

    publication = Path(publication_dir)
    approved_raw = Path(approved_raw_publication_dir)
    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise W8A16ExportHandoffError(
            f"export output already exists; refusing clobber: {output}"
        )
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise W8A16ExportHandoffError(
            f"export output parent is not a real directory: {output.parent}"
        )

    manifest, published_sha256 = _verify_publication(publication)
    _raw_manifest, raw_sha256 = _verify_publication(approved_raw)
    expected_raw = {
        "layer_config.json": DSV4_W8A16_APPROVED_LAYER_CONFIG_SHA256,
        "selection.json": DSV4_W8A16_APPROVED_SELECTION_SHA256,
        "cb_col_weights.pkl": DSV4_W8A16_APPROVED_CB_COL_WEIGHTS_SHA256,
    }
    for name, expected in expected_raw.items():
        if raw_sha256[name] != expected:
            raise W8A16ExportHandoffError(
                f"approved raw publication changed at {name}"
            )

    layer_path = publication / "layer_config.json"
    selection_path = publication / "selection.json"
    layer_payload = _json_object(layer_path, where="readmitted layer config")
    selection = _json_object(selection_path, where="readmitted selection")
    try:
        raw_assignment = load_assignment(approved_raw / "layer_config.json")
        assignment = load_assignment(layer_path)
    except Exception as exc:
        raise W8A16ExportHandoffError(
            "approved/readmitted assignments are unreadable"
        ) from exc
    assignment_sha256 = assignment_serialization_sha256(assignment)
    if (
        assignment != raw_assignment
        or len(assignment) != DSV4_TOTAL_UNITS
        or assignment_sha256 != DSV4_W8A16_APPROVED_ASSIGNMENT_SHA256
        or sum(
            fmt == "FP8_BLOCK_UE8M0_SOURCE"
            for fmt in assignment.values()
        ) != 120
    ):
        raise W8A16ExportHandoffError(
            "readmitted full qname/format map differs from the approved "
            "33,325-unit assignment"
        )
    metrics = _selection_contract(selection)
    whole = selection["whole_artifact_budget"]
    if whole.get("selection_assignment_sha256") != assignment_sha256:
        raise W8A16ExportHandoffError(
            "readmitted whole-artifact accounting binds another assignment"
        )

    metadata = layer_payload.get("__prismaquant__")
    stamp = (
        metadata.get("aura_cb_reprice")
        if isinstance(metadata, Mapping) else None
    )
    readmission = stamp.get("cpu_replay") if isinstance(stamp, Mapping) else None
    attestation = (
        stamp.get("approved_raw_assignment_attestation")
        if isinstance(stamp, Mapping) else None
    )
    if (
        not isinstance(stamp, Mapping)
        or stamp.get("schema") != CB_ANCHORED_COST_SCHEMA
        or stamp.get("cost_currency") != AURA_CURRENCY
        or stamp.get("budget_bytes") != DSV4_BUDGET_BYTES
        or selection.get("aura_cb_reprice") != stamp
        or selection.get("cost_currency") != AURA_CURRENCY
        or selection.get("feasible") is not True
        or not isinstance(readmission, Mapping)
        or readmission.get("schema") != DSV4_W8A16_READMISSION_SCHEMA
        or readmission.get("measurement_invoked") is not False
        or readmission.get("no_gpu_measurement_or_render") is not True
        or not isinstance(attestation, Mapping)
        or attestation.get("full_qname_format_map_equal") is not True
        or attestation.get("approved_assignment_sha256") != assignment_sha256
        or attestation.get("readmitted_assignment_sha256") != assignment_sha256
        or attestation.get("selection") != metrics
    ):
        raise W8A16ExportHandoffError(
            "publication lacks one matching CPU-only W8A16 readmission proof"
        )
    raw_stamp = readmission.get("approved_raw_publication")
    if (
        not isinstance(raw_stamp, Mapping)
        or Path(str(raw_stamp.get("publication", ""))).resolve(strict=False)
        != approved_raw.resolve(strict=True)
        or raw_stamp.get("assignment_sha256") != assignment_sha256
        or raw_stamp.get("selection") != metrics
        or raw_stamp.get("layer_config_sha256")
        != DSV4_W8A16_APPROVED_LAYER_CONFIG_SHA256
        or raw_stamp.get("selection_sha256")
        != DSV4_W8A16_APPROVED_SELECTION_SHA256
        or raw_stamp.get("cb_col_weights_sha256")
        != DSV4_W8A16_APPROVED_CB_COL_WEIGHTS_SHA256
    ):
        raise W8A16ExportHandoffError(
            "readmission provenance does not bind the exact approved raw "
            "publication"
        )

    runtime = _verify_runtime_contract()
    stamped_runtime = readmission.get("gridbook_runtime_pin")
    if stamped_runtime != {key: runtime[key] for key in (
        "schema", "repository", "commit", "version", "version_is_release",
        "runtime_contract_schema", "required_abi_features",
    )}:
        raise W8A16ExportHandoffError(
            "readmission was produced under a different Gridbook runtime pin"
        )

    from prismaquant.cost_streaming import (
        validate_cached_streamed_model_identity,
    )
    try:
        source_identity = validate_cached_streamed_model_identity(
            source_model_dir,
            source_identity_path,
            require_complete_checkpoint=True,
        )
    except Exception as exc:
        raise W8A16ExportHandoffError(
            "source checkpoint no longer matches its complete content identity"
        ) from exc

    from prismaquant.dspark_source_metadata import (
        discover_dspark_source_overlay_from_artifact,
    )
    try:
        overlay = discover_dspark_source_overlay_from_artifact(source_model_dir)
    except Exception as exc:
        raise W8A16ExportHandoffError(
            "DSpark source-header overlay is invalid"
        ) from exc
    routed_formats = set(assignment.values())
    if overlay is not None:
        routed_formats.update(overlay.construction_units.values())
    pending = sorted(routed_formats & ROUTE_PENDING_PASSTHROUGH_FORMATS)
    if pending:
        raise W8A16ExportHandoffError(
            f"release assignment still uses route-pending formats: {pending}"
        )

    bundle = _verify_bundle(Path(codebook_bundle_path), layer_payload)
    root = (
        Path(repo_root) if repo_root is not None
        else Path(__file__).resolve(strict=True).parent.parent
    )
    frozen = _verify_frozen_export_code(root)
    return {
        "schema": DSV4_W8A16_EXPORT_HANDOFF_SCHEMA,
        "publication": str(publication.resolve(strict=True)),
        "publication_identity_sha256": manifest["identity_sha256"],
        "published_sha256": published_sha256,
        "approved_raw_publication": str(approved_raw.resolve(strict=True)),
        "assignment_sha256": assignment_sha256,
        "unit_count": len(assignment),
        "fp8_block_w8a16_count": 120,
        "selection": metrics,
        "source_checkpoint": {
            "identity_path": str(Path(source_identity_path).resolve(strict=True)),
            "content_sha256": source_identity["content_sha256"],
            "shard_count": len(source_identity["shards"]),
        },
        "codebook_bundle": bundle,
        "gridbook_runtime_pin": runtime,
        "frozen_export_code_sha256": frozen,
        "output_path": str(output.resolve(strict=False)),
        "output_absent": True,
    }


__all__ = [
    "DSV4_W8A16_EXPORT_HANDOFF_SCHEMA",
    "W8A16ExportHandoffError",
    "verify_dsv4_w8a16_export_handoff",
]
