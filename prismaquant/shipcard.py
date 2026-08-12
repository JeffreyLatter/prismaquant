"""The ship record (`exported/shipcard.json`) — a refusal contract.

R13 (`docs/audits/architecture_re-vet_2026-07-30.md`). The build lane and the
serve lane are separated by a physical boundary: `vllm` is not importable in the
build venv, so `run-pipeline.sh` cannot run a ship gate and never should. What it
*can* do is **open a record with required, empty slots** that only the serve lane
can close. `python -m prismaquant.shipcard_cli verify` then exits non-zero until every slot holds a
record whose `model_sha` matches the artifact on disk — which turns "we never ran
the gate" from a silent omission into an explicit refusal.

Base slots (required for every artifact):

| Slot | Filled by |
|---|---|
| `native_export.eager` | `validate_native_export.py --shipcard` (eager arm) |
| `native_export.graph` | `validate_native_export.py --shipcard --no-enforce-eager` |
| `ship_gate` | `validate_quantized_model.py --shipcard` |
| `gold.kl` | `python -m prismaquant.shipcard_cli fill --slot gold.kl --record <full_kl json>` |
| `gold.ppl` | `python -m prismaquant.shipcard_cli fill --slot gold.ppl --record <ppl json>` |

Gridbook CB artifacts open one additional blocking slot,
``perf.matched_budget_parity``.  It can only be filled by the paired DSv4
performance validator after the candidate clears the predeclared served matrix
against the exact eligible container this release displaces under the same
byte budget.  The generic record importer cannot close this slot.

The two `gold.*` slots additionally require `spec_decode_detected: false` on the
record that produced the number — vLLM routes echo+logprobs through the draft
model under `--speculative-config`, so a spec-decode-on gold number is the MTP
head's NLL, not the artifact's (§7.5).

Stdlib only, no torch: the CLI must run anywhere the artifact is reachable.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA = "prismaquant.shipcard/1"

#: Every slot all serving lanes must close before an artifact is shippable.
#: Keep this base tuple stable: non-CB containers do not inherit plugin-only
#: gates merely because a Gridbook lane adds one.
REQUIRED_SLOTS: tuple[str, ...] = (
    "native_export.eager",
    "native_export.graph",
    "ship_gate",
    "gold.kl",
    "gold.ppl",
)

#: Additional slots required only for Gridbook CB artifacts.
CB_REQUIRED_SLOTS: tuple[str, ...] = (
    "perf.matched_budget_parity",
)

#: The vocabulary accepted by :func:`make_record`.  Whether a member is
#: required is artifact-specific and is resolved by :func:`required_slots`.
ALL_SLOTS: tuple[str, ...] = REQUIRED_SLOTS + CB_REQUIRED_SLOTS

#: Slots whose number is invalid if it was produced against a spec-decode serve.
GOLD_SLOTS: frozenset[str] = frozenset({"gold.kl", "gold.ppl"})

SHIPCARD_FILENAME = "shipcard.json"

# The refusal record is intentionally mutated after export as independent
# serve/gold gates close its slots.  A fixed-size JSON file keeps the exporter
# inventory and hard whole-artifact budget exact across those mutations.
# Trailing JSON whitespace is semantically inert and accepted by every reader.
SHIPCARD_RESERVED_BYTES = 256 * 1024
WEIGHT_CONTENT_MANIFEST_SCHEMA = "prismaquant.weight_content_manifest/1"
WEIGHT_STAT_ATTESTATION_SCHEMA = "prismaquant.weight_stat_attestation/1"
CB_PERFORMANCE_RESULT_SCHEMA = "prismaquant.cb_performance_parity/1"
CB_PERFORMANCE_EVIDENCE_SCHEMA = "prismaquant.cb_performance_evidence/1"
DISPLACED_CONTAINER_ELIGIBILITY_SCHEMA = (
    "prismaquant.displaced_container_eligibility/1"
)
CB_PERFORMANCE_TOOL = "validate_cb_performance.py"
CB_PERFORMANCE_TELEMETRY_KINDS = frozenset({
    "routing_per_layer_per_step",
    "expert_occupancy",
    "active_experts",
    "grouped_moe_whole_operator",
})
CB_PERFORMANCE_PHASE_METRICS: Mapping[str, tuple[tuple[str, str], ...]] = {
    "prefill": (("p95_ttft_ms", "baseline/candidate"),),
    "decode": (
        ("p95_tpot_ms", "baseline/candidate"),
        ("p95_itl_ms", "baseline/candidate"),
        ("output_throughput", "candidate/baseline"),
    ),
    "mixed": (
        ("p95_ttft_ms", "baseline/candidate"),
        ("p95_tpot_ms", "baseline/candidate"),
        ("p95_itl_ms", "baseline/candidate"),
        ("p95_e2el_ms", "baseline/candidate"),
        ("request_throughput", "candidate/baseline"),
        ("output_throughput", "candidate/baseline"),
    ),
}


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
def compute_model_sha(model_dir: str | os.PathLike) -> str:
    """Cheap, stable identity for an exported checkpoint.

    Native artifacts retain the legacy config plus per-container-size identity.
    CB artifacts additionally bind canonical ``quant_config.json`` (excluding
    only its self-referential inventory), the exporter-produced exact SHA-256
    manifest of every large safetensors container, and every ``.pqcb`` sidecar
    content hash. Routine verification validates the manifest shape/sizes and
    uses the shipcard's stat attestation instead of rereading ~100 GB.
    """
    root = Path(model_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"model dir does not exist: {root}")
    payload: dict[str, Any] = {}
    cfg = root / "config.json"
    if cfg.is_file():
        payload["config_sha"] = hashlib.sha256(cfg.read_bytes()).hexdigest()
    quant_cfg = root / "quant_config.json"
    raw_quant_cfg: dict[str, Any] | None = None
    canonical_quant_cfg: dict[str, Any] | None = None
    if quant_cfg.is_file():
        raw_quant_cfg = json.loads(quant_cfg.read_text())
        if not isinstance(raw_quant_cfg, dict):
            raise ValueError(
                f"CB quant config must be a JSON object: {quant_cfg}"
            )
        canonical_quant_cfg = dict(raw_quant_cfg)
        provenance = raw_quant_cfg.get("provenance")
        if isinstance(provenance, dict):
            canonical_provenance = dict(provenance)
            canonical_provenance.pop("artifact_inventory", None)
            canonical_quant_cfg["provenance"] = canonical_provenance
    weights = {
        p.name: p.stat().st_size
        for p in sorted(root.glob("*.safetensors"))
    }
    if not weights:
        weights = {
            p.name: p.stat().st_size
            for p in sorted(root.glob("*.gguf"))
        }
    payload["weights"] = weights
    if raw_quant_cfg is not None and canonical_quant_cfg is not None:
        manifest = (raw_quant_cfg.get("provenance") or {}).get(
            "weight_content_manifest"
        ) if isinstance(raw_quant_cfg.get("provenance"), dict) else None
        if manifest is not None:
            _validate_weight_content_manifest(manifest, weights, where=quant_cfg)
        payload["quant_config_sha"] = hashlib.sha256(
            _canonical_json(canonical_quant_cfg).encode("utf-8")
        ).hexdigest()
    codebooks = {
        p.name: {
            "bytes": p.stat().st_size,
            "sha256": _file_content_sha256(p),
        }
        for p in sorted(root.glob("*.pqcb"))
    }
    if codebooks:
        payload["codebooks"] = codebooks
    if raw_quant_cfg is not None:
        excluded = {SHIPCARD_FILENAME, "quant_config.json"}
        auxiliary = {
            path.relative_to(root).as_posix(): {
                "bytes": int(path.stat().st_size),
                "sha256": _file_content_sha256(path),
            }
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and path.name not in excluded
            and path.suffix not in {".safetensors", ".pqcb"}
        }
        if auxiliary:
            payload["auxiliary_files"] = auxiliary
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def artifact_bytes(model_dir: str | os.PathLike) -> int:
    """Exact weight + codebook payload footprint (what the box must hold)."""
    root = Path(model_dir)
    total = 0
    for pattern in ("*.safetensors", "*.gguf", "*.pqcb"):
        for p in root.glob(pattern):
            total += p.stat().st_size
    return int(total)


def _file_content_sha256(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_weight_content_manifest(model_dir: str | os.PathLike) -> dict[str, Any]:
    """Hash each finished safetensors container once at the export boundary."""
    root = Path(model_dir)
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors weights to attest under {root}")
    return {
        "schema": WEIGHT_CONTENT_MANIFEST_SCHEMA,
        "algorithm": "sha256",
        "files": {
            path.name: {
                "bytes": int(path.stat().st_size),
                "sha256": _file_content_sha256(path),
            }
            for path in files
        },
    }


def _validate_weight_content_manifest(
    manifest: object,
    weights: Mapping[str, int],
    *,
    where: str | os.PathLike,
) -> None:
    if not isinstance(manifest, Mapping) or manifest.get(
        "schema"
    ) != WEIGHT_CONTENT_MANIFEST_SCHEMA or manifest.get("algorithm") != "sha256":
        raise ValueError(f"invalid weight content manifest in {where}")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(weights):
        raise ValueError(
            f"weight content manifest file set differs from weights in {where}"
        )
    for name, expected_bytes in weights.items():
        row = files.get(name)
        if not isinstance(row, Mapping) or row.get("bytes") != expected_bytes or not (
            isinstance(row.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256")))
        ):
            raise ValueError(
                f"invalid weight content manifest entry for {name!r} in {where}"
            )


def weight_stat_attestation(model_dir: str | os.PathLike) -> dict[str, Any]:
    """Cheap post-hash mutation detector for the large weight containers."""
    root = Path(model_dir)
    files = sorted(root.glob("*.safetensors"))
    return {
        "schema": WEIGHT_STAT_ATTESTATION_SCHEMA,
        "files": {
            path.name: {
                "bytes": int((info := path.stat()).st_size),
                "mtime_ns": int(info.st_mtime_ns),
                "ctime_ns": int(info.st_ctime_ns),
            }
            for path in files
        },
    }


def assert_weight_stat_attestation(
    card: Mapping[str, Any],
    model_dir: str | os.PathLike,
) -> None:
    """Fail if a weight changed after the card bound its exact content claim."""
    expected = card.get("weight_stat_attestation")
    if expected is None:
        return  # Backward-compatible verification of historical cards.
    if not isinstance(expected, Mapping) or expected.get(
        "schema"
    ) != WEIGHT_STAT_ATTESTATION_SCHEMA:
        raise ValueError("shipcard carries an invalid weight stat attestation")
    observed = weight_stat_attestation(model_dir)
    if observed != expected:
        raise ValueError(
            "weight file stats changed after export; refusing cached content "
            "identity (run shipcard_cli reattest after a legitimate "
            "cross-filesystem copy)"
        )


def reattest_weight_stats(
    shipcard_path: str | os.PathLike,
    model_dir: str | os.PathLike | None = None,
) -> dict[str, Any]:
    """Full-hash a copied CB artifact, then refresh only its cheap stat cache."""
    path = Path(shipcard_path)
    root = Path(model_dir) if model_dir is not None else path.resolve().parent
    card = load_shipcard(path)
    quant_path = root / "quant_config.json"
    if not quant_path.is_file():
        raise ValueError("weight re-attestation requires a CB quant_config.json")
    quant_config = json.loads(quant_path.read_text(encoding="utf-8"))
    provenance = quant_config.get("provenance")
    expected = provenance.get("weight_content_manifest") if isinstance(
        provenance, Mapping
    ) else None
    if expected is None:
        raise ValueError("CB artifact has no immutable weight content manifest")
    observed = build_weight_content_manifest(root)
    if observed != expected:
        raise ValueError(
            "weight content differs from the immutable export manifest; "
            "refusing to refresh the stat attestation"
        )
    if compute_model_sha(root) != card.get("model_sha"):
        raise ValueError("copied artifact model_sha differs from the shipcard")
    card["weight_stat_attestation"] = weight_stat_attestation(root)
    card["updated"] = _now()
    write_shipcard(path, card)
    return card


def file_sha256(path: str | os.PathLike) -> str | None:
    try:
        return _file_content_sha256(path)
    except Exception:
        return None


def git_provenance(repo: str | os.PathLike | None = None) -> dict[str, Any]:
    """``{commit, dirty}`` for the tree that produced this record.

    Read-only Docker exports may have the source mounted without usable
    worktree metadata.  In that case the launch boundary can pass the same
    exact commit override used by producer identities plus an independently
    preflighted dirty bit.  When git is available, both overrides are checked
    against it rather than silently replacing contradictory observations.
    """
    root = Path(repo) if repo is not None else Path(__file__).resolve().parents[1]

    def _run(cmd: list[str]) -> str | None:
        try:
            return subprocess.run(
                cmd, cwd=root, check=True, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            ).stdout.strip()
        except Exception:
            return None

    commit_override = str(os.environ.get(
        "PRISMAQUANT_IDENTITY_GIT_COMMIT", ""
    )).strip().lower()
    if commit_override and re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit_override
    ) is None:
        raise ValueError(
            "PRISMAQUANT_IDENTITY_GIT_COMMIT must be a full 40- or "
            "64-character hexadecimal commit id"
        )
    dirty_override_raw = str(os.environ.get(
        "PRISMAQUANT_IDENTITY_GIT_DIRTY", ""
    )).strip().lower()
    dirty_values = {
        "0": False, "false": False, "no": False,
        "1": True, "true": True, "yes": True,
    }
    if dirty_override_raw and dirty_override_raw not in dirty_values:
        raise ValueError(
            "PRISMAQUANT_IDENTITY_GIT_DIRTY must be one of "
            "0/1/false/true/no/yes"
        )

    observed_commit = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--short"])
    observed_dirty = None if status is None else bool(status)
    if (
        commit_override
        and observed_commit is not None
        and observed_commit.lower() != commit_override
    ):
        raise ValueError(
            "PRISMAQUANT_IDENTITY_GIT_COMMIT contradicts the mounted "
            f"worktree HEAD {observed_commit}"
        )
    dirty_override = (
        dirty_values[dirty_override_raw] if dirty_override_raw else None
    )
    if (
        dirty_override is not None
        and observed_dirty is not None
        and observed_dirty != dirty_override
    ):
        raise ValueError(
            "PRISMAQUANT_IDENTITY_GIT_DIRTY contradicts the mounted "
            f"worktree dirty={observed_dirty}"
        )
    return {
        "commit": commit_override or observed_commit,
        "dirty": (
            dirty_override if dirty_override is not None else observed_dirty
        ),
    }


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


# ---------------------------------------------------------------------------
# Build lane
# ---------------------------------------------------------------------------
def build_shipcard(
    model_dir: str | os.PathLike,
    *,
    build: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Open a fresh record: build-lane facts filled, every slot empty."""
    root = Path(model_dir)
    from prismaquant.export_output_safety import directory_publication_target

    build_payload = dict(build or {})
    slots = ALL_SLOTS if build_payload.get("quant_method") == "gridbook" else REQUIRED_SLOTS
    card = {
        "schema": SCHEMA,
        "created": _now(),
        "model_dir": str(directory_publication_target(root)),
        "model_sha": compute_model_sha(root),
        "artifact_bytes": artifact_bytes(root),
        "reserved_file_bytes": SHIPCARD_RESERVED_BYTES,
        "build": build_payload,
        "slots": {slot: None for slot in slots},
    }
    quant_path = root / "quant_config.json"
    if quant_path.is_file():
        try:
            quant_config = json.loads(quant_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            quant_config = None
        provenance = quant_config.get("provenance") if isinstance(
            quant_config, Mapping
        ) else None
        if isinstance(provenance, Mapping) and provenance.get(
            "weight_content_manifest"
        ) is not None:
            card["weight_stat_attestation"] = weight_stat_attestation(root)
    return card


def open_cb_export_shipcard(
    model_dir: str | os.PathLike,
    quant_config: Mapping[str, Any],
    *,
    source_model: str | os.PathLike,
    layer_config_path: str | os.PathLike,
    exporter: str,
    weight_content_manifest: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Write the preliminary CB config and open its refusal record.

    CB inventory finalization must run *after* this helper.  The preliminary
    config lets :func:`compute_model_sha` bind all value-bearing CB metadata;
    the subsequently written shipcard is then part of the final recursive
    artifact inventory and of any hard whole-artifact budget check.  Inventory
    finalization changes only the field excluded from CB identity, so the
    freshly opened card remains valid after the fixed-point write.
    """
    root = Path(model_dir)
    provenance = quant_config.setdefault("provenance", {})
    if not isinstance(provenance, dict):
        raise TypeError("CB quant config provenance must be an object")
    if "weight_content_manifest" in provenance:
        raise ValueError(
            "CB quant config already carries a weight content manifest before export finalization"
        )
    if weight_content_manifest is None:
        print(
            f"[shipcard] hashing exact final weight content under {root}",
            flush=True,
        )
        manifest = build_weight_content_manifest(root)
    else:
        weights = {
            path.name: int(path.stat().st_size)
            for path in sorted(root.glob("*.safetensors"))
        }
        if not weights:
            raise FileNotFoundError(
                f"no safetensors weights to attest under {root}"
            )
        _validate_weight_content_manifest(
            weight_content_manifest, weights, where=root
        )
        # Detach the value placed into the mutable quant-config payload from
        # any mapping the caller retains.
        manifest = json.loads(json.dumps(weight_content_manifest))
        print(
            f"[shipcard] binding {len(weights)} in-stream weight SHA-256 "
            f"digest(s) under {root}",
            flush=True,
        )
    provenance["weight_content_manifest"] = manifest
    config_path = root / "quant_config.json"
    config_path.write_text(json.dumps(
        dict(quant_config), indent=2, sort_keys=True
    ))
    build = {
        "git": git_provenance(),
        "exporter": str(exporter),
        "quant_method": "gridbook",
        "source_model": str(source_model),
        "layer_config": str(layer_config_path),
        "layer_config_sha": file_sha256(layer_config_path),
        "achieved_bpp": allocator_achieved_bpp(layer_config_path),
    }
    card = build_shipcard(root, build=build)
    path = write_shipcard(root / SHIPCARD_FILENAME, card)
    return path, card


def write_shipcard(path: str | os.PathLike, card: Mapping[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(card, indent=2, default=str) + "\n").encode("utf-8")
    reserved = card.get("reserved_file_bytes")
    if reserved is not None:
        if isinstance(reserved, bool) or not isinstance(reserved, int) or reserved <= 0:
            raise ValueError("shipcard reserved_file_bytes must be a positive integer")
        if len(encoded) > reserved:
            raise ValueError(
                f"shipcard needs {len(encoded)} bytes but its fixed reservation "
                f"is {reserved} bytes; refusing to invalidate the artifact inventory"
            )
        encoded += b" " * (reserved - len(encoded))
    out.write_bytes(encoded)
    return out


def load_shipcard(path: str | os.PathLike) -> dict[str, Any]:
    card = json.loads(Path(path).read_text())
    if not isinstance(card, dict) or "slots" not in card:
        raise ValueError(f"not a shipcard: {path}")
    return card


# ---------------------------------------------------------------------------
# Serve lane
# ---------------------------------------------------------------------------
def make_record(
    *,
    slot: str,
    tool: str,
    passed: bool,
    model_sha: str | None,
    metrics: Mapping[str, Any] | None = None,
    detail: str = "",
    spec_decode_detected: bool | None = None,
    serve_fingerprint: str | None = None,
    git_commit: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One serve-lane verdict block."""
    if slot not in ALL_SLOTS:
        raise KeyError(
            f"unknown shipcard slot {slot!r}; known: {list(ALL_SLOTS)}")
    record: dict[str, Any] = {
        "slot": slot,
        "tool": tool,
        "filled_at": _now(),
        "passed": bool(passed),
        "model_sha": model_sha,
        "spec_decode_detected": spec_decode_detected,
        "serve_fingerprint": serve_fingerprint,
        "git_commit": git_commit,
        "detail": detail,
        "metrics": dict(metrics or {}),
    }
    if extra:
        record.update(dict(extra))
    return record


def fill_slot(
    path: str | os.PathLike,
    slot: str,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Append a verdict block to `slot` in the shipcard at `path` (in place)."""
    card = load_shipcard(path)
    card_model_dir = Path(path).resolve().parent
    if card.get("weight_stat_attestation") is not None:
        assert_weight_stat_attestation(card, card_model_dir)
    if slot not in card["slots"]:
        raise KeyError(
            f"shipcard {path} has no slot {slot!r}; known: "
            f"{sorted(card['slots'])}")
    card["slots"][slot] = dict(record)
    card["updated"] = _now()
    write_shipcard(path, card)
    return card


def fill_if_requested(
    path: str | os.PathLike | None,
    slot: str,
    record: Mapping[str, Any],
) -> None:
    """`fill_slot` when a `--shipcard` path was supplied; loud no-op otherwise.

    Serve-lane tools must never fail because of the record — the measurement is
    the point and the refusal lives in `verify`. Failures print and are ignored.
    """
    if not path:
        return
    try:
        fill_slot(path, slot, record)
        print(f"[shipcard] filled {slot} in {path}", flush=True)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[shipcard] WARN could not fill {slot} in {path}: {exc!r}",
              flush=True)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify(
    card: Mapping[str, Any],
    *,
    model_dir: str | os.PathLike | None = None,
    required: Iterable[str] | None = None,
) -> list[str]:
    """Return the list of reasons this artifact is not shippable (empty = OK)."""
    problems: list[str] = []
    expected_sha = card.get("model_sha")

    if model_dir is not None:
        try:
            assert_weight_stat_attestation(card, model_dir)
            on_disk = compute_model_sha(model_dir)
        except Exception as exc:
            problems.append(
                "artifact changed since the shipcard was opened or its "
                f"model_sha is not computable: {exc!r}"
            )
            on_disk = None
        if on_disk is not None and expected_sha and on_disk != expected_sha:
            problems.append(
                f"artifact changed since the shipcard was opened: "
                f"on-disk model_sha {on_disk[:12]} != shipcard "
                f"{str(expected_sha)[:12]} — re-run the serve lane")
            expected_sha = on_disk

    slots = card.get("slots") or {}
    is_gridbook_cb = _is_gridbook_card(card, model_dir=model_dir)
    if is_gridbook_cb and card.get("reserved_file_bytes") != SHIPCARD_RESERVED_BYTES:
        problems.append(
            "shipcard reserved_file_bytes is not the fixed "
            f"{SHIPCARD_RESERVED_BYTES}-byte release reservation"
        )
    if is_gridbook_cb and not isinstance(
        card.get("weight_stat_attestation"), Mapping
    ):
        problems.append(
            "Gridbook artifact lacks the required weight-stat attestation"
        )
    if required is None:
        required = required_slots(card, model_dir=model_dir)
    for slot in required:
        record = slots.get(slot)
        if not record:
            problems.append(f"{slot}: UNFILLED")
            continue
        if not isinstance(record, dict):
            problems.append(f"{slot}: malformed record ({type(record).__name__})")
            continue
        if record.get("slot") != slot:
            problems.append(
                f"{slot}: record declares slot {record.get('slot')!r}"
            )
        got = record.get("model_sha")
        if not got:
            problems.append(f"{slot}: record carries no model_sha")
        elif expected_sha and got != expected_sha:
            problems.append(
                f"{slot}: record model_sha {str(got)[:12]} != artifact "
                f"{str(expected_sha)[:12]} (record belongs to another build)")
        if record.get("passed") is not True:
            problems.append(
                f"{slot}: FAILED — {record.get('detail') or 'no detail'}")
        if is_gridbook_cb and slot in {
            "native_export.eager", "native_export.graph"
        }:
            problems.extend(_verify_gridbook_native_record(slot, record))
        if is_gridbook_cb and slot == "perf.matched_budget_parity":
            problems.extend(_verify_gridbook_performance_record(
                slot,
                record,
                model_dir=model_dir,
            ))
        if is_gridbook_cb and slot == "ship_gate":
            problems.extend(_verify_ship_gate_record(slot, record))
        if is_gridbook_cb and slot in GOLD_SLOTS:
            problems.extend(_verify_gold_record(slot, record))
        if slot in GOLD_SLOTS:
            spec = record.get("spec_decode_detected")
            if spec is None:
                problems.append(
                    f"{slot}: spec_decode_detected is unknown — a gold number "
                    "measured against a spec-decode serve is the draft model's "
                    "NLL, so 'unknown' is not acceptable (§7.5)")
            elif spec:
                problems.append(
                    f"{slot}: spec_decode_detected is TRUE — this is draft-model "
                    "NLL, not the artifact's; re-measure on a no-spec serve")
    return problems


def _verify_gold_record(slot: str, record: Mapping[str, Any]) -> list[str]:
    """Require a slot-specific finite gold metric plus measurement identity."""
    problems: list[str] = []
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        return [f"{slot}: missing structured gold metrics"]
    allowed = (
        ("kl_confident_mean", "kl_mean")
        if slot == "gold.kl"
        else ("ppl", "mean_nll")
    )
    finite = [
        key for key in allowed
        if key in metrics
        and not isinstance(metrics[key], bool)
        and isinstance(metrics[key], (int, float))
        and math.isfinite(float(metrics[key]))
        and float(metrics[key]) >= 0
    ]
    if not finite:
        problems.append(
            f"{slot}: carries no finite non-negative slot-specific metric "
            f"from {allowed}"
        )
    fingerprint = record.get("serve_fingerprint")
    if not isinstance(fingerprint, str) or re.fullmatch(
        r"[0-9a-f]{64}", fingerprint
    ) is None:
        problems.append(f"{slot}: missing exact serve fingerprint")
    commit = record.get("git_commit")
    if not isinstance(commit, str) or re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit
    ) is None:
        problems.append(f"{slot}: missing full producer git commit")
    tool = record.get("tool")
    if not isinstance(tool, str) or not tool:
        problems.append(f"{slot}: missing measurement tool identity")
    if slot == "gold.kl":
        if not any(
            isinstance(metrics.get(key), int)
            and not isinstance(metrics.get(key), bool)
            and metrics.get(key, 0) > 0
            for key in ("n_positions", "n_samples")
        ):
            problems.append(f"{slot}: missing positive KL sample/position count")
    elif not isinstance(metrics.get("n_tokens_scored"), int) or isinstance(
        metrics.get("n_tokens_scored"), bool
    ) or metrics.get("n_tokens_scored", 0) <= 0:
        problems.append(f"{slot}: missing positive scored-token count")
    return problems


def _verify_ship_gate_record(
    slot: str,
    record: Mapping[str, Any],
) -> list[str]:
    """Replay the fixed catastrophic-quality thresholds and check ledger."""
    problems: list[str] = []
    if record.get("tool") != "validate_quantized_model.py":
        problems.append(f"{slot}: not filled by validate_quantized_model.py")
    commit = record.get("git_commit")
    if not isinstance(commit, str) or re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit
    ) is None:
        problems.append(f"{slot}: missing full producer git commit")
    metrics = record.get("metrics")
    thresholds = record.get("thresholds")
    expected_checks = {
        "serve_ready", "generation_sanity", "perplexity", "mtp_acceptance"
    }
    if not isinstance(metrics, Mapping) or set(metrics) != expected_checks:
        return problems + [f"{slot}: validation check ledger is incomplete"]
    for name, row in metrics.items():
        if not isinstance(row, Mapping) or row.get("passed") is not True:
            problems.append(f"{slot}: check {name} is not a structured pass")
    expected_thresholds = {
        "max_ppl": 25.0,
        "max_mean_nll": 3.0,
        "max_p99_nll": 6.0,
        "min_gen_len": 30,
        "min_mtp_accept_p0": 0.60,
    }
    if not isinstance(thresholds, Mapping):
        problems.append(f"{slot}: missing threshold contract")
    else:
        for key, expected in expected_thresholds.items():
            value = thresholds.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) != float(expected)
            ):
                problems.append(
                    f"{slot}: threshold {key}={value!r}, expected {expected!r}"
                )
    perplexity = metrics.get("perplexity")
    if isinstance(perplexity, Mapping):
        numeric = {
            "perplexity": ("max_ppl", 25.0),
            "mean_nll_per_tok": ("max_mean_nll", 3.0),
            "max_nll_per_tok": ("max_p99_nll", 6.0),
        }
        for key, (_threshold_key, limit) in numeric.items():
            value = perplexity.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
                or float(value) > limit
            ):
                problems.append(
                    f"{slot}: perplexity metric {key} does not clear {limit}"
                )
        tokens = perplexity.get("n_tokens")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            problems.append(f"{slot}: perplexity check scored no tokens")
        if perplexity.get("spec_decode_detected") is True:
            problems.append(f"{slot}: perplexity ran under speculative decode")
        elif perplexity.get("spec_decode_detected") is not False:
            problems.append(
                f"{slot}: perplexity speculative-decode state is unknown"
            )
    else:
        problems.append(f"{slot}: missing structured perplexity evidence")
    return problems


def _verify_gridbook_native_record(
    slot: str,
    record: Mapping[str, Any],
) -> list[str]:
    """Structural proof that a CB native slot came from the exact validator."""
    problems: list[str] = []
    arm = slot.rsplit(".", 1)[-1]
    if record.get("tool") != "validate_cb_endpoint.py":
        problems.append(f"{slot}: not filled by validate_cb_endpoint.py")
    fingerprint = record.get("serve_fingerprint")
    if not isinstance(fingerprint, str) or not re.fullmatch(
        r"[0-9a-f]{64}", fingerprint
    ):
        problems.append(f"{slot}: missing exact serve fingerprint")
    if record.get("spec_decode_detected") is not False:
        problems.append(f"{slot}: speculative-decode state is not false")

    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        return problems + [f"{slot}: missing structured CB endpoint metrics"]
    expected = {
        "arm": arm,
        "enforce_eager": arm == "eager",
        "quantization": "gridbook",
        "kv_cache_dtype": "fp8",
        "tensor_parallel_size": 1,
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            problems.append(
                f"{slot}: endpoint metric {key}={metrics.get(key)!r}, expected {value!r}"
            )
    try:
        pin = json.loads((
            Path(__file__).resolve().parent
            / "gridbook_runtime"
            / "gridbook_runtime_pin.json"
        ).read_text(encoding="utf-8"))
    except Exception as exc:
        problems.append(f"{slot}: tracked Gridbook pin unreadable: {exc}")
    else:
        if metrics.get("gridbook_runtime_commit") != pin.get("commit"):
            problems.append(f"{slot}: Gridbook runtime commit is not the tracked pin")
        if metrics.get("gridbook_runtime_version") != pin.get("version"):
            problems.append(f"{slot}: Gridbook runtime version is not the tracked pin")

    contract = metrics.get("endpoint_contract")
    if not isinstance(contract, Mapping):
        problems.append(f"{slot}: missing canonical endpoint contract")
    else:
        try:
            # Runtime import avoids a module cycle: the endpoint writer imports
            # shipcard to construct records, while verify only needs this
            # stdlib structural replay after both modules are initialized.
            from .validate_cb_endpoint import validate_endpoint_contract_record

            validate_endpoint_contract_record(
                contract,
                arm=arm,
                model_sha=record.get("model_sha"),
                serve_fingerprint=fingerprint,
            )
        except Exception as exc:
            problems.append(f"{slot}: invalid endpoint contract — {exc}")
        manifest_binding = contract.get("serve_manifest")
        if not isinstance(manifest_binding, Mapping) or metrics.get(
            "serve_manifest_sha256"
        ) != manifest_binding.get("sha256"):
            problems.append(
                f"{slot}: serve-manifest digest differs from endpoint contract"
            )
        stack = contract.get("stack")
        if isinstance(stack, Mapping):
            for metric_key, stack_key in (
                ("gridbook_runtime_commit", "gridbook_runtime_commit"),
                ("gridbook_runtime_version", "gridbook_runtime_version"),
                ("vllm_version", "vllm_version"),
                ("vllm_commit", "vllm_commit"),
            ):
                if metrics.get(metric_key) != stack.get(stack_key):
                    problems.append(
                        f"{slot}: {metric_key} differs from endpoint contract"
                    )
        smoke = contract.get("endpoint_smoke")
        if isinstance(smoke, Mapping):
            for key, value in smoke.items():
                if metrics.get(key) != value:
                    problems.append(
                        f"{slot}: endpoint metric {key} differs from endpoint contract"
                    )
    graph = metrics.get("cuda_graph")
    if arm == "graph":
        if not isinstance(graph, Mapping) or not isinstance(
            graph.get("serve_log_sha256"), str
        ) or not str(graph.get("capture_marker", "")).startswith(
            "Graph capturing finished"
        ):
            problems.append(f"{slot}: missing positive CUDA-graph capture evidence")
        elif isinstance(contract, Mapping) and (
            contract.get("cuda_graph")
            != {
                "capture_marker": graph.get("capture_marker"),
                "serve_log_sha256": graph.get("serve_log_sha256"),
            }
        ):
            problems.append(
                f"{slot}: CUDA-graph evidence differs from endpoint contract"
            )
    elif graph is not None:
        problems.append(f"{slot}: eager receipt unexpectedly carries graph evidence")
    return problems


def _verify_gridbook_performance_record(
    slot: str,
    record: Mapping[str, Any],
    *,
    model_dir: str | os.PathLike | None = None,
) -> list[str]:
    """Replay the self-contained structure of the blocking CB parity proof.

    The performance validator reads the large paired report/telemetry corpus
    once.  The fixed-size shipcard persists their unique SHA-256 identities,
    the exact matrix coverage, the independently eligible displaced
    container, and the candidate inventory.  Publication replays every
    internal binding here instead of trusting a generic ``passed: true`` row.
    """
    problems: list[str] = []

    def problem(detail: str) -> None:
        problems.append(f"{slot}: {detail}")

    def sha(value: object) -> bool:
        return isinstance(value, str) and re.fullmatch(
            r"[0-9a-f]{64}", value
        ) is not None

    def positive_int(value: object) -> bool:
        return (
            not isinstance(value, bool)
            and isinstance(value, int)
            and value > 0
        )

    def finite(value: object) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None

    if record.get("slot") != slot:
        problem("record slot does not identify the parity gate")
    if record.get("tool") != CB_PERFORMANCE_TOOL:
        problem(f"not filled by {CB_PERFORMANCE_TOOL}")
    if record.get("spec_decode_detected") is not None:
        problem("performance receipt unexpectedly carries speculative-decode state")
    if record.get("serve_fingerprint") is not None:
        problem("performance receipt unexpectedly carries one-arm serve fingerprint")
    commit = record.get("git_commit")
    if not isinstance(commit, str) or re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit
    ) is None:
        problem("missing full producer git commit")

    metrics = record.get("metrics")
    evidence = record.get("evidence")
    if not isinstance(metrics, Mapping):
        return problems + [f"{slot}: missing structured parity metrics"]
    if not isinstance(evidence, Mapping):
        return problems + [f"{slot}: missing digest-bound parity evidence"]
    if metrics.get("schema") != CB_PERFORMANCE_RESULT_SCHEMA:
        problem("unsupported parity metrics schema")
    if evidence.get("schema") != CB_PERFORMANCE_EVIDENCE_SCHEMA:
        problem("unsupported parity evidence schema")
    if metrics.get("prismaquant_runtime_commit") != commit:
        problem("PrismaQuant validator commit differs from record provenance")

    try:
        pin = json.loads((
            Path(__file__).resolve().parent
            / "gridbook_runtime"
            / "gridbook_runtime_pin.json"
        ).read_text(encoding="utf-8"))
    except Exception as exc:
        problem(f"tracked Gridbook pin unreadable: {exc}")
        pin = None
    if isinstance(pin, Mapping):
        if metrics.get("gridbook_runtime_commit") != pin.get("commit"):
            problem("Gridbook runtime commit is not the tracked pin")
        if metrics.get("gridbook_runtime_version") != pin.get("version"):
            problem("Gridbook runtime version is not the tracked pin")

    digest_fields = (
        "candidate_inventory_sha256",
        "matrix_digest",
        "displaced_container_eligibility_sha256",
        "displaced_container_model_sha",
        "displaced_container_inventory_sha256",
        "displaced_container_assignment_sha256",
        "native_baseline_feasibility_sha256",
    )
    for key in digest_fields:
        if not sha(metrics.get(key)):
            problem(f"metric {key} is not a lowercase SHA-256")
    if not sha(evidence.get("comparison_manifest_sha256")):
        problem("comparison manifest is not digest-bound")

    budget = metrics.get("byte_budget")
    candidate_bytes = metrics.get("candidate_artifact_bytes")
    displaced_bytes = metrics.get("displaced_container_artifact_bytes")
    if not positive_int(budget):
        problem("byte budget is not a positive integer")
    if not positive_int(candidate_bytes) or (
        positive_int(budget) and candidate_bytes > budget
    ):
        problem("candidate artifact is not within the exact byte budget")
    if not positive_int(displaced_bytes) or (
        positive_int(budget) and displaced_bytes > budget
    ):
        problem("displaced artifact is not within the exact byte budget")

    parity_floor = finite(metrics.get("parity_floor"))
    minimum = finite(metrics.get("min_conservative_ratio"))
    tolerance = finite(metrics.get("predeclared_tolerance"))
    predeclared_at = metrics.get("predeclared_at")
    if (
        tolerance is None
        or not 0 <= tolerance <= 0.05
        or parity_floor is None
        or not math.isclose(parity_floor, 1.0 - tolerance, abs_tol=1e-12)
    ):
        problem("parity floor does not equal one minus predeclared tolerance")
    if tolerance and (
        not isinstance(metrics.get("tolerance_rationale"), str)
        or not str(metrics.get("tolerance_rationale")).strip()
    ):
        problem("nonzero performance tolerance has no rationale")
    if not isinstance(predeclared_at, str) or not predeclared_at:
        problem("performance comparison has no predeclaration timestamp")
    if parity_floor is None or not 0.95 <= parity_floor <= 1.0:
        problem("parity floor is outside the predeclared 0-5% tolerance range")
    if minimum is None or minimum <= 0 or (
        parity_floor is not None and minimum < parity_floor
    ):
        problem("conservative parity ratio does not clear the declared floor")

    coverage = metrics.get("coverage")
    cell_count = metrics.get("cell_count")
    cell_ids: list[str] = []
    expected_verdict_tuples: set[tuple[object, ...]] | None = None
    if not isinstance(coverage, Mapping):
        problem("missing exact comparison-matrix coverage")
    else:
        concurrencies = coverage.get("concurrencies")
        shipped_max = coverage.get("shipped_max_concurrency")
        valid_concurrencies = (
            isinstance(concurrencies, list)
            and all(positive_int(value) for value in concurrencies)
            and concurrencies == sorted(set(concurrencies))
            and positive_int(shipped_max)
            and shipped_max >= 8
            and shipped_max == max(concurrencies)
            and set(concurrencies) == {1, 2, 4, 8, shipped_max}
        )
        expected_count = 6 * len(concurrencies) + 2 if valid_concurrencies else None
        expected_coverage = (
            coverage.get("phases") == ["prefill", "decode", "mixed"]
            and coverage.get("chunked_prefill") == [False, True]
            and coverage.get("decode_modes") == ["plain", "shipped"]
            and coverage.get("nonzero_input_distribution") is True
            and positive_int(shipped_max)
            and valid_concurrencies
            and coverage.get("configuration_tuple_count") == expected_count
        )
        if not expected_coverage:
            problem("comparison matrix does not equal the release Cartesian product")
        if valid_concurrencies:
            expected_verdict_tuples = set()
            for concurrency in concurrencies:
                for chunked in (False, True):
                    expected_verdict_tuples.add(
                        ("prefill", concurrency, chunked, None)
                    )
                    expected_verdict_tuples.add(
                        ("mixed", concurrency, chunked, None)
                    )
                    expected_verdict_tuples.add(
                        ("decode", concurrency, chunked, "shipped")
                    )
            for chunked in (False, True):
                expected_verdict_tuples.add(("decode", 1, chunked, "plain"))
        if (
            not positive_int(cell_count)
            or cell_count != coverage.get("configuration_tuple_count")
        ):
            problem("cell count differs from exact matrix coverage")

    verdicts = metrics.get("cell_verdicts")
    verdict_minima: list[float] = []
    verdict_tuples: set[tuple[object, ...]] = set()
    verdict_ids: list[str] = []
    if not isinstance(verdicts, list) or not positive_int(cell_count) or len(
        verdicts if isinstance(verdicts, list) else []
    ) != cell_count:
        problem("cell verdict ledger does not exactly cover the matrix")
    else:
        for index, verdict in enumerate(verdicts):
            if not isinstance(verdict, Mapping):
                problem(f"cell verdict {index} is malformed")
                continue
            phase = verdict.get("phase")
            concurrency = verdict.get("concurrency")
            chunked = verdict.get("chunked_prefill")
            decode_mode = verdict.get("decode_mode")
            verdict_id = verdict.get("id")
            rows = verdict.get("metrics")
            row_minima: list[float] = []
            if not isinstance(verdict_id, str) or not verdict_id:
                problem(f"cell verdict {index} has no id")
            else:
                verdict_ids.append(verdict_id)
            if (
                phase not in {"prefill", "decode", "mixed"}
                or not positive_int(concurrency)
                or not isinstance(chunked, bool)
                or (phase == "decode" and decode_mode not in {"plain", "shipped"})
                or (phase != "decode" and decode_mode is not None)
            ):
                problem(f"cell verdict {index} has an invalid configuration")
            configuration_valid = (
                isinstance(phase, str)
                and phase in {"prefill", "decode", "mixed"}
                and positive_int(concurrency)
                and isinstance(chunked, bool)
                and (
                    (
                        phase == "decode"
                        and isinstance(decode_mode, str)
                        and decode_mode in {"plain", "shipped"}
                    )
                    or (phase != "decode" and decode_mode is None)
                )
            )
            if configuration_valid:
                verdict_tuples.add((phase, concurrency, chunked, decode_mode))
            if not isinstance(rows, list) or not rows:
                problem(f"cell verdict {index} has no metric ratios")
                continue
            expected_metrics = CB_PERFORMANCE_PHASE_METRICS.get(str(phase), ())
            observed_metrics = [
                (row.get("metric"), row.get("direction"))
                if isinstance(row, Mapping)
                else (None, None)
                for row in rows
            ]
            if observed_metrics != list(expected_metrics):
                problem(
                    f"cell verdict {index} metric names or directions differ "
                    "from the phase contract"
                )
            for row_index, row in enumerate(rows):
                ratios = row.get("paired_ratios") if isinstance(row, Mapping) else None
                if not isinstance(ratios, list) or len(ratios) < 3:
                    problem(f"cell verdict {index} metric {row_index} lacks ratios")
                    continue
                parsed = [finite(value) for value in ratios]
                if any(value is None or value <= 0 for value in parsed):
                    problem(f"cell verdict {index} metric {row_index} has invalid ratios")
                    continue
                numeric = [float(value) for value in parsed if value is not None]
                ordered = sorted(numeric)
                position = (len(ordered) - 1) * 0.05
                lower = math.floor(position)
                upper = math.ceil(position)
                p05 = ordered[lower] if lower == upper else (
                    ordered[lower] * (1 - (position - lower))
                    + ordered[upper] * (position - lower)
                )
                declared_p05 = finite(row.get("conservative_p05_ratio"))
                declared_median = finite(row.get("median_ratio"))
                if (
                    declared_p05 is None
                    or declared_median is None
                    or not math.isclose(
                        p05, declared_p05, rel_tol=1e-12, abs_tol=1e-12
                    )
                    or not math.isclose(
                        statistics.median(numeric),
                        declared_median,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                ):
                    problem(f"cell verdict {index} metric {row_index} statistics differ")
                row_minima.append(p05)
            if row_minima:
                calculated = min(row_minima)
                declared_minimum = finite(verdict.get("min_conservative_ratio"))
                if (
                    declared_minimum is None
                    or not math.isclose(
                        calculated,
                        declared_minimum,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                    or verdict.get("passed") is not (
                        parity_floor is not None and calculated >= parity_floor
                    )
                ):
                    problem(f"cell verdict {index} decision is inconsistent")
                verdict_minima.append(calculated)
        if len(verdict_ids) != len(set(verdict_ids)):
            problem("cell verdict ids are missing or not unique")
        if len(verdict_tuples) != len(verdicts):
            problem("cell verdict configurations are not unique")
        if (
            expected_verdict_tuples is None
            or verdict_tuples != expected_verdict_tuples
        ):
            problem("cell verdict configurations do not equal the release matrix")
        try:
            matrix_digest = hashlib.sha256(json.dumps(
                verdicts, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")).hexdigest()
        except (TypeError, ValueError):
            matrix_digest = None
        if matrix_digest != metrics.get("matrix_digest"):
            problem("matrix digest differs from cell verdict ledger")
        if not verdict_minima or minimum is None or not math.isclose(
            min(verdict_minima), minimum, rel_tol=1e-12, abs_tol=1e-12
        ):
            problem("global conservative ratio differs from cell verdicts")

    paired = evidence.get("paired_reports")
    report_digests: list[str] = []
    if not isinstance(paired, list) or not positive_int(cell_count) or len(
        paired if isinstance(paired, list) else []
    ) != cell_count:
        problem("paired report evidence does not exactly cover every matrix cell")
    elif isinstance(paired, list):
        for index, row in enumerate(paired):
            if not isinstance(row, Mapping):
                problem(f"paired report {index} is malformed")
                continue
            cell_id = row.get("cell_id")
            if not isinstance(cell_id, str) or not cell_id:
                problem(f"paired report {index} has no cell id")
            else:
                cell_ids.append(cell_id)
            for arm in ("candidate_sha256", "baseline_sha256"):
                value = row.get(arm)
                if not sha(value):
                    problem(f"paired report {index} {arm} is not digest-bound")
                else:
                    report_digests.append(value)
        if len(cell_ids) != len(set(cell_ids)):
            problem("paired report cell ids are not unique")
        if cell_ids != verdict_ids:
            problem("paired report ids/order differ from the cell verdict ledger")
        if len(report_digests) != len(set(report_digests)):
            problem("a benchmark report is reused across arms or matrix cells")

    telemetry = evidence.get("telemetry_sha256")
    telemetry_kinds: set[str] = set()
    telemetry_digests: list[str] = []
    if not isinstance(telemetry, list) or len(telemetry) != len(
        CB_PERFORMANCE_TELEMETRY_KINDS
    ):
        problem("telemetry evidence does not cover all four required classes")
    else:
        for index, row in enumerate(telemetry):
            if not isinstance(row, Mapping):
                problem(f"telemetry evidence {index} is malformed")
                continue
            kind = row.get("kind")
            if not isinstance(kind, str):
                problem(f"telemetry evidence {index} has no kind")
            else:
                telemetry_kinds.add(kind)
            value = row.get("sha256")
            if not sha(value):
                problem(f"telemetry evidence {index} is not digest-bound")
            else:
                telemetry_digests.append(value)
            if row.get("cell_ids") != cell_ids:
                problem(f"telemetry evidence {index} does not cover ordered cells")
        if telemetry_kinds != set(CB_PERFORMANCE_TELEMETRY_KINDS):
            problem("telemetry evidence names the wrong required classes")
        if len(telemetry_digests) != len(set(telemetry_digests)):
            problem("a telemetry payload is reused across required classes")

    displaced = evidence.get("displaced_container")
    displaced_digest: str | None = None
    if not isinstance(displaced, Mapping):
        problem("missing displaced-container eligibility proof")
    else:
        try:
            displaced_digest = hashlib.sha256(json.dumps(
                displaced,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")).hexdigest()
        except (TypeError, ValueError):
            problem("displaced-container proof is not canonical JSON")
        if displaced.get("schema") != DISPLACED_CONTAINER_ELIGIBILITY_SCHEMA:
            problem("unsupported displaced-container eligibility schema")
        if displaced.get("status") != "eligible":
            problem("displaced container is not eligible")
        for key in ("reason", "mechanism", "model_id", "cost_currency"):
            if not isinstance(displaced.get(key), str) or not str(
                displaced.get(key)
            ).strip():
                problem(f"displaced container {key} is not explicit")
        for key in (
            "model_sha",
            "artifact_inventory_sha256",
            "assignment_sha256",
            "layer_config_sha256",
            "weight_content_manifest_sha256",
            "shipcard_sha256",
            "assignment_receipt_sha256",
        ):
            if not sha(displaced.get(key)):
                problem(f"displaced container {key} is not digest-bound")
        if displaced.get("byte_budget") != budget:
            problem("displaced container binds a different byte budget")
        if displaced.get("artifact_bytes") != displaced_bytes:
            problem("displaced-container byte count differs from metrics")
        runtime = displaced.get("gridbook_runtime")
        if not isinstance(runtime, Mapping) or runtime != {
            "commit": metrics.get("gridbook_runtime_commit"),
            "version": metrics.get("gridbook_runtime_version"),
        }:
            problem("displaced container binds a different Gridbook runtime")
        endpoints = displaced.get("endpoint_record_sha256")
        if not isinstance(endpoints, Mapping) or set(endpoints) != {
            "native_export.eager", "native_export.graph"
        } or any(not sha(value) for value in (
            endpoints.values() if isinstance(endpoints, Mapping) else ()
        )):
            problem("displaced container lacks both digest-bound endpoint records")
        source = displaced.get("source_model_identity")
        if (
            not isinstance(source, Mapping)
            or source.get("schema") != "prismaquant.streamed_model.identity.v1"
            or not isinstance(source.get("resolved_commit"), str)
            or not source.get("resolved_commit")
            or not sha(source.get("content_sha256"))
            or not positive_int(source.get("checkpoint_shards"))
            or not positive_int(source.get("checkpoint_tensors"))
        ):
            problem("displaced container lacks full source-model identity")

    expected_displaced = metrics.get("displaced_container_eligibility_sha256")
    if displaced_digest != expected_displaced or evidence.get(
        "displaced_container_eligibility_sha256"
    ) != expected_displaced:
        problem("displaced-container proof digest is inconsistent")
    duplicated = {
        "model_sha": "displaced_container_model_sha",
        "artifact_inventory_sha256": "displaced_container_inventory_sha256",
        "artifact_bytes": "displaced_container_artifact_bytes",
        "assignment_sha256": "displaced_container_assignment_sha256",
        "reason": "displaced_container_reason",
    }
    if isinstance(displaced, Mapping):
        if displaced.get("model_sha") == record.get("model_sha"):
            problem("displaced container is identical to the candidate model")
        for proof_key, metric_key in duplicated.items():
            if displaced.get(proof_key) != metrics.get(metric_key):
                problem(f"displaced container {proof_key} differs from metrics")
    if evidence.get("native_baseline_feasibility_sha256") != metrics.get(
        "native_baseline_feasibility_sha256"
    ):
        problem("native infeasibility proof digest is inconsistent")

    if model_dir is not None:
        try:
            root = Path(model_dir)
            quant_config = json.loads((root / "quant_config.json").read_text(
                encoding="utf-8"
            ))
            provenance = quant_config.get("provenance")
            inventory = provenance.get("artifact_inventory") if isinstance(
                provenance, Mapping
            ) else None
            if not isinstance(inventory, Mapping):
                raise ValueError("quant_config has no finalized artifact inventory")
            inventory_digest = hashlib.sha256(json.dumps(
                inventory,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")).hexdigest()
            if inventory_digest != metrics.get("candidate_inventory_sha256"):
                raise ValueError("candidate inventory digest differs from metrics")
            if inventory.get("export_directory_bytes") != candidate_bytes:
                raise ValueError("candidate inventory byte count differs from metrics")
            if inventory.get("whole_artifact_budget_bytes") != budget:
                raise ValueError("candidate inventory binds a different budget")
            source_identity = provenance.get("source_model_identity") if isinstance(
                provenance, Mapping
            ) else None
            if not isinstance(displaced, Mapping) or displaced.get(
                "source_model_identity"
            ) != source_identity:
                raise ValueError(
                    "candidate and displaced source-model identities differ"
                )
            if inventory.get("schema") != (
                "prismaquant.cb_export_artifact_inventory.v1"
            ) or inventory.get("scope") != "all_regular_files_recursive":
                raise ValueError("candidate inventory schema or scope is invalid")
            declared_files = inventory.get("file_bytes")
            if not isinstance(declared_files, Mapping) or not declared_files:
                raise ValueError("candidate inventory file ledger is empty")
            declared: dict[str, int] = {}
            for name, size in declared_files.items():
                if (
                    not isinstance(name, str)
                    or not name
                    or name.startswith("/")
                    or ".." in Path(name).parts
                    or isinstance(size, bool)
                    or not isinstance(size, int)
                    or size < 0
                ):
                    raise ValueError("candidate inventory file ledger is malformed")
                declared[name] = size
            observed: dict[str, int] = {}
            for path in sorted(root.rglob("*")):
                if path.is_symlink():
                    raise ValueError(
                        f"candidate artifact contains symlink {path.relative_to(root)}"
                    )
                if path.is_file():
                    observed[path.relative_to(root).as_posix()] = int(
                        path.stat().st_size
                    )
            if observed != declared:
                raise ValueError(
                    "candidate recursive files differ from finalized inventory"
                )
            if sum(observed.values()) != candidate_bytes or sum(
                observed.values()
            ) > budget:
                raise ValueError(
                    "candidate recursive byte sum differs from metrics or exceeds budget"
                )
        except Exception as exc:
            problem(f"candidate artifact inventory replay failed — {exc}")
    return problems


def unfilled_slots(card: Mapping[str, Any]) -> list[str]:
    slots = card.get("slots") or {}
    return [slot for slot in required_slots(card) if not slots.get(slot)]


def _is_gridbook_card(
    card: Mapping[str, Any],
    *,
    model_dir: str | os.PathLike | None = None,
) -> bool:
    """Resolve CB identity without trusting only the mutable receipt.

    A shipcard is intentionally mutated as gates close, so publication with an
    on-disk artifact also reads ``quant_config.json``.  Removing the CB slot and
    changing ``build.quant_method`` in the receipt therefore cannot erase the
    performance obligation.
    """
    if (card.get("build") or {}).get("quant_method") == "gridbook":
        return True
    slots = card.get("slots")
    if isinstance(slots, Mapping) and any(slot in slots for slot in CB_REQUIRED_SLOTS):
        return True
    if model_dir is None:
        return False
    quant_path = Path(model_dir) / "quant_config.json"
    if not quant_path.is_file():
        return False
    try:
        payload = json.loads(quant_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, Mapping) and payload.get("quant_method") == "gridbook"


def required_slots(
    card: Mapping[str, Any],
    *,
    model_dir: str | os.PathLike | None = None,
) -> tuple[str, ...]:
    """Return the blocking slot set for this artifact/container."""
    if _is_gridbook_card(card, model_dir=model_dir):
        return ALL_SLOTS
    return REQUIRED_SLOTS


# ---------------------------------------------------------------------------
# Build-lane fact collection (used by export_native_compressed)
# ---------------------------------------------------------------------------
def kv_shared_fisher_echo(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Echo the KV-cotangent / shared-Fisher flag state (D24 caveat).

    An allocation probed with `PRISMAQUANT_ALLOW_KV_SHARED_FISHER=1` (or with the
    cotangent path switched off) rode an under-counted `k_proj`/`v_proj`
    `h_trace`. That is currently visible only in a probe log; putting it on the
    ship record makes it visible on the artifact.
    """
    env = os.environ if env is None else env
    allow = env.get("PRISMAQUANT_ALLOW_KV_SHARED_FISHER", "0")
    cotangent = env.get("PRISMAQUANT_KV_COTANGENT", "1")
    allow_on = allow not in ("", "0", "false", "False")
    cotangent_on = cotangent not in ("", "0", "false", "False")
    return {
        "PRISMAQUANT_ALLOW_KV_SHARED_FISHER": allow,
        "PRISMAQUANT_KV_COTANGENT": cotangent,
        "kv_cotangent_path_enabled": cotangent_on,
        "unvalidated_kv_fisher_correction": bool(allow_on or not cotangent_on),
        "caveat": (
            "D24: the KV-cotangent path has never been run on a real "
            "num_kv_shared_layers>0 checkpoint; ALLOW_KV_SHARED_FISHER=1 or "
            "KV_COTANGENT=0 means this allocation rode an under-counted "
            "k_proj/v_proj h_trace"
        ),
    }


def allocator_achieved_bpp(
    layer_config_path: str | os.PathLike | None,
) -> dict[str, Any]:
    """Best-effort achieved bpp, with its provenance named.

    The exporter is handed a recipe, not a bpp. The allocator's own number lives
    in `pareto.knees.json` next to `layer_config.json`; read it and *say where it
    came from* rather than recomputing an accounting-convention-sensitive number
    (CLAUDE.md principle 12: bpp labels are not comparable across eras).
    """
    if not layer_config_path:
        return {"value": None, "source": None}
    knees = Path(layer_config_path).parent / "pareto.knees.json"
    if not knees.is_file():
        return {"value": None, "source": None}
    try:
        payload = json.loads(knees.read_text())
        mode = payload.get("primary") or "log_error"
        entry = payload.get(mode) or {}
        value = entry.get("achieved_bits")
        if value is None:
            return {"value": None, "source": None}
        return {
            "value": float(value),
            "source": f"pareto.knees.json:{mode}",
            "target_bits": entry.get("target_bits"),
            "note": (
                "the allocator's achieved bpp for the knee it selected; it "
                "describes the recipe, not the exported bytes"
            ),
        }
    except Exception:
        return {"value": None, "source": None}
