"""The ship record (`exported/shipcard.json`) — a refusal contract.

R13 (`docs/audits/architecture_re-vet_2026-07-30.md`). The build lane and the
serve lane are separated by a physical boundary: `vllm` is not importable in the
build venv, so `run-pipeline.sh` cannot run a ship gate and never should. What it
*can* do is **open a record with required, empty slots** that only the serve lane
can close. `tools/shipcard.py verify` then exits non-zero until every slot holds a
record whose `model_sha` matches the artifact on disk — which turns "we never ran
the gate" from a silent omission into an explicit refusal.

Slots (all required):

| Slot | Filled by |
|---|---|
| `native_export.eager` | `validate_native_export.py --shipcard` (eager arm) |
| `native_export.graph` | `validate_native_export.py --shipcard --no-enforce-eager` |
| `ship_gate` | `validate_quantized_model.py --shipcard` |
| `gold.kl` | `tools/shipcard.py fill --slot gold.kl --record <full_kl json>` |
| `gold.ppl` | `tools/shipcard.py fill --slot gold.ppl --record <ppl json>` |

The two `gold.*` slots additionally require `spec_decode_detected: false` on the
record that produced the number — vLLM routes echo+logprobs through the draft
model under `--speculative-config`, so a spec-decode-on gold number is the MTP
head's NLL, not the artifact's (§7.5).

Stdlib only, no torch: the CLI must run anywhere the artifact is reachable.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA = "prismaquant.shipcard/1"

#: Every slot the serve lane must close before an artifact is shippable.
REQUIRED_SLOTS: tuple[str, ...] = (
    "native_export.eager",
    "native_export.graph",
    "ship_gate",
    "gold.kl",
    "gold.ppl",
)

#: Slots whose number is invalid if it was produced against a spec-decode serve.
GOLD_SLOTS: frozenset[str] = frozenset({"gold.kl", "gold.ppl"})

SHIPCARD_FILENAME = "shipcard.json"


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
def compute_model_sha(model_dir: str | os.PathLike) -> str:
    """Cheap, stable identity for an exported checkpoint.

    sha256 over the canonical JSON of `{config.json bytes sha, {weight file ->
    size}}`. Deliberately *not* a content hash of the weights: a 90 GB artifact
    would make `verify` unusable, and the pair (config, per-shard byte sizes) is
    already sensitive to any re-export — the failure this guards against is
    "the record belongs to a different build", not adversarial tampering.
    mtimes are excluded so a copied artifact keeps its identity.
    """
    root = Path(model_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"model dir does not exist: {root}")
    payload: dict[str, Any] = {}
    cfg = root / "config.json"
    if cfg.is_file():
        payload["config_sha"] = hashlib.sha256(cfg.read_bytes()).hexdigest()
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
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def artifact_bytes(model_dir: str | os.PathLike) -> int:
    """Exact on-disk footprint of the weight files (what the box must hold)."""
    root = Path(model_dir)
    total = 0
    for pattern in ("*.safetensors", "*.gguf"):
        for p in root.glob(pattern):
            total += p.stat().st_size
    return int(total)


def file_sha256(path: str | os.PathLike) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except Exception:
        return None


def git_provenance(repo: str | os.PathLike | None = None) -> dict[str, Any]:
    """`{commit, dirty}` for the tree that produced this record."""
    root = Path(repo) if repo is not None else Path(__file__).resolve().parents[1]

    def _run(cmd: list[str]) -> str | None:
        try:
            return subprocess.run(
                cmd, cwd=root, check=True, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            ).stdout.strip()
        except Exception:
            return None

    commit = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--short"])
    return {"commit": commit, "dirty": None if status is None else bool(status)}


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
    return {
        "schema": SCHEMA,
        "created": _now(),
        "model_dir": str(root.resolve()) if root.exists() else str(root),
        "model_sha": compute_model_sha(root),
        "artifact_bytes": artifact_bytes(root),
        "build": dict(build or {}),
        "slots": {slot: None for slot in REQUIRED_SLOTS},
    }


def write_shipcard(path: str | os.PathLike, card: Mapping[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card, indent=2, default=str) + "\n")
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
    if slot not in REQUIRED_SLOTS:
        raise KeyError(
            f"unknown shipcard slot {slot!r}; known: {list(REQUIRED_SLOTS)}")
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
    required: Iterable[str] = REQUIRED_SLOTS,
) -> list[str]:
    """Return the list of reasons this artifact is not shippable (empty = OK)."""
    problems: list[str] = []
    expected_sha = card.get("model_sha")

    if model_dir is not None:
        try:
            on_disk = compute_model_sha(model_dir)
        except Exception as exc:
            problems.append(f"model_sha of {model_dir} not computable: {exc!r}")
            on_disk = None
        if on_disk is not None and expected_sha and on_disk != expected_sha:
            problems.append(
                f"artifact changed since the shipcard was opened: "
                f"on-disk model_sha {on_disk[:12]} != shipcard "
                f"{str(expected_sha)[:12]} — re-run the serve lane")
            expected_sha = on_disk

    slots = card.get("slots") or {}
    for slot in required:
        record = slots.get(slot)
        if not record:
            problems.append(f"{slot}: UNFILLED")
            continue
        if not isinstance(record, dict):
            problems.append(f"{slot}: malformed record ({type(record).__name__})")
            continue
        got = record.get("model_sha")
        if not got:
            problems.append(f"{slot}: record carries no model_sha")
        elif expected_sha and got != expected_sha:
            problems.append(
                f"{slot}: record model_sha {str(got)[:12]} != artifact "
                f"{str(expected_sha)[:12]} (record belongs to another build)")
        if not record.get("passed"):
            problems.append(
                f"{slot}: FAILED — {record.get('detail') or 'no detail'}")
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


def unfilled_slots(card: Mapping[str, Any]) -> list[str]:
    slots = card.get("slots") or {}
    return [slot for slot in REQUIRED_SLOTS if not slots.get(slot)]


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
