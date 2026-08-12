#!/usr/bin/env python3
"""Serve fingerprint — mechanize the §7.4 reproducibility contract.

R15 (`docs/audits/architecture_re-vet_2026-07-30.md`). KL is bit-identical
*within* one docker session and drifts 4-8x *across* sessions: loading any CUDA
extension into the serving process shifts allocator addresses, activations get
different pointer alignments, and alignment-sensitive cuBLAS/CUTLASS heuristics
pick different kernels. On the 27B this reads as two bit-reproducible states,
conf-KL 0.01134 vs 0.01328 (+-17%), keyed purely on whether the gridbook `.so`
was resident during the dump. The rule ("A/B arms must have identical extension
residency; deltas under ~+-20% across differing stacks are not evidence") was
prose with nothing enforcing it.

This module makes the stack an object:

* `collect_manifest()` reads the **server's** address space
  (`/proc/<pid>/maps`) - it must be server-side, because the measuring client
  cannot see the server's residency, which is exactly why the drift stayed
  invisible for so long.
* `fingerprint()` = sha256 of the canonical JSON of the manifest **minus argv
  paths**, so two artifacts served the same way share a fingerprint (an A/B
  needs that) while a changed image / extension set / eager flag does not.

CLI (run inside the serving container, after READY):

    python3 /repo/tools/serve_fingerprint.py write \
        --out /dqruns/<run>/exported/serve_manifest.json --image vllm-node:latest

Stdlib only by construction: it must not import torch or vllm into the serving
container (an extra CUDA context on a 121 GiB unified pool is how boxes die),
so versions come from `importlib.metadata` and the GPU from NVML via
`nvidia-smi`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

MANIFEST_SCHEMA = "prismaquant.serve_manifest/1"
MANIFEST_FILENAME = "serve_manifest.json"

#: The extensions whose residency moves the numbers (§7.4).
EXTENSION_PATTERN = re.compile(
    r"gridbook|prismaquant|flashinfer|causal_conv1d|fla")

#: Packages whose version pins the numeric stack.
TRACKED_PACKAGES = (
    "vllm", "torch", "flashinfer-python", "gridbook", "prismaquant",
    "causal-conv1d", "flash-linear-attention", "transformers",
)

#: Keys excluded from the fingerprint: they identify the *run*, not the *stack*.
_FINGERPRINT_EXCLUDED = frozenset({
    "created", "launch_argv", "processes", "model", "container", "hostname",
    "serve_fingerprint", "schema", "served_model_name", "written_by",
})

_PATH_PLACEHOLDER = "<path>"


# ---------------------------------------------------------------------------
# Process inspection
# ---------------------------------------------------------------------------
def _read_cmdline(pid: str | int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except Exception:
        return []
    return [part for part in raw.decode("utf-8", "replace").split("\0") if part]


def find_server_pids(pattern: str = "vllm") -> list[int]:
    """Every readable process whose argv looks like the vLLM server or engine.

    Both matter: on vLLM v1 the API front-end and the EngineCore worker are
    different processes, and it is the *engine* that has the kernels resident.
    """
    pids: list[int] = []
    try:
        entries = sorted(int(p) for p in os.listdir("/proc") if p.isdigit())
    except Exception:
        return []
    for pid in entries:
        argv = _read_cmdline(pid)
        if not argv:
            continue
        joined = " ".join(argv)
        if pattern in joined or "EngineCore" in joined or "VLLM" in joined:
            pids.append(pid)
    return pids


def residency_scan(
    pids: Iterable[int | str],
) -> tuple[list[str], list[int], list[int]]:
    """`(basenames, readable_pids, unreadable_pids)` from `/proc/<pid>/maps`.

    The unreadable list is not bookkeeping: reading the maps of a root-owned
    container process from the host is denied, and that denial looks exactly
    like "no extensions are resident" — the false negative that would make two
    different stacks fingerprint identically. The caller records readability so
    an unverified scan can never masquerade as a verified empty one.
    """
    found: set[str] = set()
    readable: list[int] = []
    unreadable: list[int] = []
    for pid in pids:
        try:
            text = Path(f"/proc/{pid}/maps").read_text(errors="replace")
        except Exception:
            unreadable.append(int(pid))
            continue
        readable.append(int(pid))
        for line in text.splitlines():
            parts = line.split()
            if len(parts) < 6:
                continue
            path = parts[-1]
            if not path.startswith("/"):
                continue
            if ".so" not in path:
                continue
            if EXTENSION_PATTERN.search(path):
                found.add(os.path.basename(path))
    return sorted(found), readable, unreadable


def resident_extensions(pids: Iterable[int | str]) -> list[str]:
    """Sorted, de-duplicated basenames of the tracked `.so`s mapped by `pids`."""
    return residency_scan(pids)[0]


def package_versions(names: Sequence[str] = TRACKED_PACKAGES) -> dict[str, str]:
    """Installed versions via metadata only — never imports the package."""
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
        except Exception:
            continue
    return out


def gridbook_runtime_pin() -> dict[str, str] | None:
    """Immutable external Gridbook identity injected by the serve helper."""
    mapping = {
        "commit": "PQ_GRIDBOOK_RUNTIME_COMMIT",
        "version": "PQ_GRIDBOOK_RUNTIME_VERSION",
    }
    value = {
        field: os.environ[name]
        for field, name in mapping.items()
        if os.environ.get(name)
    }
    return value or None


def git_commit(repo: str | os.PathLike | None = None) -> str | None:
    """HEAD of the tree this tool was run from (`None` if unavailable).

    Gold-lane result JSONs carried no provenance at all before R15 — less than
    the surrogate KL JSONs, which have had `_git_provenance` for a year.
    """
    root = Path(repo) if repo is not None else Path(__file__).resolve().parents[1]
    override = os.environ.get("PRISMAQUANT_IDENTITY_GIT_COMMIT", "").strip().lower()
    if override and re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", override) is None:
        raise ValueError(
            "PRISMAQUANT_IDENTITY_GIT_COMMIT must be a full 40- or 64-hex commit"
        )
    try:
        observed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30,
        ).stdout.strip()
    except Exception:
        observed = None
    if override and observed is not None and override != observed.lower():
        raise ValueError(
            "PRISMAQUANT_IDENTITY_GIT_COMMIT contradicts the mounted checkout"
        )
    return override or observed


def artifact_binding(model_dir: str | os.PathLike) -> dict[str, Any]:
    """Bind a live server manifest to the exact mounted CB artifact."""
    root = Path(model_dir)
    from prismaquant.shipcard import compute_model_sha

    quant_path = root / "quant_config.json"
    payload = json.loads(quant_path.read_text(encoding="utf-8"))
    provenance = payload.get("provenance") if isinstance(payload, dict) else None
    inventory = provenance.get("artifact_inventory") if isinstance(
        provenance, dict
    ) else None
    if (
        not isinstance(inventory, dict)
        or inventory.get("schema")
        != "prismaquant.cb_export_artifact_inventory.v1"
        or inventory.get("scope") != "all_regular_files_recursive"
    ):
        raise ValueError("served artifact has no finalized recursive CB inventory")
    file_bytes = inventory.get("file_bytes")
    if not isinstance(file_bytes, dict) or not file_bytes:
        raise ValueError("served artifact inventory has no file ledger")
    observed: dict[str, int] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(
                f"served artifact contains symlink {path.relative_to(root)}"
            )
        if path.is_file():
            observed[path.relative_to(root).as_posix()] = int(path.stat().st_size)
    if observed != file_bytes or sum(observed.values()) != inventory.get(
        "export_directory_bytes"
    ):
        raise ValueError("served artifact files differ from finalized inventory")
    canonical = json.dumps(
        inventory, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "schema": "prismaquant.served_artifact_binding/1",
        "model_sha": compute_model_sha(root),
        "artifact_inventory_sha256": hashlib.sha256(canonical).hexdigest(),
        "artifact_bytes": sum(observed.values()),
    }


def gpu_identity() -> dict[str, Any]:
    """GPU name + driver from NVML (`nvidia-smi`) — no CUDA context created."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,driver_version",
             "--format=csv,noheader"],
            check=True, text=True, timeout=30,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        ).stdout.strip().splitlines()
    except Exception:
        return {"gpu_name": None, "driver_version": None}
    if not out:
        return {"gpu_name": None, "driver_version": None}
    first = [field.strip() for field in out[0].split(",")]
    return {
        "gpu_name": first[0] if first else None,
        "driver_version": first[1] if len(first) > 1 else None,
        "gpu_count": len(out),
    }


# ---------------------------------------------------------------------------
# argv handling
# ---------------------------------------------------------------------------
def elide_argv_paths(argv: Sequence[str]) -> list[str]:
    """Replace every path-like token with `<path>`.

    This is what makes the fingerprint a property of the *stack* rather than of
    the run: arm A and arm B of an A/B name different artifact directories and
    different output files, and must still share a fingerprint, while
    `--enforce-eager`, `--kv-cache-dtype fp8` or a changed image must not.
    """
    out: list[str] = []
    for token in argv:
        if "/" in token or token.startswith("~"):
            out.append(_PATH_PLACEHOLDER)
        else:
            out.append(token)
    return out


def _flag_value(argv: Sequence[str], flag: str) -> str | None:
    for index, token in enumerate(argv):
        if token == flag:
            return argv[index + 1] if index + 1 < len(argv) else ""
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


def _serve_model(argv: Sequence[str]) -> str | None:
    explicit = _flag_value(argv, "--model")
    if explicit:
        return explicit
    for index, token in enumerate(argv):
        if token == "serve" and index + 1 < len(argv):
            return argv[index + 1]
    return None


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def collect_manifest(
    *,
    pids: Sequence[int] | None = None,
    launch_argv: Sequence[str] | None = None,
    image: str | None = None,
    source: str = "server",
    extra: Mapping[str, Any] | None = None,
    artifact_dir: str | os.PathLike | None = None,
) -> dict[str, Any]:
    """Build the manifest for a live serving (or measuring) process."""
    if pids is None:
        pids = find_server_pids()
    pids = list(pids) or [os.getpid()]

    if launch_argv is None:
        argv = None
        for pid in pids:
            candidate = _read_cmdline(pid)
            if candidate and any("serve" == token for token in candidate):
                argv = candidate
                break
        if argv is None:
            argv = _read_cmdline(pids[0]) or list(sys.argv)
        launch_argv = argv
    launch_argv = list(launch_argv)

    enforce_eager = "--enforce-eager" in launch_argv or (
        "--enforce_eager" in launch_argv)
    extensions, readable_pids, unreadable_pids = residency_scan(pids)
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "source": source,
        "hostname": socket.gethostname(),
        "image": image or os.environ.get("PQ_SERVE_IMAGE"),
        "model": _serve_model(launch_argv),
        "served_model_name": _flag_value(launch_argv, "--served-model-name"),
        "launch_argv": launch_argv,
        "launch_flags": elide_argv_paths(launch_argv),
        "enforce_eager": bool(enforce_eager),
        "quantization": _flag_value(launch_argv, "--quantization"),
        "kv_cache_dtype": _flag_value(launch_argv, "--kv-cache-dtype"),
        "speculative_config": _flag_value(launch_argv, "--speculative-config"),
        "package_versions": package_versions(),
        "gridbook_runtime_pin": gridbook_runtime_pin(),
        "resident_extensions": extensions,
        # False whenever any inspected process's address space could not be
        # read (the host-side-of-a-container case): an unverified scan must not
        # fingerprint the same as a verified "nothing resident".
        "residency_readable": bool(readable_pids) and not unreadable_pids,
        "processes": [
            {"pid": int(pid), "cmdline": " ".join(_read_cmdline(pid))[:400]}
            for pid in pids
        ],
        "pq_env": {
            key: value for key, value in sorted(os.environ.items())
            if (
                key.startswith("PRISMAQUANT_")
                or key in {"GRIDBOOK_MXFP8_DENSE", "VLLM_USE_DEEP_GEMM"}
            ) and value
        },
    }
    manifest.update(gpu_identity())
    if artifact_dir is not None:
        manifest["artifact_binding"] = artifact_binding(artifact_dir)
    if extra:
        manifest.update(dict(extra))
    manifest["serve_fingerprint"] = fingerprint(manifest)
    return manifest


def fingerprint_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The manifest reduced to what defines the numeric stack."""
    return {
        key: value for key, value in manifest.items()
        if key not in _FINGERPRINT_EXCLUDED
    }


def fingerprint(manifest: Mapping[str, Any]) -> str:
    payload = fingerprint_payload(manifest)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def manifest_differences(
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
) -> list[str]:
    """Fingerprint-relevant keys on which two manifests disagree."""
    if not left or not right:
        return []
    a, b = fingerprint_payload(left), fingerprint_payload(right)
    return sorted(
        key for key in set(a) | set(b) if a.get(key) != b.get(key)
    )


def self_manifest(
    *,
    image: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Manifest of *this* process — for in-process measurement tools.

    `tools/measure_vllm_full_kl.py` and `tools/measure_vllm_wikitext_ppl.py`
    construct their own `LLM`, so the measuring process *is* the server and
    `/proc/self/maps` is the authoritative residency read.
    """
    return collect_manifest(
        pids=[os.getpid()],
        launch_argv=list(sys.argv),
        image=image,
        source="in_process",
        extra=extra,
    )


def load_manifest(path: str | os.PathLike) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def find_manifest(model_dir: str | os.PathLike | None) -> Path | None:
    if not model_dir:
        return None
    candidate = Path(model_dir) / MANIFEST_FILENAME
    return candidate if candidate.is_file() else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cmd_write(args: argparse.Namespace) -> int:
    manifest = collect_manifest(
        pids=[args.pid] if args.pid else None,
        image=args.image,
        artifact_dir=args.artifact_dir,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(f"[serve-manifest] {out} fingerprint="
          f"{manifest['serve_fingerprint'][:16]} "
          f"extensions={manifest['resident_extensions']}")
    if not manifest["residency_readable"]:
        print("[serve-manifest] WARN could not read every inspected process's "
              "/proc/<pid>/maps — the extension list is INCOMPLETE. Run this "
              "inside the serving container (docker exec), not on the host: "
              "an unreadable scan is not evidence that nothing is resident.")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    print(json.dumps(manifest, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p_write = sub.add_parser(
        "write", help="write serve_manifest.json for the live server")
    p_write.add_argument("--out", required=True)
    p_write.add_argument("--image", default=None,
                         help="container image tag the server runs in")
    p_write.add_argument("--pid", type=int, default=None,
                         help="inspect only this pid (default: auto-discover "
                              "the vLLM server + engine processes)")
    p_write.add_argument(
        "--artifact-dir",
        default=None,
        help="exact mounted CB artifact served by this process",
    )
    p_write.set_defaults(func=_cmd_write)

    p_show = sub.add_parser("show", help="pretty-print a manifest")
    p_show.add_argument("manifest")
    p_show.set_defaults(func=_cmd_show)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
