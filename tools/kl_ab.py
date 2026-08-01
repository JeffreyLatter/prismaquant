#!/usr/bin/env python3
"""Compare two gold-lane result JSONs — and refuse to cross serving stacks.

R15. §7.4's rule ("A/B arms must have identical extension residency; conf-KL
deltas under ~+-20% across differing serving stacks are not evidence") was prose
with nothing enforcing it, and the mechanism is dated and measured: on the 27B,
the same artifact read 0.01134 vs 0.01328 conf-KL (+-17%) keyed purely on whether
the gridbook `.so` was resident during the dump.

    python3 tools/kl_ab.py A.json B.json
    python3 tools/kl_ab.py A.json B.json --metric ppl
    python3 tools/kl_ab.py A.json B.json --allow-cross-fingerprint

Behaviour by provenance:

* **same `serve_fingerprint`** — report the delta. This is the only state in
  which a delta is evidence.
* **different fingerprints** — exit 3 without a delta, and name the manifest keys
  that differ. `--allow-cross-fingerprint` downgrades the report to a **range**:
  it prints the +-20% band, says whether the measured delta clears it, and never
  calls a within-band difference a win.
* **missing fingerprints** (legacy JSONs) — compare as before with a printed
  warning; the numbers predate the mechanism and cannot be checked.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:  # package mode (`python -m tools.kl_ab`)
    from .serve_fingerprint import manifest_differences
except ImportError:  # script mode (`python tools/kl_ab.py`)
    from serve_fingerprint import manifest_differences  # type: ignore

#: §7.4: below this relative delta, a cross-stack comparison is not evidence.
CROSS_STACK_BAND = 0.20

#: Preference order when the caller does not name a metric.
METRIC_PREFERENCE = (
    "kl_confident_mean",
    "kl_mean",
    "ppl",
    "mean_nll",
    "last_token_kl",
)


def _load(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: not a result object")
    return payload


def _pick_metric(a: Mapping[str, Any], b: Mapping[str, Any],
                 requested: str | None) -> str:
    if requested:
        for name, side in ((a, "A"), (b, "B")):
            if requested not in name:
                raise SystemExit(f"metric {requested!r} missing from {side}")
        return requested
    for key in METRIC_PREFERENCE:
        if _finite(a.get(key)) and _finite(b.get(key)):
            return key
    raise SystemExit(
        "no shared finite metric; pass --metric (looked for "
        f"{list(METRIC_PREFERENCE)})")


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _fingerprint(payload: Mapping[str, Any]) -> str | None:
    fp = payload.get("serve_fingerprint")
    if fp:
        return str(fp)
    manifest = payload.get("serve_manifest")
    if isinstance(manifest, dict) and manifest.get("serve_fingerprint"):
        return str(manifest["serve_fingerprint"])
    return None


def _manifest(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    manifest = payload.get("serve_manifest")
    return manifest if isinstance(manifest, dict) else None


def compare(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    metric: str,
    allow_cross_fingerprint: bool = False,
    label_a: str = "A",
    label_b: str = "B",
) -> tuple[int, list[str]]:
    """Return `(exit_code, lines)` — the whole verdict, no printing."""
    va, vb = float(a[metric]), float(b[metric])
    delta = vb - va
    rel = (delta / va) if va else float("nan")

    fa, fb = _fingerprint(a), _fingerprint(b)
    lines = [
        f"metric: {metric}",
        f"  {label_a}: {va:.6g}   ({a.get('model')})",
        f"  {label_b}: {vb:.6g}   ({b.get('model')})",
    ]

    for label, payload in ((label_a, a), (label_b, b)):
        if payload.get("spec_decode_detected"):
            lines.append(
                f"  !! {label} reports spec_decode_detected=true — that number "
                "is the DRAFT model's, not the artifact's (§7.5)")

    if fa and fb and fa != fb:
        differing = manifest_differences(_manifest(a), _manifest(b))
        lines.append("")
        lines.append(f"serve_fingerprint {label_a}: {fa[:16]}")
        lines.append(f"serve_fingerprint {label_b}: {fb[:16]}")
        lines.append(
            "  differing manifest keys: "
            + (", ".join(differing) if differing
               else "(manifests not embedded in the result JSONs)"))
        if not allow_cross_fingerprint:
            lines.append("")
            lines.append(
                "REFUSED: these numbers come from different serving stacks. "
                "Loading any CUDA extension shifts allocator addresses and "
                "flips alignment-sensitive kernel selection — the same 27B "
                "artifact read 0.01134 vs 0.01328 conf-KL (±17%) on that "
                "alone. Re-measure both arms on one stack, or pass "
                "--allow-cross-fingerprint to quote a range instead of a "
                "delta.")
            return 3, lines
        band = CROSS_STACK_BAND * 100.0
        lines.append("")
        lines.append(
            f"CROSS-STACK RANGE (not a delta). §7.4 band: ±{band:.0f}% of "
            f"{label_a} = [{va * (1 - CROSS_STACK_BAND):.6g}, "
            f"{va * (1 + CROSS_STACK_BAND):.6g}]")
        lines.append(f"  measured relative difference: {rel * 100:+.1f}%")
        if abs(rel) <= CROSS_STACK_BAND:
            lines.append(
                "  VERDICT: INSIDE the band — NOT EVIDENCE either way. Quote "
                f"{label_b} as 'within ±{band:.0f}% of {label_a} across "
                "differing serving stacks', never as a win or a regression.")
        else:
            lines.append(
                f"  VERDICT: outside the ±{band:.0f}% band, so the difference "
                "is unlikely to be residency alone — but it is still a "
                "cross-stack comparison; quote it as a range with the "
                "differing keys above, not as a measured delta.")
        return 0, lines

    if not fa or not fb:
        missing = [
            label for label, fp in ((label_a, fa), (label_b, fb)) if not fp
        ]
        lines.append("")
        lines.append(
            f"WARNING: no serve_fingerprint on {', '.join(missing)} (legacy "
            "JSON). Comparing anyway, but nothing verified that these ran on "
            "the same serving stack — the ±17% residency drift is invisible "
            "here. Re-measure with a current tool to get a checked delta.")
    else:
        lines.append("")
        lines.append(f"serve_fingerprint: {fa[:16]} (matched)")

    ca = (a.get("git_commit") or "")[:12]
    cb = (b.get("git_commit") or "")[:12]
    if ca and cb and ca != cb:
        lines.append(f"NOTE: different git_commit ({ca} vs {cb}) — same serving "
                     "stack, different measuring code.")

    lines.append("")
    lines.append(f"delta ({label_b} - {label_a}): {delta:+.6g}  "
                 f"({rel * 100:+.2f}%)")
    return 0, lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--metric", default=None,
                    help=f"default: first shared finite key in "
                         f"{list(METRIC_PREFERENCE)}")
    ap.add_argument("--allow-cross-fingerprint", action="store_true",
                    help="downgrade a refused cross-stack delta to an honest "
                         "±20% range")
    args = ap.parse_args(argv)

    a, b = _load(args.a), _load(args.b)
    metric = _pick_metric(a, b, args.metric)
    code, lines = compare(
        a, b,
        metric=metric,
        allow_cross_fingerprint=args.allow_cross_fingerprint,
        label_a=Path(args.a).name,
        label_b=Path(args.b).name,
    )
    print("\n".join(lines))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
