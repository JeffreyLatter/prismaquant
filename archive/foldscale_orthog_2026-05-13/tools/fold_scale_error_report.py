"""Flatten joint fold-scale search JSON into per-Linear error rows.

Input is the JSON produced by ``prismaquant.joint_smoothquant_format_search``.
The output CSV has one identity row plus one row per searched transform
(``smoothquant_max`` and/or ``awq_mean``) for every (qname, format).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _rel_gain(base: float, score: float) -> float:
    return (base - score) / max(base, 1e-30)


def _write_csv(payload: dict, output: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cluster in payload.get("clusters", []):
        cluster_key = str(cluster.get("cluster_key", ""))
        members = [str(m) for m in cluster.get("members", [])]
        per_format = cluster.get("per_format", {})
        for fmt, fmt_entry_obj in per_format.items():
            if not isinstance(fmt_entry_obj, dict):
                continue
            fmt_entry = fmt_entry_obj
            identity_by_qname = fmt_entry.get("identity_per_member_score")
            if not isinstance(identity_by_qname, dict):
                identity_by_qname = {}
            winner_mode = str(fmt_entry.get("mode", ""))
            cluster_identity_score = _safe_float(fmt_entry.get("alpha_0_score"))

            for qname in members:
                identity_score = _safe_float(identity_by_qname.get(qname))
                rows.append({
                    "cluster_key": cluster_key,
                    "qname": qname,
                    "format": str(fmt),
                    "transform": "identity",
                    "selected_for_format": winner_mode == "identity",
                    "knob": 0.0,
                    "score": identity_score,
                    "identity_score": identity_score,
                    "relative_gain_vs_identity": 0.0,
                    "cluster_score": cluster_identity_score,
                    "cluster_identity_score": cluster_identity_score,
                    "cluster_relative_gain_vs_identity": 0.0,
                    "requested_alpha_hi": _safe_float(
                        fmt_entry.get("requested_alpha_hi")
                    ),
                    "effective_alpha_hi": _safe_float(
                        fmt_entry.get("effective_alpha_hi")
                    ),
                    "evals": 0,
                })

            per_mode = fmt_entry.get("per_mode")
            if not isinstance(per_mode, dict):
                continue
            for mode, mode_entry_obj in per_mode.items():
                if not isinstance(mode_entry_obj, dict):
                    continue
                mode_entry = mode_entry_obj
                mode_scores = mode_entry.get("per_member_score")
                if not isinstance(mode_scores, dict):
                    mode_scores = {}
                mode_cluster_score = _safe_float(mode_entry.get("score"))
                for qname in members:
                    identity_score = _safe_float(identity_by_qname.get(qname))
                    score = _safe_float(mode_scores.get(qname))
                    rows.append({
                        "cluster_key": cluster_key,
                        "qname": qname,
                        "format": str(fmt),
                        "transform": str(mode),
                        "selected_for_format": str(mode) == winner_mode,
                        "knob": _safe_float(mode_entry.get("alpha")),
                        "score": score,
                        "identity_score": identity_score,
                        "relative_gain_vs_identity": _rel_gain(
                            identity_score,
                            score,
                        ),
                        "cluster_score": mode_cluster_score,
                        "cluster_identity_score": cluster_identity_score,
                        "cluster_relative_gain_vs_identity": _rel_gain(
                            cluster_identity_score,
                            mode_cluster_score,
                        ),
                        "requested_alpha_hi": _safe_float(
                            mode_entry.get("requested_alpha_hi")
                        ),
                        "effective_alpha_hi": _safe_float(
                            mode_entry.get("effective_alpha_hi")
                        ),
                        "evals": int(_safe_float(mode_entry.get("evals"))),
                    })

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cluster_key",
        "qname",
        "format",
        "transform",
        "selected_for_format",
        "knob",
        "score",
        "identity_score",
        "relative_gain_vs_identity",
        "cluster_score",
        "cluster_identity_score",
        "cluster_relative_gain_vs_identity",
        "requested_alpha_hi",
        "effective_alpha_hi",
        "evals",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)

    payload = json.loads(Path(args.input).read_text())
    rows = _write_csv(payload, Path(args.output))
    print(f"wrote {len(rows)} rows to {args.output}")

    selected = [r for r in rows if r["selected_for_format"]]
    by_key: dict[tuple[str, str], list[float]] = {}
    for row in selected:
        key = (str(row["format"]), str(row["transform"]))
        by_key.setdefault(key, []).append(float(row["relative_gain_vs_identity"]))
    for key in sorted(by_key):
        vals = by_key[key]
        mean = sum(vals) / max(len(vals), 1)
        print(
            f"{key[0]} {key[1]} selected rows={len(vals)} "
            f"mean_rel_gain={mean:.6g}"
        )


if __name__ == "__main__":
    main()
