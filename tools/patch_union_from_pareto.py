"""Patch a union cache with the (Linear, format) entries the allocator
actually picks across all Pareto target assignments.

Reads the Pareto manifest, computes the union of (qname, format) pairs
across all per-target layer_configs, and emits per-format layer_configs
listing the Linears that need each format. The driver script can then
invoke build_production_cache once per format on those subsets.
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path


def _is_nvfp4(e: dict) -> bool:
    dt = str(e.get("data_type", "")).lower()
    return (dt in {"nvfp4", "nv_fp", "nv_fp4"} and int(e.get("bits", 0)) == 4)


def _is_mxfp8(e: dict) -> bool:
    dt = str(e.get("data_type", "")).lower()
    return (dt in {"mx_fp", "mxfp8", "mxfp8_e4m3"} and int(e.get("bits", 0)) == 8)


def _is_fp8(e: dict) -> bool:
    dt = str(e.get("data_type", "")).lower()
    return (dt in {"fp8", "fp8_e4m3"} and int(e.get("bits", 0)) == 8)


def _is_bf16(e: dict) -> bool:
    dt = str(e.get("data_type", "")).lower()
    return dt in {"float", "bfloat16"} and int(e.get("bits", 16)) in (16, 0)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pareto-manifest", required=True)
    p.add_argument("--output-dir", required=True,
                   help="Directory to write per-format layer_configs.")
    args = p.parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.pareto_manifest)
    manifest = json.load(open(manifest_path))
    candidates = manifest.get("candidates", [])
    print(f"[pareto-union] {len(candidates)} Pareto targets", flush=True)

    # Per-format dict: qname -> canonical layer_config entry. The Pareto
    # assignments store qname -> format_string ("NVFP4", "FP8_E4M3", ...);
    # we expand that to the canonical entry the renderer expects.
    schemes = {
        "NVFP4": {"bits": 4, "group_size": 16, "sym": True, "data_type": "nv_fp",
                  "act_bits": 4, "act_group_size": 16, "act_sym": True,
                  "act_data_type": "nv_fp4_with_static_gs", "act_dynamic": True},
        "MXFP8": {"bits": 8, "group_size": 32, "data_type": "mx_fp",
                  "act_bits": 16, "act_data_type": "bfloat16"},
        "FP8_E4M3": {"bits": 8, "group_size": 0, "data_type": "fp8_e4m3",
                     "act_bits": 16, "act_data_type": "bfloat16"},
    }
    needed: dict[str, dict[str, dict]] = {k: {} for k in schemes}
    for cand in candidates:
        cand_path = Path(cand["path"])
        local_path = manifest_path.parent / cand_path.name
        payload = json.load(open(local_path))
        # The assignment file has top-level {achieved_bits, assignment, ...}.
        assignment = payload.get("assignment", payload)
        for qname, fmt in assignment.items():
            if not isinstance(fmt, str):
                continue
            fmt_u = fmt.upper().replace("_E4M3", "") if fmt.upper() in {"MXFP8_E4M3"} else fmt.upper()
            if fmt_u in needed:
                needed[fmt_u][qname] = dict(schemes[fmt_u])
            # BF16 / others = passthrough, no render

    n_total = sum(len(v) for v in needed.values())
    print(f"[pareto-union] union: "
          f"NVFP4={len(needed['NVFP4'])} "
          f"MXFP8={len(needed['MXFP8'])} "
          f"FP8={len(needed['FP8_E4M3'])} (total {n_total} renders)",
          flush=True)

    # Write per-format layer_configs (with passthrough entries for completeness)
    for fmt_name, qns in needed.items():
        if not qns:
            continue
        out_path = out_dir / f"union_pareto_{fmt_name.lower()}_layer_config.json"
        cfg = dict(qns)  # qname → entry mapping
        with out_path.open("w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[pareto-union] wrote {fmt_name} layer_config -> {out_path} "
              f"({len(qns)} entries)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
