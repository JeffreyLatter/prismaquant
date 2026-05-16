"""Build a production weight cache as the *union* across format choices,
derived from per-Linear output_mse already measured in cost.pkl.

This avoids the wasted compute of full format-menu rendering (which
builds every Linear × every format regardless of need). Most Linears
quantize cleanly under NVFP4 and never benefit from MXFP8 / FP8
fallback in any allocator assignment. The cost.pkl from the cost
phase already knows which Linears are "suspect" — high output_mse
under NVFP4 — so we render the fallback formats only for those.

Decision (per Linear):
  output_mse_nvfp4 ≤ T_clean              → NVFP4 only
  T_clean < output_mse_nvfp4 ≤ T_mxfp8    → NVFP4 + MXFP8  (if dim % 32 == 0)
  T_mxfp8 < output_mse_nvfp4 ≤ T_fp8      → NVFP4 + MXFP8 + FP8
  output_mse_nvfp4 > T_fp8                → all three (BF16 passthrough always available)

Thresholds default to percentiles of the per-Linear NVFP4 output_mse
distribution: T_mxfp8 = p50, T_fp8 = p75 (configurable via flags).

Phases:
  A. Render NVFP4 for every eligible Linear (one build_production_cache pass).
  B. Compute thresholds from cost.pkl + render MXFP8 for the suspect subset.
  C. Render FP8 for the very-suspect subset.

Output:
  - cache_dir/ containing the union of (qname, format) per-Linear tensors
  - union_manifest.json describing the threshold + per-Linear render plan
"""
from __future__ import annotations
import argparse, json, os, pickle, subprocess, sys, time
from pathlib import Path


def _shell(cmd: list[str], log_path: Path, label: str, env: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[union] {label} :: {' '.join(cmd[:6])}...", flush=True)
    t0 = time.time()
    with log_path.open("w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    dt = time.time() - t0
    if proc.returncode != 0:
        tail = log_path.read_text().splitlines()[-30:]
        raise SystemExit(f"[union] {label} FAILED in {dt:.1f}s; tail:\n  " + "\n  ".join(tail))
    print(f"[union] {label} done in {dt:.1f}s", flush=True)


def _load_layer_config(path: Path) -> dict[str, dict]:
    with path.open() as f:
        return json.load(f)


def _is_nvfp4(entry: dict) -> bool:
    dt = str(entry.get("data_type", "")).lower()
    bits = entry.get("bits", 0)
    group_size = entry.get("group_size", 0)
    # Production layer_config uses data_type="nv_fp" with bits=4 group_size=16.
    # Also accept the legacy aliases.
    if dt in {"nvfp4", "nv_fp", "nv_fp4"} and bits == 4:
        return True
    if dt == "uint8" and bits == 4 and group_size == 16:
        return True
    return False


def _get_output_mse(cost_entry_per_format: dict, fmt_name: str) -> float | None:
    """Find the output_mse for a given Linear at a given format.
    Tries multiple format-name variants since the cost.pkl uses some
    aliases interchangeably."""
    candidates = {
        "NVFP4": ["NVFP4", "nvfp4"],
        "MXFP8_E4M3": ["MXFP8_E4M3", "MXFP8", "mxfp8_e4m3", "mxfp8"],
        "FP8_E4M3": ["FP8_E4M3", "FP8", "fp8_e4m3", "fp8"],
    }
    for name in candidates.get(fmt_name, [fmt_name]):
        entry = cost_entry_per_format.get(name)
        if isinstance(entry, dict) and "output_mse" in entry:
            val = entry["output_mse"]
            if isinstance(val, (int, float)) and val > 0:
                return float(val)
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model dir (.safetensors source).")
    p.add_argument("--cost-pkl", required=True,
                   help="Existing cost.pkl from the cost phase.")
    p.add_argument("--input-layer-config", required=True,
                   help="JSON layer_config listing the target NVFP4 assignment "
                   "(seeds the set of NVFP4-eligible Linears).")
    p.add_argument("--output-cache-dir", required=True)
    p.add_argument("--output-pkl", required=True,
                   help="Manifest pickle written by Phase A's NVFP4 build "
                   "(used as the main cache pointer).")
    p.add_argument("--dataset", required=True)
    p.add_argument("--n-calib-samples", type=int, default=32)
    p.add_argument("--calib-seqlen", type=int, default=1024)
    p.add_argument("--build-cache-prog", default="prismaquant.build_production_cache")
    p.add_argument("--levers", default="gptq,joint_scale_opt",
                   help="--enable list passed through to build_production_cache.")
    p.add_argument("--threshold-mxfp8", type=float, default=None,
                   help="Absolute output_mse threshold above which MXFP8 is rendered. "
                   "If omitted, uses --p-mxfp8 percentile.")
    p.add_argument("--threshold-fp8", type=float, default=None,
                   help="Absolute output_mse threshold above which FP8 is rendered. "
                   "If omitted, uses --p-fp8 percentile.")
    p.add_argument("--p-mxfp8", type=float, default=0.50,
                   help="Percentile of NVFP4 output_mse above which MXFP8 is rendered.")
    p.add_argument("--p-fp8", type=float, default=0.75,
                   help="Percentile of NVFP4 output_mse above which FP8 is rendered.")
    p.add_argument("--logs-dir", default=None)
    p.add_argument("--skip-render", action="store_true",
                   help="Just print the union plan; do not actually build.")
    args = p.parse_args(argv)

    output_cache_dir = Path(args.output_cache_dir)
    output_pkl = Path(args.output_pkl)
    output_cache_dir.mkdir(parents=True, exist_ok=True)
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(args.logs_dir or (output_pkl.parent / "union_logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_pkl.parent

    # ---- Load cost.pkl + base layer_config ----
    with open(args.cost_pkl, "rb") as f:
        cost_blob = pickle.load(f)
    costs = cost_blob.get("costs", {}) if isinstance(cost_blob, dict) else {}
    if not costs:
        # Sometimes the cost.pkl is the costs dict directly
        costs = cost_blob if isinstance(cost_blob, dict) else {}
    print(f"[union] cost.pkl: {len(costs)} entries", flush=True)

    base = _load_layer_config(Path(args.input_layer_config))
    # NVFP4-eligible = every Linear that cost.pkl has an NVFP4 measurement
    # for. This is the SUPERSET — covers every Pareto target's assignment.
    # The input_layer_config is used only for per-Linear metadata (act_*,
    # passthrough overrides).
    cost_quantizable = {qn for qn in costs
                        if _get_output_mse(costs[qn], "NVFP4") is not None}
    base_nvfp4_set = {qn for qn, e in base.items() if _is_nvfp4(e)}
    # Union: anything cost.pkl can quantize OR anything the input config flags as NVFP4
    nvfp4_eligible = sorted(cost_quantizable | base_nvfp4_set)
    print(f"[union] NVFP4-eligible from cost.pkl ∪ input layer_config: {len(nvfp4_eligible)} "
          f"(cost-quantizable={len(cost_quantizable)}, input-cfg-nvfp4={len(base_nvfp4_set)})",
          flush=True)

    # ---- Decide thresholds + per-Linear render plan ----
    per_linear: dict[str, dict] = {}
    nvfp4_omses: list[float] = []
    for qn in nvfp4_eligible:
        cost_for = costs.get(qn) or costs.get(qn.replace("model.", "model.language_model.")) or {}
        nvfp4_omse = _get_output_mse(cost_for, "NVFP4")
        mxfp8_omse = _get_output_mse(cost_for, "MXFP8_E4M3")
        fp8_omse = _get_output_mse(cost_for, "FP8_E4M3")
        in_features = base[qn].get("in_features")
        # Look up in_features from cost.pkl if not in the layer_config
        if in_features is None and "in_features" in cost_for:
            in_features = cost_for["in_features"]
        per_linear[qn] = {
            "nvfp4_output_mse": nvfp4_omse,
            "mxfp8_output_mse": mxfp8_omse,
            "fp8_output_mse": fp8_omse,
            "in_features": in_features,
            "mxfp8_compatible": (in_features is None) or (in_features % 32 == 0),
            "fp8_compatible": True,
        }
        if nvfp4_omse is not None:
            nvfp4_omses.append(nvfp4_omse)

    if not nvfp4_omses:
        raise SystemExit("[union] no NVFP4 output_mse data in cost.pkl; cannot triage")
    nvfp4_omses.sort()

    if args.threshold_mxfp8 is not None:
        t_mxfp8 = args.threshold_mxfp8
    else:
        t_mxfp8 = nvfp4_omses[min(len(nvfp4_omses) - 1, int(len(nvfp4_omses) * args.p_mxfp8))]
    if args.threshold_fp8 is not None:
        t_fp8 = args.threshold_fp8
    else:
        t_fp8 = nvfp4_omses[min(len(nvfp4_omses) - 1, int(len(nvfp4_omses) * args.p_fp8))]

    mxfp8_set, fp8_set = [], []
    for qn, q in per_linear.items():
        n = q["nvfp4_output_mse"]
        if n is None:
            # Without an output_mse measurement we conservatively render MXFP8 + FP8
            mxfp8_set.append(qn)
            fp8_set.append(qn)
            continue
        if n > t_mxfp8 and q["mxfp8_compatible"]:
            mxfp8_set.append(qn)
        if n > t_fp8 and q["fp8_compatible"]:
            fp8_set.append(qn)

    n_nvfp4 = len(nvfp4_eligible)
    full_menu = n_nvfp4 * 3
    union_renders = n_nvfp4 + len(mxfp8_set) + len(fp8_set)
    print(f"[union] thresholds (output_mse): t_mxfp8={t_mxfp8:.4e}, t_fp8={t_fp8:.4e}", flush=True)
    print(f"[union] render plan:", flush=True)
    print(f"  NVFP4  : {n_nvfp4} Linears  (always)", flush=True)
    print(f"  MXFP8  : {len(mxfp8_set):>3} Linears  ({100*len(mxfp8_set)/n_nvfp4:.1f}% of eligible)", flush=True)
    print(f"  FP8    : {len(fp8_set):>3} Linears  ({100*len(fp8_set)/n_nvfp4:.1f}% of eligible)", flush=True)
    print(f"  Total renders: {union_renders} vs full-format-menu {full_menu} "
          f"({100*union_renders/full_menu:.1f}% of format-menu)", flush=True)

    # ---- Write manifest BEFORE rendering so it survives any build failure ----
    manifest = {
        "model": args.model,
        "cost_pkl": str(args.cost_pkl),
        "input_layer_config": str(args.input_layer_config),
        "output_cache_dir": str(output_cache_dir),
        "n_eligible": n_nvfp4,
        "threshold_mxfp8_output_mse": t_mxfp8,
        "threshold_fp8_output_mse": t_fp8,
        "p_mxfp8": args.p_mxfp8 if args.threshold_mxfp8 is None else None,
        "p_fp8": args.p_fp8 if args.threshold_fp8 is None else None,
        "n_mxfp8_rendered": len(mxfp8_set),
        "n_fp8_rendered": len(fp8_set),
        "render_cost_vs_format_menu_pct": 100 * union_renders / full_menu,
        "linears": {
            qn: {
                **per_linear[qn],
                "formats_rendered": (
                    ["NVFP4"]
                    + (["MXFP8_E4M3"] if qn in mxfp8_set else [])
                    + (["FP8_E4M3"] if qn in fp8_set else [])
                ),
            }
            for qn in nvfp4_eligible
        },
    }
    manifest_path = work_dir / "union_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[union] wrote manifest -> {manifest_path}", flush=True)

    if args.skip_render:
        print(f"[union] --skip-render set; stopping after plan.", flush=True)
        return 0

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "/work")

    # ---- Phase A: NVFP4 build (every eligible Linear) ----
    phaseA_cfg_path = work_dir / "union_phaseA_nvfp4_layer_config.json"
    nvfp4_scheme = {
        "bits": 4, "group_size": 16, "sym": True, "data_type": "nv_fp",
        "act_bits": 4, "act_group_size": 16, "act_sym": True,
        "act_data_type": "nv_fp4_with_static_gs", "act_dynamic": True,
    }
    phaseA_cfg = {qn: dict(nvfp4_scheme) for qn in nvfp4_eligible}
    for qn, e in base.items():
        if qn not in phaseA_cfg:
            phaseA_cfg[qn] = e
    with phaseA_cfg_path.open("w") as f:
        json.dump(phaseA_cfg, f, indent=2)

    common_cmd = [
        sys.executable, "-m", args.build_cache_prog,
        "--model", args.model,
        "--formats", "NVFP4,MXFP8_E4M3,FP8_E4M3",
        "--render-scope", "assignment",
        "--n-calib-samples", str(args.n_calib_samples),
        "--calib-seqlen", str(args.calib_seqlen),
        "--dataset", args.dataset,
        "--enable", args.levers,
        "--disable", "scale_sweep",
        "--cache-dir", str(output_cache_dir),
    ]
    cmd_A = list(common_cmd) + [
        "--output", str(output_pkl),
        "--render-layer-config", str(phaseA_cfg_path),
    ]
    _shell(cmd_A, logs_dir / "phaseA_nvfp4_build.log", "Phase A: NVFP4 build", env)

    # ---- Phase B: MXFP8 fallback (subset) ----
    if mxfp8_set:
        mxfp8_scheme = {
            "bits": 8, "group_size": 32, "data_type": "mx_fp",
            "act_bits": 16, "act_data_type": "bfloat16",
        }
        cfg = {qn: dict(mxfp8_scheme) for qn in mxfp8_set}
        for qn, e in base.items():
            if qn not in cfg:
                cfg[qn] = e
        cfg_path = work_dir / "union_phaseB_mxfp8_layer_config.json"
        with cfg_path.open("w") as f:
            json.dump(cfg, f, indent=2)
        cmd_B = list(common_cmd) + [
            "--output", str(work_dir / "union_phaseB_mxfp8.pkl"),
            "--render-layer-config", str(cfg_path),
        ]
        _shell(cmd_B, logs_dir / "phaseB_mxfp8_build.log", "Phase B: MXFP8 fallback", env)

    # ---- Phase C: FP8 fallback (subset) ----
    if fp8_set:
        fp8_scheme = {
            "bits": 8, "group_size": 0, "data_type": "fp8_e4m3",
            "act_bits": 16, "act_data_type": "bfloat16",
        }
        cfg = {qn: dict(fp8_scheme) for qn in fp8_set}
        for qn, e in base.items():
            if qn not in cfg:
                cfg[qn] = e
        cfg_path = work_dir / "union_phaseC_fp8_layer_config.json"
        with cfg_path.open("w") as f:
            json.dump(cfg, f, indent=2)
        cmd_C = list(common_cmd) + [
            "--output", str(work_dir / "union_phaseC_fp8.pkl"),
            "--render-layer-config", str(cfg_path),
        ]
        _shell(cmd_C, logs_dir / "phaseC_fp8_build.log", "Phase C: FP8 fallback", env)

    print(f"[union] complete. cache_dir={output_cache_dir}, manifest={manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
