"""Analyze the per-Linear damp sweep log to evaluate analytical damp picks.

Input: JSON-lines file written by ``_gptq_obs_rounding_nvfp4_swept`` when
``PRISMAQUANT_DAMP_SWEEP_LOG`` is set. Each line has the per-Linear H
spectrum and per-damp Hessian-weighted reconstruction error.

We do two analyses:

1. Discrete-winner correlation: which spectral feature predicts which of
   the 5 sweep candidates won?
2. Continuous-optimum fit: parabolic fit in log-damp through each
   Linear's 5 (damp, error) points to find a continuous damp*; then fit
   candidate closed-form expressions against those damp* values.

Outputs go to stdout.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict


def kind_of(name: str | None) -> str:
    if not name:
        return "?"
    m = re.search(r"layers\.\d+\.(.+)$", name)
    return m.group(1) if m else name


def continuous_optimum_log(damp_errs: list[tuple[float, float]]) -> tuple[float, float] | None:
    """Parabolic fit in (log10 damp, error). Returns (damp*, min_error).
    Returns None if too few valid points or no interior minimum."""
    pts = [(d, e) for d, e in damp_errs if math.isfinite(e) and d > 0]
    if len(pts) < 3:
        return None
    pts.sort(key=lambda x: x[0])
    # Find index of minimum
    min_idx = min(range(len(pts)), key=lambda i: pts[i][1])
    if min_idx == 0 or min_idx == len(pts) - 1:
        # Optimum is at an endpoint; not interior. Return the endpoint as best.
        return pts[min_idx]
    # Parabolic interpolation through (min_idx-1, min_idx, min_idx+1) in log-damp
    x1, y1 = math.log10(pts[min_idx - 1][0]), pts[min_idx - 1][1]
    x2, y2 = math.log10(pts[min_idx][0]), pts[min_idx][1]
    x3, y3 = math.log10(pts[min_idx + 1][0]), pts[min_idx + 1][1]
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    if abs(denom) < 1e-30:
        return pts[min_idx]
    a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    b = (x3**2 * (y1 - y2) + x2**2 * (y3 - y1) + x1**2 * (y2 - y3)) / denom
    if a <= 0:
        return pts[min_idx]
    log_x_star = -b / (2 * a)
    log_x_star = max(min(log_x_star, x3), x1)
    return (10**log_x_star, y2)  # approximate error at the optimum


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True, help="damp_log.jsonl path")
    p.add_argument("--top", type=int, default=20, help="rows to print per kind")
    args = p.parse_args(argv)

    rows = []
    with open(args.log) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    print(f"loaded {len(rows)} log entries")
    valid = [r for r in rows if r.get("best_damp") is not None]
    print(f"with best_damp: {len(valid)}")
    if not valid:
        return 1

    # === Discrete winner distribution ===
    from collections import Counter
    winner_counts = Counter(r["best_damp"] for r in valid)
    print()
    print("=== Discrete sweep-winner distribution ===")
    total = sum(winner_counts.values())
    for d in sorted(winner_counts.keys()):
        c = winner_counts[d]
        bar = "#" * int(c / total * 60)
        print(f"  damp={d:<8}  {c:>4} ({c/total*100:>5.1f}%)  {bar}")

    # === Per-kind winner distribution ===
    by_kind = defaultdict(Counter)
    for r in valid:
        by_kind[kind_of(r.get("linear_name"))][r["best_damp"]] += 1
    print()
    print("=== Per-kind sweep-winner distribution ===")
    kinds = sorted(by_kind.keys())
    damps = sorted({d for c in by_kind.values() for d in c})
    print(f"{'kind':<32}  " + "  ".join(f"d={d:<6}" for d in damps))
    for k in kinds:
        c = by_kind[k]
        ktot = sum(c.values())
        print(f"{k:<32}  " + "  ".join(f"{c[d]/ktot*100:>5.1f}%" for d in damps))

    # === Continuous-optimal damps from parabolic fit ===
    enriched = []
    for r in valid:
        per_damp = {float(k): v for k, v in r.get("per_damp_err", {}).items()}
        pts = sorted(per_damp.items())
        opt = continuous_optimum_log(pts)
        if opt is None:
            continue
        damp_star, _ = opt
        enriched.append({
            **r,
            "damp_star": damp_star,
            "lambda_max": r.get("lambda_max"),
            "lambda_min": r.get("lambda_min"),
            "mean_diag": r.get("mean_diag"),
            "kappa": (r["lambda_max"] / r["lambda_min"]
                      if r.get("lambda_min", 0) > 0 else float("inf")),
        })
    print(f"\ncontinuous fit valid: {len(enriched)}")
    if not enriched:
        return 1

    # === Hypothesis candidates ===
    # H1: damp ≈ c * (λ_min / mean_diag)
    # H2: damp ≈ c * (λ_max / mean_diag) — OBQ default scales with this
    # H3: damp ≈ c / κ
    # H4: κ-target: damp = (λ_max - K*λ_min) / (μ*(K-1)) for some K
    #
    # For each candidate, fit log10 c via least squares against log10 damp_star
    import statistics
    def fit_const(predictor):
        ratios = []
        for r in enriched:
            d = r["damp_star"]
            p_val = predictor(r)
            if p_val is None or p_val <= 0 or d <= 0:
                continue
            ratios.append(math.log10(d) - math.log10(p_val))
        if not ratios:
            return None, None
        log_c = statistics.median(ratios)
        # residuals around the fit
        residuals = [r - log_c for r in ratios]
        mse_log = sum(x*x for x in residuals) / len(residuals)
        return log_c, mse_log

    h1_log_c, h1_mse = fit_const(lambda r: r["lambda_min"] / max(r["mean_diag"], 1e-30))
    h2_log_c, h2_mse = fit_const(lambda r: r["lambda_max"] / max(r["mean_diag"], 1e-30))
    h3_log_c, h3_mse = fit_const(lambda r: 1.0 / max(r["kappa"], 1e-30))

    # For H4, fit K via grid
    def h4_pred(r, K):
        denom = r["mean_diag"] * (K - 1)
        if denom <= 0:
            return None
        num = r["lambda_max"] - K * r["lambda_min"]
        if num <= 0:
            return None  # already well-conditioned, no damp needed
        return num / denom

    best_h4_mse = float("inf")
    best_K = None
    best_h4_log_c = None
    for K in [10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]:
        log_c, mse = fit_const(lambda r, K=K: h4_pred(r, K))
        if log_c is not None and mse < best_h4_mse:
            best_h4_mse = mse
            best_K = K
            best_h4_log_c = log_c

    # Print fits
    def print_fit(name, log_c, mse, extra=""):
        if log_c is None:
            print(f"  {name}: no fit")
            return
        print(f"  {name}: log10(c) = {log_c:+6.3f}  -> c = {10**log_c:.4g};  log-MSE = {mse:.4f}{extra}")

    print()
    print("=== Closed-form fits (median log10(c), log-MSE = mean squared log10 residual) ===")
    print_fit("H1: damp = c * (λ_min / μ)", h1_log_c, h1_mse)
    print_fit("H2: damp = c * (λ_max / μ) (OBQ-style)", h2_log_c, h2_mse)
    print_fit("H3: damp = c / κ", h3_log_c, h3_mse)
    if best_K is not None:
        print_fit(f"H4: damp = c * (λ_max - K·λ_min)/(μ·(K-1))",
                  best_h4_log_c, best_h4_mse, f"  [K={best_K:.0e}]")

    # Worst-case residual for the winning hypothesis
    fits = []
    if h1_mse is not None: fits.append(("H1", h1_log_c, h1_mse, lambda r: r["lambda_min"] / r["mean_diag"]))
    if h2_mse is not None: fits.append(("H2", h2_log_c, h2_mse, lambda r: r["lambda_max"] / r["mean_diag"]))
    if h3_mse is not None: fits.append(("H3", h3_log_c, h3_mse, lambda r: 1.0 / r["kappa"]))
    if best_h4_log_c is not None:
        fits.append(("H4", best_h4_log_c, best_h4_mse,
                     lambda r, K=best_K: h4_pred(r, K)))
    if not fits:
        print("no successful fits")
        return 0
    fits.sort(key=lambda x: x[2])
    win_name, win_log_c, win_mse, win_pred = fits[0]
    print(f"\nBest fit: {win_name}, log-MSE={win_mse:.4f}")
    win_c = 10**win_log_c
    print(f"  predicted_damp(r) = {win_c:.4g} * <feature>")
    bad = []
    for r in enriched:
        p_val = win_pred(r)
        if p_val is None or p_val <= 0:
            continue
        pred = win_c * p_val
        residual_log = math.log10(r["damp_star"]) - math.log10(pred)
        bad.append((abs(residual_log), r, residual_log, pred))
    bad.sort(reverse=True)
    print(f"\nTop {min(args.top, len(bad))} worst residuals (|log10(damp* / pred)| largest):")
    for absres, r, resid, pred in bad[: args.top]:
        print(f"  {kind_of(r.get('linear_name')):<30} damp*={r['damp_star']:.4g} pred={pred:.4g} "
              f"κ={r['kappa']:.2e}  resid={resid:+.3f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
