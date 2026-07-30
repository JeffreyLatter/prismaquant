# Verbatim --propagated-sensitivity-* surface removed from
# prismaquant/allocator.py on 2026-07-30 (re-vet R4). Five CLI arguments,
# one cost-folding body, and two metadata keys
# (`propagated_sensitivity_costs`, emitted into layer_config.json meta and
# into the Pareto manifest.json). Not importable; a record.

# ---- CLI arguments ----

    ap.add_argument("--propagated-sensitivity-report", default=None,
                    help="Optional sensitivity_propagated_group_report JSON. "
                         "When provided, propagated KL is folded into the "
                         "allocator costs before candidate construction.")
    ap.add_argument("--propagated-sensitivity-scale", type=float, default=1.0,
                    help="Multiplier for --propagated-sensitivity-report "
                         "penalties. The report's current-format unit KL "
                         "is preserved at scale=1 and distributed once over "
                         "fused siblings by added-bit share.")
    ap.add_argument("--propagated-sensitivity-score-field",
                    default="propagated_kl",
                    help="Numeric report row field to fold into allocator "
                         "costs when --propagated-sensitivity-report is set.")
    ap.add_argument("--propagated-sensitivity-format-extrapolation",
                    choices=("local_mse_ratio", "current_only", "bits_interp"),
                    default="local_mse_ratio",
                    help="How to extrapolate measured current-format "
                         "propagated sensitivity across alternative "
                         "candidate formats.")
    ap.add_argument("--propagated-sensitivity-target-format", default=None,
                    help="Override target format for propagated-sensitivity "
                         "bit-share accounting. Defaults to the report's "
                         "target_format.")

# ---- cost-folding body (ran right after accounting_stats, before
#      the `if args.formats:` format-menu resolution) ----

    propagated_sensitivity_summary: dict | None = None
    if args.propagated_sensitivity_report:
        from .propagated_sensitivity_costs import (
            apply_propagated_sensitivity_penalty,
        )

        with open(args.propagated_sensitivity_report) as f:
            propagated_report = json.load(f)
        costs, propagated_sensitivity_summary = apply_propagated_sensitivity_penalty(
            costs,
            stats=stats,
            report=propagated_report,
            scale=float(args.propagated_sensitivity_scale),
            target_format=args.propagated_sensitivity_target_format,
            score_field=args.propagated_sensitivity_score_field,
            format_extrapolation=args.propagated_sensitivity_format_extrapolation,
        )
        print(
            "[alloc] propagated sensitivity costs: "
            f"report={args.propagated_sensitivity_report} "
            f"scale={propagated_sensitivity_summary['scale']:.6g} "
            f"adjusted={propagated_sensitivity_summary['adjusted_entries']} "
            f"skipped={propagated_sensitivity_summary['skipped']} "
            f"penalty={propagated_sensitivity_summary['total_scaled_member_penalty']:.6g}",
            flush=True,
        )


# ---- metadata keys removed from layer_config meta and pareto manifest ----
#         "propagated_sensitivity_costs": propagated_sensitivity_summary,
