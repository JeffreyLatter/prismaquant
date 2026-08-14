"""AQUA-AURA: price the ACTIVATION side and merge it into a weight cost.

WHY THIS IS A SEPARATE STAGE
----------------------------
The allocator's cost was weight-only. Choosing NVFP4 does not just round the
weights to 4 bits -- it is W4A4, and commits the layer's ACTIVATIONS to 4 bits
at serve time. FP8 commits them to 8. BF16 leaves them alone. A weight-only
surrogate is structurally blind to that difference (NVFP4 and NVFP4A16 render
weights bit-identically), so the DP was buying 4-bit formats at a discount to
their true cost.

This stage exists as its own step, rather than inside the cost stage, because
the A-side is genuinely separable:

  * It needs NO render. ``activation_dloss`` reads the DENSE weight (as
    ``W[o,j]^2``), the card's ``g_sq_sum``, and the format's activation grid.
    The render basis never enters, so the number is identical whether the
    W-side was rendered with RTN or with the full GPTQ+JSO production recipe.
  * It therefore costs one streaming pass over the checkpoint and a row-blocked
    ``W^2 @ var`` per unit -- minutes, not the hours a render costs -- and it can
    be recomputed against any existing cost artifact without rebuilding it.

That separability is also the reason the term matters MORE than its size on an
RTN basis suggests: GPTQ and JSO shrink the W-side substantially and do nothing
whatever to the A-side. Measured on Qwen3.8-27B, production rendering cut
NVFP4's median W-side to 0.13x its RTN value while the A-side was unchanged, so
on the shipping render the activation term is several times the weight term for
the median Linear.

WHAT IT DOES NOT DO
-------------------
It does not choose formats and it does not rewrite an allocation. It writes one
number per (unit, format) into the cost rows and lets the DP do what it already
does. Hand-promoting the units it flags would be the post-allocator rewrite the
platform vetoes (principle 1): if the allocator picks something bad, the cost
model is what is wrong.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
import time

import numpy as np

from .allocator_candidates import ACT_DLOSS_KEY

#: Return the CUDA pool to the OS once it has reserved this much. On GB10's
#: UNIFIED memory a reserved CUDA block IS host RAM, so it competes with the
#: numpy side rather than living in a separate budget.
CUDA_RESERVED_DRAIN_GIB = 8.0


def log(m: str) -> None:
    print(f"[aqua-cost] {m}", flush=True)


def build_weight_resolver(weight_map: dict) -> dict:
    """Map a card unit name to its checkpoint tensor key.

    These differ, and not cosmetically. A card's unit names come from the module
    tree the probe walked (``model.layers.7.mlp.gate_proj``), while a multimodal
    checkpoint nests the text tower one level deeper
    (``model.language_model.layers.7.mlp.gate_proj.weight``). Matching naively
    resolves a handful of units and silently prices almost nothing, which reads
    as "the menu is unavailable" rather than as a name mismatch.

    So index every ``.weight`` key under its own base name AND a de-nested
    alias. The alias is registered with ``setdefault`` so a real ``model.layers``
    key stays authoritative over one.
    """
    idx: dict[str, str] = {}
    for key in weight_map:
        if key.endswith(".weight"):
            idx[key[: -len(".weight")]] = key
    for key in weight_map:
        if not key.endswith(".weight"):
            continue
        base = key[: -len(".weight")]
        if ".language_model." in base:
            idx.setdefault(base.replace("model.language_model.", "model."), key)
    return idx


def cached_act_path(act_dir: str, name: str) -> str:
    """Where the probe parked this Linear's real input rows."""
    return os.path.join(act_dir, name.replace(".", "__") + ".pt")


def measured_act_var(spec, x_cpu, device: str):
    """Per-input-channel error variance on the layer's REAL activations.

    The synthetic path this replaces samples independent per-channel Gaussians.
    That reproduces every channel's marginal exactly and destroys the joint --
    and the joint is what an NVFP4 block scale is a function of, since 16
    consecutive channels share one FP8 scale set by the largest magnitude among
    them in that token. Real rows carry the co-occurrence; a Gaussian batch
    cannot.

    LANDMINE: the cached tensors are CPU-resident. Without the explicit
    ``.to(device)`` the quantizer runs on CPU at full numerical fidelity and no
    speed, which reads as "slow GPU" rather than "wrong device".
    """
    import torch
    x = x_cpu.to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        xq = spec.activation_quantize_dequantize(x)
        per_channel = ((x.float() - xq.float()) ** 2).mean(dim=0)
    return per_channel.double().cpu().numpy()


def activation_dloss_table(card, model_path: str, formats: list[str], *,
                           device: str = "cpu", names=None,
                           act_dir: str | None = None,
                           ) -> tuple[dict, dict, dict]:
    """``{unit: {format: act_dloss}}`` plus a report of what could not be priced.

    A format that does not quantize activations is simply absent from a unit's
    inner dict -- that is not a hole, it is the correct answer (BF16 costs
    nothing on the A-side). A format that DOES quantize activations but could
    not be priced is recorded in ``holes``, because an unpriced A-side read as
    zero is the exact mispricing this stage exists to remove.
    """
    import torch
    from safetensors import safe_open

    from .format_cost_protocol import price_activation_only
    from .format_cost_registry import RegistryFormatPlugin

    with open(os.path.join(model_path, "model.safetensors.index.json")) as fh:
        weight_map = json.load(fh)["weight_map"]
    resolver = build_weight_resolver(weight_map)

    wanted = list(names) if names is not None else [u.topology.name
                                                   for u in card.units()]
    resolvable = [n for n in wanted if n in resolver]
    log(f"weight-key resolution: {len(resolvable)}/{len(wanted)} card units "
        f"found in the checkpoint")

    # SHARD-AT-A-TIME. ``safe_open(device="cpu")`` mmaps the shard and
    # ``get_tensor`` faults its pages in; while the handle lives those pages stay
    # RESIDENT. Holding every handle grows RSS by the full bf16 size of every
    # tensor touched -- measured at 7.1 -> 48.2 GiB over 500 units on this model,
    # which is the body's weights almost exactly. Grouping by shard and closing
    # each handle bounds resident mmap to ONE shard. Order is irrelevant: each
    # unit is priced independently.
    by_shard: dict[str, list[str]] = collections.defaultdict(list)
    for name in resolvable:
        by_shard[weight_map[resolver[name]]].append(name)
    log(f"pricing the A-side of {len(resolvable)} units across "
        f"{len(by_shard)} shards, one shard resident at a time")

    table: dict[str, dict[str, float]] = {}
    holes: dict[str, list[str]] = collections.defaultdict(list)
    non_act: set[str] = set()
    t0 = time.time()
    done = 0
    var_source = collections.Counter()
    for shard in sorted(by_shard):
        with safe_open(os.path.join(model_path, shard),
                       framework="pt", device="cpu") as handle:
            for name in by_shard[shard]:
                unit = card[name]
                w_np = handle.get_tensor(resolver[name]).to(
                    torch.float32).numpy()
                # Real input rows if the probe cached them for this Linear.
                # Loaded once per unit and reused across formats -- the tensor
                # is the same batch, only the quantizer differs.
                x_cpu = None
                if act_dir:
                    p = cached_act_path(act_dir, name)
                    if os.path.exists(p):
                        blob_ = torch.load(p, map_location="cpu",
                                           weights_only=False)
                        cand = blob_.get("inputs")
                        # A shape mismatch means the cache is from a different
                        # model/shape; silently pricing on it would be worse
                        # than falling back, so require the match.
                        if (cand is not None
                                and cand.ndim == 2
                                and cand.shape[1] == unit.in_features):
                            x_cpu = cand
                row: dict[str, float] = {}
                for fmt in formats:
                    try:
                        plugin = RegistryFormatPlugin.build(
                            fmt, shape=tuple(w_np.shape), device=device)
                    except Exception as exc:
                        holes[fmt].append(f"{name}: unbuildable ({exc})")
                        continue
                    if not plugin.descriptor.quantizes_activations:
                        non_act.add(fmt)
                        del plugin
                        continue
                    v = None
                    if x_cpu is not None:
                        v = measured_act_var(plugin.spec, x_cpu, device)
                    var_source["measured" if v is not None else "modelled"] += 1
                    a = price_activation_only(unit, w_np, plugin, act_var=v)
                    if a is None:
                        holes[fmt].append(name)
                    else:
                        row[fmt] = float(a)
                    del plugin
                if row:
                    table[name] = row
                del w_np, x_cpu
                done += 1
                if torch.cuda.is_available() and (
                        torch.cuda.memory_reserved() / (1 << 30)
                        >= CUDA_RESERVED_DRAIN_GIB):
                    torch.cuda.empty_cache()
                if done % 100 == 0:
                    log(f"  priced {done}/{len(resolvable)} "
                        f"({time.time() - t0:.0f}s)")
    log(f"A-side priced for {len(table)} units in {time.time() - t0:.0f}s")
    if act_dir:
        log(f"act_var source: {dict(var_source)} (measured = real cached "
            f"activations, modelled = per-channel Gaussian fit)")
    if non_act:
        log(f"formats that leave activations alone (correctly unpriced): "
            f"{sorted(non_act)}")
    for fmt, names_ in sorted(holes.items()):
        log(f"HOLE: {fmt} quantizes activations but {len(names_)} units could "
            f"not be priced; those rows keep a weight-only cost. "
            f"e.g. {names_[:3]}")
    return (table, {k: v for k, v in holes.items()},
            {"act_var_source": dict(var_source)})


def merge_act_dloss(costs: dict, table: dict) -> dict:
    """Write ``act_dloss`` into the cost rows. Returns a merge report.

    Mutates ``costs`` in place. Rows with no priced A-side are left untouched
    rather than set to 0.0, so ``cost_entry_act_dloss``'s default and a genuine
    measured zero stay distinguishable in the artifact.
    """
    merged = 0
    unit_hits = 0
    missing_units = []
    for name, entry in costs.items():
        row = table.get(name)
        if not row:
            missing_units.append(name)
            continue
        unit_hits += 1
        for fmt, value in row.items():
            if fmt in entry and isinstance(entry[fmt], dict):
                entry[fmt][ACT_DLOSS_KEY] = float(value)
                merged += 1
    return {"units_in_cost": len(costs), "units_merged": unit_hits,
            "entries_merged": merged,
            "units_without_act_price": len(missing_units),
            "examples_without_act_price": missing_units[:5]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--card", required=True, help="sensitivity card .npz")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--cost-in", required=True, help="weight-only cost pkl")
    ap.add_argument("--cost-out", required=True,
                    help="written; --cost-in is left untouched so the "
                         "weight-only allocation stays reproducible as an arm")
    ap.add_argument("--formats", default=None,
                    help="default: every format present in the cost artifact")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--act-dir", default=None,
                    help="directory of cached real activations (the probe's "
                         "act/ dir). When given, act_var is MEASURED on each "
                         "Linear's real input rows instead of modelled from a "
                         "per-channel Gaussian fit; units with no cached rows "
                         "fall back to the model and are counted separately.")
    args = ap.parse_args()

    from .sensitivity_card import SensitivityCard

    card = SensitivityCard.from_npz(args.card)
    card.validate()
    fingerprint = card.provenance.fingerprint()
    log(f"card: {len(card)} units, fingerprint {fingerprint}")

    with open(args.cost_in, "rb") as fh:
        blob = pickle.load(fh)
    costs = blob["costs"]
    formats = ([f.strip() for f in args.formats.split(",") if f.strip()]
               if args.formats
               else sorted({f for r in costs.values() for f in r}))
    log(f"cost artifact: {len(costs)} units, formats {formats}")

    table, holes, meta = activation_dloss_table(
        card, args.model_path, formats, device=args.device,
        names=[n for n in costs], act_dir=args.act_dir)
    report = merge_act_dloss(costs, table)
    log(f"merge: {report}")

    prov = dict(blob.get("provenance") or {})
    prov["aqua_activation_cost"] = {
        "card_fingerprint": fingerprint,
        "card_path": os.path.abspath(args.card),
        "formats_priced": formats,
        "holes": {k: len(v) for k, v in holes.items()},
        "merge_report": report,
        "act_dir": os.path.abspath(args.act_dir) if args.act_dir else None,
        **meta,
    }
    blob["provenance"] = prov
    with open(args.cost_out, "wb") as fh:
        pickle.dump(blob, fh)
    log(f"wrote {args.cost_out}")

    # A one-line readout of what the DP will now see differently. Not a result --
    # the served KL A/B is -- but enough to catch a merge that did nothing.
    ratios = []
    for name, entry in costs.items():
        n = entry.get("NVFP4")
        if isinstance(n, dict) and ACT_DLOSS_KEY in n and n.get(
                "predicted_dloss", 0.0) > 0:
            ratios.append(n[ACT_DLOSS_KEY] / n["predicted_dloss"])
    if ratios:
        r = np.array(ratios)
        log(f"NVFP4 A-side / W-side over {len(r)} units: "
            f"p10={np.percentile(r, 10):.2f} med={np.median(r):.2f} "
            f"p90={np.percentile(r, 90):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
