"""Expected per-token decode READ bytes for a quantization assignment.

Why this exists
---------------
The allocator minimizes KL under a **disk-byte** budget.  Decode throughput
is not governed by disk bytes: it is governed by the bytes the serving
runtime must actually stream *per generated token*.  On a sparse MoE those
two objectives diverge violently, because a dense weight is read on every
token while a routed expert stack is read only when the router selects it.

Measured 2026-08-21 on the shipped DSv4-Flash 87 GB artifact: the dense path
is **8.3% of the checkpoint but 76.8% of decode read traffic**, and the whole
artifact costs **8.058 GB read per token** at batch 1.  A byte budget cannot
see that, so it systematically overspends decode bandwidth on the dense path.
This module is the *measurement* half of closing that gap (principle 1: it is
a measurement gap, not an optimizer gap).  It deliberately changes no
allocation; pricing read bytes inside the DP is a separate decision.

The definition
--------------
::

    read_bytes_per_token = Σ_tensor stored_bytes(tensor) × read_probability(tensor)

with exactly one read probability per tensor class (see
:data:`READ_CLASS_TABLE`, which is the single authority for that mapping):

===========================  =====================  ==========================
class                        read probability       what lands here
===========================  =====================  ==========================
``routed_experts``           ``topk / E``           routed MoE expert stacks
``dense``                    ``1.0``                allocator-assigned units
                                                    that are always active
``held_fixed``               ``1.0``                always-active tensors the
                                                    allocator never decided —
                                                    norms, biases, routers,
                                                    a pinned ``lm_head``,
                                                    grouped operands the probe
                                                    skips (DSv4 ``attn.wo_a``)
``excluded_embedding``       ``0.0`` (excluded)     the input embedding table:
                                                    one row is gathered per
                                                    token, not the table
``excluded_mtp``             ``0.0`` (excluded)     MTP / draft sidecar — read
                                                    only when spec-decode is
                                                    on; see the honest-default
                                                    note below
``excluded_non_text_graph``  ``0.0`` (excluded)     tensors the model profile
                                                    itself declines to map
                                                    into the live text graph
                                                    (vision / audio towers)
``resident_codebooks``       n/a — reported as      CB codebook tables: tiny,
                             ``resident_bytes``     cache-resident, and not
                                                    per-token stream traffic
===========================  =====================  ==========================

``topk / E`` is exact as an *expectation* under the per-layer-uniform serving
invariant PrismaQuant already enforces (experts are uniform per layer, mixed
across layers): every expert in a layer's stack stores the same bytes, so
routing skew redistributes which experts are read without changing the
expected bytes read.  Where a lane breaks that invariant — a CB split-stack
whose sub-stacks carry different bytes per expert — the number becomes an
expectation under *uniform* routing and says so in ``routing.exactness``.

Honest defaults for the genuinely ambiguous classes
---------------------------------------------------
The MTP sidecar is read every token when the artifact is served with
spec-decode and never when it is not, and nothing in the recipe says which.
It is **excluded but itemized** (``excluded.mtp_bytes``) so a caller serving
with spec-decode can add it back exactly.  The same applies to vision/audio
towers, which a text decode never touches.  Nothing is silently dropped: the
ledger reconciles against ``footprint.assignment_artifact_bytes`` to the byte
before any probability is applied, and refuses if it does not.

Scope
-----
Weights only, batch 1, greedy decode.  KV-cache traffic, activations, and
safetensors container metadata are outside it — so a figure from here is a
**lower bound** on real decode traffic, which is the honest direction for a
bandwidth ceiling.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import format_registry as fr
from . import footprint as fp
from .allocator_solver import _shape_from_stats
from .cb_export_config import CODEBOOK_TENSOR_PREFIX
from .nvfp4_cb_footprint import is_cb_format

SCHEMA = "prismaquant.read_traffic.v1"

#: Human-readable scope of every number this module produces.  It travels
#: with the value wherever it is stamped, because a published bandwidth
#: figure without its convention is not checkable (principle 12).
READ_SCOPE = (
    "weights only; expected bytes streamed per generated token at batch 1; "
    "KV cache, activations, and safetensors container metadata excluded; "
    "routed-expert stacks weighted by num_experts_per_tok / n_routed_experts"
)

#: THE mapping from tensor class to read probability.  Every consumer reads
#: it from here; there is no second copy.  ``None`` means the class carries
#: no per-token stream traffic at all and is reported separately.
READ_CLASS_TABLE: dict[str, float | None] = {
    "routed_experts": None,  # resolved at run time to topk / E
    "dense": 1.0,
    "held_fixed": 1.0,
    "excluded_embedding": 0.0,
    "excluded_mtp": 0.0,
    "excluded_non_text_graph": 0.0,
    "resident_codebooks": None,  # resident, never streamed per token
}

#: Classes whose bytes sum into ``read_bytes_per_token``.
STREAMED_CLASSES = ("dense", "routed_experts", "held_fixed")
#: Classes reported under ``excluded`` — real bytes, zero per-token traffic.
EXCLUDED_CLASSES = (
    "excluded_embedding", "excluded_mtp", "excluded_non_text_graph")

#: HF config keys that declare the routed-expert count, most specific first.
_EXPERT_COUNT_KEYS = (
    "n_routed_experts", "num_local_experts", "num_experts", "n_experts")
#: HF config keys that declare how many routed experts a token activates.
_EXPERTS_PER_TOK_KEYS = (
    "num_experts_per_tok", "num_experts_per_token", "moe_top_k",
    "num_active_experts")


class ReadTrafficError(ValueError):
    """A tensor could not be classified, priced, or reconciled.

    Always raised rather than defaulted around: a silent zero in a
    per-tensor score ranks the broken arm first (project memory,
    ``silent_zero_scores_rank_broken_arms_first``), and the same is true of
    a silently-omitted tensor in a bandwidth ledger.
    """


@dataclass(frozen=True)
class RoutingFactor:
    """The routed-expert read probability and where every term came from."""

    num_experts_per_tok: int
    n_routed_experts: int
    source: str
    exactness: str

    @property
    def read_probability(self) -> float:
        return float(self.num_experts_per_tok) / float(self.n_routed_experts)


# ---------------------------------------------------------------------------
# Routing factor
# ---------------------------------------------------------------------------

def _config_scopes(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """The config dicts a routing declaration may legitimately live in."""
    scopes: list[Mapping[str, Any]] = [config]
    for key in ("text_config", "language_model_config"):
        inner = config.get(key)
        if isinstance(inner, Mapping):
            scopes.append(inner)
    return tuple(scopes)


def _first_positive_int(
    scopes: Iterable[Mapping[str, Any]], keys: Iterable[str],
) -> tuple[int, str] | None:
    for scope_idx, scope in enumerate(scopes):
        for key in keys:
            value = scope.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            if value > 0:
                label = "config" if scope_idx == 0 else "config.text_config"
                return int(value), f"{label}.{key}"
    return None


def resolve_routing_factor(
    config: Mapping[str, Any],
    *,
    observed_expert_counts: Mapping[str, int] | None = None,
    context: str = "read_traffic",
) -> RoutingFactor:
    """Derive ``topk`` and ``E`` from the model config, and cross-check ``E``.

    Both terms are *declarations*, never guesses: an MoE model whose config
    does not state how many experts a token activates raises.  The
    ``moe_imatrix`` fallback of "assume 8" is exactly the heuristic principle
    2 forbids here, because it would silently mis-price the single largest
    term in the ledger.

    ``observed_expert_counts`` maps a tensor name to the stack depth actually
    measured for it (``stats['num_experts']`` pre-export, ``shape[0]`` on an
    exported 3-D stack).  Any disagreement with the config is a hard error:
    one of the two is describing a different model.
    """
    scopes = _config_scopes(config)
    topk = _first_positive_int(scopes, _EXPERTS_PER_TOK_KEYS)
    experts = _first_positive_int(scopes, _EXPERT_COUNT_KEYS)
    if topk is None or experts is None:
        missing = []
        if topk is None:
            missing.append(f"one of {list(_EXPERTS_PER_TOK_KEYS)}")
        if experts is None:
            missing.append(f"one of {list(_EXPERT_COUNT_KEYS)}")
        raise ReadTrafficError(
            f"[read_traffic] {context}: this checkpoint carries routed-expert "
            f"tensors but its config declares no {' and no '.join(missing)}. "
            "The routed read probability is topk/E and both terms must be "
            "declared -- refusing to assume a default, which would mis-price "
            "the largest term in the ledger."
        )
    topk_value, topk_source = topk
    experts_value, experts_source = experts
    if topk_value > experts_value:
        raise ReadTrafficError(
            f"[read_traffic] {context}: config declares "
            f"{topk_source}={topk_value} > {experts_source}={experts_value}; "
            "a token cannot activate more experts than exist."
        )
    exactness = "exact_under_per_layer_uniform_expert_stacks"
    for name, observed in sorted((observed_expert_counts or {}).items()):
        if int(observed) != experts_value:
            raise ReadTrafficError(
                f"[read_traffic] {context}: {name} carries {int(observed)} "
                f"experts but {experts_source}={experts_value}. The read "
                "probability topk/E is only meaningful when the config and "
                "the tensors describe the same model."
            )
    return RoutingFactor(
        num_experts_per_tok=topk_value,
        n_routed_experts=experts_value,
        source=f"{topk_source} / {experts_source}",
        exactness=exactness,
    )


def read_model_config(model_path: str | os.PathLike) -> dict:
    path = Path(model_path) / "config.json"
    if not path.is_file():
        raise ReadTrafficError(
            f"[read_traffic] no config.json under {str(model_path)!r}; the "
            "routed-expert read probability cannot be derived without the "
            "architecture's own expert declarations."
        )
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _strip_weight(name: str) -> str:
    return name[: -len(".weight")] if name.endswith(".weight") else name


def _has_experts_segment(name: str) -> bool:
    """True when a qname structurally sits under a routed-expert container.

    ``experts`` must be a whole path SEGMENT, which is what keeps a shared
    expert (``mlp.shared_experts.gate_proj``) out: its segment is
    ``shared_experts``, and a shared expert is read on every token.  This is
    the same structural test ``ModelProfile._packed_expert_projection_leaf``
    and ``footprint.packed_expert_alias`` already use.
    """
    return "experts" in str(name).split(".")


def _declares_routed_moe(profile) -> bool:
    """True when this profile declares a routed-expert MoE structure."""
    for accessor, empty in (
        ("per_expert_moe_regex", None),
        ("packed_expert_param_names", frozenset()),
        ("unpacked_expert_projection_names", ()),
    ):
        try:
            value = getattr(profile, accessor)()
        except Exception:
            continue
        if value and value != empty:
            return True
    return False


def _mtp_prefixes(profile) -> tuple[str, ...]:
    """Every spelling under which this profile's MTP sidecar can appear."""
    out: list[str] = []
    for value in (
        getattr(profile, "mtp_source_prefix", lambda: None)(),
        getattr(profile, "mtp_layer_prefix", lambda: None)(),
    ):
        if not value:
            continue
        text = str(value)
        out.append(text if text.endswith(".") else text + ".")
    # A recipe spells the sidecar `model.mtp.` where the checkpoint spells it
    # `mtp.`; both reach this classifier, so both are declared.
    for text in tuple(out):
        out.append("model." + text)
    return tuple(dict.fromkeys(out))


def classify_read_class(
    name: str,
    *,
    profile,
    checkpoint_key: str | None = None,
    in_assignment: bool = False,
    context: str = "read_traffic",
) -> str:
    """The read class of one tensor.  See :data:`READ_CLASS_TABLE`.

    ``name`` is the live/allocator spelling; ``checkpoint_key`` is the source
    spelling when they differ (the MTP sidecar is the case that matters — the
    profile's ``checkpoint_to_live_name`` declines it, so the manifest falls
    back to the raw key).  Both spellings are tested against every rule, so a
    class is never missed on naming alone (project memory: a Linear has three
    names).

    Raises rather than falling through when a name is structurally a routed
    expert but the profile cannot name its role: that is an undeclared
    architecture, and pricing it at ``p=1`` would silently inflate the
    ledger by the whole expert mass.
    """
    spellings = {str(name)}
    if checkpoint_key:
        spellings.add(str(checkpoint_key))
    bases = {_strip_weight(s) for s in spellings}

    if any(base.startswith(CODEBOOK_TENSOR_PREFIX) for base in bases):
        return "resident_codebooks"

    mtp_prefixes = _mtp_prefixes(profile)
    if any(base.startswith(p) for base in bases for p in mtp_prefixes):
        return "excluded_mtp"

    if any(_has_experts_segment(base) for base in bases):
        # The `experts` path SEGMENT is the structural fact, and it is the
        # same fact in all three namespaces (recipe / checkpoint / vLLM), so
        # it -- not a leaf-name table -- is what decides the read class. All
        # routed roles share one read probability, so naming the role would
        # add nothing here; what must be declared is that this architecture
        # HAS a routed MoE at all, and a name under `experts.` on a profile
        # that declares none is a contradiction, not a default.
        if not _declares_routed_moe(profile):
            raise ReadTrafficError(
                f"[read_traffic] {context}: {name!r} sits under a routed-"
                "expert container but this model profile declares no routed "
                "MoE structure (no per-expert regex, no packed-expert "
                "params, no unpacked expert projections), so its read "
                "probability is undeclared. Pricing it as always-active "
                "would inflate expected read bytes by the entire expert mass."
            )
        return "routed_experts"

    embedding = _strip_weight(str(profile.embedding_name()))
    if any(base == embedding or base.endswith("." + embedding)
           for base in bases):
        return "excluded_embedding"

    # The profile declining to map a checkpoint key into the live graph is
    # the architecture's OWN declaration that the tensor is not part of the
    # text decode path (vision/audio towers).  It is a declaration, not a
    # name test, which is why it is the rule rather than a prefix list.
    if checkpoint_key is not None:
        try:
            mapped = profile.checkpoint_to_live_name(str(checkpoint_key))
        except Exception:
            mapped = str(checkpoint_key)
        if mapped is None:
            return "excluded_non_text_graph"

    return "dense" if in_assignment else "held_fixed"


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

def _new_class_totals() -> dict[str, dict[str, Any]]:
    return {
        name: {"stored_bytes": 0, "read_bytes": 0.0, "n_tensors": 0}
        for name in READ_CLASS_TABLE
    }


def _finalize(
    class_totals: Mapping[str, Mapping[str, Any]],
    routing: RoutingFactor | None,
    *,
    reconciliation: Mapping[str, Any],
    unpriced_assignment_names: tuple[str, ...] = (),
    measured_from: str,
) -> dict:
    read_bytes = sum(
        float(class_totals[name]["read_bytes"]) for name in STREAMED_CLASSES)
    breakdown = {
        "dense": int(round(class_totals["dense"]["read_bytes"])),
        "routed": int(round(class_totals["routed_experts"]["read_bytes"])),
        "held_fixed": int(round(class_totals["held_fixed"]["read_bytes"])),
        "resident_codebooks": int(
            class_totals["resident_codebooks"]["stored_bytes"]),
    }
    return {
        "schema": SCHEMA,
        "read_bytes_per_token": int(round(read_bytes)),
        "read_gb_per_token": read_bytes / fp.GB,
        "scope": READ_SCOPE,
        "measured_from": measured_from,
        "breakdown": breakdown,
        "excluded": {
            "embedding_bytes": int(
                class_totals["excluded_embedding"]["stored_bytes"]),
            "mtp_bytes": int(class_totals["excluded_mtp"]["stored_bytes"]),
            "non_text_graph_bytes": int(
                class_totals["excluded_non_text_graph"]["stored_bytes"]),
            "note": (
                "real bytes with zero batch-1 text-decode traffic; the MTP "
                "sidecar becomes per-token traffic under spec-decode and can "
                "be added back from this figure exactly"
            ),
        },
        "routing": (
            {
                "num_experts_per_tok": routing.num_experts_per_tok,
                "n_routed_experts": routing.n_routed_experts,
                "read_probability": routing.read_probability,
                "source": routing.source,
                "exactness": routing.exactness,
            }
            if routing is not None
            else {"read_probability": None, "source": "no routed experts"}
        ),
        "classes": {
            name: {
                "stored_bytes": int(entry["stored_bytes"]),
                "read_bytes": int(round(float(entry["read_bytes"]))),
                "n_tensors": int(entry["n_tensors"]),
                "read_probability": (
                    routing.read_probability
                    if name == "routed_experts" and routing is not None
                    else READ_CLASS_TABLE[name]
                ),
            }
            for name, entry in class_totals.items()
        },
        "reconciliation": dict(reconciliation),
        "unpriced_assignment_names": list(unpriced_assignment_names),
    }


def assignment_read_traffic(
    assignment: Mapping[str, str],
    stats: Mapping[str, dict],
    *,
    model_path: str | os.PathLike,
    profile=None,
    source_manifest: Mapping[str, int] | None = None,
    source_total_bytes: int | None = None,
    config: Mapping[str, Any] | None = None,
    cb_serialization_context=None,
    per_expert_assignment: Mapping[str, str] | None = None,
    context: str = "assignment_read_traffic",
) -> dict:
    """Expected per-token decode read bytes for ``assignment``, both lanes.

    ``assignment`` is the allocator's *expanded*, post-promotion per-Linear
    recipe (fused-sibling and packed-MoE coupling already reflected) and
    ``stats`` the probe's per-Linear stats, exactly as
    :func:`footprint.assignment_artifact_bytes` takes them -- and that
    function is this one's byte authority.  Stored bytes for an assigned unit
    come from :func:`footprint.format_tensor_payload_breakdown` (the shared
    per-unit primitive that already prices NVFP4 group scales, FP8 row
    scales, and CB index/row-scale/layout bytes); stored bytes for every
    tensor the allocator never decided come from the checkpoint's own
    safetensors spans via :func:`footprint.source_tensor_span_bytes`.

    The two halves are then **reconciled against the whole-assignment
    footprint before any read probability is applied**, and a mismatch of a
    single byte raises.  That is what makes this a re-use of the footprint
    accounting rather than a second copy of it: the ledger cannot drift
    without failing.

    Returns a dict with ``read_bytes_per_token`` / ``read_gb_per_token``, the
    four-key ``breakdown`` (dense / routed / held_fixed / resident_codebooks),
    the itemized ``excluded`` bytes, the ``routing`` factor and its
    provenance, per-class totals, and the ``reconciliation`` block.  CB
    assignments require ``cb_serialization_context`` for the same reason
    ``assignment_artifact_bytes`` does.
    """
    model_path = str(model_path)
    if profile is None:
        from prismaquant.model_profiles import detect_profile_with_warning
        profile = detect_profile_with_warning(
            model_path, entrypoint="read-traffic")
    if config is None:
        config = read_model_config(model_path)
    measured_total, by_dtype = fp.source_checkpoint_bytes(model_path)
    regime = fp.source_regime(by_dtype)
    if source_total_bytes is None:
        source_total_bytes = measured_total
    if source_manifest is None:
        source_manifest = fp.source_tensor_bytes_manifest(
            model_path,
            profile.checkpoint_to_live_name,
            profile.packed_expert_parent_for_projection,
        )

    # --- byte authority -----------------------------------------------------
    totals = fp.assignment_artifact_bytes(
        assignment, stats,
        source_total_bytes=int(source_total_bytes),
        source_manifest=source_manifest,
        regime=regime,
        context=context,
        cb_serialization_context=cb_serialization_context,
        per_expert_assignment=per_expert_assignment,
    )

    merged = dict(assignment)
    if per_expert_assignment:
        merged.update(per_expert_assignment)
    unpriced = tuple(totals["missing_stats_names"])
    grouped_payload = totals.get("per_expert_format_group_payload") or {}
    grouped_qnames = {
        qname
        for group in (grouped_payload.get("groups") or {}).values()
        if is_cb_format(group["format"])
        for qname in group["member_qnames"]
    }
    cb_payload = totals.get("cb_serialized_payload") or {}
    cb_per_tensor = cb_payload.get("per_tensor") or {}

    # Mirrors footprint's own passthrough resolution exactly: the same names,
    # resolved by the same helper, so a passthrough unit is charged the
    # measured source span rather than a closed form that might disagree.
    passthrough_names = [
        qname for qname, raw in merged.items()
        if fr.canonical_format_name(raw) in _source_passthrough_formats()
    ]
    passthrough_spans = (
        fp.resolve_reencoded_source_bytes(
            source_manifest, passthrough_names, context=context)
        if passthrough_names else {}
    )

    class_totals = _new_class_totals()
    observed_expert_counts: dict[str, int] = {}
    priced_names: list[str] = []
    saw_routed = False

    def _charge(nbytes: int, klass: str) -> None:
        entry = class_totals[klass]
        entry["stored_bytes"] += int(nbytes)
        entry["n_tensors"] += 1

    # --- half one: the units the allocator decided ---------------------------
    for qname, raw_format in merged.items():
        entry = stats.get(qname)
        if entry is None and qname.endswith(".weight"):
            entry = stats.get(_strip_weight(qname))
        if not isinstance(entry, dict):
            continue  # unpriced: its source bytes stay in the floor half
        # `priced` in footprint's own loop: these are the names whose SOURCE
        # spans it removes from the floor, so they are the names whose spans
        # this ledger must treat as already covered.
        priced_names.append(qname)
        shape = _shape_from_stats(entry)
        name = fr.canonical_format_name(raw_format)
        if qname in grouped_qnames:
            continue  # priced once per physical CB sub-stack, below
        if qname in passthrough_spans:
            nbytes = int(passthrough_spans[qname])
        elif is_cb_format(name):
            item = cb_per_tensor.get(qname)
            if item is None:
                raise ReadTrafficError(
                    f"[read_traffic] {context}: {qname!r} is assigned the CB "
                    f"format {name} but the whole-assignment CB payload has "
                    "no entry for it, so its stored bytes are unknown. "
                    "Refusing to contribute a silent zero."
                )
            nbytes = int(item["tensor_payload_bytes"])
        else:
            nbytes = int(fp.format_tensor_payload_breakdown(
                name, shape, qname=qname,
                cb_serialization_context=cb_serialization_context,
            )["tensor_payload_bytes"])
        if name == "NVFP4":
            nbytes += int(fp.nvfp4_global_sidecar_bytes(
                qname, shape,
                weight_only=bool(
                    entry.get(fp.NVFP4_WEIGHT_ONLY_STATS_KEY, False)),
            ))
        klass = classify_read_class(
            qname, profile=profile, in_assignment=True, context=context)
        if klass == "routed_experts":
            saw_routed = True
            if len(shape) == 3:
                observed_expert_counts[qname] = int(shape[0])
        _charge(nbytes, klass)

    # CB split sub-stacks are physical tensors of their own; charge each once
    # to the class of its members (all members of a group share a role).
    for key, group in sorted((grouped_payload.get("groups") or {}).items()):
        members = group["member_qnames"]
        if not members:
            raise ReadTrafficError(
                f"[read_traffic] {context}: per-expert group {key!r} declares "
                "no member tensors, so it cannot be classified.")
        klass = classify_read_class(
            members[0], profile=profile, in_assignment=True, context=context)
        saw_routed |= klass == "routed_experts"
        _charge(int(group["tensor_payload_bytes"]), klass)
        if int(group["codebook_sidecar_bytes"]):
            _charge(int(group["codebook_sidecar_bytes"]), "resident_codebooks")

    cb_sidecar_bytes = int(cb_payload.get("codebook_sidecar_bytes") or 0)
    if cb_sidecar_bytes:
        _charge(cb_sidecar_bytes, "resident_codebooks")

    # --- half two: every tensor the allocator never decided ------------------
    covered_spans: set[str] = set()
    span_map = getattr(source_manifest, "spans", {}) or {}
    for qname in priced_names:
        key = qname if qname in source_manifest else _strip_weight(qname)
        covered_spans.update(span_map.get(key, ()))

    for ckpt_key, nbytes in sorted(
        fp.source_tensor_span_bytes(model_path).items()
    ):
        if fp.source_span_identity(ckpt_key) in covered_spans:
            continue
        try:
            live = profile.checkpoint_to_live_name(ckpt_key)
        except Exception:
            live = ckpt_key
        live_name = _strip_weight(live or ckpt_key)
        klass = classify_read_class(
            live_name, profile=profile, checkpoint_key=ckpt_key,
            in_assignment=False, context=context)
        saw_routed |= klass == "routed_experts"
        _charge(int(nbytes), klass)

    # --- reconcile before weighting -----------------------------------------
    ledger_total = sum(
        int(entry["stored_bytes"]) for entry in class_totals.values())
    expected = int(totals["artifact_payload_bytes"])
    if ledger_total != expected:
        raise ReadTrafficError(
            f"[read_traffic] {context}: the per-tensor read ledger totals "
            f"{ledger_total} bytes but footprint.assignment_artifact_bytes "
            f"prices the same assignment at {expected} bytes (delta "
            f"{ledger_total - expected}). One of the two is wrong about this "
            "artifact and neither number may be published. The ledger must "
            "partition exactly the bytes the export ships, or the read-bytes "
            "figure is a different artifact's."
        )

    routing = None
    if saw_routed:
        routing = resolve_routing_factor(
            config,
            observed_expert_counts=observed_expert_counts,
            context=context,
        )
    _apply_probabilities(class_totals, routing)
    return _finalize(
        class_totals, routing,
        reconciliation={
            "ledger_stored_bytes": ledger_total,
            "footprint_artifact_payload_bytes": expected,
            "agrees": True,
            "source_total_bytes": int(source_total_bytes),
            "n_priced_units": len(priced_names),
        },
        unpriced_assignment_names=unpriced,
        measured_from="allocator assignment + source checkpoint spans",
    )


def _apply_probabilities(
    class_totals: dict[str, dict[str, Any]],
    routing: RoutingFactor | None,
) -> None:
    for name, entry in class_totals.items():
        if name == "routed_experts":
            p = routing.read_probability if routing is not None else 0.0
        else:
            p = READ_CLASS_TABLE[name] or 0.0
        entry["read_bytes"] = float(entry["stored_bytes"]) * float(p)


def _source_passthrough_formats() -> frozenset[str]:
    from prismaquant.allocator_candidates import SOURCE_PASSTHROUGH_FORMATS
    return frozenset(SOURCE_PASSTHROUGH_FORMATS)


# ---------------------------------------------------------------------------
# Post-export: measure the artifact that was actually written
# ---------------------------------------------------------------------------

def exported_checkpoint_read_traffic(
    export_dir: str | os.PathLike,
    *,
    profile=None,
    config: Mapping[str, Any] | None = None,
    context: str = "exported_checkpoint_read_traffic",
) -> dict:
    """The same stat, measured from an exported checkpoint's own headers.

    This is the form the shipcard stamps, and it is the stronger of the two:
    it reads the bytes the artifact actually ships rather than the bytes a
    recipe predicts, so it cannot describe a different assignment than the
    one on disk -- the failure mode that has now shipped twice on
    ``achieved_bpp`` (see :func:`shipcard.allocator_achieved_bpp`).

    Every tensor span in every shard is classified and counted exactly once;
    the sum of stored bytes equals the checkpoint's own tensor-data total by
    construction, which is asserted.
    """
    export_dir = str(export_dir)
    if profile is None:
        from prismaquant.model_profiles import detect_profile_with_warning
        profile = detect_profile_with_warning(
            export_dir, entrypoint="read-traffic")
    if config is None:
        config = read_model_config(export_dir)

    spans = fp.source_tensor_span_bytes(export_dir)
    class_totals = _new_class_totals()
    observed_expert_counts: dict[str, int] = {}
    saw_routed = False
    for key, nbytes in sorted(spans.items()):
        # Classify on the LIVE spelling with the on-disk one alongside: a
        # multimodal checkpoint stores the embedding as
        # `model.language_model.embed_tokens.weight`, and only the profile's
        # own mapping turns that into the name the profile declares.
        try:
            live = profile.checkpoint_to_live_name(key)
        except Exception:
            live = key
        klass = classify_read_class(
            _strip_weight(live or key), profile=profile, checkpoint_key=key,
            in_assignment=False, context=context)
        # An exported artifact has no allocator/floor distinction on disk, so
        # every always-active tensor lands in `held_fixed`; the recipe-side
        # dense/held_fixed split is only available pre-export.
        saw_routed |= klass == "routed_experts"
        class_totals[klass]["stored_bytes"] += int(nbytes)
        class_totals[klass]["n_tensors"] += 1

    for shard in sorted(Path(export_dir).glob("*.safetensors")):
        header = fp._read_safetensors_header(str(shard))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            shape = tuple(int(d) for d in (meta.get("shape") or ()))
            if len(shape) == 3 and _has_experts_segment(_strip_weight(name)):
                observed_expert_counts[name] = shape[0]

    ledger_total = sum(
        int(entry["stored_bytes"]) for entry in class_totals.values())
    measured_total = fp.source_checkpoint_bytes(export_dir)[0]
    if ledger_total != measured_total:
        raise ReadTrafficError(
            f"[read_traffic] {context}: classified {ledger_total} of "
            f"{measured_total} tensor-data bytes under {export_dir!r}. Every "
            "shipped byte must be classified; an unclassified remainder is a "
            "silently omitted term in the bandwidth figure."
        )

    routing = resolve_routing_factor(
        config, observed_expert_counts=observed_expert_counts, context=context,
    ) if saw_routed else None
    _apply_probabilities(class_totals, routing)
    return _finalize(
        class_totals, routing,
        reconciliation={
            "ledger_stored_bytes": ledger_total,
            "checkpoint_tensor_data_bytes": measured_total,
            "agrees": True,
        },
        measured_from=f"exported safetensors headers under {export_dir}",
    )


# ---------------------------------------------------------------------------
# The stamped form
# ---------------------------------------------------------------------------

def _claim_from_report(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "value": report["read_gb_per_token"],
        "units": "GB per generated token (decimal GB)",
        "source": report["measured_from"],
        "scope": report["scope"],
        "breakdown": report["breakdown"],
        "excluded": report["excluded"],
        "routing": report["routing"],
        "note": (
            "expected weight bytes streamed per decode token at batch 1 -- "
            "the quantity that sets decode throughput, which a disk-byte "
            "budget cannot see on a sparse MoE"
        ),
    }


def assignment_read_traffic_claim(
    assignment: Mapping[str, str],
    stats: Mapping[str, dict],
    *,
    model_path: str | os.PathLike,
    **kwargs: Any,
) -> dict[str, Any]:
    """:func:`assignment_read_traffic` in the stamped, advisory shape.

    The recipe-side twin of :func:`read_traffic_claim`, for the stages that
    report a bpp *before* an export exists.  Advisory for the same reason: a
    selection run must not die because a bandwidth diagnostic could not be
    computed.
    """
    try:
        report = assignment_read_traffic(
            assignment, stats, model_path=model_path, **kwargs)
    except Exception as exc:
        return {
            "value": None,
            "source": None,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return _claim_from_report(report)


def read_traffic_claim(
    export_dir: str | os.PathLike | None,
    *,
    profile=None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The value-plus-provenance dict stamped beside ``achieved_bpp``.

    Advisory by construction, exactly like
    :func:`shipcard.allocator_achieved_bpp`'s cross-check: a bug in this
    accounting must never strand a finished export at card-writing time, so
    a failure is reported as a named ``reason`` rather than raised.  The
    number is either measured and complete or absent -- never partial, which
    would be the silent-zero failure this module exists to avoid.
    """
    if not export_dir:
        return {"value": None, "source": None, "reason": "no export directory"}
    try:
        report = exported_checkpoint_read_traffic(
            export_dir, profile=profile, config=config)
    except Exception as exc:  # advisory: never block a finished export
        return {
            "value": None,
            "source": None,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return _claim_from_report(report)


__all__ = [
    "SCHEMA",
    "READ_SCOPE",
    "READ_CLASS_TABLE",
    "STREAMED_CLASSES",
    "EXCLUDED_CLASSES",
    "ReadTrafficError",
    "RoutingFactor",
    "assignment_read_traffic",
    "assignment_read_traffic_claim",
    "exported_checkpoint_read_traffic",
    "read_traffic_claim",
    "resolve_routing_factor",
    "classify_read_class",
    "read_model_config",
]
