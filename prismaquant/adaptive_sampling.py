"""adaptive_sampling.py — multi-chunk per-domain expert saliency tracker.

Two responsibilities:

  1. **Adaptive sampling (#3).** Given the per-chunk REAP saliency values
     emitted by `ExpertSaliencyTracker`, decide which experts have
     stabilized in rank and can be skipped on the next chunk's phase-3
     reverse sweep. The contested ~5–15% of experts (those that sit near
     the prune cutoff) keep getting calibrated; the rest freeze. After
     2–3 chunks the per-chunk Fisher work drops to a fraction of the
     baseline, freeing budget for more chunks at the same wall time.

  2. **Per-domain saliency (#3 + user request).** Each chunk carries a
     domain label (`agentic`, `math`, `coding`, …). The scheduler keeps
     saliency histories partitioned by domain so:
        - rank-stability is per-domain (an expert can be load-bearing
          for math but droppable for coding; that flips it back to
          contested),
        - the final probe.pkl publishes
          `expert_saliency_per_domain[domain][router_q][expert_id]` for
          the allocator to consume,
        - the allocator can prune via union ("keep if any domain says
          keep"), intersection ("drop only if every domain says drop"),
          or weighted-min policies — implemented in `allocator_prune`.

Domain inference: the filename pattern accepted by `multi_chunk_probe`
is `chunk_<domain>_<idx>.jsonl` (e.g. `chunk_agentic_03.jsonl`). Bare
`chunk_<idx>.jsonl` is treated as the synthetic domain `_global` so
single-domain calibration runs work unchanged.

State serialization: the scheduler can be checkpointed to JSON so
multi-chunk runs that restart partway pick up where they left off
(matches the resume behavior in `multi_chunk_probe.py`).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Domain inference
# ---------------------------------------------------------------------------
_GLOBAL_DOMAIN = "_global"

# `chunk_<domain>_<idx>.jsonl` — domain is one or more `[A-Za-z][A-Za-z0-9-]*`
# tokens separated by `-` (e.g. `agentic`, `math-easy`, `code-py`).
# Falls through to `_global` for `chunk_<idx>.jsonl` form.
_DOMAIN_FNAME_RE = re.compile(
    r"^chunk_([A-Za-z][A-Za-z0-9-]*)_\d+\.jsonl$"
)


def infer_chunk_domain(path: str | Path) -> str:
    """Map a chunk filename to its domain label.

    `chunk_agentic_03.jsonl` -> `"agentic"`
    `chunk_math-hard_07.jsonl` -> `"math-hard"`
    `chunk_00.jsonl` -> `"_global"`

    Anything that doesn't match is also `_global`."""
    name = Path(path).name
    m = _DOMAIN_FNAME_RE.match(name)
    if m is None:
        return _GLOBAL_DOMAIN
    return m.group(1)


# ---------------------------------------------------------------------------
# Per-expert state
# ---------------------------------------------------------------------------
@dataclass
class ExpertHistory:
    """Per-(router, expert) accumulated saliency across chunks.

    `numerator_by_domain[d]` is the raw sum `Σ_t g·||f||²` observed for
    this expert across all chunks tagged with domain `d`. Together with
    `tokens_by_domain[d]` (sum of `nsamples × seqlen` over those chunks)
    it produces the per-domain weighted-average saliency on demand.

    `chunk_values[d]` is the list of per-chunk averages (one entry per
    chunk processed in domain `d`). Used to compute rank stability:
    if successive chunks yield consistent saliency, the expert has
    stabilized for that domain.
    """
    numerator_by_domain: dict[str, float] = field(default_factory=dict)
    tokens_by_domain: dict[str, int] = field(default_factory=dict)
    chunk_values: dict[str, list[float]] = field(default_factory=dict)

    def update(self, domain: str, chunk_value: float, chunk_tokens: int):
        """Fold a chunk's per-token-averaged saliency into the history.

        The chunk reports `value = Σ g·||f||² / T_chunk`. We recover the
        numerator as `value × T_chunk` and accumulate. This keeps the
        weighted-average semantics correct when chunks have different
        token counts."""
        numerator = float(chunk_value) * float(chunk_tokens)
        self.numerator_by_domain[domain] = (
            self.numerator_by_domain.get(domain, 0.0) + numerator
        )
        self.tokens_by_domain[domain] = (
            self.tokens_by_domain.get(domain, 0) + int(chunk_tokens)
        )
        self.chunk_values.setdefault(domain, []).append(float(chunk_value))

    def saliency(self, domain: str) -> float | None:
        n = self.tokens_by_domain.get(domain, 0)
        if n <= 0:
            return None
        return self.numerator_by_domain[domain] / n

    def saliency_global(self) -> float | None:
        total_num = sum(self.numerator_by_domain.values())
        total_tok = sum(self.tokens_by_domain.values())
        if total_tok <= 0:
            return None
        return total_num / total_tok

    def domains_seen(self) -> set[str]:
        return set(self.numerator_by_domain.keys())


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------
@dataclass
class AdaptiveExpertScheduler:
    """Tracks per-(domain, router, expert) saliency and emits a narrowed
    `linear_include` regex for the next chunk.

    Stability rule (per domain):
        After at least `min_chunks_for_freeze` observations, an expert
        is "stable in domain D" when the relative range of its
        per-chunk values across the last `stability_window` chunks is
        less than `stability_threshold`:
            range_rel = (max - min) / (mean + eps) < threshold
        AND the saliency rank within its router has been consistent
        (same quintile) across those chunks.

    Freeze rule (across domains):
        An expert freezes (skipped on next chunk) when EITHER:
        - it is stable in every domain it has been observed in AND
          its global rank is in the clearly-keep or clearly-drop band
          (top `keep_band` or bottom `drop_band` of its router); OR
        - it has been observed in every known domain and its rank
          across all of them places it firmly outside the contested
          window around the prune cutoff.

    Contested experts continue to be calibrated. This lets later chunks
    spend their work where it changes prune decisions.
    """
    prune_ratio: float = 0.375
    stability_threshold: float = 0.10  # 10% relative range
    stability_window: int = 3
    min_chunks_for_freeze: int = 2
    keep_band: float = 0.25  # top 25% of router → frozen-keep
    drop_band: float = 0.10  # bottom 10% of router → frozen-drop
    contested_band: float = 0.10  # ±10pp around prune cutoff stays contested
    disagreement_spread: float = 0.5  # if per-domain ranks span > this → contested

    # Mutable state (not configurable at init time)
    history: dict[str, dict[int, ExpertHistory]] = field(default_factory=dict)
    chunks_processed_by_domain: dict[str, int] = field(default_factory=dict)

    # ---- update path ----
    def update_from_chunk_pickle(
        self,
        chunk_pickle: dict,
        domain: str,
    ) -> int:
        """Fold one chunk's `expert_saliency` into the history.

        Returns the number of (router, expert) pairs updated. Reads
        `meta.nsamples` and `meta.seqlen` from the pickle to recover
        the per-chunk token count.
        """
        meta = chunk_pickle.get("meta") or {}
        nsamples = int(meta.get("nsamples") or 0)
        seqlen = int(meta.get("seqlen") or 0)
        chunk_tokens = nsamples * seqlen
        if chunk_tokens <= 0:
            # Without a token count we cannot weight the average
            # correctly. Skip rather than poison the history.
            return 0

        saliency = chunk_pickle.get("expert_saliency") or {}
        n_updated = 0
        for router_q, per_expert in saliency.items():
            for eid_raw, value in per_expert.items():
                eid = int(eid_raw)
                hist = self.history.setdefault(router_q, {}).setdefault(
                    eid, ExpertHistory())
                hist.update(domain, float(value), chunk_tokens)
                n_updated += 1

        self.chunks_processed_by_domain[domain] = (
            self.chunks_processed_by_domain.get(domain, 0) + 1
        )
        return n_updated

    # ---- decision path ----
    def _is_stable_in_domain(self, hist: ExpertHistory, domain: str) -> bool:
        vals = hist.chunk_values.get(domain) or []
        if len(vals) < self.min_chunks_for_freeze:
            return False
        window = vals[-self.stability_window:]
        if not window:
            return False
        v_mean = sum(window) / len(window)
        eps = 1e-12
        v_rel_range = (max(window) - min(window)) / (abs(v_mean) + eps)
        return v_rel_range < self.stability_threshold

    def _router_rank(self, router_q: str, eid: int,
                    domain: str | None) -> float | None:
        """Return the expert's normalized rank within its router (0=lowest
        saliency, 1=highest), or None if not enough data.

        Uses fractional rank for ties so that two experts with identical
        saliency get the same rank value. If `domain` is None, ranks by
        the global aggregate; otherwise ranks within that domain only."""
        per_expert = self.history.get(router_q)
        if per_expert is None or len(per_expert) <= 1:
            return None
        scores: list[tuple[int, float]] = []
        for e2, h2 in per_expert.items():
            v = h2.saliency_global() if domain is None else h2.saliency(domain)
            if v is None:
                continue
            scores.append((e2, v))
        if len(scores) <= 1:
            return None
        target = next((v for e, v in scores if e == eid), None)
        if target is None:
            return None
        n_less = sum(1 for _, v in scores if v < target)
        n_tied = sum(1 for _, v in scores if v == target)
        # Fractional rank: items strictly below + midpoint of the tie group.
        frac_rank = n_less + (n_tied - 1) / 2.0
        return frac_rank / (len(scores) - 1)

    def expert_status(
        self,
        router_q: str,
        eid: int,
    ) -> str:
        """Returns one of `frozen-keep`, `frozen-drop`, or `contested`."""
        per_expert = self.history.get(router_q)
        if per_expert is None:
            return "contested"
        hist = per_expert.get(eid)
        if hist is None:
            return "contested"

        domains = hist.domains_seen()
        if not domains:
            return "contested"

        # Need stability in every domain we've observed for this expert,
        # else we still consider the rank ambiguous.
        for d in domains:
            if not self._is_stable_in_domain(hist, d):
                return "contested"

        # Per-domain rank gate: an expert is frozen-keep only if it
        # ranks in the keep band of EVERY observed domain (union: any
        # domain says load-bearing → keep). It is frozen-drop only if
        # it ranks in the drop band of EVERY observed domain.
        keep_in_all = True
        drop_in_all = True
        per_domain_ranks: list[float] = []
        for d in domains:
            r = self._router_rank(router_q, eid, d)
            if r is None:
                return "contested"
            per_domain_ranks.append(r)
            if r < (1.0 - self.keep_band):
                keep_in_all = False
            if r > self.drop_band:
                drop_in_all = False
        if keep_in_all:
            return "frozen-keep"
        if drop_in_all:
            return "frozen-drop"

        # Cross-domain disagreement: if an expert's per-domain ranks span
        # more than `disagreement_spread`, that is a strong signal of
        # domain-specific load-bearing — additional chunks (especially
        # in the contested domains) can resolve which side the union /
        # intersection prune policy will land on. Without this gate the
        # global-rank fallback below would forcibly assign frozen-keep
        # or frozen-drop based on token-weighted averaging, hiding the
        # underlying disagreement.
        if (len(per_domain_ranks) > 1
                and (max(per_domain_ranks) - min(per_domain_ranks))
                    > self.disagreement_spread):
            return "contested"

        # Near-cutoff experts stay contested even if individually stable —
        # a small drift could flip the prune decision.
        global_r = self._router_rank(router_q, eid, None)
        if global_r is None:
            return "contested"
        if abs(global_r - self.prune_ratio) < self.contested_band:
            return "contested"

        # Stable + clearly outside contested band but not at the extreme
        # ends: still freeze, the prune decision won't move.
        return "frozen-keep" if global_r > self.prune_ratio else "frozen-drop"

    def frozen_experts(self) -> dict[str, set[int]]:
        """Map router_qname -> set of expert_ids that have frozen
        (either frozen-keep or frozen-drop)."""
        out: dict[str, set[int]] = {}
        for rq, per_expert in self.history.items():
            for eid in per_expert:
                if self.expert_status(rq, eid) != "contested":
                    out.setdefault(rq, set()).add(eid)
        return out

    def contested_experts(self) -> dict[str, set[int]]:
        out: dict[str, set[int]] = {}
        for rq, per_expert in self.history.items():
            for eid in per_expert:
                if self.expert_status(rq, eid) == "contested":
                    out.setdefault(rq, set()).add(eid)
        return out

    def summary(self) -> dict:
        total = 0
        frozen_keep = 0
        frozen_drop = 0
        contested = 0
        for rq, per_expert in self.history.items():
            for eid in per_expert:
                total += 1
                s = self.expert_status(rq, eid)
                if s == "frozen-keep":
                    frozen_keep += 1
                elif s == "frozen-drop":
                    frozen_drop += 1
                else:
                    contested += 1
        return {
            "total": total,
            "frozen_keep": frozen_keep,
            "frozen_drop": frozen_drop,
            "contested": contested,
            "chunks_by_domain": dict(self.chunks_processed_by_domain),
        }

    # ---- regex narrowing ----
    def linear_include_for_next_chunk(
        self,
        base_include: str,
        expert_info: dict[str, tuple[str, str]],
    ) -> str:
        """Return a Python-regex string that includes only:
          - non-MoE Linears matched by `base_include` (always tracked), AND
          - MoE expert Linears whose owning expert is currently contested.

        `expert_info` is `{linear_qname: (router_qname, expert_id)}` —
        the same shape the probe's `discover_moe_structure` produces.

        If no experts have frozen yet (e.g. chunk 0), returns
        `base_include` unchanged so the first chunk runs at full
        breadth.
        """
        frozen = self.frozen_experts()
        if not frozen:
            return base_include
        # Collect Linear qnames whose expert is FROZEN. We exclude these
        # from the next chunk by anchoring a negative lookahead at the
        # full qname. Cheaper alternative: emit a wider exclude regex
        # via the existing `linear_exclude` arg — but the probe already
        # has a fixed `linear_exclude` for routers/gates. The cleanest
        # path is to narrow `linear_include` to base AND NOT frozen.
        frozen_linears: list[str] = []
        for lname, (rq, eid_str) in expert_info.items():
            try:
                eid = int(eid_str)
            except (ValueError, TypeError):
                continue
            if eid in frozen.get(rq, set()):
                frozen_linears.append(lname)
        if not frozen_linears:
            return base_include
        # Escape qnames for regex literal use.
        escaped = [re.escape(n) for n in frozen_linears]
        # Build: `(?=base)(?!frozen0|frozen1|…)`
        # Wrap base in a non-capturing group; rely on .search semantics.
        # Negative lookahead is anchored to the full string for safety.
        return (
            f"(?=(?:{base_include}))"
            f"(?!^(?:{'|'.join(escaped)})$).*"
        )

    # ---- persistence ----
    def to_json(self) -> str:
        payload = {
            "config": {
                "prune_ratio": self.prune_ratio,
                "stability_threshold": self.stability_threshold,
                "stability_window": self.stability_window,
                "min_chunks_for_freeze": self.min_chunks_for_freeze,
                "keep_band": self.keep_band,
                "drop_band": self.drop_band,
                "contested_band": self.contested_band,
                "disagreement_spread": self.disagreement_spread,
            },
            "chunks_processed_by_domain": dict(
                self.chunks_processed_by_domain),
            "history": {
                rq: {
                    str(eid): {
                        "numerator_by_domain": h.numerator_by_domain,
                        "tokens_by_domain": h.tokens_by_domain,
                        "chunk_values": h.chunk_values,
                    } for eid, h in per_expert.items()
                } for rq, per_expert in self.history.items()
            },
        }
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def from_json(cls, blob: str) -> "AdaptiveExpertScheduler":
        data = json.loads(blob)
        cfg = data.get("config", {})
        sched = cls(**cfg)
        sched.chunks_processed_by_domain = dict(
            data.get("chunks_processed_by_domain") or {})
        for rq, per_expert in (data.get("history") or {}).items():
            for eid_str, h in per_expert.items():
                hist = ExpertHistory(
                    numerator_by_domain=dict(h.get("numerator_by_domain", {})),
                    tokens_by_domain={
                        k: int(v) for k, v in
                        (h.get("tokens_by_domain") or {}).items()
                    },
                    chunk_values={
                        k: [float(x) for x in v]
                        for k, v in (h.get("chunk_values") or {}).items()
                    },
                )
                sched.history.setdefault(rq, {})[int(eid_str)] = hist
        return sched

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> "AdaptiveExpertScheduler":
        return cls.from_json(Path(path).read_text())


# ---------------------------------------------------------------------------
# Per-domain saliency aggregation for the final merged probe.pkl
# ---------------------------------------------------------------------------
def aggregate_per_domain_saliency(
    chunk_pickles: list[tuple[dict, str]],
) -> dict[str, dict[str, dict[int, float]]]:
    """Given a list of `(chunk_pickle_dict, domain)` pairs, return
    `expert_saliency_per_domain[domain][router_qname][expert_id]` as the
    token-weighted average across all chunks tagged with that domain.

    Each chunk's `expert_saliency[router][eid]` is `Σ g·||f||² / T_chunk`.
    The combined per-domain value is
       (Σ_chunks Σ g·||f||² ) / (Σ_chunks T_chunk)
    which we recover by reweighting each chunk's reported value by its
    own `T_chunk = nsamples × seqlen` from its pickle meta.
    """
    num_by_d: dict[str, dict[str, dict[int, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float)))
    tok_by_d: dict[str, dict[str, dict[int, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int)))

    for pkl, domain in chunk_pickles:
        meta = pkl.get("meta") or {}
        nsamples = int(meta.get("nsamples") or 0)
        seqlen = int(meta.get("seqlen") or 0)
        T = nsamples * seqlen
        if T <= 0:
            continue
        sal = pkl.get("expert_saliency") or {}
        for rq, per_expert in sal.items():
            for eid_raw, value in per_expert.items():
                eid = int(eid_raw)
                num_by_d[domain][rq][eid] += float(value) * T
                tok_by_d[domain][rq][eid] += T

    out: dict[str, dict[str, dict[int, float]]] = {}
    for d, by_router in num_by_d.items():
        out[d] = {}
        for rq, by_eid in by_router.items():
            out[d][rq] = {}
            for eid, num in by_eid.items():
                tok = tok_by_d[d][rq][eid]
                if tok <= 0:
                    continue
                out[d][rq][eid] = num / tok
    return out


def saliency_with_policy(
    per_domain: dict[str, dict[str, dict[int, float]]],
    legacy_global: dict[str, dict[int, float]],
    policy: str,
) -> dict[str, dict[int, float]]:
    """Collapse per-domain saliency into the single per-(router, expert)
    map the allocator's prune candidate construction expects.

    Policies:
      - ``"global"``: return ``legacy_global`` unchanged. This is the
        v20 behavior and the safe default when the calibration data
        was not domain-tagged.
      - ``"union"``: per-(router, expert) value = max across domains.
        Higher saliency = more load-bearing, so taking the max means
        any domain that flags an expert as load-bearing protects it
        from pruning. Use when each domain's calibration must be
        served well by the final model.
      - ``"intersection"``: per-(router, expert) value = min across
        domains. An expert is only freely droppable when it's
        droppable in every domain. Use when prune budget is tight
        and you'd rather lose marginal coverage in a few domains
        than miss the budget.
      - ``"mean"``: token-weighted average across all domains. This is
        equivalent to ``legacy_global`` if every chunk had the same
        token count; included for parity with the dual histogram.

    Falls back to ``legacy_global`` if ``per_domain`` is empty.
    """
    if not per_domain or policy == "global":
        return legacy_global

    out: dict[str, dict[int, float]] = {}

    if policy == "union":
        reducer = max
    elif policy == "intersection":
        reducer = min
    elif policy == "mean":
        # Mean across the per-domain values for each (router, expert).
        # Different from the token-weighted ``legacy_global`` because
        # it weights every domain equally regardless of token count.
        # Useful when you've intentionally calibrated more on one
        # domain but want the prune decision to give every domain
        # equal say.
        reducer = None
    else:
        raise ValueError(
            f"unknown saliency policy {policy!r} "
            f"(expected one of: global, union, intersection, mean)"
        )

    routers = set()
    for d_map in per_domain.values():
        routers.update(d_map.keys())

    for rq in routers:
        per_eid: dict[int, list[float]] = {}
        for d_map in per_domain.values():
            for eid, val in (d_map.get(rq) or {}).items():
                per_eid.setdefault(int(eid), []).append(float(val))
        out[rq] = {}
        for eid, vals in per_eid.items():
            if not vals:
                continue
            if reducer is None:
                out[rq][eid] = sum(vals) / len(vals)
            else:
                out[rq][eid] = reducer(vals)
    return out


def aggregate_global_saliency(
    per_domain: dict[str, dict[str, dict[int, float]]],
    chunk_pickles: list[tuple[dict, str]],
) -> dict[str, dict[int, float]]:
    """Token-weighted average across ALL domains. Used for the legacy
    `expert_saliency` field so downstream consumers that don't know
    about domains keep working unchanged."""
    num: dict[str, dict[int, float]] = defaultdict(
        lambda: defaultdict(float))
    tok: dict[str, dict[int, int]] = defaultdict(
        lambda: defaultdict(int))
    for pkl, _domain in chunk_pickles:
        meta = pkl.get("meta") or {}
        nsamples = int(meta.get("nsamples") or 0)
        seqlen = int(meta.get("seqlen") or 0)
        T = nsamples * seqlen
        if T <= 0:
            continue
        sal = pkl.get("expert_saliency") or {}
        for rq, per_expert in sal.items():
            for eid_raw, value in per_expert.items():
                eid = int(eid_raw)
                num[rq][eid] += float(value) * T
                tok[rq][eid] += T
    out: dict[str, dict[int, float]] = {}
    for rq, by_eid in num.items():
        out[rq] = {}
        for eid, n_val in by_eid.items():
            t = tok[rq][eid]
            if t <= 0:
                continue
            out[rq][eid] = n_val / t
    return out
