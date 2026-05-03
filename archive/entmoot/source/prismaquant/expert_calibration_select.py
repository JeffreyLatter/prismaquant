"""Expert-balanced calibration sample selection.

This is a proof-of-concept helper for building better MoE calibration
sets before running the expensive REAP/Fisher passes. It assumes a cheap
router survey has already produced one JSONL row per candidate sample
with per-router expert hit mass, then selects a small subset that covers
as many underrepresented router/expert pairs as possible.

Expected survey JSONL row shape::

    {
      "id": "math-00042",
      "domain": "math",
      "text": "...",
      "hits": {
        "model.layers.12.mlp.gate": {"3": 0.71, "91": 0.14}
      }
    }

``hits`` values can be router probabilities, top-k counts, or any other
positive activation mass. The selector normalizes each expert pair by
its total available mass across the candidate pool, so rare experts are
not drowned out by high-traffic experts.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prismaquant.adaptive_sampling import _GLOBAL_DOMAIN, infer_chunk_domain


CoverageKey = tuple[str, str, int]


@dataclass
class SampleCoverage:
    """Router/expert coverage observed for one calibration candidate."""

    sample_id: str
    hits: dict[str, dict[int, float]]
    domain: str = _GLOBAL_DOMAIN
    weight: float = 1.0
    payload: dict[str, Any] = field(default_factory=dict, repr=False)

    def iter_hits(self, min_mass: float = 0.0) -> Iterable[tuple[str, int, float]]:
        for router_qname, per_expert in self.hits.items():
            for eid, mass in per_expert.items():
                if mass > min_mass:
                    yield router_qname, int(eid), float(mass)


@dataclass
class SelectionResult:
    """Result returned by ``select_expert_balanced_samples``."""

    selected: list[SampleCoverage]
    available_mass: dict[CoverageKey, float]
    selected_mass: dict[CoverageKey, float]
    scores: dict[str, float]

    @property
    def selected_ids(self) -> list[str]:
        return [sample.sample_id for sample in self.selected]

    @property
    def domain_counts(self) -> dict[str, int]:
        return dict(Counter(sample.domain for sample in self.selected))

    def coverage_summary(self, max_uncovered: int = 20) -> dict[str, Any]:
        available = self.available_mass
        selected = self.selected_mass
        fractions: list[float] = []
        uncovered: list[str] = []
        for key in sorted(available):
            total = available[key]
            if total <= 0.0:
                continue
            got = selected.get(key, 0.0)
            frac = min(got / total, 1.0)
            fractions.append(frac)
            if got <= 0.0 and len(uncovered) < max_uncovered:
                uncovered.append(_key_to_str(key))

        return {
            "selected_samples": len(self.selected),
            "selected_ids": self.selected_ids,
            "domain_counts": self.domain_counts,
            "available_pairs": len(available),
            "covered_pairs": sum(
                1 for key in available if selected.get(key, 0.0) > 0.0
            ),
            "mean_pair_fraction": (
                sum(fractions) / len(fractions) if fractions else 0.0
            ),
            "min_pair_fraction": min(fractions) if fractions else 0.0,
            "uncovered_pairs_sample": uncovered,
            "scores": dict(self.scores),
        }


def sample_from_mapping(
    row: Mapping[str, Any],
    *,
    fallback_id: str,
) -> SampleCoverage:
    """Parse a survey JSON object into ``SampleCoverage``.

    Accepted hit fields are ``hits``, ``expert_hits``, ``router_hits``,
    or ``router_counts``. Expert ids may be strings or ints.
    """

    raw_hits = (
        row.get("hits")
        or row.get("expert_hits")
        or row.get("router_hits")
        or row.get("router_counts")
    )
    hits = _normalize_hits(raw_hits)
    if not hits:
        raise ValueError(f"sample {fallback_id!r} has no positive expert hits")

    sample_id = str(
        row.get("sample_id")
        or row.get("id")
        or row.get("uid")
        or fallback_id
    )
    domain = row.get("domain")
    if not isinstance(domain, str) or not domain:
        source = row.get("source") or row.get("path") or row.get("file")
        domain = infer_chunk_domain(str(source)) if source else _GLOBAL_DOMAIN

    weight_raw = row.get("weight", 1.0)
    try:
        weight = float(weight_raw)
    except (TypeError, ValueError):
        weight = 1.0
    if not math.isfinite(weight) or weight <= 0.0:
        weight = 1.0

    return SampleCoverage(
        sample_id=sample_id,
        domain=domain,
        hits=hits,
        weight=weight,
        payload=dict(row),
    )


def load_survey_jsonl(path: str | Path) -> list[SampleCoverage]:
    """Load a router/expert survey JSONL file."""

    out: list[SampleCoverage] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"{source}:{line_no} is not a JSON object")
            out.append(
                sample_from_mapping(row, fallback_id=f"{source}:{line_no}")
            )
    return out


def select_expert_balanced_samples(
    samples: Sequence[SampleCoverage],
    *,
    budget: int,
    min_domain_counts: Mapping[str, int] | None = None,
    domain_sensitive: bool = True,
    min_mass: float = 0.0,
    coverage_power: float = 0.5,
    coverage_prior: float = 0.05,
    domain_bonus: float = 0.25,
) -> SelectionResult:
    """Greedily select samples that maximize expert coverage.

    Scoring uses fractional coverage per coverage key. A sample that
    contributes all available mass for a rare expert can outrank a sample
    that only contributes a small fraction of a common expert's mass.

    Args:
      samples: Candidate survey records.
      budget: Maximum number of samples to select.
      min_domain_counts: Optional lower bound per domain. If the
        remaining slots are needed to satisfy these minimums, selection
        is restricted to the missing domains when possible.
      domain_sensitive: When true, coverage keys are
        ``(domain, router, expert)``. When false, domains only affect
        quotas, and coverage keys are ``(_global, router, expert)``.
      min_mass: Ignore per-sample expert hits at or below this mass.
      coverage_power: Diminishing-return exponent for already covered
        expert pairs. Larger values spread coverage more aggressively.
      coverage_prior: Small prior in the denominator so first coverage
        is preferred but finite.
      domain_bonus: Additive/multiplicative boost for under-quota domains.
    """

    if budget < 0:
        raise ValueError("budget must be non-negative")
    if coverage_power < 0.0:
        raise ValueError("coverage_power must be non-negative")
    if coverage_prior <= 0.0:
        raise ValueError("coverage_prior must be positive")

    clean_min_counts = {
        str(domain): max(0, int(count))
        for domain, count in (min_domain_counts or {}).items()
    }
    available = _available_mass(samples, domain_sensitive, min_mass)
    selected: list[SampleCoverage] = []
    selected_ids: set[str] = set()
    selected_mass: defaultdict[CoverageKey, float] = defaultdict(float)
    domain_counts: Counter[str] = Counter()
    scores: dict[str, float] = {}

    max_picks = min(int(budget), len(samples))
    while len(selected) < max_picks:
        remaining_slots = max_picks - len(selected)
        missing_domains = _missing_domains(
            clean_min_counts, domain_counts, remaining_slots
        )
        candidates = [
            sample for sample in samples
            if sample.sample_id not in selected_ids
        ]
        if missing_domains:
            constrained = [
                sample for sample in candidates if sample.domain in missing_domains
            ]
            if constrained:
                candidates = constrained
        if not candidates:
            break

        best_sample: SampleCoverage | None = None
        best_score = -math.inf
        for sample in candidates:
            score = _sample_score(
                sample,
                available,
                selected_mass,
                domain_counts,
                clean_min_counts,
                domain_sensitive,
                min_mass,
                coverage_power,
                coverage_prior,
                domain_bonus,
            )
            if (
                score > best_score
                or (
                    math.isclose(score, best_score, rel_tol=0.0, abs_tol=1e-12)
                    and best_sample is not None
                    and sample.sample_id < best_sample.sample_id
                )
            ):
                best_score = score
                best_sample = sample

        if best_sample is None:
            break

        selected.append(best_sample)
        selected_ids.add(best_sample.sample_id)
        domain_counts[best_sample.domain] += 1
        scores[best_sample.sample_id] = float(best_score)
        for key, mass in _iter_key_mass(
            best_sample, domain_sensitive, min_mass
        ):
            selected_mass[key] += mass

    return SelectionResult(
        selected=selected,
        available_mass=dict(available),
        selected_mass=dict(selected_mass),
        scores=scores,
    )


def write_selected_jsonl(
    result: SelectionResult,
    path: str | Path,
    *,
    ids_only: bool = False,
) -> None:
    """Write selected survey rows or ids as JSONL."""

    target = Path(path)
    with target.open("w", encoding="utf-8") as f:
        for sample in result.selected:
            if ids_only:
                obj = {"id": sample.sample_id}
            elif sample.payload:
                obj = dict(sample.payload)
            else:
                obj = {
                    "id": sample.sample_id,
                    "domain": sample.domain,
                    "hits": {
                        router: {str(eid): mass for eid, mass in hits.items()}
                        for router, hits in sample.hits.items()
                    },
                }
            f.write(json.dumps(obj, sort_keys=True) + "\n")


def first_n_baseline_result(
    samples: Sequence[SampleCoverage],
    *,
    budget: int,
    available_mass: Mapping[CoverageKey, float],
    domain_sensitive: bool = True,
    min_mass: float = 0.0,
) -> SelectionResult:
    """Build a coverage result for the first-N input rows."""

    selected = list(samples[:max(0, min(int(budget), len(samples)))])
    return SelectionResult(
        selected=selected,
        available_mass=dict(available_mass),
        selected_mass=dict(_available_mass(selected, domain_sensitive, min_mass)),
        scores={},
    )


def coverage_lift(
    selected_summary: Mapping[str, Any],
    baseline_summary: Mapping[str, Any],
) -> dict[str, float | int]:
    """Return simple deltas between selected and baseline coverage."""

    return {
        "covered_pairs": (
            int(selected_summary.get("covered_pairs", 0))
            - int(baseline_summary.get("covered_pairs", 0))
        ),
        "mean_pair_fraction": (
            float(selected_summary.get("mean_pair_fraction", 0.0))
            - float(baseline_summary.get("mean_pair_fraction", 0.0))
        ),
        "min_pair_fraction": (
            float(selected_summary.get("min_pair_fraction", 0.0))
            - float(baseline_summary.get("min_pair_fraction", 0.0))
        ),
    }


def _normalize_hits(raw_hits: Any) -> dict[str, dict[int, float]]:
    if not isinstance(raw_hits, Mapping):
        return {}

    out: dict[str, dict[int, float]] = {}
    for router_raw, experts_raw in raw_hits.items():
        router = str(router_raw)
        per_expert: dict[int, float] = {}

        if isinstance(experts_raw, Mapping):
            expert_items = experts_raw.items()
        elif isinstance(experts_raw, Sequence) and not isinstance(
            experts_raw, (str, bytes)
        ):
            expert_items = _expert_items_from_sequence(experts_raw)
        else:
            continue

        for eid_raw, mass_raw in expert_items:
            try:
                eid = int(eid_raw)
                mass = float(mass_raw)
            except (TypeError, ValueError):
                continue
            if eid < 0 or not math.isfinite(mass) or mass <= 0.0:
                continue
            per_expert[eid] = per_expert.get(eid, 0.0) + mass

        if per_expert:
            out[router] = per_expert
    return out


def _expert_items_from_sequence(seq: Sequence[Any]) -> Iterable[tuple[Any, Any]]:
    for item in seq:
        if isinstance(item, Mapping):
            eid = (
                item.get("expert")
                if "expert" in item
                else item.get("expert_id", item.get("id"))
            )
            mass = (
                item.get("mass")
                if "mass" in item
                else item.get("count", item.get("weight", 1.0))
            )
            yield eid, mass
        elif (
            isinstance(item, Sequence)
            and not isinstance(item, (str, bytes))
            and len(item) >= 2
        ):
            yield item[0], item[1]


def _available_mass(
    samples: Sequence[SampleCoverage],
    domain_sensitive: bool,
    min_mass: float,
) -> defaultdict[CoverageKey, float]:
    available: defaultdict[CoverageKey, float] = defaultdict(float)
    for sample in samples:
        for key, mass in _iter_key_mass(sample, domain_sensitive, min_mass):
            available[key] += mass
    return available


def _iter_key_mass(
    sample: SampleCoverage,
    domain_sensitive: bool,
    min_mass: float,
) -> Iterable[tuple[CoverageKey, float]]:
    domain = sample.domain if domain_sensitive else _GLOBAL_DOMAIN
    for router_qname, eid, mass in sample.iter_hits(min_mass):
        yield (domain, router_qname, eid), mass


def _sample_score(
    sample: SampleCoverage,
    available: Mapping[CoverageKey, float],
    selected_mass: Mapping[CoverageKey, float],
    domain_counts: Mapping[str, int],
    min_domain_counts: Mapping[str, int],
    domain_sensitive: bool,
    min_mass: float,
    coverage_power: float,
    coverage_prior: float,
    domain_bonus: float,
) -> float:
    score = 0.0
    for key, mass in _iter_key_mass(sample, domain_sensitive, min_mass):
        total = available.get(key, 0.0)
        if total <= 0.0:
            continue
        fraction = min(mass / total, 1.0)
        covered_fraction = min(selected_mass.get(key, 0.0) / total, 1.0)
        score += fraction / ((coverage_prior + covered_fraction) ** coverage_power)

    missing = max(
        0,
        int(min_domain_counts.get(sample.domain, 0))
        - int(domain_counts.get(sample.domain, 0)),
    )
    if missing:
        target = max(1, int(min_domain_counts[sample.domain]))
        score = score * (1.0 + domain_bonus) + domain_bonus * missing / target

    return score * max(float(sample.weight), 0.0)


def _missing_domains(
    min_domain_counts: Mapping[str, int],
    domain_counts: Mapping[str, int],
    remaining_slots: int,
) -> set[str]:
    missing_by_domain = {
        domain: max(0, int(target) - int(domain_counts.get(domain, 0)))
        for domain, target in min_domain_counts.items()
    }
    total_missing = sum(missing_by_domain.values())
    if total_missing < remaining_slots:
        return set()
    return {domain for domain, missing in missing_by_domain.items() if missing > 0}


def _key_to_str(key: CoverageKey) -> str:
    domain, router_qname, eid = key
    return f"{domain}:{router_qname}:{eid}"


def _parse_domain_min(values: Sequence[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"invalid --domain-min {value!r}; expected DOMAIN=COUNT"
            )
        domain, count_raw = value.split("=", 1)
        out[domain] = int(count_raw)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select calibration samples that improve expert coverage.",
    )
    parser.add_argument("--survey", required=True, help="survey JSONL path")
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--output", help="selected JSONL output path")
    parser.add_argument("--summary", help="summary JSON output path")
    parser.add_argument(
        "--ids-only",
        action="store_true",
        help="write only selected ids instead of full survey rows",
    )
    parser.add_argument(
        "--domain-min",
        action="append",
        default=[],
        metavar="DOMAIN=COUNT",
        help="minimum number of selected samples for a domain",
    )
    parser.add_argument(
        "--global-coverage",
        action="store_true",
        help="cover router/expert pairs globally instead of per domain",
    )
    parser.add_argument("--min-mass", type=float, default=0.0)
    args = parser.parse_args(argv)

    samples = load_survey_jsonl(args.survey)
    result = select_expert_balanced_samples(
        samples,
        budget=args.budget,
        min_domain_counts=_parse_domain_min(args.domain_min),
        domain_sensitive=not args.global_coverage,
        min_mass=args.min_mass,
    )

    if args.output:
        write_selected_jsonl(result, args.output, ids_only=args.ids_only)

    summary = result.coverage_summary()
    baseline = first_n_baseline_result(
        samples,
        budget=args.budget,
        available_mass=result.available_mass,
        domain_sensitive=not args.global_coverage,
        min_mass=args.min_mass,
    ).coverage_summary()
    summary["first_n_baseline"] = {
        key: value for key, value in baseline.items() if key != "scores"
    }
    summary["coverage_lift_vs_first_n"] = coverage_lift(summary, baseline)
    if args.summary:
        Path(args.summary).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
