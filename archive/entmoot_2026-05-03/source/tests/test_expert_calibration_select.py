"""Tests for expert-balanced calibration sample selection."""
from __future__ import annotations

import json

import pytest

from prismaquant.expert_calibration_select import (
    SampleCoverage,
    coverage_lift,
    first_n_baseline_result,
    load_survey_jsonl,
    sample_from_mapping,
    select_expert_balanced_samples,
    write_selected_jsonl,
)


def _sample(sample_id, hits, domain="general"):
    return SampleCoverage(sample_id=sample_id, domain=domain, hits=hits)


def test_selector_prefers_rare_expert_coverage_over_common_mass():
    samples = [
        _sample("common-0", {"R": {0: 10.0}}),
        _sample("common-1", {"R": {0: 10.0}}),
        _sample("rare-1", {"R": {1: 1.0}}),
        _sample("rare-2", {"R": {2: 1.0}}),
    ]

    result = select_expert_balanced_samples(samples, budget=2)

    assert result.selected_ids == ["rare-1", "rare-2"]
    summary = result.coverage_summary()
    assert summary["available_pairs"] == 3
    assert summary["covered_pairs"] == 2


def test_first_n_baseline_reports_selector_lift():
    samples = [
        _sample("common-0", {"R": {0: 10.0}}),
        _sample("common-1", {"R": {0: 10.0}}),
        _sample("rare-1", {"R": {1: 1.0}}),
        _sample("rare-2", {"R": {2: 1.0}}),
    ]
    selected = select_expert_balanced_samples(samples, budget=2)
    baseline = first_n_baseline_result(
        samples,
        budget=2,
        available_mass=selected.available_mass,
    )

    lift = coverage_lift(selected.coverage_summary(), baseline.coverage_summary())

    assert baseline.selected_ids == ["common-0", "common-1"]
    assert lift["covered_pairs"] == 1
    assert lift["mean_pair_fraction"] > 0.0


def test_selector_respects_domain_minimum_when_slots_are_tight():
    samples = [
        _sample("code-0", {"R": {0: 1.0}}, domain="code"),
        _sample("code-1", {"R": {1: 1.0}}, domain="code"),
        _sample("code-2", {"R": {2: 1.0}}, domain="code"),
        _sample("math-0", {"R": {3: 1.0}}, domain="math"),
        _sample("math-1", {"R": {4: 1.0}}, domain="math"),
    ]

    result = select_expert_balanced_samples(
        samples,
        budget=3,
        min_domain_counts={"math": 2},
    )

    assert result.domain_counts["math"] == 2
    assert len(result.selected) == 3


def test_selector_tie_breaks_by_sample_id():
    samples = [
        _sample("c", {"R": {0: 1.0}}),
        _sample("b", {"R": {0: 1.0}}),
        _sample("a", {"R": {0: 1.0}}),
    ]

    result = select_expert_balanced_samples(samples, budget=2)

    assert result.selected_ids == ["a", "b"]


def test_sample_from_mapping_accepts_list_hit_shape_and_infers_domain():
    sample = sample_from_mapping(
        {
            "id": "row-1",
            "source": "/tmp/chunk_math-hard_03.jsonl",
            "hits": {
                "R": [
                    {"expert_id": "7", "mass": "0.25"},
                    [8, 0.5],
                    {"expert": 9, "count": 2},
                ],
            },
        },
        fallback_id="fallback",
    )

    assert sample.sample_id == "row-1"
    assert sample.domain == "math-hard"
    assert sample.hits["R"] == {7: 0.25, 8: 0.5, 9: 2.0}


def test_jsonl_roundtrip_ids_only(tmp_path):
    survey = tmp_path / "survey.jsonl"
    survey.write_text(
        "\n".join([
            json.dumps({"id": "a", "domain": "x", "hits": {"R": {"0": 1}}}),
            json.dumps({"id": "b", "domain": "x", "hits": {"R": {"1": 1}}}),
        ])
        + "\n",
        encoding="utf-8",
    )

    samples = load_survey_jsonl(survey)
    result = select_expert_balanced_samples(samples, budget=1)
    output = tmp_path / "selected.jsonl"
    write_selected_jsonl(result, output, ids_only=True)

    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert rows == [{"id": "a"}]


def test_negative_budget_raises():
    with pytest.raises(ValueError, match="budget"):
        select_expert_balanced_samples([], budget=-1)
