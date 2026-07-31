"""Mechanical half of the ARCHITECTURE.md maintenance contract (CLAUDE.md §4
principle 13, AGENTS.md rule 10): the master document's defaults table must
match `prismaquant/run-pipeline.sh`, and its structural anchors must exist.

The judgment half — prose describing behavior that changed — cannot be tested;
this file only makes silent drift of the enumerable facts impossible.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _doc() -> str:
    return (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")


def _pipeline() -> str:
    return (ROOT / "prismaquant" / "run-pipeline.sh").read_text(encoding="utf-8")


def _shell_default(script: str, name: str) -> str:
    m = re.search(rf"\$\{{{name}:=([^}}]*)\}}", script)
    assert m, f"{name} has no ':=' default in run-pipeline.sh"
    return m.group(1)


def test_architecture_doc_exists_with_provenance_stamp():
    doc = _doc()
    assert doc.startswith("# PrismaQuant Architecture")
    assert re.search(r"As of: \d{4}-\d{2}-\d{2}", doc), "provenance stamp missing"
    assert "## 0. Maintenance contract" in doc


def test_defaults_table_matches_run_pipeline():
    doc, script = _doc(), _pipeline()
    for var in (
        "FORMATS",
        "TARGET_BITS",
        "COST_MODE",
        "NSAMPLES",
        "SEQLEN",
        "PRODUCTION_CACHE_LEVERS",
    ):
        val = _shell_default(script, var)
        assert f"{var}={val}" in doc, (
            f"ARCHITECTURE.md §3.3 is stale: run-pipeline.sh has {var}={val}. "
            "Update the defaults table in the same commit as the default change."
        )


def test_target_profile_has_no_shell_default():
    """re-vet R11: TARGET_PROFILE must stay UNSET so the architecture's own
    `spec.default_serving_profile` can win. A `:=` default here silently beat
    every spec (measured: 226 Hy3 FP8 Linears -> BF16, 2026-07-11), so this
    pins the absence of one and requires the doc to say how it resolves."""
    script, doc = _pipeline(), _doc()
    assert re.search(r'\$\{TARGET_PROFILE:=\}', script), (
        "TARGET_PROFILE must have an EMPTY ':=' default in run-pipeline.sh; "
        "an architecture's spec.default_serving_profile can never win against "
        "an explicit request (serving_profiles.resolve_target_profile)."
    )
    assert f"TARGET_PROFILE_DEFAULT={_shell_default(script, 'TARGET_PROFILE_DEFAULT')}" in doc
    assert "spec-resolved" in doc, (
        "ARCHITECTURE.md §3.3 must document that TARGET_PROFILE is "
        "spec-resolved rather than shell-defaulted."
    )


def test_selection_mode_default_documented():
    """SELECTION_MODE is no longer a single ':=' default — it is surrogate,
    or validated-surrogate under a byte budget (re-vet R1)."""
    script, doc = _pipeline(), _doc()
    assert 'SELECTION_MODE:=validated-surrogate' in script
    assert 'SELECTION_MODE:=surrogate' in script
    assert "SELECTION_MODE=surrogate" in doc
    assert "TARGET_DISK_GB" in doc


def test_three_diagrams_present():
    assert _doc().count("```mermaid") == 3


def test_docs_index_leads_with_architecture():
    readme = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    assert "ARCHITECTURE.md" in readme.split("\n\n")[0] or "ARCHITECTURE.md" in readme[:500]


def test_normative_rule_files_reference_the_contract():
    for name in ("CLAUDE.md", "AGENTS.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "docs/ARCHITECTURE.md" in text, f"{name} lost the doc-sync rule"


def test_every_archive_wall_has_a_banner_readme():
    walls = [p for p in (ROOT / "archive").iterdir() if p.is_dir()]
    assert walls
    missing = [w.name for w in walls if not (w / "README.md").exists()]
    assert not missing, f"archive walls without a banner README: {missing}"
