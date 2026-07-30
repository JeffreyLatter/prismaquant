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
        "SELECTION_MODE",
        "TARGET_PROFILE",
        "NSAMPLES",
        "SEQLEN",
        "PRODUCTION_CACHE_LEVERS",
    ):
        val = _shell_default(script, var)
        assert f"{var}={val}" in doc, (
            f"ARCHITECTURE.md §3.3 is stale: run-pipeline.sh has {var}={val}. "
            "Update the defaults table in the same commit as the default change."
        )


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
