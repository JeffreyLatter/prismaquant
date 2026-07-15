import importlib.util
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_runtime_flags_doc_covers_live_prismaquant_flags():
    flags: set[str] = set()
    pattern = re.compile(r"PRISMAQUANT_[A-Z0-9_]+")
    for path in (ROOT / "prismaquant").rglob("*.py"):
        flags.update(pattern.findall(path.read_text(encoding="utf-8")))

    doc = _read("docs/runtime_flags.md")
    missing = sorted(flag for flag in flags if flag not in doc)
    assert not missing
    assert "| `PRODUCTION_RENDER_COST_SCORE_FIELD` | `weight_mse` |" in doc
    assert "`joint_mse` is the production JSO scale rule" in doc
    assert "H_DETAIL_DIR" not in doc


def test_package_readme_entrypoints_resolve_to_live_modules():
    text = _read("prismaquant/README.md")
    modules = re.findall(r"`python -m (prismaquant\.[A-Za-z0-9_]+)`", text)
    assert modules
    assert "prismaquant.polish_from_assignment" not in modules
    missing = [module for module in modules if importlib.util.find_spec(module) is None]
    assert not missing
    assert "dated `archive/` walls" in text


def test_root_readme_architecture_status_matches_in_tree_profiles():
    text = _read("README.md")
    assert "DeepSeek-V4-Flash" in text and "vendored transformer" in text
    assert "**Gemma4**" in text
    assert "**LFM2.5**" in text
    assert "GLM-4" not in text
    assert "waiting on `transformers` class" not in text
    assert "blocked on transformers" not in text


def test_audit_notes_are_not_root_level_and_scratch_is_local_only():
    assert not (ROOT / "audit_findings.md").exists()
    assert not (ROOT / "audit_questions.md").exists()
    assert (ROOT / "docs/audit_findings_2026-05-22.md").exists()
    assert (ROOT / "docs/audit_questions_2026-05-22.md").exists()
    assert not (ROOT / "scratch/smoke_graph_memory.py").exists()
    assert (ROOT / "tools/smoke_graph_memory.py").exists()


def test_claude_does_not_overstate_pipeline_enforcement():
    text = _read("CLAUDE.md")

    assert "structurally enforced\n   (`pipeline.py` `APPROVED_RESOURCE_OWNERS`)" not in text
    assert "declarative spec + owner validation, not executor" in text
    assert "runtime enforcement lives in the stage code" in text
