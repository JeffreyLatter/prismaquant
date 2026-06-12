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
    assert "| `PRODUCTION_RENDER_COST_SCORE_FIELD` | `output_mse` |" in doc
    assert "`joint_mse` is the production JSO scale rule" in doc


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
    assert "DeepSeek-V4-Flash** (vendored transformer + profile)" in text
    assert "**Gemma4**" in text
    assert "**LFM2.5**" in text
    assert "GLM-4" not in text
    assert "waiting on `transformers` class" not in text
    assert "blocked on transformers" not in text
