from pathlib import Path


def test_production_recache_default_enabled_after_smoke_ladder():
    script = (
        Path(__file__).resolve().parent.parent / "prismaquant" / "run-pipeline.sh"
    ).read_text()

    assert "PRODUCTION_CACHE:=1" in script
    assert "PRODUCTION_RECACHE:=1" in script
    assert "PRODUCTION_RECACHE=0" in script
    assert "PIPELINE_SPEC_PATH:=${WORK_DIR}/artifacts/pipeline_spec.json" in script
    assert "python3 -m prismaquant.pipeline" in script
    assert "--write-default-production" in script
    assert "--target-profile \"$TARGET_PROFILE\"" in script
    assert ': "${HADAMARD_DUQUANT' not in script
    assert "HADAMARD_DUQUANT:-" in script
    assert "archive/hdq_2026-05-14" in script
