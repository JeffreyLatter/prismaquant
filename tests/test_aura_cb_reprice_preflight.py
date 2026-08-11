"""CPU-only contract tests for the AURA-on-CB campaign driver."""
from __future__ import annotations

from pathlib import Path

from tools.aura_cb_reprice_preflight import (
    DSV4_EXPERT_UNITS,
    DSV4_NONEXPERT_UNITS,
    _config_is_dense,
    _dsv4_campaign_worker_check,
    campaign_accounting,
    dsv4_checks,
    repository_capability_checks,
)


ROOT = Path(__file__).resolve().parents[1]


def test_source_rate_domain_is_not_demand_truncated():
    plan = campaign_accounting()

    assert DSV4_EXPERT_UNITS == 43 * 256 * 3 == 33_024
    assert DSV4_NONEXPERT_UNITS == 301
    assert plan["total_units"] == 33_325
    assert plan["expert_2048x4096_units"] == 22_016
    assert plan["expert_4096x2048_units"] == 11_008
    nvfp4_formats = [f"NVFP4_CB_K{k}" for k in range(12, 19)]
    expert_fp8_formats = [
        f"FP8_CB_K{k}" for k in range(28, 34)
    ]
    nonexpert_fp8_formats = [
        f"FP8_CB_K{k}" for k in range(28, 49)
    ]
    assert plan["expert_formats"] == [
        *nvfp4_formats,
        *expert_fp8_formats,
        "MXFP4_SOURCE",
    ]
    assert plan["nonexpert_formats"] == [
        *nvfp4_formats,
        *nonexpert_fp8_formats,
        "FP8_BLOCK_UE8M0_SOURCE",
    ]
    assert plan["nvfp4_cells"] == 233_275
    assert plan["expert_fp8_cells"] == 198_144
    assert plan["nonexpert_fp8_cells"] == 6_321
    assert plan["fp8_learned_cells"] == 203_863
    assert plan["fp8_lattice_cells"] == 602
    assert plan["fp8_cells"] == 204_465
    assert plan["source_terminal_cells"] == 33_325
    assert plan["expert_cells"] == 462_336
    assert plan["nonexpert_cells"] == 8_729
    assert plan["candidate_cells"] == 471_065
    assert plan["segment_key_fields"] == [
        "family", "role", "equivalence_class",
    ]
    assert plan["plugin_equivalence_vocabulary"] == "codebook_basis"
    all_role_counts = {
        "gate_proj": 11_051,
        "up_proj": 11_051,
        "down_proj": 11_051,
        "wq_a": 43,
        "wq_b": 43,
        "wkv": 43,
        "wo_b": 43,
    }
    lattice_role_counts = {role: 43 for role in all_role_counts}
    expected_segments = {
        (family, role, basis): renders
        for family, basis, role_counts in (
            ("NVFP4_CB", "lattice", all_role_counts),
            ("FP8_CB", "learned", all_role_counts),
            ("FP8_CB", "lattice", lattice_role_counts),
        )
        for role, renders in role_counts.items()
    }
    observed_segments = {
        (
            row["family"], row["role"], row["equivalence_class"],
        ): row["renders"]
        for row in plan["anchor_segments"]
    }
    assert all(
        row["basis"] == row["equivalence_class"]
        for row in plan["anchor_segments"]
    )
    assert len(plan["anchor_segments"]) == len(observed_segments) == 21
    assert observed_segments == expected_segments
    assert all(
        row["units"] == row["renders"] for row in plan["anchor_segments"]
    )
    by_family_basis = {
        family_basis: sum(
            renders
            for (family, _role, basis), renders in observed_segments.items()
            if (family, basis) == family_basis
        )
        for family_basis in {
            (family, basis) for family, _role, basis in observed_segments
        }
    }
    assert by_family_basis == {
        ("NVFP4_CB", "lattice"): 33_325,
        ("FP8_CB", "learned"): 33_325,
        ("FP8_CB", "lattice"): 301,
    }
    assert plan["anchor_renders"] == 66_951
    assert "66,951 legal unit-family-equivalence anchors" in plan[
        "encode_seconds_formula"
    ]


def test_current_branch_is_fail_closed_on_known_dsv4_gaps():
    checks = {
        check.name: check
        for check in repository_capability_checks(ROOT, "dsv4")
    }

    # The allocator supersurrogate contract and external receipt remain fail-closed.
    for name in (
        "AURA supersurrogate allocator semantics",
        "commit-bound implementation receipt",
    ):
        assert checks[name].status == "BLOCK", (name, checks[name].detail)

    # Closed by the streamed, anchored DSv4 campaign worker and its bounded render plan.
    assert checks["DSv4 bounded campaign worker"].status == "PASS", (
        checks["DSv4 bounded campaign worker"].detail
    )

    # Closed by identity-bound per-source-class --format-plan plumbing across every cost path.
    assert checks["source-class split format plan"].status == "PASS", (
        checks["source-class split format plan"].detail
    )

    # Closed by the boundary-activation streamed adjoint plus its existing
    # qname-keyed, identity-bound per-Linear journal.
    assert checks["checkpointed KL-adjoint"].status == "PASS", (
        checks["checkpointed KL-adjoint"].detail
    )

    # Closed by pinning one expert layer around the unchanged fp32 unit-KL
    # loop and atomically publishing one identity-bound shard per serving unit.
    assert checks["checkpointed streamed expert unit-KL"].status == "PASS", (
        checks["checkpointed streamed expert unit-KL"].detail
    )

    # Closed for DSv4 too by e3516cc/0c54067 (identity-bound CB pair-shard
    # resume), hardened by 49d0edc/3a5ec22 (git-less, whole-package producer
    # identity). Asserted positively rather than dropped from the list, so the
    # test keeps covering it and a regression back to BLOCK is caught.
    assert checks["CB cache per-unit resume"].status == "PASS", (
        checks["CB cache per-unit resume"].detail
    )

    # Closed by profile-declared routed-expert discovery in both the smooth
    # AURA exclusion and empirical unit-KL path. Asserted positively rather
    # than dropped, so a regression back to rank-keyed skipping is caught.
    assert checks["DSv4 hybrid key-space"].status == "PASS", (
        checks["DSv4 hybrid key-space"].detail
    )

    # Closed 2026-08-11 by the streamed CB format-menu render path, which lets
    # the cost stage render->score->discard instead of retaining every rung
    # (a menu rung costs 2.002 B/qparam of cache, K-independent, so a 21-rung
    # 27B menu was ~619 GiB). Asserted positively rather than dropped.
    #
    # NOTE: this preflight check only verifies the *rejection* is gone -- it
    # cannot verify the render is correct. tests/test_streamed_cb_format_menu.py
    # is what actually covers menu completeness and cache-retention policy; a
    # PASS here without that file passing means nothing.
    assert checks["streamed cached-menu render"].status == "PASS", (
        checks["streamed cached-menu render"].detail
    )


def test_bounded_worker_requires_a_concrete_invoked_orchestration_seam(tmp_path):
    package = tmp_path / "prismaquant"
    package.mkdir()
    worker = package / "dsv4_aura_cb_reprice.py"
    worker.write_text(
        "def main(argv=None):\n"
        "    from prismaquant.aura_cost import run_dsv4_anchor_campaign\n"
        "    return run_dsv4_anchor_campaign(argv, control_plane=__name__)\n"
    )

    absent = _dsv4_campaign_worker_check(tmp_path)
    assert absent.status == "BLOCK"
    assert "provider module is missing" in absent.detail

    provider = package / "aura_cost.py"
    provider.write_text(
        "def run_dsv4_anchor_campaign(args, *, control_plane):\n"
        "    raise NotImplementedError('not orchestrated')\n"
    )
    placeholder = _dsv4_campaign_worker_check(tmp_path)
    assert placeholder.status == "BLOCK"
    assert "only a placeholder" in placeholder.detail

    provider.write_text(
        "def _orchestrate(args, *, control_plane):\n"
        "    return 0\n\n"
        "def run_dsv4_anchor_campaign(args, *, control_plane):\n"
        "    return _orchestrate(args, control_plane=control_plane)\n"
    )
    ready = _dsv4_campaign_worker_check(tmp_path)
    assert ready.status == "PASS", ready.detail
    assert "concrete prismaquant.aura_cost.run_dsv4_anchor_campaign" in ready.detail

    worker.write_text(
        "def _orchestrate(args, *, control_plane):\n"
        "    return 0\n\n"
        "def run_dsv4_anchor_campaign(args, *, control_plane):\n"
        "    return _orchestrate(args, control_plane=control_plane)\n\n"
        "def main(argv=None):\n"
        "    return run_dsv4_anchor_campaign(argv, control_plane=__name__)\n"
    )
    local_ready = _dsv4_campaign_worker_check(tmp_path)
    assert local_ready.status == "PASS", local_ready.detail
    assert "concrete local run_dsv4_anchor_campaign" in local_ready.detail


def test_split_plan_preflight_includes_production_render_pricing(tmp_path):
    package = tmp_path / "prismaquant"
    package.mkdir()
    for name in (
        "build_production_cache.py",
        "aura_cost.py",
        "expert_empirical_cost.py",
    ):
        (package / name).write_text('FLAG = "--format-plan"\n')
    production_render = package / "production_render_cost.py"
    production_render.write_text(
        'FLAG = "--format-plan"\n'
        'IDENTITY = "format_plan_identity_sha256"\n'
        'SCOPE = "planned_scope"\n'
    )

    checks = {
        check.name: check
        for check in repository_capability_checks(tmp_path, "dsv4")
    }
    assert checks["source-class split format plan"].status == "PASS"
    assert "production-render pricing" in checks[
        "source-class split format plan"
    ].detail

    production_render.write_text(
        'FLAG = "--format-plan"\n'
        'IDENTITY = "format_plan_identity_sha256"\n'
    )
    checks = {
        check.name: check
        for check in repository_capability_checks(tmp_path, "dsv4")
    }
    assert checks["source-class split format plan"].status == "BLOCK"


def test_dsv4_preflight_refuses_work_inside_comparison_baseline(tmp_path):
    run_root = tmp_path / "run"
    baseline_parent = run_root / "prod-cal-0p7"
    checks = {
        check.name: check
        for check in dsv4_checks(
            ROOT,
            run_root,
            baseline_parent,
            tmp_path / "dataset.jsonl",
            tmp_path / "gpu.lock",
            None,
            None,
            None,
            verify_hashes=False,
        )
    }
    assert checks["DSv4 work directory"].status == "BLOCK"
    assert "Track A baseline" in checks["DSv4 work directory"].detail


def test_dense_path_does_not_require_model_streaming_but_does_require_resume():
    checks = {
        check.name: check
        for check in repository_capability_checks(ROOT, "dense")
    }

    # Dense never needed model streaming, and the two resume capabilities
    # landed 2026-08-11 (see the dsv4 test above for the commit chain).
    assert checks["streamed cached-menu render"].status == "PASS"
    assert checks["CB cache per-unit resume"].status == "PASS"
    assert checks["checkpointed KL-adjoint"].status == "PASS"
    # The receipt stays BLOCK by design: it is an external, HEAD-bound run
    # input, never satisfied by repository state alone.
    assert checks["commit-bound implementation receipt"].status == "BLOCK"


def test_dense_config_gate_rejects_routing_without_model_loading():
    dense, dense_detail = _config_is_dense(
        {
            "model_type": "qwen3_8",
            "architectures": ["Qwen3_8ForCausalLM"],
        }
    )
    moe, moe_detail = _config_is_dense(
        {
            "model_type": "qwen3_8_moe",
            "architectures": ["Qwen3_8MoeForCausalLM"],
            "num_experts": 128,
        }
    )

    assert dense, dense_detail
    assert not moe
    assert "num_experts" in moe_detail


def test_driver_takes_atomic_mutex_before_any_gpu_container():
    driver = (ROOT / "tools/run_aura_cb_reprice.sh").read_text()
    preflight = (ROOT / "tools/aura_cb_reprice_preflight.py").read_text()

    assert "nvidia-smi" not in driver
    assert "flock -x 9" in driver
    lock_at = driver.index("flock -x 9")
    gpu_container_at = driver.index('run --rm --name "$CONTAINER_NAME" --gpus all')
    assert lock_at < gpu_container_at
    assert driver.count("run_preflight") >= 4
    assert "PRISMAQUANT_CB_ENCODE_COMPILE" in driver
    assert "AURA_CB_LAUNCH_RECEIPT" in driver
    assert "--implementation-receipt" in driver
    assert 'CALIB_NSAMPLES="${CALIB_NSAMPLES:-16}"' in driver
    assert 'CALIB_SEQLEN="${CALIB_SEQLEN:-512}"' in driver
    assert 'CALIB_SEED="${CALIB_SEED:-42}"' in driver
    assert "--resume" in driver
    assert 'if [[ ! -s "$CACHE_MANIFEST" ]]' not in driver
    assert 'if [[ ! -s "$COST_OUTPUT" ]]' not in driver
    assert "run-pipeline.sh" not in driver  # cost-only, no accidental export
    assert "\nimport torch" not in preflight
    assert "\nfrom torch" not in preflight
    assert "\nimport transformers" not in preflight
    assert "\nfrom transformers" not in preflight
