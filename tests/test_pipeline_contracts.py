from __future__ import annotations

from prismaquant.pipeline import (
    ArtifactSpec,
    MetricGateSpec,
    PipelineComponentSpec,
    compose_pipeline_spec,
    PipelineSpec,
    PipelineStageSpec,
    ResourceContract,
    default_production_pipeline_spec,
    load_pipeline_spec,
    main,
    parse_render_mechanisms,
    production_pipeline_spec_from_config,
    render_mechanism_stage_specs,
    write_pipeline_spec,
)


def test_metric_gate_selects_only_improved_linears():
    gate = MetricGateSpec(
        name="mse_improves",
        metric="output_mse",
        mode="per_item",
        direction="lower_is_better",
    )

    result = gate.evaluate(
        baseline={
            "layer.a": {"output_mse": 1.0},
            "layer.b": {"output_mse": 1.0},
            "layer.c": {"output_mse": 1.0},
        },
        candidate={
            "layer.a": {"output_mse": 0.9},
            "layer.b": {"output_mse": 1.0},
            "layer.c": {"output_mse": 1.1},
        },
    )

    assert result.passed is True
    assert result.accepted_keys() == ("layer.a",)
    assert result.rejected_keys() == ("layer.b", "layer.c")


def test_metric_gate_can_enforce_global_validation_improvement():
    gate = MetricGateSpec(
        name="kl_improves",
        metric="end_kl",
        mode="all",
        direction="lower_is_better",
    )

    passed = gate.evaluate(
        baseline={"end_kl": 0.12},
        candidate={"end_kl": 0.10},
    )
    failed = gate.evaluate(
        baseline={"end_kl": 0.12},
        candidate={"end_kl": 0.12},
    )

    assert passed.passed is True
    assert passed.accepted_keys() == ("__global__",)
    assert failed.passed is False
    assert failed.decisions[0].reason == "regressed_or_tied"


def test_metric_gate_can_allow_bounded_metric_regression():
    gate = MetricGateSpec(
        name="ppl_preserved",
        metric="ppl",
        mode="all",
        direction="lower_is_better",
        require_improvement=False,
        max_relative_regression=0.005,
    )

    tolerated = gate.evaluate(
        baseline={"ppl": 10.0},
        candidate={"ppl": 10.04},
    )
    rejected = gate.evaluate(
        baseline={"ppl": 10.0},
        candidate={"ppl": 10.10},
    )

    assert tolerated.passed is True
    assert tolerated.decisions[0].reason == "within_regression_budget"
    assert rejected.passed is False


def test_render_mechanisms_are_exposed_as_ordered_pipeline_stages():
    stages = render_mechanism_stage_specs((
        "gptq",
        "joint_scale_opt",
        "four_over_six",
    ))

    assert tuple(stage.name for stage in stages) == (
        "render.four_over_six",
        "render.joint_scale_opt",
        "render.gptq",
    )
    assert all(
        stage.resources[0].owner == "ProductionWeightCache"
        for stage in stages
    )
    assert stages[0].gates == ("gate.render.output_mse",)


def test_default_production_pipeline_contract_validates():
    spec = default_production_pipeline_spec()
    result = spec.validate()

    assert result.ok is True
    assert result.errors == ()
    stages = {stage.name: stage for stage in spec.stages}
    assert stages["cache.prefetch_assignment"].resources[0] == ResourceContract(
        resource="rendered_weights",
        owner="ProductionWeightCache",
        residency="required",
    )
    assert any(
        resource.owner == "PerturbedActivationCache"
        for resource in stages["validate.kl"].resources
    )


def test_pipeline_spec_round_trips_through_json(tmp_path):
    path = tmp_path / "pipeline.json"
    spec = default_production_pipeline_spec(render_mechanisms=("gptq",))

    write_pipeline_spec(spec, path)
    loaded = load_pipeline_spec(path)

    assert loaded.to_dict() == spec.to_dict()
    assert loaded.validate().ok is True


def test_render_mechanisms_parse_env_style_config():
    mechanisms = parse_render_mechanisms(
        "gptq,joint_scale_opt, gptq",
        disabled="joint_scale_opt",
    )

    assert mechanisms == ("gptq",)


def test_production_pipeline_spec_records_run_config():
    spec = production_pipeline_spec_from_config(
        render_mechanisms="gptq,joint_scale_opt",
        model_path="/models/qwen",
        work_dir="/runs/qwen",
        formats="NVFP4,BF16",
        target_bits=4.75,
        target_profile="vllm_packed_moe",
        calibration_modality="text-only",
        selection_mode="surrogate",
        production_cache="1",
        production_recache="1",
    )

    assert spec.validate().ok is True
    assert spec.metadata["render_mechanisms"] == ["joint_scale_opt", "gptq"]
    assert spec.metadata["target_profile"] == "vllm_packed_moe"
    assert spec.metadata["formats"] == "NVFP4,BF16"


def test_pipeline_cli_writes_validated_default_spec(tmp_path):
    path = tmp_path / "pipeline_spec.json"

    rc = main([
        "--write-default-production",
        str(path),
        "--validate",
        "--render-mechanisms",
        "gptq",
        "--target-profile",
        "research",
    ])
    loaded = load_pipeline_spec(path)

    assert rc == 0
    assert path.exists()
    assert loaded.validate().ok is True
    assert loaded.metadata["render_mechanisms"] == ["gptq"]


def _synthetic_component(insert_after: str = "cache.prefetch_assignment"):
    return PipelineComponentSpec(
        id="synthetic_research",
        artifacts=(ArtifactSpec(
            "synthetic_layer_assignment",
            "layer_config",
            version="synthetic.v1",
        ),),
        gates=(MetricGateSpec(
            name="gate.synthetic.kl",
            metric="last_token_kl",
            direction="lower_is_better",
        ),),
        stages=(PipelineStageSpec(
            name="research.synthetic.validate",
            component="synthetic:validate",
            inputs=("resident_production_weight_cache",),
            outputs=("synthetic_layer_assignment",),
            gates=("gate.synthetic.kl",),
            resources=(ResourceContract(
                resource="rendered_weights",
                owner="ProductionWeightCache",
                residency="required",
            ),),
        ),),
        insert_after=insert_after,
        status="research",
        default_enabled=False,
    )


def test_opt_in_component_composes_after_production_cache_prefetch():
    component = _synthetic_component()
    spec = production_pipeline_spec_from_config(
        render_mechanisms="gptq",
        components=(component,),
    )

    assert spec.validate().ok is True
    stage_names = [stage.name for stage in spec.stages]
    prefetch_idx = stage_names.index("cache.prefetch_assignment")
    synthetic_idx = stage_names.index("research.synthetic.validate")
    validate_idx = stage_names.index("validate.kl")

    assert prefetch_idx < synthetic_idx < validate_idx
    assert "synthetic_layer_assignment" in spec.artifact_map()
    assert spec.metadata["components"] == [{
        "id": "synthetic_research",
        "status": "research",
        "default_enabled": False,
    }]


def test_pipeline_cli_lists_no_shelved_research_components(capsys):
    list_rc = main(["--list-components"])
    listed = capsys.readouterr().out

    assert list_rc == 0
    assert listed == ""


def test_compose_pipeline_spec_rejects_missing_component_insert_point():
    base = default_production_pipeline_spec(render_mechanisms=())
    bad_component = _synthetic_component(insert_after="missing.stage")

    try:
        compose_pipeline_spec(base, (bad_component,))
    except ValueError as exc:
        assert "insert_after stage 'missing.stage' was not found" in str(exc)
    else:
        raise AssertionError("expected missing insert point to fail")


def test_pipeline_validation_rejects_parallel_rendered_weight_cache():
    spec = PipelineSpec(
        id="bad",
        artifacts=(ArtifactSpec("source", "model", provided=True),),
        stages=(PipelineStageSpec(
            name="bad.cache",
            component="bad",
            inputs=("source",),
            outputs=("cache",),
            resources=(ResourceContract(
                resource="rendered_weights",
                owner="AdHocCache",
                residency="required",
            ),),
        ),),
    )

    result = spec.validate()
    assert result.ok is False
    assert any("rendered_weights must use" in error for error in result.errors)


def test_pipeline_validation_rejects_parallel_activation_cache():
    spec = PipelineSpec(
        id="bad-activation-cache",
        artifacts=(ArtifactSpec("source", "model", provided=True),),
        stages=(PipelineStageSpec(
            name="bad.activation",
            component="bad",
            inputs=("source",),
            outputs=("acts",),
            resources=(ResourceContract(
                resource="perturbed_activations",
                owner="AdHocActivationCache",
                residency="required",
            ),),
        ),),
    )

    result = spec.validate()
    assert result.ok is False
    assert any("perturbed_activations must use" in error for error in result.errors)


def test_pipeline_validation_requires_inputs_to_be_available_in_order():
    spec = PipelineSpec(
        id="bad-order",
        artifacts=(
            ArtifactSpec("source", "model", provided=True),
            ArtifactSpec("late", "payload"),
        ),
        stages=(
            PipelineStageSpec(
                name="uses.late",
                component="consumer",
                inputs=("source", "late"),
                outputs=("out",),
            ),
            PipelineStageSpec(
                name="produces.late",
                component="producer",
                inputs=("source",),
                outputs=("late",),
            ),
        ),
    )

    result = spec.validate()
    assert result.ok is False
    assert "uses.late: input 'late' is not available" in result.errors
