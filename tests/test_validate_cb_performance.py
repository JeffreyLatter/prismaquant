"""Fail-closed tests for the paired Gridbook serving-performance gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import prismaquant.validate_cb_performance as performance_validator
from prismaquant.gridbook_runtime_pin import load_gridbook_runtime_pin
from prismaquant.shipcard import (
    SHIPCARD_RESERVED_BYTES,
    _verify_gridbook_performance_record,
    build_shipcard,
    build_weight_content_manifest,
    compute_model_sha,
    git_provenance,
    load_shipcard,
    write_shipcard,
)
from prismaquant.validate_cb_endpoint import (
    DSV4_SPARK_GPU_NAME,
    DSV4_SPARK_VLLM_IMAGE,
    DSV4_SPARK_VLLM_VERSION,
)
from prismaquant.validate_cb_performance import (
    CBPerformanceValidationError,
    DISPLACED_CONTAINER_SCHEMA,
    EVIDENCE_SCHEMA,
    MANIFEST_SCHEMA,
    RESULT_SCHEMA,
    SLOT,
    TELEMETRY_SCHEMA,
    TOOL,
    validate_cb_performance,
)
from tools.serve_fingerprint import elide_argv_paths, fingerprint


_SOURCE_IDENTITY = {
    "schema": "prismaquant.streamed_model.identity.v1",
    "content_sha256": "1" * 64,
    "resolved_commit": "deepseek-v4-source-revision",
    "checkpoint_shards": 1,
    "checkpoint_tensors": 1,
}


_ALLOWED_DIFFERENCES = [
    "artifacts.model_id",
    "artifacts.whole_served_artifact_bytes",
    "artifacts.budget_headroom_bytes",
    "artifacts.payload",
    "artifacts.inventory",
    "execution_identity.format_rung",
    "execution_identity.serialization",
    "execution_identity.quant_contract",
    "execution_identity.kernel_backend",
    "execution_identity.fallback_state",
    "execution_identity.manifest",
    "server.base_url",
    "server.served_model_name",
]


def _write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory_sha(inventory: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            inventory,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _finalize_test_inventory(root: Path, quant_config: dict, budget: int) -> None:
    quant_path = root / "quant_config.json"
    for _ in range(16):
        _write_json(quant_path, quant_config)
        file_bytes = {
            path.relative_to(root).as_posix(): path.stat().st_size
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }
        inventory = {
            "schema": "prismaquant.cb_export_artifact_inventory.v1",
            "scope": "all_regular_files_recursive",
            "file_bytes": file_bytes,
            "export_directory_bytes": sum(file_bytes.values()),
            "whole_artifact_budget_bytes": budget,
        }
        if quant_config["provenance"].get("artifact_inventory") == inventory:
            return
        quant_config["provenance"]["artifact_inventory"] = inventory
    raise AssertionError("test inventory did not converge")


def _artifact(root: Path, budget: int) -> tuple[Path, str, int, str]:
    artifact = root / "artifact"
    artifact.mkdir()
    (artifact / "config.json").write_text(
        '{"model_type":"deepseek_v4","num_hidden_layers":43,'
        '"n_routed_experts":256}\n'
    )
    (artifact / "model.safetensors").write_bytes(b"test weights")
    quant_config = {
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
        "provenance": {
            "source_model_identity": dict(_SOURCE_IDENTITY),
            "artifact_inventory": {
                "schema": "prismaquant.cb_export_artifact_inventory.v1",
                "scope": "pending_final_write",
            }
        },
    }
    _write_json(artifact / "quant_config.json", quant_config)
    model_sha = compute_model_sha(artifact)
    shipcard = artifact / "shipcard.json"
    _write_json(
        shipcard,
        {
            "schema": "prismaquant.shipcard/1",
            "model_sha": model_sha,
            "build": {"quant_method": "gridbook"},
            "reserved_file_bytes": SHIPCARD_RESERVED_BYTES,
            "slots": {SLOT: None},
        },
    )
    write_shipcard(shipcard, load_shipcard(shipcard))
    _finalize_test_inventory(artifact, quant_config, budget)
    assert compute_model_sha(artifact) == model_sha
    inventory = quant_config["provenance"]["artifact_inventory"]
    inventory_sha = _inventory_sha(inventory)
    artifact_bytes = inventory["export_directory_bytes"]
    return shipcard, model_sha, artifact_bytes, inventory_sha


def _displaced_artifact(root: Path, budget: int) -> dict:
    artifact = root / "displaced-artifact"
    artifact.mkdir()
    (artifact / "config.json").write_text(
        '{"model_type":"deepseek_v4","num_hidden_layers":43,'
        '"n_routed_experts":256}\n'
    )
    (artifact / "model.safetensors").write_bytes(b"prior aura weights")
    assignment_sha = hashlib.sha256(b"prior-a-fast-assignment").hexdigest()
    layer_config_sha = hashlib.sha256(b"prior-a-fast-layer-config").hexdigest()
    quant_config = {
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
        "provenance": {
            "assignment_sha256": assignment_sha,
            "source_model_identity": dict(_SOURCE_IDENTITY),
            "weight_content_manifest": build_weight_content_manifest(artifact),
            "artifact_inventory": {
                "schema": "prismaquant.cb_export_artifact_inventory.v1",
                "scope": "pending_final_write",
            },
        },
    }
    _write_json(artifact / "quant_config.json", quant_config)
    card = build_shipcard(
        artifact,
        build={
            "quant_method": "gridbook",
            "layer_config_sha": layer_config_sha,
        },
    )
    for arm in ("eager", "graph"):
        card["slots"][f"native_export.{arm}"] = {
            "slot": f"native_export.{arm}",
            "model_sha": card["model_sha"],
            "passed": True,
        }
    write_shipcard(artifact / "shipcard.json", card)
    _finalize_test_inventory(artifact, quant_config, budget)
    model_sha = compute_model_sha(artifact)
    assert model_sha == card["model_sha"]
    inventory = quant_config["provenance"]["artifact_inventory"]
    inventory_sha = _inventory_sha(inventory)
    artifact_bytes = inventory["export_directory_bytes"]
    model_id = "prior-a-fast-raw-cb@exact"
    reason = "on-disk A-FAST raw control displaced by the Aura DSv4Flash assignment"
    receipt = {
        "schema": "prismaquant.displaced_assignment/1",
        "status": "eligible",
        "mechanism": "a_fast_raw_control",
        "reason": reason,
        "model_id": model_id,
        "model_sha": model_sha,
        "assignment_sha256": assignment_sha,
        "artifact_inventory_sha256": inventory_sha,
        "artifact_bytes": artifact_bytes,
        "byte_budget": budget,
        "source_model_identity": dict(_SOURCE_IDENTITY),
        "layer_config_sha256": layer_config_sha,
        "cost_currency": "weight_mse",
    }
    receipt_path = root / "prior-displaced-assignment.json"
    receipt_sha = _write_json(receipt_path, receipt)
    identity = {
        "model_id": model_id,
        "model_sha": model_sha,
        "artifact_inventory_sha256": inventory_sha,
        "artifact_bytes": artifact_bytes,
    }
    declaration = {
        **identity,
        "artifact_dir": str(artifact.relative_to(root)),
        "assignment_sha256": assignment_sha,
        "mechanism": "a_fast_raw_control",
        "reason": reason,
        "assignment_receipt": {
            "path": str(receipt_path.relative_to(root)),
            "sha256": receipt_sha,
        },
    }
    return {
        "root": artifact,
        "identity": identity,
        "declaration": declaration,
        "receipt_path": receipt_path,
    }


def _metrics(*, candidate: bool) -> dict:
    latency = [9.0, 9.1, 8.9] if candidate else [10.0, 10.1, 9.9]
    throughput = [110.0, 111.0, 109.0] if candidate else [100.0, 101.0, 99.0]
    rows = {}
    for name in ("p95_ttft_ms", "p95_tpot_ms", "p95_itl_ms", "p95_e2el_ms"):
        rows[name] = {"values": latency}
    for name in (
        "request_throughput",
        "output_throughput",
        "total_token_throughput",
    ):
        rows[name] = {"values": throughput}
    return rows


_SERVER_ENVIRONMENT = {
    "GRIDBOOK_MXFP8_DENSE": "1",
    "PRISMAQUANT_CB_DECODE": "cuda",
    "PRISMAQUANT_PRELOAD_FUSED": "1",
    "VLLM_USE_DEEP_GEMM": "0",
}


def _server_argv(*, model_id: str, chunked: bool, speculative: bool) -> list[str]:
    argv = [
        "/usr/local/bin/vllm",
        "serve",
        "/model",
        "--served-model-name",
        model_id,
        "--quantization",
        "gridbook",
        "--kv-cache-dtype",
        "fp8",
        "--tensor-parallel-size",
        "1",
        "--no-enable-prefix-caching",
        "--enable-chunked-prefill"
        if chunked
        else "--no-enable-chunked-prefill",
    ]
    if speculative:
        argv.extend(
            ["--speculative-config", '{"method":"mtp","num_speculative_tokens":3}']
        )
    return argv


def _serve_manifest_attachment(
    root: Path,
    *,
    cell_id: str,
    candidate: bool,
    identity: dict,
    chunked: bool,
    speculative: bool,
) -> dict:
    pin = load_gridbook_runtime_pin()
    argv = _server_argv(
        model_id=identity["model_id"],
        chunked=chunked,
        speculative=speculative,
    )
    payload = {
        "schema": "prismaquant.serve_manifest/1",
        "created": "2026-08-12T00:30:00Z",
        "source": "server",
        "hostname": "one-spark",
        "image": DSV4_SPARK_VLLM_IMAGE,
        "model": "/model",
        "served_model_name": identity["model_id"],
        "launch_argv": argv,
        "launch_flags": elide_argv_paths(argv),
        "enforce_eager": False,
        "quantization": "gridbook",
        "kv_cache_dtype": "fp8",
        "speculative_config": (
            '{"method":"mtp","num_speculative_tokens":3}'
            if speculative
            else None
        ),
        "package_versions": {
            "gridbook": pin.version,
            "vllm": DSV4_SPARK_VLLM_VERSION,
        },
        "gridbook_runtime_pin": {"commit": pin.commit, "version": pin.version},
        "resident_extensions": ["prismaquant_cb_v2_ext.test.so"],
        "residency_readable": True,
        "processes": [
            {
                "pid": 11001 if candidate else 22001,
                "cmdline": " ".join(argv),
            }
        ],
        "pq_env": dict(_SERVER_ENVIRONMENT),
        "gpu_name": DSV4_SPARK_GPU_NAME,
        "driver_version": "release-driver",
        "gpu_count": 1,
        "artifact_binding": {
            "schema": "prismaquant.served_artifact_binding/1",
            "model_sha": identity["model_sha"],
            "artifact_inventory_sha256": identity["artifact_inventory_sha256"],
            "artifact_bytes": identity["artifact_bytes"],
        },
    }
    payload["serve_fingerprint"] = fingerprint(payload)
    arm = "candidate" if candidate else "baseline"
    path = root / "serve-manifests" / f"{cell_id}-{arm}.json"
    digest = _write_json(path, payload)
    return {
        "reference": str(path.relative_to(root)),
        "sha256": digest,
        "bytes": path.stat().st_size,
    }


def _report(
    *,
    candidate: bool,
    model_id: str,
    inventory_sha: str,
    artifact_bytes: int,
    budget: int,
    concurrency: int,
    output_len: int,
    input_range_ratio: float,
    chunked: bool,
    speculative: bool,
    server_evidence: dict,
) -> dict:
    pin = load_gridbook_runtime_pin()
    base_url = "http://candidate" if candidate else "http://baseline"
    server_argv = _server_argv(
        model_id=model_id,
        chunked=chunked,
        speculative=speculative,
    )
    return {
        "schema": "gridbook.vllm-bench-serve.v2",
        "status": "success",
        "run_label": "candidate" if candidate else "baseline",
        "evidence_scope": "single-arm-serving-measurement",
        "measurement_valid": True,
        "parity_acceptance": False,
        "release_acceptance": False,
        "release_eligible": True,
        "started_at": "2026-08-12T01:00:00Z",
        "finished_at": "2026-08-12T01:10:00Z",
        "duration_s": 600.0,
        "metadata": {
            "git": {
                "commit": pin.commit,
                "dirty": False,
                "source": "checkout",
                "release_eligible": True,
            },
            "measurement_provenance": {
                "digest_bound_inputs_verified_before_requests": True,
                "digest_bound_inputs_verified_after_requests": True,
                "git_state_verified_after_requests": True,
                "client_runtime_verified_after_requests": True,
            },
            "software": {
                "gridbook_version": pin.version,
                "runner_vllm_cli_probe": "vLLM 0.23.1 pinned client",
                "python": "3.12.0",
                "platform": "Linux",
                "machine": "aarch64",
                "hostname": "one-spark",
            },
            "artifacts": {
                "image_id": DSV4_SPARK_VLLM_IMAGE,
                "model_id": model_id,
                "benchmark_model": "base-model@revision",
                "tokenizer": "tokenizer@revision",
                "whole_served_artifact_bytes": artifact_bytes,
                "byte_budget_bytes": budget,
                "budget_headroom_bytes": budget - artifact_bytes,
                "within_byte_budget": True,
                "payload": None,
                "inventory": {
                    "sha256": inventory_sha,
                    "computed_total_bytes": artifact_bytes,
                },
                "byte_scope": "whole served artifact",
            },
            "execution_identity": {
                "format_rung": "dsv4flash-cb" if candidate else "prior-a-fast-raw-cb",
                "serialization": {
                    "layout": "cb",
                    "scale_coding": "mixed",
                },
                "quant_contract": "mixed",
                "kernel_backend": "gridbook",
                "tensor_parallel_size": 1,
                "fallback_state": "none",
                "manifest": {
                    "schema": "gridbook.execution-manifest.v1",
                    "sha256": "b" * 64 if candidate else "c" * 64,
                    "artifact_inventory_sha256": inventory_sha,
                    "coverage": "all_serving_units",
                    "tensor_parallel_size": 1,
                    "assignment_count": 1,
                    "assignments": [
                        {
                            "unit": "model.layers.0.self_attn.q_proj",
                            "format_rung": "FP8_CB_K36" if candidate else "NVFP4_CB_K32",
                            "serialized_layout": "cb",
                            "scale_coding": "e4m3" if candidate else "ue4m3",
                            "quant_contract": "W8A8" if candidate else "W4A4",
                            "kernel_backend": "gridbook",
                            "fallback_state": "none",
                        }
                    ],
                },
                "client_runtime_id": "vLLM 0.23.1 pinned client",
                "server_runtime_id": DSV4_SPARK_VLLM_VERSION,
                "hardware": {
                    "gpu_id": DSV4_SPARK_GPU_NAME,
                    "driver_version": "release-driver",
                    "accelerator_runtime": "CUDA 13",
                },
            },
            "server": {
                "base_url": base_url,
                "backend": "openai-chat",
                "endpoint": "/v1/completions",
                "served_model_name": model_id,
                "recorded_args": server_argv,
                "prefix_caching": "off",
                "evidence": {
                    "scope": "digest-bound startup evidence",
                    "attachments": [server_evidence],
                },
            },
            "dispatch": {
                "runner_environment": {"source": "benchmark process", "values": {}},
                "server_environment": {
                    "source": "explicit --server-env arguments",
                    "values": dict(_SERVER_ENVIRONMENT),
                },
            },
            "workload": {
                "dataset": "random",
                "requested_random_input_len": 32,
                "observed_input_length_contract": {"mode": "exact", "value": 31},
                "output_len": output_len,
                "num_prompts_per_block": 16,
                "warmups_per_block": 4,
                "max_concurrency": concurrency,
                "blocks": 3,
                "dataset_base_seed": 1234,
                "dataset_block_seeds": [1234, 1235, 1236],
                "sampling": {"strategy": "greedy", "temperature": 0.0},
                "request_rate": "inf",
                "request_burstiness": 1.0,
                "input_range_ratio": input_range_ratio,
                "speculative_decoding": {
                    "mode": "on" if speculative else "off",
                    "config": {"num_speculative_tokens": 3} if speculative else None,
                },
            },
        },
        "blocks": [
            {
                "index": index + 1,
                "status": "success",
                "returncode": 0,
                "validation_error": None,
                "command": [
                    "vllm",
                    "bench",
                    "serve",
                    "--base-url",
                    base_url,
                    "--endpoint",
                    "/v1/completions",
                    "--model",
                    "base-model@revision",
                    "--served-model-name",
                    model_id,
                    "--tokenizer",
                    "tokenizer@revision",
                ],
                "raw_result": {
                    metric: row["values"][index]
                    for metric, row in _metrics(candidate=candidate).items()
                },
            }
            for index in range(3)
        ],
        "summary": {"completed_blocks": 3, "metrics": _metrics(candidate=candidate)},
    }


def _comparison(tmp_path: Path) -> dict:
    budget = 10_000_000
    shipcard, model_sha, candidate_bytes, candidate_inventory = _artifact(
        tmp_path, budget
    )
    displaced = _displaced_artifact(tmp_path, budget)
    baselines = {
        phase: dict(displaced["identity"])
        for phase in ("prefill", "decode", "mixed")
    }
    candidate_identity = {
        "model_id": "candidate@exact",
        "model_sha": model_sha,
        "artifact_inventory_sha256": candidate_inventory,
        "artifact_bytes": candidate_bytes,
    }
    specs = []
    for concurrency in (1, 2, 4, 8, 16):
        for chunked in (False, True):
            suffix = "on" if chunked else "off"
            specs.extend(
                [
                    (
                        f"prefill-c{concurrency}-{suffix}",
                        "prefill",
                        concurrency,
                        chunked,
                        None,
                        1,
                        0.25 if concurrency == 2 and chunked else 0.0,
                        False,
                    ),
                    (
                        f"mixed-c{concurrency}-{suffix}",
                        "mixed",
                        concurrency,
                        chunked,
                        None,
                        128,
                        0.0,
                        False,
                    ),
                    (
                        f"decode-c{concurrency}-shipped-{suffix}",
                        "decode",
                        concurrency,
                        chunked,
                        "shipped",
                        256,
                        0.0,
                        True,
                    ),
                ]
            )
    for chunked in (False, True):
        suffix = "on" if chunked else "off"
        specs.append(
            (f"decode-c1-plain-{suffix}", "decode", 1, chunked, "plain", 256, 0.0, False)
        )
    cells = []
    for cell_id, phase, concurrency, chunked, mode, output_len, ratio, spec in specs:
        candidate_path = tmp_path / "reports" / f"{cell_id}-candidate.json"
        baseline_path = tmp_path / "reports" / f"{cell_id}-baseline.json"
        candidate_server_evidence = _serve_manifest_attachment(
            tmp_path,
            cell_id=cell_id,
            candidate=True,
            identity=candidate_identity,
            chunked=chunked,
            speculative=spec,
        )
        candidate_digest = _write_json(
            candidate_path,
            _report(
                candidate=True,
                model_id="candidate@exact",
                inventory_sha=candidate_inventory,
                artifact_bytes=candidate_bytes,
                budget=budget,
                concurrency=concurrency,
                output_len=output_len,
                input_range_ratio=ratio,
                chunked=chunked,
                speculative=spec,
                server_evidence=candidate_server_evidence,
            ),
        )
        baseline = baselines[phase]
        baseline_server_evidence = _serve_manifest_attachment(
            tmp_path,
            cell_id=cell_id,
            candidate=False,
            identity=baseline,
            chunked=chunked,
            speculative=spec,
        )
        baseline_digest = _write_json(
            baseline_path,
            _report(
                candidate=False,
                model_id=baseline["model_id"],
                inventory_sha=baseline["artifact_inventory_sha256"],
                artifact_bytes=baseline["artifact_bytes"],
                budget=budget,
                concurrency=concurrency,
                output_len=output_len,
                input_range_ratio=ratio,
                chunked=chunked,
                speculative=spec,
                server_evidence=baseline_server_evidence,
            ),
        )
        cells.append(
            {
                "id": cell_id,
                "phase": phase,
                "concurrency": concurrency,
                "chunked_prefill": chunked,
                **({"decode_mode": mode} if mode is not None else {}),
                "allowed_arm_difference_paths": _ALLOWED_DIFFERENCES,
                "candidate_report": {
                    "path": str(candidate_path.relative_to(tmp_path)),
                    "sha256": candidate_digest,
                },
                "baseline_report": {
                    "path": str(baseline_path.relative_to(tmp_path)),
                    "sha256": baseline_digest,
                },
            }
        )
    repository_commit = git_provenance()["commit"]
    cell_ids = [cell["id"] for cell in cells]
    phase_by_cell = {cell["id"]: cell["phase"] for cell in cells}
    pin = load_gridbook_runtime_pin()
    telemetry = []
    for kind in (
        "routing_per_layer_per_step",
        "expert_occupancy",
        "active_experts",
        "grouped_moe_whole_operator",
    ):
        arms = {}
        for arm in ("candidate", "baseline"):
            arm_cells = []
            for cell_id in cell_ids:
                model_id = (
                    "candidate@exact"
                    if arm == "candidate"
                    else baselines[phase_by_cell[cell_id]]["model_id"]
                )
                layers = []
                for layer_id in range(43):
                    step = {
                        "step_id": "block-1234-step-0",
                        "run_label": f"{arm}-{cell_id}",
                        "block_seed": 1234,
                        "timestamp": "2026-08-12T01:05:00Z",
                    }
                    if kind == "routing_per_layer_per_step":
                        step["expert_histogram"] = {"0": 6, "1": 6}
                    elif kind == "expert_occupancy":
                        step["expert_occupancy"] = {"0": 0.5, "1": 0.5}
                    elif kind == "active_experts":
                        step["active_experts"] = [0, 1]
                    else:
                        step.update(
                            {
                                "fallback_state": "none",
                                "routed_tokens": 12,
                                "stages_ms": {
                                    "routing": 0.1,
                                    "packing": 0.1,
                                    "launches": 0.1,
                                    "kernel": 0.2,
                                    "combine": 0.1,
                                },
                            }
                        )
                    layers.append({"layer_id": layer_id, "steps": [step]})
                arm_cells.append(
                    {"cell_id": cell_id, "model_id": model_id, "layers": layers}
                )
            arms[arm] = {"cells": arm_cells}
        payload = {
            "schema": TELEMETRY_SCHEMA,
            "kind": kind,
            "gridbook_runtime": {"commit": pin.commit, "version": pin.version},
            "cell_ids": cell_ids,
            "arms": arms,
        }
        path = tmp_path / "telemetry" / f"{kind}.json"
        digest = _write_json(path, payload)
        telemetry.append(
            {
                "kind": kind,
                "path": str(path.relative_to(tmp_path)),
                "sha256": digest,
                "cell_ids": cell_ids,
            }
        )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "predeclared_at": "2026-08-12T00:00:00Z",
        "base_model_identity": "deepseek-v4-flash@exact",
        "prismaquant_runtime": {"commit": repository_commit, "validator": TOOL},
        "gridbook_runtime": {"commit": pin.commit, "version": pin.version},
        "byte_budget": budget,
        "source_model_binding": {
            "identity_schema": "prismaquant.native_source_model/1",
            "content_sha256": _SOURCE_IDENTITY["content_sha256"],
            "identity_sha256": "2" * 64,
            "shard_count": 1,
        },
        "candidate": candidate_identity,
        "baselines": baselines,
        "displaced_container": displaced["declaration"],
        "parity_floor": 1.0,
        "predeclared_tolerance": 0.0,
        "shipped_max_concurrency": 16,
        "coverage": {
            "required_cell_ids": cell_ids,
            "phases": ["prefill", "decode", "mixed"],
            "concurrencies": [1, 2, 4, 8, 16],
            "chunked_prefill": [False, True],
            "decode_modes": ["plain", "shipped"],
            "nonzero_input_distribution": True,
        },
        "telemetry_contract": {
            "routing_per_layer_per_step": True,
            "expert_occupancy": True,
            "active_experts": True,
            "grouped_operator_includes": [
                "routing",
                "packing",
                "launches",
                "kernel",
                "combine",
            ],
        },
        "telemetry": telemetry,
        "cells": cells,
    }
    manifest_path = tmp_path / "comparison.json"
    _write_json(manifest_path, manifest)
    return {
        "shipcard": shipcard,
        "manifest": manifest_path,
        "manifest_payload": manifest,
        "displaced": displaced,
    }


def _stub_clean_provenance(monkeypatch) -> None:
    # The production validator must be run from the exact clean commit named
    # by the comparison manifest.  This synthetic fixture intentionally lives
    # in a developer worktree with pending changes, so preserve the real HEAD
    # identity used by _comparison() while stubbing only its cleanliness.
    repository_commit = performance_validator.git_provenance()["commit"]
    monkeypatch.setattr(
        performance_validator,
        "git_provenance",
        lambda: {"commit": repository_commit, "dirty": False},
    )


def _bypass_external_native_certificate(monkeypatch) -> None:
    _stub_clean_provenance(monkeypatch)
    monkeypatch.setattr(
        performance_validator,
        "_validate_native_feasibility",
        lambda *args, **kwargs: (
            "f" * 64,
            frozenset({"model.layers.0.self_attn.q_proj"}),
        ),
    )
    monkeypatch.setattr(
        performance_validator,
        "verify_shipcard",
        lambda *args, **kwargs: [],
    )


def _report_and_serve_manifest(
    fixture: dict, tmp_path: Path, *, cell_index: int = 0, arm: str = "candidate"
) -> tuple[dict, Path, dict, Path]:
    reference = fixture["manifest_payload"]["cells"][cell_index][f"{arm}_report"]
    report_path = tmp_path / reference["path"]
    report = json.loads(report_path.read_text())
    attachment = report["metadata"]["server"]["evidence"]["attachments"][0]
    serve_path = tmp_path / attachment["reference"]
    serve_manifest = json.loads(serve_path.read_text())
    return report, report_path, serve_manifest, serve_path


def _rewrite_report_and_comparison(
    fixture: dict,
    *,
    cell_index: int,
    arm: str,
    report_path: Path,
    report: dict,
) -> None:
    reference = fixture["manifest_payload"]["cells"][cell_index][f"{arm}_report"]
    reference["sha256"] = _write_json(report_path, report)
    _write_json(fixture["manifest"], fixture["manifest_payload"])


def test_paired_matrix_produces_and_fills_exact_shipcard_record(tmp_path, monkeypatch):
    fixture = _comparison(tmp_path)
    _bypass_external_native_certificate(monkeypatch)
    result = validate_cb_performance(
        fixture["shipcard"], fixture["manifest"], fill=True
    )

    assert result["schema"] == RESULT_SCHEMA
    assert result["status"] == "success"
    record = result["record"]
    assert record["slot"] == SLOT
    assert record["tool"] == TOOL
    assert record["passed"] is True
    assert record["metrics"]["schema"] == RESULT_SCHEMA
    assert record["metrics"]["cell_count"] == 32
    assert record["metrics"]["min_conservative_ratio"] > 1.0
    assert record["evidence"]["schema"] == EVIDENCE_SCHEMA
    assert len(record["evidence"]["paired_reports"]) == 32
    assert record["metrics"]["native_baseline_feasibility_sha256"] == "f" * 64
    assert record["evidence"]["displaced_container"]["schema"] == (
        DISPLACED_CONTAINER_SCHEMA
    )
    assert record["metrics"]["displaced_container_reason"] == (
        fixture["manifest_payload"]["displaced_container"]["reason"]
    )
    assert load_shipcard(fixture["shipcard"])["slots"][SLOT] == record
    assert _verify_gridbook_performance_record(
        SLOT,
        record,
        model_dir=fixture["shipcard"].parent,
    ) == []


def test_shipcard_replay_refuses_minimal_or_forged_parity_record(
    tmp_path, monkeypatch,
):
    fixture = _comparison(tmp_path)
    _bypass_external_native_certificate(monkeypatch)
    valid = validate_cb_performance(
        fixture["shipcard"], fixture["manifest"], fill=False
    )["record"]

    minimal = {
        "slot": SLOT,
        "tool": TOOL,
        "passed": True,
        "model_sha": valid["model_sha"],
    }
    problems = _verify_gridbook_performance_record(SLOT, minimal)
    assert any("structured parity metrics" in problem for problem in problems)

    forged = json.loads(json.dumps(valid))
    forged["evidence"]["displaced_container"]["artifact_bytes"] -= 1
    problems = _verify_gridbook_performance_record(
        SLOT,
        forged,
        model_dir=fixture["shipcard"].parent,
    )
    assert any("proof digest is inconsistent" in problem for problem in problems)
    assert any("byte count differs" in problem for problem in problems)


def test_bound_report_mutation_fails_before_a_ratio_can_be_used(tmp_path, monkeypatch):
    fixture = _comparison(tmp_path)
    _bypass_external_native_certificate(monkeypatch)
    first = fixture["manifest_payload"]["cells"][0]["candidate_report"]
    report_path = tmp_path / first["path"]
    report_path.write_text(report_path.read_text() + " \n")

    with pytest.raises(CBPerformanceValidationError, match="SHA-256 mismatch"):
        validate_cb_performance(fixture["shipcard"], fixture["manifest"], fill=False)


def test_report_requires_digest_bound_structured_serve_manifest(tmp_path, monkeypatch):
    fixture = _comparison(tmp_path)
    _bypass_external_native_certificate(monkeypatch)
    report, report_path, _, serve_path = _report_and_serve_manifest(
        fixture, tmp_path
    )
    serve_path.write_text("ordinary startup log, not a serve manifest\n")
    attachment = report["metadata"]["server"]["evidence"]["attachments"][0]
    attachment["sha256"] = hashlib.sha256(serve_path.read_bytes()).hexdigest()
    attachment["bytes"] = serve_path.stat().st_size
    _rewrite_report_and_comparison(
        fixture,
        cell_index=0,
        arm="candidate",
        report_path=report_path,
        report=report,
    )

    with pytest.raises(CBPerformanceValidationError, match="exactly one structured"):
        validate_cb_performance(fixture["shipcard"], fixture["manifest"], fill=False)


def test_serve_manifest_attachment_is_reread_at_parity_time(tmp_path, monkeypatch):
    fixture = _comparison(tmp_path)
    _bypass_external_native_certificate(monkeypatch)
    _, _, _, serve_path = _report_and_serve_manifest(fixture, tmp_path)
    serve_path.write_text(serve_path.read_text() + " \n")

    with pytest.raises(CBPerformanceValidationError, match="changed after measurement"):
        validate_cb_performance(fixture["shipcard"], fixture["manifest"], fill=False)


def test_candidate_report_cannot_attach_baseline_artifact_server(tmp_path, monkeypatch):
    fixture = _comparison(tmp_path)
    _bypass_external_native_certificate(monkeypatch)
    report, report_path, serve_manifest, serve_path = _report_and_serve_manifest(
        fixture, tmp_path
    )
    serve_manifest["artifact_binding"] = {
        "schema": "prismaquant.served_artifact_binding/1",
        "model_sha": fixture["displaced"]["identity"]["model_sha"],
        "artifact_inventory_sha256": fixture["displaced"]["identity"][
            "artifact_inventory_sha256"
        ],
        "artifact_bytes": fixture["displaced"]["identity"]["artifact_bytes"],
    }
    serve_manifest["serve_fingerprint"] = fingerprint(serve_manifest)
    attachment = report["metadata"]["server"]["evidence"]["attachments"][0]
    attachment["sha256"] = _write_json(serve_path, serve_manifest)
    attachment["bytes"] = serve_path.stat().st_size
    _rewrite_report_and_comparison(
        fixture,
        cell_index=0,
        arm="candidate",
        report_path=report_path,
        report=report,
    )

    with pytest.raises(CBPerformanceValidationError, match="exact expected artifact"):
        validate_cb_performance(fixture["shipcard"], fixture["manifest"], fill=False)


def test_candidate_and_baseline_cannot_reuse_one_server_process(tmp_path, monkeypatch):
    fixture = _comparison(tmp_path)
    _bypass_external_native_certificate(monkeypatch)
    candidate = _report_and_serve_manifest(fixture, tmp_path, arm="candidate")[2]
    report, report_path, baseline, serve_path = _report_and_serve_manifest(
        fixture, tmp_path, arm="baseline"
    )
    baseline["processes"] = candidate["processes"]
    baseline["serve_fingerprint"] = fingerprint(baseline)
    attachment = report["metadata"]["server"]["evidence"]["attachments"][0]
    attachment["sha256"] = _write_json(serve_path, baseline)
    attachment["bytes"] = serve_path.stat().st_size
    _rewrite_report_and_comparison(
        fixture,
        cell_index=0,
        arm="baseline",
        report_path=report_path,
        report=report,
    )

    with pytest.raises(CBPerformanceValidationError, match="one live server process"):
        validate_cb_performance(fixture["shipcard"], fixture["manifest"], fill=False)


def test_reported_server_argv_must_equal_live_server_argv(tmp_path, monkeypatch):
    fixture = _comparison(tmp_path)
    _bypass_external_native_certificate(monkeypatch)
    report, report_path, serve_manifest, serve_path = _report_and_serve_manifest(
        fixture, tmp_path
    )
    serve_manifest["launch_argv"].append("--unreported-switch")
    serve_manifest["launch_flags"] = elide_argv_paths(serve_manifest["launch_argv"])
    serve_manifest["serve_fingerprint"] = fingerprint(serve_manifest)
    attachment = report["metadata"]["server"]["evidence"]["attachments"][0]
    attachment["sha256"] = _write_json(serve_path, serve_manifest)
    attachment["bytes"] = serve_path.stat().st_size
    _rewrite_report_and_comparison(
        fixture,
        cell_index=0,
        arm="candidate",
        report_path=report_path,
        report=report,
    )

    with pytest.raises(CBPerformanceValidationError, match="exact live-server launch argv"):
        validate_cb_performance(fixture["shipcard"], fixture["manifest"], fill=False)


def test_quant_config_file_digest_cannot_impersonate_inventory_identity(
    tmp_path, monkeypatch,
):
    fixture = _comparison(tmp_path)
    _bypass_external_native_certificate(monkeypatch)
    manifest = fixture["manifest_payload"]
    quant_config_path = fixture["shipcard"].parent / "quant_config.json"
    wrong_digest = hashlib.sha256(quant_config_path.read_bytes()).hexdigest()
    assert wrong_digest != manifest["candidate"]["artifact_inventory_sha256"]
    reference = manifest["cells"][0]["candidate_report"]
    report_path = tmp_path / reference["path"]
    report = json.loads(report_path.read_text())
    report["metadata"]["artifacts"]["inventory"]["sha256"] = wrong_digest
    report["metadata"]["execution_identity"]["manifest"][
        "artifact_inventory_sha256"
    ] = wrong_digest
    reference["sha256"] = _write_json(report_path, report)
    _write_json(fixture["manifest"], manifest)

    with pytest.raises(
        CBPerformanceValidationError,
        match="candidate inventory digest differs",
    ):
        validate_cb_performance(fixture["shipcard"], fixture["manifest"], fill=False)


def test_conservative_block_ratio_below_predeclared_floor_fails(tmp_path, monkeypatch):
    fixture = _comparison(tmp_path)
    _bypass_external_native_certificate(monkeypatch)
    manifest = fixture["manifest_payload"]
    reference = manifest["cells"][2]["candidate_report"]
    report_path = tmp_path / reference["path"]
    report = json.loads(report_path.read_text())
    report["summary"]["metrics"]["p95_itl_ms"]["values"] = [20.0, 20.0, 20.0]
    for block in report["blocks"]:
        block["raw_result"]["p95_itl_ms"] = 20.0
    reference["sha256"] = _write_json(report_path, report)
    _write_json(fixture["manifest"], manifest)

    with pytest.raises(CBPerformanceValidationError, match="parity failed"):
        validate_cb_performance(fixture["shipcard"], fixture["manifest"], fill=False)


def test_declared_coverage_without_an_executed_ladder_cell_fails(tmp_path, monkeypatch):
    fixture = _comparison(tmp_path)
    _bypass_external_native_certificate(monkeypatch)
    manifest = fixture["manifest_payload"]
    removed = manifest["cells"].pop()
    manifest["coverage"]["required_cell_ids"].remove(removed["id"])
    _write_json(fixture["manifest"], manifest)

    with pytest.raises(CBPerformanceValidationError, match="Cartesian product"):
        validate_cb_performance(fixture["shipcard"], fixture["manifest"], fill=False)


def test_historical_displaced_artifact_without_current_shipcard_requires_reexport(
    tmp_path, monkeypatch,
):
    fixture = _comparison(tmp_path)
    _bypass_external_native_certificate(monkeypatch)
    displaced_root = fixture["displaced"]["root"]
    (displaced_root / "shipcard.json").unlink()
    quant_path = displaced_root / "quant_config.json"
    quant_config = json.loads(quant_path.read_text())
    _finalize_test_inventory(
        displaced_root,
        quant_config,
        fixture["manifest_payload"]["byte_budget"],
    )

    with pytest.raises(CBPerformanceValidationError, match="no current shipcard"):
        validate_cb_performance(fixture["shipcard"], fixture["manifest"], fill=False)


def test_displaced_container_requires_digest_bound_assignment_receipt(
    tmp_path, monkeypatch,
):
    fixture = _comparison(tmp_path)
    _bypass_external_native_certificate(monkeypatch)
    manifest = fixture["manifest_payload"]
    manifest["displaced_container"].pop("assignment_receipt")
    _write_json(fixture["manifest"], manifest)

    with pytest.raises(CBPerformanceValidationError, match="assignment receipt"):
        validate_cb_performance(fixture["shipcard"], fixture["manifest"], fill=False)


def test_each_phase_must_name_the_exact_displaced_container(tmp_path, monkeypatch):
    fixture = _comparison(tmp_path)
    _bypass_external_native_certificate(monkeypatch)
    manifest = fixture["manifest_payload"]
    manifest["baselines"]["decode"]["model_id"] = "some-other-container"
    _write_json(fixture["manifest"], manifest)

    with pytest.raises(CBPerformanceValidationError, match="decode baseline"):
        validate_cb_performance(fixture["shipcard"], fixture["manifest"], fill=False)


def test_comparison_without_native_infeasibility_certificate_fails_closed(
    tmp_path, monkeypatch,
):
    fixture = _comparison(tmp_path)
    _stub_clean_provenance(monkeypatch)

    with pytest.raises(
        CBPerformanceValidationError,
        match="native_baseline_feasibility reference is missing",
    ):
        validate_cb_performance(fixture["shipcard"], fixture["manifest"], fill=False)
