import json
import pickle
from pathlib import Path

from prismaquant import adjoint_l3_frontier as frontier
from prismaquant import adjoint_l3 as l3a


def test_adjoint_l3_frontier_writes_summary_assignments_and_moves(tmp_path):
    adjoint_path = tmp_path / "adjoint.json"
    probe_path = tmp_path / "probe.pkl"
    base_path = tmp_path / "base.json"
    out_dir = tmp_path / "frontier"

    adjoint_path.write_text(json.dumps({
        "schema": l3a.SCHEMA,
        "rank": 1,
        "units": {
            "layer.a": {
                "formats": {
                    "BF16": {
                        "sketch": [0.0],
                        "diagonal_cost": 0.0,
                        "bits_per_param": 16.0,
                        "memory_bytes": 16,
                    },
                    "NVFP4": {
                        "sketch": [1.0],
                        "diagonal_cost": 1.0,
                        "bits_per_param": 4.0,
                        "memory_bytes": 4,
                    },
                }
            },
            "layer.b": {
                "formats": {
                    "BF16": {
                        "sketch": [0.0],
                        "diagonal_cost": 0.0,
                        "bits_per_param": 16.0,
                        "memory_bytes": 16,
                    },
                    "NVFP4": {
                        "sketch": [1.0],
                        "diagonal_cost": 0.2,
                        "bits_per_param": 4.0,
                        "memory_bytes": 4,
                    },
                }
            },
        },
    }))
    stats = {
        "layer.a": {
            "n_params": 8,
            "_memory_bytes_by_format": {"BF16": 16, "NVFP4": 4},
        },
        "layer.b": {
            "n_params": 8,
            "_memory_bytes_by_format": {"BF16": 16, "NVFP4": 4},
        },
    }
    with probe_path.open("wb") as fh:
        pickle.dump({"stats": stats}, fh)
    base_path.write_text(json.dumps({
        "assignment": {"layer.a": "BF16", "layer.b": "BF16"}
    }))

    rc = frontier.main([
        "--adjoint-costs", str(adjoint_path),
        "--probe", str(probe_path),
        "--base-assignment", str(base_path),
        "--formats", "NVFP4,BF16",
        "--target-full-bpps", "4,10,16",
        "--output-dir", str(out_dir),
    ])

    assert rc == 0
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["schema"] == "prismaquant.adjoint_l3.frontier.v1"
    assert [row["target_bpp"] for row in summary["rows"]] == [4.0, 10.0, 16.0]
    assert all(row["target_feasible"] is True for row in summary["rows"])
    assert summary["knee"]["mode"] == "surrogate_adjoint_kneedle"
    for row in summary["rows"]:
        assert Path(row["assignment_path"]).exists()
        assert Path(row["full_assignment_path"]).exists()
        assert Path(row["move_report_path"]).exists()
    assert (out_dir / "frontier.csv").exists()

    middle = json.loads(Path(summary["rows"][1]["full_assignment_path"]).read_text())
    assert list(middle["assignment"].values()).count("NVFP4") == 1
