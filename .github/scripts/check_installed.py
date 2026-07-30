#!/usr/bin/env python3
"""Post-install gate: exercise the wheel from site-packages, not the checkout.

Run this from a directory that is NOT the repo root, so `import prismaquant`
cannot resolve to the source tree. It asserts the install is genuinely
non-editable and then does the one thing an import smoke test does not: reads
the runtime JSON specs back out of site-packages, which is where a
package-data regression actually shows up.
"""
from __future__ import annotations

import os
import subprocess
import sys

import prismaquant

where = os.path.dirname(prismaquant.__file__)
print(f"prismaquant resolved from: {where}")
if "site-packages" not in where and "dist-packages" not in where:
    sys.exit("prismaquant did not resolve from site-packages — this check must "
             "run outside the repo root, or the wheel is not installed")

# Serving-constraint specs: read from the installed package's JSON.
from prismaquant import serving_profiles as sp  # noqa: E402

for name in ("vllm_packed_moe", "vllm_qwen3_5_packed_moe", "gguf", "research"):
    profile = sp.load_serving_profile(name)
    if profile is None:
        sys.exit(f"serving profile {name!r} did not load from the install")
    print(f"  serving profile OK: {name}")

# Model-structure specs: the spec directory must exist inside the package and
# carry every arch the source tree ships.
spec_dir = os.path.join(where, "model_profiles", "specs")
specs = sorted(f for f in os.listdir(spec_dir) if f.endswith(".json"))
if not specs:
    sys.exit(f"no model-structure specs under {spec_dir} — package-data broke")
print(f"  model-structure specs OK: {len(specs)} ({', '.join(specs)})")

# run-pipeline.sh is the orchestrator, shipped as package data.
pipeline = os.path.join(where, "run-pipeline.sh")
if not os.path.isfile(pipeline):
    sys.exit(f"run-pipeline.sh missing from the install ({pipeline})")
print("  run-pipeline.sh OK")

# The CLI entry point every downstream user touches first.
r = subprocess.run([sys.executable, "-m", "prismaquant.allocator", "--help"],
                   capture_output=True, text=True)
if r.returncode != 0:
    sys.exit(f"`python -m prismaquant.allocator --help` failed:\n{r.stderr}")
print("  allocator CLI OK")

print("installed wheel verified")
