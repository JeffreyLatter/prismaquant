# Gridbook 0.8.5 block-FP8 W8A16 installed-wheel gate — 2026-08-12

Status: passed on one NVIDIA GB10 DGX Spark (sm121). This is route-existence,
resident-weight, dispatch, and operator-correctness evidence for the dedicated
source block-FP8 W8A16 lane. It is not full-artifact eager/graph generation,
matched-budget performance parity, KL, or PPL evidence; those remain separate
post-export shipcard gates.

## Immutable inputs

- Gridbook commit:
  `e992e5980c96333a48149f96392d6cff56ae9e3f` (released as 0.8.5).
- Built wheel:
  `/home/rob/gridbook-fp8-w8a16-release/dist/gridbook-0.8.5-py3-none-any.whl`.
- Wheel SHA-256:
  `51122fab1533d538230836b103cef9f438dbea015a75c671437e52392cf90d4d`.
- Container image:
  `eugr/spark-vllm@sha256:7bf752a9fa225b528b27c6a1118cb1727cddd7c383096d83281010c4f8b407bc`.
- Persistent extension cache: Docker volume `gridbook-w8a16-085-sm121`.
- Evidence directory:
  `/home/rob/dq-runs/dsv4-flash-0731/gridbook-085-gate-e992e59.30HdKZ`.

The wheel was built from the clean commit with:

```bash
/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python -m build --wheel
```

The seven test modules and `tests/conftest.py` were copied read-only from that
same checkout into the evidence directory. Pytest 8.3.5 and its dependencies
were downloaded into that directory before the network-isolated container run.

## Measured command

The successful installed-wheel GPU invocation was:

```bash
docker run --rm --pull=never --gpus all --ipc=host --network none \
  --name gridbook-085-w8a16-gate-e992e59 \
  -v /home/rob/gridbook-fp8-w8a16-release/dist:/dist:ro \
  -v /home/rob/dq-runs/dsv4-flash-0731/gridbook-085-gate-e992e59.30HdKZ:/gate:rw \
  -v gridbook-w8a16-085-sm121:/gridbook-ext:rw \
  -e PRISMAQUANT_CB_EXT_DIR=/gridbook-ext \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  --entrypoint bash \
  eugr/spark-vllm@sha256:7bf752a9fa225b528b27c6a1118cb1727cddd7c383096d83281010c4f8b407bc \
  -lc '
set -euo pipefail
python3 -m pip install --no-deps --force-reinstall -q \
  /dist/gridbook-0.8.5-py3-none-any.whl
python3 -m pip install --no-index --find-links /gate -q pytest==8.3.5
cd /gate
python3 - <<'"'"'PY'"'"'
import importlib.metadata
from pathlib import Path
import gridbook
from gridbook.runtime_contract import load_runtime_contract

origin = Path(gridbook.__file__).resolve()
assert "/dist-packages/gridbook/" in str(origin) or \
       "/site-packages/gridbook/" in str(origin), origin
assert not str(origin).startswith("/gate"), origin
assert gridbook.__version__ == "0.8.5"
assert importlib.metadata.version("gridbook") == "0.8.5"
contract = load_runtime_contract()
assert contract["schema"] == "gridbook.runtime-contract.v3"
assert contract["abi_features"]["source_fp8_block128_w8a16"] == 1
print("installed", origin, gridbook.__version__, flush=True)
PY
python3 -m pytest -q --junitxml=/gate/junit.xml \
  test_fp8_source_w8a16.py \
  test_fp8_source_loader_dtype_guard.py \
  test_fp8_source_w8a16_geometry.py \
  test_fp8_source_w8a16_cuda.py \
  test_source_passthrough.py \
  test_dsv4_woa.py \
  test_mixed_fused_linear.py 2>&1 | tee /gate/gate.log
python3 - <<'"'"'PY'"'"'
import xml.etree.ElementTree as ET
from gridbook import cuda_ext

root = ET.parse("/gate/junit.xml").getroot()
suites = [root] if root.tag == "testsuite" else list(root)
counts = {
    key: sum(int(s.attrib.get(key, 0)) for s in suites)
    for key in ("tests", "failures", "errors", "skipped")
}
assert counts["tests"] > 0
assert counts["failures"] == counts["errors"] == counts["skipped"] == 0
source = cuda_ext.require_fp8_source_w8a16_ext(
    "installed wheel release gate"
)
assert source.__gridbook_jit_capability__ == (12, 1)
assert source.__gridbook_jit_abi_schema__ == \
       cuda_ext._FP8_SOURCE_W8A16_ABI_SCHEMA
assert len(source.__gridbook_jit_identity__) == 64
grouped = cuda_ext.require_bf16_grouped_ext(
    "installed wheel release gate"
)
assert grouped is not None
print("PASS", counts, "source_identity", source.__gridbook_jit_identity__)
PY
'
```

## Result and raw evidence

Pytest reported `91 passed, 0 skipped` in 71.23 seconds. The selection covers
the source loader dtype guard, exact DSv4 dense/grouped geometry, independent
CUDA expansion/GEMV oracles, BF16 activation preservation, CUDA graph replay,
non-default streams, raw-plane residency, dispatch telemetry, source
passthrough, DSv4 `wo_a`, and mixed fused-Linears.

- Console log:
  `/home/rob/dq-runs/dsv4-flash-0731/gridbook-085-gate-e992e59.30HdKZ/gate.log`
  (`sha256:5d9d63da1ac42a96d94aa9183bd683d6cacf8041877bf9fc96e0812b620e91c6`).
- JUnit:
  `/home/rob/dq-runs/dsv4-flash-0731/gridbook-085-gate-e992e59.30HdKZ/junit.xml`
  (`sha256:ce5f9e99b3fb1a8780e219e50edbcf821afcddcb5f4c77d78845703ef56d7553`).

The JUnit root records `tests=91`, `failures=0`, `errors=0`, `skipped=0`, and
`time=71.230`. After pytest, the command also required the loaded source
extension to report JIT capability `(12, 1)`, the exact W8A16 ABI schema, a
64-hex build identity, and a resolvable Gridbook-owned grouped-BF16 bridge.
