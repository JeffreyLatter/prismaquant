# CB/CBL CUDA encode profile and optimization — 2026-08-11

Status: measured on branch `perf/ldlq-atom-compile` at base commit `c96eefc`
with the working-tree changes described below. This is a dated result record,
not a serving-gate declaration.

## Scope and method

The target was the compiled (`PRISMAQUANT_CB_ENCODE_COMPILE=1`) balanced,
scale-sweep FP8-CB product encoder. Measurements used an NVIDIA GB10 DGX
Spark (driver 595.84), torch 2.11.0+cu130, CUDA 13.0, deterministic synthetic
BF16 weights, and one positive per-input-column imatrix vector. Every CUDA run
held `/home/rob/dq-runs/gpu.lock`. No run used `/tmp`.

The DeepSeek production unit is `(2048, 4096)` and the production MoE batch is
`(E=8, 2048, 4096)`. Dense transfer checks used single 2-D tensors at the
expected Qwen range: `(4096,4096)`, `(4096,8192)`, and `(8192,8192)`. Timings
below are synchronized medians after compile warm-up. Paired arms were run in
one process and interleaved where noted.

Raw logs, profiler tables, traces, and rejected-prototype results are under:

```text
/home/rob/dq-runs/pq-cbl-export-cb-encode-20260811/
```

## Kernel profile: where the time goes

The clean one-expert baselines were 75.109 ms for K28 and 144.363 ms for K33.
Raw `torch.profiler` device totals were 75.314 and 144.741 ms respectively;
CPU totals were essentially the same (74.907 and 142.919 ms), so the steady
path is GPU-bound rather than host- or storage-bound.

| CUDA operator family | K28 time / share / calls | K33 time / share / calls |
|---|---:|---:|
| compiled fused score `min`/`argmin` reductions | 41.289 ms / 54.82% / 32 | 100.025 ms / 69.11% / 104 |
| moment GEMMs (`aten::mm`, CUTLASS SIMT SGEMM) | 11.876 ms / 15.77% / 24 | 29.597 ms / 20.45% / 72 |
| concatenation | 6.019 ms / 7.99% / 16 | 2.762 ms / 1.91% / 52 |
| elementwise multiply | 5.727 ms / 7.60% / 58 | 4.472 ms / 3.09% / 154 |
| add | 2.514 ms / 3.34% / 24 | 1.984 ms / 1.37% / 78 |
| index / gather / where | 5.817 ms / 7.72% | 3.838 ms / 2.65% |
| sum | 1.194 ms / 1.58% / 21 | 0.907 ms / 0.63% / 63 |

The call-count difference is the chunk model. With the old 64M score-element
cap, K28's four 128-entry subtables use 1024 source rows per main chunk: one
pilot plus two main chunks. K33 has one 512-entry and three 256-entry
subtables, so the largest table forces 256-row chunks: one pilot plus eight
main chunks. The compiled scorer is therefore the critical K33 lever, not
small host helpers.

Inductor output confirms that scale arithmetic and the reduction live in one
Triton kernel and that both minimum value and index are reduction state; the
`(m,K,S)` score volume is not materialized. The inspected emitted wrapper was:

```text
/home/rob/dq-runs/dsv4-quality-hybrid/cache/torchinductor/fl/cflctys7lx3fhtts2lo6bjdhzrsbvvbubielaq472udcoiuz3q37.py
```

Profiler sources are `torchprof_k28.txt/json` and `torchprof_k33.txt/json` in
the result directory. The older phase-probe helper was not used for
attribution because its per-helper synchronizations serialize nested work.

## Changes that survived production-shape checks

### 1. K28-specific compiled workspace correction

`_moment_rows_step` now recognizes exactly four K=128 product subtables (the
FP8-CB K28 layout) and uses a 32M score-element cap. This changes a width-4096
main chunk from 1024 to 512 rows. It does not alter K33 or any mixed-size rung.
Although this doubles K28's reduction launch count, each launch has a smaller
resident four-stream B working set; that is faster on GB10 and materially
reduces peak allocation.

| Production case | Before | After | Speedup | Exact fields |
|---|---:|---:|---:|:---:|
| K28 lattice, E8 × 2048 × 4096 | 558.568 ms total (69.821 ms/E) | 518.623 ms total (64.828 ms/E) | **1.0770×** | yes |
| K28 Lloyd-trained, grid-snapped tables, E8 × 2048 × 4096 | 634.061 ms total (79.258 ms/E) | 580.397 ms total (72.550 ms/E) | **1.0925×** | yes |
| K28, 4096 × 4096 | 148.662 ms | 142.500 ms | **1.0432×** | yes |
| K28, 4096 × 8192 | 304.816 ms | 277.546 ms | **1.0982×** | yes |
| K28, 8192 × 8192 (workspace arm only) | 627.049 ms | 586.982 ms | **1.0683×** | yes |

On the E8 production batch, peak allocated CUDA/UMA memory fell from 2.941 to
2.163 GiB (−26.4%). K33 retains the generic 64M cap: its measured sweep was
best around the current 32M–64M range, while 128M, 256M, and 512M raised time
from about 144 ms to 146, 165, and 165 ms.

### 2. Exact-shape pilot/first-chunk moment reuse

The balanced encoder built identical A/B moments for the calibration pilot
and first main chunk, then discarded the pilot copy. It now reuses the same
read-only moment objects only when the pilot row count and first-chunk row
count are identical. There is no slice and no GEMM-shape or summation change.

This has essentially no stacked-DS effect because one pilot is amortized over
all eight experts: paired K33 E8 timing was 1272.969 → 1271.164 ms (1.0014×,
within clock variance). It transfers to the single-tensor dense path. At K28
8192² it added 1.0075× over the workspace change, producing 627.049 → 582.621
ms cumulative (**1.0763×**). It deliberately does not fire when the pilot
spans multiple chunks.

### 3. Serialized-index compilation containment

`_vq_dist_argmin`, `_score_minargmin_batched`, and `_score_argmin` now use the
same CUDA-only containment policy as the LDLQ atom route. CPU always uses
eager index reduction because torch 2.11's CPU compiled `min(dim=-1)` can pair
a correct value with a wrong index. CUDA compiler/configuration failures now
propagate from index-producing routes rather than silently dropping the run
onto the 16–20× slower eager encoder. The mixed-rung Dynamo limits remain
256/4096 and are set fail-closed.

A counting-backend diagnostic with `dynamic=True` reused one symbolic Dynamo
graph across K=128, 256, 512, and 1024; the CUDA interleaving test then proved
the resulting K specializations return true indices on replay. That finds no
Dynamo frame thrash in the scorer wrappers. It does not assume lower-level
Inductor cache reuse is harmless—the replay exactness test remains the gate.

## Quality and determinism gates

The production E8 K28 comparison reconstructed both arms and computed:

```text
weighted MSE before = 3.983534043072723e-05
weighted MSE after  = 3.983534043072723e-05
delta               = 0.0
```

All serialized fields, including indices, scales, shape, and each codebook
subtable, were `torch.equal`. A separate check used four genuinely
Lloyd-trained, FP8-grid-snapped K28 subtables (three iterations over 16,384
training vectors per stream); its before/after weighted MSE was exactly
`3.976564039476216e-05` in both arms, and every field was exact. Its absolute
timings came from a later GPU-clock epoch and should not be read as a
learned-versus-lattice penalty; the useful number is the paired arm speedup.
The dense shape comparisons were also field-exact.

The identity test now explicitly builds the score tensor and asserts that the
returned index equals its true eager argmin for both ordinary and periodic-A
layouts. It also interleaves K=128, 256, 512, then 128 again in one CUDA
process, checks the true index after every specialization, and proves that
compiled index helpers are unreachable on CPU. This avoids the unsafe test
pattern of validating only the returned minimum value.

Final test command:

```bash
flock -x /home/rob/dq-runs/gpu.lock env \
  TMPDIR=/home/rob/dq-runs/pq-cbl-export-cb-encode-20260811/tmp \
  PYTHONPATH=/home/rob/pq-cbl-export \
  /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python -m pytest -q \
  tests/test_nvfp4_cb_encode_perf_identity.py
```

Result: **44 passed** (15 warnings) in 24.39 seconds. The JUnit record is
`final_identity_junit.xml` in the result directory. A CPU-only rerun after the
final edits also passed 23 tests with the one CUDA specialization test skipped.
The broader `tests/test_nvfp4_cb_formats.py` CPU regression passed all 136
tests (`final_formats_cpu_junit.xml`).

## Experiments that did not transfer

| Candidate | K28 result | K33 result | Decision |
|---|---:|---:|---|
| two-pass scale scan then selected assignment | 75.316 → 77.927 ms (0.966×) | 144.397 → 156.874 ms (0.920×) | reject; exact but slower |
| wider compiled four-stream scale-selection boundary | 77.639 → 77.120 ms (1.007×) | 147.788 → 238.623 ms (0.619×) | reject; indices also changed |
| static (`dynamic=False`) compilation | 75.084 → 72.107 ms (1.041×) | 143.710 → 147.702 ms (0.973×) | reject globally; K33 loss and specialization growth |
| `max-autotune-no-cudagraphs` | 1.0030× | 1.0031× | noise-sized; reject complexity |
| tensor-only m2 accumulation | no steady gain | no steady gain | useful graph prerequisite, not a speed change |
| direct elementwise replacement for skinny moment GEMM | neutral with changed indices | about 1.008× with changed indices | reject correctness failure |

`mode="reduce-overhead"` (Inductor CUDA graphs) was also tested. The narrow
compiled scorer graph reuses static output storage while prior stream outputs
are still live, and PyTorch raised its overwritten-CUDAGraph-output error.
Cloning each large result would give back the memory traffic that compilation
removed. No CUDA-graph mode was landed. Capture-OFF is the tested,
bit-exact path. Whole-encode capture remains blocked by the pilot's host scalar
conversion and, for two-tier FP4, a data-dependent host scale window.

## Remaining work

K33 is still dominated by its 512-entry stream: fused reductions plus moment
GEMMs are 89.6% of device time. The next credible kernel experiment is
per-stream row geometry: keep K512 at its measured 256-row working set while
letting the three K256 streams process 512 rows. It requires aggregating
per-stream errors without materializing/copying a new stacked score buffer;
an outer chunk increase by itself already regressed and is not sufficient.
The naive split model cuts scorer calls from 104 to 68 at 2048×4096, but
doubles resident B moments from roughly 640 MiB to 1.25 GiB and introduces
about 384 MiB of concatenation traffic, so it was not landed without a
slice-consume kernel that avoids those copies.

For dense Qwen, the current K28 change transfers at 1.04–1.10× over the tested
shape range. A whole-encode CUDA graph is worth revisiting only after m2 and
scale-window host synchronizations are removed and capture ON/OFF fields are
proved byte-identical. Learned-bundle lookup also needs to keep validated
tables resident; hashing or copying a CUDA codebook per Linear would erase a
meaningful fraction of these sub-100-ms gains and belongs in the existing
bundle/cache wiring rather than a second residency mechanism here.
