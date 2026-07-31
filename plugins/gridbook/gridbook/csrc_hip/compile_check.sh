#!/usr/bin/env bash
# Day-one bring-up check for the gridbook HIP kernels on a ROCm box.
#
# Compiles and runs, in order: the WMMA capability/layout probe, then the
# kernel self-test (parity), then optionally the benchmarks.  Everything is
# standalone hipcc — no torch, no python, no vLLM — so a failure here is a
# toolchain or kernel failure and nothing else.
#
#   ./compile_check.sh                 # probe + parity
#   ./compile_check.sh --bench         # + benchmarks
#   PQ_ARCH=gfx1100 ./compile_check.sh # another RDNA3 target
#
# Artifacts land in $PQ_BUILD (default ~/.cache/prismaquant-cb-hip-check).
# Never /tmp: repo rule, and a cleared /tmp has cost this project artifacts.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PQ_BUILD="${PQ_BUILD:-$HOME/.cache/prismaquant-cb-hip-check}"
mkdir -p "$PQ_BUILD"

# --- toolchain -------------------------------------------------------------
if ! command -v hipcc >/dev/null 2>&1; then
  echo "FAIL: hipcc not on PATH. Install a ROCm toolchain (rocm-hip-devel," \
       "hipcc) or add \$ROCM_PATH/bin to PATH." >&2
  exit 2
fi
echo "== hipcc =="; hipcc --version | head -3

# --- target arch -----------------------------------------------------------
# Prefer an explicit PQ_ARCH; otherwise ask the device.  The kernels are
# validated on gfx1151 (Strix Halo) and assume wave32 + 64 KiB LDS, which is
# every gfx11 part; a gfx9/gfx94x (CDNA, wave64) target will be rejected at
# runtime by the launcher's warpSize assert rather than miscompute.
ARCH="${PQ_ARCH:-}"
if [[ -z "$ARCH" ]] && command -v rocminfo >/dev/null 2>&1; then
  ARCH="$(rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9a-f]*' || true)"
fi
ARCH="${ARCH:-gfx1151}"
echo "== target: $ARCH =="

# --- Fedora / ROCm 7.1.1 link fix ------------------------------------------
# A plain `hipcc x.hip -o x` fails there with
#   ld.lld: error: undefined symbol: __hipUnregisterFatBinary
# because the HIP runtime is not linked by default.  Harmless elsewhere.
LDFLAGS=()
for d in /usr/lib64 /opt/rocm/lib; do
  if [[ -e "$d/libamdhip64.so" ]]; then LDFLAGS=(-L"$d" -lamdhip64); break; fi
done
echo "== link flags: ${LDFLAGS[*]:-<none>} =="

CFLAGS=(--offload-arch="$ARCH" -O3)   # NO fast-math: the activation QDQ must
                                      # match torch's rounding bit-for-bit.

build() {  # build <out> <sources...>
  local out="$1"; shift
  echo "-- building $out"
  hipcc "${CFLAGS[@]}" "${LDFLAGS[@]}" "$@" -o "$PQ_BUILD/$out"
}

# --- 1. capability + fragment-layout probe ---------------------------------
build wmma_probe "$HERE/wmma_probe.hip"
echo "== WMMA probe =="
"$PQ_BUILD/wmma_probe"
echo

# --- 2. kernel parity ------------------------------------------------------
build cb_hip_selftest "$HERE/cb_hip_selftest.hip" "$HERE/cb_gemv_hip.hip" \
      "$HERE/cb_gemm_hip.hip"
echo "== kernel self-test =="
"$PQ_BUILD/cb_hip_selftest" "$@"

echo
echo "== compile_check OK (arch $ARCH, artifacts in $PQ_BUILD) =="
echo "Next: the torch extension and the pytest gate —"
echo "  PYTHONPATH=<repo>/plugins/gridbook python -m pytest \\"
echo "    <repo>/plugins/gridbook/tests/test_hip_decode_parity.py -v"
