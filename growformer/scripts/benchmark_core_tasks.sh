#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-reports/benchmarks}"
ITER="${2:-5}"
WARMUPS="${GROWFORMER_BENCH_WARMUPS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"
source "${SCRIPT_DIR}/benchmark_common.sh"

benchmark_init "${OUT_DIR}" "${ITER}" "${WARMUPS}"

echo "Running core demo tasks (${ITER} iterations each)..."
benchmark_run "core_xor" "${BENCH_BIN}" --xor
benchmark_run "core_spiral" "${BENCH_BIN}" --spiral
benchmark_run "core_language_pipeline" "${BENCH_BIN}" --language-pipeline

benchmark_finish

