#!/usr/bin/env bash
set -euo pipefail

ITER="${1:-5}"
OUT_DIR="${2:-reports/benchmarks}"
WARMUPS="${GROWFORMER_BENCH_WARMUPS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"
source "${SCRIPT_DIR}/benchmark_common.sh"

benchmark_init "${OUT_DIR}" "${ITER}" "${WARMUPS}"

echo "Running language/code benchmarks (${ITER} iterations each)..."
benchmark_run "language_action" \
  "${BENCH_BIN}" --language-action-text "implement binary search in rust"
benchmark_run "language_codegen" \
  "${BENCH_BIN}" --language-code-text "implement a web server in rust"
benchmark_run "language_codegen_eval" \
  "${BENCH_BIN}" --language-code-eval \
  --code-eval-report "${OUT_DIR}/m5_codegen_eval_holdouts_20.json"
benchmark_run "m5_retention_eval" \
  "${BENCH_BIN}" --m5-retention-eval \
  --m5-retention-plan data/language/m5/retention_eval_splits.json \
  --m5-epochs 20 \
  --m5-lr 0.2 \
  --m5-feature-dim 512 \
  --m5-replay-per-epoch 24 \
  --m5-retention-report "${OUT_DIR}/m5_retention_report_20holdout.json"

benchmark_finish

