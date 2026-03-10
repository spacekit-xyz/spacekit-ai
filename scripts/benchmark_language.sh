#!/usr/bin/env bash
set -euo pipefail

ITER="${1:-5}"
OUT_DIR="${2:-reports/benchmarks}"
BIN="${GROWFORMER_BIN:-target/debug/growformer}"

mkdir -p "${OUT_DIR}"

echo "Building debug binary..."
cargo build -q

run_bench() {
  local name="$1"
  local cmd="$2"
  local out_file="${OUT_DIR}/${name}.time.txt"
  rm -f "${out_file}"
  for i in $(seq 1 "${ITER}"); do
    /usr/bin/time -lp bash -lc "${cmd}" >/dev/null 2>>"${out_file}"
  done
  local avg_real
  avg_real="$(awk '/^real /{sum+=$2; n+=1} END{if(n>0) printf "%.3f", sum/n; else print "0.000"}' "${out_file}")"
  local avg_rss_kb
  avg_rss_kb="$(awk '/maximum resident set size/{sum+=$1; n+=1} END{if(n>0) printf "%.0f", sum/n; else print "0"}' "${out_file}")"
  echo "${name}: avg_real=${avg_real}s avg_max_rss=${avg_rss_kb}KB"
}

echo "Running language/code benchmarks (${ITER} iterations each)..."
run_bench "language_action" "\"${BIN}\" --language-action-text \"implement binary search in rust\""
run_bench "language_codegen" "\"${BIN}\" --language-code-text \"implement a web server in rust\""
run_bench "language_codegen_eval" "\"${BIN}\" --language-code-eval --code-eval-report reports/m5_codegen_eval_holdouts_20.json"
run_bench "m5_retention_eval" "\"${BIN}\" --m5-retention-eval --m5-retention-plan data/language/m5/retention_eval_splits.json --m5-epochs 20 --m5-lr 0.2 --m5-feature-dim 512 --m5-replay-per-epoch 24 --m5-retention-report reports/m5_retention_report_20holdout.json"

echo "Benchmark files saved to ${OUT_DIR}"

