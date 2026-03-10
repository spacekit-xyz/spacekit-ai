#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-reports/benchmarks}"
BIN="${GROWFORMER_BIN:-target/debug/growformer}"
mkdir -p "${OUT_DIR}"

echo "Building debug binary..."
cargo build -q

run_once() {
  local name="$1"
  local cmd="$2"
  local out_file="${OUT_DIR}/${name}.time.txt"
  /usr/bin/time -lp bash -lc "${cmd}" >"${OUT_DIR}/${name}.stdout.txt" 2>"${out_file}"
  local real
  real="$(awk '/^real /{print $2}' "${out_file}" | tail -n 1)"
  local rss
  rss="$(awk '/maximum resident set size/{print $1}' "${out_file}" | tail -n 1)"
  echo "${name}: real=${real}s max_rss=${rss}KB"
}

echo "Running core demo tasks (single run each)..."
run_once "core_xor" "\"${BIN}\" --xor"
run_once "core_spiral" "\"${BIN}\" --spiral"
run_once "core_language_pipeline" "\"${BIN}\" --language-pipeline"

echo "Outputs and timing saved to ${OUT_DIR}"

