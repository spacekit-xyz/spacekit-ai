#!/usr/bin/env bash

# Shared release-benchmark harness. Source this file from a benchmark script,
# then call benchmark_init, benchmark_run, and benchmark_finish.

BENCH_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

benchmark_init() {
  BENCH_OUT_DIR="$1"
  BENCH_ITERATIONS="$2"
  BENCH_WARMUPS="$3"
  BENCH_BIN="${GROWFORMER_BIN:-target/release/growformer}"

  if ! [[ "${BENCH_ITERATIONS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "iterations must be a positive integer: ${BENCH_ITERATIONS}" >&2
    return 2
  fi
  if ! [[ "${BENCH_WARMUPS}" =~ ^[0-9]+$ ]]; then
    echo "warmups must be a non-negative integer: ${BENCH_WARMUPS}" >&2
    return 2
  fi

  mkdir -p "${BENCH_OUT_DIR}"
  BENCH_SAMPLES="${BENCH_OUT_DIR}/samples.tsv"
  BENCH_SUMMARY="${BENCH_OUT_DIR}/summary.tsv"
  printf "benchmark\titeration\telapsed_seconds\tmax_rss_kib\texit_status\n" >"${BENCH_SAMPLES}"

  {
    echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "profile=release"
    echo "iterations=${BENCH_ITERATIONS}"
    echo "warmups=${BENCH_WARMUPS}"
    echo "os=$(uname -s)"
    echo "arch=$(uname -m)"
    echo "rustc=$(rustc --version)"
    echo "cargo=$(cargo --version)"
    echo "git_revision=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo "binary=${BENCH_BIN}"
  } >"${BENCH_OUT_DIR}/metadata.env"

  if [[ -z "${GROWFORMER_BIN:-}" ]]; then
    echo "Building release benchmark binary..."
    cargo build --release --bin growformer
  elif [[ ! -x "${BENCH_BIN}" ]]; then
    echo "GROWFORMER_BIN is not executable: ${BENCH_BIN}" >&2
    return 2
  fi
}

benchmark_measure() {
  local stdout_file="$1"
  local timing_file="$2"
  shift 2

  python3 "${BENCH_COMMON_DIR}/benchmark_runner.py" \
    "${stdout_file}" "${timing_file}" "$@"
}

benchmark_run() {
  local name="$1"
  shift
  local -a command=("$@")

  echo "Benchmarking ${name} (${BENCH_ITERATIONS} measured, ${BENCH_WARMUPS} warmup)..."
  local warmup
  for ((warmup = 1; warmup <= BENCH_WARMUPS; warmup++)); do
    "${command[@]}" >/dev/null
  done

  local iteration result elapsed rss_kib status
  for ((iteration = 1; iteration <= BENCH_ITERATIONS; iteration++)); do
    result="$(benchmark_measure \
      "${BENCH_OUT_DIR}/${name}.${iteration}.stdout.txt" \
      "${BENCH_OUT_DIR}/${name}.${iteration}.time.txt" \
      "${command[@]}")"
    IFS=$'\t' read -r elapsed rss_kib status <<<"${result}"
    if ! [[ "${elapsed}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
      echo "${name} produced unparseable timing data; see ${BENCH_OUT_DIR}/${name}.${iteration}.time.txt" >&2
      return 1
    fi
    if ! [[ "${rss_kib}" == "NA" || "${rss_kib}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
      echo "${name} produced unparseable RSS data; see ${BENCH_OUT_DIR}/${name}.${iteration}.time.txt" >&2
      return 1
    fi
    printf "%s\t%s\t%s\t%s\t%s\n" \
      "${name}" "${iteration}" "${elapsed}" "${rss_kib}" "${status}" >>"${BENCH_SAMPLES}"
    if [[ "${status}" -ne 0 ]]; then
      echo "${name} failed on iteration ${iteration}; see ${BENCH_OUT_DIR}/${name}.${iteration}.time.txt" >&2
      return "${status}"
    fi
  done
}

benchmark_finish() {
  printf "benchmark\titerations\tmean_elapsed_seconds\tmean_max_rss_kib\n" >"${BENCH_SUMMARY}"
  awk -F'\t' '
    NR == 1 { next }
    {
      count[$1] += 1
      elapsed[$1] += $3
      if ($4 != "NA") {
        rss[$1] += $4
        rss_count[$1] += 1
      }
    }
    END {
      for (name in count) {
        if (rss_count[name] > 0) {
          printf "%s\t%d\t%.6f\t%.0f\n", name, count[name], elapsed[name]/count[name], rss[name]/rss_count[name]
        } else {
          printf "%s\t%d\t%.6f\tNA\n", name, count[name], elapsed[name]/count[name]
        }
      }
    }
  ' "${BENCH_SAMPLES}" | sort >>"${BENCH_SUMMARY}"
  echo "Benchmark samples: ${BENCH_SAMPLES}"
  echo "Benchmark summary: ${BENCH_SUMMARY}"
  echo "Environment metadata: ${BENCH_OUT_DIR}/metadata.env"
}
