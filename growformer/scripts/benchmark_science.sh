#!/usr/bin/env bash
set -euo pipefail

SUITE="${1:-routing}"
OUT_DIR="${2:-reports/benchmarks/science}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

case "${SUITE}" in
  routing)
    NAMES=("phase3g_cone" "phase3h_pseudo_label")
    FLAGS=("--phase3g-cone" "--phase3h-label-free")
    ;;
  full)
    NAMES=("phase3g_cone" "phase3h_pseudo_label" "phase4d_context_free_mnist")
    FLAGS=("--phase3g-cone" "--phase3h-label-free" "--phase4d-cf-mnist-full")
    ;;
  *)
    echo "unknown suite '${SUITE}'; expected routing or full" >&2
    exit 2
    ;;
esac

mkdir -p "${OUT_DIR}"
BIN="${GROWFORMER_DEMOS_BIN:-target/release/growformer-demos}"
if [[ -z "${GROWFORMER_DEMOS_BIN:-}" ]]; then
  cargo build --release --bin growformer-demos
elif [[ ! -x "${BIN}" ]]; then
  echo "GROWFORMER_DEMOS_BIN is not executable: ${BIN}" >&2
  exit 2
fi

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "suite=${SUITE}"
  echo "profile=release"
  echo "os=$(uname -s)"
  echo "arch=$(uname -m)"
  echo "rustc=$(rustc --version)"
  echo "cargo=$(cargo --version)"
  echo "git_revision=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
  echo "binary=${BIN}"
} >"${OUT_DIR}/metadata.env"

printf "benchmark\tflag\texit_status\tverdict\n" >"${OUT_DIR}/results.tsv"

for index in "${!NAMES[@]}"; do
  name="${NAMES[$index]}"
  flag="${FLAGS[$index]}"
  stdout_file="${OUT_DIR}/${name}.stdout.txt"
  stderr_file="${OUT_DIR}/${name}.stderr.txt"

  echo "Running ${name} (${flag})..."
  set +e
  "${BIN}" "${flag}" >"${stdout_file}" 2>"${stderr_file}"
  status=$?
  set -e

  verdict="$(awk '/OVERALL:/ { line=$0 } END { print line }' "${stdout_file}")"
  verdict="${verdict//$'\t'/ }"
  printf "%s\t%s\t%s\t%s\n" "${name}" "${flag}" "${status}" "${verdict:-missing}" \
    >>"${OUT_DIR}/results.tsv"

  if [[ "${status}" -ne 0 ]]; then
    echo "${name} exited with status ${status}" >&2
    exit "${status}"
  fi
  if [[ -z "${verdict}" ]]; then
    echo "${name} produced no OVERALL verdict" >&2
    exit 1
  fi
done

echo "Scientific benchmark artifacts: ${OUT_DIR}"
