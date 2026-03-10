#!/usr/bin/env bash
set -euo pipefail

ACTION_EVAL_DATA="${1:-}"
GENERATION_EVAL_REPORT="${2:-}"

echo "Validating M4 generation metrics"
CMD=(cargo run -- --validate-generation)
if [[ -n "${ACTION_EVAL_DATA}" ]]; then
  CMD+=(--action-eval-data "${ACTION_EVAL_DATA}")
fi
if [[ -n "${GENERATION_EVAL_REPORT}" ]]; then
  CMD+=(--generation-eval-report "${GENERATION_EVAL_REPORT}")
fi
"${CMD[@]}"

