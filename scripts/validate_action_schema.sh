#!/usr/bin/env bash
set -euo pipefail

ACTION_EVAL_DATA="${1:-}"
ACTION_EVAL_REPORT="${2:-}"

echo "Validating M3 action schema metrics"
CMD=(cargo run -- --validate-action-schema)
if [[ -n "${ACTION_EVAL_DATA}" ]]; then
  CMD+=(--action-eval-data "${ACTION_EVAL_DATA}")
fi
if [[ -n "${ACTION_EVAL_REPORT}" ]]; then
  CMD+=(--action-eval-report "${ACTION_EVAL_REPORT}")
fi
"${CMD[@]}"

