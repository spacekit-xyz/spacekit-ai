#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_PATH="${1:-checkpoints/gle_student_routing_tuned.json}"
ACTION_EVAL_DATA="${2:-}"
ACTION_EVAL_REPORT="${3:-}"

echo "== Growformer Stack Validation =="
echo "Checkpoint: ${CHECKPOINT_PATH}"

echo
echo "[1/2] Validate GLE routing gate"
cargo run -- --validate-gle "${CHECKPOINT_PATH}"

echo
echo "[2/2] Validate M3 action schema gate"
CMD=(cargo run -- --validate-action-schema)
if [[ -n "${ACTION_EVAL_DATA}" ]]; then
  CMD+=(--action-eval-data "${ACTION_EVAL_DATA}")
fi
if [[ -n "${ACTION_EVAL_REPORT}" ]]; then
  CMD+=(--action-eval-report "${ACTION_EVAL_REPORT}")
fi
"${CMD[@]}"

echo
echo "Stack validation PASSED"

