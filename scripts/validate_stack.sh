#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_PATH="${1:-checkpoints/gle_student_routing_tuned.json}"
ACTION_EVAL_DATA="${2:-}"
ACTION_EVAL_REPORT="${3:-}"
GENERATION_EVAL_REPORT="${4:-}"

echo "== Growformer Stack Validation =="
echo "Checkpoint: ${CHECKPOINT_PATH}"

echo
echo "[1/3] Validate GLE routing gate"
cargo run -- --validate-gle "${CHECKPOINT_PATH}"

echo
echo "[2/3] Validate M3 action schema gate"
CMD=(cargo run -- --validate-action-schema)
if [[ -n "${ACTION_EVAL_DATA}" ]]; then
  CMD+=(--action-eval-data "${ACTION_EVAL_DATA}")
fi
if [[ -n "${ACTION_EVAL_REPORT}" ]]; then
  CMD+=(--action-eval-report "${ACTION_EVAL_REPORT}")
fi
"${CMD[@]}"

echo
echo "[3/3] Validate M4 constrained generation gate"
CMD=(cargo run -- --validate-generation)
if [[ -n "${ACTION_EVAL_DATA}" ]]; then
  CMD+=(--action-eval-data "${ACTION_EVAL_DATA}")
fi
if [[ -n "${GENERATION_EVAL_REPORT}" ]]; then
  CMD+=(--generation-eval-report "${GENERATION_EVAL_REPORT}")
fi
"${CMD[@]}"

echo
echo "Stack validation PASSED"

