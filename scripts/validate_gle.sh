#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_PATH="${1:-checkpoints/gle_student_routing_tuned.json}"

echo "Validating GLE checkpoint: ${CHECKPOINT_PATH}"
cargo run -- --validate-gle "${CHECKPOINT_PATH}"

