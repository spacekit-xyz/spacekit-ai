#!/usr/bin/env bash
set -euo pipefail

echo "Validating M3 action schema metrics"
cargo run -- --validate-action-schema

