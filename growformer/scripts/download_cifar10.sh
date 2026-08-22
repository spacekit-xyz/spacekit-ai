#!/usr/bin/env bash
# Download CIFAR-10 via torchvision and export flat binaries for Phase 4e.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python3 scripts/export_cifar10.py "$@"
