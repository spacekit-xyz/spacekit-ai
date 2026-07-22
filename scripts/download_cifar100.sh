#!/usr/bin/env bash
# Download CIFAR-100 (binary) for future Split-CIFAR-100 promote–freeze eval.
# Phase 4c currently runs a synthetic scaffold only — this script prepares data.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${CIFAR_ROOT:-$ROOT/data/cifar-100-binary}"
mkdir -p "$(dirname "$DEST")"
URL="https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"
TMP="$(mktemp -d)"
echo "Fetching $URL → $DEST"
curl -L "$URL" -o "$TMP/cifar-100-binary.tar.gz"
tar -xzf "$TMP/cifar-100-binary.tar.gz" -C "$TMP"
rm -rf "$DEST"
mv "$TMP/cifar-100-binary" "$DEST"
rm -rf "$TMP"
echo "OK: CIFAR-100 binary at $DEST"
echo "Next: cargo run --release --bin growformer-demos -- --phase4e-split-cifar-lite"
