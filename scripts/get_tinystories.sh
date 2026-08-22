#!/usr/bin/env bash
# Download TinyStories text shards from Hugging Face.
# Default: validation only (~20 MB) for a quick CPU sanity run.
#   FULL=1 bash scripts/get_tinystories.sh  — also fetch ~1 GB train shard.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/data"
mkdir -p "${DEST}"

BASE="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"

fetch() {
  local name="$1"
  local url="${BASE}/${name}?download=true"
  echo "[get_tinystories] fetching ${name}"
  curl -L --fail --retry 3 -o "${DEST}/${name}" "${url}"
}

fetch "TinyStories-valid.txt"

if [[ "${FULL:-0}" == "1" ]]; then
  fetch "TinyStories-train.txt"
fi

echo "[get_tinystories] done → ${DEST}"
