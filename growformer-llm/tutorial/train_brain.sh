#!/usr/bin/env bash
# Train (or retrain) a project's Growformer brain from its *.gf.toml manifest.
#
# Usage:
#   bash growformer-llm/tutorial/train_brain.sh <project_dir>
#
# Writes [train].brain_output from the manifest (e.g. agent/crypto-brain.bin) and a label
# classifier (agent/label-model.json) inside the project folder. The crypto tutorial
# project trains both in about a minute on a laptop.
#
# Env: GROWFORMER  path to the growformer CLI [<repo>/target/release/growformer, built if missing]

set -euo pipefail

if [[ $# -lt 1 ]]; then
  sed -n '2,10p' "$0"
  exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
GROWFORMER="${GROWFORMER:-${REPO_ROOT}/target/release/growformer}"

P="$(cd "$1" && pwd)"
MANIFEST="$(ls "${P}"/*.gf.toml 2>/dev/null | head -1 || true)"
if [[ -z "${MANIFEST}" ]]; then
  echo "no *.gf.toml manifest in ${P}" >&2
  exit 1
fi

if [[ ! -x "${GROWFORMER}" ]]; then
  echo "[tutorial] building the growformer CLI (first run only)…"
  (cd "${REPO_ROOT}" && cargo build --release -p growformer \
    --no-default-features --features cli,native --bin growformer)
fi

cd "${P}"
"${GROWFORMER}" --train-brain --project "$(basename "${MANIFEST}")"

# Label classifier: picks the label; the brain then explains from that label's examples.
GF="${GF:-${REPO_ROOT}/target/release/gf-llm}"
if [[ ! -x "${GF}" ]]; then
  (cd "${REPO_ROOT}" && cargo build --release -p growformer-llm --bin gf-llm)
fi
LABEL_SRC="${P}/data"
if ! ls "${P}"/data/train_*.jsonl >/dev/null 2>&1; then
  # Companion-style project: only labelled chat rows (skip fragments/guardrails).
  LABEL_SRC="$(mktemp -d)"
  for f in "${P}"/data/*.jsonl; do
    case "$(basename "$f")" in *guardrails*|inference_*|eval_*|holdout_*) continue ;; esac
    head -1 "$f" | grep -q '"semantic_intent"' && ln -s "$f" "${LABEL_SRC}/"
  done
fi
"${GF}" label-train "${LABEL_SRC}" --out "${P}/agent/label-model.json"

OUT_REL="$(awk '/^\[train\]/{f=1;next} /^\[/{f=0} f && /^brain_output[[:space:]]*=/{gsub(/.*=[[:space:]]*"|".*/,"");print;exit}' "${MANIFEST}")"
echo
echo "[tutorial] brain       → ${P}/${OUT_REL}"
echo "[tutorial] label model → ${P}/agent/label-model.json"
echo "Next: bash ${HERE}/infer.sh ${P} \"your prompt\""
