#!/usr/bin/env bash
# Train (or retrain) a project's Growformer brain from its *.gf.toml manifest.
#
# Usage:
#   bash growformer-llm/tutorial/train_brain.sh <project_dir>
#
# Writes [train].brain_output from the manifest (e.g. agent/crypto-brain.bin) and a label
# classifier (agent/label-model.json) inside the project folder. When the manifest has a
# [system1] section, also trains the System 1 decision brain ([system1].brain_output). The crypto tutorial
# project trains both in about a minute on a laptop.
#
# Env: GROWFORMER  path to the growformer CLI [<repo>/target/release/growformer, built if missing]
#      S1          path to gf-system1 [<repo>/target/release/gf-system1, built if missing]

set -euo pipefail

if [[ $# -lt 1 ]]; then
  sed -n '2,12p' "$0"
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

# System 1 decision brain (typed Choice / Score / Noul answers), when configured.
if grep -q '^\[system1\]' "${MANIFEST}"; then
  S1="${S1:-${REPO_ROOT}/target/release/gf-system1}"
  if [[ ! -x "${S1}" ]]; then
    echo "[tutorial] building gf-system1 (first run only)…"
    (cd "${REPO_ROOT}" && cargo build --release -p growformer-system1 --features server,candle --bin gf-system1)
  fi
  "${S1}" train --project "${MANIFEST}"
  S1_REL="$(awk '/^\[system1\]/{f=1;next} /^\[/{f=0} f && /^brain_output[[:space:]]*=/{gsub(/.*=[[:space:]]*"|".*/,"");print;exit}' "${MANIFEST}")"
fi

OUT_REL="$(awk '/^\[train\]/{f=1;next} /^\[/{f=0} f && /^brain_output[[:space:]]*=/{gsub(/.*=[[:space:]]*"|".*/,"");print;exit}' "${MANIFEST}")"
echo
echo "[tutorial] brain       → ${P}/${OUT_REL}"
echo "[tutorial] label model → ${P}/agent/label-model.json"
if [[ -n "${S1_REL:-}" ]]; then
  echo "[tutorial] system 1    → ${P}/${S1_REL}"
  echo "           try: ${S1} decide --brain ${P}/${S1_REL} --state \"Bitcoin crashed after the ETF delay\""
fi
echo "Next: bash ${HERE}/infer.sh ${P} \"your prompt\""
