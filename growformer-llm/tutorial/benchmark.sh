#!/usr/bin/env bash
# Held-out benchmark for a Growformer project: how well do the brain, and the
# brain + label classifier, answer prompts they were NOT trained on?
#
# Usage:
#   bash growformer-llm/tutorial/benchmark.sh <project_dir> [out_dir]
#
# Steps (nothing in <project_dir> is modified):
#   1. Stratified split of the labelled JSONL rows (per semantic_intent) → train / test
#   2. Copy the project with only the train rows, train its brain + label model
#   3. Score on the test rows with `gf-llm brain-eval`:
#        brain (hybrid)  – current default path
#        label+brain     – classifier picks the label, brain explains
#   4. Print a summary table (+ JSON per mode in <out_dir>)
#
# Env:
#   SEED [7]  TEST_FRAC [0.2]
#   SPAWN     also run with this lattice merge threshold (e.g. 0.92 = old default merge,
#             2.0 = never merge); overrides [train].*_spawn_threshold in the copy's manifest
#   GF / GROWFORMER  binary paths (built if missing)

set -euo pipefail

if [[ $# -lt 1 ]]; then
  sed -n '2,20p' "$0"
  exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_ROOT="$(cd "${HERE}/.." && pwd)"
REPO_ROOT="$(cd "${LLM_ROOT}/.." && pwd)"
GF="${GF:-${REPO_ROOT}/target/release/gf-llm}"
GROWFORMER="${GROWFORMER:-${REPO_ROOT}/target/release/growformer}"
SEED="${SEED:-7}"
TEST_FRAC="${TEST_FRAC:-0.2}"

P="$(cd "$1" && pwd)"
NAME="$(basename "${P}")"
OUT="${2:-${LLM_ROOT}/agent-data/benchmark/${NAME}}"
mkdir -p "${OUT}"
OUT="$(cd "${OUT}" && pwd)"

[[ -x "${GF}" ]] || (cd "${REPO_ROOT}" && cargo build --release -p growformer-llm --bin gf-llm)
[[ -x "${GROWFORMER}" ]] || (cd "${REPO_ROOT}" && cargo build --release -p growformer \
  --no-default-features --features cli,native --bin growformer)

MANIFEST="$(basename "$(ls "${P}"/*.gf.toml | head -1)")"

# Labelled training files (same rules as train_lm.sh).
LABELLED=()
if ls "${P}"/data/train_*.jsonl >/dev/null 2>&1; then
  for f in "${P}"/data/train_*.jsonl; do LABELLED+=("$f"); done
else
  for f in "${P}"/data/*.jsonl; do
    case "$(basename "$f")" in *guardrails*|inference_*|eval_*|holdout_*) continue ;; esac
    head -1 "$f" | grep -q '"semantic_intent"' && LABELLED+=("$f")
  done
fi
SRC="${OUT}/labelled"
rm -rf "${SRC}" && mkdir -p "${SRC}"
for f in "${LABELLED[@]}"; do ln -s "$f" "${SRC}/"; done

# ── 1. split ──────────────────────────────────────────────────────────────────
"${GF}" jsonl-split "${SRC}" --train-out "${OUT}/train.jsonl" --test-out "${OUT}/test.jsonl" \
  --test-frac "${TEST_FRAC}" --seed "${SEED}"

# ── 2. project copy with train rows only ─────────────────────────────────────
make_copy() {
  local dst="$1" spawn="${2:-}"
  rm -rf "${dst}"
  mkdir -p "${dst}"
  (cd "${P}" && tar cf - --exclude='agent/*.bin' --exclude='*.bak*' .) | (cd "${dst}" && tar xf -)
  for f in "${LABELLED[@]}"; do rm -f "${dst}/data/$(basename "$f")"; done
  cp "${OUT}/train.jsonl" "${dst}/data/train_split.jsonl"
  mkdir -p "${dst}/agent"
  if [[ -n "${spawn}" ]]; then
    awk -v s="${spawn}" '/^(sentiment|chat)_spawn_threshold[[:space:]]*=/{next} {print} /^\[train\]/{print "sentiment_spawn_threshold = " s; print "chat_spawn_threshold = " s}' \
      "${dst}/${MANIFEST}" > "${dst}/${MANIFEST}.tmp" && mv "${dst}/${MANIFEST}.tmp" "${dst}/${MANIFEST}"
  fi
  echo "[bench] training brain in ${dst}"
  (cd "${dst}" && "${GROWFORMER}" --train-brain --project "${MANIFEST}" > "${dst}/train_brain.log" 2>&1)
  grep -E "gen\[g0\]: [0-9]+ lattice programs" "${dst}/train_brain.log" | head -1 || true
}

brain_of() {
  awk '/^\[infer\]/{f=1;next} /^\[/{f=0} f && /^brain[[:space:]]*=/{gsub(/.*=[[:space:]]*"|".*/,"");print;exit}' "$1/${MANIFEST}"
}

evaluate() {
  local dir="$1" tag="$2"
  local brain="${dir}/$(brain_of "${dir}")"
  (cd "${dir}" && "${GF}" brain-eval --brain "${brain}" --project "${MANIFEST}" \
      --test "${OUT}/test.jsonl" --json-out "${OUT}/${tag}-brain.json" 2>/dev/null | tail -1 \
    | sed "s/^/${tag} brain (hybrid)      : /")
  (cd "${dir}" && "${GF}" brain-eval --brain "${brain}" --project "${MANIFEST}" \
      --test "${OUT}/test.jsonl" --label-model "${OUT}/label-model.json" \
      --json-out "${OUT}/${tag}-label.json" 2>/dev/null | tail -1 \
    | sed "s/^/${tag} label+brain         : /")
}

"${GF}" label-train "${OUT}/labelled" --out "${OUT}/label-model.json" >/dev/null 2>&1 || true
# The model above saw all rows; retrain on the train split only for a fair score.
mkdir -p "${OUT}/train-only" && cp "${OUT}/train.jsonl" "${OUT}/train-only/train_split.jsonl"
"${GF}" label-train "${OUT}/train-only" --out "${OUT}/label-model.json"

make_copy "${OUT}/proj"
RESULTS="$(evaluate "${OUT}/proj" manifest)"
if [[ -n "${SPAWN:-}" ]]; then
  make_copy "${OUT}/proj-spawn" "${SPAWN}"
  RESULTS="${RESULTS}
$(evaluate "${OUT}/proj-spawn" "spawn=${SPAWN}")"
fi

echo
echo "=== held-out benchmark: ${NAME}  (seed ${SEED}, test ${TEST_FRAC}) ==="
echo "${RESULTS}"
echo "per-row JSON: ${OUT}/*.json"
