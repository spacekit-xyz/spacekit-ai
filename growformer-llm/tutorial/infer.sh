#!/usr/bin/env bash
# Run inference against a Growformer project: brain memory (product path), optionally
# with the LM trained by train_lm.sh.
#
# Usage:
#   bash growformer-llm/tutorial/infer.sh <project_dir> "prompt"      # one prompt
#   bash growformer-llm/tutorial/infer.sh <project_dir>               # interactive chat
#   PROMPTS=<file> bash growformer-llm/tutorial/infer.sh <project_dir>  # batch (one prompt per line, # = comment)
#
# Env:
#   COMPOSE  brain  — answer = retrieved brain memory (default, product path)
#            polish — brain answer, rephrased by the LM (falls back to the brain answer
#                     if the rewrite drops the label or invents content)
#            lm     — experimental: LM generates, conditioned on brain memory
#            diag   — routing / memory diagnostics (brain-infer --brain-only -v)
#   BRAIN    brain .bin (default: [infer].brain from the project's *.gf.toml)
#   LM_DIR   folder with <name>-lm.json + <name>.tok (default: where train_lm.sh wrote them)
#   HYBRID   true|false — try the raw-lattice shortcut first (default true)
#   LABEL    1 = use <project>/agent/label-model.json when present (default), 0 = brain only
#   GF       path to the gf-llm binary

set -euo pipefail

if [[ $# -lt 1 ]]; then
  sed -n '2,20p' "$0"
  exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_ROOT="$(cd "${HERE}/.." && pwd)"
REPO_ROOT="$(cd "${LLM_ROOT}/.." && pwd)"
GF="${GF:-${REPO_ROOT}/target/release/gf-llm}"

P="$(cd "$1" && pwd)"
NAME="$(basename "${P}")"
PROMPT="${2:-}"
COMPOSE="${COMPOSE:-brain}"
HYBRID="${HYBRID:-true}"
LM_DIR="${LM_DIR:-${LLM_ROOT}/agent-data/tutorial/${NAME}}"

if [[ ! -x "${GF}" ]]; then
  echo "[tutorial] building gf-llm (first run only)…"
  (cd "${REPO_ROOT}" && cargo build --release -p growformer-llm --bin gf-llm)
fi

MANIFEST="$(ls "${P}"/*.gf.toml 2>/dev/null | head -1 || true)"
if [[ -z "${MANIFEST}" ]]; then
  echo "no *.gf.toml manifest in ${P}" >&2
  exit 1
fi

if [[ -z "${BRAIN:-}" ]]; then
  # [infer] brain = "agent/xyz.bin"  (path relative to the manifest)
  REL="$(awk '/^\[infer\]/{f=1;next} /^\[/{f=0} f && /^brain[[:space:]]*=/{gsub(/.*=[[:space:]]*"|".*/,"");print;exit}' "${MANIFEST}")"
  BRAIN="${P}/${REL}"
fi
if [[ ! -f "${BRAIN}" ]]; then
  echo "brain not found: ${BRAIN} (set BRAIN=… or train one — see tutorial/README.md §5)" >&2
  exit 1
fi

LABEL_ARGS=()
LABEL_MODEL="${LABEL_MODEL:-${P}/agent/label-model.json}"
if [[ "${LABEL:-1}" == "1" && -f "${LABEL_MODEL}" ]]; then
  LABEL_ARGS=(--label-model "${LABEL_MODEL}")
fi

LM_ARGS=()
if [[ "${COMPOSE}" == "polish" || "${COMPOSE}" == "lm" ]]; then
  CKPT="${LM_DIR}/${NAME}-lm.json"
  TOK="${LM_DIR}/${NAME}.tok"
  if [[ ! -f "${CKPT}" || ! -f "${TOK}" ]]; then
    echo "no LM at ${CKPT} — run: bash ${HERE}/train_lm.sh ${P}" >&2
    exit 1
  fi
  LM_ARGS=(--checkpoint "${CKPT}" --tokenizer "${TOK}")
fi

# Relative data/ paths in the manifest resolve against the project folder.
cd "${P}"

run_one() {
  local q="$1"
  if [[ "${COMPOSE}" == "diag" ]]; then
    "${GF}" brain-infer --brain "${BRAIN}" --project "${MANIFEST}" --hybrid "${HYBRID}" \
      ${LABEL_ARGS[@]+"${LABEL_ARGS[@]}"} --prompt "${q}" --brain-only -v
  else
    "${GF}" chat --compose "${COMPOSE}" --hybrid "${HYBRID}" \
      --brain "${BRAIN}" --project "${MANIFEST}" ${LABEL_ARGS[@]+"${LABEL_ARGS[@]}"} \
      ${LM_ARGS[@]+"${LM_ARGS[@]}"} --message "${q}" 2>/dev/null \
      | sed -n '/^Assistant>/,$p'   # brain logs go to stdout; keep only the reply
  fi
}

if [[ -n "${PROMPTS:-}" ]]; then
  while IFS= read -r q || [[ -n "$q" ]]; do
    [[ -z "${q// }" || "${q}" == \#* ]] && continue
    echo "You> ${q}"
    run_one "${q}"
    echo
  done < "${PROMPTS}"
elif [[ -n "${PROMPT}" ]]; then
  run_one "${PROMPT}"
else
  exec "${GF}" chat --compose "${COMPOSE}" --hybrid "${HYBRID}" \
    --brain "${BRAIN}" --project "${MANIFEST}" ${LABEL_ARGS[@]+"${LABEL_ARGS[@]}"} \
    ${LM_ARGS[@]+"${LM_ARGS[@]}"}
fi
