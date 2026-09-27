#!/usr/bin/env bash
# Train a vanilla growformer-llm chat LM on a Growformer project's JSONL data.
#
# Usage (from anywhere):
#   bash growformer-llm/tutorial/train_lm.sh <project_dir> [out_dir]
#
#   bash growformer-llm/tutorial/train_lm.sh growformer-llm/tutorial/projects/crypto-sentiment
#   bash growformer-llm/tutorial/train_lm.sh ~/…/spacekit-projects/companions/luna
#
# A project dir is any folder with a `*.gf.toml` manifest and a `data/` folder of JSONL rows
# ({"text", "semantic_intent", "expected_response", …}).
#
# Corpus rules (auto-detected, override with CHAT_CLEAN=true|false):
#   * data/ has train_*.jsonl  → sentiment-style project: only train_*.jsonl are used and
#     replies are cleaned (keeps the "NEGATIVE (mild) — …" label, strips meta, ≤160 chars).
#   * otherwise (e.g. Luna)    → every *.jsonl whose rows carry semantic_intent is used
#     (fragment / guardrail files are skipped) and replies are kept verbatim.
#
# Env (defaults in brackets):
#   STEPS [4000]  VOCAB [1024]  SEQ_LEN [256]  D_MODEL [128]  D_FF [512]  N_HEADS [4]
#   N_BLOCKS [4]  GRAD_ACCUM [8]  LR_MAX [1e-3]  PATIENCE [5]  VAL_EVERY [200]
#   GF        path to the gf-llm binary [<repo>/target/release/gf-llm, built if missing]
#   GF_THREADS  cap worker threads (default: all cores)
#   RESUME=1  continue an interrupted run from <out>/<name>-lm.last.json

set -euo pipefail

if [[ $# -lt 1 ]]; then
  sed -n '2,25p' "$0"
  exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_ROOT="$(cd "${HERE}/.." && pwd)"
REPO_ROOT="$(cd "${LLM_ROOT}/.." && pwd)"

P="$(cd "$1" && pwd)"
NAME="$(basename "${P}")"
OUT="${2:-${LLM_ROOT}/agent-data/tutorial/${NAME}}"
mkdir -p "${OUT}"
OUT="$(cd "${OUT}" && pwd)"

STEPS="${STEPS:-4000}"
VOCAB="${VOCAB:-1024}"
SEQ_LEN="${SEQ_LEN:-256}"
D_MODEL="${D_MODEL:-128}"
D_FF="${D_FF:-512}"
N_HEADS="${N_HEADS:-4}"
N_BLOCKS="${N_BLOCKS:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR_MAX="${LR_MAX:-1e-3}"
PATIENCE="${PATIENCE:-5}"
VAL_EVERY="${VAL_EVERY:-200}"
GF="${GF:-${REPO_ROOT}/target/release/gf-llm}"

if [[ ! -x "${GF}" ]]; then
  echo "[tutorial] building gf-llm (first run only)…"
  (cd "${REPO_ROOT}" && cargo build --release -p growformer-llm --bin gf-llm)
fi

if [[ ! -d "${P}/data" ]]; then
  echo "no data/ folder in ${P}" >&2
  exit 1
fi

# ── 1. JSONL → chat corpus ────────────────────────────────────────────────────
if ls "${P}"/data/train_*.jsonl >/dev/null 2>&1; then
  SRC="${P}/data"
  CHAT_CLEAN="${CHAT_CLEAN:-true}"
else
  # Companion-style project: link only labelled chat rows (skip fragments/guardrails).
  SRC="${OUT}/jsonl"
  rm -rf "${SRC}" && mkdir -p "${SRC}"
  for f in "${P}"/data/*.jsonl; do
    case "$(basename "$f")" in
      *guardrails*|inference_*|eval_*|holdout_*) continue ;;
    esac
    if head -1 "$f" | grep -q '"semantic_intent"'; then
      ln -s "$f" "${SRC}/"
    fi
  done
  CHAT_CLEAN="${CHAT_CLEAN:-false}"
fi

echo "[tutorial] project=${NAME}  corpus from ${SRC}  (chat-clean=${CHAT_CLEAN})"
"${GF}" jsonl-to-txt "${SRC}" --chat --chat-clean "${CHAT_CLEAN}" --out "${OUT}/chat.txt"

# ── 2. Tokenize → encode → split ─────────────────────────────────────────────
"${GF}" tokenize "${OUT}/chat.txt" "${VOCAB}" "${OUT}/${NAME}.tok"
"${GF}" encode   "${OUT}/chat.txt" "${OUT}/${NAME}.tok" "${OUT}/${NAME}.bin"
"${GF}" split    "${OUT}/${NAME}.bin" "${OUT}/train.bin" "${OUT}/heldout.bin" --train-frac 0.9

# ── 3. Train (best-val checkpoint + early stopping) ──────────────────────────
CKPT="${OUT}/${NAME}-lm.json"
EXTRA=()
if [[ "${RESUME:-0}" == "1" ]]; then
  EXTRA+=(--resume "${OUT}/${NAME}-lm.last.json")
fi
"${GF}" train "${OUT}/${NAME}.tok" "${OUT}/train.bin" "${OUT}/heldout.bin" \
  --checkpoint-out "${CKPT}" \
  --turn-aligned --seq-len "${SEQ_LEN}" --steps "${STEPS}" \
  --no-param-match --d-model "${D_MODEL}" --d-ff "${D_FF}" \
  --n-heads "${N_HEADS}" --n-blocks "${N_BLOCKS}" --tie-embeddings \
  --grad-accum "${GRAD_ACCUM}" --lr-max "${LR_MAX}" \
  --val-every "${VAL_EVERY}" --patience "${PATIENCE}" \
  --subject "${NAME}" --sample-every 0 \
  ${EXTRA[@]+"${EXTRA[@]}"}

# ── 4. Held-out eval (writes bits/byte into the .gfcard.json) ────────────────
"${GF}" eval --checkpoint "${CKPT}" --tokenizer "${OUT}/${NAME}.tok" \
  --train-bin "${OUT}/train.bin" "${OUT}/heldout.bin" \
  --seq-len "${SEQ_LEN}" --windows 32 --no-ledger

cat <<EOF

[tutorial] done
  checkpoint : ${CKPT}
  tokenizer  : ${OUT}/${NAME}.tok
  card       : ${OUT}/${NAME}-lm.gfcard.json
Next:
  bash ${HERE}/infer.sh ${P} "your prompt"                 # brain answer (product path)
  COMPOSE=polish bash ${HERE}/infer.sh ${P} "your prompt"  # brain answer, LM rephrase
EOF
