#!/usr/bin/env bash
# Train a vanilla growformer-llm checkpoint on Growformer crypto/fintech JSONL.
#
# Product note: Path A (brain retrieve + label) is the portable domain response
# path. Chatbots should default to `gf-llm chat --compose brain`. This script
# builds an optional domain LM for polish / experimental lm compose.
#
# Usage (from growformer-llm/):
#   bash scripts/train_domain_vanilla.sh
#   CHAT=1 STEPS=8000 bash scripts/train_domain_vanilla.sh   # clean chat + turn-aligned
#
# Env:
#   DOMAIN     crypto | fintech | both (default: both)
#   DATA_ROOT  Growformer data root (default: ../growformer/data)
#   STEPS      train steps (default: 4000; use 8000+ for chat)
#   VOCAB      BPE target vocab (default: 2048)
#   FEATURES   cargo features (default: vanilla-lm,brain-memory)
#   CHAT=1     clean ### User:/### Assistant: corpus + --turn-aligned + seq 256
#   SEQ_LEN    override sequence length (CHAT default 256, else 128)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DATA_ROOT="${DATA_ROOT:-${ROOT}/../growformer/data}"
DOMAIN="${DOMAIN:-both}"
STEPS="${STEPS:-4000}"
VOCAB="${VOCAB:-2048}"
FEATURES="${FEATURES:-vanilla-lm,brain-memory}"
OUT_DIR="${OUT_DIR:-${ROOT}/data/domain}"
CKPT_DIR="${CKPT_DIR:-${ROOT}/agent-data}"

mkdir -p "${OUT_DIR}" "${CKPT_DIR}"

DIRS=()
case "${DOMAIN}" in
  crypto)  DIRS+=("${DATA_ROOT}/crypto") ;;
  fintech) DIRS+=("${DATA_ROOT}/fintech") ;;
  both)
    DIRS+=("${DATA_ROOT}/crypto" "${DATA_ROOT}/fintech")
    ;;
  *)
    echo "DOMAIN must be crypto|fintech|both" >&2
    exit 1
    ;;
esac

for d in "${DIRS[@]}"; do
  if [[ ! -d "${d}" ]]; then
    echo "missing data dir: ${d}" >&2
    exit 1
  fi
done

SUFFIX=""
if [[ "${CHAT:-0}" == "1" ]]; then
  SUFFIX="-chat"
fi
TXT="${OUT_DIR}/${DOMAIN}${SUFFIX}.txt"
TOK="${OUT_DIR}/${DOMAIN}${SUFFIX}.tok"
BIN="${OUT_DIR}/${DOMAIN}${SUFFIX}.bin"
TRAIN_BIN="${OUT_DIR}/${DOMAIN}${SUFFIX}-train.bin"
HELD_BIN="${OUT_DIR}/${DOMAIN}${SUFFIX}-heldout.bin"
CKPT="${CKPT_DIR}/domain-${DOMAIN}${SUFFIX}-vanilla.json"

if [[ "${CHAT:-0}" == "1" ]]; then
  SEQ_LEN="${SEQ_LEN:-256}"
  TRAIN_EXTRA=(--turn-aligned --seq-len "${SEQ_LEN}")
else
  SEQ_LEN="${SEQ_LEN:-128}"
  TRAIN_EXTRA=(--seq-len "${SEQ_LEN}")
fi

run() {
  cargo run --release --no-default-features --features "${FEATURES}" \
    --bin gf-llm -- "$@"
}

echo "[domain] jsonl → ${TXT}"
if [[ "${CHAT:-0}" == "1" ]]; then
  # Clean assistant targets by default (strip meta rationales).
  run jsonl-to-txt "${DIRS[@]}" --chat --chat-clean true --out "${TXT}"
else
  run jsonl-to-txt "${DIRS[@]}" --out "${TXT}"
fi

echo "[domain] tokenize → ${TOK}"
run tokenize "${TXT}" "${VOCAB}" "${TOK}"

echo "[domain] encode → ${BIN}"
run encode "${TXT}" "${TOK}" "${BIN}"

echo "[domain] split → train/heldout"
run split "${BIN}" "${TRAIN_BIN}" "${HELD_BIN}" --train-frac 0.9

echo "[domain] train vanilla → ${CKPT}"
run train "${TOK}" "${TRAIN_BIN}" "${HELD_BIN}" \
  --checkpoint-out "${CKPT}" \
  "${TRAIN_EXTRA[@]}" \
  --steps "${STEPS}" \
  --d-model 16 --d-ff 64 --n-blocks 4 --n-heads 4 \
  --tie-embeddings \
  --init-seed 1000 \
  --sample-every 500

echo "[domain] held-out eval"
run eval \
  --checkpoint "${CKPT}" \
  --tokenizer "${TOK}" \
  --train-bin "${TRAIN_BIN}" \
  "${HELD_BIN}" --seq-len "${SEQ_LEN}" --windows 32 \
  --run-id "domain-${DOMAIN}${SUFFIX}-vanilla" \
  --no-ledger

echo "[domain] done"
echo "  checkpoint: ${CKPT}"
echo "  tokenizer:  ${TOK}"
echo "Product chat (brain compose):"
echo "  cargo run --release --no-default-features --features ${FEATURES} --bin gf-llm -- \\"
echo "    chat --compose brain --brain <brain.bin> --project <project.gf.toml> \\"
echo "    --message 'Bitcoin crashed after the ETF delay'"
echo "Optional LM polish (after chat train):"
echo "  ... chat --compose polish --checkpoint ${CKPT} --tokenizer ${TOK} --brain ... --project ..."
