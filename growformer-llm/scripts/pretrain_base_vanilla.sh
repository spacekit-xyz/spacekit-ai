#!/usr/bin/env bash
# Pretrain a general vanilla base LM on TinyStories, to fine-tune domain
# specialists from (see train_domain_vanilla.sh with BASE_CKPT/BASE_TOK).
#
# Domain corpora here are tiny (tens to hundreds of examples); training a
# specialist from scratch mostly memorises them. Learning general English
# first and then adapting the upper blocks generalises much better.
#
# Usage (from growformer-llm/):
#   bash scripts/get_tinystories.sh            # ~20 MB validation shard
#   bash scripts/pretrain_base_vanilla.sh
#   FULL=1 bash scripts/get_tinystories.sh && TXT=data/TinyStories-train.txt STEPS=40000 \
#     bash scripts/pretrain_base_vanilla.sh
#
# Env:
#   TXT        source text (default: data/TinyStories-valid.txt)
#   STEPS      train steps (default: 20000)
#   VOCAB      BPE vocab (default: 2048) — domain fine-tunes reuse this tokenizer
#   D_MODEL / D_FF / N_HEADS / N_BLOCKS   (default: 128 / 512 / 4 / 4)
#   SEQ_LEN    (default: 128)   GRAD_ACCUM (default: 8)   LR_MAX (default: 1e-3)
#   OUT_DIR    (default: agent-data/base)
#   GF_THREADS worker threads for batch gradients (default: all cores)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

TXT="${TXT:-${ROOT}/data/TinyStories-valid.txt}"
STEPS="${STEPS:-20000}"
VOCAB="${VOCAB:-2048}"
D_MODEL="${D_MODEL:-128}"
D_FF="${D_FF:-512}"
N_HEADS="${N_HEADS:-4}"
N_BLOCKS="${N_BLOCKS:-4}"
SEQ_LEN="${SEQ_LEN:-128}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR_MAX="${LR_MAX:-1e-3}"
FEATURES="${FEATURES:-vanilla-lm,brain-memory}"
OUT_DIR="${OUT_DIR:-${ROOT}/agent-data/base}"

if [[ ! -f "${TXT}" ]]; then
  echo "missing ${TXT} — run: bash scripts/get_tinystories.sh" >&2
  exit 1
fi
mkdir -p "${OUT_DIR}"

TOK="${OUT_DIR}/base.tok"
BIN="${OUT_DIR}/base.bin"
TRAIN_BIN="${OUT_DIR}/base-train.bin"
HELD_BIN="${OUT_DIR}/base-heldout.bin"
CKPT="${OUT_DIR}/base-vanilla.json"

run() {
  cargo run --release --no-default-features --features "${FEATURES}" \
    --bin gf-llm -- "$@"
}

echo "[base] tokenize → ${TOK}"
run tokenize "${TXT}" "${VOCAB}" "${TOK}"
echo "[base] encode → ${BIN}"
run encode "${TXT}" "${TOK}" "${BIN}"
echo "[base] split → train/heldout"
run split "${BIN}" "${TRAIN_BIN}" "${HELD_BIN}" --train-frac 0.95

echo "[base] train → ${CKPT}"
run train "${TOK}" "${TRAIN_BIN}" "${HELD_BIN}" \
  --checkpoint-out "${CKPT}" \
  --seq-len "${SEQ_LEN}" \
  --steps "${STEPS}" \
  --no-param-match \
  --d-model "${D_MODEL}" --d-ff "${D_FF}" --n-blocks "${N_BLOCKS}" --n-heads "${N_HEADS}" \
  --tie-embeddings \
  --grad-accum "${GRAD_ACCUM}" \
  --lr-max "${LR_MAX}" \
  --val-every 500 --val-chunks 64 --patience 6 \
  --subject tinystories-base \
  --init-seed 1000 \
  --sample-every 2000

echo "[base] held-out eval"
run eval --checkpoint "${CKPT}" --tokenizer "${TOK}" --train-bin "${TRAIN_BIN}" \
  "${HELD_BIN}" --seq-len "${SEQ_LEN}" --windows 64 --no-ledger

echo "[base] done"
echo "  checkpoint: ${CKPT}   (final/resumable state: ${CKPT%.json}.last.json)"
echo "  tokenizer:  ${TOK}"
echo "Fine-tune a domain specialist from it:"
echo "  BASE_CKPT=${CKPT} BASE_TOK=${TOK} bash scripts/train_domain_vanilla.sh"
