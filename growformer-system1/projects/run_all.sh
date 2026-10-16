#!/usr/bin/env bash
# Build gf-system1, create the gaming / model-routing / guardrails brains with the
# open model attached, measure + calibrate on the seed sets, then serve all three.
#
# Usage (from anywhere):
#   bash growformer-system1/projects/run_all.sh            # build, set up, eval, serve
#   SERVE=0 bash growformer-system1/projects/run_all.sh    # stop after eval
#   FRESH=1 bash growformer-system1/projects/run_all.sh    # recreate brains (drops trained heads)
#
# Env: MODEL / TOKENIZER  [models/Qwen3-8B-Q4_K_M.gguf, models/tokenizer.json]
#      FEATURES           [server,candle,metal on macOS, server,candle elsewhere]
#      ADDR [127.0.0.1:8787]  LOG [decisions.jsonl]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
cd "${ROOT}"

MODEL="${MODEL:-models/Qwen3-8B-Q4_K_M.gguf}"
TOKENIZER="${TOKENIZER:-models/tokenizer.json}"
if [[ -z "${FEATURES:-}" ]]; then
  if [[ "$(uname)" == "Darwin" ]]; then FEATURES="server,candle,metal"; else FEATURES="server,candle"; fi
fi
S1="target/release/gf-system1"

for f in "${MODEL}" "${TOKENIZER}"; do
  [[ -f "$f" ]] || { echo "missing $f (set MODEL / TOKENIZER)" >&2; exit 1; }
done

# Always rebuild: an old binary lacks `init`, `--log`, rules and groups.
echo "[s1] building gf-system1 (--features ${FEATURES})…"
cargo build --release -p growformer-system1 --features "${FEATURES}" --bin gf-system1

for p in gaming model-routing guardrails; do
  D="growformer-system1/projects/${p}"
  B="${D}/agent/${p}.s1.json.gz"
  if [[ "${FRESH:-0}" == "1" || ! -f "${B}" ]]; then
    "${S1}" init --project "${D}/${p}.gf.toml"
  else
    # Keep trained heads / probes; just refresh rules and groups.
    "${S1}" update-rules --brain "${B}" --questions "${D}/system1/questions.toml"
  fi
  "${S1}" attach-lm --brain "${B}" --arch qwen3 --template chatml-nothink \
    --model "${MODEL}" --tokenizer "${TOKENIZER}"
done

# Measure the open model and fit its calibration where there are labelled rows.
for p in gaming model-routing guardrails; do
  D="growformer-system1/projects/${p}"
  if ls "${D}"/data/*.jsonl >/dev/null 2>&1; then
    echo; echo "=== ${p} ==="
    "${S1}" eval --brain "${D}/agent/${p}.s1.json.gz" --data "${D}/data" --lm --fit-lm-temperature --save
  else
    echo; echo "=== ${p}: no labelled rows yet (see ${D}/data/README.md); skipping eval"
  fi
done

[[ "${SERVE:-1}" == "1" ]] || exit 0
echo
exec "${S1}" serve \
  --brain "growformer-system1/projects/gaming/agent/gaming.s1.json.gz" \
  --brain "growformer-system1/projects/model-routing/agent/model-routing.s1.json.gz" \
  --brain "growformer-system1/projects/guardrails/agent/guardrails.s1.json.gz" \
  --addr "${ADDR:-127.0.0.1:8787}" --log "${LOG:-decisions.jsonl}"
