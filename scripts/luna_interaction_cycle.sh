#!/usr/bin/env bash
# Luna interaction → sparse CL cycle (no RL).
#
# Stages:
#   1) optional drain from storage node
#   2) audit-capture → label_queue.jsonl (skip if queue already reviewed)
#   3) promote reviewed labels → approved train/eval shard
#   4) fingerprint adapter on approved shard only
#   5) chat certify (+ optional real-traffic holdout prompts)
#
# Usage (from growformer/):
#   # Offline: you already drained + human-labeled the queue
#   STAGE=promote_adapt \
#   QUEUE=/path/to/capture_artifacts/label_queue.jsonl \
#   REVIEWED_BY=you \
#   bash scripts/luna_interaction_cycle.sh
#
#   # Full: drain → triage (stops for human label) 
#   STORAGE_URL=https://node.example STAGE=drain_audit bash scripts/luna_interaction_cycle.sh
#
# Env:
#   COMPANION     Luna root (default: $SK/spacekit-projects/companions/luna)
#   SK            SpaceKit root (default: /Users/astor/Projects/2026/spacekit)
#   BRAIN         base brain (default: $COMPANION/agent/luna-v3-3d.bin)
#   PROJECT       luna.gf.toml (default: $COMPANION/luna.gf.toml)
#   CAPTURE_DIR   capture artifacts (default: $COMPANION/capture_artifacts)
#   APPROVED_DIR  promoted shard out (default: $CAPTURE_DIR/approved_shard)
#   QUEUE         reviewed label_queue.jsonl
#   STAGE         drain_audit | promote_adapt | all  (default: promote_adapt)
#   STORAGE_URL   required for drain stages
#   REVIEWED_BY   attribution for promotion_manifest
#   FP_ALPHA      fingerprint EMA (default 0.25)
#   N             certify samples per prompt (default 6)
#   PROMOTE=1     copy adapted brain over BRAIN after PASS (keeps .bak-YYYYMMDD)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

SK="${SK:-/Users/astor/Projects/2026/spacekit}"
COMPANION="${COMPANION:-${SK}/spacekit-projects/companions/luna}"
PROJECT="${PROJECT:-${COMPANION}/luna.gf.toml}"
BRAIN="${BRAIN:-${COMPANION}/agent/luna-v3-3d.bin}"
CAPTURE_DIR="${CAPTURE_DIR:-${COMPANION}/capture_artifacts}"
APPROVED_DIR="${APPROVED_DIR:-${CAPTURE_DIR}/approved_shard}"
QUEUE="${QUEUE:-${CAPTURE_DIR}/label_queue.jsonl}"
STAGE="${STAGE:-promote_adapt}"
REVIEWED_BY="${REVIEWED_BY:-unknown}"
FP_ALPHA="${FP_ALPHA:-0.25}"
N="${N:-6}"
OUT_DIR="${OUT_DIR:-${ROOT}/agent-data/train-loop}"

echo "=== Luna interaction cycle ==="
echo "  STAGE=${STAGE}"
echo "  COMPANION=${COMPANION}"
echo "  BRAIN=${BRAIN}"
echo "  CAPTURE_DIR=${CAPTURE_DIR}"
echo "  QUEUE=${QUEUE}"

mkdir -p "${CAPTURE_DIR}" "${APPROVED_DIR}" "${OUT_DIR}"

run_drain() {
  if [[ -z "${STORAGE_URL:-}" ]]; then
    echo "STORAGE_URL required for drain" >&2
    exit 1
  fi
  python3 scripts/drain_capture.py \
    --storage-url "${STORAGE_URL}" \
    --out-dir "${CAPTURE_DIR}"
}

run_audit() {
  cargo run --release --bin growformer-demos -- \
    --audit-capture "${CAPTURE_DIR}" "${COMPANION}"
  echo ""
  echo "Human step: fill semantic_intent on ${QUEUE}"
  echo "Then re-run: STAGE=promote_adapt QUEUE=${QUEUE} REVIEWED_BY=… bash scripts/luna_interaction_cycle.sh"
}

run_promote_adapt() {
  if [[ ! -f "${QUEUE}" ]]; then
    echo "reviewed queue missing: ${QUEUE}" >&2
    echo "Run STAGE=drain_audit first, label intents, then promote_adapt." >&2
    exit 1
  fi

  # Fail fast if nothing labeled yet.
  labeled=$(python3 - <<PY
import json
from pathlib import Path
n=0
for line in Path("${QUEUE}").open():
    line=line.strip()
    if not line: continue
    o=json.loads(line)
    if (o.get("semantic_intent") or "").strip():
        n+=1
print(n)
PY
)
  if [[ "${labeled}" -lt 1 ]]; then
    echo "No labeled rows in ${QUEUE} (semantic_intent empty). Label before promote." >&2
    exit 1
  fi

  python3 scripts/promote_label_queue.py \
    --queue "${QUEUE}" \
    --companion "${COMPANION}" \
    --out-dir "${APPROVED_DIR}" \
    --reviewed-by "${REVIEWED_BY}"

  EVAL_PROMPTS=$(python3 - <<PY
import json
from pathlib import Path
m=json.loads(Path("${APPROVED_DIR}/promotion_manifest.json").read_text())
print(m.get("prompts_path",""))
PY
)

  GATE_CMD="BRAIN=\"\$BRAIN\" N=${N} node scripts/certify_chat.mjs"
  if [[ -n "${EVAL_PROMPTS}" && -f "${EVAL_PROMPTS}" ]]; then
    GATE_CMD="BRAIN=\"\$BRAIN\" N=${N} EXTRA_EVAL=\"${EVAL_PROMPTS}\" node scripts/certify_chat.mjs"
  fi

  PROJECT="${PROJECT}" \
  BRAIN="${BRAIN}" \
  DATA_DIR="${APPROVED_DIR}" \
  MODE=fingerprint \
  LOOPS=1 \
  FP_ALPHA="${FP_ALPHA}" \
  OUT_DIR="${OUT_DIR}" \
  GATE_CMD="${GATE_CMD}" \
  bash scripts/train_loop.sh

  latest="${OUT_DIR}/loop1-fingerprint.bin"
  if [[ ! -f "${latest}" ]]; then
    echo "expected adapted brain missing: ${latest}" >&2
    exit 1
  fi
  echo "[cycle] adapted brain: ${latest}"

  if [[ "${PROMOTE:-0}" == "1" ]]; then
    bak="${BRAIN}.bak-$(date +%Y%m%d)"
    cp "${BRAIN}" "${bak}"
    cp "${latest}" "${BRAIN}"
    echo "[cycle] promoted → ${BRAIN} (backup ${bak})"
  else
    echo "[cycle] not promoted (set PROMOTE=1 to install after PASS)"
  fi
}

case "${STAGE}" in
  drain_audit)
    run_drain
    run_audit
    ;;
  audit)
    run_audit
    ;;
  promote_adapt)
    run_promote_adapt
    ;;
  all)
    run_drain
    run_audit
    echo "Stopping after audit for human labeling (STAGE=all does not auto-promote unlabeled queues)."
    ;;
  *)
    echo "STAGE must be drain_audit|audit|promote_adapt|all" >&2
    exit 1
    ;;
esac
