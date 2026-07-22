#!/usr/bin/env bash
# Continual product loop for growformer brains.
#
# Modes:
#   fingerprint — EMA-augment structural + topic fingerprints (default for companions;
#                 does not retrain LearnedRouter / does not touch lattices)
#   router      — retrain LearnedRouter when multi-class groups exist; always also
#                 fingerprint-augments. Mono-class / pet_chat→legacy-group brains skip router.
#   overlay     — train overlay brain from SHARD_DIR, merge into base, export
#   full        — full --train-brain (use sparingly)
#
# Always ends with an optional GATE_CMD (frozen battery / companion matrix).
#
# Usage (from growformer/):
#   bash scripts/train_loop.sh
#   MODE=fingerprint LOOPS=2 bash scripts/train_loop.sh
#   MODE=router LOOPS=3 bash scripts/train_loop.sh
#   MODE=overlay SHARD_DIR=../path/to/new_jsonl_dir bash scripts/train_loop.sh
#
# Env:
#   PROJECT      *.gf.toml (required for most companions / domain brains)
#   BRAIN        input brain.bin (default: from [infer].brain via project, or agent/*.bin)
#   OUT_DIR      where adapted brains land (default: agent-data/train-loop)
#   MODE         fingerprint | router | overlay | full  (default: fingerprint)
#   LOOPS        iterations (default: 1)
#   SHARD_DIR    JSONL dir for overlay train (MODE=overlay)
#   DATA_DIR     JSONL dir override for fingerprint/router (approved capture shard)
#   GATE_CMD     shell command run after each loop (exit non-zero aborts)
#   ALSO_CLASSIFIER=1  also retrain ActionClassifier in router mode
#   EPOCHS       brain-epochs for router/full (default: 30)
#   FP_ALPHA     EMA blend for fingerprint mode (default: 0.25; higher = more new data)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

MODE="${MODE:-fingerprint}"
LOOPS="${LOOPS:-1}"
OUT_DIR="${OUT_DIR:-${ROOT}/agent-data/train-loop}"
EPOCHS="${EPOCHS:-30}"
FP_ALPHA="${FP_ALPHA:-0.25}"
PROJECT="${PROJECT:-}"
BRAIN="${BRAIN:-}"
SHARD_DIR="${SHARD_DIR:-}"
DATA_DIR="${DATA_DIR:-}"
GATE_CMD="${GATE_CMD:-}"

mkdir -p "${OUT_DIR}"

run_gf() {
  cargo run --release --features cli --bin growformer -- "$@"
}

if [[ -z "${PROJECT}" && -z "${BRAIN}" ]]; then
  echo "Set PROJECT=path/to/*.gf.toml and/or BRAIN=path/to/brain.bin" >&2
  exit 1
fi

current_brain="${BRAIN}"
for ((i = 1; i <= LOOPS; i++)); do
  echo ""
  echo "========== train-loop ${i}/${LOOPS} mode=${MODE} =========="
  out="${OUT_DIR}/loop${i}-${MODE}.bin"

  # Rebuild args each loop so --brain is never passed twice.
  LOOP_ARGS=()
  if [[ -n "${PROJECT}" ]]; then
    LOOP_ARGS+=(--project "${PROJECT}")
  fi
  if [[ -n "${current_brain}" ]]; then
    LOOP_ARGS+=(--brain "${current_brain}")
  fi
  # Approved capture shard: train only on that JSONL dir (not full companion data/).
  if [[ -n "${DATA_DIR}" && "${MODE}" != "overlay" ]]; then
    LOOP_ARGS+=(--data-dir "${DATA_DIR}")
  fi

  case "${MODE}" in
    fingerprint)
      run_gf "${LOOP_ARGS[@]}" \
        --train-fingerprint-adapter \
        --fingerprint-alpha "${FP_ALPHA}" \
        --brain-output "${out}"
      ;;
    router)
      EXTRA=(--train-router-adapter --brain-epochs "${EPOCHS}" --brain-output "${out}" --fingerprint-alpha "${FP_ALPHA}")
      if [[ "${ALSO_CLASSIFIER:-0}" == "1" ]]; then
        EXTRA+=(--also-classifier)
      fi
      run_gf "${LOOP_ARGS[@]}" "${EXTRA[@]}"
      ;;
    overlay)
      if [[ -z "${SHARD_DIR}" ]]; then
        echo "MODE=overlay requires SHARD_DIR=…" >&2
        exit 1
      fi
      overlay="${OUT_DIR}/loop${i}-overlay-raw.bin"
      run_gf "${LOOP_ARGS[@]}" \
        --train-brain --auto \
        --data-dir "${SHARD_DIR}" \
        --brain-output "${overlay}"
      base_for_merge="${current_brain}"
      if [[ -z "${base_for_merge}" ]]; then
        echo "MODE=overlay needs BRAIN= base to merge into" >&2
        exit 1
      fi
      run_gf --merge-brain --brain "${base_for_merge}" --overlay-brain "${overlay}" --brain-output "${out}"
      ;;
    full)
      run_gf "${LOOP_ARGS[@]}" --train-brain --auto --brain-output "${out}"
      ;;
    *)
      echo "MODE must be fingerprint|router|overlay|full" >&2
      exit 1
      ;;
  esac

  current_brain="${out}"
  echo "[train-loop] current brain → ${current_brain}"

  if [[ -n "${GATE_CMD}" ]]; then
    echo "[train-loop] gate: ${GATE_CMD}"
    # shellcheck disable=SC2086
    BRAIN="${current_brain}" eval "${GATE_CMD}"
  else
    echo "[train-loop] no GATE_CMD set — skip frozen battery (recommended before ship)"
  fi
done

echo ""
echo "[train-loop] done. Latest brain: ${current_brain}"
echo "Ship only after a frozen held-out / companion prompt matrix passes."
