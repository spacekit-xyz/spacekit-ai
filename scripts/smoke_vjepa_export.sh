#!/usr/bin/env bash
# V-JEPA export smoke: mock teacher (CI) then optional HF path.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="${1:-data/wm/vjepa_export_v1.json}"
MODE="${VJEPA_MODE:-mock}"

echo "==> export_vjepa_features.py --mode ${MODE} --out ${OUT}"
if [[ "$MODE" == "mock" && -f "$OUT" && "${FORCE_EXPORT:-0}" != "1" ]]; then
  echo "    keeping existing pinned export (set FORCE_EXPORT=1 to regenerate)"
else
  python3 scripts/export_vjepa_features.py --mode "$MODE" --out "$OUT"
fi

echo "==> Phase 3u adapters on pinned export"
cargo run --release --bin growformer-demos -- --phase3u-vjepa-wm

echo "OK: V-JEPA ${MODE} smoke finished."
echo "For Meta weights: VJEPA_MODE=hf bash scripts/smoke_vjepa_export.sh"
echo "  (requires: recent transformers with vjepa2 support; torch; model facebook/vjepa2-vitl-fpc64-256)"
echo "Real-log path: --phase5g-vjepa-real-log (frozen-vision on dumped log);"
echo "  HF: python3 scripts/export_vjepa_features.py --mode hf --log data/wm/vjepa_real_log/visuomotor_log_v1.jsonl \\"
echo "      --out data/wm/vjepa_export_hf_real_log.json"
