#!/usr/bin/env python3
"""Promote a reviewed capture label_queue into immutable train/eval JSONL shards.

Human fills `semantic_intent` on `<capture_dir>/label_queue.jsonl` (from
`growformer-demos --audit-capture`). This tool validates intents against the
companion corpus, splits holdout, and emits LanguageSample-shaped rows for
fingerprint / overlay CL — without rewriting lattices itself.

Usage:
  python3 scripts/promote_label_queue.py \\
    --queue capture_artifacts/label_queue.jsonl \\
    --companion /path/to/companions/luna \\
    --out-dir capture_artifacts/approved_shard \\
    --reviewed-by alice

Outputs:
  capture_train_<stamp>.jsonl   — training shard (fingerprint / overlay)
  eval_capture_<stamp>.jsonl    — holdout (excluded from brain training scans)
  promotion_manifest.json       — hashes + reviewer metadata
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_valid_intents(companion: Path) -> set[str]:
    intents: set[str] = set()
    data = companion / "data"
    if not data.is_dir():
        return intents
    for path in sorted(data.glob("*.jsonl")):
        name = path.name
        if name.startswith("eval_") or name == "inference_guardrails.jsonl":
            continue
        try:
            for line in path.open():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                intent = (row.get("semantic_intent") or row.get("intent") or "").strip()
                if intent:
                    intents.add(intent)
        except OSError:
            continue
    return intents


def load_queue(path: Path) -> list[dict]:
    rows = []
    for i, line in enumerate(path.open(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path}:{i}: invalid JSON: {e}") from e
        if not isinstance(row, dict):
            raise SystemExit(f"{path}:{i}: expected object")
        rows.append(row)
    return rows


def to_language_sample(
    phrase: str,
    intent: str,
    *,
    expected_response: str | None,
    stamp: str,
    idx: int,
) -> dict:
    return {
        "task_id": f"capture_{stamp}_{idx:04d}",
        "text": phrase,
        "semantic_intent": intent,
        "domain": "pet",
        "action_target": "pet_chat",
        "policy_regime": "default",
        "language_channel": "english",
        "expected_response": expected_response,
        "expected_code": None,
        "provenance": {
            "kind": "PromotedCapture",
            "reviewed": True,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", required=True, type=Path, help="reviewed label_queue.jsonl")
    ap.add_argument("--companion", required=True, type=Path, help="companion root (has data/)")
    ap.add_argument("--out-dir", required=True, type=Path, help="approved shard directory")
    ap.add_argument("--reviewed-by", default="unknown")
    ap.add_argument("--holdout-ratio", type=float, default=0.2)
    ap.add_argument("--min-labeled", type=int, default=1)
    ap.add_argument(
        "--allow-unknown-intent",
        action="store_true",
        help="keep rows whose semantic_intent is not in companion train corpus",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--stamp",
        default=None,
        help="output stamp (default: UTC YYYYMMDD_HHMMSS)",
    )
    args = ap.parse_args()

    if not args.queue.is_file():
        raise SystemExit(f"queue not found: {args.queue}")
    if not args.companion.is_dir():
        raise SystemExit(f"companion not found: {args.companion}")

    valid = load_valid_intents(args.companion)
    if not valid and not args.allow_unknown_intent:
        raise SystemExit(
            f"no semantic_intent values found under {args.companion / 'data'}; "
            "pass --allow-unknown-intent to skip validation"
        )

    raw = load_queue(args.queue)
    labeled: list[tuple[str, str, str | None]] = []
    skipped_blank = 0
    skipped_unknown = 0
    for row in raw:
        phrase = (row.get("phrase") or row.get("text") or "").strip()
        intent = (row.get("semantic_intent") or "").strip()
        if not phrase:
            continue
        if not intent:
            skipped_blank += 1
            continue
        if valid and intent not in valid and not args.allow_unknown_intent:
            skipped_unknown += 1
            continue
        resp = row.get("expected_response")
        if isinstance(resp, str):
            resp = resp.strip() or None
        else:
            resp = None
        labeled.append((phrase, intent, resp))

    # Dedup by lowercase phrase, keep first labeled.
    seen: set[str] = set()
    uniq: list[tuple[str, str, str | None]] = []
    for phrase, intent, resp in labeled:
        key = phrase.casefold()
        if key in seen:
            continue
        seen.add(key)
        uniq.append((phrase, intent, resp))

    if len(uniq) < args.min_labeled:
        raise SystemExit(
            f"need ≥{args.min_labeled} labeled rows; got {len(uniq)} "
            f"(blank_intent={skipped_blank}, unknown_intent={skipped_unknown})"
        )

    # Stable holdout split by phrase hash (reproducible).
    holdout_n = int(round(len(uniq) * max(0.0, min(1.0, args.holdout_ratio))))
    scored = sorted(
        uniq,
        key=lambda t: hashlib.sha256(t[0].casefold().encode()).hexdigest(),
    )
    eval_rows = scored[:holdout_n]
    train_rows = scored[holdout_n:]
    if not train_rows and scored:
        # Keep at least one train row when tiny queues.
        train_rows = scored[-1:]
        eval_rows = scored[:-1]

    stamp = args.stamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", stamp)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / f"capture_train_{safe}.jsonl"
    eval_path = args.out_dir / f"eval_capture_{safe}.jsonl"
    manifest_path = args.out_dir / "promotion_manifest.json"

    def write_samples(path: Path, rows: list[tuple[str, str, str | None]]) -> None:
        with path.open("w") as f:
            for i, (phrase, intent, resp) in enumerate(rows, 1):
                f.write(json.dumps(to_language_sample(phrase, intent, expected_response=resp, stamp=safe, idx=i), ensure_ascii=False))
                f.write("\n")

    print(f"queue:            {args.queue}")
    print(f"companion:        {args.companion}")
    print(f"valid intents:    {len(valid)}")
    print(f"labeled unique:   {len(uniq)} (blank={skipped_blank}, unknown={skipped_unknown})")
    print(f"train / eval:     {len(train_rows)} / {len(eval_rows)}")
    print(f"out-dir:          {args.out_dir}")

    if args.dry_run:
        print("dry-run: no files written")
        return

    write_samples(train_path, train_rows)
    write_samples(eval_path, eval_rows)

    # Optional combined prompts file for certify_chat EXTRA_EVAL (plain prompts).
    prompts_path = args.out_dir / f"eval_capture_{safe}_prompts.txt"
    with prompts_path.open("w") as f:
        for phrase, _, _ in eval_rows:
            f.write(phrase.replace("\n", " ") + "\n")

    manifest = {
        "reviewed_by": args.reviewed_by,
        "stamp": safe,
        "queue": str(args.queue),
        "queue_sha256": sha256_file(args.queue),
        "companion": str(args.companion),
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "prompts_path": str(prompts_path),
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "holdout_ratio": args.holdout_ratio,
        "skipped_blank_intent": skipped_blank,
        "skipped_unknown_intent": skipped_unknown,
        "valid_intent_count": len(valid),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Fingerprint CL uses text+semantic_intent+action_target. "
            "Overlay/full train needs expected_response filled for new knowledge."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {train_path} ({len(train_rows)})")
    print(f"wrote {eval_path} ({len(eval_rows)})")
    print(f"wrote {prompts_path}")
    print(f"wrote {manifest_path}")
    print("Next: DATA_DIR=<out-dir> MODE=fingerprint GATE_CMD=… bash scripts/train_loop.sh")


if __name__ == "__main__":
    main()
