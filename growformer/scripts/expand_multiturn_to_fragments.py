#!/usr/bin/env python3
"""Expand multi-turn companion JSONL into fragment candidates for richer compose.

Growformer chat companions (Luna) compose from typed fragments (opener/body/closer).
Multi-turn rows already teach history-aware *lattices*; this script mines additional
*fragments* so basal_ganglia / reflective fields have more per-intent competition.

Usage:
  python3 scripts/expand_multiturn_to_fragments.py \\
    --in ../../spacekit/spacekit-projects/companions/luna/data/luna_multiturn_v1.jsonl \\
    --out ../../spacekit/spacekit-projects/companions/luna/data/luna_fragments_from_multiturn.jsonl \\
    --archetype cheerful_companion

Does not overwrite an existing library — append/merge manually after review.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def split_utterance(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    # Prefer sentence-ish splits; keep short exclamations.
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    out = []
    for p in parts:
        p = p.strip()
        if len(p) < 8:
            continue
        if len(p) > 180:
            p = p[:177] + "…"
        out.append(p)
    if not out and text:
        out.append(text[:180])
    return out


def role_for_index(i: int, n: int) -> str:
    if n == 1:
        return "body"
    if i == 0:
        return "opener"
    if i == n - 1:
        return "closer"
    return "body"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--archetype", default="cheerful_companion")
    ap.add_argument("--prefix", default="mtfrag")
    ap.add_argument("--min-history", type=int, default=1, help="Only rows with history length ≥ N")
    args = ap.parse_args()

    rows = []
    with args.inp.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    frags = []
    seen = set()
    n_src = 0
    for row in rows:
        pet = row.get("pet") or {}
        history = pet.get("history") or row.get("history") or []
        if len(history) < args.min_history and (pet.get("conversation_turn") or 1) <= 1:
            continue
        n_src += 1
        intent = row.get("semantic_intent") or "open_ended_chat"
        texts = []
        # Mine assistant/pet lines from history + expected_response.
        for turn in history:
            role = (turn.get("role") or "").lower()
            if role in ("pet", "agent", "assistant"):
                texts.extend(split_utterance(turn.get("text") or ""))
        if row.get("expected_response"):
            texts.extend(split_utterance(row["expected_response"]))

        for i, text in enumerate(texts):
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            role = role_for_index(i, len(texts))
            voice = "identity" if role == "opener" else "activity"
            fid = f"{args.prefix}_{len(frags):04d}"
            frags.append(
                {
                    "fragment_id": fid,
                    "voice": voice,
                    "text": text,
                    "role": role,
                    "intent_affinity": [intent],
                    "ocean_affinity": {},
                    "state_gate": {},
                    "archetype": args.archetype,
                    "weight": 0.85 if role == "body" else 0.95,
                    "source": "expand_multiturn_to_fragments",
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for frag in frags:
            f.write(json.dumps(frag, ensure_ascii=False) + "\n")

    print(f"source multi-turn-ish rows: {n_src}", file=sys.stderr)
    print(f"wrote {len(frags)} fragments → {args.out}", file=sys.stderr)
    print(
        "Review, then concatenate into your fragments_jsonl (dedupe by text) and retrain/redeploy.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
