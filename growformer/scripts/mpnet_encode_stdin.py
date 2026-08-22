#!/usr/bin/env python3
"""Encode texts with sentence-transformers for Growformer semantic_router parity.

Reads one JSON object from stdin:
  {"model": "sentence-transformers/all-mpnet-base-v2", "texts": ["..."]}

Writes one JSON object to stdout:
  {"embeddings": [[...], ...]}   # L2-normalized (normalize_embeddings=True)

Must match spacekit-projects/coding/python/scripts/routing_lib.MpnetEncoder.encode.
"""
from __future__ import annotations

import json
import sys


def main() -> None:
    raw = sys.stdin.read()
    try:
        req = json.loads(raw)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"invalid JSON: {e}"}), file=sys.stderr)
        sys.exit(1)

    model_name = (req.get("model") or "sentence-transformers/all-mpnet-base-v2").strip()
    texts = req.get("texts")
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        print(json.dumps({"error": "texts must be a list of strings"}), file=sys.stderr)
        sys.exit(1)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            json.dumps({"error": "sentence-transformers required (pip install sentence-transformers)"}),
            file=sys.stderr,
        )
        sys.exit(1)

    model = SentenceTransformer(model_name)
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embeddings = [list(map(float, row)) for row in vectors]
    sys.stdout.write(json.dumps({"embeddings": embeddings}, separators=(",", ":")))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
