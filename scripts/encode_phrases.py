#!/usr/bin/env python3
"""Offline phrase encoder for the real-encoder experiment.

Reads a phrases JSON (exported by --export-phrases) and writes an embeddings
JSON consumable by --certify-encoder byo:<path>.

Usage:
    python scripts/encode_phrases.py phrases_to_encode.json \
        --model all-mpnet-base-v2 \
        --output embeddings_all-mpnet-base-v2.json
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Encode phrases with sentence-transformers")
    parser.add_argument("phrases_json", type=Path, help="JSON file with list of phrases to encode")
    parser.add_argument("--model", default="all-mpnet-base-v2", help="sentence-transformers model name")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path (default: embeddings_<model>.json)")
    args = parser.parse_args()

    if not args.phrases_json.exists():
        print(f"error: {args.phrases_json} not found", file=sys.stderr)
        sys.exit(1)

    phrases = json.loads(args.phrases_json.read_text())
    if not isinstance(phrases, list) or not all(isinstance(p, str) for p in phrases):
        print("error: phrases JSON must be a list of strings", file=sys.stderr)
        sys.exit(1)

    print(f"loading model '{args.model}'...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("error: pip install sentence-transformers", file=sys.stderr)
        sys.exit(1)

    model = SentenceTransformer(args.model)
    print(f"encoding {len(phrases)} phrases...")
    embeddings = model.encode(phrases, show_progress_bar=True, convert_to_numpy=True)

    result = {}
    for phrase, vec in zip(phrases, embeddings):
        result[phrase] = [round(float(v), 6) for v in vec]

    output_path = args.output or Path(f"embeddings_{args.model}.json")
    output_path.write_text(json.dumps(result, ensure_ascii=False))
    dim = len(next(iter(result.values())))
    print(f"wrote {len(result)} embeddings ({dim}-dim) to {output_path}")


if __name__ == "__main__":
    main()
