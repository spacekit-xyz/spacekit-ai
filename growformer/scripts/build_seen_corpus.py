#!/usr/bin/env python3
"""Build the disjoint-bench SEEN/forbidden denominator from the REAL Luna
training corpus (recursive-learning loop: the model's own training set is the
reference against which we prove new eval phrases are out-of-distribution).

Seen-set = authored stub (data/authored_disjoint_eval/train.jsonl)
           UNION the real Luna labeled corpus (every luna_*.jsonl row that
           carries both `text` and `semantic_intent`; response-only fragments
           with no intent are correctly excluded).

Output:
  data/authored_disjoint_eval/seen_corpus.jsonl          (deduped text+intent)
  data/authored_disjoint_eval/seen_corpus.provenance.json (lineage + counts)

Provenance note (F1/F2): these rows are the SEEN side only. They are the
disjointness reference, NEVER eval candidates — using a Luna training phrase as
an eval phrase would be training-on-the-test (the exact inversion of the bench).
"""
import glob
import json
import os
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import author_disjoint as ad  # noqa: E402

LUNA_DATA_DIR = os.environ.get(
    "LUNA_DATA_DIR",
    "/Users/astor/Projects/2026/spacekit/spacekit-projects/companions/luna/data")
STUB = "data/authored_disjoint_eval/train.jsonl"
OUT = "data/authored_disjoint_eval/seen_corpus.jsonl"
PROV = "data/authored_disjoint_eval/seen_corpus.provenance.json"


def main():
    luna_files = sorted(glob.glob(f"{LUNA_DATA_DIR}/luna_*.jsonl"))
    seen = {}  # (text_lower, intent) -> {text, intent, source_file}
    per_source = {}

    def ingest(path, tag):
        n = 0
        for text, intent in ad.load_pairs(path):
            key = (text.lower(), intent)
            if key not in seen:
                seen[key] = {"text": text, "semantic_intent": intent,
                             "seen_source": tag}
                n += 1
        per_source[tag] = per_source.get(tag, 0) + n

    ingest(STUB, "authored_stub")
    for fp in luna_files:
        ingest(fp, f"luna/{Path(fp).name}")

    rows = list(seen.values())
    Path(OUT).write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    # Provenance + per-intent counts.
    by_intent = {}
    for r in rows:
        by_intent[r["semantic_intent"]] = by_intent.get(r["semantic_intent"], 0) + 1
    prov = {
        "description": "SEEN/forbidden denominator for the disjoint capability "
                       "bench. authored stub UNION real Luna training corpus.",
        "luna_data_dir": LUNA_DATA_DIR,
        "luna_files": [Path(f).name for f in luna_files],
        "total_rows": len(rows),
        "rows_added_per_source": per_source,
        "rows_per_intent": dict(sorted(by_intent.items(),
                                        key=lambda kv: -kv[1])),
        "discipline": "SEEN side only — never use these as eval candidates "
                      "(would be training-on-the-test). F1/F2.",
    }
    Path(PROV).write_text(json.dumps(prov, indent=2) + "\n")

    print(f"seen_corpus: {len(rows)} unique rows -> {OUT}")
    for src, n in per_source.items():
        print(f"   +{n:>4} from {src}")
    print(f"   {len(by_intent)} intents covered")


if __name__ == "__main__":
    main()
