#!/usr/bin/env python3
"""Diagnostic: re-gate against the REAL Luna training corpus instead of the
363-row authored stub. Non-destructive — reports only."""
import glob
import json
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
import author_disjoint as ad

LUNA = "/Users/astor/Projects/2026/spacekit/spacekit-projects/companions/luna/data"
STUB = "data/authored_disjoint_eval/train.jsonl"
EVAL = "data/authored_disjoint_eval/eval.jsonl"

# Build the REAL seen-set from every luna_*.jsonl row that has text+intent.
real = []
for fp in sorted(glob.glob(f"{LUNA}/luna_*.jsonl")):
    try:
        real += ad.load_pairs(fp)
    except Exception as e:
        print(f"  (skip {Path(fp).name}: {e})")
stub = ad.load_pairs(STUB)
print(f"stub seen-set: {len(stub)} rows | REAL Luna seen-set: {len(real)} rows")

cf_stub, gf_stub = ad.build_index(stub)
cf_real, gf_real = ad.build_index(real)
print(f"global features  stub={len(gf_stub):>6}  real={len(gf_real):>6}")


def regate(label, pairs, cf, gf):
    p = f = s = 0
    fails = []
    for text, intent in pairs:
        d, seen, ov = ad.check_phrase(text, intent, cf, gf)
        if d:
            p += 1
            s += 1 if seen else 0
        else:
            f += 1
            fails.append((intent, text, sorted(ov)[:4]))
    print(f"\n{label}: {p} pass / {f} fail / {s} seen-elsewhere  (n={len(pairs)})")
    for intent, text, ov in fails:
        print(f"   FAIL [{intent}] \"{text}\"  leaks {ov}")
    return p, f, s


ev = ad.load_pairs(EVAL)
print("\n================ EXISTING BENCH eval.jsonl ================")
regate("vs STUB (original gate)", ev, cf_stub, gf_stub)
regate("vs REAL Luna corpus", ev, cf_real, gf_real)

# Re-gate the 26 agent survivors against the real corpus.
print("\n================ NEW AGENT SURVIVORS (8 concepts) ================")
sv = []
for fp in glob.glob("data/generated/*/survivors.jsonl"):
    for line in Path(fp).read_text().splitlines():
        if line.strip():
            v = json.loads(line)
            sv.append((v["text"], v["semantic_intent"]))
regate("vs STUB (what they passed)", sv, cf_stub, gf_stub)
regate("vs REAL Luna corpus", sv, cf_real, gf_real)
