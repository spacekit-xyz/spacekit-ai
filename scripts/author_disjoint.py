#!/usr/bin/env python3
"""Fast authoring/validation helper for wbc-disjoint eval phrases.

Replicates grounding_loop::for_each_feature exactly so candidate phrases can be
checked against their own-concept training (must be disjoint at w/b/c) and the
global training (must be seen-elsewhere) without recompiling Rust.

Usage:
  python scripts/author_disjoint.py <train.jsonl> check <eval.jsonl>
  python scripts/author_disjoint.py <train.jsonl> probe <concept> "candidate phrase"
  python scripts/author_disjoint.py <train.jsonl> forbidden <concept>
"""
import json
import sys
from pathlib import Path


def features(text):
    """Exact replica of for_each_feature: returns set of w:/b:/c: keys."""
    words = []
    cur = []
    for ch in text.lower():
        if ch.isalnum() and ch.isascii():
            cur.append(ch)
        else:
            if cur:
                words.append("".join(cur))
                cur = []
    if cur:
        words.append("".join(cur))

    feats = set()
    for i, w in enumerate(words):
        feats.add(f"w:{w}")
        if i + 1 < len(words):
            feats.add(f"b:{w}_{words[i+1]}")
        padded = f"^{w}$"
        if len(padded) >= 3:
            for j in range(len(padded) - 2):
                feats.add(f"c:{padded[j]}{padded[j+1]}{padded[j+2]}")
    return feats


def load_pairs(path):
    pairs = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        v = json.loads(line)
        t = v["text"].strip()
        intent = v.get("semantic_intent", v.get("intent", "")).strip()
        if t and intent:
            pairs.append((t, intent))
    return pairs


def build_index(train):
    concept_feats = {}
    global_feats = set()
    for text, intent in train:
        f = features(text)
        concept_feats.setdefault(intent, set()).update(f)
        global_feats.update(f)
    return concept_feats, global_feats


def check_phrase(phrase, intent, concept_feats, global_feats):
    f = features(phrase)
    own = concept_feats.get(intent, set())
    overlap = f & own
    disjoint = len(overlap) == 0
    seen_elsewhere = any((k in global_feats) and (k not in own) for k in f)
    return disjoint, seen_elsewhere, overlap


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    train_path = sys.argv[1]
    cmd = sys.argv[2]
    train = load_pairs(train_path)
    concept_feats, global_feats = build_index(train)

    if cmd == "forbidden":
        concept = sys.argv[3]
        own = concept_feats.get(concept, set())
        tris = sorted(k for k in own if k.startswith("c:"))
        words = sorted(k for k in own if k.startswith("w:"))
        print(f"[{concept}] {len(words)} words, {len(tris)} char-trigrams in own-concept training")
        print("WORDS:", " ".join(w[2:] for w in words))
        print("TRIGRAMS:", " ".join(t[2:] for t in tris))
        return

    if cmd == "probe":
        concept = sys.argv[3]
        phrase = sys.argv[4]
        disjoint, seen, overlap = check_phrase(phrase, concept, concept_feats, global_feats)
        status = "DISJOINT" if disjoint else "LEAKS"
        tag = " (seen-elsewhere)" if (disjoint and seen) else (" (novel)" if disjoint else "")
        print(f"  [{status}{tag}] \"{phrase}\" -> {concept}")
        if overlap:
            print(f"    leaks ({len(overlap)}): {', '.join(sorted(overlap))}")
        return

    if cmd == "check":
        eval_path = sys.argv[3]
        ev = load_pairs(eval_path)
        n_pass = n_fail = n_seen = 0
        for phrase, intent in ev:
            disjoint, seen, overlap = check_phrase(phrase, intent, concept_feats, global_feats)
            if disjoint:
                n_pass += 1
                if seen:
                    n_seen += 1
                tag = "seen-elsewhere" if seen else "novel"
                print(f"  PASS [{intent}] \"{phrase}\" ({tag})")
            else:
                n_fail += 1
                leak = ", ".join(sorted(overlap)[:6])
                print(f"  FAIL [{intent}] \"{phrase}\" leaks: {leak}")
        print(f"\nSummary: {n_pass} pass / {n_fail} fail / {n_seen} seen-elsewhere (need >=8)")
        return

    print(f"unknown command: {cmd}")
    sys.exit(1)


if __name__ == "__main__":
    main()
