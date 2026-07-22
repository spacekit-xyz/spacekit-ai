#!/usr/bin/env python3
"""Step 4 — human validation pass that turns generated survivors into bench rows.

The firewall's load-bearing honesty step (DATA_GENERATION_SPEC §1 F3): the LLM
only *proposes*; a human *certifies* concept-preservation, and certification is
what admits a phrase to the bench. This tool orchestrates the BLINDNESS,
ADJUDICATION, and PROMOTION so the human judgement is unbiased and auditable.

Design — BLIND FORCED-CHOICE RECONSTRUCTION (not "does this mean X?"):
  A validator sees ONLY the phrase and the closed menu of intents. They pick the
  single best intent. A "does this mean X?" yes/no leaks the answer and invites
  acquiescence bias; forced-choice is a real classification, so agreement with
  the survivor's hidden source_intent is strong evidence a stranger recovers the
  meaning from surface alone. Disagreement that *coheres* on another intent is a
  salvage (relabel), not just a reject.

Admission rule (per item):
  - >=2 validators independently choose the SAME intent, AND
  - that agreed intent's phrase still passes the wbc gate vs the real seen-corpus, AND
  - majority naturalness != "malformed".
  -> admit under the agreed intent (source_intent OR a salvaged relabel).
  Everything else is rejected (with reason) — never admitted to hit a count.

Honeypots: the worklist secretly mixes in known-true bench phrases. A validator
who misses too many honeypots is unreliable and their whole sheet is discarded
BEFORE adjudication (so a rubber-stamper can't pad the bench).

Subcommands:
  build       -> blind worklist + per-validator answer templates + secret key
  adjudicate  -> score honeypots, compute kappa, write admitted/rejected + report
  promote     -> re-gate admitted vs seen-corpus, append to eval.jsonl (authored)

Blindness contract (enforce operationally): validators must not see _key.json,
each other's answer sheets, the generator_id, or any router output.
"""
import argparse
import glob
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import author_disjoint as ad  # noqa: E402

GEN_DIR = Path("data/generated")
VAL_DIR = GEN_DIR / "validation"
EVAL = Path("data/authored_disjoint_eval/eval.jsonl")
SEEN = "data/authored_disjoint_eval/seen_corpus.jsonl"
CANON = GEN_DIR / "intent_canonicals.json"
KEY = VAL_DIR / "_key.json"            # SECRET — never show validators
ITEMS = VAL_DIR / "worklist_items.jsonl"
MENU = VAL_DIR / "intent_menu.json"
ADMITTED = VAL_DIR / "admitted.jsonl"
REJECTED = VAL_DIR / "rejected.jsonl"
REPORT = VAL_DIR / "report.json"
HONEYPOT_MISS_MAX = 0.34   # discard a sheet that misses >1/3 of honeypots


def item_id(text, salt):
    return hashlib.sha1(f"{salt}|{text}".encode()).hexdigest()[:10]


def load_pending_survivors():
    out = []
    for fp in sorted(glob.glob(str(GEN_DIR / "*" / "survivors.jsonl"))):
        for line in Path(fp).read_text().splitlines():
            if not line.strip():
                continue
            v = json.loads(line)
            if not v.get("validated_by"):  # [] / missing == pending
                out.append(v)
    return out


def cmd_build(args):
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    survivors = load_pending_survivors()
    if not survivors:
        raise SystemExit("no pending survivors (validated_by==[]) found.")
    canon = json.loads(CANON.read_text()) if CANON.exists() else {}

    # Honeypots: real bench phrases with known-true intent, mixed in blind.
    bench = ad.load_pairs(str(EVAL))
    rng = random.Random(args.seed)
    n_honey = max(args.honeypots, 0)
    honey = rng.sample(bench, min(n_honey, len(bench)))

    key, items = {}, []
    for v in survivors:
        iid = item_id(v["text"], "s")
        key[iid] = {"text": v["text"], "truth": v["semantic_intent"],
                    "kind": "candidate", "generator_id": v.get("generator_id")}
        items.append({"item_id": iid, "text": v["text"]})
    for text, intent in honey:
        iid = item_id(text, "h")
        key[iid] = {"text": text, "truth": intent, "kind": "honeypot"}
        items.append({"item_id": iid, "text": text})
    rng.shuffle(items)

    # Intent menu = the closed forced-choice set (with canonical descriptions).
    intents = sorted({k["truth"] for k in key.values()} |
                     set(canon.keys()))
    menu = {it: (canon.get(it, {}) or {}).get("canonical", "") for it in intents}

    ITEMS.write_text("\n".join(json.dumps(x) for x in items) + "\n")
    MENU.write_text(json.dumps(menu, indent=2) + "\n")
    KEY.write_text(json.dumps(key, indent=2) + "\n")
    for name in args.validators:
        tmpl = VAL_DIR / f"answers_{name}.jsonl"
        if tmpl.exists():
            print(f"  (keep existing {tmpl.name})")
            continue
        tmpl.write_text("\n".join(json.dumps({
            "item_id": x["item_id"], "text": x["text"],
            "chosen_intent": "", "naturalness": "", "confidence": ""
        }) for x in items) + "\n")
    print(f"built worklist: {len(survivors)} candidates + {len(honey)} honeypots "
          f"= {len(items)} blind items")
    print(f"  menu: {len(menu)} intents -> {MENU}")
    print(f"  validators: {', '.join(args.validators)} (fill chosen_intent + "
          f"naturalness[natural|stilted|malformed])")
    print(f"  SECRET key -> {KEY} (do NOT show validators)")


def cohen_kappa(a, b):
    """Cohen's kappa over paired categorical labels."""
    items = [k for k in a if k in b]
    if not items:
        return None
    labels = sorted({a[k] for k in items} | {b[k] for k in items})
    n = len(items)
    po = sum(1 for k in items if a[k] == b[k]) / n
    ca, cb = Counter(a[k] for k in items), Counter(b[k] for k in items)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    return None if pe == 1 else (po - pe) / (1 - pe)


def cmd_adjudicate(args):
    key = json.loads(KEY.read_text())
    sheets = {}
    for fp in sorted(glob.glob(str(VAL_DIR / "answers_*.jsonl"))):
        name = Path(fp).stem.replace("answers_", "")
        ans = {}
        for line in Path(fp).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("chosen_intent"):
                ans[r["item_id"]] = r
        if ans:
            sheets[name] = ans
    if len(sheets) < 2:
        raise SystemExit(f"need >=2 completed answer sheets; found {len(sheets)}.")

    # 1) Honeypot reliability gate — discard unreliable sheets first.
    honey = [iid for iid, k in key.items() if k["kind"] == "honeypot"]
    reliable = {}
    rel_report = {}
    for name, ans in sheets.items():
        seen = [iid for iid in honey if iid in ans]
        miss = [iid for iid in seen
                if ans[iid]["chosen_intent"] != key[iid]["truth"]]
        miss_rate = (len(miss) / len(seen)) if seen else 1.0
        rel_report[name] = {"honeypots_seen": len(seen),
                            "missed": len(miss), "miss_rate": round(miss_rate, 3)}
        if seen and miss_rate <= HONEYPOT_MISS_MAX:
            reliable[name] = ans
        else:
            print(f"  DISCARD sheet '{name}': honeypot miss_rate "
                  f"{miss_rate:.0%} > {HONEYPOT_MISS_MAX:.0%}")
    if len(reliable) < 2:
        raise SystemExit("fewer than 2 reliable sheets after honeypot gate.")

    # 2) Pairwise kappa (report inter-annotator agreement).
    names = sorted(reliable)
    kappas = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            kappas[f"{names[i]}~{names[j]}"] = cohen_kappa(
                {k: v["chosen_intent"] for k, v in reliable[names[i]].items()},
                {k: v["chosen_intent"] for k, v in reliable[names[j]].items()})

    # 3) Adjudicate candidates.
    cf, gf = ad.build_index(ad.load_pairs(SEEN))
    admitted, rejected = [], []
    for iid, k in key.items():
        if k["kind"] != "candidate":
            continue
        votes = [reliable[n][iid] for n in names if iid in reliable[n]]
        if len(votes) < 2:
            rejected.append({**k, "item_id": iid, "reason": "under-annotated"})
            continue
        choice_counts = Counter(v["chosen_intent"] for v in votes)
        agreed, n_agree = choice_counts.most_common(1)[0]
        nat = Counter(v.get("naturalness", "") for v in votes)
        if n_agree < 2:
            rejected.append({**k, "item_id": iid, "reason": "no_majority",
                             "votes": dict(choice_counts)})
            continue
        if nat.get("malformed", 0) > len(votes) / 2:
            rejected.append({**k, "item_id": iid, "reason": "malformed"})
            continue
        # Re-gate under the AGREED intent (salvage relabels are allowed but must
        # still be wbc-disjoint vs the real seen-corpus under the new label).
        d, s, ov = ad.check_phrase(k["text"], agreed, cf, gf)
        if not (d and s):
            rejected.append({**k, "item_id": iid, "reason": "gate_fail_under_agreed",
                             "agreed": agreed, "leaks": sorted(ov)[:6]})
            continue
        admitted.append({
            "text": k["text"], "semantic_intent": agreed,
            "provenance": "authored", "source_intent": k["truth"],
            "relabeled": agreed != k["truth"],
            "generator_id": k.get("generator_id"),
            "validated_by": names, "n_agree": n_agree,
        })

    ADMITTED.write_text("\n".join(json.dumps(x) for x in admitted) + "\n")
    REJECTED.write_text("\n".join(json.dumps(x) for x in rejected) + "\n")
    report = {
        "reliable_sheets": names, "reliability": rel_report,
        "pairwise_cohen_kappa": kappas,
        "candidates": sum(1 for k in key.values() if k["kind"] == "candidate"),
        "admitted": len(admitted),
        "relabeled": sum(1 for a in admitted if a["relabeled"]),
        "rejected": len(rejected),
        "reject_reasons": dict(Counter(r["reason"] for r in rejected)),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"\nadmitted -> {ADMITTED}\nrejected -> {REJECTED}\nreport -> {REPORT}")
    print("Next: review admitted.jsonl, then `promote`.")


def cmd_promote(args):
    if not ADMITTED.exists():
        raise SystemExit("run `adjudicate` first.")
    admitted = [json.loads(l) for l in ADMITTED.read_text().splitlines() if l.strip()]
    if not admitted:
        raise SystemExit("nothing admitted.")
    cf, gf = ad.build_index(ad.load_pairs(SEEN))
    existing = {(t.lower(), it) for t, it in ad.load_pairs(str(EVAL))}
    rows, skipped = [], []
    for a in admitted:
        key = (a["text"].lower(), a["semantic_intent"])
        d, s, ov = ad.check_phrase(a["text"], a["semantic_intent"], cf, gf)
        if not (d and s) or key in existing:
            skipped.append(a["text"])
            continue
        rows.append({"text": a["text"], "semantic_intent": a["semantic_intent"],
                     "provenance": "authored", "validated_by": a["validated_by"],
                     "source_intent": a.get("source_intent"),
                     "generator_id": a.get("generator_id")})
    if args.dry_run:
        print(f"[dry-run] would append {len(rows)} rows to {EVAL} "
              f"(skip {len(skipped)})")
    else:
        with open(EVAL, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        # mark promoted survivors as validated in their source files
        _mark_validated({(r["text"].lower(), r["semantic_intent"]) for r in rows},
                        admitted)
        print(f"appended {len(rows)} validated rows to {EVAL} (skip {len(skipped)})")

    # Report the disjoint bin under the real denominator after promotion.
    bin_n = sum(1 for t, it in ad.load_pairs(str(EVAL))
                if all(ad.check_phrase(t, it, cf, gf)[:2]))
    print(f"disjoint bin now: {bin_n} (resolution gate: n>=47)")
    print("Then re-run Step 5 validity gates (--real-encoder-experiment): "
          "CATA-collapse + n>=47 + wbc.")


def _mark_validated(promoted_keys, admitted):
    by_key = {(a["text"].lower(), a["semantic_intent"]): a for a in admitted}
    for fp in glob.glob(str(GEN_DIR / "*" / "survivors.jsonl")):
        lines = Path(fp).read_text().splitlines()
        changed = False
        out = []
        for line in lines:
            if not line.strip():
                continue
            v = json.loads(line)
            k = (v["text"].lower(), v.get("semantic_intent"))
            # match against source_intent too (relabels)
            for pk, a in by_key.items():
                if v["text"].lower() == pk[0] and not v.get("validated_by"):
                    v["validated_by"] = a["validated_by"]
                    v["provenance"] = "authored"
                    v["semantic_intent"] = a["semantic_intent"]
                    changed = True
                    break
            out.append(json.dumps(v))
        if changed:
            Path(fp).write_text("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="make blind worklist + answer templates")
    b.add_argument("--validators", nargs="+", default=["v1", "v2"])
    b.add_argument("--honeypots", type=int, default=8)
    b.add_argument("--seed", type=int, default=20260625)
    b.set_defaults(func=cmd_build)
    a = sub.add_parser("adjudicate", help="score + admit/reject from answer sheets")
    a.set_defaults(func=cmd_adjudicate)
    p = sub.add_parser("promote", help="append admitted rows to eval.jsonl")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_promote)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
