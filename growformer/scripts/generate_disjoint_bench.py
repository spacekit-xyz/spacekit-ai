#!/usr/bin/env python3
"""Firewalled LLM-paraphrase generation loop for the capability disjoint bench.

Implements docs/DATA_GENERATION_SPEC.md §2. Generates surface-disjoint,
concept-preserving paraphrase *candidates*, filters them through the exact
mechanical gate (author_disjoint.for_each_feature replica), expands the
forbidden list from real leaked features, and loops until a yield target.

DISCIPLINE (do not remove):
  - This produces CAPABILITY candidates only. Survivors are NOT certified data;
    they are inputs to Step 4 (>=2 human blind validators). See spec §0/§1 F3.
  - The LLM only PROPOSES. Humans certify concept-preservation. This script
    never assigns the final label.
  - Lineage is stamped (provenance=synthetic, source_intent, generator_id) so a
    survivor can never be silently used as both train and test (F1/F2).

Gate is run IN-PROCESS via author_disjoint (not by parsing CLI stdout), which
guarantees bit-parity with the Rust `for_each_feature` the certifier uses.

LLM wiring: implement `call_llm(prompt) -> list[str]`, or run in --offline mode
where each round's candidates are read from round{N}_candidates.jsonl that you
paste in by hand (lets you exercise the gate without API access).

Usage:
  python3 scripts/generate_disjoint_bench.py <concept> --canonical "<sentence>" \
      [--train data/authored_disjoint_eval/train.jsonl] [--target 40] \
      [--rounds 5] [--generator-id gpt-x] [--offline]
"""
import argparse
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import author_disjoint as ad  # exact for_each_feature replica  # noqa: E402

# Minimal stopword set so the stem-uniqueness guard targets *content* words.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "to",
    "of", "in", "on", "at", "for", "and", "or", "but", "so", "it", "its", "her",
    "him", "his", "she", "he", "they", "them", "that", "this", "these", "those",
    "with", "as", "by", "from", "just", "you", "your", "i", "im", "me", "my",
    "we", "us", "out", "up", "down", "when", "if", "then", "now", "got", "get",
}


def stem(word: str) -> str:
    """Crude suffix-stripping stemmer: scare/scared/scary -> scar*. Pilot guard,
    not linguistics. Collapses obvious inflections so two survivors don't share
    a content-word lemma (the surface-form-collapse the gate is sensitive to)."""
    w = word.lower()
    for suf in ("ingly", "edly", "ing", "ied", "ies", "ied", "ed", "es", "er",
                "est", "ly", "y", "s"):
        if len(w) - len(suf) >= 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def content_stems(text: str) -> set:
    out = set()
    for tok in ad.features(text):  # reuse the gate's exact tokenizer via w: feats
        if tok.startswith("w:"):
            word = tok[2:]
            if word not in STOPWORDS and len(word) >= 3:
                out.add(stem(word))
    return out


def build_prompt(canonical: str, forbidden_words: set, forbidden_tris: set) -> str:
    """Refined Step-2 prompt. Concept LABEL never appears here. The pilot
    (DATA_GENERATION_SPEC §2 finding) showed the binding constraint is
    ultra-common function words / endings, and that an *aggressive* compliance
    prompt + terse style + per-phrase self-check moves yield 0% -> ~20% on a 3B
    model. Those proven levers are baked in here."""
    words = " ".join(sorted(forbidden_words))
    tris = " ".join(sorted(t for t in forbidden_tris))
    return f"""Generate many different natural phrases that all mean the SAME thing.

MEANING TO PRESERVE EXACTLY:
"{canonical}"

HARD RULE — write TERSE, telegraphic phrases (3-8 words), unusual vocabulary.
NEVER use any of these words: {words}
NEVER use any letter-sequence: {tris}
ALSO never use these (they almost always leak): the, to, a, is, of, it, your,
you, and avoid any word ending in -ing or -er.

Before generating:
1. Restate the meaning in your own words (1 sentence).
2. Restate the forbidden constraint in your own words (1 sentence).

Then produce 20 phrases. After writing each, silently verify it breaks no rule
and rewrite it if it does. Requirements:
- Preserve the meaning EXACTLY (no additions, omissions, or shifts).
- Maximize surface diversity (register, syntax, vocabulary).
- No two phrases may share a content-word stem (scare/scared/scary = one stem).
- No preamble, no restating the meaning sentence, no robotic phrasing.

OUTPUT: ONLY a JSON array: [{{"text": "..."}}, ...]
"""


import os
import re
import urllib.error
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("GROWFORMER_OLLAMA_MODEL", "llama3.2:latest")

# Generator backend: "ollama" (default) or "anthropic". The Anthropic key is read
# ONLY from the environment (never a file/arg) so it cannot be committed.
GENERATOR = os.environ.get("GROWFORMER_GENERATOR", "ollama").lower()
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("GROWFORMER_ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")


def extract_phrases(raw: str) -> list:
    """Robustly pull phrase strings out of an LLM response that may include
    restate-sentences, <think> traces, ```json fences, or a bare JSON array."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"```(?:json)?|```", "", raw)
    # Prefer a JSON array of {"text": ...} (or bare strings).
    start, end = raw.find("["), raw.rfind("]")
    if start != -1 and end > start:
        try:
            arr = json.loads(raw[start:end + 1])
            out = []
            for item in arr:
                if isinstance(item, dict) and "text" in item:
                    out.append(str(item["text"]).strip())
                elif isinstance(item, str):
                    out.append(item.strip())
            if out:
                return [p for p in out if p]
        except json.JSONDecodeError:
            pass
    # Fallback: any {"text": "..."} objects scattered in prose.
    objs = re.findall(r'\{"text"\s*:\s*"([^"]+)"\}', raw)
    if objs:
        return [o.strip() for o in objs if o.strip()]
    # Last resort: bullet/numbered lines.
    lines = []
    for ln in raw.splitlines():
        ln = ln.strip().lstrip("-*0123456789. ").strip().strip('"')
        if 2 <= len(ln.split()) <= 14:
            lines.append(ln)
    return lines


def _call_ollama(prompt: str) -> list:
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 1.0, "top_p": 0.95},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        resp = json.loads(r.read().decode())
    return extract_phrases(resp.get("response", ""))


def _call_anthropic(prompt: str) -> list:
    if not ANTHROPIC_API_KEY:
        raise SystemExit(
            "ANTHROPIC_API_KEY not set. Export it in your shell:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...\n"
            "Never put the key in a file or pass it as an argument.")
    body = json.dumps({
        "model": ANTHROPIC_MODEL,
        "max_tokens": 1500,
        "temperature": 1.0,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body,
        headers={
            "content-type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        })
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:600]
        raise SystemExit(f"Anthropic API {e.code}: {detail}")
    text = "".join(b.get("text", "") for b in resp.get("content", [])
                    if b.get("type") == "text")
    return extract_phrases(text)


def call_llm(prompt: str) -> list:
    """INTEGRATION POINT. Generator is selected by GROWFORMER_GENERATOR
    (ollama|anthropic). The generator is a DIFFERENT system from the encoder
    being certified (firewall: generator != certifier) and sees only the
    canonical sentence + forbidden list (never the concept's aliases/training).
    Returns a list of phrase strings."""
    if GENERATOR == "anthropic":
        return _call_anthropic(prompt)
    return _call_ollama(prompt)


def load_offline_round(round_file: Path) -> list:
    if not round_file.exists():
        print(f"  [offline] paste candidates into {round_file} and re-run.")
        return []
    out = []
    for line in round_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line)["text"])
        except (json.JSONDecodeError, KeyError):
            out.append(line)  # tolerate a plain-text line
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("concept")
    ap.add_argument("--canonical", default=None,
                    help="one clean human sentence; NOT the concept label. "
                         "If omitted, read from --canonical-file for this concept.")
    ap.add_argument("--canonical-file",
                    default="data/generated/intent_canonicals.json",
                    help="JSON map {intent: {canonical: ...}} from "
                         "derive_canonicals.py (corpus-grounded).")
    ap.add_argument("--train", default="data/authored_disjoint_eval/seen_corpus.jsonl",
                    help="SEEN/forbidden denominator. Defaults to the real Luna "
                         "corpus UNION authored stub (build_seen_corpus.py). "
                         "Pass the bare stub only for legacy reproduction.")
    ap.add_argument("--target", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--generator-id", default=None)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    if args.canonical is None:
        cf_path = Path(args.canonical_file)
        if not cf_path.exists():
            ap.error(f"--canonical not given and {cf_path} missing "
                     f"(run scripts/derive_canonicals.py first).")
        cmap = json.loads(cf_path.read_text())
        entry = cmap.get(args.concept) or {}
        args.canonical = entry.get("canonical")
        if not args.canonical:
            ap.error(f"no canonical for '{args.concept}' in {cf_path}; "
                     f"pass --canonical or add one to derive_canonicals.py.")
    if args.generator_id is None:
        if args.offline:
            args.generator_id = "manual"
        elif GENERATOR == "anthropic":
            args.generator_id = f"anthropic/{ANTHROPIC_MODEL}"
        else:
            args.generator_id = f"ollama/{OLLAMA_MODEL}"

    train = ad.load_pairs(args.train)
    concept_feats, global_feats = ad.build_index(train)
    if args.concept not in concept_feats:
        print(f"WARNING: concept '{args.concept}' not in train; "
              f"own-concept forbidden list will be empty.")

    # Seed the forbidden list from the concept's own-concept training (Step 1).
    own = concept_feats.get(args.concept, set())
    forbidden_words = {k[2:] for k in own if k.startswith("w:")}
    forbidden_tris = {k[2:] for k in own if k.startswith("c:")}

    out_dir = Path(f"data/generated/{args.concept}")
    out_dir.mkdir(parents=True, exist_ok=True)

    survivors, seen_stems, log = [], set(), []

    for rnd in range(1, args.rounds + 1):
        print(f"\n=== Round {rnd} for {args.concept} "
              f"(have {len(survivors)}/{args.target}) ===")
        prompt = build_prompt(args.canonical, forbidden_words, forbidden_tris)
        (out_dir / f"round{rnd}_prompt.txt").write_text(prompt)

        round_file = out_dir / f"round{rnd}_candidates.jsonl"
        if args.offline:
            candidates = load_offline_round(round_file)
        else:
            candidates = call_llm(prompt)
            round_file.write_text(
                "\n".join(json.dumps({"text": c}) for c in candidates))

        kept = leaked = drift_dupes = 0
        for text in candidates:
            disjoint, seen, overlap = ad.check_phrase(
                text, args.concept, concept_feats, global_feats)
            if not (disjoint and seen):
                leaked += 1
                # Expand forbidden from the REAL leaked features (full, not the
                # CLI-truncated 6) so the next round steers away.
                for k in overlap:
                    if k.startswith("w:"):
                        forbidden_words.add(k[2:])
                    elif k.startswith("c:"):
                        forbidden_tris.add(k[2:])
                continue
            stems = content_stems(text)
            if stems & seen_stems:  # stem-uniqueness guard (surface-collapse)
                drift_dupes += 1
                continue
            seen_stems |= stems
            survivors.append(text)
            kept += 1

        log.append({"round": rnd, "n_candidates": len(candidates),
                    "kept": kept, "leaked": leaked, "stem_dupes": drift_dupes,
                    "forbidden_words": len(forbidden_words),
                    "cumulative_survivors": len(survivors)})
        print(f"  kept={kept} leaked={leaked} stem_dupes={drift_dupes} "
              f"forbidden_words={len(forbidden_words)}")
        if len(survivors) >= args.target:
            break

    # Survivors carry FULL lineage (F1) and are flagged unvalidated: they are NOT
    # bench data until >=2 humans pass them (Step 4). Provenance is 'synthetic'
    # until human validation promotes it to 'authored'.
    surv_path = out_dir / "survivors.jsonl"
    with open(surv_path, "w") as f:
        for s in survivors:
            f.write(json.dumps({
                "text": s,
                "semantic_intent": args.concept,
                "provenance": "synthetic",
                "source_intent": args.concept,
                "generator_id": args.generator_id,
                "validated_by": [],  # filled by Step 4 humans; [] = NOT yet bench
            }) + "\n")
    (out_dir / "generation_log.jsonl").write_text(
        "\n".join(json.dumps(x) for x in log))

    print(f"\nSurvivors: {len(survivors)} -> {surv_path}")
    print(f"Next: Step 4 — >=2 human blind validators confirm concept-preservation,")
    print(f"      then assemble into the bench and run --real-encoder-experiment")
    print(f"      (require CATA-collapse + resolvable n>=47 + wbc). Spec §2.4-2.5.")


if __name__ == "__main__":
    main()
