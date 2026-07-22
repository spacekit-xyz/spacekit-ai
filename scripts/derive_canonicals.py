#!/usr/bin/env python3
"""Derive per-intent canonical meanings + anchor vocab from the REAL Luna corpus
(recursive-learning loop: the product's own training data defines the concept
meaning that generated paraphrases must preserve).

For each semantic_intent we emit:
  - canonical: one clean human sentence (the meaning the generator preserves).
    Authored as a *paraphrase* of the concept, NOT a verbatim training phrase
    (so it never leaks an eval surface) and NOT the bare concept label (the
    pilot showed labels cause semantic drift). Grounded in the corpus anchors +
    examples printed during authoring.
  - anchors: top graph_anchors for the intent, auto-extracted (frequency-ranked,
    persona constants stripped) — extra grounding context.
  - examples: a few short corpus phrases (provenance: SEEN; for human reference
    only — never copied into the eval).

Output: data/generated/intent_canonicals.json
"""
import collections
import glob
import json
import os
from pathlib import Path

LUNA_DATA_DIR = os.environ.get(
    "LUNA_DATA_DIR",
    "/Users/astor/Projects/2026/spacekit/spacekit-projects/companions/luna/data")
OUT = "data/generated/intent_canonicals.json"
PERSONA_CONST = {"cheerful_companion", "siamese", "cat"}

# Corpus-grounded canonical paraphrases (one clean sentence each). Authored from
# the anchors + examples; deliberately NOT verbatim training phrases.
CANONICAL = {
    "anxiety_trigger": "something sudden and loud is happening and the pet is on edge",
    "bedtime_routine": "it is time for the pet to settle down and go to sleep",
    "bonding_request": "the owner wants the pet to come close for affection",
    "compliment": "the owner tells the pet how lovely it looks",
    "cozy_distraction": "the owner asks the pet for a gentle comforting distraction",
    "curiosity_share": "the pet notices something intriguing and wants to investigate",
    "emotional_support": "the owner feels low and wants comfort from the pet",
    "gratitude_comfort": "the owner thanks the pet for helping them feel better",
    "gratitude_simple": "the owner gives the pet a brief simple thank you",
    "greeting_check_in": "the owner greets the pet and checks if it is awake",
    "identity_intro": "someone asks the pet who or what it is",
    "lore_qa": "the owner asks the pet about its favorites or backstory",
    "grooming_offer": "the owner offers to groom or clean the pet",
    "grounding_support": "the owner is panicking and needs help calming down",
    "household_activity": "the owner narrates a chore they are doing at home",
    "mealtime_check": "the owner asks whether the pet has already eaten",
    "mealtime_request": "the owner announces it is time for the pet to eat",
    "off_topic_deflection": "the owner asks an unrelated factual question the pet deflects",
    "open_ended_chat": "the owner just wants to chat about nothing in particular",
    "play_invitation": "the owner invites the pet to play",
    "playful_attention_seeking": "the pet is being loud and demanding attention",
    "reassurance_seeking": "the owner reassures the pet that the scary thing is over",
    "reunion_warm": "the owner has just come home and greets the pet warmly",
    "routine_transition": "the owner signals it is time for the next part of the routine",
    "separation_announcement": "the owner announces they are leaving for a while",
    "simple_acknowledgment": "the owner gives a brief thanks or sign off",
    "storytelling": "the owner asks the pet to tell a story",
    "status_check": "the owner asks how the pet is doing",
    "training_command": "the owner praises or corrects the pet's behavior",
    "treat_offer": "the owner offers the pet a small treat",
    "trigger_warning": "a loud frightening event is occurring near the pet",
    "visitor_alert": "the owner warns the pet that someone is arriving",
    "weather_commentary": "the owner remarks on the weather outside",
}


def main():
    anchors = collections.defaultdict(collections.Counter)
    examples = collections.defaultdict(list)
    for fp in sorted(glob.glob(f"{LUNA_DATA_DIR}/luna_*.jsonl")):
        for line in Path(fp).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                v = json.loads(line)
            except json.JSONDecodeError:
                continue
            it, t = v.get("semantic_intent"), v.get("text")
            if not it or not t:
                continue
            ga = (v.get("pet", {}) or {}).get("graph_anchors") or v.get("graph_anchors") or []
            for a in ga:
                if a not in PERSONA_CONST:
                    anchors[it][a] += 1
            if len(examples[it]) < 5 and 2 <= len(t.split()) <= 6:
                examples[it].append(t)

    out = {}
    intents = sorted(set(CANONICAL) | set(anchors))
    missing = []
    for it in intents:
        canon = CANONICAL.get(it)
        if canon is None:
            missing.append(it)
        out[it] = {
            "canonical": canon,
            "anchors": [a for a, _ in anchors[it].most_common(6)],
            "examples": examples[it][:3],
            "canonical_source": "authored_paraphrase" if canon else "MISSING",
        }
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {len(out)} intents -> {OUT}")
    if missing:
        print(f"  NOTE: no authored canonical for: {', '.join(missing)} "
              f"(anchors/examples still emitted; add a sentence before generating)")


if __name__ == "__main__":
    main()
