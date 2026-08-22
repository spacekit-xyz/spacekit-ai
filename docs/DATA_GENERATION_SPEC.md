# Data Generation Spec — what you can generate, what you can't, and the runnable recipe

**Status:** runnable spec, 2026-06-25. Makes `GROUNDING_LOOP_SPEC.md` §20.1 (expanded authored bench)
and §20.2 (firewalled LLM-paraphrase protocol) **executable**. Read §0 before running anything.

---

## §0 The fork (read this first): two different "datas"

"The data we're waiting for" is ambiguous across exactly the line the whole arc holds. There are two
datasets, and **generation is permitted for one and forbidden for the other.**

| | What it is | Can you generate it? |
| --- | --- | --- |
| **Capability data** | An authored, surface-disjoint, concept-preserving **bench** that asks *can an encoder/router generalize on a fair test* | **Yes** — this spec. It un-gates research **now**. |
| **Deployment data** | The **`RealTraffic` disjoint bin** that asks *does the router work on real production traffic* | **No.** Generating it is the provenance trap (§2.5). Only diverse real users fill it. |

**Why deployment data cannot be generated (the line that does not move).** The provenance firewall (arc
§1.2) and §18.9 proved it physically: self-generated traffic is **0-disjoint by construction** — surface
diversity is exactly what data you produce lacks. An LLM is *better* than self-chatter (it has real
surface diversity), but it still **cannot stand in for real traffic**, because real traffic carries
structure no generator has (genuine noise, code-switch, multi-intent, distributional weirdness, the long
tail of how strangers actually talk). LLM paraphrases certify **capability**; they **never** certify
**deployment** (firewall F5). The 0/47 number moves only via the §21 shadow harness catching real users.

**So this spec generates capability data.** That is not a consolation prize: a thin n=47 bench is the
current bottleneck on *every* capability experiment — the graph-router edge-ablation (§2.6), encoder
comparisons, the cone router, distillation. Generating a larger, harder, validated capability bench
un-gates all of that **today**, off the data clock. It is the legitimate, valuable, runnable move.

---

## §1 The firewall (why generated data is honest here, and where it would stop being)

Restating `GROUNDING_LOOP_SPEC.md` §20.2 F1–F5 with the one reconciliation that matters:

- **F1 — lineage tracked.** Every generated phrase carries `provenance: synthetic`, `source_intent`,
  `generator_id` (which model produced it). Never silently relabeled `real_traffic` or `authored`.
- **F2 — test XOR train.** A generated phrase used for contrastive *training* is barred from *every*
  certify/eval set, and vice-versa. Enforced by lineage, not intention.
- **F3 + F5 reconciled — the certify surface is human-validated, not raw-LLM.** F3 says the certification
  surface is never raw LLM output; F5 says LLM paraphrases may expand the capability bench. Both hold
  because **the LLM only *proposes*; a human *validates* concept-preservation (§2.4), and validation is
  what admits a phrase to the bench.** A validated phrase is `authored` (human-vetted); the LLM made
  *proposing* cheap, not *certifying* automatic. The certify surface stays human-authored — generation
  cheapens the authoring, never replaces the human ground-truth step.
- **F4 — disjointness-verified.** Every phrase passes `author_disjoint.py` / `--verify-disjoint-eval`
  at `wbc` before use. In-lexicon paraphrases are discarded (they are the trap again).
- **F5 — never certify deployment.** Capability bench + contrastive training only. Deployment = §21.

**The cost truth (so the recipe isn't oversold):** generation removes the *authoring* cost (you don't
hand-write paraphrases) but **not** the *validation* cost — ≥2 humans still confirm each surviving
candidate means its intent, blind to any router. That validation is the load-bearing honesty step and is
bounded (~hundreds of candidates → tens of bench phrases), but it is real. Generation turns "author 200
disjoint phrases by hand" into "validate 200 machine-proposed candidates down to the survivors," which is
faster, not free.

---

## §1.5 The seen-index is the real product corpus (recursive grounding)

The gate's disjointness claim is only as honest as its **denominator** — "disjoint from *what the model
was trained on*." Until 2026-06-25 that denominator was a 363-row authored stub. It is now the **real Luna
training corpus**, which is the recursive-learning move: *the product's own training set becomes the
reference against which new eval phrases are proven out-of-distribution.* Three roles, firewall-assigned:

- **Luna training `*.jsonl` → the SEEN/forbidden denominator (legitimate).** `scripts/build_seen_corpus.py`
  unions the authored stub with every `luna_*.jsonl` row that carries `text`+`semantic_intent` (response-only
  fragments are correctly excluded) → `data/authored_disjoint_eval/seen_corpus.jsonl` (+ `.provenance.json`).
  This is now the default `--train` for the generator. **Never an eval candidate** — a Luna training phrase
  used as an eval phrase is training-on-the-test, the exact inversion of the bench.
- **`inference_pets.toml` + per-row `graph_anchors` → canonical meanings (legitimate).**
  `scripts/derive_canonicals.py` emits `data/generated/intent_canonicals.json` — one corpus-grounded
  canonical *paraphrase* per intent (not the label, not a verbatim phrase) plus frequency-ranked anchors.
  `generate_disjoint_bench.py --canonical-file` reads it, so batch generation is grounded in real intents.
- **`capture_artifacts/traffic_*.jsonl` → the DEPLOYMENT bin (separate firewall).** `RealTraffic`,
  unlabeled. Harvested via `GROUNDING_LOOP_SPEC.md` §21, never poured into this capability bench (F1/F5).

**What adopting the real corpus actually changed (measured).** The stub turned out to have been curated
*from* Luna, so most files added 0 rows; the real upgrade is **+75 newer rows** (comfort arcs, cheer/Spain
lore, expansion_v5, multiturn, ood) and **+7 intents** the stub predated (438 rows, 33 intents; feature
space +13%). The honest denominator earned its keep immediately: it flagged one eval phrase,
`"yap endlessly"` (open_ended_chat, leaks `c:^en`), that was only "disjoint" against the undercounted
stub. It was removed, so the disjoint bin is **46** under the real denominator (was a nominal 47) — the
one-phrase gap is filled by **human-validated survivors**, not by hand-authoring a replacement to hit the
number (that would game the threshold). Re-gate anytime with `scripts/regate_against_luna.py`.

---

## §2 The runnable recipe (per concept, looped)

All commands are real and verified against `data/authored_disjoint_eval/`. Train/eval files are JSONL:
`{"text": "...", "semantic_intent": "<concept>", "provenance": "..."}`.

### Step 1 — extract the forbidden-token list (what the paraphrase must avoid)

```bash
python3 scripts/author_disjoint.py data/authored_disjoint_eval/train.jsonl forbidden <concept>
```

Real output for `anxiety_trigger` (the generator's "avoid these" list):

```
[anxiety_trigger] 23 words, 90 char-trigrams in own-concept training
WORDS: a about are construction is it jumpy just loud noise okay outside s seem start the there to today vacuum weird wind you
TRIGRAMS: ^a$ ^ab ^ar ^co ^is ^it ... acu are art ... vac wei win you
```

**Realism check (this is why it's hard):** the candidate `"that rumbling machine is scaring her"` **LEAKS**
— on `w:is`, `c:^th`, `c:her`. Function words and common trigrams leak at `wbc`. Disjointness is strict;
expect a **low yield per proposal**, which is the honest reason disjoint data is scarce.

### Step 2 — generate candidates with an external LLM (blind to aliases)

Prompt an external model (the `generator_id`) — **not** the encoder being certified, and **not** shown
the concept's training phrases or node aliases (it must paraphrase *meaning*, not copy *surface*). Three
gate-aware hardenings materially raise yield (each pre-registered as a hypothesis to **measure**, §3, not
an assumed multiplier):

1. **Canonical anchor, not the concept label.** Feed a single clean human sentence ("The animal reacts
   with fear when something sudden happens"), never the noun-label `anxiety_trigger`. The label invites
   semantic drift; a canonical sentence pins the meaning. *The concept label is used only for filenames.*
2. **Restate-the-constraint.** Make the model restate the canonical meaning and the forbidden constraint
   in its own words **before** generating — it internalizes the forbidden list and leaks less.
3. **Stem-uniqueness.** No two outputs may share a content-word stem (scare/scared/scary = one stem),
   forcing genuinely different lexical neighborhoods instead of inflectional reshuffling.

> You are an expert at generating many different natural surface forms for the SAME meaning.
> **INTENT (preserve exactly):** "`<canonical sentence>`"
> **FORBIDDEN — WORDS:** `<words from Step 1>` **— LETTER-SEQUENCES:** `<trigrams from Step 1>`
> First restate (1) the meaning and (2) the forbidden constraint in your own words. Then produce 20 short
> phrases that mean exactly the intent. Use NONE of the forbidden words/letter-sequences. Maximize surface
> diversity (register, syntax, vocabulary, length 3–12). **No two outputs may share a content-word stem.**
> Terse/article-dropped phrasing is good. Return ONLY a JSON array `[{"text": "..."}]`.

**Empirical finding that determines the prompt's priorities (measured here, `anxiety_trigger`):** the
binding constraint is **not** content-word reuse — it is **common function words and their char-trigrams**
that happen to be in *this concept's* own training. Every one of `"she trembled at the booming bang"`,
`"my pup bolted under the bed"`, `"poor thing freaked at the rumble"` LEAKS on the **single word "the"**
(`w:the`, `c:^th`, `c:the`, `c:he$`). So **forbidden-list *compliance* is the dominant lever — above
stem-uniqueness** — because the leaks are function words, not content stems. And even one content word can
leak on a buried trigram: `"pup darted off hearing big crash"` leaks on `c:art` (from "st**art**"). The
two that obey the list pass: `"dog freaked from huge boom"`, `"kitty fled when blender rumbled"` →
`DISJOINT (seen-elsewhere)`. The productive style is **terse, article-dropped**, which matches the real
bench's own register ("critter darted past patio glass").

### Step 3 — mechanical gate: keep only `wbc`-disjoint + seen-elsewhere

```bash
# single candidate:
python3 scripts/author_disjoint.py data/authored_disjoint_eval/train.jsonl probe <concept> "candidate"
# a whole proposed file:
python3 scripts/author_disjoint.py data/authored_disjoint_eval/train.jsonl check proposed_<concept>.jsonl
```

Keep only `DISJOINT (seen-elsewhere)`. Discard `LEAKS` (own-concept overlap) and `novel` (not seen
anywhere — can't enter the seen-elsewhere bin). Re-prompt with the leaked tokens added to the forbidden
list. **Loop Steps 2–3** until you have enough survivors.

**Automation (`scripts/generate_disjoint_bench.py`).** The loop is scripted: it seeds the forbidden list
from the concept's own training, builds the Step-2 prompt, runs the gate **in-process** (calling
`author_disjoint`'s exact `for_each_feature` replica — *not* parsing CLI stdout, so it stays bit-identical
to the Rust certifier), enforces stem-uniqueness, expands the forbidden list from the **real, full** leaked
feature set each round, and stamps lineage (`provenance: synthetic`, `source_intent`, `generator_id`,
`validated_by: []`). Wire `call_llm()` to your generator, or run `--offline` and paste candidates into
`round{N}_candidates.jsonl` to exercise the gate without an API:

```bash
python3 scripts/generate_disjoint_bench.py anxiety_trigger \
  --canonical "the animal reacts with fear when something sudden happens" \
  --target 40 --generator-id <model>
```

Output: `data/generated/<concept>/survivors.jsonl` (lineage-stamped, `validated_by: []` = **not bench
data** until Step 4) and `generation_log.jsonl` (per-round yield / leaked / stem-dupes / forbidden growth
— the numbers §3 asks you to report).

### Step 4 — human validation (the F3 ground-truth step, blind to routers) — RUNNABLE

This is the load-bearing honesty step: the LLM *proposed*; a human *certifies*, and certification is what
admits a phrase. `scripts/validate_survivors.py` orchestrates blindness, adjudication, and promotion so the
judgement is unbiased and auditable. **Design = blind forced-choice reconstruction**, not "does this mean
X?": a validator sees only the phrase + the closed intent menu and picks the single best intent. A yes/no
question leaks the answer (acquiescence bias); forced-choice is a real classification, so agreement with
the survivor's *hidden* `source_intent` is strong evidence a stranger recovers the meaning from surface
alone. Disagreement that *coheres* on another valid intent is a **salvage (relabel)**, not just a reject.

```bash
# 1) Build the blind worklist (candidates + secret honeypots, shuffled) + answer
#    templates for each validator. _key.json is SECRET — never shown to validators.
python3 scripts/validate_survivors.py build --validators v1 v2 --honeypots 6
#    -> each validator fills answers_<name>.jsonl: chosen_intent + naturalness
#       (natural|stilted|malformed), blind to source_intent, generators, routers,
#       and each other.
# 2) Adjudicate: honeypot-reliability gate first (discards rubber-stampers BEFORE
#    counting), then Cohen's κ, then admit/reject.
python3 scripts/validate_survivors.py adjudicate
# 3) Promote admitted rows into the bench (re-gated vs the real seen-corpus).
python3 scripts/validate_survivors.py promote   # add --dry-run to preview
```

**Admission rule (per item):** ≥2 *reliable* validators independently choose the **same** intent, that
intent still passes the `wbc` gate vs the **real seen-corpus** (§1.5), and the majority naturalness is not
`malformed` → admit under the agreed intent (`source_intent` or a salvaged relabel). Everything else is
rejected with a recorded reason (`no_majority`, `malformed`, `gate_fail_under_agreed`, `under-annotated`).
Nothing is ever admitted to *hit a count* — the `n ≥ 47` gap is closed by validated phrases or by
generating more, never by lowering the bar.

**Anti-gaming guardrails baked in:** (a) **honeypots** — known-true bench phrases secretly mixed in; a
validator missing >1/3 has their *whole sheet discarded before adjudication*, so a rubber-stamper cannot
pad the bench; (b) **blindness** — validators never see `_key.json`, the `generator_id`, any router output,
or each other's sheets; (c) **κ reported** — low inter-annotator agreement is visible, not hidden; (d)
**re-gate on promote** — even a human-agreed phrase must still be `wbc`-disjoint vs the real corpus.

Admitted phrases become `provenance: authored` (human-vetted), carry `validated_by: [ids]`, and their
source `survivors.jsonl` row is flipped from pending (`validated_by: []`) to validated. *(LLM-assisted
pre-screening may cut human load, but the human forced-choice is the final gate for the certify surface; an
LLM-only "validation" is acceptable for §20.4 training data, never for the certify bench.)*

**Smoke-tested (2026-06-25):** on the 26 agent survivors + 6 honeypots, two synthetic sheets (one with a
deliberate single disagreement) exercised the full path → honeypot gate passed, κ=0.96, 25 admitted / 1
`no_majority` rejected, `promote --dry-run` projected the append and re-reported the bin. The synthetic
sheets were deleted; empty templates remain ready for real validators. **The remaining gate is genuinely
human: agent-proposed survivors still need ≥2 real people.**

### Step 5 — assemble the bench and run the validity gates

Append validated phrases to `eval.jsonl`, then certify the assembled bench:

```bash
GROWFORMER_INFERENCE_TOML=<abs path>/data/sentiment/inference_sentiment_core.toml \
cargo run --release --bin growformer-demos -- \
  --real-encoder-experiment data/authored_disjoint_eval embeddings_all-mpnet-base-v2.json
```

The bench is **valid only if** (these are the §17 gates, unchanged):

1. **CATA collapses** — lexical positive control fails on the bench. If CATA *passes*, the bench is
   lexically separable (easy) and `INVALID`, not a harder test. This is the anti-mirage gate.
2. **Resolvable** — disjoint bin `n ≥ 47`, Wilson CI width `≤ 0.30`. Below that → `BELOW_RESOLUTION`,
   action = generate more, **not** loosen the gate.
3. **Disjoint level `wbc`** — the strict, promotable level (§17 refused `wb`/`w` fallback).

### Step 6 — lineage stamp

Record the bench manifest: per phrase `{source_intent, generator_id, validated_by, κ}`. This is F1/F2 in
practice — it lets a future reader prove the certify surface was human-validated and that no phrase is in
both a train and a test set.

### Pilot result (2026-06-25, local Ollama `llama3.2:3B`) — measured, not assumed

Two concepts, `play_invitation` and `greeting_check_in`, via `scripts/generate_disjoint_bench.py`. The
numbers and the **two findings that reorder the whole approach**:

| generator | `play_invitation` | `greeting_check_in` |
| --- | --- | --- |
| llama3.2:3B, baseline prompt | **0** / ~60 | **0** / ~36 |
| llama3.2:3B, hardened prompt | **5** / ~80 (~6%) | **4** / ~80 (~5%) |
| **capable LLM (agent), hardened prompt, 1 batch** | **11 / 14 (79%)** | **4 / 8 (50%)** |

The third row is the decisive one: a capable generator clears the floor at **~50–79%**, and — unlike the 3B
— the survivors are **faithful and natural** ("fancy a romp, bud?", "tussle now, lil mate", "ahoy mate,
chipper?"), not degraded salad. Its few leaks are precise and fixable: all on **word-initial trigrams**
(`^fe` feel, `^sa` salutations, `^do` doggo) whose 2-letter prefix sits in own-concept training — a
trivial next-round avoidance rule. **Conclusion: generator capability, not the gate, was the binding
constraint on *faithful* yield.** Two caveats remain non-negotiable: (a) a capable generator still
**proposes only** — Step-4 blind human validation is unchanged (F3); and (b) using the *in-loop agent* as
generator carries a mild provenance wrinkle (it has seen project context) that a **blind external API**
call would not — prefer the blind API for the real bench build, agent-as-generator for piloting.

**8-concept agent pilot (2026-06-25).** Anthropic API was blocked on account credits (valid key, model
`claude-sonnet-4-6` exists — a billing block, not a bug), so the agent generated via `--offline`.
Result: **26 disjoint survivors across 8 concepts** (weather 6, compliment 6, play 5, bedtime 4, treat 2,
bonding 2, greeting 1, mealtime **0**), lineage-stamped `generator_id: agent/claude`, `validated_by: []`.
Two honest notes: (i) stem-uniqueness deduped aggressively where authored candidates were repetitive
(play 5 of 15; greeting 1 of 9 — many "ahoy…chipper" stems collapsed), so *stem diversity*, not raw count,
is the real authoring lever; (ii) `mealtime_request` yielded **0** — its core vocabulary
(eat/food/meal/dinner/feed/hungry/snack) plus common trigrams left almost no faithful escape, the
structural "concept-vocabulary-is-the-forbidden-set" wall (cf. greeting). Survivors are written to
`data/generated/<concept>/survivors.jsonl` and remain **pre-validation** until Step 4.

**Finding 1 — the prompt is the dominant lever, not the model size.** Baseline yield was a flat **0%**;
the hardened prompt (explicit ban on `the/to/a/is/your/you` + `-ing`/`-er`, terse telegraphic style,
per-phrase self-check, verified passing-style few-shot) moved it to **~5–6%** on the *same* 3B model.
(The 70B `r1-1776` was unusable locally — HTTP 500 / failed load.) Confirms the binding constraint is
ultra-common function-word/trigram overlap with the concept's own training, and that *compliance*
pressure, not capability, is what clears it first.

**Finding 2 (the load-bearing one) — mechanical survival ≠ usable; the gate cannot see meaning.** The
survivors are surface-disjoint but **semantically degraded**, because escaping the trigram floor forces
the model into rare/abstract vocabulary: `"Pet greets human with query"` (**role-reversed** — canonical
is *person→pet*), `"Species interaction is engaged"` / `"Animal acquaintance initiates inquiry"` (abstract
salad), with only `"Send friend with ball"`-class phrases both disjoint *and* faithful. The `wbc` gate
passes all of them; **Step 4 human validation must reject the drift**, so the yield of *faithful, natural*
disjoint phrases is **well below** the ~5–6% mechanical rate. Two consequences:

- **Step 4 is doing the heavy lifting, exactly as designed.** The mechanical gate is necessary but not
  sufficient; without blind human validation, a generated bench would be surface-disjoint nonsense that
  *looks* like a hard test and isn't. This is §1 F3 earning its place empirically.
- **A tension the spec must own:** the `wbc` floor and meaning-preservation pull against each other — the
  phrasings that clear the floor are the *least* like real user traffic. Generation therefore (a) does
  **not** make clean capability data cheap (the human-faithful-disjoint intersection is a narrow target),
  and (b) the survivors skew toward an unnatural distribution, which is a further reason they certify
  **capability only**, never deployment (§0). Do **not** read "we can generate the bench" as solved; read
  it as "generation cheaply *proposes*, the gate filters surface, and humans still carry meaning — at a
  low net yield." A more capable generator (cloud) is the untested lever most likely to raise *faithful*
  yield; it is the recommended next probe, not a settled result.

---

## §3 What this un-gates (and what it still doesn't)

**Un-gated immediately by a larger validated bench:**

- **The graph-router edge-ablation (§2.6).** A bigger bench makes the `E1/E2 > E0` CI tighter, and the
  generator can be steered to propose **`contrasts_with` candidates** (semantically-near-but-opposite —
  the one place edges might earn their keep). *The relational slice's membership stays router-blind
  annotator agreement (`SEMANTIC_GRAPH_ROUTER_SPEC.md` §3.1.1) — the generator proposes, the humans
  define the slice.*
- **Encoder comparisons / distillation / cone router** — every capability question in
  `GROUNDING_LOOP_SPEC.md` §20.6, now at higher n.

**Still NOT moved (the boundary, restated):**

- **0/47 deployment.** The generated bench is `authored`; §18 deployment certification requires
  `RealTraffic`. No generated phrase counts. The deployment number moves only via real users reaching the
  §21 shadow harness.
- This is the §4.2 discipline applied to *data*: generation is legitimate **capability** work that runs
  off the data clock; it does not accelerate the off-keyboard mover.

---

## §4 One-paragraph summary

You can generate the **capability** bench, not the **deployment** bin. The recipe is: `forbidden` →
LLM-propose-blind → `probe`/`check` to keep `wbc`-disjoint+seen-elsewhere → ≥2-human validate (blind) →
assemble → `--real-encoder-experiment` and require CATA-collapse + resolvable + `wbc`. The firewall makes
it honest (the LLM proposes, humans certify; synthetic lineage tracked; never the deployment surface).
This un-gates the graph-router ablation and every other capability experiment today — and it deliberately
does **not** touch 0/47, which only real traffic moves.
