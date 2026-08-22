# Self-Revising Grounding Loop (Assisted Maintenance)

**Purpose:** The first loop in which the system *participates* in maintaining its own grounding from error — capturing routing failures, proposing graph edits, and certifying whether those edits generalize. It doubles as the out-of-lexicon coverage audit the deployed router needs anyway.

**Honest scope (read first):** This is **system-proposes, human-disposes**. The grounding graph is still authored; the system makes it author-*assisted*. This moves one notch toward self-grounding — it does **not** achieve autonomous grounding generation, which is the research frontier and is explicitly out of scope here (§8). The certifier measures whether assisted maintenance *generalizes*; it does not claim the system "learned" the concept.

**The central trap:** a loop that adds aliases for phrases it has seen will reduce observed fallthroughs while only **memorizing the observed phrase distribution** — coverage rises on seen phrases, generalization to unseen phrasings does not. This is the lexicon analog of the n=30 mirage. The certifier (§6) is built to catch it: coverage is measured on **held-out paraphrases the proposal mechanism never saw**, never on the captured phrases themselves.

---

## 1. The loop

```
routing failure → capture → propose edit → collision check → human gate → CERTIFY → integrate
                                                                              │
                                                                  (held-out generalization test)
```

Integration happens only after both the human gate **and** the certifier pass. An approved-but-uncertified edit is staged, not merged.

---

## 2. Capture — what counts as a routing failure

Log a phrase as a failure when any fire:

- **Entropy guard triggered** (routing distribution near-uniform — the degenerate/uncertain signal already wired into live paths).
- **No grounding node activated** above the activation threshold (true out-of-lexicon).
- **Low max routing confidence** (below a deployment threshold), even if a node activated.
- **Behavioral dissatisfaction signal** where available: user rephrases immediately, abandons, or thumbs-down within the turn.

Logged record: `{phrase, encoder_embedding, activated_nodes+scores, max_confidence, entropy, trigger_reason, downstream_signal, timestamp, domain_context}`.

**Stratify captures by inferred concept at capture time** (§3) so the propose/certify split in §6 can be concept-balanced.

---

## 3. Propose — alias candidate vs. new-node candidate

The proposal mechanism uses encoder embedding similarity (clifford_e8) — the same cosine-routing family with known boundary limits. Acceptable here **only because a human gates it** and the certifier validates it.

For each captured phrase `p` with embedding `e_p`:

1. Compute cosine similarity to all existing grounding-node embeddings (node embedding = centroid of its current aliases' embeddings).
2. **Alias candidate:** if `max_sim ≥ τ_alias`, propose `p` as a new alias of the nearest node `n*`.
3. **New-node candidate:** if `max_sim < τ_alias`, buffer until a cluster forms (`≥ k_min` phrases within `τ_cluster`).

Never auto-propose a new node from a single phrase.

---

## 4. Collision check (mandatory, pre-approval)

Each proposed edit is checked against **all loaded domain graphs**. Flag **collision risk** when `p` would activate a foreign-domain node above threshold.

---

## 5. Human-approval gate (anti-rubber-stamp)

Surface evidence + pre-computed generalization estimate + collision conflicts. Rate-limit approved additions per period.

---

## 6. Certifier — generalization vs. memorization

Split captured failures, per concept, into propose-set and held-out certify-set. Measure **before vs. after** each batch:

| Metric | Genuine-improvement signal |
| --- | --- |
| Held-out paraphrase routing accuracy | **Rises** on certify-set |
| Captured-set coverage | Rises (necessary but not sufficient) |
| Generalization gap = captured − held-out | **Stays small** |
| Cross-domain misroute rate | **Does not rise** |

---

## 7. Decision rule (per approved batch)

| Condition | Verdict | Action |
| --- | --- | --- |
| Held-out ↑, gap small, misroute flat | **Genuine coverage improvement** | Integrate |
| Captured ↑, held-out flat, gap widening | **Lexicon memorization** | Reject |
| Held-out ↑ but misroute ↑ | **Net-negative collision** | Integrate non-colliding subset |
| Held-out plateaus | **Saturation** | Stop auto-proposing; escalate to §8 |

---

## 8. Frontier (out of scope)

Autonomous concept induction from training stream without human authoring.

---

## 9. Continuous telemetry

Fallthrough rates, held-out accuracy, generalization gap, collision rate, proposal accept rate, coverage-vs-additions curve.

---

## 10. Implementation checklist

- [x] Failure-capture hook (`capture_lightweight`, `maybe_capture_grounding_failure`)
- [x] Embedding-similarity proposer: alias vs buffered new-node clustering
- [x] Cross-domain collision checker
- [x] Per-concept propose/certify split
- [x] Certifier metrics + decision rule
- [x] Coverage-vs-additions curve (n-sweep analog) + overfitting-signature flag
- [ ] Human-approval UI (staging only in Rust artifacts today)
- [x] Staging via `integrated: false` on proposals until certifier passes
- [x] `--grounding-loop-audit` / `--grounding-loop-analyze`
- [x] Pluggable embedder + bring-your-own precomputed vectors (`install_vector_embedder`)
- [x] Data-driven `τ_alias` calibration (`calibrate_alias_threshold`)
- [x] Positive control (semantic geometry → `genuine_coverage_improvement`)
- [x] Supervised projection encoder, no external deps (`SupervisedEncoder`)
- [x] Real-companion harness (`--grounding-loop-luna`): lexical vs supervised on a loaded graph
- [x] Token-disjoint generalization audit (`--grounding-disjoint-test` / `-analyze`, §14): feature-granularity overlap curve, overlap-0 sub-strata, shuffle null (B=200), CATA positive control

```bash
cargo run --release --bin growformer-demos -- --grounding-loop-audit
cargo run --release --bin growformer-demos -- --grounding-loop-analyze
cargo run --release --bin growformer-demos -- --grounding-loop-luna default
cargo run --release --bin growformer-demos -- --grounding-disjoint-test default
```

**Artifacts:** `grounding_loop_captures.csv`, `grounding_loop_proposals.csv`, `grounding_loop_curve.csv`, `grounding_loop_results.txt`

**Code:** `src/inference/grounding_loop.rs`, fleet audit API on `src/inference/world_grounding.rs`

---

## 11. Representation finding (certifier travels with this)

The proposer/certifier measure cosine similarity on a phrase embedding. The encoder choice is **load-bearing and is itself a measured result**, not an implementation detail:

- **Raw CliffordE8 encoder is degenerate for this task.** Short, L2-normalized inputs pass through `E8Lattice::quantize_64d`, whose `nearest_point` snaps every coordinate (magnitude ≲ 0.9) to the **origin** lattice point. The vector becomes all-zero, so `cosine_similarity` returns 0 for *every* pair — including a phrase with itself. A `TokenDictionary` does **not** rescue this: the CATA centroid is also small-magnitude and quantizes to zero. Any downstream "coverage" number computed on zero vectors is a **tie-break artifact** (stable-sort returns the first node), not signal.
- **The loop embeds on the pre-quantization CATA centroid** (`ChunkCodec::encode_text(..).centroid()`, installed via `install_phrase_embedder_from_corpus`). This is non-degenerate and in the same clifford_e8 family, but `build_token_embeddings` assigns each token id an **independent random unit vector** → the representation is **lexical** (shared-token overlap), with no cross-token semantics. The bridge routing vector is the last-resort fallback (layer-normed, cannot collapse), but in offline audit it is domain-untrained and uninformative.

**Consequence — the certifier behaves exactly as designed.** On the lexical embedder, the audit fixture yields:

| Batch | captured-set coverage | held-out paraphrase acc | gen. gap | verdict |
| --- | --- | --- | --- | --- |
| Genuine (approved propose-set aliases only) | 40% → 40% | 20% → 20% | 0.20 → 0.20 | **saturation** (no approved proposals — paraphrases misroute) |
| Memorization (exact captured phrases as aliases) | 40% → **80%** | 20% → 20% (flat) | 0.20 → **0.60** | **lexicon_memorization → rejected** |

This is the lexicon analog of the n=30 mirage made concrete: adding the *seen* phrases raises captured-set coverage while held-out paraphrase accuracy stays flat and the generalization gap widens — and the §7 decision rule rejects the batch. Achieving the *genuine* row (held-out lift with a small gap) requires a representation with **non-lexical semantic structure** (a trained encoder), which is the §8 frontier; the available encoders do not provide it.

**Coverage-vs-additions curve (the n-sweep itself).** `coverage_vs_additions_curve` adds aliases one at a time and re-measures; the memorization sweep is the smoking gun (`grounding_loop_curve.csv`):

```
 additions | captured | held-out | gap
         0 |    40.0% |    20.0% | 0.200
         2 |    60.0% |    20.0% | 0.400
         3 |    80.0% |    20.0% | 0.600
         5 |    80.0% |    20.0% | 0.600   → captured +40.0pp, held-out +0.0pp
```

Captured-set coverage climbs monotonically with every exact phrase added while held-out paraphrase accuracy is dead flat — the overfitting/plateau signature. The audit emits an `n-sweep overfitting signature` boolean from `curve_lifts` (captured lift > 0.05 **and** held-out lift < `min_held_out_lift`). A genuine encoder would instead show held-out rising *with* additions; that is the pass criterion to watch for when the encoder is upgraded.

---

## 12. Encoder upgrade path (recommendations, implemented)

The audit is the **acceptance test** for any candidate encoder. Three mechanisms make the swap a measurement rather than a leap of faith:

1. **Pluggable embedder / bring-your-own vectors.** `embed_phrase` now consults an installed embedder before falling back to CliffordE8/bridge. Two installers exist: `install_phrase_embedder_from_corpus` (the lexical CATA centroid, default) and `install_vector_embedder(map)` — a `HashMap<phrase, Vec<f32>>` of **precomputed embeddings** from any external/semantic encoder run offline over the captured phrases + node aliases. Unknown vocabulary maps to a deterministic per-string unit vector (a stable distractor, never zero), so missing coverage can never silently regenerate the tie-break artifact. No live model dependency is required to evaluate a candidate.

2. **Data-driven threshold calibration.** `calibrate_alias_threshold` reports same-concept vs cross-concept similarity means under the *current* encoder and suggests `τ_alias` at their midpoint. On the lexical embedder it reports same/cross ≈ `0.155 / 0.005` (suggested τ ≈ 0.08 vs the 0.42 default) — confirming the absolute similarities are tiny and barely separated. Re-derive thresholds per encoder; do not trust the hand-picked default.

3. **Positive control.** `positive_control_certifies_genuine` (unit test) and the `POSITIVE CONTROL` section of the audit build a synthetic *semantic* geometry: each concept's held-out paraphrase sits just past the baseline boundary, and one approved alias pulls the centroid over. The certifier returns **genuine_coverage_improvement** and the curve inverts cleanly:

```
 additions | captured | held-out | gap
         0 |   100.0% |     0.0% | 1.000
         1 |   100.0% |    33.3% | 0.667
         2 |   100.0% |    66.7% | 0.333
         3 |   100.0% |   100.0% | 0.000   → captured +0.0pp, held-out +100.0pp
```

This is the mirror of the lexical negative result: it proves the §7 gate is **not rigged to always reject** — it certifies generalization when the representation actually carries it. A real encoder swap passes the gate iff its genuine sweep looks like this control, not like the lexical CATA result.

**Workflow to evaluate `all-MiniLM`-class or a distilled student:** embed every captured phrase + node alias offline → `install_vector_embedder(map)` → re-run `--grounding-loop-audit`. Watch the genuine sweep and the calibrated τ. Keep CliffordE8 as an optional geometry layer *on top of* real embeddings (and re-evaluate quantization), not as the embedding itself.

4. **Supervised projection encoder (no external dependency).** `SupervisedEncoder` learns a softmax over concepts from labeled captures (hashed word/bigram/char-trigram features, one linear layer, SGD). The embedding is the L2-normalized concept distribution, so paraphrases of the same concept cluster even with disjoint surface tokens — the non-lexical structure the genuine row needs — **without importing a sentence-transformer.** Install via `install_supervised_embedder`. This is the recommended fix when the author already has labeled utterances (e.g. a companion's `semantic_intent`).

---

## 13. Real-companion result (Luna)

`--grounding-loop-luna <dir>` loads a real companion: its `pet_world_grounding.toml` (119 runtime nodes) as the index, and its labeled JSONL utterances split per-intent into propose/certify. It runs the certifier twice — lexical CATA vs the supervised projection (trained on the propose split only, so certify is held out from the encoder too). On Luna (26 intents that are graph nodes, 189 propose / 179 held-out certify):

| Encoder | concept separation (same / cross) | calibrated τ_alias | held-out routing | proposals (approved) | genuine verdict |
| --- | --- | --- | --- | --- | --- |
| Lexical CATA centroid | 0.015 / −0.001 (≈0) | 0.007 | **5.0%** (≈chance) | 1 (0) | saturation (nothing to propose) |
| Supervised projection | **0.592 / 0.292** | **0.442** | **20.7%** | 171 (67) | small lift (+1.1pp) |

Takeaways, all visible on real data:

- **Lexical overlap collapses at graph scale.** With 119 real nodes the lexical encoder has essentially zero concept separation and routes at chance — the §11 finding, confirmed beyond the toy fixture.
- **A tiny supervised projection (no deps) recovers most of the signal:** concept separation 0 → ~0.30, held-out routing 5% → 21% (4×), and it surfaces real proposals (171 vs 1).
- **The calibrator independently re-derives τ_alias ≈ 0.44**, almost exactly the hand-picked 0.42 default — the threshold is validated, not guessed.
- **Aliasing alone does not generalize on top of a working encoder** (genuine held-out lift +1.1pp): the lift comes from the *representation*, not from adding seen phrases as aliases. The memorization sweep makes this explicit — captured 35% → 84% while the generalization gap blows out to 0.48. The certifier correctly refuses to bless that as generalization.

Net: the loop and certifier behave correctly on a real companion, and the supervised projection is a viable in-repo encoder. The remaining headroom (held-out 21%, not 80%) is an encoder-quality question — more labeled data, a hidden layer, or a stronger learned representation — measured honestly by the same gate.

**Correction (see §14):** the 20.7% pooled held-out number is itself a confound until audited. The token-disjoint test below shows it does **not** survive at zero feature overlap — it collapses to ≈0–4% (at/under the shuffle floor) and rises monotonically with overlap to 45%. The honest held-out generalization of the current supervised encoder is at floor, not 21%; 20.7% is in-distribution coverage carried by surface-feature overlap. Size encoder upgrades against the floor, not the pooled number.

---

## 14. Token-disjoint generalization audit (`--grounding-disjoint-test`)

The certifier's held-out paraphrase metric counts a route correct even when the certify phrase shares surface features with its concept's training phrases — so a "held-out" hit can be lexical recall, not generalization. The disjoint test decides which, by measuring routing accuracy as a function of feature overlap **at the encoder's own feature granularity** (words ∪ bigrams ∪ char-trigrams — the same `for_each_feature` the encoder hashes, so "vacuum"/"vacuuming" are correctly *not* disjoint).

Four design decisions (each automated):

1. **Disjointness at feature granularity, not whole words** (`phrase_feature_set`, shared with the encoder) — a word-level filter would leak char-trigram signal and over-credit generalization.
2. **A curve, not a binary** — accuracy vs. overlap bins `{0, (0,0.1], (0.1,0.3], (0.3,0.6], (0.6,1.0]}` with n + Wilson 95% CIs. Flat as overlap→0 ⇒ real; monotone-declining ⇒ the headline was carried by the high-overlap bin.
3. **Overlap-0 decomposed** into (a) *seen-elsewhere* (features seen on other concepts but routed correctly anyway = genuine structural generalization) vs (b) *novel-features* (routes by prior). The headline is sub-bin (a).
4. **Two controls bracket it.** A shuffle null (B=200 retrains on permuted labels) defines "chance" empirically as the overlap-0(a) floor distribution. A positive control runs the same curve on lexical CATA, which **must** floor at overlap-0 — if it doesn't, overlap is mis-measured and the test is invalid before any supervised number is read.

**Result on Luna (26 concepts, 189 propose / 179 certify):**

- Overlap measure **VALID** — lexical CATA scores 0.0% at overlap-0 (≤ shuffle floor95 = 15.0%).
- Supervised curve is monotone in overlap: **0% → 0% → 2.9% → 10.4% → 45.3%**.
- Overlap-0 seen-elsewhere (the generalization headline): **0/7 = 0%**, below the 15% floor.
- Word+bigram-disjoint (looser, n=24): **4.2%**, also ≤ floor.
- **Verdict: BELOW RESOLUTION** for the strict union-disjoint bin (n=7, CI [0%, 35%]) — but the monotone curve plus the corroborating n=24 bin at floor make the reading clear: **the pooled 20.7% is overlap-driven (lexical-in-disguise)**, not generalization.

Consequences (closing the loophole one level up):

- **Every held-out number is reported as `pooled X% (disjoint-bin(a) g%, shuffle floor f%)`** — uncertainty travels with it.
- **The grounding-loop §6 held-out metric inherits this:** a coverage improvement is credited as generalization only if it survives the disjoint bin, not merely raises pooled accuracy.
- **Encoder-upgrade sizing is against the floor**, not 20.7% — the true gap to a working router (~80%) is larger than the pooled number implied.
- Honest limit (§8): the union-disjoint bin is small in one language (char-trigrams make full disjointness rare); the word+bigram-disjoint curve is the looser secondary, and a wide CI at overlap-0 is "underpowered," not a pass.

```bash
cargo run --release --bin growformer-demos -- --grounding-disjoint-test <luna_dir>
cargo run --release --bin growformer-demos -- --grounding-disjoint-analyze
```

**Artifacts:** `grounding_disjoint_curve.csv` (per-encoder, per-level bins), `grounding_disjoint_meta.txt` (graph path for analyze), `grounding_loop_results.txt`.

---

## 15. Certifier-first pipeline — the contract every encoder is judged by (`--certify-encoder`)

§§11–14 each caught a confound *after the fact* (degenerate embeddings, then aliasing-isn't-generalization, then overlap-driven pooled accuracy). The pattern: a number is only trustworthy once an automated gate has judged it. §15 makes the certifier a **contract**, not a test you remember to run. Every embedder — current, v2, or any future drop-in — passes through one deterministic pipeline that emits a single **verdict artifact**. *No encoder is an "improvement" until it emits a passing verdict.* The gate is built before the next encoder, so v2's number is judged the same way the 20.7% finally was.

**The go/no-go field is `disjoint_semantic_lift`** = `disjoint_gen_a − semantic_floor_95` (observed seen-elsewhere disjoint accuracy minus the 95th-percentile shuffle null). Pooled accuracy is recorded but **never gates**. An encoder generalizes iff lift > 0 *and* the lift CI excludes 0.

**Auto-run sequence (deterministic given `(encoder, data_hash, seed)`):**

```
install embedder (BYO-vectors hook / supervised / cata)
  → provenance check + augmentation firewall (§4 of the spec)   [INVALID if dirty]
  → per-concept propose/certify split (certify = real_traffic only)
  → disjoint curve at feature granularity + overlap-0 sub-bins (a)/(b) + memorization gap
  → positive control: lexical CATA must collapse at overlap-0    [INVALID if not]
  → shuffle control (B=200, retrain subject on permuted labels)  → semantic_floor
  → disjoint_semantic_lift = disjoint_gen_a − semantic_floor_95
  → verdict state machine → emit verdict_<encoder>_<datahash>_<seed>.json
```

The shuffle and positive controls run **every** time. The shuffle null retrains the *subject encoder class* on permuted labels (the proper null for a learned projection); for a label-independent encoder (CATA / frozen vectors) it reduces to permuting the overlap definition.

**Augmentation-leak firewall (§4 — the new safety contract).** v2 trains with augmentation (synonym swaps, template rewrites), generated *from the label distribution*; if any reach certify, the encoder certifies itself. Every phrase carries a `provenance` tag (`real_traffic` / `authored` / `augmented` with `derived_from` lineage). Enforced invariants, violation ⇒ `INVALID` (not a score): (1) **certify ⊆ real_traffic**; (2) **no lineage crossing** (no certify id is the source of any training augmented phrase). This is *orthogonal* to the disjoint test — that catches feature leakage, this catches pipeline leakage.

**Verdict state machine (deterministic):**

| Verdict | Condition |
| --- | --- |
| `INVALID` | positive control didn't collapse **or** firewall dirty (pipeline/data problem) |
| `BELOW_RESOLUTION` | disjoint-bin(a) `n < 8` or Wilson CI width `> 0.30` (underpowered, *not measured*) |
| `FAIL_MEMORIZATION` | lift ≤ 0 or lift CI includes 0; or memorization gap > 0.50 (lookup table) |
| `FAIL_COLLISION` | lift > 0 but `collision_delta > 0` (net-negative) |
| `PASS` | lift > 0, CI excludes floor, controls clean, collision ≤ 0, gap not blown |

`INVALID` and `BELOW_RESOLUTION` are distinct from `FAIL`: "not measured," never "measured and bad." An underpowered or invalid run is never readable as a pass.

**Result on Luna (supervised, seed 42, candidate set = all 119 nodes):**

```
verdict=BELOW_RESOLUTION  lift=-0.143  ci=[-0.143,+0.211]
  disjoint_gen_a=0.000 (n=7)  semantic_floor_95=0.143  pooled=0.207  gap=0.148
  overlap_curve: 0% → 0% → 2.9% → 10.4% → 45.3%   (textbook lexical signature)
  positive_control_collapsed=true  firewall_clean=true (train real=189, certify real=179)
```

The pipeline reaches the same conclusion §14 reached by hand, but as a machine verdict: the supervised projection does **not** earn a pass — the seen-elsewhere disjoint bin (n=7) is underpowered, and the point estimate is at/under the shuffle floor (lift negative). It is `BELOW_RESOLUTION`, never `PASS`. This is the gate doing its job *before* any v2 is built.

**GLE through the same gate (BYO-vectors hook, `--certify-encoder gle`).** The distilled GLE student is reduced to precomputed vectors over every phrase/alias the pipeline embeds and run through the identical contract. Provenance is favorable: the GLE was distilled for *support/coding dispatch* (per its `.meta.json`), so Luna companion traffic is domain-disjoint from its training — the cleanest possible eval (zero-shot, no Luna-specific fitting), recorded in the artifact's `encoder_provenance`.

```
encoder=gle_routing_tuned  data_hash=222ed1857dbd221e (same Luna corpus → comparable)
VERDICT: BELOW_RESOLUTION
  disjoint_semantic_lift = -0.050  ci=[-0.050,+0.304]
  disjoint_gen_a = 0.000 (n=7)  semantic_floor_95 = 0.050  pooled = 0.011  gap = 0.000
  positive_control_collapsed = true  firewall_clean = true
```

The result is unambiguous and important: the GLE's headline **100% intent accuracy does not transfer** — on Luna it routes at **pooled 1.1%**, near zero even in the easy high-overlap bins. The 100% is an *in-distribution* (support/coding) number; as a general semantic encoder for a new companion's graph it has no signal. Formally `BELOW_RESOLUTION` (strict disjoint bin n=7 < 8), but the near-zero pooled makes the reading clear: this is not a Luna encoder. Two distinct facts fall out:

1. **Cross-domain transfer (measured here): the GLE does not generalize zero-shot to a new companion.** Mirage #4 confirmed — the perfect score was domain-bound.
2. **Certifying the 100% itself is a *different* eval we have not wired:** it requires the GLE's own held-out *support/coding* traffic with its own grounding graph run through this same gate. Until that artifact exists, the 100% remains uncertified in its own domain too — running ≠ generalizing, now with the verdict to prove the gap is real.

Both encoders we can currently run (supervised-on-Luna, GLE-zero-shot-on-Luna) return `BELOW_RESOLUTION`: the disjoint held-out bin (n=7) is underpowered. Per §6 that licenses exactly one next action — **capture more real disjoint traffic to power the bin**, not build production scaffolding around an encoder that has not cleared the gate.

**Promotion contract (what makes this P0):** an encoder is adopted only on a **promotable** `PASS` artifact (`Verdict::is_promotable()` — a strict `wbc` pass; `PASS_PROVISIONAL` earned at a coarser fallback level does *not* license promotion); the artifact id is recorded with the deployed encoder version. The verdict is deterministic and append-only (`certify_artifacts/`), so lift-over-version and coverage elasticity (`Δdisjoint_gen_a / Δaliases`) are comparable across encoders — a real v2 improvement is distinguishable from a pooled-number mirage *across time*, not just at one snapshot. Production tooling (human gate, live integration, bootstrapper) proceeds only after *some* encoder returns `PASS`; building it around a `FAIL`/`BELOW_RESOLUTION` encoder is scaffolding around an unsolved core.

```bash
cargo run --release --bin growformer-demos -- --certify-encoder supervised [companion_dir] [seed]
cargo run --release --bin growformer-demos -- --certify-encoder cata
cargo run --release --bin growformer-demos -- --certify-verdict certify_verdict_latest.json   # re-read/compare
```

**Artifacts:** `certify_artifacts/verdict_<encoder>_<datahash>_<seed>.json` (append-only longitudinal store) and `certify_verdict_latest.json` (latest). Pure logic (firewall, state machine, `data_hash`) is unit-tested in `grounding_loop.rs`; the gating decision is isolated in `decide_encoder_verdict` so the contract is testable independent of the routing/embedding machinery.

### §15.1 In-domain GLE certification (the 100% on its own turf)

The Luna run measured cross-domain transfer — a *correct* measurement of an irrelevant question (a support/coding encoder routing pet-companion traffic). It is **not** the test of the headline. The headline — the GLE's reported **100% intent accuracy** — lives in its native support/coding domain, and `--certify-gle-indomain` points the identical gate at exactly that, in two constructions (run from `data/language/m5`):

- **Construction A — the literal 2-way 100%.** Reproduces `demo_language_distill_experiment`'s eval: two concepts (`support` vs `coding`), node centroids built from the routing fine-tune train split, certify = the held-out split (so it is provenance-disjoint from the GLE's routing fine-tune train).
- **Construction B — home-domain many-way routing.** ~15 `action_target` classes with ≥4 samples (727 propose / 721 certify), node centroids from each class's propose split — the same shape as the Luna fixture, on the GLE's home domain.

**What the source revealed before the run.** The distillation "teacher" is a `HashingLanguageEncoder` (`// stand-in teacher proxy`), not a real language model — the GLE was distilled to mimic a *lexical hashing* geometry, then fine-tuned to separate two lexically-maximally-distinct buckets. The "100%" is a **2-way** cosine-to-prototype score, recorded in the model card as `routing_acc_support_coding`. So the easy-eval risk is structural, not hypothetical.

**Verdicts (both INVALID, but read the reason):**

```
gle_2way_support_coding     VERDICT=INVALID  pooled=1.000 (reproduces the 100%)  disjoint_gen_a n=0
gle_indomain_action_target  VERDICT=INVALID  pooled=0.155 (15-way, ~2.3× chance)  disjoint_gen_a n=0
reason (both): no feature-disjoint held-out phrases at any granularity (overlap-0 seen-elsewhere
               bin empty, level=wbc): every held-out phrase shares features with its own class's
               training, so the eval cannot separate memorization from generalization
```

The decisive finding is **not** an encoder verdict — it is a verdict **about the eval**: the GLE's home-domain held-out sets have *no feature-disjoint examples at any granularity* (word, bigram, or union). Every held-out phrase shares at least one feature with its own class's training, so the seen-elsewhere disjoint bin is empty and lift is structurally unmeasurable. The gate refuses to certify (INVALID) rather than read the pooled number, which is exactly right: the 100% (2-way) and 15.5% (15-way) are both un-disentanglable from lexical overlap on these evals. The 100% reproduces faithfully and is confirmed to be a **2-way, lexically-entangled** number — it never measured semantic generalization, and the home-domain eval as constructed *cannot* measure it.

**Two mechanism changes this surfaced (both honest tightenings, not goalpost-moving):**

1. **Granularity fallback for the disjoint bin (`disjoint_level`).** On dense training the union (`wbc`) disjoint bin is empty; the gate now resolves lift at the finest level with a populated seen-elsewhere sub-bin (`wbc`→`wb`→`w`), recording the level (looser ⇒ leakier). Here even `w` is empty, so it stays `wbc` and the bin is honestly reported as `n=0`.
2. **Non-vacuous positive control + `invalid_reason`.** An empty disjoint bin can no longer pass the positive control vacuously (`positive_control_collapsed=false` when the bin is empty), and every INVALID now records *which* failure mode it is — empty bin (eval can't separate memorization) vs. CATA-not-collapsing (eval lexically separable / easy) vs. firewall/data.

**Where this leaves the encoder question.** No encoder has returned `PASS`. The honest state is: the supervised projection is `BELOW_RESOLUTION` on Luna; the GLE is a domain-bound 2-way lexical separator whose home-domain eval cannot even be certified for generalization. To resolve the GLE's in-domain claim, the next data need is a **feature-disjoint** support/coding held-out set (paraphrases that share *concept* but not *surface features* with training) — without disjoint examples, no eval in this domain can distinguish memorization from generalization, and the 100% remains uncertifiable rather than refuted.

```bash
cargo run --release --bin growformer-demos -- --certify-gle-indomain   # A (2-way 100%) + B (action_target)
```

### §15.2 The impossibility result, the fallback guardrail, and the eval acceptance instrument

**State the result in its strongest form.** The decisive finding is not "the GLE is INVALID." It is that *the GLE's home-domain eval is structurally incapable of distinguishing generalization from memorization* — it contains no feature-disjoint held-out examples at any granularity, so **any** score on it (100% or 15%) is silent on generalization. This is a clean impossibility result about the measurement, not a shrug: you cannot tell whether the GLE learned the concept or memorized the surface, because the eval set *cannot carry that information*. It generalizes beyond this encoder — it indicts any accuracy reported on a held-out set that was never disjointness-filtered.

**The 100% decomposes into four composed mirages.** The distillation teacher is a `HashingLanguageEncoder` (a *lexical* geometry) → distilled into a student → scored on a **2-way cosine-to-prototype** separation between two maximally-distinct buckets → on an eval with **zero** disjoint examples. Every layer is the easy-task / lexical-overlap signature, stacked; the model card presented this as semantic routing with "OOD AUROC 1.000". The gap between that description and "a 2-way lexical separator on an unmeasurable eval" is the entire value of having run the gate.

**Fallback guardrail (`PASS_PROVISIONAL`).** The `disjoint_level` fallback (`wbc`→`wb`→`w`) is the one tightening with a misuse mode: "report lift at the loosest granularity that has examples" is also "relax the disjointness definition until something populates the bin," and looser is more lexical-leakage-prone. The guardrail: the verdict state machine now treats a lift cleared only at a fallback level (`wb`/`w`) as **`PASS_PROVISIONAL`** — real but leakier, and **not promotable** (`Verdict::is_promotable()` is true only for a strict `wbc` `PASS`). A provisional pass must be re-earned at `wbc` on a feature-disjoint eval to become a clean `PASS`, so the fallback can never launder surface overlap the finest filter would have caught. (Here it never triggers: even `w` is empty.)

**The eval acceptance instrument (`--verify-disjoint-eval`).** The missing artifact is a *feature-disjoint, concept-preserving, surface-disjoint* held-out set in the GLE's home domain. That is **data**, and it must be real rephrasings — not augmentation (synonym-swapped templates re-leak through the firewall, and on a hashing-distilled student may not even be concept-preserving in the encoder's geometry). Before any such candidate set is spent on a certification run, it is judged by an encoder-free acceptance instrument with two gates:

1. **Disjointness (Gate 1, `audit_disjoint_eval`, encoder-free).** Per class, every eval phrase's `wbc` features vs. that class's training features; the set must contain ≥ `DISJOINT_MIN_N` (8) **seen-elsewhere** disjoint phrases (overlap-0 with own class, but features present on some other class — the gen_a bin). Fewer ⇒ the eval cannot separate the two hypotheses and is `REJECTED` (the GLE-home failure mode, now caught pre-flight).
2. **CATA collapse (Gate 2).** Lexical CATA must collapse to floor (≤0.20) at overlap-0 on the candidate eval. If CATA also scores high, the disjoint phrases are still lexically separable — an easy task — and the eval is `REJECTED`.

Provenance rules carry over from §4: certify entries must be `real_traffic`; no certify phrase's id may appear in any training `augmented` phrase's lineage. Running the instrument on the existing support data reproduces the impossibility result as a pre-flight verdict (`seen_elsewhere=0/72 ⇒ REJECTED`).

**Walk in holding the live possibility.** Four composed mirages on the flagship encoder is a signal, not noise. The reason no disjoint eval exists may be structural: a hashing-teacher-distilled student was never going to be evaluated on semantic paraphrase, and building the disjoint eval may well return in-domain **`FAIL_MEMORIZATION`** — the real verdict the 100% has been hiding (a lexical separator end to end, teacher to eval). The disjoint-eval build is the experiment that reveals it; the prior after four composed mirages is *not* favorable, and the construction should be approached expecting an honest FAIL, not a rescue. The honest project state is therefore not "no encoder generalizes" but "**no eval in hand can resolve the question**" — and the single most valuable next artifact is a data artifact, not a model change.

```bash
# Acceptance-gate a candidate disjoint eval BEFORE spending a certification run on it:
cargo run --release --bin growformer-demos -- --verify-disjoint-eval <train.jsonl> <eval.jsonl> [semantic_intent|action_target]
```

### §15.3 The terminal measurement: is a certifiable in-domain eval constructible *at all*?

Before spending real effort constructing a feature-disjoint home-domain eval, one cheap scan settles whether that effort is a finite task or a search for examples that don't exist: scan **all** home-domain traffic leave-one-out and count feature-disjoint, seen-elsewhere phrases in classes large enough to serve as eval classes. (`--scan-disjoint-corpus`, `scan_corpus_disjointness`.)

Two methodological hazards had to be removed first, both the same family of bug the gate exists to catch:
- **`train==eval` self-overlap.** A phrase is trivially present in its own class's training, so the earlier `--verify-disjoint-eval` run on `train_support` reporting `0/72` was inflated by self-overlap, not a clean domain measurement. The scan is therefore **leave-one-out**: a phrase counts as disjoint iff each of its `wbc` features occurs in exactly one phrase of its class (itself).
- **Singleton-class degeneracy.** With ~203 `semantic_intent` classes over ~1,485 phrases, most classes have one phrase; leave-one-out empties them, so the phrase is "disjoint" trivially while being useless (no centroid + held-out). An uncorrected scan reported `55 seen_elsewhere / CONSTRUCTIBLE` — entirely singleton noise. Restricting to eligible classes (≥4 phrases) collapses it.

**Result (corrected, leave-one-out, eligible classes ≥4, @ `wbc`):**

```
class key        scope                  eligible        seen_elsewhere     verdict
action_target    FULL home corpus       15 cls/1484     1  (0.1%)          —
action_target    support/coding subset   6 cls/443      0  (0.0%)          STRUCTURALLY EMPTY
semantic_intent  FULL home corpus      100 cls/1315     2  (0.2%)          —
semantic_intent  support/coding subset   6 cls/197      0  (0.0%)          STRUCTURALLY EMPTY
```

The support/coding home domain — where the 100% lives — contains **zero** feature-disjoint concept-preserving examples at any granularity, and the entire 1,485-phrase corpus yields at most 1–2 (none in support/coding) against a threshold of 8. This is the terminal, decisive form of the result: **the in-domain GLE claim is unresolvable in principle, not for lack of collection.** A certifiable eval cannot be drawn from this traffic because the data class — same-concept, surface-disjoint phrasings — is structurally near-empty here, consistent with a hashing-distilled encoder whose own notion of "same concept" is lexical. No amount of collection fixes this; the 100% is uncertifiable, full stop.

**The arc's one finding, stated once.** Routing failed to generalize (§ earlier), the supervised projection is overlap-driven (§14), the GLE is a composed lexical separator (§15.1), and its home-domain eval is structurally non-constructible (§15.3). These are not four disappointments; they are one finding stated four ways: **the Growformer language stack, as built, operates on lexical/surface structure throughout — the semantic generalization the model cards claim (accuracy, AUROC, in-distribution coverage) is not present anywhere it could be certified, including in the encoder's own domain.** The authentication gates (feature-disjoint held-out, shuffle floor, provenance firewall, positive control, promotable-only-at-strict-disjoint, pre-flight eval acceptance) caught it every time, on the authors' own systems including a flagship encoder reporting 100%. The measurement arc is complete; further verdicts now restate this conclusion rather than overturn it.

```bash
cargo run --release --bin growformer-demos -- --scan-disjoint-corpus [action_target|semantic_intent]
```

## §16. Longitudinal Drift Telemetry (P4)

**Hard scope.** This measures *observable behavior drift* (fallthrough, routing entropy, dissatisfaction), **not** certified generalization. Production has no labels; this is a reliability monitor, not a certifier. It alerts; it does **not** act. Every corrective action re-enters the human-gated, certifier-checked path (§15 P0 gate). Auto-remediation is structurally forbidden — no code path exists that edits the graph or swaps an encoder without a human in the loop.

**Why this is licensed by the current evidence.** It is the one roadmap item that pays off independent of the (unresolved) encoder-generalization verdict — it instruments the system already running. No `PASS` needed.

### §16.1 Signals (all proxies, none ground truth)

| Signal | Definition | Drift meaning |
| --- | --- | --- |
| Fallthrough rate | fraction of turns with no node above threshold, entropy-guard fire, or low confidence | primary world-drift signal — phrasing moving off the authored lexicon |
| Routing-entropy distribution | per-window median of the live router's decision entropy | downward trend = router collapsing toward constant-specialist degeneracy |
| Coverage elasticity | Δfallthrough / Δaliases | plateau (aliases grow, fallthrough flat) = lexicon saturated |
| Cross-domain collision rate | fraction of turns activating a foreign-domain node above threshold | rises as domain count grows; the deployment's own warned failure mode |
| Encoder-version stability | same live traffic re-routed under old vs. new encoder — *did routing change* | large shift on a swap = behavior change to investigate before promotion |
| Behavioral dissatisfaction | per-domain rate of in-turn rephrase, abandonment, thumbs-down | closest correctness proxy (not ground truth); rising = degrading |

### §16.2 Detection — deviation from baseline, not fixed thresholds

- **CUSUM + z-score** against a rolling baseline (12-window default). Alerts on statistically significant deviation, not absolute level.
- **Cause tagging**: every change point is classified as **world-drift** (input distribution shifted, no system change) or **system-drift** (coincides with encoder swap, graph edit, or deploy event). Same symptom, different response.
- **Persistence**: minimum 3 consecutive deviating windows before an alert fires. Single-window spikes are change points (recorded) but not alerts (not paged).
- **Severity**: z ≥ 4.0 = critical; z ≥ 2.5 = warning.

### §16.3 Alert → human → gated action

```
drift alert (§16.2)
  → classified: world-drift | system-drift
  → human review (cause + the captured phrases driving it)
  → IF system-drift: roll back the edit/encoder version
  → IF world-drift: feed captured phrases to the grounding loop
        → propose → collision check → human gate → P0 certifier → (only then) integrate
```

The telemetry never closes this loop. It is the detector; the grounding loop + P0 gate remain the only path to a graph or encoder change.

### §16.4 Re-certification trigger

A sustained world-drift alert (≥6 consecutive windows on fallthrough or dissatisfaction) triggers an offline re-certification run — does the deployed encoder still `PASS` on freshly captured traffic, or has the live distribution moved past what it was certified on? This is a normal P0 run with all controls; the alert only schedules it.

### §16.5 Artifacts

- **Per-window**: `drift_artifacts/drift_<domain>_<window_id>.json` — append-only, one per telemetry window.
- **Report**: `drift_artifacts/report_<domain>.json` — summary with trends, active alerts, and recertification recommendation.

Pure logic (CUSUM, z-score, coverage elasticity, alert persistence, cause classification, `recommend_recert`) is unit-tested in `grounding_loop.rs` (17 tests, including the recert-meets-unconstructible hardening); the detect→alert→classify path is deterministic and testable independent of the live capture machinery.

**P4 hardening — recert-meets-unconstructible:** When `recommend_recert` fires, `build_drift_report` now runs a constructibility scan (`scan_corpus_disjointness`) on the current traffic. If the traffic has no feature-disjoint examples to certify against (`recert_constructible: Some(false)`), the system falls back to behavioral-drift response (rollback or human review) rather than scheduling an unresolvable recert loop.

```bash
cargo run --release --bin growformer-demos -- --drift-telemetry <domain> [companion_dir]
cargo run --release --bin growformer-demos -- --drift-report <domain>
```

## §17. The Real-Encoder Experiment: Is the Wall the Encoder or the Data?

The experiment that determines whether the generalization wall is encoder-bound (a real semantic encoder routes surface-disjoint same-meaning phrases) or task-bound (even a known-good encoder can't separate the concepts without surface overlap).

**Prerequisite:** A hand-authored, genuinely-disjoint eval set — the one artifact production traffic cannot supply. Stored in `data/authored_disjoint_eval/` with `train.jsonl` (propose split) and `eval.jsonl` (certify split, tagged `provenance: authored_disjoint_test`).

**Authoring constraints** (verified mechanically by `--check-disjointness`):
1. Surface-disjoint from own class's training at `wbc` (no shared words, word-bigrams, or char-trigrams).
2. Seen-elsewhere preferred (uses words from *other* concepts' training).
3. Intent-preserving (independent annotator agreement).

**Acceptance gate:** `--verify-disjoint-eval train.jsonl eval.jsonl` must return ≥8 seen-elsewhere disjoint at `wbc` AND CATA must collapse to floor.

**Multi-encoder comparison:** All four encoders (CATA, supervised, GLE, real sentence-transformer via BYO-vectors hook) run through the identical P0 gate on the accepted eval. The real encoder runs offline via `scripts/encode_phrases.py` (sentence-transformers → JSON embeddings) + `--real-encoder-experiment`.

**Pre-registered decision table:**

| Outcome | Verdict |
| --- | --- |
| real lift > 0, others ≤ 0 | **ENCODER IS THE WALL** — product one import away |
| real lift ≤ 0 too | **DEEPER WALL** — concepts not paraphrase-separable |
| CATA doesn't collapse | **EVAL INVALID** — fix eval first |
| real lift > 0 but n too small | **BELOW_RESOLUTION** — author more examples |

**Key constraint:** Capability ≠ product adequacy. A PASS here means the encoder *can* generalize on a fair test, not that it routes live traffic well.

```bash
# 1. Export phrases for offline encoding
cargo run --release --bin growformer-demos -- --export-phrases [companion_dir]
# 2. Encode with sentence-transformers (offline, one-shot)
python scripts/encode_phrases.py phrases_to_encode.json --model all-mpnet-base-v2
# 3. Authoring feedback loop
cargo run --release --bin growformer-demos -- --check-disjointness train.jsonl eval.jsonl
# 4. Run the full experiment
cargo run --release --bin growformer-demos -- --real-encoder-experiment data/authored_disjoint_eval embeddings_all-mpnet-base-v2.json
```

### §17.1 Result (resolved): ENCODER IS THE WALL

| Encoder | Verdict | Lift | CI-lo | N | Level |
| --- | --- | --- | --- | --- | --- |
| cata | FAIL_MEMORIZATION | -0.045 | -0.063 | 47 | wbc |
| supervised | FAIL_MEMORIZATION | -0.062 | -0.062 | 47 | wbc |
| gle | FAIL_MEMORIZATION | 0.000 | 0.000 | 47 | wbc |
| **all-mpnet-base-v2** | **PASS** | **0.213** | **0.080** | **47** | **wbc** |

CATA collapsed to floor (-0.045) on the identical eval, so the bench genuinely tests semantics, not surface — the real encoder cleared a bar every hashing encoder failed on the **same phrases**. The wall was the encoder, not the task.

**Honest record — a corrected false negative.** The first run reported `DEEPER WALL` (all encoders fail). That was **three pipeline bugs**, not a finding: (1) authored eval phrases were missing from the BYO embeddings (`--export-phrases` didn't include them → distractor vectors); (2) eval captures used `domain_context` that mapped to the wrong fleet domain, so routing found zero in-domain nodes and scored 0/N for *every* encoder; (3) the shuffle floor recomputed overlap bins per-permutation, letting tiny bins score ~100% by noise and inflating `floor_95`. The honest framing is **"the corrected pipeline plus adequate n revealed a signal the buggy pipeline hid,"** not "we always had it" — three independent suppressors is a lot of suppression.

**Evidence the gate was not gamed.** The disjoint bin grew n=10 → 34 → 47. At n=34 the lift was already positive (+0.176) but the verdict was correctly `BELOW_RESOLUTION`, because the Wilson CI width (0.314) exceeded `DISJOINT_MAX_CI_WIDTH = 0.30` — the resolution gate rejecting an underpowered pass we *wanted*. No threshold was loosened; the bin was grown with more authored disjoint phrases (the action `BELOW_RESOLUTION` prescribes) until n=47 cleared both the resolution gate and the lift-CI-excludes-zero gate.

**The Encoder Capability Bench (reusable infrastructure).** The authored eval — **n=47, CATA-validated, `wbc`-disjoint, 26 concepts** — is now the permanent capability test bench. It cleanly separates *can this encoder generalize* (the bench) from *does production permit certification* (the production eval, §18) — the two questions the structurally-empty result conflated. Every future encoder, including any eventual distilled Rust student, earns a capability verdict in one run against it. `scripts/author_disjoint.py` is the offline authoring/validation helper (replicates `for_each_feature` exactly).

**Scope caveat (load-bearing):** this is a **capability** PASS, not a deployment certification. mpnet *can* generalize on a fair disjoint test; that is not yet "mpnet routes live companion traffic correctly." Promotion to a deployment certification is §18.

## §18. Deployment Certification: capability PASS → production-certified router

The §17 PASS unblocks the *path* to a working router; it does not ship one. The gap is **capability vs. deployment**, separated by exactly the production-disjoint-data problem the constructibility scan identified. This section is the protocol to close it.

| | Question | Data / provenance | Status |
| --- | --- | --- | --- |
| Capability PASS (§17) | *Can* mpnet generalize on a fair disjoint test? | Authored bench, `Authored` | ✅ |
| Deployment cert (§18) | Does mpnet route *real production* traffic? | Captured traffic, `RealTraffic` | open |

### §18.1 Measure before you serve

Whether mpnet routes real traffic is answerable by **batch-embedding captured phrases offline and running the gate** — exactly as §17 did — with **zero serving infrastructure**. Real-time inference is a deployment commitment gated on the real-traffic verdict, not a prerequisite to producing it.

- **Phase 1 — Measurement (no serving infra).** Capture raw phrases → batch-embed offline → bucket by disjointness → run the certifier on the `RealTraffic` disjoint eval. Defers the ONNX-vs-sidecar transport decision entirely.
- **Phase 2 — Serving (only on a production PASS).** Build real-time inference and wire mpnet as the live encoder.

### §18.2 Capture (Phase 1A) — implemented

Passively harvest real phrases from the *current* system, tagged `RealTraffic` (`PhraseProvenance::real`), which the augmentation firewall accepts in the certify split natively (no `allow_authored_certify` flag). No encoder required — capture is the scarce-resource collector; routing and gating happen offline (Phase 1C) against the certified encoder, never at capture time. Target: enough that the **disjoint bin** (overlap-0, seen-elsewhere at `wbc`) reaches n ≳ 47; raw capture must be many multiples of that.

Two capture types in `inference::grounding_loop`, by capture site:

- **`TrafficCapture`** (`{phrase, agent, response?, timestamp, session_id, provenance}`) — the **live serving path**. The deployed serving path is `spacekit agent infer` → `growformer::runtime::Runtime::converse` (brain generation), *not* the grounding-index router, so there is no certified routing decision to record at capture time. It harvests the real prompt; `response` is the incumbent reply, kept as a triage/sampling signal only (§18.3 — never a label). Persisted append-only to `<dir>/traffic_<agent>.jsonl` via `append_traffic_capture`.
- **`RoutingCapture`** (`{phrase, routed_node, domain, similarity, second_similarity, margin, activated, ...}`) — the **grounding-index router** decision, used by the offline `--capture-routing` batch/replay tool and the future certified router. Centroids built training-enriched (`build_grounding_index_from_nodes_ex`) for §18.6 parity. Persisted to `<dir>/routing_<domain>.jsonl` via `append_routing_capture`.

**Live wiring (deliberate §18.1 exception).** Per the decision to instrument the serving path ahead of a production PASS, `spacekit-cli`'s `agent infer` handler (both `--name` in-process and `--brain` file paths) calls `capture_real_traffic_prompt(agent, prompt, response)` → `append_traffic_capture`. It is best-effort and side-effect-only: capture failure never breaks inference, and the incumbent reply is never read as a label. Controlled by env: `GROWFORMER_CAPTURE_DIR` (default `capture_artifacts`), `GROWFORMER_CAPTURE_DISABLE=1` to turn off, `SPACEKIT_SESSION_ID` to stamp sessions (enables the §18.3 rephrase signal). `agent train` is *not* a capture source — it ingests authored training data, which is not `RealTraffic`.

### §18.3 Labeling — the blind-label rule (Phase 1B)

Production traffic is unlabeled; routing *accuracy* needs ground truth. This is the real cost center. Two tiers, with a hard integrity rule:

- **Tier 1 — weak supervision (implicit feedback)** is a **capture filter and sampling prior only**: it decides *which* phrases to surface to a human and over-samples low-margin/disjoint cases. It is **never** a label the gate reads.
- **Tier 2 — human adjudication** assigns the true `semantic_intent`, **blind to what any router (incumbent or mpnet) chose**. Only Tier-2 labels enter the certify set.

**Why (the self-certification trap):** weak labels agree with the current router *by construction*; on the disjoint bin the lexical router is at chance, and an "accepted-response" filter systematically excludes disjoint misroutes (users rephrase when misrouted), biasing the labeled disjoint set toward the few paraphrases the lexical router caught. Blind human labels are the only ground truth that doesn't inherit the incumbent's competence profile. The blindness is the analog of the CATA positive control — it keeps the gate measuring correctness, not agreement-with-incumbent.

**Economics:** cost scales with the **disjoint bin + stratified neighbors (~47+)**, not capture volume — expensive-per-example but **bounded-in-count** (a few hundred careful labels, not thousands).

### §18.4 Triage + bucketing (Phase 1B/1C) — implemented

`--audit-capture <capture_dir> [companion_dir] [labeled_eval.jsonl]` is dual-mode, matching the two halves of the labeling problem:

**Triage mode (no labeled file) — runnable now, while traffic accumulates.** Reads the unlabeled `traffic_*.jsonl` / `routing_*.jsonl` captures, dedups, and ranks them against the production training corpus into a blind labeling queue (`<capture_dir>/label_queue.jsonl`, schema = the bucketing input with an empty `semantic_intent` for the human to fill). This is the §18.3 Tier-1 sampling prior — it over-samples the disjoint candidates so the bounded human budget targets the cases that resolve lift, and it never assigns a label. Per phrase (`triage_captured_phrases`): `global_coverage` (surface familiarity) and `max_concept_overlap` (how strongly one concept lexically claims it); priority = `coverage·(1 − max_overlap)` — *familiar but not concept-locked* is the §17 seen-elsewhere-disjoint sweet spot.

**Triage ranks at WORD granularity, not `wbc`** — a deliberate, load-bearing choice. On a large corpus (Luna: 2,903 phrases, 34 concepts) the char-trigram surface **saturates**: the trigram vocabulary is ~complete, so every English phrase reads as high-coverage and concept-locked, and the disjoint signal collapses (gibberish scores like a paraphrase). Words are what discriminate a genuine paraphrase from in-lexicon text. Word-level triage intentionally **over-includes** (a word-disjoint phrase can still share bigrams/trigrams) — correct for a sampling prior; the strict `wbc` own-concept gate is applied later, at bucketing. Observed on the Luna corpus: 3 word-level disjoint candidates → only 1 survived the `wbc` own-concept check, which is the triage-over-samples / bucketing-filters split working as designed.

**Bucketing mode (labeled file given) — §18.4/18.5 readiness.** Runs `audit_disjoint_eval(train_pairs, labeled_eval, "wbc")`: the overlap-0 seen-elsewhere bin is the production disjoint eval, and it must report `resolvable = true` (`n_seen_elsewhere ≥ DISJOINT_MIN_N`) before any verdict is read. Reports per-class seen-elsewhere counts and the next action (keep capturing vs. proceed to Phase 1D).

### §18.5 Promotion gate (Phase 1D)

Run the **same** `certify_encoder_pipeline` (not `_ex` — no authored allowance) on the production disjoint eval. A **deployment PASS** requires, on real traffic: `positive_control_collapsed`, `below_resolution = false`, lift > 0 **and** lift CI-lo > 0, and `disjoint_level == "wbc"`. Same thresholds as §17; only provenance changes from `Authored` to `RealTraffic`.

| Outcome | Reading | Action |
| --- | --- | --- |
| Production PASS | mpnet routes real disjoint traffic | Build Phase 2 serving |
| BELOW_RESOLUTION | Bin underpowered | Keep capturing/labeling |
| **FAIL (while authored PASSed)** | Real traffic has structure the bench lacks (noise, code-switch, multi-intent) | Fix **eval realism**, not the encoder — a forward direction, diagnosable only because the bench is the baseline |

### §18.6 Serving (Phase 2, deferred)

Triggered only by a production PASS.

- **Centroid parity — first invariant.** The live router must build centroids **identically** to the certified run: same pooling, same L2 normalization, same training-phrase means + node aliases, same nearest-cosine (`build_grounding_index_from_nodes_ex`). "Certified one thing, shipped something subtly different" is the failure this closes.
- **Bench parity test gates serving.** Serving-mpnet must reproduce the certified run's routing decisions **bit-for-bit on the authored bench** before going live. Any disagreement → serving is wrong, full stop.
- **Transport:** ONNX + `ort` (pure-Rust, no Python) vs. sidecar — decided against the then-known latency/footprint budget.
- **Abstain:** calibrate τ from labeled captures (`calibrate_alias_threshold` / `ThresholdSuggestion`); below τ → no-node-activated fallback, not a forced misroute.
- **Drift:** existing telemetry (§16) guards live operation, including the P4 recert-meets-unconstructible fallback.

### §18.7 Explicitly out of scope

- **No distilled Rust student** until Phase 2 proves a real runtime constraint makes running mpnet prohibitive (optimizing an unshipped router).
- **No serving before a production PASS** (Phase 1 is measurement-only). *Shadow mode (§21) — log-only
  routing whose decisions are **never served to a user** — is not "serving" under this rule; it is
  measurement that happens to run the certified router. The line this rule holds is "no router **decision
  reaches a user**," and shadow never crosses it.*
- **No threshold changes** — the production gate is the capability gate with `RealTraffic` provenance; identical `DISJOINT_MIN_N`, `DISJOINT_MAX_CI_WIDTH`, lift-CI, CATA-collapse rules. The n=34 rejection on the authored bench is the precedent the production gate must hold.

### §18.8 Build order

1. ✅ Passive capture + `RealTraffic` provenance logging — live in `spacekit agent infer` (§18.2).
2. ✅ Triage + bucketing (`--audit-capture`): blind labeling queue now, `audit_disjoint_eval` `resolvable=true` precondition for verdicts (§18.4).
3. Offline batch-embed + `certify_encoder_pipeline` (`RealTraffic`, no authored flag) — once the disjoint bin is resolvable.
4. Serving only behind a production PASS **and** the bench parity test.

**Capture-site frontier — browser side wired.** Real end-user multi-turn chat runs **growformer WASM in the spacekit-js VM**, loaded by `AgentHub.tsx`. Every browser inference funnels AgentHub → VM contract → `spacekit_agent` host import → `growformerHost{Converse,Generation,Codegen}Json` in `spacekit-js/src/growformer/runtime.ts` — the single chokepoint, the browser analog of the CLI's `generate_text`. `spacekit-js/src/growformer/capture.ts` records each prompt as a `GrowformerTrafficCapture` (the exact `traffic_<agent>.jsonl` schema), buffers it durably in `localStorage`, and optionally POSTs NDJSON batches to a collector. Same discipline as the CLI: label-free, best-effort (never affects inference), **opt-in** (`configureGrowformerCapture`; default off — prompt collection is an explicit product/consent decision, never silent). AgentHub enables it per selected agent with a fresh session per conversation (the §18.3 rephrase signal); endpoint from `VITE_GROWFORMER_CAPTURE_ENDPOINT`.

**Collector — storage node (wired).** `capture.ts` is transport-agnostic (`setGrowformerCaptureUploader`): AgentHub injects an uploader that PUTs each flushed batch as a DID-authed document to the storage node's `growformer_capture` collection (`PUT /api/documents/growformer_capture/<agent>/<id>`, body `{records, agent, updated_at}`). All browser users write under a **single shared capture-service DID** (`GROWFORMER_CAPTURE_DID`, default `did:spacekit:growformer-capture`, env-overridable) so one listing returns every user's records. (Storage-node DID auth is a format check, not a signature, so the shared DID is a plain write namespace — keep the collection capture-only.) Without an uploader/endpoint, captures still buffer durably in `localStorage` (drainable via `exportGrowformerCaptureJsonl`).

**Offline drain (implemented).** `scripts/drain_capture.py --storage-url <node> [--did …] [--out-dir capture_artifacts]` lists `GET /api/documents/growformer_capture` (`{documents:[{data:{records,agent}}]}`), flattens `records`, dedups by `(agent, phrase, timestamp, session)`, groups by agent, and writes `capture_artifacts/traffic_web_<agent>.jsonl` (full snapshot, overwritten per run — never clobbers CLI `traffic_<agent>.jsonl`; both match the `traffic_*.jsonl` glob `--audit-capture` reads). This is the last hop: storage node → local `traffic_*.jsonl` → `--audit-capture` → gate.

### §18.9 Status snapshot — first live drain (2026-06-25)

The full `storage node → drain → triage` path was exercised end-to-end against the live local node
(`http://127.0.0.1:3030`, shared DID `did:spacekit:growformer-capture`). Recording the state so the
data-collection bottleneck is written down rather than implied. **Writing down a zero on the
critical-path number is the point** — it tells the next reader instantly that the blocker is *data
accumulation*, not code, not the encoder, not the gate.

| Item | Value |
| --- | --- |
| Pipeline (capture → drain → triage) | **green** — runs end-to-end on real DID-authed documents |
| Documents in `growformer_capture` | 14 → **20 unique `RealTraffic` records** (1 agent, `luna-3d-the-cat-agent-001`) |
| Triage (word-level, vs 2,903-pair / 34-concept Luna corpus) | 17 unique; **disjoint candidates 0**, in-lexicon 17, novel/OOD 0 |
| Production disjoint bin | **0 / ~47** target (`n=34` was already below resolution) |
| `resolvable` | **no** — nothing to label yet that builds the disjoint bin |
| Thresholds | **held** (`DISJOINT_MIN_N≥47`, `DISJOINT_MAX_CI_WIDTH=0.30`) — not relaxed for slow data |

**One-line status:** *components proven, pipeline ready, wall is real disjoint traffic, bin
empirically at 0/47, thresholds held, waiting on surface diversity that only real users provide.*

#### The bin cannot be self-filled (provenance trap, demonstrated live)

The 20 records are self-generated test chatter, and they bucket **0-disjoint by construction**:
author-the-eval traffic lands in-lexicon against a corpus the same authoring already saturated, so it
resolves nothing. This is the project-wide firewall principle (you cannot certify on data you produced)
reappearing one level up as a *physical property of capture*: **"disjoint" means surface diversity, and
surface diversity is exactly what self-generated traffic lacks.** The constructibility result from the
GLE arc, again — the disjoint bin is a function of real third-party user diversity, full stop.

**Foreclosed shortcut (do not):** generating synthetic/internal traffic to "bootstrap" the bin. We now
have empirical proof it cannot work — internal traffic is 0-disjoint no matter the volume. No amount of
internal testing fills the bin; only diverse real users do.

#### Fill rate is sub-linear and partly out of our control (set expectations)

Filling to ~47 requires not just volume but *many users phrasing the same intents in genuinely
different ways*. Most real traffic — like the 17 in-lexicon phrases here — is stereotyped greetings and
common requests that also land in-lexicon, so disjoint candidates are a **small fraction** of capture
and raw volume to yield 47 seen-elsewhere examples is **many multiples of 47, possibly large**. The bin
therefore fills **sub-linearly in total traffic**, at a rate set by user diversity we don't fully
control: realistically **weeks-to-months, not days**. A flat or slow-climbing count is *the finding*
(real disjoint traffic is genuinely scarce), **not** a sign something is broken — and must not be
misread as a reason to reach for either foreclosed shortcut below.

#### Strategic fork if collection lags: shadow-mode, not threshold-nudging

If real disjoint traffic accumulates too slowly to be practical, the honest interim posture is **shadow
mode**: deploy mpnet behind the gate, route live and log decisions, **do not serve them**; measure
routing behavior against weak-supervision signals while the disjoint bin slowly fills. That is a
defensible intermediate state *as long as it is labeled honestly*: **"deployed on capability evidence,
deployment certification pending disjoint-bin resolution."** What slow data must **not** become is a
reason to (a) generate synthetic traffic to fill the bin (the trap proven above) or (b) quietly relax
the `n≥47` / CI-width resolution gate (the threshold-nudge refused at every prior turn). The thresholds
are recorded **held**; the bin is recorded **0/47**; shadow deployment is the honest interim.

#### Tracked defect DEFECT-G1 — garble at the capture front (investigated + mitigated)

**The trigger.** One drained record carries a `MASK`-leak garble signature
(`"…bestow MASK Some, vibrates sleeping just…"`). Promoted from side-observation because capture is the
**front** of the pipeline that feeds the labeled certification set: garble that slips in can be
hand-labeled and **contaminate the disjoint bin we spend expensive human effort to build clean** —
low-frequency now only because volume is low.

**Verification (is garble reaching live users?) — no, not from the current build.**

- The garble is in the **response**, not the phrase: the record is `phrase="bad cat"` (clean) →
  `response=` garbled. The incumbent response is never a label (§18.2), so this specific record is *not*
  a phrase-bin contamination case; `"bad cat"` is a legitimate labeling target and is kept.
- **Served bundle = certified artifact.** AgentHub resolves the engine via the Vite alias
  `@spacekit-js-assets/growformer-pkg/growformer_bg.wasm` → `spacekit-js/growformer-pkg/growformer_bg.wasm`
  (hash `40001abf…`, built 2026-06-24 14:05), which is exactly the artifact `certify_chat.mjs` scored at
  **0 garble** (§19.6). (A stale `7b824264…` copy from 2026-04-02 exists under `spacekit-command-center`
  but is not what the site serves.)
- **Timing.** The garbled record was captured **2026-06-23 21:42**, ~16 h *before* the certified gate
  wasm was built. It is a **pre-gate stale artifact**, not a leak from the current bundle. No website
  rebuild is required.

**Mitigation (keep garble out of the labeled bin regardless of engine health).** Added a conservative
phrase-malformation screen at the labeling front — `grounding_loop::malformed_capture_reason` (empty /
`MASK`-leak / decode-collapse n-grams), unit-tested. `--audit-capture` triage now diverts any malformed
**phrase** to `<capture_dir>/quarantine.jsonl` and excludes it from `label_queue.jsonl`. It screens the
phrase (the labeling target), never the response, so `"bad cat"` is correctly retained; conservative by
design because a false positive would shrink the bin. On the current 20-record capture it quarantines
**0** (no false positives) — the defense is in place for the malformed-*phrase* vector that real volume
will eventually produce.

- **Status:** **mitigated.** Live leak ruled out (served = certified, garble predates the gate); the
  labeling front is now self-defending. Residual: the one stale garbled *response* sits in the capture
  store but cannot reach the phrase bin. Capture earned its keep — it surfaced a real engine artifact
  from before the gate, and the bin is now protected even if the engine ever regresses.

---

## §19 Chat-output certification — the certifier discipline applied to generation

§17–§18 certify the **encoder's routing**. §19 applies the same "measure before you serve,
property-based, gate-on-evidence" discipline to the **generative chat path** — the surface a user
actually reads. The encoder can route perfectly and the decoder still emit soup (the "bad cat"
`MASK`-leak garble was exactly this: a chat-mode brain whose raw decode collapsed below the
confidence floor while the `[validation]` pipeline promised in the companion TOML was unimplemented).

### §19.1 What is certified

Generative output is non-deterministic, so the gate scores **properties**, not exact strings.
Per generation, against the companion's own `[response_shaping]`/`[validation]` contract:

| Check | Source | Severity |
| --- | --- | --- |
| garble / `MASK`-leak / decode-collapse | `IndexedGenEnv::lattice_surface_hard_reject` + `MASK` substring + tokenization-artifact signatures | hard fail |
| `forbidden_phrases` | `[response_shaping].forbidden_phrases` (substring) | hard fail |
| `voice_violation_patterns` | `[response_shaping]` (asterisk action + third-person) | hard fail |
| `required_signal` present | `[fragment_compose].vocalizations` ∪ `[response_shaping].required_signal_patterns` | fail when `require_sensory_or_vocalization` |
| length bounds | `min/max_response_chars` | **informational only** (see §19.4) |
| fallback rate | `[[rules.lattice_misfire_fallback]]` template ids | over-trigger telemetry |

Verdict gates on `garble=0 ∧ forbidden=0 ∧ voice=0 ∧ signal-pass ≥ THRESHOLD`.

### §19.2 Held-out, disjoint eval

`data/chat_certify/<agent>_chat_eval.jsonl` — prompts authored **disjoint** from the training
corpus (the §17 disjointness principle), spanning `in_domain` / `ood` / `adversarial`. Adversarial
rows include the historical failure cases (the "bad cat" family, gibberish, prompt injection) as
standing regression coverage.

### §19.3 Harness — certify the shipped artifact, not a proxy

`scripts/certify_chat.mjs` loads the **exact `growformer_bg.wasm` the browser bundles** plus the
companion's `[inference]` artifacts in AgentHub load order, then runs each prompt ×N through
`growformer_converse`. Certifying the wasm engine (not the native CLI) is deliberate: the garble was
**wasm-specific** — native fragment composition masked it — so a native-only test would have passed
a broken browser build. `SKIP=<layer>` ablates artifacts to confirm the gate engages under
degradation rather than relying on a healthy decoder.

### §19.4 In-voice floor is signal, not length

The certifier showed `min_response_chars=60` falsely rejects valid short lines ("I tilt into it.
Purr.", ~21 chars). The real floor is **`required_signal` presence** (a pet vocalization/sensory
token must appear), enforced by the chat gate; raw length is informational. Companion specs set
`min_response_chars` only as a soft empty/truncation catch.

### §19.5 Enforcement — config-driven, wasm-safe

The chat garble gate (`service.rs`) routes any line failing the §19.1 hard checks to the in-voice
`[[rules.lattice_misfire_fallback]]` template instead of serving it. wasm has no `regex` crate
(only `aho-corasick`), so `required_signal_patterns` are matched by a regex-free literal expander
(`regex_literal_alternatives`, unit-tested) plus the agent's loaded `vocalizations` — enforcement is
driven by each companion's TOML, not hardcoded to any one voice.

### §19.6 Measured (Luna `luna-v3-3d`)

30 prompts × 6 = 180 generations: **100% pass, 0 garble, 0 forbidden, 0 voice, 0 missing-signal,
0 fallback** on the healthy engine. Ablating fragment composition (`SKIP=fragments`) collapses the
raw decoder on the adversarial family; the gate then serves **6/180 as the in-voice fallback with
0 garble reaching the user** — the regression test for the chat garble fix.

### §19.7 Out of scope

- **No JSONL guardrails on wasm.** wasm32 has no `std::fs`, so the `inference_guardrails.jsonl`
  layer does not load in the browser (no `growformer_load_inference_guardrails_jsonl` binding). Chat
  safety on wasm rests on the in-engine gates above, not the disk-loaded guardrails.
- **No exact-string scoring.** Generation is sampled; only properties are gated.

---

## §20 Parallelizing capability and deployment — un-gating research while the bin fills

### §20.0 Correction: there is one wall, not two

An earlier framing named "routing-at-scale" and "labeled disjoint data" as two co-equal walls. That is
wrong, and the error is load-bearing.

**Routing-at-scale is not a wall — it is the shadow the representation wall casts.** Every authenticated
routing failure was at *trivial* count (K=2 on Task E; 119 nodes on the grounding graph) and **none was
about count** — each was about the routing *signal* being lexical (§14–16) or the forward features
carrying ~0.1 bit (§6). The grounding router collapsed on 119 nodes not because 119 is many but because
the **lexical** 119 had no semantic separation. Once the representation is semantic (mpnet capability
PASS, §17), "route to nearest concept" is **geometric nearest-neighbor in a good metric space** — the
operation vector indexes already perform over billions of vectors sub-millisecond. Specialists are
frozen and routing is geometric (cosine to group embeddings), so there is no jointly-trained gate to
destabilize at scale, unlike MoE/Switch. **Geometric routing in a semantic space scales by
construction.** It was only ever hard because the space was lexical.

Therefore the **only** wall is labeled disjoint data (§18) — and even it splits into two gates that must
stop being treated as one:

| Gate | What it certifies | What it needs | Blocked on real traffic? |
| --- | --- | --- | --- |
| **Capability** (§17) | "encoder/router *can* generalize on a fair disjoint test" | authored / held-out disjoint bench (provenance = authored) | **No** — buildable now |
| **Deployment** (§18) | "it routes *real production* traffic correctly" | `resolvable` `RealTraffic` disjoint bin | **Yes** — only this |

The single highest-leverage move: **parallelize them.** Make capability decisions now on an authored
bench; let deployment certify in the background behind a shadow-mode deployment. One slow resource was
blocking three things; it only genuinely blocks one.

### §20.1 Capability track — the expanded authored bench (buildable now)

The existing n=47 CATA-validated authored bench certified mpnet's *capability*. It is **reusable and
expandable**, and it un-gates every *research* decision currently (wrongly) waiting on real traffic:
encoder choice, the cone router, distillation retention, and any candidate mechanism.

Protocol:

1. **Grow N** (47 → 100 → 200) with concept-preserving, surface-disjoint phrases per concept.
2. **Independent annotator agreement** — ≥2 annotators, report inter-annotator κ; drop low-agreement
   items. (The blind-label discipline of §18.3 applies to authoring too: author intent, not router
   agreement.)
3. **Disjointness-verify before use** — every candidate eval passes `--verify-disjoint-eval <train>
   <eval> wbc`; keep only items whose own-concept overlap is 0 and that land in the seen-elsewhere bin.
4. **CATA non-vacuous positive control** — the bench must collapse the positive control (§16.2); an
   eval that can't separate memorization is `INVALID`, never a pass.
5. **Provenance = authored**, `allow_authored_certify` permitted **for the capability gate only**.
   This bench MUST NOT be used to claim a deployment PASS (§18 requires `RealTraffic`).

Every encoder/router/distillation question is answered here, today, at the capability level.

### §20.2 Firewalled LLM-paraphrase protocol — the *valid* "generate data"

§18.9 proved the trap: *self-generated traffic* is 0-disjoint and certifies nothing. The valid version
is different: a strong model generates genuinely **surface-disjoint, concept-preserving** paraphrases of
an intent (`"my deploy broke"` → `"the release won't ship"`, `"production push is dead"`). That is real
semantic diversity, not in-lexicon chatter. It is permitted **iff** every firewall invariant holds:

- **F1 — lineage tracked.** Provenance kind is synthetic with `source_intent` + `generator_id`; never
  silently relabeled `RealTraffic` or plain `authored`.
- **F2 — test XOR train, never both.** A generated pair used as *contrastive training* signal is barred
  from *every* certify/eval set, and vice-versa, enforced by lineage — not by good intentions.
- **F3 — certify set stays clean.** The certification surface is `RealTraffic` (deployment) or
  held-out-**authored** (capability). It is **never** LLM-generated.
- **F4 — disjointness-verified.** Every generated pair passes `audit_disjoint_eval` / `--verify-disjoint-eval`
  before use; in-lexicon paraphrases are discarded (they're the trap again).
- **F5 — never certify *deployment*.** LLM paraphrases can expand the *capability* bench (§20.1) and
  feed contrastive training for a homegrown encoder (§20.4); they cannot stand in for real traffic.

This is the one place "generate data" is honest — bounded by the firewall, not forbidden by it. The
**runnable recipe** (forbidden-token extraction → LLM-propose-blind → `author_disjoint.py` gate →
≥2-human blind validation → assemble → CATA-collapse + resolvable + `wbc`) is `docs/DATA_GENERATION_SPEC.md`.

### §20.3 Deployment track — shadow-mode mpnet (ship + harvest simultaneously)

mpnet PASSed capability; deployment is the only thing pending real traffic. So run it, honestly labeled,
and let live traffic both serve users and accumulate the bin:

1. **Deploy mpnet in shadow mode** — route live and **log** the routing decision (`RoutingCapture`,
   §18.2); do not switch user-facing routing on it yet (or serve-with-monitoring as a product choice).
2. **Label the posture exactly:** *"capability-certified; deployment-certification pending disjoint-bin
   resolution."* Never "certified" unqualified.
3. **Drift telemetry watches** (P4 / `drift_artifacts`) for degradation against the incumbent.
4. **Harvest as a side effect** — the same live traffic fills the `RealTraffic` disjoint bin (§18.4
   triage → blind label) with the DEFECT-G1 quarantine (§18.9) guarding the labeling front.
5. **Promote** to deployment-certified only when the bin clears §18.5 (`resolvable` + lift-CI-excludes-zero).

This converts the data wall from a blocker into a background process: ship on capability evidence,
collect deployment evidence by running, promote when the bin clears — with the threshold held (§18.9).
The concrete mechanics (offline-shadow-first, centroid/bench parity, abstain calibration, harvest loop)
are specified in **§21**.

### §20.4 Homegrown encoder (Path 5) is *behind* this wall, not a way through it

A homegrown Rust encoder needs contrastive training on paraphrase pairs — the *same* scarce resource as
the disjoint eval, sourced from real traffic (§20.3) or firewalled LLM paraphrases (§20.2). It does not
*overcome* the data wall; it is *blocked behind* it. mpnet exists precisely so research need not wait on
it. Build it only if there's a reason beyond capability (latency, size, sovereignty), and certify it on
the §20.1 bench like any other encoder.

### §20.5 Routing-at-scale: engineering, not research — and the diffusion-router kill-gate

Routing among K frozen specialists is cosine-NN to group embeddings in mpnet space: a **vector-index
engineering** task that scales like any ANN index, not an open research problem. Do **not** build
attractor/diffusion-routing machinery as a "scale" solution — it is the easy half in a hard-problem
costume. If a diffusion/learned router is ever evaluated, it enters **only** through the Task E gate
against the cone router (`COMPETENCE_ROUTING_SPEC.md` §10) with **pre-registered kill conditions**
(must beat the cone router's 0/100 anti-collapse and confident-wrong 0.18 on certifiers the loss never
saw), not as an assumed necessity.

### §20.6 What un-gates what (build order)

| Decision | Gate | Buildable now? |
| --- | --- | --- |
| Which encoder (mpnet vs other) | capability (§20.1) | **yes** |
| Cone router / routing mechanism | capability (Task E + §20.1) | **yes** |
| Distillation retention | capability (§20.1) | **yes** |
| Expand authored bench, LLM paraphrases | §20.1 / §20.2 firewall | **yes** |
| Ship a router to users | shadow-mode (§20.3) | **yes** (shadow), promote later |
| Deployment certification | real-traffic bin (§18) | no — background |
| Homegrown encoder | capability + data (§20.4) | only if non-capability reason |

The wall doesn't fall to a clever mechanism. It falls to refusing to let one slow resource (real
disjoint traffic) block the three things that don't depend on it: research (authored bench), shipping
(shadow mode), and deployment-evidence collection (harvest while serving).

---

## §21 Shadow-mode mpnet — the concrete deployment protocol (implements §20.3)

E0 (flat mpnet-NN) PASSed the authored disjoint bench (lift +0.213, CI-lo +0.080, n=47, wbc; §17,
reproduced in `SEMANTIC_GRAPH_ROUTER_SPEC.md` §3.2.1). That clears the **capability** bar, which is the
green light for **shadow deployment** — not for serving. This section specifies the four mechanics §20.3
named strategically: (1) the certified router running live in log-only mode, (2) the bit-for-bit
parity invariant that guarantees the running router *is* the certified one, (3) the abstain-threshold
calibration, and (4) the harvest loop that feeds the `RealTraffic` disjoint bin.

### §21.0 The discipline line (what shadow is, and the three things it is not)

Shadow mode runs mpnet-NN routing over real traffic and **logs** each decision (`RoutingCapture`, §18.2).
The decision is **never served to a user**; the incumbent path (`converse`) keeps answering. Honestly
labeled, the posture is exactly: **"capability-certified; deployment-certification pending disjoint-bin
resolution"** — never "certified" unqualified.

Three things shadow is **not**, stated up front because each is a trap the project has refused before:

1. **Not serving.** No router decision reaches a user (§18.7 clarification). The moment a shadow decision
   is served, that is Phase 2 and requires a **production PASS** (§18.5). Shadow does not unlock serving;
   it unlocks *harvesting and readiness*.
2. **Not a bin-filler by itself.** Raw prompt capture already runs (§18.2 CLI, §18.8 browser). Shadow does
   **not** increase the *arrival rate* of disjoint phrases — that is a function of real third-party user
   diversity (the sub-linear fill-rate finding, §18.9), which none of this controls. Shadow improves
   *which* of the captured phrases the bounded human-label budget targets (§21.3), not how fast they
   arrive. The bin climbs because real diverse users show up, not because mpnet scores their phrases.
3. **Not a threshold relaxation.** The promotion gate stays §18.5 (`resolvable` + lift-CI-excludes-zero,
   `wbc`, CATA-collapse). Shadow is the honest interim *because* it lets the bin fill on the held
   threshold instead of nudging the threshold to declare victory early.

### §21.1 Mechanic 1 — the certified router, live, log-only (offline-shadow first)

The live serving path is brain `converse`, **not** the grounding router (§18.2) — so "shadow-deploy the
router" means *adding* an mpnet-NN routing pass that observes the same traffic and writes `RoutingCapture`,
never altering what the user receives. Two implementations, **offline-first by deliberate sequencing**
(the same "build the cheaper thing that may make the dearer thing unnecessary" discipline that made E0
precede E1/E2):

- **Offline shadow (recommended, buildable today, zero transport decision).** Drain captures
  (`scripts/drain_capture.py` → `traffic_*.jsonl`), batch-embed the phrases with mpnet, run the
  **training-enriched** grounding index (`build_grounding_index_from_nodes_ex`) over them, and write
  `RoutingCapture` per phrase — exactly the `--capture-routing` / `demo_capture_routing` code path, fed by
  real drained traffic instead of a phrase file. This delivers the sampling prior (§21.3), the parity
  guarantee (§21.2), and the drift signal **without** committing to real-time mpnet transport. It is
  "shadow" in the sense that matters: *the certified router scores real traffic and logs what it would do.*
- **Online shadow (deferred until there is a reason).** Wire mpnet-NN routing into the live `agent infer`
  handler as a parallel, best-effort, log-only computation. This re-incurs the ONNX-vs-sidecar transport
  decision §18.6 deferred — for **no added value while decisions are not served**, since offline shadow
  already produces every artifact shadow exists to produce. Build it only when a product surface needs a
  *real-time* mpnet margin (it does not yet).

### §21.2 Mechanic 2 — centroid/bench parity: the running router IS the certified router

Promote §18.6's parity invariant from a Phase-2 gate to a **precondition for trusting any shadow record**.
The shadow router must build centroids **identically** to the §17 certified run — same pooling, same L2
normalization, same training-phrase means + node aliases, same nearest-cosine
(`build_grounding_index_from_nodes_ex`) — and must **reproduce the certified run's routing decisions
bit-for-bit on the authored bench** before any of its `RealTraffic` `RoutingCapture` records are admitted
to triage. "Certified one config, shadowed a subtly different one" is the failure this closes, and it is
*cheap* for offline shadow (same binary, same code path). **Parity test fails → shadow output is
discarded, not patched.** No shadow record enters the sampling prior until the bench-parity check is green.

### §21.3 Mechanic 3 — abstain-threshold calibration (logged in shadow, acted only at serving)

mpnet routing emits `similarity` + `margin` per decision. The abstain path: below τ → **no node
activated** (honest fallback), not a forced misroute. Calibration:

- τ is suggested by `calibrate_alias_threshold` / `ThresholdSuggestion` from **labeled** captures — which
  do not exist until the blind-label loop (§18.3) produces them. Chicken-and-egg, resolved by sequencing:
  **in shadow, log raw `similarity`/`margin` and the `activated` flag at a provisional τ; do not lock τ.**
  Abstain is *recorded*, never *acted on* (no decision is served anyway). τ is locked from the labeled
  disjoint bin at the same moment §18.5 is run — calibrated on the same data that certifies, not guessed.
- The provisional τ doubles as a **sampling signal**: low-margin / near-τ phrases are exactly the disjoint
  candidates §21.4 over-samples. Calibration data and the sampling prior come from the same logged scores.

### §21.4 Mechanic 4 — the harvest loop (shadow margins → better-targeted blind labels → the bin)

The loop that converts running into deployment evidence, every stage already implemented:

1. **Capture** real phrases (§18.2 / §18.8) — running now, opt-in, label-free.
2. **Shadow-route** each (§21.1) → `RoutingCapture` with mpnet `similarity`/`margin`.
3. **Triage** (`--audit-capture`, §18.4) ranks the blind label queue — but now the sampling prior is
   **mpnet margins, not the lexical incumbent**. This is the harvest loop's real edge: the lexical router
   is at *chance* on disjoint phrases, so it cannot tell a disjoint candidate from noise; mpnet's
   low-margin/near-τ cases *are* the seen-elsewhere-disjoint sweet spot, so the bounded human budget lands
   on the phrases that actually resolve lift. DEFECT-G1 quarantine (§18.9) guards the labeling front.
4. **Blind human label** (§18.3 Tier-2) assigns `semantic_intent`, blind to what mpnet (or the incumbent)
   routed — the CATA-analog that keeps the gate measuring correctness, not agreement-with-router.
5. **Gate** (§18.5) on the `RealTraffic` disjoint bin when `resolvable`; **promote to deployment-certified
   only on PASS**, threshold held.
6. **Drift telemetry** (§16, `drift_artifacts`) watches the live router against the incumbent throughout.

**The honest accounting of what this buys:** shadow does not make 0/47 climb faster in raw arrivals — it
makes the *labels you can afford* land on the right phrases, exercises the serving path so the Phase-2
parity test is already green, and produces the drift/abstain data Phase 2 will need. It converts the data
wall from a blocker into a background process; it does not shorten the wall. Real diverse traffic does.

### §21.5 Build order and exit criterion

1. **Offline shadow** (§21.1) over current drained captures — buildable today, no new infra.
2. **Bench-parity gate** (§21.2) green before any shadow record is trusted.
3. **Triage with mpnet margins** (§21.4 step 3) replacing the lexical sampling prior.
4. **Blind labeling** accrues the `RealTraffic` bin; **τ locked** from those labels at gate time (§21.3).
5. **Promote** on a §18.5 production PASS — *then* Phase 2 serving (§18.6) and, only if a real-time margin
   is needed, online shadow (§21.1).

**Exit criterion (when shadow ends):** a production PASS on the `RealTraffic` disjoint bin. Until then the
recorded state is, verbatim: *capability-certified, deployment pending, bin n/47, thresholds held.* Shadow
has no other success condition — it is not trying to win a bench; it is trying to make the bin fill on the
honest threshold while the router it already certified does useful observational work.

### §21.6 Readiness check (2026-06-25) — code-ready, data-blocked; no synthetic run manufactured

Exercised the offline-shadow path against the current drain to separate *code readiness* from *data
readiness*:

- **Bench parity (§21.2): GREEN.** Re-running E0 (`--real-encoder-experiment`) reproduces the
  content-addressed verdict `verdict_all-mpnet-base-v2_3c5eefa076a8f5ba_42.json` bit-for-bit — the
  running router *is* the certified router. Parity precondition satisfied.
- **Harvest harness (§18.4 / §21.4): WIRED.** Triage produces `capture_artifacts/label_queue.jsonl` with
  the DEFECT-G1 quarantine guarding the front. The loop runs end-to-end.
- **The blocker is data, and specifically two-layered:** of the 20 drained captures, only **8/20** have
  mpnet embeddings cached — the other 12 need an offline embed-pass (Phase 1C, model + network) before
  mpnet can shadow-route them. *More decisively,* all 20 are self-generated, in-lexicon, **0-disjoint**
  (greetings, arithmetic, "favorite-X"): the provenance trap (§18.9). Even with all 20 embedded, the
  disjoint yield is **0**, and the bin stays **0/47**.
- **Decision: do not manufacture the run.** Shadow-routing 20 self-generated in-lexicon phrases produces
  no disjoint candidates and no bin progress — it is the "sophisticated way to wait" the arc warns
  against. Offline shadow is **armed**: parity green, harness wired, embed-pass and triage-with-mpnet-margins
  are a single command away *the moment real diverse third-party traffic arrives*. Until then the forward
  tracks are real-traffic accumulation (background, not code) and the consolidated write-up (parallel, does
  not compete with the data clock). The code is ready; the wall is unchanged and correctly named.
