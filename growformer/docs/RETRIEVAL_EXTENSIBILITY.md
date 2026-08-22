# Retrieval extensibility (phased plan)

This document tracks how Growformer keeps **core code language-neutral** while shipping **domain/locale policy** as data or WASM.

## Phase 0 — Freeze

- **Do not** add new English substring heuristics in `group_gen.rs` for sentiment/crypto retrieval.
- **`sentiment_crypto_rescore.toml` scope:** keep it for **crypto/tape template collisions** (funding, dominance, liquidations, etc.). **Do not** grow **PR / corporate headline matrices** here—each new `when_query_any` laundry list fights OOD forever. Prefer **training gold**, **`causal` / `gfcausal_*`**, and **Layer 0 concept grounding** (see `GROWFORMER_CAUSAL_AI.md` — World grounding) so behavior generalizes.
- Locale-specific tables remain the Phase 2 escape hatch for real multilingual policy, not English headline whack-a-mole.

## Design note — verbatim lattice vs compositional replies

Forced-topic sentiment today returns **one decoded lattice program** (plus rerankers). TOML rescore rules are **narrow collision patches** for the **tape/crypto** lattice; JSONL gold and causal structure teach **what** programs mean. **Garbled decode** (wrong tokens in the witness strip) is a **generation/indexing** quality issue—fixing it with more rescore rows is the wrong layer; improve training, decode, or composition instead.

Longer term, richer answers should **compose** (e.g. witness slots, multi-sentence assembly from several indexed fragments) rather than **spit out a single stored analysis verbatim**. Until that pipeline exists, declarative rules stay bounded, auditable, and easy to locale-split.

## Training “games” vs retrieval patches

Cloze / span-prediction / extra LM objectives are **not** wired into the default Growformer categorical trainer from this repo path. What **is** in-tree: **JSONL gold**, **joint-index / `gfcausal_*` tokens**, **TOML rescore**, **inference heuristics** (e.g. press-headline → `neutral`), and **hard rejects**. Retraining updates the lattice; rules/hints apply at infer without a second train pass.

## Causal index tokens (sentiment)

Training rows may include a `causal` object; the sentiment joint index then contains `__GF_CAUSAL__ gfcausal_t_<type>_c_<connector>` before the witness marker, and optionally a second token `gfcausal_st_<causal_subtype>` (see `dimension::language`, `GROWFORMER_CAUSAL_AI.md`). At inference, `inference::causal_hints` adds matching `gfcausal_t_*` keywords from connector heuristics; subtype tokens are training-time until hints learn the same vocabulary.

## Layer 0 — grounding query expansion (MVP)

- **Table:** `data/inference/grounding_expand.toml` (`version = 1`, `[[rules]]` with `when_all`, optional `when_any` / `unless_any`, `add_keywords`).
- **Loader / API:** `inference::grounding_expand::extend_subject_keywords_with_grounding` — substring bundles on a padded lowercase query; appends keywords for BM25 (deduped).
- **Wiring:** `LatticeShortcutsPlugin::extend_subject_keywords` runs it **after** `causal_hints` (same path as word-split + causal tokens).
- **Intent:** **Sparse relational cues** that will later map to graph edges (e.g. loss + “game” → gambling frame; funding + negative + flat price → shorts squeeze vocabulary)—**not** per-brand or per-headline keyword lists. Those belong in **structured world knowledge + training**, not another curated file.
- **Spec:** `GROWFORMER_CAUSAL_AI.md` (*World grounding*), `docs/WORLD_MODELS.md` §3.1.

## Layer 0 — typed world graph (MVP scaffold)

- **Table:** `data/inference/world_grounding.toml` (`version = 1`, `[[nodes]]` with `id`, optional `aliases`, `[[nodes.edges]]` with `kind`, `target`, `weight`).
- **Loader / API:** `inference::world_grounding::extend_subject_keywords_with_world_graph` — token hits on `id` / `aliases`, bounded BFS (depth 2, cap 18 terms), appends edge `target` strings for BM25 (deduped with prior keywords).
- **Wiring:** `LatticeShortcutsPlugin::extend_subject_keywords` runs it **after** `grounding_expand` (causal → rule bundles → **typed graph**).
- **Intent:** Auditable **concept adjacency** (e.g. `launch` → `product_release`, `stablecoin` → `issuer` / `regulatory_approval`) without growing `group_gen` English. Version and extend the TOML per domain bundle; optional later: disk override, magnitude bands, sentiment-bearing edge metadata.

## Phase 1 — Neutral API (done)

- `inference::retrieval_rescore` defines `RetrievalQueryLexical`, `RetrievalCandidateLexical`, and the `SentimentRetrievalRescoreExtension` trait for future hosts.
- `group_gen` calls `apply_embedded_sentiment_crypto_rescore` with `(program_idx, score)` and a decode closure — no scenario English in the hot path.

## Phase 2 — Data-first multilingual

- Train **separate brains** per locale (`brain.en.bin`, `brain.fr.bin`, …) from JSONL in `data/fintech/locale/<lang>/`, or use one multilingual encoder (future).
- Optional: load `sentiment_crypto_rescore.<locale>.toml` instead of the embedded default when `RetrievalQueryLexical::locale` is set (host responsibility).

## Phase 3 — Declarative rules (done for EN crypto)

- Default table: `data/inference/sentiment_crypto_rescore.toml` (`version = 1`, `locale = "en"`).
- Rule fields: `when_query_all`, `when_query_any`, `unless_query_any`, `unless_query_all`, `when_program_all`, `when_program_any`, `unless_program_any`, `delta`.

## Phase 3b — Lexicon externalized (stopwords, intent names, code markers)

- `data/inference/retrieval_lexicon.toml` — `[locales.en]` holds:
  - `graph_stopwords` (program graph / IDF signatures)
  - `lex_align_stopwords` (forced-topic lexical alignment)
  - `intent_implement` / `intent_explain` (query `intent_action` tokens)
  - `code_markers_bm25`, `code_markers_rust`, `code_markers_python`, plus Python `return` / `->` disambiguators
- Loader: `inference::retrieval_lexicon` (`global()`, `global_for_locale` stub for future `[locales.fr]`).
- `ProgramGraph::build` takes `&RetrievalLexicon`; forced-topic BM25 + lang-hint paths use the same table.
- Add `[locales.fr]`, `[locales.es]`, … with the same keys; then wire `language_channel` → `global_for_locale(Some(...))`.

## Phase 4 — WASM policy module (next)

- Compile the same rule interpreter (or a generated decision tree) to `wasm32-unknown-unknown`.
- Host passes **compact features** (token histogram IDs, topic id, optional embedding bytes); guest returns score deltas.
- Pin **module hash** next to `AGENT_BRAIN` in Spacekit storage.

## Phase 5 — Spacekit contract surface

- **`spacekit-standard-library/agents/spacekit-growformer-sentiment-analysis`**: resolve manifest `{ brain_key, optional_rescore_wasm_hash, locale }` before calling `growformer_generation`.
- **`spacekit-contract-sdk`**: optional host import `retrieval_rescore_apply` (ABI TBD) alongside `growformer_generation`.
- **`spacekit-contract-language`**: document standard agent extension imports in SKCL specs.

## Related paths

- Neurokit: `growformer/data/inference/sentiment_crypto_rescore.toml`, `growformer/data/inference/retrieval_lexicon.toml`, `growformer/src/inference/retrieval_rescore.rs`, `growformer/src/inference/retrieval_lexicon.rs`
- Spacekit: `spacekit-standard-library/agents/`, `spacekit-contract-sdk`, `spacekit-contract-language`
