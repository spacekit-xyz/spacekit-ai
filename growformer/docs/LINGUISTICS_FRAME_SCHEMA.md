# Linguistics brain — frame schema (Phase 0)

Path C roadmap: scenario topics (`etf_delay_bearish`, …) become **frame IDs** in a shared pre-layer; domain brains consume frames instead of growing TOML phrase lists.

## Downstream feature contract

Every domain brain / retrieval pass receives:

| Field | Type | Role |
| --- | --- | --- |
| `primary_frame` | string | Best-matching semantic frame (e.g. `RATE_HIKE`, `ETF_DELAY_BEARISH`) |
| `polarity_bearing` | string | `negative` / `positive` / `neutral` / `mixed` from lexicon + structure |
| `speech_act` | string | `headline_wire` / `first_person_complaint` / `counterfactual` / … |
| `domain_hints` | string[] | Optional routing (`crypto`, `fintech`, `consumer_credit`) |
| `reject_frames` | string[] | Hard-reject lattice programs indexed under wrong frames |

## Seed frames (SpaceKit battery + held-out)

| Frame ID | Absorbs scenario topic | Example cues |
| --- | --- | --- |
| `ETF_DELAY_BEARISH` | `etf_delay_bearish` | ETF + delay/shelved/postpon + BTC sell-off/slid/crash |
| `MORTGAGE_RATE_COMPLAINT` | `mortgage_rate_complaint` | lender + mortgage/APR + raised/lifted + no notice/alert |
| `FEE_COMPLAINT` | `fee_complaint` | custody/account fee + non-negotiable policy |
| `COUNTERFACTUAL_POSITIVE` | (reject) | `would have` + rally/gain — not price-decline bucket |

## Phase 1 (declarative MVP) — **shipped 2026-07-04**

- `data/linguistics/frame_lexicon.toml` — frame CNF rules + reject frames
- `src/inference/frame_lexicon.rs` — compile-time loader; runs **before** domain `headline_lexical_topic` in `sentiment_lexical_topic_key`
- `brain_memory::query_hybrid` — reject-frame guard + full lattice body (not witness gloss only)

## Phase 2 (tiny promoted brain)

Train on frame classification rows only; promote-freeze; never merge with crypto/fintech sentiment lattices.

## Deferred

Phonology, historical linguistics, sociolinguistics tags beyond register — text-only path; add when ASR or locale routing needs them.
