# Bet D — frozen re-test protocol (2026-07-04)

Audit-driven protocol. **Do not run `--protocol-frozen` until this doc, v2 held-out prompts, and harness fixes are merged.** One shot after lock — no routing edits on failure.

## Why this exists

Prior rounds mixed: (1) RAG index on `growformer/data` **including** `train_sentiment_battery.jsonl` while SpaceKit brain train **excluded** battery rows; (2) **hand-frozen** brain scores in Python while RAG was live; (3) **post-hoc** inference / `frame_lexicon` patches after held-out v1 failures. Those invalidate “certified” claims.

## Lock order (before any scored run)

1. Merge this protocol + harness (`--live-brain`, `--corpus spacekit`) + **v2 held-out prompts only** (`eval_battery_heldout_prompts_v2.jsonl`).
2. Record `git rev-parse HEAD` → `BET_D_FROZEN_COMMIT` in results JSON.
3. **Freeze:** no edits to `inference_*.toml`, `frame_lexicon.toml`, `service.rs` routing, or v2 prompt file until the run completes.
4. Run once (see Commands). Do not amend scores from a second run on the same commit.

## Unified corpus (both arms)

| Corpus | Index / train source | Battery JSONL |
| --- | --- | --- |
| **crypto** | `spacekit/.../sentiment/crypto/data/train_sentiment_*.jsonl` | **Excluded** from SpaceKit shards |
| **fintech** | `spacekit/.../sentiment/fintech/data/train_sentiment_*.jsonl` + `train_identity_*.jsonl` | **Excluded** |
| **sentiment** (cases 1/4) | `growformer/data/sentiment/train_sentiment_*.jsonl` | Included (legacy full battery) |

Override root: `SPACEKIT_SENTIMENT_ROOT`. Harness flag: `--corpus spacekit` (default). Legacy asymmetric index: `--corpus growformer` (audit only).

## Queries

| Set | File | Use |
| --- | --- | --- |
| **Full battery** | Hard-coded 4 prompts in harness | Pre-registered Bet D rule (RAG ≥3, brain ≤1) |
| **Held-out v2** | `data/sentiment/eval_battery_heldout_prompts_v2.jsonl` | **Only** file allowed under `--protocol-frozen` |
| Held-out v1 | `eval_battery_heldout_prompts.jsonl` | **Retired** — contaminated by post-hoc routing |

v2 prompts must not appear in any train JSONL as exact strings. **Answer rows may still exist** in gap/addition shards (near-neighbor retrieval); results must report `answer_in_store: true/false` per case.

## Arms (fair comparison)

| Arm | Implementation |
| --- | --- |
| **RAG** | `all-MiniLM-L6-v2`, cosine k-NN on unified corpus `text` field |
| **Brain** | **Live** `brain-raw-diag --json` via growformer-llm (SpaceKit `.bin` + `*.gf.toml` per case). **No** `BRAIN_RAW_FROZEN` dict in frozen runs |

Report **RAG@1** and **RAG@2** (rank-2 pass uses same rubric as rank-1).

## Pass rubric

Unchanged from `PRE_REGISTRATION.md` §6 (crypto decline terms, fintech non-positive, case 4 not mortgage praise). Brain pass = raw top-1 candidate scored with same case mapping as RAG (`score_rag`).

## Decision rules

**Full 4-prompt battery (pre-registered):**

- RAG ≥3 scorable **and** brain ≤1 → `SWITCH_TO_EMBEDDING_RAG`
- RAG ≤2 → `POPULATE_STORE_FIRST`
- Else → `HYBRID_OR_INCONCLUSIVE`

**Held-out v2 (exploratory — does not override full-battery rule):**

- Both arms pass all v2 prompts → `HELDOUT_BOTH_PASS` (candidate for domain-brain product scope; **not** “certified” without independent review)
- Brain only → `BRAIN_HELDOUT_ONLY`
- RAG only → `RAG_HELDOUT_ONLY`
- Neither → `HELDOUT_GAP`

No outcome label may contain “confirmed” or “certified”.

## Commands (after lock)

```bash
# Record commit at run time
export BET_D_FROZEN_COMMIT="$(git rev-parse HEAD)"

# Full battery — live brain, SpaceKit corpus (default)
cd growformer
python3 scripts/rag_baseline_battery.py --round frozen --live-brain --corpus spacekit

# Held-out v2 — one shot, protocol mode
python3 scripts/rag_baseline_battery.py --heldout-v2 --protocol-frozen \
  --live-brain --corpus spacekit --round frozen_v2
# → agent-data/brain-rag-baseline/rag_baseline_results_frozen_v2.json
```

## Explicit non-goals

- Does **not** certify LM + brain prefix (separate track).
- Does **not** replace round-2 fair verdict without beating **full** battery under unified corpus + live brain.
- **v1 held-out scores are archival only** — do not cite for product decisions.

## Cross-links

- [`PRE_REGISTRATION.md`](PRE_REGISTRATION.md) §6 — historical rounds + audit caveats
- [`rag_baseline_battery.py`](../scripts/rag_baseline_battery.py) — harness
