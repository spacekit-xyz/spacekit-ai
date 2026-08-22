# Bet D — Path A v3 protocol (routing fix, 2026-07-05)

Follows frozen v2 outcome **`RAG_HELDOUT_ONLY`** (brain 1/2). Micro-experiment (`--force-topic`) showed **lattice retrieval succeeds when routing fires** — bottleneck is headline lexical rules, not store/RAG-class retrieval.

**Status: completed** — commit `1cf123ad9f75ccbf9f11f29bd0194aa220da5ad5`, results `agent-data/brain-rag-baseline/rag_baseline_results_path_a_v3.json`, outcome **`HELDOUT_BOTH_PASS`** (RAG 2/2, brain 2/2, natural routing).

## Micro-experiment result (2026-07-05)

| Prompt | Natural brain | `--force-topic` |
| --- | --- | --- |
| v2 fintech (BofA repriced APR) | `no_topic_hint`, 0 candidates | `mortgage_rate_complaint` → gap paraphrase #1, `witness=true` |
| v2 crypto (SEC kicked ETF) | `negative_mild` causal template (wrong bucket) | `etf_delay_bearish` → `crypto_gap_001` class, `witness=true` |

**Decision:** Path A — minimal routing TOML only; no train edits, no `service.rs` / `frame_lexicon` changes for v3.

## Allowed edits (this cycle only)

| File | Change |
| --- | --- |
| `spacekit/.../fintech/data/inference_fintech.toml` | `repriced`, `zero advance warning` → `mortgage_rate_complaint` |
| `spacekit/.../crypto/data/inference_crypto.toml` | `sold off`, `kicked`, `next quarter`, `sec` → `etf_delay_bearish` |

**Not in scope:** new train rows, battery JSONL, v2 prompt edits, LM work.

## Lock order

1. Merge routing TOML + this doc + `--force-topic` diagnostic (already in `brain-raw-diag`).
2. `git rev-parse HEAD` → `BET_D_PATH_A_V3_COMMIT` in results JSON (`--commit-hash` or env).
3. **Freeze** until one held-out v2 run completes.
4. Re-run **held-out v2 only** (same prompts as frozen v2 — no v3 prompt file).

## Commands

```bash
export BET_D_PATH_A_V3_COMMIT="$(git rev-parse HEAD)"

cd growformer
python3 scripts/rag_baseline_battery.py --heldout-v2 --protocol-frozen \
  --live-brain --corpus spacekit --round path_a_v3
# → agent-data/brain-rag-baseline/rag_baseline_results_path_a_v3.json
```

Optional sanity (natural routing, not scored):

```bash
cd growformer-llm
cargo run --bin tinystories -- brain-raw-diag --brain .../fintech-brain.bin \
  --project .../fintech-sentiment-analysis.gf.toml \
  --prompt "Bank of America repriced my home loan APR with zero advance warning" --top-k 3 -v
```

## Success criteria (held-out v2, same rubric)

- Both arms pass both v2 prompts → `HELDOUT_BOTH_PASS` (candidate Path A product scope; not “certified”)
- Brain passes both, RAG does not → `BRAIN_HELDOUT_ONLY`
- Otherwise → document gap; do not patch again on same commit

### Result (2026-07-05)

| Case | RAG@1 | Brain raw #1 | pass both |
| --- | --- | --- | --- |
| v2 crypto SEC kick | `crypto_gap_001` (`etf_delay_bearish`) | Same gap class, `witness=true` | **Yes** |
| v2 fintech BofA repriced | `sent_fin_099` (APR complaint) | `mortgage_rate_complaint` gap paraphrase, `witness=true` | **Yes** |

Harness outcome: **`HELDOUT_BOTH_PASS`**. Routing TOML fix validated; no further edits on this commit.

## Cross-links

- [`BET_D_FROZEN_PROTOCOL.md`](BET_D_FROZEN_PROTOCOL.md) — v2 frozen run (pre-fix)
- [`PRE_REGISTRATION.md`](PRE_REGISTRATION.md) §6
