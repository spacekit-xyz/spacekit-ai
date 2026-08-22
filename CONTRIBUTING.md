# Contributing to Growformer LLM

## Development

Keep `growformer`, `growformer-ledger`, and `growformer-llm` as sibling
checkouts while path dependencies are used.

```bash
cargo fmt --all --check
cargo check --no-default-features --features vanilla-lm,clifford-lm
cargo test --lib
cargo test
```

Changes to reported comparisons must preserve dataset splits, selection tags,
seeds, parameter budgets, and evaluation metrics. Do not replace an unfavorable
result with a new protocol without retaining and labeling the earlier result.

Do not commit corpora, tokenized datasets, checkpoints, generated ledgers,
brains, credentials, or user conversations. Add tests and update developer
documentation for public behavior changes.

Report security issues through `SECURITY.md`.
