# Contributing to Growformer LLM

## Development

Run checks from the SpaceKit AI monorepo root so Cargo resolves shared
workspace dependencies consistently.

```bash
cargo fmt --all --check
cargo check -p growformer-llm --no-default-features --features vanilla-lm,clifford-lm
cargo test -p growformer-llm --lib
cargo test -p growformer-llm
```

Changes to reported comparisons must preserve dataset splits, selection tags,
seeds, parameter budgets, and evaluation metrics. Do not replace an unfavorable
result with a new protocol without retaining and labeling the earlier result.

Do not commit corpora, tokenized datasets, checkpoints, generated ledgers,
brains, credentials, or user conversations. Add tests and update developer
documentation for public behavior changes.

Report security issues through `SECURITY.md`.
