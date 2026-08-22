# Contributing to Growformer Ledger

```bash
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo package --allow-dirty
```

The ledger's canonical byte representation is part of its integrity contract.
When adding an `EvalRecord` field, update canonical serialization, compatibility
tests, and documentation together.

Never commit production result ledgers, private datasets, credentials, or model
artifacts. Use deterministic fixtures for tests.

Report security issues through `SECURITY.md`.
