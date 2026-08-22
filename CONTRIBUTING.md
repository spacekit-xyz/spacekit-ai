# Contributing to SpaceKit AI

## Development

Use package-scoped checks while iterating:

```bash
cargo fmt --all --check
cargo check --workspace
cargo test -p growformer-ledger
cargo test -p growformer-nca
cargo test -p growformer-llm
cargo test -p growformer --lib
```

Run the complete workspace suite before merging changes that affect shared APIs
or dependency configuration.

## Research integrity

Benchmark and evaluation changes must record the revision, dataset split, seed,
configuration, hardware, metric, and comparison budget. Do not remove negative
or historical results merely because a newer experiment performs better.

Changes to the evaluation ledger's canonical serialization require
compatibility tests. Changes to public model or runtime behavior require tests
and documentation.

## Repository policy

- Preserve dependency direction documented in `README.md`.
- Keep Growformer integrations optional where practical.
- Do not commit datasets, checkpoints, trained brains, generated reports,
  credentials, or private user data.
- Use small deterministic fixtures for tests.
- Keep package licenses and public metadata accurate.
- Update root CI when adding or renaming a workspace member.

Report vulnerabilities privately according to `SECURITY.md`.
