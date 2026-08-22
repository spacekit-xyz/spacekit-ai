# Contributing to Growformer

Growformer is a research-oriented Rust crate with runtime, CLI, server, WASM,
benchmark, and reproducibility surfaces.

## Development

```bash
cargo fmt --all --check
cargo check --no-default-features --bin growformer-runtime
cargo test --lib
cargo test --test brain_merge --test causal_ordering_probe --test wasm_service_test
cargo package --allow-dirty
```

Run focused benchmark or scientific-reproduction scripts only when changing
their associated claims. Report hardware, revision, seed, dataset split, and
configuration with new results.

## Repository rules

- Do not commit downloaded datasets, model checkpoints, brains, embeddings,
  experiment output, credentials, or generated WASM packages.
- Keep small deterministic inference TOML files only when required by the
  published crate.
- Separate measured results from hypotheses and identify historical results.
- Update tests and public documentation with behavior changes.
- Keep optional features buildable with `--no-default-features`.

Report security issues through `SECURITY.md`, not a public issue.
