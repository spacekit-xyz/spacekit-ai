# Contributing to SpaceKit NCA

```bash
cargo fmt --all --check
cargo clippy --all-targets
cargo test
cargo package --allow-dirty
```

Add numerical tests for algebra, gradients, optimizers, masking, cache behavior,
and training changes. Document parameter-count and compute-budget assumptions
with benchmark claims.

Do not commit datasets, checkpoints, generated model output, credentials, or
populated environment files. Report security issues through `SECURITY.md`.
