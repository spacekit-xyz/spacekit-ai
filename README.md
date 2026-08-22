# SpaceKit AI

SpaceKit AI is the public monorepo for Growformer research and machine-learning
projects. It contains the core promote-freeze substrate, language-model
experiments, reproducible evaluation tooling, and neural cellular automata
research.

Repository: [spacekit-xyz/spacekit-ai](https://github.com/spacekit-xyz/spacekit-ai)

## Projects

| Directory | Role | Status | License |
|---|---|---|---|
| [`growformer/`](growformer/) | Promote-freeze neural substrate, specialist routing, local inference, CLI, server, and WASM runtime | Primary implementation | Apache-2.0 |
| [`growformer-ledger/`](growformer-ledger/) | Hash-chained evaluation records and paired statistical comparisons | Supporting library | MIT |
| [`growformer-llm/`](growformer-llm/) | Domain language models, chat workflows, optional Growformer memory, and historical Clifford experiments | Depends on Growformer and Ledger | Apache-2.0 |
| [`growformer-nca/`](growformer-nca/) | Experimental neural cellular automata and Clifford-algebra components | Independent research crate | MIT |

## Clone and build

```bash
git clone https://github.com/spacekit-xyz/spacekit-ai.git
cd spacekit-ai

cargo check --workspace
cargo test --workspace
```

The Growformer suite is substantially larger than the supporting crates. Use
package-scoped commands during development:

```bash
cargo test -p growformer-ledger
cargo test -p growformer-nca
cargo test -p growformer-llm
cargo test -p growformer --lib
```

## Dependency direction

```text
growformer ───────────────┐
                         ├──> growformer-llm
growformer-ledger ────────┘

growformer-nca                 standalone
```

`growformer` and `growformer-ledger` remain independently usable.
`growformer-llm` consumes their public APIs. `growformer-nca` remains isolated
until an explicit integration contract replaces duplicated experimental code.

Internal dependencies use workspace-relative paths during development.
External consumers, including SpaceKit Core, should use tagged revisions or
published crate versions rather than filesystem paths.

## Repository layout

```text
spacekit-ai/
├── Cargo.toml
├── README.md
├── CONTRIBUTING.md
├── SECURITY.md
├── growformer/
├── growformer-ledger/
├── growformer-llm/
└── growformer-nca/
```

Each project owns its detailed architecture, usage, and reproducibility
documentation. Cross-project policy and CI live at the monorepo root.

## Research and production boundaries

These projects are experimental machine-learning systems. Passing tests or
reported benchmark results do not establish general intelligence, adversarial
robustness, model safety, privacy compliance, or suitability for autonomous
high-stakes decisions.

Results must identify the source revision, dataset split, seed, configuration,
hardware, metric, and comparison budget. Historical and negative results
remain part of the research record and must be clearly labeled.

## Data and generated artifacts

Do not commit:

- downloaded corpora or benchmark datasets;
- model checkpoints, trained brains, embeddings, or tokenized corpora;
- user conversations or production inference records;
- generated experiment reports or local evaluation ledgers;
- API keys, private keys, populated environment files, or certificates;
- build output, generated WASM packages, or dependency caches.

Small deterministic fixtures and inference configuration required by a
published crate are allowed when their provenance and purpose are documented.
Large datasets and models belong in versioned release assets or an external
artifact store with checksums.

## Relationship to SpaceKit Core

[SpaceKit Core](https://github.com/spacekit-xyz/spacekit-core) is a separate
infrastructure monorepo. AI integration there must be optional and
feature-gated. SpaceKit Core must consume tagged Growformer revisions rather
than unpublished sibling paths.

Private applications and websites may consume SpaceKit AI releases, but they
must not become dependencies of these public libraries.

## Contributing and security

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development requirements. Report
suspected vulnerabilities privately according to [`SECURITY.md`](SECURITY.md).
Never place secrets, private datasets, or exploit details in public issues.

## Licensing

This monorepo contains packages under different permissive licenses. Consult
each package's `Cargo.toml` and `LICENSE` file. A root-level policy does not
override package-level licensing.
