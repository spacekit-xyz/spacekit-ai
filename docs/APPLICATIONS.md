# Applications — What Growformer and Growformer JEPA Can Do

> **Audience:** this doc answers one question for two readers — *"What can I actually
> build or use today?"* — for **developers** (who train, embed, and ship) and for
> **users** (who interact with the resulting agents and tools).
>
> **Scope:** it covers the two tracks that share one architectural core:
> - **Growformer** — the trainable *brain* runtime for language, agents, sentiment, and code.
> - **Growformer JEPA** — the *world-model* track: frozen-encoder + promotable predictor
>   adapters for agents that predict, plan, and act.
>
> **Companion docs:** [`README.md`](../README.md) (overview + CLI), [`USE_CASES.md`](../USE_CASES.md)
> (substrate-level research use cases, "Neuro" naming),
> [`docs/GROWFORMER_PUBLIC_WHITEPAPER.md`](GROWFORMER_PUBLIC_WHITEPAPER.md)
> (canonical, self-contained public claims),
> [`docs/GROWFORMER_WHITEPAPER.md`](GROWFORMER_WHITEPAPER.md)
> (technical and reproducibility companion), [`docs/WORLD_MODELS.md`](WORLD_MODELS.md) (JEPA spec),
> [`docs/AI_OPERATING_SYSTEM.md`](AI_OPERATING_SYSTEM.md) (ecosystem).

## Maturity legend

Every capability below is tagged so you know what to rely on:

| Tag | Meaning |
|-----|---------|
| **[Production]** | Implemented, tested, and on the primary product path — safe to build on. |
| **[Preview]** | Implemented and working, but limited, evolving, or partially wired. |
| **[Research-certified]** | Real, reproducible, and gated by in-repo certifiers — but a demo/host substrate, not a shipped end-user product surface. |

The honest one-line summary: **language brains are the production surface; the JEPA
world-model stack is a certified research + hosting substrate.** Both are real; they
sit at different maturity levels.

---

## The shared idea (read this first)

Both tracks are built on one principle: **parameter isolation with promote-freeze
continual learning.** You train a new specialist in an isolated *Mirror*, then
*promote* it into a frozen *Main* store. Nothing overwrites what came before, so
**you get ~0% catastrophic forgetting by construction**, and every capability is a
named, inspectable unit rather than a diffuse weight update.

That single property is why the applications below are possible: you can keep adding
domains to a brain, ship small deterministic artifacts, trace why an output happened,
and pin a deployed model so it can't silently drift.

---

# Part 1 — Growformer (language, agents, code)

Growformer turns training data into a portable **brain** (`.bin` file) that runs
anywhere: CLI, an HTTP server, a lean embeddable runtime, the browser (WASM), or
inside another Rust program. Inference is a **single forward pass** over a fixed
program layout — not an autoregressive token loop — so it is fast and small enough
for edge and browser deployment.

### Inference modes (the verbs a brain understands) — [Production]

| Mode | What it produces | Typical application |
|------|------------------|---------------------|
| `action` | Structured action JSON (intent routing, no prose) | Route a request to a tool, ticket type, or sub-agent |
| `generation` | Single-shot text | Classification-with-reasoning, templated answers |
| `converse` | Multi-turn chat with memory + personality | Companion/support agents |
| `codegen` | Code snippet + language/kind | Code assistance, stub filling |
| `paramecium` | Lattice-only answer (no neural substrate) | Ultra-small-footprint inference |

## For users — experiences Growformer powers

- **Domain companion & chat agents** *(e.g. Luna/Kitsu companion loop)* — [Production] —
  multi-turn conversation with persistent memory, anaphora handling, topic shifts, and
  an **OCEAN (Big Five) personality** that shapes tone. See [`docs/OCEAN.md`](OCEAN.md),
  [`docs/CONTINUAL_PRODUCT_LOOP.md`](CONTINUAL_PRODUCT_LOOP.md).
- **Sentiment analysis, ready-made** — [Production] — general, **fintech/TradFi**, and
  **crypto/DeFi** flavors ship as project templates (`scripts/*-sentiment-analysis.gf.toml`).
  Classify headlines/posts as positive/negative/neutral/mixed/sarcastic with reasoning.
- **Code assistance** — [Preview] — route a request and emit a code snippet from trained
  code lattices, with deterministic template fallbacks. Great for scaffolding and stub
  filling; not yet a full free-form coding model.
- **A tool-using assistant** — [Preview] — the agent can call built-in tools:
  `calculator`, `file_reader`, `code_runner` are implemented; `web_search`/`web_payment`
  are stubs.
- **A private, offline, in-browser assistant** — [Production] — because a brain is a
  small self-contained file that runs in WASM with no server, users get an assistant that
  works offline and keeps data on-device. See [`docs/iot.md`](iot.md).

## For developers — what you can build and ship

- **Train a domain brain from JSONL** — [Production] — drop `*.jsonl` samples
  (`text`, `semantic_intent`, `expected_response`, `expected_code`, `history`, …) in a data
  dir, write a `*.gf.toml` project, and run `--train-brain`. Ships as one `.bin`.
- **Embed it five ways** — [Production]:
  - **CLI** (`growformer`) for training/merge/infer and a REPL,
  - **HTTP server** (`growformer-node`) with `/v1/chat`, SSE streaming, multi-brain
    loading, `/v1/feedback`, `/v1/personality` (see [`docs/NODE.md`](NODE.md)),
  - **Lean runtime** (`growformer-runtime brain.bin "prompt"`) — inference only,
  - **WASM** (`growformer_*` exports) for browser/edge,
  - **Rust library** (`growformer::Runtime` / `LanguageService`) embedded in your app.
- **Host many brains at once** — [Production] — load a directory of `.bin` files and
  switch the active brain per request (`GROWFORMER_BRAIN_DIR`, `brain` field in a request).
- **Add capabilities without retraining from scratch** — [Production] — `--merge-brain`
  overlays specialists (e.g. a causal-relationship brain onto a sentiment brain);
  `--train-router-adapter` / `--train-fingerprint-adapter` update routing while lattices
  stay frozen.
- **Online / feedback-driven improvement** — [Preview] — submit accept/reject/correction
  feedback (`/v1/feedback`) to nudge lattice quality and routing; see
  [`docs/CONTINUUM.md`](CONTINUUM.md) for the train-while-on roadmap.
- **Traceable, certifiable outputs** — [Production] — responses carry `template_id`,
  `traceable`, and routing `reason`; a certifier suite (`--certify-encoder`,
  `--acceptance-report`, grounding-loop audits) lets you gate a brain before shipping.
- **Distribute under entitlement** — [Production] — end users invoke brains via
  `spacekit agent` with capability gating (`growformer.train/infer/merge`).

### Ready-made application templates (`scripts/*.gf.toml`)

| Template | Application |
|----------|-------------|
| `sentiment-analysis.gf.toml` | General sentiment/tone |
| `fintech-sentiment-analysis.gf.toml` | Equities/banking/corporate-finance sentiment |
| `crypto-sentiment-analysis.gf.toml` | Blockchain/DeFi/market sentiment |
| `code.gf.toml` | Code-assist brain (`code_brain = true`) |
| `causal-relationship.gf.toml` | Causal-connector classifier (mergeable overlay) |

---

# Part 2 — Growformer JEPA (world models: predict, plan, act)

Growformer JEPA applies the same promote-freeze discipline to **world models**. A
**frozen, hash-pinned encoder** maps observations to a latent `z`; **promotable adapters**
(predictor / energy / action-energy) are each trained on one dynamics regime and frozen
in; an **authenticated router** dispatches among them at inference; and a planner picks
actions via short **latent rollouts** or low-energy descent.

> **What it is not** (stated plainly, matching the whitepaper's discipline):
> not full LeCun-AMI / large-scale JEPA training; not a chat surface (Luna/chat is
> explicitly *not* a world-model certifier). WM bundles are now **DM citizens** and can
> ship in one `brain.bin` with language specialists (`--phase5a-wm-dm`, `--phase5e-wm-brain`);
> that is Preview+/Research wiring, not AMI or a shared mega-backbone.

This track is **[Research-certified]**: the toy→scene ladder (phases 3i–3w) passes
in-repo certifier gates and runs as a SpaceKit host, but it is infrastructure, not a
packaged end-user product.

## For users — experiences it enables

- **Agents that act in an environment (not chat)** — [Research-certified] — a deployed
  world-model bundle takes an observation and returns a route/plan/action, e.g. steering
  in a 2D dynamics world or acting on a **scene graph** of objects and typed relations.
- **Spatial / scene understanding** — [Research-certified] — the scene-graph world model
  reasons over explicit objects + relations (phase `3v`), hosted for a simulator (phase `3w`).
- **Predict-and-plan behavior** — [Research-certified] — action-conditioned energy
  \(E(z,a,z')\) plus a short-horizon planner choose the lowest-energy next move.

## For developers — what you can build

- **Learn a world model from observations** — [Research-certified] — train predictor/energy
  adapters per dynamics regime on synthetic or logged transitions (`--phase3i-jepa-wm`,
  `--phase3j-energy-wm`, …).
- **Plan with action-conditioned energy** — [Research-certified] — `plan_action` does
  greedy/short-horizon planning over a discrete action set (`--phase3n-action-wm`).
- **Transfer across domains** — [Research-certified] — the same certifiers hold on
  *foreign* dynamics (bounce ball, central force) via `wm_proof.rs` (`--phase3r-beyond-toy`).
- **Bridge to real vision features** — [Preview] — a **V-JEPA** export slot lets you plug
  frozen `facebook/vjepa2-*` features (or a mock bank) as the encoder (`--phase3u-vjepa-wm`).
- **Deploy a pinned, verifiable bundle** — [Research-certified] — serialize a
  `ComposedWmBundle` and serve it with `deploy_step`; the encoder **fingerprint is verified
  on load**, so a drifted model hard-fails instead of silently misbehaving.
- **Host it for SpaceKit** — [Research-certified] — a JSON-lines stdio protocol
  (`--wm-host-stdio scene|acting|deploy`) exposes `load_bundle` / `step` / `act` to an
  external simulator. See [`docs/WM_SPACEKIT_HOST.md`](WM_SPACEKIT_HOST.md); Python client
  at `scripts/wm_spacekit_client.py`.
- **Representation-learning research** — [Research-certified] — context-free routing over
  frozen specialists on MNIST (`--phase4a` … `--phase4d-cf-mnist-full`); CIFAR-10 with a
  frozen patch bank (`--phase4f-split-cifar-frozen`).
- **WM inside DimensionManager** — [Preview+] — promote pinned acting/composed bundles as
  Main *citizens* and call act/deploy through DM (`--phase5a-wm-dm`); persist via checkpoint
  and graduate into one language+WM `brain.bin` (`--phase5e-wm-brain`). Not AMI.
- **Product act-loop** — [Research-certified] — ship metric is **task return** vs random/VG
  with pin reload (`--phase5b-product-act`); live SpaceKit acting-host episode
  (`--phase5f-live-spacekit`); chat is never the WM certifier.
- **D′ real-log V-JEPA** — [Preview] — dump a visuomotor log, freeze-export, train adapters
  (`--phase5g-vjepa-real-log`); HF via `export_vjepa_features.py --mode hf --log …`.

> **Sibling project:** `growformer-llm` is a separate small-domain transformer LLM crate
> (vanilla-transformer default; optional Clifford research; optional "brain-memory" that
> retrieves from Growformer lattice brains). It shares the promote-freeze story but
> contains **no** JEPA/world-model code.

---

## Why choose this over a conventional model

These cut across both tracks and are the real "why":

- **Continual learning without forgetting** — add domains/regimes forever; old ones are
  frozen, not overwritten (Split MNIST: 97.7%, 0.0% forgetting, reported as a retention audit).
- **Small, portable, on-device** — a brain is a single `.bin`; runs in WASM/browser and on
  edge/IoT; `paramecium` mode is kilobyte-scale.
- **Traceability & certification** — outputs carry provenance; a certifier suite gates
  quality before you ship, rather than trusting aggregate accuracy.
- **Verifiable deployment** — world-model bundles are **hash-pinned**; reload after restart
  must reproduce the same fingerprint or it fails closed.
- **Specialist routing/composition** — dispatch among frozen experts (the qualified positive
  result of the routing research; see the whitepaper for what is and isn't claimed).

---

## Capability maturity at a glance

| Application area | Track | Maturity |
|------------------|-------|----------|
| Sentiment (general / fintech / crypto) | Growformer | [Production] |
| Companion / support chat + personality | Growformer | [Production] |
| Multi-brain hosting, merge, adapters | Growformer | [Production] |
| CLI / HTTP node / runtime / WASM / lib embedding | Growformer | [Production] |
| Traceability + certifier gating | Growformer | [Production] |
| Code assistance | Growformer | [Preview] |
| Tool use (calculator/file/code) | Growformer | [Preview] (web tools stubbed) |
| Online/feedback learning (Continuum) | Growformer | [Preview] |
| World-model predict / plan / act | Growformer JEPA | [Research-certified] |
| Scene-graph WM + SpaceKit host | Growformer JEPA | [Research-certified] |
| Domain transfer + pinned deploy | Growformer JEPA | [Research-certified] |
| V-JEPA real-vision bridge | Growformer JEPA | [Preview] |
| Split-CIFAR-10 frozen patch CL (`--phase4f`) | Growformer JEPA | [Preview] — PASS 10/10 (41% input-only routing; zero forgetting) |
| WM citizens in DimensionManager (`--phase5a-wm-dm`) | Growformer JEPA | [Preview+] (persist + roundtrip) |
| Language + WM `brain.bin` (`--phase5e-wm-brain`) | Growformer JEPA | [Preview+] |
| Product act-loop return metric (`--phase5b-product-act`) | Growformer JEPA | [Research-certified] |
| External product loop (`--phase5c-external-product`) | Growformer JEPA | [Research-certified] |
| Live SpaceKit episode (`--phase5f-live-spacekit`) | Growformer JEPA | [Research-certified] |
| D′ lite frozen-vision JEPA (`--phase5d-vjepa-vision`) | Growformer JEPA | [Preview] (HF V-JEPA optional) |
| D′ real-log V-JEPA (`--phase5g-vjepa-real-log`) | Growformer JEPA | [Preview] (HF via `--mode hf --log`) |
| Full-scale JEPA / LeCun AMI | Growformer JEPA | Deferred until D′ HF-at-scale + DM wiring green |

---

## Getting started (pick your path)

```bash
# Train a language brain from a project manifest
cargo run --release -- --train-brain --project scripts/crypto-sentiment-analysis.gf.toml

# Ask it something
cargo run --release -- --infer --project scripts/crypto-sentiment-analysis.gf.toml \
  --prompt "ETF inflows hit a record this week"

# Serve a brain over HTTP (chat, SSE, multi-brain)
GROWFORMER_BRAIN_PATH=brain.bin cargo run --release --bin growformer-node

# Run a certified world-model demo, then host it for a simulator
cargo run --release --bin growformer-demos -- --phase3i-jepa-wm
cargo run --release --bin growformer-demos -- --wm-host-stdio scene
```

Browse the full demo catalog (~90 entry points) with
`cargo run --bin growformer-demos -- --help` (registry: `src/demos.rs`).

---

*This document is an applications catalog. For the science and its explicit non-claims,
start with the canonical
[`docs/GROWFORMER_PUBLIC_WHITEPAPER.md`](GROWFORMER_PUBLIC_WHITEPAPER.md). The detailed
protocols and reproduction record remain in
[`docs/GROWFORMER_WHITEPAPER.md`](GROWFORMER_WHITEPAPER.md); for substrate-level research
use cases, see [`USE_CASES.md`](../USE_CASES.md).*
