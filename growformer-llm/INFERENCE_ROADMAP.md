# growformer-llm: Inference & Framework Roadmap

**growformer-llm** is a framework for building **domain specialists**, models
that deeply understand *one subject*, not general-purpose all-knowing LLMs. Every design
choice below follows from that. A specialist is small, its corpus is narrow, its inputs are
mostly in-distribution, and devs will train and ship *many* of them. That makes the wins
different from GPT-scale work: capacity-scaling matters less; **cheap deployment, conditional
compute, grounding, and routing across a fleet of specialists** matter more.

This roadmap supersedes the MoE-first ordering in `SPARSE_ACTIVATION_PLAN.md`. That plan is
still correct as a component, it becomes **Stage 3b** here, but a unified model format and a
runtime come first, because sparse activation, quantization, and serving all change the model
payload, and you should redesign that payload **once**.

---

## What "better inference" means for a specialist (vs a generalist)

| Lever | Generalist framing | **Specialist framing (this repo)** |
|---|---|---|
| Sparse activation | MoE capacity scaling (more params, flat FLOPs) | **Conditional compute**, skip/early-exit on in-domain easy tokens; **subtopic experts**; **route across a fleet of specialists** |
| Model size | bigger is better | **smaller is shippable**, quantize, prune, distill to the domain |
| Knowledge | in the weights | in the weights **+ grounding TOML** (you already have `world_grounding*.toml`) |
| Delivery | one giant served model | **many tiny models** + a router; edge/wasm/embedded |
| Eval | broad benchmarks | **bits/byte on the domain corpus** (already your north-star metric) |

The "single subject vs all-knowing" architecture is literally **sparse activation at the model
level**: instead of one dense model that knows everything, a router activates the one small
specialist that owns the subject. You already have the routing primitive
(`LmAdjustableConeRouter`, `semantic_router` in growformer), Stage 3c wires it up.

---

## Current state (verified in code)

- **Two checkpoint formats, both pretty-printed JSON:**
  - `LmCheckpoint` (`src/v2/checkpoint.rs`), `schema = 3`, Clifford/LM path; FFN variant carried by `cfg.dense_ffn` + optional `dense_fc1/fc2`.
  - `VanillaCheckpoint` (`src/v2/vanilla_checkpoint.rs`), `schema = 1`, product path.
  - No capability flags beyond the schema int + `cfg` fields; no compact/quantized payload (growformer's `language_checkpoint.json` is ~396 KB of pretty JSON, this does not scale to a fleet).
- **Inference exists and already streams:** `InferenceCache` (KV-cached) + `generate_stream` with a `TokenCallback` (`src/main.rs` `Generate`, and `tinystories generate`).
- **No delivery surface of its own:** only `tinystories`/`gf-llm`/`growformer-llm` train/eval/generate CLIs. The lean runtime, HTTP server, Docker, and wasm all live in **growformer**, and growformer-llm does not build to a growformer artifact — so it inherits none of them.
- **Domain machinery already present:** grounding TOML, the classifier `train`/`infer` path, and the cone router — the repo already leans specialist.

---

## Sequenced roadmap

### Stage 1 : Unified, versioned model container *(foundation; do first)*

One container both stacks write and the runtime reads, designed to hold everything the later
stages add so the format is not re-migrated three times.

```rust
// src/model_container.rs (new)
pub struct ModelContainer {
    pub format_version: u32,          // container envelope version (start at 1)
    pub arch: Arch,                   // Vanilla | Clifford
    pub caps: Capabilities,           // feature flags (below)
    pub card: ModelCard,             // provenance + domain identity
    pub cfg: TrainConfigV2,           // existing config (already serde)
    pub tokenizer: TokenizerRef,      // inline vocab, or a ".tok" sidecar reference
    pub payload: WeightPayload,       // JsonF32 (today) | Bincode | QuantI8
}

pub struct Capabilities {
    pub tied_embeddings: bool,
    pub ffn: FfnKind,                 // Dense | Clifford | Moe { n_experts, top_k, skip_expert }
    pub quant: QuantKind,             // None | Int8 { scale_per_channel }
    pub grounding: Vec<String>,       // grounding TOML ids this specialist expects
    pub early_exit: bool,
}

pub struct ModelCard {               // what makes a specialist shippable + discoverable
    pub subject: String,              // e.g. "fintech-sentiment" — the ONE thing it knows
    pub created_utc: String,
    pub train_steps: u64,
    pub eval_bits_per_byte: Option<f32>,
    pub base_model: Option<String>,   // for fine-tuned / distilled specialists
    pub notes: String,
}
```

- **Payload:** keep `JsonF32` as the compatibility default; add a **`Bincode`** payload (compact) and later `QuantI8`. `flate2` is already a dependency — gzip the payload for a large drop from the 396 KB-class pretty JSON.
- **Backward compatibility:** `load_container` first tries the envelope; on failure it falls back to `LmCheckpoint`(schema 2/3) and `VanillaCheckpoint`(schema 1) via thin adapters, then wraps them in a `ModelContainer` in-memory. Existing checkpoints keep loading; nothing breaks.
- **Exit check:** round-trip test per arch; every existing checkpoint in `data/` / `agent-data/` loads through the new path and produces byte-identical logits.

*Why first:* the runtime (Stage 2), quantization (Stage 4), and MoE (Stage 3b) all change the
payload and the capability set. Designing the envelope once means one migration, not three.

### Stage 2 : Lean runtime + serving *(the delivery mechanism you're missing)*

Mirror growformer's proven pattern, scoped to the transformer:

- **`gf-llm-runtime` binary** — loads a `ModelContainer`, single-shot + `--json` + REPL, reuses
  `InferenceCache` + `generate_stream` (already streaming). Build lean (no training deps),
  same spirit as `growformer-runtime`.
- **`gf-llm-node` (feature `server`)** — small axum server: `POST /generate` with token
  streaming (SSE), `GET /card` returns the `ModelCard`. Mirrors `growformer-node`.
- **wasm target** — add `crate-type = ["cdylib","rlib"]` + a `wasm.rs` inference entrypoint so a
  specialist runs in-browser/edge. Specialists are small enough that this is realistic.
- **Docker + `build-linux.sh`** — copy growformer's, retarget the binary.

*Exit check:* `gf-llm-runtime <container> "prompt"` streams a continuation; `curl` the server;
a wasm smoke build loads a tiny specialist. **This is the concrete answer to your delivery
question** — growformer-llm gains its own delivery surface instead of depending on growformer's.

### Stage 3 : Sparse activation for specialists *(recast MoE plan)*

Smallest, highest-certainty win first:

- **3a — Conditional compute / early exit.** A per-token skip gate on the FFN (identity path)
  and optional **early exit** at block *k* when the running prediction is confident. Specialists
  see mostly in-distribution tokens, so a large fraction are "easy" → real latency drop at equal
  quality. Cheapest real inference speedup; needs no new experts.
- **3b — Subtopic experts (the `SPARSE_ACTIVATION_PLAN.md` MoE).** Top-1 switch over N expert
  FFNs for a domain's sub-areas, `n_experts=1` == today's dense FFN, load-balancing aux loss.
  Now rides the Stage-1 container (`FfnKind::Moe`) so MoE specialists are directly deployable.
- **3c — Specialist-fleet routing.** Route across *multiple* `ModelContainer` specialists with
  the existing `LmAdjustableConeRouter` / `semantic_router`. This is the "single subject vs
  all-knowing" design as a system: N tiny specialists + a router, only one active per query.
  Connects to growformer's specialist-routing substrate.

*Exit check per tier:* 3a — tokens/sec up at ≥ parity bits/byte; 3b — MoE ≥ dense at matched
active FLOPs on the domain corpus, no dead experts; 3c — router picks the right specialist above
a target accuracy on held-out mixed-domain prompts.

### Stage 4 : Specialist quality levers

- **corpus-semantic init** — already a ~37 % val-ppl win; make it the framework default and
  document it as *the* specialist lever.
- **grounding TOML as domain prior** — first-class in the container (`caps.grounding`) and the
  runtime, so a specialist ships with its world model.
- **int8 quantization** — `QuantKind::Int8` payload from Stage 1; shrinks specialists for edge/wasm.
- **distillation** — distill a larger teacher into the small domain specialist (quality at tiny size).
- **domain vocab sizing** — right-size BPE per subject; a narrow domain needs far less vocab.

### Stage 5 : Framework / developer ergonomics

Make the *library* surface first-class so devs build their own specialists:

- `Model::load(path) -> Session`, `Session::generate(prompt, cfg) -> stream` — the ergonomic API
  over today's `InferenceCache` plumbing.
- Pluggable `Ffn` / `Attention` traits so a dev swaps dense/MoE/their own without forking.
- `from_pretrained` + the `ModelCard` convention for sharing specialists.
- Examples (`examples/train_a_specialist.rs`), a quickstart, and per-subject templates.

---

## Immediate next actions

1. **Stage 1 kickoff:** add `src/model_container.rs` with the envelope + capability flags +
   `ModelCard`, `save_container` / `load_container`, and adapters from the two existing schemas.
   Ship behind the existing default features; existing checkpoints must still load.
2. Add the round-trip + byte-identical-logits test (style of `checkpoint::tests::roundtrip_*`).
3. **Stage 2 kickoff:** `gf-llm-runtime` bin wrapping `InferenceCache` + `generate_stream`.
4. Then proceed into Stage 3a (conditional compute) → 3b (MoE) → 3c (fleet routing).

**Build note:** Stages 1–2 are additive (new module + new binary + `Cargo.toml` bin/feature
entries) and low-risk, but they need to compile and pass tests against the real internal APIs.
The safe way to land them is to implement + `cargo test` in the repo. I can write the actual
code for Stage 1 and the `gf-llm-runtime` bin next; it should be done with build access so the
first commit compiles green rather than landing speculative Rust.
