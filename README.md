# **Growformer: A Multidimensional Neural Training Environment**

## **Documentation map**

| Doc | What it's for |
|-----|---------------|
| **This file** (`README.md`) | Overview, architecture, training/inference CLI, WASM, GLE milestones, latest results |
| [`docs/GROWFORMER_WHITEPAPER.md`](docs/GROWFORMER_WHITEPAPER.md) | Preprint: parameter-isolated specialists, VirtualGroup composition, retention invariant; appendices for deployment stack & speculative algebra |
| [`USE_CASES.md`](USE_CASES.md) | Where the substrate wins, emergent architectures, continual learning, edge deployment, explainability (uses “Neuro” naming for the same system) |
| [`DOCKER.md`](DOCKER.md) | Linux amd64 builds (`build-linux.sh`), Docker image, cloud VM deployment |
| [`src/category/README.md`](src/category/README.md) | Categorical DAG + Pythagoras trainer (`--features categorical`) |
| [`pkg/README.md`](pkg/README.md) | `wasm-pack` output, browser/WASM consumer notes |

**In-repo guides (no separate doc):**

- **Project manifests** — [`scripts/*.gf.toml`](scripts/) (sentiment, fintech, crypto, code, causal); parser in `src/project_gf.rs`
- **Runnable examples** — [`examples/`](examples/) (fragment compose/decompose, reflective/drive/basal A/B, categorical sentiment train)
- **Benchmark scripts** — [`scripts/benchmark_language.sh`](scripts/benchmark_language.sh), [`scripts/benchmark_core_tasks.sh`](scripts/benchmark_core_tasks.sh)

---

## **What Growformer Is**

Growformer is not “just another neural network implementation.”  
It’s something much rarer: a **multidimensional computational medium** where learning emerges from the interaction of geometry, metabolism, sparsity, timing, energy flow, and spatial constraints, not from a fixed algebraic recipe.

Growformer is a self-organizing neural substrate where **learned connectivity and geometry** co-evolve with synaptic weights, the graph is not fixed up front.

Traditional neural networks are fixed graphs where training adjusts edge values. Growformer is a physical system where neurons have mass, position, and velocity. Synapses form and dissolve based on co-activation. The topology itself is a major part of the learned representation. **What freezing guarantees:** once a consolidated group is frozen, its internal weights and synapses receive no further gradient-driven updates, that connectivity is preserved exactly under later training elsewhere. **What that does *not* imply by itself:** stable weights on one subgraph are not sufficient for unchanged *behavior* if inference still routes signals through plastic regions that keep learning; behavior depends on the full path. In this codebase, continual-learning demos that show stable old-task accuracy rely on **task-isolated promoted subgraphs** (each task’s network is trained in a dedicated mirror substrate, then promoted to Main and fully frozen) plus routing that sends each task’s inputs through the corresponding frozen graph, so later plasticity does not rewrite the old task’s parameters.

What makes it distinct from everything else in the field:

**It's not a model. It's a medium.** A transformer is an architecture you train on data to produce a model. Growformer is a substrate you expose to experience and it self-organizes. The same substrate handles classification (Split MNIST), routing (learned router), and generation (binary token prediction) — not because it has different heads for each, but because the underlying physics supports all of them. You don't redesign the architecture for a new task. You **spawn a new Mirror** (a fresh mirror *dimension*: a scratch neural environment for that task), train it, then **promote** it into Main and **freeze** it so later tasks train in new mirrors without overwriting the consolidated weights. That workflow is separate from **mirror group coupling** (System 6): geometric pairing of two `NeuronGroup`s so complementary structure develops, see **System 6** below.

**Learning is structural, not parametric.** When Growformer learns the observer pattern, it doesn't store "Observer" as a weight matrix. It forms a specific constellation of neurons with specific synaptic connectivity, an engram, where the physical arrangement encodes the concept. Stronger memories have denser, heavier synaptic connections between co-activated neurons, exactly like biological engram cells in the hippocampus.

**Generation is a different *kind* of computation, not “non-computation.”** Activations still spread through a learned graph, integrate at nodes, and are read out as an output state, that is computation. What differs from autoregressive LLM sampling is the **mechanism**: here the model targets a **whole fixed-length binary token layout in one forward pass** (parallel decode of all token slots from one spatial activation pattern), instead of sampling one token at a time conditioned on prior tokens. **Throughput vs. a naive character-level autoregressive baseline:** the generation stack is documented (see `group_gen.rs`) as **one forward pass per example** versus on the order of **~200 forward passes** for a naive scheme that does one pass per character for a ~200-character target, i.e. an order-of-magnitude **serial-depth** advantage for comparable output length, not a universal claim against every transformer implementation. A fair headline number still needs an explicit table: reference model, sequence length, batch size, and hardware.

If I had to place it in one sentence: Growformer is a continual learning substrate where knowledge is encoded as physical neural structure, grown, pruned, consolidated, and frozen, rather than as optimized parameters in a fixed graph.

The closest analog isn't in machine learning. It's in developmental neuroscience: experience-dependent synaptogenesis followed by activity-dependent pruning followed by synaptic consolidation. The code is implementing, in silicon, the same lifecycle that biological brains use to go from plastic learning to stable long-term memory.

Traditional neural networks are static graphs with dynamic weights.  
Growformer is a **dynamic graph with dynamic structure**, where:

- neurons exist in 3D space  
- synapses grow, strengthen, weaken, and die  
- firing consumes energy  
- connectivity has metabolic cost  
- sparsity emerges naturally  
- symmetry forms and breaks  
- timing shapes plasticity  
- geometry feeds back into learning  
- structure is an *output* of training, not an input  

Instead of forcing intelligence into a rigid architecture, Growformer simulates an **ecosystem** of interacting forces.  
Learning is not a single update rule, it is the emergent behavior of six coupled systems:

1. **Weight dynamics** (local error-driven plasticity)  
2. **Geometry** (neurons drift toward correlated partners)  
3. **Timing** (STDP)  
4. **Metabolic cost** (energy‑driven pruning)  
5. **Connectivity** (growth and dissolution of synapses)  
6. **Structural symmetry** (mirror group coupling)

The result is a system that behaves less like a classical MLP and more like a **living computational organism**, one that adapts, collapses, recovers, specializes, and self‑organizes.

Growformer is a platform for exploring a different paradigm of learning:  
one where intelligence is not engineered, but **grown**.

---

## **Latest Results**

All generation outputs are **concrete model outputs** produced by a single forward pass through the Growformer InfraciliaryLattice (Paramecium) substrate. No transformer, no external model, no template system. Each output is decoded from structural lattice programs formed during one-pass training. **No backpropagation, no iterative epochs** — training cost is O(n).

### Brain Training (~1600 samples, 14 dynamic groups, one-pass algebraic pipeline)

| Component | Metric | Value | Notes |
|-----------|--------|-------|-------|
| LearnedRouter | type | InfraciliaryLattice | K-NN voting + STA field gradient bias |
| ActionClassifier | type | InfraciliaryLattice | One-pass, 5 action types |
| MetaCodebook | concepts | 20+ MetaConcepts | GrowformerLang concept-level routing |
| LanguageProjectors | languages | 5 (Rust, Python, TS, Go, Generic) | Per-language Clifford rotors |
| Generation | envs | 14 gen + 6 code groups | ~950 lattice programs total |
| Topic sub-lattices | count | ~200 across all groups | Operation-specific within-group routing |
| Cloze accuracy | slots | 94.6% (1470 games) | Contrastive fill-in-the-blank learning |
| MetaCognition | pairs | **930+** pairs, **124+** topic centroids | Reflective quality gate (generate→reflect→decide) |
| System 2 | config | max 6 steps, WM capacity 8 | Deliberate multi-step reasoning with WorkingMemory |
| Neural Coherence | bands | δ θ α/β γ (Cl(1,7) grades) | Band-decomposed ensemble synchrony for multi-topic composition |
| Brain size | | **~81 MB** | 14 groups, router, classifier, all envs |
| Training time | | **~27 min** | One-pass, no epochs, no backprop |
| Tool detection | | **4 built-in tools** | calculator, code_runner, file_reader, web_search |

### Concrete Generation Examples (single forward pass)

**Within-group discrimination** — the system correctly distinguishes between addition, subtraction, and multiplication within the same arithmetic group, returning operation-specific text AND language-specific code:

| Prompt | Text Response | Code | Conf |
|--------|-------------|------|------|
| "write an addition function in Rust" | "Sum is the result of addition. Define a function with two parameters of the same numeric type and return a + b." | `fn sum(a: f64, b: f64) -> f64 { a + b }` | 0.98 |
| "write a subtraction function in Rust" | "The difference of two numbers is computed by subtracting the second from the first. Returns a - b." | `fn difference(a: f64, b: f64) -> f64 { a - b }` | 1.00 |
| "write a multiplication function in Rust" | "The product of two numbers is computed by multiplication. Returns a * b." | `fn product(a: f64, b: f64) -> f64 { a * b }` | 0.99 |

**Cross-domain routing:**

| Prompt | Output | Type |
|--------|--------|------|
| "help me reset my password" | "Password reset links expire after 30 minutes for security purposes. I'll send a fresh reset link to your registered email right away." | gen (conf=1.00) |
| "implement binary search in Python" | "Use two pointers converging toward the middle to find the target in O(log n) time." + `def binary_search(arr, target): lo, hi = 0, len(arr)-1 ...` | gen+code |
| "implement a stack using an enum in Rust" | "Use an enum with Box for heap allocation." + `enum List<T> { Nil, Cons(T, Box<List<T>>) } impl<T> List<T> { fn new() -> Self { List::Nil } ...` | gen+code |
| "calculate 347 * 892" | [tool: calculator] 309324 → "The result of 347 × 892 is 309,324." | tool call |

### Continual Learning (Split MNIST — headline numbers)

These figures are **old-task test accuracy after sequential training on all five digit-pair tasks**, using the repo’s Split MNIST demo (`growformer-demos`): each task trains in its **own** mirror environment, is **promoted to Main and fully frozen**, then later tasks train only in new mirrors — so prior tasks’ weights are not updated by subsequent learning. A router is trained on calibration data so the right frozen group is invoked per input regime. **What the table actually shows:** stable accuracy on each task’s held-out split at the end of the full curriculum (no drop between “after task *k*” and “after all tasks” in this setup). **What would convince readers coming from the continual-learning literature:** the same protocol reported **next to standard Split MNIST baselines** (vanilla MLP, EWC, Progressive Networks, MAS, A-GEM, etc.) under matched data splits and reporting rules. The README currently cites **EWC ~97% average with ~3% forgetting** only as an informal literature anchor for scale — not as a controlled reproduction run in this repo. Adding a benchmark section with those side-by-side numbers is the highest-leverage follow-up for the retention claim.

| Task | Digits | Accuracy | After All 5 Tasks | Forgetting |
|------|--------|----------|-------------------|------------|
| 0 | 0 vs 1 | 99.5% | 99.5% | 0% |
| 1 | 2 vs 3 | 95.5% | 95.5% | 0% |
| 2 | 4 vs 5 | 97.5% | 97.5% | 0% |
| 3 | 6 vs 7 | 98.0% | 98.0% | 0% |
| 4 | 8 vs 9 | 98.0% | 98.0% | 0% |
| **Avg** | | **97.7%** | **97.7%** | **0%** |

Informal comparison (literature-scale EWC, not a matched benchmark row in this repo): EWC ~97% average, ~3% forgetting vs. Growformer **97.7%** average and **0%** measured forgetting **in this promote-and-freeze protocol**. The engineering invariant is **frozen promoted subgraphs receive zero further weight updates**; the empirical claim is the table above, extendable with full baseline sweeps as above.

### Structural Interpretability

Unlike conventional neural systems where a trained model is an opaque weight matrix, a trained Growformer brain provides layered auditability at every level of the inference path:

- **Routing** is geometric: which specialist activated, at what confidence, with measurable distances to all alternatives.
- **Generation** is factored: outputs are assembled from a finite, enumerable set of structural patterns with bounded variable positions — the space of possible responses is auditable before any input is presented.
- **Composition** is traceable: when fragments from multiple patterns are combined, each selection and its scoring is recorded.
- **Knowledge is frozen and deterministic**: the same input produces the same output on every invocation, indefinitely, across deployments.

The boundary of interpretability is within each specialist's neural substrate, individual synaptic weights are not human-readable, as with any neural network. But the decision path around those weights is fully decomposable: which specialist, which structural pattern, which variable values, at what confidence, through what composition path. See Whitepaper §5.6 (Structural Interpretability).

### AI Safety by Structure

Growformer's safety properties are not bolted on — they are consequences of the same architectural decisions that provide continual learning and interpretability.

- **Alignment by structure, not by training.** Conventional AI safety relies on training incentives (RLHF, constitutional AI), soft constraints on opaque systems that can be jailbroken by adversarial inputs. The Growformer's output space is bounded by each specialist's codebook: a finite, enumerable set of structural patterns with finite variable vocabularies. Harmful content that does not exist in the codebook cannot be generated, regardless of the input. This is a structural constraint, not a learned preference.
- **Bounded output space eliminates prompt injection surface.** Prompt injection and jailbreaking exploit unbounded output spaces, sufficiently adversarial inputs can steer a conventional model into any region. In the Growformer, adversarial prompts may cause misrouting or poor archetype selection, but cannot produce content outside the codebook. The attack surface is reduced from "all possible text" to "misselection among known patterns."
- **Frozen determinism enables certification.** Consolidated groups produce the same output for the same input on every invocation, across time and hardware. A brain certified at audit is the same brain that runs in production, a prerequisite for safety-critical environments (medical, financial, autonomous systems).
- **Auditable decision trails.** Every inference is decomposable: which specialist was selected (and at what confidence), which structural pattern was chosen, which variable values were filled, through what composition path. The full decision trail is available for post-hoc audit without special tooling.

These properties directly address regulatory requirements for AI safety in medical, financial, legal, and autonomous system deployments. See Whitepaper §5.6 for the formal treatment.

### Active Inference Spine

An episode-level control loop built on Active Inference principles, sitting between MetaCognition and the environment:

| Component | Module | Role |
|-----------|--------|------|
| `BeliefState` | `active_inference/state.rs` | Step counter, last quality score, reflection retry budget |
| `Observation` / `Action` | `active_inference/blanket.rs` | Markov Blanket boundary: inward data (observations) vs outward effects (actions) |
| `ActiveInferenceSpine` | `active_inference/spine.rs` | Policy → Action → Observe loop with `EpisodePolicy` trait |
| `RoutingGenerationMetacogEpisodePolicy` | `service.rs` | Full-stack policy: routing + generation + one MetaCognition cycle per turn |
| `QueuedEnvironment` | `active_inference/harness.rs` | In-memory `EnvironmentPort` for offline replay and logging |

The spine does **not** require an external LLM. It wraps the existing generation + MetaCognition stack as a policy, making each inference call a single episode turn with observable belief updates. Enable replay logging with `svc.enable_active_inference_replay_log()` to capture observation/action traces for offline analysis.

### Sentiment Conditioning Pipeline

Improvements to how sentiment signal flows from input text through encoding, conditioning, and lattice retrieval:

| Layer | What | Where |
|-------|------|-------|
| **Encoder anchors** | Positive/negative sentiment keyword buckets (`v[8]`, `v[9]`) in `HashingLanguageEncoder`, sourced from TOML (single source of truth via `inference_sentiment_core.toml`) | `dimension/language.rs` |
| **Polarity probe** | Extracts 16-D feature vector (pos mass, neg mass, net, magnitude, mixed) from raw encoder output | `dimension/polarity_probe.rs` |
| **Clifford conditioning** | Polarity features appended to the 192-D conditioning vector (indices 176–191) via `adapt_for_group_clifford` | `dimension/manager.rs` |
| **Spawn threshold** | Sentiment lattice groups use a lower spawn threshold (0.92 vs 0.97) to preserve finer polarity distinctions | `cli_impl.rs` (train path) |
| **Context guardrails** | Lexical polarity rules for domain-ambiguous words (`disgusting` in gaming vs theater, crypto promotional register → neutral), garble rejection via TOML `hard_reject_substrings` | `inference_sentiment_core.toml`, `sentiment_generation_lexicon.toml` |

### Causal AI

Causality as a first-class field on training data, with connector-aware retrieval and a path toward Brain B (relationship-only lattice). Implementation: `dimension/language.rs` (`CausalAnnotation`), `inference/causal_hints.rs`, `inference/causal_relation.rs`, `inference/world_grounding.rs`, `inference/grounding_expand.rs`; project scaffold [`scripts/causal-relationship.gf.toml`](scripts/causal-relationship.gf.toml).

- **`CausalAnnotation`** on JSONL rows: `causal_type`, `connector`, `cause_span`, `effect_span`, optional `causal_subtype`
- **`gfcausal_t_*_c_*`** index tokens injected before the BM25 witness, aligning training and inference
- **`causal_hints`**: connector detection → causal BM25 tokens at inference time (no separate brain needed)
- **Layer 0 world grounding**: `data/inference/world_grounding.toml` — typed concept nodes/edges with bounded BFS for query enrichment before lattice retrieval
- **`grounding_expand`**: sparse query expansion (e.g. loss+game → gambling lexicon) via `data/inference/grounding_expand.toml`

### Two-Level Lattice Hierarchy (E8 + Leech)

A mathematically optimal quantization hierarchy using the two provably densest sphere packings:

**E8 lattice (dimension 8, local)** — the 64d bridge embedding decomposes as 8 × 8d subspaces, each quantized to the E8 lattice (Viazovska, Fields Medal 2022):
- Optimal archetype selection: O(64) nearest-point decode, kissing number 240
- Algebraically exact Hopf transitions: E8 root inner products replace heuristic cosine similarity
- Native error correction: Extended Hamming [8,4,4] — 2-bit detection, 1-bit correction

**Leech lattice (dimension 24, global)** — the densest sphere packing in 24d (Cohn et al., 2017), constructed from 3×E8 + Golay code glue:
- **ProjectModel**: Hybrid embedding pipeline (structural AST-lite + semantic hash projection + relational graph + git co-change + test/quality + pattern fingerprint) maps files, functions, types, and modules to 24d Leech-quantized embeddings
- **CodeAnalyzer** parses 8 languages (Rust, Python, TypeScript, JavaScript, Go, C, C++, Java); auto-indexes sub-entities (functions, types) from declarations
- **GitHistory** populates edit-correlation channel from `git log` when `.git` is present
- Context-aware generation: nearest-neighbor queries condition the brain on related codebase entities
- Native error correction: Extended Golay [24,12,8] — 3-bit correction, 4-bit detection

REPL: `/index <path>` to index a project (auto-loads git history when `.git` found), `/project [file]` to query related entities. Implementation: `spectral.rs` (`E8Lattice`, `LeechLattice`, `ProjectModel`, `CodeAnalyzer`, `GitHistory`).

### Semantic Token Dictionary (Implemented)

The `TokenDictionary` orders vocabulary tokens using **distributional semantics** rather than alphabetical clustering:

1. **Co-occurrence vectors** — for each token, a vector of co-occurrence counts with every other token within a 5-token context window across the training corpus
2. **Greedy nearest-neighbor chain** — tokens are arranged so the most semantically similar token is always adjacent (cosine similarity on co-occurrence vectors)
3. **Gray coding** — adjacent tokens differ by exactly 1 bit

This means a 1-bit error in the algebraic generation lands on a **semantically related word** instead of garbage. "build" is adjacent to "construct" and "create", not "binary" (which happens to share a first letter). The result: Observer pattern generation went from conf 0.57 (wrong pattern, garbled) to **conf 0.92** (correct, fully legible) after switching from first-character clustering to semantic ordering.

### Tool-Use Agent (Implemented)

The REPL is a working **text-based agent** that routes between conversation (g0) and inline tool execution:

| Tool | Trigger | Execution |
|------|---------|-----------|
| `calculator` | "calculate 347 * 892" | Recursive-descent arithmetic parser |
| `file_reader` | "read file src/main.rs" | `std::fs::read_to_string` with 50-line preview |
| `code_runner` | "run this python: print(sum(range(100)))" | `std::process::Command` with stdout capture |
| `web_search` | "search for rust async patterns" | Stub (returns query acknowledgment) |

Tool results are fed back through `generation_with_tool_result` for a g0-quality conversational wrapper. The `ToolRegistry` supports custom tool registration for domain-specific agents.

### Base Agent + Custom Training (18 MB deployable brain)

The micro-brain at **18 MB** is small enough for browser (WASM), mobile, IoT, and edge deployment. The architecture supports a **base-agent-plus-augmentation** model:

1. **Ship the base brain** — g0 conversation, tool routing, identity (18 MB)
2. **Users train domain groups** on their own data — when new capacity is added as **separate consolidated groups** and the base is **frozen**, user updates do not rewrite base weights (same “new subgraph + freeze” idea as Split MNIST, not a blanket theorem about all possible wiring)
3. **Retention of base behavior** depends on routing and whether inference paths stay on frozen parameters; the intended deployment story matches the continual-learning demos (isolated frozen regions)
4. **Export the augmented brain** — compact because only new structure is added

### Fragment Composition

Free-text generation by composing typed sentence fragments instead of emitting whole training sentences verbatim. Three reflective voices — **Identity** (persona/OCEAN), **Activity** (action content), **Drive** (state-gated needs) — are assembled from `[opener?] body+ [coda?]` slots with intent affinity, OCEAN scoring, and runtime state gates.

| Piece | Where |
|-------|-------|
| Composer | `fragment_composer.rs` |
| TOML policy | `[fragment_compose]` in inference TOML (`inference/inference_toml.rs`) |
| Service hook | `LanguageService::try_fragment_compose` (`service.rs`) |
| Examples | `examples/decompose_fragments.rs`, `examples/fragment_compose_demo.rs` |

### Reflective Field Stack

Conditioning-space composition that pairs with fragment composition:

| Module | Role |
|--------|------|
| `reflective_field.rs` | Per-voice blend weights (`ReflectiveWeights`) from OCEAN + runtime state |
| `drive_field.rs` | State-gated needs (hunger/energy/mood) that bias Drive-voice selection |
| `basal_ganglia.rs` | Action selection / gating layer between routing and generation |

A/B harnesses: `examples/reflective_field_ab.rs`, `examples/drive_field_ab.rs`, `examples/basal_ganglia_ab.rs`.

### Brain Package + Portable Runtime

Brains ship as versioned binary packages (`brain.rs`):

- Magic `GWFBRPKG`, format v1 (header + checkpoint + personality) or v2 (+ UTF-8 TOML plugins blob)
- Parsed on `LanguageService::load_brain`; plugins manifest → `InferenceHarness`

Lean inference without the full training CLI:

| Piece | Command / API |
|-------|---------------|
| `runtime.rs` | `Runtime::from_brain_bytes` — prompt, converse, codegen; native + wasm32 |
| `growformer-runtime` | `cargo build --release --bin growformer-runtime --no-default-features` then `growformer-runtime brain.bin "prompt"` |

### Categorical Training (optional)

Category-theory scaffolding for compositional generalization (Pythagoras nodes, bifunctor forward, sentiment functor). Behind Cargo feature **`categorical`** (not in default features). See [`src/category/README.md`](src/category/README.md); example: `cargo run --example categorical_sentiment_train --features categorical`.

### ARC-AGI (experimental)

Clifford-encoded grid reasoning solvers for ARC-style tasks:

| Binary | Purpose |
|--------|---------|
| `growformer-arc` | Full ARC brain pipeline (`arc_brain.rs`, `arc_dsl.rs`) |
| `growformer-arc-agi-2x2` | 2×2 (and optional 3×3) benchmark runner (`arc_agi.rs`) |
| `growformer-demos --arc-agi` | Demo entry point |

### Distribution (SpaceKit)

Production deployments embed Growformer via the library API (`growformer::run_cli_with_entitlement`) with capability gating in `entitlement.rs` (train / infer / merge). The standalone `growformer` binary is for crate development; end users typically invoke training and inference through SpaceKit.

---

## **Origin**

This project emerged from a close reading of the Harvard/Google connectome mapping project: a 10‑year effort to map one cubic millimeter of human brain tissue. The result was 57,000 cells, 150 million synapses, and 1.4 petabytes of raw data — from a speck smaller than a grain of rice.

The mapping revealed things textbooks don’t contain. One neuron had over 5,000 connection points. Some axons had coiled into tight whorls for unknown reasons. Pairs of cell clusters grew in mirror images of each other. Jeff Lichtman, the Harvard lead, described “a chasm between what we already know and what we need to know.”

This project asks a direct question: **what would a neural network look like if it tried to close even a small part of that chasm?**

Standard artificial neurons are defined by a single scalar — a weight, updated by gradient descent. A biological neuron has at minimum six active dimensions: connection strength, geometry, timing behavior, metabolic cost, variable connectivity, and structural group relationships. We’re modeling one dimension of a six‑dimensional object.  
**This codebase implements all six.**

---

## **The Six Dimensions**

| Dimension | Type | What It Does |
|-----------|------|--------------|
| 1. Weight | `f32` | Classical synaptic strength, updated by backprop |
| 2. Geometry | `Vec3` | Position in 3D embedding space; neurons that fire together drift closer |
| 3. Timing | `f32` (last fired + decay) | Spike‑timing‑dependent plasticity; *when* a signal arrives matters |
| 4. Metabolic Cost | `f32` | Each synapse has a running cost; neurons over budget prune their weakest connections |
| 5. Connectivity Density | `Vec<Synapse>` | Connections form and dissolve dynamically based on proximity and budget |
| 6. Structural Group | `GroupId` | Neurons belong to groups that can be paired as mirrors, developing correlated structure |

---

## **Architecture**

```
growformer/
├── Cargo.toml
├── examples/                — Fragment compose/decompose, reflective/drive/basal A/B, categorical train
├── scripts/*.gf.toml        — Project manifests (sentiment, fintech, crypto, code, causal)
└── src/
    ├── lib.rs               — Library crate (shared by all binaries + WASM)
    ├── main.rs              — Dev CLI entry → `run_cli` (no entitlement gate)
    ├── cli_impl.rs          — Full train / infer / merge / init CLI (`run_from_argv`)
    ├── runtime_main.rs      — `growformer-runtime` binary entry
    ├── demos.rs             — Demos & benchmarks binary
    ├── server.rs            — HTTP server (`growformer-node`)
    ├── service.rs           — LanguageService: routing, generation, OCEAN, fragment compose
    ├── runtime.rs           — Portable inference API (`Runtime::from_brain_bytes`)
    ├── brain.rs             — Brain package envelope (GWFBRPKG v1/v2 + plugins blob)
    ├── project_gf.rs        — `*.gf.toml` project manifest parser
    ├── entitlement.rs       — SpaceKit capability gate (train / infer / merge)
    ├── tools_builtin.rs     — calculator, file_reader, code_runner, web_search executors
    ├── fragment_composer.rs — Typed fragment assembly (Identity / Activity / Drive voices)
    ├── reflective_field.rs  — Per-voice conditioning weights from OCEAN + state
    ├── drive_field.rs       — State-gated needs biasing Drive voice
    ├── basal_ganglia.rs     — Action selection / gating
    ├── micro_brain.rs       — MetaBrain + Paramecium micro-brains (centroid coordinator)
    ├── predictive_coder.rs  — Predictive coding scaffold
    ├── topic_graph.rs       — Topic relationship graph
    ├── text_autoencoder.rs  — Text autoencoder experiments
    ├── types.rs, neuron.rs, environment.rs
    ├── spectral.rs          — TokenDictionary, E8/Leech, ProjectModel, CodeAnalyzer, GitHistory
    ├── clifford.rs          — Cl(1,7) SpaceTime Algebra
    ├── growformer_lang.rs   — MetaConcept, MetaCodebook, LanguageProjector
    ├── reasoning.rs         — CognitiveMap, System 1.5, deliberate System 2 (WorkingMemory)
    ├── metacognition.rs, coherence.rs, understanding.rs, cloze.rs
    ├── arc_agi.rs, arc_brain.rs, arc_dsl.rs  — ARC-AGI solvers (cli feature)
    ├── category/            — Categorical DAG trainer (`--features categorical`; see README inside)
    ├── active_inference/    — BeliefState, Markov blanket, episode spine, replay harness
    ├── inference/
    │   ├── harness.rs, inference_toml.rs, manifest.rs, inference_guardrails.rs
    │   ├── causal_hints.rs, causal_relation.rs, world_grounding.rs, grounding_expand.rs
    │   ├── retrieval_rescore.rs, retrieval_lexicon.rs, sentiment_generation_lexicon.rs
    │   └── plugins/         — LatticeShortcutsPlugin + `default_inference_harness()`
    ├── dimension/
    │   ├── manager.rs       — DimensionManager: Main/Mirror dims, conditioning, promotion
    │   ├── main_dim.rs, mirror_dim.rs, promotion.rs, observer.rs
    │   ├── router.rs, action_classifier.rs, paramecium.rs, group_gen.rs, generation_head.rs
    │   ├── generation.rs, codegen.rs, composition.rs, embedding.rs, policy.rs
    │   ├── language.rs, polarity_probe.rs, action.rs, tool.rs
    └── systems/
        ├── metabolic.rs, growth.rs, stdp.rs, geometry.rs, whorls.rs, mirror.rs
        └── checkpoint.rs    — Legacy neural-env serialization (demos)
```

### Inference plugins (`src/inference/`)

Optional **inference-time** behavior is implemented as compile-time plugins behind **`InferenceHarness`**, not as dynamic `.so` loads. The brain package may embed a UTF-8 TOML **`BrainPluginsManifest`** (`plugins_blob` on v2 packages). Numeric gates are stored under the **`[sentiment]`** table name for existing brain exports; Rust exposes them as **`InferenceThresholds`** (`manifest.rs`).

Shortcut **lists** load at **runtime** from disk. Resolution order (native): **`--inference-toml`** / **`--project`** `*.gf.toml` **`[inference]`** table (registered via `inference::set_inference_toml_cli_paths`), then env **`GROWFORMER_INFERENCE_TOML`** (legacy: **`GROWFORMER_SENTIMENT_INFERENCE_TOML`**), then the first readable file among **default relative paths** under cwd, next to the binary, and `../data/...` from the exe. Built-in order: `data/sentiment/inference_sentiment_core.toml` (customer-safe defaults), then `data/fintech/inference_fintech.toml`. Internal sentiment fixtures + PR-wire examples live in `data/sentiment/inference_sentiment_reference.toml` (opt-in via `[inference].toml`). Prepend or reorder with comma-separated **`GROWFORMER_INFERENCE_TOML_DEFAULT_RELS`**. Merge baseline: **`--inference-defaults-toml`** / **`[inference].defaults_toml`**, then **`GROWFORMER_INFERENCE_DEFAULTS_TOML`**, then the same default-path search. **Optional guardrails JSONL** (extra `lexical_topic` / `lattice_misfire` lines, merged **after** TOML): **`--inference-guardrails-jsonl`** / **`[inference].guardrails_jsonl`**, else **`GROWFORMER_INFERENCE_GUARDRAILS_JSONL`**, else every existing file among **`data/sentiment/inference_guardrails.jsonl`** and **`data/fintech/inference_guardrails.jsonl`** under the same cwd / exe search roots (see `src/inference/inference_guardrails.rs`). **wasm32** skips JSONL (no `std::fs`) and uses a compile-time include of the sentiment TOML only.

### Project manifests (`*.gf.toml`)

Versioned TOML (currently **`schema_version = 1`**) so one file can drive train paths, **`[inference]`** rule files, and default **`--brain`** for **`--infer`**. Paths are resolved **relative to the manifest’s directory** unless absolute.

- **Scaffold:** `cargo run --release -- init [PATH] [--name MyBrain]` writes a starter file (default output `Growformer.gf.toml`).
- **Examples:** [`scripts/sentiment-analysis.gf.toml`](scripts/sentiment-analysis.gf.toml), [`scripts/fintech-sentiment-analysis.gf.toml`](scripts/fintech-sentiment-analysis.gf.toml), [`scripts/crypto-sentiment-analysis.gf.toml`](scripts/crypto-sentiment-analysis.gf.toml), [`scripts/causal-relationship.gf.toml`](scripts/causal-relationship.gf.toml) — train with  
  `cargo run --release -- --train-brain --project scripts/sentiment-analysis.gf.toml`  
  (run from the `growformer` crate root so `../data/...` resolves).
- **Code-only brain:** [`scripts/code.gf.toml`](scripts/code.gf.toml) sets **`[train].code_brain = true`**, which enables **GrowformerLang MetaCodebook (training Stage 2b)** and **per-group code lattices** from **`expected_code`** in JSONL. Default training leaves these **off** so support/sentiment brains are not pulled into code-generation plumbing. Same opt-in via **`--train-code-lattice`** without a project file.
- **Brain merge:** `--merge-brain --overlay-brain <path>` composes an overlay checkpoint onto a base brain (entitlement-gated when embedded).
- **CLI flags** still override merged values when you pass them explicitly (e.g. **`--inference-toml`** wins over **`[inference].toml`** in the project file).

Parser: **`src/project_gf.rs`**.

- **`LanguageService`** holds **`inference_harness`**, initialized with **`default_inference_harness()`** (currently registers **`LatticeShortcutsPlugin`**). Generation, meta-route guards, subject-keyword merging, coherence/metacog skips, and **`export_brain`** defaults are dispatched through the harness so **`service.rs`** stays orchestration-focused.
- **Borrowing:** Inside the main generation path, code clones **`inference_harness`** before **`active_dm_mut()`** (same pattern as hoisted plugin config) so the harness can run while the active **`DimensionManager`** is mutably borrowed.
- **Extending:** Implement **`BrainInferencePlugin`** in `src/inference/plugins/`, override only the hooks you need (defaults are no-ops), and append **`Box::new(YourPlugin)`** in **`plugins/mod.rs`** **`default_inference_harness()`**, or replace **`svc.inference_harness`** with **`InferenceHarness::new(vec![...])`** for a custom registry.

Binaries share one library (`LanguageService` / `Runtime`):

| Binary | Command | Purpose |
|--------|---------|---------|
| `growformer` | `cargo run --release` | Dev CLI: train, infer, merge, init (no entitlement gate) |
| `growformer-cli` | `cargo run --bin growformer-cli` | Same CLI surface as `growformer` (internal alias) |
| `growformer-demos` | `cargo run --bin growformer-demos` | Demos, benchmarks, M3–M6 gate commands |
| `growformer-node` | `cargo run --bin growformer-node` | HTTP API server |
| `growformer-runtime` | `cargo build --release --bin growformer-runtime --no-default-features` | Lean inference-only REPL / single-shot |
| `growformer-arc` | `cargo run --bin growformer-arc` | ARC brain pipeline |
| `growformer-arc-agi-2x2` | `cargo run --bin growformer-arc-agi-2x2` | ARC 2×2 benchmark runner |

The library compiles to `wasm32-unknown-unknown` with `--no-default-features` (optionally `--features wasm-bindgen` for JS bindings in `wasm.rs`).

---

## **WASM Support**

The Growformer library compiles to WebAssembly. Only the **lib crate** targets WASM — the CLI and Node binaries are native-only.

### Feature flags

| Feature | Default | What it enables |
|---------|---------|-----------------|
| `native` | yes | `reqwest` HTTP encoder, filesystem checkpoint loading |
| `server` | yes | `axum`/`tokio` HTTP server (`growformer-node` binary) |
| `parallel` | yes | `rayon` parallel iterators in compute path |
| `cli` | yes | `clap`, `indicatif`, `mnist`, `kiddo`, `cli_impl`, `project_gf`, ARC modules |
| `standalone_cli` | no | `growformer-cli` binary (same as `growformer` entry) |
| `categorical` | no | `category/` DAG trainer + `categorical_sentiment_train` example |
| `training` | no | `gradient_memory`, `training_objectives` (training-only helpers) |
| `wasm-bindgen` | no | `wasm.rs` JS bindings (`growformer_acceptance_report`, etc.) |

WASM / lean builds disable default features:

```bash
cargo check --lib --no-default-features --target wasm32-unknown-unknown
```

### WASM API usage

**Low-level** (`LanguageService`):

```rust
use growformer::dimension::LanguageConfig;
use growformer::service::LanguageService;

let config = LanguageConfig::default();
let mut svc = LanguageService::new_with_config(config)?;

// Optional: load GLE checkpoints from pre-fetched bytes
svc.load_gle_students_from_bytes(&[&checkpoint_bytes])?;

let action = svc.action("implement a rust web server")?;
let (_action, response) = svc.generation("help me reset my password")?;
let (_action, code) = svc.codegen("implement a rust web server")?;
```

**Portable** (`Runtime` — preferred for embeds):

```rust
use growformer::runtime::Runtime;

let mut rt = Runtime::from_brain_bytes(&brain_bytes)?;
let response = rt.prompt("help me reset my password")?;
```

With `--features wasm-bindgen`, `wasm.rs` exposes JS-callable wrappers.

### What is gated behind `#[cfg(not(target_arch = "wasm32"))]`

- `LanguageService::new_default()` (reads env vars — use `new_with_config()` instead)
- `GleStudentCheckpoint::load()` (reads filesystem — use `from_bytes()` instead)
- All `save_*`/`load_*` checkpoint functions (use `serialize_checkpoint_to_bytes()`/`deserialize_checkpoint_from_bytes()`)
- `HttpGleEncoder` (requires `reqwest`)
- `pub mod mnist` (depends on `mnist` crate filesystem I/O)

### What is gated behind `#[cfg(feature = "parallel")]`

- `rayon` parallel iterators. When disabled, all compute falls back to sequential `.iter()`.

### Entropy

WASM builds automatically pull in `getrandom` with the `js` feature for `rand` entropy via `crypto.getRandomValues()`.

---

## **Growformer Language Encoder (GLE)**

GLE is the in-house language front-end for Growformer. It is a distilled student encoder
trained and tuned for organism routing, then projected through the language bridge into the
shared 64-d routing space.

- No external transformer runtime is required for inference.
- No third-party hosted model dependency is required for routing.
- Checkpoints are private artifacts produced by this repo:
  - `checkpoints/gle_student_base.json`
  - `checkpoints/gle_student_routing_tuned.json`
- Runtime can load one or many local GLE checkpoints:
  - Single: `GROWFORMER_GLE_CHECKPOINT=checkpoints/gle_student_routing_tuned.json`
  - Ensemble: `GROWFORMER_GLE_CHECKPOINTS=checkpoints/gle_student_base.json,checkpoints/gle_student_routing_tuned.json`
  - Optional ensemble weights: `GROWFORMER_GLE_WEIGHTS=0.3,0.7`

Latest internal routing run with `gle_student_routing_tuned.json` (**in-distribution status, not a certification**):

- Intent accuracy: `100%`
- Median routing margin: `1.828`
- P10 routing margin: `1.818`
- OOD AUROC: `1.000`
- OOD FAR: `0.00%`

> **Certification status: UNCERTIFIED.** These are in-distribution status numbers, not a
> generalization certification. A 100% / AUROC-1.000 card reports none of the four things the
> certifier-first contract requires, and each is where a perfect score goes to die:
>
> 1. **Feature-disjoint held-out?** A 100% on an eval that shares surface tokens with training is
>    the 20.7% mirage at the easy end (see §14 of `docs/GROUNDING_LOOP_SPEC.md`).
> 2. **Provenance-clean eval?** If the eval set is distilled/augmented from the same distribution
>    the GLE trained on, the encoder certifies itself — the firewall's entire reason to exist (§4).
> 3. **Shuffle floor?** If a label-permuted GLE also scores high, the task is trivial and 100%
>    means nothing.
> 4. **Positive control collapses?** If lexical CATA also scores well on the same eval, the eval
>    is lexically separable and isn't testing semantics at all.
>
> The GLE is a drop-in encoder via the BYO-vectors hook and is judged by the **same**
> certifier-first gate as every other encoder — no pass because it's ours and we want it to work.
>
> **It has now been run through the gate in two places.**
>
> *Cross-domain (`--certify-encoder gle`, eval = Luna companion traffic):* `BELOW_RESOLUTION`,
> **pooled 1.1%**. This is a *correct* measurement of an irrelevant question — a support/coding
> encoder asked to route pet-companion traffic it was never built for. Not a mirage; just the
> wrong domain. It does **not** bear on the 100%.
>
> *In-domain (`--certify-gle-indomain`, the actual headline):* the gate now points at the 100% on
> its own support/coding turf, two ways — (A) the literal 2-way `support`-vs-`coding` task that
> *produced* the 100%, and (B) ~15-way `action_target` routing on the same home data. Both return
> **`INVALID`**, with the decisive reason being a verdict about the *eval*, not the encoder:
> *no feature-disjoint held-out phrases exist at any granularity* — every held-out phrase shares
> features with its own class's training, so the eval cannot separate memorization from
> generalization. Construction A reproduces **pooled 100%** faithfully and confirms it is a
> **2-way, lexically-entangled** number (the distillation teacher is a *hashing* proxy — lexical at
> the root); Construction B routes at **15.5%** (~2.3× chance) on 15 home-domain classes. The 100%
> is not refuted — it is shown to be **uncertifiable on the eval that produced it**, which is the
> sharper finding. To certify the in-domain claim at all, the data need is a *feature-disjoint*
> support/coding held-out set (concept-preserving, surface-disjoint paraphrases).
>
> No encoder has emitted a `PASS` artifact with `disjoint_semantic_lift` clear of the shuffle floor
> on provenance-clean, feature-disjoint held-out traffic, so "100% intent accuracy" still has the
> standing the 20.7% pooled number had before §14 caught it: **you can run it, you cannot yet trust
> it.** See `docs/GROUNDING_LOOP_SPEC.md` §15 / §15.1 and the certifier commands below.

This makes the language stack practical for private, low-latency, CPU-friendly routing:

`text -> GLE semantic vector -> bridge -> 64-d routing vector -> Growformer group dispatch`

Current language milestone status (**"complete" = the code path runs and passes its
in-distribution checks; it is not a held-out/disjoint generalization certification**):

- M1 (Language Embedding Foundation): complete
- M2 (Embedding-First Routing Validation): complete
- M3 (Intent-to-Action Layer): complete
- M4 (Controlled Language Generation, template-only): complete
- M5 (Continual Language Learning Integration): complete — *in-distribution only. The CMI work
  shows routing/integration over frozen specialists does **not** yet generalize on held-out, so
  "complete" here means the path runs and retains in-distribution, not that continual learning is
  solved.*
- M6 (Production Agent Modes): complete — *production status is not validation; running ≠
  generalizing.*

Gate commands (run via `growformer-demos`):

- M3: `cargo run --bin growformer-demos -- --validate-action-schema --action-eval-data data/language/stage_ab_action_eval_extended.jsonl`
- M4: `cargo run --bin growformer-demos -- --validate-generation --action-eval-data data/language/stage_ab_action_eval_extended.jsonl`
- M5: `cargo run --bin growformer-demos -- --m5-retention-eval --m5-retention-plan data/language/m5/retention_eval_splits_full.json`
- M6: `cargo run --bin growformer-demos -- --acceptance-report`

Operational commands (run via `growformer-demos`):

- Print model card: `cargo run --bin growformer-demos -- --print-gle-card checkpoints/gle_student_routing_tuned.json`
- M3 starter action JSON: `cargo run --bin growformer-demos -- --language-action-text "help me reset password"`
- M5 starter code generation: `cargo run --bin growformer-demos -- --language-code-text "implement binary search in rust"`
- M6 acceptance report: `cargo run --bin growformer-demos -- --acceptance-report --acceptance-report-path reports/m6_acceptance.json`

Encoder certification (the contract every encoder — GLE included — is judged by; see
`docs/GROUNDING_LOOP_SPEC.md` §15):

- Certify an encoder, emit verdict artifact: `cargo run --release --bin growformer-demos -- --certify-encoder supervised <companion_dir>`
- Positive-control / lexical baseline: `cargo run --release --bin growformer-demos -- --certify-encoder cata <companion_dir>`
- Certify the GLE on its **own** support/coding domain (the 100%): `cargo run --release --bin growformer-demos -- --certify-gle-indomain`
- Re-read / compare a verdict: `cargo run --release --bin growformer-demos -- --certify-verdict certify_verdict_latest.json`
- Go/no-go field: `disjoint_semantic_lift` (disjoint-bin accuracy − shuffle-floor 95th pct). No
  encoder is promoted without a `PASS` artifact; pooled accuracy never gates.

M5 datasets:

- Stage C (multi-turn stateful):
  - `data/language/m5/train_multi_turn.jsonl` (24 samples)
  - `data/language/m5/eval_multi_turn_holdout.jsonl` (20 samples)
- Stage D (adversarial/noisy):
  - `data/language/m5/train_adversarial.jsonl` (24 samples)
  - `data/language/m5/eval_adversarial_holdout.jsonl` (20 samples)
- Full 7-domain retention plan: `data/language/m5/retention_eval_splits_full.json`
  - Train order: Python -> Rust -> JavaScript -> Design Patterns -> Architectural Patterns -> Multi-turn -> Adversarial
  - Retention target: post-sequence ratio `>= 0.97` per domain
  - Latest result: `mean_retention_ratio=1.000` across all 7 domains
- Automatic Mirror spawn trigger: K=10 consecutive low-confidence routing batches
  - Configured via `DimensionManager::auto_spawn_k` and `auto_spawn_threshold`

M5 coding datasets (original):

- Python train set: `data/language/m5/train_python_coding.jsonl`
- Rust train set: `data/language/m5/train_rust_coding.jsonl`
- JavaScript train set: `data/language/m5/train_javascript_coding.jsonl`
- Holdout eval sets:
  - `data/language/m5/eval_python_holdout.jsonl`
  - `data/language/m5/eval_rust_holdout.jsonl`
  - `data/language/m5/eval_javascript_holdout.jsonl`
- Sequential retention plan:
  - `data/language/m5/retention_eval_splits.json`
  - Train order: Python -> Rust -> JavaScript
  - Retention target: post-sequence ratio `>= 0.97` per domain
- Curriculum template for systematic data collection:
  - `data/language/m5/CURRICULUM_V1_TEMPLATE.md`
- Pattern-subject datasets:
  - `data/language/m5/train_design_patterns.jsonl`
  - `data/language/m5/eval_design_patterns_holdout.jsonl`
  - `data/language/m5/train_architectural_patterns.jsonl`
  - `data/language/m5/eval_architectural_patterns_holdout.jsonl`
  - `data/language/m5/retention_patterns_eval_splits.json`
  - Current size: train `48 + 48`, holdout `24 + 24`
  - Expanded benchmark run:
    `cargo run --bin growformer-demos -- --m5-retention-eval --m5-retention-plan data/language/m5/retention_patterns_eval_splits.json --m5-epochs 100 --m5-lr 0.12 --m5-feature-dim 1024 --m5-replay-per-epoch 64 --m5-replay-prior-ratio 0.9 --m5-retention-report reports/m5_retention_patterns_report_v9.json`

Benchmarks:

- Language/code benchmark suite (repeated runs with latency + RSS):
  - `bash scripts/benchmark_language.sh 5 reports/benchmarks`
- Core task benchmark suite (single-run XOR/Spiral/Language pipeline):
  - `bash scripts/benchmark_core_tasks.sh reports/benchmarks`
- Latest measured CLI baseline on this machine (debug build):
  - `--language-action-text`: ~`1.43s` warm average, ~`9.5 MB` max RSS
  - `--language-code-text`: ~`1.43s` average, ~`9.6 MB` max RSS
  - `--language-code-eval` (60 samples): ~`1.53s` average, ~`9.9 MB` max RSS

Growformer Node (HTTP dev server):

- Start server:
  - `cargo run --bin growformer-node`
- Start server with perf JSONL logging:
  - `GROWFORMER_NODE_LOG_PATH=reports/node_perf.jsonl cargo run --bin growformer-node`
- Health:
  - `curl http://127.0.0.1:8080/v1/health`
- Chat/codegen request:
  - `curl -X POST http://127.0.0.1:8080/v1/chat -H "Content-Type: application/json" -d '{"mode":"codegen","message":"implement a web server in rust","options":{"include_raw_stdout":false}}'`
- SSE chat stream:
  - `curl -N -X POST http://127.0.0.1:8080/v1/chat/stream -H "Content-Type: application/json" -d '{"mode":"codegen","message":"implement a web server in rust"}'`
- Runtime note:
  - `growformer-node` now calls Growformer as a shared library in-process (no CLI subprocess per request).

**Multiple brains (checkpoints)**

You can load several trained brains and switch between them. Each brain is a **full checkpoint**: its own router, action classifier, generation head, codegen head, and group layout. There is **no merging** — inference always uses one active checkpoint; switching brain swaps the entire stack.

- **Routing:** Each brain has its own `LearnedRouter`, `ActionClassifier`, `DimensionManager::main` (groups, embedding library), and generation/codegen heads. When you select a brain, that brain’s router and classifier handle routing and action type; that brain’s heads produce text and code. Heads are conditioned on the **routed group** (group one-hot) so the correct region-specific attractor is used; brains trained with the current `--train-brain` pipeline include this binding. No averaging or merging of parameters across brains.
- **Loading:**
  - **Single brain (legacy):** `GROWFORMER_BRAIN_PATH=brain.bin` (or default `brain.bin`) — loads one checkpoint as `"default"`.
  - **Multiple brains:** `GROWFORMER_BRAIN_DIR=micro-brains` — loads every `*.bin` in that directory; each file is registered under its stem (e.g. `my-brain.bin` → `"my-brain"`, `user-a-brain.bin` → `"user-a"`). The first name (alphabetically) is set active at startup.
- **API:**
  - `GET /v1/brains` — returns `{ "brains": ["default", "my-brain", "user-a"], "active": "my-brain" }`.
  - In `POST /v1/chat`, optional body field `"brain": "user-a"` — uses that checkpoint for the request and sets it active for subsequent requests until changed.
- **Library:** `LanguageService` has `load_brain(data)` (single default), `load_brain_as(name, data)` (additional named checkpoint), `list_brains()`, `set_active_brain(name)`, and `active_dm()` for inspection. Inference (action, generation, codegen) always uses the active checkpoint.

Example: train or obtain `my-brain.bin` and `user-a-brain.bin`, put both in `micro-brains/`, set `GROWFORMER_BRAIN_DIR=micro-brains`, start the node — then call `GET /v1/brains` and pass `"brain": "user-a"` in the chat body when you want that subject’s stack.

M6 Agent Modes:

- Two modes share one backend:
  - **ContextFile** — retrieval-augmented agent. Injects context snippets, reads micro-brain episodic summaries (read-only).
  - **MicroBrain** — trained compact brain agent. Routes via Growformer language pipeline, consumes retrieval snippets if available.
- Shared-state contract:
  - Context-file mode may read episodic summaries via `read_episodic_summaries()`.
  - Micro-brain mode may consume retrieval snippets via `context_snippets()`.
  - Raw episodic memory is never directly mutated by context-file mode.
  - Every cross-mode handoff is logged with mode origin, confidence, reason, and timestamp.
- API endpoints:
  - `POST /v1/mode` — switch between `context_file` and `micro_brain`
  - `GET /v1/acceptance` — full M6 acceptance report JSON
  - `GET /v1/health` — now includes `agent_mode` field
  - `POST /v1/chat` — now accepts optional `agent_mode` and `context_snippets` fields
- SLO tracking:
  - Latency P95 tracked per inference call (configurable via `SloConfig`)
  - Checkpoint domain count tracked
  - Acceptance report includes pass/fail against SLO targets
- CLI: `cargo run --bin growformer-demos -- --acceptance-report --acceptance-report-path reports/m6_acceptance.json`

---

## **The Six Systems**

### **System 1 — Metabolic Pruning (`systems/metabolic.rs`)**

Each synapse has a running energy cost proportional to its effective strength. When a neuron exceeds its energy budget, it prunes its weakest synapse. This is not random dropout — it is cost‑driven selection pressure. Strong synapses are never pruned; instead the budget expands slightly, modeling the biological reality that useful connections attract more resources.

---

### **System 2 — Dynamic Synapse Growth (`systems/growth.rs`)**

Neurons within spatial proximity and with budget headroom can form new connections. Initial strength is inversely proportional to distance, with small random noise. A slow sweep removes synapses that never developed strength after a minimum age. Structure emerges under pressure rather than being fixed at initialization.

---

### **System 3 — STDP: Spike‑Timing‑Dependent Plasticity (`systems/stdp.rs`)**

Timing determines whether a synapse is strengthened or weakened:

- Pre fires **before** post → causal → strengthen  
- Post fires **before** pre → acausal → weaken  
- Outside the STDP window → no effect  

STDP runs alongside backprop. Both update synaptic strength — one from global error, one from local timing.

---

### **System 4 — Geometry Update (`systems/geometry.rs`)**

Neurons that activate together are pulled spatially closer each tick. Pull strength is proportional to synapse strength × activation correlation. A soft boundary prevents unbounded drift. Spatial position feeds back into System 2: neurons that drift into proximity become candidates for new synapses.

---

### **System 5 — Whorl Detection (`systems/whorls.rs`)**

The Harvard mapping found axons coiling into tight whorls for unknown reasons. We model this as geometric self‑reference: a cycle in the connection graph whose participants are also spatially colocated. Detected whorls are reported but not removed — they may represent stable attractor states or emergent memory loops.

---

### **System 6 — Mirror Group Coupling (`systems/mirror.rs`)**

**This is not the same thing as “spawn a Mirror” in the opening paragraphs.** Here, **mirror** means **paired `NeuronGroup`s** linked for **IFS (iterated function system) mirror coupling** during training: neurons are assigned **pairwise counterparts** across the two groups, and positions are nudged so each neuron moves toward the **individual reflection** of its partner across the midplane between the two group centroids (not collapsed to a shared centroid). Coupling is expressed **through geometry** (which then biases where new synapses can form); direct weight-copy averaging was removed so internal diversity is preserved. A `mirror_symmetry_score` tracks how symmetric the pair has become. Use this when you want **two populations to develop complementary, reflected structure** in shared training — orthogonal to the **Mirror dimension → promote → freeze** lifecycle used for new tasks in `DimensionManager`.

---

## **Training Loop**

All six systems compose in a fixed order each tick:

```rust
pub fn train_tick(&mut self, input: &[f32], target: &[f32], rng: &mut impl Rng) -> TickResult {
    let output = self.forward_pass(input, true, rng);  // temporal-gated forward (dropout on)
    let loss   = self.backprop(&output, target);       // dim 1: weight update
    record_firing(...);
    if self.config.stdp_enabled { update_stdp_layer(...); }  // dim 3
    apply_metabolic_pressure(...);                     // dim 4: prune over-budget synapses
    if tick_count % prune_interval == 0 {
        prune_three_phase(...);                        // age/dormancy-based structural pruning
        potentiate_active_synapses(...);               // strengthen well-used synapses
    }
    if tick_count % geometry_interval == 0 { update_geometry(...); }  // dim 2
    grow_synapses(...);                                // dim 5: form new connections
    update_group_centroids(...);
    apply_ifs_mirror_coupling(...);                    // dim 6: structural symmetry
    self.time += 1.0;
}
```

The forward pass itself is non‑standard: signal strength is gated by a temporal decay function so that recently fired neurons carry more weight.

---

## **Quick Start**

### Train a brain

```bash
# Auto-configure everything from the dataset (recommended)
cargo run --release -- --train-brain --auto

# Optional metadata embedded in the exported brain package header (name, description, author)
cargo run --release -- --train-brain --auto \
  --brain-name "Support assistant v1" \
  --brain-description "Trained on internal helpdesk + model routing + coding mix" \
  --brain-author "your-org"

# Manual: specify epochs and replicas
cargo run --release -- --train-brain --brain-gen-epochs 3000 --brain-gen-replicas 2

# Quick validation run (capped samples + epochs, asserts inference)
cargo run --release -- --validate-brain-training
```

### Run inference

```bash
# Interactive mode (REPL)
cargo run --release -- --infer

# Single prompt
cargo run --release -- --infer --prompt "explain the observer pattern"

# Custom brain file
cargo run --release -- --infer --brain my_custom.bin --prompt "implement binary search in Python"
```

### Auto-configuration (`--auto`)

When `--auto` is set, the system profiles the training dataset and derives all parameters automatically:
- `MAX_TOKENS` — sized to fit the longest response in the data (with headroom)
- `GEN_HIDDEN` / `GEN_K` — scaled from estimated output dimension
- `gen_epochs` — base schedule by dataset size, with class-imbalance compensation
- `router_epochs` / `classifier_epochs` — scaled by number of groups
- `replicas` — auto-detected from available CPU cores
- **Early stopping** — monitors loss plateau, stops when improvement < 0.3% over 100 epochs

No manual tuning required. Users provide training data; the system configures itself.

### Run demos & benchmarks

Demos run as a separate binary:

```bash
cargo run --bin growformer-demos -- --xor
cargo run --bin growformer-demos -- --spiral
cargo run --bin growformer-demos -- --mnist
cargo run --bin growformer-demos -- --language-pipeline
cargo run --bin growformer-demos -- --help
```

---

## **Configuration**

All behavior is controlled through `EnvironmentConfig` in `types.rs`. A subset of the main knobs:

```rust
pub struct EnvironmentConfig {
    pub max_synapses_per_neuron: usize,
    pub energy_budget_per_neuron: f32,
    pub pruning_threshold: f32,
    pub mirror_coupling_strength: f32,
    pub stdp_window: f32,
    pub geometry_influence: f32,
    pub growth_radius: f32,
    pub learning_rate: f32,
    pub a_plus: f32,
    pub a_minus: f32,
    pub tau_plus: f32,
    pub tau_minus: f32,
    pub geometry_interval: u32,
    pub geometry_noise: f32,
    pub competitive_k: usize,
    pub dropout_rate: f32,
    pub weight_decay: f32,
    pub stdp_enabled: bool,
    pub prune_early_age: u32,
    pub prune_early_threshold: f32,
    pub prune_mid_threshold: f32,
    pub prune_mid_facilitation_floor: f32,
    pub prune_long_age: u32,
    pub prune_long_dormancy: u32,
    pub prune_long_threshold: f32,
    pub prune_interval: u32,
    // ... plus physics (thermal_noise, gravity_g, k_repel, damping, etc.),
    // homeostasis (homeostasis_target, homeostasis_lr), lateral_inhibition,
    // mass–competition coupling, and more — see types.rs for the full struct.
}
```

---

## **Dependencies**

Core (always on): `rand`, `serde`, `serde_json`, `toml`, `aho-corasick`, `instant`.

Optional by feature:

| Feature | Crates |
|---------|--------|
| `cli` | `clap`, `indicatif`, `mnist`, `kiddo` |
| `parallel` | `rayon` |
| `native` | `reqwest` (blocking JSON, rustls) |
| `server` | `axum`, `tokio`, `tower-http`, `async-stream`, `futures-core`, `uuid` |
| `wasm-bindgen` | `wasm-bindgen`, `serde-wasm-bindgen`, `console_error_panic_hook` |

See `Cargo.toml` for pinned versions. Docker/Linux builds: [`DOCKER.md`](DOCKER.md).

---

## **Known Limitations and Next Steps**

- **Forward pass adjacency** is O(n²); replace with reverse adjacency index.  
- **Whorl detection** uses depth‑limited DFS; future versions should track axon geometry.  
- **STDP + backprop** interact but are not unified; a combined update rule is future work.  
- **No recurrence** yet; geometry and growth naturally want to form recurrent loops.  
- **Continual-learning evidence:** publish Split MNIST (and related) results **against standard baselines** on matched splits, plus explicit **latency tables** for generation (single-pass vs. stated autoregressive reference, length, hardware).

---

## **Relationship to the Biological Record**

The Harvard/Google mapping produced 1.4 petabytes from one cubic millimeter. Every major AI model on Earth fits in a fraction of that. The full human brain at that resolution would require ~1.4 zettabytes — roughly all data generated on Earth in a year.

This project does not close that gap.  
What it does is build a training environment where the dimensions that *create* that gap — timing, geometry, metabolic cost, variable connectivity, structural symmetry — are live variables during training rather than fixed architectural choices.

The network’s structure is an **output** of training, not just its weights.

The next step in the biological record is a mouse hippocampus: 10 cubic millimeters over five years.  
The next step here is a reverse adjacency index and a recurrent layer.

---

## **Quantum Biology and Neural Communication**

Recent experimental evidence has established that quantum effects operate at functional scales in biological systems. Quantum coherence in photosynthetic light-harvesting complexes (Fleming et al., 2007), radical-pair entanglement in avian magnetoreception, and proton tunneling in enzyme catalysis all demonstrate that the "too warm and wet" objection to biological quantum mechanics is empirically false.

In neuroscience, three hypotheses extend this to neural communication:

**Penrose-Hameroff (Orchestrated Objective Reduction):** Quantum computations in microtubules within neurons, where wavefunction collapse produces discrete moments of cognition. Bandyopadhyay's lab has measured resonance frequencies in microtubules consistent with this theory.

**Fisher (Posner molecules):** Phosphorus nuclear spins in calcium phosphate clusters serve as neural qubits with coherence times of hours to days — the most physically rigorous proposal, under active experimental testing.

**Electromagnetic toroidal coherence:** Neurons generate electromagnetic fields when they fire. Neural assemblies — groups of co-firing neurons — produce larger-scale fields that naturally adopt **toroidal geometry** (any current loop generates a toroidal magnetic field). A toroidal field is self-contained: it doesn't radiate efficiently outward, which means it could **maintain quantum coherence** by shielding entangled states from environmental decoherence. If neurons within a toroidal field region share entangled electromagnetic states, this provides a **non-electrical communication channel** — information transfer that doesn't depend on synaptic transmission or axonal conduction. This would explain:

- **The binding problem**: How distributed neural activity across separate brain regions becomes unified experience. Non-local correlations from shared entangled states within toroidal field regions provide binding without signal propagation delay.
- **Speed of cognitive processes**: Certain cognitive phenomena (gestalt perception, insight, rapid pattern recognition) occur faster than synaptic transmission chains can account for. Electromagnetic field-mediated quantum correlations operate at the speed of light within the field region.
- **Neural synchrony**: Large-scale neural oscillations (gamma, theta) may reflect the coherent electromagnetic field dynamics of toroidal regions rather than purely synaptic network effects.

**Connection to Growformer's architecture:** The mathematical structures in the Growformer map directly onto this picture:

| Biological hypothesis | Growformer implementation |
|---|---|
| Toroidal field region (coherent neural assembly) | Group (structurally isolated specialist with internal coherence) |
| Entangled states within a field region | Engram consolidation (frozen synaptic traces that deterministically influence output) |
| Non-commutative quantum field interactions between regions | Non-commutative multi-specialist composition (leader/follower ordering) |
| Field isolation between distant assemblies | **Promoted Main groups frozen after training** (no further weight updates on that subgraph); *not* a claim that IFS mirror-coupled groups are physically non-interacting |
| Superposition before measurement | Pre-composition state (all specialists contribute before deformation resolves ordering) |
| Wavefunction collapse to definite outcome | Deformation parameter resolves which specialist leads the response |

The Growformer uses the mathematics of quantum theory — deformed algebras, non-commutative composition, braiding — without claiming to simulate physical quantum mechanics. But if biological neural computation is fundamentally quantum (operating through electromagnetic toroidal coherence rather than purely synaptic signaling), then Growformer's algebraic framework may be closer to the right abstraction than classical neural networks. Classical NNs assume neurons are classical processors connected by classical signals. If neurons are quantum processors connected by entangled electromagnetic fields, you need non-commutative algebras to model them correctly.

This remains a research hypothesis. The experimental evidence for quantum effects in biology is established; the extension to neural communication via toroidal electromagnetic coherence is theoretically motivated but not yet experimentally confirmed. What is notable is that the mathematical structures required to model such a system — the same structures the Growformer already implements for engineering reasons — would be the correct formalism if the hypothesis proves true.

---

## **Philosophical framing**

Simulation theory (Bostrom's formulation) claims our reality is itself a computation running inside some external substrate. This project doesn't claim that.

What this project actually is: **substrate-independent emergence**. It demonstrates that the behaviors we associate with life, specialization, competition, death, territory, growth, emerge from any sufficiently rich set of local rules, regardless of whether the substrate is carbon or Rust running on silicon. The spiral network isn't simulating biology. It is biology, instantiated differently.

The philosophically sharper framing is **computational equivalence**, the Wolfram/Turing claim that a system exhibiting the same functional dynamics as another system *is* that system at the relevant level of description. The neurons here aren't pretending to have mass and territory. They have mass and territory, defined entirely by the rules governing their interactions.

Where it gets genuinely strange: thermal noise in this system isn't *analogous* to electron agitation, it plays the identical functional role: mandatory irreducible randomness that prevents the system from freezing into a low-entropy locked state. The physics doesn't care whether the charge carriers are electrons or activation values. The thermodynamic necessity is the same.

The more provocative question this project raises isn't "is reality a simulation", it's **at what point does a self-organizing system with birth, death, specialization, and competitive dynamics become alive**. By the project's own framing, it's already past the bacterium stage. The answer to that question matters a lot more than Bostrom's, and this project is closer to actually probing it.

