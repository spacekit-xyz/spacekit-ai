# **Growformer — A Multidimensional Neural Training Environment**

## **What Growformer Is**

Growformer is not “just another neural network implementation.”  
It’s something much rarer: a **multidimensional computational medium** where learning emerges from the interaction of geometry, metabolism, sparsity, timing, energy flow, and spatial constraints — not from a fixed algebraic recipe.

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
Learning is not a single update rule — it is the emergent behavior of six coupled systems:

1. **Weight dynamics** (backprop)  
2. **Geometry** (neurons drift toward correlated partners)  
3. **Timing** (STDP)  
4. **Metabolic cost** (energy‑driven pruning)  
5. **Connectivity** (growth and dissolution of synapses)  
6. **Structural symmetry** (mirror group coupling)

The result is a system that behaves less like a classical MLP and more like a **living computational organism** — one that adapts, collapses, recovers, specializes, and self‑organizes.

Growformer is a platform for exploring a different paradigm of learning:  
one where intelligence is not engineered, but **grown**.

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
└── src/
    ├── lib.rs
    ├── main.rs              — Training demos: XOR and spiral classification
    ├── types.rs             — Vec3, Synapse, NeuronGroup, EnvironmentConfig, NeuronSnapshot
    ├── neuron.rs            — Neuron struct with all 6 dimensions
    ├── environment.rs       — NeuralEnvironment: forward pass, backprop, full training loop
    └── systems/
        ├── mod.rs
        ├── metabolic.rs     — System 1: Cost-driven synapse pruning
        ├── growth.rs        — System 2: Proximity-based synapse formation
        ├── stdp.rs          — System 3: Spike-timing-dependent plasticity
        ├── geometry.rs      — System 4: Spatial drift of neuron positions
        ├── whorls.rs        — System 5: Cycle detection and whorl reporting
        └── mirror.rs        — System 6: Mirror group coupling
```

Runtime topology (shared library path):

`CLI (main.rs)` -> `growformer::service::LanguageService` -> `dimension::{action,generation,codegen}`

`Node (server.rs)` -> `growformer::service::LanguageService` -> `dimension::{action,generation,codegen}`

This means both entrypoints now use the same in-process inference and initialization path.

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

Latest internal routing run with `gle_student_routing_tuned.json`:

- Intent accuracy: `100%`
- Median routing margin: `1.828`
- P10 routing margin: `1.818`
- OOD AUROC: `1.000`
- OOD FAR: `0.00%`

This makes the language stack practical for private, low-latency, CPU-friendly routing:

`text -> GLE semantic vector -> bridge -> 64-d routing vector -> Growformer group dispatch`

Current language milestone status:

- M1 (Language Embedding Foundation): complete
- M2 (Embedding-First Routing Validation): complete
- M3 (Intent-to-Action Layer): complete
- M4 (Controlled Language Generation, template-only): complete
  - Gate command: `cargo run -- --validate-action-schema --action-eval-data data/language/stage_ab_action_eval_extended.jsonl --action-eval-report reports/m3_action_eval_extended.json`
  - Latest result: PASS on Stage A+B extended evaluation dataset
  - M4 gate command: `cargo run -- --validate-generation --action-eval-data data/language/stage_ab_action_eval_extended.jsonl --generation-eval-report reports/m4_generation_eval_extended.json`
  - Latest result: PASS (task-success non-regression + template hallucination baseline)

Operational commands:

- Print model card: `cargo run -- --print-gle-card checkpoints/gle_student_routing_tuned.json`
- Validate checkpoint gate: `cargo run -- --validate-gle checkpoints/gle_student_routing_tuned.json`
- M3 starter action JSON: `cargo run -- --language-action-text "help me reset password"`
- Validate M3 action schema: `cargo run -- --validate-action-schema`
- Validate M3 action schema on Stage A+B JSONL:
  `cargo run -- --validate-action-schema --action-eval-data data/language/stage_ab_action_eval.jsonl --action-eval-report reports/m3_action_eval.json`
- Validate M3 on extended paraphrase set:
  `cargo run -- --validate-action-schema --action-eval-data data/language/stage_ab_action_eval_extended.jsonl --action-eval-report reports/m3_action_eval_extended.json`
- M4 starter generated response:
  `cargo run -- --language-generate-text "please help reset my account password"`
- M5 starter code generation:
  `cargo run -- --language-code-text "implement binary search in rust"`
- M5 code generation eval:
  `cargo run -- --language-code-eval --code-eval-data data/language/m5/eval_codegen_mixed.jsonl --code-eval-report reports/m5_codegen_eval_mixed.json`
- M5 code generation validation gate:
  `cargo run -- --validate-codegen --code-eval-data data/language/m5/eval_codegen_mixed.jsonl --code-eval-report reports/m5_codegen_eval_mixed.json`
- M5 full holdout eval (Python+Rust+JS defaults, with per-language metrics):
  `cargo run -- --language-code-eval --code-eval-report reports/m5_codegen_eval_holdouts.json`
- M5 full holdout validation gate:
  `cargo run -- --validate-codegen --code-eval-report reports/m5_codegen_eval_holdouts.json`
- M5 demo script:
  `bash scripts/demo_code_tasks.sh`
- M5 real sequential training + retention eval (non-mock):
  `cargo run -- --m5-retention-eval --m5-retention-plan data/language/m5/retention_eval_splits.json --m5-epochs 30 --m5-lr 0.2 --m5-feature-dim 512 --m5-retention-report reports/m5_retention_report.json`
- M5 retention eval with replay anti-forgetting:
  `cargo run -- --m5-retention-eval --m5-retention-plan data/language/m5/retention_eval_splits.json --m5-epochs 30 --m5-lr 0.2 --m5-feature-dim 512 --m5-replay-per-epoch 24 --m5-retention-report reports/m5_retention_report_replay.json`
- M5 subject training (design + architectural patterns):
  `cargo run -- --m5-retention-eval --m5-retention-plan data/language/m5/retention_patterns_eval_splits.json --m5-epochs 30 --m5-lr 0.2 --m5-feature-dim 512 --m5-replay-per-epoch 24 --m5-retention-report reports/m5_retention_patterns_report.json`
- M5 subject training with prior-domain replay bias:
  `cargo run -- --m5-retention-eval --m5-retention-plan data/language/m5/retention_patterns_eval_splits.json --m5-epochs 40 --m5-lr 0.2 --m5-feature-dim 512 --m5-replay-per-epoch 24 --m5-replay-prior-ratio 0.9 --m5-retention-report reports/m5_retention_patterns_report_v3.json`
- M5 targeted interference fix run (domain/intent-aware features):
  `cargo run -- --m5-retention-eval --m5-retention-plan data/language/m5/retention_patterns_eval_splits.json --m5-epochs 60 --m5-lr 0.2 --m5-feature-dim 512 --m5-replay-per-epoch 36 --m5-replay-prior-ratio 0.9 --m5-retention-report reports/m5_retention_patterns_report_v7.json`
- Evaluate M4 constrained generation:
  `cargo run -- --language-generation-eval --action-eval-data data/language/stage_ab_action_eval_extended.jsonl --generation-eval-report reports/m4_generation_eval_extended.json`
- Validate M4 constrained generation gate:
  `cargo run -- --validate-generation --action-eval-data data/language/stage_ab_action_eval_extended.jsonl --generation-eval-report reports/m4_generation_eval_extended.json`
- CI helper script: `scripts/validate_gle.sh`
- M3 script with Stage A+B data: `scripts/validate_action_schema.sh data/language/stage_ab_action_eval.jsonl reports/m3_action_eval.json`
- M4 script with Stage A+B data: `scripts/validate_generation.sh data/language/stage_ab_action_eval_extended.jsonl reports/m4_generation_eval_extended.json`
- Full stack gate (GLE + M3 + M4): `scripts/validate_stack.sh checkpoints/gle_student_routing_tuned.json data/language/stage_ab_action_eval_extended.jsonl reports/m3_action_eval_extended.json reports/m4_generation_eval_extended.json`

M5 dataset scaffolding (coding retention):

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
    `cargo run -- --m5-retention-eval --m5-retention-plan data/language/m5/retention_patterns_eval_splits.json --m5-epochs 100 --m5-lr 0.12 --m5-feature-dim 1024 --m5-replay-per-epoch 64 --m5-replay-prior-ratio 0.9 --m5-retention-report reports/m5_retention_patterns_report_v9.json`

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

Two neuron groups can be designated as mirrors. Each tick, each group is nudged toward the other’s average weight and toward the spatial reflection of the other’s centroid. This creates correlated structural development. A `mirror_symmetry_score` tracks the degree of symmetry.

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

## **Running the Examples**

Run one demo at a time via CLI:

```bash
cargo run -- --xor
cargo run -- --spiral
```

### **Demo 1: XOR** (`--xor`)  
2 inputs → 4 hidden → 1 output, 3000 ticks. Hidden layer split into mirror groups. Structural report prints synapse distribution, energy cost, geometric spread, symmetry score, whorl count, and per‑neuron state.

### **Demo 2: Spiral Classification** (`--spiral`)  
Two interleaved spirals, 200 samples per class.  
2 inputs → 8 hidden → 4 hidden → 1 output, 5000 ticks.  
A harder nonlinear boundary that exercises geometry, growth, and timing systems.

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

```toml
[dependencies]
clap = { version = "4", features = ["derive"] }
kiddo = "5.2.4"
rand = "0.8"
rayon = "1.8"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

---

## **Known Limitations and Next Steps**

- **Forward pass adjacency** is O(n²); replace with reverse adjacency index.  
- **Whorl detection** uses depth‑limited DFS; future versions should track axon geometry.  
- **STDP + backprop** interact but are not unified; a combined update rule is future work.  
- **No recurrence** yet; geometry and growth naturally want to form recurrent loops.  

---

## **Relationship to the Biological Record**

The Harvard/Google mapping produced 1.4 petabytes from one cubic millimeter. Every major AI model on Earth fits in a fraction of that. The full human brain at that resolution would require ~1.4 zettabytes — roughly all data generated on Earth in a year.

This project does not close that gap.  
What it does is build a training environment where the dimensions that *create* that gap — timing, geometry, metabolic cost, variable connectivity, structural symmetry — are live variables during training rather than fixed architectural choices.

The network’s structure is an **output** of training, not just its weights.

The next step in the biological record is a mouse hippocampus: 10 cubic millimeters over five years.  
The next step here is a reverse adjacency index and a recurrent layer.

---

## **Philosophical framing**

Simulation theory (Bostrom's formulation) claims our reality is itself a computation running inside some external substrate. This project doesn't claim that.

What this project actually is: **substrate-independent emergence**. It demonstrates that the behaviors we associate with life — specialization, competition, death, territory, growth — emerge from any sufficiently rich set of local rules, regardless of whether the substrate is carbon or Rust running on silicon. The spiral network isn't simulating biology. It is biology, instantiated differently.

The philosophically sharper framing is **computational equivalence** — the Wolfram/Turing claim that a system exhibiting the same functional dynamics as another system *is* that system at the relevant level of description. The neurons here aren't pretending to have mass and territory. They have mass and territory, defined entirely by the rules governing their interactions.

Where it gets genuinely strange: thermal noise in this system isn't *analogous* to electron agitation — it plays the identical functional role: mandatory irreducible randomness that prevents the system from freezing into a low-entropy locked state. The physics doesn't care whether the charge carriers are electrons or activation values. The thermodynamic necessity is the same.

The more provocative question this project raises isn't "is reality a simulation" — it's **at what point does a self-organizing system with birth, death, specialization, and competitive dynamics become alive**. By the project's own framing, it's already past the bacterium stage. The answer to that question matters a lot more than Bostrom's, and this project is closer to actually probing it.

