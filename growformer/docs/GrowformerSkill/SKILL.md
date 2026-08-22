---
name: growformer
description: >
  Expert knowledge base for the Growformer biologically-inspired neural network project ("growformer").
  A Rust implementation of a self-organizing neural environment — the Growformer architecture —
  with physics-based geometry, KWTA competition, mass-competition coupling, pruning, and backprop.
  Includes a full language pipeline (GLE encoder, language bridge, action routing, codegen, generation),
  WASM library support, HTTP server (growformer-node), and dual agent modes (ContextFile, MicroBrain).
  Use this skill whenever the user mentions: growformer project, Growformer, Rivera Model, spiral
  classification, KWTA, neural environment, EnvironmentConfig, NeuralEnvironment, synapse pruning,
  mass decay, Brain API, continual learning, concentric circles, Phase 2, Phase 3, frozen flag,
  gradient gate, mirror dimension, GlobalObserver, DimensionManager, fractal topology, NOW model,
  OCEAN profile, group embedding, VirtualGroup, EpisodicMemory, LearnedRouter, Phase 3c, composition,
  set_router, train_and_set_router, neurogenesis, reserve pool, GLE, Growformer Language Encoder,
  LanguageBridge, LanguageRuntime, LanguageService, language routing, action routing, codegen,
  code generation, M5 retention, M6 agent mode, ContextFile, MicroBrain, acceptance report,
  growformer-node, WASM, wasm32, SLO, auto-spawn, mirror spawn trigger, or asks about
  debugging/tuning this codebase.
  Also trigger when the user pastes training logs showing loss, syn, sparse, mass, lr, A_retain columns.
---

# Growformer — Growformer Project Knowledge Base

## What This Project Is

A biologically-inspired neural architecture in Rust called the **Growformer**.
Not a standard backprop network, which combines gradient descent with physical simulation: neurons have geometry, mass, synapses grow/prune based on activity, and competition emerges from spatial dynamics.

- **Architecture name:** Growformer  
- **Architecture category:** Dynamically Structured Physical Neural System (DSPNS) — not an LLM,
not neuromorphic, not a SOM. Topology is a real-time consequence of physical simulation, not design.  
- **Runtime API:** LanguageService (service.rs) for inference; BrainAPI for brain management  
- **Validated architecture:** 2→16→16→1 (spiral, circles), 2→16→16→2 (continual learning), 2→4→1 (XOR), 64→256→256→1408 (language generation per group)  
- **Validated accuracy:** 90–92.6% on double-spiral; Split MNIST 97.3% avg, 0% forgetting  
- **Generation:** Single-pass binary token prediction via per-group `GroupGenEnv`. Gen g0 eval loss 0.1340, gen g1 ~0.23, code g1 0.3437. Concrete substrate output (no templates, no external models).  
- **Key breakthrough:** KWTA residual (0.02) — preserving 2% activation for losers prevents
gradient starvation and unlocked 90%+ from a 75.4% ceiling. Binary token prediction (200x faster than character-level) — predicts all tokens in a single forward pass.

**Growth today:** The system grows *synapses* and forms functional structure from undifferentiated substrate (pruning, potentiation, geometry). Engram consolidation protects learned memory traces. **Single-neuron neurogenesis** is implemented. The **Growformer** name reflects both synaptic/structure growth and neuron growth when neurogenesis is enabled.

---

## Neurogenesis and the reserve pool (minimal path implemented)

**Current behavior:** Layer sizes are set at build via `mirror_layer_sizes`. Synapses grow and prune. **Single-neuron neurogenesis is implemented:** a Mirror can add one neuron to a hidden layer mid-training when a trigger fires (epoch + loss or residual streak). New neurons are allocated on demand by default, or promoted from an optional reserve pool if configured.

**Implemented:** 
- (1) **Trigger:** `MirrorDimension::try_neurogenesis_trigger(epoch_trigger, loss_threshold, current_loss, rng)` — after `epoch_trigger` epochs, if `current_loss > loss_threshold` and not already triggered, adds one neuron to the last hidden layer. 
- (2) **Refine trigger (residual):** `try_neurogenesis_trigger_residual(residual_threshold, min_epochs_high, current_loss, rng)` — add one neuron if loss has been above threshold for at least `min_epochs_high` consecutive epochs; streak resets when loss ≤ threshold. 
- (3) **Single-neuron insertion:** `NeuralEnvironment::insert_neuron_at_layer(layer_idx, rng, reserve_pool)` — creates one neuron (or promotes from optional `reserve_pool`), appends to layer, adds synapses. 
- (4) **Reserve pool:** Set `DimensionManagerConfig::reserve_pool_size > 0`; mirrors spawn via `new_with_reserve_pool` and promote from pool when neurogenesis fires. 
- (5) **Integration:** `--neurogenesis` demo; spiral run with trigger, loss decreases after event.

**Order of implementation (minimal path) and what is next:**

1. ~~**Trigger**~~ — Done: epoch + loss policy; call `try_mirror_neurogenesis` after each epoch.
2. ~~**Single-neuron insertion**~~ — Done: `insert_neuron_at_layer` in `environment.rs`.
3. ~~**Integration**~~ — Done: `--neurogenesis` demo; spiral run with trigger, loss decreases after event.
4. ~~**Refine trigger**~~ — Done: residual-based trigger; `try_neurogenesis_trigger_residual` (streak of high loss).
5. ~~**Reserve pool**~~ — Done: `reserve_pool_size` in config; `new_with_reserve_pool`; promote from pool in `insert_neuron_at_layer`.

**Reserve pool priority:** On-demand allocation works and the current neurogenesis result does not depend on the pool to be valid. The main benefit of a reserve pool is **warm start**: a pool neuron could accumulate baseline activity from passive exposure to training dynamics before recruitment, making integration smoother than a cold, freshly allocated neuron. Worth implementing for that reason; not required for validity of the current result.

---

## Architecture Naming

| Term | Meaning |
|------|---------|
| Growformer | The architecture — mass-competition, KWTA, physics geometry, three-phase pruning. Grows synapses/structure; single-neuron neurogenesis + optional reserve pool. Substrate-based generation via binary token prediction. |
| Brain API | The runtime interface — Brain::from_config(), Brain::step(), Brain::snapshot() |
| GroupGenEnv | Per-group generation environment — wraps NeuralEnvironment for binary token prediction; 128 token slots × 11 bits |
| TokenDictionary | Per-group vocabulary — maps token IDs to strings for generation decoding |
| NeuralEnvironment | The implementation — internal to Brain API |
| Rivera Model | Project name — earned: Split MNIST 0% forgetting, 97.3% avg; whitepaper-publishable |
| DSPNS | Dynamically Structured Physical Neural System — formal category name |
| DimensionManager | Phase 3 top-level runtime — owns Main + Mirror dimensions + GlobalObserver + LanguageRuntime |
| GLE | Growformer Language Encoder — distilled student encoder for text→embedding routing |
| LanguageBridge | Projects encoder output (384-d) → 64-d routing space with EMA smoothing |
| LanguageService | Shared library service (service.rs) — wraps DimensionManager for action/generation/codegen |
| AgentMode | M6 dual modes: ContextFile (retrieval-augmented) and MicroBrain (trained model routing) |
| growformer-node | HTTP server (server.rs) — axum-based REST API for chat, codegen, mode switching |

---

## File Map

```
src/
  lib.rs            — crate root; feature-gated modules; maybe_par_iter! macro for WASM/native
  main.rs           — Production CLI: --train-brain [--auto], --infer [--prompt]; auto-config,
                      data profiling, convergence monitor, brain training pipeline
  demos.rs          — Separate binary (growformer-demos): XOR, spiral, circles, MLP, Phase 2,
                      --fractal, --phase3c, --neurogenesis, --language-*, --m5-retention-eval,
                      --acceptance-report, all benchmarks and evaluations
  server.rs         — growformer-node HTTP server (axum); /v1/health, /v1/chat, /v1/chat/stream,
                      /v1/mode, /v1/acceptance
  service.rs        — LanguageService shared library; AgentMode (ContextFile/MicroBrain);
                      SLO tracking; handoff logging; acceptance report; brain load/switch
  spectral.rs       — TokenDictionary for binary token prediction; tokenizer
  types.rs          — EnvironmentConfig; Neuron/Synapse with frozen, consolidation, branch_id
  environment.rs    — forward, backprop, KWTA, freeze_consolidated_pathway(), engram consolidation
  neuron.rs         — Neuron (frozen, winning_branch)
  dimension/
    mod.rs          — pub re-exports for all dimension types
    manager.rs      — DimensionManager: top-level runtime; auto-spawn trigger; episodic summaries;
                      group_gen_envs, group_code_envs for per-group substrate generation;
                      route_text_to_action_stateless for independent inference
    main_dim.rs     — MainDimension: frozen consolidated groups
    mirror_dim.rs   — MirrorDimension: isolated full-plasticity training
    group_gen.rs    — GroupGenEnv: per-group binary token prediction via NeuralEnvironment;
                      GenEnvOverrides for auto-configured hidden/k/max_tokens;
                      encode_target, decode_output, train_step, eval_loss
    promotion.rs    — PromotionGateConfig, evaluate_promotion, promote
    observer.rs     — GlobalObserver: routing, promotion gating, activity history
    embedding.rs    — GroupEmbedding, cosine_similarity, hidden_activation_vector
    router.rs       — LearnedRouter (MLP), attend_by_query
    composition.rs  — VirtualGroup (blend frozen groups), Episode, EpisodicMemory
    language.rs     — GLE encoder, LanguageBridge, LanguageRuntime, EMA smoother, OOD rejection,
                      route_language_embedding, multi-checkpoint ensemble, WASM from_bytes path
    action.rs       — ActionJson, ActionType, ActionPayload, action_from_routing (M3)
    codegen.rs      — CodeGeneration, generate_code_from_action (M5)
    generation.rs   — GeneratedResponse, render_action_template (M4 legacy)
    policy.rs       — ContinuousPolicy
  systems/
    mod.rs          — pub re-exports
    checkpoint.rs   — save/load checkpoint functions; WASM serialize_to_bytes/deserialize_from_bytes
    growth.rs       — synapse pruning, potentiation, floor; group synapse gate
    geometry.rs     — physics simulation, reaction-diffusion inhibition; group repulsion penalty
    metabolic.rs    — energy/mass budget; frozen neuron exclusion
    mirror.rs       — mirror group coupling; frozen neuron exclusion
    stdp.rs         — STDP (disabled); frozen exclusion
    whorls.rs       — geometric pattern detection
  mnist.rs          — MNIST loading/filtering (native only, gated behind cfg)

data/language/
  stage_ab_action_eval.jsonl          — M3 Stage A+B eval
  stage_ab_action_eval_extended.jsonl — M3/M4 extended paraphrase eval
  m5/                                 — M5 training/eval datasets
    train_python_coding.jsonl, train_rust_coding.jsonl, train_javascript_coding.jsonl
    train_design_patterns.jsonl, train_architectural_patterns.jsonl
    train_multi_turn.jsonl (Stage C), train_adversarial.jsonl (Stage D)
    eval_*_holdout.jsonl              — holdout eval sets per domain
    retention_eval_splits.json        — 3-domain retention plan (coding only)
    retention_eval_splits_full.json   — 7-domain retention plan (all domains incl Stage C/D)
    retention_patterns_eval_splits.json — patterns-only retention plan
    CURRICULUM_V1_TEMPLATE.md

tests/
  wasm_service_test.rs — integration test for WASM-compatible LanguageService API

docs/
  LANGUAGE.md        — M0-M6 spec and implementation status
  NODE.md            — growformer-node HTTP API documentation
  ARCHITECTURE.md    — architecture overview
  GrowformerSkill/   — this skill + parameter_guide.md + results_table.md

outputs/
  brain_api.rs                — complete Brain API / Growformer runtime
  demo_continual_learning.rs  — Phase 2 demo with frozen flag gradient gate
  phase2_checkpoint.rs        — checkpoint save/load, demo_phase2_train_a/b
  geometry.rs                 — group boundary: 3x repulsion between task groups
  growth.rs                   — group boundary: cross-group synapse gate
  fractal_topology_spec.md    — Phase 3 full spec (standalone drop-in module)
  growformer_spec.docx        — full architecture spec v0.3
  rivera_visualizer.jsx       — Three.js + React real-time neural visualizer
```

---

## Validated Parameters (90%+ Checkpoint)

```rust
// Phase 2 config — dropout MUST be 0.0 for continual learning (see failure mode #12)
learning_rate: 0.15,
weight_decay: 0.0000025,
bias_decay: 0.0,           // NEVER enable
dropout_rate: 0.0,         // 0.0 for Phase 2+ — 0.1 for single-task Phase 1 only
competitive_k: 4,
lateral_inhibition: 0.12,
lr_decay: 0.00008,
sigma_inhib: 2.0,
debye_length: 1.5,
thermal_noise: 0.02,
k_repel: 0.2,
gravity_g: 0.05,
damping: 0.2,
mass_win_threshold: 0.15,
mass_decay: 0.00009,
mass_growth: 0.0005,
homeostasis_lr: 0.0,       // DISABLED
growth_radius: 2.0,
prune_interval: 500,
weight_clamp: 5.0,
max_synapses_per_neuron: 64,
energy_budget_per_neuron: 100.0,
pruning_threshold: 0.001,
mirror_coupling_strength: 0.001,
geometry_interval: 500,
stdp_enabled: false,
```

**CRITICAL — KWTA residual in environment.rs:**
```rust
if i >= k {
    if let Some(n) = self.neurons.get_mut(nid) {
        n.activation *= 0.02;  // NOT 0.0 — this single line unlocked 90%+
    }
}
```

---

## Seed Results

| Seed | Results |
|------|---------|
| 42 | 90.4%, 92.6%, 90.1% |
| 7 | 92.2%, 91.8% |

Good seed signature: loss ~0.084 at epoch 500. Bad seed: loss >0.21 at epoch 500 — try different seed.

---

## Benchmark Results

| Task | Model | Accuracy | Active Neurons | Synapses |
|------|-------|----------|----------------|----------|
| Double spiral | Growformer seed=42 | 90.4–92.6% | 9–11 of 16 | 139–163 |
| Double spiral | MLP baseline | 90.4% | 16 of 16 | ~272 stable |
| Circles noise=0.05 | Growformer | 100% | 16 of 16 | ~185 |
| Circles noise=0.25 | Growformer | 97.9% | 16 of 16 | ~227 |
| Phase 2 Task A retention | Growformer frozen flag | 94.5% → 91.5% | — | — |
| Phase 2 Task B (shared env) | Growformer | 52.1% | — | — |
| Phase 3 Task A (Mirror) | Growformer fractal | 92.5% | — | — |
| Phase 3 Task B (Mirror) | Growformer fractal | 99.1% | — | — |
| Phase 3 A_retain after Task B | Growformer fractal | 92.5% → 92.5% (0% forgetting) | — | — |
| Phase 3 routing (no context) | LearnedRouter 400 ep, lr=0.15 | Spiral→0, Circles→1 (correct) | — | — |
| Phase 3 Context routing | Both tasks | 1.000 margin (perfect) | — | — |
| Phase 3c Task C composition | VirtualGroup 30 samples | 80–90% train; held-out ~77%+ | — | — |
| Phase 3c Task D (3-group) | VirtualGroup 40 samples | 70–77% train; held-out reported | — | — |
| **Split MNIST (5 tasks)** | Growformer sequential | **97.3% avg** | **0% forgetting** | **~70KB checkpoint** |
| M2 language routing | GLE → bridge → cosine | 100% intent, 1.828 median margin | — | — |
| M5 coding retention (3 lang) | M5Learner sequential | mean_retention=1.000 | — | — |
| M5 full 7-domain retention | M5Learner + replay | mean_retention=1.000 | — | — |
| **Gen g0 (support)** | GroupGenEnv binary token | **eval loss 0.1340** | 232 frozen neurons | 8182 synapses |
| **Gen g1 (coding/general)** | GroupGenEnv binary token | **eval loss ~0.23** | — | ~105K synapses |
| **Code g1** | GroupGenEnv binary token | **eval loss 0.3437** | 232 frozen neurons | 8647 synapses |
| M6 acceptance | SLO check | PASS (P95 latency 1.6ms) | — | — |
| CLI baseline (debug build) | --language-action-text | ~1.43s, ~9.5 MB RSS | — | — |

**Phase 2 headline:** 3.9% forgetting — well below 10% threshold. PASS.
Task B 52.1% is due to shared physics substrate suppression. Fixed by Mirror Dimension (Phase 3).

**Phase 3 headline:** 0% forgetting. 92.5% spiral, 99.1% circles, simultaneous. Learned router: single forward → logits → argmax (400 ep, lr=0.15, hidden=16). Router updatable via `set_router(router)` or `train_and_set_router` after adding groups. Phase 3c: composition (VirtualGroup) and episodic memory (store/recall, second-presentation on held-out). **Split MNIST:** 97.3% average, 0% forgetting; every task matches baseline exactly (Main Dimension received zero gradient; no mechanism for forgetting). Rivera Model name fully earned — publishable today.

**Language pipeline headline:** M1-M6 complete. GLE routing at 100% intent accuracy with 1.828 median margin. **Substrate-based generation** via per-group GroupGenEnv using single-pass binary token prediction (200x faster than char-level): gen g0 eval loss 0.1340, gen g1 ~0.23. All generation outputs are concrete model outputs — produced by a single forward pass through Growformer substrate. M5 real learning retains 100% across 7 sequential domains. M6 dual agent modes (ContextFile + MicroBrain) with SLO-enforced acceptance report PASS. Auto-configuration (`--auto`) profiles dataset and derives training parameters, network size, and early stopping thresholds.

---

## Split MNIST — Sequential Continual Learning (Complete)

**Result:** Every task matches its baseline exactly. Not approximately — exactly. Retention equals promotion-time accuracy because the Main Dimension received zero gradient across all five sequential training runs. There was nothing to forget because there was no mechanism for forgetting to occur.

### Complete Split MNIST Result

| Task | Digits | Accuracy | After All 5 Tasks | Forgetting |
|------|--------|----------|-------------------|------------|
| 0 | 0 vs 1 | 97.6% | 97.6% | 0% |
| 1 | 2 vs 3 | 96.7% | 96.7% | 0% |
| 2 | 4 vs 5 | 98.6% | 98.6% | 0% |
| 3 | 6 vs 7 | 97.1% | 97.1% | 0% |
| 4 | 8 vs 9 | 96.3% | 96.3% | 0% |
| **Avg** | | **97.3%** | **97.3%** | **0%** |

### Comparison

- **EWC:** 97% average, ~3% forgetting. Achieves result through mathematical regularization that still permits degradation.
- **Growformer:** 97.3% average, 0% forgetting. Achieves result through structural isolation that makes degradation impossible. Better on both metrics simultaneously.
- **Progressive Neural Networks:** Zero forgetting but unbounded linear memory growth — a full new column per task, no pruning. Growformer promotes a pruned sparse group per task: five tasks, ~70KB checkpoint. That gap is the efficiency argument made concrete.

**Whitepaper:** Results section is complete. Toy task validation, real benchmark validation (Split MNIST), compositional generalization, self-organizing routing, neurogenesis, and a checkpoint that fits in 70KB. Every claim in the paper has a number behind it. Rivera Model name is fully earned. Publishable today.

---

## Phase 2 — Continual Learning

### Architecture

```
[2 → 16 → 16 → 2]
Group A: layer1[0..8] + layer2[0..8] → output[0]   (Task A: spiral)
Group B: layer1[8..16] + layer2[8..16] → output[1]  (Task B: circles)
Shared substrate: input layer only. No shared hidden neurons between tasks.
Cross-head synapses severed before training begins.
```

### Gradient Gate — Frozen Flag (final correct implementation)

`frozen: bool` on Neuron and Synapse, checked inside backprop and all plasticity systems.
No restore loop. No snapshot fields. Prevents writes — never undoes them.

**freeze_consolidated_pathway() sets frozen=true on:**
- Every neuron in group_a_ids (layer1[0..8] + layer2[0..8])
- output_0 neuron
- Every synapse from input neurons targeting group_a_layer1 neurons

**Frozen checks — must be present in ALL of these:**
- `environment.rs` backprop: skip weight/synapse update if `n.frozen` or `syn.frozen`
- `environment.rs` forward: skip mass and homeostasis if `n.frozen`
- `geometry.rs` update_geometry: skip position integration if `n.frozen`
- `stdp.rs`: skip if frozen
- `metabolic.rs`: exclude from pruning and synapse aging/depression pass
- `growth.rs`: no synapse growth or pruning on frozen neurons/synapses
- `mirror.rs`: skip frozen neurons

**Key rule:** frozen blocks writes only, not reads. Frozen neurons still compute activations
and contribute to the forward pass. Group A's signal still reaches output[0].

### Zero-Gradient on Inactive Head

```rust
// Task A — output[1] zero gradient:
let current_out = env.predict(input);
let target_both = [target[0], current_out[1]];

// Task B — output[0] zero gradient:
let current_out = env.predict(input);
let target_both = [current_out[0], target[0]];
```

dropout_rate MUST be 0.0, otherwise predict() and train_tick() use different masks,
producing nonzero gradient that accumulates to saturation by epoch 500.

### Cross-Head Synapse Severing

Run immediately after group assignment, before any training:
```rust
for &nid in &group_a_ids {
    if let Some(n) = env.neurons.get_mut(&nid) {
        n.synapses.retain(|s| s.target != output_1);
    }
}
for &nid in &group_b_ids {
    if let Some(n) = env.neurons.get_mut(&nid) {
        n.synapses.retain(|s| s.target != output_0);
    }
}
```

### Checkpoint System

```bash
cargo run -- train-a    # trains Task A, saves task_a_checkpoint.json (~20 min)
cargo run -- train-b    # loads checkpoint, tests Task B fix (~20 min, not 40)
cargo run               # full run, no checkpoint
```

Checkpoint stores full neuron state with frozen flags embedded in serialized neurons.
No separate snapshot fields needed.

### Phase 2 Failure Modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| A_retain=50% at epoch 0 | Shared layer2 | Split layer2 between groups |
| A_retain=50% at epoch 0 | Neutral target (0.5) on inactive head | Match target to current prediction |
| A_retain=50% at epoch 500 (exactly) | Dropout mask mismatch | dropout_rate: 0.0 |
| A_retain=50% at epoch 500 | Group B → output[0] synapses training | Sever cross-head synapses; use full neuron clone |
| A_retain drops ~epoch 1000 | Neuron geometry drift | Frozen flag on geometry integration |
| A_retain slow decline | Synapse depression accumulating | Frozen synapses excluded from aging in metabolic.rs |
| Task B 52.1% | Group A mass dominates KWTA | Mirror Dimension (Phase 3) |

---

## Phase 3b — Routing Results

### Confirmed Results

| Input | Routed To | Margin | Status |
|-------|-----------|--------|--------|
| Spiral test point | Group 0 ✓ | 0.051 | Correct, fragile |
| Circles test point | Group 1 ✓ | 0.664 | Correct, robust |
| Spiral + context | Group 0 ✓ | 1.000 | Perfect |
| Circles + context | Group 1 ✓ | 1.000 | Perfect |

### Key Findings

**Context routing at 1.000** proves the group embeddings are completely orthogonal when queried
correctly. The information for perfect routing exists in the embedding space.

**Spiral fragility (0.051)** is a test input location problem, not an embedding problem.
The specific test point sits near a region where both groups activate similarly. Points deeper
into spiral arms will produce wider margins. The embedding quality is confirmed sound by the
1.000 context result.

**Circles robustness (0.664)** reflects that the radial boundary produces activation patterns
more geometrically distinctive than the spiral boundary from a single test point.

### Router Architecture

- **Learned router (primary):** MLP input_dim→hidden(16)→num_groups, lr=0.15. Single forward → logits → argmax. Trained with `train_and_set_router(data_per_group, rng, 400)`; shuffled epochs. When `num_groups`/`input_dim` match, observer uses router; else falls back to cosine over embeddings.
- **Dynamic update:** `set_router(router)` to pass in a pre-built router (e.g. after adding a 3rd group); or call `train_and_set_router` again with new group count. Existing main groups never retrained.
- **Fallback:** cosine similarity between input activation and group embedding (used when no learned router or size mismatch).

### Spiral Routing — Improvement Path

Spiral margin ~0.051 (correct but fragile on one test point). Use multiple test points or deeper spiral-arm points; learned router (400 ep) gives correct routing; more epochs or margin target can flip spiral to wrong group — keep 400 ep, one-hot target.

---

## Phase 3c — Composition + Episodic (COMPLETE)

### VirtualGroup

- Blends 2+ frozen groups: `output = sum_g (w_g * group_g.predict(input))`. Only blend weights train (tiny problem).
- `train_composition(group_ids, data, lr, epochs)` → (VirtualGroup, accuracy). Use 30–40 samples.
- Task C: spiral-gated circles (inner r&lt;0.4 → spiral, outer → circles). Task D: 3-way radius gate (spiral/circles/moons).

### EpisodicMemory

- **Store:** `store_composition_episode(virtual_group, data, accuracy, residual)`. Signature = mean input coords. Stored when composition acc ≥ 80% (Task C) or ≥ 75% (Task D).
- **Retrieve:** `episodic_retrieve(signature, threshold)` → `Option<&Episode>`. Episode: input_signature, group_ids, blend_weights, accuracy, residual.
- **Second presentation:** Retrieve by train signature, evaluate retrieved blend on held-out data. Task C and Task D both report held-out accuracy (e.g. "Task D held-out: retrieved composition accuracy = X% (n=60) [train Y%]").

### Demos

- `--fractal`: Demo 6 — spiral + circles mirrors, promote, train router (400 ep), infer with/without context.
- `--phase3c`: Same plus Task C (2-group composition), Task D (moons, 3-group composition), episodic store, second-presentation and held-out tests, timed memory recall.

### Adding groups after deployment

- Existing main groups: never retrained. Router: pass in new router via `set_router(router)` or retrain with `train_and_set_router` for new group count.

---

```rust
// geometry.rs — 3x repulsion between different task groups
let group_penalty = match (group_id, other_group) {
    (Some(g1), Some(g2)) if g1 != g2 => 3.0,
    _ => 1.0,
};

// growth.rs — block cross-group synapse growth
let cross_group = src_group.is_some()
    && tgt_group.is_some()
    && src_group != tgt_group;
if cross_group { continue; }
```

---

## Phase 3 — Fractal Topology & Mirror Dimension

### Core Insight

Main dimension = frozen consolidated knowledge only, never trains.
Mirror dimension = isolated full-plasticity environment per task.
Promotion gate = gatekeeper between mirror and main.
GlobalObserver = integrating window across all dimensions.

### Architecture

```
Layer 3: GlobalObserver — watches all, routes inference, gates promotion
Layer 2: DimensionManager
  ┌──────────────────────┐  ┌─────────────────────────────┐
  │  Main Dimension      │  │  Mirror Dimension(s)         │
  │  All groups frozen   │  │  Full plasticity             │
  │  Inference only      │  │  No Group A neurons inside   │
  └──────────────────────┘  └─────────────────────────────┘
Layer 1: NeuralEnvironment (unchanged)
```

### Fractal Property

Observer → Training Space → Consolidation → Promotion Gate at every scale:

| Scale | Observer | Consolidation |
|-------|----------|---------------|
| Neuron | Activation history | `frozen: bool` |
| Group | Group embedding vector | `freeze_consolidated_pathway()` |
| Global | GlobalObserver | Main Dimension |

### Key Structs

```rust
pub struct DimensionManager {
    pub main: MainDimension,
    pub mirrors: HashMap<String, MirrorDimension>,
    pub observer: GlobalObserver,
    pub episodic_memory: EpisodicMemory,
    pub config: DimensionManagerConfig,
    pub language_runtime: LanguageRuntime,
    low_confidence_streak: u32,        // M5 auto-spawn tracker
    pub auto_spawn_threshold: f32,     // default 0.15
    pub auto_spawn_k: u32,            // default 10 (K=10 consecutive low-confidence batches)
}

pub struct MainDimension {
    pub groups: HashMap<GroupId, FrozenGroupEnv>,  // all frozen=true
    pub embedding_library: Vec<GroupEmbedding>,
    pub group_order: Vec<GroupId>,                 // append-only
}

pub struct MirrorDimension {
    pub task_name: String,
    pub env: NeuralEnvironment,             // full plasticity
    pub main_query_fn: Option<...>,         // read-only channel to Main
    pub epochs_trained: u32,
    pub best_accuracy: f32,
}

pub struct GlobalObserver {
    pub embedding_library: Vec<GroupEmbedding>,
    pub group_activity: HashMap<GroupId, VecDeque<f32>>,
    pub routing_config: RoutingConfig,
    pub learned_router: Option<LearnedRouter>,  // single forward → logits → argmax
    pub last_chosen_group_id: Option<GroupId>,
    pub last_routing_scores: Option<Vec<(GroupId, f32, f32, f32, f32)>>,
    pub promotion_gate_config: PromotionGateConfig,
    pub coherence: f32,
}

// Language pipeline (M1-M4)
pub struct LanguageRuntime {
    pub config: LanguageConfig,
    pub encoder: HashingLanguageEncoder,  // GLE text→embedding
    pub bridge: LanguageBridge,           // 384-d → 64-d projection
    pub smoother: EmaSmoother,            // EMA over turn embeddings (alpha=0.2)
    preloaded_students: Vec<GleStudentCheckpoint>,
}

// Shared service (M6)
pub struct LanguageService {
    pub dm: DimensionManager,
    pub support_gid: GroupId,
    pub coding_gid: GroupId,
    pub calibration: CalibrationReport,
    pub mode: AgentMode,          // ContextFile or MicroBrain
    pub slo_config: SloConfig,    // latency P95, memory, checkpoint limits
    latency_log: Vec<f64>,
    handoff_log: Vec<HandoffLogEntry>,
    context_snippets: Vec<String>,
}
```

### Promotion Gate Criteria

| Criterion | Threshold |
|-----------|-----------|
| Task accuracy | > 85% |
| Redundancy cosine vs existing groups | < 0.80 |
| Geometric separation | > 1.5 units |
| Stability window | 50 epoch plateau |

### Group Embeddings

```rust
pub struct GroupEmbedding {
    pub group_id: GroupId,
    pub vector: Vec<f32>,           // mean activation over calibration samples
    pub accuracy: f32,
    pub intrinsic_dim: Option<f32>,
    pub description: Option<String>,
    pub metatags: Vec<String>,
    pub tag_vector: Vec<f32>,       // hashed tag embedding for context routing
    pub language_vector: Vec<f32>,  // 64-d vector for language routing (set via set_group_language_vector)
}
// vector computed once at promotion. language_vector set via representative prompts.
```

### Learning Schedule

| Residual | Operation | Epochs |
|----------|-----------|--------|
| < 0.05 | Pure recall | 0 |
| < 0.15 | Reweight router | 50 |
| < 0.30 | Compose groups (VirtualGroup) | 200 |
| > 0.30 | Spawn new mirror | 4000 |

### New Files for Phase 3

```
src/dimension/
  mod.rs, main_dim.rs, mirror_dim.rs, promotion.rs,
  observer.rs, manager.rs, embedding.rs, router.rs,
  composition.rs   // VirtualGroup, Episode, EpisodicMemory
```

Zero breaking changes. Only add `pub mod dimension;` to lib.rs.

### DimensionManager API

```rust
// Core inference and routing
DimensionManager::new(config)
dm.infer(&input)
dm.infer_with_context(&input, context_tags)
dm.last_chosen_group_id() / dm.last_routing_scores()

// Mirror lifecycle
dm.spawn_mirror("task", seed)
dm.train_mirror_epoch("task", data, rng, batch_size)
dm.evaluate_promotions(calibration_data)
dm.force_promote("task", calibration_data) -> GroupId
dm.try_mirror_neurogenesis("task", epoch_trigger, loss_threshold, current_loss, rng)
dm.try_mirror_neurogenesis_residual("task", threshold, min_epochs, loss, rng)

// Router
dm.train_and_set_router(data_per_group, rng, epochs)
dm.set_router(router)

// Composition + episodic memory
dm.train_composition(group_ids, data, lr, epochs)  // -> (VirtualGroup, accuracy)
dm.store_composition_episode(&vg, data, accuracy, residual)
dm.predict_with_composition(input, &vg)
dm.predict_with_episode(input, &episode)
dm.episodic_retrieve(signature, threshold) -> Option<&Episode>
dm.episodic_summaries() -> Vec<EpisodicSummary>    // M6 shared-state read
dm.evaluate_main_group(group_id, data) -> accuracy
dm.list_groups() / dm.list_mirrors()

// Language pipeline (M1-M4)
dm.configure_language(LanguageConfig)
dm.calibrate_language_bridge(&dataset, &requirements) -> Result<CalibrationReport>
dm.route_text(text) -> Result<LanguageRoutingDecision>
dm.route_text_stateless(text) -> Result<LanguageRoutingDecision>
dm.route_text_to_action(text) -> Result<ActionJson>
dm.route_text_to_action_with_threshold(text, ood) -> Result<ActionJson>
dm.set_group_language_vector(group_id, vec)
dm.set_group_language_vector_from_texts(group_id, &texts)
dm.build_group_language_vector_from_texts(&texts) -> Result<Vec<f32>>

// M5 auto-spawn
dm.track_confidence_for_auto_spawn(&routing) -> Option<String>  // K=10 streak → suggested name
dm.route_text_with_spawn_check(text) -> Result<(LanguageRoutingDecision, Option<String>)>
dm.low_confidence_streak() -> u32
dm.checkpoint_size_summary() -> CheckpointSizeSummary
```

### LanguageService API (service.rs)

```rust
// Construction
LanguageService::new_default() -> Result<Self>            // native only (reads env vars)
LanguageService::new_with_config(LanguageConfig) -> Result<Self>  // WASM-safe

// Inference
svc.action(text) -> Result<ActionJson>
svc.generation(text) -> Result<(ActionJson, GeneratedResponse)>
svc.codegen(text) -> Result<(ActionJson, Option<CodeGeneration>)>
svc.load_gle_students_from_bytes(&[&[u8]]) -> Result<usize>

// M6 agent modes
svc.set_mode(AgentMode, confidence, reason)
svc.active_mode() -> AgentMode
svc.handoff_log() -> &[HandoffLogEntry]
svc.push_context_snippet(String)      // context-file mode: inject retrieval snippets
svc.context_snippets() -> &[String]
svc.clear_context_snippets()
svc.read_episodic_summaries() -> Vec<EpisodicSummary>  // read-only cross-mode access
svc.route_with_spawn_check(text) -> Result<(LanguageRoutingDecision, Option<String>)>

// M6 SLO + acceptance
svc.slo_snapshot() -> SloSnapshot
svc.acceptance_report() -> AcceptanceReport

// Brain management
svc.active_dm_mut() -> &mut DimensionManager   // public access for inference pipelines
```

---

## NOW Model Observer Windows

| Level | Timescale | Component |
|-------|-----------|-----------|
| L1: Micro | 1 tick | Neuron activation, synapse facilitation |
| L2: Local | 1 epoch | NeuralEnvironment: KWTA, mass, backprop |
| L3: Group | 10–100 epochs | Consolidated group, frozen pathway, embedding |
| L4: Mirror | 100–4000 epochs | MirrorDimension: isolated training environment |
| L5: Promotion | Task completion | PromotionGate: gating new knowledge into main |
| L6: Global | Lifetime | GlobalObserver: cross-group coherence, episodic memory |
| L7: Composition | On demand | VirtualGroup: blending frozen groups |

**Key insight:** Every Phase 2 failure was a boundary violation — a lower observer level
(backprop) writing to state owned by a higher level (consolidated group). The frozen flag
is the NOW model's architectural invariant made concrete.

---

## OCEAN Profile (Specified, Not Yet Implemented)

| Trait | Config Parameter | Effect |
|-------|-----------------|--------|
| Openness | `growth_radius`, `pruning_threshold` | Exploration vs exploitation |
| Conscientiousness | `facilitation_bonus`, consolidation threshold | Strictness before consolidating |
| Extraversion | `mass_win_threshold`, `lateral_inhibition` | Baseline activation energy |
| Agreeableness | `group_boundary_penalty` | Inter-group cooperation |
| Neuroticism | `thermal_noise`, `mass_decay` | Stability vs sensitivity |

Status: Specified. Needs empirical validation before implementation.
Gate: does high neuroticism produce higher variance Phase 2 retention across seeds?

---

## Architecture Category

**Formal:** Dynamically Structured Physical Neural System (DSPNS)

Four properties never combined before:
1. Physics-embodied representation (geometry, mass, velocity are computational variables)
2. Metabolically-constrained plasticity (energy budgets)
3. Consolidation-based continual learning (structural freezing, not regularization)
4. Intrinsic-dimensionality-aware self-organization

**Not an LLM.** Sparse inference — only k/N neurons active per forward pass. At scale,
inactive groups consume zero compute. Power scales with active synapses, not parameter count.

---

## Failure Modes Encyclopedia

1. **Hard-Zero KWTA** — `n.activation=0.0` kills 75% gradient. Fix: `*= 0.02`
2. **Input Synapse Cap** — `strength_cap=0.5` starves input layer. Fix: uniform `weight_clamp`
3. **Input Weight Decay Split** — Fix: uniform `weight_decay: 0.0000025`
4. **File Drift** — Fix: `outputs/neuro/src/` is source of truth
5. **Missing Synapse Floor** — Fix: `if neuron.synapses.len() <= 2 { continue; }`
6. **Mass Stuck 1.56** — Fix: `mass_decay: 0.00009`
7. **Always-On Attractor** — Fix: remove warmup, prune_stop_tick, bias_decay
8. **Local KWTA** — Permanently closed. Geometrically incompatible at 16 neurons
9. **Inner Oversampling** — Makes things worse (64.5% → 58.2%)
10. **generate_spiral_data(n)** — n is per-class, total = 2n
11. **Circles growth_radius=0.0** — Fix: `growth_radius: 2.0`
12. **Dropout in Continual Learning** — A_retain=50% exactly at epoch 500. Fix: `dropout_rate: 0.0`
13. **Shared Layer2** — A_retain=50% at epoch 0. Fix: split layer2 between groups
14. **Neutral Target Gradient Pollution** — Fix: match target to current prediction
15. **Restore-Loop Incompleteness** — effective_strength = strength × facilitation × depression — all fields drift. Fix: frozen flag architecture
16. **Group B KWTA Suppression** — Task B 52.1%. Fix: Mirror Dimension (Phase 3)

---

## Key Architecture Rules

1. KWTA residual = 0.02, never 0.0
2. Uniform weight decay — no input/hidden split
3. No input synapse strength cap
4. Minimum synapse floor = 2 in growth.rs
5. No bias_decay, no warmup, no prune_stop_tick
6. growth_radius = 2.0 for all tasks
7. dropout_rate = 0.0 for continual learning (Phase 2+)
8. Cross-head synapses severed before training begins (Phase 2+)
9. Frozen flag blocks writes only — frozen neurons still fire and contribute to forward pass
10. outputs/ is source of truth — sync before each run
11. Neuron count per Mirror is fixed at build; synapses grow/prune. Neurogenesis (single-neuron insertion, optional reserve pool) is implemented — trigger by epoch+loss or residual streak; reserve pool gives warm start when provided.

---

## Phase Transitions

| Phase | Status | Gate |
|-------|--------|------|
| Phase 1: Digital Bacterium | COMPLETE | 90%+ spiral, 100% circles, Brain API |
| Phase 2: Continual Learner | COMPLETE | 3.9% forgetting, frozen flag architecture |
| Phase 2b: Hypernetwork Layer | SUBSUMED | Residual-gated schedule lives in DimensionManager |
| Phase 3a: Mirror Dimension | COMPLETE | 0% forgetting, 92.5% spiral, 99.1% circles |
| Phase 3b: Self-Organizing Router | COMPLETE | LearnedRouter (400 ep, lr=0.15); spiral→0, circles→1; set_router for dynamic update |
| Phase 3c: Composition + Episodic | COMPLETE | VirtualGroup (2- and 3-group), EpisodicMemory, Task C/D, second-presentation and held-out accuracy |
| Phase 3d: GlobalObserver + NOW | COMPLETE | Routing, promotion gating, activity history, coherence; full NOW hierarchy mapped |
| Phase 3e: Split MNIST | COMPLETE | 5-task sequential: 97.3% avg, 0% forgetting, ~70KB; Main Dimension zero gradient |
| M1: Language Embedding | COMPLETE | GLE encoder + LanguageBridge; text→384-d→64-d routing vector |
| M2: Routing Validation | COMPLETE | 100% intent accuracy, 1.828 median margin, 1.000 OOD AUROC |
| M3: Intent-to-Action | COMPLETE | ActionJson output; validated on Stage A+B extended dataset |
| M4: Substrate Generation | COMPLETE | Per-group GroupGenEnv; binary token prediction (128 slots × 11 bits = 1408 output neurons); gen g0 eval 0.1340, gen g1 ~0.23; engram consolidation + priority replay |
| M5: Continual Language Learning | COMPLETE | 7-domain sequential retention (coding×3 + patterns×2 + Stage C/D); mean_retention_ratio=1.000; auto-spawn trigger K=10 |
| M6: Production Agent Modes | COMPLETE | ContextFile + MicroBrain modes; shared-state contract; handoff logging; SLO tracking; acceptance report PASS |
| WASM Library | COMPLETE | lib compiles to wasm32-unknown-unknown with --no-default-features; feature-gated native/server/parallel/cli |
| Auto-Configuration | COMPLETE | `--auto` flag: DataProfile → AutoConfig → GenEnvOverrides; convergence monitor with early stopping |
| CLI Refactor | COMPLETE | `main.rs` = train+infer only; `demos.rs` = all benchmarks/demos as separate binary |
| Phase 4: Open-Ended Evolver | FUTURE | Optimizes own physics |
| Continuum (train-while-on) | FUTURE | Feedback-driven online learning; see `docs/CONTINUUM.md` |

**Phase 4 vs Continuum:** Phase 4 (Evolver) = meta-learning of architecture/physics (e.g. growth_radius, lr). Continuum = parameter updates from user feedback and optional persistence; implementable with current architecture and can feed Phase 4 later as a signal.

**Rivera Model name: fully earned.** Split MNIST 97.3% avg, 0% forgetting; multi-task learned, retained, routed without task label; composition from few examples; episodic store/recall with held-out generalization; ~70KB for 5 tasks. Language pipeline complete through M6 with dual agent modes and production SLO tracking.

---

## Reading Training Logs

| Field | Healthy | Alarm |
|-------|---------|-------|
| loss | ~0.084 at epoch 500 | >0.20 = bad basin, restart |
| syn | 250–300 early, 140–200 final | <130 = over-pruning |
| sparse | 0.10–0.25 | 0.00 = collapse |
| mass | 1.4–2.1 | flat 1.56 = mass_decay wrong |
| A_retain | flat across all epochs | drops at 500 = frozen flag incomplete |
| A_retain | exactly 50.0% | dropout mask mismatch or major write path leak |

---

## Language Pipeline (M1-M6)

### Architecture

```
text → GLE encoder (384-d) → LanguageBridge (Linear 384→64 + LayerNorm) → 64-d routing vector
  → cosine similarity vs group language vectors → LanguageRoutingDecision
  → action_from_routing → ActionJson → render_action_template / generate_code_from_action
```

### GLE (Growformer Language Encoder)

In-house distilled student encoder. No external transformer runtime required.
- Hashing-based encoder (HashingLanguageEncoder) for lightweight inference
- Optional disk checkpoints: `GleStudentCheckpoint::load(path)`
- Optional HTTP endpoint: `GROWFORMER_GLE_HTTP_ENDPOINT`
- Multi-checkpoint ensemble: weighted average of embeddings
- WASM-safe: `GleStudentCheckpoint::from_bytes(&[u8])`

### LanguageBridge

Projects encoder output into 64-d routing space:
- `Linear(input_dim → 64) + LayerNorm + confidence head`
- EMA smoothing over turn embeddings (alpha=0.2) for multi-turn stability
- OOD rejection: prompts below `ood_similarity_threshold` (0.15) are rejected

### Routing Flow

`route_text(text)` → `LanguageRoutingDecision`:
- `chosen_group_id: Option<GroupId>` — None if OOD rejected
- `best_similarity`, `second_similarity`, `margin`, `confidence`
- `rejected_as_ood: bool`

### Action Types (M3)

`ActionJson` contains:
- `action_type`: CodingAssist, CustomerSupport, KnowledgeQA, SafetyRefusal, ProceduralGuide, Conversational, MultiTurnFollowup, AdversarialBlock, Fallback
- `confidence`, `routed_group_id`, `text_summary`
- `payload`: Optional typed payload (e.g. `CodingAssist { language_hint, task_hint }`)

### Code Generation (M5)

`generate_code_from_action(&action, text)` → `Option<CodeGeneration>`:
- `language`: python, rust, javascript (inferred from text)
- `kind`: implementation, debug, optimize, refactor, explain, generic
- `code`: deterministic stub templates based on language+kind

**Substrate generation (GroupGenEnv — binary token prediction):**
Each generation group owns a dedicated `GroupGenEnv` wrapping a full `NeuralEnvironment`. The generation pipeline:
1. Input: 64-d conditioning vector (language bridge embedding)
2. Substrate: NeuralEnvironment with configurable hidden size (default 256), KWTA competition, physics-based dynamics
3. Output: 1408 binary neurons (128 token slots × 11 bits per token)
4. Decoding: sigmoid → binary threshold → token ID lookup in per-group `TokenDictionary`
5. Training: BCE loss per bit, teacher-forced from target token sequences; engram consolidation protects learned memory traces; priority replay revisits high-loss samples

**Key properties:**
- **200x faster** than sequential character-level prediction — all tokens predicted in a single forward pass
- Per-group dictionaries (up to 2048 entries, 11 bits) — vocabulary tuned to each group's training data
- `GenEnvOverrides` allows auto-configuration of hidden size, k, max_tokens, max_synapses, energy budget
- Engram consolidation: synapses between co-active neurons during training are protected from pruning via `consolidation` weight
- Priority replay: high-loss samples are revisited more frequently to break plateaus
- All outputs are **concrete model outputs** — no templates, no external models, no post-processing

Real learning path (not mocked):
- `M5Learner`: multi-head linear model (lang_head + task_head) over hashing-based text features
- Sequential training: Python → Rust → JS → Design Patterns → Architectural Patterns → Multi-turn → Adversarial
- Anti-forgetting: replay buffer with configurable `--m5-replay-per-epoch` and `--m5-replay-prior-ratio`
- Domain/intent-aware feature injection for cross-domain separability
- Retention target: ≥97% per domain; achieved: 100% across all 7 domains

### Auto Mirror Spawn (M5)

When routing confidence stays below `auto_spawn_threshold` (0.15) for K=10 consecutive calls:
- `track_confidence_for_auto_spawn(&routing)` returns `Some(suggested_mirror_name)`
- Caller decides whether to actually spawn the mirror
- Streak resets on any above-threshold routing or after trigger fires

### M6 Agent Modes

Two modes on the same backend:

**ContextFile mode:**
- Retrieval-augmented: injects context snippets via `push_context_snippet()`
- Reads micro-brain episodic summaries via `read_episodic_summaries()` (read-only)
- Never directly mutates episodic memory

**MicroBrain mode:**
- Uses trained Growformer language pipeline for routing
- May consume retrieval snippets from context-file mode
- Default mode at startup

**Handoff contract:**
- Every mode switch logged: `HandoffLogEntry { from_mode, to_mode, confidence, reason, timestamp_ms }`
- Accessible via `svc.handoff_log()`

**SLO tracking:**
- Latency P95 computed from per-inference measurements
- Checkpoint domain count tracked
- Configurable via `SloConfig { latency_p95_ms, max_memory_bytes, max_checkpoint_domains }`

**Acceptance report:**
- `svc.acceptance_report()` → `AcceptanceReport` with understanding, generation, continual-learning, system, and mode metrics
- CLI: `cargo run -- --acceptance-report`
- API: `GET /v1/acceptance`

---

## Auto-Configuration (`--auto`)

The `--auto` flag on `--train-brain` profiles the training dataset and derives optimal parameters:

**DataProfile** (computed by `profile_training_data`):
- Per-group: sample count, mean/max token length, class imbalance ratio
- Global: total samples, unique intents, vocabulary coverage

**AutoConfig** (computed by `auto_configure`):
- `router_epochs`, `classifier_epochs`, `classifier_lr` — scaled to dataset size
- `gen_epochs` — per group, scaled to sample count and token complexity
- `k_replicas` — parallelism level
- `GenEnvOverrides` — per group: hidden size, competitive_k, max_synapses, energy_budget, max_tokens

**Convergence Monitor** (early stopping):
- Tracks rolling loss window during training
- Stops early when improvement falls below `es_min_imp` over `es_window` epochs (after `es_min_ep` minimum epochs)
- Prevents wasted compute when the substrate has converged

### CLI Architecture

```
cargo run --release -- --train-brain [--auto]     # Train a brain.bin
cargo run --release -- --infer [--prompt "..."]    # Load brain.bin, interactive or single-shot inference
cargo run --bin growformer-demos -- [flags]        # Benchmarks, demos, evaluations
```

---

## WASM Support

The library compiles to `wasm32-unknown-unknown` with `--no-default-features`.

| Feature | Default | What it enables |
|---------|---------|-----------------|
| `native` | yes | reqwest HTTP encoder, filesystem checkpoint loading |
| `server` | yes | axum/tokio HTTP server (growformer-node binary) |
| `parallel` | yes | rayon parallel iterators in compute path |
| `cli` | yes | clap, indicatif, mnist, kiddo for the CLI binary |

WASM-safe API path:
- `LanguageService::new_with_config(LanguageConfig)` (no env vars)
- `svc.load_gle_students_from_bytes(&[&checkpoint_bytes])` (no filesystem)
- `serialize_checkpoint_to_bytes` / `deserialize_checkpoint_from_bytes` (no std::fs)
- `maybe_par_iter!` macro falls back to sequential `.iter()` when `parallel` feature is off

---

## Growformer Node (HTTP Server)

axum-based REST server in `src/server.rs`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/health` | GET | Status, runtime mode, active agent mode |
| `/v1/chat` | POST | Inference: action, generation, or codegen |
| `/v1/chat/stream` | POST | SSE streaming variant of /v1/chat |
| `/v1/mode` | POST | Switch agent mode (context_file / micro_brain) |
| `/v1/acceptance` | GET | M6 acceptance report JSON |

Chat request accepts optional `agent_mode` and `context_snippets` fields.
Response includes `agent_mode` alongside existing `mode`, `latency_ms`, `perf`.

Env vars: `GROWFORMER_NODE_ADDR`, `GROWFORMER_NODE_TOKEN`, `GROWFORMER_NODE_LOG_PATH`,
`GROWFORMER_GLE_CHECKPOINT`, `GROWFORMER_GLE_CHECKPOINTS`, `GROWFORMER_GLE_WEIGHTS`.

---

## Reference Files

- `docs/LANGUAGE.md` — M0-M6 milestone spec and implementation status
- `docs/NODE.md` — growformer-node HTTP API documentation
- `docs/ARCHITECTURE.md` — architecture overview (generation architecture, latest benchmarks, concrete output examples)
- `docs/ROADMAP.md` — current focus, completed phases, future phases
- `docs/CONTINUUM.md` — online / continual learning spec (train-while-on, feedback, persistence)
- `docs/fractal_topology_spec.md` — Phase 3 spec
- `docs/phase3c.md` — Phase 3c composition + episodic (Task C/D, held-out, adding groups)
- `README.md` — quick start, auto-configuration, CLI reference, latest results with concrete generation examples
- `PITCH.md` — investor pitch deck aligned with all docs
- `outputs/phase2_checkpoint.rs` — checkpoint save/load
- `outputs/demo_continual_learning.rs` — Phase 2 frozen flag demo