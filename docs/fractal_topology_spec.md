# GROWFORMER — Fractal Topology & Mirror Dimension
## Architecture Specification v1.0 — Phase 3 Foundation

> Standalone drop-in module for the Growformer project.
> Covers: GlobalObserver · Mirror Dimension · Promotion Gate · Fractal NOW Windows · Integration Points

---

## Table of Contents

1. [Motivation & Problem Statement](#1-motivation--problem-statement)
2. [Architecture Overview](#2-architecture-overview)
3. [Main Dimension](#3-main-dimension)
4. [Mirror Dimension](#4-mirror-dimension)
5. [Promotion Gate](#5-promotion-gate)
6. [GlobalObserver](#6-globalobserver)
7. [DimensionManager](#7-dimensionmanager)
8. [NOW Model Observer Window Mapping](#8-now-model-observer-window-mapping)
9. [Integration Points](#9-integration-points)
10. [Implementation Order](#10-implementation-order)
11. [Demo Function](#11-demo-function)
12. [Glossary](#12-glossary)

---

## 1. Motivation & Problem Statement

Phase 2 proved the Growformer can retain Task A knowledge while learning Task B, achieving **3.9% forgetting** — well below the 10% threshold. However Task B accuracy reached only **52.1%**.

The cause is shared physics substrate. Group B neurons compete with frozen Group A neurons for KWTA slots, inhibition budget, and mass dynamics in the same environment. Group A's high-mass neurons (2.58) dominate the competitive landscape and suppress Group B before it can differentiate.

Partitioning neurons inside a single environment is the wrong abstraction. **The environment itself is the interference.** Each task requires its own complete physics space — its own competitive dynamics, its own geometry, its own mass budget — with no shared substrate at all.

> **Core Insight:** The main dimension only ever contains frozen consolidated knowledge. All training happens in an isolated mirror dimension. When a mirror task meets the promotion threshold, it is promoted to the main dimension as a new frozen group. The main dimension never trains — it only receives promotions.

---

## 2. Architecture Overview

### 2.1 The Three Layers

The Fractal Topology system consists of three layers, each a complete observer unit at a different scale. The pattern **Observer → Training Space → Consolidation → Promotion Gate** repeats at every scale.

```
┌────────────────────────────────────────────────────────────────────┐
│  LAYER 3: GlobalObserver                                           │
│  Watches all dimensions. Maintains embedding library.              │
│  Routes inference. Gates promotion. Detects novelty.               │
├────────────────────────────────────────────────────────────────────┤
│  LAYER 2: DimensionManager                                         │
│  ┌───────────────────┐  ┌───────────────────────────┐             │
│  │  Main Dimension   │  │  Mirror Dimension(s)       │             │
│  │  Frozen only      │  │  Full plasticity           │             │
│  │  All groups frozen│  │  One active task per mirror│             │
│  │  Inference only   │  │  No Group A neurons inside │             │
│  └───────────────────┘  └───────────────────────────┘             │
├────────────────────────────────────────────────────────────────────┤
│  LAYER 1: NeuralEnvironment (existing, unchanged)                  │
│  Physics, KWTA, pruning, backprop, frozen flags                    │
└────────────────────────────────────────────────────────────────────┘
```

### 2.2 The Fractal Property

Every level of the hierarchy has identical structure: an observer, a training space, a consolidation store, and a promotion gate.

| Scale  | Observer                           | Consolidation                          |
|--------|------------------------------------|----------------------------------------|
| Neuron | Activation history, facilitation   | `frozen: bool` flag on Neuron/Synapse  |
| Group  | Group embedding vector             | `freeze_consolidated_pathway()`        |
| Global | GlobalObserver                     | Main Dimension (frozen-only store)     |

> **Fractal definition:** Observer → Training Space → Consolidation → Promotion Gate, repeated at neuron scale, group scale, and global scale. The same pattern at every level of the hierarchy.

---

## 3. Main Dimension

### 3.1 Role

The Main Dimension is the consolidated knowledge store. It contains only proven, frozen groups. It **never trains**. It only receives promoted groups from the Mirror Dimension via the Promotion Gate. At inference time, the GlobalObserver queries the Main Dimension for relevant group activations and composes them.

### 3.2 Struct Definition

```rust
pub struct MainDimension {
    // One NeuralEnvironment per promoted group.
    // Each environment is fully frozen (all neurons frozen=true).
    // Indexed by GroupId for O(1) lookup.
    pub groups: HashMap<GroupId, FrozenGroupEnv>,

    // Shared embedding space — all group vectors registered here.
    // Used by GlobalObserver for cosine similarity routing.
    pub embedding_library: Vec<GroupEmbedding>,

    // Creation order — defines output head indices for multi-task inference.
    pub group_order: Vec<GroupId>,
}

pub struct FrozenGroupEnv {
    pub group_id: GroupId,
    pub task_name: String,
    pub env: NeuralEnvironment,     // fully frozen — all neurons frozen=true
    pub embedding: GroupEmbedding,  // registered at promotion time
    pub accuracy: f32,              // task accuracy at consolidation
    pub promoted_at_epoch: u64,
}
```

### 3.3 Invariants

- No neuron in the Main Dimension has `frozen=false` at any time.
- No training loop runs against any Main Dimension environment.
- Embeddings are computed once at promotion and never recomputed.
- `group_order` is append-only — groups are never removed after promotion.

### 3.4 Inference from Main Dimension

At inference time the GlobalObserver queries all relevant `FrozenGroupEnv`s, collects their output head activations, and composes them.

```rust
impl MainDimension {
    pub fn query(&mut self, input: &[f32], group_ids: &[GroupId])
        -> Vec<(GroupId, Vec<f32>)>
    {
        group_ids.iter().filter_map(|&gid| {
            self.groups.get_mut(&gid).map(|fg| {
                let out = fg.env.predict(input);
                (gid, out)
            })
        }).collect()
    }
}
```

---

## 4. Mirror Dimension

### 4.1 Role

The Mirror Dimension is a complete, isolated `NeuralEnvironment` dedicated to training one task at a time. It has its own neurons, its own KWTA competition, its own mass dynamics, and its own geometry. **No Group A neurons exist inside it.** No frozen groups compete for resources. Task B gets the full environment budget.

### 4.2 Struct Definition

```rust
pub struct MirrorDimension {
    pub task_name: String,
    pub env: NeuralEnvironment,   // full plasticity — no frozen neurons
    pub config: EnvironmentConfig,

    // Training metadata
    pub epochs_trained: u32,
    pub best_accuracy: f32,
    pub current_accuracy: f32,

    // Read-only query channel to Main Dimension.
    // Mirror can ask: 'what does Group A activate on this input?'
    // Used to avoid redundant representation during training.
    // Write direction: mirror → main via PromotionGate only.
    pub main_query_fn: Option<Box<dyn Fn(&[f32]) -> Vec<(GroupId, Vec<f32>)>>>,
}
```

### 4.3 Isolation Guarantees

- Mirror Dimension environment is constructed **fresh** for each new task — not cloned from Main.
- No neuron IDs are shared between Mirror and Main Dimension environments.
- Physics simulation (KWTA, mass, geometry) runs fully independently in Mirror.
- Mirror may query Main **read-only** via `main_query_fn`. It cannot write to Main.
- Mirror is destroyed after successful promotion. A new Mirror is created for the next task.

### 4.4 Read-Only Query Channel

The Mirror can query the Main Dimension to understand what existing consolidated groups already represent. This prevents the Mirror from relearning what Group A already knows and encourages complementary representation.

```rust
// In Mirror training loop, optional complementarity signal:
let main_activations = (self.main_query_fn)(input);
// main_activations: Vec<(GroupId, Vec<f32>)>
//
// Mirror uses this to measure overlap with existing knowledge.
// High overlap → Mirror learns to complement, not duplicate.
// Low overlap  → Mirror learns freely (genuinely new domain).
//
// This is read-only. Mirror cannot modify Main state.
```

### 4.5 Multiple Mirrors

The DimensionManager can maintain multiple Mirror Dimensions simultaneously — one per task being learned in parallel. Each Mirror is fully isolated from all others. The GlobalObserver tracks all active Mirrors and evaluates each independently for promotion readiness.

---

## 5. Promotion Gate

### 5.1 Role

The Promotion Gate is the gatekeeper between Mirror and Main Dimension. Nothing enters the Main Dimension without passing through it. It evaluates a trained Mirror against four criteria and either promotes, rejects, or requests continued training.

### 5.2 Promotion Criteria

| Criterion               | Threshold          | Rationale                                                               |
|-------------------------|--------------------|-------------------------------------------------------------------------|
| Task accuracy           | > 85%              | Mirror must have genuinely learned the task                             |
| Retention compatibility | cosine < 0.80      | Must not duplicate an existing group — no redundant representation      |
| Geometric separation    | > 1.5 units        | Mirror group centroid must be spatially distinct in embedding space     |
| Stability window        | 50 epoch plateau   | Accuracy must not be improving — consolidation requires settled weights |

### 5.3 Promotion Process

```rust
impl PromotionGate {
    pub fn evaluate(&self, mirror: &MirrorDimension, main: &MainDimension)
        -> PromotionDecision
    {
        // 1. Accuracy check
        if mirror.best_accuracy < self.accuracy_threshold {
            return PromotionDecision::ContinueTraining {
                reason: "accuracy below threshold",
            };
        }

        // 2. Redundancy check
        let mirror_emb = compute_group_embedding(&mirror.env, &mirror.calibration_data);
        for existing in &main.embedding_library {
            let sim = cosine_similarity(&mirror_emb.vector, &existing.vector);
            if sim > self.redundancy_threshold {
                return PromotionDecision::Reject {
                    reason: "redundant with existing group",
                    similar_to: existing.group_id,
                };
            }
        }

        // 3. Stability check
        if !mirror.is_stable(self.stability_window) {
            return PromotionDecision::ContinueTraining {
                reason: "accuracy still improving",
            };
        }

        PromotionDecision::Promote
    }

    pub fn promote(&self, mirror: MirrorDimension, main: &mut MainDimension,
                   observer: &mut GlobalObserver) -> GroupId
    {
        // Freeze all neurons in mirror environment
        for neuron in mirror.env.neurons.values_mut() {
            neuron.frozen = true;
            for syn in &mut neuron.synapses { syn.frozen = true; }
        }

        // Compute and register embedding
        let embedding = compute_group_embedding(&mirror.env, &mirror.calibration_data);
        let group_id = main.register_group(mirror.env, embedding.clone());

        // Notify GlobalObserver
        observer.register_group(group_id, embedding);

        // Mirror is consumed — destroy it
        group_id
    }
}
```

### 5.4 Rejection Handling

| Rejection Reason          | Action                                                                              |
|---------------------------|-------------------------------------------------------------------------------------|
| Accuracy below threshold  | Continue training in Mirror — no action                                             |
| Redundant with Group A    | Attempt composition: VirtualGroup(Mirror + GroupA). Promote VirtualGroup if it works |
| Still improving           | Continue training — evaluate again after stability window                           |
| Geometric overlap         | Increase `group_boundary_penalty` in Mirror config, retrain final epochs            |

---

## 6. GlobalObserver

### 6.1 Role

The GlobalObserver is the highest NOW window. It watches all dimensions simultaneously, maintains the shared embedding space, routes inference inputs to relevant groups, gates promotion, detects novelty, and provides temporal continuity across tasks. **It does not train — it observes, decides, and coordinates.**

### 6.2 Struct Definition

```rust
pub struct GlobalObserver {
    // Embedding library — all promoted groups registered here
    pub embedding_library: Vec<GroupEmbedding>,

    // Activity history — what groups have been recently active
    // Slow integration window: temporal continuity across tasks
    pub group_activity: HashMap<GroupId, VecDeque<f32>>,
    pub activity_window: usize,  // e.g. 1000 recent activations

    // Episodic memory — successful compositions indexed by input signature
    pub episodic_memory: EpisodicMemory,

    // Router — maps input to group relevance weights
    pub router: GroupRouter,

    // Promotion gate
    pub promotion_gate: PromotionGate,

    // Active mirrors being tracked
    pub active_mirrors: Vec<MirrorDimension>,

    // Coherence score — are active groups agreeing?
    pub coherence: f32,
}
```

### 6.3 Core Methods

```rust
impl GlobalObserver {

    // === INFERENCE ===

    // Route an input to relevant groups, compose outputs
    pub fn infer(&mut self, input: &[f32], main: &mut MainDimension) -> Vec<f32> {
        // 1. Check episodic memory for known input signature
        if let Some(episode) = self.episodic_memory.find(input, threshold=0.95) {
            if episode.residual < 0.05 { return self.recall(episode, main, input); }
        }

        // 2. Route: which groups are relevant?
        let attention = self.router.attend(input, &self.embedding_library);

        // 3. Query relevant groups from Main Dimension
        let relevant: Vec<GroupId> = attention.iter()
            .filter(|(_, w)| *w > 0.1)
            .map(|(gid, _)| *gid).collect();
        let outputs = main.query(input, &relevant);

        // 4. Compose outputs weighted by attention
        let result = self.compose(&outputs, &attention);

        // 5. Update activity history (temporal continuity)
        self.update_activity(&relevant);

        result
    }

    // === LEARNING DECISIONS ===

    // Decide what to do with a new task
    pub fn decide_learning_op(&self, input_signature: &[f32], residual: f32)
        -> LearningOp
    {
        match residual {
            r if r < 0.05 => LearningOp::Recall,
            r if r < 0.15 => LearningOp::ReweightRouter,
            r if r < 0.30 => LearningOp::Compose,
            _             => LearningOp::SpawnMirror,
        }
    }

    // === PROMOTION ===

    // Evaluate all active mirrors for promotion readiness
    pub fn evaluate_mirrors(&mut self, main: &mut MainDimension) {
        let mut to_promote = vec![];
        for mirror in &self.active_mirrors {
            match self.promotion_gate.evaluate(mirror, main) {
                PromotionDecision::Promote => to_promote.push(mirror.task_name.clone()),
                _ => {}
            }
        }
        for name in to_promote {
            self.promote_mirror(&name, main);
        }
    }

    // === COHERENCE ===

    // Measure agreement between active group outputs
    pub fn measure_coherence(&self, outputs: &[(GroupId, Vec<f32>)]) -> f32 {
        // Pairwise cosine similarity of output vectors.
        // High coherence: groups agree on input interpretation.
        // Low coherence: groups disagree → novel input or composition failure.
        todo!()
    }
}
```

---

## 7. DimensionManager

### 7.1 Role

The DimensionManager is the top-level runtime struct. It owns the Main Dimension, all active Mirror Dimensions, and the GlobalObserver. It is the **single entry point** for all external interaction — training, inference, checkpointing. The caller never interacts with individual dimensions directly.

### 7.2 Struct Definition

```rust
pub struct DimensionManager {
    pub main: MainDimension,
    pub mirrors: HashMap<String, MirrorDimension>,  // task_name → mirror
    pub observer: GlobalObserver,
    pub config: DimensionManagerConfig,
}

pub struct DimensionManagerConfig {
    pub mirror_config: EnvironmentConfig,    // config for new mirrors
    pub mirror_layer_sizes: Vec<usize>,       // e.g. [2, 16, 16, 1]
    pub promotion_check_interval: u32,        // epochs between gate checks
    pub max_concurrent_mirrors: usize,        // parallel task limit
    pub calibration_samples: usize,           // samples for embedding computation
}
```

### 7.3 Public API

```rust
impl DimensionManager {

    // Create a new DimensionManager with empty Main Dimension
    pub fn new(config: DimensionManagerConfig) -> Self;

    // === INFERENCE ===
    // Route input through GlobalObserver to relevant Main Dimension groups
    pub fn infer(&mut self, input: &[f32]) -> Vec<f32>;

    // === TRAINING ===
    // Spawn a new Mirror Dimension for a task
    pub fn spawn_mirror(&mut self, task_name: &str, seed: u64) -> &mut MirrorDimension;

    // Train one tick in a named mirror
    pub fn train_mirror_tick(
        &mut self,
        task_name: &str,
        input: &[f32],
        target: &[f32],
        rng: &mut impl Rng,
    ) -> TrainResult;

    // Train a full epoch in a named mirror
    pub fn train_mirror_epoch(
        &mut self,
        task_name: &str,
        data: &[([f32; 2], [f32; 1])],
        rng: &mut impl Rng,
    ) -> EpochResult;

    // Check all mirrors for promotion readiness
    pub fn evaluate_promotions(&mut self);

    // Force promote a mirror (used in testing / demos)
    pub fn force_promote(&mut self, task_name: &str) -> GroupId;

    // === CHECKPOINTING ===
    pub fn save(&self, path: &str);
    pub fn load(path: &str) -> Self;

    // === INTROSPECTION ===
    pub fn list_groups(&self) -> Vec<GroupSummary>;
    pub fn list_mirrors(&self) -> Vec<MirrorSummary>;
    pub fn coherence(&self) -> f32;
}
```

---

## 8. NOW Model Observer Window Mapping

The Nested Observer Windows model proposes that subjective experience emerges from stacked hierarchical observer windows at different spatial and temporal scales. The Fractal Topology maps each NOW level to a concrete architectural component.

| NOW Level         | Timescale          | Growformer Component                                          |
|-------------------|--------------------|---------------------------------------------------------------|
| L1: Micro         | 1 tick             | Individual neuron activation, synapse facilitation            |
| L2: Local         | 1 epoch            | NeuralEnvironment: KWTA, mass, backprop                       |
| L3: Group         | 10–100 epochs      | Consolidated group, frozen pathway, group embedding           |
| L4: Mirror        | 100–4000 epochs    | MirrorDimension: isolated task training environment           |
| L5: Promotion     | Task completion    | PromotionGate: gating of new knowledge into main store        |
| L6: Global        | Lifetime           | GlobalObserver: cross-group coherence, episodic memory        |
| L7: Composition   | On demand          | VirtualGroup: blending frozen groups for novel tasks          |

### 8.1 Temporal Continuity

The NOW model identifies the highest observer window as the source of temporal continuity — the sense that now connects to the recent past. In the Fractal Topology, the GlobalObserver's activity history (`group_activity: HashMap<GroupId, VecDeque<f32>>`) provides this property. The system knows not just what groups know, but what they have recently been activated on, across all tasks and all time.

### 8.2 The Many Minds Problem — Solved

Without a GlobalObserver, each consolidated group is an independent observer with no shared reference frame. Group A knows spirals, Group B knows circles, and neither knows the other exists.

The Fractal Topology solves this at the architectural level: all groups share a single embedding library owned by the GlobalObserver. Coherence is measured continuously. Composition is available on demand.

---

## 9. Integration Points

This module is designed to **drop into the existing Growformer project with minimal changes to existing files**. All new code lives in new files. Existing `NeuralEnvironment` is unchanged except for the `frozen` flag already added in Phase 2.

### 9.1 New Files

| File                            | Contents                                                         |
|---------------------------------|------------------------------------------------------------------|
| `src/dimension/mod.rs`          | Module root. Re-exports all public types.                        |
| `src/dimension/main_dim.rs`     | `MainDimension`, `FrozenGroupEnv` structs and impl.             |
| `src/dimension/mirror_dim.rs`   | `MirrorDimension` struct and impl.                               |
| `src/dimension/promotion.rs`    | `PromotionGate`, `PromotionDecision`, promotion logic.           |
| `src/dimension/observer.rs`     | `GlobalObserver` struct and impl.                                |
| `src/dimension/manager.rs`      | `DimensionManager` — top-level entry point.                     |
| `src/dimension/embedding.rs`    | `GroupEmbedding`, `compute_group_embedding()`, `cosine_similarity()`. |
| `src/dimension/router.rs`       | `GroupRouter` — heuristic and learned variants.                  |
| `src/dimension/episodic.rs`     | `EpisodicMemory`, `Episode` structs.                             |
| `src/dimension/composition.rs`  | `VirtualGroup`, `compose_groups()`, `find_composition()`.        |

### 9.2 Changes to Existing Files

| File                | Change                                                                         |
|---------------------|--------------------------------------------------------------------------------|
| `src/lib.rs`        | Add: `pub mod dimension;`                                                      |
| `src/neuron.rs`     | Already done: `frozen: bool` on `Neuron` and `Synapse`. No further changes.   |
| `src/environment.rs`| Already done: `freeze_consolidated_pathway()`, frozen checks in backprop.     |
| `src/main.rs`       | Add `demo_fractal()` using `DimensionManager` API. Existing demos unchanged.  |

> **Zero breaking changes.** `NeuralEnvironment`, `Neuron`, `Synapse`, and all existing systems are unchanged. The dimension module wraps `NeuralEnvironment` — it does not modify it. All existing demos continue to work.

### 9.3 Cargo.toml

No new dependencies required. `serde` and `serde_json` are already present. All new structs derive `Serialize`/`Deserialize` following existing patterns.

---

## 10. Implementation Order

Each step is independently testable. Do not proceed to the next step until the current step has a passing test.

### Step 1 — GroupEmbedding (`src/dimension/embedding.rs`)
1. Implement `GroupEmbedding` struct with `vector`, `task_signature`, `accuracy`, `group_id` fields.
2. Implement `compute_group_embedding()` — run calibration data through env, mean-pool activations.
3. Implement `cosine_similarity()` and `retrieve_relevant_groups()`.
4. **Test:** compute embeddings for Task A (spiral) and Task B (circles). Verify dissimilar (cosine < 0.5).

### Step 2 — MainDimension (`src/dimension/main_dim.rs`)
1. Implement `MainDimension` and `FrozenGroupEnv` structs.
2. Implement `register_group()` — takes a frozen `NeuralEnvironment`, stores it.
3. Implement `query()` — runs `predict()` on relevant frozen environments.
4. **Test:** register Task A checkpoint as a `FrozenGroupEnv`. Query with spiral input. Verify 90%+ accuracy.

### Step 3 — MirrorDimension (`src/dimension/mirror_dim.rs`)
1. Implement `MirrorDimension` struct with fresh `NeuralEnvironment` construction.
2. Implement `train_epoch()` and `is_stable()` methods.
3. Implement read-only `main_query_fn` channel (optional, can be `None` initially).
4. **Test:** train circles from scratch in Mirror. Verify 90%+ without any Group A interference.

### Step 4 — PromotionGate (`src/dimension/promotion.rs`)
1. Implement `PromotionGate` with configurable thresholds.
2. Implement `evaluate()` returning `PromotionDecision`.
3. Implement `promote()` — freeze Mirror, compute embedding, register with `MainDimension`.
4. **Test:** train circles to 90%+, call `evaluate()`, verify `Promote` decision. Call `promote()`, verify `MainDimension` has two groups.

### Step 5 — GlobalObserver (`src/dimension/observer.rs`)
1. Implement `GlobalObserver` with embedding library, activity history, episodic memory stubs.
2. Implement `infer()` using heuristic cosine router (no learned weights yet).
3. Implement `evaluate_mirrors()` calling `PromotionGate` on all active mirrors.
4. **Test:** infer on spiral input — verify Group A activates. Infer on circles input — verify Group B activates.

### Step 6 — DimensionManager (`src/dimension/manager.rs`)
1. Implement `DimensionManager` as thin coordinator over all components.
2. Implement public API: `infer()`, `spawn_mirror()`, `train_mirror_epoch()`, `evaluate_promotions()`.
3. Implement `save()`/`load()` using `serde_json`.
4. **Test:** full Phase 2 demo using `DimensionManager` API. Task A in Mirror → promoted. Task B in new Mirror. Verify retention >85% and Task B accuracy >85%.

### Step 7 — EpisodicMemory & VirtualGroup (`episodic.rs`, `composition.rs`)
1. Implement `EpisodicMemory` with `find()` by cosine similarity on input signatures.
2. Implement `VirtualGroup` with blending weight training.
3. Implement `find_composition()` with greedy residual-driven group selection.
4. **Test:** create a task combining spiral + circles features. Verify composition finds GroupA + GroupB without spawning a new mirror.

---

## 11. Demo Function

Add to `main.rs` alongside existing demos.

```rust
fn demo_fractal_continual_learning() {
    println!("--- Demo 6: Fractal Continual Learning ---");

    let config = DimensionManagerConfig {
        mirror_config: phase2_base_config(),
        mirror_layer_sizes: vec![2, 16, 16, 1],
        promotion_check_interval: 500,
        max_concurrent_mirrors: 2,
        calibration_samples: 100,
    };

    let mut dm = DimensionManager::new(config);
    let mut rng = StdRng::seed_from_u64(42);
    let mut data_rng = StdRng::seed_from_u64(99);

    // === TASK A: Spiral ===
    // Train in isolated Mirror with full environment budget
    let spiral_data = generate_spiral_data(400, &mut data_rng);
    dm.spawn_mirror("spiral", 42);

    for epoch in 0..4000 {
        let result = dm.train_mirror_epoch("spiral", &spiral_data, &mut rng);
        if epoch % 500 == 0 {
            println!("  [spiral] epoch {} | loss={:.4} | acc={:.1}%",
                epoch, result.loss, result.accuracy * 100.0);
        }
        if epoch % 500 == 0 { dm.evaluate_promotions(); }
    }

    // Force promote if not auto-promoted
    let spiral_group = dm.force_promote("spiral");
    println!("Task A promoted as Group {}", spiral_group);

    // === TASK B: Circles ===
    // Completely fresh Mirror — zero competition from Task A
    let circles_data = generate_concentric_circles_data(400, &mut data_rng);
    dm.spawn_mirror("circles", 43);  // different seed

    for epoch in 0..4000 {
        let result = dm.train_mirror_epoch("circles", &circles_data, &mut rng);
        if epoch % 500 == 0 {
            // Retention: query Main Dimension for Task A — unaffected by Task B
            let spiral_acc = dm.evaluate_main_group(spiral_group, &spiral_data);
            println!("  [circles] epoch {} | loss={:.4} | acc={:.1}% | A_retain={:.1}%",
                epoch, result.loss, result.accuracy * 100.0, spiral_acc * 100.0);
        }
    }

    let circles_group = dm.force_promote("circles");
    println!("Task B promoted as Group {}", circles_group);

    // === RESULTS ===
    let final_spiral  = dm.evaluate_main_group(spiral_group, &spiral_data);
    let final_circles = dm.evaluate_main_group(circles_group, &circles_data);
    println!("\nFinal Task A: {:.1}%", final_spiral * 100.0);
    println!("Final Task B: {:.1}%", final_circles * 100.0);

    // === INFERENCE TEST ===
    // GlobalObserver routes without task label — no caller-specified task
    let spiral_out  = dm.infer(&[0.3_f32, 0.4]);
    let circles_out = dm.infer(&[0.0_f32, 0.9]);
    println!("Routed spiral input  → output: {:.3}", spiral_out[0]);
    println!("Routed circles input → output: {:.3}", circles_out[0]);
}
```

---

## 12. Glossary

| Term                   | Definition                                                                                                          |
|------------------------|---------------------------------------------------------------------------------------------------------------------|
| `MainDimension`        | Consolidated knowledge store. Contains only frozen promoted groups. Never trains.                                   |
| `MirrorDimension`      | Isolated training environment for one task. Full plasticity. No shared substrate with Main.                        |
| `PromotionGate`        | Evaluates Mirror against accuracy, redundancy, geometric, and stability criteria before allowing promotion.         |
| `DimensionManager`     | Top-level runtime. Owns Main, all Mirrors, and GlobalObserver. Single public API entry point.                      |
| `GlobalObserver`       | Highest NOW window. Routes inference, gates promotion, maintains coherence and episodic memory across all tasks.    |
| `GroupEmbedding`       | Fixed vector encoding a group's mean activation pattern. Computed once at promotion. Never recomputed.              |
| `GroupRouter`          | Attention mechanism mapping input to group relevance weights via cosine similarity over the embedding library. |

### Learned router and data reference

To integrate a **learned router** and reference data from the neural router:

- **id** — `GroupId` (u32). Primary key for each group. Use as the **target** for the learned router: training data are `(input, target_group_id)`. At calibration or after promotion we have task-labeled data (e.g. "spiral" → group 0); resolve task name to `GroupId` via `main.group_order` / `DimensionManager` and record `(input, group_id)` to train the router.
- **desc** — Optional human-readable `description` per group (e.g. "Spiral 2D binary classification"). Stored on `GroupEmbedding`; use for logging, debugging, or future text-conditioned routing.
- **metatags** — Optional `Vec<String>` per group (e.g. `["spiral", "classification", "2d"]`). Use to **filter** which groups participate in routing (e.g. only groups with tag `"classification"`) or as extra input to the router; supports composition and discovery.

**Integration:** `LearnedRouter` in `router.rs` is a small MLP `[input_dim, hidden, num_groups]`. Train with `train_step(input, target_group_id)`; infer with `choose_group(input)` (argmax logits). Observer can use learned router when present (replace or blend with heuristic). When `num_groups` increases (new promotion), rebuild or extend the router output layer.

---
| `EpisodicMemory`       | Persistent composition history indexed by input signature. Zero-training recall for previously seen problem types.  |
| `VirtualGroup`         | Composition of two or more frozen groups with trainable blending weights. Component groups stay frozen.             |
| Fractal property       | Observer → Training Space → Consolidation → Promotion Gate repeating at neuron, group, and global scale.           |
| Read-only channel      | Mirror may query Main Dimension activations to avoid redundant representation. Cannot write to Main.                |
| Temporal continuity    | GlobalObserver activity history spanning all tasks. Provides the NOW model's highest observer window property.      |

---

*Growformer Fractal Topology Spec v1.0 — Phase 3 Foundation*
