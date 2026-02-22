# **Neuro — A Multidimensional Neural Training Environment in Rust**

## **What Neuro Is**

Neuro is not “just another neural network implementation.”  
It’s something much rarer: a **multidimensional computational medium** where learning emerges from the interaction of geometry, metabolism, sparsity, timing, energy flow, and spatial constraints — not from a fixed algebraic recipe.

Traditional neural networks are static graphs with dynamic weights.  
Neuro is a **dynamic graph with dynamic structure**, where:

- neurons exist in 3D space  
- synapses grow, strengthen, weaken, and die  
- firing consumes energy  
- connectivity has metabolic cost  
- sparsity emerges naturally  
- symmetry forms and breaks  
- timing shapes plasticity  
- geometry feeds back into learning  
- structure is an *output* of training, not an input  

Instead of forcing intelligence into a rigid architecture, Neuro simulates an **ecosystem** of interacting forces.  
Learning is not a single update rule — it is the emergent behavior of six coupled systems:

1. **Weight dynamics** (backprop)  
2. **Geometry** (neurons drift toward correlated partners)  
3. **Timing** (STDP)  
4. **Metabolic cost** (energy‑driven pruning)  
5. **Connectivity** (growth and dissolution of synapses)  
6. **Structural symmetry** (mirror group coupling)

The result is a system that behaves less like a classical MLP and more like a **living computational organism** — one that adapts, collapses, recovers, specializes, and self‑organizes.

Neuro is a platform for exploring a different paradigm of learning:  
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
neuro/
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
    let output = self.forward(input);            // temporal-gated forward pass
    let loss   = self.backprop(&output, target); // dim 1: weight update
    update_stdp_layer(...);                      // dim 3: timing-dependent update
    apply_metabolic_pressure(...);               // dim 4: prune over-budget synapses
    update_geometry(...);                        // dim 2: spatial drift
    grow_synapses(...);                          // dim 5: form new connections
    apply_mirror_coupling(...);                  // dim 6: structural symmetry
    self.time += 1.0;
}
```

The forward pass itself is non‑standard: signal strength is gated by a temporal decay function so that recently fired neurons carry more weight.

---

## **Running the Examples**

```bash
cargo run
```

Two demos run in sequence:

### **Demo 1: XOR**  
2 inputs → 4 hidden → 1 output, 3000 ticks. Hidden layer split into mirror groups. Structural report prints synapse distribution, energy cost, geometric spread, symmetry score, whorl count, and per‑neuron state.

### **Demo 2: Spiral Classification**  
Two interleaved spirals, 200 samples per class.  
2 inputs → 8 hidden → 4 hidden → 1 output, 5000 ticks.  
A harder nonlinear boundary that exercises geometry, growth, and timing systems.

---

## **Configuration**

All behavior is controlled through `EnvironmentConfig`:

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
}
```

---

## **Dependencies**

```toml
[dependencies]
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

