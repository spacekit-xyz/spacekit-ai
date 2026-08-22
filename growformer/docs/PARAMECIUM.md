# Growformer Paramecium

## The Biological Premise

Paramecium caudatum are single-celled organisms roughly the width of a human hair.

They have no brain, no neurons, no synapses, and no central nervous system of any kind.

But what they do have is ~100,000 microtubules…

With that substrate alone, they can:

→ Swim in controlled helical trajectories
→ Modulate speed continuously
→ Execute graded avoidance reactions (reverse, pivot, resume)
→ Escape predators with emergency burst reversals
→ Fire localized volleys of 8,000 trichocyst harpoons
→ Navigate toward food via chemotaxis
→ Orient in electric fields (galvanotaxis)
→ Orient to gravity (gravitaxis)
→ Sense and navigate thermal gradients
→ Sense and navigate toward light
→ Detect and follow surfaces (thigmotaxis)
→ Forage biofilms
→ Generate feeding currents and sort particles at the cytostome
→ Engage in reciprocal sex with mating-type recognition, nuclear exchange, and complete genomic reconstruction
→ Self-fertilize when no partner is available (autogamy)
→ Habituate to repeated stimuli (primitive learning)
→ Inherit cortical MT architecture epigenetically independent of the genome

17 distinct behaviors. One lattice. Zero neurons.

The coordination layer is the infraciliary lattice — a microtubule-based grid connecting all 5,000 ciliary basal bodies into a single cell-wide network.

Every cilium is a terminal node on a microtubule mesh that coordinates metachronal waves across the entire cell surface — thousands of appendages phase-locked into coherent motion by a substrate that predates the nervous system by a billion years.

The neuron didn't invent computation. It inherited microtubules.

---

## The Implementation

`src/dimension/paramecium.rs` — lattice-only sub-neuronal inference. No `NeuralEnvironment`. No synapses. No backpropagation.

### Primitives

| Biological | Implementation | Purpose |
|---|---|---|
| Cilium basal body | `BehavioralProgram` | A node on the lattice with a stored response pattern |
| Infraciliary lattice | `InfraciliaryLattice` | The complete computation substrate |
| Metachronal wave | `WaveState` | Phase-locked EMA field propagating across all nodes |
| Chemotaxis | `nearest_program()` (cosine similarity) | Gradient sensing to select the right program |
| Habituation | `habituation` counter on each program | Dampen response to repeated identical stimuli |
| Burst reversal | `avoidance_reaction()` | Emergency switch to lowest-habituation program |
| Trichocyst volley | `trichocyst_volley()` | Fire top-K programs simultaneously |
| Autogamy | `autogamy()` | Self-reorganize: merge similar programs, prune dead ones |
| Development | `develop()` | Build programs from data via EMA centroid alignment |

### Inference Loop

```
Input text
  → encode_and_bridge (384d → 64d)
  → E8 lattice quantization
  → cosine similarity to all program centroids (chemotaxis)
  → wave propagation from best match
  → habituation check (dampen if same program fires again)
  → wave-modulated selection (neighboring programs can override)
  → fire selected program → decode token sequence
  → online EMA centroid drift (learning without backprop)
  → wave decay for next cycle
```

### Training

No gradient descent. No error backpropagation. The lattice self-organizes through **wave-phase alignment**:

1. Present a (embedding, response) sample
2. Find nearest existing program by cosine similarity
3. If similar enough: shift the program's EMA centroid toward the new input
4. If too distant: spawn a new program at this location
5. Coherence tracks how tightly clustered the activating inputs are

This is the paramecium learning model: repeated exposure to a stimulus shifts the attractor. No teacher signal needed beyond "this input produced this response."

### Usage

**REPL:**
```
/paramecium what is the observer pattern?
/pm explain binary search
```

**API:**
```json
POST /v1/chat
{ "mode": "paramecium", "message": "what is the observer pattern?" }
```

**Programmatic:**
```rust
let mut lattice = InfraciliaryLattice::new(dictionary);
lattice.develop(&samples, 0.95);
let response = lattice.respond(&embedding);
```

### Build from existing brain

The paramecium auto-builds from any loaded brain's codebook — extracting archetype centroids and token sequences as behavioral programs:

```rust
svc.build_paramecium(); // extracts from active brain
let (action, resp) = svc.paramecium_respond("what is the observer pattern?")?;
```

### Memory footprint

A 50-program paramecium lattice uses **under 100KB**. Compare:

| System | Size |
|---|---|
| GPT-3 | 350 GB |
| GPT-4 | ~1.8 TB |
| Growformer micro-brain | 18 MB |
| **Growformer Paramecium** | **< 100 KB** |

### Learning-side: how the paramecium improves training

The paramecium is not just a fast inference path. It provides signals that guide the neuronal substrate's learning:

| Method | Signal | Used by |
|---|---|---|
| `novelty_score()` | How well-known a sample is (0.0=novel, 1.0=redundant) | Curriculum ordering in early epochs |
| `curriculum_order()` | Rank a batch hardest-first | Pre-loss training phase (epochs 0–50) |
| `discover_groups()` | Bottom-up cluster count from data | Diagnostic: compare hand-labeled vs discovered groups |
| `archetype_seeds()` | Lattice centroids as codebook initializers | Codebook construction warm-start |
| `wave_conditioning()` | Wave-phase pattern as routing signal | Supplementary conditioning for generation envs |

**Curriculum learning**: Before the neural substrate has trained (no loss data), the lattice already knows which samples are novel and which are redundant. Novel samples get presented first in early epochs. Once loss-based replay kicks in (epoch 50+), novelty blends with per-sample loss for priority replay — samples that are both high-loss and lattice-novel get more training steps.

**Group discovery**: The lattice's `develop()` naturally discovers clusters via spawn threshold. Running `discover_groups()` on all training embeddings reports how many natural clusters exist vs the hand-labeled group structure. When these diverge significantly, it signals the group topology should be reconsidered.

**Post-training paramecium**: After `train_brain` completes, a final paramecium is built from the trained codebooks and stored with the brain. This means the lattice is immediately available for continuum learning at runtime — providing curriculum signals for online feedback training, not just fast inference.

### Where it fits in the architecture

```
┌─────────────────────────────────────────────────┐
│  AI Operating System (orchestration, policy)     │
├─────────────────────────────────────────────────┤
│  Neuronal Groups (domain specialists, 18MB)      │
│  ├── g0: support/general                         │
│  ├── g1: patterns                                │
│  ├── g2: coding                                  │
│  └── g3: reasoning                               │
├─────────────────────────────────────────────────┤
│  Paramecium (sub-neuronal substrate, <100KB)     │
│  ├── gradient sensing (chemotaxis)               │
│  ├── wave coordination (metachronal)             │
│  ├── habituation (primitive learning)            │
│  ├── burst reversal (emergency avoidance)        │
│  ├── novelty detection (curriculum signal) ──────┼──→ training
│  ├── group discovery (cluster topology)   ──────┼──→ training
│  └── wave conditioning (routing signal)   ──────┼──→ training
└─────────────────────────────────────────────────┘
```

The paramecium sits below the neuronal groups in the hierarchy, but its influence flows **upward into training**, not just downward into inference. It's the organism's first contact with data — fast enough to pre-screen, structured enough to provide curriculum signals, and persistent enough to guide continuum learning.

Same architecture at three scales. Same principle: intelligence emerges from the composition of specialized substrates, not from one substrate being enormous.
