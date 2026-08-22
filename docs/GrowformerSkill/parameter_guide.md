# EnvironmentConfig Parameter Guide

Every field with validated values, failed experiments, and notes.

---

## Learning

| Parameter | Validated | Notes |
|-----------|-----------|-------|
| learning_rate | 0.15 (spiral/circles), 0.05 (XOR) | High early lr critical for gradient flow through lateral inhibition |
| lr_decay | 0.00008 | Exponential: lr * exp(-lr_decay * epoch). 0.0 for XOR |
| weight_decay | 0.0000025 (spiral/circles), 0.0 (XOR) | UNIFORM — same for input and hidden synapses. Never split by is_input |
| bias_decay | 0.0 | NEVER ENABLE. Any value annihilates hidden biases over 3.2M steps |
| dropout_rate | 0.1 | Applied 100% to layer-1, 50% to deeper layers |
| weight_clamp | 5.0 (spiral/circles), 50.0 (XOR) | XOR needs 50+ to push past 0.36/0.64 plateau |

**Input weight decay history:**
- 0% decay (eff_decay=0.0): input energy=38, accuracy=59% — too high, sigmoid saturates
- 10% decay (eff_decay*0.1): input energy=24, accuracy=62% — still too high
- 100% decay (uniform): input energy=3–8, accuracy=64.5%+ — correct equilibrium
- **Rule: never split weight decay by layer. Uniform 0.0000025 produces correct input energy.**

---

## Competition — THE most important section

| Parameter | Validated | Notes |
|-----------|-----------|-------|
| competitive_k | 4 | Global KWTA: keep top-4 of 16. Do NOT lower for easier tasks — let pruning handle sparsity |
| **KWTA residual** | **0.02** | **In environment.rs: n.activation *= 0.02 (NOT 0.0). This single change: +15% accuracy** |
| kwta_radius | 0.0 | Local KWTA. KEEP AT 0.0. All radius values tested and failed permanently |
| kwta_suppression | 0.2 | Irrelevant while kwta_radius=0.0 |
| lateral_inhibition | 0.12 | Reaction-diffusion inhibition. Active from epoch 0, no warmup |
| sigma_inhib | 2.0 | Inhibition spatial decay length |

**KWTA residual is not in EnvironmentConfig — it is hardcoded in environment.rs:**
```rust
// Global KWTA fallback block in forward_pass():
for (i, (nid, _)) in acts.iter().enumerate() {
    if i >= k {
        if let Some(n) = self.neurons.get_mut(nid) {
            n.activation *= 0.02;  // MUST be 0.02, never 0.0
        }
    }
}
```

**Why 0.02 matters:** Hard-zero (0.0) means 12 of 16 hidden neurons receive zero gradient
every forward pass. Only the 4 KWTA winners' input synapses update. Over millions of ticks,
12 neurons' input synapses decay toward zero — input energy collapses, network can't learn.
With 0.02, losers still pass tiny gradient that prevents synapse decay while maintaining competition.

---

## Physics / Geometry

| Parameter | Validated | Notes |
|-----------|-----------|-------|
| thermal_noise | 0.02 | Mandatory symmetry breaking. Johnson-Nyquist analog — prevents mirror-locked collapse |
| k_repel | 0.2 | Same-layer repulsion coefficient |
| gravity_g | 0.05 | Pulls neurons toward layer centroid |
| damping | 0.2 | Velocity damping per tick: v *= (1 - damping) |
| debye_length | 1.5 | Screening length — smooth decay replaces hard repulsion_radius cutoff |
| geometry_interval | 500 | Ticks between physics updates |
| geometry_noise | 0.0 | Replaced by thermal_noise — keep at 0.0 |
| physics_dt | 0.05 | Integration timestep |

---

## Mass-Competition Coupling

| Parameter | Validated | Notes |
|-----------|-----------|-------|
| mass_growth | 0.0005 | Winners gain this mass per sample above win_threshold |
| mass_decay | 0.00009 | **Must scale with sample count** — see formula below |
| mass_win_threshold | 0.15 | Activation above this = winner this sample |
| mass_min | 0.3 | Losers stay as light exploratory drifters |
| mass_max | 3.0 | Winners cap — prevents single hub monopoly |

**mass_decay calibration (CRITICAL):**
- 400 total samples → 0.00015
- 800 total samples → 0.00009
- 1600 total samples → 0.000045
- Formula: mass_decay ≈ 0.00015 × (400 / n_samples)
- Wrong mass_decay signature: mass column reads exactly 1.56–1.57 every epoch (no variation)

---

## Homeostasis

| Parameter | Validated | Notes |
|-----------|-----------|-------|
| homeostasis_lr | 0.0 | DISABLED. When enabled, equalizes all neurons to same bias — kills diversity |
| homeostasis_target | 0.30 | Irrelevant while homeostasis_lr=0.0 |
| homeostasis_tau | 0.001 | EMA decay for running activation (irrelevant while disabled) |

---

## Pruning — Three-Phase

| Parameter | Validated | Notes |
|-----------|-----------|-------|
| prune_interval | 500 | Ticks between pruning runs |
| prune_early_age | 100 | Phase 1: cull silent trial connections |
| prune_early_threshold | 0.003 | Phase 1 strength threshold |
| prune_mid_threshold | 0.01 | Phase 2 strength threshold |
| prune_mid_facilitation_floor | 1.02 | Phase 2: facilitation>1.02 means synapse was used |
| prune_long_age | 2000 | Phase 3: structural pruning of old dormant synapses |
| prune_long_dormancy | 1000 (spiral), 100000 (XOR) | Ticks inactive before phase-3 candidate |
| prune_long_threshold | 0.005 | Phase 3 strength threshold |
| prune_stop_tick | 0 | KEEP AT 0 (disabled). Any value creates always-on attractor |

**MINIMUM SYNAPSE FLOOR — required in growth.rs prune_three_phase:**
```rust
if input_ids.contains(&neuron.id) { continue; }  // protect input neurons
let min_synapses = 2;
if neuron.synapses.len() <= min_synapses { continue; }  // prevent neuron death
```
Without this: KWTA losers accumulate low facilitation, all synapses pruned, neuron permanently
disconnected, zero gradient, dead weight. A neuron with 2 synapses can still learn; 0 cannot recover.

---

## Synapses / Growth

| Parameter | Validated | Notes |
|-----------|-----------|-------|
| max_synapses_per_neuron | 64 | Capacity cap |
| growth_radius | 2.0 | New synapse growth radius. Was 0.0 in circles (copy-paste bug) |
| pruning_threshold | 0.001 | Minimum strength for metabolic pruning |
| facilitation_bonus | 0.002 | LTP consolidation bonus for highly-used synapses |
| energy_budget_per_neuron | 100.0 | High value effectively disables metabolic pruning as primary gate |

**Input synapse protection:**
- Protected from PRUNING: yes — input_ids guard prevents removal
- Protected from WEIGHT DECAY: no — receives same uniform decay as all synapses
- Protected from STRENGTH CAP: no — same weight_clamp as all synapses
- History: input cap (0.5) was tested and catastrophically failed — collapsed energy to 0.4–2.0

---

## Other

| Parameter | Validated | Notes |
|-----------|-----------|-------|
| mirror_coupling_strength | 0.001 (spiral), 0.0 (XOR) | Appears non-functional as tuning lever. XOR must be 0.0 |
| stdp_enabled | false | Not yet tested in production |
| stdp_window | 20.0 | STDP timing window (inactive) |

---

## BrainConfig Extensions (Brain API layer)

| Parameter | Default | Notes |
|-----------|---------|-------|
| seed | 42 | RNG seed. 0 = system entropy. Best seeds: 42, 7 |
| reward_scale | 0.0 | RL reward modulates lr: lr *= (1 + reward_scale * reward.clamp(-1,1)) |
| auto_mirror_groups | true | Auto-create mirror groups on first hidden layer |
| reserve_pool_size | 6 | Neurons held out of mirror groups for future task spawning |
| layer_sizes | [2,16,16,1] | Network topology. Last dim = max concurrent task heads |

---

## XOR Config (working state)

```rust
learning_rate: 0.05,
weight_decay: 0.0,
bias_decay: 0.0,
lr_decay: 0.0,
dropout_rate: 0.0,
competitive_k: 0,
lateral_inhibition: 0.0,
mirror_coupling_strength: 0.0,   // MUST be 0.0 — mirror symmetry blocks XOR
weight_clamp: 50.0,              // MUST be 50.0 — 5.0 hits ceiling at 0.36/0.64
prune_long_dormancy: 100_000,    // effectively disable phase-3 pruning
// architecture: [2, 4, 1], seed: 42, 20000 epochs
```

---

## MLP Baseline Config (for comparison runs)

```rust
// Vanilla MLP — all Growformer mechanisms disabled:
learning_rate: 0.15,
weight_decay: 0.0000025,
bias_decay: 0.0,
lr_decay: 0.00008,
competitive_k: 0,           // no KWTA
lateral_inhibition: 0.0,    // no inhibition
dropout_rate: 0.0,          // no dropout
thermal_noise: 0.0,         // no physics
gravity_g: 0.0,
k_repel: 0.0,
mass_decay: 0.0,
mass_growth: 0.0,
homeostasis_lr: 0.0,
prune_interval: 999_999,    // no pruning
geometry_interval: 999_999,
// No mirror groups — remove create_group/pair_mirror_groups calls
// Use separate rng for weights vs data to ensure same data across runs
```