# Ocean Model

**What OCEAN is:**

The Big Five personality model — Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism. Each is a continuous dimension, not a category. A person scores somewhere on each axis, and the combination produces stable behavioral tendencies that persist across contexts.

**Why it maps onto the Growformer naturally:**

The architecture already has analog dimensions that behave like personality traits. They're just not named or exposed as a coherent profile yet.

| OCEAN dimension | Growformer analog | Current parameter |
|----------------|-------------------|-------------------|
| Openness | Willingness to spawn new groups vs recall | residual spawn threshold |
| Conscientiousness | How thoroughly a group consolidates before freezing | prune_mid_facilitation_floor |
| Extraversion | How aggressively groups compete for activation | competitive_k |
| Agreeableness | Cross-group coherence tolerance | GlobalObserver coherence threshold |
| Neuroticism | Sensitivity to residual error — how easily destabilized | thermal_noise + lr sensitivity |

These parameters already exist. OCEAN would be a named abstraction layer over them — a BrainPersonality struct that sets a coherent bundle of underlying parameters rather than requiring the caller to tune each one independently.

**The deeper argument for it:**

Right now the Brain API exposes raw EnvironmentConfig parameters. A caller building an agent has to understand what `thermal_noise: 0.02` means to tune it. An OCEAN profile is a semantically meaningful interface — a caller can say "this agent should be highly curious and somewhat impulsive" and the profile translates that into appropriate parameter bundles.

This is exactly what personality does biologically — it's a stable high-level description of low-level neural tendencies. Dopaminergic sensitivity, serotonin baseline, cortisol reactivity — all measurable, all underlying specific behavioral patterns. OCEAN is just the named abstraction over those.

**What each dimension would control concretely:**

**Openness (exploration vs exploitation):**
```rust
high_openness: {
    spawn_threshold: 0.30,      // spawns new groups readily
    growth_radius: 3.0,         // synapses reach further
    thermal_noise: 0.04,        // more symmetry breaking
    prune_long_dormancy: 500,   // prunes faster, stays flexible
}
low_openness: {
    spawn_threshold: 0.60,      // strongly prefers recall
    growth_radius: 1.5,
    thermal_noise: 0.01,
    prune_long_dormancy: 2000,
}
```

**Conscientiousness (consolidation thoroughness):**
```rust
high_conscientiousness: {
    prune_mid_facilitation_floor: 1.5,  // harder to consolidate
    finetune_epochs: 50,                 // brief fine-tune windows
    rollback_threshold: 0.02,           // strict rollback
    prune_interval: 250,                // prunes more frequently
}
low_conscientiousness: {
    prune_mid_facilitation_floor: 1.02,
    finetune_epochs: 500,
    rollback_threshold: 0.10,
}
```

**Extraversion (competitive dominance):**
```rust
high_extraversion: {
    competitive_k: 6,           // more winners per pass
    mass_win_threshold: 0.10,   // easier to be a winner
    lateral_inhibition: 0.06,   // less suppression of neighbors
    mirror_coupling: 0.003,     // stronger group coherence
}
low_extraversion: {
    competitive_k: 2,
    mass_win_threshold: 0.20,
    lateral_inhibition: 0.18,
}
```

**Agreeableness (cross-group cooperation tolerance):**
```rust
high_agreeableness: {
    group_repulsion_penalty: 1.5,   // groups stay closer together
    coherence_threshold: 0.60,      // tolerates more disagreement
    composition_bias: 0.3,          // prefers composition over spawning
}
low_agreeableness: {
    group_repulsion_penalty: 5.0,   // groups push far apart
    coherence_threshold: 0.85,
    composition_bias: -0.2,         // prefers independent specialists
}
```

**Neuroticism (stability vs reactivity):**
```rust
high_neuroticism: {
    lr_scale_on_error: 2.0,     // large errors cause large lr spikes
    mass_decay: 0.00015,        // mass destabilizes faster
    dropout_rate: 0.2,          // more stochastic
    thermal_noise: 0.05,        // more volatile geometry
}
low_neuroticism: {
    lr_scale_on_error: 1.1,
    mass_decay: 0.00006,
    dropout_rate: 0.05,
    thermal_noise: 0.01,
}
```

**The implementation:**

```rust
pub struct OceanProfile {
    pub openness: f32,          // 0.0 - 1.0
    pub conscientiousness: f32,
    pub extraversion: f32,
    pub agreeableness: f32,
    pub neuroticism: f32,
}

impl OceanProfile {
    pub fn apply(&self, config: &mut EnvironmentConfig) {
        // Interpolate between low and high parameter bundles
        // for each dimension
    }
    
    // Named presets
    pub fn scientist() -> Self {
        Self { openness: 0.85, conscientiousness: 0.80,
               extraversion: 0.35, agreeableness: 0.55, neuroticism: 0.25 }
    }
    
    pub fn explorer() -> Self {
        Self { openness: 0.95, conscientiousness: 0.30,
               extraversion: 0.70, agreeableness: 0.65, neuroticism: 0.50 }
    }
    
    pub fn guardian() -> Self {
        Self { openness: 0.20, conscientiousness: 0.90,
               extraversion: 0.30, agreeableness: 0.75, neuroticism: 0.15 }
    }
}
```

**The philosophical dimension:**

This is where it gets genuinely interesting. OCEAN in humans is not chosen — it emerges from genetics, development, and experience. A high-openness person didn't decide to be curious; their neural architecture makes exploration intrinsically rewarding.

If the Growformer's OCEAN profile is fixed at initialization and stable across tasks — if a high-openness brain consistently spawns new groups while a low-openness brain consistently prefers recall, regardless of the specific task — then the profile is genuinely personality-like. It's a stable dispositional tendency, not a per-task setting.

The test would be: initialize two brains with identical architectures but different OCEAN profiles. Train both on the same task sequence. Do they develop different group structures, different composition strategies, different forgetting rates? If yes, the profile produces meaningfully different cognitive styles, not just different parameter settings.

**Note: One concern to name honestly:**

OCEAN in humans predicts behavior but doesn't cause it directly — it's a description of tendencies, not a mechanism. Applying it to the Growformer risks being cosmetic — renaming parameters without adding genuine structure. The architecture needs to be rich enough that the same underlying parameters genuinely produce different emergent behaviors, not just different loss curves.

The current architecture is probably rich enough. The mass-competition-geometry loop, KWTA dynamics, and pruning cascade all interact nonlinearly. Small parameter changes can produce qualitatively different emergent structures. If that's true, OCEAN profiles are a real interface to real behavioral diversity, not just a naming convention.

Worth building after Phase 3b. The GlobalObserver is the right place to read the profile — it's the highest integrating layer and the one that decides between recall, composition, fine-tuning, and spawning. OCEAN shapes those decisions at the policy level, not the mechanism level.