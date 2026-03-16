use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::neuron::Neuron;
pub type NeuronId = u32;
pub type GroupId = u32;

/// One supervised sample: input (variable length, e.g. 2 for spiral/circles or 64 for MNIST) and binary target.
pub type Sample = (Vec<f32>, [f32; 1]);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }

    pub fn distance(&self, other: &Vec3) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    pub fn magnitude(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn magnitude_sq(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }
}

impl std::ops::Add for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, rhs: f32) -> Vec3 {
        Vec3::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl std::ops::AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Vec3) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

/// A single synapse — stateful, not just a weight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Synapse {
    pub target: NeuronId,
    pub strength: f32,
    pub timing_offset: f32, // STDP: relative arrival offset
    pub facilitation: f32,  // strengthens with repeated use
    pub depression: f32,    // weakens under sustained load
    pub age: u32,           // ticks since formation

    /// Tick of last meaningful signal transmission.
    /// Used by three-phase pruning to distinguish dormant synapses
    /// from active ones regardless of absolute age.
    /// A synapse at age 5000 that fired last tick is structurally
    /// stable; one at age 5000 that last fired 3000 ticks ago is
    /// a candidate for long-term structural pruning.
    pub last_active: u32,

    /// If true, backprop and plasticity never update this synapse (e.g. input→consolidated pathway).
    pub frozen: bool,

    /// Engram consolidation: accumulates when pre and post neurons co-fire (both in KWTA winners).
    /// High consolidation = synapse is part of a memory trace; gets reduced LR and pruning immunity.
    /// Biological analog: engram-engram synapses are stronger and denser (Rajasethupathy et al.).
    #[serde(default)]
    pub consolidation: f32,

    /// Dendritic branch index on the target neuron (0-based).
    /// When dendritic_branches > 1, inputs are summed per-branch; the winning branch
    /// (highest local sum) drives the soma. Gradient flows only through the winning branch.
    /// Biological analog: compartmentalized dendritic computation.
    #[serde(default)]
    pub branch_id: u8,
}

impl Synapse {
    pub fn new(target: NeuronId, strength: f32) -> Self {
        Self {
            target,
            strength,
            timing_offset: 0.0,
            facilitation: 1.0,
            depression: 1.0,
            age: 0,
            last_active: 0,
            frozen: false,
            consolidation: 0.0,
            branch_id: 0,
        }
    }

    /// Effective signal transmission — preserves sign for inhibitory synapses
    pub fn effective_strength(&self) -> f32 {
        let magnitude = self.strength.abs() * self.facilitation * self.depression;
        magnitude.clamp(0.0, 2.0) * self.strength.signum()
    }

    /// Unsigned cost for metabolic budget calculations
    pub fn metabolic_cost(&self) -> f32 {
        self.strength.abs() * self.facilitation * self.depression
    }
}

/// A group of neurons that can develop mirror-symmetry relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronGroup {
    pub id: GroupId,
    pub members: Vec<NeuronId>,
    pub mirror_group: Option<GroupId>,
    pub centroid: Vec3,
}

impl NeuronGroup {
    pub fn new(id: GroupId) -> Self {
        Self {
            id,
            members: Vec::new(),
            mirror_group: None,
            centroid: Vec3::zero(),
        }
    }
}

/// Configuration for the full environment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentConfig {
    pub max_synapses_per_neuron: usize,
    pub energy_budget_per_neuron: f32,
    pub pruning_threshold: f32,
    pub mirror_coupling_strength: f32,
    pub stdp_window: f32,
    pub geometry_influence: f32,
    pub growth_radius: f32,
    pub learning_rate: f32,
    /// Decay applied to neuron bias each step. Stronger than weight_decay keeps
    /// neurons in the selective mid-range of sigmoid rather than always-on/always-off.
    /// Rule: set to 3-10x weight_decay. Prevents the 0.147-convergence collapse.
    pub bias_decay: f32,
    pub weight_decay: f32,
    pub a_plus: f32,
    pub a_minus: f32,
    pub tau_plus: f32,
    pub tau_minus: f32,
    /// How many ticks between geometry updates.
    /// Geometry runs every tick by default which causes spatial collapse
    /// in long training runs. Set to 50-200 for stable spread.
    pub geometry_interval: u32,
    /// Std deviation of Gaussian noise added to neuron positions each geometry update.
    /// This is the primary symmetry-breaking mechanism — even tiny values (0.001–0.01)
    /// break the perfect positional symmetry of same-layer neurons, allowing gradient
    /// descent to differentiate them over time.
    pub geometry_noise: f32,

    /// Lateral inhibition strength (Option A competition).
    /// Each neuron's activation is reduced by this fraction of the layer mean.
    /// 0.0 = disabled. 0.1–0.3 is a good range.
    /// Runs before winner-takes-more so the k-gate sees inhibited values.
    pub lateral_inhibition: f32,

    /// Learning rate decay factor applied each epoch via set_epoch().
    /// effective_lr = learning_rate * lr_decay^epoch
    /// 0.0 = no decay (fixed lr). 0.001 = moderate annealing.
    /// Allows aggressive early exploration with fine-grained late convergence.
    pub lr_decay: f32,

    pub tick_count: u64,

    // -------------------------------------------------------------------------
    // Physics-based geometry (N-body particle system)
    // Pass 1: stable dynamics with constant mass
    // Pass 2: mass tied to competitive success (feature hubs emerge)
    // -------------------------------------------------------------------------
    pub physics_dt: f32,         // integration timestep (0.01 safe for k_repel < 1.0)
    pub gravity_g: f32,          // laminar gravity toward layer centroid
    pub k_repel: f32,            // same-layer repulsion coefficient
    pub repulsion_radius: f32,   // repulsion cutoff distance
    pub damping: f32,            // velocity damping per tick: v *= (1 - damping)
    pub thermal_noise: f32,      // velocity noise std — replaces geometry_noise
    pub hebbian_attraction: f32, // correlated partner attraction strength

    // -------------------------------------------------------------------------
    // Reaction-diffusion lateral inhibition (Turing pattern mechanism)
    // -------------------------------------------------------------------------
    /// Spatial decay length for inhibitory signal.
    /// Inhibition from neuron j falls off as exp(-dist/sigma_inhib).
    /// Controls receptive field scale — smaller = many small fields,
    /// larger = few broad fields. Typical: 0.5–2.0 (in geometry units).
    pub sigma_inhib: f32,

    // -------------------------------------------------------------------------
    // Debye-screened repulsion (replaces hard cutoff)
    // -------------------------------------------------------------------------
    /// Debye screening length. Repulsion decays as exp(-dist/debye_length).
    /// Adaptive density: dense regions get stronger screening (shorter effective
    /// range), sparse regions get weaker screening (longer effective range).
    /// Replaces repulsion_radius hard cutoff with physically smooth decay.
    pub debye_length: f32,

    /// Strength bonus applied to synapses that survive phase-2 pruning with
    /// high facilitation. Completes the activity-dependent plasticity loop:
    /// pruning removes unused connections, this strengthens frequently-used ones.
    /// Biological analog: LTP consolidation — surviving synapses grow larger
    /// dendritic spines and more AMPA receptors.
    /// Typical: 0.001–0.005. Applied every prune_interval ticks.
    pub facilitation_bonus: f32,

    // -------------------------------------------------------------------------
    // Local KWTA — neighborhood competition
    // -------------------------------------------------------------------------
    /// Radius within which neurons compete for territory.
    /// A neuron survives iff no neighbor within this radius has higher activation.
    /// This produces local maxima as winners — non-overlapping spatial territories.
    /// Set to 0.0 to use global competitive_k instead.
    /// Typical: 1.0–2.0 (relative to geometric spread of ~3.0).
    pub kwta_radius: f32,
    /// Activation multiplier for local KWTA losers. 0.0 = hard zero (original),
    /// 0.2 = soft suppression — losers keep 20% signal and gradient.
    /// Soft suppression allows losers to still learn toward winning on other samples.
    pub kwta_suppression: f32,

    // -------------------------------------------------------------------------
    // Mass–competition coupling (Pass 2 geometry)
    // -------------------------------------------------------------------------
    /// Per-sample mass growth rate for neurons above the win threshold.
    /// Winners accumulate mass → become gravitational hubs → attract correlated partners.
    /// Typical: 0.0001–0.001. Applied every forward pass.
    pub mass_growth: f32,

    /// Per-sample mass decay rate for all neurons (losers decay faster than they grow).
    /// Applied every forward pass: mass *= (1 - mass_decay).
    /// Typical: 0.00005–0.0002. Slightly less than mass_growth so winners net-gain.
    pub mass_decay: f32,

    /// Activation threshold above which a neuron is considered a winner this sample.
    /// After reaction-diffusion inhibition, winners have activation above this value.
    /// Typical: 0.4 (slightly below sigmoid midpoint — rewards any meaningful firing).
    pub mass_win_threshold: f32,

    /// Minimum neuron mass — losers don't vanish entirely, stay as exploratory drifters.
    pub mass_min: f32,

    /// Maximum neuron mass — winners become hubs but don't dominate gravity unboundedly.
    pub mass_max: f32,

    /// Mass-based consolidation: scale effective learning rate by 1/(1 + k*mass).
    /// High-mass neurons (spiral hubs) get smaller LR during Task B; low-mass stay plastic.
    /// 0.0 = disabled. Typical: 2.0–5.0 for continual learning.
    pub mass_consolidation_k: f32,

    // -------------------------------------------------------------------------
    // Homeostatic plasticity
    // -------------------------------------------------------------------------
    /// Target mean activation for hidden neurons. Homeostatic adjustment
    /// nudges each neuron's bias toward maintaining this firing rate.
    /// Biological analog: synaptic scaling — neurons regulate excitability
    /// to maintain a stable mean firing rate regardless of input statistics.
    /// Typical: 0.3–0.4 (slightly below sigmoid midpoint for sparse coding).
    pub homeostasis_target: f32,

    /// Learning rate for homeostatic bias adjustment. Much weaker than gradient
    /// lr — should not overwhelm task learning, only prevent runaway drift.
    /// Typical: 0.0001–0.001. Applied every forward pass.
    pub homeostasis_lr: f32,

    /// Exponential moving average decay for homeostatic tracking.
    /// tau=0.001 means ~1000-sample window. Must be << 1.
    /// Separates fast gradient learning from slow homeostatic correction.
    pub homeostasis_tau: f32,


    /// 0 = disabled. 4–8 is a good starting point for a 16-neuron hidden layer.
    /// Forces specialisation by preventing all neurons from firing identically.
    pub competitive_k: usize,

    // -------------------------------------------------------------------------
    // Three-phase pruning — maps to LTP consolidation windows
    //
    // Phase 1 (early): young synapses that never fired get a fast cull.
    //   Biological analog: early LTP maintenance window (~minutes).
    //   Many trial connections form; most fail to carry signal and are retracted.
    //
    // Phase 2 (mid): survived the cull but never consolidated — low strength
    //   AND low facilitation means no Hebbian reinforcement occurred.
    //   Biological analog: late LTP consolidation (~hours, BDNF-dependent).
    //
    // Phase 3 (long): established synapses that have gone dormant.
    //   Uses dormancy (ticks since last_active) not just absolute age.
    //   Biological analog: structural plasticity (~days/weeks).
    // -------------------------------------------------------------------------

    /// Phase 1 age cutoff (ticks). Synapses younger than this are in the early window.
    pub prune_early_age: u32,
    /// Phase 1 strength floor. Synapses below this AND younger than early_age are culled.
    pub prune_early_threshold: f32,

    /// Phase 2 age range start (= early_age). End = long_age.
    /// Mid-phase synapses are pruned if strength < mid_threshold AND facilitation < mid_facilitation_floor.
    pub prune_mid_threshold: f32,
    pub prune_mid_facilitation_floor: f32,

    /// How often (in ticks) to run three-phase pruning.
    /// Running every tick is too aggressive — a synapse at age 50 hasn't
    /// had enough samples to develop signal. Schedule: every 200–500 ticks.
    /// XOR with no growth: set very high (50000) to effectively disable it.
    pub prune_interval: u32,
    /// Phase 3 dormancy window. Prune if (current_tick - last_active) > this AND strength < long_threshold.
    /// Stop all pruning after this tick count. The network has found its structure
    /// by epoch ~3000-4000 (tick ~1.2M-1.6M). Continued pruning after that destroys
    /// learned connectivity rather than cleaning dead weight.
    /// Set to 0 to disable (prune always). At 400 samples × 500 prune_interval = 200k ticks/epoch,
    /// stopping at epoch 3000 = tick 1,200,000.
    pub prune_stop_tick: u64,
    /// Maximum absolute value for weight/bias clamps during backprop.
    /// Default 5.0 (validated for spiral). XOR needs 15.0+ to push past 0.36/0.64 plateau.
    pub weight_clamp: f32,
    pub prune_long_dormancy: u32,
    pub prune_long_threshold: f32,
    pub prune_long_age: u32,
    pub stdp_enabled: bool,
    pub dropout_rate: f32,

    /// Use binary cross-entropy gradient at the output layer instead of MSE.
    /// With sigmoid activation, MSE gradient = (o-t)*o*(1-o) which vanishes
    /// as outputs saturate. BCE gradient = (o-t) stays strong at saturation.
    /// Enable for binary-target tasks (generation). Default false (MSE, validated
    /// for spiral/MNIST classification).
    pub output_bce: bool,

    // -------------------------------------------------------------------------
    // Engram-level consolidation (memory trace protection)
    //
    // Synapses between co-activated neurons (both in KWTA winners) accumulate
    // consolidation. High-consolidation synapses: reduced LR (resist overwriting),
    // pruning immunity. Biological analog: engram-engram synapses are stronger
    // and denser; memory strength correlates with synaptic connectivity.
    // -------------------------------------------------------------------------
    /// Enable engram consolidation. When true, co-activated synapses accumulate
    /// consolidation and get protected. Use for generation tasks with many patterns.
    #[serde(default)]
    pub engram_enabled: bool,
    /// Activation threshold for "fired" (participated in this forward pass).
    /// Neurons with activation >= this are considered engram participants.
    #[serde(default)]
    pub engram_activation_threshold: f32,
    /// Per-tick increment when pre and post both fired. Cap at engram_cap.
    #[serde(default)]
    pub engram_increment: f32,
    /// Maximum consolidation value (0.0 to 1.0).
    #[serde(default)]
    pub engram_cap: f32,
    /// At consolidation=1.0, effective LR is scaled by (1 - engram_lr_scale).
    /// 0.8 means 20% of normal LR for fully consolidated synapses.
    #[serde(default)]
    pub engram_lr_scale: f32,
    /// Synapses with consolidation >= this are never pruned.
    #[serde(default)]
    pub engram_prune_threshold: f32,

    // -------------------------------------------------------------------------
    // Dendritic branches — compartmentalized input integration
    //
    // Each hidden neuron has N dendritic branches. Incoming synapses are assigned
    // to branches; each branch sums its inputs independently. The soma fires
    // based on the winning branch (highest local sum). Backprop flows only
    // through the winning branch's synapses.
    //
    // This allows a single neuron to participate in N independent engrams without
    // interference: branch A detects pattern X, branch B detects pattern Y.
    // Multiplies effective per-neuron capacity by the branch count.
    //
    // Biological analog: dendritic compartments with local nonlinear integration,
    // branch-specific plasticity, and dendritic spikes.
    // -------------------------------------------------------------------------
    /// Number of dendritic branches per hidden neuron.
    /// 1 = disabled (all synapses on one branch, identical to current behavior).
    /// 4 = each neuron has 4 independent pattern detectors.
    #[serde(default = "default_dendritic_branches")]
    pub dendritic_branches: usize,
}

fn default_dendritic_branches() -> usize { 1 }

impl Default for EnvironmentConfig {
    fn default() -> Self {
        Self {
            max_synapses_per_neuron: 64,
            energy_budget_per_neuron: 10.0,
            pruning_threshold: 0.05,
            mirror_coupling_strength: 0.001,
            stdp_window: 20.0,
            geometry_influence: 0.001,
            growth_radius: 2.0,
            learning_rate: 0.01,
            a_plus: 0.002,
            a_minus: 0.002,
            tau_plus: 20.0,
            tau_minus: 20.0,
            geometry_interval: 100,
            stdp_enabled: false,
            weight_decay: 0.0000025,  // scaled: 0.001 epoch-decay / 400 samples
            bias_decay: 0.000025,    // scaled: 0.01/400 — 10× weight_decay at epoch level
            dropout_rate: 0.1,
            geometry_noise: 0.005,
            competitive_k: 0,
            tick_count: 0,

            // Three-phase pruning defaults
            prune_early_age: 100,
            prune_early_threshold: 0.003,   // very low — only cull truly silent trial connections

            prune_mid_threshold: 0.01,
            prune_mid_facilitation_floor: 1.02, // facilitation > 1.02 = synapse was used at least once

            prune_long_age: 2000,
            prune_stop_tick: 0,
            weight_clamp: 5.0,   // spiral validated at 5.0 — XOR config overrides to 15.0         // disabled by default — set per task
            prune_long_dormancy: 1000,
            prune_long_threshold: 0.005,
            // prune_long_age: 2000,
            
            homeostasis_tau: 0.001,
            prune_interval: 500,

            // Physics-based geometry (Pass 1: constant mass)
            physics_dt: 0.05,
            gravity_g: 0.02,
            k_repel: 0.5,
            repulsion_radius: 3.0,
            damping: 0.15,
            thermal_noise: 0.002,
            hebbian_attraction: 0.001,

            // Reaction-diffusion inhibition
            sigma_inhib: 1.5,     // inhibition decay length — tune to geometry spread / sqrt(n)

            // Debye-screened repulsion
            debye_length: 2.0,    // screening length — replaces hard repulsion_radius cutoff
            facilitation_bonus: 0.002, // LTP consolidation bonus for highly-used synapses
            kwta_radius: 0.0,
            kwta_suppression: 0.2,  // soft suppression default — 20% signal preserved for losers              // disabled by default — set in spiral config

            // Mass–competition coupling
            mass_growth: 0.0005,          // winners gain this much mass per sample
            mass_decay: 0.00015,          // all neurons lose this fraction per sample
            mass_win_threshold: 0.4,      // activation above this = winner this sample
            mass_min: 0.3,                // losers stay as light exploratory drifters
            mass_max: 3.0,                // winners cap — prevents single hub monopoly
            mass_consolidation_k: 0.0,    // 0 = disabled; set 2–5 for continual learning

            // Homeostatic plasticity
            homeostasis_target: 0.35,     // target mean activation per neuron
            homeostasis_lr: 0.0003,       // gentle bias nudge — weaker than gradient lr
            lateral_inhibition: 0.0, // disabled by default — set 0.1–0.2 for hidden layers
            lr_decay: 0.0,
            output_bce: false,

            // Engram consolidation — disabled by default; GroupGenEnv enables
            engram_enabled: false,
            engram_activation_threshold: 0.1,
            engram_increment: 0.05,
            engram_cap: 1.0,
            engram_lr_scale: 0.8,
            engram_prune_threshold: 0.4,

            // Dendritic branches — 1 = disabled (current behavior)
            dendritic_branches: 1,
        }
    }
}

/// Full snapshot of a neuron at a point in time — for observation/export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronSnapshot {
    pub id: NeuronId,
    pub weight: f32,
    pub geometry: Vec3,
    pub synapse_count: usize,
    pub energy_used: f32,
    pub group_id: Option<GroupId>,
    pub last_fired: f64,
    pub whorled: bool,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase2Checkpoint {
    pub neurons: HashMap<NeuronId, Neuron>,
    pub groups: HashMap<GroupId, NeuronGroup>,
    pub layers: Vec<Vec<NeuronId>>,
    pub layer_of: HashMap<NeuronId, usize>,
    pub current_lr: f32,
    pub config: EnvironmentConfig,
    pub group_a_ids: Vec<NeuronId>,
    pub group_b_ids: Vec<NeuronId>,
    pub consolidated_snapshot: Vec<(NeuronId, f32, Vec<(NeuronId, f32)>)>,
    pub output_0_incoming: Vec<(NeuronId, f32)>,
    pub task_a_accuracy: f32,
    pub seed: u64,
}