use serde::{Deserialize, Serialize};
use crate::types::*;

/// A full multidimensional neuron
/// Each field is an active dimension — not metadata, not decoration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neuron {
    pub id: NeuronId,

    // Dimension 1: Classical weight (bias term for this neuron)
    pub weight: f32,

    // Dimension 2: Position in 3D embedding space
    pub geometry: Vec3,

    /// Velocity vector for physics-based position integration.
    /// Updated each geometry tick: v += a * dt, pos += v * dt.
    /// Damping bleeds velocity each tick to prevent oscillation.
    pub velocity: Vec3,

    /// Particle mass. Pass 1: constant at 1.0.
    /// Pass 2: tied to competitive success — winners gain mass (become
    /// stable hubs), losers lose mass (become exploratory drifters).
    pub mass: f32,

    // Dimension 3: Temporal state
    pub last_fired: f64,
    pub decay_rate: f32,
    pub activation: f32, // current activation value after last forward pass

    /// Exponential moving average of activation — slow timescale homeostatic signal.
    /// Homeostasis nudges bias based on this, not instantaneous activation.
    /// This separates fast gradient-driven learning from slow drift correction.
    /// Updated each sample: running_act = (1 - tau) * running_act + tau * activation
    /// tau = homeostasis_tau (small, ~0.001) gives ~1000-sample averaging window.
    pub running_activation: f32,

    // Dimension 4: Metabolic cost
    pub energy_cost: f32,
    pub energy_budget: f32,

    // Dimension 5: Variable connectivity
    pub synapses: Vec<Synapse>,

    // Dimension 6: Structural group membership
    pub group_id: Option<GroupId>,
    pub mirror_partner: Option<NeuronId>,

    /// If true, no plasticity or gradient update may modify this neuron (bias, geometry, mass, synapses).
    /// Set by freeze_consolidated_pathway(); backprop and all plasticity systems skip frozen neurons.
    pub frozen: bool,

    /// Which dendritic branch won during the last forward pass (0-based).
    /// Used by backprop to route gradient only through the winning branch's synapses.
    #[serde(default)]
    pub winning_branch: u8,
}

impl Neuron {
    pub fn new(id: NeuronId, geometry: Vec3, config: &EnvironmentConfig) -> Self {
        Self {
            id,
            weight: 0.0,
            geometry,
            velocity: Vec3::zero(),
            mass: 1.0,
            last_fired: 0.0,
            decay_rate: 0.1,
            activation: 0.0,
            running_activation: 0.35, // initialise at target to avoid early homeostatic shock
            energy_cost: 0.5,
            energy_budget: config.energy_budget_per_neuron,
            synapses: Vec::new(),
            group_id: None,
            mirror_partner: None,
            frozen: false,
            winning_branch: 0,
        }
    }

    /// Total metabolic cost at this moment — always non-negative
    pub fn current_energy_cost(&self) -> f32 {
        self.synapses.iter().map(|s| s.metabolic_cost() * self.energy_cost).sum()
    }

    /// Whether this neuron is over its energy budget
    pub fn over_budget(&self) -> bool {
        self.current_energy_cost() > self.energy_budget
    }

    /// Returns true if the synapse to `target` already exists
    pub fn has_synapse_to(&self, target: NeuronId) -> bool {
        self.synapses.iter().any(|s| s.target == target)
    }

    /// Add a new synapse. Allows negative strength for inhibitory connections.
    pub fn add_synapse(&mut self, target: NeuronId, strength: f32, max: usize) -> bool {
        if self.synapses.len() >= max || self.has_synapse_to(target) || target == self.id {
            return false;
        }
        self.synapses.push(Synapse::new(target, strength));
        true
    }

    /// Add a new synapse assigned to a specific dendritic branch of the target.
    pub fn add_synapse_to_branch(&mut self, target: NeuronId, strength: f32, max: usize, branch_id: u8) -> bool {
        if self.synapses.len() >= max || self.has_synapse_to(target) || target == self.id {
            return false;
        }
        let mut syn = Synapse::new(target, strength);
        syn.branch_id = branch_id;
        self.synapses.push(syn);
        true
    }

    /// Apply sigmoid activation
    pub fn activate(&mut self, input: f32) {
        self.activation = sigmoid(input + self.weight);
    }

    /// Produce a snapshot for serialization/logging
    pub fn snapshot(&self) -> NeuronSnapshot {
        let whorled = self.detect_whorl();
        NeuronSnapshot {
            id: self.id,
            weight: self.weight,
            geometry: self.geometry,
            synapse_count: self.synapses.len(),
            energy_used: self.current_energy_cost(),
            group_id: self.group_id,
            last_fired: self.last_fired,
            whorled,
        }
    }

    /// Detect geometric self-coiling (whorls):
    /// A neuron is considered whorled if it has synapses looping back to neurons
    /// that are geometrically very close but functionally distant in the graph
    pub fn detect_whorl(&self) -> bool {
        // Simplified detection: high synapse count with low geometric spread
        // In a full impl you'd check actual partner positions
        self.synapses.len() > 20
            && self.synapses.iter().map(|s| s.effective_strength()).sum::<f32>()
                / self.synapses.len() as f32
                > 0.8
    }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid_derivative(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1.0 - s)
}