// =============================================================================
// Phase 2 Checkpoint — Save / Load
//
// Saves the full trained Task A state to disk so Task B debugging runs
// skip the 20-minute Task A training phase entirely.
//
// USAGE:
//   // In main(), switch between modes via command line:
//   cargo run -- train-a     // trains Task A, saves checkpoint, exits
//   cargo run -- train-b     // loads checkpoint, runs Task B only (~20 min)
//   cargo run                // full run, no checkpoint (default)
//
// INTEGRATION:
//   1. Add phase2_checkpoint.rs contents to main.rs (or as a module)
//   2. Update main() with the mode switch (see bottom of this file)
//   3. Split demo_continual_learning() into:
//        demo_phase2_train_a() — trains A, saves checkpoint
//        demo_phase2_train_b() — loads checkpoint, trains B
// =============================================================================

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use crate::types::{NeuronId, GroupId, Synapse};
use crate::neuron::Neuron;
use crate::types::NeuronGroup;
use rand::rngs::StdRng;
use rand::SeedableRng;

// =============================================================================
// Checkpoint struct — everything needed to reconstruct post-Task-A state
// =============================================================================

#[derive(Serialize, Deserialize)]
pub struct Phase2Checkpoint {
    // Full neuron state — weights, geometry, mass, synapses, group_id
    pub neurons: HashMap<NeuronId, Neuron>,

    // Group assignments — which neurons belong to which group
    pub groups: HashMap<GroupId, NeuronGroup>,

    // Layer structure — neuron IDs per layer
    pub layers: Vec<Vec<NeuronId>>,

    // Layer lookup — NeuronId → layer index
    pub layer_of: HashMap<NeuronId, usize>,

    // Learning rate at time of consolidation
    pub current_lr: f32,

    // Group membership for demo reconstruction
    pub group_a_ids: Vec<NeuronId>,
    pub group_b_ids: Vec<NeuronId>,
    pub group_a: GroupId,
    pub group_b: GroupId,

    // Frozen snapshots — full Neuron clones so restore overwrites every field (incl. facilitation, depression, etc.)
    pub consolidated_neurons: Vec<Neuron>,
    pub output_0_neuron: Neuron,
    /// Every synapse that targets output_0 (full Synapse: strength, facilitation, depression, etc.).
    /// Needed because Group B and other neurons' synapses to output_0 would otherwise drift during Task B.
    pub output_0_incoming_synapses: Vec<(NeuronId, Synapse)>,

    // Output head neuron IDs
    pub output_0: NeuronId,
    pub output_1: NeuronId,

    // Metadata
    pub task_a_accuracy: f32,         // e.g. 0.911
    pub task_a_correct: usize,        // e.g. 729
    pub task_a_total: usize,          // e.g. 800
    pub seed: u64,                    // weight_rng seed used
    pub data_seed: u64,               // data_rng seed used
    pub trained_epochs: u32,          // Task A epochs completed
}

// =============================================================================
// save_phase2_checkpoint
//
// Call this immediately after the consolidation block in demo_phase2_train_a().
// Serializes the full environment and frozen snapshots to JSON on disk.
// =============================================================================

pub fn save_phase2_checkpoint(
    env: &crate::environment::NeuralEnvironment,
    group_a_ids: &[NeuronId],
    group_b_ids: &[NeuronId],
    group_a: GroupId,
    group_b: GroupId,
    consolidated_neurons: &[Neuron],
    output_0_neuron: &Neuron,
    output_0_incoming_synapses: &[(NeuronId, Synapse)],
    output_0: NeuronId,
    output_1: NeuronId,
    task_a_correct: usize,
    task_a_total: usize,
    seed: u64,
    data_seed: u64,
    trained_epochs: u32,
    path: &str,
) {
    let checkpoint = Phase2Checkpoint {
        neurons:   env.neurons.clone(),
        groups:    env.groups.clone(),
        layers:    env.layers.clone(),
        layer_of:  env.layer_of.clone(),
        current_lr: env.current_lr,
        group_a_ids: group_a_ids.to_vec(),
        group_b_ids: group_b_ids.to_vec(),
        group_a,
        group_b,
        consolidated_neurons: consolidated_neurons.to_vec(),
        output_0_neuron: output_0_neuron.clone(),
        output_0_incoming_synapses: output_0_incoming_synapses.to_vec(),
        output_0,
        output_1,
        task_a_accuracy: task_a_correct as f32 / task_a_total as f32,
        task_a_correct,
        task_a_total,
        seed,
        data_seed,
        trained_epochs,
    };

    let json = serde_json::to_string_pretty(&checkpoint)
        .expect("Checkpoint serialization failed");

    std::fs::write(path, &json)
        .expect(&format!("Failed to write checkpoint to {}", path));

    let size_kb = json.len() / 1024;
    println!("  Checkpoint saved: {} ({} KB)", path, size_kb);
    println!("  Task A: {}/{} ({:.1}%)",
        task_a_correct, task_a_total,
        100.0 * task_a_correct as f32 / task_a_total as f32);
}

// =============================================================================
// load_phase2_checkpoint
//
// Call at the start of demo_phase2_train_b() instead of building and
// training Task A. Reconstructs a NeuralEnvironment from the checkpoint.
//
// Returns the environment and all variables needed by the Task B loop.
// =============================================================================

pub fn load_phase2_checkpoint(
    path: &str,
    config: &crate::types::EnvironmentConfig,
) -> (
    crate::environment::NeuralEnvironment,
    Vec<NeuronId>,          // group_a_ids
    Vec<NeuronId>,          // group_b_ids
    GroupId,                // group_a
    GroupId,                // group_b
    Vec<Neuron>,            // consolidated_neurons
    Neuron,                 // output_0_neuron
    Vec<(NeuronId, Synapse)>, // output_0_incoming_synapses
    NeuronId,               // output_0
    NeuronId,               // output_1
    u64,                    // data_seed (for recreating Task B data)
) {
    let json = std::fs::read_to_string(path)
        .expect(&format!("Checkpoint not found: {}\nRun with 'train-a' first.", path));

    let ckpt: Phase2Checkpoint = serde_json::from_str(&json)
        .expect("Checkpoint deserialization failed — file may be corrupted");

    // Reconstruct NeuralEnvironment from checkpoint.
    // build_layers() sets up input_ids, output_ids, layer_of internals.
    // We then overwrite neurons and groups with the trained checkpoint state.
    let mut env = crate::environment::NeuralEnvironment::new(config.clone());

    // Derive layer sizes from checkpoint layers
    let layer_sizes: Vec<usize> = ckpt.layers.iter().map(|l| l.len()).collect();
    let mut rng = StdRng::seed_from_u64(ckpt.seed);
    env.build_layers(&layer_sizes, &mut rng);

    // Overwrite with trained checkpoint state
    env.neurons    = ckpt.neurons;
    env.groups     = ckpt.groups;
    env.layers     = ckpt.layers;
    env.layer_of   = ckpt.layer_of;
    env.current_lr = ckpt.current_lr;

    println!("Checkpoint loaded: {}", path);
    println!("  Task A was: {}/{} ({:.1}%) after {} epochs",
        ckpt.task_a_correct, ckpt.task_a_total,
        100.0 * ckpt.task_a_accuracy,
        ckpt.trained_epochs);
    println!("  Skipping Task A training — proceeding directly to Task B.\n");

    (
        env,
        ckpt.group_a_ids,
        ckpt.group_b_ids,
        ckpt.group_a,
        ckpt.group_b,
        ckpt.consolidated_neurons,
        ckpt.output_0_neuron,
        ckpt.output_0_incoming_synapses,
        ckpt.output_0,
        ckpt.output_1,
        ckpt.data_seed,
    )
}