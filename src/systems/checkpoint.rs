// =============================================================================
// Phase 2 Checkpoint — Save / Load
//
// Saves the full trained Task A state to disk so Task B debugging runs
// skip the 20-minute Task A training phase entirely.
//
// The gradient gate is implemented by the frozen flag on neurons/synapses.
// Call env.freeze_consolidated_pathway() before save so saved state has
// frozen=true on the consolidated pathway. No restore loop — backprop
// and all plasticity systems skip frozen state.
// =============================================================================

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use crate::types::{NeuronId, GroupId};
use crate::neuron::Neuron;
use crate::types::NeuronGroup;
use rand::rngs::StdRng;
use rand::SeedableRng;

// =============================================================================
// Checkpoint struct — env state + metadata (frozen flags live on neurons/synapses)
// =============================================================================

#[derive(Serialize, Deserialize)]
pub struct Phase2Checkpoint {
    pub neurons: HashMap<NeuronId, Neuron>,
    pub groups: HashMap<GroupId, NeuronGroup>,
    pub layers: Vec<Vec<NeuronId>>,
    pub layer_of: HashMap<NeuronId, usize>,
    pub current_lr: f32,

    pub group_a_ids: Vec<NeuronId>,
    pub group_b_ids: Vec<NeuronId>,
    pub group_a: GroupId,
    pub group_b: GroupId,
    pub output_0: NeuronId,
    pub output_1: NeuronId,

    pub task_a_accuracy: f32,
    pub task_a_correct: usize,
    pub task_a_total: usize,
    pub seed: u64,
    pub data_seed: u64,
    pub trained_epochs: u32,
}

// =============================================================================
// save_phase2_checkpoint — call after freeze_consolidated_pathway() in train-a
// =============================================================================

pub fn save_phase2_checkpoint(
    env: &crate::environment::NeuralEnvironment,
    group_a_ids: &[NeuronId],
    group_b_ids: &[NeuronId],
    group_a: GroupId,
    group_b: GroupId,
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
// load_phase2_checkpoint — reconstructs env; frozen flags come from saved state
// =============================================================================

pub fn load_phase2_checkpoint(
    path: &str,
    config: &crate::types::EnvironmentConfig,
) -> (
    crate::environment::NeuralEnvironment,
    Vec<NeuronId>,
    Vec<NeuronId>,
    GroupId,
    GroupId,
    NeuronId,
    NeuronId,
    u64,
) {
    let json = std::fs::read_to_string(path)
        .expect(&format!("Checkpoint not found: {}\nRun with 'train-a' first.", path));

    let ckpt: Phase2Checkpoint = serde_json::from_str(&json)
        .expect("Checkpoint deserialization failed — file may be corrupted");

    let mut env = crate::environment::NeuralEnvironment::new(config.clone());
    let layer_sizes: Vec<usize> = ckpt.layers.iter().map(|l| l.len()).collect();
    let mut rng = StdRng::seed_from_u64(ckpt.seed);
    env.build_layers(&layer_sizes, &mut rng);

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
        ckpt.output_0,
        ckpt.output_1,
        ckpt.data_seed,
    )
}
