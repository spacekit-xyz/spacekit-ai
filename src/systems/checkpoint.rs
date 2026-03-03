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

    env.sync_input_output_ids_from_layers();
    env.sync_next_neuron_id_from_neurons();

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

// =============================================================================
// Unit tests — JSON round-trip of HashMap<NeuronId, Neuron>
// Ensures serde_json key format (e.g. u32 keys as strings) doesn't corrupt
// checkpoint load (e.g. 50% retention on load).
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Vec3, EnvironmentConfig, Synapse};

    fn make_test_neuron(id: NeuronId, weight: f32, frozen: bool, synapses: Vec<Synapse>) -> Neuron {
        let config = EnvironmentConfig::default();
        let mut n = Neuron::new(id, Vec3::new(id as f32, 0.0, 0.0), &config);
        n.weight = weight;
        n.frozen = frozen;
        n.synapses = synapses;
        n
    }

    #[test]
    fn test_neurons_map_json_roundtrip() {
        let mut neurons: HashMap<NeuronId, Neuron> = HashMap::new();
        neurons.insert(0, make_test_neuron(0, -0.5, true, vec![
            Synapse { target: 1, strength: 0.3, frozen: true, ..Synapse::new(1, 0.3) },
        ]));
        neurons.insert(1, make_test_neuron(1, 0.2, false, vec![
            Synapse { target: 2, strength: -0.1, frozen: false, ..Synapse::new(2, -0.1) },
        ]));
        neurons.insert(2, make_test_neuron(2, 0.0, true, vec![]));

        let json = serde_json::to_string(&neurons).expect("serialize neurons map");
        let restored: HashMap<NeuronId, Neuron> = serde_json::from_str(&json).expect("deserialize neurons map");

        assert_eq!(restored.len(), neurons.len(), "key count must match");
        for (id, orig) in &neurons {
            let r = restored.get(id).expect("every original key must be present after round-trip");
            assert_eq!(r.id, orig.id, "neuron id");
            assert_eq!(r.weight, orig.weight, "neuron weight");
            assert_eq!(r.frozen, orig.frozen, "neuron frozen");
            assert_eq!(r.synapses.len(), orig.synapses.len(), "synapse count");
            for (i, (rs, os)) in r.synapses.iter().zip(orig.synapses.iter()).enumerate() {
                assert_eq!(rs.target, os.target, "synapse[{}] target", i);
                assert_eq!(rs.strength, os.strength, "synapse[{}] strength", i);
                assert_eq!(rs.frozen, os.frozen, "synapse[{}] frozen", i);
            }
        }
    }

    #[test]
    fn test_phase2_checkpoint_neurons_roundtrip() {
        let config = EnvironmentConfig::default();
        let mut neurons: HashMap<NeuronId, Neuron> = HashMap::new();
        for id in 0..5u32 {
            let mut n = Neuron::new(id, Vec3::new(id as f32, 0.0, 0.0), &config);
            n.weight = id as f32 * 0.1;
            n.frozen = id % 2 == 0;
            if id < 4 {
                n.add_synapse(id + 1, 0.2, 64);
            }
            neurons.insert(id, n);
        }

        let checkpoint = Phase2Checkpoint {
            neurons: neurons.clone(),
            groups: HashMap::new(),
            layers: vec![vec![0, 1], vec![2, 3], vec![4]],
            layer_of: [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2)].into_iter().collect(),
            current_lr: 0.01,
            group_a_ids: vec![0, 2],
            group_b_ids: vec![1, 3],
            group_a: 0,
            group_b: 1,
            output_0: 4,
            output_1: 4,
            task_a_accuracy: 0.9,
            task_a_correct: 720,
            task_a_total: 800,
            seed: 42,
            data_seed: 99,
            trained_epochs: 4000,
        };

        let json = serde_json::to_string_pretty(&checkpoint).expect("serialize checkpoint");
        let loaded: Phase2Checkpoint = serde_json::from_str(&json).expect("deserialize checkpoint");

        assert_eq!(loaded.neurons.len(), checkpoint.neurons.len());
        for (id, orig) in &checkpoint.neurons {
            let r = loaded.neurons.get(id).expect("key present after round-trip");
            assert_eq!(r.id, orig.id);
            assert_eq!(r.weight, orig.weight);
            assert_eq!(r.frozen, orig.frozen);
            assert_eq!(r.synapses.len(), orig.synapses.len());
        }
    }
}
