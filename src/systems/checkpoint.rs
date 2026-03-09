// =============================================================================
// Phase 2 Checkpoint — Save / Load
//
// Saves the full trained Task A state to disk so Task B debugging runs
// skip the 20-minute Task A training phase entirely.
//
// MnistCheckpoint — Save Main (five frozen groups) + baseline accs for retention eval.
// =============================================================================

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use crate::types::{NeuronId, GroupId};
use crate::neuron::Neuron;
use crate::types::NeuronGroup;
use crate::dimension::MainDimension;
use crate::dimension::LanguageRuntime;
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
// MNIST Checkpoint — save Main (five frozen groups) for retention evaluation
// =============================================================================

#[derive(Serialize, Deserialize)]
pub struct MnistCheckpoint {
    pub main: MainDimension,
    pub group_order: Vec<GroupId>,
    /// Baseline accuracies at save time (one per task) for retention comparison.
    pub baseline_accs: Vec<f32>,
}

pub fn save_mnist_checkpoint(
    main: &MainDimension,
    group_order: &[GroupId],
    baseline_accs: &[f32],
    path: &str,
) {
    let checkpoint = MnistCheckpoint {
        main: main.clone(),
        group_order: group_order.to_vec(),
        baseline_accs: baseline_accs.to_vec(),
    };
    let json = serde_json::to_string_pretty(&checkpoint).expect("MnistCheckpoint serialization failed");
    std::fs::write(path, &json).unwrap_or_else(|e| panic!("Failed to write {}: {}", path, e));
    println!("  Checkpoint saved: {} ({} KB)", path, json.len() / 1024);
}

pub fn load_mnist_checkpoint(path: &str) -> (MainDimension, Vec<GroupId>, Vec<f32>) {
    let json = std::fs::read_to_string(path).unwrap_or_else(|_| panic!("Checkpoint not found: {}\nRun --mnist first and complete all 5 tasks.", path));
    let ckpt: MnistCheckpoint = serde_json::from_str(&json).expect("MnistCheckpoint deserialization failed");
    (ckpt.main, ckpt.group_order, ckpt.baseline_accs)
}

// =============================================================================
// Language Checkpoint — save calibrated bridge + per-group language vectors
// =============================================================================

#[derive(Serialize, Deserialize)]
pub struct LanguageCheckpoint {
    pub runtime: LanguageRuntime,
    pub group_language_vectors: HashMap<GroupId, Vec<f32>>,
}

pub fn save_language_checkpoint(
    runtime: &LanguageRuntime,
    group_language_vectors: &HashMap<GroupId, Vec<f32>>,
    path: &str,
) {
    let checkpoint = LanguageCheckpoint {
        runtime: runtime.clone(),
        group_language_vectors: group_language_vectors.clone(),
    };
    let json = serde_json::to_string_pretty(&checkpoint).expect("LanguageCheckpoint serialization failed");
    std::fs::write(path, &json).unwrap_or_else(|e| panic!("Failed to write {}: {}", path, e));
    println!("  Language checkpoint saved: {} ({} KB)", path, json.len() / 1024);
}

pub fn load_language_checkpoint(path: &str) -> (LanguageRuntime, HashMap<GroupId, Vec<f32>>) {
    let json = std::fs::read_to_string(path).unwrap_or_else(|_| panic!("Language checkpoint not found: {}", path));
    let ckpt: LanguageCheckpoint = serde_json::from_str(&json).expect("LanguageCheckpoint deserialization failed");
    (ckpt.runtime, ckpt.group_language_vectors)
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

    /// Proves the MNIST checkpoint can be written to disk and loaded back.
    /// Uses a temp file so it works on any system without polluting the repo.
    #[test]
    fn test_mnist_checkpoint_write_and_load() {
        use crate::dimension::MainDimension;
        use crate::dimension::embedding::GroupEmbedding;
        use crate::environment::NeuralEnvironment;
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        use std::fs;

        let config = EnvironmentConfig::default();
        let mut rng = StdRng::seed_from_u64(42);
        let mut env = NeuralEnvironment::new(config);
        env.build_layers(&[64, 32, 32, 1], &mut rng);
        env.freeze_all();

        let calibration: Vec<crate::types::Sample> = (0..5)
            .map(|i| (vec![0.0_f32; 64], [if i % 2 == 0 { 0.0 } else { 1.0 }]))
            .collect();
        let vector = crate::dimension::embedding::compute_group_embedding(&mut env, &calibration);
        let embedding = GroupEmbedding {
            group_id: 0,
            vector,
            task_name: "test_task".to_string(),
            accuracy: 0.95,
            intrinsic_dim: None,
            description: None,
            metatags: vec![],
            tag_vector: vec![],
            language_vector: vec![],
        };

        let mut main = MainDimension::new();
        main.register_group(0, "test_task".to_string(), env, embedding, 0.95, 100);
        let group_order = vec![0];
        let baseline_accs = vec![0.95, 0.92, 0.98, 0.97, 0.96];

        let path = std::env::temp_dir().join("growformer_mnist_checkpoint_test.json");
        let path_str = path.to_str().expect("temp path");

        save_mnist_checkpoint(&main, &group_order, &baseline_accs, path_str);

        assert!(path.exists(), "checkpoint file must exist after save");
        let meta = fs::metadata(path_str).expect("metadata");
        assert!(meta.len() > 100, "checkpoint file must be non-trivial ({} bytes)", meta.len());

        let (loaded_main, loaded_order, loaded_accs) = load_mnist_checkpoint(path_str);

        assert_eq!(loaded_order, group_order);
        assert_eq!(loaded_accs, baseline_accs);
        assert_eq!(loaded_main.group_order.len(), main.group_order.len());
        assert!(loaded_main.groups.contains_key(&0));

        fs::remove_file(path_str).ok();
    }

    #[test]
    fn test_language_checkpoint_write_and_load() {
        use crate::dimension::{LanguageConfig, LanguageRuntime};
        use std::fs;

        let runtime = LanguageRuntime::new(LanguageConfig::default());
        let mut vectors = HashMap::new();
        vectors.insert(0u32, vec![0.1f32; 64]);
        vectors.insert(1u32, vec![0.2f32; 64]);

        let path = std::env::temp_dir().join("growformer_language_checkpoint_test.json");
        let path_str = path.to_str().expect("temp path");
        save_language_checkpoint(&runtime, &vectors, path_str);
        assert!(path.exists(), "language checkpoint must exist");

        let (loaded_runtime, loaded_vectors) = load_language_checkpoint(path_str);
        assert_eq!(loaded_runtime.config.bridge_output_dim, runtime.config.bridge_output_dim);
        assert_eq!(loaded_vectors.len(), vectors.len());
        assert_eq!(loaded_vectors.get(&0).map(|v| v.len()), Some(64));

        fs::remove_file(path_str).ok();
    }
}
