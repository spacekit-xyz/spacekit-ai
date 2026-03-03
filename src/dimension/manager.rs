//! DimensionManager — top-level entry point. Owns Main, Mirrors, Observer.

use std::collections::HashMap;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use crate::environment::NeuralEnvironment;
use crate::types::EnvironmentConfig;
use crate::types::GroupId;

use super::embedding::{compute_group_embedding, GroupEmbedding};
use super::main_dim::MainDimension;
use super::mirror_dim::{MirrorDimension, EpochResult};
use super::observer::GlobalObserver;
use super::promotion::PromotionGateConfig;

#[derive(Clone, Serialize, Deserialize)]
pub struct DimensionManagerConfig {
    pub mirror_config: EnvironmentConfig,
    pub mirror_layer_sizes: Vec<usize>,
    pub promotion_check_interval: u32,
    pub max_concurrent_mirrors: usize,
    pub calibration_samples: usize,
}

impl Default for DimensionManagerConfig {
    fn default() -> Self {
        Self {
            mirror_config: EnvironmentConfig::default(),
            mirror_layer_sizes: vec![2, 16, 16, 1],
            promotion_check_interval: 500,
            max_concurrent_mirrors: 2,
            calibration_samples: 100,
        }
    }
}

pub struct DimensionManager {
    pub main: MainDimension,
    pub mirrors: HashMap<String, MirrorDimension>,
    pub observer: GlobalObserver,
    pub config: DimensionManagerConfig,
    next_group_id: GroupId,
}

impl DimensionManager {
    pub fn new(config: DimensionManagerConfig) -> Self {
        Self {
            main: MainDimension::new(),
            mirrors: HashMap::new(),
            observer: GlobalObserver::new(PromotionGateConfig::default()),
            config,
            next_group_id: 0,
        }
    }

    /// Single entry point for inference.
    pub fn infer(&mut self, input: &[f32]) -> Vec<f32> {
        self.observer.infer(input, &mut self.main)
    }

    /// Spawn a new Mirror for a task. Fails if at max_concurrent_mirrors or name exists.
    pub fn spawn_mirror(&mut self, task_name: &str, seed: u64) -> Option<&mut MirrorDimension> {
        if self.mirrors.len() >= self.config.max_concurrent_mirrors {
            return None;
        }
        if self.mirrors.contains_key(task_name) {
            return None;
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut env = NeuralEnvironment::new(self.config.mirror_config.clone());
        env.build_layers(&self.config.mirror_layer_sizes, &mut rng);
        let mirror = MirrorDimension::new(
            task_name.to_string(),
            env,
            self.config.mirror_config.clone(),
        );
        self.mirrors.insert(task_name.to_string(), mirror);
        self.mirrors.get_mut(task_name)
    }

    /// Train one epoch in a named mirror.
    pub fn train_mirror_epoch(
        &mut self,
        task_name: &str,
        data: &[([f32; 2], [f32; 1])],
        rng: &mut impl Rng,
    ) -> Option<EpochResult> {
        self.mirrors.get_mut(task_name).map(|m| m.train_epoch(data, rng))
    }

    /// Check all mirrors for promotion; promote those that pass.
    pub fn evaluate_promotions(&mut self, calibration_data: &[([f32; 2], [f32; 1])]) {
        let _ = self.observer.evaluate_mirrors(
            &mut self.mirrors,
            &mut self.main,
            calibration_data,
            &mut self.next_group_id,
        );
    }

    /// Force promote a mirror (for demos / testing). Consumes the mirror and registers in Main.
    pub fn force_promote(&mut self, task_name: &str, calibration_data: &[([f32; 2], [f32; 1])]) -> Option<GroupId> {
        let mirror = self.mirrors.remove(task_name)?;
        let mut env = mirror.env;
        env.freeze_all();
        let vector = compute_group_embedding(&mut env, calibration_data);
        let embedding = GroupEmbedding {
            group_id: self.next_group_id,
            vector,
            task_name: mirror.task_name.clone(),
            accuracy: mirror.best_accuracy,
            intrinsic_dim: None,
        };
        self.main.register_group(
            self.next_group_id,
            mirror.task_name,
            env,
            embedding,
            mirror.best_accuracy,
            mirror.epochs_trained as u64,
        );
        self.observer.embedding_library = self.main.embedding_library.clone();
        let gid = self.next_group_id;
        self.next_group_id = self.next_group_id.saturating_add(1);
        Some(gid)
    }

    /// Evaluate accuracy of a specific main-dimension group on data.
    pub fn evaluate_main_group(
        &mut self,
        group_id: GroupId,
        data: &[([f32; 2], [f32; 1])],
    ) -> f32 {
        let fg = match self.main.groups.get_mut(&group_id) {
            Some(f) => f,
            None => return 0.0,
        };
        let mut correct = 0usize;
        for (input, target) in data {
            let out = fg.env.predict(input);
            if out.len() >= 1 && (out[0] - target[0]).abs() < 0.5 {
                correct += 1;
            }
        }
        if data.is_empty() {
            0.0
        } else {
            correct as f32 / data.len() as f32
        }
    }

    pub fn list_groups(&self) -> Vec<GroupSummary> {
        self.main.group_order.iter().filter_map(|&gid| {
            self.main.groups.get(&gid).map(|fg| GroupSummary {
                group_id: gid,
                task_name: fg.task_name.clone(),
                accuracy: fg.accuracy,
                promoted_at_epoch: fg.promoted_at_epoch,
            })
        }).collect()
    }

    pub fn list_mirrors(&self) -> Vec<MirrorSummary> {
        self.mirrors.iter().map(|(name, m)| MirrorSummary {
            task_name: name.clone(),
            epochs_trained: m.epochs_trained,
            best_accuracy: m.best_accuracy,
        }).collect()
    }

    pub fn coherence(&self) -> f32 {
        self.observer.coherence
    }

    /// Group id that was chosen on the last infer() (for logging routing).
    pub fn last_chosen_group_id(&self) -> Option<GroupId> {
        self.observer.last_chosen_group_id
    }

    /// Per-group (gid, self_sim, cross_sim, margin, score) from last infer(); None if no infer yet.
    pub fn last_routing_scores(&self) -> Option<&[(GroupId, f32, f32, f32, f32)]> {
        self.observer.last_routing_scores.as_deref()
    }
}

#[derive(Debug, Clone)]
pub struct GroupSummary {
    pub group_id: GroupId,
    pub task_name: String,
    pub accuracy: f32,
    pub promoted_at_epoch: u64,
}

#[derive(Debug, Clone)]
pub struct MirrorSummary {
    pub task_name: String,
    pub epochs_trained: u32,
    pub best_accuracy: f32,
}
