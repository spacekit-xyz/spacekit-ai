//! DimensionManager — top-level entry point. Owns Main, Mirrors, Observer.

use std::collections::HashMap;

use rand::Rng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use crate::environment::NeuralEnvironment;
use crate::types::EnvironmentConfig;
use crate::types::GroupId;

use super::embedding::{compute_group_embedding, build_tag_vector, GroupEmbedding, TAG_VECTOR_DIM};
use super::main_dim::MainDimension;
use super::mirror_dim::{MirrorDimension, EpochResult};
use super::observer::GlobalObserver;
use super::promotion::PromotionGateConfig;
use super::router::LearnedRouter;

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

    /// Single entry point for inference. Pass context_tags to re-rank by tag-vector similarity.
    pub fn infer(&mut self, input: &[f32]) -> Vec<f32> {
        self.observer.infer(input, &mut self.main, None)
    }

    /// Infer with optional context tags for tag-vector re-rank (e.g. ["spiral"] or ["circles"]).
    pub fn infer_with_context(&mut self, input: &[f32], context_tags: Option<&[String]>) -> Vec<f32> {
        self.observer.infer(input, &mut self.main, context_tags)
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
        let metatags = vec![mirror.task_name.clone()];
        let embedding = GroupEmbedding {
            group_id: self.next_group_id,
            vector,
            task_name: mirror.task_name.clone(),
            accuracy: mirror.best_accuracy,
            intrinsic_dim: None,
            description: None,
            metatags: metatags.clone(),
            tag_vector: build_tag_vector(&metatags, TAG_VECTOR_DIM),
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

    /// Train a learned router on labeled data and set it on the observer.
    /// Each entry is (samples, group_index): group_index is the index into main.group_order (0 = first group, etc.).
    /// Call after promotion so main.group_order.len() == data_per_group.len(). Uses input_dim = 2, hidden = 16, lr = 0.15.
    /// Samples are shuffled each epoch to avoid biasing toward the last class.
    pub fn train_and_set_router(
        &mut self,
        data_per_group: &[(&[([f32; 2], [f32; 1])], usize)],
        rng: &mut impl Rng,
        epochs: usize,
    ) {
        if data_per_group.is_empty() || self.main.group_order.len() != data_per_group.len() {
            return;
        }
        let input_dim = 2usize;
        let num_groups = data_per_group.len();
        let mut router = LearnedRouter::new(input_dim, num_groups, 16, rng);
        let mut samples: Vec<([f32; 2], usize)> = Vec::new();
        for (data, group_index) in data_per_group {
            if *group_index >= num_groups {
                continue;
            }
            for (input, _target) in data.iter() {
                samples.push((*input, *group_index));
            }
        }
        for _ in 0..epochs {
            samples.shuffle(rng);
            for (input, group_index) in &samples {
                router.train_step(input, *group_index as GroupId, rng);
            }
        }
        self.observer.learned_router = Some(router);
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
