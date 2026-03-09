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

use super::composition::{EpisodicMemory, Episode, VirtualGroup};
use super::action::{ActionJson, action_from_routing};
use super::embedding::{compute_group_embedding, build_tag_vector, GroupEmbedding, TAG_VECTOR_DIM};
use super::language::{
    CalibrationDataset, CalibrationReport, CalibrationRequirements, LanguageConfig,
    LanguageRoutingDecision, LanguageRuntime, route_language_embedding,
};
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
    /// If > 0, mirrors spawn with a reserve pool of this many neurons for neurogenesis (promote from pool).
    pub reserve_pool_size: usize,
}

impl Default for DimensionManagerConfig {
    fn default() -> Self {
        Self {
            mirror_config: EnvironmentConfig::default(),
            mirror_layer_sizes: vec![2, 16, 16, 1],
            promotion_check_interval: 500,
            max_concurrent_mirrors: 2,
            calibration_samples: 100,
            reserve_pool_size: 0,
        }
    }
}

pub struct DimensionManager {
    pub main: MainDimension,
    pub mirrors: HashMap<String, MirrorDimension>,
    pub observer: GlobalObserver,
    pub episodic_memory: EpisodicMemory,
    pub config: DimensionManagerConfig,
    pub language_runtime: LanguageRuntime,
    next_group_id: GroupId,
}

impl DimensionManager {
    pub fn new(config: DimensionManagerConfig) -> Self {
        Self {
            main: MainDimension::new(),
            mirrors: HashMap::new(),
            observer: GlobalObserver::new(PromotionGateConfig::default()),
            episodic_memory: EpisodicMemory::new(),
            config,
            language_runtime: LanguageRuntime::new(LanguageConfig::default()),
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
        let mirror = if self.config.reserve_pool_size > 0 {
            MirrorDimension::new_with_reserve_pool(
                task_name.to_string(),
                env,
                self.config.mirror_config.clone(),
                self.config.reserve_pool_size,
            )
        } else {
            MirrorDimension::new(
                task_name.to_string(),
                env,
                self.config.mirror_config.clone(),
            )
        };
        self.mirrors.insert(task_name.to_string(), mirror);
        self.mirrors.get_mut(task_name)
    }

    /// Train one epoch in a named mirror.
    /// Train one epoch on the mirror. If `batch_size` is `Some(b)` with `b > 1`, uses minibatch SGD
    /// (B clones in parallel, then average params) for faster multi-core runs.
    pub fn train_mirror_epoch(
        &mut self,
        task_name: &str,
        data: &[crate::types::Sample],
        rng: &mut impl Rng,
        batch_size: Option<usize>,
    ) -> Option<EpochResult> {
        self.mirrors.get_mut(task_name).map(|m| {
            match batch_size {
                Some(b) if b > 1 => {
                    let epoch = m.epochs_trained;
                    m.train_epoch_minibatch(data, b, epoch, rng)
                }
                _ => m.train_epoch(data, rng),
            }
        })
    }

    /// If mirror has reached epoch_trigger and current_loss > loss_threshold and not yet triggered,
    /// insert one neuron into its last hidden layer (neurogenesis). Returns true if a neuron was added.
    pub fn try_mirror_neurogenesis(
        &mut self,
        task_name: &str,
        epoch_trigger: u32,
        loss_threshold: f32,
        current_loss: f32,
        rng: &mut impl Rng,
    ) -> bool {
        self.mirrors
            .get_mut(task_name)
            .map_or(false, |m| m.try_neurogenesis_trigger(epoch_trigger, loss_threshold, current_loss, rng))
    }

    /// Residual-based neurogenesis: add one neuron if loss has been above threshold for at least
    /// min_epochs_high consecutive epochs. Returns true if a neuron was added.
    pub fn try_mirror_neurogenesis_residual(
        &mut self,
        task_name: &str,
        residual_threshold: f32,
        min_epochs_high: u32,
        current_loss: f32,
        rng: &mut impl Rng,
    ) -> bool {
        self.mirrors.get_mut(task_name).map_or(false, |m| {
            m.try_neurogenesis_trigger_residual(residual_threshold, min_epochs_high, current_loss, rng)
        })
    }

    /// Check all mirrors for promotion; promote those that pass.
    pub fn evaluate_promotions(&mut self, calibration_data: &[crate::types::Sample]) {
        let _ = self.observer.evaluate_mirrors(
            &mut self.mirrors,
            &mut self.main,
            calibration_data,
            &mut self.next_group_id,
        );
    }

    /// Force promote a mirror (for demos / testing). Consumes the mirror and registers in Main.
    pub fn force_promote(&mut self, task_name: &str, calibration_data: &[crate::types::Sample]) -> Option<GroupId> {
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
            language_vector: vec![],
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
        data: &[crate::types::Sample],
    ) -> f32 {
        let fg = match self.main.groups.get_mut(&group_id) {
            Some(f) => f,
            None => return 0.0,
        };
        let mut correct = 0usize;
        for (input, target) in data {
            let out = fg.env.predict(input.as_slice());
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
    /// Call after promotion so main.group_order.len() == data_per_group.len(). Uses input_dim from config.mirror_layer_sizes[0], hidden = 16, lr = 0.15.
    /// Samples are shuffled each epoch to avoid biasing toward the last class.
    pub fn train_and_set_router(
        &mut self,
        data_per_group: &[(&[crate::types::Sample], usize)],
        rng: &mut impl Rng,
        epochs: usize,
    ) {
        if data_per_group.is_empty() || self.main.group_order.len() != data_per_group.len() {
            return;
        }
        let input_dim = self.config.mirror_layer_sizes.first().copied().unwrap_or(2);
        let num_groups = data_per_group.len();
        let mut router = LearnedRouter::new(input_dim, num_groups, 16, rng);
        let mut samples: Vec<(Vec<f32>, usize)> = Vec::new();
        for (data, group_index) in data_per_group {
            if *group_index >= num_groups {
                continue;
            }
            for (input, _target) in data.iter() {
                samples.push((input.clone(), *group_index));
            }
        }
        for _ in 0..epochs {
            samples.shuffle(rng);
            for (input, group_index) in &samples {
                router.train_step(input.as_slice(), *group_index as GroupId, rng);
            }
        }
        self.observer.learned_router = Some(router);
    }

    /// Set the learned router directly (e.g. after adding a new group: train a router with
    /// num_groups = main.group_order.len() elsewhere and pass it in, or load from checkpoint).
    /// Replaces any existing router. No retraining of existing main groups.
    pub fn set_router(&mut self, router: LearnedRouter) {
        self.observer.learned_router = Some(router);
    }

    /// Create a VirtualGroup for the given group IDs and train blend weights on data.
    /// Returns (trained VirtualGroup, accuracy on data). Use small data (e.g. 20–50 samples).
    pub fn train_composition(
        &mut self,
        group_ids: &[GroupId],
        data: &[crate::types::Sample],
        lr: f32,
        epochs: usize,
    ) -> (VirtualGroup, f32) {
        let mut vg = VirtualGroup::new(group_ids.to_vec());
        for _ in 0..epochs {
            for (input, target) in data {
                vg.train_step(&mut self.main, input.as_slice(), target, lr);
            }
        }
        let mut correct = 0usize;
        for (input, target) in data {
            let out = vg.predict(&mut self.main, input.as_slice());
            if out.len() >= 1 && (out[0] - target[0]).abs() < 0.5 {
                correct += 1;
            }
        }
        let acc = if data.is_empty() {
            0.0
        } else {
            correct as f32 / data.len() as f32
        };
        (vg, acc)
    }

    /// Store a successful composition in episodic memory. Signature = mean of input coords (any dimension).
    pub fn store_composition_episode(
        &mut self,
        virtual_group: &VirtualGroup,
        data: &[crate::types::Sample],
        accuracy: f32,
        residual: f32,
    ) {
        if data.is_empty() {
            return;
        }
        let dim = data[0].0.len();
        let mut input_signature = vec![0.0f32; dim];
        for (input, _) in data {
            for (i, &v) in input.iter().enumerate() {
                if i < dim {
                    input_signature[i] += v;
                }
            }
        }
        let n = data.len() as f32;
        for v in &mut input_signature {
            *v /= n;
        }
        self.episodic_memory.store(Episode {
            input_signature,
            group_ids: virtual_group.group_ids.clone(),
            blend_weights: virtual_group.blend_weights.clone(),
            accuracy,
            residual,
        });
    }

    /// Infer using a VirtualGroup (blend of frozen groups). For Phase 3c demo.
    pub fn predict_with_composition(&mut self, input: &[f32], virtual_group: &VirtualGroup) -> Vec<f32> {
        virtual_group.predict(&mut self.main, input)
    }

    /// Retrieve a stored composition by signature similarity (e.g. [x, y] or mean of batch).
    pub fn episodic_retrieve(&self, signature: &[f32], threshold: f32) -> Option<&Episode> {
        self.episodic_memory.retrieve(signature, threshold)
    }

    /// Infer using a retrieved episode (blend weights from episodic memory). For memory-recall path.
    pub fn predict_with_episode(&mut self, input: &[f32], episode: &Episode) -> Vec<f32> {
        let vg = VirtualGroup {
            group_ids: episode.group_ids.clone(),
            blend_weights: episode.blend_weights.clone(),
        };
        vg.predict(&mut self.main, input)
    }

    /// Replace language runtime config (encoder preset, bridge dim, EMA alpha, OOD threshold).
    pub fn configure_language(&mut self, config: LanguageConfig) {
        self.language_runtime = LanguageRuntime::new(config);
    }

    /// Run one-time global bridge calibration and freeze the bridge.
    pub fn calibrate_language_bridge(
        &mut self,
        dataset: &CalibrationDataset,
        requirements: &CalibrationRequirements,
    ) -> Result<CalibrationReport, String> {
        self.language_runtime.calibrate(dataset, requirements)
    }

    /// Attach a calibrated 64-d language routing vector to an existing group.
    pub fn set_group_language_vector(
        &mut self,
        group_id: GroupId,
        language_vector: Vec<f32>,
    ) -> Result<(), String> {
        if language_vector.len() != self.language_runtime.config.bridge_output_dim {
            return Err(format!(
                "language vector must be {} dims, got {}",
                self.language_runtime.config.bridge_output_dim,
                language_vector.len()
            ));
        }
        if let Some(group) = self.main.groups.get_mut(&group_id) {
            group.embedding.language_vector = language_vector.clone();
        } else {
            return Err(format!("group {} not found", group_id));
        }
        if let Some(lib) = self.main.embedding_library.iter_mut().find(|e| e.group_id == group_id) {
            lib.language_vector = language_vector;
            Ok(())
        } else {
            Err(format!("embedding library entry missing for group {}", group_id))
        }
    }

    /// Route raw text to a promoted group using encoder -> bridge -> 64-d cosine routing.
    pub fn route_text(&mut self, text: &str) -> Result<LanguageRoutingDecision, String> {
        let bridged = self.language_runtime.bridge_text(text)?;
        Ok(route_language_embedding(
            &self.main.embedding_library,
            &bridged.routed_vector,
            bridged.confidence,
            self.language_runtime.config.ood_similarity_threshold,
        ))
    }

    /// Stateless routing for independent single-turn evaluation.
    pub fn route_text_stateless(&self, text: &str) -> Result<LanguageRoutingDecision, String> {
        let bridged = self.language_runtime.bridge_text_stateless(text)?;
        Ok(route_language_embedding(
            &self.main.embedding_library,
            &bridged.routed_vector,
            bridged.confidence,
            self.language_runtime.config.ood_similarity_threshold,
        ))
    }

    /// M3 deterministic path: text -> routing -> structured action JSON.
    pub fn route_text_to_action(&mut self, text: &str) -> Result<ActionJson, String> {
        let routing = self.route_text(text)?;
        Ok(action_from_routing(&self.main, &routing, text))
    }

    /// M3 deterministic path with an explicit OOD threshold override.
    pub fn route_text_to_action_with_threshold(
        &self,
        text: &str,
        ood_threshold: f32,
    ) -> Result<ActionJson, String> {
        let bridged = self.language_runtime.bridge_text_stateless(text)?;
        let routing = route_language_embedding(
            &self.main.embedding_library,
            &bridged.routed_vector,
            bridged.confidence,
            ood_threshold,
        );
        Ok(action_from_routing(&self.main, &routing, text))
    }

    /// Build one group language vector by averaging bridged vectors over representative prompts.
    pub fn build_group_language_vector_from_texts(
        &mut self,
        texts: &[String],
    ) -> Result<Vec<f32>, String> {
        if texts.is_empty() {
            return Err("texts must not be empty".to_string());
        }
        let dim = self.language_runtime.config.bridge_output_dim;
        let mut acc = vec![0.0f32; dim];
        let mut n = 0f32;
        for t in texts {
            let out = self.language_runtime.bridge_text_stateless(t)?;
            if out.routed_vector.len() != dim {
                return Err("bridged vector has unexpected dimension".to_string());
            }
            for (a, v) in acc.iter_mut().zip(out.routed_vector.iter()) {
                *a += *v;
            }
            n += 1.0;
        }
        for a in &mut acc {
            *a /= n;
        }
        Ok(acc)
    }

    /// Convenience API: build and attach a group language vector from prompts.
    pub fn set_group_language_vector_from_texts(
        &mut self,
        group_id: GroupId,
        texts: &[String],
    ) -> Result<(), String> {
        let v = self.build_group_language_vector_from_texts(texts)?;
        self.set_group_language_vector(group_id, v)
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
