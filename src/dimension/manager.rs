//! DimensionManager — top-level entry point. Owns Main, Mirrors, Observer.

use std::collections::HashMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use rand::Rng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use crate::environment::NeuralEnvironment;
use crate::types::EnvironmentConfig;
use crate::types::GroupId;

use super::composition::{EpisodicMemory, Episode, VirtualGroup};
use super::action::{ActionJson, ActionType, action_from_routing};
use super::action_classifier::ActionClassifier;
use super::generation_head::GenerationHead;
use super::group_gen::GroupGenEnv;
use crate::spectral::TokenDictionary;
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

#[derive(Serialize, Deserialize)]
pub struct DimensionManager {
    pub main: MainDimension,
    pub mirrors: HashMap<String, MirrorDimension>,
    pub observer: GlobalObserver,
    pub episodic_memory: EpisodicMemory,
    pub config: DimensionManagerConfig,
    pub language_runtime: LanguageRuntime,
    pub action_classifier: Option<ActionClassifier>,
    pub generation_head: Option<GenerationHead>,
    pub codegen_head: Option<GenerationHead>,
    #[serde(default)]
    pub group_gen_envs: HashMap<usize, GroupGenEnv>,
    #[serde(default)]
    pub group_code_envs: HashMap<usize, GroupGenEnv>,
    #[serde(default)]
    pub gen_dictionary: Option<TokenDictionary>,
    #[serde(default)]
    pub code_dictionary: Option<TokenDictionary>,
    next_group_id: GroupId,
    low_confidence_streak: u32,
    pub auto_spawn_threshold: f32,
    pub auto_spawn_k: u32,
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
            action_classifier: None,
            generation_head: None,
            codegen_head: None,
            group_gen_envs: HashMap::new(),
            group_code_envs: HashMap::new(),
            gen_dictionary: None,
            code_dictionary: None,
            next_group_id: 0,
            low_confidence_streak: 0,
            auto_spawn_threshold: 0.15,
            auto_spawn_k: 10,
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

    /// Train a LearnedRouter on (embedding, group_index) pairs from language data.
    /// `samples` is (embedding_vec, group_index). Returns (train_loss, accuracy).
    /// When `batch_size` is `Some(b)`, runs minibatch SGD: B clones train in parallel, then params are averaged (uses multiple cores).
    pub fn train_language_router(
        &mut self,
        samples: &[(Vec<f32>, usize)],
        epochs: usize,
        rng: &mut impl Rng,
        batch_size: Option<usize>,
    ) -> (f32, f32) {
        let num_groups = self.main.group_order.len();
        if num_groups == 0 || samples.is_empty() {
            return (0.0, 0.0);
        }
        let input_dim = samples[0].0.len();
        let hidden = 64.min(input_dim);
        let mut router = LearnedRouter::new(input_dim, num_groups, hidden, rng);
        let mut indices: Vec<usize> = (0..samples.len()).collect();
        let mut last_loss = 0.0f32;

        let use_parallel = batch_size.map_or(0, |b| b).saturating_sub(1) > 0;

        for epoch in 0..epochs {
            indices.shuffle(rng);
            if use_parallel {
                let b = batch_size.unwrap_or(1);
                for chunk in indices.chunks(b) {
                    let batch_data: Vec<(&[f32], usize)> = chunk
                        .iter()
                        .map(|&i| (samples[i].0.as_slice(), samples[i].1))
                        .collect();
                    let mut clones: Vec<LearnedRouter> =
                        (0..batch_data.len()).map(|_| router.clone()).collect();
                    let seed_base = (epoch as u64).wrapping_mul(1_000_000).wrapping_add(chunk[0] as u64);
                    let losses: Vec<f32> = crate::maybe_par_iter_mut!(clones)
                        .zip(batch_data)
                        .enumerate()
                        .map(|(i, (clone, (emb, group_idx)))| {
                            let mut thread_rng = StdRng::seed_from_u64(seed_base.wrapping_add(i as u64));
                            clone.train_step(emb, group_idx as GroupId, &mut thread_rng)
                        })
                        .collect();
                    last_loss = losses.iter().sum::<f32>() / losses.len().max(1) as f32;
                    if let Some(avg_router) = LearnedRouter::average_from(&clones) {
                        router = avg_router;
                    }
                }
            } else {
                let mut total_loss = 0.0f32;
                for &i in &indices {
                    let (ref emb, group_idx) = samples[i];
                    total_loss += router.train_step(emb, group_idx as GroupId, rng);
                }
                last_loss = total_loss / indices.len() as f32;
            }
        }

        let mut correct = 0usize;
        for (emb, expected) in samples {
            if let Some(chosen) = router.choose_group(emb) {
                if chosen as usize == *expected {
                    correct += 1;
                }
            }
        }
        let accuracy = correct as f32 / samples.len() as f32;
        self.observer.learned_router = Some(router);
        (last_loss, accuracy)
    }

    /// Train the action classifier from (embedding, ActionType) pairs with balanced class sampling.
    /// Returns (loss, accuracy).
    pub fn train_action_classifier(
        &mut self,
        samples: &[(Vec<f32>, ActionType)],
        epochs: usize,
        lr: f32,
    ) -> (f32, f32) {
        if samples.is_empty() {
            return (0.0, 0.0);
        }
        let input_dim = samples[0].0.len();
        let mut clf = ActionClassifier::new(input_dim, 48);

        // Group indices by class for balanced sampling
        use super::action_classifier::NUM_ACTION_TYPES;
        let mut by_class: Vec<Vec<usize>> = vec![vec![]; NUM_ACTION_TYPES];
        for (i, (_, at)) in samples.iter().enumerate() {
            let idx = match at {
                ActionType::SupportTicket => 0,
                ActionType::CodingAssist => 1,
                ActionType::GeneralAssist => 2,
                ActionType::Fallback => 3,
            };
            by_class[idx].push(i);
        }
        let active_classes: Vec<usize> = by_class.iter().enumerate()
            .filter(|(_, v)| !v.is_empty())
            .map(|(i, _)| i)
            .collect();
        let samples_per_class_per_epoch = 50;

        let mut last_loss = 0.0f32;
        let mut rng_seed = 77u64;
        for _ in 0..epochs {
            let mut total_loss = 0.0f32;
            let mut steps = 0usize;
            for &cls in &active_classes {
                let pool = &by_class[cls];
                for k in 0..samples_per_class_per_epoch {
                    rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let idx = pool[(rng_seed as usize / 7) % pool.len()];
                    let (ref emb, ref at) = samples[idx];
                    total_loss += clf.train_step(emb, at, lr);
                    steps += 1;
                    let _ = k;
                }
            }
            last_loss = total_loss / steps.max(1) as f32;
        }
        let mut correct = 0usize;
        for (emb, expected) in samples {
            if clf.predict(emb) == *expected {
                correct += 1;
            }
        }
        let accuracy = correct as f32 / samples.len() as f32;
        self.action_classifier = Some(clf);
        (last_loss, accuracy)
    }

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

    /// Route raw text to a promoted group using encoder -> bridge -> routing.
    /// When a trained LearnedRouter is present and dimensions match, uses the router (MLP);
    /// otherwise falls back to cosine similarity over embedding_library centroids.
    pub fn route_text(&mut self, text: &str) -> Result<LanguageRoutingDecision, String> {
        let bridged = self.language_runtime.bridge_text(text)?;
        let vec = &bridged.routed_vector;
        let confidence = bridged.confidence;
        let ood_threshold = self.language_runtime.config.ood_similarity_threshold;

        let use_router = self.observer.learned_router.as_ref().map_or(false, |r| {
            r.input_dim == vec.len() && r.num_groups == self.main.group_order.len()
        });

        if use_router {
            let router = self.observer.learned_router.as_mut().unwrap();
            let logits = router.predict_logits(vec);
            if logits.len() == self.main.group_order.len() {
                let mut indexed: Vec<(usize, f32)> =
                    logits.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let (best_idx, best_logit) = indexed.first().copied().unwrap_or((0, -1e9));
                let second_logit = indexed.get(1).map(|x| x.1).unwrap_or(-1e9);
                let margin = best_logit - second_logit;
                let chosen_group_id = self.main.group_order.get(best_idx).copied();
                // Router logits are not on cosine scale; only reject when we have no valid group.
                let rejected_as_ood = chosen_group_id.is_none();
                return Ok(LanguageRoutingDecision {
                    chosen_group_id: if rejected_as_ood { None } else { chosen_group_id },
                    best_similarity: best_logit,
                    second_similarity: second_logit,
                    margin,
                    confidence,
                    rejected_as_ood,
                });
            }
        }

        Ok(route_language_embedding(
            &self.main.embedding_library,
            vec,
            confidence,
            ood_threshold,
        ))
    }

    /// Stateless routing for independent single-turn evaluation.
    /// Uses the learned router when available (same as route_text but without EMA smoothing).
    pub fn route_text_stateless(&mut self, text: &str) -> Result<LanguageRoutingDecision, String> {
        let bridged = self.language_runtime.bridge_text_stateless(text)?;
        let vec = &bridged.routed_vector;
        let confidence = bridged.confidence;

        let use_router = self.observer.learned_router.as_ref().map_or(false, |r| {
            r.input_dim == vec.len() && r.num_groups == self.main.group_order.len()
        });

        if use_router {
            let router = self.observer.learned_router.as_mut().unwrap();
            let logits = router.predict_logits(vec);
            if logits.len() == self.main.group_order.len() {
                let mut indexed: Vec<(usize, f32)> =
                    logits.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let (best_idx, best_logit) = indexed.first().copied().unwrap_or((0, -1e9));
                let second_logit = indexed.get(1).map(|x| x.1).unwrap_or(-1e9);
                let margin = best_logit - second_logit;
                let chosen_group_id = self.main.group_order.get(best_idx).copied();
                let rejected_as_ood = chosen_group_id.is_none();
                return Ok(LanguageRoutingDecision {
                    chosen_group_id: if rejected_as_ood { None } else { chosen_group_id },
                    best_similarity: best_logit,
                    second_similarity: second_logit,
                    margin,
                    confidence,
                    rejected_as_ood,
                });
            }
        }

        Ok(route_language_embedding(
            &self.main.embedding_library,
            vec,
            confidence,
            self.language_runtime.config.ood_similarity_threshold,
        ))
    }

    /// M3 deterministic path: text -> routing -> structured action JSON.
    /// Uses stateful EMA bridging — suitable for multi-turn conversations.
    pub fn route_text_to_action(&mut self, text: &str) -> Result<ActionJson, String> {
        let routing = self.route_text(text)?;
        let mut action = action_from_routing(&self.main, &routing, text);

        if let Some(ref clf) = self.action_classifier {
            if let Ok(bridged) = self.language_runtime.bridge_text_stateless(text) {
                let (predicted_type, conf) = clf.predict_with_confidence(&bridged.routed_vector);
                if conf > 0.4 {
                    action.action_type = predicted_type;
                    action.confidence = conf;
                    action.reason = "classified".to_string();
                }
            }
        }

        Ok(action)
    }

    /// Stateless action routing for independent single-turn prompts.
    /// No EMA smoothing — each call is independent.
    pub fn route_text_to_action_stateless(&mut self, text: &str) -> Result<ActionJson, String> {
        let routing = self.route_text_stateless(text)?;
        let mut action = action_from_routing(&self.main, &routing, text);

        if let Some(ref clf) = self.action_classifier {
            if let Ok(bridged) = self.language_runtime.bridge_text_stateless(text) {
                let (predicted_type, conf) = clf.predict_with_confidence(&bridged.routed_vector);
                if conf > 0.4 {
                    action.action_type = predicted_type;
                    action.confidence = conf;
                    action.reason = "classified".to_string();
                }
            }
        }

        Ok(action)
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

    /// Track routing confidence and detect when K consecutive route calls
    /// fall below `auto_spawn_threshold`. Returns `Some(task_name)` when the
    /// spawn trigger fires, or `None` if confidence is still acceptable.
    /// Caller is responsible for actually spawning the mirror if desired.
    pub fn track_confidence_for_auto_spawn(
        &mut self,
        routing: &LanguageRoutingDecision,
    ) -> Option<String> {
        let below = routing.rejected_as_ood
            || routing.best_similarity < self.auto_spawn_threshold;
        if below {
            self.low_confidence_streak += 1;
        } else {
            self.low_confidence_streak = 0;
        }
        if self.low_confidence_streak >= self.auto_spawn_k {
            self.low_confidence_streak = 0;
            let name = format!("auto_spawn_{}", self.next_group_id);
            Some(name)
        } else {
            None
        }
    }

    /// Route text and auto-check for mirror spawn trigger.
    /// Returns (routing_decision, Option<suggested_mirror_name>).
    pub fn route_text_with_spawn_check(
        &mut self,
        text: &str,
    ) -> Result<(LanguageRoutingDecision, Option<String>), String> {
        let routing = self.route_text(text)?;
        let spawn = self.track_confidence_for_auto_spawn(&routing);
        Ok((routing, spawn))
    }

    pub fn low_confidence_streak(&self) -> u32 {
        self.low_confidence_streak
    }

    /// Summarize episodic memory for cross-mode read access (M6 shared-state contract).
    pub fn episodic_summaries(&self) -> Vec<EpisodicSummary> {
        self.episodic_memory
            .episodes
            .iter()
            .enumerate()
            .map(|(i, ep)| EpisodicSummary {
                index: i,
                group_ids: ep.group_ids.clone(),
                accuracy: ep.accuracy,
                residual: ep.residual,
            })
            .collect()
    }

    /// Count of checkpoint-worthy state: groups, mirrors, episodes.
    pub fn checkpoint_size_summary(&self) -> CheckpointSizeSummary {
        CheckpointSizeSummary {
            promoted_groups: self.main.group_order.len(),
            active_mirrors: self.mirrors.len(),
            episodic_episodes: self.episodic_memory.episodes.len(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicSummary {
    pub index: usize,
    pub group_ids: Vec<GroupId>,
    pub accuracy: f32,
    pub residual: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointSizeSummary {
    pub promoted_groups: usize,
    pub active_mirrors: usize,
    pub episodic_episodes: usize,
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
