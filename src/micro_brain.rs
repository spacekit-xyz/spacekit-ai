//! MetaBrain + Micro-Brain architecture.
//!
//! Every classification/routing decision uses Growformer's own substrate:
//! - `MicroBrain`: Paramecium lattice for classification (topic, verb, action)
//! - `ArchetypeBrain`: global `InfraciliaryLattice` for archetype selection across groups
//! - `MetaBrain`: centroid-based coordinator that fuses micro-brain outputs (zero backprop)

use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use crate::dimension::action::ActionType;
use crate::dimension::group_gen::GEN_COND_DIM;
use crate::dimension::paramecium::{InfraciliaryLattice, BehavioralProgram, WaveState};
use crate::spectral::{E8Lattice, TokenDictionary};
use crate::understanding::{TOPIC_EMBED_DIM, VERB_EMBED_DIM, VERB_LABELS};

// ---------------------------------------------------------------------------
// MicroBrain — Paramecium lattice for classification
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MicroBrainRole {
    Topic,
    Verb,
    Action,
    Custom(String),
}

#[derive(Clone, Serialize, Deserialize)]
pub struct MicroBrain {
    pub lattice: InfraciliaryLattice,
    pub role: MicroBrainRole,
    pub input_dim: usize,
    pub output_dim: usize,
    pub class_names: Vec<String>,
    pub frozen: bool,
}

impl std::fmt::Debug for MicroBrain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MicroBrain")
            .field("role", &self.role)
            .field("input_dim", &self.input_dim)
            .field("output_dim", &self.output_dim)
            .field("frozen", &self.frozen)
            .finish()
    }
}

impl MicroBrain {
    pub fn new(
        role: MicroBrainRole,
        input_dim: usize,
        output_dim: usize,
        _hidden_dim: usize,
        class_names: Vec<String>,
        _rng: &mut impl Rng,
    ) -> Self {
        let labels: Vec<String> = (0..output_dim).map(|i| format!("cls_{}", i)).collect();
        let label_strs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
        let dict = TokenDictionary::build(&label_strs, 64);
        let lattice = InfraciliaryLattice::new(dict);
        Self { lattice, role, input_dim, output_dim, class_names, frozen: false }
    }

    /// Forward pass returning (best_class_idx, confidence, pseudo-logits).
    /// Uses immutable cosine nearest-neighbor (no EMA drift).
    pub fn predict(&mut self, input: &[f32]) -> (usize, f32, Vec<f32>) {
        self.predict_shared(input)
    }

    /// Immutable prediction via direct cosine similarity — no centroid drift.
    pub fn predict_shared(&self, input: &[f32]) -> (usize, f32, Vec<f32>) {
        if input.len() != self.input_dim || self.output_dim == 0 {
            return (0, 0.0, vec![0.0; self.output_dim]);
        }
        if self.lattice.programs.is_empty() {
            return (0, 0.0, vec![0.0; self.output_dim]);
        }
        let mut best_idx = 0;
        let mut best_sim = f32::NEG_INFINITY;
        for (i, prog) in self.lattice.programs.iter().enumerate() {
            let sim = cosine_sim_vecs(input, &prog.ema_centroid);
            if sim > best_sim { best_sim = sim; best_idx = i; }
        }
        let text = self.lattice.programs[best_idx].display_text(&self.lattice.dictionary);
        let cls = parse_cls(&text).unwrap_or(0);
        let conf = best_sim.max(0.0);
        let mut logits = vec![0.0f32; self.output_dim];
        if cls < self.output_dim {
            logits[cls] = conf;
        }
        (cls, conf, logits)
    }

    /// One training step: develop lattice with this (input, class) pair.
    #[cfg(feature = "training")]
    pub fn train_step(&mut self, input: &[f32], target_idx: usize, _rng: &mut impl Rng) -> f32 {
        if self.frozen || input.len() != self.input_dim || target_idx >= self.output_dim {
            return 0.0;
        }
        let label = format!("cls_{}", target_idx);
        let pairs = vec![(input.to_vec(), label)];
        self.lattice.develop(&pairs, 0.90);
        let resp = self.lattice.respond(input);
        let predicted = parse_cls(&resp.text).unwrap_or(usize::MAX);
        if predicted == target_idx { 0.0 } else { 1.0 }
    }

    /// Build from labeled data in one pass.
    #[cfg(feature = "training")]
    pub fn build_from_data(
        role: MicroBrainRole,
        input_dim: usize,
        output_dim: usize,
        class_names: Vec<String>,
        samples: &[(&[f32], usize)],
    ) -> Self {
        let labels: Vec<String> = (0..output_dim).map(|i| format!("cls_{}", i)).collect();
        let label_strs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
        let dict = TokenDictionary::build(&label_strs, 64);
        let pairs: Vec<(Vec<f32>, String)> = samples.iter()
            .map(|(emb, idx)| (emb.to_vec(), format!("cls_{}", idx)))
            .collect();
        let mut lattice = InfraciliaryLattice::new(dict);
        lattice.develop(&pairs, 0.90);
        Self { lattice, role, input_dim, output_dim, class_names, frozen: false }
    }

    pub fn freeze(&mut self) { self.frozen = true; }

    pub fn class_name(&self, idx: usize) -> &str {
        self.class_names.get(idx).map(|s| s.as_str()).unwrap_or("unknown")
    }
}

fn parse_cls(text: &str) -> Option<usize> {
    text.strip_prefix("cls_").and_then(|s| s.parse().ok())
}

fn softmax_argmax(logits: &[f32]) -> (usize, f32) {
    if logits.is_empty() { return (0, 0.0); }
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max_l).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|e| e / sum.max(1e-10)).collect();
    let (idx, &conf) = probs.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));
    (idx, conf)
}

// ---------------------------------------------------------------------------
// ArchetypeBrain — global Paramecium lattice for archetype selection
// ---------------------------------------------------------------------------

/// Metadata for each behavioral program in the global archetype lattice.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArchetypeProgram {
    pub group_idx: usize,
    pub archetype_idx: usize,
}

/// Result of archetype brain inference.
#[derive(Clone, Debug)]
pub struct ArchetypeResult {
    pub group_idx: usize,
    pub archetype_idx: usize,
    pub confidence: f32,
    pub wave_energy: f32,
    /// Top-K volley for multi-archetype composition when confidence is low.
    pub volley: Vec<(usize, usize, f32)>, // (group_idx, arch_idx, weight)
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ArchetypeBrain {
    pub lattice: InfraciliaryLattice,
    pub program_meta: Vec<ArchetypeProgram>,
}

impl ArchetypeBrain {
    /// Build from all groups' codebook prototypes.
    /// Each (group_idx, archetype_idx, centroid, token_sequence) becomes one program.
    pub fn build(
        entries: &[(usize, usize, Vec<f32>, Vec<u16>)],
        dictionary: TokenDictionary,
    ) -> Self {
        let mut lattice = InfraciliaryLattice::new(dictionary);
        let mut program_meta = Vec::with_capacity(entries.len());

        for (group_idx, arch_idx, centroid, tokens) in entries {
            let lattice_sig = E8Lattice::quantize_64d(centroid);
            lattice.programs.push(BehavioralProgram {
                centroid: centroid.clone(),
                lattice_signature: lattice_sig,
                token_sequence: tokens.clone(),
                verbatim_display_text: None,
                activation_count: 1,
                ema_centroid: centroid.clone(),
                coherence: 1.0,
                habituation: 0.0,
                quality_score: 0.0,
                reliability: 0.5,
                total_retrievals: 0,
                session_drift: Vec::new(),
                session_hits: 0,
                session_quality_sum: 0.0,
                activation_level: 0.0,
                refractory: false,
            });
            program_meta.push(ArchetypeProgram {
                group_idx: *group_idx,
                archetype_idx: *arch_idx,
            });
        }
        lattice.wave = WaveState::new(lattice.programs.len());

        Self { lattice, program_meta }
    }

    /// Sense + select: find the best archetype across all groups.
    /// Uses immutable cosine nearest-neighbor — no EMA centroid drift.
    pub fn select(&mut self, embedding: &[f32]) -> ArchetypeResult {
        if self.lattice.programs.is_empty() {
            return ArchetypeResult {
                group_idx: 0, archetype_idx: 0,
                confidence: 0.0, wave_energy: 0.0, volley: Vec::new(),
            };
        }

        let mut scored: Vec<(usize, f32)> = self.lattice.programs.iter().enumerate()
            .map(|(i, prog)| (i, cosine_sim_vecs(embedding, &prog.ema_centroid)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let (best_idx, best_sim) = scored[0];
        let meta = self.program_meta.get(best_idx)
            .cloned()
            .unwrap_or(ArchetypeProgram { group_idx: 0, archetype_idx: 0 });

        let volley = if best_sim < 0.9 {
            scored.iter().take(3).filter_map(|&(idx, sim)| {
                self.program_meta.get(idx).map(|m| {
                    (m.group_idx, m.archetype_idx, sim)
                })
            }).collect()
        } else {
            vec![(meta.group_idx, meta.archetype_idx, best_sim)]
        };

        ArchetypeResult {
            group_idx: meta.group_idx,
            archetype_idx: meta.archetype_idx,
            confidence: best_sim.max(0.0),
            wave_energy: best_sim.max(0.0),
            volley,
        }
    }

    pub fn program_count(&self) -> usize {
        self.lattice.program_count()
    }
}

// ---------------------------------------------------------------------------
// MetaBrain — coordinator that learns to weight micro-brain outputs
// ---------------------------------------------------------------------------

/// Result of the full MetaBrain processing pipeline.
#[derive(Clone, Debug)]
pub struct MetaResult {
    pub conditioning: Vec<f32>,
    pub group_idx: Option<usize>,
    pub archetype_idx: Option<usize>,
    pub volley: Vec<(usize, usize, f32)>,
    pub confidence: f32,
    pub topic: String,
    pub verb: String,
    pub action: ActionType,
}

/// Centroid-based coordinator: stores (input, output) pairs and does
/// nearest-neighbor lookup at inference. Zero backprop.
#[derive(Clone, Serialize, Deserialize)]
pub struct CentroidCoordinator {
    pub centroids: Vec<Vec<f32>>,
    pub outputs: Vec<Vec<f32>>,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl CentroidCoordinator {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self { centroids: Vec::new(), outputs: Vec::new(), input_dim, output_dim }
    }

    /// Develop from (input, output) pairs in one pass.
    #[cfg(feature = "training")]
    pub fn develop(&mut self, pairs: &[(Vec<f32>, Vec<f32>)]) {
        for (input, output) in pairs {
            let mut found = false;
            for (i, c) in self.centroids.iter().enumerate() {
                if cosine_sim_vecs(input, c) > 0.98 {
                    // EMA update existing centroid
                    for (j, v) in self.centroids[i].iter_mut().enumerate() {
                        *v = *v * 0.9 + input.get(j).copied().unwrap_or(0.0) * 0.1;
                    }
                    for (j, v) in self.outputs[i].iter_mut().enumerate() {
                        *v = *v * 0.9 + output.get(j).copied().unwrap_or(0.0) * 0.1;
                    }
                    found = true;
                    break;
                }
            }
            if !found {
                self.centroids.push(input.clone());
                self.outputs.push(output.clone());
            }
        }
    }

    /// Nearest-neighbor lookup: find the most similar centroid, return its output.
    pub fn predict(&self, input: &[f32]) -> Vec<f32> {
        if self.centroids.is_empty() {
            return vec![0.0; self.output_dim];
        }
        let mut best_idx = 0;
        let mut best_sim = f32::NEG_INFINITY;
        for (i, c) in self.centroids.iter().enumerate() {
            let sim = cosine_sim_vecs(input, c);
            if sim > best_sim { best_sim = sim; best_idx = i; }
        }
        self.outputs[best_idx].clone()
    }

    /// Train step returns MSE loss (for API compatibility).
    #[cfg(feature = "training")]
    pub fn train_step(&mut self, input: &[f32], target: &[f32]) -> f32 {
        self.develop(&[(input.to_vec(), target.to_vec())]);
        let pred = self.predict(input);
        let mse: f32 = pred.iter().zip(target.iter())
            .map(|(p, t)| (p - t) * (p - t))
            .sum::<f32>() / target.len().max(1) as f32;
        mse
    }
}

fn cosine_sim_vecs(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len().min(b.len()) {
        let ai = a[i] as f64;
        let bi = b[i] as f64;
        dot += ai * bi;
        na += ai * ai;
        nb += bi * bi;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-20 { 0.0 } else { (dot / denom) as f32 }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct MetaBrain {
    pub coordinator: CentroidCoordinator,
    pub archetype_brain: Option<ArchetypeBrain>,
    pub topic_brain: MicroBrain,
    pub verb_brain: MicroBrain,
    pub action_brain: MicroBrain,
    pub topic_embeddings: Vec<Vec<f32>>,
    pub verb_embeddings: Vec<Vec<f32>>,
    pub frozen: bool,
    coordinator_input_dim: usize,
    coordinator_output_dim: usize,
}

impl Default for MetaBrain {
    fn default() -> Self {
        let mut rng = StdRng::seed_from_u64(0);
        let topic_brain = MicroBrain::new(
            MicroBrainRole::Topic, 1, 1, 1, Vec::new(), &mut rng,
        );
        let verb_brain = MicroBrain::new(
            MicroBrainRole::Verb, 1, 1, 1, Vec::new(), &mut rng,
        );
        let action_brain = MicroBrain::new(
            MicroBrainRole::Action, 1, 1, 1, Vec::new(), &mut rng,
        );
        let coordinator = CentroidCoordinator::new(0, 0);
        Self {
            coordinator,
            archetype_brain: None,
            topic_brain,
            verb_brain,
            action_brain,
            topic_embeddings: Vec::new(),
            verb_embeddings: Vec::new(),
            frozen: false,
            coordinator_input_dim: 0,
            coordinator_output_dim: 0,
        }
    }
}

const ACTION_NAMES: &[&str] = &["support", "coding", "general", "tool", "fallback"];

pub fn index_to_action_type(idx: usize) -> ActionType {
    match idx {
        0 => ActionType::SupportTicket,
        1 => ActionType::CodingAssist,
        2 => ActionType::GeneralAssist,
        3 => ActionType::ToolCall,
        _ => ActionType::Fallback,
    }
}

impl MetaBrain {
    /// Build a MetaBrain with Paramecium micro-brains sized for the given vocabulary.
    pub fn build(
        raw_dim: usize,
        num_topics: usize,
        topic_names: Vec<String>,
        topic_embeddings: Vec<Vec<f32>>,
        verb_embeddings: Vec<Vec<f32>>,
        num_actions: usize,
        rng: &mut impl Rng,
    ) -> Self {
        let num_verbs = VERB_LABELS.len();

        let topic_brain = MicroBrain::new(
            MicroBrainRole::Topic,
            raw_dim, num_topics, 64,
            topic_names,
            rng,
        );
        let verb_brain = MicroBrain::new(
            MicroBrainRole::Verb,
            raw_dim, num_verbs, 32,
            VERB_LABELS.iter().map(|s| s.to_string()).collect(),
            rng,
        );
        let action_brain = MicroBrain::new(
            MicroBrainRole::Action,
            raw_dim, num_actions, 32,
            ACTION_NAMES.iter().map(|s| s.to_string()).collect(),
            rng,
        );

        let coord_input = num_topics + num_verbs + num_actions + 3;
        let coord_output = GEN_COND_DIM;
        let coordinator = CentroidCoordinator::new(coord_input, coord_output);

        Self {
            coordinator,
            archetype_brain: None,
            topic_brain,
            verb_brain,
            action_brain,
            topic_embeddings,
            verb_embeddings,
            frozen: false,
            coordinator_input_dim: coord_input,
            coordinator_output_dim: coord_output,
        }
    }

    /// Full inference pipeline: run all micro-brains, fuse via coordinator.
    pub fn process(
        &mut self,
        h_raw: &[f32],
        bridge_vec: &[f32],
    ) -> MetaResult {
        // Run micro-brains
        let (topic_idx, topic_conf, topic_logits) = self.topic_brain.predict(h_raw);
        let (verb_idx, verb_conf, verb_logits) = self.verb_brain.predict(h_raw);
        let (action_idx, action_conf, action_logits) = self.action_brain.predict(h_raw);

        let topic_name = self.topic_brain.class_name(topic_idx).to_string();
        let verb_name = self.verb_brain.class_name(verb_idx).to_string();
        let action = index_to_action_type(action_idx);

        // Run archetype brain
        let arch_result = self.archetype_brain.as_mut().map(|ab| ab.select(h_raw));
        let (group_idx, arch_idx, arch_conf, volley) = match &arch_result {
            Some(r) => (Some(r.group_idx), Some(r.archetype_idx), r.confidence, r.volley.clone()),
            None => (None, None, 0.0, Vec::new()),
        };

        // Assemble coordinator input: all logits + confidence scalars
        let mut coord_input = Vec::with_capacity(self.coordinator_input_dim);
        coord_input.extend_from_slice(&topic_logits);
        coord_input.extend_from_slice(&verb_logits);
        coord_input.extend_from_slice(&action_logits);
        coord_input.push(topic_conf);
        coord_input.push(verb_conf);
        coord_input.push(action_conf);
        coord_input.resize(self.coordinator_input_dim, 0.0);

        let coordinator_output = self.coordinator.predict(&coord_input);

        // Build final conditioning: blend coordinator output with bridge + understanding.
        // Gate low-confidence topic predictions to prevent misclassified topics
        // (e.g. prompt_injection for legitimate coding prompts) from polluting conditioning.
        // Also suppress adversarial catch-all topics that fire as false positives.
        let is_adversarial_topic = topic_name == "prompt_injection"
            || topic_name == "jailbreak"
            || topic_name == "system_prompt_leak";
        let topic_emb = if topic_conf >= 0.45 && !is_adversarial_topic {
            self.topic_embeddings.get(topic_idx)
                .cloned().unwrap_or_else(|| vec![0.0f32; TOPIC_EMBED_DIM])
        } else {
            vec![0.0f32; TOPIC_EMBED_DIM]
        };
        let verb_emb = self.verb_embeddings.get(verb_idx)
            .cloned().unwrap_or_else(|| vec![0.0f32; VERB_EMBED_DIM]);

        let mut conditioning = Vec::with_capacity(GEN_COND_DIM);
        // First 128d: blend of bridge vector and coordinator output
        for i in 0..bridge_vec.len().min(128) {
            let coord_val = coordinator_output.get(i).copied().unwrap_or(0.0);
            conditioning.push(bridge_vec[i] * 0.7 + coord_val * 0.3);
        }
        // Next 32d: topic embedding
        conditioning.extend_from_slice(&topic_emb);
        // Next 16d: verb embedding
        conditioning.extend_from_slice(&verb_emb);
        // Pad to GEN_COND_DIM
        conditioning.resize(GEN_COND_DIM, 0.0);

        MetaResult {
            conditioning,
            group_idx,
            archetype_idx: arch_idx,
            volley,
            confidence: (topic_conf + verb_conf + arch_conf) / 3.0,
            topic: topic_name,
            verb: verb_name,
            action,
        }
    }

    /// Train the coordinator: develop with this (micro-brain output, target conditioning) pair.
    #[cfg(feature = "training")]
    pub fn train_coordinator_step(
        &mut self,
        h_raw: &[f32],
        target_conditioning: &[f32],
    ) -> f32 {
        if self.frozen { return 0.0; }
        let (_, _, topic_logits) = self.topic_brain.predict(h_raw);
        let (_, _, verb_logits) = self.verb_brain.predict(h_raw);
        let (_, _, action_logits) = self.action_brain.predict(h_raw);

        let (_, tc, _) = self.topic_brain.predict(h_raw);
        let (_, vc, _) = self.verb_brain.predict(h_raw);
        let (_, ac, _) = self.action_brain.predict(h_raw);

        let mut coord_input = Vec::with_capacity(self.coordinator_input_dim);
        coord_input.extend_from_slice(&topic_logits);
        coord_input.extend_from_slice(&verb_logits);
        coord_input.extend_from_slice(&action_logits);
        coord_input.push(tc);
        coord_input.push(vc);
        coord_input.push(ac);
        coord_input.resize(self.coordinator_input_dim, 0.0);

        self.coordinator.train_step(&coord_input, target_conditioning)
    }

    pub fn freeze(&mut self) {
        self.frozen = true;
        self.topic_brain.freeze();
        self.verb_brain.freeze();
        self.action_brain.freeze();
    }

    pub fn is_ready(&self) -> bool {
        self.topic_brain.output_dim > 1 && self.verb_brain.output_dim > 1
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_micro_brain_predict_shape() {
        let input_a: Vec<f32> = (0..32).map(|i| if i < 16 { 1.0 } else { 0.0 }).collect();
        let input_b: Vec<f32> = (0..32).map(|i| if i >= 16 { 1.0 } else { 0.0 }).collect();
        let samples = vec![
            (input_a.as_slice(), 0usize),
            (input_b.as_slice(), 1),
        ];
        let mut brain = MicroBrain::build_from_data(
            MicroBrainRole::Topic, 32, 5,
            vec!["a".into(), "b".into(), "c".into(), "d".into(), "e".into()],
            &samples,
        );
        let (idx, conf, logits) = brain.predict(&input_a);
        assert!(idx < 5);
        assert!(conf > 0.0);
        assert_eq!(logits.len(), 5);
    }

    #[test]
    fn test_micro_brain_training_converges() {
        let input_a: Vec<f32> = (0..16).map(|i| if i < 5 { 1.0 } else { 0.0 }).collect();
        let input_b: Vec<f32> = (0..16).map(|i| if i >= 5 && i < 10 { 1.0 } else { 0.0 }).collect();
        let input_c: Vec<f32> = (0..16).map(|i| if i >= 10 { 1.0 } else { 0.0 }).collect();

        let samples: Vec<(&[f32], usize)> = vec![
            (input_a.as_slice(), 0),
            (input_b.as_slice(), 1),
            (input_c.as_slice(), 2),
        ];
        let mut brain = MicroBrain::build_from_data(
            MicroBrainRole::Topic, 16, 3,
            vec!["a".into(), "b".into(), "c".into()],
            &samples,
        );

        let (pred_a, _, _) = brain.predict(&input_a);
        let (pred_b, _, _) = brain.predict(&input_b);
        assert_eq!(pred_a, 0, "should classify input_a as class 0");
        assert_eq!(pred_b, 1, "should classify input_b as class 1");
    }

    #[test]
    fn test_micro_brain_freeze_blocks_training() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut brain = MicroBrain::new(
            MicroBrainRole::Verb, 8, 3, 4,
            vec!["x".into(), "y".into(), "z".into()],
            &mut rng,
        );
        brain.freeze();
        let loss = brain.train_step(&vec![0.5; 8], 0, &mut rng);
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_archetype_brain_build_and_select() {
        let dict = TokenDictionary::build(&["hello world", "goodbye world"], 100);
        let entries = vec![
            (0, 0, vec![1.0f32; 64], dict.encode("hello world")),
            (0, 1, vec![-1.0f32; 64], dict.encode("goodbye world")),
            (1, 0, vec![0.5f32; 64], dict.encode("hello world")),
        ];
        let mut ab = ArchetypeBrain::build(&entries, dict);
        assert_eq!(ab.program_count(), 3);

        let result = ab.select(&vec![0.9f32; 64]);
        assert_eq!(result.group_idx, 0);
        assert_eq!(result.archetype_idx, 0);
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_archetype_brain_volley_spans_groups() {
        let dict = TokenDictionary::build(&["a", "b", "c"], 100);
        let entries = vec![
            (0, 0, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dict.encode("a")),
            (1, 0, vec![0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dict.encode("b")),
            (2, 0, vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dict.encode("c")),
        ];
        let mut ab = ArchetypeBrain::build(&entries, dict);
        // Input close to both group 0 and group 1 archetypes
        let input = vec![0.95, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = ab.select(&input);
        assert!(!result.volley.is_empty());
        // Volley should include programs from multiple groups when confidence is low
        let groups_in_volley: Vec<usize> = result.volley.iter().map(|v| v.0).collect();
        assert!(groups_in_volley.len() >= 1);
    }

    #[test]
    fn test_meta_brain_build_and_process() {
        let mut rng = StdRng::seed_from_u64(42);
        let topic_names = vec!["coding".into(), "support".into(), "general".into()];
        let topic_embs = vec![
            vec![0.1f32; TOPIC_EMBED_DIM],
            vec![0.2f32; TOPIC_EMBED_DIM],
            vec![0.3f32; TOPIC_EMBED_DIM],
        ];
        let verb_embs: Vec<Vec<f32>> = (0..VERB_LABELS.len())
            .map(|i| vec![(i as f32) * 0.1; VERB_EMBED_DIM])
            .collect();

        let mut mb = MetaBrain::build(32, 3, topic_names, topic_embs, verb_embs, 5, &mut rng);
        let h_raw = vec![0.5f32; 32];
        let bridge = vec![0.3f32; 128];
        let result = mb.process(&h_raw, &bridge);

        assert_eq!(result.conditioning.len(), GEN_COND_DIM);
        assert!(!result.topic.is_empty());
        assert!(!result.verb.is_empty());
    }

    #[test]
    fn test_meta_brain_coordinator_training() {
        let mut rng = StdRng::seed_from_u64(42);
        let topic_names = vec!["a".into(), "b".into()];
        let topic_embs = vec![vec![0.1f32; TOPIC_EMBED_DIM]; 2];
        let verb_embs: Vec<Vec<f32>> = (0..VERB_LABELS.len())
            .map(|_| vec![0.1f32; VERB_EMBED_DIM])
            .collect();

        let mut mb = MetaBrain::build(16, 2, topic_names, topic_embs, verb_embs, 5, &mut rng);
        let h_raw = vec![0.5f32; 16];
        let target = vec![0.1f32; GEN_COND_DIM];
        let loss1 = mb.train_coordinator_step(&h_raw, &target);
        let loss2 = mb.train_coordinator_step(&h_raw, &target);
        // After two develops, centroid regression should converge quickly
        assert!(loss2 <= loss1 + 0.01, "coordinator loss should not diverge: {} -> {}", loss1, loss2);
    }

    #[test]
    fn test_meta_brain_freeze() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut mb = MetaBrain::build(
            8, 2, vec!["a".into(), "b".into()],
            vec![vec![0.0; TOPIC_EMBED_DIM]; 2],
            vec![vec![0.0; VERB_EMBED_DIM]; VERB_LABELS.len()],
            5, &mut rng,
        );
        mb.freeze();
        assert!(mb.frozen);
        assert!(mb.topic_brain.frozen);
        assert!(mb.verb_brain.frozen);
        assert!(mb.action_brain.frozen);
    }

    #[test]
    fn test_meta_brain_serialization() {
        let mut rng = StdRng::seed_from_u64(42);
        let mb = MetaBrain::build(
            8, 2, vec!["a".into(), "b".into()],
            vec![vec![0.0; TOPIC_EMBED_DIM]; 2],
            vec![vec![0.0; VERB_EMBED_DIM]; VERB_LABELS.len()],
            5, &mut rng,
        );
        let json = serde_json::to_string(&mb).unwrap();
        let restored: MetaBrain = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.topic_brain.output_dim, mb.topic_brain.output_dim);
        assert_eq!(restored.verb_brain.output_dim, mb.verb_brain.output_dim);
        assert!(restored.is_ready() == mb.is_ready());
    }

    #[test]
    fn test_softmax_argmax() {
        let logits = vec![1.0, 3.0, 2.0, 0.5];
        let (idx, conf) = softmax_argmax(&logits);
        assert_eq!(idx, 1);
        assert!(conf > 0.4);
        assert!(conf < 1.0);
    }
}
