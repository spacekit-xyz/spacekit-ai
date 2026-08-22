//! Understanding Layer — semantic intent and action-verb conditioning.
//!
//! Two classifiers, frozen after training like Main Dimension:
//!   1. Topic classifier: raw 768d → 72 topic intents (32d embedding each)
//!   2. Verb classifier:  raw 768d → 8 action-verbs   (16d embedding each)
//!
//! The combined 48d understanding vector is concatenated with the 128d bridge
//! vector to form a 176d (padded to 192d) conditioning signal that tells the
//! generation head both WHAT the topic is and WHAT TO DO with it.
//!
//! With Cl(1,7) SpaceTime Algebra, the verb embedding also produces a
//! **goal magnitude** scalar that feeds the timelike (e_0) dimension of the
//! Clifford embedding, giving the system causal/sequential direction.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const TOPIC_EMBED_DIM: usize = 32;
pub const VERB_EMBED_DIM: usize = 16;
pub const UNDERSTANDING_DIM: usize = TOPIC_EMBED_DIM + VERB_EMBED_DIM; // 48

pub const VERB_LABELS: &[&str] = &[
    "explain",
    "design",
    "implement",
    "debug",
    "optimize",
    "test",
    "refactor",
    "compare",
];

pub fn intent_to_verb(intent: &str) -> &'static str {
    let s = intent.to_ascii_lowercase();
    if s.contains("debug") {
        return "debug";
    }
    if s.contains("test") {
        return "test";
    }
    if s.contains("refactor") {
        return "refactor";
    }
    if s.contains("optim") {
        return "optimize";
    }
    if s.contains("implement") || s.contains("coding_impl") {
        return "implement";
    }
    if s.contains("design") || s.contains("architect") || s.contains("compos") {
        return "design";
    }
    if s.contains("compare") || s.contains("vs") {
        return "compare";
    }
    "explain"
}

/// Paramecium-based classifier for understanding layer.
/// Replaces the SmallMlp — one-pass `develop()` instead of iterative backprop.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SmallMlp {
    pub lattice: crate::dimension::paramecium::InfraciliaryLattice,
    pub labels: Vec<String>,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
}

impl SmallMlp {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, _seed: u64) -> Self {
        let labels: Vec<String> = (0..output_dim).map(|i| format!("cls_{}", i)).collect();
        let label_strs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
        let dict = crate::spectral::TokenDictionary::build(&label_strs, 64);
        let lattice = crate::dimension::paramecium::InfraciliaryLattice::new(dict);
        Self {
            lattice,
            labels,
            input_dim,
            hidden_dim,
            output_dim,
        }
    }

    /// Build from labeled data in one pass.
    pub fn build(input_dim: usize, output_dim: usize, samples: &[(&[f32], usize)]) -> Self {
        let labels: Vec<String> = (0..output_dim).map(|i| format!("cls_{}", i)).collect();
        let label_strs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
        let dict = crate::spectral::TokenDictionary::build(&label_strs, 64);
        let pairs: Vec<(Vec<f32>, String)> = samples
            .iter()
            .map(|(emb, idx)| (emb.to_vec(), format!("cls_{}", idx)))
            .collect();
        let mut lattice = crate::dimension::paramecium::InfraciliaryLattice::new(dict);
        lattice.develop(&pairs, 0.90);
        Self {
            lattice,
            labels,
            input_dim,
            hidden_dim: 0,
            output_dim,
        }
    }

    pub fn predict(&self, input: &[f32]) -> usize {
        self.predict_shared(input)
    }

    pub fn predict_with_confidence(&self, input: &[f32]) -> (usize, f32) {
        self.predict_with_confidence_shared(input)
    }

    /// Immutable prediction via direct cosine similarity (no EMA drift).
    fn predict_shared(&self, input: &[f32]) -> usize {
        if self.lattice.programs.is_empty() {
            return 0;
        }
        let mut best_idx = 0;
        let mut best_sim = f32::NEG_INFINITY;
        for (i, prog) in self.lattice.programs.iter().enumerate() {
            let sim = cosine_sim(input, &prog.ema_centroid);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }
        let text = self.lattice.programs[best_idx].display_text(&self.lattice.dictionary);
        Self::parse_cls(&text).unwrap_or(0)
    }

    fn predict_with_confidence_shared(&self, input: &[f32]) -> (usize, f32) {
        if self.lattice.programs.is_empty() {
            return (0, 0.0);
        }
        let mut best_idx = 0;
        let mut best_sim = f32::NEG_INFINITY;
        for (i, prog) in self.lattice.programs.iter().enumerate() {
            let sim = cosine_sim(input, &prog.ema_centroid);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }
        let text = self.lattice.programs[best_idx].display_text(&self.lattice.dictionary);
        let idx = Self::parse_cls(&text).unwrap_or(0);
        (idx, best_sim.max(0.0))
    }

    pub fn train_step(&mut self, input: &[f32], target_idx: usize, _lr: f32) -> f32 {
        let label = format!("cls_{}", target_idx);
        let pairs = vec![(input.to_vec(), label)];
        self.lattice.develop(&pairs, 0.90);
        let predicted = self.predict(input);
        if predicted == target_idx {
            0.0
        } else {
            1.0
        }
    }

    fn parse_cls(text: &str) -> Option<usize> {
        text.strip_prefix("cls_").and_then(|s| s.parse().ok())
    }
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
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
    if denom < 1e-20 {
        0.0
    } else {
        (dot / denom) as f32
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnderstandingLayer {
    pub topic_names: Vec<String>,
    pub topic_embeddings: Vec<Vec<f32>>,
    pub topic_classifier: SmallMlp,

    pub verb_names: Vec<String>,
    pub verb_embeddings: Vec<Vec<f32>>,
    pub verb_classifier: SmallMlp,

    pub frozen: bool,
}

impl Default for UnderstandingLayer {
    fn default() -> Self {
        Self {
            topic_names: Vec::new(),
            topic_embeddings: Vec::new(),
            topic_classifier: SmallMlp::new(1, 1, 1, 0),
            verb_names: Vec::new(),
            verb_embeddings: Vec::new(),
            verb_classifier: SmallMlp::new(1, 1, 1, 0),
            frozen: false,
        }
    }
}

impl UnderstandingLayer {
    /// Build from training data: (raw_embedding, semantic_intent) pairs.
    pub fn build(samples: &[(&[f32], &str)], raw_dim: usize) -> Self {
        // Build topic vocabulary
        let mut topic_set: Vec<String> = Vec::new();
        let mut topic_map: HashMap<String, usize> = HashMap::new();
        for &(_, intent) in samples {
            if !topic_map.contains_key(intent) {
                topic_map.insert(intent.to_string(), topic_set.len());
                topic_set.push(intent.to_string());
            }
        }
        let num_topics = topic_set.len().max(2);

        // Build verb vocabulary
        let verb_names: Vec<String> = VERB_LABELS.iter().map(|s| s.to_string()).collect();
        let verb_map: HashMap<String, usize> = verb_names
            .iter()
            .enumerate()
            .map(|(i, v)| (v.clone(), i))
            .collect();
        let num_verbs = verb_names.len();

        // Build topic classifier — one-pass Paramecium develop
        let topic_samples: Vec<(&[f32], usize)> = samples
            .iter()
            .filter_map(|&(raw, intent)| topic_map.get(intent).map(|&idx| (raw, idx)))
            .collect();
        let topic_clf = SmallMlp::build(raw_dim, num_topics, &topic_samples);
        println!(
            "    [understanding/topic] Paramecium one-pass: {} programs from {} samples",
            topic_clf.lattice.program_count(),
            topic_samples.len()
        );

        // Build verb classifier — one-pass Paramecium develop
        let verb_samples: Vec<(&[f32], usize)> = samples
            .iter()
            .map(|&(raw, intent)| {
                let verb = intent_to_verb(intent);
                let idx = verb_map.get(verb).copied().unwrap_or(0);
                (raw, idx)
            })
            .collect();
        let verb_clf = SmallMlp::build(raw_dim, num_verbs, &verb_samples);
        println!(
            "    [understanding/verb] Paramecium one-pass: {} programs from {} samples",
            verb_clf.lattice.program_count(),
            verb_samples.len()
        );

        // Compute topic embeddings: mean of raw embeddings per topic
        let mut topic_sums: Vec<Vec<f32>> = vec![vec![0.0f32; TOPIC_EMBED_DIM]; num_topics];
        let mut topic_counts: Vec<usize> = vec![0; num_topics];
        for &(raw, intent) in samples {
            if let Some(&idx) = topic_map.get(intent) {
                for (j, s) in topic_sums[idx].iter_mut().enumerate() {
                    *s += raw.get(j).copied().unwrap_or(0.0);
                }
                topic_counts[idx] += 1;
            }
        }
        let topic_embeddings: Vec<Vec<f32>> = topic_sums
            .iter()
            .zip(topic_counts.iter())
            .map(|(sum, &count)| {
                let n = count.max(1) as f32;
                let mut emb: Vec<f32> = sum.iter().map(|s| s / n).collect();
                let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
                for x in emb.iter_mut() {
                    *x /= norm;
                }
                emb
            })
            .collect();

        // Compute verb embeddings: mean of raw embeddings per verb
        let mut verb_sums: Vec<Vec<f32>> = vec![vec![0.0f32; VERB_EMBED_DIM]; num_verbs];
        let mut verb_counts: Vec<usize> = vec![0; num_verbs];
        for &(raw, intent) in samples {
            let verb = intent_to_verb(intent);
            if let Some(&idx) = verb_map.get(verb) {
                for (j, s) in verb_sums[idx].iter_mut().enumerate() {
                    *s += raw.get(j).copied().unwrap_or(0.0);
                }
                verb_counts[idx] += 1;
            }
        }
        let verb_embeddings: Vec<Vec<f32>> = verb_sums
            .iter()
            .zip(verb_counts.iter())
            .map(|(sum, &count)| {
                let n = count.max(1) as f32;
                let mut emb: Vec<f32> = sum.iter().map(|s| s / n).collect();
                let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
                for x in emb.iter_mut() {
                    *x /= norm;
                }
                emb
            })
            .collect();

        // Evaluate
        let mut topic_correct = 0usize;
        let mut verb_correct = 0usize;
        for &(raw, intent) in samples {
            let pred_topic = topic_clf.predict(raw);
            if let Some(&true_idx) = topic_map.get(intent) {
                if pred_topic == true_idx {
                    topic_correct += 1;
                }
            }
            let pred_verb = verb_clf.predict(raw);
            let true_verb = intent_to_verb(intent);
            if let Some(&true_idx) = verb_map.get(true_verb) {
                if pred_verb == true_idx {
                    verb_correct += 1;
                }
            }
        }
        let n = samples.len().max(1);
        println!(
            "    [understanding] topic accuracy: {:.1}% ({}/{}), verb accuracy: {:.1}% ({}/{})",
            topic_correct as f32 / n as f32 * 100.0,
            topic_correct,
            n,
            verb_correct as f32 / n as f32 * 100.0,
            verb_correct,
            n
        );

        Self {
            topic_names: topic_set,
            topic_embeddings,
            topic_classifier: topic_clf,
            verb_names,
            verb_embeddings,
            verb_classifier: verb_clf,
            frozen: false,
        }
    }

    /// Build with MicroBrain classifiers — now delegates to Paramecium-based `build()`.
    /// MicroBrain (NeuralEnvironment) is no longer used; Paramecium handles everything.
    pub fn build_with_micro_brains(samples: &[(&[f32], &str)], raw_dim: usize) -> Self {
        Self::build(samples, raw_dim)
    }

    /// Classify input and return (topic_embedding, verb_embedding, topic_name, verb_name).
    pub fn classify(&self, raw: &[f32]) -> (Vec<f32>, Vec<f32>, String, String) {
        let (topic_idx, _) = self.topic_classifier.predict_with_confidence(raw);
        let (verb_idx, _) = self.verb_classifier.predict_with_confidence(raw);

        let topic_emb = self
            .topic_embeddings
            .get(topic_idx)
            .cloned()
            .unwrap_or_else(|| vec![0.0f32; TOPIC_EMBED_DIM]);
        let verb_emb = self
            .verb_embeddings
            .get(verb_idx)
            .cloned()
            .unwrap_or_else(|| vec![0.0f32; VERB_EMBED_DIM]);

        let topic_name = self
            .topic_names
            .get(topic_idx)
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());
        let verb_name = self
            .verb_names
            .get(verb_idx)
            .cloned()
            .unwrap_or_else(|| "explain".to_string());

        (topic_emb, verb_emb, topic_name, verb_name)
    }

    /// Get the combined understanding vector (48d) for conditioning.
    pub fn conditioning_vector(&self, raw: &[f32]) -> Vec<f32> {
        let (topic_emb, verb_emb, _, _) = self.classify(raw);
        let mut out = Vec::with_capacity(UNDERSTANDING_DIM);
        out.extend_from_slice(&topic_emb);
        out.extend_from_slice(&verb_emb);
        out
    }

    /// Immutable version using SmallMlp fallback only (safe for shared references).
    pub fn conditioning_vector_shared(&self, raw: &[f32]) -> Vec<f32> {
        let (topic_idx, _) = self.topic_classifier.predict_with_confidence(raw);
        let (verb_idx, _) = self.verb_classifier.predict_with_confidence(raw);
        let topic_emb = self
            .topic_embeddings
            .get(topic_idx)
            .cloned()
            .unwrap_or_else(|| vec![0.0f32; TOPIC_EMBED_DIM]);
        let verb_emb = self
            .verb_embeddings
            .get(verb_idx)
            .cloned()
            .unwrap_or_else(|| vec![0.0f32; VERB_EMBED_DIM]);
        let mut out = Vec::with_capacity(UNDERSTANDING_DIM);
        out.extend_from_slice(&topic_emb);
        out.extend_from_slice(&verb_emb);
        out
    }

    /// Compute the goal magnitude for the timelike (e_0) dimension.
    /// Action verbs that imply directed change (implement, debug, optimize, refactor)
    /// produce higher magnitudes than passive verbs (explain, compare).
    /// This feeds the timelike axis of the Cl(1,7) embedding.
    pub fn goal_magnitude(&self, raw: &[f32]) -> f32 {
        let (verb_idx, verb_conf) = self.verb_classifier.predict_with_confidence(raw);
        let verb_name = self
            .verb_names
            .get(verb_idx)
            .map(|s| s.as_str())
            .unwrap_or("explain");
        let directedness = match verb_name {
            "implement" => 1.0,
            "debug" => 0.9,
            "optimize" => 0.85,
            "refactor" => 0.8,
            "test" => 0.7,
            "design" => 0.6,
            "compare" => 0.3,
            "explain" => 0.2,
            _ => 0.1,
        };
        directedness * verb_conf.max(0.0).min(1.0)
    }

    pub fn freeze(&mut self) {
        self.frozen = true;
    }

    pub fn is_empty(&self) -> bool {
        self.topic_names.is_empty()
    }

    pub fn topic_count(&self) -> usize {
        self.topic_names.len()
    }

    pub fn verb_count(&self) -> usize {
        self.verb_names.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_samples() -> (Vec<Vec<f32>>, Vec<&'static str>) {
        let intents = vec![
            "coding_implementation",
            "coding_debug",
            "coding_optimize",
            "coding_testing",
            "coding_refactor",
            "behavioral",
            "microservices",
            "physics",
            "advice",
            "billing_issue",
        ];
        let mut raw_vecs = Vec::new();
        for (i, _) in intents.iter().enumerate() {
            let mut v = vec![0.0f32; 64];
            for j in 0..64 {
                v[j] = ((i * 7 + j) as f32 * 0.1).sin();
            }
            raw_vecs.push(v);
        }
        (raw_vecs, intents)
    }

    #[test]
    fn test_build_understanding_layer() {
        let (raw_vecs, intents) = make_samples();
        let samples: Vec<(&[f32], &str)> = raw_vecs
            .iter()
            .zip(intents.iter())
            .map(|(r, &i)| (r.as_slice(), i))
            .collect();
        let layer = UnderstandingLayer::build(&samples, 64);
        assert_eq!(layer.topic_count(), 10);
        assert_eq!(layer.verb_count(), 8);
        assert!(!layer.is_empty());
    }

    #[test]
    fn test_classify_returns_correct_dims() {
        let (raw_vecs, intents) = make_samples();
        let samples: Vec<(&[f32], &str)> = raw_vecs
            .iter()
            .zip(intents.iter())
            .map(|(r, &i)| (r.as_slice(), i))
            .collect();
        let mut layer = UnderstandingLayer::build(&samples, 64);
        let (topic_emb, verb_emb, topic_name, verb_name) = layer.classify(&raw_vecs[0]);
        assert_eq!(topic_emb.len(), TOPIC_EMBED_DIM);
        assert_eq!(verb_emb.len(), VERB_EMBED_DIM);
        assert!(!topic_name.is_empty());
        assert!(VERB_LABELS.contains(&verb_name.as_str()));
    }

    #[test]
    fn test_conditioning_vector_dim() {
        let (raw_vecs, intents) = make_samples();
        let samples: Vec<(&[f32], &str)> = raw_vecs
            .iter()
            .zip(intents.iter())
            .map(|(r, &i)| (r.as_slice(), i))
            .collect();
        let mut layer = UnderstandingLayer::build(&samples, 64);
        let cond = layer.conditioning_vector(&raw_vecs[0]);
        assert_eq!(cond.len(), UNDERSTANDING_DIM);
    }

    #[test]
    fn test_freeze_layer() {
        let (raw_vecs, intents) = make_samples();
        let samples: Vec<(&[f32], &str)> = raw_vecs
            .iter()
            .zip(intents.iter())
            .map(|(r, &i)| (r.as_slice(), i))
            .collect();
        let mut layer = UnderstandingLayer::build(&samples, 64);
        assert!(!layer.frozen);
        layer.freeze();
        assert!(layer.frozen);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let (raw_vecs, intents) = make_samples();
        let samples: Vec<(&[f32], &str)> = raw_vecs
            .iter()
            .zip(intents.iter())
            .map(|(r, &i)| (r.as_slice(), i))
            .collect();
        let mut layer = UnderstandingLayer::build(&samples, 64);
        let json = serde_json::to_string(&layer).unwrap();
        let mut restored: UnderstandingLayer = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.topic_count(), layer.topic_count());
        assert_eq!(restored.verb_count(), layer.verb_count());
        let (t1, v1, _, _) = layer.classify(&raw_vecs[0]);
        let (t2, v2, _, _) = restored.classify(&raw_vecs[0]);
        assert_eq!(t1, t2);
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_intent_to_verb_mapping() {
        assert_eq!(intent_to_verb("coding_implementation"), "implement");
        assert_eq!(intent_to_verb("coding_debug"), "debug");
        assert_eq!(intent_to_verb("coding_testing"), "test");
        assert_eq!(intent_to_verb("coding_refactor"), "refactor");
        assert_eq!(intent_to_verb("coding_optimize"), "optimize");
        assert_eq!(intent_to_verb("microservices"), "explain");
        assert_eq!(intent_to_verb("architectural_composition"), "design");
        assert_eq!(intent_to_verb("physics"), "explain");
    }

    #[test]
    fn test_small_mlp_predict_shape() {
        let mlp = SmallMlp::new(64, 32, 10, 42);
        let input = vec![0.1f32; 64];
        let (idx, conf) = mlp.predict_with_confidence(&input);
        assert!(idx < 10);
        let _ = conf;
    }

    #[test]
    fn test_small_mlp_build_classifies() {
        let input_a: Vec<f32> = (0..16).map(|i| if i < 8 { 1.0 } else { 0.0 }).collect();
        let input_b: Vec<f32> = (0..16).map(|i| if i >= 8 { 1.0 } else { 0.0 }).collect();
        let samples: Vec<(&[f32], usize)> = vec![(input_a.as_slice(), 0), (input_b.as_slice(), 1)];
        let mlp = SmallMlp::build(16, 4, &samples);
        assert_eq!(mlp.predict(&input_a), 0);
        assert_eq!(mlp.predict(&input_b), 1);
    }

    #[test]
    fn test_different_intents_get_different_embeddings() {
        let (raw_vecs, intents) = make_samples();
        let samples: Vec<(&[f32], &str)> = raw_vecs
            .iter()
            .zip(intents.iter())
            .map(|(r, &i)| (r.as_slice(), i))
            .collect();
        let mut layer = UnderstandingLayer::build(&samples, 64);
        let c0 = layer.conditioning_vector(&raw_vecs[0]); // coding_implementation
        let c1 = layer.conditioning_vector(&raw_vecs[1]); // coding_debug
        let diff: f32 = c0.iter().zip(c1.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 0.01,
            "different intents should produce different conditioning: diff={}",
            diff
        );
    }
}
