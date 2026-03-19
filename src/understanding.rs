//! Understanding Layer — semantic intent and action-verb conditioning.
//!
//! Two classifiers, frozen after training like Main Dimension:
//!   1. Topic classifier: raw 768d → 72 topic intents (32d embedding each)
//!   2. Verb classifier:  raw 768d → 8 action-verbs   (16d embedding each)
//!
//! The combined 48d understanding vector is concatenated with the 128d bridge
//! vector to form a 176d (padded to 192d) conditioning signal that tells the
//! generation head both WHAT the topic is and WHAT TO DO with it.

use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::micro_brain::{MicroBrain, MicroBrainRole};

pub const TOPIC_EMBED_DIM: usize = 32;
pub const VERB_EMBED_DIM: usize = 16;
pub const UNDERSTANDING_DIM: usize = TOPIC_EMBED_DIM + VERB_EMBED_DIM; // 48

pub const VERB_LABELS: &[&str] = &[
    "explain", "design", "implement", "debug", "optimize", "test", "refactor", "compare",
];

pub fn intent_to_verb(intent: &str) -> &'static str {
    let s = intent.to_ascii_lowercase();
    if s.contains("debug") { return "debug"; }
    if s.contains("test") { return "test"; }
    if s.contains("refactor") { return "refactor"; }
    if s.contains("optim") { return "optimize"; }
    if s.contains("implement") || s.contains("coding_impl") { return "implement"; }
    if s.contains("design") || s.contains("architect") || s.contains("compos") { return "design"; }
    if s.contains("compare") || s.contains("vs") { return "compare"; }
    "explain"
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SmallMlp {
    pub w1: Vec<Vec<f32>>,
    pub b1: Vec<f32>,
    pub w2: Vec<Vec<f32>>,
    pub b2: Vec<f32>,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
}

impl SmallMlp {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, seed: u64) -> Self {
        let mut rng_state = seed;
        let mut rand_f32 = || -> f32 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bits = ((rng_state >> 33) as u32) as f32 / u32::MAX as f32;
            (bits - 0.5) * 2.0 * (6.0 / (input_dim + hidden_dim) as f32).sqrt()
        };

        let w1: Vec<Vec<f32>> = (0..hidden_dim)
            .map(|_| (0..input_dim).map(|_| rand_f32()).collect())
            .collect();
        let b1 = vec![0.0f32; hidden_dim];
        let w2: Vec<Vec<f32>> = (0..output_dim)
            .map(|_| (0..hidden_dim).map(|_| rand_f32()).collect())
            .collect();
        let b2 = vec![0.0f32; output_dim];
        Self { w1, b1, w2, b2, input_dim, hidden_dim, output_dim }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut hidden = self.b1.clone();
        for (h, row) in hidden.iter_mut().zip(self.w1.iter()) {
            for (w, x) in row.iter().zip(input.iter()) {
                *h += w * x;
            }
            if *h < 0.0 { *h *= 0.01; } // LeakyReLU
        }
        let mut output = self.b2.clone();
        for (o, row) in output.iter_mut().zip(self.w2.iter()) {
            for (w, h) in row.iter().zip(hidden.iter()) {
                *o += w * h;
            }
        }
        output
    }

    pub fn predict(&self, input: &[f32]) -> usize {
        let logits = self.forward(input);
        logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    pub fn predict_with_confidence(&self, input: &[f32]) -> (usize, f32) {
        let logits = self.forward(input);
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|l| (l - max_l).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();
        let (idx, &conf) = probs.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));
        (idx, conf)
    }

    pub fn train_step(&mut self, input: &[f32], target_idx: usize, lr: f32) -> f32 {
        let logits = self.forward(input);
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|l| (l - max_l).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();
        let loss = -(probs[target_idx].max(1e-10)).ln();

        // dL/d_logits = probs - one_hot(target)
        let mut d_logits = probs.clone();
        d_logits[target_idx] -= 1.0;

        // Hidden activations (recompute)
        let mut hidden = self.b1.clone();
        for (h, row) in hidden.iter_mut().zip(self.w1.iter()) {
            for (w, x) in row.iter().zip(input.iter()) {
                *h += w * x;
            }
            if *h < 0.0 { *h *= 0.01; }
        }
        let mut hidden_pre = self.b1.clone();
        for (h, row) in hidden_pre.iter_mut().zip(self.w1.iter()) {
            for (w, x) in row.iter().zip(input.iter()) {
                *h += w * x;
            }
        }

        // Backprop through w2, b2
        let mut d_hidden = vec![0.0f32; self.hidden_dim];
        for (o_idx, dl) in d_logits.iter().enumerate() {
            self.b2[o_idx] -= lr * dl;
            for (h_idx, h_val) in hidden.iter().enumerate() {
                self.w2[o_idx][h_idx] -= lr * dl * h_val;
                d_hidden[h_idx] += self.w2[o_idx][h_idx] * dl;
            }
        }

        // Backprop through LeakyReLU + w1, b1
        for (h_idx, dh) in d_hidden.iter().enumerate() {
            let gate = if hidden_pre[h_idx] >= 0.0 { 1.0 } else { 0.01 };
            let grad = dh * gate;
            self.b1[h_idx] -= lr * grad;
            for (i_idx, x) in input.iter().enumerate() {
                self.w1[h_idx][i_idx] -= lr * grad * x;
            }
        }

        loss
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

    /// MicroBrain replacements — used when present, SmallMlp used as fallback.
    #[serde(default)]
    pub topic_brain: Option<MicroBrain>,
    #[serde(default)]
    pub verb_brain: Option<MicroBrain>,

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
            topic_brain: None,
            verb_brain: None,
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
        let verb_map: HashMap<String, usize> = verb_names.iter().enumerate()
            .map(|(i, v)| (v.clone(), i))
            .collect();
        let num_verbs = verb_names.len();

        // Initialize classifiers
        let mut topic_clf = SmallMlp::new(raw_dim, 64, num_topics, 42);
        let mut verb_clf = SmallMlp::new(raw_dim, 32, num_verbs, 137);

        // Train topic classifier
        let topic_epochs = 400;
        let topic_lr = 0.01;
        for epoch in 0..topic_epochs {
            let lr = topic_lr * (1.0 - epoch as f32 / topic_epochs as f32 * 0.5);
            let mut total_loss = 0.0f32;
            for &(raw, intent) in samples {
                if let Some(&idx) = topic_map.get(intent) {
                    total_loss += topic_clf.train_step(raw, idx, lr);
                }
            }
            if epoch % 100 == 0 || epoch == topic_epochs - 1 {
                println!("    [understanding/topic] epoch {}/{} loss={:.4}",
                    epoch, topic_epochs, total_loss / samples.len().max(1) as f32);
            }
        }

        // Train verb classifier
        let verb_epochs = 300;
        let verb_lr = 0.015;
        for epoch in 0..verb_epochs {
            let lr = verb_lr * (1.0 - epoch as f32 / verb_epochs as f32 * 0.5);
            let mut total_loss = 0.0f32;
            for &(raw, intent) in samples {
                let verb = intent_to_verb(intent);
                if let Some(&idx) = verb_map.get(verb) {
                    total_loss += verb_clf.train_step(raw, idx, lr);
                }
            }
            if epoch % 100 == 0 || epoch == verb_epochs - 1 {
                println!("    [understanding/verb] epoch {}/{} loss={:.4}",
                    epoch, verb_epochs, total_loss / samples.len().max(1) as f32);
            }
        }

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
        let topic_embeddings: Vec<Vec<f32>> = topic_sums.iter().zip(topic_counts.iter())
            .map(|(sum, &count)| {
                let n = count.max(1) as f32;
                let mut emb: Vec<f32> = sum.iter().map(|s| s / n).collect();
                let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
                for x in emb.iter_mut() { *x /= norm; }
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
        let verb_embeddings: Vec<Vec<f32>> = verb_sums.iter().zip(verb_counts.iter())
            .map(|(sum, &count)| {
                let n = count.max(1) as f32;
                let mut emb: Vec<f32> = sum.iter().map(|s| s / n).collect();
                let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
                for x in emb.iter_mut() { *x /= norm; }
                emb
            })
            .collect();

        // Evaluate
        let mut topic_correct = 0usize;
        let mut verb_correct = 0usize;
        for &(raw, intent) in samples {
            let pred_topic = topic_clf.predict(raw);
            if let Some(&true_idx) = topic_map.get(intent) {
                if pred_topic == true_idx { topic_correct += 1; }
            }
            let pred_verb = verb_clf.predict(raw);
            let true_verb = intent_to_verb(intent);
            if let Some(&true_idx) = verb_map.get(true_verb) {
                if pred_verb == true_idx { verb_correct += 1; }
            }
        }
        let n = samples.len().max(1);
        println!("    [understanding] topic accuracy: {:.1}% ({}/{}), verb accuracy: {:.1}% ({}/{})",
            topic_correct as f32 / n as f32 * 100.0, topic_correct, n,
            verb_correct as f32 / n as f32 * 100.0, verb_correct, n);

        Self {
            topic_names: topic_set,
            topic_embeddings,
            topic_classifier: topic_clf,
            verb_names,
            verb_embeddings,
            verb_classifier: verb_clf,
            topic_brain: None,
            verb_brain: None,
            frozen: false,
        }
    }

    /// Build with MicroBrain classifiers (NeuralEnvironment) instead of SmallMlp.
    /// Falls back to SmallMlp training first, then replaces with MicroBrain.
    pub fn build_with_micro_brains(samples: &[(&[f32], &str)], raw_dim: usize) -> Self {
        let mut layer = Self::build(samples, raw_dim);

        let mut rng = StdRng::seed_from_u64(42);
        let num_topics = layer.topic_names.len().max(2);
        let num_verbs = layer.verb_names.len();

        let mut topic_brain = MicroBrain::new(
            MicroBrainRole::Topic, raw_dim, num_topics, 64,
            layer.topic_names.clone(), &mut rng,
        );
        let mut verb_brain = MicroBrain::new(
            MicroBrainRole::Verb, raw_dim, num_verbs, 32,
            layer.verb_names.clone(), &mut rng,
        );

        let mut topic_map: HashMap<String, usize> = HashMap::new();
        for (i, name) in layer.topic_names.iter().enumerate() {
            topic_map.insert(name.clone(), i);
        }
        let verb_map: HashMap<String, usize> = layer.verb_names.iter().enumerate()
            .map(|(i, v)| (v.clone(), i)).collect();

        let topic_epochs = 400;
        for epoch in 0..topic_epochs {
            let mut total_loss = 0.0f32;
            for &(raw, intent) in samples {
                if let Some(&idx) = topic_map.get(intent) {
                    total_loss += topic_brain.train_step(raw, idx, &mut rng);
                }
            }
            if epoch % 100 == 0 || epoch == topic_epochs - 1 {
                println!("    [understanding/topic-brain] epoch {}/{} loss={:.4}",
                    epoch, topic_epochs, total_loss / samples.len().max(1) as f32);
            }
        }

        let verb_epochs = 300;
        for epoch in 0..verb_epochs {
            let mut total_loss = 0.0f32;
            for &(raw, intent) in samples {
                let verb = intent_to_verb(intent);
                if let Some(&idx) = verb_map.get(verb) {
                    total_loss += verb_brain.train_step(raw, idx, &mut rng);
                }
            }
            if epoch % 100 == 0 || epoch == verb_epochs - 1 {
                println!("    [understanding/verb-brain] epoch {}/{} loss={:.4}",
                    epoch, verb_epochs, total_loss / samples.len().max(1) as f32);
            }
        }

        // Evaluate MicroBrain accuracy
        let mut topic_correct = 0usize;
        let mut verb_correct = 0usize;
        for &(raw, intent) in samples {
            let (pred_topic, _, _) = topic_brain.predict(raw);
            if let Some(&true_idx) = topic_map.get(intent) {
                if pred_topic == true_idx { topic_correct += 1; }
            }
            let (pred_verb, _, _) = verb_brain.predict(raw);
            let true_verb = intent_to_verb(intent);
            if let Some(&true_idx) = verb_map.get(true_verb) {
                if pred_verb == true_idx { verb_correct += 1; }
            }
        }
        let n = samples.len().max(1);
        println!("    [understanding/brain] topic accuracy: {:.1}% ({}/{}), verb accuracy: {:.1}% ({}/{})",
            topic_correct as f32 / n as f32 * 100.0, topic_correct, n,
            verb_correct as f32 / n as f32 * 100.0, verb_correct, n);

        layer.topic_brain = Some(topic_brain);
        layer.verb_brain = Some(verb_brain);
        layer
    }

    /// Classify input and return (topic_embedding, verb_embedding, topic_name, verb_name).
    /// Prefers MicroBrain classifiers when available, falls back to SmallMlp.
    pub fn classify(&mut self, raw: &[f32]) -> (Vec<f32>, Vec<f32>, String, String) {
        let topic_idx = if let Some(ref mut brain) = self.topic_brain {
            let (idx, _, _) = brain.predict(raw);
            idx
        } else {
            let (idx, _) = self.topic_classifier.predict_with_confidence(raw);
            idx
        };
        let verb_idx = if let Some(ref mut brain) = self.verb_brain {
            let (idx, _, _) = brain.predict(raw);
            idx
        } else {
            let (idx, _) = self.verb_classifier.predict_with_confidence(raw);
            idx
        };

        let topic_emb = self.topic_embeddings.get(topic_idx)
            .cloned()
            .unwrap_or_else(|| vec![0.0f32; TOPIC_EMBED_DIM]);
        let verb_emb = self.verb_embeddings.get(verb_idx)
            .cloned()
            .unwrap_or_else(|| vec![0.0f32; VERB_EMBED_DIM]);

        let topic_name = self.topic_names.get(topic_idx)
            .cloned().unwrap_or_else(|| "unknown".to_string());
        let verb_name = self.verb_names.get(verb_idx)
            .cloned().unwrap_or_else(|| "explain".to_string());

        (topic_emb, verb_emb, topic_name, verb_name)
    }

    /// Get the combined understanding vector (48d) for conditioning.
    pub fn conditioning_vector(&mut self, raw: &[f32]) -> Vec<f32> {
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
        let topic_emb = self.topic_embeddings.get(topic_idx)
            .cloned().unwrap_or_else(|| vec![0.0f32; TOPIC_EMBED_DIM]);
        let verb_emb = self.verb_embeddings.get(verb_idx)
            .cloned().unwrap_or_else(|| vec![0.0f32; VERB_EMBED_DIM]);
        let mut out = Vec::with_capacity(UNDERSTANDING_DIM);
        out.extend_from_slice(&topic_emb);
        out.extend_from_slice(&verb_emb);
        out
    }

    pub fn freeze(&mut self) {
        self.frozen = true;
        if let Some(ref mut b) = self.topic_brain { b.freeze(); }
        if let Some(ref mut b) = self.verb_brain { b.freeze(); }
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
            "coding_implementation", "coding_debug", "coding_optimize",
            "coding_testing", "coding_refactor", "behavioral", "microservices",
            "physics", "advice", "billing_issue",
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
        let samples: Vec<(&[f32], &str)> = raw_vecs.iter()
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
        let samples: Vec<(&[f32], &str)> = raw_vecs.iter()
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
        let samples: Vec<(&[f32], &str)> = raw_vecs.iter()
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
        let samples: Vec<(&[f32], &str)> = raw_vecs.iter()
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
        let samples: Vec<(&[f32], &str)> = raw_vecs.iter()
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
    fn test_small_mlp_forward_shape() {
        let mlp = SmallMlp::new(64, 32, 10, 42);
        let input = vec![0.1f32; 64];
        let output = mlp.forward(&input);
        assert_eq!(output.len(), 10);
    }

    #[test]
    fn test_small_mlp_training_reduces_loss() {
        let mut mlp = SmallMlp::new(16, 8, 4, 42);
        let input: Vec<f32> = (0..16).map(|i| (i as f32 * 0.1).sin()).collect();
        let first_loss = mlp.train_step(&input, 2, 0.01);
        let mut last_loss = first_loss;
        for _ in 0..200 {
            last_loss = mlp.train_step(&input, 2, 0.01);
        }
        assert!(last_loss < first_loss * 0.5,
            "loss should decrease: first={} last={}", first_loss, last_loss);
    }

    #[test]
    fn test_different_intents_get_different_embeddings() {
        let (raw_vecs, intents) = make_samples();
        let samples: Vec<(&[f32], &str)> = raw_vecs.iter()
            .zip(intents.iter())
            .map(|(r, &i)| (r.as_slice(), i))
            .collect();
        let mut layer = UnderstandingLayer::build(&samples, 64);
        let c0 = layer.conditioning_vector(&raw_vecs[0]); // coding_implementation
        let c1 = layer.conditioning_vector(&raw_vecs[1]); // coding_debug
        let diff: f32 = c0.iter().zip(c1.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.01, "different intents should produce different conditioning: diff={}", diff);
    }
}
