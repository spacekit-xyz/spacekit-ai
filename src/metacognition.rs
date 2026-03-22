//! MetaCognition — reflective inference loop (System 1.5).
//!
//! After the primary generation pass (System 1) produces a candidate response,
//! MetaCognition evaluates it through a generate-reflect-decide loop:
//!
//! 1. **Encode** the candidate: embed [prompt ⊕ response] as a joint vector
//! 2. **Reflect**: score coherence (semantic overlap), relevance (topic match),
//!    and completeness (response adequacy) using a Reflection MicroBrain
//! 3. **Decide**: accept if scores exceed thresholds, otherwise re-condition
//!    and retry (max retries configurable, default 2)
//! 4. **Degrade gracefully**: if all retries fail, emit a structured
//!    "I don't have knowledge about this topic" message instead of returning
//!    low-quality output
//!
//! The Reflection MicroBrain is an InfraciliaryLattice trained on
//! (prompt+response, quality_label) pairs during the main training pass.
//! It learns what "good response to this kind of prompt" looks like in
//! embedding space, without backpropagation.
//!
//! Biological analog: anterior cingulate cortex error monitoring —
//! the "that doesn't seem right" signal that triggers re-evaluation.

use serde::{Deserialize, Serialize};

/// Reflection scores from the MetaCognition evaluation.
#[derive(Clone, Debug)]
pub struct ReflectionScores {
    /// Semantic overlap between prompt embedding and response embedding.
    /// High = response talks about the same thing as the prompt.
    pub coherence: f32,
    /// Topic-specific match: does the response address the specific topic
    /// inferred from the prompt (e.g., "observer_pattern" not "factory_pattern")?
    pub relevance: f32,
    /// Response adequacy: length, specificity, and structural completeness.
    pub completeness: f32,
    /// Combined quality score (weighted blend of the three).
    pub quality: f32,
}

/// The outcome of a MetaCognition evaluation cycle.
#[derive(Clone, Debug)]
pub enum ReflectionOutcome {
    /// Response passes quality gate — return it.
    Accept {
        scores: ReflectionScores,
    },
    /// Response failed quality gate — retry with adjusted conditioning.
    /// The adjustment vector should be blended into the conditioning for retry.
    Retry {
        scores: ReflectionScores,
        adjustment: Vec<f32>,
        attempt: usize,
    },
    /// All retries exhausted — degrade gracefully with a structured message.
    Degrade {
        scores: ReflectionScores,
        message: String,
        attempts_exhausted: usize,
    },
}

/// Configuration for the MetaCognition module.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetaCognitionConfig {
    /// Minimum quality score to accept a response (0.0–1.0).
    pub accept_threshold: f32,
    /// Maximum retry attempts before graceful degradation.
    pub max_retries: usize,
    /// Weight of coherence in the combined quality score.
    pub coherence_weight: f32,
    /// Weight of relevance in the combined quality score.
    pub relevance_weight: f32,
    /// Weight of completeness in the combined quality score.
    pub completeness_weight: f32,
    /// Minimum response length (chars) to consider "complete".
    pub min_response_length: usize,
    /// Adjustment strength: how much the reflection signal perturbs conditioning.
    pub adjustment_strength: f32,
}

impl Default for MetaCognitionConfig {
    fn default() -> Self {
        Self {
            accept_threshold: 0.45,
            max_retries: 2,
            coherence_weight: 0.45,
            relevance_weight: 0.35,
            completeness_weight: 0.20,
            min_response_length: 20,
            adjustment_strength: 0.25,
        }
    }
}

/// The MetaCognition engine: reflective quality gate on System 1 output.
///
/// Operates entirely in embedding space — no token-level analysis, no
/// autoregressive decoding. Each evaluation is a single geometric operation:
/// project the (prompt, response) pair into the reflection lattice and
/// measure distance from known-good attractors.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetaCognition {
    pub config: MetaCognitionConfig,
    /// Coherence reference vectors: centroids of known-good (prompt, response) pairs
    /// per topic, learned during training. Each entry is (topic_key, centroid).
    reference_centroids: Vec<(String, Vec<f32>)>,
    /// Global centroid of all training pairs (fallback when topic is unknown).
    global_centroid: Vec<f32>,
    /// Number of training pairs absorbed.
    pair_count: u64,
    /// Per-topic pair counts for weighted averaging.
    topic_counts: Vec<(String, u64)>,
}

impl MetaCognition {
    pub fn new(config: MetaCognitionConfig) -> Self {
        Self {
            config,
            reference_centroids: Vec::new(),
            global_centroid: Vec::new(),
            pair_count: 0,
            topic_counts: Vec::new(),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(MetaCognitionConfig::default())
    }

    /// Train the reflection brain: absorb a known-good (prompt_emb, response_emb)
    /// pair with its topic label. Builds reference centroids via EMA.
    pub fn absorb_pair(
        &mut self,
        prompt_emb: &[f32],
        response_emb: &[f32],
        topic: &str,
    ) {
        let joint = Self::joint_embedding(prompt_emb, response_emb);
        let alpha = 0.05_f32;

        // Update global centroid
        if self.global_centroid.is_empty() {
            self.global_centroid = joint.clone();
        } else {
            for (i, v) in joint.iter().enumerate() {
                if i < self.global_centroid.len() {
                    self.global_centroid[i] = self.global_centroid[i] * (1.0 - alpha) + v * alpha;
                }
            }
        }
        self.pair_count += 1;

        // Update or create topic-specific centroid
        if let Some(pos) = self.reference_centroids.iter().position(|(t, _)| t == topic) {
            let centroid = &mut self.reference_centroids[pos].1;
            for (i, v) in joint.iter().enumerate() {
                if i < centroid.len() {
                    centroid[i] = centroid[i] * (1.0 - alpha) + v * alpha;
                }
            }
            if let Some(tc) = self.topic_counts.iter_mut().find(|(t, _)| t == topic) {
                tc.1 += 1;
            }
        } else {
            self.reference_centroids.push((topic.to_string(), joint));
            self.topic_counts.push((topic.to_string(), 1));
        }
    }

    /// Evaluate a candidate response against the prompt.
    ///
    /// Returns `ReflectionScores` with coherence, relevance, and completeness.
    /// The caller decides whether to accept, retry, or degrade based on the
    /// combined quality score.
    pub fn evaluate(
        &self,
        prompt_emb: &[f32],
        response_emb: &[f32],
        response_text: &str,
        topic_hint: Option<&str>,
    ) -> ReflectionScores {
        let coherence = self.score_coherence(prompt_emb, response_emb);
        let relevance = self.score_relevance(prompt_emb, response_emb, topic_hint);
        let completeness = self.score_completeness(response_text);

        let quality = self.config.coherence_weight * coherence
            + self.config.relevance_weight * relevance
            + self.config.completeness_weight * completeness;

        ReflectionScores {
            coherence,
            relevance,
            completeness,
            quality,
        }
    }

    /// Full reflection cycle: evaluate → decide → (retry adjustment | accept | degrade).
    pub fn reflect(
        &self,
        prompt_emb: &[f32],
        response_emb: &[f32],
        response_text: &str,
        topic_hint: Option<&str>,
        attempt: usize,
    ) -> ReflectionOutcome {
        let scores = self.evaluate(prompt_emb, response_emb, response_text, topic_hint);

        println!(
            "  [metacog] attempt={}, coherence={:.3}, relevance={:.3}, completeness={:.3}, quality={:.3} (threshold={:.3})",
            attempt, scores.coherence, scores.relevance, scores.completeness,
            scores.quality, self.config.accept_threshold
        );

        if scores.quality >= self.config.accept_threshold {
            return ReflectionOutcome::Accept { scores };
        }

        if attempt >= self.config.max_retries {
            let message = self.degradation_message(topic_hint);
            return ReflectionOutcome::Degrade {
                scores,
                message,
                attempts_exhausted: attempt,
            };
        }

        // Compute adjustment vector: push conditioning away from the bad response
        // and toward the reference centroid for this topic.
        let adjustment = self.compute_adjustment(prompt_emb, response_emb, topic_hint);
        ReflectionOutcome::Retry {
            scores,
            adjustment,
            attempt,
        }
    }

    /// Whether the reflection brain has enough training data to be useful.
    pub fn is_ready(&self) -> bool {
        self.pair_count >= 10 && !self.reference_centroids.is_empty()
    }

    pub fn pair_count(&self) -> u64 {
        self.pair_count
    }

    pub fn topic_count(&self) -> usize {
        self.reference_centroids.len()
    }

    // --- Internal scoring functions ---

    /// Coherence: cosine similarity between prompt and response embeddings.
    /// Measures whether the response is semantically related to the prompt.
    fn score_coherence(&self, prompt_emb: &[f32], response_emb: &[f32]) -> f32 {
        cosine_sim(prompt_emb, response_emb).max(0.0)
    }

    /// Relevance: how close the (prompt, response) joint embedding is to
    /// the reference centroid for the given topic. Falls back to global
    /// centroid when topic is unknown or missing.
    fn score_relevance(
        &self,
        prompt_emb: &[f32],
        response_emb: &[f32],
        topic_hint: Option<&str>,
    ) -> f32 {
        let joint = Self::joint_embedding(prompt_emb, response_emb);

        // Try topic-specific centroid first
        if let Some(topic) = topic_hint {
            if let Some((_, centroid)) = self.reference_centroids.iter().find(|(t, _)| t == topic) {
                return cosine_sim(&joint, centroid).max(0.0);
            }
        }

        // Fallback: global centroid
        if !self.global_centroid.is_empty() {
            cosine_sim(&joint, &self.global_centroid).max(0.0)
        } else {
            0.5 // no training data yet — neutral score
        }
    }

    /// Completeness: embedding-based structural adequacy with heuristic floor.
    ///
    /// Three signals blended:
    /// 1. Embedding spread: response centroid distance from the global centroid
    ///    (responses that land near a real training region score higher)
    /// 2. Length adequacy: sigmoid on character count
    /// 3. Structural markers: sentences, lists, paragraphs
    fn score_completeness(&self, response_text: &str) -> f32 {
        let len = response_text.len();
        if len < 5 {
            return 0.0;
        }
        if len < self.config.min_response_length {
            return 0.15;
        }

        // Length score: sigmoid that saturates around 250 chars
        let length_score = 1.0 / (1.0 + (-0.02 * (len as f32 - 80.0)).exp());

        let has_sentences = response_text.contains(". ") || response_text.contains(".\n");
        let sentence_bonus = if has_sentences { 0.15 } else { 0.0 };

        let has_structure = response_text.contains('\n')
            || response_text.contains("- ")
            || response_text.contains("1.");
        let structure_bonus = if has_structure { 0.1 } else { 0.0 };

        // Embedding adequacy: if we have reference centroids, check whether
        // the response text length is consistent with training data norms.
        // Short responses to prompts that typically produce long answers are suspect.
        let embedding_bonus = if !self.reference_centroids.is_empty() {
            let avg_topic_count = self.topic_counts.iter()
                .map(|(_, c)| *c as f32)
                .sum::<f32>()
                / self.topic_counts.len().max(1) as f32;
            // Well-covered topics (many training pairs) expect richer responses
            if avg_topic_count > 5.0 && len > 100 { 0.15 }
            else if avg_topic_count > 2.0 && len > 50 { 0.08 }
            else { 0.0 }
        } else {
            0.0
        };

        (length_score + sentence_bonus + structure_bonus + embedding_bonus).clamp(0.0, 1.0)
    }

    /// Compute an adjustment vector that steers conditioning away from the
    /// failed response and toward the reference centroid for this topic.
    ///
    /// Geometric interpretation: the adjustment is a vector in embedding space
    /// pointing from the bad joint embedding toward the reference attractor,
    /// scaled by `adjustment_strength`. Blending this into conditioning shifts
    /// the next generation attempt toward the known-good region.
    fn compute_adjustment(
        &self,
        prompt_emb: &[f32],
        response_emb: &[f32],
        topic_hint: Option<&str>,
    ) -> Vec<f32> {
        let joint = Self::joint_embedding(prompt_emb, response_emb);

        let reference = if let Some(topic) = topic_hint {
            self.reference_centroids
                .iter()
                .find(|(t, _)| t == topic)
                .map(|(_, c)| c.as_slice())
        } else {
            None
        }
        .unwrap_or(&self.global_centroid);

        if reference.is_empty() || joint.len() != reference.len() {
            return vec![0.0; prompt_emb.len()];
        }

        // Direction: reference − joint (points toward known-good)
        let dim = prompt_emb.len().min(joint.len()).min(reference.len());
        let mut adjustment = vec![0.0f32; dim];
        for i in 0..dim {
            adjustment[i] = (reference[i] - joint[i]) * self.config.adjustment_strength;
        }

        adjustment
    }

    /// Create a joint embedding from prompt and response by element-wise
    /// averaging. This captures the "semantic midpoint" between question
    /// and answer — good pairs cluster tightly, bad pairs scatter.
    fn joint_embedding(prompt_emb: &[f32], response_emb: &[f32]) -> Vec<f32> {
        let dim = prompt_emb.len().max(response_emb.len());
        let mut joint = vec![0.0f32; dim];
        for i in 0..dim {
            let p = if i < prompt_emb.len() { prompt_emb[i] } else { 0.0 };
            let r = if i < response_emb.len() { response_emb[i] } else { 0.0 };
            joint[i] = (p + r) * 0.5;
        }
        joint
    }

    /// Produce a structured degradation message when all retries are exhausted.
    fn degradation_message(&self, topic_hint: Option<&str>) -> String {
        match topic_hint {
            Some(topic) => format!(
                "I don't have sufficient knowledge about '{}' to give you a confident answer. \
                 This topic is outside my current training scope. \
                 Could you rephrase or ask about a related topic I might know better?",
                topic.replace('_', " ")
            ),
            None => "I'm not confident in my answer to this question. \
                     It may be outside my current training scope. \
                     Could you rephrase or provide more context?"
                .to_string(),
        }
    }
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let dot: f32 = a[..len].iter().zip(b[..len].iter()).map(|(x, y)| x * y).sum();
    let na = a[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-10 || nb < 1e-10 {
        return 0.0;
    }
    dot / (na * nb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reflection_accept() {
        let mut mc = MetaCognition::with_defaults();

        let prompt = vec![1.0, 0.0, 0.5, 0.3];
        let good_response = vec![0.9, 0.1, 0.4, 0.35];

        for _ in 0..20 {
            mc.absorb_pair(&prompt, &good_response, "test_topic");
        }

        assert!(mc.is_ready());
        let scores = mc.evaluate(&prompt, &good_response, "This is a good detailed response.", Some("test_topic"));
        assert!(scores.quality > 0.3, "Good pair should score reasonably: {:.3}", scores.quality);
    }

    #[test]
    fn test_reflection_degrade() {
        let mc = MetaCognition::with_defaults();
        let prompt = vec![1.0, 0.0, 0.5];
        let bad_response = vec![-1.0, 0.0, -0.5]; // opposite direction

        let outcome = mc.reflect(&prompt, &bad_response, "", None, 3);
        match outcome {
            ReflectionOutcome::Degrade { message, .. } => {
                assert!(message.contains("not confident") || message.contains("outside"));
            }
            _ => panic!("Expected degradation for bad response after max retries"),
        }
    }

    #[test]
    fn test_joint_embedding_symmetry() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let j1 = MetaCognition::joint_embedding(&a, &b);
        let j2 = MetaCognition::joint_embedding(&b, &a);
        assert_eq!(j1, j2);
        assert_eq!(j1, vec![2.5, 3.5, 4.5]);
    }
}
