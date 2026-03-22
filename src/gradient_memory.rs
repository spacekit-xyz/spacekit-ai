//! SpaceTime Gradient Memory (STGM)
//!
//! Test-time optimization of memory tokens in Cl(1,7) SpaceTime Algebra.
//! Instead of retrieving a single stored program, STGM initializes memory
//! from k-nearest programs and runs gradient descent in STA space to compose
//! a novel, query-optimized representation that decodes to tokens.
//!
//! Cognitive architecture mapping:
//!   PFC goal representation  → query embedding (sustained across steps)
//!   Working memory buffer    → memory slots (persist across optimization)
//!   Predictive coding        → loss computation = prediction error
//!   Conflict monitoring (ACC)→ coherence threshold → escalate steps
//!   Hierarchical control     → goal → topic → slots → tokens
//!   Basal ganglia gating     → soft attention over source programs

use crate::clifford::{embed_bridge_vector, extract_conditioning, Multivector};
use crate::coherence::{band_coherence_mv, BandCoherence};

/// Simple LCG for fast, deterministic-per-step random perturbations.
/// Avoids pulling in rand for this hot path.
struct FastRng(u64);
impl FastRng {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn rademacher(&mut self) -> f32 {
        if self.next_u32() & 1 == 0 { 1.0 } else { -1.0 }
    }
}

/// Source program data used to seed memory.
#[derive(Clone)]
pub struct MemorySource {
    pub centroid: Vec<f32>,
    pub token_ids: Vec<u16>,
    pub similarity: f32,
}

/// Configuration for the gradient memory optimization loop.
#[derive(Debug, Clone)]
pub struct GradientMemoryConfig {
    /// Number of optimization steps (default 8).
    pub max_steps: usize,
    /// Learning rate for SPSA gradient steps.
    pub lr: f32,
    /// SPSA perturbation magnitude.
    pub epsilon: f32,
    /// Coherence threshold: stop early if exceeded.
    pub coherence_target: f32,
    /// Conflict escalation: multiply steps when coherence below this.
    pub conflict_threshold: f32,
    /// Weight for relevance loss (band coherence with query).
    pub w_relevance: f32,
    /// Weight for composition coherence (inter-source alignment).
    pub w_composition: f32,
    /// Weight for diversity (anti-collapse between memory slots).
    pub w_diversity: f32,
}

impl Default for GradientMemoryConfig {
    fn default() -> Self {
        Self {
            max_steps: 8,
            lr: 0.05,
            epsilon: 0.02,
            coherence_target: 0.85,
            conflict_threshold: 0.5,
            w_relevance: 0.55,
            w_composition: 0.30,
            w_diversity: 0.15,
        }
    }
}

/// Result of a gradient memory optimization run.
#[derive(Debug, Clone)]
pub struct GradientMemoryResult {
    /// Optimized conditioning vector in bridge space.
    pub conditioning: Vec<f32>,
    /// Per-position soft weights over source programs (for token decoding).
    pub blend_weights: Vec<f32>,
    /// Final band coherence with the query.
    pub coherence: BandCoherence,
    /// Number of optimization steps taken.
    pub steps_taken: usize,
    /// Whether conflict monitoring escalated the step count.
    pub escalated: bool,
    /// Loss trajectory (for diagnostics).
    pub loss_history: Vec<f32>,
}

/// SpaceTime Gradient Memory: test-time optimization in Cl(1,7).
///
/// The memory state is a multivector in STA space that gets optimized
/// via SPSA gradient descent to maximize relevance to the query while
/// maintaining coherence across source programs.
pub struct GradientMemory {
    /// STA multivector representing the current memory state.
    state: Multivector,
    /// Bridge-space representation (extracted from STA state).
    bridge_state: Vec<f32>,
    /// Source programs used to seed memory.
    sources: Vec<MemorySource>,
    /// Soft attention weights over sources (sum to 1).
    attention: Vec<f32>,
    /// Configuration.
    config: GradientMemoryConfig,
}

impl GradientMemory {
    /// Initialize memory from source programs and query embedding.
    ///
    /// Seeds the STA memory state as a similarity-weighted combination
    /// of source program centroids, biased toward the query.
    pub fn new(
        query: &[f32],
        sources: Vec<MemorySource>,
        target_dim: usize,
        config: GradientMemoryConfig,
    ) -> Self {
        let n = sources.len().max(1);
        let mut attention: Vec<f32> = sources.iter()
            .map(|s| s.similarity.max(0.01))
            .collect();
        let sum: f32 = attention.iter().sum();
        for w in &mut attention { *w /= sum; }

        // Initialize bridge state as weighted combination of source centroids + query
        let dim = query.len().max(1);
        let mut bridge_state = vec![0.0f32; dim];
        let query_weight = 0.3;
        let source_weight = 1.0 - query_weight;

        for (i, v) in query.iter().enumerate() {
            bridge_state[i] += v * query_weight;
        }
        for (src, &w) in sources.iter().zip(attention.iter()) {
            for (i, v) in src.centroid.iter().enumerate() {
                if i < dim {
                    bridge_state[i] += v * w * source_weight;
                }
            }
        }

        let state = embed_bridge_vector(&bridge_state);
        let _ = target_dim; // used during decode, not init

        Self {
            state,
            bridge_state,
            sources,
            attention,
            config,
        }
    }

    /// Run the test-time optimization loop.
    ///
    /// Each step:
    ///   1. Compute multi-objective loss (relevance + composition + diversity)
    ///   2. SPSA gradient estimate on the STA multivector
    ///   3. Update memory state
    ///   4. Check early stopping (coherence target) or conflict escalation
    pub fn optimize(&mut self, query: &[f32], target_dim: usize) -> GradientMemoryResult {
        let query_mv = embed_bridge_vector(query);
        let mut steps = self.config.max_steps;
        let mut escalated = false;
        let mut loss_history = Vec::with_capacity(steps);

        // Initial coherence check for conflict monitoring
        let initial_coh = band_coherence_mv(&self.state, &query_mv);
        if initial_coh.combined < self.config.conflict_threshold {
            steps = (steps * 2).min(24);
            escalated = true;
        }

        let n_components = self.state.components.len();
        let mut best_loss = f32::MAX;
        let mut best_state = self.state.clone();
        let mut best_attention = self.attention.clone();

        // Seed RNG from query hash for reproducibility per query
        let seed = query.iter().take(8).enumerate()
            .fold(0u64, |acc, (i, v)| acc ^ ((*v).to_bits() as u64).wrapping_shl(i as u32 * 8));
        let mut rng = FastRng::new(seed.wrapping_add(42));

        for step in 0..steps {
            let current_loss = self.compute_loss(&query_mv);
            loss_history.push(current_loss);

            if current_loss < best_loss {
                best_loss = current_loss;
                best_state = self.state.clone();
                best_attention = self.attention.clone();
            }

            // Early stopping: coherence target reached
            let coh = band_coherence_mv(&self.state, &query_mv);
            if coh.combined >= self.config.coherence_target && step >= 2 {
                break;
            }

            // SPSA gradient step on STA components
            self.spsa_step(&query_mv, n_components, &mut rng);

            // Also optimize attention weights
            self.spsa_attention_step(&query_mv, &mut rng);

            // Rebuild bridge state from STA
            self.bridge_state = extract_conditioning(&self.state, target_dim);
        }

        // Use best state found
        self.state = best_state;
        self.attention = best_attention;
        self.bridge_state = extract_conditioning(&self.state, target_dim);

        let final_coh = band_coherence_mv(&self.state, &query_mv);

        GradientMemoryResult {
            conditioning: self.bridge_state.clone(),
            blend_weights: self.attention.clone(),
            coherence: final_coh,
            steps_taken: loss_history.len(),
            escalated,
            loss_history,
        }
    }

    /// Multi-objective loss function.
    ///
    /// - Relevance: band coherence between memory and query (want high)
    /// - Composition: coherence between memory and source centroids (want high)
    /// - Diversity: variance in attention weights (want high, avoid collapse)
    fn compute_loss(&self, query_mv: &Multivector) -> f32 {
        // Relevance: negative band coherence with query
        let relevance = band_coherence_mv(&self.state, query_mv);
        let relevance_loss = 1.0 - relevance.combined;

        // Composition: average coherence with source programs
        let composition_loss = if self.sources.is_empty() {
            0.0
        } else {
            let mut total = 0.0f32;
            for src in &self.sources {
                let src_mv = embed_bridge_vector(&src.centroid);
                let coh = band_coherence_mv(&self.state, &src_mv);
                total += 1.0 - coh.combined;
            }
            total / self.sources.len() as f32
        };

        // Diversity: entropy of attention weights (high entropy = diverse)
        let diversity_loss = {
            let entropy: f32 = self.attention.iter()
                .filter(|&&w| w > 1e-8)
                .map(|&w| -w * w.ln())
                .sum();
            let max_entropy = (self.attention.len() as f32).max(1.0).ln();
            if max_entropy > 0.0 { 1.0 - entropy / max_entropy } else { 0.0 }
        };

        self.config.w_relevance * relevance_loss
            + self.config.w_composition * composition_loss
            + self.config.w_diversity * diversity_loss
    }

    /// SPSA gradient step on the STA multivector components.
    fn spsa_step(&mut self, query_mv: &Multivector, n_components: usize, rng: &mut FastRng) {
        let eps = self.config.epsilon;
        let lr = self.config.lr;

        // Random Rademacher perturbation direction (different each step)
        let perturbation: Vec<f32> = (0..n_components)
            .map(|_| rng.rademacher())
            .collect();

        // Forward perturbation
        let mut state_plus = self.state.clone();
        for i in 0..n_components {
            state_plus.components[i] += eps * perturbation[i];
        }

        // Backward perturbation
        let mut state_minus = self.state.clone();
        for i in 0..n_components {
            state_minus.components[i] -= eps * perturbation[i];
        }

        // Compute losses
        let saved_state = self.state.clone();

        self.state = state_plus;
        let loss_plus = self.compute_loss(query_mv);

        self.state = state_minus;
        let loss_minus = self.compute_loss(query_mv);

        self.state = saved_state;

        // SPSA gradient estimate and update
        let grad_scale = (loss_plus - loss_minus) / (2.0 * eps);
        for i in 0..n_components {
            self.state.components[i] -= lr * grad_scale * perturbation[i];
        }
    }

    /// SPSA step on the attention weights over source programs.
    fn spsa_attention_step(&mut self, query_mv: &Multivector, rng: &mut FastRng) {
        if self.sources.len() <= 1 { return; }

        let eps = self.config.epsilon;
        let lr = self.config.lr * 0.5;
        let n = self.attention.len();

        let perturbation: Vec<f32> = (0..n)
            .map(|_| rng.rademacher())
            .collect();

        // Perturb attention and recompute STA state
        let saved_attention = self.attention.clone();
        let saved_state = self.state.clone();

        // Plus perturbation
        let mut att_plus = saved_attention.clone();
        for i in 0..n { att_plus[i] = (att_plus[i] + eps * perturbation[i]).max(0.01); }
        let sum: f32 = att_plus.iter().sum();
        for w in &mut att_plus { *w /= sum; }
        self.attention = att_plus;
        self.rebuild_state_from_attention();
        let loss_plus = self.compute_loss(query_mv);

        // Minus perturbation
        let mut att_minus = saved_attention.clone();
        for i in 0..n { att_minus[i] = (att_minus[i] - eps * perturbation[i]).max(0.01); }
        let sum: f32 = att_minus.iter().sum();
        for w in &mut att_minus { *w /= sum; }
        self.attention = att_minus;
        self.rebuild_state_from_attention();
        let loss_minus = self.compute_loss(query_mv);

        // Restore and update
        self.attention = saved_attention;
        self.state = saved_state;

        let grad_scale = (loss_plus - loss_minus) / (2.0 * eps);
        for i in 0..n {
            self.attention[i] = (self.attention[i] - lr * grad_scale * perturbation[i]).max(0.01);
        }
        let sum: f32 = self.attention.iter().sum();
        for w in &mut self.attention { *w /= sum; }

        self.rebuild_state_from_attention();
    }

    /// Reconstruct the STA state from current attention weights over sources.
    fn rebuild_state_from_attention(&mut self) {
        let query_weight = 0.3;
        let source_weight = 1.0 - query_weight;
        let dim = self.bridge_state.len();
        let mut new_bridge = vec![0.0f32; dim];

        // Keep the query component from the original bridge state
        for i in 0..dim {
            new_bridge[i] = self.bridge_state[i] * query_weight;
        }
        for (src, &w) in self.sources.iter().zip(self.attention.iter()) {
            for (i, v) in src.centroid.iter().enumerate() {
                if i < dim {
                    new_bridge[i] += v * w * source_weight;
                }
            }
        }

        self.state = embed_bridge_vector(&new_bridge);
    }

    /// Decode the optimized memory into a token sequence.
    ///
    /// Uses attention-weighted blending of source program token sequences.
    /// For each output position, selects the token from the highest-weighted
    /// source that has a token at that position, producing a novel composition.
    pub fn decode_tokens(&self) -> Vec<u16> {
        if self.sources.is_empty() { return Vec::new(); }

        // Determine output length from weighted sources
        let max_len: usize = self.sources.iter()
            .map(|s| s.token_ids.len())
            .max()
            .unwrap_or(0);

        let target_len = {
            let weighted_len: f32 = self.sources.iter()
                .zip(self.attention.iter())
                .map(|(s, &w)| s.token_ids.len() as f32 * w)
                .sum();
            (weighted_len.round() as usize).min(max_len)
        };

        let mut output = Vec::with_capacity(target_len);

        for pos in 0..target_len {
            // For each position, pick token from highest-scoring source
            // that has a token at this position
            let mut best_token = 0u16;
            let mut best_weight = -1.0f32;

            for (src_idx, src) in self.sources.iter().enumerate() {
                if pos < src.token_ids.len() {
                    let w = self.attention[src_idx];
                    if w > best_weight {
                        best_weight = w;
                        best_token = src.token_ids[pos];
                    }
                }
            }

            output.push(best_token);
        }

        output
    }

    /// Decode using sentence-level interleaving: pick whole sentences
    /// from sources based on attention weights, producing more coherent
    /// compositions than token-level blending.
    pub fn decode_sentences(
        &self,
        dictionary: &crate::spectral::TokenDictionary,
    ) -> String {
        if self.sources.is_empty() { return String::new(); }

        let source_texts: Vec<String> = self.sources.iter()
            .map(|s| dictionary.decode(&s.token_ids))
            .collect();

        let source_sentences: Vec<Vec<String>> = source_texts.iter()
            .map(|text| {
                text.split(". ")
                    .filter(|s| s.len() > 3)
                    .map(|s| s.trim().to_string())
                    .collect()
            })
            .collect();

        // Greedily select sentences: round-robin weighted by attention
        let max_sentences = 4;
        let mut result_sentences: Vec<String> = Vec::new();
        let mut used: Vec<usize> = vec![0; self.sources.len()]; // next sentence index per source

        for _ in 0..max_sentences {
            let mut best_src = 0;
            let mut best_score = -1.0f32;

            for (idx, (sents, &w)) in source_sentences.iter().zip(self.attention.iter()).enumerate() {
                if used[idx] < sents.len() {
                    let novelty = if result_sentences.is_empty() { 1.0 } else {
                        let candidate = &sents[used[idx]];
                        let overlap: f32 = result_sentences.iter()
                            .map(|existing| bigram_overlap(candidate, existing))
                            .max_by(|a, b| a.partial_cmp(b).unwrap())
                            .unwrap_or(0.0);
                        1.0 - overlap
                    };
                    let score = w * novelty;
                    if score > best_score {
                        best_score = score;
                        best_src = idx;
                    }
                }
            }

            if best_score <= 0.0 { break; }
            if let Some(sent) = source_sentences[best_src].get(used[best_src]) {
                result_sentences.push(sent.clone());
            }
            used[best_src] += 1;
        }

        let joined = result_sentences.join(". ");
        if !joined.is_empty() && !joined.ends_with('.') {
            format!("{}.", joined)
        } else {
            joined
        }
    }
}

/// Bigram overlap ratio between two strings (0.0 = no overlap, 1.0 = identical).
fn bigram_overlap(a: &str, b: &str) -> f32 {
    let a_words: Vec<&str> = a.split_whitespace().collect();
    let b_words: Vec<&str> = b.split_whitespace().collect();
    if a_words.len() < 2 || b_words.len() < 2 { return 0.0; }

    let a_bigrams: std::collections::HashSet<(&str, &str)> = a_words.windows(2)
        .map(|w| (w[0], w[1]))
        .collect();
    let b_bigrams: std::collections::HashSet<(&str, &str)> = b_words.windows(2)
        .map(|w| (w[0], w[1]))
        .collect();

    let intersection = a_bigrams.intersection(&b_bigrams).count();
    let union = a_bigrams.union(&b_bigrams).count();
    if union == 0 { 0.0 } else { intersection as f32 / union as f32 }
}

// ===========================================================================
// Unit tests
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedding(seed: f32, dim: usize) -> Vec<f32> {
        (0..dim).map(|i| ((i as f32 + seed) * 0.1).sin()).collect()
    }

    fn make_sources(n: usize, dim: usize) -> Vec<MemorySource> {
        (0..n).map(|i| {
            MemorySource {
                centroid: make_embedding(i as f32 * 50.0, dim),
                token_ids: vec![1, 2, 3, 4, 5],
                similarity: 0.9 - (i as f32 * 0.1),
            }
        }).collect()
    }

    #[test]
    fn test_gradient_memory_initializes() {
        let query = make_embedding(0.5, 128);
        let sources = make_sources(3, 128);
        let gm = GradientMemory::new(&query, sources, 128, GradientMemoryConfig::default());

        assert_eq!(gm.sources.len(), 3);
        assert_eq!(gm.attention.len(), 3);
        let sum: f32 = gm.attention.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "attention should sum to 1.0: {}", sum);
    }

    #[test]
    fn test_optimize_reduces_loss() {
        let query = make_embedding(0.5, 128);
        let sources = make_sources(3, 128);
        let mut gm = GradientMemory::new(&query, sources, 128, GradientMemoryConfig::default());

        let result = gm.optimize(&query, 128);

        assert!(result.steps_taken >= 2, "should take multiple steps");
        assert!(!result.loss_history.is_empty());
        let first_loss = result.loss_history[0];
        let last_loss = result.loss_history.last().copied().unwrap();
        assert!(last_loss <= first_loss + 0.1,
            "loss should not increase much: {} → {}", first_loss, last_loss);
    }

    #[test]
    fn test_optimize_produces_conditioning() {
        let query = make_embedding(0.5, 128);
        let sources = make_sources(3, 128);
        let mut gm = GradientMemory::new(&query, sources, 128, GradientMemoryConfig::default());

        let result = gm.optimize(&query, 128);

        assert_eq!(result.conditioning.len(), 128);
        assert!(result.conditioning.iter().any(|&v| v != 0.0),
            "conditioning should not be all zeros");
    }

    #[test]
    fn test_conflict_escalation() {
        // Use very distant query to trigger low initial coherence
        let query = make_embedding(999.0, 128);
        let mut sources = make_sources(2, 128);
        for s in &mut sources { s.similarity = 0.1; }

        let config = GradientMemoryConfig {
            max_steps: 4,
            conflict_threshold: 0.99, // very high → will escalate
            ..Default::default()
        };
        let mut gm = GradientMemory::new(&query, sources, 128, config);
        let result = gm.optimize(&query, 128);

        assert!(result.escalated, "should escalate when initial coherence is low");
        assert!(result.steps_taken > 4, "escalated should take more steps");
    }

    #[test]
    fn test_early_stopping() {
        // Use query that matches sources well
        let sources = make_sources(2, 128);
        let query = sources[0].centroid.clone();

        let config = GradientMemoryConfig {
            max_steps: 20,
            coherence_target: 0.3, // very low threshold → stop early
            ..Default::default()
        };
        let mut gm = GradientMemory::new(&query, sources, 128, config);
        let result = gm.optimize(&query, 128);

        assert!(result.steps_taken < 20, "should stop early: took {} steps", result.steps_taken);
    }

    #[test]
    fn test_decode_tokens() {
        let query = make_embedding(0.5, 128);
        let mut sources = make_sources(3, 128);
        sources[0].token_ids = vec![10, 20, 30, 40];
        sources[1].token_ids = vec![50, 60, 70];
        sources[2].token_ids = vec![80, 90];

        let gm = GradientMemory::new(&query, sources, 128, GradientMemoryConfig::default());
        let tokens = gm.decode_tokens();

        assert!(!tokens.is_empty(), "should decode to non-empty tokens");
        assert!(tokens.len() <= 4, "output length should be bounded by weighted avg");
    }

    #[test]
    fn test_attention_stays_normalized() {
        let query = make_embedding(0.5, 128);
        let sources = make_sources(4, 128);
        let mut gm = GradientMemory::new(&query, sources, 128, GradientMemoryConfig::default());

        gm.optimize(&query, 128);

        let sum: f32 = gm.attention.iter().sum();
        assert!((sum - 1.0).abs() < 0.05, "attention should stay normalized: {}", sum);
        assert!(gm.attention.iter().all(|&w| w >= 0.0), "attention should be non-negative");
    }

    #[test]
    fn test_blend_weights_reflect_similarity() {
        let query = make_embedding(0.5, 128);
        let sources = vec![
            MemorySource { centroid: make_embedding(0.5, 128), token_ids: vec![1, 2], similarity: 0.95 },
            MemorySource { centroid: make_embedding(100.0, 128), token_ids: vec![3, 4], similarity: 0.3 },
        ];
        let gm = GradientMemory::new(&query, sources, 128, GradientMemoryConfig::default());

        assert!(gm.attention[0] > gm.attention[1],
            "higher-similarity source should have higher initial attention: {} vs {}",
            gm.attention[0], gm.attention[1]);
    }

    #[test]
    fn test_single_source_works() {
        let query = make_embedding(0.5, 128);
        let sources = vec![
            MemorySource { centroid: make_embedding(1.0, 128), token_ids: vec![1, 2, 3], similarity: 0.9 },
        ];
        let mut gm = GradientMemory::new(&query, sources, 128, GradientMemoryConfig::default());
        let result = gm.optimize(&query, 128);

        assert_eq!(result.blend_weights.len(), 1);
        assert!((result.blend_weights[0] - 1.0).abs() < 0.01);
    }
}
