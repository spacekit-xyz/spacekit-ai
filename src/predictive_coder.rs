//! Predictive Coding — iterative generation refinement via STA error decomposition.
//!
//! Instead of a flat "generate → retry with weak adjustment" loop, the predictive
//! coder decomposes the error signal across Cl(1,7) STA grades:
//!
//!   Grade 0 (δ): scalar magnitude — overall activation/confidence mismatch
//!   Grade 1 (θ): vector direction — semantic intent drift
//!   Grade 2 (α/β): bivector structure — relational/structural errors
//!   Grade 3+ (γ): higher-order — compositional pattern mismatch
//!
//! Each grade gets a targeted correction with its own learning rate, mirroring
//! how the brain uses different frequency bands for different aspects of processing:
//!   δ (1-4 Hz)  — gross arousal / context switching
//!   θ (4-8 Hz)  — working memory, intentional navigation
//!   α/β (8-30 Hz) — sensorimotor binding, structural coordination
//!   γ (30-100 Hz) — fine-grained feature binding, compositional assembly
//!
//! Biological analog: cortical predictive coding (Rao & Ballard 1999).
//! Forward model generates prediction → comparator computes error →
//! error decomposed by cortical layer → targeted feedback corrections.

use crate::clifford::{
    embed_bridge_vector, extract_conditioning, Multivector, GRADE_DIMS, GRADE_OFFSETS,
};
use crate::coherence::{band_coherence_mv, BandCoherence};

/// Per-grade learning rates for targeted correction.
/// Higher grades get stronger correction (they carry more
/// compositional information and are more volatile).
const GRADE_LR: [f32; 9] = [
    0.02, // grade 0: scalar — gentle (overall activation)
    0.08, // grade 1: vector — moderate (semantic direction)
    0.12, // grade 2: bivector — strong (relational structure)
    0.15, // grade 3: trivector — strong (compositional)
    0.10, // grade 4
    0.08, // grade 5
    0.06, // grade 6
    0.04, // grade 7
    0.02, // grade 8: pseudoscalar
];

/// Configuration for the predictive coding loop.
#[derive(Debug, Clone)]
pub struct PredictiveCodingConfig {
    /// Maximum refinement cycles.
    pub max_cycles: usize,
    /// Overall learning rate multiplier.
    pub lr_scale: f32,
    /// Accept threshold: stop when combined coherence exceeds this.
    pub accept_coherence: f32,
    /// Minimum improvement between cycles to continue (stall detection).
    pub min_improvement: f32,
    /// Weight for forward model error (how wrong was the prediction).
    pub w_forward_error: f32,
    /// Weight for goal alignment (how close to query intent).
    pub w_goal_alignment: f32,
}

impl Default for PredictiveCodingConfig {
    fn default() -> Self {
        Self {
            max_cycles: 3,
            lr_scale: 1.0,
            accept_coherence: 0.75,
            min_improvement: 0.02,
            w_forward_error: 0.6,
            w_goal_alignment: 0.4,
        }
    }
}

/// Decomposed error signal across STA grades.
#[derive(Debug, Clone)]
pub struct GradeError {
    /// Per-grade error magnitudes (9 grades in Cl(1,7)).
    pub magnitudes: [f32; 9],
    /// Per-grade correction directions (full multivector).
    pub correction: Multivector,
    /// Which grade has the largest error.
    pub dominant_grade: usize,
    /// Total error magnitude.
    pub total_error: f32,
}

/// Result of a predictive coding refinement cycle.
#[derive(Debug, Clone)]
pub struct RefinementResult {
    /// Adjusted conditioning vector.
    pub conditioning: Vec<f32>,
    /// Band coherence after refinement.
    pub coherence: BandCoherence,
    /// Number of cycles performed.
    pub cycles: usize,
    /// Per-cycle error magnitudes (for diagnostics).
    pub error_history: Vec<f32>,
    /// Whether refinement improved on the original.
    pub improved: bool,
    /// Grade error decomposition from the last cycle.
    pub last_grade_error: Option<GradeError>,
}

/// The predictive coder: iterative refinement of generation conditioning.
pub struct PredictiveCoder {
    config: PredictiveCodingConfig,
}

impl PredictiveCoder {
    pub fn new(config: PredictiveCodingConfig) -> Self {
        Self { config }
    }

    /// Run the predictive coding loop.
    ///
    /// Given:
    ///   - goal_embedding: the query/intent embedding (what we want)
    ///   - initial_conditioning: the conditioning vector used for first generation
    ///   - response_embedding: embedding of the generated response
    ///
    /// Returns an adjusted conditioning vector that corrects for the
    /// prediction error, decomposed by STA grade.
    pub fn refine(
        &self,
        goal_embedding: &[f32],
        initial_conditioning: &[f32],
        response_embedding: &[f32],
    ) -> RefinementResult {
        let goal_mv = embed_bridge_vector(goal_embedding);
        let mut cond = initial_conditioning.to_vec();
        let mut cond_mv = embed_bridge_vector(&cond);
        let target_dim = initial_conditioning.len();

        let initial_coherence = band_coherence_mv(&cond_mv, &goal_mv);
        let mut best_coherence = initial_coherence.combined;
        let mut best_cond = cond.clone();
        let mut error_history = Vec::new();
        let mut last_grade_error = None;

        for cycle in 0..self.config.max_cycles {
            let resp_mv = if cycle == 0 {
                embed_bridge_vector(response_embedding)
            } else {
                cond_mv.clone()
            };

            // Decompose prediction error across STA grades
            let grade_error = self.decompose_error(&goal_mv, &resp_mv, &cond_mv);
            error_history.push(grade_error.total_error);

            // Early accept
            let current_coh = band_coherence_mv(&cond_mv, &goal_mv);
            if current_coh.combined >= self.config.accept_coherence {
                last_grade_error = Some(grade_error);
                break;
            }

            // Stall detection
            if cycle > 0 {
                let prev = error_history[cycle - 1];
                let curr = grade_error.total_error;
                if prev - curr < self.config.min_improvement {
                    last_grade_error = Some(grade_error);
                    break;
                }
            }

            // Apply grade-specific corrections
            self.apply_correction(&mut cond_mv, &grade_error);
            cond = extract_conditioning(&cond_mv, target_dim);

            let new_coh = band_coherence_mv(&cond_mv, &goal_mv);
            if new_coh.combined > best_coherence {
                best_coherence = new_coh.combined;
                best_cond = cond.clone();
            }

            last_grade_error = Some(grade_error);
        }

        let final_mv = embed_bridge_vector(&best_cond);
        let final_coh = band_coherence_mv(&final_mv, &goal_mv);
        let improved = final_coh.combined > initial_coherence.combined;

        RefinementResult {
            conditioning: best_cond,
            coherence: final_coh,
            cycles: error_history.len(),
            error_history,
            improved,
            last_grade_error,
        }
    }

    /// Decompose the error between goal, response, and conditioning
    /// into per-grade error signals.
    ///
    /// Two error sources:
    ///   1. Forward model error: response ≠ conditioning (what we predicted
    ///      the response would be vs what it actually was)
    ///   2. Goal alignment error: response ≠ goal (response doesn't match intent)
    fn decompose_error(
        &self,
        goal_mv: &Multivector,
        response_mv: &Multivector,
        conditioning_mv: &Multivector,
    ) -> GradeError {
        let mut correction = Multivector::zero();
        let mut magnitudes = [0.0f32; 9];
        let mut total = 0.0f32;
        let mut max_grade = 0;
        let mut max_mag = 0.0f32;

        for grade in 0..9 {
            let offset = GRADE_OFFSETS[grade];
            let dim = GRADE_DIMS[grade];

            // Forward model error: what the conditioning predicted vs actual response
            let mut forward_err = 0.0f32;
            for i in 0..dim {
                let diff =
                    conditioning_mv.components[offset + i] - response_mv.components[offset + i];
                forward_err += diff * diff;
            }
            forward_err = forward_err.sqrt();

            // Goal alignment error: response vs goal
            let mut goal_err = 0.0f32;
            for i in 0..dim {
                let diff = goal_mv.components[offset + i] - response_mv.components[offset + i];
                goal_err += diff * diff;
            }
            goal_err = goal_err.sqrt();

            // Combined error for this grade
            let grade_error =
                self.config.w_forward_error * forward_err + self.config.w_goal_alignment * goal_err;
            magnitudes[grade] = grade_error;
            total += grade_error;

            if grade_error > max_mag {
                max_mag = grade_error;
                max_grade = grade;
            }

            // Correction direction: push conditioning toward goal, weighted by error
            let lr = GRADE_LR[grade] * self.config.lr_scale;
            for i in 0..dim {
                let goal_dir =
                    goal_mv.components[offset + i] - conditioning_mv.components[offset + i];
                correction.components[offset + i] = goal_dir * lr * grade_error;
            }
        }

        GradeError {
            magnitudes,
            correction,
            dominant_grade: max_grade,
            total_error: total,
        }
    }

    /// Apply the grade-decomposed correction to the conditioning multivector.
    fn apply_correction(&self, cond_mv: &mut Multivector, error: &GradeError) {
        for i in 0..cond_mv.components.len() {
            cond_mv.components[i] += error.correction.components[i];
        }
    }
}

// ===========================================================================
// Differentiable Slot Optimization
// ===========================================================================

/// Soft probability distribution over slot options (differentiable decode).
///
/// Instead of picking the single nearest-neighbor in `soft_decode_index`,
/// this returns a full probability distribution that can be used for
/// gradient-based optimization of slot bits.
pub fn soft_decode_distribution(bits: &[f32], num_options: usize, temperature: f32) -> Vec<f32> {
    if num_options <= 1 {
        return vec![1.0];
    }

    let n_bits = bits_for_count(num_options);
    let mut distances = Vec::with_capacity(num_options);

    for cand in 0..num_options {
        let mut dist = 0.0f32;
        for i in 0..n_bits {
            let target = if (cand >> i) & 1 == 1 { 1.0f32 } else { 0.0 };
            let d = bits.get(i).copied().unwrap_or(0.0) - target;
            dist += d * d;
        }
        distances.push(dist);
    }

    // Temperature-scaled softmax: P(val) = exp(-dist/T) / sum(exp(-dist/T))
    let t = temperature.max(0.01);
    let max_neg_dist = distances
        .iter()
        .map(|d| -d / t)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = distances
        .iter()
        .map(|d| ((-d / t) - max_neg_dist).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    } else {
        let uniform = 1.0 / num_options as f32;
        probs.fill(uniform);
    }

    probs
}

/// Expected token from soft distribution over slot vocabulary.
/// Returns the token with highest probability and the confidence.
pub fn soft_slot_token(bits: &[f32], slot_vocab: &[u16], temperature: f32) -> (u16, f32) {
    if slot_vocab.is_empty() {
        return (0, 0.0);
    }
    let probs = soft_decode_distribution(bits, slot_vocab.len(), temperature);
    let (best_idx, &best_prob) = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    (slot_vocab[best_idx.min(slot_vocab.len() - 1)], best_prob)
}

/// Optimize slot bits via SPSA to maximize quality.
///
/// Given initial slot bits, iteratively perturbs and evaluates using
/// a scoring function, converging to slot values that produce better tokens.
pub fn optimize_slot_bits(
    initial_bits: &[f32],
    num_steps: usize,
    lr: f32,
    epsilon: f32,
    score_fn: &dyn Fn(&[f32]) -> f32,
) -> Vec<f32> {
    let n = initial_bits.len();
    let mut bits = initial_bits.to_vec();
    let mut best_bits = bits.clone();
    let mut best_score = score_fn(&bits);

    let mut rng_state: u64 = initial_bits
        .iter()
        .fold(0u64, |acc, v| acc.wrapping_add(v.to_bits() as u64))
        .wrapping_add(7919);

    for _ in 0..num_steps {
        // Rademacher perturbation
        let perturbation: Vec<f32> = (0..n)
            .map(|_| {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                if (rng_state >> 33) & 1 == 0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();

        // Forward and backward
        let mut plus = bits.clone();
        let mut minus = bits.clone();
        for i in 0..n {
            plus[i] = (plus[i] + epsilon * perturbation[i]).clamp(0.0, 1.0);
            minus[i] = (minus[i] - epsilon * perturbation[i]).clamp(0.0, 1.0);
        }

        let score_plus = score_fn(&plus);
        let score_minus = score_fn(&minus);

        // Gradient ascent (maximize score)
        let grad_scale = (score_plus - score_minus) / (2.0 * epsilon);
        for i in 0..n {
            bits[i] = (bits[i] + lr * grad_scale * perturbation[i]).clamp(0.0, 1.0);
        }

        let score = score_fn(&bits);
        if score > best_score {
            best_score = score;
            best_bits = bits.clone();
        }
    }

    best_bits
}

fn bits_for_count(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    (usize::BITS - (n - 1).leading_zeros()) as usize
}

// ===========================================================================
// Schema Abstraction
// ===========================================================================

/// A schema is a reusable template extracted from common patterns
/// across multiple programs. It has fixed positions (invariant tokens)
/// and typed slots (positions that vary across programs).
#[derive(Debug, Clone)]
pub struct Schema {
    /// Human-readable label (derived from dominant tokens).
    pub label: String,
    /// Total token length of the schema.
    pub length: usize,
    /// Fixed positions: (position, token_id).
    pub fixed: Vec<(usize, u16)>,
    /// Variable slots: (position, allowed_tokens).
    pub slots: Vec<SchemaSlot>,
    /// Number of programs this schema was extracted from.
    pub support: usize,
    /// Average quality score of source programs.
    pub avg_quality: f32,
}

/// A typed slot in a schema — a position where content varies.
#[derive(Debug, Clone)]
pub struct SchemaSlot {
    pub position: usize,
    /// Tokens observed at this position (with frequency counts).
    pub candidates: Vec<(u16, u32)>,
}

impl SchemaSlot {
    /// Most common token at this slot.
    pub fn mode_token(&self) -> u16 {
        self.candidates
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(tok, _)| *tok)
            .unwrap_or(0)
    }

    /// Number of distinct tokens observed.
    pub fn variety(&self) -> usize {
        self.candidates.len()
    }
}

/// Extract schemas from a collection of programs.
///
/// Groups programs by similarity, then for each group finds
/// positions that are invariant (fixed) vs positions that vary (slots).
pub fn extract_schemas(
    programs: &[(Vec<u16>, f32)], // (token_sequence, quality_score)
    min_support: usize,
    similarity_threshold: f32,
) -> Vec<Schema> {
    if programs.is_empty() {
        return Vec::new();
    }

    // Group programs by token-level Jaccard similarity
    let mut used = vec![false; programs.len()];
    let mut groups: Vec<Vec<usize>> = Vec::new();

    for i in 0..programs.len() {
        if used[i] {
            continue;
        }
        let mut group = vec![i];
        used[i] = true;

        for j in (i + 1)..programs.len() {
            if used[j] {
                continue;
            }
            let sim = token_jaccard(&programs[i].0, &programs[j].0);
            if sim >= similarity_threshold {
                group.push(j);
                used[j] = true;
            }
        }
        if group.len() >= min_support {
            groups.push(group);
        }
    }

    // For each group, extract fixed/variable positions
    groups
        .iter()
        .map(|group| {
            let sequences: Vec<&Vec<u16>> = group.iter().map(|&i| &programs[i].0).collect();
            let min_len = sequences.iter().map(|s| s.len()).min().unwrap_or(0);
            let max_len = sequences.iter().map(|s| s.len()).max().unwrap_or(0);
            let use_len = min_len; // conservative: only schema up to shortest

            let mut fixed = Vec::new();
            let mut slots = Vec::new();
            let threshold = (group.len() as f32 * 0.8).ceil() as usize; // 80% agreement = fixed

            for pos in 0..use_len {
                let mut token_counts: std::collections::HashMap<u16, u32> =
                    std::collections::HashMap::new();
                for seq in &sequences {
                    if pos < seq.len() {
                        *token_counts.entry(seq[pos]).or_insert(0) += 1;
                    }
                }

                let (most_common, max_count) = token_counts
                    .iter()
                    .max_by_key(|(_, &c)| c)
                    .map(|(&t, &c)| (t, c as usize))
                    .unwrap_or((0, 0));

                if max_count >= threshold {
                    fixed.push((pos, most_common));
                } else {
                    let candidates: Vec<(u16, u32)> = token_counts.into_iter().collect();
                    slots.push(SchemaSlot {
                        position: pos,
                        candidates,
                    });
                }
            }

            let avg_quality =
                group.iter().map(|&i| programs[i].1).sum::<f32>() / group.len() as f32;

            // Label from first few fixed tokens
            let label = fixed
                .iter()
                .take(4)
                .map(|(_, t)| format!("{}", t))
                .collect::<Vec<_>>()
                .join("-");

            Schema {
                label: if label.is_empty() {
                    "abstract".to_string()
                } else {
                    label
                },
                length: max_len,
                fixed,
                slots,
                support: group.len(),
                avg_quality,
            }
        })
        .collect()
}

/// Fill a schema's slots using a conditioning vector.
///
/// For each slot, scores candidate tokens by how well they align
/// with the conditioning signal (via cosine similarity of a simple
/// token-position hash embedding — this is a heuristic).
pub fn fill_schema(schema: &Schema, conditioning: &[f32]) -> Vec<u16> {
    let mut tokens = vec![0u16; schema.length];

    // Place fixed tokens
    for &(pos, tok) in &schema.fixed {
        if pos < tokens.len() {
            tokens[pos] = tok;
        }
    }

    // Fill slots based on conditioning signal
    for slot in &schema.slots {
        if slot.position >= tokens.len() || slot.candidates.is_empty() {
            continue;
        }

        // Score each candidate by hash-based alignment with conditioning
        let pos_hash = slot.position as f32 * 0.1;
        let cond_signal: f32 = conditioning
            .iter()
            .enumerate()
            .map(|(i, &v)| v * ((i as f32 * 0.07 + pos_hash).sin()))
            .sum::<f32>();

        let best = slot
            .candidates
            .iter()
            .max_by(|(tok_a, count_a), (tok_b, count_b)| {
                let score_a =
                    *count_a as f32 + 0.3 * (cond_signal * *tok_a as f32 * 0.001).sin().abs();
                let score_b =
                    *count_b as f32 + 0.3 * (cond_signal * *tok_b as f32 * 0.001).sin().abs();
                score_a.partial_cmp(&score_b).unwrap()
            })
            .map(|(tok, _)| *tok)
            .unwrap_or(0);

        tokens[slot.position] = best;
    }

    // Truncate trailing zeros
    while tokens.last() == Some(&0) && tokens.len() > 1 {
        tokens.pop();
    }

    tokens
}

/// Token-level Jaccard similarity between two sequences.
fn token_jaccard(a: &[u16], b: &[u16]) -> f32 {
    use std::collections::HashSet;
    let set_a: HashSet<u16> = a.iter().copied().collect();
    let set_b: HashSet<u16> = b.iter().copied().collect();
    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
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

    // --- Predictive Coding tests ---

    #[test]
    fn test_predictive_coder_refines() {
        let goal = make_embedding(1.0, 128);
        let cond = make_embedding(2.0, 128);
        let response = make_embedding(3.0, 128);

        let pc = PredictiveCoder::new(PredictiveCodingConfig::default());
        let result = pc.refine(&goal, &cond, &response);

        assert!(!result.error_history.is_empty());
        assert_eq!(result.conditioning.len(), 128);
    }

    #[test]
    fn test_grade_error_decomposition() {
        let goal = make_embedding(1.0, 128);
        let response = make_embedding(5.0, 128);
        let cond = make_embedding(3.0, 128);

        let pc = PredictiveCoder::new(PredictiveCodingConfig::default());
        let result = pc.refine(&goal, &cond, &response);

        if let Some(ge) = result.last_grade_error {
            assert!(ge.total_error > 0.0, "should have nonzero error");
            assert!(ge.dominant_grade < 9, "dominant grade in range");
            let sum: f32 = ge.magnitudes.iter().sum();
            assert!(
                (sum - ge.total_error).abs() < 0.01,
                "magnitudes should sum to total"
            );
        }
    }

    #[test]
    fn test_predictive_coder_identical_goal_stops_early() {
        let goal = make_embedding(1.0, 128);
        let pc = PredictiveCoder::new(PredictiveCodingConfig {
            accept_coherence: 0.3,
            ..Default::default()
        });
        let result = pc.refine(&goal, &goal, &goal);

        assert!(
            result.cycles <= 2,
            "should stop early when goal matches: {} cycles",
            result.cycles
        );
    }

    #[test]
    fn test_predictive_coder_stall_detection() {
        let goal = make_embedding(1.0, 128);
        let cond = make_embedding(1.0001, 128); // very close
        let response = make_embedding(1.0002, 128);

        let pc = PredictiveCoder::new(PredictiveCodingConfig {
            max_cycles: 10,
            min_improvement: 0.5, // very high → will stall immediately
            ..Default::default()
        });
        let result = pc.refine(&goal, &cond, &response);

        assert!(
            result.cycles < 10,
            "should detect stall early: {} cycles",
            result.cycles
        );
    }

    // --- Differentiable Slot Decode tests ---

    #[test]
    fn test_soft_decode_distribution_sums_to_one() {
        let bits = vec![0.8, 0.2, 0.6];
        let probs = soft_decode_distribution(&bits, 8, 1.0);

        assert_eq!(probs.len(), 8);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "should sum to 1.0: {}", sum);
    }

    #[test]
    fn test_soft_decode_low_temperature_concentrates() {
        let bits = vec![1.0, 0.0, 1.0]; // binary: 5 (101)
        let cold = soft_decode_distribution(&bits, 8, 0.01);
        let warm = soft_decode_distribution(&bits, 8, 10.0);

        let cold_max = cold.iter().cloned().fold(0.0f32, f32::max);
        let warm_max = warm.iter().cloned().fold(0.0f32, f32::max);

        assert!(
            cold_max > warm_max + 0.1,
            "low temperature should concentrate: cold_max={}, warm_max={}",
            cold_max,
            warm_max
        );
    }

    #[test]
    fn test_soft_decode_picks_correct_value() {
        let bits = vec![1.0, 0.0, 1.0]; // should decode to 5 (= 1 + 4)
        let probs = soft_decode_distribution(&bits, 8, 0.1);
        let best = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        assert_eq!(best, 5, "should decode to 5: got {}", best);
    }

    #[test]
    fn test_soft_slot_token() {
        let bits = vec![0.0]; // should pick index 0
        let vocab = vec![42u16, 99, 101];
        let (token, conf) = soft_slot_token(&bits, &vocab, 0.1);
        assert_eq!(token, 42, "should pick first vocab entry");
        assert!(conf > 0.5, "should be confident: {}", conf);
    }

    #[test]
    fn test_optimize_slot_bits_improves() {
        let initial = vec![0.5, 0.5, 0.5]; // maximally uncertain
        let target_val = 5; // 101 in binary

        let score_fn = |bits: &[f32]| -> f32 {
            let mut score = 0.0f32;
            for i in 0..3 {
                let target = if (target_val >> i) & 1 == 1 { 1.0 } else { 0.0 };
                score -= (bits.get(i).copied().unwrap_or(0.0) - target).powi(2);
            }
            score
        };

        let initial_score = score_fn(&initial);
        let optimized = optimize_slot_bits(&initial, 20, 0.1, 0.05, &score_fn);
        let final_score = score_fn(&optimized);

        assert!(
            final_score > initial_score,
            "optimization should improve: {} → {}",
            initial_score,
            final_score
        );
    }

    // --- Schema Abstraction tests ---

    #[test]
    fn test_extract_schemas_finds_patterns() {
        let programs: Vec<(Vec<u16>, f32)> = vec![
            (vec![10, 20, 30, 40, 50], 0.8),
            (vec![10, 20, 31, 40, 50], 0.7),
            (vec![10, 20, 32, 40, 50], 0.9),
        ];

        let schemas = extract_schemas(&programs, 2, 0.5);

        assert!(!schemas.is_empty(), "should extract at least one schema");
        let s = &schemas[0];
        assert!(s.support >= 3, "all three should be in the group");
        assert!(s.fixed.len() >= 3, "positions 0,1,3,4 should be fixed");
        assert!(!s.slots.is_empty(), "position 2 should be a slot");
    }

    #[test]
    fn test_extract_schemas_min_support() {
        let programs: Vec<(Vec<u16>, f32)> = vec![(vec![10, 20], 0.5), (vec![99, 88], 0.5)];

        let schemas = extract_schemas(&programs, 3, 0.5);
        assert!(
            schemas.is_empty(),
            "should not extract with support < min_support"
        );
    }

    #[test]
    fn test_fill_schema_places_fixed_tokens() {
        let schema = Schema {
            label: "test".to_string(),
            length: 5,
            fixed: vec![(0, 10), (1, 20), (3, 40), (4, 50)],
            slots: vec![SchemaSlot {
                position: 2,
                candidates: vec![(30, 5), (31, 3), (32, 2)],
            }],
            support: 3,
            avg_quality: 0.8,
        };

        let cond = make_embedding(1.0, 128);
        let tokens = fill_schema(&schema, &cond);

        assert_eq!(tokens[0], 10);
        assert_eq!(tokens[1], 20);
        assert_eq!(tokens[3], 40);
        assert_eq!(tokens[4], 50);
        assert!(
            tokens[2] == 30 || tokens[2] == 31 || tokens[2] == 32,
            "slot should be filled with a candidate: {}",
            tokens[2]
        );
    }

    #[test]
    fn test_schema_slot_mode_token() {
        let slot = SchemaSlot {
            position: 0,
            candidates: vec![(10, 5), (20, 8), (30, 3)],
        };
        assert_eq!(slot.mode_token(), 20, "should return most frequent");
    }

    #[test]
    fn test_token_jaccard() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![3, 4, 5, 6, 7];
        let sim = token_jaccard(&a, &b);
        // intersection = {3,4,5} = 3, union = {1,2,3,4,5,6,7} = 7
        assert!((sim - 3.0 / 7.0).abs() < 0.01);
    }
}
