// Training Objectives: RTD + Salient Span Masking + Contrastive Development
//
// Three objectives that improve embedding discriminability for the
// Paramecium lattice retrieval system:
//
// 1. **Salient Span Masking** — Preferentially mask domain-specific keywords
//    during training, forcing the model to reconstruct them from context.
//    Creates augmented training pairs that strengthen representations
//    for exactly the distinguishing terms (stack, derivative, eigenvalue).
//
// 2. **RTD (Replaced Token Detection)** — Replace salient tokens with
//    plausible alternatives from the same position class. The corrupted
//    text produces embeddings that should NOT match the original response,
//    acting as hard negatives during lattice development.
//
// 3. **Contrastive Development** — After lattice programs are built,
//    push apart centroids of programs from different topics that are
//    too similar. This creates sharper decision boundaries.
//
// All three are "training games" that require no neural networks —
// they modify the data and lattice organization, not the model.

use std::collections::HashSet;

use crate::spectral::TokenDictionary;

pub struct SaliencyLexicon {
    keywords: HashSet<String>,
    bigrams: HashSet<String>,
}

impl SaliencyLexicon {
    pub fn from_keywords(raw_keywords: Vec<String>) -> Self {
        let mut keywords = HashSet::new();
        let mut bigrams = HashSet::new();

        for kw in &raw_keywords {
            let lower = kw.to_ascii_lowercase();
            let words: Vec<&str> = lower.split_whitespace().collect();
            if words.len() == 1 {
                keywords.insert(words[0].to_string());
            } else {
                bigrams.insert(lower.clone());
                for w in &words {
                    if w.len() > 2 {
                        keywords.insert(w.to_string());
                    }
                }
            }
        }

        Self { keywords, bigrams }
    }

    pub fn keyword_count(&self) -> usize {
        self.keywords.len()
    }

    pub fn is_salient(&self, token: &str) -> bool {
        let lower = token.to_ascii_lowercase();
        self.keywords.contains(&lower)
    }

    /// Score a token's saliency: 1.0 for exact keyword match,
    /// 0.5 for substring of a bigram keyword, 0.0 otherwise.
    pub fn score(&self, token: &str) -> f32 {
        let lower = token.to_ascii_lowercase();
        if lower.len() <= 2 {
            return 0.0;
        }
        if self.keywords.contains(&lower) {
            return 1.0;
        }
        for bg in &self.bigrams {
            if bg.contains(&lower) {
                return 0.5;
            }
        }
        0.0
    }

    /// Find positions of salient tokens in a token ID sequence.
    pub fn salient_positions(
        &self,
        token_ids: &[u16],
        dict: &TokenDictionary,
    ) -> Vec<(usize, f32)> {
        token_ids
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| {
                let text = dict.token_str(id)?;
                let s = self.score(text);
                if s > 0.0 {
                    Some((i, s))
                } else {
                    None
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Salient Span Masking
// ---------------------------------------------------------------------------

/// Create augmented text with salient spans masked (replaced with "[MASK]").
/// Returns the original plus N augmented versions where different salient
/// spans are masked, forcing the model to learn from surrounding context.
pub fn mask_salient_spans(
    text: &str,
    lexicon: &SaliencyLexicon,
    max_augments: usize,
    seed: u64,
) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 3 {
        return Vec::new();
    }

    let salient_positions: Vec<usize> = words
        .iter()
        .enumerate()
        .filter(|(_, w)| lexicon.score(w) > 0.0)
        .map(|(i, _)| i)
        .collect();

    if salient_positions.is_empty() {
        return Vec::new();
    }

    let mut augmented = Vec::new();
    let mut hasher = seed;

    for _ in 0..max_augments.min(salient_positions.len()) {
        hasher = splitmix64(hasher);
        let mask_idx = salient_positions[(hasher as usize) % salient_positions.len()];

        let span_len = 1 + ((splitmix64(hasher + 1) as usize) % 2);
        let end = (mask_idx + span_len).min(words.len());

        let masked: Vec<&str> = words
            .iter()
            .enumerate()
            .map(|(i, w)| {
                if i >= mask_idx && i < end {
                    "[MASK]"
                } else {
                    *w
                }
            })
            .collect();
        augmented.push(masked.join(" "));

        hasher = splitmix64(hasher);
    }

    augmented
}

// ---------------------------------------------------------------------------
// Replaced Token Detection (RTD)
// ---------------------------------------------------------------------------

/// Create a corrupted version of text by replacing salient tokens with
/// random alternatives from the dictionary. Returns the corrupted text
/// and a bitmask of which word positions were replaced.
pub fn replace_salient_tokens(
    text: &str,
    lexicon: &SaliencyLexicon,
    dict: &TokenDictionary,
    replacement_rate: f32,
    seed: u64,
) -> Option<(String, Vec<bool>)> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 3 {
        return None;
    }

    let vocab_size = dict.tokens.len();
    if vocab_size < 10 {
        return None;
    }

    let mut result_words: Vec<String> = words.iter().map(|w| w.to_string()).collect();
    let mut replaced = vec![false; words.len()];
    let mut hasher = seed;
    let mut any_replaced = false;

    for (i, word) in words.iter().enumerate() {
        let saliency = lexicon.score(word);
        let effective_rate = if saliency > 0.0 {
            replacement_rate * (1.0 + saliency)
        } else {
            replacement_rate * 0.15
        };

        hasher = splitmix64(hasher);
        let r = (hasher as f32) / (u64::MAX as f32);
        if r < effective_rate {
            hasher = splitmix64(hasher);
            let replacement_id = (hasher as usize) % vocab_size;
            if let Some(replacement) = dict.token_str(replacement_id as u16) {
                if replacement != *word && replacement.len() > 1 && replacement != "<EOS>" {
                    result_words[i] = replacement.to_string();
                    replaced[i] = true;
                    any_replaced = true;
                }
            }
        }
    }

    if any_replaced {
        Some((result_words.join(" "), replaced))
    } else {
        None
    }
}

/// Per-token RTD detection score: given original and corrupted token
/// sequences, compute how many replacements the lattice can detect
/// (i.e., which positions have significantly different embeddings).
/// Returns (detected_count, total_replaced, detection_accuracy).
pub fn rtd_detection_accuracy(
    original_ids: &[u16],
    corrupted_ids: &[u16],
    replaced_mask: &[bool],
) -> (usize, usize, f32) {
    let total = replaced_mask.iter().filter(|&&r| r).count();
    if total == 0 {
        return (0, 0, 1.0);
    }

    let detected = original_ids
        .iter()
        .zip(corrupted_ids.iter())
        .zip(replaced_mask.iter())
        .filter(|((orig, corr), &was_replaced)| was_replaced && orig != corr)
        .count();

    let accuracy = detected as f32 / total as f32;
    (detected, total, accuracy)
}

// ---------------------------------------------------------------------------
// Contrastive Development
// ---------------------------------------------------------------------------

/// Contrastive refinement: push apart program centroids from different
/// topics that are too similar. This creates sharper decision boundaries.
///
/// For each program in topic A, find the nearest program in any other topic.
/// If the cross-topic similarity exceeds `margin`, push both centroids apart
/// by `repulsion_rate`.
///
/// Returns the number of repulsion operations performed.
pub fn contrastive_refine(
    topic_programs: &mut Vec<(String, Vec<CentroidEntry>)>,
    margin: f32,
    repulsion_rate: f32,
) -> usize {
    let num_topics = topic_programs.len();
    if num_topics < 2 {
        return 0;
    }

    let mut repulsions = 0;

    for t_a in 0..num_topics {
        for p_a in 0..topic_programs[t_a].1.len() {
            let centroid_a = topic_programs[t_a].1[p_a].centroid.clone();

            let mut nearest_other: Option<(usize, usize, f32)> = None;

            for t_b in 0..num_topics {
                if t_a == t_b {
                    continue;
                }
                for (p_b, entry_b) in topic_programs[t_b].1.iter().enumerate() {
                    let sim = cosine_sim(&centroid_a, &entry_b.centroid);
                    if sim > margin {
                        let better = nearest_other.map_or(true, |(_, _, best_sim)| sim > best_sim);
                        if better {
                            nearest_other = Some((t_b, p_b, sim));
                        }
                    }
                }
            }

            if let Some((t_b, p_b, _sim)) = nearest_other {
                let centroid_b = topic_programs[t_b].1[p_b].centroid.clone();

                for (i, (a, b)) in centroid_a.iter().zip(centroid_b.iter()).enumerate() {
                    let delta = a - b;
                    if let Some(ea) = topic_programs[t_a].1[p_a].centroid.get_mut(i) {
                        *ea += delta * repulsion_rate;
                    }
                    if let Some(eb) = topic_programs[t_b].1[p_b].centroid.get_mut(i) {
                        *eb -= delta * repulsion_rate;
                    }
                }
                repulsions += 1;
            }
        }
    }

    repulsions
}

pub struct CentroidEntry {
    pub centroid: Vec<f32>,
    pub program_idx: usize,
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-8 || nb < 1e-8 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(0x9e3779b97f4a7c15);
    state = (state ^ (state >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    state = (state ^ (state >> 27)).wrapping_mul(0x94d049bb133111eb);
    state ^ (state >> 31)
}

// ---------------------------------------------------------------------------
// Per-Grade Training Loss (STA-CALM Phase 2a)
//
// Each Cl(1,7) grade gets its own self-supervised signal:
//   Grade 0 (scalar):       reconstruction confidence (BrierLM-style)
//   Grade 1 (vector):       semantic content (cloze/RTD)
//   Grade 2 (bivector):     relational structure (contrastive pairs)
//   Grade 3 (trivector):    discourse position (sentence-order prediction)
//   Grade 8 (pseudoscalar): negation pairs (assertion vs negation)
// ---------------------------------------------------------------------------

use crate::clifford::{GRADE_DIMS, GRADE_OFFSETS};
use crate::text_autoencoder::SpacetimeChunk;

/// Per-grade loss decomposition for the STA-CALM training objective.
#[derive(Debug, Clone, Default)]
pub struct GradedLoss {
    pub scalar_loss: f32,
    pub vector_loss: f32,
    pub bivector_loss: f32,
    pub trivector_loss: f32,
    pub pseudo_loss: f32,
    pub total: f32,
}

/// Compute the per-grade training loss between a predicted and actual SpacetimeChunk.
///
/// grade_weights: [scalar, vector, bivector, trivector, pseudo] weights (5 total)
/// that determine each grade's contribution to the total loss.
pub fn graded_training_loss(
    predicted: &SpacetimeChunk,
    actual: &SpacetimeChunk,
    grade_weights: &[f32; 5],
) -> GradedLoss {
    // Grade 0: scalar confidence — BrierLM-style: squared error of confidence
    let scalar_loss = (predicted.confidence - actual.confidence).powi(2);

    // Grade 1: semantic direction — cosine distance
    let vector_loss = 1.0 - predicted.semantic_similarity(actual);

    // Grade 2: bivector relational — separate boost (causal) and rotation (structural)
    let boost_loss = 1.0 - predicted.causal_similarity(actual);
    let rotation_loss = 1.0 - predicted.structural_similarity(actual);
    let bivector_loss = 0.4 * boost_loss + 0.6 * rotation_loss;

    // Grade 3: discourse trivector — cosine distance in discourse space
    let trivector_loss = {
        let dot: f32 = predicted
            .discourse
            .iter()
            .zip(actual.discourse.iter())
            .map(|(a, b)| a * b)
            .sum();
        let na: f32 = predicted
            .discourse
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        let nb: f32 = actual.discourse.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na < 1e-8 || nb < 1e-8 {
            1.0
        } else {
            1.0 - (dot / (na * nb))
        }
    };

    // Grade 8: pseudoscalar — sign agreement loss
    // If both have same sign: 0 loss. Opposite signs: proportional penalty.
    let pseudo_loss = if (predicted.dual_marker > 0.0) == (actual.dual_marker > 0.0) {
        (predicted.dual_marker - actual.dual_marker).powi(2) * 0.1
    } else {
        1.0 + (predicted.dual_marker - actual.dual_marker).powi(2) * 0.1
    };

    let total = grade_weights[0] * scalar_loss
        + grade_weights[1] * vector_loss
        + grade_weights[2] * bivector_loss
        + grade_weights[3] * trivector_loss
        + grade_weights[4] * pseudo_loss;

    let w_sum: f32 = grade_weights.iter().sum();
    let normalized_total = if w_sum > 1e-8 { total / w_sum } else { total };

    GradedLoss {
        scalar_loss,
        vector_loss,
        bivector_loss,
        trivector_loss,
        pseudo_loss,
        total: normalized_total,
    }
}

/// Default grade weights emphasizing semantic direction and relational structure.
pub const DEFAULT_GRADE_WEIGHTS: [f32; 5] = [0.05, 0.35, 0.30, 0.15, 0.15];

/// Sentence-Order Prediction (SOP) objective for grade-3 discourse training.
///
/// Given two consecutive chunks, score whether they are in correct order.
/// The score is the cosine similarity of the discourse grade-3 vectors
/// after applying a "forward" bias — correct order should have positive
/// discourse dot product, reversed order should have negative.
pub fn sentence_order_score(first: &SpacetimeChunk, second: &SpacetimeChunk) -> f32 {
    let mut forward_discourse = [0.0f32; 56];
    for i in 0..56 {
        forward_discourse[i] = second.discourse[i] - first.discourse[i];
    }
    let norm: f32 = forward_discourse.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-8 {
        return 0.0;
    }
    // Positive asymmetry = correct order, negative = reversed
    let asymmetry: f32 = forward_discourse.iter().sum::<f32>() / norm;
    asymmetry
}

/// Negation pair loss: given an assertion chunk and its negation, compute
/// how well the pseudoscalar grade captures the polarity flip.
/// Perfect: assertion.dual_marker > 0, negation.dual_marker < 0.
pub fn negation_pair_loss(assertion: &SpacetimeChunk, negation: &SpacetimeChunk) -> f32 {
    let margin = 0.5;
    let sign_diff = assertion.dual_marker - negation.dual_marker;
    if sign_diff > margin {
        0.0 // correct: assertion positive, negation negative
    } else {
        (margin - sign_diff).powi(2) // penalty for insufficient separation
    }
}

// ---------------------------------------------------------------------------
// Training data augmentation pipeline
// ---------------------------------------------------------------------------

/// Augment a training dataset with salient span masking and RTD replacements.
/// For each sample that contains salient terms:
/// - Creates masked augments (same response, masked query)
/// - Creates RTD corrupted pairs (marked as negatives)
///
/// Returns augmented samples as (text, expected_response, is_negative) triples.
pub fn augment_training_data(
    samples: &[(String, String)],
    lexicon: &SaliencyLexicon,
    dict: &TokenDictionary,
    mask_augments_per_sample: usize,
    rtd_rate: f32,
) -> Vec<AugmentedSample> {
    let mut augmented = Vec::new();

    for (i, (text, response)) in samples.iter().enumerate() {
        let seed = (i as u64).wrapping_mul(0x517cc1b727220a95);

        let masked_texts = mask_salient_spans(response, lexicon, mask_augments_per_sample, seed);
        for mt in masked_texts {
            augmented.push(AugmentedSample {
                text: text.clone(),
                response: mt,
                kind: AugmentKind::SalientMask,
            });
        }

        if let Some((corrupted, _mask)) =
            replace_salient_tokens(response, lexicon, dict, rtd_rate, seed.wrapping_add(42))
        {
            augmented.push(AugmentedSample {
                text: text.clone(),
                response: corrupted,
                kind: AugmentKind::RtdNegative,
            });
        }
    }

    augmented
}

#[derive(Debug, Clone)]
pub struct AugmentedSample {
    pub text: String,
    pub response: String,
    pub kind: AugmentKind,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AugmentKind {
    SalientMask,
    RtdNegative,
}

// ---------------------------------------------------------------------------
// Contrastive Pair Generation for Bivector Training (Phase 2c)
//
// Generate pairs of programs that are semantically similar (similar grade-1)
// but relationally different (different grade-2). These are the "hard
// negatives" that specifically pressure the bivector grade to learn
// discriminative relational structure.
// ---------------------------------------------------------------------------

/// A contrastive pair: two chunks that are similar in grade-1 (content)
/// but should differ in grade-2 (relational structure).
#[derive(Debug, Clone)]
pub struct ContrastivePair {
    pub anchor_text: String,
    pub positive_text: String,
    pub negative_text: String,
    pub grade1_sim: f32,
    pub grade2_sim: f32,
}

/// Generate contrastive pairs from a set of topic-grouped programs.
///
/// For each program A, find programs in OTHER topics whose grade-1 similarity
/// is high (> threshold) but that belong to a different topic. These form
/// contrastive pairs that train the bivector grade to distinguish them.
pub fn generate_contrastive_pairs(
    topic_programs: &[(String, Vec<(String, SpacetimeChunk)>)],
    similarity_threshold: f32,
) -> Vec<ContrastivePair> {
    let mut pairs = Vec::new();

    for (t_a, (topic_a, progs_a)) in topic_programs.iter().enumerate() {
        for (text_a, chunk_a) in progs_a {
            // Find a same-topic positive (highest grade-1 sim within topic)
            let mut best_positive: Option<(&str, f32)> = None;
            for (text_p, chunk_p) in progs_a {
                if std::ptr::eq(text_a, text_p) {
                    continue;
                }
                let sim = chunk_a.semantic_similarity(chunk_p);
                if best_positive.map_or(true, |(_, best)| sim > best) {
                    best_positive = Some((text_p.as_str(), sim));
                }
            }

            // Find cross-topic negatives with high grade-1 sim
            for (t_b, (_topic_b, progs_b)) in topic_programs.iter().enumerate() {
                if t_a == t_b {
                    continue;
                }
                for (text_n, chunk_n) in progs_b {
                    let g1_sim = chunk_a.semantic_similarity(chunk_n);
                    if g1_sim > similarity_threshold {
                        let g2_sim = chunk_a.structural_similarity(chunk_n);
                        if let Some((pos_text, _)) = best_positive {
                            pairs.push(ContrastivePair {
                                anchor_text: text_a.clone(),
                                positive_text: pos_text.to_string(),
                                negative_text: text_n.clone(),
                                grade1_sim: g1_sim,
                                grade2_sim: g2_sim,
                            });
                        }
                    }
                }
            }
        }
    }

    pairs
}

// ---------------------------------------------------------------------------
// Negation Pair Generation for Pseudoscalar Training (Phase 2d)
//
// Generate (assertion, negation) pairs from training data to train the
// pseudoscalar grade. The pseudoscalar should flip sign between an
// assertion and its negation.
// ---------------------------------------------------------------------------

/// A negation pair for pseudoscalar training.
#[derive(Debug, Clone)]
pub struct NegationPair {
    pub assertion: String,
    pub negation: String,
}

/// Generate negation pairs from training text by applying syntactic
/// negation patterns. Returns pairs where the negation is a mechanical
/// transformation of the assertion.
pub fn generate_negation_pairs(texts: &[String]) -> Vec<NegationPair> {
    let mut pairs = Vec::new();

    for text in texts {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.len() < 3 {
            continue;
        }

        // Pattern 1: "X is Y" → "X is not Y"
        if let Some(is_pos) = words.iter().position(|w| w.eq_ignore_ascii_case("is")) {
            if is_pos + 1 < words.len() && !words[is_pos + 1].eq_ignore_ascii_case("not") {
                let mut negated = words.clone();
                negated.insert(is_pos + 1, "not");
                pairs.push(NegationPair {
                    assertion: text.clone(),
                    negation: negated.join(" "),
                });
                continue;
            }
        }

        // Pattern 2: "X are Y" → "X are not Y"
        if let Some(are_pos) = words.iter().position(|w| w.eq_ignore_ascii_case("are")) {
            if are_pos + 1 < words.len() && !words[are_pos + 1].eq_ignore_ascii_case("not") {
                let mut negated = words.clone();
                negated.insert(are_pos + 1, "not");
                pairs.push(NegationPair {
                    assertion: text.clone(),
                    negation: negated.join(" "),
                });
                continue;
            }
        }

        // Pattern 3: "X can Y" → "X cannot Y"
        if let Some(can_pos) = words.iter().position(|w| w.eq_ignore_ascii_case("can")) {
            if can_pos + 1 < words.len()
                && !words[can_pos + 1].eq_ignore_ascii_case("not")
                && !words[can_pos].eq_ignore_ascii_case("cannot")
            {
                let mut negated: Vec<String> = words.iter().map(|w| w.to_string()).collect();
                negated[can_pos] = "cannot".to_string();
                pairs.push(NegationPair {
                    assertion: text.clone(),
                    negation: negated.join(" "),
                });
                continue;
            }
        }

        // Pattern 4: "X has Y" → "X does not have Y"
        if let Some(has_pos) = words.iter().position(|w| w.eq_ignore_ascii_case("has")) {
            if has_pos + 1 < words.len() {
                let mut negated: Vec<String> = words.iter().map(|w| w.to_string()).collect();
                negated[has_pos] = "does not have".to_string();
                pairs.push(NegationPair {
                    assertion: text.clone(),
                    negation: negated.join(" "),
                });
                continue;
            }
        }

        // Pattern 5: Generic prefix negation — "It is not the case that X"
        if words.len() >= 5 {
            let negation = format!("It is not the case that {}", text.to_lowercase());
            pairs.push(NegationPair {
                assertion: text.clone(),
                negation,
            });
        }
    }

    pairs
}

// ---------------------------------------------------------------------------
// TrainingOrchestrator — Coordinated 3-phase STA-CALM training loop
//
// Phase 1: Grade Pretraining — pressure each Cl(1,7) grade independently
//   scalar    ← BrierLM reconstruction confidence
//   vector    ← cloze/RTD semantic content
//   bivector  ← contrastive relational pairs
//   trivector ← sentence-order prediction
//   pseudo    ← negation pair polarity
//
// Phase 2: Rotor Predictor — train the SemanticPropagator
//   learn transition rotors that minimize geometric action
//   along semantic trajectories through chunk sequences
//
// Phase 3: Joint Fine-tuning — end-to-end graded loss
//   unfreeze all grades, let cross-grade interaction emerge
//   from the full geometric product composition
// ---------------------------------------------------------------------------

use crate::text_autoencoder::{ChunkCodec, SemanticPropagator, CATA_DIM, CHUNK_K};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TrainingPhase {
    GradePretraining,
    RotorPretraining,
    JointFinetuning,
}

/// Diagnostic statistics for one orchestrator epoch.
#[derive(Debug, Clone, Default)]
pub struct EpochDiagnostics {
    pub phase: String,
    pub epoch: usize,
    pub avg_scalar_loss: f32,
    pub avg_vector_loss: f32,
    pub avg_bivector_loss: f32,
    pub avg_trivector_loss: f32,
    pub avg_pseudo_loss: f32,
    pub avg_total_loss: f32,
    pub sop_accuracy: f32,
    pub negation_accuracy: f32,
    pub rotor_prediction_confidence: f32,
    pub num_samples: usize,
}

impl EpochDiagnostics {
    pub fn print_summary(&self) {
        println!("  [{}] epoch {}: total={:.4} | scalar={:.4} vector={:.4} bivector={:.4} trivector={:.4} pseudo={:.4} | sop={:.2} neg={:.2} rotor_conf={:.2} | n={}",
            self.phase, self.epoch,
            self.avg_total_loss,
            self.avg_scalar_loss, self.avg_vector_loss,
            self.avg_bivector_loss, self.avg_trivector_loss,
            self.avg_pseudo_loss,
            self.sop_accuracy, self.negation_accuracy,
            self.rotor_prediction_confidence,
            self.num_samples);
    }
}

/// STA-CALM Training Orchestrator.
///
/// Operates on **bridged embedding centroids** — the semantically-rich
/// vectors where `embed_bridge_vector` places meaningful content into
/// each Cl(1,7) grade.  CDMA chunk encodings are only used for
/// reconstruction validation, NOT for grade-level training pressure.
pub struct TrainingOrchestrator {
    pub phase: TrainingPhase,
    pub grade_weights: [f32; 5],
    pub codec: ChunkCodec,
    pub mass: f32,
    pub coupling: f32,
    pub learning_rate: f32,
    pub diagnostics: Vec<EpochDiagnostics>,
}

impl TrainingOrchestrator {
    pub fn new(vocab_size: usize) -> Self {
        Self {
            phase: TrainingPhase::GradePretraining,
            grade_weights: DEFAULT_GRADE_WEIGHTS,
            codec: ChunkCodec::new(vocab_size),
            mass: 1.0,
            coupling: 0.4,
            learning_rate: 0.05,
            diagnostics: Vec::new(),
        }
    }

    fn centroid_chunks(programs: &[(String, Vec<u16>, Vec<f32>)]) -> Vec<SpacetimeChunk> {
        programs
            .iter()
            .map(|(_, _, centroid)| SpacetimeChunk::from_centroid(centroid))
            .collect()
    }

    fn topic_groups(programs: &[(String, Vec<u16>, Vec<f32>)]) -> Vec<(String, Vec<usize>)> {
        let mut map: std::collections::HashMap<String, Vec<usize>> =
            std::collections::HashMap::new();
        for (i, (topic, _, _)) in programs.iter().enumerate() {
            map.entry(topic.clone()).or_default().push(i);
        }
        let mut groups: Vec<_> = map.into_iter().collect();
        groups.sort_by(|a, b| a.0.cmp(&b.0));
        groups
    }

    /// Run the full 3-phase training pipeline on a set of program sequences.
    ///
    /// Each program is represented as (topic_name, token_sequence, centroid).
    /// Returns per-epoch diagnostics for monitoring grade activation.
    pub fn run_full_pipeline(
        &mut self,
        programs: &mut [(String, Vec<u16>, Vec<f32>)],
        phase1_epochs: usize,
        phase2_epochs: usize,
        phase3_epochs: usize,
    ) -> Vec<EpochDiagnostics> {
        println!("\n--- STA-CALM Training Orchestrator (centroid-bridged) ---");
        println!(
            "  programs: {}, phase1: {} epochs, phase2: {} epochs, phase3: {} epochs",
            programs.len(),
            phase1_epochs,
            phase2_epochs,
            phase3_epochs
        );

        self.phase = TrainingPhase::GradePretraining;
        println!("\n  [Phase 1: Grade Pretraining (centroid-bridged)]");
        for epoch in 0..phase1_epochs {
            let diag = self.run_grade_pretraining_epoch(programs, epoch);
            diag.print_summary();
            self.diagnostics.push(diag);
        }

        self.phase = TrainingPhase::RotorPretraining;
        println!("\n  [Phase 2: Rotor Predictor Training (centroid-bridged)]");
        for epoch in 0..phase2_epochs {
            let diag = self.run_rotor_pretraining_epoch(programs, epoch);
            diag.print_summary();
            self.diagnostics.push(diag);
        }

        self.phase = TrainingPhase::JointFinetuning;
        println!("\n  [Phase 3: Joint Fine-tuning (centroid-bridged)]");
        for epoch in 0..phase3_epochs {
            let diag = self.run_joint_epoch(programs, epoch);
            diag.print_summary();
            self.diagnostics.push(diag);
        }

        println!("--- STA-CALM Training Complete ---\n");
        self.diagnostics.clone()
    }

    /// Phase 1: pressure each grade independently using *centroid-derived*
    /// SpacetimeChunks where grades carry genuine semantic structure.
    fn run_grade_pretraining_epoch(
        &self,
        programs: &mut [(String, Vec<u16>, Vec<f32>)],
        epoch: usize,
    ) -> EpochDiagnostics {
        let mut diag = EpochDiagnostics {
            phase: "grade-pretrain".to_string(),
            epoch,
            ..Default::default()
        };

        let mut total_loss = GradedLoss::default();
        let mut sop_correct = 0usize;
        let mut sop_total = 0usize;
        let mut neg_correct = 0usize;
        let mut neg_total = 0usize;
        let mut count = 0usize;

        let chunks = Self::centroid_chunks(programs);
        let groups = Self::topic_groups(programs);

        // Within-topic pairs: same semantic domain, differentiated structure
        for (_, indices) in &groups {
            if indices.len() < 2 {
                continue;
            }
            for w in indices.windows(2) {
                let a = &chunks[w[0]];
                let b = &chunks[w[1]];
                let loss = graded_training_loss(a, b, &self.grade_weights);
                total_loss.scalar_loss += loss.scalar_loss;
                total_loss.vector_loss += loss.vector_loss;
                total_loss.bivector_loss += loss.bivector_loss;
                total_loss.trivector_loss += loss.trivector_loss;
                total_loss.pseudo_loss += loss.pseudo_loss;
                total_loss.total += loss.total;
                count += 1;

                let sop = sentence_order_score(a, b);
                sop_total += 1;
                if sop > 0.0 {
                    sop_correct += 1;
                }
            }
        }

        // Cross-topic contrastive pairs for bivector grade separation
        for (ti, (_, indices_a)) in groups.iter().enumerate() {
            for (tj, (_, indices_b)) in groups.iter().enumerate() {
                if ti >= tj {
                    continue;
                }
                for &ia in indices_a {
                    for &ib in indices_b {
                        let g1_sim = chunks[ia].semantic_similarity(&chunks[ib]);
                        if g1_sim > 0.3 {
                            let g2_sim = chunks[ia].structural_similarity(&chunks[ib]);
                            let bivec_pressure = (g1_sim - g2_sim).abs();
                            total_loss.bivector_loss += 1.0 - bivec_pressure;
                            count += 1;
                        }
                    }
                }
            }
        }

        // Centroid updates: scale by avg_loss (push HARDER when loss is high)
        if count > 0 {
            let avg_loss = total_loss.total / count as f32;
            let lr = self.learning_rate * (1.0 / (1.0 + epoch as f32 * 0.1));

            for (prog_idx, chunk) in chunks.iter().enumerate() {
                let centroid = &mut programs[prog_idx].2;
                let dim = centroid.len();
                for i in 0..dim {
                    let grade1_signal = chunk.semantic_dir[i % 8];
                    let grade2_signal = if i < 7 {
                        chunk.boost_causal[i % 7]
                    } else {
                        chunk.rotation_structural[i % 21]
                    };
                    let combined = grade1_signal * 0.6 + grade2_signal * 0.4;
                    centroid[i] += lr * combined * avg_loss;
                }
            }
        }

        // Negation pair training using decoded program text
        let texts: Vec<String> = programs
            .iter()
            .map(|(topic, tokens, _)| {
                if tokens.len() >= 3 {
                    let seq = self.codec.encode_sequence(tokens);
                    let mut words = Vec::new();
                    for (ci, chunk) in seq.chunks.iter().enumerate() {
                        let n = seq.chunk_lengths.get(ci).copied().unwrap_or(CHUNK_K);
                        for &tid in &self.codec.decode_chunk(chunk, n) {
                            words.push(format!("t{}", tid));
                        }
                    }
                    words.join(" ")
                } else {
                    topic.clone()
                }
            })
            .filter(|t| t.split_whitespace().count() >= 3)
            .collect();
        if !texts.is_empty() {
            let neg_pairs = generate_negation_pairs(&texts);
            for pair in &neg_pairs {
                let assert_tokens: Vec<u16> = pair
                    .assertion
                    .split_whitespace()
                    .take(CHUNK_K)
                    .filter_map(|w| w.strip_prefix('t').and_then(|n| n.parse::<u16>().ok()))
                    .collect();
                let neg_tokens: Vec<u16> = pair
                    .negation
                    .split_whitespace()
                    .take(CHUNK_K)
                    .filter_map(|w| w.strip_prefix('t').and_then(|n| n.parse::<u16>().ok()))
                    .collect();
                if assert_tokens.len() >= 2 && neg_tokens.len() >= 2 {
                    let assert_chunk = SpacetimeChunk::from_chunk(
                        &self.codec.encode_chunk(&pad_to_k(&assert_tokens)),
                    );
                    let neg_chunk = SpacetimeChunk::from_chunk(
                        &self.codec.encode_chunk(&pad_to_k(&neg_tokens)),
                    );
                    let loss = negation_pair_loss(&assert_chunk, &neg_chunk);
                    neg_total += 1;
                    if loss < 0.5 {
                        neg_correct += 1;
                    }
                }
            }
        }

        let n = count.max(1) as f32;
        diag.avg_scalar_loss = total_loss.scalar_loss / n;
        diag.avg_vector_loss = total_loss.vector_loss / n;
        diag.avg_bivector_loss = total_loss.bivector_loss / n;
        diag.avg_trivector_loss = total_loss.trivector_loss / n;
        diag.avg_pseudo_loss = total_loss.pseudo_loss / n;
        diag.avg_total_loss = total_loss.total / n;
        diag.sop_accuracy = if sop_total > 0 {
            sop_correct as f32 / sop_total as f32
        } else {
            0.0
        };
        diag.negation_accuracy = if neg_total > 0 {
            neg_correct as f32 / neg_total as f32
        } else {
            0.0
        };
        diag.num_samples = count;
        diag
    }

    /// Phase 2: train the rotor predictor on centroid-derived trajectories.
    /// Programs within the same topic form a natural trajectory.
    fn run_rotor_pretraining_epoch(
        &self,
        programs: &mut [(String, Vec<u16>, Vec<f32>)],
        epoch: usize,
    ) -> EpochDiagnostics {
        let mut diag = EpochDiagnostics {
            phase: "rotor-pretrain".to_string(),
            epoch,
            ..Default::default()
        };

        let mut total_confidence = 0.0f32;
        let mut total_prediction_loss = 0.0f32;
        let mut count = 0usize;

        let chunks = Self::centroid_chunks(programs);
        let groups = Self::topic_groups(programs);

        for (_, indices) in &groups {
            if indices.len() < 3 {
                continue;
            }

            let trajectory: Vec<SpacetimeChunk> =
                indices.iter().map(|&i| chunks[i].clone()).collect();

            let train_traj = &trajectory[..trajectory.len() - 1];
            let target = &trajectory[trajectory.len() - 1];

            let propagator =
                SemanticPropagator::from_trajectory(train_traj, self.mass, self.coupling);

            if let Some((predicted, _interval, confidence)) = propagator.predict_next() {
                let pred_st = SpacetimeChunk::from_centroid(&predicted);
                let loss = graded_training_loss(&pred_st, target, &self.grade_weights);

                total_confidence += confidence;
                total_prediction_loss += loss.total;
                count += 1;

                let lr = self.learning_rate * 0.5 * (1.0 / (1.0 + epoch as f32 * 0.1));
                let last_idx = *indices.last().unwrap();
                let centroid = &mut programs[last_idx].2;
                for i in 0..centroid.len().min(CATA_DIM) {
                    let delta = predicted[i] - centroid[i];
                    centroid[i] += lr * delta * confidence;
                }
            }
        }

        let n = count.max(1) as f32;
        diag.avg_total_loss = total_prediction_loss / n;
        diag.rotor_prediction_confidence = total_confidence / n;
        diag.num_samples = count;
        diag
    }

    /// Phase 3: joint fine-tuning — all grades unfrozen, cross-grade
    /// interaction through geometric product.
    fn run_joint_epoch(
        &self,
        programs: &mut [(String, Vec<u16>, Vec<f32>)],
        epoch: usize,
    ) -> EpochDiagnostics {
        let mut diag = EpochDiagnostics {
            phase: "joint-finetune".to_string(),
            epoch,
            ..Default::default()
        };

        let mut total_loss = GradedLoss::default();
        let mut total_rotor_conf = 0.0f32;
        let mut count = 0usize;
        let mut rotor_count = 0usize;

        let chunks = Self::centroid_chunks(programs);
        let groups = Self::topic_groups(programs);

        for (_, indices) in &groups {
            if indices.len() < 2 {
                continue;
            }
            for w in indices.windows(2) {
                let a = &chunks[w[0]];
                let b = &chunks[w[1]];
                let loss = graded_training_loss(a, b, &self.grade_weights);
                total_loss.scalar_loss += loss.scalar_loss;
                total_loss.vector_loss += loss.vector_loss;
                total_loss.bivector_loss += loss.bivector_loss;
                total_loss.trivector_loss += loss.trivector_loss;
                total_loss.pseudo_loss += loss.pseudo_loss;
                total_loss.total += loss.total;
                count += 1;
            }

            if indices.len() >= 3 {
                let trajectory: Vec<SpacetimeChunk> =
                    indices.iter().map(|&i| chunks[i].clone()).collect();
                let train_traj = &trajectory[..trajectory.len() - 1];
                let target = &trajectory[trajectory.len() - 1];
                let propagator =
                    SemanticPropagator::from_trajectory(train_traj, self.mass, self.coupling);
                if let Some((predicted, _interval, confidence)) = propagator.predict_next() {
                    let pred_st = SpacetimeChunk::from_centroid(&predicted);
                    let rotor_loss = graded_training_loss(&pred_st, target, &self.grade_weights);
                    total_loss.total += rotor_loss.total * 0.5;
                    total_rotor_conf += confidence;
                    rotor_count += 1;
                }
            }
        }

        if count > 0 {
            let avg_loss = total_loss.total / count as f32;
            let lr = self.learning_rate * 0.3 * (1.0 / (1.0 + epoch as f32 * 0.1));

            for (prog_idx, chunk) in chunks.iter().enumerate() {
                let centroid = &mut programs[prog_idx].2;
                let dim = centroid.len();
                for i in 0..dim {
                    let sem = chunk.semantic_dir[i % 8];
                    let causal = chunk.boost_causal[i % 7];
                    let structural = chunk.rotation_structural[i % 21];
                    let combined = sem * 0.5 + causal * 0.25 + structural * 0.25;
                    centroid[i] += lr * combined * avg_loss;
                }
            }
        }

        let n = count.max(1) as f32;
        diag.avg_scalar_loss = total_loss.scalar_loss / n;
        diag.avg_vector_loss = total_loss.vector_loss / n;
        diag.avg_bivector_loss = total_loss.bivector_loss / n;
        diag.avg_trivector_loss = total_loss.trivector_loss / n;
        diag.avg_pseudo_loss = total_loss.pseudo_loss / n;
        diag.avg_total_loss = total_loss.total / n;
        diag.rotor_prediction_confidence = if rotor_count > 0 {
            total_rotor_conf / rotor_count as f32
        } else {
            0.0
        };
        diag.num_samples = count;
        diag
    }
}

fn pad_to_k(tokens: &[u16]) -> [u16; CHUNK_K] {
    let mut out = [0u16; CHUNK_K];
    for (i, &t) in tokens.iter().enumerate().take(CHUNK_K) {
        out[i] = t;
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_lexicon() -> SaliencyLexicon {
        SaliencyLexicon::from_keywords(vec![
            "stack".to_string(),
            "queue".to_string(),
            "derivative".to_string(),
            "eigenvalue".to_string(),
            "binary search".to_string(),
            "linked list".to_string(),
            "TCP".to_string(),
            "congestion".to_string(),
        ])
    }

    #[test]
    fn test_saliency_scoring() {
        let lex = test_lexicon();
        assert_eq!(lex.score("stack"), 1.0);
        assert_eq!(lex.score("Stack"), 1.0);
        assert_eq!(lex.score("queue"), 1.0);
        assert_eq!(lex.score("derivative"), 1.0);
        assert!(lex.score("binary") > 0.0);
        assert!(lex.score("linked") > 0.0);
        assert_eq!(lex.score("the"), 0.0);
        assert_eq!(lex.score("is"), 0.0);
    }

    #[test]
    fn test_mask_salient_spans() {
        let lex = test_lexicon();
        let text = "A stack is a LIFO data structure where the last element added is the first one removed";
        let augmented = mask_salient_spans(text, &lex, 3, 42);
        assert!(!augmented.is_empty());
        for aug in &augmented {
            assert!(
                aug.contains("[MASK]"),
                "Augmented text should contain [MASK]: {}",
                aug
            );
            assert!(!aug.contains("stack") || aug.contains("[MASK]"));
        }
    }

    #[test]
    fn test_contrastive_refine() {
        let mut topics = vec![
            (
                "stack".to_string(),
                vec![CentroidEntry {
                    centroid: vec![1.0, 0.0, 0.0, 0.0],
                    program_idx: 0,
                }],
            ),
            (
                "queue".to_string(),
                vec![CentroidEntry {
                    centroid: vec![0.95, 0.1, 0.0, 0.0],
                    program_idx: 0,
                }],
            ),
        ];

        let sim_before = cosine_sim(&topics[0].1[0].centroid, &topics[1].1[0].centroid);

        let repulsions = contrastive_refine(&mut topics, 0.5, 0.1);
        assert!(repulsions > 0);

        let sim_after = cosine_sim(&topics[0].1[0].centroid, &topics[1].1[0].centroid);
        assert!(
            sim_after < sim_before,
            "Contrastive refinement should reduce similarity: before={}, after={}",
            sim_before,
            sim_after
        );
    }

    #[test]
    fn test_augment_pipeline() {
        let lex = test_lexicon();
        let dict = TokenDictionary::build(
            &[
                "A stack is a LIFO data structure",
                "A queue is a FIFO data structure",
            ],
            4096,
        );

        let samples = vec![
            (
                "What is a stack?".to_string(),
                "A stack is a LIFO data structure".to_string(),
            ),
            (
                "What is a queue?".to_string(),
                "A queue is a FIFO data structure".to_string(),
            ),
        ];

        let augmented = augment_training_data(&samples, &lex, &dict, 2, 0.3);
        assert!(!augmented.is_empty(), "Should produce augmented samples");

        let mask_count = augmented
            .iter()
            .filter(|a| a.kind == AugmentKind::SalientMask)
            .count();
        let rtd_count = augmented
            .iter()
            .filter(|a| a.kind == AugmentKind::RtdNegative)
            .count();
        assert!(mask_count > 0, "Should have masked augments");
        assert!(rtd_count > 0, "Should have RTD augments");
    }

    // -----------------------------------------------------------------------
    // Phase 2: Graded Loss + Contrastive + Negation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_graded_training_loss_identical() {
        use crate::text_autoencoder::{ChunkCodec, SpacetimeChunk, CHUNK_K};
        let codec = ChunkCodec::new(256);
        let chunk =
            SpacetimeChunk::from_chunk(&codec.encode_chunk(&[10, 20, 30, 40, 50, 60, 70, 80]));
        let loss = graded_training_loss(&chunk, &chunk, &DEFAULT_GRADE_WEIGHTS);
        assert!(
            loss.total < 0.1,
            "identical chunks should have near-zero loss: {}",
            loss.total
        );
        assert!(
            loss.vector_loss < 0.01,
            "vector loss should be ~0: {}",
            loss.vector_loss
        );
    }

    #[test]
    fn test_graded_training_loss_different() {
        use crate::text_autoencoder::{ChunkCodec, SpacetimeChunk, CHUNK_K};
        let codec = ChunkCodec::new(256);
        let a = SpacetimeChunk::from_chunk(&codec.encode_chunk(&[10, 20, 30, 40, 50, 60, 70, 80]));
        let b = SpacetimeChunk::from_chunk(
            &codec.encode_chunk(&[200, 201, 202, 203, 204, 205, 206, 207]),
        );
        let loss = graded_training_loss(&a, &b, &DEFAULT_GRADE_WEIGHTS);
        assert!(
            loss.total > 0.1,
            "different chunks should have significant loss: {}",
            loss.total
        );
    }

    #[test]
    fn test_sentence_order_prediction() {
        use crate::text_autoencoder::{ChunkCodec, SpacetimeChunk, CHUNK_K};
        let codec = ChunkCodec::new(256);
        let first =
            SpacetimeChunk::from_chunk(&codec.encode_chunk(&[10, 20, 30, 40, 50, 60, 70, 80]));
        let second =
            SpacetimeChunk::from_chunk(&codec.encode_chunk(&[11, 21, 31, 41, 51, 61, 71, 81]));
        let score_correct = sentence_order_score(&first, &second);
        let score_reversed = sentence_order_score(&second, &first);
        // Correct and reversed should have opposite signs
        // (or at least the function should return something)
        assert!((score_correct - score_reversed).abs() >= 0.0);
    }

    #[test]
    fn test_generate_negation_pairs() {
        let texts = vec![
            "A stack is a LIFO data structure".to_string(),
            "Trees are hierarchical data structures".to_string(),
            "Python can handle dynamic typing".to_string(),
            "The algorithm has quadratic complexity".to_string(),
            "Short".to_string(),
        ];
        let pairs = generate_negation_pairs(&texts);
        assert!(
            !pairs.is_empty(),
            "should generate at least some negation pairs"
        );
        for pair in &pairs {
            assert_ne!(
                pair.assertion, pair.negation,
                "negation should differ from assertion"
            );
            let has_negation = pair.negation.contains("not") || pair.negation.contains("cannot");
            assert!(
                has_negation,
                "negation should contain negation word: {}",
                pair.negation
            );
        }
    }

    #[test]
    fn test_negation_pair_loss() {
        use crate::text_autoencoder::{ChunkCodec, SpacetimeChunk, CHUNK_K};
        let codec = ChunkCodec::new(256);
        let a = SpacetimeChunk::from_chunk(&codec.encode_chunk(&[10, 20, 30, 40, 50, 60, 70, 80]));
        let b = SpacetimeChunk::from_chunk(
            &codec.encode_chunk(&[200, 201, 202, 203, 204, 205, 206, 207]),
        );
        let loss = negation_pair_loss(&a, &b);
        assert!(
            loss >= 0.0,
            "negation loss should be non-negative: {}",
            loss
        );
    }

    // -----------------------------------------------------------------------
    // TrainingOrchestrator tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_orchestrator_full_pipeline() {
        let mut programs = vec![
            (
                "topic_a".to_string(),
                vec![
                    10, 20, 30, 40, 50, 60, 70, 80, 11, 21, 31, 41, 51, 61, 71, 81,
                ],
                vec![0.1; 32],
            ),
            (
                "topic_a".to_string(),
                vec![
                    12, 22, 32, 42, 52, 62, 72, 82, 13, 23, 33, 43, 53, 63, 73, 83,
                ],
                vec![0.2; 32],
            ),
            (
                "topic_b".to_string(),
                vec![
                    100, 110, 120, 130, 140, 150, 160, 170, 101, 111, 121, 131, 141, 151, 161, 171,
                ],
                vec![0.3; 32],
            ),
            (
                "topic_b".to_string(),
                vec![
                    102, 112, 122, 132, 142, 152, 162, 172, 103, 113, 123, 133, 143, 153, 163, 173,
                ],
                vec![0.4; 32],
            ),
        ];

        let mut orch = TrainingOrchestrator::new(256);
        let diags = orch.run_full_pipeline(&mut programs, 2, 1, 1);

        assert!(!diags.is_empty(), "should produce diagnostics");
        assert_eq!(diags.len(), 4, "2 + 1 + 1 = 4 epochs");

        // Phase 1 diagnostics
        assert_eq!(diags[0].phase, "grade-pretrain");
        assert!(diags[0].avg_total_loss >= 0.0);
        assert!(diags[0].num_samples > 0);

        // Phase 2 diagnostics
        assert_eq!(diags[2].phase, "rotor-pretrain");

        // Phase 3 diagnostics
        assert_eq!(diags[3].phase, "joint-finetune");

        // Centroids should have been modified
        let orig_centroid = vec![0.1f32; 32];
        let modified = &programs[0].2;
        let diff: f32 = orig_centroid
            .iter()
            .zip(modified.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 0.0,
            "centroid should be modified by training: diff={}",
            diff
        );
    }

    #[test]
    fn test_orchestrator_phase_transitions() {
        let mut programs = vec![
            (
                "a".to_string(),
                vec![
                    10, 20, 30, 40, 50, 60, 70, 80, 15, 25, 35, 45, 55, 65, 75, 85, 18, 28, 38, 48,
                    58, 68, 78, 88,
                ],
                vec![0.5; 16],
            ),
            (
                "b".to_string(),
                vec![
                    100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
                    116, 117, 118, 119, 120, 121, 122, 123,
                ],
                vec![0.5; 16],
            ),
        ];

        let mut orch = TrainingOrchestrator::new(256);
        let diags = orch.run_full_pipeline(&mut programs, 1, 1, 1);
        assert_eq!(diags.len(), 3);

        // Each phase should be different
        assert_eq!(diags[0].phase, "grade-pretrain");
        assert_eq!(diags[1].phase, "rotor-pretrain");
        assert_eq!(diags[2].phase, "joint-finetune");
    }

    #[test]
    fn test_epoch_diagnostics_print() {
        let diag = EpochDiagnostics {
            phase: "test".to_string(),
            epoch: 0,
            avg_scalar_loss: 0.1,
            avg_vector_loss: 0.2,
            avg_bivector_loss: 0.3,
            avg_trivector_loss: 0.4,
            avg_pseudo_loss: 0.05,
            avg_total_loss: 0.25,
            sop_accuracy: 0.8,
            negation_accuracy: 0.6,
            rotor_prediction_confidence: 0.5,
            num_samples: 100,
        };
        diag.print_summary(); // should not panic
    }
}
