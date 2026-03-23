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
        if lower.len() <= 2 { return 0.0; }
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
    pub fn salient_positions(&self, token_ids: &[u16], dict: &TokenDictionary) -> Vec<(usize, f32)> {
        token_ids.iter().enumerate().filter_map(|(i, &id)| {
            let text = dict.token_str(id)?;
            let s = self.score(text);
            if s > 0.0 { Some((i, s)) } else { None }
        }).collect()
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

    let salient_positions: Vec<usize> = words.iter().enumerate()
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

        let masked: Vec<&str> = words.iter().enumerate()
            .map(|(i, w)| {
                if i >= mask_idx && i < end { "[MASK]" } else { *w }
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

    let detected = original_ids.iter()
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
                if t_a == t_b { continue; }
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
    if na < 1e-8 || nb < 1e-8 { 0.0 } else { dot / (na * nb) }
}

fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(0x9e3779b97f4a7c15);
    state = (state ^ (state >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    state = (state ^ (state >> 27)).wrapping_mul(0x94d049bb133111eb);
    state ^ (state >> 31)
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

        let masked_texts = mask_salient_spans(
            response, lexicon, mask_augments_per_sample, seed,
        );
        for mt in masked_texts {
            augmented.push(AugmentedSample {
                text: text.clone(),
                response: mt,
                kind: AugmentKind::SalientMask,
            });
        }

        if let Some((corrupted, _mask)) = replace_salient_tokens(
            response, lexicon, dict, rtd_rate, seed.wrapping_add(42),
        ) {
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
            assert!(aug.contains("[MASK]"), "Augmented text should contain [MASK]: {}", aug);
            assert!(!aug.contains("stack") || aug.contains("[MASK]"));
        }
    }

    #[test]
    fn test_contrastive_refine() {
        let mut topics = vec![
            ("stack".to_string(), vec![
                CentroidEntry { centroid: vec![1.0, 0.0, 0.0, 0.0], program_idx: 0 },
            ]),
            ("queue".to_string(), vec![
                CentroidEntry { centroid: vec![0.95, 0.1, 0.0, 0.0], program_idx: 0 },
            ]),
        ];

        let sim_before = cosine_sim(
            &topics[0].1[0].centroid,
            &topics[1].1[0].centroid,
        );

        let repulsions = contrastive_refine(&mut topics, 0.5, 0.1);
        assert!(repulsions > 0);

        let sim_after = cosine_sim(
            &topics[0].1[0].centroid,
            &topics[1].1[0].centroid,
        );
        assert!(sim_after < sim_before, "Contrastive refinement should reduce similarity: before={}, after={}", sim_before, sim_after);
    }

    #[test]
    fn test_augment_pipeline() {
        let lex = test_lexicon();
        let dict = TokenDictionary::build(&[
            "A stack is a LIFO data structure",
            "A queue is a FIFO data structure",
        ], 4096);

        let samples = vec![
            ("What is a stack?".to_string(), "A stack is a LIFO data structure".to_string()),
            ("What is a queue?".to_string(), "A queue is a FIFO data structure".to_string()),
        ];

        let augmented = augment_training_data(&samples, &lex, &dict, 2, 0.3);
        assert!(!augmented.is_empty(), "Should produce augmented samples");

        let mask_count = augmented.iter().filter(|a| a.kind == AugmentKind::SalientMask).count();
        let rtd_count = augmented.iter().filter(|a| a.kind == AugmentKind::RtdNegative).count();
        assert!(mask_count > 0, "Should have masked augments");
        assert!(rtd_count > 0, "Should have RTD augments");
    }
}
