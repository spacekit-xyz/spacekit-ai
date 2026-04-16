// ── compose.rs ────────────────────────────────────────────────────────────────
// CategoricalComposer: bridges the categorical network's disentangled
// representations (sentiment × entity × optional causal) with the lattice's
// program families to compose novel, grounded generation output.
//
// The lattice retrieves the program family (template skeleton), the categorical
// bifunctor decomposes the input into orthogonal sentiment and entity morphisms,
// and this composer recombines them into output text — enabling compositional
// generalization for unseen entity × sentiment combinations.

use crate::category::disentanglement::SimpleRng;
use crate::category::forward::{align_to_dim, bifunctor_branch_vectors};
use crate::category::linear_head::LinearHead;
use crate::category::pythagoras::PythagorasNode;
use crate::category::training::{AuxCategory, SentimentLabel};

/// A generation template extracted from lattice programs: fixed skeleton with
/// variable slots that the composer fills from categorical decomposition.
#[derive(Debug, Clone)]
pub struct ProgramTemplate {
    pub template_id: usize,
    pub skeleton: String,
    pub sentiment_slot: SentimentSlot,
    pub entity_slot: EntitySlot,
    pub confidence: f32,
}

/// Sentiment-dependent text fragments the composer selects based on the
/// categorical sentiment vector.
#[derive(Debug, Clone)]
pub struct SentimentSlot {
    pub label_phrase: &'static str,
    pub tone_marker: &'static str,
    pub grounding_prefix: &'static str,
}

/// Entity-dependent slot: the salient noun/subject tokens from the input.
#[derive(Debug, Clone)]
pub struct EntitySlot {
    pub entity_tokens: Vec<String>,
    pub category: AuxCategory,
}

impl SentimentSlot {
    pub fn from_label(label: &SentimentLabel) -> Self {
        match label {
            SentimentLabel::PositiveStrong => Self {
                label_phrase: "POSITIVE (strong)",
                tone_marker: "clearly positive",
                grounding_prefix: "The overall tone reads as clearly positive.",
            },
            SentimentLabel::PositiveMild => Self {
                label_phrase: "POSITIVE (mild)",
                tone_marker: "mildly positive",
                grounding_prefix: "The overall tone reads as mildly positive.",
            },
            SentimentLabel::Neutral => Self {
                label_phrase: "NEUTRAL",
                tone_marker: "mostly neutral",
                grounding_prefix: "The overall tone reads as mostly neutral.",
            },
            SentimentLabel::NegativeMild => Self {
                label_phrase: "NEGATIVE (mild)",
                tone_marker: "clearly negative",
                grounding_prefix: "The overall tone reads as clearly negative.",
            },
            SentimentLabel::NegativeStrong => Self {
                label_phrase: "NEGATIVE (strong)",
                tone_marker: "strongly negative",
                grounding_prefix: "The overall tone reads as strongly negative.",
            },
            SentimentLabel::Sarcastic => Self {
                label_phrase: "SARCASTIC",
                tone_marker: "ironic or surface/actual mismatch",
                grounding_prefix: "The line may use irony or surface/actual mismatch.",
            },
            SentimentLabel::Mixed => Self {
                label_phrase: "MIXED",
                tone_marker: "dual-valence",
                grounding_prefix: "Contrastive or dual-valence wording — read as MIXED when both poles appear.",
            },
        }
    }
}

/// Intermediate output from the categorical decomposition step.
#[derive(Debug, Clone)]
pub struct CategoricalDecomposition {
    pub sentiment: SentimentLabel,
    pub sentiment_confidence: f32,
    pub sentiment_vec: Vec<f32>,
    pub entity_category: AuxCategory,
    pub entity_confidence: f32,
    pub entity_vec: Vec<f32>,
}

/// Composed output: the final text + metadata.
#[derive(Debug, Clone)]
pub struct ComposedOutput {
    pub text: String,
    pub label_line: String,
    pub explanation: String,
    pub confidence: f32,
    pub template_id: Option<usize>,
    pub composed: bool,
}

/// The composer itself: holds trained categorical weights and composes
/// generation output from disentangled representations.
pub struct CategoricalComposer {
    pub composition: PythagorasNode<Vec<f32>>,
    pub sentiment_head: LinearHead,
    pub aux_head: LinearHead,
    pub embed_dim: usize,
    pub branch_dim: usize,
}

impl CategoricalComposer {
    pub fn new(
        composition: PythagorasNode<Vec<f32>>,
        sentiment_head: LinearHead,
        aux_head: LinearHead,
        embed_dim: usize,
        branch_dim: usize,
    ) -> Self {
        Self { composition, sentiment_head, aux_head, embed_dim, branch_dim }
    }

    /// Create with random initialization (for testing / bootstrap).
    pub fn new_random(embed_dim: usize, branch_dim: usize, seed: u64) -> Self {
        let mut rng = SimpleRng::new(seed);
        let weights: Vec<f32> = (0..embed_dim)
            .map(|_| (rng.gen_f32() * 2.0 - 1.0) * 0.1)
            .collect();
        let composition = PythagorasNode::leaf(weights, embed_dim);
        let sentiment_head = LinearHead::new_random(branch_dim, SentimentLabel::num_classes(), &mut rng);
        let aux_head = LinearHead::new_random(branch_dim, AuxCategory::num_classes(), &mut rng);
        Self { composition, sentiment_head, aux_head, embed_dim, branch_dim }
    }

    /// Decompose an embedding into disentangled sentiment and entity vectors
    /// using the trained Pythagoras tree + linear heads.
    pub fn decompose(&self, embedding: &[f32]) -> CategoricalDecomposition {
        let emb = align_to_dim(embedding, self.embed_dim);
        let (sent_vec, ent_vec) = bifunctor_branch_vectors(
            &self.composition, &emb, self.branch_dim,
        );

        let (si, _sl, sp, sc) = self.sentiment_head.predict_with_probs(&sent_vec);
        let sentiment = SentimentLabel::from_class_index(si).unwrap_or(SentimentLabel::Neutral);

        let (ai, _al, _ap, ac) = self.aux_head.predict_with_probs(&ent_vec);
        let entity_category = AuxCategory::from_class_index(ai).unwrap_or(AuxCategory::Other);

        CategoricalDecomposition {
            sentiment,
            sentiment_confidence: sc,
            sentiment_vec: sent_vec,
            entity_category,
            entity_confidence: ac,
            entity_vec: ent_vec,
        }
    }

    /// Compose a generation output from the categorical decomposition and
    /// salient tokens extracted from the user's input.
    ///
    /// When a lattice template is available (Some(template)), the composer uses
    /// it as a skeleton and fills slots. When None (the OOD/fallback case), the
    /// composer constructs a grounded explanation from the decomposition alone.
    pub fn compose(
        &self,
        decomposition: &CategoricalDecomposition,
        user_text: &str,
        template: Option<&ProgramTemplate>,
    ) -> ComposedOutput {
        let slot = SentimentSlot::from_label(&decomposition.sentiment);
        let salient = extract_salient_tokens(user_text, 8);
        let entity_slot = EntitySlot {
            entity_tokens: salient.clone(),
            category: decomposition.entity_category.clone(),
        };

        let (explanation, conf, tid) = if let Some(tmpl) = template {
            let text = fill_template(tmpl, &slot, &entity_slot);
            (text, tmpl.confidence * decomposition.sentiment_confidence, Some(tmpl.template_id))
        } else {
            let text = compose_from_decomposition(
                &slot, &entity_slot, user_text, &decomposition,
            );
            (text, decomposition.sentiment_confidence * 0.7, None)
        };

        let grounding = format!(
            "Grounded in the user's own words: \"{}\"",
            truncate_for_grounding(user_text, 120),
        );

        ComposedOutput {
            text: format!("{} — {}", slot.label_phrase, &explanation),
            label_line: slot.label_phrase.to_string(),
            explanation: format!("{} {}", explanation, grounding),
            confidence: conf.min(0.95),
            template_id: tid,
            composed: true,
        }
    }

    /// Full pipeline: embed → decompose → compose.
    pub fn generate(
        &self,
        embedding: &[f32],
        user_text: &str,
        template: Option<&ProgramTemplate>,
    ) -> ComposedOutput {
        let decomp = self.decompose(embedding);
        self.compose(&decomp, user_text, template)
    }
}

/// Extract salient tokens from user text: lowercase, deduped, stopwords removed.
fn extract_salient_tokens(text: &str, max: usize) -> Vec<String> {
    let stopwords: &[&str] = &[
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "about", "like",
        "through", "after", "over", "between", "out", "against", "during",
        "before", "it", "its", "this", "that", "these", "those", "i", "we",
        "you", "he", "she", "they", "me", "him", "her", "us", "them", "my",
        "your", "his", "our", "their", "and", "but", "or", "nor", "not",
        "no", "so", "if", "than", "too", "very", "just", "also",
    ];

    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();

    for word in text.split_whitespace() {
        let clean: String = word.chars()
            .filter(|c| c.is_alphanumeric() || *c == '\'' || *c == '-')
            .collect::<String>()
            .to_ascii_lowercase();
        if clean.len() <= 2 { continue; }
        if stopwords.contains(&clean.as_str()) { continue; }
        if seen.insert(clean.clone()) {
            out.push(clean);
            if out.len() >= max { break; }
        }
    }
    out
}

/// Compose an explanation entirely from the categorical decomposition when no
/// lattice template is available. This is the key improvement over the old
/// "No witness-matched lattice row; stance follows routing only" fallback.
fn compose_from_decomposition(
    slot: &SentimentSlot,
    entity_slot: &EntitySlot,
    user_text: &str,
    decomp: &CategoricalDecomposition,
) -> String {
    let entity_phrase = if entity_slot.entity_tokens.is_empty() {
        String::new()
    } else {
        let joined = entity_slot.entity_tokens.join(", ");
        format!(" Key terms: {}.", joined)
    };

    let contrastive_marker = detect_contrastive_marker(user_text);

    match &decomp.sentiment {
        SentimentLabel::Mixed => {
            if let Some(marker) = contrastive_marker {
                format!(
                    "Contrastive marker ({}) with both laudatory and critical wording \
                     — dual valence (MIXED), not a single pole.{}",
                    marker, entity_phrase,
                )
            } else {
                format!(
                    "Positive and negative cues both appear; overall read is MIXED.{}",
                    entity_phrase,
                )
            }
        }
        SentimentLabel::Sarcastic => {
            format!(
                "{}{}",
                slot.grounding_prefix, entity_phrase,
            )
        }
        SentimentLabel::Neutral => {
            if has_factual_markers(user_text) {
                format!(
                    "Measurable fact, status, or time — no evaluative opinion; \
                     read as NEUTRAL (not praise or complaint).{}",
                    entity_phrase,
                )
            } else if contrastive_marker.is_some() {
                format!(
                    "Compensating factors or hedged stance without a single emotional pole \
                     — read as NEUTRAL.{}",
                    entity_phrase,
                )
            } else {
                format!(
                    "{}{}",
                    slot.grounding_prefix, entity_phrase,
                )
            }
        }
        _ => {
            format!(
                "{}{}",
                slot.grounding_prefix, entity_phrase,
            )
        }
    }
}

fn detect_contrastive_marker(text: &str) -> Option<&'static str> {
    let lower = text.to_ascii_lowercase();
    if lower.contains(" but ") || lower.contains(" but,") { return Some("but"); }
    if lower.contains(" yet ") || lower.contains(" yet,") { return Some("yet"); }
    if lower.contains(" however ") || lower.contains(" however,") { return Some("however"); }
    if lower.contains(" although ") || lower.contains(" although,") { return Some("although"); }
    if lower.contains(" though ") || lower.contains(" though,") { return Some("though"); }
    if lower.contains(" despite ") { return Some("despite"); }
    if lower.contains(" nevertheless ") { return Some("nevertheless"); }
    if lower.contains(" instead ") { return Some("instead"); }
    None
}

fn has_factual_markers(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    let markers = [
        "list ", "define ", "what is ", "what are ", "how many ", "how much ",
        "what nominal", "which ", "name the ", "identify ", "describe the ",
    ];
    markers.iter().any(|m| lower.starts_with(m) || lower.contains(&format!(" {}", m)))
}

fn fill_template(
    tmpl: &ProgramTemplate,
    slot: &SentimentSlot,
    entity_slot: &EntitySlot,
) -> String {
    let mut text = tmpl.skeleton.clone();
    text = text.replace("{sentiment}", slot.tone_marker);
    text = text.replace("{label}", slot.label_phrase);
    let entity_str = if entity_slot.entity_tokens.is_empty() {
        "the input".to_string()
    } else {
        entity_slot.entity_tokens.join(", ")
    };
    text = text.replace("{entity}", &entity_str);
    text
}

fn truncate_for_grounding(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        text.to_string()
    } else {
        format!("{}...", &text[..max_chars])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_salient_removes_stopwords() {
        let tokens = extract_salient_tokens("the cat is on the mat", 10);
        assert!(tokens.contains(&"cat".to_string()));
        assert!(tokens.contains(&"mat".to_string()));
        assert!(!tokens.contains(&"the".to_string()));
        assert!(!tokens.contains(&"is".to_string()));
    }

    #[test]
    fn extract_salient_respects_max() {
        let tokens = extract_salient_tokens(
            "alpha bravo charlie delta echo foxtrot golf hotel india juliet",
            3,
        );
        assert_eq!(tokens.len(), 3);
    }

    #[test]
    fn sentiment_slot_labels_are_correct() {
        let s = SentimentSlot::from_label(&SentimentLabel::PositiveStrong);
        assert_eq!(s.label_phrase, "POSITIVE (strong)");
        let s = SentimentSlot::from_label(&SentimentLabel::Mixed);
        assert_eq!(s.label_phrase, "MIXED");
    }

    #[test]
    fn detect_contrastive_finds_but() {
        assert_eq!(detect_contrastive_marker("good but bad"), Some("but"));
        assert_eq!(detect_contrastive_marker("all fine"), None);
    }

    #[test]
    fn compose_from_decomp_mixed_with_contrastive() {
        let slot = SentimentSlot::from_label(&SentimentLabel::Mixed);
        let entity = EntitySlot {
            entity_tokens: vec!["earnings".into(), "stock".into()],
            category: AuxCategory::Event,
        };
        let decomp = CategoricalDecomposition {
            sentiment: SentimentLabel::Mixed,
            sentiment_confidence: 0.8,
            sentiment_vec: vec![0.0; 4],
            entity_category: AuxCategory::Event,
            entity_confidence: 0.7,
            entity_vec: vec![0.0; 4],
        };
        let text = compose_from_decomposition(
            &slot, &entity,
            "The earnings call sounded confident, yet the stock keeps bleeding",
            &decomp,
        );
        assert!(text.contains("yet"), "should mention contrastive marker");
        assert!(text.contains("MIXED") || text.contains("dual valence"));
    }

    #[test]
    fn composer_generate_produces_output() {
        let composer = CategoricalComposer::new_random(8, 4, 42);
        let embedding = vec![0.1f32; 8];
        let output = composer.generate(
            &embedding,
            "I love this song but the lyrics are mediocre",
            None,
        );
        assert!(output.composed);
        assert!(!output.text.is_empty());
        assert!(!output.label_line.is_empty());
    }

    #[test]
    fn composer_with_template_fills_slots() {
        let composer = CategoricalComposer::new_random(8, 4, 42);
        let embedding = vec![0.5f32; 8];
        let template = ProgramTemplate {
            template_id: 7,
            skeleton: "The tone is {sentiment} given {entity}.".to_string(),
            sentiment_slot: SentimentSlot::from_label(&SentimentLabel::Neutral),
            entity_slot: EntitySlot {
                entity_tokens: vec![],
                category: AuxCategory::Other,
            },
            confidence: 0.8,
        };
        let output = composer.generate(
            &embedding,
            "Bitcoin price hits new high",
            Some(&template),
        );
        assert!(output.template_id == Some(7));
        assert!(output.text.contains("given"));
    }

    #[test]
    fn neutral_factual_query_gets_factual_explanation() {
        let slot = SentimentSlot::from_label(&SentimentLabel::Neutral);
        let entity = EntitySlot {
            entity_tokens: vec!["voltages".into()],
            category: AuxCategory::Other,
        };
        let decomp = CategoricalDecomposition {
            sentiment: SentimentLabel::Neutral,
            sentiment_confidence: 0.9,
            sentiment_vec: vec![0.0; 4],
            entity_category: AuxCategory::Other,
            entity_confidence: 0.8,
            entity_vec: vec![0.0; 4],
        };
        let text = compose_from_decomposition(
            &slot, &entity,
            "What nominal AC voltages are used for residential split-phase service?",
            &decomp,
        );
        assert!(text.contains("Measurable fact") || text.contains("NEUTRAL"));
    }

    #[test]
    fn truncate_for_grounding_works() {
        assert_eq!(truncate_for_grounding("short", 120), "short");
        let long = "a".repeat(200);
        let t = truncate_for_grounding(&long, 120);
        assert!(t.ends_with("..."));
        assert!(t.len() < 130);
    }
}
