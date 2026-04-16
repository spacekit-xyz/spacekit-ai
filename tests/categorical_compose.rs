// tests/categorical_compose.rs — integration tests for CategoricalComposer
// Verifies compositional generation for unseen entity × sentiment combinations.

#![cfg(feature = "categorical")]

use growformer::category::compose::{
    CategoricalComposer, CategoricalDecomposition, ComposedOutput, ProgramTemplate,
};
use growformer::category::training::{AuxCategory, SentimentLabel};
use growformer::category::compose::{EntitySlot, SentimentSlot};

fn make_composer() -> CategoricalComposer {
    CategoricalComposer::new_random(16, 8, 77)
}

// ── Compositional generalization: unseen entity × sentiment ──────────────────

#[test]
fn unseen_entity_positive_strong() {
    let composer = make_composer();
    let emb = vec![0.5f32; 16];
    let out = composer.generate(&emb, "that play was disgusting", None);
    assert!(out.composed, "output should be composed");
    assert!(!out.text.is_empty());
    assert!(out.confidence > 0.0);
}

#[test]
fn unseen_entity_negative_mild() {
    let composer = make_composer();
    let emb = vec![-0.3f32; 16];
    let out = composer.generate(
        &emb,
        "Bolt laid off a third of its staff, citing AI adoption",
        None,
    );
    assert!(out.composed);
    assert!(out.explanation.len() > 10);
}

#[test]
fn contrastive_mixed_with_but() {
    let composer = make_composer();
    let emb = vec![0.1f32; 16];
    let out = composer.generate(
        &emb,
        "The earnings call sounded confident, but the stock keeps bleeding after hours",
        None,
    );
    assert!(out.composed);
    let lower = out.explanation.to_ascii_lowercase();
    assert!(
        lower.contains("but") || lower.contains("contrastive") || lower.contains("dual"),
        "should detect contrastive marker: {:?}",
        out.explanation
    );
}

#[test]
fn factual_query_with_neutral_decomp_gets_factual_explanation() {
    let decomp = CategoricalDecomposition {
        sentiment: SentimentLabel::Neutral,
        sentiment_confidence: 0.9,
        sentiment_vec: vec![0.0; 8],
        entity_category: AuxCategory::Other,
        entity_confidence: 0.8,
        entity_vec: vec![0.0; 8],
    };
    let composer = make_composer();
    let out = composer.compose(
        &decomp,
        "What nominal AC voltages are used for residential split-phase service?",
        None,
    );
    assert!(out.composed);
    let lower = out.explanation.to_ascii_lowercase();
    assert!(
        lower.contains("neutral") || lower.contains("measurable") || lower.contains("fact"),
        "factual query with neutral decomp should get factual explanation: {:?}",
        out.explanation
    );
}

#[test]
fn sarcastic_gets_irony_mention() {
    let decomp = CategoricalDecomposition {
        sentiment: SentimentLabel::Sarcastic,
        sentiment_confidence: 0.85,
        sentiment_vec: vec![0.0; 8],
        entity_category: AuxCategory::Event,
        entity_confidence: 0.7,
        entity_vec: vec![0.0; 8],
    };
    let composer = make_composer();
    let out = composer.compose(
        &decomp,
        "Third email, still no answer. You're doing great.",
        None,
    );
    assert!(out.composed);
    assert!(out.text.contains("SARCASTIC"));
    let lower = out.explanation.to_ascii_lowercase();
    assert!(lower.contains("irony") || lower.contains("mismatch"));
}

// ── Template-based composition ───────────────────────────────────────────────

#[test]
fn template_fills_sentiment_and_entity() {
    let composer = make_composer();
    let emb = vec![0.2f32; 16];
    let template = ProgramTemplate {
        template_id: 42,
        skeleton: "The overall sentiment is {sentiment}, focusing on {entity}.".to_string(),
        sentiment_slot: SentimentSlot::from_label(&SentimentLabel::Neutral),
        entity_slot: EntitySlot {
            entity_tokens: vec![],
            category: AuxCategory::Other,
        },
        confidence: 0.85,
    };
    let out = composer.generate(
        &emb,
        "Bitcoin price hits new high after ETF approval",
        Some(&template),
    );
    assert!(out.composed);
    assert_eq!(out.template_id, Some(42));
    assert!(out.text.contains("focusing on"));
}

// ── Confidence bounds ────────────────────────────────────────────────────────

#[test]
fn confidence_is_bounded() {
    let composer = make_composer();
    let emb = vec![1.0f32; 16];
    let out = composer.generate(&emb, "some text", None);
    assert!(out.confidence <= 0.95, "confidence should be capped at 0.95");
    assert!(out.confidence >= 0.0);
}

// ── Grounding text present ───────────────────────────────────────────────────

#[test]
fn output_contains_grounding() {
    let composer = make_composer();
    let emb = vec![0.1f32; 16];
    let out = composer.generate(
        &emb,
        "I keep going back to this song",
        None,
    );
    assert!(
        out.explanation.contains("Grounded in the user"),
        "composed output should include grounding: {:?}",
        out.explanation
    );
}

// ── Seven sentiment labels all produce valid output ──────────────────────────

#[test]
fn all_seven_labels_produce_composed_output() {
    let labels = [
        SentimentLabel::PositiveStrong,
        SentimentLabel::PositiveMild,
        SentimentLabel::Neutral,
        SentimentLabel::NegativeMild,
        SentimentLabel::NegativeStrong,
        SentimentLabel::Sarcastic,
        SentimentLabel::Mixed,
    ];
    let composer = make_composer();
    for label in &labels {
        let decomp = CategoricalDecomposition {
            sentiment: label.clone(),
            sentiment_confidence: 0.8,
            sentiment_vec: vec![0.0; 8],
            entity_category: AuxCategory::Other,
            entity_confidence: 0.6,
            entity_vec: vec![0.0; 8],
        };
        let out = composer.compose(&decomp, "test input text here", None);
        assert!(out.composed, "label {:?} should produce composed output", label);
        assert!(!out.text.is_empty(), "label {:?} should produce non-empty text", label);
        assert!(!out.label_line.is_empty(), "label {:?} should have label_line", label);
    }
}
