// ── sentiment.rs ──────────────────────────────────────────────────────────────
// Sentiment/entity disentanglement via bifunctor split.
// The Pythagoras tree's left child = sentiment morphism,
// right child = entity morphism.
// Natural transformations between entity categories enable
// generalization to unseen days/objects.

use crate::category::training::SentimentLabel;
use crate::category::{Layer, NaturalTransform};

// ── ParsedInput ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ParsedInput {
    /// Full token embedding of the input sentence.
    pub embedding: Vec<f32>,
    /// Raw text for inspection.
    pub raw: String,
    /// Known entity category, if pre-tagged.
    pub entity_category: Option<crate::category::training::AuxCategory>,
}

impl ParsedInput {
    pub fn new(raw: impl Into<String>, embedding: Vec<f32>) -> Self {
        Self {
            embedding,
            raw: raw.into(),
            entity_category: None,
        }
    }

    pub fn with_category(mut self, cat: crate::category::training::AuxCategory) -> Self {
        self.entity_category = Some(cat);
        self
    }
}

// ── Disentanglement loss ───────────────────────────────────────────────────────

/// Penalizes shared information between the sentiment and entity branches.
/// During training, minimizing this alongside the main loss encourages
/// the Pythagoras left/right children to learn independent representations.
pub fn disentanglement_loss(sentiment_embedding: &[f32], entity_embedding: &[f32]) -> f32 {
    debug_assert_eq!(
        sentiment_embedding.len(),
        entity_embedding.len(),
        "Sentiment and entity embeddings must have equal dimension"
    );

    let dot: f32 = sentiment_embedding
        .iter()
        .zip(entity_embedding.iter())
        .map(|(a, b)| a * b)
        .sum();

    let norm_s: f32 = sentiment_embedding
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();
    let norm_e: f32 = entity_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_s == 0.0 || norm_e == 0.0 {
        return 0.0;
    }

    // Cosine similarity — we want this ≈ 0 (orthogonal branches)
    (dot / (norm_s * norm_e)).abs()
}

/// Combined training loss: task loss + λ * disentanglement loss.
pub fn combined_loss(
    task_loss: f32,
    sentiment_emb: &[f32],
    entity_emb: &[f32],
    lambda: f32,
) -> f32 {
    task_loss + lambda * disentanglement_loss(sentiment_emb, entity_emb)
}

// ── SentimentFunctor ──────────────────────────────────────────────────────────

/// The sentiment functor F: maps any ParsedInput → SentimentLabel
/// independently of which entity appears in the input.
/// Internal Pythagoras split: left=sentiment, right=entity.
pub struct SentimentFunctor {
    /// Left branch weights — learn "I hate/love X" pattern.
    sentiment_weights: Vec<f32>,
    /// Right branch weights — learn entity category membership.
    entity_weights: Vec<f32>,
    /// Threshold for classification.
    threshold: f32,
}

impl SentimentFunctor {
    pub fn new(sentiment_weights: Vec<f32>, entity_weights: Vec<f32>, threshold: f32) -> Self {
        Self {
            sentiment_weights,
            entity_weights,
            threshold,
        }
    }

    /// Extract the sentiment branch embedding.
    pub fn sentiment_embedding(&self, input: &ParsedInput) -> Vec<f32> {
        input
            .embedding
            .iter()
            .zip(self.sentiment_weights.iter().cycle())
            .map(|(x, w)| x * w)
            .collect()
    }

    /// Extract the entity branch embedding.
    pub fn entity_embedding(&self, input: &ParsedInput) -> Vec<f32> {
        input
            .embedding
            .iter()
            .zip(self.entity_weights.iter().cycle())
            .map(|(x, w)| x * w)
            .collect()
    }

    /// Score: positive = positive sentiment, negative = negative.
    pub fn sentiment_score(&self, input: &ParsedInput) -> f32 {
        self.sentiment_embedding(input).iter().sum::<f32>() / self.sentiment_weights.len() as f32
    }
}

impl Layer<ParsedInput, SentimentLabel> for SentimentFunctor {
    fn forward(&self, input: ParsedInput) -> SentimentLabel {
        let score = self.sentiment_score(&input);
        let t = self.threshold;
        // Functor law: score depends only on sentiment branch,
        // not which entity (day of week, weather, etc.) is present.
        if score > t * 2.0 {
            SentimentLabel::PositiveStrong
        } else if score > t {
            SentimentLabel::PositiveMild
        } else if score < -t * 2.0 {
            SentimentLabel::NegativeStrong
        } else if score < -t {
            SentimentLabel::NegativeMild
        } else if score > 0.0 {
            SentimentLabel::Mixed
        } else if score < 0.0 {
            SentimentLabel::Sarcastic
        } else {
            SentimentLabel::Neutral
        }
    }
}

// ── Day-of-week natural transformation ───────────────────────────────────────

/// Days of the week as objects in a category.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DayOfWeek {
    Monday,
    Tuesday,
    Wednesday,
    Thursday,
    Friday,
    Saturday,
    Sunday,
}

impl DayOfWeek {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().trim_end_matches('s') {
            "monday" => Some(Self::Monday),
            "tuesday" => Some(Self::Tuesday),
            "wednesday" => Some(Self::Wednesday),
            "thursday" => Some(Self::Thursday),
            "friday" => Some(Self::Friday),
            "saturday" => Some(Self::Saturday),
            "sunday" => Some(Self::Sunday),
            _ => None,
        }
    }

    pub fn is_weekend(&self) -> bool {
        matches!(self, Self::Saturday | Self::Sunday)
    }
}

/// Natural transformation: any DayOfWeek → DayOfWeek.
/// Commutativity: swapping the day before or after sentiment extraction
/// gives the same result — this is what we train the model to achieve.
pub struct DaySubstitution;

impl NaturalTransform<DayOfWeek, DayOfWeek> for DaySubstitution {
    /// Identity transformation: the object changes, sentiment morphism stays constant.
    fn transform(day: DayOfWeek) -> DayOfWeek {
        day
    }
}

/// Alias used by GrowformerTrainer for inference — delegates to AuxCategory::infer.
pub fn entity_to_aux_category(entity: &str) -> crate::category::training::AuxCategory {
    crate::category::training::AuxCategory::infer(entity)
}
