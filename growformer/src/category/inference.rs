// ── inference.rs ────────────────────────────────────────────────────────────────
// Rich inference output: logits, probabilities, and trained aux head vs heuristic.

use crate::category::forward::{align_to_dim, bifunctor_branch_vectors};
use crate::category::linear_head::LinearHead;
use crate::category::pythagoras::PythagorasNode;
use crate::category::sentiment::entity_to_aux_category;
use crate::category::training::{AuxCategory, SentimentLabel};

/// Compact classification: sentiment from sentiment head, entity category from **aux head**
/// when produced by `infer_head` / `infer_head_detail` (not the string heuristic).
#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub input: String,
    pub sentiment: SentimentLabel,
    pub entity: String,
    pub inferred_category: AuxCategory,
}

impl InferenceResult {
    pub fn display(&self) {
        println!(
            "  \"{}\"\n   → {:?}  |  entity=\"{}\" ({:?})\n",
            self.input, self.sentiment, self.entity, self.inferred_category
        );
    }
}

/// Full diagnostics for `GrowformerTrainer` head + bifunctor inference.
#[derive(Debug, Clone)]
pub struct InferenceDetail {
    pub input: String,
    pub sentiment: SentimentLabel,
    pub sentiment_logits: Vec<f32>,
    pub sentiment_probs: Vec<f32>,
    pub sentiment_confidence: f32,
    /// Last whitespace token, `s`-stripped (same convention as training heuristics).
    pub entity: String,
    /// Rule-based category from the entity string (no aux head).
    pub aux_heuristic: AuxCategory,
    /// `aux_head` argmax — matches training supervision target space.
    pub aux_predicted: AuxCategory,
    pub aux_logits: Vec<f32>,
    pub aux_probs: Vec<f32>,
    pub aux_confidence: f32,
}

impl InferenceDetail {
    /// Compact view: sentiment from sentiment head, entity type from **aux head** (not heuristic).
    pub fn to_result(&self) -> InferenceResult {
        InferenceResult {
            input: self.input.clone(),
            sentiment: self.sentiment.clone(),
            entity: self.entity.clone(),
            inferred_category: self.aux_predicted.clone(),
        }
    }

    pub fn display(&self) {
        println!(
            "  \"{}\"\n   sentiment: {:?}  (conf={:.3})\n   aux head:  {:?}  (conf={:.3})\n   aux heuristic (entity): {:?}  entity=\"{}\"\n",
            self.input,
            self.sentiment,
            self.sentiment_confidence,
            self.aux_predicted,
            self.aux_confidence,
            self.aux_heuristic,
            self.entity,
        );
    }
}

fn tail_entity(input: &str) -> String {
    let raw = input.split_whitespace().last().unwrap_or("");
    raw.trim_matches(|c: char| !c.is_alphanumeric())
        .trim_end_matches('s')
        .to_string()
}

/// Run bifunctor + both heads; `embedding` is aligned to `embed_dim` before the tree.
pub fn infer_from_embedding(
    input: impl Into<String>,
    embedding: &[f32],
    embed_dim: usize,
    branch_dim: usize,
    composition: &PythagorasNode<Vec<f32>>,
    sentiment_head: &LinearHead,
    aux_head: &LinearHead,
) -> Result<InferenceDetail, &'static str> {
    if embed_dim == 0 {
        return Err("embed_dim must be positive");
    }
    let input = input.into();
    let emb = align_to_dim(embedding, embed_dim);
    let (sent_vec, ent_vec) = bifunctor_branch_vectors(composition, &emb, branch_dim);

    let (si, sl, sp, sc) = sentiment_head.predict_with_probs(&sent_vec);
    let sentiment = SentimentLabel::from_class_index(si).unwrap_or(SentimentLabel::Neutral);

    let (ai, al, ap, ac) = aux_head.predict_with_probs(&ent_vec);
    let aux_predicted = AuxCategory::from_class_index(ai).unwrap_or(AuxCategory::Other);

    let entity = tail_entity(&input);
    let aux_heuristic = entity_to_aux_category(&entity);

    Ok(InferenceDetail {
        input,
        sentiment,
        sentiment_logits: sl,
        sentiment_probs: sp,
        sentiment_confidence: sc,
        entity,
        aux_heuristic,
        aux_predicted,
        aux_logits: al,
        aux_probs: ap,
        aux_confidence: ac,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::category::disentanglement::SimpleRng;
    use crate::category::training::SentimentLabel;

    #[test]
    fn infer_from_embedding_runs() {
        let mut rng = SimpleRng::new(3);
        let d = 8usize;
        let b = 4usize;
        let comp = PythagorasNode::leaf(vec![0.5f32; d], d);
        let sh = LinearHead::new_random(b, SentimentLabel::num_classes(), &mut rng);
        let ah = LinearHead::new_random(b, AuxCategory::num_classes(), &mut rng);
        let emb = vec![0.1f32; d];
        let r = infer_from_embedding("I hate mondays", &emb, d, b, &comp, &sh, &ah).unwrap();
        assert_eq!(r.sentiment_logits.len(), SentimentLabel::num_classes());
        assert_eq!(r.aux_logits.len(), AuxCategory::num_classes());
        let s: f32 = r.sentiment_probs.iter().sum();
        assert!((s - 1.0).abs() < 1e-4);
    }
}
