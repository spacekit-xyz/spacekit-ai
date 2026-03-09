//! GroupRouter — maps input to group relevance (heuristic or learned).
//!
//! **Learned router**: Small MLP `input -> logits over groups`. Training data referenced by
//! **id** (GroupId): each sample is `(input, target_group_id)`. At calibration / after promotion
//! we have task-labeled data (e.g. "spiral" -> group 0), so we can collect (input, group_id) and
//! train the router. **desc** (GroupEmbedding.description) and **metatags** (GroupEmbedding.metatags)
//! can filter which groups participate in routing or be used as extra input in future extensions.

use crate::environment::NeuralEnvironment;
use crate::types::{EnvironmentConfig, GroupId};
use rand::Rng;
use serde::{Deserialize, Serialize};

use super::embedding::{retrieve_relevant_groups, GroupEmbedding};

// ---------------------------------------------------------------------------
// Learned router: neural net input -> logits per group
// ---------------------------------------------------------------------------

/// Learned router: MLP with shape [input_dim, hidden_size, num_groups].
/// Train with (input, target_group_id); infer returns logits; argmax = chosen group.
/// Reference training data by **id** (GroupId). When num_groups increases (new promotion),
/// rebuild or extend the output layer (current impl uses fixed num_groups at creation).
#[derive(Clone, Serialize, Deserialize)]
pub struct LearnedRouter {
    pub env: NeuralEnvironment,
    pub num_groups: usize,
    pub input_dim: usize,
}

impl LearnedRouter {
    /// Build a router MLP: input_dim -> hidden -> num_groups. Uses a config with higher learning rate
    /// (0.15) so the small classifier converges quickly on labeled (input, group) data.
    pub fn new(
        input_dim: usize,
        num_groups: usize,
        hidden_size: usize,
        rng: &mut impl Rng,
    ) -> Self {
        let mut config = EnvironmentConfig::default();
        config.learning_rate = 0.15;
        let mut env = NeuralEnvironment::new(config);
        env.build_layers(&[input_dim, hidden_size, num_groups], rng);
        LearnedRouter {
            env,
            num_groups,
            input_dim,
        }
    }

    /// Logits over groups (same order as main.group_order). Empty if input len != input_dim.
    pub fn predict_logits(&mut self, input: &[f32]) -> Vec<f32> {
        if input.len() != self.input_dim || self.num_groups == 0 {
            return vec![];
        }
        self.env.predict(input)
    }

    /// One training step: forward + backprop with one-hot target for group_id. Returns loss.
    /// Use when you have labeled data (input, group_id); group_id is the **id** reference.
    pub fn train_step(
        &mut self,
        input: &[f32],
        target_group_id: GroupId,
        _rng: &mut impl Rng,
    ) -> f32 {
        if input.len() != self.input_dim {
            return 0.0;
        }
        let gid = target_group_id as usize;
        if gid >= self.num_groups {
            return 0.0;
        }
        let output = self.env.forward(input);
        let mut target = vec![0.0f32; self.num_groups];
        target[gid] = 1.0;
        self.env.backprop(&output, &target)
    }

    /// Chosen group id (argmax of logits). None if no groups or empty logits.
    pub fn choose_group(&mut self, input: &[f32]) -> Option<GroupId> {
        let logits = self.predict_logits(input);
        if logits.is_empty() {
            return None;
        }
        let (idx, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;
        Some(idx as GroupId)
    }
}

// ---------------------------------------------------------------------------
// Query-based attention (cosine similarity over embeddings)
// ---------------------------------------------------------------------------

/// Given a query vector (e.g. from running input through a reference env), return top-k groups by cosine similarity.
/// Routing in production uses LearnedRouter when set, or GlobalObserver's cosine over hidden activations.
pub fn attend_by_query(
    query: &[f32],
    embeddings: &[GroupEmbedding],
    top_k: usize,
) -> Vec<(GroupId, f32)> {
    retrieve_relevant_groups(query, embeddings, top_k)
}

// ---------------------------------------------------------------------------
// Tag-based re-rank and stickiness (Markov-like prior) — no training, scalable
// ---------------------------------------------------------------------------

/// Add stickiness: boost score for the group that was last chosen (Markov-like temporal prior).
pub fn apply_stickiness(
    scores: &mut [(GroupId, f32)],
    last_chosen_group_id: Option<GroupId>,
    stickiness: f32,
) {
    if stickiness <= 0.0 || last_chosen_group_id.is_none() {
        return;
    }
    let last = last_chosen_group_id.unwrap();
    for (gid, s) in scores.iter_mut() {
        if *gid == last {
            *s += stickiness;
            break;
        }
    }
}

/// Re-rank by tag-vector similarity to query (from context_tags via build_tag_vector).
pub fn apply_tag_rerank(
    scores: &mut [(GroupId, f32)],
    embeddings: &[GroupEmbedding],
    query_tag_vector: &[f32],
    weight: f32,
) {
    if weight <= 0.0 || query_tag_vector.is_empty() {
        return;
    }
    for (gid, s) in scores.iter_mut() {
        if let Some(emb) = embeddings.iter().find(|e| e.group_id == *gid) {
            if !emb.tag_vector.is_empty() && emb.tag_vector.len() == query_tag_vector.len() {
                let sim = super::embedding::cosine_similarity(&emb.tag_vector, query_tag_vector);
                *s += weight * sim;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::embedding::{build_tag_vector, GroupEmbedding, TAG_VECTOR_DIM};

    #[test]
    fn test_apply_stickiness_boosts_last_chosen() {
        let mut scores = vec![(0u32, 1.0f32), (1u32, 1.0f32)];
        apply_stickiness(&mut scores, Some(0), 0.2);
        assert_eq!(scores[0].1, 1.2);
        assert_eq!(scores[1].1, 1.0);
        let best = scores.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0;
        assert_eq!(best, 0);
    }

    #[test]
    fn test_apply_stickiness_none_no_change() {
        let mut scores = vec![(0u32, 1.0f32), (1u32, 1.0f32)];
        apply_stickiness(&mut scores, None, 0.2);
        assert_eq!(scores[0].1, 1.0);
        assert_eq!(scores[1].1, 1.0);
    }

    #[test]
    fn test_apply_tag_rerank_boosts_matching_group() {
        let spiral_tv = build_tag_vector(&[String::from("spiral")], TAG_VECTOR_DIM);
        let circles_tv = build_tag_vector(&[String::from("circles")], TAG_VECTOR_DIM);
        let query = build_tag_vector(&[String::from("spiral")], TAG_VECTOR_DIM);
        let embeddings = vec![
            GroupEmbedding {
                group_id: 0,
                vector: vec![],
                task_name: "spiral".into(),
                accuracy: 0.9,
                intrinsic_dim: None,
                description: None,
                metatags: vec!["spiral".into()],
                tag_vector: spiral_tv,
                language_vector: vec![],
            },
            GroupEmbedding {
                group_id: 1,
                vector: vec![],
                task_name: "circles".into(),
                accuracy: 0.9,
                intrinsic_dim: None,
                description: None,
                metatags: vec!["circles".into()],
                tag_vector: circles_tv,
                language_vector: vec![],
            },
        ];
        let mut scores = vec![(0u32, 1.0f32), (1u32, 1.0f32)];
        apply_tag_rerank(&mut scores, &embeddings, &query, 0.5);
        assert!(scores[0].1 > scores[1].1);
        let best = scores.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0;
        assert_eq!(best, 0);
    }
}
