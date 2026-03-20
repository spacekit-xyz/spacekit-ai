//! GroupRouter — maps input to group relevance (Paramecium lattice or heuristic).
//!
//! **Learned router**: Paramecium lattice trained via `develop()` in one pass.
//! Each behavioral program stores (embedding, group_id). At inference, wave-propagation
//! selects the nearest program and returns its group_id.

use crate::types::GroupId;
use crate::spectral::TokenDictionary;
use crate::dimension::paramecium::InfraciliaryLattice;
use rand::Rng;
use serde::{Deserialize, Serialize};

use super::embedding::{retrieve_relevant_groups, GroupEmbedding};

// ---------------------------------------------------------------------------
// Learned router: Paramecium lattice input -> group selection
// ---------------------------------------------------------------------------

/// Learned router: Paramecium lattice developed from (embedding, group_label) pairs.
/// One-pass training via `develop()`. Inference via wave-propagation nearest-program.
#[derive(Clone, Serialize, Deserialize)]
pub struct LearnedRouter {
    pub lattice: InfraciliaryLattice,
    pub num_groups: usize,
    pub input_dim: usize,
}

impl LearnedRouter {
    pub fn new(
        input_dim: usize,
        num_groups: usize,
        _hidden_size: usize,
        _rng: &mut impl Rng,
    ) -> Self {
        let dict = TokenDictionary::build(&[], 64);
        let lattice = InfraciliaryLattice::new(dict);
        LearnedRouter {
            lattice,
            num_groups,
            input_dim,
        }
    }

    /// Build router from labeled training data in one pass.
    pub fn build(
        input_dim: usize,
        num_groups: usize,
        samples: &[(Vec<f32>, GroupId)],
    ) -> Self {
        let group_labels: Vec<String> = (0..num_groups).map(|g| format!("group_{}", g)).collect();
        let dict = TokenDictionary::build(
            &group_labels.iter().map(|s| s.as_str()).collect::<Vec<_>>(), 64,
        );
        let pairs: Vec<(Vec<f32>, String)> = samples.iter()
            .map(|(emb, gid)| (emb.clone(), format!("group_{}", gid)))
            .collect();
        let mut lattice = InfraciliaryLattice::new(dict);
        lattice.develop(&pairs, 0.90);
        LearnedRouter {
            lattice,
            num_groups,
            input_dim,
        }
    }

    /// Logits over groups via direct cosine nearest-neighbor (no EMA drift).
    pub fn predict_logits(&mut self, input: &[f32]) -> Vec<f32> {
        if input.len() != self.input_dim || self.num_groups == 0 {
            return vec![];
        }
        let mut logits = vec![0.0f32; self.num_groups];
        if self.lattice.programs.is_empty() {
            return logits;
        }
        let mut best_idx = 0;
        let mut best_sim = f32::NEG_INFINITY;
        for (i, prog) in self.lattice.programs.iter().enumerate() {
            let sim = cosine_sim(input, &prog.ema_centroid);
            if sim > best_sim { best_sim = sim; best_idx = i; }
        }
        let text = self.lattice.dictionary.decode(&self.lattice.programs[best_idx].token_sequence);
        if let Some(gid) = Self::parse_group_id(&text) {
            if gid < self.num_groups {
                logits[gid] = best_sim.max(0.0);
            }
        }
        logits
    }

    /// One training step: develop lattice with this (input, group_id) pair.
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
        let pairs = vec![(input.to_vec(), format!("group_{}", gid))];
        self.lattice.develop(&pairs, 0.90);

        let resp = self.lattice.respond(input);
        let predicted = Self::parse_group_id(&resp.text).unwrap_or(usize::MAX);
        if predicted == gid { 0.0 } else { 1.0 }
    }

    /// Chosen group id (from nearest lattice program).
    pub fn choose_group(&mut self, input: &[f32]) -> Option<GroupId> {
        let logits = self.predict_logits(input);
        if logits.is_empty() {
            return None;
        }
        let (idx, &val) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;
        if val > 0.0 { Some(idx as GroupId) } else { None }
    }

    /// Merge multiple routers by combining their lattice programs.
    pub fn average_from(routers: &[Self]) -> Option<Self> {
        if routers.is_empty() {
            return None;
        }
        let mut base = routers[0].clone();
        for r in &routers[1..] {
            for prog in &r.lattice.programs {
                let text = base.lattice.dictionary.decode(&prog.token_sequence);
                let pairs = vec![(prog.ema_centroid.clone(), text)];
                base.lattice.develop(&pairs, 0.95);
            }
        }
        Some(base)
    }

    fn parse_group_id(text: &str) -> Option<usize> {
        text.strip_prefix("group_").and_then(|s| s.parse().ok())
    }

    pub fn program_count(&self) -> usize {
        self.lattice.program_count()
    }
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-10 { 0.0 } else { dot / denom }
}

// ---------------------------------------------------------------------------
// Query-based attention (cosine similarity over embeddings)
// ---------------------------------------------------------------------------

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

    #[test]
    fn test_router_build_and_classify() {
        let samples: Vec<(Vec<f32>, GroupId)> = vec![
            (vec![1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0),
            (vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1),
            (vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2),
            (vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 3),
        ];
        let mut router = LearnedRouter::build(16, 4, &samples);
        for (emb, expected_gid) in &samples {
            let chosen = router.choose_group(emb);
            assert_eq!(chosen, Some(*expected_gid),
                "router should route {:?} to group {}", emb, expected_gid);
        }
    }
}
