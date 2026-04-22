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

    /// Logits over groups via K-nearest neighbor voting with field gradient bias.
    ///
    /// Instead of picking a single nearest program (fragile when groups overlap),
    /// the top-K programs vote for their groups. Votes are weighted by:
    ///   - Cosine similarity to input (proximity)
    ///   - Field gradient alignment (directional flow toward the program)
    ///
    /// This enables correct routing even when "write an addition function" and
    /// "implement a linked list" have very similar embeddings: the gradient
    /// breaks the tie by pointing toward the correct cluster.
    pub fn predict_logits(&mut self, input: &[f32]) -> Vec<f32> {
        if input.len() != self.input_dim || self.num_groups == 0 {
            return vec![];
        }
        let mut logits = vec![0.0f32; self.num_groups];
        if self.lattice.programs.is_empty() {
            return logits;
        }

        let k = 7.min(self.lattice.programs.len());

        // Score all programs by cosine similarity.
        let mut scored: Vec<(usize, f32)> = self.lattice.programs.iter().enumerate()
            .map(|(i, prog)| (i, cosine_sim(input, &prog.ema_centroid)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Compute field gradient: ∇F at the query point, where each program
        // is a source with strength proportional to its similarity.
        // All accumulation in f64 to eliminate FMA vs non-FMA divergence (native vs WASM).
        let dim = input.len();
        let mut gradient = vec![0.0f64; dim];
        let mut weight_sum = 0.0f64;
        for &(idx, sim) in scored.iter().take(k * 3) {
            if sim < 0.01 { break; }
            let centroid = &self.lattice.programs[idx].ema_centroid;
            let mut disp_norm_sq = 0.0f64;
            for j in 0..dim.min(centroid.len()) {
                let d = (input[j] - centroid[j]) as f64;
                disp_norm_sq += d * d;
            }
            if disp_norm_sq < 1e-20 { continue; }
            let green_w = (sim as f64) / disp_norm_sq;
            for j in 0..dim.min(centroid.len()) {
                gradient[j] += green_w * (input[j] - centroid[j]) as f64;
            }
            weight_sum += green_w;
        }
        if weight_sum > 1e-20 {
            for v in &mut gradient { *v /= weight_sum; }
        }
        let grad_mag: f64 = gradient.iter().map(|x| x * x).sum::<f64>().sqrt();

        // K-NN voting with gradient alignment bias (f64 accumulators).
        let mut logits_f64 = vec![0.0f64; self.num_groups];
        for &(idx, sim) in scored.iter().take(k) {
            if sim < 0.0 { continue; }
            let text = self.lattice.programs[idx].display_text(&self.lattice.dictionary);
            if let Some(gid) = Self::parse_group_id(&text) {
                if gid < self.num_groups {
                    let grad_bonus = if grad_mag > 1e-12 {
                        let centroid = &self.lattice.programs[idx].ema_centroid;
                        let min_dim = dim.min(centroid.len()).min(gradient.len());
                        let dot: f64 = (0..min_dim)
                            .map(|j| (centroid[j] - input[j]) as f64 * gradient[j])
                            .sum();
                        let alignment = (dot / grad_mag).clamp(-1.0, 1.0);
                        (alignment + 1.0) / 2.0
                    } else {
                        0.5
                    };

                    let vote = (sim as f64).max(0.0) * (0.65 + 0.35 * grad_bonus);
                    logits_f64[gid] += vote;
                }
            }
        }
        for (dst, src) in logits.iter_mut().zip(logits_f64.iter()) {
            *dst = *src as f32;
        }

        logits
    }

    /// One training step: develop lattice with this (input, group_id) pair.
    #[cfg(feature = "training")]
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
                let text = prog.display_text(&base.lattice.dictionary);
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
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len().min(b.len()) {
        let ai = a[i] as f64;
        let bi = b[i] as f64;
        dot += ai * bi;
        na += ai * ai;
        nb += bi * bi;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-20 { 0.0 } else { (dot / denom) as f32 }
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
