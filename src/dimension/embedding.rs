//! Group embeddings for routing and redundancy checks.
//! Computed once at promotion; mean-pool of hidden activations over calibration data.

use crate::environment::NeuralEnvironment;
use crate::types::GroupId;
use serde::{Deserialize, Serialize};

/// Fixed vector encoding a group's mean activation pattern.
/// Computed once at promotion. Never recomputed.
/// Use **id** (group_id) to reference this group in the learned router and in training data.
/// **description** and **metatags** support logging, filtering, and future text-conditioned routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupEmbedding {
    pub group_id: GroupId,
    /// Mean-pooled hidden-layer activations over calibration samples.
    pub vector: Vec<f32>,
    pub task_name: String,
    pub accuracy: f32,
    /// Optional: intrinsic dimensionality (e.g. PCA). For future use.
    pub intrinsic_dim: Option<f32>,
    /// Human-readable description (e.g. "Spiral 2D binary classification"). For logging / routing.
    pub description: Option<String>,
    /// Tags for filtering or conditioning (e.g. ["spiral", "classification", "2d"]).
    pub metatags: Vec<String>,
    /// Vectorized metatags for search/re-rank (same dim for all groups; empty = no tag routing).
    #[serde(default)]
    pub tag_vector: Vec<f32>,
    /// Optional language-space vector from global frozen bridge (default 64 dims).
    #[serde(default)]
    pub language_vector: Vec<f32>,
}

/// Default dimension for tag vectors. Fixed size; no growing vocab — scales to any number of tags.
pub const TAG_VECTOR_DIM: usize = 64;

/// Deterministic hash of a string for tag vector index.
fn hash_tag(s: &str) -> u64 {
    let mut h: u64 = 5381;
    for b in s.bytes() {
        h = h.wrapping_mul(33).wrapping_add(b as u64);
    }
    h
}

/// Build a fixed-size vector from metatags (hash-based; no vocab). L2-normalized for cosine.
/// Use for tag-space search and re-rank. Same dim for all groups.
pub fn build_tag_vector(metatags: &[String], dim: usize) -> Vec<f32> {
    if dim == 0 {
        return vec![];
    }
    let mut v = vec![0.0f32; dim];
    for tag in metatags {
        let h = hash_tag(tag);
        let i0 = (h as usize) % dim;
        let i1 = ((h >> 16) as usize) % dim;
        v[i0] += 1.0;
        if i1 != i0 {
            v[i1] += 1.0;
        }
    }
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-10 {
        for x in &mut v {
            *x /= n;
        }
    }
    v
}

/// Hidden activation vector for one forward pass.
/// Same layer (1..n_layers-1), same neuron order, and no extra normalization as group embedding.
/// Call after env.predict(input). Router compares this to GroupEmbedding.vector for cosine score.
pub fn hidden_activation_vector(env: &NeuralEnvironment) -> Vec<f32> {
    let n_layers = env.layers.len();
    if n_layers < 2 {
        return vec![];
    }
    let hidden_ids: Vec<_> = env.layers[1..n_layers - 1]
        .iter()
        .flat_map(|layer| layer.iter().copied())
        .collect();
    hidden_ids
        .iter()
        .map(|&id| env.neurons.get(&id).map(|n| n.activation).unwrap_or(0.0))
        .collect()
}

/// Run calibration data through env, collect hidden activations per sample, mean-pool.
/// Returns the embedding vector (same length as total hidden neurons).
/// Caller sets group_id, task_name, accuracy when building GroupEmbedding.
pub fn compute_group_embedding(
    env: &mut NeuralEnvironment,
    calibration_data: &[crate::types::Sample],
) -> Vec<f32> {
    if calibration_data.is_empty() {
        return vec![];
    }
    let n_layers = env.layers.len();
    if n_layers < 2 {
        return vec![];
    }
    // Hidden layers: indices 1 .. n_layers-1
    let hidden_ids: Vec<_> = env.layers[1..n_layers - 1]
        .iter()
        .flat_map(|layer| layer.iter().copied())
        .collect();
    let dim = hidden_ids.len();
    let mut sum: Vec<f32> = vec![0.0; dim];
    for (input, _) in calibration_data {
        env.predict(input.as_slice());
        for (i, &id) in hidden_ids.iter().enumerate() {
            if let Some(n) = env.neurons.get(&id) {
                sum[i] += n.activation;
            }
        }
    }
    let n = calibration_data.len() as f32;
    sum.iter().map(|s| s / n).collect()
}

/// Cosine similarity between two vectors. Returns value in [-1, 1].
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let denom = na * nb;
    if denom < 1e-10 {
        return 0.0;
    }
    (dot / denom).clamp(-1.0, 1.0)
}

/// Return group indices sorted by relevance (cosine similarity to query vector, descending).
pub fn retrieve_relevant_groups(
    query: &[f32],
    embeddings: &[GroupEmbedding],
    top_k: usize,
) -> Vec<(GroupId, f32)> {
    let mut scored: Vec<_> = embeddings
        .iter()
        .map(|e| (e.group_id, cosine_similarity(query, &e.vector)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(top_k).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::EnvironmentConfig;
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn spiral_like(n: usize, _rng: &mut StdRng) -> Vec<crate::types::Sample> {
        use std::f32::consts::PI;
        let mut out = Vec::with_capacity(n * 2);
        for i in 0..n {
            let t = (i as f32 / n as f32) * 2.0 * PI;
            let r = 0.5 + (i as f32 * 0.001) % 0.2;
            out.push((vec![r * t.cos(), r * t.sin()], [0.0]));
        }
        for i in 0..n {
            let t = (i as f32 / n as f32) * 2.0 * PI + PI;
            let r = 0.5 + 0.3 + (i as f32 * 0.001) % 0.2;
            out.push((vec![r * t.cos(), r * t.sin()], [1.0]));
        }
        out
    }

    fn circles_like(n: usize, rng: &mut StdRng) -> Vec<crate::types::Sample> {
        use std::f32::consts::PI;
        let mut out = Vec::with_capacity(n * 2);
        for _ in 0..n {
            let theta = rng.gen::<f32>() * 2.0 * PI;
            let r = 0.5 + rng.gen_range(-0.05..0.05);
            out.push((vec![r * theta.cos(), r * theta.sin()], [0.0]));
        }
        for _ in 0..n {
            let theta = rng.gen::<f32>() * 2.0 * PI;
            let r = 1.0 + rng.gen_range(-0.05..0.05);
            out.push((vec![r * theta.cos(), r * theta.sin()], [1.0]));
        }
        out
    }

    #[test]
    fn test_embedding_compute_and_cosine() {
        let config = EnvironmentConfig::default();
        let mut rng = StdRng::seed_from_u64(42);
        let mut env_a = NeuralEnvironment::new(config.clone());
        env_a.build_layers(&[2, 16, 16, 1], &mut rng);
        let mut env_b = NeuralEnvironment::new(config);
        env_b.build_layers(&[2, 16, 16, 1], &mut rng);

        let spiral_data = spiral_like(50, &mut rng);
        let circles_data = circles_like(50, &mut rng);

        let emb_a = compute_group_embedding(&mut env_a, &spiral_data);
        let emb_b = compute_group_embedding(&mut env_b, &circles_data);

        assert!(!emb_a.is_empty());
        assert_eq!(emb_a.len(), emb_b.len());
        let sim = cosine_similarity(&emb_a, &emb_b);
        assert!(sim >= -1.0 && sim <= 1.0);
        // Dissimilarity (cosine < 0.5) requires trained nets; validated in Step 2/4 integration.
    }

    #[test]
    fn test_build_tag_vector_dim_and_normalized() {
        let v = build_tag_vector(&[String::from("spiral")], TAG_VECTOR_DIM);
        assert_eq!(v.len(), TAG_VECTOR_DIM);
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(n > 0.99 && n < 1.01, "tag vector should be L2-normalized, got norm {}", n);
    }

    #[test]
    fn test_build_tag_vector_deterministic() {
        let a = build_tag_vector(&[String::from("spiral")], TAG_VECTOR_DIM);
        let b = build_tag_vector(&[String::from("spiral")], TAG_VECTOR_DIM);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-5);
        }
    }

    #[test]
    fn test_build_tag_vector_different_tags_different_vectors() {
        let spiral = build_tag_vector(&[String::from("spiral")], TAG_VECTOR_DIM);
        let circles = build_tag_vector(&[String::from("circles")], TAG_VECTOR_DIM);
        let sim = cosine_similarity(&spiral, &circles);
        assert!(sim < 1.0, "spiral and circles tag vectors should differ, cosine was {}", sim);
    }

    #[test]
    fn test_build_tag_vector_same_query_high_similarity() {
        let spiral_vec = build_tag_vector(&[String::from("spiral")], TAG_VECTOR_DIM);
        let query = build_tag_vector(&[String::from("spiral")], TAG_VECTOR_DIM);
        assert!(!spiral_vec.is_empty() && !query.is_empty());
        let sim = cosine_similarity(&spiral_vec, &query);
        assert!(sim > 0.99, "same tag should give cosine ~1, got {}", sim);
    }

    #[test]
    fn test_build_tag_vector_empty_dim() {
        let v = build_tag_vector(&[String::from("a")], 0);
        assert!(v.is_empty());
    }

    #[test]
    fn test_hidden_activation_same_layout_as_embedding() {
        let config = EnvironmentConfig::default();
        let mut rng = StdRng::seed_from_u64(42);
        let mut env = NeuralEnvironment::new(config);
        env.build_layers(&[2, 16, 16, 1], &mut rng);
        let one_sample: Vec<crate::types::Sample> = vec![(vec![0.1_f32, 0.2], [0.0])];
        let embedding = compute_group_embedding(&mut env, &one_sample);
        env.predict(&[0.1_f32, 0.2]);
        let hidden = hidden_activation_vector(&env);
        assert_eq!(hidden.len(), embedding.len());
        let cos = cosine_similarity(&hidden, &embedding);
        assert!((cos - 1.0).abs() < 1e-5, "single sample: activation should equal embedding, cos={}", cos);
    }
}
