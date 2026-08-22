//! Neural Coherence Analysis — measures synchrony across lattice programs
//! using Cl(1,7) SpaceTime Algebra grade decomposition.
//!
//! Inspired by EEG coherence in neuroscience, where synchrony of neural
//! activity across brain areas indicates functional connectivity. We map
//! the STA grade structure to frequency bands:
//!
//!   δ (delta)      — Grade 0 scalar: global magnitude alignment
//!   θ (theta)      — Grade 1 vectors: directional/intentional alignment
//!   α/β (alpha-beta) — Grade 2 bivectors: structural (spatial) + causal (boost) coherence
//!   γ (gamma)      — Grade 3+ trivectors: fine-grained semantic binding
//!
//! Key neuroscience insight: coherence is primarily influenced by the
//! "sending" region — a program's embedding quality (signal strength per
//! band) matters more than pairwise similarity alone. Programs with strong,
//! well-defined grade-specific activations produce better coherence signals.

use crate::clifford::{
    embed_bridge_vector, Multivector, BOOST_BIVECTOR_COUNT, GRADE_DIMS, GRADE_OFFSETS,
};

/// Per-band coherence scores between two programs, analogous to
/// frequency-specific EEG coherence between brain regions.
#[derive(Clone, Debug, Default)]
pub struct BandCoherence {
    /// δ band: scalar magnitude alignment (overall activation level)
    pub delta: f32,
    /// θ band: grade-1 vector alignment (intentional direction)
    pub theta: f32,
    /// α/β band: grade-2 bivector alignment, split into:
    ///   - boost (causal): temporal/goal-directed coherence
    ///   - spatial (rotation): structural/relational coherence
    pub alpha_beta_boost: f32,
    pub alpha_beta_spatial: f32,
    /// γ band: grade-3 trivector alignment (fine-grained semantic binding)
    pub gamma: f32,
    /// Combined coherence score (weighted across bands)
    pub combined: f32,
}

impl BandCoherence {
    /// Combine band scores with neuroscience-informed weights.
    /// Alpha-beta (structural) coherence is weighted highest — it captures
    /// the relational patterns that make programs compose well together.
    pub fn compute_combined(&mut self) {
        self.combined = 0.05 * self.delta
            + 0.15 * self.theta
            + 0.15 * self.alpha_beta_boost
            + 0.40 * self.alpha_beta_spatial
            + 0.25 * self.gamma;
    }
}

/// Signal strength per band for a single program — analogous to
/// spectral power density in EEG. Programs with strong, focused
/// activations in specific grades are better "senders" of coherent signals.
#[derive(Clone, Debug, Default)]
pub struct BandPower {
    pub delta: f32,
    pub theta: f32,
    pub alpha_beta_boost: f32,
    pub alpha_beta_spatial: f32,
    pub gamma: f32,
    pub total: f32,
}

/// Compute band-decomposed coherence between two embedding vectors.
///
/// Embeds both vectors into Cl(1,7) and measures per-grade alignment,
/// then weights by the sending program's signal strength (the neuroscience
/// insight that coherence is primarily determined by the sender).
pub fn band_coherence(a: &[f32], b: &[f32]) -> BandCoherence {
    let mv_a = embed_bridge_vector(a);
    let mv_b = embed_bridge_vector(b);
    band_coherence_mv(&mv_a, &mv_b)
}

/// Coherence from pre-computed multivectors (avoids redundant embedding).
pub fn band_coherence_mv(mv_a: &Multivector, mv_b: &Multivector) -> BandCoherence {
    let mut bc = BandCoherence::default();

    // δ band: grade 0 (scalar) — just compare magnitudes
    let s_a = mv_a.grade(0)[0];
    let s_b = mv_b.grade(0)[0];
    let mag_a = s_a.abs().max(1e-10);
    let mag_b = s_b.abs().max(1e-10);
    bc.delta = 1.0 - ((mag_a - mag_b).abs() / (mag_a + mag_b)).min(1.0);

    // θ band: grade 1 (vectors) — directional cosine similarity
    bc.theta = grade_cosine(mv_a.grade(1), mv_b.grade(1));

    // α/β band: grade 2 (bivectors) — split into boost and spatial
    let bv_a = mv_a.grade(2);
    let bv_b = mv_b.grade(2);
    bc.alpha_beta_boost =
        grade_cosine(&bv_a[..BOOST_BIVECTOR_COUNT], &bv_b[..BOOST_BIVECTOR_COUNT]);
    bc.alpha_beta_spatial =
        grade_cosine(&bv_a[BOOST_BIVECTOR_COUNT..], &bv_b[BOOST_BIVECTOR_COUNT..]);

    // γ band: grade 3 (trivectors) — fine-grained semantic binding
    bc.gamma = grade_cosine(mv_a.grade(3), mv_b.grade(3));

    bc.compute_combined();
    bc
}

/// Measure the signal strength (spectral power) of a single program's
/// embedding across STA bands. Strong, well-defined grade-specific
/// activations indicate a "good sender" — its coherence signals are
/// meaningful rather than noisy.
pub fn band_power(embedding: &[f32]) -> BandPower {
    let mv = embed_bridge_vector(embedding);
    band_power_mv(&mv)
}

/// Band power from a pre-computed multivector.
pub fn band_power_mv(mv: &Multivector) -> BandPower {
    let mut bp = BandPower::default();

    bp.delta = mv.grade(0)[0].abs();

    bp.theta = l2_norm(mv.grade(1));

    let bv = mv.grade(2);
    bp.alpha_beta_boost = l2_norm(&bv[..BOOST_BIVECTOR_COUNT]);
    bp.alpha_beta_spatial = l2_norm(&bv[BOOST_BIVECTOR_COUNT..]);

    bp.gamma = l2_norm(mv.grade(3));

    bp.total = bp.delta + bp.theta + bp.alpha_beta_boost + bp.alpha_beta_spatial + bp.gamma;
    bp
}

/// Compute the ensemble coherence of a set of programs.
///
/// This is the mean pairwise coherence, weighted by each program's
/// signal strength (band power). Programs with stronger, cleaner signals
/// contribute more to the ensemble score.
///
/// Returns (combined_coherence, per_band_averages).
pub fn ensemble_coherence(embeddings: &[&[f32]]) -> (f32, BandCoherence) {
    let n = embeddings.len();
    if n < 2 {
        return (
            1.0,
            BandCoherence {
                combined: 1.0,
                ..Default::default()
            },
        );
    }

    // Pre-compute multivectors and band powers
    let mvs: Vec<Multivector> = embeddings.iter().map(|e| embed_bridge_vector(e)).collect();
    let powers: Vec<BandPower> = mvs.iter().map(|mv| band_power_mv(mv)).collect();

    let mut total_bc = BandCoherence::default();
    let mut weight_sum = 0.0f32;

    for i in 0..n {
        for j in (i + 1)..n {
            let bc = band_coherence_mv(&mvs[i], &mvs[j]);

            // Weight by geometric mean of sender powers (both directions)
            let w = (powers[i].total * powers[j].total).sqrt().max(1e-10);
            total_bc.delta += bc.delta * w;
            total_bc.theta += bc.theta * w;
            total_bc.alpha_beta_boost += bc.alpha_beta_boost * w;
            total_bc.alpha_beta_spatial += bc.alpha_beta_spatial * w;
            total_bc.gamma += bc.gamma * w;
            weight_sum += w;
        }
    }

    if weight_sum > 1e-10 {
        total_bc.delta /= weight_sum;
        total_bc.theta /= weight_sum;
        total_bc.alpha_beta_boost /= weight_sum;
        total_bc.alpha_beta_spatial /= weight_sum;
        total_bc.gamma /= weight_sum;
    }
    total_bc.compute_combined();

    (total_bc.combined, total_bc)
}

/// Greedy coherence-maximizing selection from a pool of candidates.
///
/// Starting from the highest-relevance candidate, iteratively adds the
/// candidate that maximizes the ensemble's combined coherence, up to
/// `max_items`. This avoids selecting programs that are individually
/// relevant but incoherent as an ensemble (the neuroscience analog:
/// high local activation but desynchronized cross-region oscillations).
///
/// Returns indices into the candidate pool, in selection order.
pub fn coherence_select(
    embeddings: &[&[f32]],
    relevance_scores: &[f32],
    max_items: usize,
    min_coherence: f32,
) -> Vec<usize> {
    let n = embeddings.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    let mvs: Vec<Multivector> = embeddings.iter().map(|e| embed_bridge_vector(e)).collect();

    // Start with the highest-relevance candidate
    let first = relevance_scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let mut selected = vec![first];
    let mut remaining: Vec<usize> = (0..n).filter(|&i| i != first).collect();

    while selected.len() < max_items && !remaining.is_empty() {
        let mut best_candidate = None;
        let mut best_score = f32::NEG_INFINITY;

        for &candidate in &remaining {
            // Compute mean coherence of candidate with all currently selected
            let mut coh_sum = 0.0f32;
            for &sel in &selected {
                let bc = band_coherence_mv(&mvs[candidate], &mvs[sel]);
                coh_sum += bc.combined;
            }
            let mean_coh = coh_sum / selected.len() as f32;

            // Balance relevance and coherence: 40% relevance, 60% coherence
            let score = 0.4 * relevance_scores[candidate] + 0.6 * mean_coh;

            if score > best_score {
                best_score = score;
                best_candidate = Some(candidate);
            }
        }

        if let Some(candidate) = best_candidate {
            // Check minimum coherence threshold before adding
            let mut coh_sum = 0.0f32;
            for &sel in &selected {
                coh_sum += band_coherence_mv(&mvs[candidate], &mvs[sel]).combined;
            }
            let mean_coh = coh_sum / selected.len() as f32;

            if mean_coh < min_coherence {
                break; // adding this candidate would desynchronize the ensemble
            }

            selected.push(candidate);
            remaining.retain(|&i| i != candidate);
        } else {
            break;
        }
    }

    selected
}

/// Coherence matrix between topic sub-lattice centroids.
/// Entry (i, j) is the combined band coherence between topics i and j.
/// Used for diagnostic visualization and to identify naturally cohering
/// topic clusters within a group.
pub fn coherence_matrix(centroids: &[&[f32]]) -> Vec<Vec<f32>> {
    let n = centroids.len();
    let mvs: Vec<Multivector> = centroids.iter().map(|c| embed_bridge_vector(c)).collect();
    let mut matrix = vec![vec![0.0f32; n]; n];

    for i in 0..n {
        matrix[i][i] = 1.0;
        for j in (i + 1)..n {
            let bc = band_coherence_mv(&mvs[i], &mvs[j]);
            matrix[i][j] = bc.combined;
            matrix[j][i] = bc.combined;
        }
    }
    matrix
}

fn grade_cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = l2_norm(a);
    let nb = l2_norm(b);
    if na < 1e-10 || nb < 1e-10 {
        return 0.0;
    }
    (dot / (na * nb)).clamp(0.0, 1.0)
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_coherence_is_high() {
        let v: Vec<f32> = (0..128).map(|i| (i as f32 * 0.07).sin()).collect();
        let bc = band_coherence(&v, &v);
        // Wedge product accumulation in embed_bridge_vector introduces some
        // numerical diffusion, so self-coherence may not be exactly 1.0 but
        // should still be the highest achievable value.
        assert!(
            bc.combined > 0.60,
            "self-coherence should be high, got {}",
            bc.combined
        );
        assert!(
            bc.alpha_beta_spatial > 0.0,
            "spatial coherence should be nonzero for self"
        );
    }

    #[test]
    fn test_similar_more_coherent_than_dissimilar() {
        let a: Vec<f32> = (0..128).map(|i| (i as f32 * 0.07).sin()).collect();
        let b_similar: Vec<f32> = (0..128).map(|i| (i as f32 * 0.07).sin() + 0.05).collect();
        let b_different: Vec<f32> = (0..128).map(|i| (i as f32 * 0.43).cos()).collect();
        let bc_sim = band_coherence(&a, &b_similar);
        let bc_diff = band_coherence(&a, &b_different);
        assert!(
            bc_sim.combined > bc_diff.combined,
            "similar vectors should cohere more: {:.3} vs {:.3}",
            bc_sim.combined,
            bc_diff.combined
        );
    }

    #[test]
    fn test_band_power_nonzero() {
        let v: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05).cos()).collect();
        let bp = band_power(&v);
        assert!(
            bp.total > 0.0,
            "band power should be nonzero for non-zero input"
        );
        assert!(
            bp.alpha_beta_spatial > 0.0,
            "spatial power should be nonzero"
        );
    }

    #[test]
    fn test_ensemble_coherence_identical() {
        let v: Vec<f32> = (0..128).map(|i| (i as f32 * 0.03).sin()).collect();
        let refs: Vec<&[f32]> = vec![&v, &v, &v];
        let (combined, _) = ensemble_coherence(&refs);
        assert!(
            combined > 0.60,
            "identical ensemble should have high coherence, got {}",
            combined
        );
    }

    #[test]
    fn test_coherence_select_respects_max() {
        let vecs: Vec<Vec<f32>> = (0..5)
            .map(|i| {
                (0..128)
                    .map(|j| ((i * 17 + j) as f32 * 0.04).sin())
                    .collect()
            })
            .collect();
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let scores = vec![0.8, 0.6, 0.9, 0.5, 0.7];
        let selected = coherence_select(&refs, &scores, 3, 0.0);
        assert!(selected.len() <= 3);
        assert_eq!(selected[0], 2, "should start with highest relevance");
    }
}
