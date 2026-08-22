// ── disentanglement.rs ────────────────────────────────────────────────────────
// Full three-term disentanglement loss stack:
//   1. Cosine separation      — push sentiment/entity branches apart per sample
//   2. Orthogonality in expectation — mean(s·e) → 0 across the batch
//   3. Contrastive alignment  — same-sentiment embeddings cluster,
//                               different-sentiment embeddings repel
//
// Cross-branch dropout is implemented here and called from GrowformerNode.
// Each term can be individually weighted and annealed across training stages.

use crate::category::training::SentimentLabel;

// ── Vector helpers ────────────────────────────────────────────────────────────

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let na = norm(a);
    let nb = norm(b);
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot(a, b) / (na * nb)
}

/// Clamp a value to [min, max].
fn clamp(v: f32, min: f32, max: f32) -> f32 {
    v.max(min).min(max)
}

// ── DisentanglementWeights ────────────────────────────────────────────────────

/// Per-term weights for the disentanglement loss.
/// All weights can be annealed by the curriculum scheduler.
#[derive(Debug, Clone)]
pub struct DisentanglementWeights {
    /// Weight for per-sample cosine separation (term 1).
    pub cosine: f32,
    /// Weight for batch-level orthogonality in expectation (term 2).
    pub ortho: f32,
    /// Weight for contrastive alignment across same/different sentiments (term 3).
    pub contrastive: f32,
    /// Contrastive margin: different-class pairs are only penalized if similarity
    /// exceeds this margin.
    pub contrastive_margin: f32,
}

impl Default for DisentanglementWeights {
    fn default() -> Self {
        Self {
            cosine: 0.1,
            ortho: 0.1,
            contrastive: 0.2,
            contrastive_margin: 0.3,
        }
    }
}

impl DisentanglementWeights {
    pub fn stage_1() -> Self {
        // Scaffold: light disentanglement, auxiliary labels carry more weight
        Self {
            cosine: 0.05,
            ortho: 0.05,
            contrastive: 0.05,
            contrastive_margin: 0.5,
        }
    }

    pub fn stage_2() -> Self {
        // Loosen: full disentanglement pressure, no auxiliary labels
        Self {
            cosine: 0.15,
            ortho: 0.15,
            contrastive: 0.3,
            contrastive_margin: 0.3,
        }
    }

    pub fn stage_3() -> Self {
        // Harden: reduced contrastive (clusters already formed), tighter ortho
        Self {
            cosine: 0.1,
            ortho: 0.2,
            contrastive: 0.1,
            contrastive_margin: 0.2,
        }
    }
}

// ── DisentanglementLoss ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DisentanglementLoss {
    pub weights: DisentanglementWeights,
}

impl DisentanglementLoss {
    pub fn new(weights: DisentanglementWeights) -> Self {
        Self { weights }
    }

    /// Compute the full three-term loss over a batch.
    ///
    /// # Arguments
    /// - `sentiment_batch`: left-branch embeddings, one per sample
    /// - `entity_batch`:    right-branch embeddings, one per sample
    /// - `sentiment_labels`: ground-truth sentiment labels (used for contrastive)
    ///
    /// Returns `(total_loss, term1_cosine, term2_ortho, term3_contrastive)`
    /// so callers can log each term independently.
    pub fn compute(
        &self,
        sentiment_batch: &[Vec<f32>],
        entity_batch: &[Vec<f32>],
        sentiment_labels: &[SentimentLabel],
    ) -> LossBreakdown {
        assert_eq!(
            sentiment_batch.len(),
            entity_batch.len(),
            "Sentiment and entity batch sizes must match"
        );
        assert_eq!(
            sentiment_batch.len(),
            sentiment_labels.len(),
            "Batch size and label count must match"
        );

        if sentiment_batch.is_empty() {
            return LossBreakdown::zero();
        }

        let cosine = self.cosine_term(sentiment_batch, entity_batch);
        let ortho = self.orthogonality_term(sentiment_batch, entity_batch);
        let contrast = self.contrastive_term(sentiment_batch, sentiment_labels);

        let total = self.weights.cosine * cosine
            + self.weights.ortho * ortho
            + self.weights.contrastive * contrast;

        LossBreakdown {
            total,
            cosine,
            ortho,
            contrastive: contrast,
        }
    }

    // ── Term 1: per-sample cosine separation ─────────────────────────────────

    /// Mean absolute cosine similarity between sentiment and entity embeddings.
    /// We want this → 0 (branches point in orthogonal directions).
    fn cosine_term(&self, s: &[Vec<f32>], e: &[Vec<f32>]) -> f32 {
        let sum: f32 = s
            .iter()
            .zip(e.iter())
            .map(|(si, ei)| cosine_sim(si, ei).abs())
            .sum();
        sum / s.len() as f32
    }

    // ── Term 2: orthogonality in expectation ─────────────────────────────────

    /// Mean dot product across batch → 0.
    /// Stricter than per-sample cosine: enforces that the *average* inner product
    /// is zero, not just that each pair is roughly orthogonal.
    fn orthogonality_term(&self, s: &[Vec<f32>], e: &[Vec<f32>]) -> f32 {
        let n = s.len() as f32;
        let sum: f32 = s.iter().zip(e.iter()).map(|(si, ei)| dot(si, ei)).sum();
        (sum / n).abs()
    }

    // ── Term 3: contrastive alignment on sentiment branch ────────────────────

    /// Supervised contrastive loss on the sentiment branch only.
    /// - Same-label pairs: minimise distance (loss = 1 - sim)
    /// - Different-label pairs: push apart if similarity > margin
    ///
    /// Operates over all unique pairs in the batch — O(n²) but batches are
    /// small enough that this is fine during training.
    fn contrastive_term(&self, s: &[Vec<f32>], labels: &[SentimentLabel]) -> f32 {
        let mut loss = 0.0f32;
        let mut count = 0usize;

        for i in 0..s.len() {
            for j in (i + 1)..s.len() {
                let sim = cosine_sim(&s[i], &s[j]);
                let same = labels[i] == labels[j];
                if same {
                    // Pull same-class embeddings together
                    loss += 1.0 - sim;
                } else {
                    // Push different-class embeddings apart beyond the margin
                    loss += clamp(sim - self.weights.contrastive_margin, 0.0, 1.0);
                }
                count += 1;
            }
        }

        if count == 0 {
            0.0
        } else {
            loss / count as f32
        }
    }
}

// ── LossBreakdown ─────────────────────────────────────────────────────────────

/// Per-term loss values for logging and diagnostics.
#[derive(Debug, Clone, Default)]
pub struct LossBreakdown {
    pub total: f32,
    pub cosine: f32,
    pub ortho: f32,
    pub contrastive: f32,
}

impl LossBreakdown {
    pub fn zero() -> Self {
        Self::default()
    }

    pub fn display(&self) {
        println!(
            "  dis_total={:.4}  cosine={:.4}  ortho={:.4}  contrastive={:.4}",
            self.total, self.cosine, self.ortho, self.contrastive
        );
    }
}

// ── Cross-branch dropout ──────────────────────────────────────────────────────

/// Result of a forward pass through the bifunctor split with dropout applied.
#[derive(Debug)]
pub struct BifunctorOutput<A> {
    pub left: Option<A>,  // sentiment branch (None = dropped)
    pub right: Option<A>, // entity branch    (None = dropped)
    pub both_active: bool,
}

impl<A: Default + Clone> BifunctorOutput<A> {
    /// Return left output, falling back to right if dropped, then default.
    pub fn sentiment_or_fallback(&self) -> A {
        self.left
            .clone()
            .or_else(|| self.right.clone())
            .unwrap_or_default()
    }

    /// Return right output, falling back to left if dropped, then default.
    pub fn entity_or_fallback(&self) -> A {
        self.right
            .clone()
            .or_else(|| self.left.clone())
            .unwrap_or_default()
    }
}

/// Apply cross-branch dropout. Never drops both branches simultaneously.
/// p=0.0 disables dropout (eval mode).
pub fn cross_branch_dropout<A: Clone>(
    left: A,
    right: A,
    dropout_p: f32,
    rng: &mut SimpleRng,
) -> BifunctorOutput<A> {
    if dropout_p <= 0.0 {
        return BifunctorOutput {
            left: Some(left),
            right: Some(right),
            both_active: true,
        };
    }
    let r = rng.gen_f32();
    if r < dropout_p / 2.0 {
        BifunctorOutput {
            left: None,
            right: Some(right),
            both_active: false,
        }
    } else if r < dropout_p {
        BifunctorOutput {
            left: Some(left),
            right: None,
            both_active: false,
        }
    } else {
        BifunctorOutput {
            left: Some(left),
            right: Some(right),
            both_active: true,
        }
    }
}

// ── SimpleRng ─────────────────────────────────────────────────────────────────
// Minimal LCG-based RNG so callers don't need the `rand` crate.
// Replace with rand::thread_rng() in production.

pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        // Splitmix64
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Generate a float in [0, 1).
    pub fn gen_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

// ── Combined loss helper ───────────────────────────────────────────────────────

/// Task loss + full disentanglement stack.
pub fn combined_loss_full(
    task_loss: f32,
    sentiment_batch: &[Vec<f32>],
    entity_batch: &[Vec<f32>],
    sentiment_labels: &[SentimentLabel],
    dis: &DisentanglementLoss,
) -> (f32, LossBreakdown) {
    let breakdown = dis.compute(sentiment_batch, entity_batch, sentiment_labels);
    (task_loss + breakdown.total, breakdown)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::category::training::SentimentLabel as SL;

    #[test]
    fn cosine_sim_parallel_is_one() {
        let v = vec![1.0f32, 2.0, 3.0];
        assert!((cosine_sim(&v, &v) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cross_branch_dropout_never_drops_both() {
        let mut rng = SimpleRng::new(123);
        for _ in 0..200 {
            let o = cross_branch_dropout(vec![1.0f32], vec![2.0f32], 0.9, &mut rng);
            assert!(o.left.is_some() || o.right.is_some());
        }
    }

    #[test]
    fn combined_loss_full_increases_with_correlated_branches() {
        let dis = DisentanglementLoss::new(DisentanglementWeights::default());
        let v = vec![1.0f32, 0.0, 0.0];
        let labels = [SL::Neutral];
        let (t1, _) = combined_loss_full(0.0, &[v.clone()], &[v.clone()], &labels, &dis);
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        let (t2, _) = combined_loss_full(0.0, &[a], &[b], &labels, &dis);
        assert!(
            t2 < t1,
            "orthogonal branches should lower disentanglement loss"
        );
    }
}
