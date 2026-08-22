// ── linear_head.rs ────────────────────────────────────────────────────────────
// Small softmax classifier on branch vectors; SGD step for use in the trainer.

use crate::category::disentanglement::SimpleRng;

/// Row-major weights: `w[k * in_dim + i]` connects input `i` to logit `k`.
#[derive(Debug, Clone)]
pub struct LinearHead {
    pub in_dim: usize,
    pub out_dim: usize,
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

impl LinearHead {
    pub fn new_zeros(in_dim: usize, out_dim: usize) -> Self {
        Self {
            in_dim,
            out_dim,
            w: vec![0.0f32; out_dim * in_dim],
            b: vec![0.0f32; out_dim],
        }
    }

    pub fn new_random(in_dim: usize, out_dim: usize, rng: &mut SimpleRng) -> Self {
        let scale = (1.0f32 / in_dim.max(1) as f32).sqrt();
        let w: Vec<f32> = (0..(out_dim * in_dim))
            .map(|_| (rng.gen_f32() * 2.0 - 1.0) * scale)
            .collect();
        Self {
            in_dim,
            out_dim,
            w,
            b: vec![0.0f32; out_dim],
        }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.in_dim);
        let mut logits = vec![0.0f32; self.out_dim];
        for k in 0..self.out_dim {
            let mut acc = self.b[k];
            let row = k * self.in_dim;
            for i in 0..self.in_dim {
                acc += self.w[row + i] * x[i];
            }
            logits[k] = acc;
        }
        logits
    }

    pub fn softmax(logits: &[f32]) -> Vec<f32> {
        let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|&z| (z - m).exp()).collect();
        let sum: f32 = exp.iter().sum::<f32>().max(1e-10);
        exp.iter().map(|e| e / sum).collect()
    }

    /// Cross-entropy loss only (no weight update).
    pub fn cross_entropy(&self, x: &[f32], target: usize) -> f32 {
        let probs = Self::softmax(&self.forward(x));
        -probs[target].max(1e-10).ln()
    }

    pub fn predict_class(&self, x: &[f32]) -> usize {
        self.predict_with_probs(x).0
    }

    /// Argmax class index, logits, softmax probabilities, and probability at the argmax.
    pub fn predict_with_probs(&self, x: &[f32]) -> (usize, Vec<f32>, Vec<f32>, f32) {
        let logits = self.forward(x);
        let probs = Self::softmax(&logits);
        let idx = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let conf = probs.get(idx).copied().unwrap_or(0.0);
        (idx, logits, probs, conf)
    }

    /// ∂CE/∂x for softmax cross-entropy (matches [`Self::step_ce`] before the weight update).
    pub fn grad_input_ce(&self, x: &[f32], target: usize) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.in_dim);
        let logits = self.forward(x);
        let mut probs = Self::softmax(&logits);
        probs[target] -= 1.0;
        let mut gx = vec![0.0f32; self.in_dim];
        for k in 0..self.out_dim {
            let dl = probs[k];
            let row = k * self.in_dim;
            for i in 0..self.in_dim {
                gx[i] += dl * self.w[row + i];
            }
        }
        gx
    }

    /// One SGD step: cross-entropy gradient on logits. Returns the loss value.
    pub fn step_ce(&mut self, x: &[f32], target: usize, lr: f32) -> f32 {
        debug_assert_eq!(x.len(), self.in_dim);
        let logits = self.forward(x);
        let mut probs = Self::softmax(&logits);
        let loss = -probs[target].max(1e-10).ln();
        probs[target] -= 1.0;

        for k in 0..self.out_dim {
            let dl = probs[k];
            self.b[k] -= lr * dl;
            let row = k * self.in_dim;
            for i in 0..self.in_dim {
                self.w[row + i] -= lr * dl * x[i];
            }
        }
        loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let p = LinearHead::softmax(&[1.0f32, 2.0f32, 3.0f32]);
        let s: f32 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn predict_with_probs_confidence_is_argmax_prob() {
        let h = LinearHead::new_zeros(2, 3);
        let x = vec![1.0f32, 0.0];
        let (idx, _, probs, conf) = h.predict_with_probs(&x);
        assert_eq!(
            idx,
            probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        );
        assert!((probs[idx] - conf).abs() < 1e-5);
    }

    #[test]
    fn step_ce_reduces_loss_on_repeat() {
        let mut rng = SimpleRng::new(7);
        let mut h = LinearHead::new_random(4, 3, &mut rng);
        let x = vec![0.2f32, -0.3, 0.5, 0.1];
        let t = 1usize;
        let l0 = h.cross_entropy(&x, t);
        for _ in 0..80 {
            h.step_ce(&x, t, 0.2);
        }
        let l1 = h.cross_entropy(&x, t);
        assert!(l1 < l0, "expected loss to drop, got {} -> {}", l0, l1);
    }
}
