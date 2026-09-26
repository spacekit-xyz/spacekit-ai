//! Real-valued train ops shared by the vanilla LM core (no Clifford types).

use serde::{Deserialize, Serialize};

use crate::real_linear::LinearReal;

// ─── Adam config ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdamConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    /// AdamW: apply `weight_decay` directly to the weights (`w -= lr·wd·w`)
    /// instead of folding it into the gradient (classic L2, which Adam's
    /// per-parameter scaling largely cancels). When true, biases are not decayed.
    #[serde(default)]
    pub decoupled_weight_decay: bool,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            decoupled_weight_decay: false,
        }
    }
}

// ─── Cross-entropy ────────────────────────────────────────────────────────────

/// Softmax cross-entropy. Returns `(loss, dL/dlogits)`.
///
/// The loss is computed in log-space (`logsumexp − logit[target]`) so it stays
/// finite even when the target's probability underflows f32 (the old
/// `-ln(prob)` form returned `inf` there and poisoned the running loss).
/// The exp-sum is accumulated in f64 for large vocabularies.
pub fn cross_entropy(logits: &[f32], target: usize) -> (f32, Vec<f32>) {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f64 = exps.iter().map(|&e| e as f64).sum();
    let log_z = max as f64 + sum.ln();
    let loss = (log_z - logits[target] as f64) as f32;
    let inv = (1.0 / sum) as f32;
    let mut grad: Vec<f32> = exps.iter().map(|&e| e * inv).collect();
    grad[target] -= 1.0;
    (loss, grad)
}

// ─── Real head gradients ──────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct RealHeadGrad {
    pub d_weights: Vec<Vec<f32>>,
    pub d_bias: Vec<f32>,
}

impl RealHeadGrad {
    pub fn zeros(out_dim: usize, in_features: usize) -> Self {
        Self {
            d_weights: vec![vec![0.0; in_features]; out_dim],
            d_bias: vec![0.0; out_dim],
        }
    }

    pub fn accumulate(&mut self, other: &RealHeadGrad) {
        for o in 0..self.d_weights.len() {
            for j in 0..self.d_weights[o].len() {
                self.d_weights[o][j] += other.d_weights[o][j];
            }
            self.d_bias[o] += other.d_bias[o];
        }
    }

    pub fn scale(&mut self, s: f32) {
        for row in &mut self.d_weights {
            for w in row {
                *w *= s;
            }
        }
        for b in &mut self.d_bias {
            *b *= s;
        }
    }

    /// Sum of squared entries (for a global grad norm across many tensors).
    pub fn sq_norm(&self) -> f32 {
        let mut sq = 0.0f32;
        for row in &self.d_weights {
            for &w in row {
                sq += w * w;
            }
        }
        for &b in &self.d_bias {
            sq += b * b;
        }
        sq
    }

    pub fn bias_sq_norm(&self) -> f32 {
        self.d_bias.iter().map(|b| b * b).sum()
    }

    pub fn norm(&self) -> f32 {
        let mut sq = 0.0f32;
        for row in &self.d_weights {
            for &w in row {
                sq += w * w;
            }
        }
        for &b in &self.d_bias {
            sq += b * b;
        }
        sq.sqrt()
    }

    pub fn clip_norm(&mut self, max_norm: f32) {
        let n = self.norm();
        if n > max_norm && n > 0.0 {
            self.scale(max_norm / n);
        }
    }
}

pub fn real_linear_backward(
    weights: &[Vec<f32>],
    input: &[f32],
    grad_out: &[f32],
    grad: &mut RealHeadGrad,
) -> Vec<f32> {
    let out_dim = weights.len();
    let in_features = input.len();
    let mut grad_input = vec![0.0f32; in_features];

    for o in 0..out_dim {
        let g = grad_out[o];
        if g == 0.0 {
            continue;
        }
        grad.d_bias[o] += g;
        let w = &weights[o];
        let dw = &mut grad.d_weights[o];
        for j in 0..in_features {
            dw[j] += g * input[j];
            grad_input[j] += g * w[j];
        }
    }
    grad_input
}

// ─── Real head Adam ───────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RealHeadOptimizer {
    pub w_m: Vec<Vec<f32>>,
    pub w_v: Vec<Vec<f32>>,
    pub b_m: Vec<f32>,
    pub b_v: Vec<f32>,
    pub step: u64,
    pub cfg: AdamConfig,
}

impl RealHeadOptimizer {
    pub fn new(out_dim: usize, in_features: usize, cfg: AdamConfig) -> Self {
        Self {
            w_m: vec![vec![0.0; in_features]; out_dim],
            w_v: vec![vec![0.0; in_features]; out_dim],
            b_m: vec![0.0; out_dim],
            b_v: vec![0.0; out_dim],
            step: 0,
            cfg,
        }
    }

    /// Coupled (classic L2) coefficient folded into the gradient; 0 under AdamW.
    #[inline]
    fn l2_coef(&self) -> f32 {
        if self.cfg.decoupled_weight_decay {
            0.0
        } else {
            self.cfg.weight_decay
        }
    }

    /// Decoupled (AdamW) per-step shrink factor `lr·wd`; 0 under classic L2.
    #[inline]
    fn decoupled_decay(&self) -> f32 {
        if self.cfg.decoupled_weight_decay {
            self.cfg.lr * self.cfg.weight_decay
        } else {
            0.0
        }
    }

    pub fn step(&mut self, head: &mut LinearReal, grad: &RealHeadGrad) {
        self.step += 1;
        let t = self.step as f32;
        let bc1 = 1.0 - self.cfg.beta1.powf(t);
        let bc2 = 1.0 - self.cfg.beta2.powf(t);
        let l2 = self.l2_coef();
        let decay = self.decoupled_decay();

        for o in 0..head.out_dim {
            {
                let g = grad.d_bias[o] + self.l2_coef() * head.bias[o];
                self.b_m[o] = self.cfg.beta1 * self.b_m[o] + (1.0 - self.cfg.beta1) * g;
                self.b_v[o] = self.cfg.beta2 * self.b_v[o] + (1.0 - self.cfg.beta2) * g * g;
                let m_hat = self.b_m[o] / bc1;
                let v_hat = self.b_v[o] / bc2;
                head.bias[o] -= self.cfg.lr * m_hat / (v_hat.sqrt() + self.cfg.eps);
            }
            let w = &mut head.weights[o];
            let wm = &mut self.w_m[o];
            let wv = &mut self.w_v[o];
            let dw = &grad.d_weights[o];
            for j in 0..head.in_features {
                let g = dw[j] + l2 * w[j];
                wm[j] = self.cfg.beta1 * wm[j] + (1.0 - self.cfg.beta1) * g;
                wv[j] = self.cfg.beta2 * wv[j] + (1.0 - self.cfg.beta2) * g * g;
                let m_hat = wm[j] / bc1;
                let v_hat = wv[j] / bc2;
                w[j] -= self.cfg.lr * m_hat / (v_hat.sqrt() + self.cfg.eps) + decay * w[j];
            }
        }
    }

    pub fn step_bias_only(&mut self, head: &mut LinearReal, grad: &RealHeadGrad) {
        self.step += 1;
        let t = self.step as f32;
        let bc1 = 1.0 - self.cfg.beta1.powf(t);
        let bc2 = 1.0 - self.cfg.beta2.powf(t);
        for o in 0..head.out_dim {
            let g = grad.d_bias[o] + self.l2_coef() * head.bias[o];
            self.b_m[o] = self.cfg.beta1 * self.b_m[o] + (1.0 - self.cfg.beta1) * g;
            self.b_v[o] = self.cfg.beta2 * self.b_v[o] + (1.0 - self.cfg.beta2) * g * g;
            let m_hat = self.b_m[o] / bc1;
            let v_hat = self.b_v[o] / bc2;
            head.bias[o] -= self.cfg.lr * m_hat / (v_hat.sqrt() + self.cfg.eps);
        }
    }
}

/// Linear warmup from `lr_min` to `lr_max`, then cosine decay back to `lr_min`
/// at `total_steps`. `t` is the 0-based index of the update about to be applied.
pub fn cosine_lr_with_warmup(
    t: u64,
    warmup_steps: u64,
    total_steps: u64,
    lr_max: f32,
    lr_min: f32,
) -> f32 {
    if t < warmup_steps {
        lr_min + (lr_max - lr_min) * (t as f32 / warmup_steps as f32)
    } else {
        let progress = ((t - warmup_steps) as f32
            / total_steps.saturating_sub(warmup_steps).max(1) as f32)
            .min(1.0);
        let cos_val = (std::f32::consts::PI * progress).cos();
        lr_min + 0.5 * (lr_max - lr_min) * (1.0 + cos_val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_entropy_is_finite_when_target_prob_underflows() {
        let mut logits = vec![0.0f32; 8];
        logits[0] = 200.0; // target 1 has p ≈ e^-200 → underflows f32
        let (loss, grad) = cross_entropy(&logits, 1);
        assert!(loss.is_finite(), "loss must stay finite, got {loss}");
        assert!((loss - 200.0).abs() < 1e-3);
        assert!((grad[1] + 1.0).abs() < 1e-6);
        assert!((grad[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cross_entropy_matches_naive_on_small_logits() {
        let logits = [0.1f32, -0.4, 1.3, 0.2];
        let (loss, grad) = cross_entropy(&logits, 2);
        let z: f32 = logits.iter().map(|l| l.exp()).sum();
        let naive = -(logits[2].exp() / z).ln();
        assert!((loss - naive).abs() < 1e-6);
        let gs: f32 = grad.iter().sum();
        assert!(gs.abs() < 1e-6, "softmax grad sums to 0");
    }

    #[test]
    fn cosine_schedule_clamps_past_total() {
        let lr = cosine_lr_with_warmup(5_000, 10, 1_000, 1e-3, 1e-5);
        assert!((lr - 1e-5).abs() < 1e-9);
    }

    #[test]
    fn adamw_decays_weights_but_not_bias() {
        let mut head = LinearReal::new_dims(2, 1, 0);
        head.weights[0] = vec![1.0, 1.0];
        head.bias[0] = 1.0;
        let mut opt = RealHeadOptimizer::new(
            1,
            2,
            AdamConfig {
                lr: 0.1,
                weight_decay: 0.5,
                decoupled_weight_decay: true,
                ..Default::default()
            },
        );
        let zero = RealHeadGrad::zeros(1, 2);
        opt.step(&mut head, &zero);
        assert!((head.weights[0][0] - 0.95).abs() < 1e-6);
        assert!((head.bias[0] - 1.0).abs() < 1e-6);
    }
}
