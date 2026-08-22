// optim.rs — Adam optimiser for multivector parameters
//
// Each of the 16 float components of every Multivector weight is treated as
// an independent scalar parameter.  Adam maintains per-component first and
// second moment estimates (m, v) and applies the bias-corrected update.
//
// Usage pattern:
//   1. Create one MvAdamState per learnable Multivector (or use ModelOptimizer
//      which manages a table for an entire layer).
//   2. After computing GradLinear from backprop::linear_backward, call
//      adam_step for each parameter multivector.
//   3. Replace the parameter in-place with the returned updated value.

use crate::backprop::GradLinear;
use crate::Multivector;

// Real Adam config / head optimiser live in real_ops (vanilla-first).
pub use crate::real_ops::{cosine_lr_with_warmup, AdamConfig, RealHeadOptimizer};

// ─── Per-parameter state ──────────────────────────────────────────────────────

/// Adam moment estimates for a single Multivector (16 components).
#[derive(Clone, Debug)]
pub struct MvAdamState {
    pub m: [f32; 16], // first moment  (mean of gradients)
    pub v: [f32; 16], // second moment (uncentred variance)
    pub step: u64,    // number of updates applied so far
}

impl MvAdamState {
    pub fn zero() -> Self {
        Self {
            m: [0.0; 16],
            v: [0.0; 16],
            step: 0,
        }
    }
}

// ─── Core Adam step ───────────────────────────────────────────────────────────

/// Apply one Adam update to `param` given gradient `grad`.
///
/// Returns the updated parameter and mutates `state` in-place.
pub fn adam_step(
    param: &Multivector,
    grad: &Multivector,
    state: &mut MvAdamState,
    cfg: &AdamConfig,
) -> Multivector {
    state.step += 1;
    let t = state.step as f32;
    let b1 = cfg.beta1;
    let b2 = cfg.beta2;

    let bc1 = 1.0 - b1.powf(t); // bias correction 1
    let bc2 = 1.0 - b2.powf(t); // bias correction 2

    let mut new_c = [0.0f32; 16];

    for k in 0..16 {
        let g = grad.c[k] + cfg.weight_decay * param.c[k];

        // Update biased moments
        state.m[k] = b1 * state.m[k] + (1.0 - b1) * g;
        state.v[k] = b2 * state.v[k] + (1.0 - b2) * g * g;

        // Bias-corrected estimates
        let m_hat = state.m[k] / bc1;
        let v_hat = state.v[k] / bc2;

        new_c[k] = param.c[k] - cfg.lr * m_hat / (v_hat.sqrt() + cfg.eps);
    }

    Multivector { c: new_c }
}

// ─── Layer-level optimiser ────────────────────────────────────────────────────

/// Holds Adam states for every weight/bias multivector in a CliffordLinear layer.
pub struct LayerOptimizer {
    pub weight_states: Vec<Vec<MvAdamState>>, // [out_dim][in_dim]
    pub bias_states: Vec<MvAdamState>,        // [out_dim]
    pub cfg: AdamConfig,
}

impl LayerOptimizer {
    pub fn new(out_dim: usize, in_dim: usize, cfg: AdamConfig) -> Self {
        Self {
            weight_states: vec![vec![MvAdamState::zero(); in_dim]; out_dim],
            bias_states: vec![MvAdamState::zero(); out_dim],
            cfg,
        }
    }

    /// Apply Adam to every weight and bias of a layer.
    ///
    /// Mutates `weights` and `biases` in place using the gradients in `grad`.
    pub fn step(
        &mut self,
        weights: &mut Vec<Vec<Multivector>>,
        biases: &mut Vec<Multivector>,
        grad: &GradLinear,
    ) {
        let out_dim = weights.len();
        let in_dim = weights[0].len();

        for d in 0..out_dim {
            biases[d] = adam_step(
                &biases[d],
                &grad.d_biases[d],
                &mut self.bias_states[d],
                &self.cfg,
            );
            for i in 0..in_dim {
                weights[d][i] = adam_step(
                    &weights[d][i],
                    &grad.d_weights[d][i],
                    &mut self.weight_states[d][i],
                    &self.cfg,
                );
            }
        }
    }
}

// Learning-rate schedule + real head optimiser: see `real_ops` (re-exported above).

/// Gradient norm across all components of a GradLinear (useful for grad clipping).
pub fn grad_norm(grad: &GradLinear) -> f32 {
    let mut sq: f32 = 0.0;
    for row in &grad.d_weights {
        for mv in row {
            sq += mv.c.iter().map(|&c| c * c).sum::<f32>();
        }
    }
    for mv in &grad.d_biases {
        sq += mv.c.iter().map(|&c| c * c).sum::<f32>();
    }
    sq.sqrt()
}

/// Clip gradients in GradLinear so their norm does not exceed `max_norm`.
pub fn clip_grad_norm(grad: &mut GradLinear, max_norm: f32) {
    let norm = grad_norm(grad);
    if norm > max_norm {
        let scale = max_norm / norm;
        for row in &mut grad.d_weights {
            for mv in row {
                mv.c.iter_mut().for_each(|c| *c *= scale);
            }
        }
        for mv in &mut grad.d_biases {
            mv.c.iter_mut().for_each(|c| *c *= scale);
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adam_decreases_loss_simple() {
        // A trivial convex problem: minimise ||param||²
        // grad = 2 * param, optimum at 0.
        let cfg = AdamConfig {
            lr: 0.1,
            ..Default::default()
        };
        let mut state = MvAdamState::zero();
        let mut param = Multivector::scalar(1.0);

        for _ in 0..200 {
            let mut grad = Multivector::zero();
            grad.c[0] = 2.0 * param.c[0]; // dL/dparam = 2*param
            param = adam_step(&param, &grad, &mut state, &cfg);
        }

        assert!(
            param.c[0].abs() < 0.05,
            "Adam should converge near 0, got {}",
            param.c[0]
        );
    }

    #[test]
    fn lr_schedule_warmup() {
        let lr = cosine_lr_with_warmup(0, 100, 1000, 1e-3, 1e-5);
        assert!((lr - 1e-5).abs() < 1e-9, "step 0 should be lr_min");
        let lr = cosine_lr_with_warmup(100, 100, 1000, 1e-3, 1e-5);
        assert!((lr - 1e-3).abs() < 1e-6, "end of warmup should be lr_max");
    }

    #[test]
    fn grad_clipping_respects_max_norm() {
        let mut grad = crate::backprop::GradLinear::zeros(2, 2);
        // Set a large gradient
        grad.d_weights[0][0].c[0] = 100.0;
        clip_grad_norm(&mut grad, 1.0);
        let norm = grad_norm(&grad);
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "norm after clipping should be 1, got {norm}"
        );
    }
}
