// backprop.rs — Gradients and backward pass for Clifford LLM layers
//
// The geometric product is bilinear, so gradients follow the standard
// product rule.  Given C = geo(A, B):
//
//   dL/dA[i] = Σ_j  B[j] * cayley[i][j].sign * dL/dC[ cayley[i][j].blade ]
//   dL/dB[j] = Σ_i  A[i] * cayley[i][j].sign * dL/dC[ cayley[i][j].blade ]
//
// These are computed via the Cayley table without any additional overhead.
// For training use an outer loop that calls layer_backward to accumulate
// GradLinear structs, then pass them to optim::adam_step.

use crate::cayley_const::{CliffordAlgebraConst, CAYLEY_STA};
use crate::{CliffordLinear, Multivector};
use std::sync::Arc;

// Re-export real ops used by Clifford tape/train (canonical defs in real_ops).
pub use crate::real_ops::{cross_entropy, real_linear_backward, RealHeadGrad};

// ─── Gradient types ───────────────────────────────────────────────────────────

/// Gradient of a single CliffordLinear layer.
/// Mirrors the weight/bias structure — same dimensions, components are ∂L/∂param.
pub struct GradLinear {
    pub d_weights: Vec<Vec<Multivector>>, // [out_dim][in_dim]
    pub d_biases: Vec<Multivector>,       // [out_dim]
}

impl GradLinear {
    pub fn zeros(out_dim: usize, in_dim: usize) -> Self {
        Self {
            d_weights: vec![vec![Multivector::zero(); in_dim]; out_dim],
            d_biases: vec![Multivector::zero(); out_dim],
        }
    }

    /// Accumulate gradients from another GradLinear (useful for batch averaging).
    pub fn accumulate(&mut self, other: &GradLinear) {
        for d in 0..self.d_weights.len() {
            for i in 0..self.d_weights[d].len() {
                for k in 0..16 {
                    self.d_weights[d][i].c[k] += other.d_weights[d][i].c[k];
                }
            }
            for k in 0..16 {
                self.d_biases[d].c[k] += other.d_biases[d].c[k];
            }
        }
    }

    /// Scale all gradients by a scalar (e.g. 1/batch_size).
    pub fn scale(&mut self, s: f32) {
        for row in &mut self.d_weights {
            for mv in row {
                mv.c.iter_mut().for_each(|v| *v *= s);
            }
        }
        for mv in &mut self.d_biases {
            mv.c.iter_mut().for_each(|v| *v *= s);
        }
    }
}

// ─── Primitive: backward through a single geometric product ──────────────────

/// Given C = geo(A, B) and grad_C = dL/dC,
/// returns (grad_A = dL/dA, grad_B = dL/dB).
pub fn geo_product_backward(
    a: &Multivector,
    b: &Multivector,
    grad_c: &Multivector,
) -> (Multivector, Multivector) {
    let mut grad_a = [0.0f32; 16];
    let mut grad_b = [0.0f32; 16];

    for i in 0..16 {
        for j in 0..16 {
            let cell = CAYLEY_STA[i][j];
            let sign = cell.sign as f32;
            let k = cell.blade as usize;
            let g_k = grad_c.c[k];

            // dL/dA[i] += B[j] * sign * dL/dC[k]
            grad_a[i] += b.c[j] * sign * g_k;
            // dL/dB[j] += A[i] * sign * dL/dC[k]
            grad_b[j] += a.c[i] * sign * g_k;
        }
    }

    (Multivector { c: grad_a }, Multivector { c: grad_b })
}

// ─── CliffordLinear backward ──────────────────────────────────────────────────

/// Backward pass through a CliffordLinear layer.
///
/// Forward:  out[d] = Σ_i geo(W[d][i], x[i]) + bias[d]
///
/// Returns:
///   - GradLinear:       ∂L/∂W and ∂L/∂bias for this layer
///   - Vec<Multivector>: ∂L/∂x   to propagate to the layer below
///
/// `inputs`   — the x slice passed during the forward call  (length in_dim)
/// `grad_out` — dL/dOut, one multivector per output position (length out_dim)
pub fn linear_backward(
    weights: &[Vec<Multivector>], // [out_dim][in_dim]
    inputs: &[Multivector],       // [in_dim]
    grad_out: &[Multivector],     // [out_dim]
) -> (GradLinear, Vec<Multivector>) {
    let out_dim = weights.len();
    let in_dim = inputs.len();
    let mut grad = GradLinear::zeros(out_dim, in_dim);
    let mut grad_x = vec![Multivector::zero(); in_dim];

    for d in 0..out_dim {
        // bias gradient: dL/dbias[d] = grad_out[d]
        grad.d_biases[d] = grad_out[d].clone();

        for i in 0..in_dim {
            let (dw, dx) = geo_product_backward(&weights[d][i], &inputs[i], &grad_out[d]);
            // Accumulate weight gradient
            for k in 0..16 {
                grad.d_weights[d][i].c[k] += dw.c[k];
            }
            // Accumulate input gradient
            for k in 0..16 {
                grad_x[i].c[k] += dx.c[k];
            }
        }
    }

    (grad, grad_x)
}

// ─── Loss: cross-entropy over scalar logits ───────────────────────────────────

/// Computes cross-entropy loss and the gradient of the loss w.r.t. logits.
/// (Canonical definition in `real_ops`; re-exported above.)

/// Scatter scalar logit gradients back into multivector grad_out for the output head.
/// The head extracts grade-0 (scalar part), so only component 0 of each
/// output multivector gradient is non-zero.
///
/// `grad_logits` — dL/d(logit[d]) for each output dim d
pub fn scalar_head_backward(grad_logits: &[f32]) -> Vec<Multivector> {
    grad_logits
        .iter()
        .map(|&g| {
            let mut mv = Multivector::zero();
            mv.c[0] = g; // grade-0 only
            mv
        })
        .collect()
}

// ─── Real output head backward (Clifford flatten path) ────────────────────────

/// Backward through the real output head for one position (Clifford flatten path).
///
/// Forward:  logit[o] = bias[o] + Σ_j W[o][j] · flat[j]      (flat = flatten(head_input))
pub fn real_head_backward(
    weights: &[Vec<f32>],
    head_input: &[Multivector],
    grad_logits: &[f32],
    grad: &mut RealHeadGrad,
) -> Vec<Multivector> {
    let flat: Vec<f32> = head_input.iter().flat_map(|mv| mv.c).collect();
    let grad_flat = real_linear_backward(weights, &flat, grad_logits, grad);
    crate::ffn::unflatten_mvs(&grad_flat, head_input.len())
}

// ─── Layer-norm backward (simplified) ────────────────────────────────────────

/// Backward through the flat layer-norm used in CliffordLayerNorm.
///
/// For layer-norm y = (x − μ) / σ * γ + β, the gradient w.r.t. x is:
///   dx = (1/σ) * (dL/dy * γ − mean(dL/dy * γ) − y_hat * mean(dL/dy * γ * y_hat))
///
/// Returns grad_x as a flat Vec<f32> of length 16 × d_model.
pub fn layer_norm_backward(x_hat: &[f32], gamma: &[f32], grad_out: &[f32], std: f32) -> Vec<f32> {
    crate::clifford_layer_norm::backward_flat(x_hat, gamma, grad_out, std)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Numerical gradient check for geo_product_backward.
    /// Perturbs each component of A by ε and checks finite-difference ≈ analytic grad.
    #[test]
    fn numerical_grad_check_a() {
        let mut a = Multivector::zero();
        let mut b = Multivector::zero();
        let mut grad_c = Multivector::zero();
        // Arbitrary values
        a.c[1] = 0.7;
        a.c[6] = -0.3;
        b.c[2] = 1.2;
        b.c[5] = 0.4;
        grad_c.c[0] = 1.0;
        grad_c.c[3] = -0.5;

        let alg = CliffordAlgebraConst::new();
        let (grad_a, _) = geo_product_backward(&a, &b, &grad_c);

        let eps = 1e-4;
        for i in 0..16 {
            let mut a_plus = a.clone();
            a_plus.c[i] += eps;
            let mut a_minus = a.clone();
            a_minus.c[i] -= eps;
            let c_plus = alg.geo_product(&a_plus, &b);
            let c_minus = alg.geo_product(&a_minus, &b);
            // Directional loss: L = Σ_k grad_c[k] * C[k]
            let loss_plus: f32 = (0..16).map(|k| grad_c.c[k] * c_plus.c[k]).sum();
            let loss_minus: f32 = (0..16).map(|k| grad_c.c[k] * c_minus.c[k]).sum();
            let fd = (loss_plus - loss_minus) / (2.0 * eps);
            assert!(
                (grad_a.c[i] - fd).abs() < 1e-3,
                "grad_A[{i}]: analytic={:.6}, fd={:.6}",
                grad_a.c[i],
                fd
            );
        }
    }

    #[test]
    fn cross_entropy_grad_sums_to_zero() {
        let logits = vec![1.0f32, 2.0, 0.5, -1.0];
        let (_, grad) = cross_entropy(&logits, 1);
        // Softmax grad sums to 0 except for the +1 shift which we subtracted
        let sum: f32 = grad.iter().sum();
        assert!(
            sum.abs() < 1e-6,
            "cross-entropy grad should sum to 0, got {sum}"
        );
    }
}
