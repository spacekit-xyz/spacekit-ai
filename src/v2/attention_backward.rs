// attention_backward.rs — Full backward pass through CliffordAttention
//
// The forward pass produces:
//
//     Q[i][d] = (W_Q ⊛ x[i])[d]            — d_model multivectors per position
//     K[j][d] = (W_K ⊛ x[j])[d]
//     V[j][d] = (W_V ⊛ x[j])[d]
//
//     score[i][j] = (1/scale) Σ_d ⟨Q[i][d] ⊛ K[j][d]⟩₀
//     w[i][j]     = softmax_j(score[i][j])
//     agg[i][d]   = Σ_j w[i][j] · V[j][d]
//     out[i]      = W_O ⊛ agg[i]
//
// Backward derivation (all gradients aligned with dL/dout[i] from upstream):
//
//   1. w_o backward (standard CliffordLinear backward):
//        grad_W_O, grad_agg[i]   ← linear_backward(W_O, agg[i], grad_out[i])
//
//   2. Aggregation backward.  Since agg[i][d] = Σ_j w[i][j] · V[j][d]:
//        grad_V[j][d] = Σ_i w[i][j] · grad_agg[i][d]
//        grad_w[i][j] = Σ_d ⟨grad_agg[i][d], V[j][d]⟩    (component dot product)
//
//   3. Softmax backward — standard Jacobian:
//        grad_score[i] = w[i] ⊙ (grad_w[i] − Σ_j w[i][j] · grad_w[i][j])
//
//   4. Score backward.  Since score[i][j] = (1/scale) Σ_d ⟨Q[i][d] ⊛ K[j][d]⟩₀:
//        The gradient of the score is a *grade-0-only* multivector C̃ at the
//        output of geo_product(Q[i][d], K[j][d]) — namely
//          grad_C̃ = (1/scale) · grad_score[i][j] · e_0
//        Then geo_product_backward(Q[i][d], K[j][d], grad_C̃) gives gradients
//        on both Q[i][d] and K[j][d].  Future positions (causal mask) get zero
//        gradient automatically because grad_score there is zero.
//
//   5. W_Q, W_K, W_V backward — standard CliffordLinear backwards seeded with
//      the grad_Q, grad_K, grad_V multivectors from step 4 and step 2.
//
// Returns gradients for all four projections plus dL/dx (input to attention)
// so they can flow into the upstream layer-norm and residual.

use crate::Multivector;
use crate::CliffordAttention;
use crate::backprop::{GradLinear, geo_product_backward, linear_backward};
use super::tape::AttentionTape;

/// Full set of gradients produced by attention_backward.
pub struct AttentionGrads {
    pub w_q:   GradLinear,
    pub w_k:   GradLinear,
    pub w_v:   GradLinear,
    pub w_o:   GradLinear,
    /// dL/d(attention input) — propagate to the layer below.  [seq][d_model]
    pub grad_input: Vec<Vec<Multivector>>,
}

/// Backward through one block's CliffordAttention using the recorded tape.
///
/// `attn`     — the attention layer (for its weights)
/// `tape`     — the AttentionTape recorded during the forward pass
/// `grad_out` — dL/d(attention output) one multivector per position. [seq][d_model]
pub fn attention_backward(
    attn:     &CliffordAttention,
    tape:     &AttentionTape,
    grad_out: &[Vec<Multivector>],
) -> AttentionGrads {
    let seq      = tape.input.len();
    let d_model  = attn.d_model;
    let n_heads  = attn.n_heads;
    let head_dim = attn.head_dim;
    let scale    = ((head_dim * 16) as f32).sqrt();

    // ── Step 1: w_o backward ──────────────────────────────────────────────────
    let mut grad_wo  = GradLinear::zeros(d_model, d_model);
    let mut grad_agg = vec![vec![Multivector::zero(); d_model]; seq];

    for i in 0..seq {
        let (g_wo, g_agg) = linear_backward(
            &attn.w_o.weights,
            &tape.agg[i],
            &grad_out[i],
        );
        grad_wo.accumulate(&g_wo);
        // g_agg has length d_model
        for d in 0..d_model {
            for k in 0..16 { grad_agg[i][d].c[k] += g_agg[d].c[k]; }
        }
    }

    // ── Step 2: aggregation backward (per head) ───────────────────────────────
    // Channel d belongs to head h = d / head_dim and uses weights[h].
    //   grad_V[j][d] = Σ_i w[h][i][j] · grad_agg[i][d]
    //   grad_w[h][i][j] = Σ_{d∈h} ⟨grad_agg[i][d], V[j][d]⟩
    let mut grad_v = vec![vec![Multivector::zero(); d_model]; seq];
    let mut grad_w = vec![vec![vec![0.0f32; seq]; seq]; n_heads];

    for h in 0..n_heads {
        let d0 = h * head_dim;
        let d1 = d0 + head_dim;
        for i in 0..seq {
            for j in 0..seq {
                let w_ij = tape.weights[h][i][j];

                // grad_w[h][i][j] = Σ_{d∈h} ⟨grad_agg[i][d], V[j][d]⟩
                let mut gw = 0.0f32;
                for d in d0..d1 {
                    let ga = &grad_agg[i][d].c;
                    let vd = &tape.v[j][d].c;
                    for k in 0..16 { gw += ga[k] * vd[k]; }
                }
                grad_w[h][i][j] = gw;

                if w_ij == 0.0 { continue; } // masked / numerically zero
                for d in d0..d1 {
                    for k in 0..16 {
                        grad_v[j][d].c[k] += w_ij * grad_agg[i][d].c[k];
                    }
                }
            }
        }
    }

    // ── Step 3: softmax backward (per head, per row) ──────────────────────────
    let mut grad_score = vec![vec![vec![0.0f32; seq]; seq]; n_heads];
    for h in 0..n_heads {
        for i in 0..seq {
            let dot: f32 = (0..seq).map(|l| tape.weights[h][i][l] * grad_w[h][i][l]).sum();
            for j in 0..seq {
                grad_score[h][i][j] = tape.weights[h][i][j] * (grad_w[h][i][j] - dot);
            }
        }
    }

    // ── Step 4: score → Q, K backward (per head) ──────────────────────────────
    // score[h][i][j] = (1/scale) Σ_{d∈h} ⟨geo(Q[i][d], K[j][d])⟩₀
    let mut grad_q = vec![vec![Multivector::zero(); d_model]; seq];
    let mut grad_k = vec![vec![Multivector::zero(); d_model]; seq];

    let inv_scale = 1.0 / scale;
    for h in 0..n_heads {
        let d0 = h * head_dim;
        let d1 = d0 + head_dim;
        for i in 0..seq {
            for j in 0..seq {
                let gs = grad_score[h][i][j];
                if gs == 0.0 { continue; }  // skips masked future positions

                // Build the "grade-0-only" gradient multivector:  C̃ = (gs/scale) · 1
                let mut grad_c = Multivector::zero();
                grad_c.c[0] = gs * inv_scale;

                for d in d0..d1 {
                    let (g_qid, g_kjd) = geo_product_backward(
                        &tape.q[i][d],
                        &tape.k[j][d],
                        &grad_c,
                    );
                    for k in 0..16 {
                        grad_q[i][d].c[k] += g_qid.c[k];
                        grad_k[j][d].c[k] += g_kjd.c[k];
                    }
                }
            }
        }
    }

    // ── Step 5: Q, K, V backward through their CliffordLinear layers ──────────
    let mut grad_wq = GradLinear::zeros(d_model, d_model);
    let mut grad_wk = GradLinear::zeros(d_model, d_model);
    let mut grad_wv = GradLinear::zeros(d_model, d_model);
    let mut grad_input = vec![vec![Multivector::zero(); d_model]; seq];

    for i in 0..seq {
        let (g_wq, g_xq) = linear_backward(&attn.w_q.weights, &tape.input[i], &grad_q[i]);
        let (g_wk, g_xk) = linear_backward(&attn.w_k.weights, &tape.input[i], &grad_k[i]);
        let (g_wv, g_xv) = linear_backward(&attn.w_v.weights, &tape.input[i], &grad_v[i]);

        grad_wq.accumulate(&g_wq);
        grad_wk.accumulate(&g_wk);
        grad_wv.accumulate(&g_wv);

        for d in 0..d_model {
            for k in 0..16 {
                grad_input[i][d].c[k] += g_xq[d].c[k] + g_xk[d].c[k] + g_xv[d].c[k];
            }
        }
    }

    AttentionGrads {
        w_q: grad_wq,
        w_k: grad_wk,
        w_v: grad_wv,
        w_o: grad_wo,
        grad_input,
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::{CliffordAlgebra, CliffordAttention};
    use crate::cayley_const::CliffordAlgebraConst;
    use crate::v2::tape::attention_forward_taped;

    /// Sum of all components of grad_out is the loss for the finite-difference check.
    fn dummy_loss(out: &[Vec<Multivector>]) -> f32 {
        let mut s = 0.0;
        for row in out { for mv in row { for k in 0..16 { s += mv.c[k]; } } }
        s
    }

    fn dummy_grad_out(seq: usize, d_model: usize) -> Vec<Vec<Multivector>> {
        // dL/dout = all-ones for the dummy_loss above
        let mut out = vec![vec![Multivector::zero(); d_model]; seq];
        for row in &mut out {
            for mv in row {
                for k in 0..16 { mv.c[k] = 1.0; }
            }
        }
        out
    }

    #[test]
    fn finite_diff_grad_input() {
        // Numerical check: perturb each component of attn input and verify the
        // analytic gradient matches the finite difference.
        let alg_arc = Arc::new(CliffordAlgebra::sta());
        let alg     = CliffordAlgebraConst::new();
        let d_model = 4;
        let seq     = 3;
        let attn    = CliffordAttention::new(d_model, 2, alg_arc.clone());

        // Random-ish input
        let mut x: Vec<Vec<Multivector>> = (0..seq).map(|i| {
            (0..d_model).map(|d| {
                let mut mv = Multivector::zero();
                for k in 0..16 { mv.c[k] = ((i * d + k) as f32 * 0.1).sin(); }
                mv
            }).collect()
        }).collect();

        // Forward and backward
        let tape = attention_forward_taped(&alg, &attn, &x, true);
        let grad_out = dummy_grad_out(seq, d_model);
        let grads = attention_backward(&attn, &tape, &grad_out);

        // Check input gradient at a few sample positions
        let eps = 1e-3;
        let checks = [(0usize, 0usize, 0usize), (1, 2, 5), (2, 1, 6)];
        for (i, d, k) in checks {
            x[i][d].c[k] += eps;
            let tape_plus = attention_forward_taped(&alg, &attn, &x, true);
            let loss_plus = dummy_loss(&tape_plus.output);

            x[i][d].c[k] -= 2.0 * eps;
            let tape_minus = attention_forward_taped(&alg, &attn, &x, true);
            let loss_minus = dummy_loss(&tape_minus.output);

            x[i][d].c[k] += eps; // restore

            let fd = (loss_plus - loss_minus) / (2.0 * eps);
            let analytic = grads.grad_input[i][d].c[k];

            // Tolerance is loose because of float accumulation across the chain
            assert!(
                (fd - analytic).abs() < 0.05 + 0.05 * analytic.abs(),
                "grad_input mismatch at ({i},{d},{k}): fd={fd:.4} analytic={analytic:.4}"
            );
        }
    }

    #[test]
    fn finite_diff_grad_wq() {
        // Check a few entries of the W_Q gradient against finite differences.
        let alg_arc = Arc::new(CliffordAlgebra::sta());
        let alg     = CliffordAlgebraConst::new();
        let d_model = 4;
        let seq     = 3;
        let mut attn = CliffordAttention::new(d_model, 2, alg_arc.clone());

        let x: Vec<Vec<Multivector>> = (0..seq).map(|i| {
            (0..d_model).map(|d| {
                let mut mv = Multivector::zero();
                for k in 0..16 { mv.c[k] = ((i * d + k) as f32 * 0.13).cos() * 0.3; }
                mv
            }).collect()
        }).collect();

        let tape = attention_forward_taped(&alg, &attn, &x, true);
        let grad_out = dummy_grad_out(seq, d_model);
        let grads = attention_backward(&attn, &tape, &grad_out);

        let eps = 1e-3;
        let checks = [(0usize, 0usize, 0usize), (1, 2, 5), (3, 1, 6)];
        for (out_d, in_d, k) in checks {
            let original = attn.w_q.weights[out_d][in_d].c[k];

            attn.w_q.weights[out_d][in_d].c[k] = original + eps;
            let t1 = attention_forward_taped(&alg, &attn, &x, true);
            let l1 = dummy_loss(&t1.output);

            attn.w_q.weights[out_d][in_d].c[k] = original - eps;
            let t2 = attention_forward_taped(&alg, &attn, &x, true);
            let l2 = dummy_loss(&t2.output);

            attn.w_q.weights[out_d][in_d].c[k] = original;

            let fd = (l1 - l2) / (2.0 * eps);
            let analytic = grads.w_q.d_weights[out_d][in_d].c[k];

            assert!(
                (fd - analytic).abs() < 0.05 + 0.05 * analytic.abs(),
                "grad_W_Q mismatch at ({out_d},{in_d},{k}): fd={fd:.4} analytic={analytic:.4}"
            );
        }
    }
}
