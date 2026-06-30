// block_backward.rs — End-to-end backward through one CliffordBlock
//
// Uses BlockTape from tape.rs and AttentionGrads from attention_backward.rs to
// compute gradients for *every* parameter in the block:
//
//   norm1.gamma, norm1.beta
//   attn.w_q, attn.w_k, attn.w_v, attn.w_o
//   norm2.gamma, norm2.beta
//   ffn.fc1, ffn.fc2
//
// Returns those gradients plus dL/d(block input) so the gradient can keep
// flowing to the layer below (next block, positional encoding, or embedding).
//
// Forward graph (pre-norm + residuals):
//
//   block_input ──┬──────────────────────────────┐
//                 │                              │
//              norm1 ──> attn ──> after_res1 ──┐ │
//                                              │ │
//                                              ▼ ▼
//                                          (residual add)
//                                              │
//                            ┌─────────────────┤
//                            │                 │
//                         norm2 ──> ffn ──> (residual add) ──> after_res2
//
// Backward walks this in reverse, splitting gradients at each residual.

use crate::{Multivector, CliffordBlock};
use crate::backprop::{GradLinear, linear_backward, layer_norm_backward};
use super::attention_backward::{attention_backward, AttentionGrads};
use super::tape::BlockTape;

/// Full gradient bundle for a single transformer block.
pub struct BlockGrads {
    pub norm1_gamma: Vec<f32>,
    pub norm1_beta:  Vec<f32>,
    pub attn:        AttentionGrads,
    pub norm2_gamma: Vec<f32>,
    pub norm2_beta:  Vec<f32>,
    pub ffn_fc1:     GradLinear,
    pub ffn_fc2:     GradLinear,
    /// dL/d(block input) — flows to the layer below.   [seq][d_model]
    pub grad_input:  Vec<Vec<Multivector>>,
}

/// Backward through one transformer block.
///
/// `block`    — the block itself (parameters)
/// `tape`     — recorded forward pass for this block
/// `grad_out` — dL/d(block output, i.e. tape.after_res2).  [seq][d_model]
pub fn block_backward(
    block:    &CliffordBlock,
    tape:     &BlockTape,
    grad_out: &[Vec<Multivector>],
) -> BlockGrads {
    let seq     = tape.block_input.len();
    let d_model = tape.block_input[0].len();
    let _n_comp = d_model * 16;

    // ── 1. Residual 2 split ───────────────────────────────────────────────────
    // after_res2 = after_res1 + ffn_out
    // grad_after_res1 ← grad_out;   grad_ffn_out ← grad_out
    let grad_ffn_out: Vec<Vec<Multivector>> = grad_out.to_vec();
    let mut grad_after_res1: Vec<Vec<Multivector>> = grad_out.to_vec();

    // ── 2. FFN backward ───────────────────────────────────────────────────────
    // Through fc2:  grad_h_post[i], grad_fc2 ← linear_backward(W_fc2, h_post[i], grad_ffn_out[i])
    let mut grad_fc2  = GradLinear::zeros(d_model, block.ffn.fc2.in_dim);
    let mut grad_h_post = vec![vec![Multivector::zero(); block.ffn.fc2.in_dim]; seq];

    for i in 0..seq {
        let (g_fc2, g_h) = linear_backward(
            &block.ffn.fc2.weights,
            &tape.ffn.hidden_post[i],
            &grad_ffn_out[i],
        );
        grad_fc2.accumulate(&g_fc2);
        for d in 0..block.ffn.fc2.in_dim {
            for k in 0..16 { grad_h_post[i][d].c[k] += g_h[d].c[k]; }
        }
    }

    // ReLU backward:  grad_h_pre[i][d][k] = grad_h_post[i][d][k] if h_pre > 0 else 0
    let mut grad_h_pre = vec![vec![Multivector::zero(); block.ffn.fc2.in_dim]; seq];
    for i in 0..seq {
        for d in 0..block.ffn.fc2.in_dim {
            for k in 0..16 {
                grad_h_pre[i][d].c[k] = if tape.ffn.hidden_pre[i][d].c[k] > 0.0 {
                    grad_h_post[i][d].c[k]
                } else { 0.0 };
            }
        }
    }

    // Through fc1
    let mut grad_fc1   = GradLinear::zeros(block.ffn.fc2.in_dim, d_model);
    let mut grad_ffn_in = vec![vec![Multivector::zero(); d_model]; seq];

    for i in 0..seq {
        let (g_fc1, g_x) = linear_backward(
            &block.ffn.fc1.weights,
            &tape.ffn.input[i],
            &grad_h_pre[i],
        );
        grad_fc1.accumulate(&g_fc1);
        for d in 0..d_model {
            for k in 0..16 { grad_ffn_in[i][d].c[k] += g_x[d].c[k]; }
        }
    }

    // ── 3. Norm 2 backward ────────────────────────────────────────────────────
    // grad_ffn_in is dL/d(norm2 output).  Convert through layer_norm_backward
    // to get dL/d(after_res1) which we then accumulate with the residual path.
    let (grad_n2_gamma, grad_n2_beta, grad_after_res1_from_norm) =
        layer_norm_backward_with_params(
            &tape.norm2_stats,
            &block.norm2.gamma,
            &grad_ffn_in,
            d_model,
        );

    // Sum residual path and norm-path gradients into grad_after_res1
    for i in 0..seq {
        for d in 0..d_model {
            for k in 0..16 {
                grad_after_res1[i][d].c[k] += grad_after_res1_from_norm[i][d].c[k];
            }
        }
    }

    // ── 4. Residual 1 split ───────────────────────────────────────────────────
    // after_res1 = block_input + attn_out
    // grad_attn_out ← grad_after_res1;   grad_block_input(partial) ← grad_after_res1
    let grad_attn_out: Vec<Vec<Multivector>> = grad_after_res1.clone();
    let mut grad_block_input: Vec<Vec<Multivector>> = grad_after_res1.clone();

    // ── 5. Attention backward (full Q/K/V/O) ──────────────────────────────────
    let attn_grads = attention_backward(&block.attn, &tape.attn, &grad_attn_out);

    // The attention's grad_input is dL/d(norm1 output).
    // ── 6. Norm 1 backward ────────────────────────────────────────────────────
    let (grad_n1_gamma, grad_n1_beta, grad_block_input_from_norm) =
        layer_norm_backward_with_params(
            &tape.norm1_stats,
            &block.norm1.gamma,
            &attn_grads.grad_input,
            d_model,
        );

    // Add the norm path into the residual path gradient
    for i in 0..seq {
        for d in 0..d_model {
            for k in 0..16 {
                grad_block_input[i][d].c[k] += grad_block_input_from_norm[i][d].c[k];
            }
        }
    }

    BlockGrads {
        norm1_gamma: grad_n1_gamma,
        norm1_beta:  grad_n1_beta,
        attn:        attn_grads,
        norm2_gamma: grad_n2_gamma,
        norm2_beta:  grad_n2_beta,
        ffn_fc1:     grad_fc1,
        ffn_fc2:     grad_fc2,
        grad_input:  grad_block_input,
    }
}

// ─── Layer-norm backward with gamma/beta gradients ───────────────────────────
//
// For each position, applies the layer_norm_backward formula and additionally
// computes dL/dγ = Σ grad_out * x_hat   and   dL/dβ = Σ grad_out.
// Returns (grad_gamma, grad_beta, grad_x_as_multivectors).

fn layer_norm_backward_with_params(
    stats:    &[super::tape::LayerNormStats],
    gamma:    &[f32],
    grad_out: &[Vec<Multivector>],
    d_model:  usize,
) -> (Vec<f32>, Vec<f32>, Vec<Vec<Multivector>>) {
    let seq    = stats.len();
    let n_comp = d_model * 16;

    let mut grad_gamma = vec![0.0f32; n_comp];
    let mut grad_beta  = vec![0.0f32; n_comp];
    let mut grad_x = vec![vec![Multivector::zero(); d_model]; seq];

    for i in 0..seq {
        // Flatten grad_out[i] to length n_comp
        let g_flat: Vec<f32> = grad_out[i].iter().flat_map(|mv| mv.c).collect();

        // dL/dγ[k] += g_flat[k] * x_hat[k]
        // dL/dβ[k] += g_flat[k]
        for k in 0..n_comp {
            grad_gamma[k] += g_flat[k] * stats[i].x_hat[k];
            grad_beta[k]  += g_flat[k];
        }

        // dL/dx using the existing backprop function
        let g_x_flat = layer_norm_backward(
            &stats[i].x_hat,
            gamma,
            &g_flat,
            stats[i].std,
        );

        // Reshape back into [d_model] multivectors
        for d in 0..d_model {
            for k in 0..16 {
                grad_x[i][d].c[k] = g_x_flat[d * 16 + k];
            }
        }
    }

    (grad_gamma, grad_beta, grad_x)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::{
        CliffordAlgebra, CliffordAttention, CliffordFFN, CliffordLayerNorm,
    };
    use crate::cayley_const::CliffordAlgebraConst;
    use crate::v2::tape::block_forward_taped;

    fn dummy_loss(out: &[Vec<Multivector>]) -> f32 {
        out.iter().flatten().flat_map(|mv| mv.c).sum()
    }

    fn dummy_grad_out(seq: usize, d_model: usize) -> Vec<Vec<Multivector>> {
        (0..seq).map(|_| (0..d_model).map(|_| {
            let mut mv = Multivector::zero();
            for k in 0..16 { mv.c[k] = 1.0; }
            mv
        }).collect()).collect()
    }

    fn build_block(d_model: usize, d_ff: usize, alg_arc: Arc<CliffordAlgebra>) -> CliffordBlock {
        CliffordBlock {
            attn:  CliffordAttention::new(d_model, 2, alg_arc.clone()),
            ffn:   CliffordFFN::new(d_model, d_ff, alg_arc.clone()),
            norm1: CliffordLayerNorm::new(d_model),
            norm2: CliffordLayerNorm::new(d_model),
        }
    }

    #[test]
    fn end_to_end_finite_diff_grad_input() {
        let alg_arc = Arc::new(CliffordAlgebra::sta());
        let alg     = CliffordAlgebraConst::new();
        let d_model = 4;
        let d_ff    = 8;
        let seq     = 3;
        let block   = build_block(d_model, d_ff, alg_arc.clone());

        let mut x: Vec<Vec<Multivector>> = (0..seq).map(|i| {
            (0..d_model).map(|d| {
                let mut mv = Multivector::zero();
                for k in 0..16 { mv.c[k] = ((i * d + k) as f32 * 0.07).sin() * 0.5; }
                mv
            }).collect()
        }).collect();

        let tape = block_forward_taped(&alg, &block, &x, true);
        let grad_out = dummy_grad_out(seq, d_model);
        let grads = block_backward(&block, &tape, &grad_out);

        let eps = 1e-3;
        let checks = [(0usize, 0usize, 0usize), (1, 1, 3), (2, 2, 7)];
        for (i, d, k) in checks {
            x[i][d].c[k] += eps;
            let t1 = block_forward_taped(&alg, &block, &x, true);
            let l1 = dummy_loss(&t1.after_res2);

            x[i][d].c[k] -= 2.0 * eps;
            let t2 = block_forward_taped(&alg, &block, &x, true);
            let l2 = dummy_loss(&t2.after_res2);

            x[i][d].c[k] += eps;

            let fd = (l1 - l2) / (2.0 * eps);
            let analytic = grads.grad_input[i][d].c[k];

            // Generous tolerance — there's a long chain of multivector ops
            assert!(
                (fd - analytic).abs() < 0.1 + 0.1 * analytic.abs(),
                "block grad_input mismatch at ({i},{d},{k}): fd={fd:.4} analytic={analytic:.4}"
            );
        }
    }
}
