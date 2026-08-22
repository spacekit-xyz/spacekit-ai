// block_backward.rs — End-to-end backward through one CliffordBlock
//
// Uses BlockTape from tape.rs and AttentionGrads from attention_backward.rs to
// compute gradients for *every* parameter in the block:
//
//   norm1.gamma, norm1.beta
//   attn.w_q, attn.w_k, attn.w_v, attn.w_o
//   norm2.gamma, norm2.beta
//   ffn (Clifford or dense param-matched)
//
// Returns those gradients plus dL/d(block input) so the gradient can keep
// flowing to the layer below (next block, positional encoding, or embedding).

use super::attention_backward::{attention_backward, AttentionGrads};
use super::tape::{BlockTape, FfnHidden};
use crate::backprop::{
    layer_norm_backward, linear_backward, real_linear_backward, GradLinear, RealHeadGrad,
};
use crate::ffn::{flatten_mvs, unflatten_mvs, FfnVariant};
use crate::{CliffordBlock, Multivector};

/// FFN parameter gradients (Clifford geo-linear or dense real-linear).
pub enum FfnGrad {
    Clifford(GradLinear, GradLinear),
    Dense(RealHeadGrad, RealHeadGrad),
}

impl FfnGrad {
    pub fn scale(&mut self, s: f32) {
        match self {
            Self::Clifford(g1, g2) => {
                g1.scale(s);
                g2.scale(s);
            }
            Self::Dense(g1, g2) => {
                g1.scale(s);
                g2.scale(s);
            }
        }
    }

    pub fn clip_norm(&mut self, max_norm: f32) {
        match self {
            Self::Clifford(g1, g2) => {
                crate::optim::clip_grad_norm(g1, max_norm);
                crate::optim::clip_grad_norm(g2, max_norm);
            }
            Self::Dense(g1, g2) => {
                g1.clip_norm(max_norm);
                g2.clip_norm(max_norm);
            }
        }
    }
}

/// Full gradient bundle for a single transformer block.
pub struct BlockGrads {
    pub norm1_gamma: Vec<f32>,
    pub norm1_beta: Vec<f32>,
    pub attn: AttentionGrads,
    pub norm2_gamma: Vec<f32>,
    pub norm2_beta: Vec<f32>,
    pub ffn: FfnGrad,
    /// dL/d(block input) — flows to the layer below.   [seq][d_model]
    pub grad_input: Vec<Vec<Multivector>>,
}

/// Backward through one transformer block.
pub fn block_backward(
    block: &CliffordBlock,
    tape: &BlockTape,
    grad_out: &[Vec<Multivector>],
) -> BlockGrads {
    let seq = tape.block_input.len();
    let d_model = tape.block_input[0].len();

    // ── 1. Residual 2 split ───────────────────────────────────────────────────
    let grad_ffn_out: Vec<Vec<Multivector>> = grad_out.to_vec();
    let mut grad_after_res1: Vec<Vec<Multivector>> = grad_out.to_vec();

    // ── 2. FFN backward ───────────────────────────────────────────────────────
    let (ffn_grad, grad_ffn_in) = ffn_backward(&block.ffn, &tape.ffn, &grad_ffn_out);

    // ── 3. Norm 2 backward ────────────────────────────────────────────────────
    let (grad_n2_gamma, grad_n2_beta, grad_after_res1_from_norm) = layer_norm_backward_with_params(
        &tape.norm2_stats,
        &block.norm2.gamma,
        &grad_ffn_in,
        d_model,
    );

    for i in 0..seq {
        for d in 0..d_model {
            for k in 0..16 {
                grad_after_res1[i][d].c[k] += grad_after_res1_from_norm[i][d].c[k];
            }
        }
    }

    // ── 4. Residual 1 split ───────────────────────────────────────────────────
    let grad_attn_out: Vec<Vec<Multivector>> = grad_after_res1.clone();
    let mut grad_block_input: Vec<Vec<Multivector>> = grad_after_res1.clone();

    // ── 5. Attention backward ─────────────────────────────────────────────────
    let attn_grads = attention_backward(&block.attn, &tape.attn, &grad_attn_out);

    // ── 6. Norm 1 backward ────────────────────────────────────────────────────
    let (grad_n1_gamma, grad_n1_beta, grad_block_input_from_norm) = layer_norm_backward_with_params(
        &tape.norm1_stats,
        &block.norm1.gamma,
        &attn_grads.grad_input,
        d_model,
    );

    for i in 0..seq {
        for d in 0..d_model {
            for k in 0..16 {
                grad_block_input[i][d].c[k] += grad_block_input_from_norm[i][d].c[k];
            }
        }
    }

    BlockGrads {
        norm1_gamma: grad_n1_gamma,
        norm1_beta: grad_n1_beta,
        attn: attn_grads,
        norm2_gamma: grad_n2_gamma,
        norm2_beta: grad_n2_beta,
        ffn: ffn_grad,
        grad_input: grad_block_input,
    }
}

fn ffn_backward(
    ffn: &FfnVariant,
    tape: &super::tape::FfnTape,
    grad_out: &[Vec<Multivector>],
) -> (FfnGrad, Vec<Vec<Multivector>>) {
    let seq = grad_out.len();
    match (ffn, &tape.hidden) {
        (
            FfnVariant::Clifford(f),
            FfnHidden::Clifford {
                hidden_pre,
                hidden_post,
            },
        ) => {
            let d_ff = f.fc2.in_dim;
            let d_model = f.fc1.in_dim;
            let mut grad_fc2 = GradLinear::zeros(d_model, d_ff);
            let mut grad_h_post = vec![vec![Multivector::zero(); d_ff]; seq];

            for i in 0..seq {
                let (g_fc2, g_h) = linear_backward(&f.fc2.weights, &hidden_post[i], &grad_out[i]);
                grad_fc2.accumulate(&g_fc2);
                for d in 0..d_ff {
                    for k in 0..16 {
                        grad_h_post[i][d].c[k] += g_h[d].c[k];
                    }
                }
            }

            let mut grad_h_pre = vec![vec![Multivector::zero(); d_ff]; seq];
            for i in 0..seq {
                for d in 0..d_ff {
                    for k in 0..16 {
                        grad_h_pre[i][d].c[k] = if hidden_pre[i][d].c[k] > 0.0 {
                            grad_h_post[i][d].c[k]
                        } else {
                            0.0
                        };
                    }
                }
            }

            let mut grad_fc1 = GradLinear::zeros(d_ff, d_model);
            let mut grad_ffn_in = vec![vec![Multivector::zero(); d_model]; seq];
            for i in 0..seq {
                let (g_fc1, g_x) = linear_backward(&f.fc1.weights, &tape.input[i], &grad_h_pre[i]);
                grad_fc1.accumulate(&g_fc1);
                for d in 0..d_model {
                    for k in 0..16 {
                        grad_ffn_in[i][d].c[k] += g_x[d].c[k];
                    }
                }
            }
            (FfnGrad::Clifford(grad_fc1, grad_fc2), grad_ffn_in)
        }
        (FfnVariant::Dense(f), FfnHidden::Dense { input_flat, pre }) => {
            let mut grad_fc2 = RealHeadGrad::zeros(f.fc2.out_dim, f.fc2.in_features);
            let mut grad_fc1 = RealHeadGrad::zeros(f.fc1.out_dim, f.fc1.in_features);
            let mut grad_ffn_in = Vec::with_capacity(seq);

            for i in 0..seq {
                let g_out_flat = flatten_mvs(&grad_out[i]);
                let post: Vec<f32> = pre[i].iter().map(|v| v.max(0.0)).collect();
                let g_h = real_linear_backward(&f.fc2.weights, &post, &g_out_flat, &mut grad_fc2);
                let g_pre: Vec<f32> = g_h
                    .iter()
                    .zip(pre[i].iter())
                    .map(|(g, p)| if *p > 0.0 { *g } else { 0.0 })
                    .collect();
                let g_in_flat =
                    real_linear_backward(&f.fc1.weights, &input_flat[i], &g_pre, &mut grad_fc1);
                grad_ffn_in.push(unflatten_mvs(&g_in_flat, f.d_model));
            }
            (FfnGrad::Dense(grad_fc1, grad_fc2), grad_ffn_in)
        }
        _ => panic!("FFN variant / tape hidden mismatch — forward/backward desync"),
    }
}

fn layer_norm_backward_with_params(
    stats: &[super::tape::LayerNormStats],
    gamma: &[f32],
    grad_out: &[Vec<Multivector>],
    d_model: usize,
) -> (Vec<f32>, Vec<f32>, Vec<Vec<Multivector>>) {
    let seq = stats.len();
    let n_comp = d_model * 16;

    let mut grad_gamma = vec![0.0f32; n_comp];
    let mut grad_beta = vec![0.0f32; n_comp];
    let mut grad_x = vec![vec![Multivector::zero(); d_model]; seq];

    for i in 0..seq {
        let g_flat: Vec<f32> = grad_out[i].iter().flat_map(|mv| mv.c).collect();

        for k in 0..n_comp {
            grad_gamma[k] += g_flat[k] * stats[i].x_hat[k];
            grad_beta[k] += g_flat[k];
        }

        let g_x_flat = layer_norm_backward(&stats[i].x_hat, gamma, &g_flat, stats[i].std);

        for d in 0..d_model {
            for k in 0..16 {
                grad_x[i][d].c[k] = g_x_flat[d * 16 + k];
            }
        }
    }

    (grad_gamma, grad_beta, grad_x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cayley_const::CliffordAlgebraConst;
    use crate::ffn::FfnVariant;
    use crate::v2::tape::block_forward_taped;
    use crate::{CliffordAlgebra, CliffordAttention, CliffordLayerNorm};
    use std::sync::Arc;

    fn dummy_loss(out: &[Vec<Multivector>]) -> f32 {
        out.iter().flatten().flat_map(|mv| mv.c).sum()
    }

    fn dummy_grad_out(seq: usize, d_model: usize) -> Vec<Vec<Multivector>> {
        (0..seq)
            .map(|_| {
                (0..d_model)
                    .map(|_| {
                        let mut mv = Multivector::zero();
                        for k in 0..16 {
                            mv.c[k] = 1.0;
                        }
                        mv
                    })
                    .collect()
            })
            .collect()
    }

    fn build_block(d_model: usize, d_ff: usize, alg_arc: Arc<CliffordAlgebra>) -> CliffordBlock {
        CliffordBlock {
            attn: CliffordAttention::new(d_model, 2, alg_arc.clone()),
            ffn: FfnVariant::clifford(d_model, d_ff, alg_arc.clone()),
            norm1: CliffordLayerNorm::new(d_model),
            norm2: CliffordLayerNorm::new(d_model),
        }
    }

    #[test]
    fn end_to_end_finite_diff_grad_input() {
        let alg_arc = Arc::new(CliffordAlgebra::sta());
        let alg = CliffordAlgebraConst::new();
        let d_model = 4;
        let d_ff = 8;
        let seq = 3;
        let block = build_block(d_model, d_ff, alg_arc.clone());

        let mut x: Vec<Vec<Multivector>> = (0..seq)
            .map(|i| {
                (0..d_model)
                    .map(|d| {
                        let mut mv = Multivector::zero();
                        for k in 0..16 {
                            mv.c[k] = ((i * d + k) as f32 * 0.07).sin() * 0.5;
                        }
                        mv
                    })
                    .collect()
            })
            .collect();

        let tape = block_forward_taped(
            &alg,
            &block,
            &x,
            true,
            crate::AttentionScoreMode::InnerProduct,
        );
        let grad_out = dummy_grad_out(seq, d_model);
        let grads = block_backward(&block, &tape, &grad_out);

        let eps = 1e-3;
        let checks = [(0usize, 0usize, 0usize), (1, 1, 3), (2, 2, 7)];
        for (i, d, k) in checks {
            x[i][d].c[k] += eps;
            let t1 = block_forward_taped(
                &alg,
                &block,
                &x,
                true,
                crate::AttentionScoreMode::InnerProduct,
            );
            let l1 = dummy_loss(&t1.after_res2);

            x[i][d].c[k] -= 2.0 * eps;
            let t2 = block_forward_taped(
                &alg,
                &block,
                &x,
                true,
                crate::AttentionScoreMode::InnerProduct,
            );
            let l2 = dummy_loss(&t2.after_res2);

            x[i][d].c[k] += eps;

            let fd = (l1 - l2) / (2.0 * eps);
            let analytic = grads.grad_input[i][d].c[k];

            assert!(
                (fd - analytic).abs() < 0.1 + 0.1 * analytic.abs(),
                "block grad_input mismatch at ({i},{d},{k}): fd={fd:.4} analytic={analytic:.4}"
            );
        }
    }
}
