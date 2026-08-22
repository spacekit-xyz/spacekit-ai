// tape.rs — Forward-pass activation tape for end-to-end backward
//
// The current trainer manually re-runs sub-computations during backward because
// it has no way to retrieve the activations produced during the forward pass.
// This tape records every multivector activation, every softmax weight, and
// every layer-norm statistic so the backward pass can simply pop the relevant
// records in reverse order.
//
// Memory cost for a forward pass with seq tokens, d_model multivectors, b blocks:
//
//   embedding       seq × d_model × 16
//   per block:
//     norm1 stats    seq × (16d + 1 std + 1 mean) ≈ seq × (16d + 2)
//     attn Q/K/V    3 × seq × d_model × 16
//     attn scores    seq × seq
//     attn weights   seq × seq
//     attn agg out   seq × d_model × 16
//     attn out       seq × d_model × 16
//     residual1 out  seq × d_model × 16
//     norm2 stats    seq × (16d + 2)
//     ffn h_pre      seq × d_ff × 16   (pre-ReLU, needed for ReLU mask)
//     ffn h_post     seq × d_ff × 16
//     ffn out        seq × d_model × 16
//   head out         seq × vocab
//
// For the small demo config (seq 128, d_model 8, blocks 2, d_ff 32, vocab 4k)
// this is ~50 KB per block plus 4 MB embedding/head — entirely tractable.
use crate::Multivector;

// ─── Layer-norm statistics ───────────────────────────────────────────────────

/// Cached layer-norm statistics for one position.
/// Needed to recompute the norm gradient without redoing the mean/var pass.
#[derive(Clone, Debug)]
pub struct LayerNormStats {
    /// Normalised values (x − μ)/σ before applying γ,β.  Length 16 × d_model.
    pub x_hat: Vec<f32>,
    /// Mean over the position's 16 × d_model components.
    pub mean: f32,
    /// Standard deviation (sqrt(var + eps)).
    pub std: f32,
}

// ─── Attention activations for one block ─────────────────────────────────────

/// Everything the attention backward needs from a single block's forward pass.
///
/// All arrays are stored at the position-major / per-block granularity that
/// the attention forward produced them.  Nothing is recomputed during backward.
#[derive(Clone, Debug)]
pub struct AttentionTape {
    /// Input to attention (after norm1).         [seq][d_model]
    pub input: Vec<Vec<Multivector>>,
    /// Q, K, V projections.                       [seq][d_model]
    pub q: Vec<Vec<Multivector>>,
    pub k: Vec<Vec<Multivector>>,
    pub v: Vec<Vec<Multivector>>,
    /// Pre-softmax scores per head (after scaling + causal mask). [n_heads][seq][seq]
    pub scores: Vec<Vec<Vec<f32>>>,
    /// Post-softmax attention weights per head.   [n_heads][seq][seq]
    pub weights: Vec<Vec<Vec<f32>>>,
    /// Weighted V aggregation (input to w_o).     [seq][d_model]
    pub agg: Vec<Vec<Multivector>>,
    /// Final attention output (after w_o).        [seq][d_model]
    pub output: Vec<Vec<Multivector>>,
    /// Score kernel used in this forward pass.
    pub score_mode: crate::AttentionScoreMode,
}

// ─── FFN activations for one block ────────────────────────────────────────────

#[derive(Clone, Debug)]
pub enum FfnHidden {
    Clifford {
        hidden_pre: Vec<Vec<Multivector>>,
        hidden_post: Vec<Vec<Multivector>>,
    },
    Dense {
        input_flat: Vec<Vec<f32>>,
        pre: Vec<Vec<f32>>,
    },
}

#[derive(Clone, Debug)]
pub struct FfnTape {
    /// Input to FFN (after norm2).               [seq][d_model]
    pub input: Vec<Vec<Multivector>>,
    pub hidden: FfnHidden,
    /// Final FFN output (after fc2).             [seq][d_model]
    pub output: Vec<Vec<Multivector>>,
}

// ─── Per-block tape ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct BlockTape {
    /// Input to the block (= output of previous block, or embedding).  [seq][d_model]
    pub block_input: Vec<Vec<Multivector>>,
    /// Layer-norm 1 stats, one per position.
    pub norm1_stats: Vec<LayerNormStats>,
    pub attn: AttentionTape,
    /// After first residual add.                  [seq][d_model]
    pub after_res1: Vec<Vec<Multivector>>,
    /// Layer-norm 2 stats, one per position.
    pub norm2_stats: Vec<LayerNormStats>,
    pub ffn: FfnTape,
    /// After second residual add (= output of block).  [seq][d_model]
    pub after_res2: Vec<Vec<Multivector>>,
}

// ─── Full forward tape ────────────────────────────────────────────────────────

/// Records every activation needed to backprop the entire model.
#[derive(Clone, Debug, Default)]
pub struct Tape {
    /// Token ids fed to the model — needed to route the embedding gradient.
    pub token_ids: Vec<usize>,
    /// Output of the embedding lookup (pre-positional encoding).  [seq][d_model]
    pub embedded: Vec<Vec<Multivector>>,
    /// Output of positional encoding (= input to first block).    [seq][d_model]
    pub post_pe: Vec<Vec<Multivector>>,
    /// One BlockTape per transformer block.
    pub blocks: Vec<BlockTape>,
    /// Final-norm statistics, one per position (norm applied before the head).
    pub final_norm_stats: Vec<LayerNormStats>,
    /// Input to the output head (= final-norm output).            [seq][d_model]
    pub head_input: Vec<Vec<Multivector>>,
    /// Final logits (real head projection of the flattened head_input). [seq][vocab]
    pub logits: Vec<Vec<f32>>,
}

impl Tape {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reserve space for `n_blocks` block tapes.  Call before forward.
    pub fn reserve(&mut self, n_blocks: usize) {
        self.blocks.reserve(n_blocks);
    }

    /// Sequence length (matches across all tape components).
    pub fn seq_len(&self) -> usize {
        self.token_ids.len()
    }
}

// ─── Forward-with-tape variants ──────────────────────────────────────────────
//
// These mirror the operations in clifford_llm.rs but record everything into
// a Tape as they go.  Called by training code; never used at pure inference.

use crate::cayley_const::CliffordAlgebraConst;
use crate::ffn::{flatten_mvs, unflatten_mvs};
use crate::mask::mask_scores;
use crate::{
    CliffordAttention, CliffordBlock, CliffordFFN, CliffordLLM, CliffordLayerNorm, FfnVariant,
};

/// Layer-norm forward with stats recorded (metric-weighted over blade components).
pub fn norm_forward_taped(
    ln: &CliffordLayerNorm,
    x: &[Multivector],
) -> (Vec<Multivector>, LayerNormStats) {
    let (output, stats) =
        crate::clifford_layer_norm::forward_multivectors(&ln.gamma, &ln.beta, ln.eps, x);
    (
        output,
        LayerNormStats {
            x_hat: stats.x_hat,
            mean: stats.mean,
            std: stats.std,
        },
    )
}

/// Attention forward, recording Q/K/V/scores/weights for the backward pass.
pub fn attention_forward_taped(
    alg: &CliffordAlgebraConst,
    attn: &CliffordAttention,
    x: &[Vec<Multivector>],
    causal: bool,
    score_mode: crate::AttentionScoreMode,
) -> AttentionTape {
    let seq = x.len();
    let n_heads = attn.n_heads;
    let head_dim = attn.head_dim;
    let scale = ((head_dim * 16) as f32).sqrt();

    let q: Vec<Vec<Multivector>> = x.iter().map(|xi| attn.w_q.forward(xi)).collect();
    let k: Vec<Vec<Multivector>> = x.iter().map(|xi| attn.w_k.forward(xi)).collect();
    let v: Vec<Vec<Multivector>> = x.iter().map(|xi| attn.w_v.forward(xi)).collect();

    // Per-head scores: geometric default uses ⟨Q,K⟩ = (Q⊛K̃)₀; dot ablation uses Q·K.
    let mut scores = vec![vec![vec![0.0f32; seq]; seq]; n_heads];
    for h in 0..n_heads {
        let d0 = h * head_dim;
        let d1 = d0 + head_dim;
        for i in 0..seq {
            for j in 0..seq {
                let s: f32 = (d0..d1)
                    .map(|d| crate::attention_pair_score(alg, &q[i][d], &k[j][d], score_mode))
                    .sum();
                scores[h][i][j] = s / scale;
            }
        }
        if causal {
            mask_scores(&mut scores[h], None);
        }
    }

    // Softmax per head, per row.
    let weights: Vec<Vec<Vec<f32>>> = scores
        .iter()
        .map(|head| head.iter().map(|row| softmax(row)).collect())
        .collect();

    // Aggregate V: channel d uses its own head's attention weights.
    let agg: Vec<Vec<Multivector>> = (0..seq)
        .map(|i| {
            (0..attn.d_model)
                .map(|d| {
                    let h = d / head_dim;
                    (0..seq).fold(Multivector::zero(), |acc, j| {
                        let scaled = v[j][d].scale(weights[h][i][j]);
                        Multivector {
                            c: std::array::from_fn(|k| acc.c[k] + scaled.c[k]),
                        }
                    })
                })
                .collect()
        })
        .collect();

    // Final w_o projection
    let output: Vec<Vec<Multivector>> = agg.iter().map(|ai| attn.w_o.forward(ai)).collect();

    AttentionTape {
        input: x.to_vec(),
        q,
        k,
        v,
        scores,
        weights,
        agg,
        output,
        score_mode,
    }
}

/// FFN forward with intermediate activations recorded.
pub fn ffn_forward_taped(ffn: &FfnVariant, x: &[Vec<Multivector>]) -> FfnTape {
    match ffn {
        FfnVariant::Clifford(f) => ffn_forward_taped_clifford(f, x),
        FfnVariant::Dense(f) => {
            let input_flat: Vec<Vec<f32>> = x.iter().map(|xi| flatten_mvs(xi)).collect();
            let pre: Vec<Vec<f32>> = input_flat
                .iter()
                .map(|flat| f.fc1.forward_flat(flat))
                .collect();
            let output: Vec<Vec<Multivector>> = pre
                .iter()
                .map(|p| {
                    let post: Vec<f32> = p.iter().map(|v| v.max(0.0)).collect();
                    unflatten_mvs(&f.fc2.forward_flat(&post), f.d_model)
                })
                .collect();
            FfnTape {
                input: x.to_vec(),
                hidden: FfnHidden::Dense { input_flat, pre },
                output,
            }
        }
    }
}

fn ffn_forward_taped_clifford(ffn: &CliffordFFN, x: &[Vec<Multivector>]) -> FfnTape {
    let hidden_pre: Vec<Vec<Multivector>> = x.iter().map(|xi| ffn.fc1.forward(xi)).collect();

    let hidden_post: Vec<Vec<Multivector>> = hidden_pre
        .iter()
        .map(|row| row.iter().map(|mv| mv.map(|c| c.max(0.0))).collect())
        .collect();

    let output: Vec<Vec<Multivector>> = hidden_post.iter().map(|h| ffn.fc2.forward(h)).collect();

    FfnTape {
        input: x.to_vec(),
        hidden: FfnHidden::Clifford {
            hidden_pre,
            hidden_post,
        },
        output,
    }
}

/// One full block forward producing a BlockTape.
pub fn block_forward_taped(
    alg: &CliffordAlgebraConst,
    block: &CliffordBlock,
    input: &[Vec<Multivector>],
    causal: bool,
    score_mode: crate::AttentionScoreMode,
) -> BlockTape {
    // Norm 1
    let mut norm1_stats = Vec::with_capacity(input.len());
    let n1: Vec<Vec<Multivector>> = input
        .iter()
        .map(|xi| {
            let (out, stats) = norm_forward_taped(&block.norm1, xi);
            norm1_stats.push(stats);
            out
        })
        .collect();

    // Attention
    let attn = attention_forward_taped(alg, &block.attn, &n1, causal, score_mode);

    // Residual 1
    let after_res1: Vec<Vec<Multivector>> = input
        .iter()
        .zip(attn.output.iter())
        .map(|(xi, ai)| {
            xi.iter()
                .zip(ai.iter())
                .map(|(a, b)| Multivector {
                    c: std::array::from_fn(|k| a.c[k] + b.c[k]),
                })
                .collect()
        })
        .collect();

    // Norm 2
    let mut norm2_stats = Vec::with_capacity(after_res1.len());
    let n2: Vec<Vec<Multivector>> = after_res1
        .iter()
        .map(|xi| {
            let (out, stats) = norm_forward_taped(&block.norm2, xi);
            norm2_stats.push(stats);
            out
        })
        .collect();

    // FFN
    let ffn = ffn_forward_taped(&block.ffn, &n2);

    // Residual 2
    let after_res2: Vec<Vec<Multivector>> = after_res1
        .iter()
        .zip(ffn.output.iter())
        .map(|(xi, fi)| {
            xi.iter()
                .zip(fi.iter())
                .map(|(a, b)| Multivector {
                    c: std::array::from_fn(|k| a.c[k] + b.c[k]),
                })
                .collect()
        })
        .collect();

    BlockTape {
        block_input: input.to_vec(),
        norm1_stats,
        attn,
        after_res1,
        norm2_stats,
        ffn,
        after_res2,
    }
}

/// Full model forward producing a complete Tape ready for backward.
///
/// Includes positional encoding and the output head's multivector projection
/// before the scalar extraction that produces logits.
pub fn model_forward_taped(
    alg: &CliffordAlgebraConst,
    model: &CliffordLLM,
    token_ids: &[usize],
    causal: bool,
    dot_scores: bool,
) -> Tape {
    let score_mode = crate::AttentionScoreMode::from_dot_flag(dot_scores);
    use crate::positional::RotorPositionalEncoding;

    // 1. Embedding lookup
    let embedded: Vec<Vec<Multivector>> = token_ids
        .iter()
        .map(|&id| model.embedding[id].clone())
        .collect();

    // 2. Positional encoding
    let pe = RotorPositionalEncoding::new(model.embedding[0].len());
    let post_pe = pe.encode(alg, &embedded);

    // 3. Blocks
    let mut blocks = Vec::with_capacity(model.blocks.len());
    let mut x = post_pe.clone();
    for block in &model.blocks {
        let bt = block_forward_taped(alg, block, &x, causal, score_mode);
        x = bt.after_res2.clone();
        blocks.push(bt);
    }

    // 4. Final layer norm, then output head — flatten each position's d_model
    //    multivectors to 16·d_model real features and project to vocab logits.
    let mut final_norm_stats = Vec::with_capacity(x.len());
    let head_input: Vec<Vec<Multivector>> = x
        .iter()
        .map(|xi| {
            let (out, stats) = norm_forward_taped(&model.final_norm, xi);
            final_norm_stats.push(stats);
            out
        })
        .collect();

    let logits: Vec<Vec<f32>> = head_input.iter().map(|xi| model.head.forward(xi)).collect();

    Tape {
        token_ids: token_ids.to_vec(),
        embedded,
        post_pe,
        blocks,
        final_norm_stats,
        head_input,
        logits,
    }
}

/// Inference/training-aligned logits: embedding → rotor PE → blocks (optional causal) → head.
/// Prefer this over [`CliffordLLM::forward`] for v2 LM checkpoints (matches [`model_forward_taped`]).
pub fn model_forward_logits(
    alg: &CliffordAlgebraConst,
    model: &CliffordLLM,
    token_ids: &[usize],
    causal: bool,
    dot_scores: bool,
) -> Vec<Vec<f32>> {
    model_forward_taped(alg, model, token_ids, causal, dot_scores).logits
}

// ─── Helper ───────────────────────────────────────────────────────────────────

fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 || !sum.is_finite() {
        // All-masked row (e.g. position 0 with future-only attention): uniform fallback
        return vec![1.0 / x.len() as f32; x.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CliffordAlgebra, LinearReal};
    use std::sync::Arc;

    #[test]
    fn tape_shapes_match_forward() {
        let alg_arc = Arc::new(CliffordAlgebra::sta());
        let alg = CliffordAlgebraConst::new();
        let d_model = 4;
        let vocab = 16;

        let attn = CliffordAttention::new(d_model, 2, alg_arc.clone());
        let ffn = FfnVariant::clifford(d_model, 8, alg_arc.clone());
        let norm1 = CliffordLayerNorm::new(d_model);
        let norm2 = CliffordLayerNorm::new(d_model);
        let block = CliffordBlock {
            attn,
            ffn,
            norm1,
            norm2,
        };

        let head = LinearReal::new(d_model, vocab);
        let model = CliffordLLM {
            embedding: vec![vec![Multivector::scalar(0.1); d_model]; vocab],
            blocks: vec![block],
            final_norm: CliffordLayerNorm::new(d_model),
            head,
            algebra: alg_arc,
        };

        let ids = vec![1usize, 2, 3, 4];
        let tape = model_forward_taped(&alg, &model, &ids, true, false);

        assert_eq!(tape.seq_len(), 4);
        assert_eq!(tape.embedded.len(), 4);
        assert_eq!(tape.embedded[0].len(), d_model);
        assert_eq!(tape.blocks.len(), 1);
        // scores are now [n_heads][seq][seq]
        assert_eq!(tape.blocks[0].attn.scores.len(), 2); // n_heads
        assert_eq!(tape.blocks[0].attn.scores[0].len(), 4); // seq
        assert_eq!(tape.blocks[0].attn.scores[0][0].len(), 4);
        let row_sum: f32 = tape.blocks[0].attn.weights[0][0].iter().sum();
        assert!((row_sum - 1.0).abs() < 1e-5); // softmax row-stochastic
        assert_eq!(tape.logits.len(), 4);
        assert_eq!(tape.logits[0].len(), vocab);
    }

    #[test]
    fn causal_mask_zeros_future_weights() {
        let alg_arc = Arc::new(CliffordAlgebra::sta());
        let alg = CliffordAlgebraConst::new();
        let attn = CliffordAttention::new(4, 2, alg_arc.clone());

        let x: Vec<Vec<Multivector>> = (0..3)
            .map(|i| vec![Multivector::scalar((i + 1) as f32); 4])
            .collect();
        let tape = attention_forward_taped(
            &alg,
            &attn,
            &x,
            true,
            crate::AttentionScoreMode::InnerProduct,
        );

        // Check every head: position 0 attends only to position 0; position 2 sees all.
        for head in &tape.weights {
            assert!(head[0][1].abs() < 1e-6);
            assert!(head[0][2].abs() < 1e-6);
            assert!(head[2][0] > 0.0);
            assert!(head[2][1] > 0.0);
            assert!(head[2][2] > 0.0);
        }
    }
}
