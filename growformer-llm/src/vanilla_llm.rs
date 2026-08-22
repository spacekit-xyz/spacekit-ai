//! Param-matched vanilla transformer (row 2): real embeddings, standard LN, dot attention, ReLU FFN.

use crate::real_linear::LinearReal;
use crate::standard_layer_norm::{self, StandardLayerNorm};

fn relu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

fn softmax_row(scores: &[f32]) -> Vec<f32> {
    let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|&s| (s - m).exp()).collect();
    let z: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / z).collect()
}

/// Fixed sinusoidal positional encoding (no learnable params).
pub fn add_sinusoidal_pe(x: &mut [Vec<f32>]) {
    let seq = x.len();
    if seq == 0 {
        return;
    }
    let d = x[0].len();
    for pos in 0..seq {
        for i in 0..d {
            let angle = pos as f32 / 10_000_f32.powf(2.0 * (i / 2) as f32 / d as f32);
            let v = if i % 2 == 0 { angle.sin() } else { angle.cos() };
            x[pos][i] += v;
        }
    }
}

pub struct VanillaAttention {
    pub w_q: LinearReal,
    pub w_k: LinearReal,
    pub w_v: LinearReal,
    pub w_o: LinearReal,
    pub d_model: usize,
    pub n_heads: usize,
    pub head_dim: usize,
}

impl VanillaAttention {
    pub fn new(d_model: usize, n_heads: usize, seed: u64) -> Self {
        assert_eq!(d_model % n_heads, 0);
        let head_dim = d_model / n_heads;
        Self {
            w_q: LinearReal::new_dims(d_model, d_model, seed),
            w_k: LinearReal::new_dims(d_model, d_model, seed ^ 0xA1),
            w_v: LinearReal::new_dims(d_model, d_model, seed ^ 0xA2),
            w_o: LinearReal::new_dims(d_model, d_model, seed ^ 0xA3),
            d_model,
            n_heads,
            head_dim,
        }
    }

    pub fn forward(&self, x: &[Vec<f32>], causal: bool) -> Vec<Vec<f32>> {
        let seq = x.len();
        let d = self.d_model;
        let scale = (self.head_dim as f32).sqrt();
        let q: Vec<Vec<f32>> = x.iter().map(|xi| self.w_q.forward_flat(xi)).collect();
        let k: Vec<Vec<f32>> = x.iter().map(|xi| self.w_k.forward_flat(xi)).collect();
        let v: Vec<Vec<f32>> = x.iter().map(|xi| self.w_v.forward_flat(xi)).collect();

        let mut out = vec![vec![0.0f32; d]; seq];
        for h in 0..self.n_heads {
            let d0 = h * self.head_dim;
            let d1 = d0 + self.head_dim;
            for i in 0..seq {
                let mut scores = vec![0.0f32; seq];
                for j in 0..seq {
                    if causal && j > i {
                        scores[j] = f32::NEG_INFINITY;
                        continue;
                    }
                    let mut s = 0.0f32;
                    for t in 0..self.head_dim {
                        s += q[i][d0 + t] * k[j][d0 + t];
                    }
                    scores[j] = s / scale;
                }
                let w = softmax_row(&scores);
                for j in 0..seq {
                    for t in 0..self.head_dim {
                        out[i][d0 + t] += w[j] * v[j][d0 + t];
                    }
                }
            }
        }
        out.iter().map(|o| self.w_o.forward_flat(o)).collect()
    }
}

pub struct VanillaFFN {
    pub fc1: LinearReal,
    pub fc2: LinearReal,
}

impl VanillaFFN {
    pub fn new(d_model: usize, d_ff: usize, seed: u64) -> Self {
        Self {
            fc1: LinearReal::new_dims(d_model, d_ff, seed),
            fc2: LinearReal::new_dims(d_ff, d_model, seed ^ 0xFF01),
        }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let h = self.fc1.forward_flat(x);
        let h: Vec<f32> = h.iter().map(|&v| relu(v)).collect();
        self.fc2.forward_flat(&h)
    }
}

pub struct VanillaBlock {
    pub norm1: StandardLayerNorm,
    pub attn: VanillaAttention,
    pub norm2: StandardLayerNorm,
    pub ffn: VanillaFFN,
}

impl VanillaBlock {
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize, seed: u64) -> Self {
        Self {
            norm1: StandardLayerNorm::new(d_model),
            attn: VanillaAttention::new(d_model, n_heads, seed),
            norm2: StandardLayerNorm::new(d_model),
            ffn: VanillaFFN::new(d_model, d_ff, seed ^ 0xB10C),
        }
    }
}

pub struct VanillaLLM {
    pub embedding: Vec<Vec<f32>>,
    pub blocks: Vec<VanillaBlock>,
    pub final_norm: StandardLayerNorm,
    pub head: LinearReal,
    pub d_model: usize,
}

impl VanillaLLM {
    pub fn new(
        vocab: usize,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        n_blocks: usize,
        seed: u64,
    ) -> Self {
        let blocks: Vec<_> = (0..n_blocks)
            .map(|b| VanillaBlock::new(d_model, n_heads, d_ff, seed ^ (b as u64 + 1) * 0x9E37))
            .collect();
        Self {
            embedding: vec![vec![0.0; d_model]; vocab],
            blocks,
            final_norm: StandardLayerNorm::new(d_model),
            head: LinearReal::new_dims(d_model, vocab, seed ^ 0xEAD0),
            d_model,
        }
    }

    pub fn sync_tied_head(&mut self) {
        let vocab = self.embedding.len();
        for v in 0..vocab {
            self.head.weights[v].copy_from_slice(&self.embedding[v]);
        }
    }
}

/// Inference forward — logits per position.
pub fn vanilla_forward_logits(model: &VanillaLLM, ids: &[usize], causal: bool) -> Vec<Vec<f32>> {
    let mut x: Vec<Vec<f32>> = ids.iter().map(|&id| model.embedding[id].clone()).collect();
    add_sinusoidal_pe(&mut x);
    let d = model.d_model;

    for block in &model.blocks {
        let mut attn_in = Vec::with_capacity(x.len());
        for row in &x {
            let (y, _) =
                standard_layer_norm::forward(row, &block.norm1.gamma, &block.norm1.beta, 1e-5);
            attn_in.push(y);
        }
        let attn_out = block.attn.forward(&attn_in, causal);
        for t in 0..x.len() {
            for i in 0..d {
                x[t][i] += attn_out[t][i];
            }
        }
        for t in 0..x.len() {
            let (y, _) =
                standard_layer_norm::forward(&x[t], &block.norm2.gamma, &block.norm2.beta, 1e-5);
            let delta = block.ffn.forward(&y);
            for i in 0..d {
                x[t][i] += delta[i];
            }
        }
    }

    x.iter()
        .map(|row| {
            let (y, _) = standard_layer_norm::forward(
                row,
                &model.final_norm.gamma,
                &model.final_norm.beta,
                1e-5,
            );
            model.head.forward_flat(&y)
        })
        .collect()
}
