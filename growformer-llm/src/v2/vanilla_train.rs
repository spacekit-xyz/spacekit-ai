//! Full training loop for row-2 param-matched vanilla transformer.

use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::real_linear::LinearReal;
use crate::real_ops::{
    cosine_lr_with_warmup, cross_entropy, real_linear_backward, AdamConfig, RealHeadGrad,
    RealHeadOptimizer,
};
use crate::standard_layer_norm::{self, StandardNormStats};
use crate::vanilla_llm::{
    vanilla_forward_logits, VanillaAttention, VanillaBlock, VanillaFFN, VanillaLLM,
};
use serde::{Deserialize, Serialize};

use super::data::TrainExample;
use crate::lm_config::TrainConfigV2;

const LN_EPS: f32 = 1e-5;

// ─── Initialisation ──────────────────────────────────────────────────────────

fn fill_real_linear_random(layer: &mut LinearReal, rng: &mut StdRng, fan_in: usize) {
    let std = (1.0 / fan_in as f32).sqrt();
    let bound = std * 3.0f32.sqrt();
    for row in &mut layer.weights {
        for w in row {
            *w = rng.gen_range(-bound..bound);
        }
    }
    for b in &mut layer.bias {
        *b = 0.0;
    }
}

pub fn randomize_vanilla_model(model: &mut VanillaLLM, seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let dm = model.d_model;
    let emb_bound = 0.02f32 * 3.0f32.sqrt();
    for row in &mut model.embedding {
        for w in row {
            *w = rng.gen_range(-emb_bound..emb_bound);
        }
    }
    for block in &mut model.blocks {
        fill_real_linear_random(&mut block.attn.w_q, &mut rng, dm);
        fill_real_linear_random(&mut block.attn.w_k, &mut rng, dm);
        fill_real_linear_random(&mut block.attn.w_v, &mut rng, dm);
        fill_real_linear_random(&mut block.attn.w_o, &mut rng, dm);
        fill_real_linear_random(&mut block.ffn.fc1, &mut rng, dm);
        let d_ff = block.ffn.fc1.out_dim;
        fill_real_linear_random(&mut block.ffn.fc2, &mut rng, d_ff);
    }
    fill_real_linear_random(&mut model.head, &mut rng, dm);
}

fn gaussian_unit_vec(seed: u64, n: usize) -> Vec<f32> {
    use std::f32::consts::PI;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut vals = vec![0.0f32; n];
    let mut i = 0;
    while i < n {
        let u1: f32 = rng.gen_range(1e-7f32..1.0);
        let u2: f32 = rng.gen_range(0.0f32..1.0);
        let r = (-2.0 * u1.ln()).sqrt();
        vals[i] = r * (2.0 * PI * u2).cos();
        if i + 1 < n {
            vals[i + 1] = r * (2.0 * PI * u2).sin();
        }
        i += 2;
    }
    let norm: f32 = vals.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for v in &mut vals {
            *v /= norm;
        }
    }
    vals
}

#[inline]
fn token_seed(seed: u64, v: usize) -> u64 {
    seed ^ (v as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// Corpus-semantic embedding init (random indexing), `d_model`-dim per token.
pub fn corpus_semantic_init_vanilla(
    model: &mut VanillaLLM,
    tokens: &[u32],
    seed: u64,
    window: usize,
    scale: f32,
) {
    let vocab = model.embedding.len();
    let dm = model.d_model;
    if vocab == 0 || dm == 0 {
        return;
    }

    let idx: Vec<Vec<f32>> = (0..vocab)
        .map(|v| gaussian_unit_vec(token_seed(seed, v), dm))
        .collect();

    let mut ctx = vec![vec![0.0f32; dm]; vocab];
    let len = tokens.len();
    for i in 0..len {
        let t = tokens[i] as usize;
        if t >= vocab {
            continue;
        }
        for off in 1..=window {
            let w = 1.0 / off as f32;
            if i >= off {
                let nb = tokens[i - off] as usize;
                if nb < vocab {
                    let (src, dst) = (&idx[nb], &mut ctx[t]);
                    for k in 0..dm {
                        dst[k] += w * src[k];
                    }
                }
            }
            if i + off < len {
                let nb = tokens[i + off] as usize;
                if nb < vocab {
                    let (src, dst) = (&idx[nb], &mut ctx[t]);
                    for k in 0..dm {
                        dst[k] += w * src[k];
                    }
                }
            }
        }
    }

    const SELF_ANCHOR: f32 = 1.0;
    for v in 0..vocab {
        let mut c = std::mem::take(&mut ctx[v]);
        for k in 0..dm {
            c[k] += SELF_ANCHOR * idx[v][k];
        }
        let norm: f32 = c.iter().map(|x| x * x).sum::<f32>().sqrt();
        let s = if norm > 1e-8 { scale / norm } else { 0.0 };
        for k in 0..dm {
            model.embedding[v][k] = c[k] * s;
        }
    }
}

// ─── Tape ────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct VanillaAttnTape {
    pub input: Vec<Vec<f32>>,
    pub q: Vec<Vec<f32>>,
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    pub weights: Vec<Vec<Vec<f32>>>,
    pub agg: Vec<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub struct VanillaFfnTape {
    pub inputs: Vec<Vec<f32>>,
    pub hidden_pre: Vec<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub struct VanillaBlockTape {
    pub block_input: Vec<Vec<f32>>,
    pub norm1_stats: Vec<StandardNormStats>,
    pub attn: VanillaAttnTape,
    pub norm2_stats: Vec<StandardNormStats>,
    pub ffn: VanillaFfnTape,
}

#[derive(Clone, Debug)]
pub struct VanillaTape {
    pub logits: Vec<Vec<f32>>,
    pub head_input: Vec<Vec<f32>>,
    pub final_norm_stats: Vec<StandardNormStats>,
    pub blocks: Vec<VanillaBlockTape>,
    pub embed_pe: Vec<Vec<f32>>,
}

fn softmax_row(scores: &[f32]) -> Vec<f32> {
    let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|&s| (s - m).exp()).collect();
    let z: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / z).collect()
}

fn attention_forward_taped(
    attn: &VanillaAttention,
    x: &[Vec<f32>],
    causal: bool,
) -> (Vec<Vec<f32>>, VanillaAttnTape) {
    let seq = x.len();
    let d = attn.d_model;
    let scale = (attn.head_dim as f32).sqrt();
    let q: Vec<Vec<f32>> = x.iter().map(|xi| attn.w_q.forward_flat(xi)).collect();
    let k: Vec<Vec<f32>> = x.iter().map(|xi| attn.w_k.forward_flat(xi)).collect();
    let v: Vec<Vec<f32>> = x.iter().map(|xi| attn.w_v.forward_flat(xi)).collect();

    let mut agg = vec![vec![0.0f32; d]; seq];
    let mut weights = vec![vec![vec![0.0f32; seq]; seq]; attn.n_heads];

    for h in 0..attn.n_heads {
        let d0 = h * attn.head_dim;
        let _d1 = d0 + attn.head_dim;
        for i in 0..seq {
            let mut scores = vec![0.0f32; seq];
            for j in 0..seq {
                if causal && j > i {
                    scores[j] = f32::NEG_INFINITY;
                    continue;
                }
                let mut s = 0.0f32;
                for t in 0..attn.head_dim {
                    s += q[i][d0 + t] * k[j][d0 + t];
                }
                scores[j] = s / scale;
            }
            let w = softmax_row(&scores);
            weights[h][i] = w.clone();
            for j in 0..seq {
                for t in 0..attn.head_dim {
                    agg[i][d0 + t] += w[j] * v[j][d0 + t];
                }
            }
        }
    }

    let out: Vec<Vec<f32>> = agg.iter().map(|o| attn.w_o.forward_flat(o)).collect();
    (
        out,
        VanillaAttnTape {
            input: x.to_vec(),
            q,
            k,
            v,
            weights,
            agg,
        },
    )
}

fn ffn_forward_taped(ffn: &VanillaFFN, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let hidden_pre = ffn.fc1.forward_flat(x);
    let hidden_post: Vec<f32> = hidden_pre.iter().map(|&v| v.max(0.0)).collect();
    let out = ffn.fc2.forward_flat(&hidden_post);
    (out, hidden_pre)
}

fn block_forward_taped(block: &VanillaBlock, x: &mut [Vec<f32>], causal: bool) -> VanillaBlockTape {
    let seq = x.len();
    let d = block.attn.d_model;
    let block_input = x.to_vec();

    let mut norm1_stats = Vec::with_capacity(seq);
    let mut attn_in = Vec::with_capacity(seq);
    for row in x.iter() {
        let (y, stats) =
            standard_layer_norm::forward(row, &block.norm1.gamma, &block.norm1.beta, LN_EPS);
        norm1_stats.push(stats);
        attn_in.push(y);
    }

    let (attn_out, attn_tape) = attention_forward_taped(&block.attn, &attn_in, causal);
    for t in 0..seq {
        for i in 0..d {
            x[t][i] += attn_out[t][i];
        }
    }

    let mut norm2_stats = Vec::with_capacity(seq);
    let mut ffn_inputs = Vec::with_capacity(seq);
    let mut ffn_hidden_pre = Vec::with_capacity(seq);
    for t in 0..seq {
        let (y, stats) =
            standard_layer_norm::forward(&x[t], &block.norm2.gamma, &block.norm2.beta, LN_EPS);
        norm2_stats.push(stats);
        let (delta, hpre) = ffn_forward_taped(&block.ffn, &y);
        ffn_inputs.push(y);
        ffn_hidden_pre.push(hpre);
        for i in 0..d {
            x[t][i] += delta[i];
        }
    }

    VanillaBlockTape {
        block_input,
        norm1_stats,
        attn: attn_tape,
        norm2_stats,
        ffn: VanillaFfnTape {
            inputs: ffn_inputs,
            hidden_pre: ffn_hidden_pre,
        },
    }
}

fn model_forward_taped(model: &VanillaLLM, ids: &[usize], causal: bool) -> VanillaTape {
    let mut x = model.embed_with_positions(ids);
    let embed_pe = x.clone();

    let mut block_tapes = Vec::with_capacity(model.blocks.len());
    for block in &model.blocks {
        block_tapes.push(block_forward_taped(block, &mut x, causal));
    }

    let mut final_norm_stats = Vec::with_capacity(x.len());
    let mut head_input = Vec::with_capacity(x.len());
    let mut logits = Vec::with_capacity(x.len());
    for row in &x {
        let (y, stats) = standard_layer_norm::forward(
            row,
            &model.final_norm.gamma,
            &model.final_norm.beta,
            LN_EPS,
        );
        final_norm_stats.push(stats);
        head_input.push(y.clone());
        logits.push(model.head.forward_flat(&y));
    }

    VanillaTape {
        logits,
        head_input,
        final_norm_stats,
        blocks: block_tapes,
        embed_pe,
    }
}

// ─── Backward ────────────────────────────────────────────────────────────────

fn ln_param_grads_with_gamma(
    stats: &[StandardNormStats],
    gamma: &[f32],
    grad_out: &[Vec<f32>],
) -> (Vec<f32>, Vec<f32>, Vec<Vec<f32>>) {
    let d = gamma.len();
    let mut dgamma = vec![0.0f32; d];
    let mut dbeta = vec![0.0f32; d];
    let mut grad_in = Vec::with_capacity(grad_out.len());
    for (t, g) in grad_out.iter().enumerate() {
        for i in 0..d {
            dgamma[i] += g[i] * stats[t].x_hat[i];
            dbeta[i] += g[i];
        }
        grad_in.push(standard_layer_norm::backward(
            &stats[t].x_hat,
            gamma,
            g,
            stats[t].std,
        ));
    }
    (dgamma, dbeta, grad_in)
}

fn attention_backward(
    attn: &VanillaAttention,
    tape: &VanillaAttnTape,
    grad_out: &[Vec<f32>],
) -> (
    RealHeadGrad,
    RealHeadGrad,
    RealHeadGrad,
    RealHeadGrad,
    Vec<Vec<f32>>,
) {
    let seq = tape.input.len();
    let d = attn.d_model;
    let n_heads = attn.n_heads;
    let head_dim = attn.head_dim;
    let scale = (head_dim as f32).sqrt();
    let inv_scale = 1.0 / scale;

    let mut grad_wq = RealHeadGrad::zeros(d, d);
    let mut grad_wk = RealHeadGrad::zeros(d, d);
    let mut grad_wv = RealHeadGrad::zeros(d, d);
    let mut grad_wo = RealHeadGrad::zeros(d, d);
    let mut grad_agg = vec![vec![0.0f32; d]; seq];

    for i in 0..seq {
        let g = real_linear_backward(&attn.w_o.weights, &tape.agg[i], &grad_out[i], &mut grad_wo);
        for j in 0..d {
            grad_agg[i][j] += g[j];
        }
    }

    let mut grad_v = vec![vec![0.0f32; d]; seq];
    let mut grad_w = vec![vec![vec![0.0f32; seq]; seq]; n_heads];

    for h in 0..n_heads {
        let d0 = h * head_dim;
        let d1 = d0 + head_dim;
        for i in 0..seq {
            for j in 0..seq {
                let w_ij = tape.weights[h][i][j];
                let mut gw = 0.0f32;
                for t in d0..d1 {
                    gw += grad_agg[i][t] * tape.v[j][t];
                }
                grad_w[h][i][j] = gw;
                if w_ij == 0.0 {
                    continue;
                }
                for t in d0..d1 {
                    grad_v[j][t] += w_ij * grad_agg[i][t];
                }
            }
        }
    }

    let mut grad_score = vec![vec![vec![0.0f32; seq]; seq]; n_heads];
    for h in 0..n_heads {
        for i in 0..seq {
            let dot: f32 = (0..seq)
                .map(|l| tape.weights[h][i][l] * grad_w[h][i][l])
                .sum();
            for j in 0..seq {
                grad_score[h][i][j] = tape.weights[h][i][j] * (grad_w[h][i][j] - dot);
            }
        }
    }

    let mut grad_q = vec![vec![0.0f32; d]; seq];
    let mut grad_k = vec![vec![0.0f32; d]; seq];
    for h in 0..n_heads {
        let d0 = h * head_dim;
        let d1 = d0 + head_dim;
        for i in 0..seq {
            for j in 0..seq {
                let gs = grad_score[h][i][j];
                if gs == 0.0 {
                    continue;
                }
                let g = gs * inv_scale;
                for t in d0..d1 {
                    grad_q[i][t] += g * tape.k[j][t];
                    grad_k[j][t] += g * tape.q[i][t];
                }
            }
        }
    }

    let mut grad_input = vec![vec![0.0f32; d]; seq];
    for i in 0..seq {
        let gq = real_linear_backward(&attn.w_q.weights, &tape.input[i], &grad_q[i], &mut grad_wq);
        let gk = real_linear_backward(&attn.w_k.weights, &tape.input[i], &grad_k[i], &mut grad_wk);
        let gv = real_linear_backward(&attn.w_v.weights, &tape.input[i], &grad_v[i], &mut grad_wv);
        for j in 0..d {
            grad_input[i][j] += gq[j] + gk[j] + gv[j];
        }
    }

    (grad_wq, grad_wk, grad_wv, grad_wo, grad_input)
}

fn ffn_backward(
    ffn: &VanillaFFN,
    tape: &VanillaFfnTape,
    grad_out: &[Vec<f32>],
) -> (RealHeadGrad, RealHeadGrad, Vec<Vec<f32>>) {
    let seq = grad_out.len();
    let d_model = ffn.fc1.in_features;
    let d_ff = ffn.fc1.out_dim;
    let mut grad_fc2 = RealHeadGrad::zeros(d_model, d_ff);
    let mut grad_fc1 = RealHeadGrad::zeros(d_ff, d_model);
    let mut grad_in = vec![vec![0.0f32; d_model]; seq];

    for i in 0..seq {
        let post: Vec<f32> = tape.hidden_pre[i].iter().map(|&v| v.max(0.0)).collect();
        let g_h = real_linear_backward(&ffn.fc2.weights, &post, &grad_out[i], &mut grad_fc2);
        let g_pre: Vec<f32> = g_h
            .iter()
            .zip(&tape.hidden_pre[i])
            .map(|(&g, &h)| if h > 0.0 { g } else { 0.0 })
            .collect();
        let g_x = real_linear_backward(&ffn.fc1.weights, &tape.inputs[i], &g_pre, &mut grad_fc1);
        grad_in[i] = g_x;
    }

    (grad_fc1, grad_fc2, grad_in)
}

struct VanillaBlockGrads {
    norm1_gamma: Vec<f32>,
    norm1_beta: Vec<f32>,
    w_q: RealHeadGrad,
    w_k: RealHeadGrad,
    w_v: RealHeadGrad,
    w_o: RealHeadGrad,
    norm2_gamma: Vec<f32>,
    norm2_beta: Vec<f32>,
    fc1: RealHeadGrad,
    fc2: RealHeadGrad,
    grad_input: Vec<Vec<f32>>,
}

fn block_backward(
    block: &VanillaBlock,
    tape: &VanillaBlockTape,
    grad_out: &[Vec<f32>],
) -> VanillaBlockGrads {
    let seq = tape.block_input.len();
    let d = block.attn.d_model;

    let grad_ffn_out: Vec<Vec<f32>> = grad_out.to_vec();
    let mut grad_after_res1: Vec<Vec<f32>> = grad_out.to_vec();

    let (grad_fc1, grad_fc2, grad_ffn_in) = ffn_backward(&block.ffn, &tape.ffn, &grad_ffn_out);

    let (grad_n2_gamma, grad_n2_beta, grad_from_n2) =
        ln_param_grads_with_gamma(&tape.norm2_stats, &block.norm2.gamma, &grad_ffn_in);
    for i in 0..seq {
        for j in 0..d {
            grad_after_res1[i][j] += grad_from_n2[i][j];
        }
    }

    let grad_attn_out: Vec<Vec<f32>> = grad_after_res1.clone();
    let mut grad_block_input: Vec<Vec<f32>> = grad_after_res1;

    let (grad_wq, grad_wk, grad_wv, grad_wo, grad_from_attn) =
        attention_backward(&block.attn, &tape.attn, &grad_attn_out);

    let (grad_n1_gamma, grad_n1_beta, grad_from_n1) =
        ln_param_grads_with_gamma(&tape.norm1_stats, &block.norm1.gamma, &grad_from_attn);
    for i in 0..seq {
        for j in 0..d {
            grad_block_input[i][j] += grad_from_n1[i][j];
        }
    }

    VanillaBlockGrads {
        norm1_gamma: grad_n1_gamma,
        norm1_beta: grad_n1_beta,
        w_q: grad_wq,
        w_k: grad_wk,
        w_v: grad_wv,
        w_o: grad_wo,
        norm2_gamma: grad_n2_gamma,
        norm2_beta: grad_n2_beta,
        fc1: grad_fc1,
        fc2: grad_fc2,
        grad_input: grad_block_input,
    }
}

// ─── Sparse real embedding grad / optimiser ──────────────────────────────────

#[derive(Clone, Debug)]
pub struct VanillaEmbeddingGrad {
    pub d_model: usize,
    pub grads: HashMap<usize, Vec<f32>>,
}

impl VanillaEmbeddingGrad {
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,
            grads: HashMap::new(),
        }
    }

    pub fn accumulate(&mut self, token_id: usize, grad: &[f32]) {
        debug_assert_eq!(grad.len(), self.d_model);
        let entry = self
            .grads
            .entry(token_id)
            .or_insert_with(|| vec![0.0; self.d_model]);
        for i in 0..self.d_model {
            entry[i] += grad[i];
        }
    }

    pub fn scale(&mut self, s: f32) {
        for entry in self.grads.values_mut() {
            for v in entry {
                *v *= s;
            }
        }
    }

    pub fn sq_norm(&self) -> f32 {
        self.grads
            .values()
            .map(|g| g.iter().map(|v| v * v).sum::<f32>())
            .sum()
    }

    pub fn merge(&mut self, other: &VanillaEmbeddingGrad) {
        for (&tid, grad) in &other.grads {
            self.accumulate(tid, grad);
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VanillaEmbedAdamState {
    m: Vec<f32>,
    v: Vec<f32>,
    step: u64,
}

/// Sparse ("lazy") Adam over embedding rows: only rows seen in a step update,
/// each with its own bias-correction step count. Embeddings are never decayed.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VanillaEmbeddingOptimizer {
    pub d_model: usize,
    pub cfg: AdamConfig,
    states: HashMap<usize, VanillaEmbedAdamState>,
}

impl VanillaEmbeddingOptimizer {
    pub fn new(d_model: usize, cfg: AdamConfig) -> Self {
        Self {
            d_model,
            cfg,
            states: HashMap::new(),
        }
    }

    pub fn step(&mut self, embedding: &mut [Vec<f32>], grad: &VanillaEmbeddingGrad) {
        for (&token_id, token_grad) in &grad.grads {
            let state = self
                .states
                .entry(token_id)
                .or_insert_with(|| VanillaEmbedAdamState {
                    m: vec![0.0; self.d_model],
                    v: vec![0.0; self.d_model],
                    step: 0,
                });
            state.step += 1;
            let t = state.step as f32;
            let bc1 = 1.0 - self.cfg.beta1.powf(t);
            let bc2 = 1.0 - self.cfg.beta2.powf(t);
            for d in 0..self.d_model {
                let g = token_grad[d];
                state.m[d] = self.cfg.beta1 * state.m[d] + (1.0 - self.cfg.beta1) * g;
                state.v[d] = self.cfg.beta2 * state.v[d] + (1.0 - self.cfg.beta2) * g * g;
                let m_hat = state.m[d] / bc1;
                let v_hat = state.v[d] / bc2;
                embedding[token_id][d] -= self.cfg.lr * m_hat / (v_hat.sqrt() + self.cfg.eps);
            }
        }
    }
}

// ─── Optimiser state ─────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VanillaBlockOptimizer {
    pub wq: RealHeadOptimizer,
    pub wk: RealHeadOptimizer,
    pub wv: RealHeadOptimizer,
    pub wo: RealHeadOptimizer,
    pub fc1: RealHeadOptimizer,
    pub fc2: RealHeadOptimizer,
    pub norm1_gamma_m: Vec<f32>,
    pub norm1_gamma_v: Vec<f32>,
    pub norm1_beta_m: Vec<f32>,
    pub norm1_beta_v: Vec<f32>,
    pub norm2_gamma_m: Vec<f32>,
    pub norm2_gamma_v: Vec<f32>,
    pub norm2_beta_m: Vec<f32>,
    pub norm2_beta_v: Vec<f32>,
    pub step: u64,
}

impl VanillaBlockOptimizer {
    /// `adam` is the matrix config (AdamW decay applies to Q/K/V/O/FFN only;
    /// LayerNorm params use a decay-free copy — see `apply_grads_vanilla`).
    pub fn new(cfg: &TrainConfigV2, adam: AdamConfig) -> Self {
        let dm = cfg.d_model;
        Self {
            wq: RealHeadOptimizer::new(dm, dm, adam.clone()),
            wk: RealHeadOptimizer::new(dm, dm, adam.clone()),
            wv: RealHeadOptimizer::new(dm, dm, adam.clone()),
            wo: RealHeadOptimizer::new(dm, dm, adam.clone()),
            fc1: RealHeadOptimizer::new(cfg.d_ff, dm, adam.clone()),
            fc2: RealHeadOptimizer::new(dm, cfg.d_ff, adam.clone()),
            norm1_gamma_m: vec![0.0; dm],
            norm1_gamma_v: vec![0.0; dm],
            norm1_beta_m: vec![0.0; dm],
            norm1_beta_v: vec![0.0; dm],
            norm2_gamma_m: vec![0.0; dm],
            norm2_gamma_v: vec![0.0; dm],
            norm2_beta_m: vec![0.0; dm],
            norm2_beta_v: vec![0.0; dm],
            step: 0,
        }
    }
}

fn adam_step_scalar(
    params: &mut [f32],
    grads: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    step: u64,
    cfg: &AdamConfig,
) {
    let t = step as f32;
    let bc1 = 1.0 - cfg.beta1.powf(t);
    let bc2 = 1.0 - cfg.beta2.powf(t);
    // LayerNorm gamma/beta: never weight-decayed (decaying gamma toward 0
    // shrinks every residual branch).
    for i in 0..params.len() {
        let g = grads[i];
        m[i] = cfg.beta1 * m[i] + (1.0 - cfg.beta1) * g;
        v[i] = cfg.beta2 * v[i] + (1.0 - cfg.beta2) * g * g;
        let m_hat = m[i] / bc1;
        let v_hat = v[i] / bc2;
        params[i] -= cfg.lr * m_hat / (v_hat.sqrt() + cfg.eps);
    }
}

pub struct VanillaModelState {
    pub model: VanillaLLM,
    pub cfg: TrainConfigV2,
    pub step: u64,
    pub block_opts: Vec<VanillaBlockOptimizer>,
    pub head_opt: RealHeadOptimizer,
    pub embed_opt: VanillaEmbeddingOptimizer,
    pub fnorm_gamma_m: Vec<f32>,
    pub fnorm_gamma_v: Vec<f32>,
    pub fnorm_beta_m: Vec<f32>,
    pub fnorm_beta_v: Vec<f32>,
    pub fnorm_step: u64,
    /// Global grad norm of the last update (pre-clip), for logging.
    pub last_grad_norm: f32,
}

/// Serializable snapshot of every optimiser moment, so `--resume` continues
/// exactly where a run stopped instead of restarting Adam from zero.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VanillaOptimState {
    pub step: u64,
    pub block_opts: Vec<VanillaBlockOptimizer>,
    pub head_opt: RealHeadOptimizer,
    pub embed_opt: VanillaEmbeddingOptimizer,
    pub fnorm_gamma_m: Vec<f32>,
    pub fnorm_gamma_v: Vec<f32>,
    pub fnorm_beta_m: Vec<f32>,
    pub fnorm_beta_v: Vec<f32>,
    pub fnorm_step: u64,
}

/// Matrix-param Adam config derived from the training config (AdamW when
/// `weight_decay > 0`).
pub fn vanilla_adam_config(cfg: &TrainConfigV2) -> AdamConfig {
    AdamConfig {
        lr: cfg.lr_max,
        beta2: cfg.adam_beta2,
        weight_decay: cfg.weight_decay,
        decoupled_weight_decay: true,
        ..Default::default()
    }
}

/// Same betas/lr as `base`, but no weight decay (LayerNorm, biases, embeddings).
fn no_decay(base: &AdamConfig) -> AdamConfig {
    AdamConfig {
        weight_decay: 0.0,
        ..base.clone()
    }
}

impl VanillaModelState {
    fn fresh_optimisers(
        cfg: &TrainConfigV2,
    ) -> (
        Vec<VanillaBlockOptimizer>,
        RealHeadOptimizer,
        VanillaEmbeddingOptimizer,
    ) {
        let adam = vanilla_adam_config(cfg);
        let block_opts: Vec<_> = (0..cfg.n_blocks)
            .map(|_| VanillaBlockOptimizer::new(cfg, adam.clone()))
            .collect();
        let head_opt = RealHeadOptimizer::new(cfg.vocab_size, cfg.d_model, adam.clone());
        let embed_opt = VanillaEmbeddingOptimizer::new(cfg.d_model, no_decay(&adam));
        (block_opts, head_opt, embed_opt)
    }

    pub fn new(mut cfg: TrainConfigV2) -> Self {
        if cfg.embed_scale <= 0.0 {
            cfg.embed_scale = (cfg.d_model as f32).sqrt();
        }
        let (block_opts, head_opt, embed_opt) = Self::fresh_optimisers(&cfg);
        let mut model = VanillaLLM::new(
            cfg.vocab_size,
            cfg.d_model,
            cfg.n_heads,
            cfg.d_ff,
            cfg.n_blocks,
            cfg.init_seed,
        );
        model.embed_scale = cfg.embed_scale;
        randomize_vanilla_model(&mut model, cfg.init_seed);
        if cfg.tie_embeddings {
            model.sync_tied_head();
        }
        let dm = cfg.d_model;
        let mut st = Self {
            model,
            cfg,
            step: 0,
            block_opts,
            head_opt,
            embed_opt,
            fnorm_gamma_m: vec![0.0; dm],
            fnorm_gamma_v: vec![0.0; dm],
            fnorm_beta_m: vec![0.0; dm],
            fnorm_beta_v: vec![0.0; dm],
            fnorm_step: 0,
            last_grad_norm: 0.0,
        };
        st.update_lr();
        st
    }

    /// Weights-only load (fresh optimiser moments). Use `restore_optim` to
    /// resume a run exactly.
    pub fn from_loaded(cfg: TrainConfigV2, mut model: VanillaLLM, step: u64) -> Self {
        let (block_opts, head_opt, embed_opt) = Self::fresh_optimisers(&cfg);
        model.embed_scale = if cfg.embed_scale > 0.0 {
            cfg.embed_scale
        } else {
            1.0
        };
        let dm = cfg.d_model;
        let mut st = Self {
            model,
            cfg,
            step,
            block_opts,
            head_opt,
            embed_opt,
            fnorm_gamma_m: vec![0.0; dm],
            fnorm_gamma_v: vec![0.0; dm],
            fnorm_beta_m: vec![0.0; dm],
            fnorm_beta_v: vec![0.0; dm],
            fnorm_step: 0,
            last_grad_norm: 0.0,
        };
        st.update_lr();
        st
    }

    /// Rebuild optimisers after changing `cfg` training knobs (lr / wd / β₂),
    /// e.g. when fine-tuning from a base checkpoint.
    pub fn reset_optimisers(&mut self) {
        let (b, h, e) = Self::fresh_optimisers(&self.cfg);
        self.block_opts = b;
        self.head_opt = h;
        self.embed_opt = e;
        let dm = self.cfg.d_model;
        self.fnorm_gamma_m = vec![0.0; dm];
        self.fnorm_gamma_v = vec![0.0; dm];
        self.fnorm_beta_m = vec![0.0; dm];
        self.fnorm_beta_v = vec![0.0; dm];
        self.fnorm_step = 0;
        self.update_lr();
    }

    pub fn optim_snapshot(&self) -> VanillaOptimState {
        VanillaOptimState {
            step: self.step,
            block_opts: self.block_opts.clone(),
            head_opt: self.head_opt.clone(),
            embed_opt: self.embed_opt.clone(),
            fnorm_gamma_m: self.fnorm_gamma_m.clone(),
            fnorm_gamma_v: self.fnorm_gamma_v.clone(),
            fnorm_beta_m: self.fnorm_beta_m.clone(),
            fnorm_beta_v: self.fnorm_beta_v.clone(),
            fnorm_step: self.fnorm_step,
        }
    }

    pub fn restore_optim(&mut self, o: VanillaOptimState) -> Result<(), String> {
        if o.block_opts.len() != self.cfg.n_blocks {
            return Err("optimizer state block count mismatch".into());
        }
        if o.head_opt.w_m.len() != self.cfg.vocab_size || o.fnorm_gamma_m.len() != self.cfg.d_model
        {
            return Err("optimizer state shape mismatch".into());
        }
        if o.step != self.step {
            return Err(format!(
                "optimizer state step {} != checkpoint step {}",
                o.step, self.step
            ));
        }
        self.block_opts = o.block_opts;
        self.head_opt = o.head_opt;
        self.embed_opt = o.embed_opt;
        self.fnorm_gamma_m = o.fnorm_gamma_m;
        self.fnorm_gamma_v = o.fnorm_gamma_v;
        self.fnorm_beta_m = o.fnorm_beta_m;
        self.fnorm_beta_v = o.fnorm_beta_v;
        self.fnorm_step = o.fnorm_step;
        self.update_lr();
        Ok(())
    }

    /// Current learning rate (the one the next update will use).
    pub fn current_lr(&self) -> f32 {
        self.head_opt.cfg.lr
    }

    /// Set every optimiser's LR for the update at index `self.step`
    /// (0-based), so step 0 starts at `lr_min` and warms up.
    pub fn update_lr(&mut self) {
        let lr = cosine_lr_with_warmup(
            self.step,
            self.cfg.warmup_steps,
            self.cfg.total_steps,
            self.cfg.lr_max,
            self.cfg.lr_min,
        );
        for b in &mut self.block_opts {
            b.wq.cfg.lr = lr;
            b.wk.cfg.lr = lr;
            b.wv.cfg.lr = lr;
            b.wo.cfg.lr = lr;
            b.fc1.cfg.lr = lr;
            b.fc2.cfg.lr = lr;
        }
        self.head_opt.cfg.lr = lr;
        self.embed_opt.cfg.lr = lr;
    }
}

struct VanillaStepGrads {
    head: RealHeadGrad,
    fnorm_dgamma: Vec<f32>,
    fnorm_dbeta: Vec<f32>,
    blocks: Vec<VanillaBlockGrads>,
    embed: VanillaEmbeddingGrad,
    /// Summed (not averaged) NLL over `n_tokens` supervised positions.
    loss: f32,
    n_tokens: usize,
    valid: bool,
}

impl VanillaStepGrads {
    fn zeros(cfg: &TrainConfigV2) -> Self {
        let dm = cfg.d_model;
        Self {
            head: RealHeadGrad::zeros(cfg.vocab_size, dm),
            fnorm_dgamma: vec![0.0; dm],
            fnorm_dbeta: vec![0.0; dm],
            blocks: (0..cfg.n_blocks)
                .map(|_| VanillaBlockGrads {
                    norm1_gamma: vec![0.0; dm],
                    norm1_beta: vec![0.0; dm],
                    w_q: RealHeadGrad::zeros(dm, dm),
                    w_k: RealHeadGrad::zeros(dm, dm),
                    w_v: RealHeadGrad::zeros(dm, dm),
                    w_o: RealHeadGrad::zeros(dm, dm),
                    norm2_gamma: vec![0.0; dm],
                    norm2_beta: vec![0.0; dm],
                    fc1: RealHeadGrad::zeros(cfg.d_ff, dm),
                    fc2: RealHeadGrad::zeros(dm, cfg.d_ff),
                    grad_input: Vec::new(),
                })
                .collect(),
            embed: VanillaEmbeddingGrad::new(dm),
            loss: 0.0,
            n_tokens: 0,
            valid: false,
        }
    }

    /// Squared global L2 norm over every tensor the optimiser will update.
    fn sq_norm(&self, cfg: &TrainConfigV2) -> f32 {
        let mut sq = if cfg.tie_embeddings {
            // Tied: head weight grads were merged into `embed`; only the bias
            // is stepped on the head itself.
            self.head.bias_sq_norm()
        } else {
            self.head.sq_norm()
        };
        sq += self.fnorm_dgamma.iter().map(|v| v * v).sum::<f32>();
        sq += self.fnorm_dbeta.iter().map(|v| v * v).sum::<f32>();
        for (b, g) in self.blocks.iter().enumerate() {
            if b < cfg.freeze_blocks {
                continue;
            }
            sq += g.w_q.sq_norm() + g.w_k.sq_norm() + g.w_v.sq_norm() + g.w_o.sq_norm();
            sq += g.fc1.sq_norm() + g.fc2.sq_norm();
            for v in g
                .norm1_gamma
                .iter()
                .chain(&g.norm1_beta)
                .chain(&g.norm2_gamma)
                .chain(&g.norm2_beta)
            {
                sq += v * v;
            }
        }
        if cfg.train_embeddings && !cfg.freeze_embeddings {
            sq += self.embed.sq_norm();
        }
        sq
    }

    fn add(&mut self, o: &VanillaStepGrads) {
        self.head.accumulate(&o.head);
        for i in 0..self.fnorm_dgamma.len() {
            self.fnorm_dgamma[i] += o.fnorm_dgamma[i];
            self.fnorm_dbeta[i] += o.fnorm_dbeta[i];
        }
        for b in 0..self.blocks.len() {
            let sb = &mut self.blocks[b];
            let ob = &o.blocks[b];
            for i in 0..sb.norm1_gamma.len() {
                sb.norm1_gamma[i] += ob.norm1_gamma[i];
                sb.norm1_beta[i] += ob.norm1_beta[i];
                sb.norm2_gamma[i] += ob.norm2_gamma[i];
                sb.norm2_beta[i] += ob.norm2_beta[i];
            }
            sb.w_q.accumulate(&ob.w_q);
            sb.w_k.accumulate(&ob.w_k);
            sb.w_v.accumulate(&ob.w_v);
            sb.w_o.accumulate(&ob.w_o);
            sb.fc1.accumulate(&ob.fc1);
            sb.fc2.accumulate(&ob.fc2);
        }
        self.embed.merge(&o.embed);
        self.loss += o.loss;
        self.n_tokens += o.n_tokens;
        self.valid |= o.valid;
    }

    fn scale(&mut self, s: f32) {
        self.head.scale(s);
        for v in &mut self.fnorm_dgamma {
            *v *= s;
        }
        for v in &mut self.fnorm_dbeta {
            *v *= s;
        }
        for b in &mut self.blocks {
            for v in &mut b.norm1_gamma {
                *v *= s;
            }
            for v in &mut b.norm1_beta {
                *v *= s;
            }
            for v in &mut b.norm2_gamma {
                *v *= s;
            }
            for v in &mut b.norm2_beta {
                *v *= s;
            }
            b.w_q.scale(s);
            b.w_k.scale(s);
            b.w_v.scale(s);
            b.w_o.scale(s);
            b.fc1.scale(s);
            b.fc2.scale(s);
        }
        self.embed.scale(s);
    }
}

/// Forward + backward for one example. Gradients and loss are **sums** over
/// the example's supervised positions (`n_tokens`); the caller normalises by
/// the total token count of the whole step so every token weighs the same,
/// regardless of how many supervised positions each example has.
fn compute_grads_vanilla(state: &VanillaModelState, example: &TrainExample) -> VanillaStepGrads {
    let seq = example.len();
    let dm = state.cfg.d_model;
    let vocab = state.cfg.vocab_size;
    let mut out = VanillaStepGrads::zeros(&state.cfg);

    let tape = model_forward_taped(&state.model, &example.full_ids, true);
    let loss_mask = example.loss_mask();
    let mut total_loss = 0.0f32;
    let mut n_loss = 0usize;
    let mut grad_logits = vec![vec![0.0f32; vocab]; seq];

    for t in 0..seq {
        if !loss_mask[t] || t + 1 >= seq {
            continue;
        }
        let target = example.full_ids[t + 1];
        let (loss, gl) = cross_entropy(&tape.logits[t], target);
        total_loss += loss;
        n_loss += 1;
        grad_logits[t] = gl;
    }
    if n_loss == 0 {
        return out;
    }

    let mut grad_head = std::mem::replace(&mut out.head, RealHeadGrad::zeros(0, 0));
    let mut grad_x_final = vec![vec![0.0f32; dm]; seq];
    for t in 0..seq {
        if grad_logits[t].iter().all(|&g| g == 0.0) {
            continue;
        }
        let gx = real_linear_backward(
            &state.model.head.weights,
            &tape.head_input[t],
            &grad_logits[t],
            &mut grad_head,
        );
        for j in 0..dm {
            grad_x_final[t][j] += gx[j];
        }
    }

    let mut fnorm_dgamma = vec![0.0f32; dm];
    let mut fnorm_dbeta = vec![0.0f32; dm];
    let mut grad_x = vec![vec![0.0f32; dm]; seq];
    for t in 0..seq {
        if grad_x_final[t].iter().all(|&g| g == 0.0) {
            continue;
        }
        let stats = &tape.final_norm_stats[t];
        for i in 0..dm {
            fnorm_dgamma[i] += grad_x_final[t][i] * stats.x_hat[i];
            fnorm_dbeta[i] += grad_x_final[t][i];
        }
        grad_x[t] = standard_layer_norm::backward(
            &stats.x_hat,
            &state.model.final_norm.gamma,
            &grad_x_final[t],
            stats.std,
        );
    }

    let train_embed = state.cfg.train_embeddings && !state.cfg.freeze_embeddings;
    // Blocks below `lowest_needed` are frozen and nothing beneath them trains,
    // so backprop can stop early.
    let lowest_needed = if train_embed {
        0
    } else {
        state.cfg.freeze_blocks.min(state.cfg.n_blocks)
    };
    let mut block_grads: Vec<Option<VanillaBlockGrads>> =
        (0..state.cfg.n_blocks).map(|_| None).collect();
    for b in (lowest_needed..state.cfg.n_blocks).rev() {
        let block = &state.model.blocks[b];
        let block_tape = &tape.blocks[b];
        let mut grads = block_backward(block, block_tape, &grad_x);
        grad_x = std::mem::take(&mut grads.grad_input);
        block_grads[b] = Some(grads);
    }
    for (i, g) in block_grads.into_iter().enumerate() {
        if let Some(g) = g {
            out.blocks[i] = g;
        }
    }

    let tied = state.cfg.tie_embeddings;
    let mut embed_grad = VanillaEmbeddingGrad::new(dm);
    if train_embed {
        // x = embed_scale · E[id] + PE  ⇒  dL/dE[id] = embed_scale · dL/dx
        let es = state.model.embed_scale;
        for t in 0..seq {
            if es == 1.0 {
                embed_grad.accumulate(example.full_ids[t], &grad_x[t]);
            } else {
                let g: Vec<f32> = grad_x[t].iter().map(|v| v * es).collect();
                embed_grad.accumulate(example.full_ids[t], &g);
            }
        }
        if tied {
            for v in 0..vocab {
                let row = &grad_head.d_weights[v];
                if row.iter().all(|&g| g == 0.0) {
                    continue;
                }
                embed_grad.accumulate(v, row);
            }
        }
    }

    out.head = grad_head;
    out.fnorm_dgamma = fnorm_dgamma;
    out.fnorm_dbeta = fnorm_dbeta;
    out.embed = embed_grad;
    out.loss = total_loss;
    out.n_tokens = n_loss;
    out.valid = true;
    out
}

fn apply_grads_vanilla(state: &mut VanillaModelState, grads: &VanillaStepGrads) {
    let tied = state.cfg.tie_embeddings;

    // LayerNorm params share the matrices' LR/betas but are never decayed.
    let ln_cfg = no_decay(&state.head_opt.cfg);

    state.fnorm_step += 1;
    adam_step_scalar(
        &mut state.model.final_norm.gamma,
        &grads.fnorm_dgamma,
        &mut state.fnorm_gamma_m,
        &mut state.fnorm_gamma_v,
        state.fnorm_step,
        &ln_cfg,
    );
    adam_step_scalar(
        &mut state.model.final_norm.beta,
        &grads.fnorm_dbeta,
        &mut state.fnorm_beta_m,
        &mut state.fnorm_beta_v,
        state.fnorm_step,
        &ln_cfg,
    );

    for b in 0..state.cfg.n_blocks {
        if b < state.cfg.freeze_blocks {
            continue;
        }
        let pg = &grads.blocks[b];
        let opt = &mut state.block_opts[b];
        opt.step += 1;
        let lr_step = opt.step;
        let bm = &mut state.model.blocks[b];

        opt.wq.step(&mut bm.attn.w_q, &pg.w_q);
        opt.wk.step(&mut bm.attn.w_k, &pg.w_k);
        opt.wv.step(&mut bm.attn.w_v, &pg.w_v);
        opt.wo.step(&mut bm.attn.w_o, &pg.w_o);
        opt.fc1.step(&mut bm.ffn.fc1, &pg.fc1);
        opt.fc2.step(&mut bm.ffn.fc2, &pg.fc2);

        adam_step_scalar(
            &mut bm.norm1.gamma,
            &pg.norm1_gamma,
            &mut opt.norm1_gamma_m,
            &mut opt.norm1_gamma_v,
            lr_step,
            &ln_cfg,
        );
        adam_step_scalar(
            &mut bm.norm1.beta,
            &pg.norm1_beta,
            &mut opt.norm1_beta_m,
            &mut opt.norm1_beta_v,
            lr_step,
            &ln_cfg,
        );
        adam_step_scalar(
            &mut bm.norm2.gamma,
            &pg.norm2_gamma,
            &mut opt.norm2_gamma_m,
            &mut opt.norm2_gamma_v,
            lr_step,
            &ln_cfg,
        );
        adam_step_scalar(
            &mut bm.norm2.beta,
            &pg.norm2_beta,
            &mut opt.norm2_beta_m,
            &mut opt.norm2_beta_v,
            lr_step,
            &ln_cfg,
        );
    }

    if tied {
        state
            .head_opt
            .step_bias_only(&mut state.model.head, &grads.head);
    } else {
        state.head_opt.step(&mut state.model.head, &grads.head);
    }

    if state.cfg.train_embeddings && !state.cfg.freeze_embeddings {
        state
            .embed_opt
            .step(&mut state.model.embedding, &grads.embed);
    }

    if tied {
        state.model.sync_tied_head();
    }
}

/// Worker threads for per-example gradient work. `GF_THREADS` overrides.
fn worker_threads(n_items: usize) -> usize {
    let hw = std::env::var("GF_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        });
    hw.min(n_items).max(1)
}

/// Compute per-example grads in parallel (examples are independent given the
/// current weights) and sum them.
fn compute_grads_batch(state: &VanillaModelState, examples: &[TrainExample]) -> VanillaStepGrads {
    let n_threads = worker_threads(examples.len());
    if n_threads <= 1 {
        let mut acc = VanillaStepGrads::zeros(&state.cfg);
        for ex in examples {
            let g = compute_grads_vanilla(state, ex);
            if g.valid {
                acc.add(&g);
            }
        }
        return acc;
    }
    let chunk = examples.len().div_ceil(n_threads);
    let partials: Vec<VanillaStepGrads> = std::thread::scope(|scope| {
        let handles: Vec<_> = examples
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    let mut acc = VanillaStepGrads::zeros(&state.cfg);
                    for ex in part {
                        let g = compute_grads_vanilla(state, ex);
                        if g.valid {
                            acc.add(&g);
                        }
                    }
                    acc
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("grad worker panicked"))
            .collect()
    });
    let mut it = partials.into_iter();
    let mut acc = it
        .next()
        .unwrap_or_else(|| VanillaStepGrads::zeros(&state.cfg));
    for p in it {
        acc.add(&p);
    }
    acc
}

/// One optimiser update over `examples` (a batch / accumulated micro-batches).
///
/// 1. Per-example grads are computed in parallel and **summed**.
/// 2. The sum is divided by the step's total supervised-token count, so the
///    update is the true mean over tokens (padding-heavy chunks don't get the
///    same weight as full ones).
/// 3. A single global L2 clip is applied across every trainable tensor
///    (per-tensor / per-example clipping distorted the update direction).
/// 4. The LR for this update index is set *before* stepping, so step 0 runs
///    at `lr_min` (warmup start) rather than `lr_max`.
///
/// Returns the mean token NLL of the batch.
pub fn train_step_vanilla_accum(state: &mut VanillaModelState, examples: &[TrainExample]) -> f32 {
    if examples.is_empty() {
        return 0.0;
    }
    let mut acc = compute_grads_batch(state, examples);
    if !acc.valid || acc.n_tokens == 0 {
        return 0.0;
    }
    let inv = 1.0 / acc.n_tokens as f32;
    acc.scale(inv);
    let mean_loss = acc.loss * inv;

    let norm = acc.sq_norm(&state.cfg).sqrt();
    state.last_grad_norm = norm;
    let clip = state.cfg.grad_clip;
    if clip > 0.0 && norm.is_finite() && norm > clip {
        acc.scale(clip / norm);
    }
    if !norm.is_finite() {
        eprintln!(
            "[train] warning: non-finite grad norm at step {} — update skipped",
            state.step
        );
        return mean_loss;
    }

    state.update_lr();
    apply_grads_vanilla(state, &acc);
    state.step += 1;
    state.update_lr();
    mean_loss
}

/// Summed NLL and supervised-token count for one example (for token-weighted
/// validation means and bits/byte).
pub fn eval_vanilla_nll_sum(state: &VanillaModelState, ex: &TrainExample) -> (f64, usize) {
    let logits = vanilla_forward_logits(&state.model, &ex.full_ids, true);
    let mask = ex.loss_mask();
    let mut total = 0.0f64;
    let mut n = 0usize;
    for t in 0..ex.len() {
        if !mask[t] || t + 1 >= ex.len() {
            continue;
        }
        let (loss, _) = cross_entropy(&logits[t], ex.full_ids[t + 1]);
        total += loss as f64;
        n += 1;
    }
    (total, n)
}

pub fn eval_vanilla_lm_loss(state: &VanillaModelState, ex: &TrainExample) -> f32 {
    let (total, n) = eval_vanilla_nll_sum(state, ex);
    if n == 0 {
        0.0
    } else {
        (total / n as f64) as f32
    }
}

/// Token-weighted mean NLL over a fixed validation set, evaluated in parallel.
/// Returns `(mean_nll, summed_nll, n_tokens)`.
pub fn eval_vanilla_set(state: &VanillaModelState, set: &[TrainExample]) -> (f32, f64, usize) {
    if set.is_empty() {
        return (0.0, 0.0, 0);
    }
    let n_threads = worker_threads(set.len());
    let chunk = set.len().div_ceil(n_threads);
    let parts: Vec<(f64, usize)> = std::thread::scope(|scope| {
        let hs: Vec<_> = set
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    part.iter().fold((0.0f64, 0usize), |(s, n), ex| {
                        let (a, b) = eval_vanilla_nll_sum(state, ex);
                        (s + a, n + b)
                    })
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("eval worker"))
            .collect()
    });
    let (sum, n) = parts
        .into_iter()
        .fold((0.0f64, 0usize), |(s, n), (a, b)| (s + a, n + b));
    let mean = if n == 0 { 0.0 } else { (sum / n as f64) as f32 };
    (mean, sum, n)
}

/// Early-stopping bookkeeping on a validation metric (lower is better).
#[derive(Clone, Debug)]
pub struct EarlyStopper {
    pub best: f32,
    pub best_step: u64,
    pub bad_evals: usize,
    /// Consecutive non-improving evals tolerated before stopping (0 = never stop).
    pub patience: usize,
    /// Minimum decrease that counts as an improvement.
    pub min_delta: f32,
}

impl EarlyStopper {
    pub fn new(patience: usize, min_delta: f32) -> Self {
        Self {
            best: f32::INFINITY,
            best_step: 0,
            bad_evals: 0,
            patience,
            min_delta,
        }
    }

    /// Record a validation value. Returns `true` if it is a new best.
    pub fn observe(&mut self, step: u64, value: f32) -> bool {
        if value.is_finite() && value < self.best - self.min_delta {
            self.best = value;
            self.best_step = step;
            self.bad_evals = 0;
            true
        } else {
            self.bad_evals += 1;
            false
        }
    }

    pub fn should_stop(&self) -> bool {
        self.patience > 0 && self.bad_evals >= self.patience
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(tied: bool) -> TrainConfigV2 {
        let mut c = TrainConfigV2::small(20);
        c.d_model = 8;
        c.n_heads = 2;
        c.d_ff = 16;
        c.n_blocks = 2;
        c.tie_embeddings = tied;
        c.warmup_steps = 5;
        c.total_steps = 100;
        c.lr_max = 1e-2;
        c.lr_min = 1e-5;
        c
    }

    fn ex(ids: &[usize]) -> TrainExample {
        TrainExample::lm_sequence(ids.to_vec())
    }

    fn summed_loss(st: &VanillaModelState, e: &TrainExample) -> f64 {
        eval_vanilla_nll_sum(st, e).0
    }

    /// Central difference of the summed loss w.r.t. one scalar, taking the
    /// closest agreement over a few step sizes (ReLU kinks and f32 round-off
    /// make any single `h` noisy).
    fn fd_close(
        st: &mut VanillaModelState,
        e: &TrainExample,
        analytic: f32,
        set: &dyn Fn(&mut VanillaModelState, f32),
        orig: f32,
    ) -> (bool, f32) {
        let mut best = f32::INFINITY;
        let mut best_num = 0.0;
        for h in [3e-3f32, 1e-3, 3e-4] {
            set(st, orig + h);
            let lp = summed_loss(st, e);
            set(st, orig - h);
            let lm = summed_loss(st, e);
            set(st, orig);
            let num = ((lp - lm) / (2.0 * h as f64)) as f32;
            let err = (analytic - num).abs() / (1.0 + num.abs());
            if err < best {
                best = err;
                best_num = num;
            }
        }
        (best < 1e-2, best_num)
    }

    /// Finite-difference check of the analytic gradient, including the
    /// sqrt(d_model) embedding scale and the tied head.
    #[test]
    fn gradients_match_finite_differences() {
        for tied in [false, true] {
            let mut st = VanillaModelState::new(cfg(tied));
            assert!((st.model.embed_scale - 8f32.sqrt()).abs() < 1e-6);
            let e = ex(&[4, 7, 9, 4, 11, 12, 7]);
            let g = compute_grads_vanilla(&st, &e);

            for (tok, d) in [(7usize, 0usize), (7, 3), (9, 5)] {
                let analytic = g.embed.grads.get(&tok).map(|r| r[d]).unwrap_or(0.0);
                let orig = st.model.embedding[tok][d];
                let set = move |s: &mut VanillaModelState, v: f32| {
                    s.model.embedding[tok][d] = v;
                    if s.cfg.tie_embeddings {
                        s.model.sync_tied_head();
                    }
                };
                let (ok, num) = fd_close(&mut st, &e, analytic, &set, orig);
                assert!(
                    ok,
                    "tied={tied} embed[{tok}][{d}] analytic={analytic} numeric={num}"
                );
            }

            let analytic = g.blocks[1].w_q.d_weights[1][2];
            let orig = st.model.blocks[1].attn.w_q.weights[1][2];
            let set =
                |s: &mut VanillaModelState, v: f32| s.model.blocks[1].attn.w_q.weights[1][2] = v;
            let (ok, num) = fd_close(&mut st, &e, analytic, &set, orig);
            assert!(ok, "tied={tied} w_q analytic={analytic} numeric={num}");

            let analytic = g.blocks[0].norm1_gamma[1];
            let orig = st.model.blocks[0].norm1.gamma[1];
            let set = |s: &mut VanillaModelState, v: f32| s.model.blocks[0].norm1.gamma[1] = v;
            let (ok, num) = fd_close(&mut st, &e, analytic, &set, orig);
            assert!(
                ok,
                "tied={tied} norm1.gamma analytic={analytic} numeric={num}"
            );
        }
    }

    #[test]
    fn first_update_uses_warmup_lr_not_lr_max() {
        let mut st = VanillaModelState::new(cfg(false));
        assert!(
            (st.current_lr() - 1e-5).abs() < 1e-9,
            "step 0 must start at lr_min"
        );
        train_step_vanilla_accum(&mut st, &[ex(&[3, 4, 5, 6])]);
        assert_eq!(st.step, 1);
        let expect = cosine_lr_with_warmup(1, 5, 100, 1e-2, 1e-5);
        assert!((st.current_lr() - expect).abs() < 1e-9);
    }

    #[test]
    fn global_clip_bounds_total_norm() {
        let mut c = cfg(false);
        c.grad_clip = 1e-3;
        let st = VanillaModelState::new(c);
        let mut g = compute_grads_batch(&st, &[ex(&[3, 4, 5, 6, 7])]);
        let n = g.sq_norm(&st.cfg).sqrt();
        assert!(n > 1e-3);
        g.scale(1e-3 / n);
        assert!((g.sq_norm(&st.cfg).sqrt() - 1e-3).abs() < 1e-6);
    }

    #[test]
    fn step_loss_is_token_weighted() {
        let st = VanillaModelState::new(cfg(false));
        let long = ex(&[3, 4, 5, 6, 7, 8, 9, 10, 11]);
        let short = ex(&[12, 13, 14]);
        let (sl, nl) = eval_vanilla_nll_sum(&st, &long);
        let (ss, ns) = eval_vanilla_nll_sum(&st, &short);
        let g = compute_grads_batch(&st, &[long, short]);
        assert_eq!(g.n_tokens, nl + ns);
        let want = ((sl + ss) / (nl + ns) as f64) as f32;
        assert!((g.loss / g.n_tokens as f32 - want).abs() < 1e-4);
    }

    #[test]
    fn parallel_batch_matches_serial() {
        let st = VanillaModelState::new(cfg(true));
        let exs: Vec<_> = (0..6)
            .map(|k| ex(&[3 + k, 4 + k, 5 + k, 6 + k, 7 + k]))
            .collect();
        let par = compute_grads_batch(&st, &exs);
        let mut ser = VanillaStepGrads::zeros(&st.cfg);
        for e in &exs {
            ser.add(&compute_grads_vanilla(&st, e));
        }
        assert_eq!(par.n_tokens, ser.n_tokens);
        let d = (par.sq_norm(&st.cfg) - ser.sq_norm(&st.cfg)).abs();
        assert!(d < 1e-3 * ser.sq_norm(&st.cfg).max(1.0));
    }

    #[test]
    fn training_reduces_loss() {
        let mut c = cfg(true);
        c.total_steps = 150;
        let mut st = VanillaModelState::new(c);
        let data = [ex(&[3, 4, 5, 6, 3, 4, 5, 6]), ex(&[7, 8, 9, 7, 8, 9, 7, 8])];
        let before = eval_vanilla_set(&st, &data).0;
        for _ in 0..150 {
            train_step_vanilla_accum(&mut st, &data);
        }
        let after = eval_vanilla_set(&st, &data).0;
        assert!(after < before * 0.5, "loss {before} -> {after}");
    }

    #[test]
    fn early_stopper_patience() {
        let mut es = EarlyStopper::new(2, 0.0);
        assert!(es.observe(100, 3.0));
        assert!(!es.observe(200, 3.1));
        assert!(!es.should_stop());
        assert!(!es.observe(300, 3.0));
        assert!(es.should_stop());
        assert_eq!(es.best_step, 100);
    }
}
