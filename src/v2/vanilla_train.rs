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
    add_sinusoidal_pe, vanilla_forward_logits, VanillaAttention, VanillaBlock, VanillaFFN,
    VanillaLLM,
};

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
        let d1 = d0 + attn.head_dim;
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
    let mut x: Vec<Vec<f32>> = ids.iter().map(|&id| model.embedding[id].clone()).collect();
    add_sinusoidal_pe(&mut x);
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

    pub fn merge(&mut self, other: &VanillaEmbeddingGrad) {
        for (&tid, grad) in &other.grads {
            self.accumulate(tid, grad);
        }
    }
}

struct VanillaEmbedAdamState {
    m: Vec<f32>,
    v: Vec<f32>,
    step: u64,
}

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
                let g = token_grad[d] + self.cfg.weight_decay * embedding[token_id][d];
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
    for i in 0..params.len() {
        let g = grads[i] + cfg.weight_decay * params[i];
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
}

impl VanillaModelState {
    pub fn new(cfg: TrainConfigV2) -> Self {
        let adam = AdamConfig {
            lr: cfg.lr_max,
            ..Default::default()
        };
        let block_opts: Vec<_> = (0..cfg.n_blocks)
            .map(|_| VanillaBlockOptimizer::new(&cfg, adam.clone()))
            .collect();
        let head_opt = RealHeadOptimizer::new(cfg.vocab_size, cfg.d_model, adam.clone());
        let embed_opt = VanillaEmbeddingOptimizer::new(cfg.d_model, adam);
        let mut model = VanillaLLM::new(
            cfg.vocab_size,
            cfg.d_model,
            cfg.n_heads,
            cfg.d_ff,
            cfg.n_blocks,
            cfg.init_seed,
        );
        randomize_vanilla_model(&mut model, cfg.init_seed);
        if cfg.tie_embeddings {
            model.sync_tied_head();
        }
        let dm = cfg.d_model;
        Self {
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
        }
    }

    pub fn from_loaded(cfg: TrainConfigV2, model: VanillaLLM, step: u64) -> Self {
        let adam = AdamConfig {
            lr: cfg.lr_max,
            ..Default::default()
        };
        let block_opts: Vec<_> = (0..cfg.n_blocks)
            .map(|_| VanillaBlockOptimizer::new(&cfg, adam.clone()))
            .collect();
        let head_opt = RealHeadOptimizer::new(cfg.vocab_size, cfg.d_model, adam.clone());
        let embed_opt = VanillaEmbeddingOptimizer::new(cfg.d_model, adam);
        let dm = cfg.d_model;
        Self {
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
        }
    }

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
    loss: f32,
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
            valid: false,
        }
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
        self.loss *= s;
    }
}

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
    total_loss /= n_loss as f32;
    let scale = 1.0 / n_loss as f32;

    let mut grad_head = out.head;
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
    grad_head.scale(scale);
    grad_head.clip_norm(state.cfg.grad_clip);

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
    for g in &mut fnorm_dgamma {
        *g *= scale;
    }
    for g in &mut fnorm_dbeta {
        *g *= scale;
    }

    let mut block_grads = Vec::with_capacity(state.cfg.n_blocks);
    for b in (0..state.cfg.n_blocks).rev() {
        let block = &state.model.blocks[b];
        let block_tape = &tape.blocks[b];
        let mut grads = block_backward(block, block_tape, &grad_x);

        grads.w_q.scale(scale);
        grads.w_k.scale(scale);
        grads.w_v.scale(scale);
        grads.w_o.scale(scale);
        grads.fc1.scale(scale);
        grads.fc2.scale(scale);
        for g in &mut grads.norm1_gamma {
            *g *= scale;
        }
        for g in &mut grads.norm1_beta {
            *g *= scale;
        }
        for g in &mut grads.norm2_gamma {
            *g *= scale;
        }
        for g in &mut grads.norm2_beta {
            *g *= scale;
        }

        grads.w_q.clip_norm(state.cfg.grad_clip);
        grads.w_k.clip_norm(state.cfg.grad_clip);
        grads.w_v.clip_norm(state.cfg.grad_clip);
        grads.w_o.clip_norm(state.cfg.grad_clip);
        grads.fc1.clip_norm(state.cfg.grad_clip);
        grads.fc2.clip_norm(state.cfg.grad_clip);

        grad_x = std::mem::take(&mut grads.grad_input);
        block_grads.push((b, grads));
    }
    block_grads.reverse();
    for (i, (_, g)) in block_grads.into_iter().enumerate() {
        out.blocks[i] = g;
    }

    let tied = state.cfg.tie_embeddings;
    let mut embed_grad = VanillaEmbeddingGrad::new(dm);
    if state.cfg.train_embeddings && !state.cfg.freeze_embeddings {
        for t in 0..seq {
            embed_grad.accumulate(example.full_ids[t], &grad_x[t]);
        }
        embed_grad.scale(scale);
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
    out.valid = true;
    out
}

fn apply_grads_vanilla(state: &mut VanillaModelState, grads: &VanillaStepGrads) {
    let tied = state.cfg.tie_embeddings;

    state.fnorm_step += 1;
    let fnorm_lr = state.head_opt.cfg.lr;
    let fcfg = AdamConfig {
        lr: fnorm_lr,
        ..Default::default()
    };
    adam_step_scalar(
        &mut state.model.final_norm.gamma,
        &grads.fnorm_dgamma,
        &mut state.fnorm_gamma_m,
        &mut state.fnorm_gamma_v,
        state.fnorm_step,
        &fcfg,
    );
    adam_step_scalar(
        &mut state.model.final_norm.beta,
        &grads.fnorm_dbeta,
        &mut state.fnorm_beta_m,
        &mut state.fnorm_beta_v,
        state.fnorm_step,
        &fcfg,
    );

    for b in 0..state.cfg.n_blocks {
        if b < state.cfg.freeze_blocks {
            continue;
        }
        let pg = &grads.blocks[b];
        let opt = &mut state.block_opts[b];
        opt.step += 1;
        let lr_step = opt.step;
        let cfg = opt.wq.cfg.clone();
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
            &cfg,
        );
        adam_step_scalar(
            &mut bm.norm1.beta,
            &pg.norm1_beta,
            &mut opt.norm1_beta_m,
            &mut opt.norm1_beta_v,
            lr_step,
            &cfg,
        );
        adam_step_scalar(
            &mut bm.norm2.gamma,
            &pg.norm2_gamma,
            &mut opt.norm2_gamma_m,
            &mut opt.norm2_gamma_v,
            lr_step,
            &cfg,
        );
        adam_step_scalar(
            &mut bm.norm2.beta,
            &pg.norm2_beta,
            &mut opt.norm2_beta_m,
            &mut opt.norm2_beta_v,
            lr_step,
            &cfg,
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

pub fn train_step_vanilla_accum(state: &mut VanillaModelState, examples: &[TrainExample]) -> f32 {
    if examples.is_empty() {
        return 0.0;
    }
    let mut acc = VanillaStepGrads::zeros(&state.cfg);
    let mut n_valid = 0usize;
    for ex in examples {
        let g = compute_grads_vanilla(state, ex);
        if !g.valid {
            continue;
        }
        acc.add(&g);
        n_valid += 1;
    }
    if n_valid == 0 {
        return 0.0;
    }
    acc.scale(1.0 / n_valid as f32);
    apply_grads_vanilla(state, &acc);
    state.step += 1;
    state.update_lr();
    acc.loss
}

pub fn eval_vanilla_lm_loss(state: &VanillaModelState, ex: &TrainExample) -> f32 {
    let logits = vanilla_forward_logits(&state.model, &ex.full_ids, true);
    let mask = ex.loss_mask();
    let mut total = 0.0f32;
    let mut n = 0usize;
    for t in 0..ex.len() {
        if !mask[t] || t + 1 >= ex.len() {
            continue;
        }
        let (loss, _) = cross_entropy(&logits[t], ex.full_ids[t + 1]);
        total += loss;
        n += 1;
    }
    if n == 0 {
        0.0
    } else {
        total / n as f32
    }
}
