// train_v2.rs — Full end-to-end training loop using the tape
//
// This replaces the manual gradient routing in train.rs with a clean pipeline:
//
//   1. Forward pass produces a Tape with every activation recorded
//   2. Cross-entropy at every loss-masked position produces grad_logits
//   3. Output head backward → grad on head_input
//   4. For each block in reverse: block_backward → grad_input
//   5. Final grad_input is dL/d(post_positional_encoding); for now we treat
//      positional encoding as a fixed function and accumulate that gradient
//      directly into the embedding (an approximation — the rotor sandwich is
//      invertible so the magnitudes are similar)
//   6. Apply Adam to every parameter that received a gradient
//
// All gradients are accumulated across the loss positions and scaled by 1/n_loss.

use super::block_backward::{block_backward, BlockGrads, FfnGrad};
use super::data::TrainExample;
use super::embedding::{EmbeddingGrad, EmbeddingOptimizer};
use super::tape::model_forward_taped;
use crate::backprop::{
    cross_entropy, layer_norm_backward, real_head_backward, GradLinear, RealHeadGrad,
};
use crate::cayley_const::CliffordAlgebraConst;
use crate::ffn::{matched_dense_ffn_hidden, FfnVariant};
use crate::optim::{
    clip_grad_norm, cosine_lr_with_warmup, AdamConfig, LayerOptimizer, RealHeadOptimizer,
};
use crate::CliffordLinear;
use crate::{CliffordAlgebra, CliffordLLM, LinearReal, Multivector};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Arc;

// ─── Random initialisation (symmetry breaking) ───────────────────────────────

/// Fill a Clifford linear layer's weights with uniform noise of the given std
/// (uniform on [-√3·std, √3·std] has variance std²) and zero its biases.
fn fill_linear_random(layer: &mut CliffordLinear, rng: &mut StdRng, std: f32) {
    let bound = std * 3.0f32.sqrt();
    for row in &mut layer.weights {
        for mv in row {
            for k in 0..16 {
                mv.c[k] = rng.gen_range(-bound..bound);
            }
        }
    }
    for b in &mut layer.bias {
        *b = Multivector::zero();
    }
}

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

/// Randomly initialise every learnable parameter of the model.
///
/// Linear layers use fan-in scaling `std = 1/√(fan_in · 16)` (the 16 accounts
/// for the blades mixed by each geometric product); embeddings use a small
/// fixed std; the real head uses `1/√in_features`.  Deterministic given `seed`.
pub fn randomize_model(model: &mut CliffordLLM, seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);

    // Embeddings — small uniform across all blades.
    let emb_bound = 0.02f32 * 3.0f32.sqrt();
    for row in &mut model.embedding {
        for mv in row {
            for k in 0..16 {
                mv.c[k] = rng.gen_range(-emb_bound..emb_bound);
            }
        }
    }

    // Transformer blocks.
    for block in &mut model.blocks {
        let dm = block.attn.d_model;
        let std_dm = (1.0 / (dm * 16) as f32).sqrt();
        fill_linear_random(&mut block.attn.w_q, &mut rng, std_dm);
        fill_linear_random(&mut block.attn.w_k, &mut rng, std_dm);
        fill_linear_random(&mut block.attn.w_v, &mut rng, std_dm);
        fill_linear_random(&mut block.attn.w_o, &mut rng, std_dm);

        match &mut block.ffn {
            FfnVariant::Clifford(f) => {
                let fc1_in = f.fc1.in_dim;
                let fc2_in = f.fc2.in_dim;
                fill_linear_random(&mut f.fc1, &mut rng, (1.0 / (fc1_in * 16) as f32).sqrt());
                fill_linear_random(&mut f.fc2, &mut rng, (1.0 / (fc2_in * 16) as f32).sqrt());
            }
            FfnVariant::Dense(f) => {
                let fan1 = f.fc1.in_features;
                let fan2 = f.fc2.in_features;
                fill_real_linear_random(&mut f.fc1, &mut rng, fan1);
                fill_real_linear_random(&mut f.fc2, &mut rng, fan2);
            }
        }
    }

    // Real output head.
    let head_bound = (1.0 / model.head.in_features as f32).sqrt() * 3.0f32.sqrt();
    for row in &mut model.head.weights {
        for w in row {
            *w = rng.gen_range(-head_bound..head_bound);
        }
    }
    for b in &mut model.head.bias {
        *b = 0.0;
    }
}

/// Deterministic unit-norm Gaussian vector of length `n`, seeded per token.
/// Box–Muller from a seeded stream — reproducible and decorrelated across ids.
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

/// Write a flat `16·d_model` vector into one embedding row's blades.
fn write_row(row: &mut [Multivector], flat: &[f32], scale: f32) {
    let dm = row.len();
    for d in 0..dm {
        for k in 0..16 {
            row[d].c[k] = flat[d * 16 + k] * scale;
        }
    }
}

/// Structured token-embedding initialisation (ported from growformer's
/// `ChunkCodec::build_token_embeddings`).
///
/// Each token id gets a **deterministic, unit-norm Gaussian** vector spread over
/// all `16·d_model` blade components, then scaled by `scale`.  Unlike the tiny
/// uniform init (where every token starts near-identical and near-zero), this
/// gives every token a distinct, well-separated identity from step 0 — and, with
/// weight tying, a usable output classifier from step 0 (each head row is a
/// distinct unit vector).  `scale = 1` makes logits ~unit-variance after the
/// final layer norm.
pub fn structured_embedding_init(model: &mut CliffordLLM, seed: u64, scale: f32) {
    let dm = model.embedding.first().map(|r| r.len()).unwrap_or(0);
    let n = dm * 16;
    for (v, row) in model.embedding.iter_mut().enumerate() {
        let vals = gaussian_unit_vec(token_seed(seed, v), n);
        write_row(row, &vals, scale);
    }
}

/// Corpus-semantic token-embedding initialisation via **random indexing**
/// (the dependency-free distributional method that mirrors growformer's
/// CDMA spread-spectrum + co-occurrence smoothing).
///
/// 1. Each token id gets a random unit-norm *index* vector (its identity).
/// 2. For every occurrence in `tokens`, the index vectors of its ±`window`
///    neighbours are added (distance-weighted) into the token's *context*
///    vector.  Tokens that share contexts therefore end up with similar
///    context vectors — the distributional hypothesis, in one pass.
/// 3. A `self_anchor` keeps each token distinct (and is the fallback for
///    tokens absent from the corpus); the result is L2-normalised and scaled.
///
/// This gives the embedding table a genuine semantic prior (unlike the random
/// `structured_embedding_init`), which is where growformer's encoder gets its
/// training-quality edge.
pub fn corpus_semantic_init(
    model: &mut CliffordLLM,
    tokens: &[u32],
    seed: u64,
    window: usize,
    scale: f32,
) {
    let vocab = model.embedding.len();
    let dm = model.embedding.first().map(|r| r.len()).unwrap_or(0);
    let n = dm * 16;
    if vocab == 0 || n == 0 {
        return;
    }

    // 1. Random index vectors (token identities).
    let idx: Vec<Vec<f32>> = (0..vocab)
        .map(|v| gaussian_unit_vec(token_seed(seed, v), n))
        .collect();

    // 2. Co-occurrence accumulation into context vectors.
    let mut ctx = vec![vec![0.0f32; n]; vocab];
    let len = tokens.len();
    for i in 0..len {
        let t = tokens[i] as usize;
        if t >= vocab {
            continue;
        }
        for off in 1..=window {
            let w = 1.0 / off as f32; // closer neighbours weigh more
            if i >= off {
                let nb = tokens[i - off] as usize;
                if nb < vocab {
                    let (src, dst) = (&idx[nb], &mut ctx[t]);
                    for k in 0..n {
                        dst[k] += w * src[k];
                    }
                }
            }
            if i + off < len {
                let nb = tokens[i + off] as usize;
                if nb < vocab {
                    let (src, dst) = (&idx[nb], &mut ctx[t]);
                    for k in 0..n {
                        dst[k] += w * src[k];
                    }
                }
            }
        }
    }

    // 3. Self anchor + normalise + write.
    const SELF_ANCHOR: f32 = 1.0;
    for v in 0..vocab {
        let mut c = std::mem::take(&mut ctx[v]);
        for k in 0..n {
            c[k] += SELF_ANCHOR * idx[v][k];
        }
        let norm: f32 = c.iter().map(|x| x * x).sum::<f32>().sqrt();
        let s = if norm > 1e-8 { scale / norm } else { 0.0 };
        write_row(&mut model.embedding[v], &c, s);
    }
}

// ─── Training configuration (carried in lm_config; re-exported for compat) ───

pub use crate::lm_config::TrainConfigV2;

// ─── Model + optimiser state ─────────────────────────────────────────────────

/// Adam state for one block's FFN (Clifford or dense param-matched).
pub enum FfnBlockOptimizer {
    Clifford {
        fc1: LayerOptimizer,
        fc2: LayerOptimizer,
    },
    Dense {
        fc1: RealHeadOptimizer,
        fc2: RealHeadOptimizer,
    },
}

impl FfnBlockOptimizer {
    fn step_clifford(&mut self, f: &mut crate::CliffordFFN, g1: &GradLinear, g2: &GradLinear) {
        let Self::Clifford { fc1, fc2 } = self else {
            panic!("FFN optimizer/variant mismatch");
        };
        fc1.step(&mut f.fc1.weights, &mut f.fc1.bias, g1);
        fc2.step(&mut f.fc2.weights, &mut f.fc2.bias, g2);
    }

    fn step_dense(&mut self, f: &mut crate::DenseFFN, g1: &RealHeadGrad, g2: &RealHeadGrad) {
        let Self::Dense { fc1, fc2 } = self else {
            panic!("FFN optimizer/variant mismatch");
        };
        fc1.step(&mut f.fc1, g1);
        fc2.step(&mut f.fc2, g2);
    }
}

/// Per-block optimiser bundle.
pub struct BlockOptimizer {
    pub wq: LayerOptimizer,
    pub wk: LayerOptimizer,
    pub wv: LayerOptimizer,
    pub wo: LayerOptimizer,
    pub ffn: FfnBlockOptimizer,
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

impl BlockOptimizer {
    pub fn new(cfg: &TrainConfigV2, adam: AdamConfig) -> Self {
        let n = cfg.d_model * 16;
        let ffn = if cfg.dense_ffn {
            let hidden = matched_dense_ffn_hidden(cfg.d_model, cfg.d_ff);
            FfnBlockOptimizer::Dense {
                fc1: RealHeadOptimizer::new(hidden, n, adam.clone()),
                fc2: RealHeadOptimizer::new(n, hidden, adam.clone()),
            }
        } else {
            FfnBlockOptimizer::Clifford {
                fc1: LayerOptimizer::new(cfg.d_ff, cfg.d_model, adam.clone()),
                fc2: LayerOptimizer::new(cfg.d_model, cfg.d_ff, adam.clone()),
            }
        };
        Self {
            wq: LayerOptimizer::new(cfg.d_model, cfg.d_model, adam.clone()),
            wk: LayerOptimizer::new(cfg.d_model, cfg.d_model, adam.clone()),
            wv: LayerOptimizer::new(cfg.d_model, cfg.d_model, adam.clone()),
            wo: LayerOptimizer::new(cfg.d_model, cfg.d_model, adam.clone()),
            ffn,
            norm1_gamma_m: vec![0.0; n],
            norm1_gamma_v: vec![0.0; n],
            norm1_beta_m: vec![0.0; n],
            norm1_beta_v: vec![0.0; n],
            norm2_gamma_m: vec![0.0; n],
            norm2_gamma_v: vec![0.0; n],
            norm2_beta_m: vec![0.0; n],
            norm2_beta_v: vec![0.0; n],
            step: 0,
        }
    }
}

/// Adam step for a scalar parameter vector (used for layer-norm γ, β).
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

pub struct ModelStateV2 {
    pub model: CliffordLLM,
    pub alg: CliffordAlgebraConst,
    pub block_opts: Vec<BlockOptimizer>,
    pub head_opt: RealHeadOptimizer,
    pub embed_opt: EmbeddingOptimizer,
    // Adam state for the final layer-norm γ/β (scalar params, len 16·d_model).
    pub fnorm_gamma_m: Vec<f32>,
    pub fnorm_gamma_v: Vec<f32>,
    pub fnorm_beta_m: Vec<f32>,
    pub fnorm_beta_v: Vec<f32>,
    pub fnorm_step: u64,
    pub cfg: TrainConfigV2,
    pub step: u64,
}

impl ModelStateV2 {
    pub fn new(cfg: TrainConfigV2) -> Self {
        let alg_arc = Arc::new(CliffordAlgebra::sta());
        let alg = CliffordAlgebraConst::new();
        let adam = AdamConfig {
            lr: cfg.lr_max,
            ..Default::default()
        };

        let blocks: Vec<_> = (0..cfg.n_blocks)
            .map(|_| crate::CliffordBlock {
                attn: crate::CliffordAttention::new(cfg.d_model, cfg.n_heads, alg_arc.clone()),
                ffn: if cfg.dense_ffn {
                    FfnVariant::dense_matched(cfg.d_model, cfg.d_ff, cfg.init_seed ^ 0xDE5EFF10)
                } else {
                    FfnVariant::clifford(cfg.d_model, cfg.d_ff, alg_arc.clone())
                },
                norm1: crate::CliffordLayerNorm::new(cfg.d_model),
                norm2: crate::CliffordLayerNorm::new(cfg.d_model),
            })
            .collect();

        let block_opts: Vec<_> = (0..cfg.n_blocks)
            .map(|_| BlockOptimizer::new(&cfg, adam.clone()))
            .collect();

        let head = LinearReal::new(cfg.d_model, cfg.vocab_size);
        let head_opt = RealHeadOptimizer::new(cfg.vocab_size, cfg.d_model * 16, adam.clone());
        let embed_opt = EmbeddingOptimizer::new(cfg.d_model, adam);

        let embedding = vec![vec![Multivector::scalar(0.01); cfg.d_model]; cfg.vocab_size];

        let mut model = CliffordLLM {
            embedding,
            blocks,
            final_norm: crate::CliffordLayerNorm::new(cfg.d_model),
            head,
            algebra: alg_arc,
        };
        // Symmetry breaking: without random init every output channel and every
        // token embedding is identical, the body receives zero gradient, and the
        // model cannot represent token identity.  Seeded for reproducibility.
        randomize_model(&mut model, cfg.init_seed);
        if cfg.structured_init {
            structured_embedding_init(&mut model, cfg.init_seed ^ 0xE8E8_E8E8, 1.0);
        }
        if cfg.tie_embeddings {
            model.sync_tied_head();
        }

        let n = cfg.d_model * 16;
        Self {
            model,
            alg,
            block_opts,
            head_opt,
            embed_opt,
            fnorm_gamma_m: vec![0.0; n],
            fnorm_gamma_v: vec![0.0; n],
            fnorm_beta_m: vec![0.0; n],
            fnorm_beta_v: vec![0.0; n],
            fnorm_step: 0,
            cfg,
            step: 0,
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
            match &mut b.ffn {
                FfnBlockOptimizer::Clifford { fc1, fc2 } => {
                    fc1.cfg.lr = lr;
                    fc2.cfg.lr = lr;
                }
                FfnBlockOptimizer::Dense { fc1, fc2 } => {
                    fc1.cfg.lr = lr;
                    fc2.cfg.lr = lr;
                }
            }
        }
        self.head_opt.cfg.lr = lr;
        self.embed_opt.cfg.lr = lr;
    }
}

// ─── One training step ────────────────────────────────────────────────────────

/// FFN parameter gradients for one block (variant-specific).
pub enum FfnParamGrads {
    Clifford {
        fc1: GradLinear,
        fc2: GradLinear,
    },
    Dense {
        fc1: RealHeadGrad,
        fc2: RealHeadGrad,
    },
}

impl FfnParamGrads {
    fn zeros(cfg: &TrainConfigV2) -> Self {
        if cfg.dense_ffn {
            let hidden = matched_dense_ffn_hidden(cfg.d_model, cfg.d_ff);
            let n = cfg.d_model * 16;
            Self::Dense {
                fc1: RealHeadGrad::zeros(hidden, n),
                fc2: RealHeadGrad::zeros(n, hidden),
            }
        } else {
            Self::Clifford {
                fc1: GradLinear::zeros(cfg.d_ff, cfg.d_model),
                fc2: GradLinear::zeros(cfg.d_model, cfg.d_ff),
            }
        }
    }

    fn add(&mut self, o: &FfnParamGrads) {
        match (self, o) {
            (Self::Clifford { fc1, fc2 }, Self::Clifford { fc1: o1, fc2: o2 }) => {
                fc1.accumulate(o1);
                fc2.accumulate(o2);
            }
            (Self::Dense { fc1, fc2 }, Self::Dense { fc1: o1, fc2: o2 }) => {
                fc1.accumulate(o1);
                fc2.accumulate(o2);
            }
            _ => panic!("FFN param grad variant mismatch"),
        }
    }

    fn scale(&mut self, s: f32) {
        match self {
            Self::Clifford { fc1, fc2 } => {
                fc1.scale(s);
                fc2.scale(s);
            }
            Self::Dense { fc1, fc2 } => {
                fc1.scale(s);
                fc2.scale(s);
            }
        }
    }
}

/// Parameter gradients for one transformer block (no `grad_input`, so it can be
/// summed across microbatches of differing sequence length).
pub struct BlockParamGrads {
    pub norm1_gamma: Vec<f32>,
    pub norm1_beta: Vec<f32>,
    pub norm2_gamma: Vec<f32>,
    pub norm2_beta: Vec<f32>,
    pub w_q: GradLinear,
    pub w_k: GradLinear,
    pub w_v: GradLinear,
    pub w_o: GradLinear,
    pub ffn: FfnParamGrads,
}

impl BlockParamGrads {
    fn zeros(cfg: &TrainConfigV2) -> Self {
        let n = cfg.d_model * 16;
        Self {
            norm1_gamma: vec![0.0; n],
            norm1_beta: vec![0.0; n],
            norm2_gamma: vec![0.0; n],
            norm2_beta: vec![0.0; n],
            w_q: GradLinear::zeros(cfg.d_model, cfg.d_model),
            w_k: GradLinear::zeros(cfg.d_model, cfg.d_model),
            w_v: GradLinear::zeros(cfg.d_model, cfg.d_model),
            w_o: GradLinear::zeros(cfg.d_model, cfg.d_model),
            ffn: FfnParamGrads::zeros(cfg),
        }
    }

    fn from_block(g: BlockGrads) -> Self {
        let ffn = match g.ffn {
            FfnGrad::Clifford(g1, g2) => FfnParamGrads::Clifford { fc1: g1, fc2: g2 },
            FfnGrad::Dense(g1, g2) => FfnParamGrads::Dense { fc1: g1, fc2: g2 },
        };
        Self {
            norm1_gamma: g.norm1_gamma,
            norm1_beta: g.norm1_beta,
            norm2_gamma: g.norm2_gamma,
            norm2_beta: g.norm2_beta,
            w_q: g.attn.w_q,
            w_k: g.attn.w_k,
            w_v: g.attn.w_v,
            w_o: g.attn.w_o,
            ffn,
        }
    }

    fn add(&mut self, o: &BlockParamGrads) {
        for i in 0..self.norm1_gamma.len() {
            self.norm1_gamma[i] += o.norm1_gamma[i];
            self.norm1_beta[i] += o.norm1_beta[i];
            self.norm2_gamma[i] += o.norm2_gamma[i];
            self.norm2_beta[i] += o.norm2_beta[i];
        }
        self.w_q.accumulate(&o.w_q);
        self.w_k.accumulate(&o.w_k);
        self.w_v.accumulate(&o.w_v);
        self.w_o.accumulate(&o.w_o);
        self.ffn.add(&o.ffn);
    }

    fn scale(&mut self, s: f32) {
        for v in self.norm1_gamma.iter_mut() {
            *v *= s;
        }
        for v in self.norm1_beta.iter_mut() {
            *v *= s;
        }
        for v in self.norm2_gamma.iter_mut() {
            *v *= s;
        }
        for v in self.norm2_beta.iter_mut() {
            *v *= s;
        }
        self.w_q.scale(s);
        self.w_k.scale(s);
        self.w_v.scale(s);
        self.w_o.scale(s);
        self.ffn.scale(s);
    }
}

/// All parameter gradients produced by one (or several, accumulated) microbatch
/// passes — decoupled from optimiser state so they can be averaged before the
/// single Adam step that gradient accumulation requires.
pub struct StepGrads {
    pub head: RealHeadGrad,
    pub fnorm_dgamma: Vec<f32>,
    pub fnorm_dbeta: Vec<f32>,
    pub blocks: Vec<BlockParamGrads>,
    pub embed: EmbeddingGrad,
    pub loss: f32,
    pub valid: bool,
}

impl StepGrads {
    fn zeros(cfg: &TrainConfigV2) -> Self {
        let n = cfg.d_model * 16;
        Self {
            head: RealHeadGrad::zeros(cfg.vocab_size, cfg.d_model * 16),
            fnorm_dgamma: vec![0.0; n],
            fnorm_dbeta: vec![0.0; n],
            blocks: (0..cfg.n_blocks)
                .map(|_| BlockParamGrads::zeros(cfg))
                .collect(),
            embed: EmbeddingGrad::new(cfg.d_model),
            loss: 0.0,
            valid: false,
        }
    }

    /// Accumulate another microbatch's grads (sum).
    fn add(&mut self, o: &StepGrads) {
        self.head.accumulate(&o.head);
        for i in 0..self.fnorm_dgamma.len() {
            self.fnorm_dgamma[i] += o.fnorm_dgamma[i];
            self.fnorm_dbeta[i] += o.fnorm_dbeta[i];
        }
        for b in 0..self.blocks.len() {
            self.blocks[b].add(&o.blocks[b]);
        }
        self.embed.merge(&o.embed);
        self.loss += o.loss;
    }

    /// Average over `n` accumulated microbatches.
    fn scale(&mut self, s: f32) {
        self.head.scale(s);
        for v in self.fnorm_dgamma.iter_mut() {
            *v *= s;
        }
        for v in self.fnorm_dbeta.iter_mut() {
            *v *= s;
        }
        for b in &mut self.blocks {
            b.scale(s);
        }
        self.embed.scale(s);
        self.loss *= s;
    }
}

/// Pure gradient computation for a single microbatch — does **not** touch
/// optimiser state or model parameters.  Gradients are scaled by `1/n_loss`
/// (per-token mean) and the head/block linears are clipped, matching the
/// single-step path.  Returns `valid = false` when the example has no loss
/// positions.
pub fn compute_grads_v2(state: &ModelStateV2, example: &TrainExample) -> StepGrads {
    let seq = example.len();
    let dm = state.cfg.d_model;
    let vocab = state.cfg.vocab_size;
    let mut out = StepGrads::zeros(&state.cfg);

    // ── 1. Forward with tape ─────────────────────────────────────────────────
    let tape = model_forward_taped(
        &state.alg,
        &state.model,
        &example.full_ids,
        true,
        state.cfg.dot_attention,
    );

    // ── 2. Loss + grad_logits at every loss-masked position ───────────────────
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

    // ── 3. Output head backward (real projection) ────────────────────────────
    let mut grad_head = out.head; // reuse the pre-zeroed allocation
    let mut grad_x_final = vec![vec![Multivector::zero(); dm]; seq];

    for t in 0..seq {
        if grad_logits[t].iter().all(|&g| g == 0.0) {
            continue;
        }
        let gx = real_head_backward(
            &state.model.head.weights,
            &tape.head_input[t],
            &grad_logits[t],
            &mut grad_head,
        );
        for d in 0..dm {
            for k in 0..16 {
                grad_x_final[t][d].c[k] += gx[d].c[k];
            }
        }
    }
    grad_head.scale(scale);
    grad_head.clip_norm(state.cfg.grad_clip);

    // ── 3b. Final layer-norm backward ────────────────────────────────────────
    let n_comp = dm * 16;
    let mut fnorm_dgamma = vec![0.0f32; n_comp];
    let mut fnorm_dbeta = vec![0.0f32; n_comp];
    let mut grad_x = vec![vec![Multivector::zero(); dm]; seq];
    for t in 0..seq {
        let g_flat: Vec<f32> = grad_x_final[t].iter().flat_map(|mv| mv.c).collect();
        if g_flat.iter().all(|&g| g == 0.0) {
            continue;
        }
        let stats = &tape.final_norm_stats[t];
        for k in 0..n_comp {
            fnorm_dgamma[k] += g_flat[k] * stats.x_hat[k];
            fnorm_dbeta[k] += g_flat[k];
        }
        let g_x = layer_norm_backward(
            &stats.x_hat,
            &state.model.final_norm.gamma,
            &g_flat,
            stats.std,
        );
        for d in 0..dm {
            for k in 0..16 {
                grad_x[t][d].c[k] = g_x[d * 16 + k];
            }
        }
    }
    for g in &mut fnorm_dgamma {
        *g *= scale;
    }
    for g in &mut fnorm_dbeta {
        *g *= scale;
    }

    // ── 4. Each block backward in reverse order ──────────────────────────────
    for b in (0..state.cfg.n_blocks).rev() {
        let block = &state.model.blocks[b];
        let block_tape = &tape.blocks[b];

        let mut grads = block_backward(block, block_tape, &grad_x);

        grads.attn.w_q.scale(scale);
        grads.attn.w_k.scale(scale);
        grads.attn.w_v.scale(scale);
        grads.attn.w_o.scale(scale);
        grads.ffn.scale(scale);
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

        // Clip linear gradients (γ/β are small and rarely need clipping).  We
        // clip for every block; frozen ones simply never get applied.
        clip_grad_norm(&mut grads.attn.w_q, state.cfg.grad_clip);
        clip_grad_norm(&mut grads.attn.w_k, state.cfg.grad_clip);
        clip_grad_norm(&mut grads.attn.w_v, state.cfg.grad_clip);
        clip_grad_norm(&mut grads.attn.w_o, state.cfg.grad_clip);
        grads.ffn.clip_norm(state.cfg.grad_clip);

        grad_x = std::mem::take(&mut grads.grad_input);
        out.blocks[b] = BlockParamGrads::from_block(grads);
    }

    // ── 5. Embedding gradient (sparse) ───────────────────────────────────────
    let tied = state.cfg.tie_embeddings;
    let mut embed_grad = EmbeddingGrad::new(dm);
    if state.cfg.train_embeddings && !state.cfg.freeze_embeddings {
        for t in 0..seq {
            embed_grad.accumulate(example.full_ids[t], &grad_x[t]);
        }
        embed_grad.scale(scale); // input-lookup path

        if tied {
            // Output-projection path: dL/d(embedding[v]) += grad_head.d_weights[v]
            // (already scaled + clipped).  Reshape 16·dm reals → dm multivectors.
            for v in 0..vocab {
                let row = &grad_head.d_weights[v];
                if row.iter().all(|&g| g == 0.0) {
                    continue;
                }
                let mut mvs = vec![Multivector::zero(); dm];
                for d in 0..dm {
                    for k in 0..16 {
                        mvs[d].c[k] = row[d * 16 + k];
                    }
                }
                embed_grad.accumulate(v, &mvs);
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

/// Apply one (already-averaged) set of gradients to the model via Adam.
/// Advances every optimiser clock exactly once.  Does **not** touch
/// `state.step`/`update_lr` — the caller owns the LR schedule.
fn apply_grads_v2(state: &mut ModelStateV2, grads: &StepGrads) {
    let tied = state.cfg.tie_embeddings;

    // ── Final layer-norm ─────────────────────────────────────────────────────
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

    // ── Blocks (frozen ones are skipped; their Adam clock does not advance) ───
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
        opt.wq
            .step(&mut bm.attn.w_q.weights, &mut bm.attn.w_q.bias, &pg.w_q);
        opt.wk
            .step(&mut bm.attn.w_k.weights, &mut bm.attn.w_k.bias, &pg.w_k);
        opt.wv
            .step(&mut bm.attn.w_v.weights, &mut bm.attn.w_v.bias, &pg.w_v);
        opt.wo
            .step(&mut bm.attn.w_o.weights, &mut bm.attn.w_o.bias, &pg.w_o);
        match (&mut bm.ffn, &mut opt.ffn, &pg.ffn) {
            (
                FfnVariant::Clifford(f),
                FfnBlockOptimizer::Clifford { .. },
                FfnParamGrads::Clifford { fc1, fc2 },
            ) => {
                opt.ffn.step_clifford(f, fc1, fc2);
            }
            (
                FfnVariant::Dense(f),
                FfnBlockOptimizer::Dense { .. },
                FfnParamGrads::Dense { fc1, fc2 },
            ) => {
                opt.ffn.step_dense(f, fc1, fc2);
            }
            _ => panic!("FFN variant / optimizer / grad mismatch at block {b}"),
        }

        adam_step_scalar(
            &mut state.model.blocks[b].norm1.gamma,
            &pg.norm1_gamma,
            &mut opt.norm1_gamma_m,
            &mut opt.norm1_gamma_v,
            lr_step,
            &cfg,
        );
        adam_step_scalar(
            &mut state.model.blocks[b].norm1.beta,
            &pg.norm1_beta,
            &mut opt.norm1_beta_m,
            &mut opt.norm1_beta_v,
            lr_step,
            &cfg,
        );
        adam_step_scalar(
            &mut state.model.blocks[b].norm2.gamma,
            &pg.norm2_gamma,
            &mut opt.norm2_gamma_m,
            &mut opt.norm2_gamma_v,
            lr_step,
            &cfg,
        );
        adam_step_scalar(
            &mut state.model.blocks[b].norm2.beta,
            &pg.norm2_beta,
            &mut opt.norm2_beta_m,
            &mut opt.norm2_beta_v,
            lr_step,
            &cfg,
        );
    }

    // ── Head ─────────────────────────────────────────────────────────────────
    if tied {
        state
            .head_opt
            .step_bias_only(&mut state.model.head, &grads.head);
    } else {
        state.head_opt.step(&mut state.model.head, &grads.head);
    }

    // ── Embeddings (sparse) ──────────────────────────────────────────────────
    if state.cfg.train_embeddings && !state.cfg.freeze_embeddings {
        state
            .embed_opt
            .step(&mut state.model.embedding, &grads.embed);
    }

    // Keep the head weight mirror consistent with the updated shared embedding.
    if tied {
        state.model.sync_tied_head();
    }
}

/// One optimiser step from a single microbatch.
pub fn train_step_v2(state: &mut ModelStateV2, example: &TrainExample) -> f32 {
    let grads = compute_grads_v2(state, example);
    if !grads.valid {
        return 0.0;
    }
    apply_grads_v2(state, &grads);
    state.step += 1;
    state.update_lr();
    grads.loss
}

/// One optimiser step from `examples.len()` microbatches whose gradients are
/// averaged (effective batch size = number of valid microbatches).  This is the
/// gradient-accumulation entry point: it reduces step-to-step gradient noise
/// without the memory cost of a true batched forward pass.  Returns the mean
/// loss over the valid microbatches.
pub fn train_step_v2_accum(state: &mut ModelStateV2, examples: &[TrainExample]) -> f32 {
    if examples.is_empty() {
        return 0.0;
    }
    let mut acc = StepGrads::zeros(&state.cfg);
    let mut n_valid = 0usize;
    for ex in examples {
        let g = compute_grads_v2(state, ex);
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
    apply_grads_v2(state, &acc);
    state.step += 1;
    state.update_lr();
    acc.loss
}

/// Single optimizer step on the **output head only** — one taped forward over the sequence,
/// cross-entropy masked positions, backward **through `head` only**.
///
/// Blocks and embeddings are not updated (no attention/FFN backward), so this is much cheaper
/// than [`train_step_v2`] and closer in spirit to a shallow “one forward + short backward” tick.
/// The transformer body stays frozen; loss can still move while the head adapts to frozen features.
pub fn train_step_v2_head_only(state: &mut ModelStateV2, example: &TrainExample) -> f32 {
    let seq = example.len();
    let dm = state.cfg.d_model;
    let vocab = state.cfg.vocab_size;

    let tape = model_forward_taped(
        &state.alg,
        &state.model,
        &example.full_ids,
        true,
        state.cfg.dot_attention,
    );

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
        return 0.0;
    }
    total_loss /= n_loss as f32;
    let scale = 1.0 / n_loss as f32;

    let mut grad_head = RealHeadGrad::zeros(vocab, dm * 16);

    for t in 0..seq {
        if grad_logits[t].iter().all(|&g| g == 0.0) {
            continue;
        }
        let _ = real_head_backward(
            &state.model.head.weights,
            &tape.head_input[t],
            &grad_logits[t],
            &mut grad_head,
        );
    }
    grad_head.scale(scale);
    grad_head.clip_norm(state.cfg.grad_clip);

    state.head_opt.step(&mut state.model.head, &grad_head);

    state.step += 1;
    state.update_lr();
    total_loss
}

// ─── Public training loop ────────────────────────────────────────────────────

pub fn train_v2(dataset: &super::data::Dataset, state: &mut ModelStateV2) {
    if dataset.train.is_empty() {
        eprintln!("[train_v2] no training examples");
        return;
    }

    let cfg = state.cfg.clone();
    let mut step = 0usize;
    let mut running = 0.0f32;

    for epoch in 0..cfg.epochs {
        let shuffled = dataset.shuffled_train(epoch as u64 + 7);

        for ex in &shuffled {
            let loss = train_step_v2(state, ex);
            running += loss;
            step += 1;

            if step % cfg.log_every == 0 {
                eprintln!(
                    "[train_v2] epoch={} step={} loss={:.4} lr={:.2e}",
                    epoch + 1,
                    step,
                    running / cfg.log_every as f32,
                    cosine_lr_with_warmup(
                        state.step,
                        cfg.warmup_steps,
                        cfg.total_steps,
                        cfg.lr_max,
                        cfg.lr_min,
                    ),
                );
                running = 0.0;
            }
        }
        eprintln!("[train_v2] epoch {} complete ({} steps)", epoch + 1, step);
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::data::{encode_record, RawRecord, Tokenizer};

    fn tiny(text: &str, resp: &str) -> RawRecord {
        RawRecord {
            task_id: "t".into(),
            text: text.into(),
            semantic_intent: "p".into(),
            domain: "d".into(),
            action_target: "a".into(),
            policy_regime: "r".into(),
            language_channel: "english".into(),
            code_language: None,
            split: "train".into(),
            expected_response: resp.into(),
            expected_code: None,
        }
    }

    #[test]
    fn end_to_end_loss_decreases() {
        // Single example, many steps — full-graph Adam should beat CE noticeably.
        let recs = vec![tiny("a b c", "x y z")];
        let mut tok = Tokenizer::new();
        tok.fit(&recs);
        let ex = encode_record(&recs[0], &tok, 32).unwrap();

        let mut cfg = TrainConfigV2::small(tok.vocab_size());
        cfg.warmup_steps = 0;
        cfg.lr_max = 2e-2;
        cfg.lr_min = 1e-4;
        cfg.total_steps = 50_000;
        let mut state = ModelStateV2::new(cfg);
        let vs = tok.vocab_size();
        let dm = state.cfg.d_model;
        state.model.embedding = vec![vec![Multivector::scalar(0.01); dm]; vs];

        let initial = train_step_v2(&mut state, &ex);
        for _ in 0..120 {
            train_step_v2(&mut state, &ex);
        }
        let after = train_step_v2(&mut state, &ex);

        assert!(
            after < initial * 0.75,
            "loss should drop: initial={initial:.4} after={after:.4}"
        );
    }
}
