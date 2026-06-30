// clifford_llm.rs — Sketch of a Clifford algebra LLM using STA (Cl(1,3))
//
// Algebra: Cl(1,3), signature (+,-,-,-)
// Basis blades (16 total):
//   grade 0: [1]
//   grade 1: [e0, e1, e2, e3]
//   grade 2: [e01, e02, e03, e12, e13, e23]
//   grade 3: [e012, e013, e023, e123]
//   grade 4: [e0123]
//
// The key replacement over a standard transformer:
//   standard:  y = W x           (matrix-vector)
//   clifford:  y = W ⊛ x         (geometric product of multivectors)
//   attention: score = <Q_i · K_j>₀   (grade-0 / scalar part)

use std::sync::Arc;
use std::ops::{Add, Mul};

// ─── 1. Multivector ───────────────────────────────────────────────────────────

/// A multivector in Cl(1,3) — 16 real components.
/// Index ordering matches the grade table above.
#[derive(Clone, Debug, Default)]
pub struct Multivector {
    pub c: [f32; 16],
}

impl Multivector {
    pub fn zero() -> Self { Self { c: [0.0; 16] } }
    pub fn scalar(v: f32) -> Self { let mut m = Self::zero(); m.c[0] = v; m }

    /// Scale every component
    pub fn scale(&self, s: f32) -> Self {
        let mut out = self.c;
        out.iter_mut().for_each(|x| *x *= s);
        Self { c: out }
    }

    /// Apply a function to every component (e.g. ReLU)
    pub fn map(&self, f: impl Fn(f32) -> f32) -> Self {
        let mut c = self.c;
        c.iter_mut().for_each(|x| *x = f(*x));
        Self { c }
    }

    /// Grade-0 (scalar) part — used for attention scores
    pub fn scalar_part(&self) -> f32 { self.c[0] }

    /// Squared norm: <M̃ M>₀  (sum of squares with metric signs)
    pub fn norm_sq(&self, alg: &CliffordAlgebra) -> f32 {
        alg.geo_product(&alg.reverse(self), self).c[0]
    }
}

impl Add for Multivector {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut c = self.c;
        for i in 0..16 { c[i] += rhs.c[i]; }
        Self { c }
    }
}

// ─── 2. Clifford Algebra (Cayley table) ──────────────────────────────────────

/// Each entry in the Cayley table: (sign ∈ {-1,+1}, output blade index 0..15)
#[derive(Clone, Copy, Debug)]
pub struct CayleyEntry(pub f32, pub usize);

#[derive(Debug)]
pub struct CliffordAlgebra {
    /// cayley[i][j] = how basis_i · basis_j maps to a single output blade
    pub cayley: [[CayleyEntry; 16]; 16],
}

impl CliffordAlgebra {
    /// Build STA = Cl(1,3): e0²=+1, e1²=e2²=e3²=-1
    /// In a real crate you'd do this at compile time with const fn or a build script.
    pub fn sta() -> Self {
        // Represented as bit-masks: blade k ↔ the set bits of k indicate
        // which basis vectors are present (e.g. e01 = 0b0011 = 3).
        // The metric signs for squaring each basis vector:
        let metric: [f32; 4] = [1.0, -1.0, -1.0, -1.0]; // e0²,e1²,e2²,e3²

        let mut cayley = [[CayleyEntry(1.0, 0); 16]; 16];
        for i in 0..16usize {
            for j in 0..16usize {
                let (sign, blade) = geometric_product_blades(i, j, &metric);
                cayley[i][j] = CayleyEntry(sign, blade);
            }
        }
        Self { cayley }
    }

    /// Geometric product: a ⊛ b
    pub fn geo_product(&self, a: &Multivector, b: &Multivector) -> Multivector {
        let mut out = [0.0f32; 16];
        for i in 0..16 {
            if a.c[i] == 0.0 { continue; }
            for j in 0..16 {
                if b.c[j] == 0.0 { continue; }
                let CayleyEntry(sign, k) = self.cayley[i][j];
                out[k] += sign * a.c[i] * b.c[j];
            }
        }
        Multivector { c: out }
    }

    /// Reverse: flip sign of grade-2 and grade-3 blades
    /// Grade k gets sign (-1)^(k(k-1)/2)
    pub fn reverse(&self, a: &Multivector) -> Multivector {
        let signs: [f32; 16] = [
            1., 1., 1., 1., 1.,        // grade 0,1
           -1.,-1.,-1.,-1.,-1.,-1.,   // grade 2
           -1.,-1.,-1.,-1.,           // grade 3
            1.,                        // grade 4
        ];
        let mut c = a.c;
        for i in 0..16 { c[i] *= signs[i]; }
        Multivector { c }
    }

    /// Inner product: scalar part of a ⊛ reverse(b)
    pub fn inner_product(&self, a: &Multivector, b: &Multivector) -> f32 {
        self.geo_product(a, &self.reverse(b)).c[0]
    }
}

/// Bit-mask geometric product for two basis blades in Cl(p,q).
/// Returns (sign, output_blade_index).
fn geometric_product_blades(mut a: usize, mut b: usize, metric: &[f32]) -> (f32, usize) {
    // Count swaps needed to merge sorted lists of indices (bubble sort parity)
    let blade_a = a;
    let blade_b = b;
    let mut sign = 1.0f32;
    let mut tmp = blade_b;
    // Count reorder swaps
    let mut a_bits = blade_a;
    while a_bits != 0 {
        let bit = a_bits & (!a_bits + 1);      // lowest set bit
        let idx = bit.trailing_zeros() as usize;
        // Count bits in blade_b that are less than idx (they pass through)
        let lower = blade_b & (bit - 1);
        if lower.count_ones() % 2 == 1 { sign = -sign; }
        if blade_b & bit != 0 {
            // Two identical basis vectors → apply metric signature
            sign *= metric[idx];
        }
        a_bits &= a_bits - 1;
    }
    let result_blade = blade_a ^ blade_b; // XOR = symmetric difference
    (sign, result_blade)
}

// ─── 3. Clifford Linear Layer ─────────────────────────────────────────────────

/// Replaces: y_d = Σ_i W[d,i] · x[i]  (scalar multiply)
/// With:     y_d = Σ_i alg.geo(W[d,i], x[i])   (geometric product)
#[derive(Debug)]
pub struct CliffordLinear {
    pub out_dim: usize,
    pub in_dim:  usize,
    /// weights[out][in]: multivector weight for each (output, input) pair
    pub weights: Vec<Vec<Multivector>>,
    pub bias:    Vec<Multivector>,
    pub algebra: Arc<CliffordAlgebra>,
}

impl CliffordLinear {
    pub fn new(in_dim: usize, out_dim: usize, algebra: Arc<CliffordAlgebra>) -> Self {
        // Initialize weights to grade-1 unit vectors + small noise (todo: kaiming)
        let weights = (0..out_dim).map(|_| {
            (0..in_dim).map(|_| Multivector::scalar(0.01)).collect()
        }).collect();
        let bias = vec![Multivector::zero(); out_dim];
        Self { out_dim, in_dim, weights, bias, algebra }
    }

    pub fn forward(&self, x: &[Multivector]) -> Vec<Multivector> {
        assert_eq!(x.len(), self.in_dim);
        (0..self.out_dim).map(|d| {
            let sum = x.iter().enumerate().fold(Multivector::zero(), |acc, (i, xi)| {
                acc + self.algebra.geo_product(&self.weights[d][i], xi)
            });
            sum + self.bias[d].clone()
        }).collect()
    }
}

// ─── 4. Clifford Layer Norm ───────────────────────────────────────────────────

/// Normalise over the 16×d_model component space per-position.
/// Treats the entire multivector sequence as a flat vector for mean/var.
pub struct CliffordLayerNorm {
    pub d_model: usize,
    pub eps: f32,
    pub gamma: Vec<f32>,   // 16 × d_model learnable scales
    pub beta:  Vec<f32>,   // 16 × d_model learnable biases
}

impl CliffordLayerNorm {
    pub fn new(d_model: usize) -> Self {
        let n = d_model * 16;
        Self { d_model, eps: 1e-5, gamma: vec![1.0; n], beta: vec![0.0; n] }
    }

    pub fn forward(&self, x: &[Multivector]) -> Vec<Multivector> {
        let flat: Vec<f32> = x.iter().flat_map(|mv| mv.c).collect();
        let n = flat.len() as f32;
        let mean = flat.iter().sum::<f32>() / n;
        let var  = flat.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let std  = (var + self.eps).sqrt();

        flat.iter().enumerate()
            .map(|(i, v)| (v - mean) / std * self.gamma[i] + self.beta[i])
            .collect::<Vec<_>>()
            .chunks(16)
            .map(|chunk| {
                let mut c = [0.0f32; 16];
                c.copy_from_slice(chunk);
                Multivector { c }
            })
            .collect()
    }
}

// ─── 5. Clifford Multi-Head Attention ─────────────────────────────────────────

pub struct CliffordAttention {
    pub d_model: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub w_q: CliffordLinear,
    pub w_k: CliffordLinear,
    pub w_v: CliffordLinear,
    pub w_o: CliffordLinear,
    pub algebra: Arc<CliffordAlgebra>,
}

impl CliffordAttention {
    pub fn new(d_model: usize, n_heads: usize, algebra: Arc<CliffordAlgebra>) -> Self {
        let head_dim = d_model / n_heads;
        let alg = algebra.clone();
        Self {
            d_model, n_heads, head_dim,
            w_q: CliffordLinear::new(d_model, d_model, alg.clone()),
            w_k: CliffordLinear::new(d_model, d_model, alg.clone()),
            w_v: CliffordLinear::new(d_model, d_model, alg.clone()),
            w_o: CliffordLinear::new(d_model, d_model, alg.clone()),
            algebra,
        }
    }

    /// x: [seq_len][d_model] multivectors.  Genuine multi-head: each head mixes
    /// only its own channel slice `[h·head_dim, (h+1)·head_dim)` and has its own
    /// softmax.  (No causal mask here — this v1 path is for shape/demo use; the
    /// trained LM uses the taped forward in `v2::tape`.)
    pub fn forward(&self, x: &[Vec<Multivector>]) -> Vec<Vec<Multivector>> {
        let seq   = x.len();
        let scale = ((self.head_dim * 16) as f32).sqrt();

        let q: Vec<Vec<Multivector>> = x.iter().map(|xi| self.w_q.forward(xi)).collect();
        let k: Vec<Vec<Multivector>> = x.iter().map(|xi| self.w_k.forward(xi)).collect();
        let v: Vec<Vec<Multivector>> = x.iter().map(|xi| self.w_v.forward(xi)).collect();

        // Per-head attention weights.
        let weights: Vec<Vec<Vec<f32>>> = (0..self.n_heads).map(|h| {
            let d0 = h * self.head_dim;
            let d1 = d0 + self.head_dim;
            (0..seq).map(|i| {
                let raw: Vec<f32> = (0..seq).map(|j| {
                    let s: f32 = (d0..d1)
                        .map(|d| self.algebra.geo_product(&q[i][d], &k[j][d]).scalar_part())
                        .sum();
                    s / scale
                }).collect();
                softmax(&raw)
            }).collect()
        }).collect();

        let mut out_seq = Vec::with_capacity(seq);
        for i in 0..seq {
            let attn_out: Vec<Multivector> = (0..self.d_model).map(|d| {
                let h = d / self.head_dim;
                (0..seq).fold(Multivector::zero(), |acc, j| {
                    acc + v[j][d].scale(weights[h][i][j])
                })
            }).collect();
            out_seq.push(self.w_o.forward(&attn_out));
        }
        out_seq
    }
}

// ─── 6. Clifford FFN ──────────────────────────────────────────────────────────

/// Two-layer feed-forward network.
/// Non-linearity: component-wise ReLU on each of the 16 multivector components.
/// You could also try the geometric sigmoid: σ(‖M‖) · M/‖M‖
pub struct CliffordFFN {
    pub fc1: CliffordLinear,
    pub fc2: CliffordLinear,
}

impl CliffordFFN {
    pub fn new(d_model: usize, d_ff: usize, algebra: Arc<CliffordAlgebra>) -> Self {
        Self {
            fc1: CliffordLinear::new(d_model, d_ff,    algebra.clone()),
            fc2: CliffordLinear::new(d_ff,    d_model, algebra),
        }
    }

    pub fn forward(&self, x: &[Multivector]) -> Vec<Multivector> {
        let h: Vec<Multivector> = self.fc1.forward(x)
            .into_iter()
            .map(|mv| mv.map(|c| c.max(0.0)))   // component-wise ReLU
            .collect();
        self.fc2.forward(&h)
    }
}

// ─── 7. Transformer Block ─────────────────────────────────────────────────────

pub struct CliffordBlock {
    pub attn:  CliffordAttention,
    pub ffn:   CliffordFFN,
    pub norm1: CliffordLayerNorm,
    pub norm2: CliffordLayerNorm,
}

impl CliffordBlock {
    pub fn forward(&self, x: &[Vec<Multivector>]) -> Vec<Vec<Multivector>> {
        // Pre-norm + residual (GPT-2 style)
        let n1: Vec<Vec<Multivector>> = x.iter()
            .map(|xi| self.norm1.forward(xi))
            .collect();
        let a = self.attn.forward(&n1);
        // Residual add (component-wise)
        let x2: Vec<Vec<Multivector>> = x.iter().zip(a.iter())
            .map(|(xi, ai)| xi.iter().zip(ai.iter()).map(|(a, b)| a.clone() + b.clone()).collect())
            .collect();

        let n2: Vec<Vec<Multivector>> = x2.iter()
            .map(|xi| self.norm2.forward(xi))
            .collect();
        let f: Vec<Vec<Multivector>> = n2.iter()
            .map(|xi| self.ffn.forward(xi))
            .collect();
        x2.iter().zip(f.iter())
            .map(|(xi, fi)| xi.iter().zip(fi.iter()).map(|(a, b)| a.clone() + b.clone()).collect())
            .collect()
    }
}

// ─── 7b. Real-valued output head ──────────────────────────────────────────────

/// Output projection from the residual stream to vocabulary logits.
///
/// The previous head was a `CliffordLinear` whose 16-component output was then
/// collapsed to its grade-0 part — meaning 15/16 of the geometric-product
/// compute (and gradient) was discarded.  This head instead **flattens** the
/// `d_model` multivectors into `16 · d_model` real features and applies a plain
/// real matrix.  It is cheaper and strictly more expressive: every blade of
/// every channel can contribute to every logit.
#[derive(Debug, Clone)]
pub struct LinearReal {
    pub out_dim:     usize,        // vocab_size
    pub in_features: usize,        // 16 × d_model
    pub weights:     Vec<Vec<f32>>, // [out_dim][in_features]
    pub bias:        Vec<f32>,      // [out_dim]
}

impl LinearReal {
    /// `d_model` multivectors in → `out_dim` logits out.  Zero-initialised;
    /// call a randomiser before training to break symmetry.
    pub fn new(d_model: usize, out_dim: usize) -> Self {
        let in_features = d_model * 16;
        Self {
            out_dim,
            in_features,
            weights: vec![vec![0.0; in_features]; out_dim],
            bias:    vec![0.0; out_dim],
        }
    }

    /// Flatten `x` (d_model multivectors → 16·d_model floats) and project to logits.
    pub fn forward(&self, x: &[Multivector]) -> Vec<f32> {
        debug_assert_eq!(x.len() * 16, self.in_features);
        let flat: Vec<f32> = x.iter().flat_map(|mv| mv.c).collect();
        (0..self.out_dim)
            .map(|o| {
                let w = &self.weights[o];
                let mut s = self.bias[o];
                for j in 0..self.in_features {
                    s += w[j] * flat[j];
                }
                s
            })
            .collect()
    }
}

// ─── 8. Full Model ────────────────────────────────────────────────────────────

pub struct CliffordLLM {
    /// Token embedding table: vocab_size × d_model Multivectors
    /// Your existing STA encoder populates these
    pub embedding:   Vec<Vec<Multivector>>,
    pub blocks:      Vec<CliffordBlock>,
    /// Final layer norm applied to the residual stream before the head (GPT-2
    /// `ln_f`).  Without it the unbounded residual stream makes logits explode.
    pub final_norm:  CliffordLayerNorm,
    pub head:        LinearReal,   // residual stream (16·d_model reals) → vocab logits
    pub algebra:     Arc<CliffordAlgebra>,
}

impl CliffordLLM {
    /// Weight tying: mirror the embedding table into the output-head weights so
    /// `logit[v] = bias[v] + <flatten(final_norm(x)), flatten(embedding[v])>`.
    ///
    /// The head and embedding then share one matrix (a strong prior + parameter
    /// saving for small models).  Call this after any embedding update and after
    /// loading a tied checkpoint so `head.weights` stays consistent; the forward
    /// and backward paths can then read `head.weights` unchanged.
    pub fn sync_tied_head(&mut self) {
        let vocab = self.embedding.len();
        debug_assert_eq!(self.head.weights.len(), vocab);
        for v in 0..vocab {
            let emb = &self.embedding[v];
            let row = &mut self.head.weights[v];
            debug_assert_eq!(row.len(), emb.len() * 16);
            for d in 0..emb.len() {
                for k in 0..16 {
                    row[d * 16 + k] = emb[d].c[k];
                }
            }
        }
    }

    /// Single forward pass — returns logits[seq_len][vocab_size]
    pub fn forward(&self, token_ids: &[usize]) -> Vec<Vec<f32>> {
        // 1. Embed tokens using your STA language encoder
        let mut x: Vec<Vec<Multivector>> = token_ids.iter()
            .map(|&id| self.embedding[id].clone())
            .collect();

        // 2. Run through Clifford transformer blocks
        for block in &self.blocks {
            x = block.forward(&x);
        }

        // 3. Final norm, then project to logits: flatten each position's
        //    d_model multivectors into 16·d_model real features and apply the
        //    real output head.
        x.iter()
            .map(|xi| self.head.forward(&self.final_norm.forward(xi)))
            .collect()
    }
}

// ─── 9. Utilities ─────────────────────────────────────────────────────────────

fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

// ─── 10. Quick smoke test ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometric_product_e01_eq_neg_e10() {
        let alg = CliffordAlgebra::sta();
        let mut e0 = Multivector::zero(); e0.c[1] = 1.0;  // e0 blade
        let mut e1 = Multivector::zero(); e1.c[2] = 1.0;  // e1 blade

        let e01 = alg.geo_product(&e0, &e1);
        let e10 = alg.geo_product(&e1, &e0);

        // e01 should equal -e10
        for i in 0..16 {
            assert!((e01.c[i] + e10.c[i]).abs() < 1e-6,
                "e01[{}]={}, e10[{}]={}", i, e01.c[i], i, e10.c[i]);
        }
    }

    #[test]
    fn forward_pass_shape() {
        let alg = Arc::new(CliffordAlgebra::sta());
        let d_model = 8;
        let vocab   = 32;
        let seq_len = 4;

        // Dummy embedding table
        let embedding = vec![vec![Multivector::scalar(0.1); d_model]; vocab];
        let blocks = vec![]; // 0 blocks for shape check
        let head = LinearReal::new(d_model, vocab);

        let model = CliffordLLM {
            embedding,
            blocks,
            final_norm: CliffordLayerNorm::new(d_model),
            head,
            algebra: alg,
        };
        let ids: Vec<usize> = (0..seq_len).collect();
        let logits = model.forward(&ids);

        assert_eq!(logits.len(), seq_len);
        assert_eq!(logits[0].len(), vocab);
    }
}

// ─── Notes for your existing STA encoder ─────────────────────────────────────
//
// Your encoder presumably maps a token to a sequence of Multivectors in Cl(1,3).
// Drop those directly into `embedding[token_id]` — no change needed.
//
// Things to implement next for real training:
//   1. Backprop through geo_product (it's bilinear, so gradients are clean)
//   2. Optimizer step (Adam works fine, treat each component independently)
//   3. Positional encoding in the multivector domain — you can encode position
//      as a rotor R = exp(-θ e12 / 2) and apply R x R̃ (sandwich product)
//   4. Causal mask: set scores[j] = -inf for j > i before softmax
//   5. KV cache for inference