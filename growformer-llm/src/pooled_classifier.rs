//! Mean-pooled byte embeddings + Clifford linear head + optional scalar grounding injection.
//! End-to-end differentiable with existing `linear_backward` / `cross_entropy`.

use std::sync::Arc;

use crate::backprop::{cross_entropy, linear_backward, scalar_head_backward};
use crate::{CliffordAlgebra, CliffordLinear, Multivector};

pub struct PooledClassifier {
    pub embedding: Vec<Vec<Multivector>>,
    pub head: CliffordLinear,
    /// Per `d_model` slot: linear mix of grounding vector into scalar channel before head.
    pub ground_w: Vec<Vec<f32>>,
    pub ground_b: Vec<f32>,
    pub d_model: usize,
    pub n_classes: usize,
    pub vocab_size: usize,
    pub ground_dim: usize,
}

impl PooledClassifier {
    pub fn new(
        algebra: Arc<CliffordAlgebra>,
        vocab_size: usize,
        d_model: usize,
        n_classes: usize,
        ground_dim: usize,
    ) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let embedding: Vec<Vec<Multivector>> = (0..vocab_size)
            .map(|_| {
                (0..d_model)
                    .map(|_| {
                        let mut mv = Multivector::zero();
                        for k in 0..16 {
                            mv.c[k] = rng.gen_range(-0.03f32..0.03f32);
                        }
                        mv
                    })
                    .collect()
            })
            .collect();

        let head = CliffordLinear::new(d_model, n_classes, algebra);
        let ground_w: Vec<Vec<f32>> = (0..d_model)
            .map(|_| {
                (0..ground_dim)
                    .map(|_| rng.gen_range(-0.05f32..0.05f32))
                    .collect()
            })
            .collect();
        let ground_b = vec![0.0f32; d_model];

        Self {
            embedding,
            head,
            ground_w,
            ground_b,
            d_model,
            n_classes,
            vocab_size,
            ground_dim,
        }
    }

    pub fn forward_logits(&self, byte_ids: &[usize], g: &[f32]) -> (Vec<f32>, Vec<Multivector>) {
        assert_eq!(g.len(), self.ground_dim);
        let seq_len = byte_ids.len().max(1);
        let mut pooled: Vec<Multivector> = (0..self.d_model).map(|_| Multivector::zero()).collect();

        for &id in byte_ids {
            let row = &self.embedding[id];
            for d in 0..self.d_model {
                for k in 0..16 {
                    pooled[d].c[k] += row[d].c[k];
                }
            }
        }
        let inv = 1.0f32 / seq_len as f32;
        for d in 0..self.d_model {
            for k in 0..16 {
                pooled[d].c[k] *= inv;
            }
        }

        let mut aug = pooled.clone();
        for d in 0..self.d_model {
            let inj = self.ground_b[d]
                + self.ground_w[d]
                    .iter()
                    .zip(g.iter())
                    .map(|(&w, &gv)| w * gv)
                    .sum::<f32>();
            aug[d].c[0] += inj;
        }

        let head_out = self.head.forward(&aug);
        let logits: Vec<f32> = head_out.iter().map(|mv| mv.scalar_part()).collect();
        (logits, aug)
    }

    /// Single-example backward: returns grads for head, embedding slice, ground weights.
    pub fn backward_one(
        &self,
        byte_ids: &[usize],
        g: &[f32],
        label: usize,
    ) -> (
        f32,
        crate::backprop::GradLinear,
        Vec<Vec<Multivector>>,
        Vec<Vec<f32>>,
        Vec<f32>,
    ) {
        let (logits, aug) = self.forward_logits(byte_ids, g);
        let (loss, grad_logits) = cross_entropy(&logits, label);
        let grad_head_out = scalar_head_backward(&grad_logits);
        let (grad_head, grad_aug) = linear_backward(&self.head.weights, &aug, &grad_head_out);

        let seq_len = byte_ids.len().max(1);
        let inv = 1.0f32 / seq_len as f32;

        let mut grad_emb = vec![vec![Multivector::zero(); self.d_model]; self.vocab_size];
        for &id in byte_ids {
            for d in 0..self.d_model {
                for k in 0..16 {
                    grad_emb[id][d].c[k] += grad_aug[d].c[k] * inv;
                }
            }
        }

        let mut grad_gw: Vec<Vec<f32>> = (0..self.d_model)
            .map(|d| {
                (0..self.ground_dim)
                    .map(|j| grad_aug[d].c[0] * g[j])
                    .collect()
            })
            .collect();
        let mut grad_gb: Vec<f32> = (0..self.d_model).map(|d| grad_aug[d].c[0]).collect();

        // Finite grounding can inflate grads — gentle scaling keeps Adam stable on tiny data.
        let gs = 0.25f32;
        for row in &mut grad_gw {
            for v in row {
                *v *= gs;
            }
        }
        for v in &mut grad_gb {
            *v *= gs;
        }

        (loss, grad_head, grad_emb, grad_gw, grad_gb)
    }
}
