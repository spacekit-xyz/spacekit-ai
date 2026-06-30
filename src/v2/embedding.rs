// embedding.rs — Trainable embedding table for the Clifford LLM
//
// In the original sketch the embedding is `Vec<Vec<Multivector>>` populated
// once from the STA encoder and never updated.  This module wraps it in a
// thin struct that:
//
//   1. Reads embedding[token_id] for the forward lookup
//   2. Accumulates gradients per-token-id during backward
//   3. Applies Adam-style updates only to token ids that actually appeared
//      in the batch (sparse update — efficient for large vocabularies)

use std::collections::HashMap;
use crate::Multivector;
use crate::optim::{AdamConfig, MvAdamState, adam_step};

// ─── Gradient accumulator ────────────────────────────────────────────────────

/// Sparse gradient buffer for the embedding table.
///
/// Only token ids that appeared in the batch get an entry, which makes the
/// memory cost O(unique_tokens × d_model × 16) rather than O(vocab × d_model × 16).
pub struct EmbeddingGrad {
    pub d_model: usize,
    /// token_id → accumulated gradient over d_model multivectors
    pub grads: HashMap<usize, Vec<Multivector>>,
}

impl EmbeddingGrad {
    pub fn new(d_model: usize) -> Self {
        Self { d_model, grads: HashMap::new() }
    }

    /// Accumulate gradient at a single token position.
    ///
    /// `token_id`   — the id whose embedding was looked up
    /// `grad`       — dL/d(embedding[token_id])  (length d_model)
    pub fn accumulate(&mut self, token_id: usize, grad: &[Multivector]) {
        debug_assert_eq!(grad.len(), self.d_model);
        let entry = self.grads.entry(token_id)
            .or_insert_with(|| vec![Multivector::zero(); self.d_model]);
        for d in 0..self.d_model {
            for k in 0..16 { entry[d].c[k] += grad[d].c[k]; }
        }
    }

    /// Average over a batch of `n` samples (divide every accumulated gradient).
    pub fn scale(&mut self, s: f32) {
        for entry in self.grads.values_mut() {
            for mv in entry { for k in 0..16 { mv.c[k] *= s; } }
        }
    }

    /// Number of distinct token ids with accumulated gradient.
    pub fn n_updated(&self) -> usize { self.grads.len() }

    /// Merge another sparse embedding gradient into this one (sum per token id).
    /// Used for gradient accumulation across microbatches.
    pub fn merge(&mut self, other: &EmbeddingGrad) {
        for (&tid, grad) in &other.grads {
            self.accumulate(tid, grad);
        }
    }
}

// ─── Embedding optimiser state ───────────────────────────────────────────────

/// Adam moment estimates for the embedding table.
///
/// Stored sparsely: state is created on demand the first time a token id is
/// updated.  This means a vocabulary of 50k where only 2k tokens appear in
/// training uses ~25× less optimiser memory than a dense allocation.
pub struct EmbeddingOptimizer {
    pub d_model: usize,
    pub cfg: AdamConfig,
    /// token_id → 16-component Adam state per model dimension
    pub states: HashMap<usize, Vec<MvAdamState>>,
}

impl EmbeddingOptimizer {
    pub fn new(d_model: usize, cfg: AdamConfig) -> Self {
        Self { d_model, cfg, states: HashMap::new() }
    }

    /// Apply Adam updates to `embedding` for every token id present in `grad`.
    ///
    /// Token ids absent from the batch are left untouched.
    pub fn step(
        &mut self,
        embedding: &mut Vec<Vec<Multivector>>,
        grad:      &EmbeddingGrad,
    ) {
        for (&token_id, token_grad) in &grad.grads {
            let state = self.states.entry(token_id)
                .or_insert_with(|| vec![MvAdamState::zero(); self.d_model]);

            for d in 0..self.d_model {
                embedding[token_id][d] = adam_step(
                    &embedding[token_id][d],
                    &token_grad[d],
                    &mut state[d],
                    &self.cfg,
                );
            }
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_accumulation_only_touches_used_ids() {
        let mut g = EmbeddingGrad::new(4);
        let some_grad = vec![Multivector::scalar(0.5); 4];

        g.accumulate(7, &some_grad);
        g.accumulate(7, &some_grad);   // same id twice
        g.accumulate(42, &some_grad);

        assert_eq!(g.n_updated(), 2);
        // token 7 accumulated twice
        assert!((g.grads[&7][0].c[0] - 1.0).abs() < 1e-6);
        // token 42 accumulated once
        assert!((g.grads[&42][0].c[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn step_updates_only_seen_tokens() {
        let d_model = 4;
        let mut embedding = vec![vec![Multivector::scalar(0.1); d_model]; 10];
        let mut opt = EmbeddingOptimizer::new(d_model, AdamConfig {
            lr: 0.1, ..Default::default()
        });

        let mut grad = EmbeddingGrad::new(d_model);
        grad.accumulate(3, &vec![Multivector::scalar(1.0); d_model]);

        let before_3 = embedding[3][0].c[0];
        let before_5 = embedding[5][0].c[0];

        opt.step(&mut embedding, &grad);

        assert!(embedding[3][0].c[0] < before_3, "seen token should move");
        assert!((embedding[5][0].c[0] - before_5).abs() < 1e-9,
            "unseen token must not move");
    }
}
