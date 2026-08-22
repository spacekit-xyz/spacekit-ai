//! FFN variants for the param-matched dense ablation (FFN-only swap at block boundary).

use std::sync::Arc;

use crate::{CliffordAlgebra, CliffordFFN, CliffordLinear, LinearReal, Multivector};

// ─── Multivector flatten (same layout as LinearReal head) ─────────────────────

pub fn flatten_mvs(x: &[Multivector]) -> Vec<f32> {
    x.iter().flat_map(|mv| mv.c).collect()
}

pub fn unflatten_mvs(flat: &[f32], d_model: usize) -> Vec<Multivector> {
    debug_assert_eq!(flat.len(), d_model * 16);
    flat.chunks(16)
        .map(|chunk| {
            let mut c = [0.0f32; 16];
            c.copy_from_slice(chunk);
            Multivector { c }
        })
        .collect()
}

// ─── Param budgets (weights + biases, every learnable scalar) ────────────────

pub fn clifford_ffn_scalars(d_model: usize, d_ff: usize) -> usize {
    16 * (2 * d_model * d_ff + d_ff + d_model)
}

pub fn dense_ffn_scalars(d_model: usize, hidden: usize) -> usize {
    let in_ = 16 * d_model;
    2 * in_ * hidden + hidden + in_
}

/// Hidden width `H` that minimises |dense_total − clifford_total| (integer `H` only).
pub fn matched_dense_ffn_hidden(d_model: usize, d_ff: usize) -> usize {
    let target = clifford_ffn_scalars(d_model, d_ff);
    let lo = d_ff.saturating_sub(4).max(1);
    let hi = d_ff + 8;
    let mut best_h = d_ff;
    let mut best_diff = usize::MAX;
    for h in lo..=hi {
        let diff = target.abs_diff(dense_ffn_scalars(d_model, h));
        if diff < best_diff {
            best_diff = diff;
            best_h = h;
        }
    }
    best_h
}

pub fn assert_ffn_param_match(d_model: usize, d_ff: usize, hidden: usize) {
    let clifford = clifford_ffn_scalars(d_model, d_ff);
    let dense = dense_ffn_scalars(d_model, hidden);
    if clifford == dense {
        eprintln!("[ffn] param budgets matched: {clifford} scalars (H={hidden})");
    } else {
        eprintln!(
            "[ffn] param budget: clifford={clifford} dense={dense} (H={hidden}, Δ={}) — \
             integer H may not hit exact total; closest in search window",
            clifford as i64 - dense as i64
        );
    }
}

// ─── Dense FFN (real linear on flattened 16·d_model residual) ────────────────

pub struct DenseFFN {
    pub fc1: LinearReal,
    pub fc2: LinearReal,
    pub d_model: usize,
    pub hidden: usize,
}

impl DenseFFN {
    pub fn new(d_model: usize, hidden: usize, init_seed: u64) -> Self {
        let in_ = 16 * d_model;
        Self {
            fc1: LinearReal::new_dims(in_, hidden, init_seed),
            fc2: LinearReal::new_dims(hidden, in_, init_seed ^ 0x9E37_79B9_7F4A_7C15),
            d_model,
            hidden,
        }
    }

    pub fn forward(&self, x: &[Multivector]) -> Vec<Multivector> {
        let flat = flatten_mvs(x);
        let pre = self.fc1.forward_flat(&flat);
        let post: Vec<f32> = pre.iter().map(|v| v.max(0.0)).collect();
        let out = self.fc2.forward_flat(&post);
        unflatten_mvs(&out, self.d_model)
    }
}

// ─── Block-level FFN variant ─────────────────────────────────────────────────

pub enum FfnVariant {
    Clifford(CliffordFFN),
    Dense(DenseFFN),
}

impl FfnVariant {
    pub fn clifford(d_model: usize, d_ff: usize, algebra: Arc<CliffordAlgebra>) -> Self {
        Self::Clifford(CliffordFFN::new(d_model, d_ff, algebra))
    }

    pub fn dense_matched(d_model: usize, d_ff: usize, init_seed: u64) -> Self {
        let hidden = matched_dense_ffn_hidden(d_model, d_ff);
        assert_ffn_param_match(d_model, d_ff, hidden);
        Self::Dense(DenseFFN::new(d_model, hidden, init_seed))
    }

    pub fn is_dense(&self) -> bool {
        matches!(self, Self::Dense(_))
    }

    pub fn forward(&self, x: &[Multivector]) -> Vec<Multivector> {
        match self {
            Self::Clifford(f) => f.forward(x),
            Self::Dense(f) => f.forward(x),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_hidden_search_is_close_at_defaults() {
        let h = matched_dense_ffn_hidden(16, 64);
        let c = clifford_ffn_scalars(16, 64);
        let d = dense_ffn_scalars(16, h);
        assert!(c.abs_diff(d) <= 100, "c={c} d={d} h={h}");
    }
}
