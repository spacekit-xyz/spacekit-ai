//! Real-valued linear layer (shared by vanilla LM and Clifford output head).

/// Real matrix: `out_dim × in_features` plus bias.
///
/// Used as the vanilla transformer’s Q/K/V/O/FFN/head layers, and as the
/// Clifford LM’s flattened multivector → vocab head.
#[derive(Debug, Clone)]
pub struct LinearReal {
    pub out_dim: usize,
    pub in_features: usize,
    pub weights: Vec<Vec<f32>>, // [out_dim][in_features]
    pub bias: Vec<f32>,         // [out_dim]
}

impl LinearReal {
    /// Clifford head helper: `d_model` multivectors → `16 · d_model` features → `out_dim`.
    pub fn new(d_model: usize, out_dim: usize) -> Self {
        Self::new_dims(d_model * 16, out_dim, 0)
    }

    /// General real linear: `in_features` → `out_dim`. `seed` reserved for init helpers.
    pub fn new_dims(in_features: usize, out_dim: usize, _seed: u64) -> Self {
        Self {
            out_dim,
            in_features,
            weights: vec![vec![0.0; in_features]; out_dim],
            bias: vec![0.0; out_dim],
        }
    }

    pub fn weight_scalars(&self) -> usize {
        self.out_dim * self.in_features + self.out_dim
    }

    /// Real matmul: `out[o] = bias[o] + Σ_j W[o][j] · x[j]`.
    pub fn forward_flat(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.in_features);
        (0..self.out_dim)
            .map(|o| {
                let w = &self.weights[o];
                let mut s = self.bias[o];
                for j in 0..self.in_features {
                    s += w[j] * x[j];
                }
                s
            })
            .collect()
    }
}
