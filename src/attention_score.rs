//! Attention score kernels for Clifford multi-head attention.
//!
//! - [`AttentionScoreMode::InnerProduct`]: scalar part of `Q ⊛ K̃` (Clifford inner product)
//! - [`AttentionScoreMode::Dot`]: Euclidean dot on all 16 blade components (row 3b ablation)

use serde::{Deserialize, Serialize};

use crate::Multivector;
use crate::cayley_const::CliffordAlgebraConst;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttentionScoreMode {
    /// ⟨Q, K⟩ = grade-0 part of `Q ⊛ reverse(K)` — metric-aware pairing.
    #[serde(rename = "inner")]
    InnerProduct,
    /// `Σ_k Q.c[k] · K.c[k]` — ablation row 3b.
    #[serde(rename = "dot")]
    Dot,
}

impl AttentionScoreMode {
    pub fn from_dot_flag(dot_attention: bool) -> Self {
        if dot_attention {
            Self::Dot
        } else {
            Self::InnerProduct
        }
    }
}

impl Default for AttentionScoreMode {
    fn default() -> Self {
        Self::InnerProduct
    }
}

/// Per (Q, K) pair score before head scaling / softmax.
#[inline]
pub fn attention_pair_score(
    alg: &CliffordAlgebraConst,
    q: &Multivector,
    k: &Multivector,
    mode: AttentionScoreMode,
) -> f32 {
    match mode {
        AttentionScoreMode::Dot => q.c.iter().zip(k.c.iter()).map(|(a, b)| a * b).sum(),
        AttentionScoreMode::InnerProduct => alg.inner_product(q, k),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Multivector;
    use crate::cayley_const::CliffordAlgebraConst;

    #[test]
    fn inner_product_mode_calls_algebra_inner_product() {
        let alg = CliffordAlgebraConst::new();
        let mut q = Multivector::zero();
        let mut k = Multivector::zero();
        q.c[1] = 0.5;
        k.c[6] = 0.2;
        k.c[2] = 0.3;
        let expected = alg.inner_product(&q, &k);
        let got = attention_pair_score(&alg, &q, &k, AttentionScoreMode::InnerProduct);
        assert!((got - expected).abs() < 1e-6);
        let dot = attention_pair_score(&alg, &q, &k, AttentionScoreMode::Dot);
        assert!((dot - expected).abs() > 1e-6 || expected.abs() < 1e-6);
    }
}
