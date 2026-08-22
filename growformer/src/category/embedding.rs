// ── embedding.rs ───────────────────────────────────────────────────────────────
// Pluggable sentence → `Vec<f32>` for `TrainingRecord.embedding` and training parity.

/// Fill [`crate::category::training::TrainingRecord::embedding`] before or after load.
pub trait SentenceEmbedder {
    fn embed_dim(&self) -> usize;
    fn embed(&self, text: &str) -> Vec<f32>;
}

/// Deterministic character-hash (same as [`crate::category::forward::char_hash_embed`]).
#[derive(Debug, Clone)]
pub struct CharHashEmbedder {
    pub dim: usize,
}

impl CharHashEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl SentenceEmbedder for CharHashEmbedder {
    fn embed_dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        crate::category::forward::char_hash_embed(text, self.dim)
    }
}

/// Hashed bag-of-tokens with L2 normalization — richer than char hash, still dependency-free.
#[derive(Debug, Clone)]
pub struct TokenHashEmbedder {
    pub dim: usize,
}

impl TokenHashEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    fn fnv1a64(bytes: &[u8]) -> u64 {
        const FNV_OFFSET: u64 = 1469598103934665603;
        const FNV_PRIME: u64 = 1099511628211;
        let mut h = FNV_OFFSET;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
        h
    }
}

impl SentenceEmbedder for TokenHashEmbedder {
    fn embed_dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        let dim = self.dim.max(1);
        let mut v = vec![0.0f32; dim];
        for w in text.split_whitespace() {
            let lw = w.to_lowercase();
            let h = Self::fnv1a64(lw.as_bytes());
            let i = (h as usize) % dim;
            v[i] += 1.0;
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_embed_deterministic() {
        let e = TokenHashEmbedder::new(32);
        let a = e.embed("hello world hello");
        let b = e.embed("hello world hello");
        assert_eq!(a, b);
        assert!(
            (a.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 1e-4
                || a.iter().all(|&x| x == 0.0)
        );
    }
}
