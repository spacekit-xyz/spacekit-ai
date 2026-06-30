// kv_cache.rs — Key-Value cache for autoregressive inference
//
// During training the full sequence is processed in one shot.  During
// inference (token-by-token generation) recomputing K and V for all past
// positions on every step wastes O(n²) work.  The KV cache stores the K and V
// multivectors from all previous steps so each new token only needs to compute
// its own Q, K, V and can attend to cached past K/V directly.
//
// Cache layout:
//   KVCache
//     └── layers: Vec<LayerKVCache>   (one per CliffordBlock)
//           ├── k: Vec<Vec<Multivector>>  [seq_so_far][d_model]
//           └── v: Vec<Vec<Multivector>>  [seq_so_far][d_model]

use crate::Multivector;

// ─── Per-layer cache ──────────────────────────────────────────────────────────

/// Key and value cache for a single transformer layer.
#[derive(Clone, Debug, Default)]
pub struct LayerKVCache {
    /// Accumulated K multivectors: one Vec<Multivector> per past token.
    pub k: Vec<Vec<Multivector>>,
    /// Accumulated V multivectors: one Vec<Multivector> per past token.
    pub v: Vec<Vec<Multivector>>,
}

impl LayerKVCache {
    pub fn new() -> Self { Self::default() }

    /// Append K and V for one new token position.
    pub fn push(&mut self, k: Vec<Multivector>, v: Vec<Multivector>) {
        self.k.push(k);
        self.v.push(v);
    }

    /// Number of cached token positions.
    #[inline]
    pub fn seq_len(&self) -> usize { self.k.len() }

    /// Return a slice of all cached K vectors — used as the full key sequence
    /// when computing attention for the new token.
    #[inline]
    pub fn all_k(&self) -> &[Vec<Multivector>] { &self.k }

    /// Return a slice of all cached V vectors.
    #[inline]
    pub fn all_v(&self) -> &[Vec<Multivector>] { &self.v }

    /// Truncate the cache to `len` positions (e.g. for sliding-window attention).
    pub fn truncate(&mut self, len: usize) {
        self.k.truncate(len);
        self.v.truncate(len);
    }

    /// Clear the cache entirely (start of a new sequence).
    pub fn clear(&mut self) {
        self.k.clear();
        self.v.clear();
    }
}

// ─── Full model cache ─────────────────────────────────────────────────────────

/// KV cache for the entire model.  Holds one LayerKVCache per transformer block.
pub struct KVCache {
    pub layers:  Vec<LayerKVCache>,
    pub max_len: usize,   // maximum sequence length before eviction
}

impl KVCache {
    /// Allocate a KV cache for `n_layers` transformer blocks and a maximum
    /// context length of `max_len` tokens.
    pub fn new(n_layers: usize, max_len: usize) -> Self {
        Self {
            layers:  (0..n_layers).map(|_| LayerKVCache::new()).collect(),
            max_len,
        }
    }

    /// Append the K and V vectors for one new token across all layers simultaneously.
    ///
    /// `layer_kvs` — Vec of (k, v) pairs, one per layer (length must equal n_layers).
    pub fn push_all(&mut self, layer_kvs: Vec<(Vec<Multivector>, Vec<Multivector>)>) {
        assert_eq!(layer_kvs.len(), self.layers.len(),
            "layer_kvs length must match number of cached layers");
        for (layer, (k, v)) in self.layers.iter_mut().zip(layer_kvs.into_iter()) {
            layer.push(k, v);
        }
        // Evict if we exceeded max_len (sliding window: drop oldest)
        let seq = self.seq_len();
        if seq > self.max_len {
            let trim_to = self.max_len;
            // Shift out the oldest token by removing index 0 from each layer
            for layer in &mut self.layers {
                let excess = layer.seq_len() - trim_to;
                layer.k.drain(0..excess);
                layer.v.drain(0..excess);
            }
        }
    }

    /// Number of token positions currently cached (same for all layers).
    pub fn seq_len(&self) -> usize {
        self.layers.first().map(|l| l.seq_len()).unwrap_or(0)
    }

    /// Clear all layers (reset to empty context).
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    /// Access the cache for layer `i`.
    #[inline]
    pub fn layer(&self, i: usize) -> &LayerKVCache { &self.layers[i] }

    /// Mutable access to the cache for layer `i`.
    #[inline]
    pub fn layer_mut(&mut self, i: usize) -> &mut LayerKVCache { &mut self.layers[i] }
}

// ─── Cached attention helper ──────────────────────────────────────────────────

/// Compute attention scores for a *single new query token* against all cached
/// key vectors (past + current).  Returns the weighted sum of cached V vectors.
///
/// This is the inner loop of cached autoregressive generation.
///
/// `q_new`   — the query multivectors for the new token  [d_model]
/// `cache`   — the LayerKVCache containing all past K and V
/// `k_new`   — the key multivectors for the new token    [d_model]
/// `v_new`   — the value multivectors for the new token  [d_model]
/// `scale`   — 1/√(head_dim × 16)
///
/// The function appends k_new and v_new to the cache *before* computing
/// attention so the new token can attend to itself (causal, present = visible).
pub fn cached_attention_step(
    alg:   &crate::cayley_const::CliffordAlgebraConst,
    cache: &mut LayerKVCache,
    q_new: &[Multivector],
    k_new: Vec<Multivector>,
    v_new: Vec<Multivector>,
    scale: f32,
) -> Vec<Multivector> {
    // Append current token's K, V to the cache
    cache.push(k_new, v_new);

    let seq = cache.seq_len();
    let d   = q_new.len();

    // Compute attention scores: score[j] = (1/scale) Σ_d <Q[d] ⊛ K_j[d]>₀
    let scores: Vec<f32> = (0..seq).map(|j| {
        let s: f32 = q_new.iter().zip(cache.k[j].iter())
            .map(|(qi, kj)| alg.geo_product(qi, kj).c[0])
            .sum();
        s / scale
    }).collect();

    // Softmax (no causal mask needed — cache only holds past + present)
    let scores = softmax(&scores);

    // Weighted sum of V
    (0..d).map(|dim| {
        (0..seq).fold(Multivector::zero(), |acc, j| {
            let scaled = cache.v[j][dim].scale(scores[j]);
            Multivector {
                c: std::array::from_fn(|k| acc.c[k] + scaled.c[k]),
            }
        })
    }).collect()
}

fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_mv(v: f32) -> Vec<Multivector> {
        vec![Multivector::scalar(v); 4] // d_model = 4
    }

    #[test]
    fn push_and_len() {
        let mut cache = LayerKVCache::new();
        assert_eq!(cache.seq_len(), 0);
        cache.push(dummy_mv(1.0), dummy_mv(0.5));
        cache.push(dummy_mv(2.0), dummy_mv(0.8));
        assert_eq!(cache.seq_len(), 2);
    }

    #[test]
    fn clear_resets_to_zero() {
        let mut cache = LayerKVCache::new();
        cache.push(dummy_mv(1.0), dummy_mv(1.0));
        cache.clear();
        assert_eq!(cache.seq_len(), 0);
    }

    #[test]
    fn kvcache_evicts_oldest_on_overflow() {
        let mut cache = KVCache::new(1, 3); // max 3 tokens
        for t in 0..5u32 {
            cache.push_all(vec![(dummy_mv(t as f32), dummy_mv(t as f32))]);
        }
        // Should have trimmed to 3
        assert_eq!(cache.seq_len(), 3, "cache should be trimmed to max_len=3");
        // Oldest remaining K scalar should be 2.0 (tokens 0,1 evicted)
        assert!((cache.layers[0].k[0][0].c[0] - 2.0).abs() < 1e-6,
            "oldest remaining token should be t=2");
    }

    #[test]
    fn kvcache_multi_layer_consistency() {
        let mut cache = KVCache::new(3, 10);
        cache.push_all(vec![
            (dummy_mv(1.0), dummy_mv(1.0)),
            (dummy_mv(2.0), dummy_mv(2.0)),
            (dummy_mv(3.0), dummy_mv(3.0)),
        ]);
        assert_eq!(cache.seq_len(), 1);
        assert!((cache.layer(0).k[0][0].c[0] - 1.0).abs() < 1e-6);
        assert!((cache.layer(1).k[0][0].c[0] - 2.0).abs() < 1e-6);
        assert!((cache.layer(2).k[0][0].c[0] - 3.0).abs() < 1e-6);
    }
}
