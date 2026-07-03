// inference.rs — KV-cached autoregressive forward (O(seq·layers) per new token)
//
// `model_forward_logits` recomputes the full sequence on every generate step
// (O(seq²·layers)).  This module processes only *new* suffix tokens, reusing
// cached K/V per layer — the standard transformer inference optimisation.

use crate::{
    cayley_const::CliffordAlgebraConst, kv_cache::KVCache, CliffordAttention, CliffordBlock,
    CliffordLLM, Multivector, AttentionScoreMode, attention_pair_score,
};
use crate::positional::RotorPositionalEncoding;

/// Stateful KV cache for incremental generation.
pub struct InferenceCache {
    kv:       KVCache,
    position: usize,
    pe:       RotorPositionalEncoding,
    dot_scores: bool,
}

impl InferenceCache {
    pub fn new(n_blocks: usize, max_seq_len: usize, d_model: usize, dot_scores: bool) -> Self {
        Self {
            kv: KVCache::new(n_blocks, max_seq_len),
            position: 0,
            pe: RotorPositionalEncoding::new(d_model),
            dot_scores,
        }
    }

    pub fn reset(&mut self) {
        self.kv.clear();
        self.position = 0;
    }

    /// Process any new tokens in `token_ids[self.position..]` and return logits
    /// for each newly processed position (one row per token).  Matches
    /// [`super::tape::model_forward_logits`] on the full prefix when accumulated.
    pub fn forward_extend(
        &mut self,
        alg: &CliffordAlgebraConst,
        model: &CliffordLLM,
        token_ids: &[usize],
    ) -> Vec<Vec<f32>> {
        if token_ids.len() < self.position {
            self.reset();
        }
        let mut out = Vec::new();
        for &tid in &token_ids[self.position..] {
            out.push(self.step(alg, model, tid));
        }
        out
    }

    /// Logits for the last token only (convenience wrapper for single-token steps).
    pub fn forward_last(
        &mut self,
        alg: &CliffordAlgebraConst,
        model: &CliffordLLM,
        token_ids: &[usize],
    ) -> Option<Vec<f32>> {
        self.forward_extend(alg, model, token_ids).pop()
    }

    pub fn position(&self) -> usize {
        self.position
    }

    fn step(
        &mut self,
        alg: &CliffordAlgebraConst,
        model: &CliffordLLM,
        token_id: usize,
    ) -> Vec<f32> {
        let pos = self.position;
        let emb = model.embedding[token_id].clone();
        let mut h = self.pe.encode_position(alg, &emb, pos);

        for (layer_idx, block) in model.blocks.iter().enumerate() {
            h = block_forward_cached(alg, block, &h, self.kv.layer_mut(layer_idx), self.dot_scores);
        }

        let normed = model.final_norm.forward(&h);
        let logits = model.head.forward(&normed);
        self.position += 1;
        logits
    }
}

fn block_forward_cached(
    alg: &CliffordAlgebraConst,
    block: &CliffordBlock,
    x: &[Multivector],
    cache: &mut crate::kv_cache::LayerKVCache,
    dot_scores: bool,
) -> Vec<Multivector> {
    let n1 = block.norm1.forward(x);
    let q = block.attn.w_q.forward(&n1);
    let k = block.attn.w_k.forward(&n1);
    let v = block.attn.w_v.forward(&n1);
    let attn_out = cached_multihead_attention(alg, cache, &block.attn, &q, k, v, dot_scores);

    let res1: Vec<Multivector> = x
        .iter()
        .zip(attn_out.iter())
        .map(|(a, b)| Multivector {
            c: std::array::from_fn(|k| a.c[k] + b.c[k]),
        })
        .collect();

    let n2 = block.norm2.forward(&res1);
    let ffn_out = block.ffn.forward(&n2);

    res1.iter()
        .zip(ffn_out.iter())
        .map(|(a, b)| Multivector {
            c: std::array::from_fn(|k| a.c[k] + b.c[k]),
        })
        .collect()
}

/// Multi-head attention for one new query token against cached K/V (+ self).
fn cached_multihead_attention(
    alg: &CliffordAlgebraConst,
    cache: &mut crate::kv_cache::LayerKVCache,
    attn: &CliffordAttention,
    q_new: &[Multivector],
    k_new: Vec<Multivector>,
    v_new: Vec<Multivector>,
    dot_scores: bool,
) -> Vec<Multivector> {
    let score_mode = AttentionScoreMode::from_dot_flag(dot_scores);
    cache.push(k_new, v_new);
    let seq = cache.seq_len();
    let n_heads = attn.n_heads;
    let head_dim = attn.head_dim;
    let d_model = attn.d_model;
    let scale = ((head_dim * 16) as f32).sqrt();

    // Per-head softmax weights over all cached positions (past + present).
    let mut head_weights: Vec<Vec<f32>> = Vec::with_capacity(n_heads);
    for h in 0..n_heads {
        let d0 = h * head_dim;
        let d1 = d0 + head_dim;
        let mut scores = vec![0.0f32; seq];
        for j in 0..seq {
            scores[j] = (d0..d1)
                .map(|d| attention_pair_score(alg, &q_new[d], &cache.k[j][d], score_mode))
                .sum::<f32>()
                / scale;
        }
        head_weights.push(softmax(&scores));
    }

    let agg: Vec<Multivector> = (0..d_model)
        .map(|d| {
            let h = d / head_dim;
            let w = &head_weights[h];
            (0..seq).fold(Multivector::zero(), |acc, j| acc + cache.v[j][d].scale(w[j]))
        })
        .collect();

    attn.w_o.forward(&agg)
}

fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 || !sum.is_finite() {
        return vec![1.0 / x.len() as f32; x.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::train_v2::{randomize_model, TrainConfigV2, ModelStateV2};
    use crate::v2::tape::model_forward_logits;

    fn tiny_model(vocab: usize) -> (CliffordAlgebraConst, CliffordLLM) {
        let mut cfg = TrainConfigV2::small(vocab);
        cfg.d_model = 8;
        cfg.n_heads = 2;
        cfg.n_blocks = 2;
        let mut st = ModelStateV2::new(cfg);
        randomize_model(&mut st.model, 42);
        (st.alg, st.model)
    }

    #[test]
    fn cached_matches_full_forward() {
        let (alg, model) = tiny_model(32);
        let ids: Vec<usize> = vec![1, 5, 9, 2, 7];

        let full = model_forward_logits(&alg, &model, &ids, true, false);

        let mut cache = InferenceCache::new(model.blocks.len(), 128, 8, false);
        let mut cached = Vec::new();
        for i in 1..=ids.len() {
            cached.extend(cache.forward_extend(&alg, &model, &ids[..i]));
        }

        assert_eq!(cached.len(), full.len());
        for (a, b) in cached.iter().zip(full.iter()) {
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(b.iter()) {
                assert!(
                    (x - y).abs() < 1e-4,
                    "logit mismatch: cached={x} full={y}"
                );
            }
        }
    }

    #[test]
    fn incremental_extend_matches_batch() {
        let (alg, model) = tiny_model(24);
        let ids: Vec<usize> = vec![3, 8, 11, 4];

        let full = model_forward_logits(&alg, &model, &ids, true, false);

        let mut cache = InferenceCache::new(model.blocks.len(), 64, 8, false);
        let batch = cache.forward_extend(&alg, &model, &ids);

        assert_eq!(batch.len(), full.len());
        for t in 0..full.len() {
            for v in 0..full[t].len() {
                assert!(
                    (batch[t][v] - full[t][v]).abs() < 1e-4,
                    "t={t} v={v}"
                );
            }
        }
    }
}
