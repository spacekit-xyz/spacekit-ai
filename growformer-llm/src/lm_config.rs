//! Shared LM training configuration (vanilla-first core).

use serde::{Deserialize, Serialize};

/// Training / checkpoint config for both vanilla and (optional) Clifford LMs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfigV2 {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub n_blocks: usize,
    pub max_seq: usize,
    pub batch_size: usize,
    pub epochs: usize,
    pub lr_max: f32,
    pub lr_min: f32,
    pub warmup_steps: u64,
    pub total_steps: u64,
    pub grad_clip: f32,
    pub log_every: usize,
    pub val_every: usize,
    /// If true, also update the embedding table.
    pub train_embeddings: bool,
    /// Seed for random weight/embedding initialisation.
    #[serde(default = "default_init_seed")]
    pub init_seed: u64,
    #[serde(default)]
    pub freeze_embeddings: bool,
    #[serde(default)]
    pub freeze_blocks: usize,
    #[serde(default)]
    pub tie_embeddings: bool,
    #[serde(default)]
    pub structured_init: bool,
    #[serde(default = "default_grad_accum")]
    pub grad_accum: usize,
    /// Clifford ablation: dense real FFN (research-only).
    #[serde(default)]
    pub dense_ffn: bool,
    /// Clifford ablation: dot attention scores (research-only).
    #[serde(default)]
    pub dot_attention: bool,
    /// Product default: vanilla transformer (Bet B winner).
    #[serde(default = "default_vanilla")]
    pub vanilla: bool,
    /// Clifford reference `d_model` before param-budget matching (0 if N/A).
    #[serde(default)]
    pub clifford_ref_d_model: usize,
}

fn default_grad_accum() -> usize {
    1
}

fn default_init_seed() -> u64 {
    0x5EED_1234_ABCD_0001
}

fn default_vanilla() -> bool {
    true
}

impl TrainConfigV2 {
    /// Small vanilla config (product default after Bet B).
    pub fn small(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            d_model: 8,
            n_heads: 2,
            d_ff: 32,
            n_blocks: 2,
            max_seq: 128,
            batch_size: 4,
            epochs: 10,
            lr_max: 3e-4,
            lr_min: 1e-5,
            warmup_steps: 100,
            total_steps: 2000,
            grad_clip: 1.0,
            log_every: 10,
            val_every: 100,
            train_embeddings: true,
            init_seed: default_init_seed(),
            freeze_embeddings: false,
            freeze_blocks: 0,
            tie_embeddings: false,
            structured_init: false,
            grad_accum: 1,
            dense_ffn: false,
            dot_attention: false,
            vanilla: true,
            clifford_ref_d_model: 0,
        }
    }

    /// Historical Clifford small config (research / clifford-lm feature).
    pub fn small_clifford(vocab_size: usize) -> Self {
        let mut c = Self::small(vocab_size);
        c.vanilla = false;
        c
    }
}
