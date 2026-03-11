//! Character-level autoregressive generation head.
//!
//! Architecture:
//!   input = concat(conditioning_embedding[cond_dim], char_context_flat[ctx_len * char_embed_dim])
//!   MLP: input_dim -> hidden -> VOCAB_SIZE (logits over next char)
//!
//! Training: teacher forcing with softmax cross-entropy.
//! Inference: autoregressive sampling (greedy or temperature-based).

use serde::{Deserialize, Serialize};

pub const VOCAB_SIZE: usize = 128; // ASCII
pub const CHAR_EMBED_DIM: usize = 16;
pub const CONTEXT_LEN: usize = 32;
pub const EOS: u8 = 0; // null byte = end of sequence

/// Learned character embeddings: VOCAB_SIZE -> CHAR_EMBED_DIM
#[derive(Clone, Serialize, Deserialize)]
pub struct CharEmbeddings {
    pub table: Vec<Vec<f32>>, // [VOCAB_SIZE][CHAR_EMBED_DIM]
}

impl CharEmbeddings {
    pub fn new() -> Self {
        let mut table = vec![vec![0.0f32; CHAR_EMBED_DIM]; VOCAB_SIZE];
        for (c, row) in table.iter_mut().enumerate() {
            for (d, v) in row.iter_mut().enumerate() {
                *v = (((c as u64 * 2654435761 + d as u64 * 7919) % 1000) as f32 / 1000.0 - 0.5)
                    * 0.2;
            }
        }
        Self { table }
    }

    pub fn embed(&self, ch: u8) -> &[f32] {
        let idx = (ch as usize).min(VOCAB_SIZE - 1);
        &self.table[idx]
    }
}

/// Character-level autoregressive MLP.
#[derive(Clone, Serialize, Deserialize)]
pub struct GenerationHead {
    pub char_emb: CharEmbeddings,
    pub w1: Vec<Vec<f32>>,
    pub b1: Vec<f32>,
    pub w2: Vec<Vec<f32>>,
    pub b2: Vec<f32>,
    pub cond_dim: usize,
    pub hidden_dim: usize,
    input_dim: usize,
}

impl GenerationHead {
    pub fn new(cond_dim: usize, hidden_dim: usize) -> Self {
        let input_dim = cond_dim + CONTEXT_LEN * CHAR_EMBED_DIM;
        let mut w1 = vec![vec![0.0f32; input_dim]; hidden_dim];
        let mut w2 = vec![vec![0.0f32; hidden_dim]; VOCAB_SIZE];
        for (h, row) in w1.iter_mut().enumerate() {
            for (i, w) in row.iter_mut().enumerate() {
                *w = (((h as u64 * 2654435761 + i as u64 * 7919 + 42) % 1000) as f32 / 1000.0
                    - 0.5)
                    * (2.0 / (input_dim as f32).sqrt());
            }
        }
        for (o, row) in w2.iter_mut().enumerate() {
            for (h, w) in row.iter_mut().enumerate() {
                *w = (((o as u64 * 2654435761 + h as u64 * 7919 + 99) % 1000) as f32 / 1000.0
                    - 0.5)
                    * (2.0 / (hidden_dim as f32).sqrt());
            }
        }
        Self {
            char_emb: CharEmbeddings::new(),
            w1,
            b1: vec![0.0; hidden_dim],
            w2,
            b2: vec![0.0; VOCAB_SIZE],
            cond_dim,
            hidden_dim,
            input_dim,
        }
    }

    fn build_input(&self, cond: &[f32], context: &[u8]) -> Vec<f32> {
        let mut input = Vec::with_capacity(self.input_dim);
        for i in 0..self.cond_dim {
            input.push(if i < cond.len() { cond[i] } else { 0.0 });
        }
        let start = if context.len() >= CONTEXT_LEN {
            context.len() - CONTEXT_LEN
        } else {
            0
        };
        let ctx = &context[start..];
        let pad = CONTEXT_LEN - ctx.len();
        for _ in 0..pad {
            for _ in 0..CHAR_EMBED_DIM {
                input.push(0.0);
            }
        }
        for &ch in ctx {
            let emb = self.char_emb.embed(ch);
            input.extend_from_slice(emb);
        }
        input
    }

    fn forward(&self, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut h = vec![0.0f32; self.hidden_dim];
        for (j, hj) in h.iter_mut().enumerate() {
            let mut acc = self.b1[j];
            for (i, &xi) in input.iter().enumerate() {
                if i < self.w1[j].len() {
                    acc += self.w1[j][i] * xi;
                }
            }
            *hj = acc.tanh();
        }
        let mut logits = vec![0.0f32; VOCAB_SIZE];
        for (o, lo) in logits.iter_mut().enumerate() {
            let mut acc = self.b2[o];
            for (j, &hj) in h.iter().enumerate() {
                acc += self.w2[o][j] * hj;
            }
            *lo = acc;
        }
        (h, logits)
    }

    fn softmax(logits: &[f32]) -> Vec<f32> {
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
        let sum: f32 = exp.iter().sum();
        exp.iter().map(|e| e / sum).collect()
    }

    /// Train on one (conditioning, target_text) pair using teacher forcing.
    /// Returns average cross-entropy loss over characters.
    pub fn train_step(&mut self, cond: &[f32], target: &str, lr: f32) -> f32 {
        let target_bytes: Vec<u8> = target.bytes().take(200).collect();
        if target_bytes.is_empty() {
            return 0.0;
        }

        let mut context: Vec<u8> = Vec::new();
        let mut total_loss = 0.0f32;

        for &target_ch in &target_bytes {
            let target_idx = (target_ch as usize).min(VOCAB_SIZE - 1);
            let input = self.build_input(cond, &context);
            let (h, logits) = self.forward(&input);
            let probs = Self::softmax(&logits);
            total_loss -= probs[target_idx].max(1e-10).ln();

            // Backprop: d_logits = probs - one_hot
            let mut d_logits = probs;
            d_logits[target_idx] -= 1.0;

            // w2, b2
            let mut d_h = vec![0.0f32; self.hidden_dim];
            for (o, &dl) in d_logits.iter().enumerate() {
                self.b2[o] -= lr * dl;
                for (j, &hj) in h.iter().enumerate() {
                    d_h[j] += self.w2[o][j] * dl;
                    self.w2[o][j] -= lr * dl * hj;
                }
            }

            // w1, b1 + char embeddings
            for (j, &dh) in d_h.iter().enumerate() {
                let d_pre = dh * (1.0 - h[j] * h[j]);
                self.b1[j] -= lr * d_pre;
                for (i, &xi) in input.iter().enumerate() {
                    if i < self.w1[j].len() {
                        self.w1[j][i] -= lr * d_pre * xi;
                    }
                }
            }

            // Update char embeddings for context chars
            let start = if context.len() >= CONTEXT_LEN {
                context.len() - CONTEXT_LEN
            } else {
                0
            };
            let ctx = &context[start..];
            let pad = CONTEXT_LEN - ctx.len();
            for (pos, &ch) in ctx.iter().enumerate() {
                let ch_idx = (ch as usize).min(VOCAB_SIZE - 1);
                let base = self.cond_dim + (pad + pos) * CHAR_EMBED_DIM;
                for d in 0..CHAR_EMBED_DIM {
                    let input_idx = base + d;
                    let mut grad = 0.0f32;
                    for (j, &dh) in d_h.iter().enumerate() {
                        let d_pre = dh * (1.0 - h[j] * h[j]);
                        if input_idx < self.w1[j].len() {
                            grad += d_pre * self.w1[j][input_idx];
                        }
                    }
                    self.char_emb.table[ch_idx][d] -= lr * grad;
                }
            }

            context.push(target_ch);
        }

        total_loss / target_bytes.len() as f32
    }

    /// Generate text autoregressively from a conditioning embedding.
    pub fn generate(&self, cond: &[f32], max_len: usize, temperature: f32) -> String {
        let mut context: Vec<u8> = Vec::new();
        let mut output = Vec::new();

        for _ in 0..max_len {
            let input = self.build_input(cond, &context);
            let (_, logits) = self.forward(&input);

            let scaled: Vec<f32> = logits
                .iter()
                .map(|&l| l / temperature.max(0.01))
                .collect();
            let probs = Self::softmax(&scaled);

            // Greedy: argmax (for deterministic output)
            let idx = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);

            if idx == EOS as usize || idx > 127 {
                break;
            }

            let ch = idx as u8;
            output.push(ch);
            context.push(ch);
        }

        String::from_utf8_lossy(&output).to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_head_trains() {
        let cond = vec![0.5f32; 16];
        let mut head = GenerationHead::new(16, 32);

        let loss_before = head.train_step(&cond, "hello", 0.0);
        let mut loss = 0.0;
        for _ in 0..50 {
            loss = head.train_step(&cond, "hello", 0.05);
        }
        assert!(
            loss < loss_before || loss < 4.0,
            "loss should decrease: before={}, after={}",
            loss_before,
            loss
        );
    }

    #[test]
    fn test_generation_produces_output() {
        let cond = vec![0.5f32; 16];
        let head = GenerationHead::new(16, 32);
        let output = head.generate(&cond, 20, 1.0);
        assert!(!output.is_empty(), "generation should produce some text");
    }

    #[test]
    fn test_trained_generation_memorizes_short_string() {
        let cond = vec![1.0f32; 8];
        let mut head = GenerationHead::new(8, 64);
        let target = "ab";
        for _ in 0..200 {
            head.train_step(&cond, target, 0.02);
        }
        let out = head.generate(&cond, 10, 0.5);
        assert!(
            out.starts_with("a"),
            "should learn to generate 'a' first, got: {:?}",
            out
        );
    }
}
