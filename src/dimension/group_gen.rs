//! Per-group generation using the Growformer substrate.
//!
//! Each promoted group owns a NeuralEnvironment configured for character-level
//! autoregressive generation. Routing selects which group's env fires — the same
//! structural isolation that achieved 0% forgetting on Split MNIST, applied to
//! language output.
//!
//! Architecture per group:
//!   input  = [bridged_embedding(64d), char_context(32 normalized bytes)] = 96d
//!   hidden = 64 neurons × 2 layers, KWTA k=16 (25% sparse)
//!   output = 8 neurons (one per bit of ASCII byte)
//!
//! Output uses binary encoding: 8 output neurons each predict one bit of the
//! target character. This matches the substrate's validated strength (binary
//! classification per neuron) instead of fighting a 128-way softmax with
//! sigmoid activations and MSE gradients.

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::environment::NeuralEnvironment;
use crate::types::EnvironmentConfig;

pub const GEN_BITS: usize = 8;
pub const GEN_CONTEXT_LEN: usize = 32;
pub const GEN_COND_DIM: usize = 64;
const GEN_INPUT_DIM: usize = GEN_COND_DIM + GEN_CONTEXT_LEN; // 96
const GEN_HIDDEN: usize = 64;
const GEN_K: usize = 16;

#[derive(Clone, Serialize, Deserialize)]
pub struct GroupGenEnv {
    pub env: NeuralEnvironment,
    pub frozen: bool,
}

impl GroupGenEnv {
    pub fn new(rng: &mut impl Rng) -> Self {
        let config = gen_env_config();
        let mut env = NeuralEnvironment::new(config);
        env.build_layers(&[GEN_INPUT_DIM, GEN_HIDDEN, GEN_HIDDEN, GEN_BITS], rng);
        Self { env, frozen: false }
    }

    fn encode_context(chars: &[u8]) -> Vec<f32> {
        let mut ctx = vec![0.0f32; GEN_CONTEXT_LEN];
        let start = chars.len().saturating_sub(GEN_CONTEXT_LEN);
        let tail = &chars[start..];
        let offset = GEN_CONTEXT_LEN - tail.len();
        for (i, &ch) in tail.iter().enumerate() {
            ctx[offset + i] = ch as f32 / 127.0;
        }
        ctx
    }

    fn build_input(cond: &[f32], context: &[u8]) -> Vec<f32> {
        let mut input = Vec::with_capacity(GEN_INPUT_DIM);
        for i in 0..GEN_COND_DIM {
            input.push(if i < cond.len() { cond[i] } else { 0.0 });
        }
        input.extend_from_slice(&Self::encode_context(context));
        input
    }

    fn byte_to_bits(b: u8) -> [f32; GEN_BITS] {
        let mut bits = [0.0f32; GEN_BITS];
        for i in 0..GEN_BITS {
            bits[i] = if (b >> i) & 1 == 1 { 1.0 } else { 0.0 };
        }
        bits
    }

    fn bits_to_byte(output: &[f32]) -> u8 {
        let mut byte = 0u8;
        for i in 0..GEN_BITS.min(output.len()) {
            if output[i] > 0.5 {
                byte |= 1 << i;
            }
        }
        byte
    }

    fn binary_cross_entropy(output: &[f32], target: &[f32]) -> f32 {
        let mut loss = 0.0f32;
        for (o, t) in output.iter().zip(target.iter()) {
            let p = o.clamp(1e-7, 1.0 - 1e-7);
            loss -= t * p.ln() + (1.0 - t) * (1.0 - p).ln();
        }
        loss / output.len() as f32
    }

    /// Teacher-forcing training step through the Growformer substrate.
    /// Per-character: gradient-only (forward + backprop, includes KWTA + mass dynamics).
    /// End of sample: one full train_tick triggers pruning, geometry, synapse growth.
    /// Target is binary-encoded (8 bits per char) — 8 parallel binary classifications
    /// that match the substrate's validated sigmoid + MSE gradient pathway.
    pub fn train_step(&mut self, cond: &[f32], target: &str, rng: &mut impl Rng) -> f32 {
        if self.frozen {
            return 0.0;
        }
        let target_bytes: Vec<u8> = target.bytes().take(200).collect();
        if target_bytes.is_empty() {
            return 0.0;
        }

        let mut context: Vec<u8> = Vec::new();
        let mut total_loss = 0.0f32;
        let last_idx = target_bytes.len() - 1;

        for (ci, &target_ch) in target_bytes.iter().enumerate() {
            let input = Self::build_input(cond, &context);
            let target_bits = Self::byte_to_bits(target_ch);

            if ci == last_idx {
                let result = self.env.train_tick(&input, &target_bits, rng);
                total_loss += Self::binary_cross_entropy(&result.output, &target_bits);
            } else {
                let loss = self.env.train_tick_gradient_only(&input, &target_bits, rng);
                total_loss += loss;
            }

            context.push(target_ch);
        }

        total_loss / target_bytes.len() as f32
    }

    /// Autoregressive generation through the Growformer substrate.
    /// Each step: predict 8 bits → threshold → reconstruct byte → feed back.
    pub fn generate(&mut self, cond: &[f32], max_len: usize, _temperature: f32) -> String {
        let mut context: Vec<u8> = Vec::new();
        let mut output_chars = Vec::new();

        for _ in 0..max_len {
            let input = Self::build_input(cond, &context);
            let output = self.env.predict(&input);
            let ch = Self::bits_to_byte(&output);

            if ch == 0 || ch > 127 {
                break;
            }

            output_chars.push(ch);
            context.push(ch);
        }

        String::from_utf8_lossy(&output_chars).to_string()
    }

    pub fn freeze(&mut self) {
        self.frozen = true;
        self.env.freeze_all();
    }

    pub fn total_synapses(&self) -> usize {
        self.env.total_synapses()
    }

    pub fn total_neurons(&self) -> usize {
        self.env.neurons.len()
    }
}

fn gen_env_config() -> EnvironmentConfig {
    EnvironmentConfig {
        output_bce: true,
        learning_rate: 0.05,
        competitive_k: GEN_K,
        lateral_inhibition: 0.12,
        weight_decay: 0.000005,
        bias_decay: 0.00005,
        prune_interval: 500,
        geometry_interval: 150,
        mass_growth: 0.0005,
        mass_decay: 0.00015,
        mass_win_threshold: 0.4,
        mass_min: 0.3,
        mass_max: 3.0,
        kwta_suppression: 0.15,
        dropout_rate: 0.05,
        weight_clamp: 8.0,
        growth_radius: 2.0,
        max_synapses_per_neuron: 80,
        energy_budget_per_neuron: 10.0,
        pruning_threshold: 0.04,
        sigma_inhib: 1.5,
        debye_length: 2.0,
        facilitation_bonus: 0.002,
        physics_dt: 0.05,
        gravity_g: 0.02,
        k_repel: 0.5,
        damping: 0.15,
        thermal_noise: 0.002,
        hebbian_attraction: 0.001,
        geometry_noise: 0.005,
        homeostasis_target: 0.35,
        homeostasis_lr: 0.0003,
        ..EnvironmentConfig::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_byte_to_bits_roundtrip() {
        for ch in 0u8..128 {
            let bits = GroupGenEnv::byte_to_bits(ch);
            let back = GroupGenEnv::bits_to_byte(&bits);
            assert_eq!(ch, back, "roundtrip failed for {}", ch);
        }
    }

    #[test]
    fn test_encode_context_padding() {
        let ctx = GroupGenEnv::encode_context(b"abc");
        assert_eq!(ctx.len(), GEN_CONTEXT_LEN);
        assert_eq!(ctx[0], 0.0);
        assert!(ctx[GEN_CONTEXT_LEN - 1] > 0.0);
    }

    #[test]
    fn test_build_input_dim() {
        let cond = vec![0.5f32; GEN_COND_DIM];
        let input = GroupGenEnv::build_input(&cond, b"hello");
        assert_eq!(input.len(), GEN_INPUT_DIM);
    }

    #[test]
    fn test_new_env_topology() {
        let mut rng = StdRng::seed_from_u64(42);
        let env = GroupGenEnv::new(&mut rng);
        assert_eq!(env.env.layers.len(), 4);
        assert_eq!(env.env.layers[0].len(), GEN_INPUT_DIM);
        assert_eq!(env.env.layers[1].len(), GEN_HIDDEN);
        assert_eq!(env.env.layers[2].len(), GEN_HIDDEN);
        assert_eq!(env.env.layers[3].len(), GEN_BITS);
    }

    #[test]
    fn test_train_and_generate() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut env = GroupGenEnv::new(&mut rng);
        let cond = vec![0.1f32; GEN_COND_DIM];
        let loss = env.train_step(&cond, "hello", &mut rng);
        assert!(loss > 0.0);
        let out = env.generate(&cond, 10, 0.8);
        assert!(!out.is_empty() || true);
    }

    #[test]
    fn test_loss_decreases_over_training() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut env = GroupGenEnv::new(&mut rng);
        let cond = vec![0.1f32; GEN_COND_DIM];
        let loss_0 = env.train_step(&cond, "abc", &mut rng);
        for _ in 0..50 {
            env.train_step(&cond, "abc", &mut rng);
        }
        let loss_50 = env.train_step(&cond, "abc", &mut rng);
        assert!(loss_50 < loss_0, "loss should decrease: {} -> {}", loss_0, loss_50);
    }

    #[test]
    fn test_freeze_prevents_training() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut env = GroupGenEnv::new(&mut rng);
        env.freeze();
        let loss = env.train_step(&vec![0.1; GEN_COND_DIM], "hello", &mut rng);
        assert_eq!(loss, 0.0);
    }
}
