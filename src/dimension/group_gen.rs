//! Per-group generation using the Growformer substrate with dictionary-based
//! binary token prediction.
//!
//! Instead of autoregressive character-by-character generation (200 forward
//! passes per sample), this encodes the target as a flat binary vector of
//! token IDs and predicts all tokens in a SINGLE forward pass.
//!
//! Architecture per group:
//!   input  = bridged_embedding (64d)
//!   hidden = 128 neurons × 2 layers, KWTA k=32 (25% sparse)
//!   output = MAX_TOKENS × bits_per_token neurons (adaptive to dict size)
//!
//! bits_per_token is computed from the dictionary size:
//!   ≤256 entries → 8 bits → 256 output neurons
//!   ≤1024 entries → 10 bits → 320 output neurons
//!   ≤2048 entries → 11 bits → 352 output neurons
//!
//! Per-group dictionaries keep vocabulary tight and domain-specific.
//! Pruning stops after a warmup fraction to preserve learned capacity.
//!
//! Training: ONE train_tick per sample (not per character).
//! Inference: ONE forward pass → decode all tokens → dictionary lookup → text.

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::environment::NeuralEnvironment;
use crate::spectral::{TokenDictionary, hamming_parity_bits, hamming_encode, hamming_decode};
use crate::types::EnvironmentConfig;

pub const GEN_COND_DIM: usize = 64;
pub const MAX_TOKENS: usize = 128;
pub const GEN_HIDDEN: usize = 256;
pub const GEN_K: usize = 64;

/// Overrides for auto-configured training. When `None`, defaults are used.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GenEnvOverrides {
    pub max_tokens: Option<usize>,
    pub hidden: Option<usize>,
    pub k: Option<usize>,
    pub max_synapses: Option<usize>,
    pub energy_budget: Option<f32>,
}

/// Compute bits needed for a dictionary of the given size.
pub fn bits_for_dict(dict_len: usize) -> usize {
    if dict_len <= 1 {
        return 1;
    }
    let max_id = dict_len - 1;
    (usize::BITS - max_id.leading_zeros()) as usize
}

#[derive(Clone, Serialize, Deserialize)]
pub struct GroupGenEnv {
    pub env: NeuralEnvironment,
    pub dictionary: TokenDictionary,
    pub bits_per_token: usize,
    /// Total bits per token slot including ECC parity bits.
    pub coded_bits_per_token: usize,
    pub output_dim: usize,
    pub frozen: bool,
}

impl GroupGenEnv {
    pub fn new(dictionary: TokenDictionary, rng: &mut impl Rng) -> Self {
        Self::new_with_overrides(dictionary, &GenEnvOverrides::default(), rng)
    }

    pub fn new_with_overrides(dictionary: TokenDictionary, ov: &GenEnvOverrides, rng: &mut impl Rng) -> Self {
        let max_tok = ov.max_tokens.unwrap_or(MAX_TOKENS);
        let hidden = ov.hidden.unwrap_or(GEN_HIDDEN);
        let k = ov.k.unwrap_or(GEN_K);
        let bits_per_token = bits_for_dict(dictionary.len());
        let parity_bits = hamming_parity_bits(bits_per_token);
        let coded_bits_per_token = bits_per_token + parity_bits;
        let output_dim = max_tok * coded_bits_per_token;
        let mut config = gen_env_config();
        config.competitive_k = k;
        if let Some(ms) = ov.max_synapses { config.max_synapses_per_neuron = ms; }
        if let Some(eb) = ov.energy_budget { config.energy_budget_per_neuron = eb; }
        let mut env = NeuralEnvironment::new(config);
        env.build_layers(&[GEN_COND_DIM, hidden, hidden, output_dim], rng);
        Self {
            env,
            dictionary,
            bits_per_token,
            coded_bits_per_token,
            output_dim,
            frozen: false,
        }
    }

    /// Set pruning to stop after this many ticks.
    /// Call before training begins to lock in the warmup window.
    pub fn set_prune_stop_tick(&mut self, tick: u64) {
        self.env.config.prune_stop_tick = tick;
    }

    /// Encode a token ID as Gray-coded + Hamming ECC binary values.
    /// Gray coding ensures adjacent token IDs differ by only 1 bit.
    /// Hamming parity bits enable single-error correction at decode time.
    fn id_to_bits(&self, id: u16) -> Vec<f32> {
        let gray = self.dictionary.to_gray_id(id);
        let mut data_bits = Vec::with_capacity(self.bits_per_token);
        for i in 0..self.bits_per_token {
            data_bits.push(if (gray >> i) & 1 == 1 { 1u8 } else { 0u8 });
        }
        let codeword = hamming_encode(&data_bits, self.bits_per_token);
        codeword.iter().map(|&b| b as f32).collect()
    }

    /// Hard decode: threshold each bit at 0.5, reconstruct Gray code, convert
    /// back to token ID. Kept as fallback; soft decode is preferred.
    #[allow(dead_code)]
    fn bits_to_id_hard(bits: &[f32], dict: &TokenDictionary, dict_size: usize) -> u16 {
        let mut gray = 0u16;
        for (i, &b) in bits.iter().enumerate() {
            if b > 0.5 {
                gray |= 1 << i;
            }
        }
        let id = dict.from_gray_id(gray);
        id.min(dict_size.saturating_sub(1) as u16)
    }

    /// Soft decode: find the dictionary entry whose Gray-coded binary
    /// representation is closest to the raw sigmoid outputs (Euclidean
    /// distance). Tolerates 1-3 bit errors that hard thresholding would
    /// propagate. With Gray coding, a 1-bit error maps to an adjacent
    /// (semantically similar) token instead of an arbitrary one.
    fn bits_to_id_soft(bits: &[f32], dict: &TokenDictionary, dict_size: usize, bpt: usize) -> u16 {
        let mut best_id = 0u16;
        let mut best_dist = f32::MAX;
        for candidate in 0..dict_size as u16 {
            let gray = dict.to_gray_id(candidate);
            let mut dist = 0.0f32;
            for i in 0..bpt {
                let target_bit = if (gray >> i) & 1 == 1 { 1.0f32 } else { 0.0 };
                let d = bits.get(i).copied().unwrap_or(0.0) - target_bit;
                dist += d * d;
            }
            if dist < best_dist {
                best_dist = dist;
                best_id = candidate;
            }
        }
        best_id
    }

    /// Encode a full target text into a flat binary target vector (with ECC).
    fn encode_target(&self, text: &str) -> Vec<f32> {
        let token_ids = self.dictionary.encode(text);
        let mut target = vec![0.0f32; self.output_dim];
        let cbpt = self.coded_bits_per_token;
        let max_tok = self.output_dim / cbpt.max(1);
        for (pos, &id) in token_ids.iter().take(max_tok).enumerate() {
            let bits = self.id_to_bits(id);
            let offset = pos * cbpt;
            let len = bits.len().min(cbpt);
            target[offset..offset + len].copy_from_slice(&bits[..len]);
        }
        target
    }

    /// Decode a flat binary output vector into text via dictionary lookup.
    /// Pipeline: raw sigmoids → Hamming ECC correction → Gray decode →
    /// nearest-neighbor soft decode. Stops at EOS or low confidence.
    fn decode_output(&self, output: &[f32]) -> String {
        let dict_size = self.dictionary.len();
        let cbpt = self.coded_bits_per_token;
        let bpt = self.bits_per_token;
        let mut ids = Vec::new();
        let max_tok = self.output_dim / cbpt.max(1);
        let min_tokens = 3;
        for pos in 0..max_tok {
            let offset = pos * cbpt;
            if offset + cbpt > output.len() {
                break;
            }
            let slot = &output[offset..offset + cbpt];

            let max_confidence = slot
                .iter()
                .map(|&v| (v - 0.5).abs())
                .fold(0.0f32, f32::max);
            if pos >= min_tokens && max_confidence < 0.05 {
                break;
            }

            // Step 1: Hard-threshold to get codeword bits for ECC
            let hard_bits: Vec<u8> = slot.iter().map(|&v| if v > 0.5 { 1u8 } else { 0u8 }).collect();
            // Step 2: Hamming correction on the codeword
            let corrected_data = hamming_decode(&hard_bits, bpt);
            // Step 3: Reconstruct soft values from corrected data for soft decode
            let corrected_soft: Vec<f32> = corrected_data.iter().map(|&b| b as f32).collect();
            // Step 4: Soft decode (nearest-neighbor on data bits, Gray-aware)
            let id = Self::bits_to_id_soft(&corrected_soft, &self.dictionary, dict_size, bpt);

            if pos >= min_tokens && id == 0 {
                break;
            }
            if id > 0 {
                ids.push(id);
            }
        }
        self.dictionary.decode(&ids)
    }

    fn binary_cross_entropy(output: &[f32], target: &[f32]) -> f32 {
        let mut loss = 0.0f32;
        let mut count = 0;
        for (o, t) in output.iter().zip(target.iter()) {
            let p = o.clamp(1e-7, 1.0 - 1e-7);
            loss -= t * p.ln() + (1.0 - t) * (1.0 - p).ln();
            count += 1;
        }
        if count > 0 {
            loss / count as f32
        } else {
            0.0
        }
    }

    /// Single-pass training: one train_tick per sample.
    /// Encodes entire target as binary token IDs, trains in one substrate tick.
    /// Then runs a second gradient-only pass with an EOS-emphasized target to
    /// strongly push trailing positions toward zero.
    pub fn train_step(&mut self, cond: &[f32], target: &str, rng: &mut impl Rng) -> f32 {
        if self.frozen {
            return 0.0;
        }
        let target_vec = self.encode_target(target);
        if target_vec.iter().all(|&v| v == 0.0) {
            return 0.0;
        }

        let mut input = vec![0.0f32; GEN_COND_DIM];
        for (i, v) in cond.iter().enumerate().take(GEN_COND_DIM) {
            input[i] = *v;
        }

        // Primary learning tick: full substrate dynamics + content
        let result = self.env.train_tick(&input, &target_vec, rng);
        let loss = Self::binary_cross_entropy(&result.output, &target_vec);

        // Count actual content tokens to find where EOS begins
        let token_ids = self.dictionary.encode(target);
        let max_tok = self.output_dim / self.coded_bits_per_token.max(1);
        let content_tokens = token_ids.len().min(max_tok);
        if content_tokens < max_tok {
            let eos_target = vec![0.0f32; self.output_dim];
            self.env.train_tick_gradient_only(&input, &eos_target, rng);
        }

        loss
    }

    /// Single-pass generation: one forward pass, decode all tokens at once.
    pub fn generate(&mut self, cond: &[f32], _max_len: usize, _temperature: f32) -> String {
        let mut input = vec![0.0f32; GEN_COND_DIM];
        for (i, v) in cond.iter().enumerate().take(GEN_COND_DIM) {
            input[i] = *v;
        }

        let output = self.env.predict(&input);
        self.decode_output(&output)
    }

    /// Evaluate loss without modifying the network.
    pub fn eval_loss(&mut self, cond: &[f32], target: &str) -> f32 {
        let target_vec = self.encode_target(target);
        if target_vec.iter().all(|&v| v == 0.0) {
            return 0.0;
        }

        let mut input = vec![0.0f32; GEN_COND_DIM];
        for (i, v) in cond.iter().enumerate().take(GEN_COND_DIM) {
            input[i] = *v;
        }

        let output = self.env.predict(&input);
        Self::binary_cross_entropy(&output, &target_vec)
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

    pub fn max_tokens(&self) -> usize {
        self.output_dim / self.coded_bits_per_token.max(1)
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
        max_synapses_per_neuron: 200,
        energy_budget_per_neuron: 25.0,
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
        // Engram consolidation: protect memory traces from overwriting and pruning.
        // Co-activated synapses accumulate consolidation; high consolidation = reduced LR + prune immunity.
        engram_enabled: true,
        engram_activation_threshold: 0.5,
        engram_increment: 0.005,
        engram_cap: 1.0,
        engram_lr_scale: 0.85,
        engram_prune_threshold: 0.3,
        dendritic_branches: 1,
        ..EnvironmentConfig::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn test_dict() -> TokenDictionary {
        TokenDictionary::build(
            &[
                "reset your password",
                "check your email",
                "update your profile settings",
                "contact customer support team",
                "how do I change my account name",
            ],
            500,
        )
    }

    #[test]
    fn test_id_to_bits_roundtrip() {
        let dict = test_dict();
        let mut rng = StdRng::seed_from_u64(42);
        let env = GroupGenEnv::new(dict, &mut rng);
        let bpt = env.bits_per_token;
        let dict_size = env.dictionary.len();
        for id in 0u16..(dict_size as u16) {
            let coded_bits = env.id_to_bits(id);
            // Decode through ECC + soft decode (same pipeline as decode_output)
            let hard: Vec<u8> = coded_bits.iter().map(|&v| if v > 0.5 { 1u8 } else { 0u8 }).collect();
            let corrected = hamming_decode(&hard, bpt);
            let soft: Vec<f32> = corrected.iter().map(|&b| b as f32).collect();
            let back = GroupGenEnv::bits_to_id_soft(&soft, &env.dictionary, dict_size, bpt);
            assert_eq!(id, back, "roundtrip failed for {}", id);
        }
    }

    #[test]
    fn test_bits_for_dict() {
        assert_eq!(bits_for_dict(2), 1);
        assert_eq!(bits_for_dict(256), 8);
        assert_eq!(bits_for_dict(512), 9);
        assert_eq!(bits_for_dict(1024), 10);
        assert_eq!(bits_for_dict(1025), 11);
        assert_eq!(bits_for_dict(2048), 11);
    }

    #[test]
    fn test_encode_decode_target() {
        let dict = test_dict();
        let mut rng = StdRng::seed_from_u64(42);
        let env = GroupGenEnv::new(dict, &mut rng);

        let text = "reset your password";
        let target = env.encode_target(text);
        assert_eq!(target.len(), env.output_dim);

        let decoded = env.decode_output(&target);
        assert_eq!(decoded, text, "encode/decode roundtrip failed");
    }

    #[test]
    fn test_new_env_topology() {
        let dict = test_dict();
        let expected_bits = bits_for_dict(dict.len());
        let expected_parity = hamming_parity_bits(expected_bits);
        let expected_coded = expected_bits + expected_parity;
        let expected_out = MAX_TOKENS * expected_coded;
        let mut rng = StdRng::seed_from_u64(42);
        let env = GroupGenEnv::new(dict, &mut rng);
        assert_eq!(env.bits_per_token, expected_bits);
        assert_eq!(env.coded_bits_per_token, expected_coded);
        assert_eq!(env.output_dim, expected_out);
        assert_eq!(env.env.layers.len(), 4);
        assert_eq!(env.env.layers[0].len(), GEN_COND_DIM);
        assert_eq!(env.env.layers[1].len(), GEN_HIDDEN);
        assert_eq!(env.env.layers[2].len(), GEN_HIDDEN);
        assert_eq!(env.env.layers[3].len(), expected_out);
    }

    #[test]
    fn test_train_and_generate() {
        let dict = test_dict();
        let mut rng = StdRng::seed_from_u64(42);
        let mut env = GroupGenEnv::new(dict, &mut rng);
        let cond = vec![0.1f32; GEN_COND_DIM];
        let loss = env.train_step(&cond, "reset your password", &mut rng);
        assert!(loss > 0.0, "loss should be positive: {}", loss);
        let _out = env.generate(&cond, 100, 0.8);
    }

    #[test]
    fn test_loss_decreases_over_training() {
        let dict = test_dict();
        let mut rng = StdRng::seed_from_u64(42);
        let mut env = GroupGenEnv::new(dict, &mut rng);
        let cond = vec![0.1f32; GEN_COND_DIM];
        let target = "reset your password";
        let loss_0 = env.train_step(&cond, target, &mut rng);
        for _ in 0..100 {
            env.train_step(&cond, target, &mut rng);
        }
        let loss_100 = env.train_step(&cond, target, &mut rng);
        println!("loss: {} -> {}", loss_0, loss_100);
        assert!(
            loss_100 < loss_0,
            "loss should decrease: {} -> {}",
            loss_0,
            loss_100
        );
    }

    #[test]
    fn test_freeze_prevents_training() {
        let dict = test_dict();
        let mut rng = StdRng::seed_from_u64(42);
        let mut env = GroupGenEnv::new(dict, &mut rng);
        env.freeze();
        let loss = env.train_step(&vec![0.1; GEN_COND_DIM], "hello", &mut rng);
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_single_pass_speed() {
        let dict = test_dict();
        let mut rng = StdRng::seed_from_u64(42);
        let mut env = GroupGenEnv::new(dict, &mut rng);
        let cond = vec![0.1f32; GEN_COND_DIM];
        let target = "update your profile settings";

        let start = std::time::Instant::now();
        let mut total_loss = 0.0f32;
        for _ in 0..100 {
            total_loss += env.train_step(&cond, target, &mut rng);
        }
        let elapsed = start.elapsed();
        println!(
            "100 train steps in {:?} ({:.1}ms/step), avg loss={:.4}",
            elapsed,
            elapsed.as_millis() as f64 / 100.0,
            total_loss / 100.0
        );
    }
}
