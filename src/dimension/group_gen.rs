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
use std::collections::HashMap;

use crate::environment::NeuralEnvironment;
use crate::spectral::{
    TokenDictionary, hamming_parity_bits, hamming_encode, hamming_decode,
    tokenize, syntax_role, structural_signature, SyntaxRole,
};
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

// ---------------------------------------------------------------------------
// Algebraic Codebook — factored generation via group-theory decomposition
// ---------------------------------------------------------------------------

/// Bits needed to represent `n` distinct values.
fn bits_for_count(n: usize) -> usize {
    if n <= 1 { return 1; }
    (usize::BITS - (n - 1).leading_zeros()) as usize
}

/// A variable position in a response archetype.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArchetypeSlot {
    pub position: usize,
    pub vocab: Vec<u16>,
    pub bits: usize,
}

/// A structural response pattern: fixed tokens + variable slots.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResponseArchetype {
    pub fixed: Vec<(usize, u16)>,
    pub slots: Vec<ArchetypeSlot>,
    pub length: usize,
}

/// Factored representation of the response space for a group.
/// Decomposes response prediction from O(max_tokens × bits_per_token) into
/// O(archetype_bits + num_slots × slot_bits), typically ~80-120 bits
/// instead of ~1400-1900.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlgebraicCodebook {
    pub archetypes: Vec<ResponseArchetype>,
    pub archetype_bits: usize,
    pub max_slot_count: usize,
    pub slot_bit_widths: Vec<usize>,
    pub total_bits: usize,
}

impl AlgebraicCodebook {
    /// Build a codebook from training texts for a single group.
    /// Clusters responses into archetypes, extracts fixed/variable positions.
    pub fn build(texts: &[&str], dictionary: &TokenDictionary, max_archetypes: usize) -> Self {
        let seqs: Vec<Vec<u16>> = texts.iter().map(|t| dictionary.encode(t)).collect();
        if seqs.is_empty() {
            return Self::empty();
        }

        let max_len = seqs.iter().map(|s| s.len()).max().unwrap_or(0);
        let padded: Vec<Vec<u16>> = seqs.iter().map(|s| {
            let mut p = s.clone();
            p.resize(max_len, 0);
            p
        }).collect();

        let clusters = Self::cluster_responses(&padded, max_archetypes.min(padded.len()));
        let archetypes: Vec<ResponseArchetype> = clusters.iter().map(|indices| {
            let cluster_seqs: Vec<&Vec<u16>> = indices.iter().map(|&i| &padded[i]).collect();
            Self::extract_archetype(&cluster_seqs, max_len)
        }).collect();

        let archetype_bits = bits_for_count(archetypes.len().max(1));
        let max_slot_count = archetypes.iter().map(|a| a.slots.len()).max().unwrap_or(0);

        let mut slot_bit_widths = vec![0usize; max_slot_count];
        for arch in &archetypes {
            for (i, slot) in arch.slots.iter().enumerate() {
                slot_bit_widths[i] = slot_bit_widths[i].max(slot.bits);
            }
        }

        let total_bits = archetype_bits + slot_bit_widths.iter().sum::<usize>();

        Self { archetypes, archetype_bits, max_slot_count, slot_bit_widths, total_bits }
    }

    pub fn empty() -> Self {
        Self { archetypes: vec![], archetype_bits: 1, max_slot_count: 0, slot_bit_widths: vec![], total_bits: 1 }
    }

    /// Encode a token sequence into algebraic bits.
    pub fn encode(&self, token_ids: &[u16]) -> Vec<f32> {
        let (arch_idx, slot_values) = self.match_best(token_ids);
        let mut bits = vec![0.0f32; self.total_bits];

        for i in 0..self.archetype_bits {
            bits[i] = if (arch_idx >> i) & 1 == 1 { 1.0 } else { 0.0 };
        }

        let mut offset = self.archetype_bits;
        for (slot_idx, &val) in slot_values.iter().enumerate() {
            let sbits = self.slot_bit_widths.get(slot_idx).copied().unwrap_or(0);
            for i in 0..sbits {
                if offset + i < bits.len() {
                    bits[offset + i] = if (val >> i) & 1 == 1 { 1.0 } else { 0.0 };
                }
            }
            offset += sbits;
        }
        bits
    }

    /// Decode algebraic bits back to token IDs using nearest-neighbor soft
    /// decode for both archetype selection and slot filling.
    pub fn decode(&self, bits: &[f32]) -> Vec<u16> {
        if self.archetypes.is_empty() {
            return vec![];
        }

        // Soft decode archetype: find closest match
        let arch_bits = &bits[..self.archetype_bits.min(bits.len())];
        let arch_idx = Self::soft_decode_index(arch_bits, self.archetypes.len());
        let arch = &self.archetypes[arch_idx];

        let mut tokens = vec![0u16; arch.length];
        for &(pos, tok) in &arch.fixed {
            if pos < tokens.len() {
                tokens[pos] = tok;
            }
        }

        let mut offset = self.archetype_bits;
        for (slot_idx, slot) in arch.slots.iter().enumerate() {
            let sbits = self.slot_bit_widths.get(slot_idx).copied().unwrap_or(0);
            let end = (offset + sbits).min(bits.len());
            let slot_bits = if offset < bits.len() { &bits[offset..end] } else { &[] as &[f32] };
            let val = Self::soft_decode_index(slot_bits, slot.vocab.len());
            if slot.position < tokens.len() && val < slot.vocab.len() {
                tokens[slot.position] = slot.vocab[val];
            }
            offset += sbits;
        }

        while tokens.last() == Some(&0) {
            tokens.pop();
        }
        tokens.retain(|&t| t != 0);
        tokens
    }

    /// Nearest-neighbor decode for a small index: try all candidates and
    /// pick the one whose binary encoding is closest to the raw sigmoids.
    fn soft_decode_index(bits: &[f32], num_options: usize) -> usize {
        if num_options <= 1 { return 0; }
        let nbits = bits_for_count(num_options);
        let mut best = 0usize;
        let mut best_dist = f32::MAX;
        for candidate in 0..num_options {
            let mut dist = 0.0f32;
            for i in 0..nbits {
                let target = if (candidate >> i) & 1 == 1 { 1.0f32 } else { 0.0 };
                let actual = bits.get(i).copied().unwrap_or(0.0);
                let d = actual - target;
                dist += d * d;
            }
            if dist < best_dist {
                best_dist = dist;
                best = candidate;
            }
        }
        best
    }

    /// Find the best matching archetype and slot values for a token sequence.
    /// Scores by (matching fixed) - 2*(mismatching fixed) + (slot hits) to
    /// prevent a high-fixed-count archetype from stealing texts that belong
    /// to a different cluster.
    fn match_best(&self, token_ids: &[u16]) -> (usize, Vec<usize>) {
        let mut best_arch = 0;
        let mut best_score = i64::MIN;

        for (idx, arch) in self.archetypes.iter().enumerate() {
            let mut score: i64 = 0;
            for &(pos, tok) in &arch.fixed {
                if token_ids.get(pos).copied() == Some(tok) {
                    score += 2;
                } else {
                    score -= 3;
                }
            }
            for slot in &arch.slots {
                let actual = token_ids.get(slot.position).copied().unwrap_or(0);
                if slot.vocab.contains(&actual) {
                    score += 1;
                }
            }
            if score > best_score {
                best_score = score;
                best_arch = idx;
            }
        }

        let arch = &self.archetypes[best_arch];
        let mut slot_values: Vec<usize> = arch.slots.iter().map(|slot| {
            let actual_tok = token_ids.get(slot.position).copied().unwrap_or(0);
            slot.vocab.iter().position(|&t| t == actual_tok).unwrap_or(0)
        }).collect();
        slot_values.resize(self.max_slot_count, 0);

        (best_arch, slot_values)
    }

    /// Greedy clustering by positional token overlap.
    fn cluster_responses(padded: &[Vec<u16>], max_k: usize) -> Vec<Vec<usize>> {
        let n = padded.len();
        if n == 0 { return vec![]; }
        let max_k = max_k.min(n).max(1);
        if max_k == 1 || n <= 3 {
            return vec![(0..n).collect()];
        }

        // Pick medoids: first is index 0, then greedily pick most different
        let mut medoids = vec![0usize];
        for _ in 1..max_k {
            let mut best_idx = 0;
            let mut best_min_dist = 0usize;
            for i in 0..n {
                if medoids.contains(&i) { continue; }
                let min_overlap = medoids.iter().map(|&m| Self::overlap(&padded[i], &padded[m])).min().unwrap_or(0);
                let dist = padded[i].len().saturating_sub(min_overlap);
                if dist > best_min_dist {
                    best_min_dist = dist;
                    best_idx = i;
                }
            }
            if best_min_dist == 0 { break; }
            medoids.push(best_idx);
        }

        let mut clusters: Vec<Vec<usize>> = vec![vec![]; medoids.len()];
        for i in 0..n {
            let mut best_k = 0;
            let mut best_overlap = 0;
            for (k, &m) in medoids.iter().enumerate() {
                let ov = Self::overlap(&padded[i], &padded[m]);
                if ov > best_overlap {
                    best_overlap = ov;
                    best_k = k;
                }
            }
            clusters[best_k].push(i);
        }
        clusters.retain(|c| !c.is_empty());
        clusters
    }

    fn overlap(a: &[u16], b: &[u16]) -> usize {
        a.iter().zip(b.iter()).filter(|(x, y)| x == y && **x != 0).count()
    }

    /// Build a syntax-aware codebook for code groups. Instead of pure positional
    /// overlap, clusters by **structural signature** (keywords + punctuation kept,
    /// identifiers/literals replaced with role placeholders). Keywords and
    /// structural punctuation are auto-fixed; only identifiers and literals
    /// become slots. Dramatically reduces slot count for code.
    pub fn build_syntax_aware(texts: &[&str], dictionary: &TokenDictionary, max_archetypes: usize) -> Self {
        if texts.is_empty() {
            return Self::empty();
        }

        // Tokenize raw text to get syntax roles alongside dictionary encoding
        let raw_tokens: Vec<Vec<String>> = texts.iter().map(|t| tokenize(t)).collect();
        let signatures: Vec<Vec<String>> = raw_tokens.iter().map(|toks| structural_signature(toks)).collect();
        let seqs: Vec<Vec<u16>> = texts.iter().map(|t| dictionary.encode(t)).collect();
        let roles: Vec<Vec<SyntaxRole>> = raw_tokens.iter().map(|toks| {
            toks.iter().map(|t| syntax_role(t)).collect()
        }).collect();

        // Cluster by structural signature similarity
        let sig_strings: Vec<String> = signatures.iter().map(|s| s.join(" ")).collect();
        let mut sig_clusters: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, sig) in sig_strings.iter().enumerate() {
            sig_clusters.entry(sig.clone()).or_default().push(i);
        }

        // If too many unique signatures, merge the smallest/most-similar clusters
        // until we reach max_archetypes. Merges smallest clusters first (preserves
        // the most common structural patterns as distinct archetypes).
        let mut cluster_list: Vec<Vec<usize>> = sig_clusters.into_values().collect();
        cluster_list.sort_by_key(|c| std::cmp::Reverse(c.len()));

        while cluster_list.len() > max_archetypes {
            // Merge the two smallest clusters
            let last = cluster_list.pop().unwrap();
            if let Some(second_last) = cluster_list.last_mut() {
                second_last.extend(last);
            } else {
                cluster_list.push(last);
                break;
            }
            // Re-sort so smallest are at the end
            cluster_list.sort_by_key(|c| std::cmp::Reverse(c.len()));
        }

        let max_len = seqs.iter().map(|s| s.len()).max().unwrap_or(0);

        // Extract archetypes using syntax roles
        let archetypes: Vec<ResponseArchetype> = cluster_list.iter().map(|indices| {
            let cluster_seqs: Vec<&Vec<u16>> = indices.iter().map(|&i| &seqs[i]).collect();
            let cluster_roles: Vec<&Vec<SyntaxRole>> = indices.iter().map(|&i| &roles[i]).collect();
            Self::extract_archetype_syntax(&cluster_seqs, &cluster_roles, max_len)
        }).collect();

        let archetype_bits = bits_for_count(archetypes.len().max(1));
        let max_slot_count = archetypes.iter().map(|a| a.slots.len()).max().unwrap_or(0);

        let mut slot_bit_widths = vec![0usize; max_slot_count];
        for arch in &archetypes {
            for (i, slot) in arch.slots.iter().enumerate() {
                slot_bit_widths[i] = slot_bit_widths[i].max(slot.bits);
            }
        }

        let total_bits = archetype_bits + slot_bit_widths.iter().sum::<usize>();

        Self { archetypes, archetype_bits, max_slot_count, slot_bit_widths, total_bits }
    }

    /// Extract an archetype using syntax role awareness. Keywords and structural
    /// punctuation are always fixed (regardless of frequency). Only identifiers
    /// and literals at variable positions become slots.
    fn extract_archetype_syntax(
        seqs: &[&Vec<u16>],
        roles: &[&Vec<SyntaxRole>],
        max_len: usize,
    ) -> ResponseArchetype {
        let n = seqs.len().max(1);
        let length = seqs.iter().map(|s| {
            s.iter().rposition(|&t| t != 0).map(|p| p + 1).unwrap_or(0)
        }).max().unwrap_or(0).min(max_len);

        let mut fixed = Vec::new();
        let mut slots = Vec::new();

        for pos in 0..length {
            let mut freq: HashMap<u16, usize> = HashMap::new();
            let mut role_freq: HashMap<SyntaxRole, usize> = HashMap::new();
            for (si, seq) in seqs.iter().enumerate() {
                let tok = seq.get(pos).copied().unwrap_or(0);
                *freq.entry(tok).or_default() += 1;
                if let Some(role_vec) = roles.get(si) {
                    if let Some(&role) = role_vec.get(pos) {
                        *role_freq.entry(role).or_default() += 1;
                    }
                }
            }

            let (&most_common, &count) = freq.iter().max_by_key(|(_, &c)| c).unwrap();
            let dominant_role = role_freq.iter().max_by_key(|(_, &c)| c).map(|(&r, _)| r);

            // Keywords and structure are ALWAYS fixed if they appear in majority
            let is_structural = matches!(
                dominant_role,
                Some(SyntaxRole::Keyword) | Some(SyntaxRole::Structure) | Some(SyntaxRole::Operator)
            );

            if is_structural && count as f32 / n as f32 > 0.3 && most_common != 0 {
                // Structural token: fix even at lower threshold (30% vs 50%)
                fixed.push((pos, most_common));
            } else if count as f32 / n as f32 > 0.5 && most_common != 0 {
                // Non-structural but consistent: also fix
                fixed.push((pos, most_common));
            } else {
                let mut vocab: Vec<u16> = freq.keys().copied().filter(|&t| t != 0).collect();
                vocab.sort();
                vocab.dedup();
                if vocab.is_empty() { continue; }
                let bits = bits_for_count(vocab.len().max(2));
                slots.push(ArchetypeSlot { position: pos, vocab, bits });
            }
        }

        ResponseArchetype { fixed, slots, length }
    }

    /// Extract an archetype from a cluster of aligned token sequences.
    fn extract_archetype(seqs: &[&Vec<u16>], max_len: usize) -> ResponseArchetype {
        let n = seqs.len().max(1);
        let length = seqs.iter().map(|s| {
            s.iter().rposition(|&t| t != 0).map(|p| p + 1).unwrap_or(0)
        }).max().unwrap_or(0).min(max_len);

        let mut fixed = Vec::new();
        let mut slots = Vec::new();

        for pos in 0..length {
            let mut freq: HashMap<u16, usize> = HashMap::new();
            for seq in seqs {
                let tok = seq.get(pos).copied().unwrap_or(0);
                *freq.entry(tok).or_default() += 1;
            }

            let (&most_common, &count) = freq.iter().max_by_key(|(_, &c)| c).unwrap();

            if count as f32 / n as f32 > 0.5 && most_common != 0 {
                fixed.push((pos, most_common));
            } else {
                let mut vocab: Vec<u16> = freq.keys().copied().filter(|&t| t != 0).collect();
                vocab.sort();
                vocab.dedup();
                if vocab.is_empty() { continue; }
                let bits = bits_for_count(vocab.len().max(2));
                slots.push(ArchetypeSlot { position: pos, vocab, bits });
            }
        }

        ResponseArchetype { fixed, slots, length }
    }
}

// ---------------------------------------------------------------------------
// GroupGenEnv
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize, Deserialize)]
pub struct GroupGenEnv {
    pub env: NeuralEnvironment,
    pub dictionary: TokenDictionary,
    pub bits_per_token: usize,
    /// Total bits per token slot including ECC parity bits (raw binary mode).
    pub coded_bits_per_token: usize,
    pub output_dim: usize,
    pub frozen: bool,
    /// When present, generation uses algebraic (factored) encoding instead
    /// of raw binary token prediction. Reduces output from ~1900 to ~80-120 bits.
    #[serde(default)]
    pub codebook: Option<AlgebraicCodebook>,
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
            codebook: None,
        }
    }

    /// Create an algebraic generation environment. The codebook factorizes
    /// the response space so the substrate only predicts ~80-120 bits
    /// (archetype + slot values) instead of ~1400-1900 raw token bits.
    /// Requires training texts to extract archetypes before construction.
    pub fn new_algebraic(
        dictionary: TokenDictionary,
        codebook: AlgebraicCodebook,
        ov: &GenEnvOverrides,
        rng: &mut impl Rng,
    ) -> Self {
        let hidden = ov.hidden.unwrap_or(GEN_HIDDEN);
        let k = ov.k.unwrap_or(GEN_K);
        let output_dim = codebook.total_bits;
        let bits_per_token = bits_for_dict(dictionary.len());
        let coded_bits_per_token = bits_per_token; // not used in algebraic mode
        let mut config = gen_env_config();
        config.competitive_k = k.min(hidden / 2);
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
            codebook: Some(codebook),
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

    /// Encode a full target text into a flat binary target vector.
    /// Uses algebraic codebook when present, otherwise raw binary with ECC.
    fn encode_target(&self, text: &str) -> Vec<f32> {
        let token_ids = self.dictionary.encode(text);

        if let Some(ref cb) = self.codebook {
            return cb.encode(&token_ids);
        }

        // Raw binary path (with ECC)
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

    /// Decode a flat binary output vector into text.
    /// Uses algebraic codebook when present, otherwise raw binary pipeline
    /// (ECC correction → Gray decode → soft decode).
    fn decode_output(&self, output: &[f32]) -> String {
        if let Some(ref cb) = self.codebook {
            let ids = cb.decode(output);
            return self.dictionary.decode(&ids);
        }

        // Raw binary path
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

            let hard_bits: Vec<u8> = slot.iter().map(|&v| if v > 0.5 { 1u8 } else { 0u8 }).collect();
            let corrected_data = hamming_decode(&hard_bits, bpt);
            let corrected_soft: Vec<f32> = corrected_data.iter().map(|&b| b as f32).collect();
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
        if self.codebook.is_none() && target_vec.iter().all(|&v| v == 0.0) {
            return 0.0;
        }

        let mut input = vec![0.0f32; GEN_COND_DIM];
        for (i, v) in cond.iter().enumerate().take(GEN_COND_DIM) {
            input[i] = *v;
        }

        let result = self.env.train_tick(&input, &target_vec, rng);
        let loss = Self::binary_cross_entropy(&result.output, &target_vec);

        // EOS reinforcement only needed in raw binary mode (not algebraic)
        if self.codebook.is_none() {
            let token_ids = self.dictionary.encode(target);
            let max_tok = self.output_dim / self.coded_bits_per_token.max(1);
            let content_tokens = token_ids.len().min(max_tok);
            if content_tokens < max_tok {
                let eos_target = vec![0.0f32; self.output_dim];
                self.env.train_tick_gradient_only(&input, &eos_target, rng);
            }
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
        if self.codebook.is_none() && target_vec.iter().all(|&v| v == 0.0) {
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

    // --- Algebraic Codebook Tests ---

    fn support_texts() -> Vec<&'static str> {
        vec![
            "To reset your password, go to Settings > Security > Reset password",
            "To reset your password, navigate to Settings > Security > Change password",
            "To reset your password, visit Settings and click Security then Reset",
            "You can change your email in Settings > Profile > Email address",
            "You can change your email in Settings > Profile > Update email",
            "You can update your email under Settings > Profile > Email",
            "Contact support at help@example.com for billing questions",
            "Contact support at help@example.com for account issues",
            "Reach out to help@example.com for billing concerns",
        ]
    }

    #[test]
    fn test_codebook_build() {
        let texts = support_texts();
        let dict = TokenDictionary::build(&texts, 500);
        let cb = AlgebraicCodebook::build(&texts, &dict, 8);
        assert!(!cb.archetypes.is_empty(), "should have at least 1 archetype");
        assert!(cb.total_bits > 0, "total bits should be > 0");
        assert!(cb.total_bits < 500, "total bits should be much less than raw binary (was {})", cb.total_bits);
        println!("codebook: {} archetypes, {} max slots, {} total bits",
            cb.archetypes.len(), cb.max_slot_count, cb.total_bits);
        for (i, arch) in cb.archetypes.iter().enumerate() {
            println!("  arch[{}]: {} fixed, {} slots, len={}",
                i, arch.fixed.len(), arch.slots.len(), arch.length);
        }
    }

    #[test]
    fn test_codebook_encode_decode_roundtrip() {
        let texts = support_texts();
        let dict = TokenDictionary::build(&texts, 500);
        let cb = AlgebraicCodebook::build(&texts, &dict, 8);

        for &text in &texts {
            let token_ids = dict.encode(text);
            let bits = cb.encode(&token_ids);
            assert_eq!(bits.len(), cb.total_bits, "encoded length mismatch");
            let decoded_ids = cb.decode(&bits);
            // Compare token IDs directly (text roundtrip may lose whitespace nuances)
            assert_eq!(decoded_ids, token_ids,
                "token ID roundtrip failed for: {}\n  got: {:?}\n  exp: {:?}", text, decoded_ids, token_ids);
        }
    }

    #[test]
    fn test_algebraic_env_smaller_output() {
        let texts = support_texts();
        let dict = TokenDictionary::build(&texts, 500);
        let cb = AlgebraicCodebook::build(&texts, &dict, 8);
        let raw_output = MAX_TOKENS * bits_for_dict(dict.len());
        let algebraic_output = cb.total_bits;
        println!("raw output_dim={}, algebraic output_dim={}, reduction={}x",
            raw_output, algebraic_output, raw_output / algebraic_output.max(1));
        assert!(algebraic_output < raw_output / 2,
            "algebraic should be at least 2x smaller: {} vs {}", algebraic_output, raw_output);
    }

    #[test]
    fn test_algebraic_env_train_and_generate() {
        let texts = support_texts();
        let dict = TokenDictionary::build(&texts, 500);
        let cb = AlgebraicCodebook::build(&texts, &dict, 8);
        let mut rng = StdRng::seed_from_u64(42);
        let ov = GenEnvOverrides::default();
        let mut env = GroupGenEnv::new_algebraic(dict, cb, &ov, &mut rng);
        assert!(env.codebook.is_some());

        let cond = vec![0.1f32; GEN_COND_DIM];
        let target = "To reset your password, go to Settings > Security > Reset password";
        let loss = env.train_step(&cond, target, &mut rng);
        assert!(loss > 0.0, "loss should be positive: {}", loss);
        let _out = env.generate(&cond, 100, 0.8);
        println!("generated: {}", _out);
    }

    #[test]
    fn test_algebraic_loss_decreases() {
        let texts = support_texts();
        let dict = TokenDictionary::build(&texts, 500);
        let cb = AlgebraicCodebook::build(&texts, &dict, 8);
        let mut rng = StdRng::seed_from_u64(42);
        let ov = GenEnvOverrides::default();
        let mut env = GroupGenEnv::new_algebraic(dict, cb, &ov, &mut rng);
        let cond = vec![0.1f32; GEN_COND_DIM];
        let target = "To reset your password, go to Settings > Security > Reset password";

        let loss_0 = env.train_step(&cond, target, &mut rng);
        for _ in 0..200 {
            env.train_step(&cond, target, &mut rng);
        }
        let loss_200 = env.train_step(&cond, target, &mut rng);
        println!("algebraic loss: {} -> {}", loss_0, loss_200);
        assert!(loss_200 < loss_0, "loss should decrease: {} -> {}", loss_0, loss_200);
    }

    #[test]
    fn test_algebraic_encode_decode_via_env() {
        let texts = support_texts();
        let dict = TokenDictionary::build(&texts, 500);
        let cb = AlgebraicCodebook::build(&texts, &dict, 8);
        let mut rng = StdRng::seed_from_u64(42);
        let ov = GenEnvOverrides::default();
        let env = GroupGenEnv::new_algebraic(dict.clone(), cb, &ov, &mut rng);

        for &text in &texts {
            let target = env.encode_target(text);
            assert_eq!(target.len(), env.output_dim);
            let decoded = env.decode_output(&target);
            let expected = dict.decode(&dict.encode(text));
            assert_eq!(decoded, expected, "env encode/decode roundtrip failed for: {}", text);
        }
    }

    // --- Syntax-Aware Codebook Tests ---

    fn code_texts() -> Vec<&'static str> {
        vec![
            "def binary_search(arr, target): lo, hi = 0, len(arr)-1",
            "def bubble_sort(arr): n = len(arr)",
            "def fibonacci(n): if n <= 1: return n",
            "def factorial(n): if n <= 1: return 1",
            "def linear_search(arr, target): for i in range(len(arr))",
            "class Observer: def __init__(self): self._observers = []",
            "class Subject: def __init__(self): self._state = None",
            "class Singleton: _instance = None",
            "fn main() { let x = 42; println!(x); }",
            "fn helper(n: i32) -> i32 { if n <= 1 { return 1; } n * helper(n-1) }",
        ]
    }

    #[test]
    fn test_syntax_aware_codebook_build() {
        let texts = code_texts();
        let dict = TokenDictionary::build(&texts, 500);
        let cb_stat = AlgebraicCodebook::build(&texts, &dict, 16);
        let cb_syn = AlgebraicCodebook::build_syntax_aware(&texts, &dict, 16);

        println!("statistical: {} archetypes, {} max slots, {} total bits",
            cb_stat.archetypes.len(), cb_stat.max_slot_count, cb_stat.total_bits);
        println!("syntax-aware: {} archetypes, {} max slots, {} total bits",
            cb_syn.archetypes.len(), cb_syn.max_slot_count, cb_syn.total_bits);

        for (i, arch) in cb_syn.archetypes.iter().enumerate() {
            println!("  syn arch[{}]: {} fixed, {} slots, len={}",
                i, arch.fixed.len(), arch.slots.len(), arch.length);
        }

        // Syntax-aware should have fewer total bits (more fixed tokens from keywords)
        println!("reduction: statistical={} bits, syntax-aware={} bits",
            cb_stat.total_bits, cb_syn.total_bits);
    }

    #[test]
    fn test_syntax_aware_roundtrip() {
        let texts = code_texts();
        let dict = TokenDictionary::build(&texts, 500);
        let cb = AlgebraicCodebook::build_syntax_aware(&texts, &dict, 16);

        for &text in &texts {
            let token_ids = dict.encode(text);
            let bits = cb.encode(&token_ids);
            assert_eq!(bits.len(), cb.total_bits, "encoded length mismatch for: {}", text);
            let decoded_ids = cb.decode(&bits);
            assert_eq!(decoded_ids, token_ids,
                "syntax-aware roundtrip failed for: {}\n  got: {:?}\n  exp: {:?}", text, decoded_ids, token_ids);
        }
    }

    #[test]
    fn test_syntax_aware_env_train() {
        let texts = code_texts();
        let dict = TokenDictionary::build(&texts, 500);
        let cb = AlgebraicCodebook::build_syntax_aware(&texts, &dict, 16);
        println!("syntax-aware code env: {} bits (raw would be {})",
            cb.total_bits, MAX_TOKENS * bits_for_dict(dict.len()));
        let mut rng = StdRng::seed_from_u64(42);
        let ov = GenEnvOverrides::default();
        let mut env = GroupGenEnv::new_algebraic(dict, cb, &ov, &mut rng);

        let cond = vec![0.1f32; GEN_COND_DIM];
        let target = "def binary_search(arr, target): lo, hi = 0, len(arr)-1";
        let loss_0 = env.train_step(&cond, target, &mut rng);
        for _ in 0..200 {
            env.train_step(&cond, target, &mut rng);
        }
        let loss_200 = env.train_step(&cond, target, &mut rng);
        println!("syntax-aware code loss: {} -> {}", loss_0, loss_200);
        assert!(loss_200 < loss_0, "loss should decrease: {} -> {}", loss_0, loss_200);
    }

    #[test]
    fn test_syntax_role_classification() {
        use crate::spectral::{syntax_role, SyntaxRole};
        assert_eq!(syntax_role("def"), SyntaxRole::Keyword);
        assert_eq!(syntax_role("fn"), SyntaxRole::Keyword);
        assert_eq!(syntax_role("class"), SyntaxRole::Keyword);
        assert_eq!(syntax_role("return"), SyntaxRole::Keyword);
        assert_eq!(syntax_role("if"), SyntaxRole::Keyword);
        assert_eq!(syntax_role("("), SyntaxRole::Structure);
        assert_eq!(syntax_role(")"), SyntaxRole::Structure);
        assert_eq!(syntax_role("{"), SyntaxRole::Structure);
        assert_eq!(syntax_role(":"), SyntaxRole::Structure);
        assert_eq!(syntax_role("="), SyntaxRole::Operator);
        assert_eq!(syntax_role("=="), SyntaxRole::Operator);
        assert_eq!(syntax_role("->"), SyntaxRole::Operator);
        assert_eq!(syntax_role("42"), SyntaxRole::Literal);
        assert_eq!(syntax_role("0"), SyntaxRole::Literal);
        assert_eq!(syntax_role("binary_search"), SyntaxRole::Identifier);
        assert_eq!(syntax_role("arr"), SyntaxRole::Identifier);
        assert_eq!(syntax_role("myVar"), SyntaxRole::Identifier);
    }
}
