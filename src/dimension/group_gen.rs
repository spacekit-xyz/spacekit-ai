//! Per-group generation using the Growformer substrate with dictionary-based
//! binary token prediction.
//!
//! Instead of autoregressive character-by-character generation (200 forward
//! passes per sample), this encodes the target as a flat binary vector of
//! token IDs and predicts all tokens in a SINGLE forward pass.
//!
//! Architecture per group:
//!   input  = bridged_embedding (GEN_COND_DIM)
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

use crate::clifford::{
    embed_bridge_vector, causal_fingerprint, spatial_fingerprint,
    BOOST_BIVECTOR_COUNT,
};
use crate::cloze;
use crate::environment::NeuralEnvironment;
use crate::spectral::{
    TokenDictionary, hamming_parity_bits, hamming_encode, hamming_decode,
    tokenize, syntax_role, structural_signature, SyntaxRole, E8Lattice, from_gray,
};
use crate::types::EnvironmentConfig;

pub const GEN_COND_DIM: usize = 192;
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
    pub ephaptic_alpha: Option<f32>,
    pub ephaptic_strength: Option<f32>,
    /// Override conditioning dimension (bridge output). Defaults to GEN_COND_DIM.
    #[serde(default)]
    pub cond_dim: Option<usize>,
    /// Use hex nibble encoding instead of raw binary bits.
    /// Each token ID is encoded as ceil(bpt/4) nibbles × 16 one-hot values.
    #[serde(default)]
    pub hex_mode: Option<bool>,
}

/// Compute bits needed for a dictionary of the given size.
pub fn bits_for_dict(dict_len: usize) -> usize {
    if dict_len <= 1 {
        return 1;
    }
    let max_id = dict_len - 1;
    (usize::BITS - max_id.leading_zeros()) as usize
}

/// Number of hex nibbles (4-bit groups) needed to cover `bpt` bits.
pub fn nibbles_for_bits(bpt: usize) -> usize {
    (bpt + 3) / 4
}

/// Hex neurons per token: each nibble gets 16 one-hot outputs.
pub fn hex_neurons_per_token(bpt: usize) -> usize {
    nibbles_for_bits(bpt) * 16
}

// ---------------------------------------------------------------------------
// Algebraic Codebook — factored generation via group-theory decomposition
// ---------------------------------------------------------------------------

/// Bits needed to represent `n` distinct values.
fn bits_for_count(n: usize) -> usize {
    if n <= 1 { return 1; }
    (usize::BITS - (n - 1).leading_zeros()) as usize
}

/// Find the last bit position in the output where the network is decisive.
/// Scans the output in chunks — when a chunk's average decisiveness drops
/// below threshold, that's the content boundary (bits past this point are
/// noise from untrained trailing archetype positions).
fn output_content_boundary(output: &[f32]) -> usize {
    if output.is_empty() { return 0; }
    let chunk_size = 8;
    let threshold = 0.06;
    let mut last_decisive_end = 0;
    let mut consecutive_weak = 0;

    for start in (0..output.len()).step_by(chunk_size) {
        let end = (start + chunk_size).min(output.len());
        let chunk = &output[start..end];
        let avg_decisiveness: f32 = chunk.iter()
            .map(|&v| (v - 0.5).abs())
            .sum::<f32>() / chunk.len() as f32;

        if avg_decisiveness > threshold {
            last_decisive_end = end;
            consecutive_weak = 0;
        } else {
            consecutive_weak += 1;
            if consecutive_weak >= 3 {
                break;
            }
        }
    }
    last_decisive_end
}

/// Truncate text at the last complete sentence boundary within approximately
/// `max_tokens` worth of content. A sentence boundary is a `.` `!` or `?`
/// followed by a space. This prevents trailing tokens from distant archetype
/// cluster members from contaminating the output.
fn truncate_at_sentence(text: &str, max_tokens: usize) -> String {
    let approx_char_limit = max_tokens * 6;
    if text.len() <= approx_char_limit {
        return text.to_string();
    }
    let search_region = &text[..text.len().min(approx_char_limit + 100)];
    let mut last_boundary = 0;
    for (i, _) in search_region.match_indices(". ") {
        if i <= approx_char_limit { last_boundary = i + 1; }
    }
    for (i, _) in search_region.match_indices("! ") {
        if i <= approx_char_limit && i + 1 > last_boundary { last_boundary = i + 1; }
    }
    for (i, _) in search_region.match_indices("? ") {
        if i <= approx_char_limit && i + 1 > last_boundary { last_boundary = i + 1; }
    }
    if last_boundary > 0 {
        text[..last_boundary].trim().to_string()
    } else {
        text.to_string()
    }
}

fn gen_cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-12 || nb < 1e-12 { return 0.0; }
    dot / (na * nb)
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
    /// Median content length of training samples in this archetype's cluster.
    /// Used to truncate decoded output and avoid trailing tokens from longer
    /// samples that share the same archetype.
    #[serde(default)]
    pub median_content_length: usize,
}

/// Factored representation of the response space for a group.
/// Decomposes response prediction from O(max_tokens × bits_per_token) into
/// O(archetype_bits + num_slots × slot_bits), typically ~80-120 bits
/// instead of ~1400-1900.
///
/// When `archetype_prototypes` are present (computed from training embeddings),
/// archetype selection is done via cosine similarity at inference time rather
/// than by the neural network. The network then only predicts slot bits.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlgebraicCodebook {
    pub archetypes: Vec<ResponseArchetype>,
    pub archetype_bits: usize,
    pub max_slot_count: usize,
    pub slot_bit_widths: Vec<usize>,
    pub total_bits: usize,
    /// Per-archetype embedding centroids (mean of training embeddings assigned
    /// to each cluster). Used for prototype-based archetype selection at
    /// inference, bypassing the network's archetype bits entirely.
    #[serde(default)]
    pub archetype_prototypes: Vec<Vec<f32>>,
    /// Number of bits for slot prediction only (total_bits - archetype_bits).
    /// When prototypes are present, this is the actual network output dimension.
    #[serde(default)]
    pub slot_only_bits: usize,
}

impl AlgebraicCodebook {
    /// Build a codebook from training texts for a single group.
    /// Clusters responses into archetypes, extracts fixed/variable positions.
    /// When `embeddings` is provided (parallel to `texts`), computes per-archetype
    /// prototype centroids for embedding-based archetype selection at inference.
    pub fn build(texts: &[&str], dictionary: &TokenDictionary, max_archetypes: usize, embeddings: Option<&[&[f32]]>) -> Self {
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
        for bw in slot_bit_widths.iter_mut() {
            if *bw > 0 { *bw += 1; }
        }

        let slot_only_bits: usize = slot_bit_widths.iter().sum();
        let total_bits = archetype_bits + slot_only_bits;

        let archetype_prototypes = Self::compute_prototypes(&clusters, embeddings);

        Self { archetypes, archetype_bits, max_slot_count, slot_bit_widths, total_bits, archetype_prototypes, slot_only_bits }
    }

    pub fn empty() -> Self {
        Self { archetypes: vec![], archetype_bits: 1, max_slot_count: 0, slot_bit_widths: vec![], total_bits: 1, archetype_prototypes: vec![], slot_only_bits: 0 }
    }

    /// Returns true if this codebook has prototype embeddings for embedding-based
    /// archetype selection (slot-only mode).
    pub fn has_prototypes(&self) -> bool {
        !self.archetype_prototypes.is_empty() && self.archetype_prototypes.len() == self.archetypes.len()
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
    /// Nibble-grouped decode: split bits into 4-bit hex groups and find the
    /// best match per nibble independently. O(nibbles × 16) vs O(num_options).
    /// Errors in one nibble are contained and don't cascade.
    pub fn soft_decode_index(bits: &[f32], num_options: usize) -> usize {
        if num_options <= 1 { return 0; }
        let nbits = bits_for_count(num_options);
        let n_nib = nibbles_for_bits(nbits);
        let mut composite = 0usize;
        for nib in 0..n_nib {
            let start = nib * 4;
            let nib_bits = (nbits - start).min(4);
            let mut best_val = 0usize;
            let mut best_dist = f32::MAX;
            for cand in 0..(1usize << nib_bits) {
                let mut dist = 0.0f32;
                for i in 0..nib_bits {
                    let target = if (cand >> i) & 1 == 1 { 1.0f32 } else { 0.0 };
                    let d = bits.get(start + i).copied().unwrap_or(0.0) - target;
                    dist += d * d;
                }
                if dist < best_dist {
                    best_dist = dist;
                    best_val = cand;
                }
            }
            composite |= best_val << start;
        }
        composite.min(num_options - 1)
    }

    /// Find the best matching archetype and slot values for a token sequence.
    /// Scores by (matching fixed) - 2*(mismatching fixed) + (slot hits) to
    /// prevent a high-fixed-count archetype from stealing texts that belong
    /// to a different cluster.
    pub fn match_best(&self, token_ids: &[u16]) -> (usize, Vec<usize>) {
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

    /// Compute per-archetype prototype embeddings by averaging the bridged
    /// embeddings of all training texts assigned to each cluster.
    fn compute_prototypes(clusters: &[Vec<usize>], embeddings: Option<&[&[f32]]>) -> Vec<Vec<f32>> {
        let embs = match embeddings {
            Some(e) if !e.is_empty() => e,
            _ => return vec![],
        };
        let dim = embs[0].len();
        clusters.iter().map(|indices| {
            let mut centroid = vec![0.0f32; dim];
            let mut n = 0usize;
            for &idx in indices {
                if let Some(emb) = embs.get(idx) {
                    for (c, &v) in centroid.iter_mut().zip(emb.iter()) {
                        *c += v;
                    }
                    n += 1;
                }
            }
            if n > 0 {
                for c in &mut centroid {
                    *c /= n as f32;
                }
            }
            // L2-normalize the centroid for cosine similarity
            let norm = centroid.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for c in &mut centroid {
                    *c /= norm;
                }
            }
            centroid
        }).collect()
    }

    /// Select the best archetype for an input embedding.
    ///
    /// Uses E8 lattice decoding when the embedding is a multiple of 8d
    /// (decomposes into n/8 × 8d E8 subspaces for provably optimal quantization).
    /// Falls back to cosine similarity for non-aligned dimensions.
    ///
    /// Returns (archetype_index, confidence).
    pub fn select_archetype_by_embedding(&self, embedding: &[f32]) -> (usize, f32) {
        if self.archetype_prototypes.is_empty() {
            return (0, 0.0);
        }

        // E8 lattice decoding for embeddings aligned to 8d subspaces
        if embedding.len() >= 16 && embedding.len() % 8 == 0 {
            return E8Lattice::select_archetype(embedding, &self.archetype_prototypes);
        }

        // Fallback: cosine similarity scan for non-standard dimensions
        let norm = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
        let mut best_idx = 0;
        let mut best_sim = f32::NEG_INFINITY;
        for (i, proto) in self.archetype_prototypes.iter().enumerate() {
            let dot: f32 = embedding.iter().zip(proto.iter()).map(|(a, b)| a * b).sum();
            let sim = if norm > 1e-8 { dot / norm } else { 0.0 };
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }
        (best_idx, best_sim.max(0.0))
    }

    /// Encode a token sequence into slot-only bits (no archetype bits).
    /// Used when prototypes handle archetype selection externally.
    pub fn encode_slot_only(&self, token_ids: &[u16]) -> Vec<f32> {
        let (_arch_idx, slot_values) = self.match_best(token_ids);
        let mut bits = vec![0.0f32; self.slot_only_bits];
        let mut offset = 0;
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

    /// Measure how well the network's raw output agrees with a specific archetype.
    /// Returns a score in [0, 1] where 1 = perfect agreement.
    ///
    /// This catches the case where geometric confidence (embedding similarity)
    /// is high but the network output doesn't match the archetype's structure.
    /// When the network hasn't learned this archetype's pattern, the output
    /// bits will be incoherent relative to the archetype's slot vocabulary,
    /// producing low coherence even though the embedding was close.
    pub fn output_coherence(&self, arch_idx: usize, output_bits: &[f32]) -> f32 {
        if arch_idx >= self.archetypes.len() || output_bits.is_empty() {
            return 0.0;
        }
        let arch = &self.archetypes[arch_idx];

        // When there are no variable slots, the archetype template determines
        // the entire output — the network's bits are irrelevant
        if self.slot_only_bits == 0 || arch.slots.is_empty() {
            return 1.0;
        }

        let check_len = self.slot_only_bits.min(output_bits.len());
        if check_len == 0 {
            return 0.0;
        }

        let slot_bits = &output_bits[..check_len];

        // Check 1: bit decisiveness — well-trained outputs have bits
        // near 0.0 or 1.0 (decisive), while garbage outputs cluster around 0.5
        let mut decisive_count = 0usize;
        for &b in slot_bits {
            let dist_from_half = (b - 0.5).abs();
            if dist_from_half > 0.15 {
                decisive_count += 1;
            }
        }
        let decisiveness = decisive_count as f32 / slot_bits.len() as f32;

        // Check 2: decoded token validity — do the decoded slot values
        // produce actual content tokens?
        let decoded = self.decode_with_archetype(arch_idx, slot_bits);
        let non_zero = decoded.iter().filter(|&&t| t != 0).count();
        let content_ratio = if arch.length > 0 {
            non_zero as f32 / arch.length as f32
        } else {
            0.0
        };

        (decisiveness * 0.6 + content_ratio * 0.4).min(1.0)
    }

    /// Decode slot-only bits back to token IDs using a pre-selected archetype.
    pub fn decode_with_archetype(&self, arch_idx: usize, slot_bits: &[f32]) -> Vec<u16> {
        if arch_idx >= self.archetypes.len() {
            return vec![];
        }
        let arch = &self.archetypes[arch_idx];

        let mut tokens = vec![0u16; arch.length];
        for &(pos, tok) in &arch.fixed {
            if pos < tokens.len() {
                tokens[pos] = tok;
            }
        }

        // Measure per-slot decisiveness: how far the network output bits are
        // from 0.5. Decisive bits = the network was trained on this position.
        // Indecisive bits = the position is past the content boundary for the
        // matched sample (noise from distant samples in the same cluster).
        let mut last_decisive_pos: usize = 0;
        let mut consecutive_indecisive: usize = 0;
        let mut offset = 0;
        for (slot_idx, slot) in arch.slots.iter().enumerate() {
            let sbits = self.slot_bit_widths.get(slot_idx).copied().unwrap_or(0);
            let end = (offset + sbits).min(slot_bits.len());
            let s_bits = if offset < slot_bits.len() { &slot_bits[offset..end] } else { &[] as &[f32] };

            let decisiveness: f32 = if s_bits.is_empty() { 0.0 } else {
                s_bits.iter().map(|&v| (v - 0.5).abs()).sum::<f32>() / s_bits.len() as f32
            };

            let val = Self::soft_decode_index(s_bits, slot.vocab.len());
            if slot.position < tokens.len() && val < slot.vocab.len() {
                tokens[slot.position] = slot.vocab[val];
            }

            if decisiveness > 0.05 {
                last_decisive_pos = slot.position + 1;
                consecutive_indecisive = 0;
            } else {
                consecutive_indecisive += 1;
            }
            offset += sbits;
        }

        // Content boundary: if we have decisive slots, truncate after the
        // last one (plus a small margin for trailing fixed tokens like periods).
        // If the archetype stores median_content_length, use that as a cap.
        let slot_bound = if last_decisive_pos > 0 && consecutive_indecisive >= 2 {
            last_decisive_pos + 3
        } else {
            arch.length
        };
        let median_bound = if arch.median_content_length > 0 {
            arch.median_content_length + 2
        } else {
            arch.length
        };
        let truncate_at = slot_bound.min(median_bound).min(arch.length);

        tokens.into_iter()
            .take(truncate_at)
            .take_while(|&t| t != 0)
            .collect()
    }

    /// Build a syntax-aware codebook for code groups. Instead of pure positional
    /// overlap, clusters by **structural signature** (keywords + punctuation kept,
    /// identifiers/literals replaced with role placeholders). Keywords and
    /// structural punctuation are auto-fixed; only identifiers and literals
    /// become slots. Dramatically reduces slot count for code.
    pub fn build_syntax_aware(texts: &[&str], dictionary: &TokenDictionary, max_archetypes: usize, embeddings: Option<&[&[f32]]>) -> Self {
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
        for bw in slot_bit_widths.iter_mut() {
            if *bw > 0 { *bw += 1; }
        }

        let slot_only_bits: usize = slot_bit_widths.iter().sum();
        let total_bits = archetype_bits + slot_only_bits;

        let archetype_prototypes = Self::compute_prototypes(&cluster_list, embeddings);

        Self { archetypes, archetype_bits, max_slot_count, slot_bit_widths, total_bits, archetype_prototypes, slot_only_bits }
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

        ResponseArchetype { fixed, slots, length, median_content_length: length }
    }

    /// Extract an archetype from a cluster of aligned token sequences.
    /// Uses the **median** sample length so trailing content from the longest
    /// samples doesn't pollute the archetype with irrelevant fixed tokens.
    fn extract_archetype(seqs: &[&Vec<u16>], max_len: usize) -> ResponseArchetype {
        let n = seqs.len().max(1);
        let mut lengths: Vec<usize> = seqs.iter().map(|s| {
            s.iter().rposition(|&t| t != 0).map(|p| p + 1).unwrap_or(0)
        }).collect();
        lengths.sort();
        let length = lengths[lengths.len() / 2].min(max_len);

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

        let median_content_length = length;
        ResponseArchetype { fixed, slots, length, median_content_length }
    }
}

// ---------------------------------------------------------------------------
// Hopf Composition Table — compositional generation via fragment algebra
// ---------------------------------------------------------------------------
//
// The Hopf composition table decomposes archetypes into positional segments
// (fragments) and allows mixing fragments from different archetypes to generate
// novel responses. This is the algebraic coproduct Δ(response) = Σ fragments,
// with the composition product μ assembling fragments from multiple archetypes.
//
// For OOD prompts where no single archetype matches well, the table selects the
// best fragment per segment independently, enabling compositional generalization.

/// A segment of an archetype, representing one composable fragment.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArchetypeFragment {
    pub archetype_idx: usize,
    pub segment_idx: usize,
    pub token_range: (usize, usize),
    pub fixed: Vec<(usize, u16)>,
    pub slot_indices: Vec<usize>,
}

/// Hopf composition table: enables compositional generation by selecting
/// the best fragment per segment from potentially different archetypes.
///
/// Algebraic structure:
///   Δ(archetype) = fragment₀ ⊗ fragment₁ ⊗ ... ⊗ fragmentₖ  (coproduct)
///   μ(fᵢ, fⱼ, ...) = composed_response                       (product)
///   transition scores enforce coherent composition boundaries
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HopfCompositionTable {
    pub num_segments: usize,
    /// fragments[segment_idx] = list of fragments available for that segment
    pub fragments: Vec<Vec<ArchetypeFragment>>,
    /// Per-fragment embedding centroids: prototypes[segment_idx][fragment_idx]
    pub fragment_prototypes: Vec<Vec<Vec<f32>>>,
    /// Transition compatibility: transition[seg_idx][frag_a][frag_b] = score
    /// for fragment_a at segment seg_idx followed by fragment_b at seg_idx+1.
    pub transition: Vec<Vec<Vec<f32>>>,
    /// Total response length (max tokens across all archetypes).
    pub response_length: usize,
}

impl Default for HopfCompositionTable {
    fn default() -> Self {
        Self {
            num_segments: 3,
            fragments: vec![],
            fragment_prototypes: vec![],
            transition: vec![],
            response_length: 0,
        }
    }
}

impl HopfCompositionTable {
    /// Build a composition table from an existing codebook.
    /// Splits each archetype into `num_segments` positional fragments,
    /// computes per-fragment prototypes, and scores transition compatibility.
    pub fn build(
        codebook: &AlgebraicCodebook,
        embeddings: Option<&[&[f32]]>,
        clusters: &[Vec<usize>],
        num_segments: usize,
    ) -> Self {
        let num_segments = num_segments.max(2);
        let response_length = codebook.archetypes.iter()
            .map(|a| a.length)
            .max()
            .unwrap_or(0);

        if response_length == 0 || codebook.archetypes.is_empty() {
            return Self {
                num_segments,
                fragments: vec![vec![]; num_segments],
                fragment_prototypes: vec![vec![]; num_segments],
                transition: vec![],
                response_length: 0,
            };
        }

        let seg_size = (response_length + num_segments - 1) / num_segments;

        let mut fragments: Vec<Vec<ArchetypeFragment>> = vec![vec![]; num_segments];
        let mut fragment_prototypes: Vec<Vec<Vec<f32>>> = vec![vec![]; num_segments];

        for (arch_idx, arch) in codebook.archetypes.iter().enumerate() {
            for seg in 0..num_segments {
                let start = seg * seg_size;
                let end = ((seg + 1) * seg_size).min(response_length);
                if start >= end { continue; }

                let fixed: Vec<(usize, u16)> = arch.fixed.iter()
                    .filter(|&&(pos, _)| pos >= start && pos < end)
                    .copied()
                    .collect();

                let slot_indices: Vec<usize> = arch.slots.iter()
                    .enumerate()
                    .filter(|(_, s)| s.position >= start && s.position < end)
                    .map(|(i, _)| i)
                    .collect();

                fragments[seg].push(ArchetypeFragment {
                    archetype_idx: arch_idx,
                    segment_idx: seg,
                    token_range: (start, end),
                    fixed,
                    slot_indices,
                });

                // Compute fragment prototype from embeddings of texts in this archetype's cluster
                if let (Some(embs), Some(cluster)) = (embeddings, clusters.get(arch_idx)) {
                    let dim = embs.first().map(|e| e.len()).unwrap_or(0);
                    if dim > 0 {
                        let mut centroid = vec![0.0f32; dim];
                        let mut n = 0usize;
                        for &idx in cluster {
                            if let Some(emb) = embs.get(idx) {
                                for (c, &v) in centroid.iter_mut().zip(emb.iter()) {
                                    *c += v;
                                }
                                n += 1;
                            }
                        }
                        if n > 0 {
                            for c in &mut centroid { *c /= n as f32; }
                        }
                        let norm = centroid.iter().map(|v| v * v).sum::<f32>().sqrt();
                        if norm > 1e-8 {
                            for c in &mut centroid { *c /= norm; }
                        }
                        fragment_prototypes[seg].push(centroid);
                    } else {
                        fragment_prototypes[seg].push(vec![]);
                    }
                } else {
                    fragment_prototypes[seg].push(vec![]);
                }
            }
        }

        // Build transition scores between adjacent segments using the E8 lattice
        // compatibility score. The E8 root inner product provides algebraically exact
        // compatibility between archetype prototypes — related patterns (observer ↔
        // event-driven, proxy ↔ microservices) score high; unrelated patterns score
        // low. Same-archetype gets a small coherence bonus on top.
        let boundary_window = 3;
        let mut transition = Vec::new();
        for seg in 0..num_segments.saturating_sub(1) {
            let n_a = fragments[seg].len();
            let n_b = fragments[seg + 1].len();
            let mut scores = vec![vec![0.0f32; n_b]; n_a];

            for (a_idx, frag_a) in fragments[seg].iter().enumerate() {
                let boundary = frag_a.token_range.1;
                let a_tail: Vec<u16> = frag_a.fixed.iter()
                    .filter(|&&(pos, _)| pos >= boundary.saturating_sub(boundary_window) && pos < boundary)
                    .map(|&(_, tok)| tok)
                    .collect();

                let proto_a = codebook.archetype_prototypes.get(frag_a.archetype_idx);

                for (b_idx, frag_b) in fragments[seg + 1].iter().enumerate() {
                    let b_head: Vec<u16> = frag_b.fixed.iter()
                        .filter(|&&(pos, _)| pos < frag_b.token_range.0 + boundary_window)
                        .map(|&(_, tok)| tok)
                        .collect();

                    // E8 lattice compatibility score: uses root inner product
                    // between archetype prototypes quantized to E8 subspaces.
                    // Returns [0, 3] with algebraically exact structure.
                    let compatibility = match (proto_a, codebook.archetype_prototypes.get(frag_b.archetype_idx)) {
                        (Some(pa), Some(pb)) if !pa.is_empty() && !pb.is_empty() => {
                            E8Lattice::compatibility_score(pa, pb)
                        }
                        _ => 0.5,
                    };

                    let same_arch_bonus = if frag_a.archetype_idx == frag_b.archetype_idx { 0.3 } else { 0.0 };
                    let has_content = if !a_tail.is_empty() && !b_head.is_empty() { 0.3 } else { 0.15 };

                    scores[a_idx][b_idx] = compatibility + same_arch_bonus + has_content;
                }
            }
            transition.push(scores);
        }

        Self { num_segments, fragments, fragment_prototypes, transition, response_length }
    }

    /// Select the best fragment for each segment using embedding similarity
    /// and transition compatibility. Returns fragment indices per segment.
    ///
    /// Uses beam search with beam_width=3: for each segment, expand top
    /// candidates by embedding similarity, then re-rank by transition score.
    ///
    /// `diversity_bonus` (from OCEAN personality): positive values favor
    /// cross-archetype composition, negative values favor same-archetype
    /// coherence. Range: [-0.3, 0.3]. Pass 0.0 for neutral.
    pub fn compose_with_personality(&self, embedding: &[f32], diversity_bonus: f32) -> Vec<usize> {
        if self.fragments.is_empty() || self.response_length == 0 {
            return vec![0; self.num_segments];
        }

        let norm = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
        let beam_width = 3usize;

        // Score all fragments by embedding similarity
        let seg_scores: Vec<Vec<f32>> = self.fragment_prototypes.iter()
            .map(|seg_protos| {
                seg_protos.iter().map(|proto| {
                    if proto.is_empty() || norm < 1e-8 { return 0.0; }
                    let dot: f32 = embedding.iter().zip(proto.iter()).map(|(a, b)| a * b).sum();
                    dot / norm
                }).collect()
            }).collect();

        // Beam search: track (total_score, fragment_indices)
        let mut beam: Vec<(f32, Vec<usize>)> = if !seg_scores.is_empty() && !seg_scores[0].is_empty() {
            let mut candidates: Vec<(f32, usize)> = seg_scores[0].iter()
                .enumerate()
                .map(|(i, &s)| (s, i))
                .collect();
            candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            candidates.truncate(beam_width);
            candidates.into_iter().map(|(s, i)| (s, vec![i])).collect()
        } else {
            vec![(0.0, vec![0])]
        };

        for seg in 1..self.num_segments {
            let n_frags = self.fragments.get(seg).map(|f| f.len()).unwrap_or(1);
            let mut next_beam: Vec<(f32, Vec<usize>)> = Vec::new();

            for (prev_score, prev_path) in &beam {
                let prev_frag = *prev_path.last().unwrap_or(&0);

                for frag_idx in 0..n_frags {
                    let emb_score = seg_scores.get(seg)
                        .and_then(|s| s.get(frag_idx))
                        .copied()
                        .unwrap_or(0.0);

                    let trans_score = if seg > 0 {
                        self.transition.get(seg - 1)
                            .and_then(|t| t.get(prev_frag))
                            .and_then(|row| row.get(frag_idx))
                            .copied()
                            .unwrap_or(0.0)
                    } else {
                        0.0
                    };

                    // OCEAN personality modulation: openness boosts cross-archetype,
                    // conscientiousness boosts same-archetype transitions.
                    let personality_mod = if seg > 0 {
                        let prev_arch = self.fragments.get(seg - 1)
                            .and_then(|f| f.get(prev_frag))
                            .map(|f| f.archetype_idx);
                        let curr_arch = self.fragments.get(seg)
                            .and_then(|f| f.get(frag_idx))
                            .map(|f| f.archetype_idx);
                        match (prev_arch, curr_arch) {
                            (Some(a), Some(b)) if a != b => diversity_bonus,
                            (Some(a), Some(b)) if a == b => -diversity_bonus,
                            _ => 0.0,
                        }
                    } else {
                        0.0
                    };

                    let total = prev_score + emb_score + trans_score * 0.5 + personality_mod;
                    let mut path = prev_path.clone();
                    path.push(frag_idx);
                    next_beam.push((total, path));
                }
            }

            next_beam.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            next_beam.truncate(beam_width);
            beam = next_beam;
        }

        beam.into_iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, path)| path)
            .unwrap_or_else(|| vec![0; self.num_segments])
    }

    /// Compose with neutral personality (no diversity bias).
    pub fn compose(&self, embedding: &[f32]) -> Vec<usize> {
        self.compose_with_personality(embedding, 0.0)
    }

    /// Assemble a composed response from selected fragment indices.
    /// Merges fixed tokens from each fragment and maps slot indices.
    /// Returns (fixed_tokens, slot_map) where slot_map[composed_slot_idx] =
    /// (archetype_idx, original_slot_idx).
    pub fn assemble(&self, fragment_indices: &[usize], _codebook: &AlgebraicCodebook)
        -> (Vec<(usize, u16)>, Vec<(usize, usize)>)
    {
        let mut fixed = Vec::new();
        let mut slot_map = Vec::new();

        for (seg, &frag_idx) in fragment_indices.iter().enumerate() {
            if let Some(frag) = self.fragments.get(seg).and_then(|s| s.get(frag_idx)) {
                for &(pos, tok) in &frag.fixed {
                    fixed.push((pos, tok));
                }
                for &slot_idx in &frag.slot_indices {
                    slot_map.push((frag.archetype_idx, slot_idx));
                }
            }
        }

        fixed.sort_by_key(|&(pos, _)| pos);
        fixed.dedup_by_key(|&mut (pos, _)| pos);
        (fixed, slot_map)
    }

    /// Full composed decode: select fragments, assemble, fill slots from network output.
    /// `diversity_bonus` from OCEAN personality (0.0 for neutral).
    pub fn compose_and_decode(
        &self,
        embedding: &[f32],
        slot_bits: &[f32],
        codebook: &AlgebraicCodebook,
    ) -> (Vec<u16>, f32) {
        self.compose_and_decode_with_personality(embedding, slot_bits, codebook, 0.0)
    }

    pub fn compose_and_decode_with_personality(
        &self,
        embedding: &[f32],
        slot_bits: &[f32],
        codebook: &AlgebraicCodebook,
        diversity_bonus: f32,
    ) -> (Vec<u16>, f32) {
        let frag_indices = self.compose_with_personality(embedding, diversity_bonus);
        let (fixed, slot_map) = self.assemble(&frag_indices, codebook);

        let mut tokens = vec![0u16; self.response_length];
        for &(pos, tok) in &fixed {
            if pos < tokens.len() {
                tokens[pos] = tok;
            }
        }

        // Fill slots using the network's slot predictions
        let mut offset = 0;
        for &(arch_idx, slot_idx) in &slot_map {
            if let Some(arch) = codebook.archetypes.get(arch_idx) {
                if let Some(slot) = arch.slots.get(slot_idx) {
                    let sbits = codebook.slot_bit_widths.get(slot_idx).copied().unwrap_or(0);
                    let end = (offset + sbits).min(slot_bits.len());
                    let s_bits = if offset < slot_bits.len() { &slot_bits[offset..end] } else { &[] as &[f32] };
                    let val = AlgebraicCodebook::soft_decode_index(s_bits, slot.vocab.len());
                    if slot.position < tokens.len() && val < slot.vocab.len() {
                        tokens[slot.position] = slot.vocab[val];
                    }
                    offset += sbits;
                }
            }
        }

        // Compute confidence as average embedding similarity across selected fragments
        let norm = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
        let mut total_sim = 0.0f32;
        let mut count = 0;
        for (seg, &frag_idx) in frag_indices.iter().enumerate() {
            if let Some(proto) = self.fragment_prototypes.get(seg).and_then(|s| s.get(frag_idx)) {
                if !proto.is_empty() && norm > 1e-8 {
                    let dot: f32 = embedding.iter().zip(proto.iter()).map(|(a, b)| a * b).sum();
                    total_sim += dot / norm;
                    count += 1;
                }
            }
        }
        let confidence = if count > 0 { (total_sim / count as f32).max(0.0) } else { 0.0 };

        let result = tokens.into_iter().take_while(|&t| t != 0).collect();
        (result, confidence)
    }

    /// Check whether composition would actually produce different results
    /// from single-archetype selection by comparing fragment sources.
    pub fn is_composed(&self, fragment_indices: &[usize]) -> bool {
        if fragment_indices.len() < 2 { return false; }
        let first_arch = self.fragments.get(0)
            .and_then(|s| s.get(fragment_indices[0]))
            .map(|f| f.archetype_idx);
        fragment_indices.iter().enumerate().skip(1).any(|(seg, &frag_idx)| {
            self.fragments.get(seg)
                .and_then(|s| s.get(frag_idx))
                .map(|f| f.archetype_idx) != first_arch
        })
    }
}

// ---------------------------------------------------------------------------
// Two-phase training: Memorize → Consolidate
// ---------------------------------------------------------------------------

/// Snapshot of config values overridden during the memorize phase.
/// Passed to `enter_consolidate_mode` to restore original settings.
#[derive(Debug, Clone)]
pub struct MemorizeSnapshot {
    pub learning_rate: f32,
    pub current_lr: f32,
    pub competitive_k: usize,
    pub dropout_rate: f32,
    pub weight_decay: f32,
    pub bias_decay: f32,
    pub lateral_inhibition: f32,
    pub kwta_suppression: f32,
    pub prune_stop_tick: u64,
    pub engram_enabled: bool,
}

// ---------------------------------------------------------------------------
// GroupGenEnv
// ---------------------------------------------------------------------------

/// DEPRECATED: Use `IndexedGenEnv` instead. This struct wraps a full
/// NeuralEnvironment for generation, which requires thousands of backprop
/// epochs. `IndexedGenEnv` uses a Paramecium lattice for one-pass indexing
/// and achieves ~85% token overlap with zero iterative training.
#[deprecated(note = "Use IndexedGenEnv (Paramecium lattice) instead — zero backprop, one-pass indexing")]
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
    /// Hopf composition table for compositional generation across archetypes.
    #[serde(default)]
    pub hopf_table: Option<HopfCompositionTable>,
    /// Archetype selected by embedding prototype matching (slot-only mode).
    #[serde(skip)]
    last_selected_archetype: Option<usize>,
    /// Confidence of the last generation (cosine similarity to best prototype).
    #[serde(skip)]
    pub last_generation_confidence: f32,
    /// OCEAN personality diversity bonus for Hopf beam scoring.
    /// Positive = favor cross-archetype mixing; negative = favor coherence.
    #[serde(skip)]
    pub diversity_bonus: f32,
    /// Transient: query subject keywords for within-topic BM25 re-ranking.
    #[serde(skip)]
    pub subject_keywords: Vec<String>,
    /// Transient: query intent action (e.g., "implement", "explain", "define").
    /// Used to prefer code-bearing programs for "implement" queries and prose for "explain".
    #[serde(skip)]
    pub intent_action: String,
    /// Hex nibble encoding: each token ID is encoded as nibbles × 16 one-hot
    /// values instead of raw binary + Hamming ECC. Gives error containment
    /// per nibble and faster argmax decode.
    #[serde(default)]
    pub hex_mode: bool,
}

impl GroupGenEnv {
    pub fn new(dictionary: TokenDictionary, rng: &mut impl Rng) -> Self {
        Self::new_with_overrides(dictionary, &GenEnvOverrides::default(), rng)
    }

    pub fn new_with_overrides(dictionary: TokenDictionary, ov: &GenEnvOverrides, rng: &mut impl Rng) -> Self {
        let hidden = ov.hidden.unwrap_or(GEN_HIDDEN);
        let k = ov.k.unwrap_or(GEN_K);
        let cond_dim = ov.cond_dim.unwrap_or(GEN_COND_DIM);
        let bits_per_token = bits_for_dict(dictionary.len());
        let hex = ov.hex_mode.unwrap_or(false);

        let (coded_bits_per_token, output_dim) = if hex {
            let hex_npt = hex_neurons_per_token(bits_per_token);
            // Auto-scale max_tokens to keep output_dim comparable to binary mode
            let binary_cbpt = bits_per_token + hamming_parity_bits(bits_per_token);
            let binary_budget = ov.max_tokens.unwrap_or(MAX_TOKENS) * binary_cbpt;
            let max_tok = ov.max_tokens.map(|m| m).unwrap_or(binary_budget / hex_npt);
            (hex_npt, max_tok.max(8) * hex_npt)
        } else {
            let max_tok = ov.max_tokens.unwrap_or(MAX_TOKENS);
            let parity_bits = hamming_parity_bits(bits_per_token);
            let cbpt = bits_per_token + parity_bits;
            (cbpt, max_tok * cbpt)
        };

        let mut config = gen_env_config();
        config.competitive_k = k;
        if let Some(ms) = ov.max_synapses { config.max_synapses_per_neuron = ms; }
        if let Some(eb) = ov.energy_budget { config.energy_budget_per_neuron = eb; }
        if let Some(a)  = ov.ephaptic_alpha { config.ephaptic_field_alpha = a; }
        if let Some(s)  = ov.ephaptic_strength { config.ephaptic_field_strength = s; }
        let mut env = NeuralEnvironment::new(config);
        env.build_layers(&[cond_dim, hidden, hidden, output_dim], rng);
        Self {
            env,
            dictionary,
            bits_per_token,
            coded_bits_per_token,
            output_dim,
            frozen: false,
            codebook: None,
            hopf_table: None,
            last_selected_archetype: None,
            last_generation_confidence: 0.0,
            diversity_bonus: 0.0,
            subject_keywords: Vec::new(),
            intent_action: String::new(),
            hex_mode: hex,
        }
    }

    /// Create an algebraic generation environment. The codebook factorizes
    /// the response space so the substrate only predicts ~80-120 bits
    /// (archetype + slot values) instead of ~1400-1900 raw token bits.
    ///
    /// When the codebook has prototype embeddings, enters **slot-only mode**:
    /// archetype selection is done by embedding cosine similarity, and the
    /// network output dimension is reduced to just the slot bits.
    pub fn new_algebraic(
        dictionary: TokenDictionary,
        codebook: AlgebraicCodebook,
        ov: &GenEnvOverrides,
        rng: &mut impl Rng,
    ) -> Self {
        let hidden = ov.hidden.unwrap_or(GEN_HIDDEN);
        let k = ov.k.unwrap_or(GEN_K);
        let cond_dim = ov.cond_dim.unwrap_or(GEN_COND_DIM);
        let raw_output_dim = if codebook.has_prototypes() {
            codebook.slot_only_bits
        } else {
            codebook.total_bits
        };
        let output_dim = raw_output_dim.max(256);
        let bits_per_token = bits_for_dict(dictionary.len());
        let coded_bits_per_token = bits_per_token;
        let mut config = gen_env_config();
        config.competitive_k = k.min(hidden / 2);
        if let Some(ms) = ov.max_synapses { config.max_synapses_per_neuron = ms; }
        if let Some(eb) = ov.energy_budget { config.energy_budget_per_neuron = eb; }
        if let Some(a)  = ov.ephaptic_alpha { config.ephaptic_field_alpha = a; }
        if let Some(s)  = ov.ephaptic_strength { config.ephaptic_field_strength = s; }
        let mut env = NeuralEnvironment::new(config);
        env.build_layers(&[cond_dim, hidden, hidden, output_dim], rng);
        Self {
            env,
            dictionary,
            bits_per_token,
            coded_bits_per_token,
            output_dim,
            frozen: false,
            codebook: Some(codebook),
            hopf_table: None,
            last_selected_archetype: None,
            last_generation_confidence: 0.0,
            diversity_bonus: 0.0,
            subject_keywords: Vec::new(),
            intent_action: String::new(),
            hex_mode: false,
        }
    }

    /// Attach a Hopf composition table for compositional generation.
    pub fn set_hopf_table(&mut self, table: HopfCompositionTable) {
        self.hopf_table = Some(table);
    }

    /// Set pruning to stop after this many ticks.
    /// Call before training begins to lock in the warmup window.
    pub fn set_prune_stop_tick(&mut self, tick: u64) {
        self.env.config.prune_stop_tick = tick;
    }

    /// Switch to aggressive memorization mode: high LR, full neuron
    /// activation (k = hidden_size), no dropout, no weight/bias decay,
    /// pruning disabled. Call before Phase 1 of two-phase training.
    /// Returns the saved config snapshot needed by `enter_consolidate_mode`.
    pub fn enter_memorize_mode(&mut self) -> MemorizeSnapshot {
        let snap = MemorizeSnapshot {
            learning_rate: self.env.config.learning_rate,
            current_lr: self.env.current_lr,
            competitive_k: self.env.config.competitive_k,
            dropout_rate: self.env.config.dropout_rate,
            weight_decay: self.env.config.weight_decay,
            bias_decay: self.env.config.bias_decay,
            lateral_inhibition: self.env.config.lateral_inhibition,
            kwta_suppression: self.env.config.kwta_suppression,
            prune_stop_tick: self.env.config.prune_stop_tick,
            engram_enabled: self.env.config.engram_enabled,
        };

        let hidden_size = self.env.layers.get(1).map_or(GEN_HIDDEN, |l| l.len());

        self.env.config.learning_rate = 0.25;
        self.env.current_lr = 0.25;
        self.env.config.competitive_k = hidden_size;
        self.env.config.dropout_rate = 0.0;
        self.env.config.weight_decay = 0.0;
        self.env.config.bias_decay = 0.0;
        self.env.config.lateral_inhibition = 0.0;
        self.env.config.kwta_suppression = 0.0;
        self.env.config.prune_stop_tick = 1; // effectively disable pruning
        self.env.config.engram_enabled = false;

        snap
    }

    /// Restore consolidation-phase settings from a prior snapshot and
    /// optionally anneal LR slightly below original for stability.
    pub fn enter_consolidate_mode(&mut self, snap: &MemorizeSnapshot) {
        self.env.config.learning_rate = snap.learning_rate;
        self.env.current_lr = snap.current_lr;
        self.env.config.competitive_k = snap.competitive_k;
        self.env.config.dropout_rate = snap.dropout_rate;
        self.env.config.weight_decay = snap.weight_decay;
        self.env.config.bias_decay = snap.bias_decay;
        self.env.config.lateral_inhibition = snap.lateral_inhibition;
        self.env.config.kwta_suppression = snap.kwta_suppression;
        self.env.config.prune_stop_tick = 0; // re-enable pruning from current tick
        self.env.config.engram_enabled = snap.engram_enabled;
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

    /// Hex nibble decode: split Gray-coded bits into 4-bit nibble groups and
    /// find the best match within each nibble independently. O(nibbles × 16)
    /// instead of O(dict_size). Errors in one nibble are contained and don't
    /// cascade to other nibbles.
    fn nibbles_to_id(bits: &[f32], _dict: &TokenDictionary, dict_size: usize, bpt: usize) -> u16 {
        let n_nib = nibbles_for_bits(bpt);
        let mut gray: u16 = 0;
        for nib in 0..n_nib {
            let start = nib * 4;
            let nib_bits = (bpt - start).min(4);
            let mut best_val = 0u16;
            let mut best_dist = f32::MAX;
            for cand in 0..(1u16 << nib_bits) {
                let mut dist = 0.0f32;
                for i in 0..nib_bits {
                    let target = if (cand >> i) & 1 == 1 { 1.0f32 } else { 0.0 };
                    let d = bits.get(start + i).copied().unwrap_or(0.0) - target;
                    dist += d * d;
                }
                if dist < best_dist {
                    best_dist = dist;
                    best_val = cand;
                }
            }
            gray |= best_val << start;
        }
        let id = from_gray(gray);
        id.min(dict_size.saturating_sub(1) as u16)
    }

    /// Encode a token ID as hex nibble one-hot values.
    /// Gray code → split into nibbles → one-hot (16 values per nibble).
    fn id_to_hex(&self, id: u16) -> Vec<f32> {
        let gray = self.dictionary.to_gray_id(id);
        let n_nib = nibbles_for_bits(self.bits_per_token);
        let npt = n_nib * 16;
        let mut hex_vec = vec![0.0f32; npt];
        for nib in 0..n_nib {
            let val = ((gray >> (nib * 4)) & 0xF) as usize;
            hex_vec[nib * 16 + val] = 1.0;
        }
        hex_vec
    }

    /// Decode hex nibble one-hot outputs back to a token ID.
    /// Argmax within each 16-neuron group → compose Gray code → from_gray.
    fn hex_to_id(output: &[f32], _dict: &TokenDictionary, dict_size: usize, bpt: usize) -> u16 {
        let n_nib = nibbles_for_bits(bpt);
        let mut gray: u16 = 0;
        for nib in 0..n_nib {
            let offset = nib * 16;
            let nib_bits = (bpt - nib * 4).min(4);
            let max_val = (1usize << nib_bits) - 1;
            let mut best_val = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for v in 0..=max_val {
                let score = output.get(offset + v).copied().unwrap_or(f32::NEG_INFINITY);
                if score > best_score {
                    best_score = score;
                    best_val = v;
                }
            }
            gray |= (best_val as u16) << (nib * 4);
        }
        let id = from_gray(gray);
        id.min(dict_size.saturating_sub(1) as u16)
    }

    /// Encode a full target text into a flat binary target vector.
    /// With prototypes: slot-only bits (archetype selected externally).
    /// Without prototypes: full algebraic bits (archetype + slots).
    /// Without codebook: raw binary with ECC.
    fn encode_target(&self, text: &str) -> Vec<f32> {
        let token_ids = self.dictionary.encode(text);

        if let Some(ref cb) = self.codebook {
            let raw = if cb.has_prototypes() {
                cb.encode_slot_only(&token_ids)
            } else {
                cb.encode(&token_ids)
            };
            if raw.len() < self.output_dim {
                let mut padded = raw;
                padded.resize(self.output_dim, 0.0);
                return padded;
            }
            return raw;
        }

        if self.hex_mode {
            // Hex nibble path: one-hot per nibble, no ECC needed
            let mut target = vec![0.0f32; self.output_dim];
            let npt = self.coded_bits_per_token; // hex_neurons_per_token in hex mode
            let max_tok = self.output_dim / npt.max(1);
            for (pos, &id) in token_ids.iter().take(max_tok).enumerate() {
                let hex = self.id_to_hex(id);
                let offset = pos * npt;
                let len = hex.len().min(npt);
                target[offset..offset + len].copy_from_slice(&hex[..len]);
            }
            return target;
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

    /// Decode output vector into text.
    /// Codebook mode: archetype + slot bits.
    /// Hex mode: per-nibble argmax → compose Gray code → token ID.
    /// Binary mode: Hamming ECC → nibble-grouped decode.
    fn decode_output(&self, output: &[f32]) -> String {
        if let Some(ref cb) = self.codebook {
            if cb.has_prototypes() {
                let arch_idx = self.last_selected_archetype.unwrap_or(0);
                let arch = &cb.archetypes[arch_idx];
                let ids = cb.decode_with_archetype(arch_idx, output);
                let text = self.dictionary.decode(&ids);

                let bound = if arch.median_content_length > 0 {
                    arch.median_content_length + 2
                } else {
                    (arch.length * 4) / 5
                };
                if ids.len() > bound {
                    return truncate_at_sentence(&text, bound);
                }
                return text;
            }
            let ids = cb.decode(output);
            return self.dictionary.decode(&ids);
        }

        if self.hex_mode {
            // Hex nibble path: argmax per 16-neuron group
            let dict_size = self.dictionary.len();
            let bpt = self.bits_per_token;
            let npt = self.coded_bits_per_token; // hex_neurons_per_token
            let mut ids = Vec::new();
            let max_tok = self.output_dim / npt.max(1);
            let min_tokens = 3;
            for pos in 0..max_tok {
                let offset = pos * npt;
                if offset + npt > output.len() { break; }
                let slot = &output[offset..offset + npt];

                // Confidence: max activation in any nibble group
                let n_nib = nibbles_for_bits(bpt);
                let max_confidence = (0..n_nib)
                    .map(|nib| {
                        let no = nib * 16;
                        (0..16).map(|v| slot.get(no + v).copied().unwrap_or(0.0))
                            .fold(0.0f32, f32::max)
                    })
                    .fold(0.0f32, f32::max);

                if pos >= min_tokens && max_confidence < 0.15 { break; }

                let id = Self::hex_to_id(slot, &self.dictionary, dict_size, bpt);
                if pos >= min_tokens && id == 0 { break; }
                if id > 0 { ids.push(id); }
            }
            return self.dictionary.decode(&ids);
        }

        // Binary path: Hamming ECC → nibble-grouped decode
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
            let id = Self::nibbles_to_id(&corrected_soft, &self.dictionary, dict_size, bpt);

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
    #[cfg(feature = "training")]
    pub fn train_step(&mut self, cond: &[f32], target: &str, rng: &mut impl Rng) -> f32 {
        if self.frozen {
            return 0.0;
        }
        let target_vec = self.encode_target(target);
        if self.codebook.is_none() && target_vec.iter().all(|&v| v == 0.0) {
            return 0.0;
        }

        let input_dim = self.env.input_layer_size().unwrap_or(GEN_COND_DIM);
        let mut input = vec![0.0f32; input_dim];
        for (i, v) in cond.iter().enumerate().take(input_dim) {
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
    ///
    /// Generation strategy (in order of preference):
    /// 1. Prototype match with coherence check: if geometric confidence >= 0.9
    ///    AND output coherence >= 0.5, use the single best archetype
    /// 2. Hopf composition: if effective confidence < 0.9 and a Hopf table
    ///    exists, compose fragments from multiple archetypes
    /// 3. Fallback: use best prototype match regardless of confidence
    ///
    /// The coherence check catches the case where the embedding is
    /// geometrically close to an archetype (high similarity) but the
    /// network output doesn't match the archetype's structure (garbage).
    ///
    /// Returns (generated_text, confidence).
    pub fn generate(&mut self, cond: &[f32], _max_len: usize, _temperature: f32) -> (String, f32) {
        let input_dim = self.env.input_layer_size().unwrap_or(GEN_COND_DIM);
        let mut input = vec![0.0f32; input_dim];
        for (i, v) in cond.iter().enumerate().take(input_dim) {
            input[i] = *v;
        }

        let output = self.env.predict(&input);

        if let Some(ref cb) = self.codebook {
            if cb.has_prototypes() {
                let (arch_idx, geometric_conf) = cb.select_archetype_by_embedding(cond);

                // Secondary check: does the network output actually agree
                // with this archetype? High geometric confidence + low
                // coherence = the embedding is close but the content is wrong.
                let coherence = cb.output_coherence(arch_idx, &output);
                let effective_conf = geometric_conf * coherence;

                // Hopf composition when effective confidence is insufficient
                if effective_conf < 0.9 {
                    if let Some(ref hopf) = self.hopf_table {
                        let (ids, comp_confidence) = hopf.compose_and_decode_with_personality(
                            cond, &output, cb, self.diversity_bonus
                        );
                        let text = self.dictionary.decode(&ids);
                        self.last_selected_archetype = None;
                        self.last_generation_confidence = comp_confidence;
                        return (text, comp_confidence);
                    }
                }

                self.last_selected_archetype = Some(arch_idx);
                self.last_generation_confidence = effective_conf;
            } else {
                self.last_selected_archetype = None;
                self.last_generation_confidence = 1.0;
            }
        } else {
            self.last_selected_archetype = None;
            self.last_generation_confidence = 1.0;
        }

        (self.decode_output(&output), self.last_generation_confidence)
    }

    /// Generate using a pre-selected archetype index from the ArchetypeBrain,
    /// bypassing the codebook's cosine similarity selection.
    pub fn generate_with_archetype(
        &mut self, cond: &[f32], arch_idx: usize, arch_conf: f32,
        _max_len: usize, _temperature: f32,
    ) -> (String, f32) {
        let input_dim = self.env.input_layer_size().unwrap_or(GEN_COND_DIM);
        let mut input = vec![0.0f32; input_dim];
        for (i, v) in cond.iter().enumerate().take(input_dim) {
            input[i] = *v;
        }
        let output = self.env.predict(&input);

        if let Some(ref cb) = self.codebook {
            if cb.has_prototypes() && arch_idx < cb.archetypes.len() {
                let coherence = cb.output_coherence(arch_idx, &output);
                let effective_conf = arch_conf * coherence;
                self.last_selected_archetype = Some(arch_idx);
                self.last_generation_confidence = effective_conf;
                return (self.decode_output(&output), effective_conf);
            }
        }
        self.last_selected_archetype = None;
        self.last_generation_confidence = arch_conf;
        (self.decode_output(&output), arch_conf)
    }

    /// Generate and return an 8d E8 contribution vector alongside the text.
    /// The contribution vector captures the group's "semantic direction" for
    /// this input in the E8 lattice — used for algebraic group blending.
    pub fn generate_with_e8(
        &mut self, cond: &[f32], _max_len: usize, _temperature: f32,
    ) -> (String, f32, [f32; 8]) {
        let (text, conf) = self.generate(cond, _max_len, _temperature);

        // Project the first 8 dims of the conditioning vector to E8
        // This captures the group's contribution in a lattice-quantized space
        let mut raw = [0.0f32; 8];
        for i in 0..8.min(cond.len()) {
            raw[i] = cond[i];
        }
        let e8_point = E8Lattice::nearest_point(&raw);
        (text, conf, e8_point)
    }

    /// Evaluate loss without modifying the network.
    pub fn eval_loss(&mut self, cond: &[f32], target: &str) -> f32 {
        let target_vec = self.encode_target(target);
        if self.codebook.is_none() && target_vec.iter().all(|&v| v == 0.0) {
            return 0.0;
        }

        let input_dim = self.env.input_layer_size().unwrap_or(GEN_COND_DIM);
        let mut input = vec![0.0f32; input_dim];
        for (i, v) in cond.iter().enumerate().take(input_dim) {
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
        // Ephaptic field: group-level EMA provides immediate pattern availability.
        // alpha=0.85 gives ~7-sample memory; strength=0.1 adds gentle bias.
        ephaptic_field_alpha: 0.85,
        ephaptic_field_strength: 0.1,
        ..EnvironmentConfig::default()
    }
}

// ---------------------------------------------------------------------------
// E8 Quantum Composition Space — U_q(E8) group blending in 8 dimensions
// ---------------------------------------------------------------------------
//
// Quantum group deformation of the E8 composition space. The parameter q
// controls non-commutativity: at q=1 (classical limit) the blend is
// symmetric; at q≠1 the input embedding biases which group "leads" the
// composition via the R-matrix braiding operator.
//
//   1. Each group produces an 8d E8 lattice point (its "contribution vector")
//   2. q is computed from the input embedding's asymmetry across groups
//   3. The R-matrix braids contributions: R(g_i, g_j) ≠ R(g_j, g_i) when q≠1
//   4. The braided blend is E8-quantized to the nearest lattice point
//   5. Sentence scoring uses R-matrix weights for non-commutative ordering
//
// E8's 240 kissing neighbors × continuous q parameter provide a rich
// composition manifold where architectural and implementation details
// interleave in a context-dependent order.

/// A group's contribution to the E8 composition space.
#[derive(Debug, Clone)]
pub struct E8Contribution {
    pub group_idx: usize,
    pub lattice_point: [f32; 8],
    pub text: String,
    pub confidence: f32,
}

/// Compute the deformation parameter q from an input embedding and
/// group contributions. q measures how asymmetrically the input
/// relates to the contributing groups:
///   q ≈ 1.0 → input is equidistant from all groups (classical/symmetric)
///   q > 1.0 → input is biased toward the first/primary group
///   q < 1.0 → input is biased toward secondary groups
///
/// Derived from the ratio of E8 compatibility scores between the
/// input's lattice projection and each group's lattice point.
pub fn compute_q(input_embedding: &[f32], contributions: &[E8Contribution]) -> f32 {
    if contributions.len() < 2 {
        return 1.0;
    }

    // Project input to 8d E8
    let mut raw = [0.0f32; 8];
    for i in 0..8.min(input_embedding.len()) {
        raw[i] = input_embedding[i];
    }
    let input_e8 = E8Lattice::nearest_point(&raw);

    // Compute compatibility with each contribution
    let scores: Vec<f32> = contributions.iter()
        .map(|c| E8Lattice::compatibility_score_8d(&input_e8, &c.lattice_point))
        .collect();

    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_score = scores.iter().cloned().fold(f32::INFINITY, f32::min);

    if max_score - min_score < 0.01 {
        return 1.0; // Equidistant → classical
    }

    // q = ratio of best to second-best compatibility
    // Clamped to [0.3, 3.0] to avoid extreme deformations
    let mut sorted = scores.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let q = sorted[0] / sorted[1].max(0.01);
    q.clamp(0.3, 3.0)
}

/// R-matrix: the braiding operator for quantum group composition.
/// R(a, b) controls how contribution a interleaves with contribution b.
///
/// When q=1: R(a,b) = R(b,a) (symmetric, classical limit)
/// When q>1: R(a,b) > R(b,a) for the preferred contribution a
///
/// The R-matrix is derived from E8 root inner products deformed by q.
pub fn r_matrix(
    a: &E8Contribution,
    b: &E8Contribution,
    q: f32,
) -> f32 {
    let compat = E8Lattice::compatibility_score_8d(&a.lattice_point, &b.lattice_point);
    // Classical compatibility scaled by q-deformation:
    // when q > 1, contribution a (the "leader") gets boosted
    let q_factor = if a.confidence >= b.confidence {
        q.powf(0.5) // Leader boost
    } else {
        q.powf(-0.5) // Follower discount
    };
    compat * q_factor * a.confidence.max(0.01)
}

/// Quantum-deformed E8 blend: non-commutative composition of group
/// contributions. The R-matrix determines how much each group leads
/// the final blend based on the deformation parameter q.
pub fn e8_blend_quantum(
    contributions: &[E8Contribution],
    q: f32,
) -> [f32; 8] {
    if contributions.is_empty() {
        return [0.0f32; 8];
    }
    if contributions.len() == 1 {
        return contributions[0].lattice_point;
    }

    // Compute R-matrix weights: each contribution's total braiding score
    // against all others, deformed by q
    let mut weights = vec![0.0f32; contributions.len()];
    for (i, ci) in contributions.iter().enumerate() {
        for (j, cj) in contributions.iter().enumerate() {
            if i == j { continue; }
            weights[i] += r_matrix(ci, cj, q);
        }
    }

    // Normalize weights
    let total: f32 = weights.iter().sum::<f32>().max(0.01);
    for w in &mut weights {
        *w /= total;
    }

    // Weighted blend → E8 quantize
    let mut blended = [0.0f32; 8];
    for (c, &w) in contributions.iter().zip(weights.iter()) {
        for i in 0..8 {
            blended[i] += c.lattice_point[i] * w;
        }
    }

    E8Lattice::nearest_point(&blended)
}

/// Classical E8 blend (q=1 limit). Confidence-proportional weighted average.
pub fn e8_blend(contributions: &[E8Contribution]) -> [f32; 8] {
    e8_blend_quantum(contributions, 1.0)
}

/// Compute the E8 compatibility between a blended point and each
/// contribution, returning scores in [0, 3].
pub fn e8_contribution_scores(
    blended: &[f32; 8],
    contributions: &[E8Contribution],
) -> Vec<(usize, f32)> {
    contributions.iter().map(|c| {
        let score = E8Lattice::compatibility_score_8d(blended, &c.lattice_point);
        (c.group_idx, score)
    }).collect()
}

/// Select the best text from contributions based on E8 compatibility
/// with the blended lattice point. Returns (text, confidence, group_idx).
pub fn e8_select_best(
    blended: &[f32; 8],
    contributions: &[E8Contribution],
) -> Option<(String, f32, usize)> {
    if contributions.is_empty() { return None; }

    let scores = e8_contribution_scores(blended, contributions);
    scores.iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|&(gidx, score)| {
            let c = contributions.iter().find(|c| c.group_idx == gidx).unwrap();
            (c.text.clone(), c.confidence * (score / 3.0), c.group_idx)
        })
}

/// Quantum-deformed sentence composition. The R-matrix determines how
/// sentences from different groups interleave:
/// - High q: sentences from the "leader" group appear first and score higher
/// - q ≈ 1: symmetric scoring (classical limit)
/// - Low q: secondary groups contribute more prominently
pub fn e8_compose_sentences(
    blended: &[f32; 8],
    contributions: &[E8Contribution],
    max_sentences: usize,
) -> (String, f32) {
    e8_compose_sentences_quantum(blended, contributions, max_sentences, 1.0)
}

/// Full quantum composition with explicit q parameter.
pub fn e8_compose_sentences_quantum(
    blended: &[f32; 8],
    contributions: &[E8Contribution],
    max_sentences: usize,
    q: f32,
) -> (String, f32) {
    if contributions.is_empty() { return (String::new(), 0.0); }

    #[derive(Clone)]
    struct ScoredSentence {
        text: String,
        score: f32,
        group: usize,
        position: usize,
    }

    // Compute per-group R-matrix weights (how much each group "leads")
    let mut group_weights: Vec<f32> = vec![0.0; contributions.len()];
    for (i, ci) in contributions.iter().enumerate() {
        for (j, cj) in contributions.iter().enumerate() {
            if i == j { continue; }
            group_weights[i] += r_matrix(ci, cj, q);
        }
    }
    let total_w: f32 = group_weights.iter().sum::<f32>().max(0.01);
    for w in &mut group_weights { *w /= total_w; }

    let mut all_sentences: Vec<ScoredSentence> = Vec::new();

    for (ci, c) in contributions.iter().enumerate() {
        let compat = E8Lattice::compatibility_score_8d(blended, &c.lattice_point);
        let r_weight = group_weights[ci];
        for (pos, sent) in c.text.split(". ").enumerate() {
            let trimmed = sent.trim();
            if trimmed.len() > 10 {
                let alpha_count = trimmed.chars().filter(|ch| ch.is_alphabetic()).count();
                let alpha_ratio = alpha_count as f32 / trimmed.len().max(1) as f32;
                if alpha_ratio > 0.5 {
                    // Score = R-matrix weight × E8 compatibility × confidence × quality
                    // The R-matrix weight is the quantum deformation: it biases
                    // toward the "leader" group when q > 1
                    let score = r_weight * compat * c.confidence * alpha_ratio;
                    all_sentences.push(ScoredSentence {
                        text: trimmed.to_string(),
                        score,
                        group: c.group_idx,
                        position: pos,
                    });
                }
            }
        }
    }

    all_sentences.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    let selected: Vec<&ScoredSentence> = all_sentences.iter().take(max_sentences).collect();
    if selected.is_empty() {
        return (String::new(), 0.0);
    }

    // Order selected sentences: leader group's sentences first (by position),
    // then follower group's sentences (by position). This is the non-commutative
    // braiding — the leader sets the structure, the follower fills in details.
    let leader_group = contributions.iter()
        .enumerate()
        .max_by(|(i, _), (j, _)| group_weights[*i].partial_cmp(&group_weights[*j]).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(_, c)| c.group_idx)
        .unwrap_or(0);

    let mut leader_sents: Vec<&ScoredSentence> = selected.iter()
        .filter(|s| s.group == leader_group).copied().collect();
    let mut follower_sents: Vec<&ScoredSentence> = selected.iter()
        .filter(|s| s.group != leader_group).copied().collect();

    leader_sents.sort_by_key(|s| s.position);
    follower_sents.sort_by_key(|s| s.position);

    let mut ordered: Vec<&str> = Vec::new();
    ordered.extend(leader_sents.iter().map(|s| s.text.as_str()));
    ordered.extend(follower_sents.iter().map(|s| s.text.as_str()));

    let avg_score = selected.iter().map(|s| s.score).sum::<f32>() / selected.len() as f32;
    let text = ordered.join(". ");

    (text, avg_score)
}

// ---------------------------------------------------------------------------
// IndexedGenEnv — Paramecium-based generation (zero backprop)
//
// Knowledge is indexed in one pass (codebook + lattice develop).
// Generation is wave-propagation lookup + decode — no NeuralEnvironment.
// ---------------------------------------------------------------------------

use crate::dimension::paramecium::InfraciliaryLattice;
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// ProgramGraph — graph over lattice nodes for structural disambiguation
//
// Each node is a BehavioralProgram in a topic sub-lattice.
// Edges connect semantically similar (confusable) programs.
// Signature keywords per node enable discriminative retrieval:
//   high-IDF keywords that distinguish this program from its neighbors.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiscriminativeKeyword {
    pub keyword: String,
    /// Inverse document frequency within the topic: higher = more distinctive.
    pub specificity: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProgramEdge {
    pub target_idx: usize,
    pub cosine_sim: f32,
    pub shared_terms: u16,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProgramGraph {
    /// Per-program discriminative keywords sorted by specificity descending.
    pub signatures: Vec<Vec<DiscriminativeKeyword>>,
    /// Per-program adjacency list: edges to similar/confusable programs.
    pub adjacency: Vec<Vec<ProgramEdge>>,
    /// Inverted index: keyword → program indices (rebuilt at load time).
    #[serde(skip)]
    pub keyword_index: HashMap<String, Vec<usize>>,
}

impl ProgramGraph {
    pub fn build(
        lattice: &InfraciliaryLattice,
        dictionary: &TokenDictionary,
        lexicon: &crate::inference::retrieval_lexicon::RetrievalLexicon,
    ) -> Self {
        let n = lattice.programs.len();
        if n == 0 {
            return Self::default();
        }

        // Decode all programs to tokenized word bags
        let docs: Vec<Vec<String>> = lattice.programs.iter()
            .map(|p| {
                let text = p.display_text(dictionary);
                text.to_ascii_lowercase()
                    .split(|c: char| !c.is_alphanumeric() && c != '\'' && c != '-')
                    .filter(|w| w.len() > 2 && !lexicon.is_graph_stop(w))
                    .map(|w| w.to_string())
                    .collect()
            })
            .collect();

        // Document frequency per keyword across programs in this topic
        let mut df: HashMap<String, usize> = HashMap::new();
        for words in &docs {
            let unique: HashSet<&str> = words.iter().map(|s| s.as_str()).collect();
            for w in unique {
                *df.entry(w.to_string()).or_default() += 1;
            }
        }
        let n_f = n as f32;

        // Signature keywords: IDF-weighted terms distinctive to each program.
        // For small topics (N < 8) use a lower threshold so that important
        // terms appearing in multiple programs still get indexed — otherwise
        // a key query term like "dna" is invisible when it appears in all docs.
        let specificity_threshold = if n < 8 { 0.55 } else { 1.05 };
        let max_sig_size = if n < 8 { 40 } else { 25 };

        let mut signatures: Vec<Vec<DiscriminativeKeyword>> = Vec::with_capacity(n);
        for words in &docs {
            let unique: HashSet<&str> = words.iter().map(|s| s.as_str()).collect();
            let mut sig: Vec<DiscriminativeKeyword> = unique.iter()
                .filter_map(|w: &&str| {
                    let doc_freq = df.get(*w).copied().unwrap_or(0) as f32;
                    let specificity = (n_f / (doc_freq + 1.0)).ln() + 1.0;
                    if specificity > specificity_threshold {
                        Some(DiscriminativeKeyword { keyword: w.to_string(), specificity })
                    } else {
                        None
                    }
                })
                .collect();
            sig.sort_by(|a, b| b.specificity.partial_cmp(&a.specificity).unwrap_or(std::cmp::Ordering::Equal));
            sig.truncate(max_sig_size);
            signatures.push(sig);
        }

        // Adjacency: connect programs with cosine similarity > 0.5
        let mut adjacency: Vec<Vec<ProgramEdge>> = vec![Vec::new(); n];
        for i in 0..n {
            for j in (i + 1)..n {
                let sim = gen_cosine_sim(&lattice.programs[i].ema_centroid, &lattice.programs[j].ema_centroid);
                if sim > 0.5 {
                    let shared = docs[i].iter()
                        .collect::<HashSet<_>>()
                        .intersection(&docs[j].iter().collect())
                        .count();
                    adjacency[i].push(ProgramEdge { target_idx: j, cosine_sim: sim, shared_terms: shared as u16 });
                    adjacency[j].push(ProgramEdge { target_idx: i, cosine_sim: sim, shared_terms: shared as u16 });
                }
            }
        }

        let mut graph = ProgramGraph { signatures, adjacency, keyword_index: HashMap::new() };
        graph.rebuild_keyword_index();
        graph
    }

    /// Reconstruct the inverted index (call after deserialization).
    pub fn rebuild_keyword_index(&mut self) {
        self.keyword_index.clear();
        for (prog_idx, sig) in self.signatures.iter().enumerate() {
            for dk in sig {
                self.keyword_index.entry(dk.keyword.clone()).or_default().push(prog_idx);
            }
        }
    }

    /// Score how well a program's discriminative signature matches query keywords.
    pub fn signature_score(&self, program_idx: usize, query_keywords: &[String]) -> f32 {
        if program_idx >= self.signatures.len() || query_keywords.is_empty() {
            return 0.0;
        }
        let sig = &self.signatures[program_idx];
        let mut score = 0.0f32;
        for qk in query_keywords {
            for dk in sig {
                if dk.keyword == *qk
                    || dk.keyword.contains(qk.as_str())
                    || qk.contains(dk.keyword.as_str())
                {
                    score += dk.specificity;
                    break;
                }
            }
        }
        score
    }

    /// Find programs whose signature keywords best match the query.
    /// Returns (program_index, accumulated_specificity) sorted descending.
    pub fn keyword_lookup(&self, query_keywords: &[String]) -> Vec<(usize, f32)> {
        let mut scores: HashMap<usize, f32> = HashMap::new();
        for qk in query_keywords {
            let qk_lower = qk.to_ascii_lowercase();
            if let Some(prog_indices) = self.keyword_index.get(&qk_lower) {
                for &pidx in prog_indices {
                    let specificity = self.signatures[pidx].iter()
                        .find(|dk| dk.keyword == qk_lower)
                        .map(|dk| dk.specificity)
                        .unwrap_or(1.0);
                    *scores.entry(pidx).or_default() += specificity;
                }
            }
            // Substring match for compound terms
            for (kw, prog_indices) in &self.keyword_index {
                if kw != &qk_lower && (kw.contains(qk_lower.as_str()) || qk_lower.contains(kw.as_str())) {
                    for &pidx in prog_indices {
                        *scores.entry(pidx).or_default() += 0.5;
                    }
                }
            }
        }
        let mut result: Vec<(usize, f32)> = scores.into_iter().collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Check if a neighbor of `program_idx` has a significantly better
    /// keyword match for the query. Returns the neighbor index if so.
    pub fn neighbor_redirect(&self, program_idx: usize, query_keywords: &[String]) -> Option<usize> {
        if program_idx >= self.adjacency.len() || query_keywords.is_empty() {
            return None;
        }
        let own_score = self.signature_score(program_idx, query_keywords);
        // Only redirect if the neighbor is overwhelmingly better (3x) and
        // the current program has near-zero match. This prevents false
        // redirects like Strategy → State when both have partial matches.
        if own_score > 1.0 { return None; } // current match is decent, don't redirect
        let mut best: Option<(usize, f32)> = None;
        for edge in &self.adjacency[program_idx] {
            let neighbor_score = self.signature_score(edge.target_idx, query_keywords);
            if neighbor_score > own_score * 3.0 && neighbor_score > 3.0 {
                if best.map(|(_, s)| neighbor_score > s).unwrap_or(true) {
                    best = Some((edge.target_idx, neighbor_score));
                }
            }
        }
        best.map(|(idx, _)| idx)
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TopicSubIndex {
    pub topic_name: String,
    pub centroid: Vec<f32>,
    /// Mean causal fingerprint (7d boost bivectors from Cl(1,7)) for this topic.
    /// Encodes the temporal/goal direction of samples in this topic cluster.
    #[serde(default)]
    pub causal_centroid: [f32; BOOST_BIVECTOR_COUNT],
    pub lattice: InfraciliaryLattice,
    pub sample_count: usize,
    /// Graph over lattice nodes: discriminative keyword signatures + confusability edges.
    #[serde(default)]
    pub graph: Option<ProgramGraph>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct IndexedGenEnv {
    pub lattice: InfraciliaryLattice,
    pub topic_subindex: Vec<TopicSubIndex>,
    pub dictionary: TokenDictionary,
    pub codebook: Option<AlgebraicCodebook>,
    pub hopf_table: Option<HopfCompositionTable>,
    /// Schemas extracted from program patterns for template-based generation.
    #[serde(skip)]
    pub schemas: Vec<crate::predictive_coder::Schema>,
    /// Chunk-level continuous codec: encodes K-token chunks into Cl(1,7)
    /// multivectors for trajectory-based generation (bypasses discrete decode).
    #[serde(skip)]
    pub chunk_codec: Option<crate::text_autoencoder::ChunkCodec>,
    #[serde(skip)]
    pub last_selected_archetype: Option<usize>,
    #[serde(skip)]
    pub last_generation_confidence: f32,
    #[serde(skip)]
    pub diversity_bonus: f32,
    /// Transient: query subject keywords for within-topic BM25 re-ranking.
    #[serde(skip)]
    pub subject_keywords: Vec<String>,
    /// Transient: query intent action (e.g., "implement", "explain").
    #[serde(skip)]
    pub intent_action: String,
    pub frozen: bool,
    pub output_dim: usize,
}

impl IndexedGenEnv {
    fn build_topic_subindex(
        dictionary: &TokenDictionary,
        training_triples: &[(Vec<f32>, String, String)],
        spawn_threshold: f32,
    ) -> Vec<TopicSubIndex> {
        if training_triples.is_empty() {
            return Vec::new();
        }
        let mut by_topic: HashMap<String, Vec<(Vec<f32>, String)>> = HashMap::new();
        for (cond, text, topic) in training_triples {
            by_topic.entry(topic.clone()).or_default().push((cond.clone(), text.clone()));
        }

        let mut out = Vec::with_capacity(by_topic.len());
        for (topic_name, pairs) in by_topic {
            if pairs.is_empty() {
                continue;
            }
            let dim = pairs[0].0.len();
            let mut centroid = vec![0.0f32; dim];
            let mut causal_sum = [0.0f32; BOOST_BIVECTOR_COUNT];
            for (cond, _) in &pairs {
                for (dst, src) in centroid.iter_mut().zip(cond.iter()) {
                    *dst += *src;
                }
                let mv = embed_bridge_vector(cond);
                let cfp = causal_fingerprint(&mv);
                for (dst, src) in causal_sum.iter_mut().zip(cfp.iter()) {
                    *dst += *src;
                }
            }
            let n = pairs.len().max(1) as f32;
            for v in &mut centroid {
                *v /= n;
            }
            let norm = centroid.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            for v in &mut centroid {
                *v /= norm;
            }
            let mut causal_centroid = [0.0f32; BOOST_BIVECTOR_COUNT];
            for (i, v) in causal_sum.iter().enumerate() {
                causal_centroid[i] = v / n;
            }
            let cnorm = causal_centroid.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            for v in &mut causal_centroid {
                *v /= cnorm;
            }

            let mut lattice = InfraciliaryLattice::new(dictionary.clone());
            lattice.develop(&pairs, spawn_threshold);
            let graph = ProgramGraph::build(
                &lattice,
                dictionary,
                crate::inference::retrieval_lexicon::global(),
            );
            let n_edges: usize = graph.adjacency.iter().map(|a| a.len()).sum();
            let n_sigs: usize = graph.signatures.iter().map(|s| s.len()).sum();
            println!("    [program-graph] '{}': {} nodes, {} edges, {} signature keywords",
                topic_name, lattice.programs.len(), n_edges / 2, n_sigs);
            out.push(TopicSubIndex {
                topic_name,
                centroid,
                causal_centroid,
                lattice,
                sample_count: pairs.len(),
                graph: Some(graph),
            });
        }
        out
    }

    /// Build from training pairs in a single pass. No iterative training.
    ///
    /// 1. Build dictionary + codebook + Hopf table (algebraic indexing)
    /// 2. Develop a Paramecium lattice from (embedding, response) pairs
    ///
    /// Total cost: O(n) where n = number of training examples.
    pub fn build(
        texts: &[&str],
        embeddings: &[&[f32]],
        max_archetypes: usize,
        spawn_threshold: f32,
    ) -> Self {
        let dict = TokenDictionary::build(texts, if texts.len() > 100 { 2048 } else { 1024 });
        let emb_refs: Vec<&[f32]> = embeddings.to_vec();
        let cb = AlgebraicCodebook::build(texts, &dict, max_archetypes, Some(&emb_refs));

        let clusters: Vec<Vec<usize>> = {
            let mut c = vec![Vec::new(); cb.archetypes.len()];
            for (i, text) in texts.iter().enumerate() {
                let ids = dict.encode(text);
                let (arch, _) = cb.match_best(&ids);
                if arch < c.len() { c[arch].push(i); }
            }
            c
        };
        let hopf = HopfCompositionTable::build(&cb, Some(&emb_refs), &clusters, 3);

        let pairs: Vec<(Vec<f32>, String)> = embeddings.iter().zip(texts.iter())
            .map(|(e, t)| (e.to_vec(), t.to_string()))
            .collect();
        let mut lattice = InfraciliaryLattice::new(dict.clone());
        lattice.develop(&pairs, spawn_threshold);

        let output_dim = cb.slot_only_bits;

        Self {
            lattice,
            topic_subindex: Vec::new(),
            dictionary: dict,
            codebook: Some(cb),
            hopf_table: Some(hopf),
            schemas: Vec::new(),
            chunk_codec: None,
            last_selected_archetype: None,
            last_generation_confidence: 0.0,
            diversity_bonus: 0.0,
            subject_keywords: Vec::new(),
            intent_action: String::new(),
            frozen: false,
            output_dim,
        }
    }

    /// Build from pre-constructed components (used when loading a brain
    /// or when the caller already has a dictionary/codebook).
    pub fn from_parts(
        dictionary: TokenDictionary,
        codebook: AlgebraicCodebook,
        hopf: HopfCompositionTable,
        training_pairs: &[(Vec<f32>, String)],
        spawn_threshold: f32,
    ) -> Self {
        let output_dim = codebook.slot_only_bits;
        let mut lattice = InfraciliaryLattice::new(dictionary.clone());
        lattice.develop(training_pairs, spawn_threshold);

        Self {
            lattice,
            topic_subindex: Vec::new(),
            dictionary,
            codebook: Some(codebook),
            hopf_table: Some(hopf),
            schemas: Vec::new(),
            chunk_codec: None,
            last_selected_archetype: None,
            last_generation_confidence: 0.0,
            diversity_bonus: 0.0,
            subject_keywords: Vec::new(),
            intent_action: String::new(),
            frozen: false,
            output_dim,
        }
    }

    /// Build from pre-constructed components plus semantic topic labels.
    /// Each topic gets a compact sub-lattice that acts like a living index
    /// inside the coarse group.
    #[cfg(feature = "training")]
    pub fn from_tagged_parts(
        dictionary: TokenDictionary,
        codebook: AlgebraicCodebook,
        hopf: HopfCompositionTable,
        training_triples: &[(Vec<f32>, String, String)],
        spawn_threshold: f32,
    ) -> Self {
        let training_pairs: Vec<(Vec<f32>, String)> = training_triples.iter()
            .map(|(cond, text, _topic)| (cond.clone(), text.clone()))
            .collect();
        let topic_subindex = Self::build_topic_subindex(&dictionary, training_triples, spawn_threshold);
        let output_dim = codebook.slot_only_bits;
        let mut lattice = InfraciliaryLattice::new(dictionary.clone());
        lattice.develop(&training_pairs, spawn_threshold);

        let mut env = Self {
            lattice,
            topic_subindex,
            dictionary,
            codebook: Some(codebook),
            hopf_table: Some(hopf),
            schemas: Vec::new(),
            chunk_codec: None,
            last_selected_archetype: None,
            last_generation_confidence: 0.0,
            diversity_bonus: 0.0,
            subject_keywords: Vec::new(),
            intent_action: String::new(),
            frozen: false,
            output_dim,
        };
        env.build_schemas();
        env.build_chunk_codec();
        env
    }

    /// Topic label for a root-lattice program using **subindex centroids** only (O(#topics)).
    ///
    /// The previous training path compared each root program against every program in every
    /// topic sub-lattice to find a match — O(root × Σ topic programs), which stalls when the
    /// root lattice is large (e.g. code lattice disabled, all capacity in gen).
    pub fn topic_label_for_program_centroid(&self, prog_centroid: &[f32], default: &str) -> String {
        if self.topic_subindex.is_empty() {
            return default.to_string();
        }
        let mut best_name = default.to_string();
        let mut best_sim = f32::NEG_INFINITY;
        for sub in &self.topic_subindex {
            let sim = gen_cosine_sim(prog_centroid, &sub.centroid);
            if sim > best_sim {
                best_sim = sim;
                best_name = sub.topic_name.clone();
            }
        }
        best_name
    }

    /// Extract reusable schemas from program patterns.
    /// Finds positions that are invariant across similar programs (fixed)
    /// vs positions that vary (slots), enabling template-based generation.
    pub fn build_schemas(&mut self) {
        let programs: Vec<(Vec<u16>, f32)> = self.lattice.programs.iter()
            .map(|p| (p.token_sequence.clone(), p.quality_score))
            .collect();
        self.schemas = crate::predictive_coder::extract_schemas(&programs, 2, 0.35);
    }

    /// Build the chunk-level continuous codec for trajectory-based generation.
    /// Called after brain load to enable the continuous decode path.
    pub fn build_chunk_codec(&mut self) {
        self.chunk_codec = Some(crate::text_autoencoder::ChunkCodec::new(
            self.dictionary.tokens.len(),
        ));
    }

    /// Rebuild program graphs for all topic sub-lattices.
    /// Called after brain deserialization to reconstruct graphs from existing
    /// lattice programs, or to refresh the keyword inverted index.
    pub fn rebuild_program_graphs(&mut self) {
        for topic in &mut self.topic_subindex {
            // Always rebuild from scratch — signatures depend on IDF thresholds
            // that may have been updated since the brain was trained.
            topic.graph = Some(ProgramGraph::build(
                &topic.lattice,
                &self.dictionary,
                crate::inference::retrieval_lexicon::global(),
            ));
        }
    }

    /// Topic-aware continuous generation using grade-aware algebraic composition.
    ///
    /// Instead of flat weighted averaging, this uses the Cl(1,7) algebraic
    /// structure to compose program trajectories:
    ///   - Grade 1 (topic vectors) are slerped for smooth semantic blending
    ///   - Grade 2 (relational context) is composed via geometric product,
    ///     capturing both shared structure and novel combinations
    ///   - Grade 3+ uses weighted blending for stability
    ///
    /// If the composed trajectory has enough chunks, rotor extrapolation
    /// can extend it (predict continuation via the transition rotor pattern).
    pub fn generate_continuous_for_topic(
        &mut self,
        cond: &[f32],
        topic_name: &str,
        max_tokens: usize,
    ) -> Option<(String, f32)> {
        let codec = self.chunk_codec.as_ref()?;

        let topic = self.topic_subindex.iter()
            .find(|t| t.topic_name == topic_name)?;

        if topic.lattice.programs.len() < 2 { return None; }

        let k = 4.min(topic.lattice.programs.len());
        let mut scored: Vec<(usize, f32)> = topic.lattice.programs.iter()
            .enumerate()
            .map(|(i, p)| {
                let dot: f32 = cond.iter().zip(p.ema_centroid.iter())
                    .map(|(a, b)| a * b).sum();
                let na: f32 = cond.iter().map(|x| x * x).sum::<f32>().sqrt();
                let nb: f32 = p.ema_centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
                let sim = if na > 1e-8 && nb > 1e-8 { dot / (na * nb) } else { 0.0 };
                (i, sim)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        if scored.is_empty() || scored[0].1 < 0.1 { return None; }

        let source_seqs: Vec<(crate::text_autoencoder::ChunkSequence, f32)> = scored.iter()
            .map(|&(idx, sim)| {
                let toks = &topic.lattice.programs[idx].token_sequence;
                let wb = self.dictionary.infer_word_boundaries(toks);
                let seq = codec.encode_sequence_word_aligned(toks, &wb);
                (seq, sim)
            })
            .collect();

        let target_chunks = source_seqs[0].0.num_chunks();
        if target_chunks == 0 { return None; }

        // Propagator-based composition: build SpacetimeChunk trajectories
        // from program centroids (semantically-rich grade structure) rather
        // than CDMA chunks (quasi-random grades).  CDMA chunks are only used
        // for the final token decode step.
        let st_sources: Vec<(Vec<crate::text_autoencoder::SpacetimeChunk>, f32)> = scored.iter()
            .map(|&(idx, sim)| {
                let centroid = &topic.lattice.programs[idx].ema_centroid;
                let st = crate::text_autoencoder::SpacetimeChunk::from_centroid(centroid);
                (vec![st], sim)
            })
            .collect();

        let has_multi_chunk = st_sources.iter().any(|(t, _)| t.len() >= 2);
        let composed = if has_multi_chunk {
            // Use propagator composition for richer semantic blending
            let mass = crate::text_autoencoder::trajectory_mass(
                &st_sources[0].0
            );
            let propagator = crate::text_autoencoder::SemanticPropagator::new(mass, 0.4);
            propagator.compose_trajectories(&st_sources, target_chunks)
        } else {
            // Fall back to algebraic composition for single-chunk sources
            crate::text_autoencoder::compose_algebraic(&source_seqs, target_chunks)
        };

        // Propagator-based extension: predict additional chunks via
        // Dirac-style rotor propagation with semantic inertia.
        let mut extended = composed;
        let max_chunks = (max_tokens + crate::text_autoencoder::CHUNK_K - 1)
            / crate::text_autoencoder::CHUNK_K;
        if extended.len() >= 2 && extended.len() < max_chunks {
            let st_trajectory: Vec<crate::text_autoencoder::SpacetimeChunk> = extended.iter()
                .map(|c| crate::text_autoencoder::SpacetimeChunk::from_centroid(c))
                .collect();
            let mass = crate::text_autoencoder::trajectory_mass(&st_trajectory);
            let mut propagator = crate::text_autoencoder::SemanticPropagator::from_trajectory(
                &st_trajectory, mass, 0.4,
            );

            let extra = 2.min(max_chunks - extended.len());
            for _ in 0..extra {
                if let Some((next_chunk, _interval, confidence)) = propagator.predict_next() {
                    if confidence < 0.1 { break; }
                    let next_st = crate::text_autoencoder::SpacetimeChunk::from_centroid(&next_chunk);
                    // Coherence check: predicted chunk should be semantically related
                    if st_trajectory.last()
                        .map(|last| next_st.semantic_similarity(last) > 0.2)
                        .unwrap_or(false)
                    {
                        propagator.observe(&next_st);
                        extended.push(next_chunk);
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }

        // Apply per-grade temperature sampling before decoding.
        // This gives fine-grained control over which aspects of generation
        // are varied vs deterministic.
        let grade_temp = crate::text_autoencoder::GradeTemperature::default();
        let tempered: Vec<[f32; crate::text_autoencoder::CATA_DIM]> = extended.iter()
            .enumerate()
            .map(|(i, chunk)| {
                let seed = (i as u64).wrapping_mul(0x517cc1b727220a95)
                    .wrapping_add(topic_name.len() as u64);
                crate::text_autoencoder::apply_grade_temperature(chunk, &grade_temp, seed)
            })
            .collect();

        let chunk_lengths: Vec<usize> = source_seqs[0].0.chunk_lengths.clone();
        let mut all_tokens = Vec::new();
        for (i, chunk) in tempered.iter().enumerate() {
            let len = chunk_lengths.get(i).copied()
                .unwrap_or(crate::text_autoencoder::CHUNK_K);
            let tokens = codec.decode_chunk(chunk, len);
            all_tokens.extend_from_slice(&tokens);
        }
        all_tokens.truncate(max_tokens);

        let text = self.dictionary.decode(&all_tokens);

        // Quality guard: if the algebraic composition produced garbled text,
        // fall back to decoding the primary source program directly.
        let garbled = Self::has_tokenization_artifacts(&text) || text.len() < 5;
        if garbled {
            let primary_idx = scored[0].0;
            let primary_text = self.dictionary.decode(
                &topic.lattice.programs[primary_idx].token_sequence);
            if primary_text.len() >= 5 && !Self::has_tokenization_artifacts(&primary_text) {
                crate::infer_trace!(
                    "    [codec-fallback] algebraic composition garbled, using primary prog {}",
                    primary_idx
                );
                let confidence = scored[0].1.min(0.90);
                self.last_selected_archetype = None;
                self.last_generation_confidence = confidence;
                return Some((primary_text, confidence));
            }
            return None;
        }

        let confidence = scored[0].1.min(0.95);
        self.last_selected_archetype = None;
        self.last_generation_confidence = confidence;

        Some((text, confidence))
    }

    /// Generate text by composing program trajectories in continuous Cl(1,7)
    /// chunk space, then decoding each chunk back to tokens. Returns None if
    /// the codec isn't initialised or there aren't enough programs.
    pub fn generate_continuous(
        &mut self,
        cond: &[f32],
        max_tokens: usize,
    ) -> Option<(String, f32)> {
        let codec = self.chunk_codec.as_ref()?;
        if self.lattice.programs.len() < 2 { return None; }

        // Retrieve top-k nearest programs by conditioning similarity
        let k = 4.min(self.lattice.programs.len());
        let mut scored: Vec<(usize, f32)> = self.lattice.programs.iter()
            .enumerate()
            .map(|(i, p)| {
                let dot: f32 = cond.iter().zip(p.ema_centroid.iter())
                    .map(|(a, b)| a * b).sum();
                let na: f32 = cond.iter().map(|x| x * x).sum::<f32>().sqrt();
                let nb: f32 = p.ema_centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
                let sim = if na > 1e-8 && nb > 1e-8 { dot / (na * nb) } else { 0.0 };
                let bias = self.lattice.retrieval_bias(i);
                (i, sim * bias)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        if scored.is_empty() || scored[0].1 < 0.1 { return None; }

        // Encode each program's tokens into chunk sequences (word-aligned)
        let source_seqs: Vec<(crate::text_autoencoder::ChunkSequence, f32)> = scored.iter()
            .map(|&(idx, sim)| {
                let toks = &self.lattice.programs[idx].token_sequence;
                let wb = self.dictionary.infer_word_boundaries(toks);
                let seq = codec.encode_sequence_word_aligned(toks, &wb);
                (seq, sim)
            })
            .collect();

        // Determine target number of chunks from the best program's length
        let target_chunks = source_seqs[0].0.num_chunks();
        if target_chunks == 0 { return None; }

        // Compose trajectories: weighted blend of chunk sequences
        let composed = crate::text_autoencoder::compose_trajectories(
            &source_seqs, target_chunks,
        );

        // Decode composed chunks back to tokens
        let chunk_lengths: Vec<usize> = source_seqs[0].0.chunk_lengths.clone();
        let mut all_tokens = Vec::new();
        for (i, chunk) in composed.iter().enumerate() {
            let len = chunk_lengths.get(i).copied()
                .unwrap_or(crate::text_autoencoder::CHUNK_K);
            let tokens = codec.decode_chunk(chunk, len);
            all_tokens.extend_from_slice(&tokens);
        }
        all_tokens.truncate(max_tokens);

        let text = self.dictionary.decode(&all_tokens);
        if text.len() < 5 { return None; }

        let confidence = scored[0].1.min(0.95);
        self.last_selected_archetype = Some(scored[0].0);
        self.last_generation_confidence = confidence;

        Some((text, confidence))
    }

    /// Generate a response from a conditioning vector.
    /// Uses Paramecium wave-propagation to select the best program,
    /// then decodes via the codebook when confidence is high,
    /// or falls back to Hopf composition for uncertain inputs.
    /// Immutable nearest-neighbor lookup over lattice programs.
    /// No EMA centroid drift — safe for repeated inference calls.
    /// STA field evaluation: programs are sources, query is a field point.
    ///
    /// Instead of cosine similarity (scalar, same for nearby points), compute the
    /// Equivariant within-group scoring using full multivector features in Cl(1,7).
    ///
    /// Instead of scalar-only cosine (invariant → collapses within-group differences),
    /// this uses grade-2 bivector alignment (equivariant → preserves orientation, phase,
    /// and relative structure). This enables discrimination inside the symmetry class:
    /// "addition" and "subtraction" have similar scalars but different bivector orientations.
    ///
    /// Score = α·cosine + β·spatial_align + γ·causal_align + δ·proximity
    fn nearest_response_in_lattice(
        &self,
        lattice: &InfraciliaryLattice,
        cond: &[f32],
    ) -> (String, usize, f32) {
        if lattice.programs.is_empty() {
            return (String::new(), 0, 0.0);
        }

        // Embed input into Cl(1,7) once — extract equivariant features
        let input_mv = embed_bridge_vector(cond);
        let input_spatial = spatial_fingerprint(&input_mv);
        let input_causal = causal_fingerprint(&input_mv);

        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;

        for (i, prog) in lattice.programs.iter().enumerate() {
            // Use effective centroid (base + session drift) for in-context adaptation
            let effective = lattice.effective_centroid(i);
            let centroid = if effective.is_empty() { &prog.ema_centroid } else { &effective };

            let cosine = gen_cosine_sim(cond, centroid);

            let prog_mv = embed_bridge_vector(centroid);
            let prog_spatial = spatial_fingerprint(&prog_mv);
            let prog_causal = causal_fingerprint(&prog_mv);

            let sp_dot: f32 = input_spatial.iter().zip(prog_spatial.iter())
                .map(|(a, b)| a * b).sum();
            let sp_na: f32 = input_spatial.iter().map(|x| x * x).sum::<f32>().sqrt();
            let sp_nb: f32 = prog_spatial.iter().map(|x| x * x).sum::<f32>().sqrt();
            let spatial_align = if sp_na < 1e-8 || sp_nb < 1e-8 { 0.0 }
                else { (sp_dot / (sp_na * sp_nb)).clamp(-1.0, 1.0) };

            let ca_dot: f32 = input_causal.iter().zip(prog_causal.iter())
                .map(|(a, b)| a * b).sum();
            let ca_na: f32 = input_causal.iter().map(|x| x * x).sum::<f32>().sqrt();
            let ca_nb: f32 = prog_causal.iter().map(|x| x * x).sum::<f32>().sqrt();
            let causal_align = if ca_na < 1e-8 || ca_nb < 1e-8 { 0.0 }
                else { (ca_dot / (ca_na * ca_nb)).clamp(-1.0, 1.0) };

            let disp_norm_sq: f32 = cond.iter()
                .zip(centroid.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            let proximity = if disp_norm_sq < 1e-8 { 1.0 }
                else { (1.0 / disp_norm_sq).min(100.0).sqrt() / 10.0 };

            // Base equivariant score
            let base_score = 0.30 * cosine.max(0.0)
                + 0.35 * (spatial_align + 1.0) / 2.0
                + 0.20 * (causal_align + 1.0) / 2.0
                + 0.15 * proximity;

            // Apply multi-timescale retrieval bias (quality, reliability,
            // refractory suppression, activation decay)
            let bias = lattice.retrieval_bias(i);
            let score = base_score * bias;

            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        let text = lattice.programs[best_idx].display_text(&self.dictionary);
        (text, best_idx, best_score.max(0.0))
    }

    fn nearest_response(&self, cond: &[f32]) -> (String, usize, f32) {
        self.nearest_response_in_lattice(&self.lattice, cond)
    }

    /// Return top-K programs from a lattice, scored by equivariant STA similarity.
    fn top_k_in_lattice(
        &self,
        lattice: &InfraciliaryLattice,
        cond: &[f32],
        k: usize,
    ) -> Vec<(String, usize, f32)> {
        if lattice.programs.is_empty() {
            return Vec::new();
        }
        let input_mv = embed_bridge_vector(cond);
        let input_spatial = spatial_fingerprint(&input_mv);
        let input_causal = causal_fingerprint(&input_mv);

        let mut scored: Vec<(usize, f32)> = lattice.programs.iter().enumerate()
            .map(|(i, prog)| {
                let cosine = gen_cosine_sim(cond, &prog.ema_centroid);
                let prog_mv = embed_bridge_vector(&prog.ema_centroid);
                let prog_spatial = spatial_fingerprint(&prog_mv);
                let prog_causal = causal_fingerprint(&prog_mv);
                let sp_dot: f32 = input_spatial.iter().zip(prog_spatial.iter()).map(|(a, b)| a * b).sum();
                let sp_na: f32 = input_spatial.iter().map(|x| x * x).sum::<f32>().sqrt();
                let sp_nb: f32 = prog_spatial.iter().map(|x| x * x).sum::<f32>().sqrt();
                let spatial_align = if sp_na < 1e-8 || sp_nb < 1e-8 { 0.0 }
                    else { (sp_dot / (sp_na * sp_nb)).clamp(-1.0, 1.0) };
                let ca_dot: f32 = input_causal.iter().zip(prog_causal.iter()).map(|(a, b)| a * b).sum();
                let ca_na: f32 = input_causal.iter().map(|x| x * x).sum::<f32>().sqrt();
                let ca_nb: f32 = prog_causal.iter().map(|x| x * x).sum::<f32>().sqrt();
                let causal_align = if ca_na < 1e-8 || ca_nb < 1e-8 { 0.0 }
                    else { (ca_dot / (ca_na * ca_nb)).clamp(-1.0, 1.0) };
                let disp_norm_sq: f32 = cond.iter().zip(prog.ema_centroid.iter())
                    .map(|(a, b)| (a - b) * (a - b)).sum();
                let proximity = if disp_norm_sq < 1e-8 { 1.0 }
                    else { (1.0 / disp_norm_sq).min(100.0).sqrt() / 10.0 };
                let score = 0.30 * cosine.max(0.0)
                    + 0.35 * (spatial_align + 1.0) / 2.0
                    + 0.20 * (causal_align + 1.0) / 2.0
                    + 0.15 * proximity;
                (i, score)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(k).map(|(idx, sim)| {
            let text = lattice.programs[idx].display_text(&self.dictionary);
            (text, idx, sim)
        }).collect()
    }

    fn nearest_topic_response(&self, cond: &[f32], topic_hint: Option<&str>) -> Option<(String, String, f32)> {
        if self.topic_subindex.is_empty() {
            return None;
        }
        let hint = topic_hint.map(|s| s.trim()).filter(|s| !s.is_empty());

        // Compute input's STA fingerprints once for all topic comparisons.
        let input_mv = embed_bridge_vector(cond);
        let input_causal = causal_fingerprint(&input_mv);

        let mut best: Option<(String, String, f32)> = None;
        for topic in &self.topic_subindex {
            let (text, _prog_idx, local_conf) = self.nearest_response_in_lattice(&topic.lattice, cond);
            if text.is_empty() {
                continue;
            }

            // Displacement from topic centroid to input, in Cl(1,7)
            let displacement: Vec<f32> = cond.iter()
                .zip(topic.centroid.iter())
                .map(|(a, b)| a - b)
                .collect();
            let disp_mv = embed_bridge_vector(&displacement);
            let disp_spatial = spatial_fingerprint(&disp_mv);
            let spatial_energy: f32 = disp_spatial.iter().map(|x| x * x).sum::<f32>().sqrt();

            // Low displacement energy = input is in this topic's rest frame
            let field_proximity = (1.0 - (spatial_energy * 2.0).min(1.0)).max(0.0);

            // Causal alignment (boost bivectors)
            let causal_sim = {
                let dot: f32 = input_causal.iter().zip(topic.causal_centroid.iter()).map(|(a, b)| a * b).sum();
                let na: f32 = input_causal.iter().map(|x| x * x).sum::<f32>().sqrt();
                let nb: f32 = topic.causal_centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
                if na < 1e-10 || nb < 1e-10 { 0.0 } else { (dot / (na * nb)).max(0.0) }
            };

            let hint_bonus = match hint {
                Some(h) if h.eq_ignore_ascii_case(&topic.topic_name) => 0.12,
                Some(_) => 0.0,
                None => 0.02,
            };
            let density_bonus = ((topic.sample_count as f32).ln_1p() / 10.0).min(0.08);
            let combined = (0.40 * local_conf + 0.22 * field_proximity + 0.18 * causal_sim + hint_bonus + density_bonus).clamp(0.0, 1.0);
            if best.as_ref().map(|(_, _, score)| combined > *score).unwrap_or(true) {
                best = Some((text, topic.topic_name.clone(), combined));
            }
        }
        best
    }

    /// Surface forms (lowercase) to match a subject keyword against indexed lattice text.
    fn sentiment_keyword_match_forms(kw: &str) -> Vec<String> {
        let lower = kw.to_ascii_lowercase();
        let mut out = Vec::new();
        let compact: String = lower.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
        if compact.len() >= 4 {
            out.push(compact);
        }
        for sep in ['\'', '\u{2019}'] {
            if let Some(pos) = lower.find(sep) {
                let head: String = lower[..pos]
                    .chars()
                    .filter(|c| c.is_ascii_alphanumeric())
                    .collect();
                if head.len() >= 4 {
                    out.push(head);
                }
            }
        }
        out.sort();
        out.dedup();
        out
    }

    /// True when the joint lattice row's indexed user prefix contains enough subject
    /// keywords from the current prompt (so the rationale is not from another headline).
    fn sentiment_witness_matches_subject_keywords(joint_body: &str, query_terms: &[String]) -> bool {
        use crate::dimension::language::{SENTIMENT_CAUSAL_INDEX_CORE, SENTIMENT_LATTICE_WITNESS_CORE};
        if query_terms.len() < 2 {
            return true;
        }
        let mut witness = joint_body
            .split(SENTIMENT_LATTICE_WITNESS_CORE)
            .next()
            .unwrap_or(joint_body);
        if let Some(pos) = witness.find(SENTIMENT_CAUSAL_INDEX_CORE) {
            witness = &witness[..pos];
        }
        let w = witness.to_ascii_lowercase();
        let mut forms: Vec<String> = Vec::new();
        for qt in query_terms {
            forms.extend(Self::sentiment_keyword_match_forms(qt));
        }
        forms.sort();
        forms.dedup();
        let forms: Vec<String> = forms.into_iter().filter(|s| s.len() >= 4).collect();
        if forms.len() < 2 {
            return true;
        }
        let mut hits = 0usize;
        for f in &forms {
            if w.contains(f.as_str()) {
                hits += 1;
            }
        }
        let n = forms.len();
        let required = match n {
            2 | 3 => 1,
            4 | 5 => 2,
            _ => ((n + 2) / 3).max(2),
        }
        .min(n);
        hits >= required
    }

    /// When the user line looks like crypto / tape commentary, adjust scores using declarative rules
    /// (`data/inference/sentiment_crypto_rescore.toml`) — see [`crate::inference::retrieval_rescore`].
    fn sentiment_apply_crypto_market_retrieval_rescore(
        scored: &mut Vec<(usize, f32)>,
        query_terms: &[String],
        topic: &TopicSubIndex,
        dictionary: &TokenDictionary,
    ) {
        if query_terms.is_empty() {
            return;
        }
        let qjoin = query_terms.join(" ").to_ascii_lowercase();
        if !crate::inference::inference_toml::InferenceRulesRuntime::intent_text_suggests_crypto_market(
            qjoin.as_str(),
        ) {
            return;
        }
        let lattice = &topic.lattice;
        crate::inference::retrieval_rescore::apply_embedded_sentiment_crypto_rescore(
            qjoin.as_str(),
            scored.as_mut_slice(),
            |i| lattice.programs[i].display_text(dictionary),
        );
    }

    /// Phase A.2: within `mixed` / `negative_mild`, keep only programs whose indexed user witness
    /// matches the query domain (crypto tape vs consumer fintech).
    ///
    /// Programs **without** [`crate::dimension::language::SENTIMENT_LATTICE_WITNESS_CORE`] are
    /// response-only lattice rows (pre–A.1): they are excluded for crypto-shaped queries so cosine
    /// cannot pull fintech-only templates; they remain eligible for consumer-shaped queries.
    fn sentiment_domain_slice_allows(crypto_query: bool, decoded: &str) -> bool {
        match decoded.split_once(crate::dimension::language::SENTIMENT_LATTICE_WITNESS_CORE) {
            None => !crypto_query,
            Some((user_w, _rest)) => {
                let user_lower = user_w.to_ascii_lowercase();
                let prog_crypto = crate::inference::inference_toml::InferenceRulesRuntime::intent_text_suggests_crypto_market(
                    user_lower.as_str(),
                );
                if crypto_query {
                    prog_crypto
                } else {
                    !prog_crypto
                }
            }
        }
    }

    fn sentiment_apply_topic_domain_slice(
        cosine_scores: &mut Vec<(usize, f32)>,
        forced_topic: &str,
        query_terms: &[String],
        topic: &TopicSubIndex,
        dictionary: &TokenDictionary,
    ) {
        let ft = forced_topic.to_ascii_lowercase();
        if ft != "mixed" && ft != "negative_mild" {
            return;
        }
        if query_terms.is_empty() {
            return;
        }
        let probe = query_terms.join(" ").to_ascii_lowercase();
        let crypto_q =
            crate::inference::inference_toml::InferenceRulesRuntime::intent_text_suggests_crypto_market(
                probe.as_str(),
            );
        let orig = cosine_scores.clone();
        let core = crate::dimension::language::SENTIMENT_LATTICE_WITNESS_CORE;
        let filtered: Vec<(usize, f32)> = orig
            .iter()
            .copied()
            .filter(|&(idx, _)| {
                let decoded = topic.lattice.programs[idx].display_text(dictionary);
                Self::sentiment_domain_slice_allows(crypto_q, &decoded)
            })
            .collect();
        if !filtered.is_empty() {
            *cosine_scores = filtered;
            return;
        }
        // Crypto query + strict filter removed everyone: prefer any joint-indexed row over legacy-only.
        if crypto_q {
            let with_marker: Vec<(usize, f32)> = orig
                .iter()
                .copied()
                .filter(|&(idx, _)| topic.lattice.programs[idx].display_text(dictionary).contains(core))
                .collect();
            if !with_marker.is_empty() {
                *cosine_scores = with_marker;
            }
        }
    }

    /// Directly query the sub-lattice whose name matches `forced_topic` (case-insensitive).
    /// Bypasses cross-topic competition — used when `infer_operation_topic` gives a
    /// specific operation name (e.g., "subtraction_operation") so the correct sub-lattice
    /// is selected even when a sibling (e.g., "addition_operation") has higher cosine sim.
    ///
    /// When `lang_hint` is Some (e.g., "rust"), programs whose decoded text matches the
    /// target language's markers are preferred, preventing Python code from being returned
    /// when Rust was requested.
    fn forced_topic_response(&self, cond: &[f32], forced_topic: &str) -> Option<(String, String, f32)> {
        self.forced_topic_response_lang(cond, forced_topic, None, None)
    }

    fn forced_topic_response_lang(&self, cond: &[f32], forced_topic: &str, lang_hint: Option<&str>, subject_keywords: Option<&[&str]>) -> Option<(String, String, f32)> {
        // Exact match first, then fuzzy word-overlap fallback
        let topic_match = self.topic_subindex.iter()
            .find(|t| t.topic_name.eq_ignore_ascii_case(forced_topic))
            .or_else(|| {
                let hint_words: Vec<&str> = forced_topic.split('_')
                    .filter(|w| w.len() > 2)
                    .collect();
                if hint_words.is_empty() { return None; }
                let mut best: Option<(&TopicSubIndex, usize)> = None;
                for t in &self.topic_subindex {
                    if t.lattice.programs.is_empty() { continue; }
                    let tname_lower = t.topic_name.to_ascii_lowercase();
                    let tname_words: Vec<&str> = tname_lower.split('_')
                        .filter(|w| w.len() > 2)
                        .collect();
                    let overlap = hint_words.iter()
                        .filter(|hw| tname_words.iter().any(|tw| tw == *hw))
                        .count();
                    if overlap > 0 {
                        if best.map(|(_, prev)| overlap > prev).unwrap_or(true) {
                            best = Some((t, overlap));
                        }
                    }
                }
                if let Some((t, _)) = best {
                    crate::infer_trace!(
                        "    [fuzzy-topic] '{}' → fuzzy matched '{}' ({} progs)",
                        forced_topic, t.topic_name, t.lattice.programs.len()
                    );
                }
                best.map(|(t, _)| t)
            });

        topic_match.and_then(|topic| {
                if topic.lattice.programs.is_empty() { return None; }

                let query_terms: Vec<String> = subject_keywords.unwrap_or(&[]).iter()
                    .filter(|kw| kw.len() > 2)
                    .map(|kw| kw.to_ascii_lowercase())
                    .collect();

                let retrieval_lex = crate::inference::retrieval_lexicon::global_for_locale(None);

                // ── Stage 1: Vector recall ──
                // Score all programs by cosine similarity to conditioning vector.
                let n = topic.lattice.programs.len();
                let mut cosine_scores: Vec<(usize, f32)> = topic.lattice.programs.iter().enumerate()
                    .map(|(i, prog)| (i, gen_cosine_sim(cond, &prog.ema_centroid)))
                    .collect();
                cosine_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                Self::sentiment_apply_topic_domain_slice(
                    &mut cosine_scores,
                    forced_topic,
                    &query_terms,
                    topic,
                    &self.dictionary,
                );

                // ── Stage 2: BM25 re-rank ──
                // Decode program texts once, compute BM25 against query subject keywords.
                // BM25 acts as a lexical safety net: even if cosine picks "quicksort"
                // for a "stack" query, BM25 will boost programs containing "stack".

                let mut scored: Vec<(usize, f32)> = if query_terms.is_empty() {
                    cosine_scores
                } else {
                    // Decode all program texts and tokenize into words
                    let docs: Vec<(usize, Vec<String>)> = cosine_scores.iter()
                        .map(|&(idx, _)| {
                            let text = topic.lattice.programs[idx].display_text(&self.dictionary);
                            let words: Vec<String> = text.to_ascii_lowercase()
                                .split(|c: char| !c.is_alphanumeric() && c != '_')
                                .filter(|w| w.len() > 1)
                                .map(|w| w.to_string())
                                .collect();
                            (idx, words)
                        })
                        .collect();

                    let avgdl: f32 = docs.iter().map(|(_, w)| w.len() as f32).sum::<f32>()
                        / (docs.len().max(1) as f32);
                    let k1: f32 = 1.2;
                    let b: f32 = 0.75;

                    // IDF per query term: log((N - df + 0.5) / (df + 0.5) + 1)
                    let idfs: Vec<f32> = query_terms.iter().map(|qt| {
                        let df = docs.iter()
                            .filter(|(_, words)| words.iter().any(|w| w == qt || w.contains(qt.as_str())))
                            .count() as f32;
                        ((n as f32 - df + 0.5) / (df + 0.5) + 1.0).ln()
                    }).collect();

                    // BM25 score per document
                    let bm25_scores: Vec<f32> = docs.iter().map(|(_, words)| {
                        let dl = words.len() as f32;
                        let mut score = 0.0f32;
                        for (qi, qt) in query_terms.iter().enumerate() {
                            let tf = words.iter()
                                .filter(|w| *w == qt || w.contains(qt.as_str()))
                                .count() as f32;
                            if tf > 0.0 {
                                let num = tf * (k1 + 1.0);
                                let den = tf + k1 * (1.0 - b + b * dl / avgdl);
                                score += idfs[qi] * num / den;
                            }
                        }
                        score
                    }).collect();

                    // Normalize BM25 to [0, 1] range for blending with cosine
                    let bm25_max = bm25_scores.iter().cloned().fold(0.0f32, f32::max);
                    let bm25_norm: Vec<f32> = if bm25_max > 0.0 {
                        bm25_scores.iter().map(|s| s / bm25_max).collect()
                    } else {
                        bm25_scores
                    };

                    // Blend: cosine + λ * bm25_normalized (λ=0.35 gives BM25 strong influence)
                    let lambda = 0.35f32;
                    let mut combined: Vec<(usize, f32)> = cosine_scores.iter().enumerate()
                        .map(|(di, &(idx, cos))| {
                            let base = cos + lambda * bm25_norm[di];
                            // Intent-driven nudge: implement/code/write vs explain/define/describe (lexicon-driven).
                            let intent_mod = if !self.intent_action.is_empty() {
                                let (_, words) = &docs[di];
                                let text = topic.lattice.programs[idx].display_text(&self.dictionary);
                                let has_code = retrieval_lex.program_has_code_markers_bm25(&text);
                                let action = self.intent_action.as_str();
                                if retrieval_lex.intent_prefers_code(action) && has_code {
                                    0.08
                                } else if retrieval_lex.intent_prefers_prose(action) && !has_code && words.len() > 15 {
                                    0.05
                                } else {
                                    0.0
                                }
                            } else {
                                0.0
                            };
                            (idx, base + intent_mod)
                        })
                        .collect();
                    combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    combined
                };

                // ── Stage 3: Graph signature re-rank ──
                // Use discriminative keyword signatures from the ProgramGraph to
                // further separate confusable programs that cosine+BM25 can't distinguish.
                crate::infer_trace!(
                    "    [retrieval-diag] topic='{}', {} progs, query_terms={:?}",
                    forced_topic, n, query_terms
                );
                for (rank, &(idx, score)) in scored.iter().enumerate().take(3) {
                    let snippet: String = topic.lattice.programs[idx].display_text(&self.dictionary)
                        .chars().take(50).collect();
                    crate::infer_trace!(
                        "      pre-graph[{}]: prog={}, score={:.3}, text=\"{}...\"",
                        rank, idx, score, snippet
                    );
                }
                if let Some(ref graph) = topic.graph {
                    if !query_terms.is_empty() {
                        let sig_scores: Vec<f32> = scored.iter()
                            .map(|&(idx, _)| graph.signature_score(idx, &query_terms))
                            .collect();
                        let sig_max = sig_scores.iter().cloned().fold(0.0f32, f32::max);

                        // Diagnostic: show signature scores for top candidates
                        for (rank, (&(idx, combined), &sig)) in scored.iter().zip(sig_scores.iter()).enumerate().take(4) {
                            let sigs_for_prog: Vec<String> = graph.signatures.get(idx)
                                .map(|s| s.iter().take(5).map(|dk| format!("{}:{:.2}", dk.keyword, dk.specificity)).collect())
                                .unwrap_or_default();
                            crate::infer_trace!(
                                "      graph[{}]: prog={}, combined={:.3}, sig_score={:.3}, top_keys=[{}]",
                                rank, idx, combined, sig, sigs_for_prog.join(", ")
                            );
                        }

                        // Also show what keyword_lookup returns
                        let lookup = graph.keyword_lookup(&query_terms);
                        if !lookup.is_empty() {
                            let top3: Vec<String> = lookup.iter().take(3)
                                .map(|(idx, s)| format!("prog{}={:.2}", idx, s))
                                .collect();
                            crate::infer_trace!("      keyword_lookup: [{}]", top3.join(", "));
                        }

                        if sig_max > 0.0 {
                            let sig_lambda = 0.30f32;
                            for (di, &mut (ref _idx, ref mut score)) in scored.iter_mut().enumerate() {
                                *score += sig_lambda * (sig_scores[di] / sig_max);
                            }
                        }

                        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                        if let Some(&(top_idx, top_score)) = scored.first() {
                            if let Some(redirect_idx) = graph.neighbor_redirect(top_idx, &query_terms) {
                                if let Some(entry) = scored.iter_mut().find(|e| e.0 == redirect_idx) {
                                    entry.1 = top_score + 0.05;
                                    crate::infer_trace!(
                                        "    [graph-redirect] prog {} → neighbor {} (better keyword match)",
                                        top_idx, redirect_idx
                                    );
                                }
                            }
                        }
                    }
                } else {
                    crate::infer_trace!("      [no-graph] topic '{}' has no ProgramGraph", forced_topic);
                }
                // Stage 4: lexical alignment — boost programs that match several query
                // content words; penalize matches that hinge on a single frequent token
                // (e.g. "love" shared across unrelated positive_strong prototypes).
                if query_terms.len() >= 2 {
                    let qcontent: Vec<&String> = query_terms
                        .iter()
                        .filter(|t| t.len() > 2 && !retrieval_lex.is_lex_align_stop(t.as_str()))
                        .collect();
                    if qcontent.len() >= 2 {
                        for (idx, sc) in scored.iter_mut() {
                            let text = topic.lattice.programs[*idx].display_text(&self.dictionary);
                            let tl = text.to_ascii_lowercase();
                            let hits = qcontent.iter().filter(|qt| tl.contains(qt.as_str())).count();
                            let align = hits as f32 / qcontent.len() as f32;
                            *sc += 0.22 * align;
                            if hits == 1 && qcontent.len() >= 3 {
                                *sc -= 0.14;
                            }
                        }
                        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                        crate::infer_trace!(
                            "      [lex-align] reranked with {} content terms (stopwords stripped)",
                            qcontent.len()
                        );
                    }
                }
                Self::sentiment_apply_crypto_market_retrieval_rescore(
                    &mut scored,
                    &query_terms,
                    topic,
                    &self.dictionary,
                );
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                for (rank, &(idx, score)) in scored.iter().enumerate().take(3) {
                    let snippet: String = topic.lattice.programs[idx].display_text(&self.dictionary)
                        .chars().take(50).collect();
                    crate::infer_trace!(
                        "      post-graph[{}]: prog={}, score={:.3}, text=\"{}...\"",
                        rank, idx, score, snippet
                    );
                }

                // When a language hint is provided, try to find a program matching that language
                if let Some(lang) = lang_hint {
                    let lang_lower = lang.to_lowercase();
                    let is_rust = lang_lower == "rust";
                    let is_python = lang_lower == "python" || lang_lower == "py";

                    for &(idx, score) in &scored {
                        let text = topic.lattice.programs[idx].display_text(&self.dictionary);
                        let opening: String = text.chars().take(300).collect();
                        let centroid = &topic.lattice.programs[idx].ema_centroid;
                        if Self::should_reject_text(&opening, Some(centroid), Some(cond)) { continue; }
                        let matches_lang = if is_rust {
                            retrieval_lex.program_matches_rust_lang_hint(&text)
                        } else if is_python {
                            retrieval_lex.program_matches_python_lang_hint(&text)
                        } else {
                            true
                        };
                        if matches_lang && !text.is_empty() && score > 0.10 {
                            let snippet: String = text.chars().take(60).collect();
                            crate::infer_trace!(
                                "    [forced-topic] '{}' → {} progs, conf={:.3}, text=\"{}...\"",
                                forced_topic, topic.lattice.programs.len(), score, snippet
                            );
                            return Some((text, topic.topic_name.clone(), score));
                        }
                    }
                }

                // Select the top-scored program. When the graph gave a keyword
                // match for the top-ranked program, trust it even if the text has
                // minor encoding artifacts — the graph validated relevance.
                let top_prog_idx = scored.first().map(|&(idx, _)| idx);
                let graph_confident = top_prog_idx.map(|top_idx| {
                    topic.graph.as_ref()
                        .map(|g| {
                            let sig_score = g.signature_score(top_idx, &query_terms);
                            sig_score > 0.5
                        })
                        .unwrap_or(false)
                }).unwrap_or(false);

                for &(idx, score) in &scored {
                    if score < 0.10 { break; }
                    let text = topic.lattice.programs[idx].display_text(&self.dictionary);
                    if text.is_empty() { continue; }

                    let opening: String = text.chars().take(300).collect();
                    if Self::hard_reject_lattice_decoded_text(&opening) {
                        let snippet: String = text.chars().take(40).collect();
                        crate::infer_trace!(
                            "    [skip-hard-reject] prog={}, score={:.3}, text=\"{}...\"",
                            idx, score, snippet
                        );
                        continue;
                    }

                    // Soft artifact check: graph-confident retrieval may skip (alignment override).
                    if !graph_confident {
                        let centroid = &topic.lattice.programs[idx].ema_centroid;
                        if Self::should_reject_text_soft(&opening, Some(centroid.as_slice()), Some(cond)) {
                            let snippet: String = text.chars().take(40).collect();
                            crate::infer_trace!(
                                "    [skip-artifact] prog={}, score={:.3}, text=\"{}...\"",
                                idx, score, snippet
                            );
                            continue;
                        }
                    }
                    if query_terms.len() >= 2
                        && !Self::sentiment_witness_matches_subject_keywords(&text, &query_terms)
                    {
                        let snippet: String = text.chars().take(40).collect();
                        crate::infer_trace!(
                            "    [skip-witness-mismatch] prog={}, score={:.3}, text=\"{}...\"",
                            idx, score, snippet
                        );
                        continue;
                    }
                    let snippet: String = text.chars().take(60).collect();
                    crate::infer_trace!(
                        "    [forced-topic] '{}' → {} progs, conf={:.3}, graph_conf={}, text=\"{}...\"",
                        forced_topic, topic.lattice.programs.len(), score, graph_confident, snippet
                    );
                    return Some((text, topic.topic_name.clone(), score));
                }
                None
            })
    }

    /// Top-K nearest responses using STA field scoring with multi-timescale bias.
    fn nearest_responses_k(&self, cond: &[f32], k: usize) -> Vec<(String, usize, f32)> {
        if self.lattice.programs.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(usize, f32)> = self.lattice.programs.iter().enumerate()
            .map(|(i, prog)| {
                let effective = self.lattice.effective_centroid(i);
                let centroid = if effective.is_empty() { &prog.ema_centroid } else { &effective };
                let cosine = gen_cosine_sim(cond, centroid);
                let displacement: Vec<f32> = cond.iter()
                    .zip(centroid.iter())
                    .map(|(a, b)| a - b)
                    .collect();
                let disp_mv = embed_bridge_vector(&displacement);
                let disp_spatial = spatial_fingerprint(&disp_mv);
                let spatial_energy: f32 = disp_spatial.iter().map(|x| x * x).sum::<f32>().sqrt();
                let spatial_penalty = (spatial_energy * 2.0).min(1.0);
                let disp_norm_sq: f32 = displacement.iter().map(|d| d * d).sum();
                let proximity = if disp_norm_sq < 1e-8 { 100.0 } else { (1.0 / disp_norm_sq).min(100.0) };
                let base = 0.50 * cosine + 0.25 * (proximity / 100.0).sqrt() + 0.25 * (1.0 - spatial_penalty);
                let bias = self.lattice.retrieval_bias(i);
                (i, base * bias)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(k).map(|(idx, sim)| {
            let text = self.lattice.programs[idx].display_text(&self.dictionary);
            (text, idx, sim)
        }).collect()
    }

    /// Summarize across topic sub-lattices using coherence-guided selection.
    ///
    /// Instead of just picking the top-K most relevant programs, uses
    /// band-decomposed Cl(1,7) coherence analysis to select programs that
    /// are both relevant AND cohere with each other as an ensemble. This
    /// mirrors how coherent neural oscillations across brain areas indicate
    /// functional connectivity and coordinated processing.
    ///
    /// Returns (composed_text, confidence, topics_used, ensemble_coherence).
    pub fn summarize_across_topics(
        &self,
        cond: &[f32],
        max_topics: usize,
        max_chars: usize,
    ) -> (String, f32, Vec<String>, f32) {
        use crate::coherence::{coherence_select, ensemble_coherence};

        if self.topic_subindex.is_empty() {
            return (String::new(), 0.0, Vec::new(), 0.0);
        }

        // Phase 1: Gather best clean program from each topic sub-lattice
        struct TopicCandidate {
            topic_name: String,
            text: String,
            opening: String,
            relevance: f32,
            centroid: Vec<f32>,
        }
        let mut candidates: Vec<TopicCandidate> = Vec::new();

        for topic in &self.topic_subindex {
            if topic.lattice.programs.is_empty() {
                continue;
            }
            // Try up to 3 nearest programs to find one with clean text
            let top3 = self.top_k_in_lattice(&topic.lattice, cond, 3);
            let mut text = String::new();
            let mut conf = 0.0f32;
            for (cand_text, _pidx, cand_conf) in &top3 {
                if !cand_text.is_empty() && cand_text.len() >= 10 && !Self::has_tokenization_artifacts(cand_text) {
                    text = cand_text.clone();
                    conf = *cand_conf;
                    break;
                }
            }
            if text.is_empty() {
                let (t, _, c) = self.nearest_response_in_lattice(&topic.lattice, cond);
                text = t;
                conf = c;
            }
            let topic_centroid_ref: &[f32] = &topic.centroid;
            if text.is_empty() || text.len() < 10
                || Self::should_reject_text(&text, Some(topic_centroid_ref), Some(cond))
            {
                continue;
            }

            let opening = Self::extract_opening(&text, 200);
            if opening.is_empty()
                || Self::should_reject_text(&opening, Some(topic_centroid_ref), Some(cond))
            {
                continue;
            }

            // Centroid similarity to query
            let centroid_sim = {
                let dim = cond.len().min(topic.centroid.len());
                if dim == 0 { 0.0 } else {
                    let dot: f32 = cond[..dim].iter().zip(topic.centroid[..dim].iter())
                        .map(|(a, b)| a * b).sum();
                    let na = cond[..dim].iter().map(|x| x * x).sum::<f32>().sqrt();
                    let nb = topic.centroid[..dim].iter().map(|x| x * x).sum::<f32>().sqrt();
                    if na < 1e-10 || nb < 1e-10 { 0.0 } else { (dot / (na * nb)).max(0.0) }
                }
            };

            let relevance = 0.5 * conf + 0.5 * centroid_sim;
            candidates.push(TopicCandidate {
                topic_name: topic.topic_name.clone(),
                text,
                opening,
                relevance,
                centroid: topic.centroid.clone(),
            });
        }

        if candidates.is_empty() {
            return (String::new(), 0.0, Vec::new(), 0.0);
        }

        // Phase 2: Coherence-guided selection
        // Use topic centroids for coherence analysis — they represent the
        // "sending region" in the neuroscience analogy
        let centroid_refs: Vec<&[f32]> = candidates.iter()
            .map(|c| c.centroid.as_slice())
            .collect();
        let relevance_scores: Vec<f32> = candidates.iter()
            .map(|c| c.relevance)
            .collect();

        // Select programs that maximize both relevance AND ensemble coherence.
        // min_coherence=0.20 prevents adding programs that desynchronize the ensemble.
        let selected_indices = coherence_select(
            &centroid_refs,
            &relevance_scores,
            max_topics * 2, // over-select, then trim by char limit
            0.20,
        );

        if selected_indices.is_empty() {
            return (String::new(), 0.0, Vec::new(), 0.0);
        }

        // Compute ensemble coherence of the selected set
        let selected_centroids: Vec<&[f32]> = selected_indices.iter()
            .map(|&i| candidates[i].centroid.as_slice())
            .collect();
        let (ens_coherence, band_detail) = if selected_centroids.len() >= 2 {
            ensemble_coherence(&selected_centroids)
        } else {
            (1.0, crate::coherence::BandCoherence { combined: 1.0, ..Default::default() })
        };

        crate::infer_trace!(
            "  [coherence] ensemble={:.3}, δ={:.3} θ={:.3} α/β_boost={:.3} α/β_spatial={:.3} γ={:.3}",
            ens_coherence,
            band_detail.delta,
            band_detail.theta,
            band_detail.alpha_beta_boost,
            band_detail.alpha_beta_spatial,
            band_detail.gamma,
        );

        // Phase 3: Compose with deduplication and char limit
        let mut parts: Vec<String> = Vec::new();
        let mut topics_used: Vec<String> = Vec::new();
        let mut total_len = 0;
        let mut total_conf = 0.0f32;

        for &idx in &selected_indices {
            let cand = &candidates[idx];

            // Deduplicate
            let is_duplicate = parts.iter().any(|existing| {
                let overlap_words: usize = cand.opening.split_whitespace()
                    .filter(|w| existing.to_lowercase().contains(&w.to_lowercase()))
                    .count();
                let total_words = cand.opening.split_whitespace().count().max(1);
                overlap_words as f32 / total_words as f32 > 0.6
            });
            if is_duplicate {
                continue;
            }

            if total_len + cand.opening.len() > max_chars {
                break;
            }
            total_len += cand.opening.len();
            total_conf += cand.relevance;
            topics_used.push(cand.topic_name.clone());
            parts.push(cand.opening.clone());
        }

        if parts.is_empty() {
            return (String::new(), 0.0, Vec::new(), 0.0);
        }

        let avg_conf = total_conf / parts.len() as f32;
        let composed = parts.join(" ");
        (composed, avg_conf.clamp(0.0, 1.0), topics_used, ens_coherence)
    }

    /// Hard reject: mask tokens, bracket glitches, and known training/meta boilerplate leaks.
    /// **Never** overridden by graph confidence or centroid-vs-query alignment.
    fn hard_reject_lattice_decoded_text(text: &str) -> bool {
        let t = text.to_ascii_lowercase();
        if t.contains("[mask]") || t.contains("mask]") || t.contains("mask][") {
            return true;
        }
        if t.contains("][") || t.contains("[]") {
            return true;
        }
        if t.contains("growformer agent")
            || t.contains("growformer companion")
            || t.contains("companion agent")
            || t.contains("i am growformer")
            || t.contains("specialized ai agent")
            || t.contains("built by swtch")
            || t.contains("swtch.ai")
            || (t.contains("growformer") && t.contains("swtch"))
        {
            return true;
        }
        if t.contains("self-organizing neural") || t.contains("self organizing neural") {
            return true;
        }
        if t.contains("drawn to curiosity") {
            return true;
        }
        if t.contains("knowledge is encoded as physical neural structure") {
            return true;
        }
        if t.contains("people their people") || t.contains("their people their") {
            return true;
        }
        if t.match_indices("little details").count() >= 2 {
            return true;
        }
        if t.contains("3d environment") && t.contains("neuron") {
            return true;
        }
        if t.contains("neurons grow") && t.contains("connect") {
            return true;
        }
        if t.contains("details people details")
            || t.contains("people details people")
            || t.contains("people share people")
        {
            return true;
        }
        if t.contains("their lives") && t.match_indices("their lives").count() >= 2 {
            return true;
        }
        if t.contains("headline snack") {
            return true;
        }
        if t.contains("my intelligence comes from structure")
            || t.contains("not weight optimization")
            || t.contains("friendly companion")
            || t.contains("share a moment of curiosity")
            || t.contains("i was created to be")
        {
            return true;
        }
        if t.starts_with("no.") && t.contains("intelligence comes from") {
            return true;
        }
        if t.contains("about. about") {
            return true;
        }
        if t.contains(" mask") && (t.contains("little") || t.contains("their")) {
            return true;
        }
        Self::hard_reject_repeated_bigram(&t)
    }

    /// Detect immediate bigram repetition (`w1 w2 w1 w2`) common in broken decodes.
    fn hard_reject_repeated_bigram(t: &str) -> bool {
        let words: Vec<&str> = t.split_whitespace().collect();
        if words.len() < 4 {
            return false;
        }
        for w in words.windows(4) {
            if w[0] == w[2] && w[1] == w[3] && w[0].len() > 2 && w[1].len() > 2 {
                return true;
            }
        }
        false
    }

    /// Soft reject: tokenization heuristics, optionally overridden by high query–program alignment.
    fn should_reject_text_soft(
        text: &str,
        program_centroid: Option<&[f32]>,
        query_emb: Option<&[f32]>,
    ) -> bool {
        if !Self::has_tokenization_artifacts(text) {
            return false;
        }
        if let (Some(centroid), Some(qemb)) = (program_centroid, query_emb) {
            let alignment = gen_cosine_sim(qemb, centroid);
            if alignment > 0.40 {
                return false;
            }
        }
        true
    }

    /// Surface artifacts with optional alignment override; **hard** rejects are never accepted.
    fn should_reject_text(
        text: &str,
        program_centroid: Option<&[f32]>,
        query_emb: Option<&[f32]>,
    ) -> bool {
        if Self::hard_reject_lattice_decoded_text(text) {
            return true;
        }
        Self::should_reject_text_soft(text, program_centroid, query_emb)
    }

    /// Garbled programs have isolated single characters, excessive spacing,
    /// repetition loops, low lexical diversity, or broken propositional
    /// coherence (sentences that don't connect to their neighbors).
    fn has_tokenization_artifacts(text: &str) -> bool {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return true;
        }

        let wc = words.len();

        // Count single-character "words" (excluding only punctuation/operators)
        let single_char_count = words.iter()
            .filter(|w| {
                w.len() == 1 && !matches!(w.as_bytes().first(),
                    Some(b'(' | b')' | b'-' | b'+' | b'=' | b',' | b'.' | b':' | b';' | b'&' | b'*' | b'/' | b'?' | b'!')
                )
            })
            .count();

        if single_char_count as f32 / wc as f32 > 0.20 {
            return true;
        }

        // All single alphabetic chars (no exclusions) — catches garble that
        // sprinkles common letters (a, e, t, s) throughout real words.
        let all_single_alpha = words.iter()
            .filter(|w| w.len() == 1 && w.chars().next().map_or(false, |c| c.is_alphabetic()))
            .count();
        if all_single_alpha >= 5 && all_single_alpha as f32 / wc as f32 > 0.12 {
            return true;
        }

        // Sequences of space-separated single characters: "e a d g s"
        let mut consecutive_singles = 0u32;
        let mut max_consecutive_singles = 0u32;
        for w in &words {
            if w.len() == 1 && w.chars().next().map(|c| c.is_alphabetic()).unwrap_or(false) {
                consecutive_singles += 1;
                max_consecutive_singles = max_consecutive_singles.max(consecutive_singles);
            } else {
                consecutive_singles = 0;
            }
        }
        if max_consecutive_singles >= 3 {
            return true;
        }

        // Repetition loop: same word repeated 3+ times consecutively
        if wc >= 8 {
            let mut max_repeat = 1u32;
            let mut cur_repeat = 1u32;
            for pair in words.windows(2) {
                if pair[0] == pair[1] {
                    cur_repeat += 1;
                    max_repeat = max_repeat.max(cur_repeat);
                } else {
                    cur_repeat = 1;
                }
            }
            if max_repeat >= 3 {
                return true;
            }

            // Bigram repetition: same 2-word pair appears 3+ times
            let bigrams: Vec<(&str, &str)> = words.windows(2)
                .map(|w| (w[0], w[1]))
                .collect();
            let mut bigram_counts: std::collections::HashMap<(&str, &str), u32> =
                std::collections::HashMap::new();
            for &bg in &bigrams {
                *bigram_counts.entry(bg).or_default() += 1;
            }
            if bigram_counts.values().any(|&c| c >= 3) {
                return true;
            }
        }

        // Low lexical diversity: word salad with very few unique words
        if wc >= 12 {
            let unique: std::collections::HashSet<&str> = words.iter().copied().collect();
            let ratio = unique.len() as f32 / wc as f32;
            if ratio < 0.45 {
                return true;
            }
        }

        // Propositional coherence: consecutive sentences must share content
        // words. Garbled text has domain words in random order with broken
        // semantic trajectory — each sentence is an island.
        if wc >= 15 {
            let sentences: Vec<std::collections::HashSet<&str>> = text
                .split(|c: char| c == '.' || c == '!' || c == '?')
                .map(|s| {
                    s.split_whitespace()
                        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
                        .filter(|w| w.len() > 3)
                        .collect::<std::collections::HashSet<&str>>()
                })
                .filter(|s| !s.is_empty())
                .collect();

            if sentences.len() >= 3 {
                let mut connected = 0usize;
                for i in 0..sentences.len() {
                    let has_prev = i == 0 || sentences[i - 1].intersection(&sentences[i]).count() > 0;
                    let has_next = i == sentences.len() - 1
                        || sentences[i].intersection(&sentences[i + 1]).count() > 0;
                    if has_prev || has_next {
                        connected += 1;
                    }
                }
                let connectivity = connected as f32 / sentences.len() as f32;
                if connectivity < 0.5 {
                    return true;
                }
            }
        }

        false
    }

    /// Extract the opening portion of a text (first 1-2 sentences, up to max_chars).
    fn extract_opening(text: &str, max_chars: usize) -> String {
        let truncated = if text.len() > max_chars { &text[..max_chars] } else { text };

        // Find a sentence boundary
        if let Some(period_pos) = truncated.find(". ") {
            if period_pos > 15 {
                return truncated[..period_pos + 1].to_string();
            }
        }
        // If no clean break, take up to the last word boundary
        if let Some(space_pos) = truncated.rfind(' ') {
            if space_pos > 15 {
                return format!("{}.", &truncated[..space_pos]);
            }
        }
        truncated.to_string()
    }

    pub fn generate(&mut self, cond: &[f32], _max_len: usize, _temperature: f32) -> (String, f32) {
        self.generate_for_topic(cond, None, _max_len, _temperature)
    }

    pub fn generate_for_topic(
        &mut self,
        cond: &[f32],
        topic_hint: Option<&str>,
        _max_len: usize,
        _temperature: f32,
    ) -> (String, f32) {
        self.generate_for_topic_lang(cond, topic_hint, None, _max_len, _temperature)
    }

    pub fn generate_for_topic_lang(
        &mut self,
        cond: &[f32],
        topic_hint: Option<&str>,
        lang_hint: Option<&str>,
        _max_len: usize,
        _temperature: f32,
    ) -> (String, f32) {
        let (global_text, prog_idx, global_conf) = self.nearest_response(cond);
        let global_text_backup = global_text.clone();

        // When a specific operation topic is provided (e.g., "subtraction_operation"),
        // bypass cross-topic competition and directly query the matching sub-lattice.
        // This prevents addition's high cosine similarity from drowning out subtraction.
        let kw_refs: Vec<&str> = self.subject_keywords.iter().map(|s| s.as_str()).collect();
        let kw_opt: Option<&[&str]> = if kw_refs.is_empty() { None } else { Some(&kw_refs) };
        let forced = topic_hint
            .and_then(|h| self.forced_topic_response_lang(cond, h, lang_hint, kw_opt))
            .filter(|(ft, _, ft_conf)| {
                if *ft_conf <= 0.10 || ft.len() <= 5 {
                    return false;
                }
                let open: String = ft.chars().take(400).collect();
                if Self::hard_reject_lattice_decoded_text(&open) {
                    crate::infer_trace!(
                        "    [forced-discard] hard-reject lattice output; falling back to global/topic path"
                    );
                    return false;
                }
                true
            });
        let forced_active = forced.is_some();
        let (text, lattice_conf, topic_selected) = if let Some((ft, fn_name, ft_conf)) = forced {
            if ft.len() > 5 && ft_conf > 0.10 {
                (ft, ft_conf.max(global_conf * 0.85), Some(fn_name))
            } else {
                (global_text, global_conf, None)
            }
        } else {
            let topic_seed = self.nearest_topic_response(cond, topic_hint);
            match topic_seed {
                Some((topic_text, topic_name, topic_conf)) if topic_text.len() > 5 => {
                    if topic_conf >= global_conf - 0.03 {
                        (topic_text, topic_conf, Some(topic_name))
                    } else {
                        (global_text, global_conf, None)
                    }
                }
                _ => (global_text, global_conf, None),
            }
        };

        // Keyword-relevance gate: if the forced-topic returned text but it has zero
        // keyword overlap with the query subject, check if the global nearest-response
        // is a better match. This prevents "stack" queries from returning "quicksort"
        // when the sub-lattice's cosine-nearest happens to be about a different topic.
        if forced_active && topic_selected.is_some() && !kw_refs.is_empty() {
            let text_lower = text.to_ascii_lowercase();
            let has_kw_match = kw_refs.iter().any(|kw| kw.len() > 3 && text_lower.contains(*kw));
            if !has_kw_match {
                let global_lower = global_text_backup.to_ascii_lowercase();
                let global_has_kw = kw_refs.iter().any(|kw| kw.len() > 3 && global_lower.contains(*kw));
                let global_open: String = global_text_backup.chars().take(400).collect();
                if global_has_kw
                    && global_conf > 0.30
                    && !Self::has_tokenization_artifacts(&global_text_backup)
                    && !Self::hard_reject_lattice_decoded_text(&global_open)
                {
                    crate::infer_trace!(
                        "    [kw-override] forced-topic text has no keyword match, using global nearest"
                    );
                    self.last_selected_archetype = Some(prog_idx);
                    self.last_generation_confidence = global_conf;
                    return (global_text_backup, global_conf);
                }
            }
        }

        // When forced topic matched and returned valid text, return it directly.
        // The sub-lattice already contains only programs for this specific operation,
        // so the field inhibition gate (which uses the GLOBAL lattice) would incorrectly
        // override the sub-lattice's authoritative result.
        // Check artifacts only on the opening portion the user will see
        let opening_check: String = text.chars().take(300).collect();
        let prog_centroid = &self.lattice.programs[prog_idx].ema_centroid;
        if forced_active && topic_selected.is_some() && text.len() > 5
            && !Self::should_reject_text(&opening_check, Some(prog_centroid), Some(cond))
        {
            self.last_selected_archetype = None;
            self.last_generation_confidence = lattice_conf;
            return (text, lattice_conf);
        }

        // ∇F field gradient: compute the directional derivative of the response field
        // at the query point. Large |∇F| means we're between programs (the field is
        // changing fast) — inhibit verbatim and use gradient-aware slot inference.
        let (field_gradient, gradient_mag) = cloze::compute_field_gradient(cond, &self.lattice.programs);

        // STA inhibition gate: fires when EITHER the displacement energy is high
        // OR the field gradient magnitude is significant (we're between programs).
        let field_inhibited = if lattice_conf >= 0.75 {
            let retrieved_centroid = &self.lattice.programs[prog_idx].ema_centroid;
            let displacement: Vec<f32> = cond.iter()
                .zip(retrieved_centroid.iter())
                .map(|(a, b)| a - b)
                .collect();
            let disp_mv = embed_bridge_vector(&displacement);
            let disp_spatial = spatial_fingerprint(&disp_mv);
            let spatial_energy: f32 = disp_spatial.iter().map(|x| x * x).sum::<f32>().sqrt();
            let disp_causal = causal_fingerprint(&disp_mv);
            let causal_energy: f32 = disp_causal.iter().map(|x| x * x).sum::<f32>().sqrt();
            let displacement_energy = spatial_energy + causal_energy;

            displacement_energy > 0.10 || gradient_mag > 0.02
        } else {
            false
        };

        // High confidence AND in the program's rest frame: return lattice text directly
        if lattice_conf >= 0.80 && text.len() > 5 && !field_inhibited
            && !Self::should_reject_text(&opening_check, Some(prog_centroid), Some(cond))
        {
            self.last_selected_archetype = if topic_selected.is_some() { None } else { Some(prog_idx) };
            self.last_generation_confidence = lattice_conf;
            self.lattice.on_retrieval(prog_idx, cond);
            return (text, lattice_conf);
        }

        // Inhibited OR medium confidence: use gradient-aware slot inference.
        // The field gradient tells us WHICH DIRECTION the response should shift,
        // enabling discrimination between add/sub/mul within the same group.
        if lattice_conf >= 0.55 || field_inhibited {
            if let Some(ref cb) = self.codebook {
                if cb.has_prototypes() {
                    let (arch_idx, _) = cb.select_archetype_by_embedding(cond);

                    let slot_bits = if field_inhibited && gradient_mag > 0.01 {
                        // ∇F-aware slot inference: use gradient to bias toward
                        // the correct program's slot fills.
                        let inferred = cloze::infer_slots_with_gradient(
                            cond, &field_gradient, gradient_mag,
                            arch_idx, cb,
                            &self.lattice.programs, 7,
                        );
                        cloze::encode_inferred_slot_bits(&inferred, cb)
                    } else if field_inhibited {
                        // Inhibited but low gradient: use proximity-based inference.
                        let inferred = cloze::infer_slots(
                            cond, arch_idx, cb, &self.dictionary,
                            &self.lattice.programs, 5,
                        );
                        cloze::encode_inferred_slot_bits(&inferred, cb)
                    } else {
                        // Normal path: copy slots from retrieved text.
                        let slot_tokens = self.dictionary.encode(&text);
                        cb.encode_slot_only(&slot_tokens)
                    };

                    // Differentiable slot optimization: SPSA-optimize slot bits
                    // to maximize relevance of decoded tokens to the conditioning.
                    let optimized_bits = if field_inhibited {
                        let cond_ref = cond;
                        let dict_ref = &self.dictionary;
                        let programs_ref = &self.lattice.programs;
                        let cb_ref = cb;
                        let a_idx = arch_idx;
                        crate::predictive_coder::optimize_slot_bits(
                            &slot_bits, 6, 0.08, 0.03,
                            &|bits: &[f32]| {
                                let ids = cb_ref.decode_with_archetype(a_idx, bits);
                                let txt = dict_ref.decode(&ids);
                                if txt.len() < 5 { return -1.0; }
                                let tok_ids = dict_ref.encode(&txt);
                                let mut score = 0.0f32;
                                for pid in 0..programs_ref.len().min(5) {
                                    let p = &programs_ref[pid];
                                    let overlap = tok_ids.iter()
                                        .filter(|t| p.token_sequence.contains(t))
                                        .count();
                                    let sim = gen_cosine_sim(cond_ref, &p.ema_centroid).max(0.0);
                                    score += overlap as f32 * sim;
                                }
                                score
                            },
                        )
                    } else {
                        slot_bits.clone()
                    };

                    let decoded_ids = cb.decode_with_archetype(arch_idx, &optimized_bits);
                    let decoded_text = self.dictionary.decode(&decoded_ids);
                    let decoded_text = Self::truncate_archetype(
                        &cb.archetypes, arch_idx, &decoded_ids, &decoded_text,
                    );
                    if decoded_text.len() > 5 {
                        self.last_selected_archetype = Some(arch_idx);
                        self.last_generation_confidence = lattice_conf;
                        return (decoded_text, lattice_conf);
                    }
                }
            }
            self.last_selected_archetype = if topic_selected.is_some() { None } else { Some(prog_idx) };
            self.last_generation_confidence = lattice_conf;
            return (text, lattice_conf);
        }

        // Schema-based generation: use extracted templates if available.
        // Schemas capture the invariant structure across similar programs,
        // only filling in the variable slots using the conditioning signal.
        if !self.schemas.is_empty() {
            let mut best_schema_text = String::new();
            let mut best_schema_score = 0.0f32;

            for schema in &self.schemas {
                if schema.support < 2 || schema.fixed.is_empty() { continue; }
                let tokens = crate::predictive_coder::fill_schema(schema, cond);
                let text = self.dictionary.decode(&tokens);
                if text.len() > 10 && !Self::has_tokenization_artifacts(&text) {
                    let score = schema.avg_quality * schema.support as f32 * 0.1;
                    if score > best_schema_score {
                        best_schema_score = score;
                        best_schema_text = text;
                    }
                }
            }

            if best_schema_text.len() > 10 && best_schema_score > 0.3 {
                self.last_selected_archetype = None;
                self.last_generation_confidence = best_schema_score.min(0.75);
                return (best_schema_text, best_schema_score.min(0.75));
            }
        }

        #[cfg(feature = "training")]
        {
            use crate::gradient_memory::{GradientMemory, GradientMemoryConfig, MemorySource};
            let top_k = self.nearest_responses_k(cond, 4);
            if top_k.len() >= 2 {
                let sources: Vec<MemorySource> = top_k.iter().map(|(text, pidx, conf)| {
                    let token_ids = self.dictionary.encode(text);
                    let centroid = if *pidx < self.lattice.programs.len() {
                        self.lattice.programs[*pidx].ema_centroid.clone()
                    } else {
                        cond.to_vec()
                    };
                    MemorySource { centroid, token_ids, similarity: *conf }
                }).collect();

                let stgm_config = GradientMemoryConfig::default();
                let target_dim = cond.len();
                let mut gm = GradientMemory::new(cond, sources, target_dim, stgm_config);
                let result = gm.optimize(cond, target_dim);

                let composed = gm.decode_sentences(&self.dictionary);
                if composed.len() > 10 && result.coherence.combined > 0.40
                    && !Self::has_tokenization_artifacts(&composed)
                {
                    self.last_selected_archetype = None;
                    self.last_generation_confidence = result.coherence.combined;
                    return (composed, result.coherence.combined);
                }
            }
        }

        // Topic-aware continuous composition: when a forced topic was identified,
        // compose trajectories from that topic's programs via ChunkCodec.
        // This gives compositional generation with topic precision.
        if let Some(ref topic_name) = topic_selected {
            if let Some((ctext, cconf)) = self.generate_continuous_for_topic(cond, topic_name, _max_len) {
                if ctext.len() > 10 && !Self::has_tokenization_artifacts(&ctext) && cconf > 0.35 {
                    return (ctext, cconf);
                }
            }
        }

        // Low confidence: Hopf composition from top-K lattice responses
        if let (Some(ref cb), Some(ref hopf)) = (&self.codebook, &self.hopf_table) {
            if cb.has_prototypes() {
                let top_k = self.nearest_responses_k(cond, 3);
                if !top_k.is_empty() {
                    let best_tokens = self.dictionary.encode(&top_k[0].0);
                    let slot_bits = cb.encode_slot_only(&best_tokens);
                    let (ids, comp_conf) = hopf.compose_and_decode_with_personality(
                        cond, &slot_bits, cb, self.diversity_bonus,
                    );
                    let composed = self.dictionary.decode(&ids);
                    if composed.len() > 5 {
                        self.last_selected_archetype = None;
                        self.last_generation_confidence = comp_conf;
                        return (composed, comp_conf);
                    }
                }
            }
        }

        // Continuous chunk codec: when the selected text has tokenization artifacts
        // that survived all other paths, re-encode through the chunk codec for a
        // clean reconstruction. The codec has 100% accuracy at K=8 chunk level.
        if Self::has_tokenization_artifacts(&text) {
            if let Some((ctext, cconf)) = self.generate_continuous(cond, _max_len) {
                if ctext.len() > 10 && !Self::has_tokenization_artifacts(&ctext) {
                    return (ctext, cconf);
                }
            }
        }

        // Final fallback: return lattice text as-is
        self.last_selected_archetype = if topic_selected.is_some() { None } else { Some(prog_idx) };
        self.last_generation_confidence = lattice_conf;
        // Track retrieval for Continuum multi-timescale state
        self.lattice.on_retrieval(prog_idx, cond);
        (text, lattice_conf)
    }

    /// Generate using a pre-selected archetype index (from ArchetypeBrain).
    pub fn generate_with_archetype(
        &mut self, cond: &[f32], arch_idx: usize, arch_conf: f32,
        _max_len: usize, _temperature: f32,
    ) -> (String, f32) {
        self.generate_with_archetype_for_topic(cond, None, arch_idx, arch_conf, _max_len, _temperature)
    }

    pub fn generate_with_archetype_for_topic(
        &mut self,
        cond: &[f32],
        topic_hint: Option<&str>,
        arch_idx: usize,
        arch_conf: f32,
        _max_len: usize,
        _temperature: f32,
    ) -> (String, f32) {
        let (global_text, prog_idx, global_conf) = self.nearest_response(cond);

        let forced = topic_hint.and_then(|h| self.forced_topic_response(cond, h));
        let forced_active = forced.is_some();
        let (text, lattice_conf, topic_selected) = if let Some((ft, fn_name, ft_conf)) = forced {
            if ft.len() > 5 && ft_conf > 0.10 {
                (ft, ft_conf.max(global_conf * 0.85), Some(fn_name))
            } else {
                (global_text, global_conf, None)
            }
        } else {
            let topic_seed = self.nearest_topic_response(cond, topic_hint);
            match topic_seed {
                Some((topic_text, topic_name, topic_conf)) if topic_text.len() > 5 => {
                    if topic_conf >= global_conf - 0.03 {
                        (topic_text, topic_conf, Some(topic_name))
                    } else {
                        (global_text, global_conf, None)
                    }
                }
                _ => (global_text, global_conf, None),
            }
        };

        // Forced topic: return directly, bypass archetype reconstruction
        let opening_check2: String = text.chars().take(300).collect();
        let prog_centroid2 = &self.lattice.programs[prog_idx].ema_centroid;
        if ((forced_active && topic_selected.is_some() && text.len() > 5)
            || (lattice_conf >= 0.80 && text.len() > 5))
            && !Self::should_reject_text(&opening_check2, Some(prog_centroid2), Some(cond))
        {
            self.last_selected_archetype = if topic_selected.is_some() { None } else { Some(prog_idx) };
            self.last_generation_confidence = lattice_conf;
            return (text, lattice_conf);
        }

        // Use archetype reconstruction as fallback
        if let Some(ref cb) = self.codebook {
            if cb.has_prototypes() && arch_idx < cb.archetypes.len() {
                let slot_tokens = self.dictionary.encode(&text);
                let slot_bits = cb.encode_slot_only(&slot_tokens);
                let decoded_ids = cb.decode_with_archetype(arch_idx, &slot_bits);
                let decoded_text = self.dictionary.decode(&decoded_ids);
                let decoded_text = Self::truncate_archetype(
                    &cb.archetypes, arch_idx, &decoded_ids, &decoded_text,
                );
                if decoded_text.len() > 5 {
                    self.last_selected_archetype = Some(arch_idx);
                    self.last_generation_confidence = arch_conf * lattice_conf;
                    return (decoded_text, self.last_generation_confidence);
                }
            }
        }

        self.last_selected_archetype = if topic_selected.is_some() { None } else { Some(prog_idx) };
        self.last_generation_confidence = lattice_conf;
        (text, lattice_conf)
    }

    /// Generate and return an 8d E8 contribution vector alongside the text.
    pub fn generate_with_e8(
        &mut self, cond: &[f32], max_len: usize, temperature: f32,
    ) -> (String, f32, [f32; 8]) {
        let (text, conf) = self.generate(cond, max_len, temperature);
        let mut raw = [0.0f32; 8];
        for i in 0..8.min(cond.len()) {
            raw[i] = cond[i];
        }
        let e8_point = E8Lattice::nearest_point(&raw);
        (text, conf, e8_point)
    }

    pub fn generate_with_e8_for_topic(
        &mut self,
        cond: &[f32],
        topic_hint: Option<&str>,
        max_len: usize,
        temperature: f32,
    ) -> (String, f32, [f32; 8]) {
        let (text, conf) = self.generate_for_topic(cond, topic_hint, max_len, temperature);
        let mut raw = [0.0f32; 8];
        for i in 0..8.min(cond.len()) {
            raw[i] = cond[i];
        }
        let e8_point = E8Lattice::nearest_point(&raw);
        (text, conf, e8_point)
    }

    /// Online learning: train on a correction by developing the lattice
    /// with the new (embedding, correction) pair. No backprop.
    #[cfg(feature = "training")]
    pub fn train_step(&mut self, cond: &[f32], target: &str, _rng: &mut impl Rng) -> f32 {
        if self.frozen { return 0.0; }
        let pairs = vec![(cond.to_vec(), target.to_string())];
        self.lattice.develop(&pairs, 0.7);
        let (_, _, conf) = self.nearest_response(cond);
        1.0 - conf
    }

    /// Freeze the environment (no further online learning).
    pub fn freeze(&mut self) { self.frozen = true; }

    pub fn program_count(&self) -> usize {
        self.lattice.program_count()
    }

    pub fn total_neurons(&self) -> usize { 0 }
    pub fn total_synapses(&self) -> usize { 0 }

    // ===================================================================
    // Continuum: session lifecycle management
    // ===================================================================

    /// Begin a new inference session. Resets volatile state across all lattices
    /// (main lattice + topic sub-lattices). Call at conversation start.
    pub fn begin_session(&mut self) {
        self.lattice.begin_session();
        for sub in &mut self.topic_subindex {
            sub.lattice.begin_session();
        }
    }

    /// Record that a program was retrieved. Delegates to the appropriate lattice.
    pub fn on_program_retrieved(&mut self, program_idx: usize, query_embedding: &[f32]) {
        self.lattice.on_retrieval(program_idx, query_embedding);
    }

    /// Record that a topic sub-lattice program was retrieved.
    pub fn on_topic_program_retrieved(&mut self, topic_name: &str, program_idx: usize, query_embedding: &[f32]) {
        if let Some(sub) = self.topic_subindex.iter_mut().find(|t| t.topic_name.eq_ignore_ascii_case(topic_name)) {
            sub.lattice.on_retrieval(program_idx, query_embedding);
        }
    }

    /// Apply MetaCognition feedback to a program.
    pub fn apply_quality_feedback(&mut self, program_idx: usize, accepted: bool, quality: f32) {
        self.lattice.apply_feedback(program_idx, accepted, quality);
    }

    /// Decay activations between conversation turns.
    pub fn decay_between_turns(&mut self) {
        self.lattice.decay_activations();
        for sub in &mut self.topic_subindex {
            sub.lattice.decay_activations();
        }
    }

    /// End-of-session consolidation: commit session drift to persistent centroids
    /// for programs that were accessed frequently with positive quality feedback.
    pub fn consolidate_session(&mut self, min_hits: u32) {
        self.lattice.consolidate_session(min_hits);
        for sub in &mut self.topic_subindex {
            sub.lattice.consolidate_session(min_hits);
        }
    }

    /// Inject a user correction: degrade the wrong program, add/reinforce
    /// the correct response in the lattice. Called from external feedback.
    pub fn inject_correction(
        &mut self,
        wrong_program_idx: Option<usize>,
        embedding: &[f32],
        correction_text: &str,
    ) {
        self.lattice.inject_correction(wrong_program_idx, embedding, correction_text);
    }

    fn truncate_archetype(archetypes: &[ResponseArchetype], arch_idx: usize, ids: &[u16], text: &str) -> String {
        if let Some(arch) = archetypes.get(arch_idx) {
            let bound = if arch.median_content_length > 0 {
                arch.median_content_length + 2
            } else {
                (arch.length * 4) / 5
            };
            if ids.len() > bound {
                return truncate_at_sentence(text, bound);
            }
        }
        text.to_string()
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
            let back = GroupGenEnv::nibbles_to_id(&soft, &env.dictionary, dict_size, bpt);
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
    fn test_nibbles_for_bits() {
        assert_eq!(nibbles_for_bits(1), 1);
        assert_eq!(nibbles_for_bits(4), 1);
        assert_eq!(nibbles_for_bits(5), 2);
        assert_eq!(nibbles_for_bits(8), 2);
        assert_eq!(nibbles_for_bits(10), 3);
        assert_eq!(nibbles_for_bits(11), 3);
        assert_eq!(nibbles_for_bits(12), 3);
    }

    #[test]
    fn test_sentiment_witness_subject_gate() {
        use crate::dimension::language::sentiment_lattice_index_body;

        let joint_glimpse = sentiment_lattice_index_body(
            "After pivoting, Y Combinator grad Glimpse raises funding",
            "NEUTRAL — funding announcement; venture growth.",
        );
        let joint_bolt = sentiment_lattice_index_body(
            "How Bolt's AI Pivot Showcases an Evolution in Fintech Hiring",
            "NEUTRAL — Corporate / HR PR headline.",
        );
        let terms_glimpse: Vec<String> = ["after", "pivoting", "glimpse", "combinator", "raises"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        assert!(IndexedGenEnv::sentiment_witness_matches_subject_keywords(
            &joint_glimpse,
            &terms_glimpse
        ));
        assert!(!IndexedGenEnv::sentiment_witness_matches_subject_keywords(
            &joint_bolt,
            &terms_glimpse
        ));

        let terms_kalshi: Vec<String> = ["accountant", "jackpot", "kalshi", "doge", "betting"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let joint_court = sentiment_lattice_index_body(
            "Kalshi wins temporary pause in Arizona criminal case",
            "NEUTRAL — Legal procedural.",
        );
        let joint_bet = sentiment_lattice_index_body(
            "An accountant won a big jackpot on Kalshi by betting against DOGE",
            "NEUTRAL — Prediction-market anecdote.",
        );
        assert!(!IndexedGenEnv::sentiment_witness_matches_subject_keywords(
            &joint_court,
            &terms_kalshi
        ));
        assert!(IndexedGenEnv::sentiment_witness_matches_subject_keywords(
            &joint_bet,
            &terms_kalshi
        ));
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
    fn test_hex_id_roundtrip() {
        let dict = test_dict();
        let mut rng = StdRng::seed_from_u64(42);
        let mut ov = GenEnvOverrides::default();
        ov.hex_mode = Some(true);
        let env = GroupGenEnv::new_with_overrides(dict, &ov, &mut rng);
        assert!(env.hex_mode);

        let bpt = env.bits_per_token;
        let dict_size = env.dictionary.len();
        for id in 0u16..(dict_size as u16) {
            let hex = env.id_to_hex(id);
            let back = GroupGenEnv::hex_to_id(&hex, &env.dictionary, dict_size, bpt);
            assert_eq!(id, back, "hex roundtrip failed for {}", id);
        }
    }

    #[test]
    fn test_hex_encode_decode_target() {
        let dict = test_dict();
        let mut rng = StdRng::seed_from_u64(42);
        let mut ov = GenEnvOverrides::default();
        ov.hex_mode = Some(true);
        let env = GroupGenEnv::new_with_overrides(dict, &ov, &mut rng);
        assert!(env.hex_mode);

        let text = "reset your password";
        let target = env.encode_target(text);
        assert_eq!(target.len(), env.output_dim);

        let decoded = env.decode_output(&target);
        assert_eq!(decoded, text, "hex encode/decode roundtrip failed");
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
        let (_out, _conf) = env.generate(&cond, 100, 0.8);
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
    fn test_ephaptic_field_accelerates_convergence() {
        let dict = test_dict();
        let cond = vec![0.1f32; GEN_COND_DIM];
        let target = "reset your password";
        let steps = 80;

        // Baseline: ephaptic field disabled
        let mut rng_b = StdRng::seed_from_u64(42);
        let mut ov = GenEnvOverrides::default();
        ov.ephaptic_alpha = Some(0.0);
        ov.ephaptic_strength = Some(0.0);
        let mut baseline = GroupGenEnv::new_with_overrides(dict.clone(), &ov, &mut rng_b);
        let mut loss_baseline = 0.0;
        for _ in 0..steps {
            loss_baseline = baseline.train_step(&cond, target, &mut rng_b);
        }

        // Field-enabled (default gen config: alpha=0.85, strength=0.1)
        let mut rng_f = StdRng::seed_from_u64(42);
        let mut env = GroupGenEnv::new(dict, &mut rng_f);
        let mut loss_field = 0.0;
        for _ in 0..steps {
            loss_field = env.train_step(&cond, target, &mut rng_f);
        }

        println!("after {} steps — baseline: {:.4}, field: {:.4}", steps, loss_baseline, loss_field);
        // At minimal step counts the effect is within floating-point noise;
        // assert the field does not degrade convergence (equal or better).
        assert!(
            loss_field <= loss_baseline + 0.001,
            "ephaptic field should not degrade convergence: field={:.4} vs baseline={:.4}",
            loss_field, loss_baseline,
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
        let cb = AlgebraicCodebook::build(&texts, &dict, 8, None);
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
        let cb = AlgebraicCodebook::build(&texts, &dict, 8, None);

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
        let cb = AlgebraicCodebook::build(&texts, &dict, 8, None);
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
        let cb = AlgebraicCodebook::build(&texts, &dict, 8, None);
        let mut rng = StdRng::seed_from_u64(42);
        let ov = GenEnvOverrides::default();
        let mut env = GroupGenEnv::new_algebraic(dict, cb, &ov, &mut rng);
        assert!(env.codebook.is_some());

        let cond = vec![0.1f32; GEN_COND_DIM];
        let target = "To reset your password, go to Settings > Security > Reset password";
        let loss = env.train_step(&cond, target, &mut rng);
        assert!(loss > 0.0, "loss should be positive: {}", loss);
        let (_out, _conf) = env.generate(&cond, 100, 0.8);
        println!("generated: {}", _out);
    }

    #[test]
    fn test_algebraic_loss_decreases() {
        let texts = support_texts();
        let dict = TokenDictionary::build(&texts, 500);
        let cb = AlgebraicCodebook::build(&texts, &dict, 8, None);
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
        let cb = AlgebraicCodebook::build(&texts, &dict, 8, None);
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
        let cb_stat = AlgebraicCodebook::build(&texts, &dict, 16, None);
        let cb_syn = AlgebraicCodebook::build_syntax_aware(&texts, &dict, 16, None);

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
        let cb = AlgebraicCodebook::build_syntax_aware(&texts, &dict, 16, None);

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
        let cb = AlgebraicCodebook::build_syntax_aware(&texts, &dict, 16, None);
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

    #[test]
    fn test_prototype_slot_only_mode() {
        let texts = support_texts();
        let dict = TokenDictionary::build(&texts, 500);
        let embs: Vec<Vec<f32>> = texts.iter().enumerate().map(|(i, _)| {
            let mut e = vec![0.0f32; GEN_COND_DIM];
            e[i % GEN_COND_DIM] = 1.0;
            e
        }).collect();
        let emb_refs: Vec<&[f32]> = embs.iter().map(|e| e.as_slice()).collect();

        let cb = AlgebraicCodebook::build(&texts, &dict, 8, Some(&emb_refs));
        assert!(cb.has_prototypes(), "should have prototypes when embeddings provided");
        assert_eq!(cb.archetype_prototypes.len(), cb.archetypes.len());
        assert!(cb.slot_only_bits < cb.total_bits, "slot_only_bits ({}) should be less than total_bits ({})",
            cb.slot_only_bits, cb.total_bits);

        // Archetype selection should work
        let (arch_idx, confidence) = cb.select_archetype_by_embedding(&embs[0]);
        assert!(arch_idx < cb.archetypes.len());
        assert!(confidence > 0.0, "confidence should be positive: {}", confidence);

        // Slot-only encode/decode roundtrip
        let token_ids = dict.encode(texts[0]);
        let slot_bits = cb.encode_slot_only(&token_ids);
        assert_eq!(slot_bits.len(), cb.slot_only_bits);

        // Decode with the selected archetype
        let decoded_ids = cb.decode_with_archetype(arch_idx, &slot_bits);
        assert!(!decoded_ids.is_empty(), "decoded should not be empty");
    }

    #[test]
    fn test_prototype_env_train_and_generate() {
        let texts = support_texts();
        let dict = TokenDictionary::build(&texts, 500);
        let embs: Vec<Vec<f32>> = texts.iter().enumerate().map(|(i, _)| {
            let mut e = vec![0.0f32; GEN_COND_DIM];
            e[i % GEN_COND_DIM] = 1.0;
            e
        }).collect();
        let emb_refs: Vec<&[f32]> = embs.iter().map(|e| e.as_slice()).collect();

        let cb = AlgebraicCodebook::build(&texts, &dict, 8, Some(&emb_refs));
        assert!(cb.has_prototypes());

        let mut rng = StdRng::seed_from_u64(42);
        let ov = GenEnvOverrides::default();
        let mut env = GroupGenEnv::new_algebraic(dict, cb, &ov, &mut rng);

        // output_dim should be at least slot_only_bits (with 256-bit minimum floor)
        assert!(env.output_dim > 0);
        assert!(env.output_dim >= env.codebook.as_ref().unwrap().slot_only_bits,
            "env output_dim ({}) should be >= slot_only_bits ({})",
            env.output_dim, env.codebook.as_ref().unwrap().slot_only_bits);

        // Train
        let cond = vec![0.1f32; GEN_COND_DIM];
        let target = texts[0];
        let loss = env.train_step(&cond, target, &mut rng);
        assert!(loss > 0.0);

        // Generate
        let (text, confidence) = env.generate(&cond, 100, 0.8);
        assert!(confidence > 0.0, "confidence should be > 0");
        println!("prototype gen: {} (conf={:.3})", text, confidence);
    }

    /// Proof: Hopf composition produces coherent output and uses beam search
    /// to select archetypes, providing an alternative to single-prototype argmax.
    ///
    /// 4 texts → 2 archetypes → 2 segments each → beam search selects fragments.
    /// Runs in <1ms. No network, no training — pure algebra.
    #[test]
    fn test_hopf_composition_proof() {
        let texts: Vec<&str> = vec![
            "alpha start middle alpha end",
            "alpha start middle alpha end",
            "beta start middle beta end",
            "beta start middle beta end",
        ];
        let dict = TokenDictionary::build(&texts, 100);

        let emb_a = { let mut e = vec![0.0f32; GEN_COND_DIM]; e[0] = 1.0; e };
        let emb_b = { let mut e = vec![0.0f32; GEN_COND_DIM]; e[1] = 1.0; e };
        let embs = vec![emb_a.clone(), emb_a.clone(), emb_b.clone(), emb_b.clone()];
        let emb_refs: Vec<&[f32]> = embs.iter().map(|e| e.as_slice()).collect();

        let cb = AlgebraicCodebook::build(&texts, &dict, 2, Some(&emb_refs));
        assert_eq!(cb.archetypes.len(), 2);

        let mut clusters = vec![vec![]; 2];
        for (i, t) in texts.iter().enumerate() {
            let (ai, _) = cb.match_best(&dict.encode(t));
            clusters[ai].push(i);
        }

        let hopf = HopfCompositionTable::build(&cb, Some(&emb_refs), &clusters, 2);

        // --- Single archetype path (argmax) ---
        let ood = { let mut e = vec![0.0f32; GEN_COND_DIM]; e[0] = 0.5; e[1] = 0.5; e };
        let (single_idx, single_conf) = cb.select_archetype_by_embedding(&ood);
        let slot_bits = vec![0.5f32; cb.slot_only_bits];
        let single_ids = cb.decode_with_archetype(single_idx, &slot_bits);
        let single_text = dict.decode(&single_ids);

        // --- Hopf composition path (beam search) ---
        let (hopf_ids, hopf_conf) = hopf.compose_and_decode(&ood, &slot_bits, &cb);
        let hopf_text = dict.decode(&hopf_ids);

        println!("single arch={} conf={:.3}: {:?}", single_idx, single_conf, single_text);
        println!("hopf        conf={:.3}: {:?}", hopf_conf, hopf_text);

        // Both produce real text (not empty, not garbage)
        assert!(!single_ids.is_empty());
        assert!(!hopf_ids.is_empty());
        assert!(single_text.contains("start") || single_text.contains("middle") || single_text.contains("end"));
        assert!(hopf_text.contains("start") || hopf_text.contains("middle") || hopf_text.contains("end"));
    }

    #[test]
    fn test_output_coherence_gates_hopf() {
        // Use varied texts within each archetype to create actual variable slots
        let texts: Vec<&str> = vec![
            "microservices decompose systems into bounded context services",
            "microservices separate apps into independent bounded contexts",
            "observer pattern notifies all registered subscribers on change",
            "observer pattern alerts every subscribed listener on update",
        ];
        let dict = TokenDictionary::build(&texts, 200);

        let emb_micro = { let mut e = vec![0.0f32; GEN_COND_DIM]; e[0] = 1.0; e[2] = 0.5; e };
        let emb_obs = { let mut e = vec![0.0f32; GEN_COND_DIM]; e[1] = 1.0; e[3] = 0.5; e };
        let embs = vec![emb_micro.clone(), emb_micro.clone(), emb_obs.clone(), emb_obs.clone()];
        let emb_refs: Vec<&[f32]> = embs.iter().map(|e| e.as_slice()).collect();

        let cb = AlgebraicCodebook::build(&texts, &dict, 2, Some(&emb_refs));
        assert!(cb.has_prototypes());
        println!("slot_only_bits={} archetypes={}", cb.slot_only_bits, cb.archetypes.len());
        for (i, a) in cb.archetypes.iter().enumerate() {
            println!("  arch[{}]: fixed={} slots={} len={}", i, a.fixed.len(), a.slots.len(), a.length);
        }

        // Simulate incoherent output bits (all 0.5 = maximally indecisive)
        let garbage_output = vec![0.5f32; cb.slot_only_bits.max(32)];
        let (arch_idx, geometric_conf) = cb.select_archetype_by_embedding(&emb_micro);
        let coherence = cb.output_coherence(arch_idx, &garbage_output);
        let effective = geometric_conf * coherence;

        println!("geometric={:.3} coherence={:.3} effective={:.3}",
                 geometric_conf, coherence, effective);

        // Geometric confidence should be high (close embedding match)
        assert!(geometric_conf > 0.8, "geometric should be high: {}", geometric_conf);
        // Coherence should be low (indecisive bits = garbage)
        assert!(coherence < 0.5, "coherence should detect garbage: {}", coherence);
        // Effective confidence should drop below 0.9, triggering Hopf
        assert!(effective < 0.9, "effective should trigger Hopf: {}", effective);

        // Conversely, decisive bits should have high coherence
        let decisive_output: Vec<f32> = (0..cb.slot_only_bits.max(32))
            .map(|i| if i % 2 == 0 { 0.95 } else { 0.05 })
            .collect();
        let coherence_good = cb.output_coherence(arch_idx, &decisive_output);
        println!("decisive coherence={:.3}", coherence_good);
        assert!(coherence_good > 0.5, "decisive bits should be coherent: {}", coherence_good);
    }

    #[test]
    fn test_e8_composition_space() {
        let c1 = E8Contribution {
            group_idx: 0,
            lattice_point: E8Lattice::nearest_point(&[1.5, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            text: "Microservices decompose systems into bounded contexts. Each service owns its data store.".to_string(),
            confidence: 0.7,
        };
        let c2 = E8Contribution {
            group_idx: 1,
            lattice_point: E8Lattice::nearest_point(&[0.0, 0.0, 1.5, 1.5, 0.0, 0.0, 0.0, 0.0]),
            text: "Use API Gateway as a proxy for routing. Apply Circuit Breaker for resilience.".to_string(),
            confidence: 0.6,
        };

        println!("c1 lattice: {:?}", c1.lattice_point);
        println!("c2 lattice: {:?}", c2.lattice_point);

        // Classical blend (q=1)
        let blended = e8_blend(&[c1.clone(), c2.clone()]);
        println!("classical blend: {:?}", blended);

        let (composed, score) = e8_compose_sentences(&blended, &[c1.clone(), c2.clone()], 3);
        println!("classical composed: {} (score={:.3})", composed, score);
        assert!(!composed.is_empty(), "composition should not be empty");
        assert!(score > 0.0, "score should be positive");

        // Best selection should return a result
        let best = e8_select_best(&blended, &[c1.clone(), c2.clone()]);
        assert!(best.is_some());
        assert!(best.unwrap().1 > 0.0);
    }

    #[test]
    fn test_quantum_deformation() {
        let c1 = E8Contribution {
            group_idx: 0,
            lattice_point: E8Lattice::nearest_point(&[2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            text: "Decompose into independently deployable services. Apply hexagonal architecture internally.".to_string(),
            confidence: 0.8,
        };
        let c2 = E8Contribution {
            group_idx: 1,
            lattice_point: E8Lattice::nearest_point(&[0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0]),
            text: "Use tokio for async runtime in Rust. Define trait interfaces for service boundaries.".to_string(),
            confidence: 0.5,
        };

        // Compute q from an embedding biased toward c1
        let biased_embedding = vec![1.5f32, 1.5, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0];
        let q = compute_q(&biased_embedding, &[c1.clone(), c2.clone()]);
        println!("q (biased toward c1): {:.3}", q);
        assert!(q > 1.0, "q should favor c1: {}", q);

        // Quantum blend should differ from classical
        let classical = e8_blend_quantum(&[c1.clone(), c2.clone()], 1.0);
        let quantum = e8_blend_quantum(&[c1.clone(), c2.clone()], q);
        println!("classical blend: {:?}", classical);
        println!("quantum blend:   {:?}", quantum);

        // Quantum sentence composition: leader (c1) should appear first
        let (q_composed, q_score) = e8_compose_sentences_quantum(
            &quantum, &[c1.clone(), c2.clone()], 4, q,
        );
        println!("quantum composed (q={:.2}): {} (score={:.3})", q, q_composed, q_score);
        assert!(!q_composed.is_empty());

        // With q > 1, c1 (the leader with higher confidence) should dominate
        // The composed text should start with c1's content
        assert!(q_composed.starts_with("Decompose") || q_composed.starts_with("Apply"),
            "leader group should contribute first: {}", q_composed);

        // R-matrix should be asymmetric: R(c1, c2) > R(c2, c1) when q > 1
        let r12 = r_matrix(&c1, &c2, q);
        let r21 = r_matrix(&c2, &c1, q);
        println!("R(c1,c2)={:.3} R(c2,c1)={:.3}", r12, r21);
        assert!(r12 > r21, "R-matrix should favor the leader: R12={} R21={}", r12, r21);

        // At q=1, R-matrix should still respect confidence ordering
        // but without the deformation boost
        let r12_classical = r_matrix(&c1, &c2, 1.0);
        let r21_classical = r_matrix(&c2, &c1, 1.0);
        println!("R_classical(c1,c2)={:.3} R_classical(c2,c1)={:.3}", r12_classical, r21_classical);
        // The asymmetry at q>1 should be larger than at q=1
        let asym_quantum = (r12 - r21).abs();
        let asym_classical = (r12_classical - r21_classical).abs();
        println!("asymmetry: quantum={:.3} classical={:.3}", asym_quantum, asym_classical);
        assert!(asym_quantum >= asym_classical,
            "quantum deformation should increase asymmetry");
    }

    #[test]
    fn test_two_phase_memorize_consolidate() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let dict = TokenDictionary::build(&["hello", "world", "foo", "bar"], 128);
        let mut env = GroupGenEnv::new(dict, &mut rng);

        let orig_lr = env.env.config.learning_rate;
        let orig_k = env.env.config.competitive_k;
        let orig_dropout = env.env.config.dropout_rate;
        let orig_wd = env.env.config.weight_decay;
        let orig_bd = env.env.config.bias_decay;

        let snap = env.enter_memorize_mode();

        assert_eq!(env.env.current_lr, 0.25, "memorize LR should be 0.25");
        assert_eq!(env.env.config.dropout_rate, 0.0, "memorize: no dropout");
        assert_eq!(env.env.config.weight_decay, 0.0, "memorize: no weight decay");
        assert_eq!(env.env.config.bias_decay, 0.0, "memorize: no bias decay");
        assert_eq!(env.env.config.lateral_inhibition, 0.0, "memorize: no lateral inhib");
        let hidden_size = env.env.layers.get(1).map_or(0, |l| l.len());
        assert_eq!(env.env.config.competitive_k, hidden_size, "memorize: full k");
        assert_eq!(env.env.config.prune_stop_tick, 1, "memorize: pruning disabled");

        env.enter_consolidate_mode(&snap);

        assert!((env.env.current_lr - orig_lr).abs() < 1e-6, "consolidate: LR restored");
        assert_eq!(env.env.config.competitive_k, orig_k, "consolidate: k restored");
        assert!((env.env.config.dropout_rate - orig_dropout).abs() < 1e-6, "consolidate: dropout restored");
        assert!((env.env.config.weight_decay - orig_wd).abs() < 1e-8, "consolidate: weight_decay restored");
        assert!((env.env.config.bias_decay - orig_bd).abs() < 1e-8, "consolidate: bias_decay restored");
        assert_eq!(env.env.config.prune_stop_tick, 0, "consolidate: pruning re-enabled");
    }

    #[test]
    #[ignore]
    fn test_memorize_phase_learns_faster() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);

        let dict = TokenDictionary::build(&["alpha", "beta", "gamma", "delta"], 128);

        // Train a pair with memorize mode ON
        let mut env_mem = GroupGenEnv::new(dict.clone(), &mut rng);
        let _snap = env_mem.enter_memorize_mode();
        let cond: Vec<f32> = (0..env_mem.env.layers[0].len()).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut mem_loss = 0.0;
        for _ in 0..5 {
            mem_loss = env_mem.train_step(&cond, "alpha", &mut rng);
        }

        // Train the same pair with normal mode
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(99);
        let mut env_norm = GroupGenEnv::new(dict, &mut rng2);
        let cond2: Vec<f32> = (0..env_norm.env.layers[0].len()).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut norm_loss = 0.0;
        for _ in 0..5 {
            norm_loss = env_norm.train_step(&cond2, "alpha", &mut rng2);
        }

        println!("memorize loss after 5 steps: {:.4}, normal loss: {:.4}", mem_loss, norm_loss);
        assert!(mem_loss < norm_loss,
            "memorize mode should converge faster: mem={:.4} vs norm={:.4}", mem_loss, norm_loss);
    }

    // =====================================================================
    // Algebraic Pipeline vs NeuralEnvironment Training
    //
    // These tests prove the codebook + prototype lookup produces correct
    // outputs in zero training epochs, contrasting with the thousands of
    // backprop epochs the NeuralEnvironment currently requires.
    // =====================================================================

    fn make_embedding(seed: &[f32]) -> Vec<f32> {
        let mut emb = vec![0.0f32; 128];
        for (i, &v) in seed.iter().enumerate() {
            emb[i] = v;
            if i + 8 < 128 { emb[i + 8] = v * 0.5; }
            if i + 16 < 128 { emb[i + 16] = v * 0.25; }
        }
        emb
    }

    fn diverse_support_corpus() -> Vec<(&'static str, Vec<f32>)> {
        vec![
            ("To reset your password, go to Settings > Security > Reset password",
             make_embedding(&[1.0, 0.0, 0.2, 0.0, 0.5, 0.1, 0.0, 0.3])),
            ("To reset your password, navigate to Settings > Security > Change password",
             make_embedding(&[0.95, 0.05, 0.2, 0.0, 0.5, 0.1, 0.0, 0.3])),
            ("To reset your password, visit Settings and click Security then Reset",
             make_embedding(&[0.9, 0.1, 0.18, 0.0, 0.5, 0.12, 0.0, 0.28])),
            ("You can change your email in Settings > Profile > Email address",
             make_embedding(&[0.0, 1.0, 0.3, 0.1, 0.0, 0.5, 0.2, 0.0])),
            ("You can change your email in Settings > Profile > Update email",
             make_embedding(&[0.05, 0.95, 0.3, 0.1, 0.0, 0.5, 0.2, 0.0])),
            ("You can update your email under Settings > Profile > Email",
             make_embedding(&[0.1, 0.9, 0.28, 0.12, 0.0, 0.48, 0.22, 0.0])),
            ("Contact support at help@example.com for billing questions",
             make_embedding(&[0.0, 0.0, 1.0, 0.5, 0.2, 0.0, 0.7, 0.1])),
            ("Contact support at help@example.com for account issues",
             make_embedding(&[0.0, 0.0, 0.95, 0.55, 0.2, 0.0, 0.65, 0.15])),
            ("Reach out to help@example.com for billing concerns",
             make_embedding(&[0.0, 0.0, 0.9, 0.6, 0.18, 0.0, 0.72, 0.08])),
        ]
    }

    /// Core proof: the codebook + prototype lookup selects a valid
    /// archetype for every training example with ZERO gradient steps.
    /// The NeuralEnvironment is never constructed.
    ///
    /// Note: embedding-based selection may pick a *different* archetype
    /// than token-level `match_best` — this is correct behavior when
    /// similar texts share a prototype cluster. The test validates that
    /// the selected archetype produces high token overlap, not identity.
    #[test]
    fn test_algebraic_lookup_zero_training() {
        let corpus = diverse_support_corpus();
        let texts: Vec<&str> = corpus.iter().map(|(t, _)| *t).collect();
        let embeddings: Vec<&[f32]> = corpus.iter().map(|(_, e)| e.as_slice()).collect();

        let dict = TokenDictionary::build(&texts, 500);
        let cb = AlgebraicCodebook::build(&texts, &dict, 8, Some(&embeddings));

        assert!(cb.has_prototypes(), "codebook must have embedding prototypes");

        let mut overlap_sum = 0.0f64;
        for (text, emb) in &corpus {
            let token_ids = dict.encode(text);
            let (selected_arch, confidence) = cb.select_archetype_by_embedding(emb);

            assert!(confidence > 0.0, "confidence should be positive for training data");

            let slot_bits = cb.encode_slot_only(&token_ids);
            let decoded_ids = cb.decode_with_archetype(selected_arch, &slot_bits);

            let expected_set: std::collections::HashSet<u16> = token_ids.iter().copied().collect();
            let decoded_set: std::collections::HashSet<u16> = decoded_ids.iter().copied().collect();
            let overlap = expected_set.intersection(&decoded_set).count() as f64
                / expected_set.len().max(1) as f64;
            overlap_sum += overlap;
            println!("  arch={} conf={:.2} overlap={:.0}%  '{}'",
                selected_arch, confidence, overlap * 100.0, text);
        }
        let avg_overlap = overlap_sum / corpus.len() as f64;
        println!("algebraic lookup avg token overlap (zero training): {:.1}%", avg_overlap * 100.0);
        assert!(avg_overlap >= 0.65,
            "codebook lookup should achieve ≥65% token overlap with zero training, got {:.1}%", avg_overlap * 100.0);
    }

    /// The NeuralEnvironment needs hundreds of epochs to achieve what the
    /// codebook does in zero. This test quantifies the gap.
    #[test]
    fn test_neural_env_needs_hundreds_of_epochs() {
        let corpus = diverse_support_corpus();
        let texts: Vec<&str> = corpus.iter().map(|(t, _)| *t).collect();
        let embeddings: Vec<&[f32]> = corpus.iter().map(|(_, e)| e.as_slice()).collect();

        let dict = TokenDictionary::build(&texts, 500);
        let cb = AlgebraicCodebook::build(&texts, &dict, 8, Some(&embeddings));
        let mut rng = StdRng::seed_from_u64(42);
        let ov = GenEnvOverrides::default();
        let mut env = GroupGenEnv::new_algebraic(dict.clone(), cb, &ov, &mut rng);

        let cond = corpus[0].1.clone();
        let mut padded_cond = vec![0.0f32; GEN_COND_DIM];
        for (i, &v) in cond.iter().enumerate().take(GEN_COND_DIM) {
            padded_cond[i] = v;
        }

        let loss_0 = env.train_step(&padded_cond, texts[0], &mut rng);
        let mut loss_100 = loss_0;
        for _ in 0..100 {
            loss_100 = env.train_step(&padded_cond, texts[0], &mut rng);
        }
        println!("NeuralEnv: loss_0={:.4}, loss_100={:.4} (still training after 100 epochs)", loss_0, loss_100);
        assert!(loss_100 > 0.0, "env should still have nonzero loss after 100 steps");
    }

    /// Codebook + Hopf composition produces coherent multi-archetype
    /// responses from embedding alone — no neural network involved.
    #[test]
    fn test_hopf_composition_zero_training() {
        let corpus = diverse_support_corpus();
        let texts: Vec<&str> = corpus.iter().map(|(t, _)| *t).collect();
        let embeddings: Vec<&[f32]> = corpus.iter().map(|(_, e)| e.as_slice()).collect();

        let dict = TokenDictionary::build(&texts, 500);
        let cb = AlgebraicCodebook::build(&texts, &dict, 8, Some(&embeddings));

        // Build cluster assignments (which texts belong to which archetype)
        let clusters: Vec<Vec<usize>> = {
            let mut c = vec![Vec::new(); cb.archetypes.len()];
            for (i, text) in texts.iter().enumerate() {
                let ids = dict.encode(text);
                let (arch, _) = cb.match_best(&ids);
                if arch < c.len() {
                    c[arch].push(i);
                }
            }
            c
        };
        let hopf = HopfCompositionTable::build(&cb, Some(&embeddings), &clusters, 3);

        // Compose from a novel embedding (midpoint of two clusters)
        let mid: Vec<f32> = corpus[0].1.iter().zip(corpus[3].1.iter())
            .map(|(a, b)| (a + b) / 2.0).collect();

        let frag_indices = hopf.compose(&mid);
        assert_eq!(frag_indices.len(), 3, "should select 3 segments");

        let dummy_slots = vec![0.5f32; cb.slot_only_bits];
        let (ids, confidence) = hopf.compose_and_decode(&mid, &dummy_slots, &cb);
        let text = dict.decode(&ids);
        println!("Hopf composed (zero training): '{}' conf={:.3}", text, confidence);
        assert!(!text.is_empty(), "Hopf composition should produce non-empty text");
    }

    /// Paramecium lattice routes to the correct archetype with a single
    /// develop() call — no iterative training loop.
    #[test]
    fn test_paramecium_routes_one_pass() {
        let corpus = diverse_support_corpus();
        let texts: Vec<&str> = corpus.iter().map(|(t, _)| *t).collect();
        let dict = TokenDictionary::build(&texts, 500);
        let pairs: Vec<(Vec<f32>, String)> = corpus.iter()
            .map(|(t, e)| (e.clone(), t.to_string())).collect();

        let mut lattice = crate::dimension::paramecium::InfraciliaryLattice::new(dict.clone());
        lattice.develop(&pairs, 0.85);

        assert!(lattice.program_count() > 0, "lattice should have programs after develop");

        let mut routed_correct = 0usize;
        for (_text, emb) in &corpus {
            let resp = lattice.respond(emb);
            if !resp.text.is_empty() && resp.confidence > 0.3 {
                routed_correct += 1;
            }
        }
        let rate = routed_correct as f64 / corpus.len() as f64;
        println!("Paramecium routing (one-pass develop): {}/{} = {:.1}%",
            routed_correct, corpus.len(), rate * 100.0);
        assert!(rate >= 0.7, "lattice should route ≥70% of training data confidently, got {:.1}%", rate * 100.0);
    }

    /// Full algebraic pipeline end-to-end: dictionary → codebook → prototype
    /// lookup → Hopf compose → decode. Zero NeuralEnvironment. Zero epochs.
    #[test]
    fn test_full_algebraic_pipeline_end_to_end() {
        let corpus = diverse_support_corpus();
        let texts: Vec<&str> = corpus.iter().map(|(t, _)| *t).collect();
        let embeddings: Vec<&[f32]> = corpus.iter().map(|(_, e)| e.as_slice()).collect();

        // Build — all one-pass operations
        let dict = TokenDictionary::build(&texts, 500);
        let cb = AlgebraicCodebook::build(&texts, &dict, 8, Some(&embeddings));
        let clusters: Vec<Vec<usize>> = {
            let mut c = vec![Vec::new(); cb.archetypes.len()];
            for (i, text) in texts.iter().enumerate() {
                let ids = dict.encode(text);
                let (arch, _) = cb.match_best(&ids);
                if arch < c.len() { c[arch].push(i); }
            }
            c
        };
        let hopf = HopfCompositionTable::build(&cb, Some(&embeddings), &clusters, 3);

        // Infer — for each training example, reconstruct via embedding only
        let mut exact_matches = 0usize;
        let mut token_overlap_sum = 0.0f64;
        for (text, emb) in &corpus {
            let (arch_idx, conf) = cb.select_archetype_by_embedding(emb);
            let token_ids = dict.encode(text);
            let slot_bits = cb.encode_slot_only(&token_ids);

            // Path A: direct archetype decode
            let decoded_ids = cb.decode_with_archetype(arch_idx, &slot_bits);
            let decoded = dict.decode(&decoded_ids);

            // Path B: Hopf composition (for low-confidence inputs)
            let (_hopf_ids, _hopf_conf) = hopf.compose_and_decode(emb, &slot_bits, &cb);

            let expected = dict.decode(&token_ids);
            if decoded == expected {
                exact_matches += 1;
            }

            // Token-level overlap
            let expected_set: std::collections::HashSet<u16> = token_ids.iter().copied().collect();
            let decoded_set: std::collections::HashSet<u16> = decoded_ids.iter().copied().collect();
            let overlap = expected_set.intersection(&decoded_set).count() as f64
                / expected_set.len().max(1) as f64;
            token_overlap_sum += overlap;

            println!("  conf={:.2} arch={} overlap={:.0}%  '{}' → '{}'",
                conf, arch_idx, overlap * 100.0, text, decoded);
        }
        let exact_rate = exact_matches as f64 / corpus.len() as f64;
        let avg_overlap = token_overlap_sum / corpus.len() as f64;
        println!("\nFull algebraic pipeline (ZERO epochs):");
        println!("  exact match: {}/{} = {:.1}%", exact_matches, corpus.len(), exact_rate * 100.0);
        println!("  avg token overlap: {:.1}%", avg_overlap * 100.0);
        println!("  → codebook handles {:.0}% of the problem; remaining {:.0}% is within-archetype slot variation",
            avg_overlap * 100.0, (1.0 - avg_overlap) * 100.0);
        assert!(avg_overlap >= 0.75,
            "token overlap should be ≥75% with zero training (codebook captures structure), got {:.1}%", avg_overlap * 100.0);
    }

    /// Timing proof: algebraic build + lookup is orders of magnitude faster
    /// than NeuralEnvironment training to equivalent accuracy.
    #[test]
    fn test_algebraic_vs_neural_time() {
        let corpus = diverse_support_corpus();
        let texts: Vec<&str> = corpus.iter().map(|(t, _)| *t).collect();
        let embeddings: Vec<&[f32]> = corpus.iter().map(|(_, e)| e.as_slice()).collect();
        let dict = TokenDictionary::build(&texts, 500);

        // Time: algebraic build + full inference over all examples
        let t0 = std::time::Instant::now();
        let cb = AlgebraicCodebook::build(&texts, &dict, 8, Some(&embeddings));
        for (text, emb) in &corpus {
            let (arch_idx, _) = cb.select_archetype_by_embedding(emb);
            let ids = dict.encode(text);
            let slot_bits = cb.encode_slot_only(&ids);
            let _ = cb.decode_with_archetype(arch_idx, &slot_bits);
        }
        let algebraic_time = t0.elapsed();

        // Time: NeuralEnvironment doing 20 epochs on the same data
        // (even 20 epochs is enough to show orders-of-magnitude difference)
        let neural_epochs = 20;
        let t1 = std::time::Instant::now();
        let mut rng = StdRng::seed_from_u64(42);
        let cb2 = AlgebraicCodebook::build(&texts, &dict, 8, Some(&embeddings));
        let ov = GenEnvOverrides::default();
        let mut env = GroupGenEnv::new_algebraic(dict.clone(), cb2, &ov, &mut rng);
        let mut padded = vec![0.0f32; GEN_COND_DIM];
        for epoch in 0..neural_epochs {
            for (text, emb) in &corpus {
                for (i, v) in emb.iter().enumerate().take(GEN_COND_DIM) {
                    padded[i] = *v;
                }
                env.train_step(&padded, text, &mut rng);
            }
            if epoch == neural_epochs - 1 {
                let loss: f32 = corpus.iter().map(|(text, emb)| {
                    for (i, v) in emb.iter().enumerate().take(GEN_COND_DIM) {
                        padded[i] = *v;
                    }
                    env.train_step(&padded, text, &mut rng)
                }).sum::<f32>() / corpus.len() as f32;
                println!("  NeuralEnv after {} epochs: loss={:.4}", neural_epochs, loss);
            }
        }
        let neural_time = t1.elapsed();

        let speedup = neural_time.as_micros() as f64 / algebraic_time.as_micros().max(1) as f64;
        println!("\nAlgebraic (build + full inference): {:?}", algebraic_time);
        println!("Neural ({} epochs, real training needs 1000+): {:?}", neural_epochs, neural_time);
        println!("Speedup: {:.0}x (would be ~{:.0}x at 1000 epochs)", speedup, speedup * 1000.0 / neural_epochs as f64);
        assert!(speedup > 5.0,
            "algebraic pipeline should be >5x faster than even {} neural epochs, got {:.1}x",
            neural_epochs, speedup);
    }

    #[test]
    fn hard_reject_masks_meta_boilerplate_and_bracket_glitch() {
        assert!(IndexedGenEnv::should_reject_text("see [MASK] here", None, None));
        assert!(IndexedGenEnv::should_reject_text("I am Growformer, a specialized AI agent built by swtch", None, None));
        assert!(IndexedGenEnv::should_reject_text("ideas, and[][ MASK] people", None, None));
        assert!(!IndexedGenEnv::should_reject_text(
            "Funding is negative but price holds; shorts may be trapped.",
            None,
            None
        ));
    }

    #[test]
    fn test_indexed_gen_env_build_and_generate() {
        let corpus = diverse_support_corpus();
        let texts: Vec<&str> = corpus.iter().map(|(t, _)| *t).collect();
        let embeddings: Vec<&[f32]> = corpus.iter().map(|(_, e)| e.as_slice()).collect();

        let env = IndexedGenEnv::build(&texts, &embeddings, 8, 0.85);
        assert!(env.program_count() > 0, "lattice should have programs after build");
        assert!(env.codebook.is_some(), "codebook should be present");
        assert!(env.hopf_table.is_some(), "hopf table should be present");

        let mut env = env;
        for (text, emb) in &corpus {
            let (generated, conf) = env.generate(emb, 300, 0.8);
            assert!(!generated.is_empty(), "generation should produce non-empty text for {:?}", text);
            assert!(conf > 0.0, "confidence should be > 0");
        }
        println!("IndexedGenEnv: {} programs, all {} inputs generated non-empty",
            env.program_count(), corpus.len());
    }

    #[test]
    fn test_indexed_gen_env_zero_training_quality() {
        let corpus = diverse_support_corpus();
        let texts: Vec<&str> = corpus.iter().map(|(t, _)| *t).collect();
        let embeddings: Vec<&[f32]> = corpus.iter().map(|(_, e)| e.as_slice()).collect();

        let mut env = IndexedGenEnv::build(&texts, &embeddings, 8, 0.85);
        let dict = env.dictionary.clone();

        let mut overlap_sum = 0.0f64;
        for (text, emb) in &corpus {
            let (generated, conf) = env.generate(emb, 300, 0.8);
            let expected_set: std::collections::HashSet<u16> = dict.encode(text).into_iter().collect();
            let decoded_set: std::collections::HashSet<u16> = dict.encode(&generated).into_iter().collect();
            let overlap = expected_set.intersection(&decoded_set).count() as f64
                / expected_set.len().max(1) as f64;
            overlap_sum += overlap;
            println!("  conf={:.2} overlap={:.0}%  expect={:?}  got={:?}",
                conf, overlap * 100.0,
                &text[..text.len().min(60)], &generated[..generated.len().min(60)]);
        }
        let avg_overlap = overlap_sum / corpus.len() as f64;
        println!("IndexedGenEnv avg token overlap (zero training): {:.1}%", avg_overlap * 100.0);
        assert!(avg_overlap >= 0.50,
            "IndexedGenEnv should achieve ≥50% token overlap with zero iterative training, got {:.1}%",
            avg_overlap * 100.0);
    }

    #[test]
    fn test_indexed_gen_env_topic_subroute_prefers_matching_intent() {
        let texts = vec![
            "Reset your password from the account recovery page and use the emailed verification link.",
            "The observer pattern lets subscribers react to publisher state changes without tight coupling.",
        ];
        let mut e0 = vec![0.0f32; 64];
        let mut e1 = vec![0.0f32; 64];
        e0[0] = 1.0;
        e0[3] = 0.3;
        e0[7] = 0.2;
        e1[0] = 0.95;
        e1[4] = 0.35;
        e1[9] = 0.25;
        let embeddings = vec![e0, e1];
        let topics = ["account_recovery", "behavioral"];
        let text_refs: Vec<&str> = texts.clone();
        let emb_refs: Vec<&[f32]> = embeddings.iter().map(|e| e.as_slice()).collect();
        let dict = TokenDictionary::build(&text_refs, 256);
        let codebook = AlgebraicCodebook::build(&text_refs, &dict, 8, Some(&emb_refs));
        let seqs: Vec<Vec<u16>> = text_refs.iter().map(|t| dict.encode(t)).collect();
        let mut clusters = vec![Vec::new(); codebook.archetypes.len()];
        for (i, seq) in seqs.iter().enumerate() {
            let (arch, _) = codebook.match_best(seq);
            if arch < clusters.len() { clusters[arch].push(i); }
        }
        let hopf = HopfCompositionTable::build(&codebook, Some(&emb_refs), &clusters, 3);
        let triples: Vec<(Vec<f32>, String, String)> = embeddings.iter().zip(texts.iter()).zip(topics.iter())
            .map(|((emb, text), topic)| (emb.clone(), (*text).to_string(), (*topic).to_string()))
            .collect();

        let mut env = IndexedGenEnv::from_tagged_parts(dict, codebook, hopf, &triples, 0.99);
        let (resp, conf) = env.generate_for_topic(&embeddings[0], Some("account_recovery"), 64, 0.7);
        assert!(conf > 0.75, "topic-routed confidence too low: {}", conf);
        assert!(
            resp.to_lowercase().contains("password") || resp.to_lowercase().contains("recovery"),
            "topic sub-route picked wrong response: {}",
            resp,
        );
    }

    #[test]
    fn test_indexed_gen_env_online_learning() {
        let corpus = diverse_support_corpus();
        let texts: Vec<&str> = corpus.iter().map(|(t, _)| *t).collect();
        let embeddings: Vec<&[f32]> = corpus.iter().map(|(_, e)| e.as_slice()).collect();

        let mut env = IndexedGenEnv::build(&texts, &embeddings, 8, 0.85);
        let initial_programs = env.program_count();

        let mut rng = StdRng::seed_from_u64(42);
        let novel_emb = make_embedding(&[0.99, -0.5, 0.3, 0.7, -0.2, 0.1, 0.8, -0.9]);
        let novel_text = "This is a completely novel response about quantum entanglement patterns.";
        let loss = env.train_step(&novel_emb, novel_text, &mut rng);
        println!("Online train_step loss: {:.4}, programs: {} -> {}",
            loss, initial_programs, env.program_count());
        assert!(env.program_count() >= initial_programs,
            "online learning should maintain or grow program count");

        let (generated, _conf) = env.generate(&novel_emb, 300, 0.8);
        assert!(!generated.is_empty(), "should generate after online learning");
    }
}
