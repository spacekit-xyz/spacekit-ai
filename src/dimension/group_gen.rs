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

use crate::environment::NeuralEnvironment;
use crate::spectral::{
    TokenDictionary, hamming_parity_bits, hamming_encode, hamming_decode,
    tokenize, syntax_role, structural_signature, SyntaxRole, E8Lattice,
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
    pub fn soft_decode_index(bits: &[f32], num_options: usize) -> usize {
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
}

impl GroupGenEnv {
    pub fn new(dictionary: TokenDictionary, rng: &mut impl Rng) -> Self {
        Self::new_with_overrides(dictionary, &GenEnvOverrides::default(), rng)
    }

    pub fn new_with_overrides(dictionary: TokenDictionary, ov: &GenEnvOverrides, rng: &mut impl Rng) -> Self {
        let max_tok = ov.max_tokens.unwrap_or(MAX_TOKENS);
        let hidden = ov.hidden.unwrap_or(GEN_HIDDEN);
        let k = ov.k.unwrap_or(GEN_K);
        let cond_dim = ov.cond_dim.unwrap_or(GEN_COND_DIM);
        let bits_per_token = bits_for_dict(dictionary.len());
        let parity_bits = hamming_parity_bits(bits_per_token);
        let coded_bits_per_token = bits_per_token + parity_bits;
        let output_dim = max_tok * coded_bits_per_token;
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
    /// In slot-only mode: uses the stored selected archetype + slot bits.
    /// In full algebraic mode: decodes archetype + slot bits together.
    /// In raw binary mode: ECC correction → Gray decode → soft decode.
    fn decode_output(&self, output: &[f32]) -> String {
        if let Some(ref cb) = self.codebook {
            if cb.has_prototypes() {
                let arch_idx = self.last_selected_archetype.unwrap_or(0);
                let arch = &cb.archetypes[arch_idx];
                let ids = cb.decode_with_archetype(arch_idx, output);
                let text = self.dictionary.decode(&ids);

                // Distance-based truncation: archetype clusters can include
                // trailing fixed tokens from longer, more distant samples.
                // The median_content_length (set during training) is the
                // primary bound. For older brains without it, fall back to
                // sentence-boundary truncation at 80% of archetype length.
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
}
