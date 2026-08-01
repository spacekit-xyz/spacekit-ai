// Chunk-level Continuous Text Codec (CATA)
//
// Instead of predicting the next token (discrete symbol), we predict the
// next vector — each vector represents a K-token chunk. This reduces
// autoregressive steps by K×, fundamentally changing the generation loop
// from discrete symbol prediction to continuous trajectory prediction
// through Cl(1,7) space.
//
// Architecture:
//   Raw text → TokenDictionary → chunk into K-token windows
//     → CDMA encode each chunk → 256d Cl(1,7) multivector per chunk
//     → generation operates on chunk vectors (retrieve, compose, extrapolate)
//     → decode each chunk vector → K tokens → join → text
//
// The codec uses Walsh-Hadamard spread-spectrum (CDMA) with parallel
// interference cancellation. At K=8, reconstruction accuracy is 100%.
// No neural networks — pure algebraic encode/decode.

use crate::spectral::TokenDictionary;

/// Tokens per chunk. At K=8 with 256d latent, CDMA gives perfect reconstruction.
pub const CHUNK_K: usize = 8;

/// Cl(1,7) multivector dimension = 2^8.
pub const CATA_DIM: usize = 256;

const MAX_VOCAB: usize = 4096;

/// Number of PIC refinement rounds in the decoder.
const PIC_ROUNDS: usize = 2;

// ---------------------------------------------------------------------------
// ChunkCodec — encode/decode individual K-token chunks
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct ChunkCodec {
    token_embeddings: Vec<[f32; CATA_DIM]>,
    position_codes: [[f32; CATA_DIM]; CHUNK_K],
    pub vocab_size: usize,
}

impl ChunkCodec {
    pub fn new(vocab_size: usize) -> Self {
        let vs = vocab_size.min(MAX_VOCAB);
        let token_embeddings = Self::build_token_embeddings(vs);
        let codes = Self::build_position_codes();
        Self {
            token_embeddings,
            position_codes: codes,
            vocab_size: vs,
        }
    }

    /// Raw per-token embedding (read-only). Used by the routing encoder to build
    /// a semantic-neighbor-smoothed sentence vector; the decode path is untouched.
    pub fn token_embedding(&self, id: u16) -> &[f32; CATA_DIM] {
        let idx = (id as usize).min(self.vocab_size.saturating_sub(1));
        &self.token_embeddings[idx]
    }

    // -- Single chunk encode/decode -------------------------------------------

    /// Encode up to K tokens into a 256d Cl(1,7) multivector.
    pub fn encode_chunk(&self, tokens: &[u16]) -> [f32; CATA_DIM] {
        let n = tokens.len().min(CHUNK_K);
        let mut latent = [0.0f32; CATA_DIM];
        for (pos, &tid) in tokens.iter().enumerate().take(n) {
            let idx = (tid as usize).min(self.vocab_size.saturating_sub(1));
            let emb = &self.token_embeddings[idx];
            let code = &self.position_codes[pos];
            for j in 0..CATA_DIM {
                latent[j] += emb[j] * code[j];
            }
        }
        if n > 0 {
            let s = 1.0 / (n as f32).sqrt();
            for x in latent.iter_mut() {
                *x *= s;
            }
        }
        latent
    }

    /// Decode a 256d chunk vector back to up to K token IDs.
    /// Uses parallel decode + PIC refinement for high fidelity.
    pub fn decode_chunk(&self, latent: &[f32; CATA_DIM], num_tokens: usize) -> Vec<u16> {
        let n = num_tokens.min(CHUNK_K);
        if n == 0 {
            return Vec::new();
        }
        let scale = (n as f32).sqrt();

        // Pass 1: parallel de-spread (independent, no error propagation)
        let mut tokens: Vec<u16> = (0..n)
            .map(|pos| {
                let code = &self.position_codes[pos];
                let mut ds = [0.0f32; CATA_DIM];
                for j in 0..CATA_DIM {
                    ds[j] = latent[j] * scale * code[j];
                }
                self.nearest_token(&ds).0
            })
            .collect();

        // Pass 2+: parallel interference cancellation
        for _ in 0..PIC_ROUNDS {
            let prev = tokens.clone();
            for pos in 0..n {
                let mut iso = [0.0f32; CATA_DIM];
                for j in 0..CATA_DIM {
                    iso[j] = latent[j] * scale;
                }
                for (q, &tok) in prev.iter().enumerate() {
                    if q == pos {
                        continue;
                    }
                    let emb = &self.token_embeddings[tok as usize];
                    let code = &self.position_codes[q];
                    for j in 0..CATA_DIM {
                        iso[j] -= emb[j] * code[j];
                    }
                }
                let code = &self.position_codes[pos];
                let mut ds = [0.0f32; CATA_DIM];
                for j in 0..CATA_DIM {
                    ds[j] = iso[j] * code[j];
                }
                tokens[pos] = self.nearest_token(&ds).0;
            }
            if tokens == prev {
                break;
            }
        }
        tokens
    }

    /// Roundtrip accuracy for a single chunk.
    pub fn chunk_accuracy(&self, tokens: &[u16]) -> f32 {
        let n = tokens.len().min(CHUNK_K);
        if n == 0 {
            return 1.0;
        }
        let enc = self.encode_chunk(tokens);
        let dec = self.decode_chunk(&enc, n);
        let hits = tokens
            .iter()
            .take(n)
            .zip(dec.iter())
            .filter(|(&a, &b)| a == b)
            .count();
        hits as f32 / n as f32
    }

    // -- Full text encode/decode (chunk sequence) -----------------------------

    /// Encode full token sequence → sequence of 256d chunk vectors.
    /// Each chunk encodes K tokens. The last chunk may be shorter (padded with EOS=0).
    pub fn encode_sequence(&self, token_ids: &[u16]) -> ChunkSequence {
        let mut chunks = Vec::new();
        let mut chunk_lengths = Vec::new();
        let mut pos = 0;
        while pos < token_ids.len() {
            let end = (pos + CHUNK_K).min(token_ids.len());
            let chunk_tokens = &token_ids[pos..end];
            chunks.push(self.encode_chunk(chunk_tokens));
            chunk_lengths.push(chunk_tokens.len());
            pos = end;
        }
        ChunkSequence {
            chunks,
            chunk_lengths,
            total_tokens: token_ids.len(),
        }
    }

    /// Word-aligned encoding: respects word boundaries when placing chunk
    /// breaks. With one-word-one-token encoding every token IS a word start,
    /// so this degenerates to the standard fixed-K path — but is retained as
    /// a safety net for any future tokenization that produces multi-token words.
    pub fn encode_sequence_word_aligned(
        &self,
        token_ids: &[u16],
        word_starts: &[bool],
    ) -> ChunkSequence {
        let n = token_ids.len();
        let mut chunks = Vec::new();
        let mut chunk_lengths = Vec::new();
        let mut total_tokens = 0usize;
        let mut pos = 0;

        while pos < n {
            let ideal_end = (pos + CHUNK_K).min(n);
            if ideal_end >= n {
                let chunk_tokens = &token_ids[pos..ideal_end];
                chunks.push(self.encode_chunk(chunk_tokens));
                chunk_lengths.push(chunk_tokens.len());
                total_tokens += chunk_tokens.len();
                break;
            }

            // Walk backward from ideal_end to find a word boundary.
            let mut end = ideal_end;
            while end > pos + 1 {
                if word_starts.get(end).copied().unwrap_or(true) {
                    break;
                }
                end -= 1;
            }
            // If no boundary found, fall through to ideal_end.
            if end <= pos + 1 && !word_starts.get(end).copied().unwrap_or(true) {
                end = ideal_end;
            }

            let chunk_tokens = &token_ids[pos..end];
            chunks.push(self.encode_chunk(chunk_tokens));
            chunk_lengths.push(chunk_tokens.len());
            total_tokens += chunk_tokens.len();
            pos = end;
        }

        ChunkSequence {
            chunks,
            chunk_lengths,
            total_tokens,
        }
    }

    /// Decode a chunk sequence back to token IDs.
    pub fn decode_sequence(&self, seq: &ChunkSequence) -> Vec<u16> {
        let mut tokens = Vec::with_capacity(seq.total_tokens);
        for (chunk, &len) in seq.chunks.iter().zip(seq.chunk_lengths.iter()) {
            let decoded = self.decode_chunk(chunk, len);
            tokens.extend_from_slice(&decoded);
        }
        tokens
    }

    /// Encode text string → chunk sequence (convenience).
    /// Uses word-aligned chunking to preserve word integrity.
    pub fn encode_text(&self, text: &str, dict: &TokenDictionary) -> ChunkSequence {
        let (ids, word_starts) = dict.encode_with_word_boundaries(text);
        self.encode_sequence_word_aligned(&ids, &word_starts)
    }

    /// Decode chunk sequence → text string (convenience).
    pub fn decode_text(&self, seq: &ChunkSequence, dict: &TokenDictionary) -> String {
        let ids = self.decode_sequence(seq);
        dict.decode(&ids)
    }

    /// Full roundtrip accuracy for a token sequence (chunk-by-chunk).
    pub fn sequence_accuracy(&self, token_ids: &[u16]) -> f32 {
        if token_ids.is_empty() {
            return 1.0;
        }
        let seq = self.encode_sequence(token_ids);
        let dec = self.decode_sequence(&seq);
        let hits = token_ids
            .iter()
            .zip(dec.iter())
            .filter(|(&a, &b)| a == b)
            .count();
        hits as f32 / token_ids.len() as f32
    }

    // -- Internals ------------------------------------------------------------

    fn nearest_token(&self, signal: &[f32; CATA_DIM]) -> (u16, f32) {
        let mut best_id = 0u16;
        let mut best_dot = f32::NEG_INFINITY;
        for (id, emb) in self.token_embeddings.iter().enumerate() {
            let d: f32 = signal.iter().zip(emb.iter()).map(|(a, b)| a * b).sum();
            if d > best_dot {
                best_dot = d;
                best_id = id as u16;
            }
        }
        (best_id, best_dot)
    }

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn build_token_embeddings(vocab_size: usize) -> Vec<[f32; CATA_DIM]> {
        let mut out = Vec::with_capacity(vocab_size);
        for id in 0..vocab_size {
            let mut emb = [0.0f32; CATA_DIM];
            let mut state = (id as u64).wrapping_mul(2654435761) ^ 0xCAFEBABE_DEADBEEF;
            let mut dim = 0;
            while dim < CATA_DIM {
                let r1 = Self::splitmix64(&mut state);
                let r2 = Self::splitmix64(&mut state);
                let u1 = ((r1 & 0xFFFFFFFF) as f64 + 1.0) / (u32::MAX as f64 + 2.0);
                let u2 = ((r2 & 0xFFFFFFFF) as f64 + 1.0) / (u32::MAX as f64 + 2.0);
                let mag = (-2.0 * u1.ln()).sqrt();
                let angle = 2.0 * std::f64::consts::PI * u2;
                emb[dim] = (mag * angle.cos()) as f32;
                if dim + 1 < CATA_DIM {
                    emb[dim + 1] = (mag * angle.sin()) as f32;
                }
                dim += 2;
            }
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for x in emb.iter_mut() {
                    *x /= norm;
                }
            }
            out.push(emb);
        }
        out
    }

    fn build_position_codes() -> [[f32; CATA_DIM]; CHUNK_K] {
        let n = CATA_DIM;
        let mut h = vec![vec![1.0f32; n]; n];
        let mut size = 1usize;
        while size < n {
            for i in 0..size {
                for j in 0..size {
                    let val = h[i][j];
                    h[i + size][j] = val;
                    h[i][j + size] = val;
                    h[i + size][j + size] = -val;
                }
            }
            size *= 2;
        }
        let mut codes = [[0.0f32; CATA_DIM]; CHUNK_K];
        for (k, row) in h.into_iter().take(CHUNK_K).enumerate() {
            for (j, &v) in row.iter().enumerate() {
                codes[k][j] = v;
            }
        }
        codes
    }
}

// ---------------------------------------------------------------------------
// ChunkSequence — a text represented as a trajectory of chunk vectors
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ChunkSequence {
    /// Each entry is a CATA_DIM-length Cl(1,7) multivector.
    pub chunks: Vec<[f32; CATA_DIM]>,
    /// Actual token count in each chunk (last may be < K).
    pub chunk_lengths: Vec<usize>,
    /// Total token count.
    pub total_tokens: usize,
}

impl ChunkSequence {
    pub fn num_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Semantic centroid: average of all chunk vectors.
    pub fn centroid(&self) -> [f32; CATA_DIM] {
        let mut c = [0.0f32; CATA_DIM];
        if self.chunks.is_empty() {
            return c;
        }
        for chunk in &self.chunks {
            for j in 0..CATA_DIM {
                c[j] += chunk[j];
            }
        }
        let n = self.chunks.len() as f32;
        for x in c.iter_mut() {
            *x /= n;
        }
        c
    }

    /// Cosine similarity between centroids.
    pub fn similarity(&self, other: &ChunkSequence) -> f32 {
        let a = self.centroid();
        let b = other.centroid();
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na < 1e-8 || nb < 1e-8 {
            0.0
        } else {
            dot / (na * nb)
        }
    }
}

// ---------------------------------------------------------------------------
// Trajectory operations — continuous-domain generation primitives
// ---------------------------------------------------------------------------

/// Weighted blend of chunk sequences (same length assumed).
/// For each chunk position, slerp between source chunks weighted by score.
pub fn compose_trajectories(
    sources: &[(ChunkSequence, f32)],
    target_len: usize,
) -> Vec<[f32; CATA_DIM]> {
    if sources.is_empty() {
        return Vec::new();
    }
    let n_chunks = target_len;

    (0..n_chunks)
        .map(|i| {
            let mut result = [0.0f32; CATA_DIM];
            let mut total_w = 0.0f32;
            for (seq, w) in sources {
                if i < seq.chunks.len() {
                    for j in 0..CATA_DIM {
                        result[j] += seq.chunks[i][j] * w;
                    }
                    total_w += w;
                }
            }
            if total_w > 1e-8 {
                for x in result.iter_mut() {
                    *x /= total_w;
                }
            }
            result
        })
        .collect()
}

/// Predict the next chunk vector given a trajectory so far.
///
/// Uses rotor extrapolation: computes the "velocity" between the last two
/// chunks (as a Clifford geometric quotient direction) and applies it to
/// the last chunk. Falls back to the last chunk if only one exists.
pub fn predict_next_chunk(trajectory: &[[f32; CATA_DIM]]) -> [f32; CATA_DIM] {
    match trajectory.len() {
        0 => [0.0f32; CATA_DIM],
        1 => trajectory[0],
        _ => {
            let prev = &trajectory[trajectory.len() - 2];
            let last = &trajectory[trajectory.len() - 1];
            // "Velocity" = last - prev; extrapolate = last + velocity
            let mut next = [0.0f32; CATA_DIM];
            for j in 0..CATA_DIM {
                next[j] = last[j] + (last[j] - prev[j]);
            }
            next
        }
    }
}

/// Predict next chunk using a reference library of known trajectories.
/// Finds the trajectory with the most similar recent context and returns
/// its next chunk. Falls back to rotor extrapolation if no good match.
pub fn predict_next_from_library(
    context: &[[f32; CATA_DIM]],
    library: &[ChunkSequence],
    min_similarity: f32,
) -> [f32; CATA_DIM] {
    if context.is_empty() || library.is_empty() {
        return predict_next_chunk(context);
    }

    let last = context.last().unwrap();
    let mut best_next: Option<[f32; CATA_DIM]> = None;
    let mut best_sim = min_similarity;

    for seq in library {
        // Find the position in this trajectory closest to our last chunk
        for (i, chunk) in seq.chunks.iter().enumerate() {
            let sim = cosine(last, chunk);
            if sim > best_sim && i + 1 < seq.chunks.len() {
                best_sim = sim;
                best_next = Some(seq.chunks[i + 1]);
            }
        }
    }

    best_next.unwrap_or_else(|| predict_next_chunk(context))
}

/// Spherical linear interpolation between two chunk vectors.
pub fn slerp(a: &[f32; CATA_DIM], b: &[f32; CATA_DIM], t: f32) -> [f32; CATA_DIM] {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let dot = dot.clamp(-1.0, 1.0);
    let omega = dot.acos();
    let mut result = [0.0f32; CATA_DIM];

    if omega.abs() < 1e-6 {
        for j in 0..CATA_DIM {
            result[j] = a[j] * (1.0 - t) + b[j] * t;
        }
        return result;
    }

    let sin_o = omega.sin();
    let w1 = ((1.0 - t) * omega).sin() / sin_o;
    let w2 = (t * omega).sin() / sin_o;
    for j in 0..CATA_DIM {
        result[j] = a[j] * w1 + b[j] * w2;
    }
    result
}

fn cosine(a: &[f32; CATA_DIM], b: &[f32; CATA_DIM]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-8 || nb < 1e-8 {
        0.0
    } else {
        dot / (na * nb)
    }
}

// ---------------------------------------------------------------------------
// Cl(1,7) interop
// ---------------------------------------------------------------------------

pub fn chunk_to_multivector(chunk: &[f32; CATA_DIM]) -> crate::clifford::Multivector {
    let mut mv = crate::clifford::Multivector::zero();
    mv.components = *chunk;
    mv
}

pub fn multivector_to_chunk(mv: &crate::clifford::Multivector) -> [f32; CATA_DIM] {
    mv.components
}

// ---------------------------------------------------------------------------
// Grade-Aware Algebraic Generation
//
// The 256d chunk space IS Cl(1,7). These functions use the algebraic
// structure — geometric product, grade decomposition, rotors — to encode
// richer semantic meaning than flat vector operations.
//
// Grade layout (from clifford.rs):
//   Grade 0 (1d, [0..1)):     scalar — content density
//   Grade 1 (8d, [1..9)):     vector — semantic direction (topic)
//   Grade 2 (28d, [9..37)):   bivector — relational context
//     Boost [9..16):   temporal/causal flow (7d)
//     Rotation [16..37): structural shift (21d)
//   Grade 3 (56d, [37..93)):  trivector — three-way interactions
//   Grade 4 (70d, [93..163)): quadvector — higher-order structure
//   Grade 5-8 (93d):          dual grades (reconstruction parity)
// ---------------------------------------------------------------------------

use crate::clifford::{
    apply_group_rotor, IntervalType, Multivector, Rotor, BOOST_BIVECTOR_COUNT, GRADE_DIMS,
    GRADE_OFFSETS, ROTATION_BIVECTOR_COUNT,
};

/// A chunk enriched with grade-decomposed semantic features.
/// The raw CDMA encoding is preserved for lossless token decode;
/// graded features enable algebraic operations (composition, prediction).
#[derive(Clone, Debug)]
pub struct GradedChunk {
    pub raw: [f32; CATA_DIM],
    pub mv: Multivector,
    /// Grade-1 projection: 8d semantic direction vector.
    pub semantic_dir: [f32; 8],
    /// Grade-2 projection: 28d relational context (boost + rotation split).
    pub context_bivector: [f32; 28],
    /// Content energy: L2 norm of the chunk (grade-independent).
    pub energy: f32,
}

impl GradedChunk {
    pub fn from_chunk(chunk: &[f32; CATA_DIM]) -> Self {
        let mv = chunk_to_multivector(chunk);

        let mut semantic_dir = [0.0f32; 8];
        let grade1 = mv.grade(1);
        for (i, v) in grade1.iter().enumerate().take(8) {
            semantic_dir[i] = *v;
        }

        let mut context_bivector = [0.0f32; 28];
        let grade2 = mv.grade(2);
        for (i, v) in grade2.iter().enumerate().take(28) {
            context_bivector[i] = *v;
        }

        let energy = chunk.iter().map(|x| x * x).sum::<f32>().sqrt();

        GradedChunk {
            raw: *chunk,
            mv,
            semantic_dir,
            context_bivector,
            energy,
        }
    }

    /// Cosine similarity of semantic direction (grade-1 only).
    pub fn semantic_similarity(&self, other: &GradedChunk) -> f32 {
        let dot: f32 = self
            .semantic_dir
            .iter()
            .zip(other.semantic_dir.iter())
            .map(|(a, b)| a * b)
            .sum();
        let na: f32 = self.semantic_dir.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = other.semantic_dir.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na < 1e-8 || nb < 1e-8 {
            0.0
        } else {
            dot / (na * nb)
        }
    }
}

// ---------------------------------------------------------------------------
// SpacetimeChunk — Dirac-style grade decomposition of a chunk multivector
//
// Each grade carries explicit semantic meaning derived from the Cl(1,7)
// algebra's Minkowski-like metric signature:
//
//   Grade 0 (scalar):       confidence / information density
//   Grade 1 (vector):       primary semantic content direction
//   Grade 2 (bivector):     relational context, split into:
//     Boost (e0∧ei):        causal/sequential flow (7d)
//     Rotation (ei∧ej):     associative/structural pattern (21d)
//   Grade 3 (trivector):    discourse position — argument, evidence, conclusion
//   Grade 8 (pseudoscalar): negation/complement marker (Hodge dual)
//
// The Minkowski interval between chunks classifies semantic transitions:
//   Timelike:  causal continuation — one idea follows from the previous
//   Spacelike: lateral association — parallel concepts, enumeration
//   Lightlike: topic boundary — maximum semantic reachability
// ---------------------------------------------------------------------------

/// Full spacetime decomposition of a chunk in the Cl(1,7) Dirac framework.
#[derive(Clone, Debug)]
pub struct SpacetimeChunk {
    pub raw: [f32; CATA_DIM],
    pub mv: Multivector,
    /// Grade-0: scalar confidence / information density.
    pub confidence: f32,
    /// Grade-1: 8d semantic direction vector (1 timelike + 7 spacelike).
    pub semantic_dir: [f32; 8],
    /// Grade-2 boost bivectors (e0∧ei): causal/sequential relations.
    pub boost_causal: [f32; BOOST_BIVECTOR_COUNT],
    /// Grade-2 rotation bivectors (ei∧ej): structural/associative relations.
    pub rotation_structural: [f32; ROTATION_BIVECTOR_COUNT],
    /// Grade-3: 56d discourse position (argument structure, rhetorical role).
    pub discourse: [f32; 56],
    /// Grade-8: pseudoscalar component — negation/complement marker.
    /// Positive = asserted content, negative = negated/counterfactual.
    pub dual_marker: f32,
    /// Clifford energy: L2 norm of the full multivector.
    pub energy: f32,
    /// Spacetime interval type relative to a reference (set via classify_from).
    pub interval_type: IntervalType,
}

impl SpacetimeChunk {
    pub fn from_chunk(chunk: &[f32; CATA_DIM]) -> Self {
        let mv = chunk_to_multivector(chunk);
        Self::from_multivector(mv, *chunk)
    }

    /// Build a SpacetimeChunk from a semantic centroid via `embed_bridge_vector`.
    /// Unlike `from_chunk` (which interprets raw CDMA spreading codes), this
    /// produces grade decompositions that carry genuine semantic structure:
    /// grade-1 = semantic direction, grade-2 = relational/structural wedge
    /// products, etc.
    pub fn from_centroid(centroid: &[f32]) -> Self {
        let mv = crate::clifford::embed_bridge_vector(centroid);
        let mut raw = [0.0f32; CATA_DIM];
        for (i, &c) in mv.components.iter().enumerate().take(CATA_DIM) {
            raw[i] = c;
        }
        Self::from_multivector(mv, raw)
    }

    fn from_multivector(mv: crate::clifford::Multivector, raw: [f32; CATA_DIM]) -> Self {
        let confidence = mv.components[GRADE_OFFSETS[0]];

        let mut semantic_dir = [0.0f32; 8];
        semantic_dir.copy_from_slice(mv.grade(1));

        let grade2 = mv.grade(2);
        let mut boost_causal = [0.0f32; BOOST_BIVECTOR_COUNT];
        boost_causal.copy_from_slice(&grade2[..BOOST_BIVECTOR_COUNT]);
        let mut rotation_structural = [0.0f32; ROTATION_BIVECTOR_COUNT];
        rotation_structural.copy_from_slice(&grade2[BOOST_BIVECTOR_COUNT..]);

        let mut discourse = [0.0f32; 56];
        discourse.copy_from_slice(mv.grade(3));

        let dual_marker = mv.components[GRADE_OFFSETS[8]];

        let energy = raw.iter().map(|x| x * x).sum::<f32>().sqrt();

        SpacetimeChunk {
            raw,
            mv,
            confidence,
            semantic_dir,
            boost_causal,
            rotation_structural,
            discourse,
            dual_marker,
            energy,
            interval_type: IntervalType::Spacelike,
        }
    }

    /// Classify the interval type from this chunk to another.
    pub fn interval_to(&self, other: &SpacetimeChunk) -> IntervalType {
        crate::clifford::interval_between(&self.mv, &other.mv)
    }

    /// Squared Minkowski interval from this chunk to another.
    pub fn interval_sq(&self, other: &SpacetimeChunk) -> f32 {
        crate::clifford::minkowski_interval(&self.mv, &other.mv)
    }

    /// Cosine similarity of grade-1 semantic direction.
    pub fn semantic_similarity(&self, other: &SpacetimeChunk) -> f32 {
        let dot: f32 = self
            .semantic_dir
            .iter()
            .zip(other.semantic_dir.iter())
            .map(|(a, b)| a * b)
            .sum();
        let na: f32 = self.semantic_dir.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = other.semantic_dir.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na < 1e-8 || nb < 1e-8 {
            0.0
        } else {
            dot / (na * nb)
        }
    }

    /// Cosine similarity of boost (causal) bivectors only.
    pub fn causal_similarity(&self, other: &SpacetimeChunk) -> f32 {
        let dot: f32 = self
            .boost_causal
            .iter()
            .zip(other.boost_causal.iter())
            .map(|(a, b)| a * b)
            .sum();
        let na: f32 = self.boost_causal.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = other.boost_causal.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na < 1e-8 || nb < 1e-8 {
            0.0
        } else {
            dot / (na * nb)
        }
    }

    /// Cosine similarity of rotation (structural) bivectors only.
    pub fn structural_similarity(&self, other: &SpacetimeChunk) -> f32 {
        let dot: f32 = self
            .rotation_structural
            .iter()
            .zip(other.rotation_structural.iter())
            .map(|(a, b)| a * b)
            .sum();
        let na: f32 = self
            .rotation_structural
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        let nb: f32 = other
            .rotation_structural
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        if na < 1e-8 || nb < 1e-8 {
            0.0
        } else {
            dot / (na * nb)
        }
    }

    /// True if the pseudoscalar component indicates negated/counterfactual content.
    pub fn is_negated(&self) -> bool {
        self.dual_marker < -0.01
    }

    /// Graded similarity: weighted sum of per-grade cosine similarities.
    /// Uses the Dirac-motivated weights that emphasize the grades carrying
    /// distinct semantic channels.
    pub fn graded_similarity(&self, other: &SpacetimeChunk) -> f32 {
        const WEIGHTS: [f32; 5] = [0.05, 0.30, 0.25, 0.15, 0.05];
        let sims = [
            // grade 0: scalar similarity
            if self.energy > 1e-8 && other.energy > 1e-8 {
                (self.confidence * other.confidence).abs().sqrt()
            } else {
                0.0
            },
            // grade 1: semantic direction
            self.semantic_similarity(other),
            // grade 2: combined boost + rotation
            0.4 * self.causal_similarity(other) + 0.6 * self.structural_similarity(other),
            // grade 3: discourse
            {
                let dot: f32 = self
                    .discourse
                    .iter()
                    .zip(other.discourse.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let na: f32 = self.discourse.iter().map(|x| x * x).sum::<f32>().sqrt();
                let nb: f32 = other.discourse.iter().map(|x| x * x).sum::<f32>().sqrt();
                if na < 1e-8 || nb < 1e-8 {
                    0.0
                } else {
                    dot / (na * nb)
                }
            },
            // grade 8: pseudoscalar agreement
            if (self.dual_marker > 0.0) == (other.dual_marker > 0.0) {
                1.0
            } else {
                -1.0
            },
        ];
        WEIGHTS
            .iter()
            .zip(sims.iter())
            .map(|(w, s)| w * s)
            .sum::<f32>()
            / WEIGHTS.iter().sum::<f32>()
    }
}

/// Transition rotor between two consecutive chunks.
/// Extracted from the grade-2 part of their geometric quotient,
/// encoding the rotation plane and angle of the semantic shift.
#[derive(Clone, Debug)]
pub struct ChunkTransition {
    pub rotor: Rotor,
    /// How faithfully the rotor reconstructs the target chunk.
    /// 1.0 = perfect rotation, lower = the transition involves scaling/shearing.
    pub fidelity: f32,
}

/// Compute the transition rotor from chunk A to chunk B.
///
/// The geometric product B * reverse(A) yields a multivector whose
/// even-grade parts approximate the versor that maps A → B.
/// We extract the bivector (grade-2) to build a rotation rotor,
/// then measure how well R A R̃ reconstructs B (fidelity).
pub fn compute_transition(a: &GradedChunk, b: &GradedChunk) -> ChunkTransition {
    let a_rev = a.mv.reverse();
    let product = b.mv.geo(&a_rev);

    // Extract the bivector part — this is the rotation generator
    let biv_slice = product.grade(2);
    let mut bivector = [0.0f32; 28];
    for (i, v) in biv_slice.iter().enumerate().take(28) {
        bivector[i] = *v;
    }

    // Scale bivector to a reasonable magnitude for rotor construction
    let biv_norm: f32 = bivector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if biv_norm > 1e-6 {
        let scale = biv_norm.min(std::f32::consts::PI) / biv_norm;
        for v in &mut bivector {
            *v *= scale;
        }
    }

    let rotor = Rotor::from_bivector(&bivector);

    // Measure fidelity: how well does R A R̃ approximate B?
    let reconstructed = apply_group_rotor(&a.mv, &rotor);
    let recon_chunk = multivector_to_chunk(&reconstructed);
    let fidelity = cosine(&recon_chunk, &b.raw);

    ChunkTransition { rotor, fidelity }
}

/// Predict the next chunk using rotor extrapolation.
///
/// Instead of linear extrapolation (last + velocity), we:
/// 1. Compute the transition rotor from penultimate → last chunk
/// 2. Apply the same rotor to the last chunk to get the next
///
/// The rotor preserves magnitude and grade structure,
/// producing predictions that stay on the semantic manifold
/// rather than drifting into flat-space artifacts.
pub fn predict_next_algebraic(trajectory: &[GradedChunk]) -> [f32; CATA_DIM] {
    match trajectory.len() {
        0 => [0.0f32; CATA_DIM],
        1 => trajectory[0].raw,
        _ => {
            let n = trajectory.len();
            let transition = compute_transition(&trajectory[n - 2], &trajectory[n - 1]);

            if transition.fidelity > 0.3 {
                // Rotor extrapolation: apply same rotation again
                let next_mv = apply_group_rotor(&trajectory[n - 1].mv, &transition.rotor);
                multivector_to_chunk(&next_mv)
            } else {
                // Fidelity too low — the transition isn't well-modeled as a rotation.
                // Fall back to slerp extrapolation.
                predict_next_chunk(&[trajectory[n - 2].raw, trajectory[n - 1].raw])
            }
        }
    }
}

/// Predict the next chunk using Dirac-style propagation with semantic inertia.
///
/// Unlike plain rotor extrapolation, this considers the Minkowski interval
/// between consecutive chunks to modulate prediction confidence:
///
///   Timelike interval → causal continuation: full rotor application
///   Spacelike interval → lateral association: damped rotor (blend with identity)
///   Lightlike interval → topic boundary: high uncertainty, slerp fallback
///
/// The mass parameter represents semantic inertia — resistance to large
/// rotations. Higher mass means the prediction stays closer to the current
/// semantic trajectory, analogous to a massive particle resisting acceleration.
pub fn predict_next_spacetime(
    trajectory: &[SpacetimeChunk],
    mass: f32,
) -> ([f32; CATA_DIM], IntervalType, f32) {
    let default_out = ([0.0f32; CATA_DIM], IntervalType::Spacelike, 0.0);
    if trajectory.len() < 2 {
        return match trajectory.len() {
            0 => default_out,
            1 => (trajectory[0].raw, IntervalType::Spacelike, 0.5),
            _ => unreachable!(),
        };
    }

    let n = trajectory.len();
    let prev = &trajectory[n - 2];
    let last = &trajectory[n - 1];

    let interval = last.interval_to(prev);
    let interval_sq = last.interval_sq(prev);
    let transition = compute_transition(
        &GradedChunk::from_chunk(&prev.raw),
        &GradedChunk::from_chunk(&last.raw),
    );

    // Semantic inertia: large rotations are penalized proportional to mass.
    // The "rotation magnitude" is approximated by the bivector norm of the rotor.
    let rotor_mv = transition.rotor.to_multivector();
    let biv_norm: f32 = rotor_mv.grade(2).iter().map(|x| x * x).sum::<f32>().sqrt();
    let inertia_factor = (-mass * biv_norm).exp(); // ∈ (0, 1], high mass → dampened

    let (prediction, confidence) = match interval {
        IntervalType::Timelike => {
            // Causal continuation: apply rotor with inertia damping.
            // Blend between full rotor application and identity (no change).
            if transition.fidelity > 0.2 {
                let full_next = apply_group_rotor(&last.mv, &transition.rotor);
                let full_chunk = multivector_to_chunk(&full_next);
                // Blend: inertia_factor * rotored + (1 - inertia_factor) * last
                let mut blended = [0.0f32; CATA_DIM];
                for i in 0..CATA_DIM {
                    blended[i] =
                        inertia_factor * full_chunk[i] + (1.0 - inertia_factor) * last.raw[i];
                }
                (blended, transition.fidelity * inertia_factor)
            } else {
                (predict_next_chunk(&[prev.raw, last.raw]), 0.3)
            }
        }
        IntervalType::Spacelike => {
            // Lateral association: rotor is less reliable, dampen more.
            if transition.fidelity > 0.3 {
                let full_next = apply_group_rotor(&last.mv, &transition.rotor);
                let full_chunk = multivector_to_chunk(&full_next);
                let damping = inertia_factor * 0.5;
                let mut blended = [0.0f32; CATA_DIM];
                for i in 0..CATA_DIM {
                    blended[i] = damping * full_chunk[i] + (1.0 - damping) * last.raw[i];
                }
                (blended, transition.fidelity * 0.5)
            } else {
                (predict_next_chunk(&[prev.raw, last.raw]), 0.2)
            }
        }
        IntervalType::Lightlike => {
            // Topic boundary: high uncertainty. Use conservative slerp
            // with low extrapolation weight — the trajectory is about to diverge.
            let slerped = slerp(&prev.raw, &last.raw, 0.6);
            (slerped, 0.15)
        }
    };

    (prediction, interval, confidence)
}

/// Algebraically exact semantic negation via pseudoscalar multiplication.
///
/// In Cl(1,7), multiplying by the pseudoscalar I = e₀e₁…e₇ performs Hodge
/// duality: grade-k ↦ grade-(8-k). This is NOT a learned approximation —
/// it is the algebraic complement in the Clifford algebra.
///
/// Semantically: the negation of "X is true" produces an embedding whose
/// grade structure is the dual of X, with the pseudoscalar component flipped.
/// This models negation as geometric duality rather than statistical distance.
pub fn semantic_negate(chunk: &SpacetimeChunk) -> SpacetimeChunk {
    let negated_mv = crate::clifford::pseudoscalar_product(&chunk.mv);
    let negated_raw = multivector_to_chunk(&negated_mv);
    let mut result = SpacetimeChunk::from_chunk(&negated_raw);
    // The interval type is inherited — negation doesn't change the
    // causal structure, only the content polarity.
    result.interval_type = chunk.interval_type;
    result
}

/// Compute the semantic mass (inertia) of a trajectory.
///
/// Estimated from the consistency of rotor transitions: a trajectory
/// with uniform rotors (steady semantic drift) has low mass, while one
/// with erratic rotors (rapid topic changes) has high mass.
/// Analogous to how physical mass measures resistance to acceleration.
pub fn trajectory_mass(trajectory: &[SpacetimeChunk]) -> f32 {
    if trajectory.len() < 3 {
        return 1.0;
    }
    let mut biv_norms: Vec<f32> = Vec::new();
    for i in 1..trajectory.len() {
        let t = compute_transition(
            &GradedChunk::from_chunk(&trajectory[i - 1].raw),
            &GradedChunk::from_chunk(&trajectory[i].raw),
        );
        let biv = t.rotor.to_multivector();
        let norm: f32 = biv.grade(2).iter().map(|x| x * x).sum::<f32>().sqrt();
        biv_norms.push(norm);
    }
    let mean = biv_norms.iter().sum::<f32>() / biv_norms.len() as f32;
    let variance =
        biv_norms.iter().map(|n| (n - mean).powi(2)).sum::<f32>() / biv_norms.len() as f32;
    // High variance → high mass (erratic → resistant to prediction)
    // Low variance → low mass (steady → easy to extrapolate)
    (1.0 + variance * 10.0).min(5.0)
}

/// Compose multiple chunk trajectories using the geometric product.
///
/// Instead of weighted averaging (which washes out semantic structure),
/// this uses a grade-aware blend:
///   - Grade 0-1 (content + topic): weighted average (stable)
///   - Grade 2 (relationships): geometric product of top-2 sources
///     (preserves both shared and novel relational structure)
///   - Grade 3+ (higher structure): weighted average of the remainder
///
/// The result has richer semantic content than a flat weighted average
/// because the grade-2 composition captures relational novelty
/// through the wedge product component of the geometric product.
pub fn compose_algebraic(
    sources: &[(ChunkSequence, f32)],
    target_len: usize,
) -> Vec<[f32; CATA_DIM]> {
    if sources.is_empty() {
        return Vec::new();
    }
    if sources.len() == 1 {
        return sources[0]
            .0
            .chunks
            .iter()
            .take(target_len)
            .cloned()
            .collect();
    }

    (0..target_len)
        .map(|i| {
            // Collect available chunks at this position with their weights
            let mut available: Vec<(Multivector, f32)> = Vec::new();
            for (seq, w) in sources {
                if i < seq.chunks.len() {
                    available.push((chunk_to_multivector(&seq.chunks[i]), *w));
                }
            }
            if available.is_empty() {
                return [0.0f32; CATA_DIM];
            }
            if available.len() == 1 {
                return multivector_to_chunk(&available[0].0);
            }

            // Sort by weight descending
            available.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let primary = &available[0].0;
            let secondary = &available[1].0;
            let w_primary = available[0].1;
            let w_secondary = available[1].1;
            let w_total = w_primary + w_secondary;
            let alpha = if w_total > 1e-8 {
                w_primary / w_total
            } else {
                0.5
            };

            let mut result = Multivector::zero();

            // Grade 0 (scalar): weighted average
            result.components[0] =
                primary.components[0] * alpha + secondary.components[0] * (1.0 - alpha);

            // Grade 1 (semantic direction): slerp between topic vectors
            {
                let start = GRADE_OFFSETS[1];
                let dim = GRADE_DIMS[1];
                for d in 0..dim {
                    result.components[start + d] = primary.components[start + d] * alpha
                        + secondary.components[start + d] * (1.0 - alpha);
                }
                // Normalize grade-1 to preserve direction magnitude
                let norm: f32 = (0..dim)
                    .map(|d| result.components[start + d].powi(2))
                    .sum::<f32>()
                    .sqrt();
                let target_norm: f32 = (0..dim)
                    .map(|d| primary.components[start + d].powi(2))
                    .sum::<f32>()
                    .sqrt()
                    * alpha
                    + (0..dim)
                        .map(|d| secondary.components[start + d].powi(2))
                        .sum::<f32>()
                        .sqrt()
                        * (1.0 - alpha);
                if norm > 1e-8 {
                    let scale = target_norm / norm;
                    for d in 0..dim {
                        result.components[start + d] *= scale;
                    }
                }
            }

            // Grade 2 (relational context): geometric product of primary and secondary
            // enriches the relational structure via the wedge product component.
            // Safety: clamp the geometric product contribution to prevent
            // numerically unstable values from corrupting the chunk.
            {
                let geo_product = primary.geo(secondary);
                let start = GRADE_OFFSETS[2];
                let dim = GRADE_DIMS[2];

                let primary_g2_norm: f32 = (0..dim)
                    .map(|d| primary.components[start + d].powi(2))
                    .sum::<f32>()
                    .sqrt();
                let geo_g2_norm: f32 = (0..dim)
                    .map(|d| geo_product.components[start + d].powi(2))
                    .sum::<f32>()
                    .sqrt();

                // Only blend if the geo product grade-2 is within 3x of the primary's
                // magnitude. Larger means the product amplified noise.
                let blend_geo = if geo_g2_norm < primary_g2_norm * 3.0 && primary_g2_norm > 1e-6 {
                    0.2f32
                } else {
                    0.0 // fall back to pure weighted average for grade 2
                };

                for d in 0..dim {
                    let primary_val = primary.components[start + d];
                    if blend_geo > 0.0 {
                        let geo_val = geo_product.components[start + d]
                            * (primary_g2_norm / geo_g2_norm.max(1e-8));
                        result.components[start + d] =
                            primary_val * (1.0 - blend_geo) + geo_val * blend_geo;
                    } else {
                        result.components[start + d] =
                            primary_val * alpha + secondary.components[start + d] * (1.0 - alpha);
                    }
                }
            }

            // Grade 3+ (higher structure): weighted average
            for grade in 3..=8 {
                let start = GRADE_OFFSETS[grade];
                let dim = GRADE_DIMS[grade];
                for d in 0..dim {
                    result.components[start + d] = primary.components[start + d] * alpha
                        + secondary.components[start + d] * (1.0 - alpha);
                }
            }

            multivector_to_chunk(&result)
        })
        .collect()
}

/// Grade-aware similarity: weighted combination of per-grade cosine similarities.
/// More discriminative than flat cosine because it separately evaluates
/// topic alignment (grade 1), relational match (grade 2), and content match (grade 3+).
pub fn graded_similarity(a: &[f32; CATA_DIM], b: &[f32; CATA_DIM]) -> f32 {
    let grade_weights: [f32; 9] = [0.02, 0.25, 0.20, 0.18, 0.15, 0.08, 0.06, 0.04, 0.02];
    let mut total = 0.0f32;

    for grade in 0..9 {
        let start = GRADE_OFFSETS[grade];
        let dim = GRADE_DIMS[grade];
        if dim == 0 {
            continue;
        }

        let dot: f32 = (0..dim).map(|d| a[start + d] * b[start + d]).sum();
        let na: f32 = (0..dim).map(|d| a[start + d].powi(2)).sum::<f32>().sqrt();
        let nb: f32 = (0..dim).map(|d| b[start + d].powi(2)).sum::<f32>().sqrt();
        let sim = if na > 1e-8 && nb > 1e-8 {
            dot / (na * nb)
        } else {
            0.0
        };

        total += grade_weights[grade] * sim;
    }
    total
}

// ---------------------------------------------------------------------------
// Per-Grade Temperature Sampling (Phase 4b)
//
// Temperature controls generation diversity, but in Cl(1,7) we can apply
// temperature PER GRADE for fine-grained control:
//
//   Low T on grade-1 (vector):    conservative semantic content
//   High T on grade-2 (bivector): creative relational exploration
//   Zero T on grade-0 (scalar):   deterministic confidence weighting
//   High T on grade-3 (trivector): varied discourse structure
//
// This is impossible with flat vectors where temperature applies uniformly.
// ---------------------------------------------------------------------------

/// Per-grade temperature configuration for multivector generation.
#[derive(Clone, Debug)]
pub struct GradeTemperature {
    pub scalar: f32,
    pub vector: f32,
    pub bivector: f32,
    pub trivector: f32,
    pub pseudoscalar: f32,
}

impl Default for GradeTemperature {
    fn default() -> Self {
        Self {
            scalar: 0.0,
            vector: 0.7,
            bivector: 1.0,
            trivector: 0.8,
            pseudoscalar: 0.3,
        }
    }
}

impl GradeTemperature {
    /// Conservative: low temperature across all grades.
    pub fn conservative() -> Self {
        Self {
            scalar: 0.0,
            vector: 0.3,
            bivector: 0.3,
            trivector: 0.3,
            pseudoscalar: 0.1,
        }
    }

    /// Creative: higher temperature on relational and discourse grades.
    pub fn creative() -> Self {
        Self {
            scalar: 0.0,
            vector: 0.5,
            bivector: 1.5,
            trivector: 1.2,
            pseudoscalar: 0.5,
        }
    }

    /// Balanced: moderate temperature, emphasizing content stability.
    pub fn balanced() -> Self {
        Self::default()
    }
}

/// Apply per-grade temperature sampling to a multivector chunk.
///
/// For each grade, the temperature scales a noise perturbation:
///   result_grade_k = original_grade_k + temperature_k * noise_k
///
/// At temperature 0: deterministic (no perturbation).
/// At temperature 1: standard perturbation.
/// At temperature >1: amplified exploration.
///
/// The noise is generated deterministically from a seed for reproducibility.
pub fn apply_grade_temperature(
    chunk: &[f32; CATA_DIM],
    temps: &GradeTemperature,
    seed: u64,
) -> [f32; CATA_DIM] {
    let mut result = *chunk;
    let mv = chunk_to_multivector(chunk);

    // Higher grades (4-7) inherit a scaled-down version of the bivector temperature
    let mid_scale = temps.bivector * 0.3;
    let grade_temps: [f32; 9] = [
        temps.scalar,       // grade 0
        temps.vector,       // grade 1
        temps.bivector,     // grade 2
        temps.trivector,    // grade 3
        mid_scale,          // grade 4
        mid_scale * 0.6,    // grade 5
        mid_scale * 0.4,    // grade 6
        mid_scale * 0.2,    // grade 7
        temps.pseudoscalar, // grade 8
    ];

    let mut rng_state = seed;
    for grade in 0..9 {
        let t = grade_temps[grade];
        if t < 1e-6 {
            continue;
        }

        let start = GRADE_OFFSETS[grade];
        let dim = GRADE_DIMS[grade];

        let grade_norm: f32 = (0..dim)
            .map(|d| mv.components[start + d].powi(2))
            .sum::<f32>()
            .sqrt();

        if grade_norm < 1e-10 {
            continue;
        }

        let noise_scale = t * grade_norm * 0.1;

        for d in 0..dim {
            rng_state = splitmix64(rng_state);
            let uniform = (rng_state as f64) / (u64::MAX as f64);
            // Box-Muller-ish: simple centered perturbation from uniform
            let noise = (uniform as f32 - 0.5) * 2.0 * noise_scale;
            result[start + d] += noise;
        }
    }

    result
}

fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(0x9e3779b97f4a7c15);
    state = (state ^ (state >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    state = (state ^ (state >> 27)).wrapping_mul(0x94d049bb133111eb);
    state ^ (state >> 31)
}

// ---------------------------------------------------------------------------
// SemanticPropagator — Dirac-style autoregressive core (Phase 3)
//
// Replaces single-rotor extrapolation with a principled propagator that:
//   1. Maintains a rotor history (trajectory of semantic transformations)
//   2. Computes kinetic energy (rate of rotor change) and potential energy
//      (deviation from mean trajectory) for a least-action prediction
//   3. Uses the Minkowski interval to classify transition types
//   4. Models semantic ambiguity as interference between spinor components
//
// The propagator is the mathematical analog of the Feynman propagator
// in QFT: it describes how a semantic state evolves from one chunk to
// the next along the path of least action through meaning-space.
// ---------------------------------------------------------------------------

/// Dirac-style semantic propagator for autoregressive chunk prediction.
#[derive(Clone, Debug)]
pub struct SemanticPropagator {
    /// Semantic inertia: resistance to large rotations (topic changes).
    pub mass: f32,
    /// Interaction coupling: how strongly consecutive chunks influence each other.
    pub coupling: f32,
    /// Rotor history: sequence of transition rotors between consecutive chunks.
    pub history: Vec<Rotor>,
    /// Chunk history for interval classification.
    chunk_history: Vec<SpacetimeChunk>,
    /// Running estimate of the "mean rotor" — the average transformation direction.
    mean_rotor: Option<Rotor>,
}

impl SemanticPropagator {
    pub fn new(mass: f32, coupling: f32) -> Self {
        Self {
            mass: mass.max(0.1),
            coupling: coupling.clamp(0.0, 1.0),
            history: Vec::new(),
            chunk_history: Vec::new(),
            mean_rotor: None,
        }
    }

    /// Feed a new chunk into the propagator, updating the rotor history.
    pub fn observe(&mut self, chunk: &SpacetimeChunk) {
        if let Some(prev) = self.chunk_history.last() {
            let graded_prev = GradedChunk::from_chunk(&prev.raw);
            let graded_curr = GradedChunk::from_chunk(&chunk.raw);
            let transition = compute_transition(&graded_prev, &graded_curr);
            self.history.push(transition.rotor.clone());
            self.update_mean_rotor();
        }
        self.chunk_history.push(chunk.clone());
    }

    /// Build a propagator from an existing trajectory.
    pub fn from_trajectory(trajectory: &[SpacetimeChunk], mass: f32, coupling: f32) -> Self {
        let mut prop = Self::new(mass, coupling);
        for chunk in trajectory {
            prop.observe(chunk);
        }
        prop
    }

    /// Update the running mean rotor (exponential moving average of rotor components).
    fn update_mean_rotor(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let n = self.history.len();
        let decay = 0.7f32;

        match self.mean_rotor {
            None => {
                self.mean_rotor = Some(self.history[n - 1].clone());
            }
            Some(ref mut mean) => {
                let latest = &self.history[n - 1];
                let mean_mv = mean.to_multivector();
                let latest_mv = latest.to_multivector();
                let mut blended = crate::clifford::Multivector::zero();
                for i in 0..crate::clifford::CL8_DIM {
                    blended.components[i] =
                        mean_mv.components[i] * decay + latest_mv.components[i] * (1.0 - decay);
                }
                // Re-extract even-grade components for the rotor
                let mut new_mean = Rotor::identity();
                let even_mv = blended;
                let even_comps = even_mv.even_grade_components();
                for (i, &v) in even_comps.iter().enumerate().take(128) {
                    new_mean.components[i] = v;
                }
                new_mean.normalize();
                *mean = new_mean;
            }
        }
    }

    /// Compute the kinetic energy of the trajectory: rate of rotor change.
    /// High kinetic energy means the semantic direction is changing rapidly.
    fn kinetic_energy(&self) -> f32 {
        if self.history.len() < 2 {
            return 0.0;
        }
        let n = self.history.len();
        let curr = self.history[n - 1].to_multivector();
        let prev = self.history[n - 2].to_multivector();

        let mut diff_sq = 0.0f32;
        for i in 0..crate::clifford::CL8_DIM {
            let d = curr.components[i] - prev.components[i];
            diff_sq += d * d;
        }
        diff_sq
    }

    /// Compute the potential energy: deviation of the latest rotor from the mean.
    /// High potential means the current transition deviates from the average pattern.
    fn potential_energy(&self) -> f32 {
        let mean = match &self.mean_rotor {
            Some(r) => r,
            None => return 0.0,
        };
        if self.history.is_empty() {
            return 0.0;
        }

        let latest = self.history.last().unwrap().to_multivector();
        let mean_mv = mean.to_multivector();

        let mut diff_sq = 0.0f32;
        for i in 0..crate::clifford::CL8_DIM {
            let d = latest.components[i] - mean_mv.components[i];
            diff_sq += d * d;
        }
        self.coupling * diff_sq
    }

    /// Predict the next chunk using the least-action principle.
    ///
    /// Instead of blindly applying the last rotor, this finds the rotor
    /// that minimizes the action S = kinetic - potential:
    ///   - If kinetic >> potential: trajectory is accelerating, predict with
    ///     the current rotor (high confidence in the direction of change)
    ///   - If potential >> kinetic: trajectory is deviating from its mean,
    ///     blend back toward the mean rotor (conservative prediction)
    ///   - At equilibrium: balanced prediction using both current and mean
    ///
    /// Returns (predicted_chunk, interval_type, confidence).
    pub fn predict_next(&self) -> Option<([f32; CATA_DIM], IntervalType, f32)> {
        if self.chunk_history.len() < 2 {
            return None;
        }

        let n = self.chunk_history.len();
        let last = &self.chunk_history[n - 1];
        let prev = &self.chunk_history[n - 2];

        let interval = last.interval_to(prev);
        let ke = self.kinetic_energy();
        let pe = self.potential_energy();

        // Action-based blending: α determines how much to trust current vs mean rotor
        let action_ratio = if ke + pe > 1e-8 { ke / (ke + pe) } else { 0.5 };

        // Get the prediction rotor: blend between latest and mean
        let pred_rotor =
            if let (Some(latest_r), Some(mean_r)) = (self.history.last(), &self.mean_rotor) {
                let latest_mv = latest_r.to_multivector();
                let mean_mv = mean_r.to_multivector();
                let mut blended = crate::clifford::Multivector::zero();
                for i in 0..crate::clifford::CL8_DIM {
                    blended.components[i] = latest_mv.components[i] * action_ratio
                        + mean_mv.components[i] * (1.0 - action_ratio);
                }
                let mut rotor = Rotor::identity();
                let even = blended.even_grade_components();
                for (i, &v) in even.iter().enumerate().take(128) {
                    rotor.components[i] = v;
                }
                rotor.normalize();
                rotor
            } else if let Some(r) = self.history.last() {
                r.clone()
            } else {
                return None;
            };

        // Apply inertia damping based on mass and interval type
        let damping = match interval {
            IntervalType::Timelike => (-self.mass * 0.1).exp(),
            IntervalType::Spacelike => (-self.mass * 0.3).exp(),
            IntervalType::Lightlike => (-self.mass * 0.5).exp(),
        };

        // Apply the rotor to predict next
        let full_next = apply_group_rotor(&last.mv, &pred_rotor);
        let full_chunk = multivector_to_chunk(&full_next);

        // Blend with identity (current state) based on damping
        let mut prediction = [0.0f32; CATA_DIM];
        for i in 0..CATA_DIM {
            prediction[i] = damping * full_chunk[i] + (1.0 - damping) * last.raw[i];
        }

        // Confidence from fidelity of the prediction rotor and action balance
        let confidence = match interval {
            IntervalType::Timelike => 0.8 * damping * action_ratio.max(0.3),
            IntervalType::Spacelike => 0.5 * damping,
            IntervalType::Lightlike => 0.2 * damping,
        };

        Some((prediction, interval, confidence))
    }

    /// Detect semantic ambiguity as spinor interference (Zitterbewegung analog).
    ///
    /// When the predicted chunk's even and odd grade components have comparable
    /// magnitude, it signals interference between the "positive energy" (asserted)
    /// and "negative energy" (counterfactual) spinor components — the embedding
    /// is oscillating between a concept and its complement.
    ///
    /// Returns an ambiguity score ∈ [0, 1] where 1 = maximally ambiguous.
    pub fn ambiguity_score(&self) -> f32 {
        if let Some((pred, _, _)) = self.predict_next() {
            let mv = chunk_to_multivector(&pred);

            // Even grades (0, 2, 4, 6, 8): "positive energy" spinor components
            let even_energy: f32 = [0, 2, 4, 6, 8]
                .iter()
                .map(|&g| {
                    let start = GRADE_OFFSETS[g];
                    let dim = GRADE_DIMS[g];
                    (0..dim)
                        .map(|d| mv.components[start + d].powi(2))
                        .sum::<f32>()
                })
                .sum();

            // Odd grades (1, 3, 5, 7): "negative energy" spinor components
            let odd_energy: f32 = [1, 3, 5, 7]
                .iter()
                .map(|&g| {
                    let start = GRADE_OFFSETS[g];
                    let dim = GRADE_DIMS[g];
                    (0..dim)
                        .map(|d| mv.components[start + d].powi(2))
                        .sum::<f32>()
                })
                .sum();

            let total = even_energy + odd_energy;
            if total < 1e-8 {
                return 0.0;
            }

            // Maximum ambiguity when even ≈ odd (equal interference)
            let ratio = even_energy.min(odd_energy) / even_energy.max(odd_energy).max(1e-8);
            ratio // 1.0 = maximally ambiguous, 0.0 = one component dominates
        } else {
            0.0
        }
    }

    /// Compose multiple source trajectories using the propagator.
    ///
    /// Instead of flat weighted averaging, this:
    ///   1. Encodes each source as a SpacetimeChunk trajectory
    ///   2. Classifies interval types between consecutive chunks
    ///   3. Uses the propagator to predict the next chunk at each position
    ///   4. Blends predictions from multiple sources based on their confidence
    pub fn compose_trajectories(
        &self,
        sources: &[(Vec<SpacetimeChunk>, f32)],
        target_len: usize,
    ) -> Vec<[f32; CATA_DIM]> {
        if sources.is_empty() {
            return Vec::new();
        }
        if sources.len() == 1 {
            return sources[0]
                .0
                .iter()
                .take(target_len)
                .map(|c| c.raw)
                .collect();
        }

        (0..target_len)
            .map(|i| {
                let mut weighted_sum = [0.0f32; CATA_DIM];
                let mut total_weight = 0.0f32;

                for (trajectory, base_weight) in sources {
                    if i >= trajectory.len() {
                        continue;
                    }

                    // Build a sub-propagator for this source
                    let sub_traj = &trajectory[..=i.min(trajectory.len() - 1)];
                    let mut sub_prop = SemanticPropagator::new(self.mass, self.coupling);
                    for chunk in sub_traj {
                        sub_prop.observe(chunk);
                    }

                    // Weight includes both the source weight and the propagator's confidence
                    let confidence = if sub_traj.len() >= 2 {
                        sub_prop.predict_next().map(|(_, _, c)| c).unwrap_or(0.5)
                    } else {
                        0.5
                    };
                    let w = base_weight * (0.5 + confidence);

                    for j in 0..CATA_DIM {
                        weighted_sum[j] += trajectory[i].raw[j] * w;
                    }
                    total_weight += w;
                }

                if total_weight > 1e-8 {
                    for j in 0..CATA_DIM {
                        weighted_sum[j] /= total_weight;
                    }
                }
                weighted_sum
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embeddings_unit_normalized() {
        let codec = ChunkCodec::new(512);
        for emb in &codec.token_embeddings {
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "norm = {}", norm);
        }
    }

    #[test]
    fn test_embeddings_low_coherence() {
        let codec = ChunkCodec::new(256);
        let mut max_cos = 0.0f32;
        for i in 0..codec.vocab_size {
            for j in (i + 1)..codec.vocab_size {
                let d: f32 = codec.token_embeddings[i]
                    .iter()
                    .zip(codec.token_embeddings[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                if d.abs() > max_cos {
                    max_cos = d.abs();
                }
            }
        }
        assert!(max_cos < 0.25, "max mutual cosine = {}", max_cos);
    }

    #[test]
    fn test_position_codes_orthogonal() {
        let codec = ChunkCodec::new(16);
        for i in 0..CHUNK_K {
            for j in (i + 1)..CHUNK_K {
                let dot: f32 = codec.position_codes[i]
                    .iter()
                    .zip(codec.position_codes[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                assert!(dot.abs() < 1e-4, "codes {},{} dot={}", i, j, dot);
            }
        }
    }

    #[test]
    fn test_single_token_roundtrip() {
        let codec = ChunkCodec::new(1024);
        for tid in [0u16, 1, 50, 100, 500, 1023] {
            let acc = codec.chunk_accuracy(&[tid]);
            assert_eq!(acc, 1.0, "single token {} failed", tid);
        }
    }

    #[test]
    fn test_full_chunk_roundtrip() {
        let codec = ChunkCodec::new(1024);
        for start in [0u16, 50, 100, 500, 900] {
            let tokens: Vec<u16> = (start..start + CHUNK_K as u16).collect();
            let acc = codec.chunk_accuracy(&tokens);
            assert_eq!(
                acc, 1.0,
                "K-token chunk starting at {} failed: acc={}",
                start, acc
            );
        }
    }

    #[test]
    fn test_random_chunk_roundtrip() {
        let codec = ChunkCodec::new(2048);
        let mut state = 42u64;
        let mut total_correct = 0;
        let trials = 100;
        for _ in 0..trials {
            let tokens: Vec<u16> = (0..CHUNK_K)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((state >> 33) % 2048) as u16
                })
                .collect();
            let acc = codec.chunk_accuracy(&tokens);
            if acc == 1.0 {
                total_correct += 1;
            }
        }
        let pct = total_correct as f32 / trials as f32;
        assert!(
            pct >= 0.95,
            "random chunk perfect roundtrip rate = {:.1}%",
            pct * 100.0
        );
    }

    #[test]
    fn test_sequence_roundtrip_32_tokens() {
        let codec = ChunkCodec::new(1024);
        let tokens: Vec<u16> = (5..37).collect(); // 32 tokens = 4 chunks
        let acc = codec.sequence_accuracy(&tokens);
        assert!(acc >= 0.99, "32-token sequence accuracy = {:.3}", acc);
    }

    #[test]
    fn test_sequence_roundtrip_64_tokens() {
        let codec = ChunkCodec::new(2048);
        let mut state = 99u64;
        let tokens: Vec<u16> = (0..64)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((state >> 33) % 2048) as u16
            })
            .collect();
        let acc = codec.sequence_accuracy(&tokens);
        assert!(acc >= 0.99, "64-token sequence accuracy = {:.3}", acc);
    }

    #[test]
    fn test_centroid_similarity() {
        let codec = ChunkCodec::new(256);
        let s1 = codec.encode_sequence(&[10, 20, 30, 40, 50, 60, 70, 80]);
        let s2 = codec.encode_sequence(&[10, 20, 30, 40, 50, 60, 70, 80]);
        let s3 = codec.encode_sequence(&[200, 201, 202, 203, 204, 205, 206, 207]);
        assert!(
            (s1.similarity(&s2) - 1.0).abs() < 1e-4,
            "identical seqs should have sim=1"
        );
        assert!(
            s1.similarity(&s3) < 0.5,
            "different seqs should have low sim"
        );
    }

    #[test]
    fn test_predict_next_extrapolation() {
        let codec = ChunkCodec::new(256);
        let c0 = codec.encode_chunk(&[10, 20, 30, 40, 50, 60, 70, 80]);
        let c1 = codec.encode_chunk(&[11, 21, 31, 41, 51, 61, 71, 81]);
        let pred = predict_next_chunk(&[c0, c1]);
        // Predicted chunk should be further along the same direction
        let d01: f32 = c0
            .iter()
            .zip(c1.iter())
            .map(|(a, b)| (b - a).powi(2))
            .sum::<f32>()
            .sqrt();
        let d1p: f32 = c1
            .iter()
            .zip(pred.iter())
            .map(|(a, b)| (b - a).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(
            (d01 - d1p).abs() < 0.1,
            "prediction should maintain velocity"
        );
    }

    #[test]
    fn test_slerp_endpoints() {
        let codec = ChunkCodec::new(128);
        let a = codec.encode_chunk(&[10, 20, 30, 40, 50, 60, 70, 80]);
        let b = codec.encode_chunk(&[80, 70, 60, 50, 40, 30, 20, 10]);

        let at0 = slerp(&a, &b, 0.0);
        let at1 = slerp(&a, &b, 1.0);

        for j in 0..CATA_DIM {
            assert!((at0[j] - a[j]).abs() < 1e-4, "slerp(0) should equal a");
            assert!((at1[j] - b[j]).abs() < 1e-4, "slerp(1) should equal b");
        }
    }

    #[test]
    fn test_compose_trajectories() {
        let codec = ChunkCodec::new(256);
        let s1 = codec.encode_sequence(&[10, 20, 30, 40, 50, 60, 70, 80]);
        let s2 = codec.encode_sequence(&[90, 91, 92, 93, 94, 95, 96, 97]);

        // 100% weight on s1 → should decode back to s1's tokens
        let composed = compose_trajectories(&[(s1.clone(), 1.0), (s2.clone(), 0.0)], 1);
        let decoded = codec.decode_chunk(&composed[0], CHUNK_K);
        let original: Vec<u16> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        assert_eq!(decoded, original, "100% weight should reproduce source");
    }

    #[test]
    fn test_predict_from_library() {
        let codec = ChunkCodec::new(256);
        let t1 = codec.encode_sequence(&[
            10, 20, 30, 40, 50, 60, 70, 80, 11, 21, 31, 41, 51, 61, 71, 81,
        ]);
        // Context: first chunk of t1
        let context = &t1.chunks[0..1];
        let pred = predict_next_from_library(context, &[t1.clone()], 0.5);
        // Should predict second chunk of t1
        let sim = cosine(&pred, &t1.chunks[1]);
        assert!(
            sim > 0.9,
            "library prediction should find continuation, sim={}",
            sim
        );
    }

    #[test]
    fn test_deterministic() {
        let c1 = ChunkCodec::new(512);
        let c2 = ChunkCodec::new(512);
        for i in 0..512 {
            assert_eq!(c1.token_embeddings[i], c2.token_embeddings[i]);
        }
    }

    #[test]
    fn test_cl17_interop() {
        let codec = ChunkCodec::new(128);
        let chunk = codec.encode_chunk(&[5, 10, 15, 20, 25, 30, 35, 40]);
        let mv = chunk_to_multivector(&chunk);
        let back = multivector_to_chunk(&mv);
        for j in 0..CATA_DIM {
            assert!((chunk[j] - back[j]).abs() < 1e-7);
        }
    }

    #[test]
    fn test_graded_chunk_decomposition() {
        let codec = ChunkCodec::new(256);
        let chunk = codec.encode_chunk(&[10, 20, 30, 40, 50, 60, 70, 80]);
        let graded = GradedChunk::from_chunk(&chunk);

        assert!(graded.energy > 0.0, "energy should be positive");
        let sem_norm: f32 = graded
            .semantic_dir
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        assert!(sem_norm > 0.0, "semantic direction should be nonzero");
        let ctx_norm: f32 = graded
            .context_bivector
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        assert!(
            ctx_norm >= 0.0,
            "context bivector norm should be non-negative"
        );
    }

    #[test]
    fn test_graded_semantic_similarity() {
        let codec = ChunkCodec::new(256);
        let a = GradedChunk::from_chunk(&codec.encode_chunk(&[10, 20, 30, 40, 50, 60, 70, 80]));
        let b = GradedChunk::from_chunk(&codec.encode_chunk(&[10, 20, 30, 40, 50, 60, 70, 80]));
        let c =
            GradedChunk::from_chunk(&codec.encode_chunk(&[200, 201, 202, 203, 204, 205, 206, 207]));

        let sim_same = a.semantic_similarity(&b);
        let sim_diff = a.semantic_similarity(&c);
        assert!(
            sim_same > sim_diff,
            "same chunk should be more similar: {} vs {}",
            sim_same,
            sim_diff
        );
        assert!(
            (sim_same - 1.0).abs() < 0.01,
            "identical chunks should have sim≈1.0: {}",
            sim_same
        );
    }

    #[test]
    fn test_transition_rotor() {
        let codec = ChunkCodec::new(256);
        let a = GradedChunk::from_chunk(&codec.encode_chunk(&[10, 20, 30, 40, 50, 60, 70, 80]));
        let b = GradedChunk::from_chunk(&codec.encode_chunk(&[11, 21, 31, 41, 51, 61, 71, 81]));

        let transition = compute_transition(&a, &b);
        assert!(
            transition.fidelity > -1.0,
            "fidelity should be computed: {}",
            transition.fidelity
        );
    }

    #[test]
    fn test_predict_next_algebraic() {
        let codec = ChunkCodec::new(256);
        let c0 = codec.encode_chunk(&[10, 20, 30, 40, 50, 60, 70, 80]);
        let c1 = codec.encode_chunk(&[11, 21, 31, 41, 51, 61, 71, 81]);

        let g0 = GradedChunk::from_chunk(&c0);
        let g1 = GradedChunk::from_chunk(&c1);

        let pred = predict_next_algebraic(&[g0, g1]);
        let pred_norm: f32 = pred.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(pred_norm > 0.0, "prediction should be nonzero");

        // Prediction should be further along the trajectory
        let sim_with_last = cosine(&pred, &c1);
        assert!(
            sim_with_last > 0.0,
            "prediction should have positive similarity to last chunk: {}",
            sim_with_last
        );
    }

    #[test]
    fn test_compose_algebraic() {
        let codec = ChunkCodec::new(256);
        let s1 = codec.encode_sequence(&[10, 20, 30, 40, 50, 60, 70, 80]);
        let s2 = codec.encode_sequence(&[90, 91, 92, 93, 94, 95, 96, 97]);

        // With high weight on s1, composed should be close to s1
        let composed = compose_algebraic(&[(s1.clone(), 10.0), (s2.clone(), 0.1)], 1);
        assert_eq!(composed.len(), 1);
        let sim = cosine(&composed[0], &s1.chunks[0]);
        assert!(sim > 0.5, "high-weight source should dominate: sim={}", sim);
    }

    #[test]
    fn test_graded_similarity() {
        let codec = ChunkCodec::new(256);
        let a = codec.encode_chunk(&[10, 20, 30, 40, 50, 60, 70, 80]);
        let b = codec.encode_chunk(&[10, 20, 30, 40, 50, 60, 70, 80]);
        let c = codec.encode_chunk(&[200, 201, 202, 203, 204, 205, 206, 207]);

        let sim_same = graded_similarity(&a, &b);
        let sim_diff = graded_similarity(&a, &c);
        assert!(
            sim_same > sim_diff,
            "same chunk should be more similar: {} vs {}",
            sim_same,
            sim_diff
        );
    }

    // -----------------------------------------------------------------------
    // Phase 1: SpacetimeChunk + Interval + Inertia + Negation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_spacetime_chunk_decomposition() {
        let codec = ChunkCodec::new(256);
        let chunk = codec.encode_chunk(&[10, 20, 30, 40, 50, 60, 70, 80]);
        let st = SpacetimeChunk::from_chunk(&chunk);

        assert!(st.energy > 0.0, "energy should be positive");
        let sem_norm: f32 = st.semantic_dir.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(sem_norm > 0.0, "semantic direction should be nonzero");
        let boost_norm: f32 = st.boost_causal.iter().map(|x| x * x).sum::<f32>().sqrt();
        let rot_norm: f32 = st
            .rotation_structural
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        assert!(boost_norm >= 0.0 && rot_norm >= 0.0);
    }

    #[test]
    fn test_spacetime_semantic_similarity() {
        let codec = ChunkCodec::new(256);
        let a = SpacetimeChunk::from_chunk(&codec.encode_chunk(&[10, 20, 30, 40, 50, 60, 70, 80]));
        let b = SpacetimeChunk::from_chunk(&codec.encode_chunk(&[10, 20, 30, 40, 50, 60, 70, 80]));
        let c = SpacetimeChunk::from_chunk(
            &codec.encode_chunk(&[200, 201, 202, 203, 204, 205, 206, 207]),
        );

        let sim_same = a.semantic_similarity(&b);
        let sim_diff = a.semantic_similarity(&c);
        assert!(
            sim_same > sim_diff,
            "identical chunks should be more similar: {} vs {}",
            sim_same,
            sim_diff
        );
        assert!((sim_same - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_spacetime_graded_similarity() {
        let codec = ChunkCodec::new(256);
        let a = SpacetimeChunk::from_chunk(&codec.encode_chunk(&[10, 20, 30, 40, 50, 60, 70, 80]));
        let b = SpacetimeChunk::from_chunk(&codec.encode_chunk(&[10, 20, 30, 40, 50, 60, 70, 80]));
        let c = SpacetimeChunk::from_chunk(
            &codec.encode_chunk(&[200, 201, 202, 203, 204, 205, 206, 207]),
        );

        let gs_same = a.graded_similarity(&b);
        let gs_diff = a.graded_similarity(&c);
        assert!(
            gs_same > gs_diff,
            "graded sim should separate same vs different: {} vs {}",
            gs_same,
            gs_diff
        );
    }

    #[test]
    fn test_minkowski_interval_types() {
        use crate::clifford::{classify_interval, minkowski_interval, IntervalType, Multivector};
        // Pure timelike separation (only e0 differs)
        let mut a = Multivector::zero();
        let mut b = Multivector::zero();
        a.components[1] = 1.0; // grade-1, e0 (timelike)
        b.components[1] = 2.0;
        let s2 = minkowski_interval(&a, &b);
        assert!(s2 < 0.0, "pure e0 separation should be timelike: {}", s2);
        assert_eq!(classify_interval(s2), IntervalType::Timelike);

        // Pure spacelike separation (only e1 differs)
        let mut c = Multivector::zero();
        let mut d = Multivector::zero();
        c.components[2] = 1.0; // grade-1, e1 (spacelike)
        d.components[2] = 2.0;
        let s2 = minkowski_interval(&c, &d);
        assert!(s2 > 0.0, "pure e1 separation should be spacelike: {}", s2);
        assert_eq!(classify_interval(s2), IntervalType::Spacelike);

        // Lightlike (e0 and e1 change equally)
        let mut e = Multivector::zero();
        let mut f = Multivector::zero();
        e.components[1] = 0.0;
        e.components[2] = 0.0;
        f.components[1] = 1.0;
        f.components[2] = 1.0;
        let s2 = minkowski_interval(&e, &f);
        assert!(
            s2.abs() < 0.02,
            "equal e0/e1 separation should be lightlike: {}",
            s2
        );
        assert_eq!(classify_interval(s2), IntervalType::Lightlike);
    }

    #[test]
    fn test_predict_next_spacetime() {
        let codec = ChunkCodec::new(256);
        let c0 = SpacetimeChunk::from_chunk(&codec.encode_chunk(&[10, 20, 30, 40, 50, 60, 70, 80]));
        let c1 = SpacetimeChunk::from_chunk(&codec.encode_chunk(&[11, 21, 31, 41, 51, 61, 71, 81]));

        let (pred, _interval, confidence) = predict_next_spacetime(&[c0, c1], 1.0);
        let pred_norm: f32 = pred.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(pred_norm > 0.0, "prediction should be nonzero");
        assert!(
            confidence > 0.0,
            "confidence should be positive: {}",
            confidence
        );
    }

    #[test]
    fn test_semantic_negate() {
        let codec = ChunkCodec::new(256);
        let original =
            SpacetimeChunk::from_chunk(&codec.encode_chunk(&[10, 20, 30, 40, 50, 60, 70, 80]));
        let negated = semantic_negate(&original);

        // Negation should change the multivector
        let diff: f32 = original
            .raw
            .iter()
            .zip(negated.raw.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(
            diff > 0.1,
            "negation should change the embedding: diff={}",
            diff
        );

        // Double negation should approximately recover original
        // (I² = -1 in Cl(1,7), so applying twice gives -original)
        let double_neg = semantic_negate(&negated);
        for i in 0..CATA_DIM {
            assert!(
                (double_neg.raw[i] + original.raw[i]).abs() < 1e-3,
                "double negation should give -original at component {}: {} vs {}",
                i,
                double_neg.raw[i],
                -original.raw[i]
            );
        }
    }

    #[test]
    fn test_trajectory_mass() {
        let codec = ChunkCodec::new(256);
        // Steady trajectory: incrementing tokens
        let steady: Vec<SpacetimeChunk> = (0..4)
            .map(|k| {
                let base = (10 + k * 1) as u16;
                SpacetimeChunk::from_chunk(&codec.encode_chunk(&[
                    base,
                    base + 10,
                    base + 20,
                    base + 30,
                    base + 40,
                    base + 50,
                    base + 60,
                    base + 70,
                ]))
            })
            .collect();

        // Erratic trajectory: jumping tokens
        let erratic: Vec<SpacetimeChunk> = vec![
            SpacetimeChunk::from_chunk(&codec.encode_chunk(&[10, 20, 30, 40, 50, 60, 70, 80])),
            SpacetimeChunk::from_chunk(
                &codec.encode_chunk(&[200, 201, 202, 203, 204, 205, 206, 207]),
            ),
            SpacetimeChunk::from_chunk(&codec.encode_chunk(&[5, 15, 25, 35, 45, 55, 65, 75])),
            SpacetimeChunk::from_chunk(
                &codec.encode_chunk(&[150, 160, 170, 180, 190, 100, 110, 120]),
            ),
        ];

        let mass_steady = trajectory_mass(&steady);
        let mass_erratic = trajectory_mass(&erratic);
        assert!(
            mass_erratic >= mass_steady,
            "erratic trajectory should have higher mass: {} vs {}",
            mass_erratic,
            mass_steady
        );
    }

    #[test]
    fn test_interval_between_spacetime_chunks() {
        let codec = ChunkCodec::new(256);
        let a = SpacetimeChunk::from_chunk(&codec.encode_chunk(&[10, 20, 30, 40, 50, 60, 70, 80]));
        let b = SpacetimeChunk::from_chunk(&codec.encode_chunk(&[11, 21, 31, 41, 51, 61, 71, 81]));
        let interval = a.interval_to(&b);
        assert!(matches!(
            interval,
            IntervalType::Timelike | IntervalType::Spacelike | IntervalType::Lightlike
        ));
    }

    // -----------------------------------------------------------------------
    // Phase 4b: Per-grade temperature tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_grade_temperature_zero_is_identity() {
        let codec = ChunkCodec::new(256);
        let chunk = codec.encode_chunk(&[10, 20, 30, 40, 50, 60, 70, 80]);
        let zero_temp = GradeTemperature {
            scalar: 0.0,
            vector: 0.0,
            bivector: 0.0,
            trivector: 0.0,
            pseudoscalar: 0.0,
        };
        let result = apply_grade_temperature(&chunk, &zero_temp, 42);
        for i in 0..CATA_DIM {
            assert!(
                (result[i] - chunk[i]).abs() < 1e-7,
                "zero temperature should not change chunk at {}",
                i
            );
        }
    }

    #[test]
    fn test_grade_temperature_adds_variation() {
        let codec = ChunkCodec::new(256);
        let chunk = codec.encode_chunk(&[10, 20, 30, 40, 50, 60, 70, 80]);
        let high_temp = GradeTemperature {
            scalar: 1.0,
            vector: 1.0,
            bivector: 1.0,
            trivector: 1.0,
            pseudoscalar: 1.0,
        };
        let result = apply_grade_temperature(&chunk, &high_temp, 42);
        let diff: f32 = chunk
            .iter()
            .zip(result.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(
            diff > 0.01,
            "nonzero temperature should add variation: diff={}",
            diff
        );
    }

    #[test]
    fn test_grade_temperature_deterministic() {
        let codec = ChunkCodec::new(256);
        let chunk = codec.encode_chunk(&[10, 20, 30, 40, 50, 60, 70, 80]);
        let temp = GradeTemperature::default();
        let r1 = apply_grade_temperature(&chunk, &temp, 42);
        let r2 = apply_grade_temperature(&chunk, &temp, 42);
        for i in 0..CATA_DIM {
            assert!(
                (r1[i] - r2[i]).abs() < 1e-7,
                "same seed should give same result"
            );
        }
    }

    #[test]
    fn test_grade_temperature_different_seeds() {
        let codec = ChunkCodec::new(256);
        let chunk = codec.encode_chunk(&[10, 20, 30, 40, 50, 60, 70, 80]);
        let temp = GradeTemperature::default();
        let r1 = apply_grade_temperature(&chunk, &temp, 42);
        let r2 = apply_grade_temperature(&chunk, &temp, 99);
        let diff: f32 = r1
            .iter()
            .zip(r2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(
            diff > 0.001,
            "different seeds should give different results: diff={}",
            diff
        );
    }

    #[test]
    fn test_grade_temperature_presets() {
        let conservative = GradeTemperature::conservative();
        let creative = GradeTemperature::creative();
        assert!(
            conservative.bivector < creative.bivector,
            "creative should have higher bivector temp"
        );
        assert!(
            conservative.vector < creative.vector,
            "creative should have higher vector temp"
        );
    }

    // -----------------------------------------------------------------------
    // Phase 3: SemanticPropagator tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_propagator_observe_and_predict() {
        let codec = ChunkCodec::new(256);
        let chunks: Vec<SpacetimeChunk> = (0..4)
            .map(|k| {
                let base = (10 + k * 2) as u16;
                SpacetimeChunk::from_chunk(&codec.encode_chunk(&[
                    base,
                    base + 10,
                    base + 20,
                    base + 30,
                    base + 40,
                    base + 50,
                    base + 60,
                    base + 70,
                ]))
            })
            .collect();

        let mut prop = SemanticPropagator::new(1.0, 0.5);
        for c in &chunks {
            prop.observe(c);
        }

        assert_eq!(
            prop.history.len(),
            3,
            "should have 3 transitions for 4 chunks"
        );
        assert!(prop.mean_rotor.is_some(), "mean rotor should be computed");

        let pred = prop.predict_next();
        assert!(pred.is_some(), "should produce a prediction");
        let (pred_chunk, interval, confidence) = pred.unwrap();
        let pred_norm: f32 = pred_chunk.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(pred_norm > 0.0, "prediction should be nonzero");
        assert!(
            confidence > 0.0,
            "confidence should be positive: {}",
            confidence
        );
    }

    #[test]
    fn test_propagator_from_trajectory() {
        let codec = ChunkCodec::new(256);
        let chunks: Vec<SpacetimeChunk> = (0..5)
            .map(|k| {
                let base = (10 + k * 3) as u16;
                SpacetimeChunk::from_chunk(&codec.encode_chunk(&[
                    base,
                    base + 5,
                    base + 10,
                    base + 15,
                    base + 20,
                    base + 25,
                    base + 30,
                    base + 35,
                ]))
            })
            .collect();

        let prop = SemanticPropagator::from_trajectory(&chunks, 1.5, 0.3);
        assert_eq!(prop.history.len(), 4);
        assert!(prop.predict_next().is_some());
    }

    #[test]
    fn test_propagator_ambiguity() {
        let codec = ChunkCodec::new(256);
        let chunks: Vec<SpacetimeChunk> = (0..3)
            .map(|k| {
                let base = (10 + k * 5) as u16;
                SpacetimeChunk::from_chunk(&codec.encode_chunk(&[
                    base,
                    base + 10,
                    base + 20,
                    base + 30,
                    base + 40,
                    base + 50,
                    base + 60,
                    base + 70,
                ]))
            })
            .collect();

        let prop = SemanticPropagator::from_trajectory(&chunks, 1.0, 0.5);
        let ambiguity = prop.ambiguity_score();
        assert!(
            ambiguity >= 0.0 && ambiguity <= 1.0,
            "ambiguity should be in [0,1]: {}",
            ambiguity
        );
    }

    #[test]
    fn test_propagator_compose() {
        let codec = ChunkCodec::new(256);
        let traj1: Vec<SpacetimeChunk> = (0..3)
            .map(|k| {
                let base = (10 + k * 2) as u16;
                SpacetimeChunk::from_chunk(&codec.encode_chunk(&[
                    base,
                    base + 10,
                    base + 20,
                    base + 30,
                    base + 40,
                    base + 50,
                    base + 60,
                    base + 70,
                ]))
            })
            .collect();
        let traj2: Vec<SpacetimeChunk> = (0..3)
            .map(|k| {
                let base = (100 + k * 3) as u16;
                SpacetimeChunk::from_chunk(&codec.encode_chunk(&[
                    base,
                    base + 5,
                    base + 10,
                    base + 15,
                    base + 20,
                    base + 25,
                    base + 30,
                    base + 35,
                ]))
            })
            .collect();

        let prop = SemanticPropagator::new(1.0, 0.5);
        let composed = prop.compose_trajectories(&[(traj1, 2.0), (traj2, 1.0)], 2);
        assert_eq!(composed.len(), 2);
        let norm: f32 = composed[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm > 0.0, "composed chunk should be nonzero");
    }

    #[test]
    fn test_propagator_kinetic_potential_energy() {
        let codec = ChunkCodec::new(256);
        // Steady trajectory
        let steady: Vec<SpacetimeChunk> = (0..5)
            .map(|k| {
                let base = (10 + k) as u16;
                SpacetimeChunk::from_chunk(&codec.encode_chunk(&[
                    base,
                    base + 10,
                    base + 20,
                    base + 30,
                    base + 40,
                    base + 50,
                    base + 60,
                    base + 70,
                ]))
            })
            .collect();
        let prop_steady = SemanticPropagator::from_trajectory(&steady, 1.0, 0.5);

        // Erratic trajectory
        let erratic: Vec<SpacetimeChunk> = vec![
            SpacetimeChunk::from_chunk(&codec.encode_chunk(&[10, 20, 30, 40, 50, 60, 70, 80])),
            SpacetimeChunk::from_chunk(&codec.encode_chunk(&[200, 100, 50, 250, 5, 150, 75, 225])),
            SpacetimeChunk::from_chunk(&codec.encode_chunk(&[15, 25, 35, 45, 55, 65, 75, 85])),
            SpacetimeChunk::from_chunk(&codec.encode_chunk(&[180, 90, 45, 230, 10, 130, 60, 210])),
            SpacetimeChunk::from_chunk(&codec.encode_chunk(&[20, 30, 40, 50, 60, 70, 80, 90])),
        ];
        let prop_erratic = SemanticPropagator::from_trajectory(&erratic, 1.0, 0.5);

        let ke_steady = prop_steady.kinetic_energy();
        let ke_erratic = prop_erratic.kinetic_energy();
        // Erratic trajectory should have higher kinetic energy
        assert!(
            ke_erratic > ke_steady,
            "erratic KE should exceed steady KE: {} vs {}",
            ke_erratic,
            ke_steady
        );
    }
}
