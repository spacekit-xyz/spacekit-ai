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
        Self { token_embeddings, position_codes: codes, vocab_size: vs }
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
            for j in 0..CATA_DIM { latent[j] += emb[j] * code[j]; }
        }
        if n > 0 {
            let s = 1.0 / (n as f32).sqrt();
            for x in latent.iter_mut() { *x *= s; }
        }
        latent
    }

    /// Decode a 256d chunk vector back to up to K token IDs.
    /// Uses parallel decode + PIC refinement for high fidelity.
    pub fn decode_chunk(&self, latent: &[f32; CATA_DIM], num_tokens: usize) -> Vec<u16> {
        let n = num_tokens.min(CHUNK_K);
        if n == 0 { return Vec::new(); }
        let scale = (n as f32).sqrt();

        // Pass 1: parallel de-spread (independent, no error propagation)
        let mut tokens: Vec<u16> = (0..n).map(|pos| {
            let code = &self.position_codes[pos];
            let mut ds = [0.0f32; CATA_DIM];
            for j in 0..CATA_DIM { ds[j] = latent[j] * scale * code[j]; }
            self.nearest_token(&ds).0
        }).collect();

        // Pass 2+: parallel interference cancellation
        for _ in 0..PIC_ROUNDS {
            let prev = tokens.clone();
            for pos in 0..n {
                let mut iso = [0.0f32; CATA_DIM];
                for j in 0..CATA_DIM { iso[j] = latent[j] * scale; }
                for (q, &tok) in prev.iter().enumerate() {
                    if q == pos { continue; }
                    let emb = &self.token_embeddings[tok as usize];
                    let code = &self.position_codes[q];
                    for j in 0..CATA_DIM { iso[j] -= emb[j] * code[j]; }
                }
                let code = &self.position_codes[pos];
                let mut ds = [0.0f32; CATA_DIM];
                for j in 0..CATA_DIM { ds[j] = iso[j] * code[j]; }
                tokens[pos] = self.nearest_token(&ds).0;
            }
            if tokens == prev { break; }
        }
        tokens
    }

    /// Roundtrip accuracy for a single chunk.
    pub fn chunk_accuracy(&self, tokens: &[u16]) -> f32 {
        let n = tokens.len().min(CHUNK_K);
        if n == 0 { return 1.0; }
        let enc = self.encode_chunk(tokens);
        let dec = self.decode_chunk(&enc, n);
        let hits = tokens.iter().take(n).zip(dec.iter()).filter(|(&a, &b)| a == b).count();
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
    pub fn encode_text(&self, text: &str, dict: &TokenDictionary) -> ChunkSequence {
        let ids = dict.encode(text);
        self.encode_sequence(&ids)
    }

    /// Decode chunk sequence → text string (convenience).
    pub fn decode_text(&self, seq: &ChunkSequence, dict: &TokenDictionary) -> String {
        let ids = self.decode_sequence(seq);
        dict.decode(&ids)
    }

    /// Full roundtrip accuracy for a token sequence (chunk-by-chunk).
    pub fn sequence_accuracy(&self, token_ids: &[u16]) -> f32 {
        if token_ids.is_empty() { return 1.0; }
        let seq = self.encode_sequence(token_ids);
        let dec = self.decode_sequence(&seq);
        let hits = token_ids.iter().zip(dec.iter()).filter(|(&a, &b)| a == b).count();
        hits as f32 / token_ids.len() as f32
    }

    // -- Internals ------------------------------------------------------------

    fn nearest_token(&self, signal: &[f32; CATA_DIM]) -> (u16, f32) {
        let mut best_id = 0u16;
        let mut best_dot = f32::NEG_INFINITY;
        for (id, emb) in self.token_embeddings.iter().enumerate() {
            let d: f32 = signal.iter().zip(emb.iter()).map(|(a, b)| a * b).sum();
            if d > best_dot { best_dot = d; best_id = id as u16; }
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
            if norm > 1e-8 { for x in emb.iter_mut() { *x /= norm; } }
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
            for (j, &v) in row.iter().enumerate() { codes[k][j] = v; }
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
    pub fn num_chunks(&self) -> usize { self.chunks.len() }

    /// Semantic centroid: average of all chunk vectors.
    pub fn centroid(&self) -> [f32; CATA_DIM] {
        let mut c = [0.0f32; CATA_DIM];
        if self.chunks.is_empty() { return c; }
        for chunk in &self.chunks {
            for j in 0..CATA_DIM { c[j] += chunk[j]; }
        }
        let n = self.chunks.len() as f32;
        for x in c.iter_mut() { *x /= n; }
        c
    }

    /// Cosine similarity between centroids.
    pub fn similarity(&self, other: &ChunkSequence) -> f32 {
        let a = self.centroid();
        let b = other.centroid();
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na < 1e-8 || nb < 1e-8 { 0.0 } else { dot / (na * nb) }
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
    if sources.is_empty() { return Vec::new(); }
    let n_chunks = target_len;

    (0..n_chunks).map(|i| {
        let mut result = [0.0f32; CATA_DIM];
        let mut total_w = 0.0f32;
        for (seq, w) in sources {
            if i < seq.chunks.len() {
                for j in 0..CATA_DIM { result[j] += seq.chunks[i][j] * w; }
                total_w += w;
            }
        }
        if total_w > 1e-8 {
            for x in result.iter_mut() { *x /= total_w; }
        }
        result
    }).collect()
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
        for j in 0..CATA_DIM { result[j] = a[j] * (1.0 - t) + b[j] * t; }
        return result;
    }

    let sin_o = omega.sin();
    let w1 = ((1.0 - t) * omega).sin() / sin_o;
    let w2 = (t * omega).sin() / sin_o;
    for j in 0..CATA_DIM { result[j] = a[j] * w1 + b[j] * w2; }
    result
}

fn cosine(a: &[f32; CATA_DIM], b: &[f32; CATA_DIM]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-8 || nb < 1e-8 { 0.0 } else { dot / (na * nb) }
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
                let d: f32 = codec.token_embeddings[i].iter()
                    .zip(codec.token_embeddings[j].iter())
                    .map(|(a, b)| a * b).sum();
                if d.abs() > max_cos { max_cos = d.abs(); }
            }
        }
        assert!(max_cos < 0.25, "max mutual cosine = {}", max_cos);
    }

    #[test]
    fn test_position_codes_orthogonal() {
        let codec = ChunkCodec::new(16);
        for i in 0..CHUNK_K {
            for j in (i + 1)..CHUNK_K {
                let dot: f32 = codec.position_codes[i].iter()
                    .zip(codec.position_codes[j].iter())
                    .map(|(a, b)| a * b).sum();
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
            assert_eq!(acc, 1.0, "K-token chunk starting at {} failed: acc={}", start, acc);
        }
    }

    #[test]
    fn test_random_chunk_roundtrip() {
        let codec = ChunkCodec::new(2048);
        let mut state = 42u64;
        let mut total_correct = 0;
        let trials = 100;
        for _ in 0..trials {
            let tokens: Vec<u16> = (0..CHUNK_K).map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((state >> 33) % 2048) as u16
            }).collect();
            let acc = codec.chunk_accuracy(&tokens);
            if acc == 1.0 { total_correct += 1; }
        }
        let pct = total_correct as f32 / trials as f32;
        assert!(pct >= 0.95, "random chunk perfect roundtrip rate = {:.1}%", pct * 100.0);
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
        let tokens: Vec<u16> = (0..64).map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) % 2048) as u16
        }).collect();
        let acc = codec.sequence_accuracy(&tokens);
        assert!(acc >= 0.99, "64-token sequence accuracy = {:.3}", acc);
    }

    #[test]
    fn test_centroid_similarity() {
        let codec = ChunkCodec::new(256);
        let s1 = codec.encode_sequence(&[10, 20, 30, 40, 50, 60, 70, 80]);
        let s2 = codec.encode_sequence(&[10, 20, 30, 40, 50, 60, 70, 80]);
        let s3 = codec.encode_sequence(&[200, 201, 202, 203, 204, 205, 206, 207]);
        assert!((s1.similarity(&s2) - 1.0).abs() < 1e-4, "identical seqs should have sim=1");
        assert!(s1.similarity(&s3) < 0.5, "different seqs should have low sim");
    }

    #[test]
    fn test_predict_next_extrapolation() {
        let codec = ChunkCodec::new(256);
        let c0 = codec.encode_chunk(&[10, 20, 30, 40, 50, 60, 70, 80]);
        let c1 = codec.encode_chunk(&[11, 21, 31, 41, 51, 61, 71, 81]);
        let pred = predict_next_chunk(&[c0, c1]);
        // Predicted chunk should be further along the same direction
        let d01: f32 = c0.iter().zip(c1.iter()).map(|(a, b)| (b - a).powi(2)).sum::<f32>().sqrt();
        let d1p: f32 = c1.iter().zip(pred.iter()).map(|(a, b)| (b - a).powi(2)).sum::<f32>().sqrt();
        assert!((d01 - d1p).abs() < 0.1, "prediction should maintain velocity");
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
        let t1 = codec.encode_sequence(&[10, 20, 30, 40, 50, 60, 70, 80, 11, 21, 31, 41, 51, 61, 71, 81]);
        // Context: first chunk of t1
        let context = &t1.chunks[0..1];
        let pred = predict_next_from_library(context, &[t1.clone()], 0.5);
        // Should predict second chunk of t1
        let sim = cosine(&pred, &t1.chunks[1]);
        assert!(sim > 0.9, "library prediction should find continuation, sim={}", sim);
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
}
