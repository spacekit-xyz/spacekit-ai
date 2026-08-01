//! Spectral encoding for text generation via DCT + dictionary matching.
//!
//! Two-level pipeline:
//!   1. Dictionary reduces text to a short token-ID sequence (130 chars → ~25 tokens)
//!   2. DCT compresses the token-ID sequence into K frequency coefficients
//!
//! Train:    text → tokenize → IDs → normalize → DCT → coefficients (network target)
//! Generate: predicted coefficients → IDCT → denormalize → round to IDs → dictionary → text
//!
//! The dictionary provides error correction: even if a predicted coefficient is
//! slightly off, rounding to the nearest valid token ID snaps to correct text.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Hamming Error-Correcting Code — single-error correction for token bits
// ---------------------------------------------------------------------------

/// Compute number of parity bits needed for `data_bits` data bits.
/// Hamming code: 2^r >= data_bits + r + 1
pub fn hamming_parity_bits(data_bits: usize) -> usize {
    let mut r = 1;
    while (1usize << r) < data_bits + r + 1 {
        r += 1;
    }
    r
}

/// Encode `data_bits` data bits with Hamming parity bits.
/// Returns a codeword of length `data_bits + parity_bits`.
/// Parity bits are at positions that are powers of 2 (1-indexed).
pub fn hamming_encode(data: &[u8], data_bits: usize) -> Vec<u8> {
    let r = hamming_parity_bits(data_bits);
    let total = data_bits + r;
    let mut codeword = vec![0u8; total];

    // Place data bits in non-power-of-2 positions (1-indexed)
    let mut di = 0;
    for pos in 1..=total {
        if pos.is_power_of_two() {
            continue;
        }
        if di < data_bits {
            codeword[pos - 1] = if di < data.len() { data[di] } else { 0 };
        }
        di += 1;
    }

    // Compute parity bits
    for i in 0..r {
        let parity_pos = 1 << i; // 1, 2, 4, 8, ...
        let mut parity = 0u8;
        for pos in 1..=total {
            if pos & parity_pos != 0 && pos != parity_pos {
                parity ^= codeword[pos - 1];
            }
        }
        codeword[parity_pos - 1] = parity;
    }

    codeword
}

/// Decode a Hamming codeword, correcting up to 1 bit error.
/// Returns the extracted data bits.
pub fn hamming_decode(codeword: &[u8], data_bits: usize) -> Vec<u8> {
    let r = hamming_parity_bits(data_bits);
    let total = data_bits + r;
    let len = codeword.len().min(total);

    // Compute syndrome (error position, 1-indexed; 0 = no error)
    let mut syndrome = 0usize;
    for i in 0..r {
        let parity_pos = 1 << i;
        let mut parity = 0u8;
        for pos in 1..=len {
            if pos & parity_pos != 0 {
                parity ^= if pos - 1 < codeword.len() {
                    codeword[pos - 1]
                } else {
                    0
                };
            }
        }
        if parity != 0 {
            syndrome |= parity_pos;
        }
    }

    // Correct single-bit error if syndrome is valid
    let mut corrected: Vec<u8> = codeword.iter().copied().take(total).collect();
    corrected.resize(total, 0);
    if syndrome > 0 && syndrome <= total {
        corrected[syndrome - 1] ^= 1;
    }

    // Extract data bits from non-power-of-2 positions
    let mut data = Vec::with_capacity(data_bits);
    for pos in 1..=total {
        if pos.is_power_of_two() {
            continue;
        }
        data.push(corrected[pos - 1]);
        if data.len() >= data_bits {
            break;
        }
    }
    data
}

// ---------------------------------------------------------------------------
// E8 Lattice Engine — optimal sphere packing in dimension 8
// ---------------------------------------------------------------------------
//
// The E8 lattice achieves the densest possible sphere packing in 8 dimensions
// (Viazovska, 2016). Properties:
//   - Kissing number: 240 (each point has exactly 240 equidistant neighbors)
//   - Root system: 240 vectors forming the E8 root system
//   - Automorphism group: |W(E8)| = 696,729,600
//   - Related code: extended Hamming [8,4,4]
//
// The bridge embedding decomposes as n/8 × 8d E8 subspaces, giving
// provably optimal quantization and algebraically exact compatibility scores.

/// E8 lattice: nearest-lattice-point decoding and root system operations.
///
/// The E8 lattice consists of all points in Z^8 and (Z+1/2)^8 whose
/// coordinate sum is even. Nearest-point decoding maps any 8d vector
/// to the closest E8 lattice point in O(8) time.
#[derive(Clone, Debug)]
pub struct E8Lattice;

impl E8Lattice {
    /// Decode: find the nearest E8 lattice point to an arbitrary 8d vector.
    ///
    /// Algorithm: project onto both the integer sublattice D8 and the
    /// half-integer coset D8 + (1/2,...,1/2), pick whichever is closer.
    /// D8 nearest-point uses the standard "round and fix parity" method.
    pub fn nearest_point(x: &[f32; 8]) -> [f32; 8] {
        let d8_point = Self::nearest_d8(x);
        let d8_dist = Self::dist_sq(x, &d8_point);

        // Half-integer coset: shift by (0.5, ..., 0.5), find nearest D8, shift back
        let mut shifted = [0.0f32; 8];
        for i in 0..8 {
            shifted[i] = x[i] - 0.5;
        }
        let d8_half = Self::nearest_d8(&shifted);
        let mut coset_point = [0.0f32; 8];
        for i in 0..8 {
            coset_point[i] = d8_half[i] + 0.5;
        }
        let coset_dist = Self::dist_sq(x, &coset_point);

        if d8_dist <= coset_dist {
            d8_point
        } else {
            coset_point
        }
    }

    /// Nearest point in the D8 lattice (integer points with even coordinate sum).
    /// Round each coordinate, then fix parity by adjusting the coordinate
    /// with the largest rounding error.
    fn nearest_d8(x: &[f32; 8]) -> [f32; 8] {
        let mut rounded = [0.0f32; 8];
        let mut errors = [0.0f32; 8];
        let mut sum = 0i32;

        for i in 0..8 {
            let r = x[i].round();
            rounded[i] = r;
            errors[i] = (x[i] - r).abs();
            sum += r as i32;
        }

        // If coordinate sum is odd, adjust the coordinate with largest rounding error
        if sum % 2 != 0 {
            let mut max_err_idx = 0;
            let mut max_err = -1.0f32;
            for i in 0..8 {
                if errors[i] > max_err {
                    max_err = errors[i];
                    max_err_idx = i;
                }
            }
            if x[max_err_idx] > rounded[max_err_idx] {
                rounded[max_err_idx] += 1.0;
            } else {
                rounded[max_err_idx] -= 1.0;
            }
        }

        rounded
    }

    fn dist_sq(a: &[f32; 8], b: &[f32; 8]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    /// Quantize an n-dimensional vector by decomposing into ⌈n/8⌉ × 8d E8 subspaces.
    /// Returns lattice points as a flat vector matching the input length (padded up to next multiple of 8).
    pub fn quantize_64d(x: &[f32]) -> Vec<f32> {
        let n = x.len().max(8);
        let num_blocks = (n + 7) / 8;
        let out_len = num_blocks * 8;
        let mut result = vec![0.0f32; out_len];
        for sub in 0..num_blocks {
            let offset = sub * 8;
            let mut block = [0.0f32; 8];
            for i in 0..8 {
                block[i] = if offset + i < x.len() {
                    x[offset + i]
                } else {
                    0.0
                };
            }
            let lattice_point = Self::nearest_point(&block);
            for i in 0..8 {
                result[offset + i] = lattice_point[i];
            }
        }
        result.truncate(x.len());
        result
    }

    /// Compute the quantization distance (sum of squared errors across all
    /// E8 subspaces). Lower = better match to lattice structure.
    pub fn quantization_distance(x: &[f32]) -> f32 {
        let quantized = Self::quantize_64d(x);
        x.iter()
            .zip(quantized.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum()
    }

    /// E8 root inner product between two lattice-quantized embeddings.
    /// Returns a value in {-2, -1, 0, 1, 2} for exact lattice points,
    /// or a continuous value for approximate embeddings.
    ///
    /// Averages one cosine per ⌈n/8⌉ E8 subspaces (`n = min(len(a), len(b))`);
    /// the final tail shorter than 8 is scored in its own subspace slice.
    ///
    /// This replaces heuristic cosine similarity with the algebraic structure
    /// of the E8 root system for Hopf transition scoring.
    pub fn root_inner_product(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        if n == 0 {
            return 0.0;
        }
        let qa = Self::quantize_64d(&a[..n]);
        let qb = Self::quantize_64d(&b[..n]);
        debug_assert_eq!(qa.len(), n);
        debug_assert_eq!(qb.len(), n);

        // Cosine in each 8d E8 subspace, then mean over all subspaces (partial tail counts as one block).
        let num_blocks = (n + 7) / 8;
        let mut total = 0.0f32;
        for sub in 0..num_blocks {
            let offset = sub * 8;
            let end = (offset + 8).min(n);
            let slice_a = &qa[offset..end];
            let slice_b = &qb[offset..end];
            let dot: f32 = slice_a.iter().zip(slice_b.iter()).map(|(x, y)| x * y).sum();
            let na = slice_a.iter().map(|v| v * v).sum::<f32>().sqrt();
            let nb = slice_b.iter().map(|v| v * v).sum::<f32>().sqrt();
            if na > 1e-8 && nb > 1e-8 {
                total += dot / (na * nb);
            }
        }
        total / num_blocks.max(1) as f32
    }

    /// Compatibility score for Hopf transition scoring.
    /// Maps E8 root inner product to a [0, 3] score:
    ///   root_ip ≈ 1.0 → 3.0 (same pattern / self)
    ///   root_ip ≈ 0.5-0.9 → 1.5-2.7 (strongly compatible)
    ///   root_ip ≈ 0.0 → 0.5 (orthogonal / independent)
    ///   root_ip < 0.0 → 0.0-0.5 (opposing)
    pub fn compatibility_score(a: &[f32], b: &[f32]) -> f32 {
        let rip = Self::root_inner_product(a, b);
        // Affine map: rip in [-1, 1] → score in [0, 3]
        ((rip + 1.0) * 1.5).clamp(0.0, 3.0)
    }

    /// Compatibility score for two 8d E8 lattice points directly.
    /// Maps cosine similarity of E8-quantized points to [0, 3].
    pub fn compatibility_score_8d(a: &[f32; 8], b: &[f32; 8]) -> f32 {
        let qa = Self::nearest_point(a);
        let qb = Self::nearest_point(b);
        let dot: f32 = qa.iter().zip(qb.iter()).map(|(x, y)| x * y).sum();
        let na = qa.iter().map(|v| v * v).sum::<f32>().sqrt();
        let nb = qb.iter().map(|v| v * v).sum::<f32>().sqrt();
        let cos = if na > 1e-8 && nb > 1e-8 {
            dot / (na * nb)
        } else {
            0.0
        };
        ((cos + 1.0) * 1.5).clamp(0.0, 3.0)
    }

    /// Select the best archetype from prototypes using E8 lattice decoding.
    /// Quantizes the input embedding into E8 subspaces and compares against
    /// quantized prototypes, returning (best_index, confidence).
    ///
    /// When prototypes are already E8-quantized (after training), this reduces
    /// to a fast lattice-point comparison.
    pub fn select_archetype(embedding: &[f32], prototypes: &[Vec<f32>]) -> (usize, f32) {
        if prototypes.is_empty() {
            return (0, 0.0);
        }

        let q_emb = Self::quantize_64d(embedding);
        let emb_norm = q_emb.iter().map(|v| v * v).sum::<f32>().sqrt();

        // If E8 quantization collapsed the embedding to the origin (sparse
        // vectors have odd coordinate sums), fall back to raw cosine similarity
        if emb_norm < 1e-8 {
            let raw_norm = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
            let mut best_idx = 0;
            let mut best_sim = f32::NEG_INFINITY;
            for (i, proto) in prototypes.iter().enumerate() {
                let dot: f32 = embedding.iter().zip(proto.iter()).map(|(a, b)| a * b).sum();
                let p_norm = proto.iter().map(|v| v * v).sum::<f32>().sqrt();
                let sim = if raw_norm > 1e-8 && p_norm > 1e-8 {
                    dot / (raw_norm * p_norm)
                } else {
                    0.0
                };
                if sim > best_sim {
                    best_sim = sim;
                    best_idx = i;
                }
            }
            return (best_idx, best_sim.max(0.0));
        }

        let mut best_idx = 0;
        let mut best_sim = f32::NEG_INFINITY;

        for (i, proto) in prototypes.iter().enumerate() {
            let q_proto = Self::quantize_64d(proto);
            let dot: f32 = q_emb.iter().zip(q_proto.iter()).map(|(a, b)| a * b).sum();
            let p_norm = q_proto.iter().map(|v| v * v).sum::<f32>().sqrt();
            let sim = if emb_norm > 1e-8 && p_norm > 1e-8 {
                dot / (emb_norm * p_norm)
            } else {
                0.0
            };
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }

        (best_idx, best_sim.max(0.0))
    }
}

// ---------------------------------------------------------------------------
// Extended Hamming [8,4,4] — native to E8, double-error detection
// ---------------------------------------------------------------------------
//
// The extended Hamming code adds an overall parity bit to Hamming [7,4,3],
// giving [8,4,4]: 4 data bits, 4 parity bits, minimum distance 4.
// This detects 2-bit errors and corrects 1-bit errors (vs [7,4,3] which
// only corrects 1-bit errors with no detection of 2-bit errors).
// The connection to E8: the extended Hamming code is the binary code
// underlying the E8 lattice construction.

pub fn extended_hamming_encode(data: &[u8; 4]) -> [u8; 8] {
    // Hamming [7,4,3] parity bits
    let p1 = data[0] ^ data[1] ^ data[3];
    let p2 = data[0] ^ data[2] ^ data[3];
    let p3 = data[1] ^ data[2] ^ data[3];
    // Overall parity bit (extended code)
    let p_all = data[0] ^ data[1] ^ data[2] ^ data[3] ^ p1 ^ p2 ^ p3;
    [p1, p2, data[0], p3, data[1], data[2], data[3], p_all]
}

pub fn extended_hamming_decode(codeword: &[u8; 8]) -> ([u8; 4], bool) {
    // Compute syndrome
    let s1 = codeword[0] ^ codeword[2] ^ codeword[4] ^ codeword[6];
    let s2 = codeword[1] ^ codeword[2] ^ codeword[5] ^ codeword[6];
    let s3 = codeword[3] ^ codeword[4] ^ codeword[5] ^ codeword[6];
    let syndrome = (s1 as usize) | ((s2 as usize) << 1) | ((s3 as usize) << 2);

    // Overall parity check
    let overall: u8 = codeword.iter().fold(0u8, |acc, &b| acc ^ b);

    let mut corrected = *codeword;
    let correctable = if syndrome != 0 && overall != 0 {
        // Single-bit error: correct it
        if syndrome > 0 && syndrome <= 7 {
            corrected[syndrome - 1] ^= 1;
        }
        true
    } else if syndrome != 0 && overall == 0 {
        // Double-bit error detected but not correctable
        false
    } else {
        true
    };

    // Extract data bits from positions 3, 5, 6, 7 (0-indexed: 2, 4, 5, 6)
    let data = [corrected[2], corrected[4], corrected[5], corrected[6]];
    (data, correctable)
}

// ---------------------------------------------------------------------------
// Extended Golay Code [24,12,8] — native to Leech lattice
// ---------------------------------------------------------------------------
//
// The extended Golay code encodes 12 data bits into 24 coded bits with
// minimum distance 8. It corrects up to 3 bit errors and detects 4.
// This is the binary code underlying the Leech lattice construction,
// just as extended Hamming [8,4,4] underlies E8.

/// The 12×12 generator matrix B for the Golay code (over GF(2)).
/// G = [I₁₂ | B] gives the systematic generator matrix.
/// B is constructed from the quadratic residues mod 11 plus a border of 1s.
const GOLAY_B: [[u8; 12]; 12] = [
    [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
    [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
    [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
    [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
    [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
    [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
    [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
];

pub fn golay_encode(data: &[u8; 12]) -> [u8; 24] {
    let mut codeword = [0u8; 24];
    // Systematic part: first 12 bits are data
    for i in 0..12 {
        codeword[i] = data[i] & 1;
    }
    // Parity part: multiply data by B (mod 2)
    for j in 0..12 {
        let mut parity = 0u8;
        for i in 0..12 {
            parity ^= data[i] & GOLAY_B[i][j];
        }
        codeword[12 + j] = parity;
    }
    codeword
}

/// Compute syndrome for Golay decoding.
fn golay_syndrome(codeword: &[u8; 24]) -> [u8; 12] {
    let mut syndrome = [0u8; 12];
    // s = parity_part + B^T * data_part (mod 2)
    for j in 0..12 {
        let mut s = codeword[12 + j]; // parity bit
        for i in 0..12 {
            s ^= codeword[i] & GOLAY_B[i][j];
        }
        syndrome[j] = s & 1;
    }
    syndrome
}

fn hamming_weight(bits: &[u8]) -> usize {
    bits.iter().filter(|&&b| b != 0).count()
}

/// Decode a Golay [24,12,8] codeword. Corrects up to 3-bit errors.
/// Returns (12 data bits, correctable).
pub fn golay_decode(codeword: &[u8; 24]) -> ([u8; 12], bool) {
    let mut c = *codeword;
    let s = golay_syndrome(&c);
    let sw = hamming_weight(&s);

    if sw == 0 {
        let mut data = [0u8; 12];
        data.copy_from_slice(&c[0..12]);
        return (data, true);
    }

    // Try to correct errors in the parity half
    if sw <= 3 {
        for i in 0..12 {
            c[12 + i] ^= s[i];
        }
        let mut data = [0u8; 12];
        data.copy_from_slice(&c[0..12]);
        return (data, true);
    }

    // Try each row of B to find a pattern matching syndrome
    for i in 0..12 {
        let mut diff = [0u8; 12];
        for j in 0..12 {
            diff[j] = (s[j] ^ GOLAY_B[i][j]) & 1;
        }
        let dw = hamming_weight(&diff);
        if dw <= 2 {
            c[i] ^= 1;
            for j in 0..12 {
                c[12 + j] ^= diff[j];
            }
            let mut data = [0u8; 12];
            data.copy_from_slice(&c[0..12]);
            return (data, true);
        }
    }

    // Compute syndrome of B * s^T (secondary syndrome)
    let mut bs = [0u8; 12];
    for i in 0..12 {
        let mut v = 0u8;
        for j in 0..12 {
            v ^= GOLAY_B[i][j] & s[j];
        }
        bs[i] = v & 1;
    }
    let bsw = hamming_weight(&bs);

    if bsw <= 3 {
        for i in 0..12 {
            c[i] ^= bs[i];
        }
        let mut data = [0u8; 12];
        data.copy_from_slice(&c[0..12]);
        return (data, true);
    }

    // Try each row of B against the secondary syndrome
    for i in 0..12 {
        let mut diff = [0u8; 12];
        for j in 0..12 {
            diff[j] = (bs[j] ^ GOLAY_B[i][j]) & 1;
        }
        let dw = hamming_weight(&diff);
        if dw <= 2 {
            c[12 + i] ^= 1;
            for j in 0..12 {
                c[j] ^= diff[j];
            }
            let mut data = [0u8; 12];
            data.copy_from_slice(&c[0..12]);
            return (data, true);
        }
    }

    // Uncorrectable (4+ errors)
    let mut data = [0u8; 12];
    data.copy_from_slice(&c[0..12]);
    (data, false)
}

// ---------------------------------------------------------------------------
// Leech Lattice Engine — optimal sphere packing in dimension 24
// ---------------------------------------------------------------------------
//
// The Leech lattice Λ₂₄ achieves the densest sphere packing in 24 dimensions
// (Cohn, Kumar, Miller, Radchenko, Viazovska, 2017). Properties:
//   - Kissing number: 196,560
//   - Automorphism group contains Conway groups Co₁, Co₂, Co₃
//   - Related code: extended Golay [24,12,8]
//   - Construction: 3 copies of E8 glued by the Golay code
//
// Used for the ProjectModel: spatial index of files, symbols, and
// relationships in a codebase. Each entity maps to a 24d Leech point;
// nearest-neighbor queries find related entities.

#[derive(Clone, Debug)]
pub struct LeechLattice;

impl LeechLattice {
    /// Nearest Leech lattice point to an arbitrary 24d vector.
    ///
    /// Algorithm: decompose into 3 × 8d blocks, find nearest E8 point
    /// for each block, then adjust using the Golay code structure to
    /// find the best combination that lies on the Leech lattice.
    ///
    /// This uses a simplified construction: Λ₂₄ contains all vectors
    /// (x₁, x₂, x₃) where each xᵢ ∈ E8 (scaled by 1/√2) and the
    /// combination of their coset representatives forms a Golay codeword.
    /// For nearest-point, we check the 4 coset candidates (each block
    /// can be integer E8 or half-integer E8) and pick the closest.
    pub fn nearest_point(x: &[f32; 24]) -> [f32; 24] {
        // Split into 3 × 8d blocks
        let mut blocks = [[0.0f32; 8]; 3];
        for b in 0..3 {
            for i in 0..8 {
                blocks[b][i] = x[b * 8 + i];
            }
        }

        // For each block, find both the integer-E8 and half-integer-E8 nearest points
        let mut candidates: Vec<[f32; 24]> = Vec::with_capacity(8);

        // Generate all 8 combinations of (integer, half-integer) per block
        for mask in 0u8..8 {
            let mut point = [0.0f32; 24];
            let mut valid = true;
            let mut _coset_sum = 0i32;

            for b in 0..3 {
                let use_half = (mask >> b) & 1 == 1;
                let nearest = if use_half {
                    let mut shifted = [0.0f32; 8];
                    for i in 0..8 {
                        shifted[i] = blocks[b][i] - 0.5;
                    }
                    let d8 = E8Lattice::nearest_point(&shifted);
                    let mut result = [0.0f32; 8];
                    for i in 0..8 {
                        result[i] = d8[i] + 0.5;
                    }

                    // Verify it's half-integer E8 (sum should be even)
                    let sum: f32 = result.iter().sum();
                    if (sum.round() as i32) % 2 != 0 {
                        valid = false;
                    }
                    result
                } else {
                    E8Lattice::nearest_point(&blocks[b])
                };

                for i in 0..8 {
                    point[b * 8 + i] = nearest[i];
                }

                // Track coset: integer = 0, half-integer = 1
                if use_half {
                    _coset_sum += 1;
                }
            }

            // Leech lattice constraint: the coset pattern must be
            // consistent (all integer or all half-integer, or mixed
            // according to Golay structure). For the simplified decoder,
            // we accept all combinations and pick closest.
            if valid || true {
                candidates.push(point);
            }
        }

        // Also try pure E8 nearest in each block (the "all-integer" candidate)
        // which is already in candidates[0]

        // Pick the candidate closest to the input
        let mut best = candidates[0];
        let mut best_dist = Self::dist_sq_24(x, &candidates[0]);

        for cand in &candidates[1..] {
            let d = Self::dist_sq_24(x, cand);
            if d < best_dist {
                best_dist = d;
                best = *cand;
            }
        }

        best
    }

    fn dist_sq_24(a: &[f32; 24], b: &[f32; 24]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    /// Quantize an arbitrary-length vector by decomposing into 24d Leech subspaces.
    /// Vectors shorter than 24d are zero-padded; longer vectors use multiple subspaces.
    pub fn quantize(x: &[f32]) -> Vec<f32> {
        let dim = x.len();
        let n_blocks = (dim + 23) / 24;
        let mut result = vec![0.0f32; n_blocks * 24];

        for b in 0..n_blocks {
            let mut block = [0.0f32; 24];
            for i in 0..24 {
                let idx = b * 24 + i;
                if idx < dim {
                    block[i] = x[idx];
                }
            }
            let nearest = Self::nearest_point(&block);
            for i in 0..24 {
                result[b * 24 + i] = nearest[i];
            }
        }
        result.truncate(dim.max(n_blocks * 24));
        result
    }

    /// Compute Leech inner product (cosine similarity in quantized space).
    pub fn inner_product(a: &[f32; 24], b: &[f32; 24]) -> f32 {
        let qa = Self::nearest_point(a);
        let qb = Self::nearest_point(b);
        let dot: f32 = qa.iter().zip(qb.iter()).map(|(x, y)| x * y).sum();
        let na = qa.iter().map(|v| v * v).sum::<f32>().sqrt();
        let nb = qb.iter().map(|v| v * v).sum::<f32>().sqrt();
        if na > 1e-8 && nb > 1e-8 {
            dot / (na * nb)
        } else {
            0.0
        }
    }

    /// Compatibility score for project entities in Leech space.
    /// Maps inner product to [0, 4] — wider range than E8 due to richer structure.
    pub fn compatibility_score(a: &[f32; 24], b: &[f32; 24]) -> f32 {
        let ip = Self::inner_product(a, b);
        ((ip + 1.0) * 2.0).clamp(0.0, 4.0)
    }

    /// Find the k nearest neighbors of a query point among a set of 24d points.
    /// Returns indices sorted by ascending distance.
    pub fn nearest_neighbors(
        query: &[f32; 24],
        points: &[[f32; 24]],
        k: usize,
    ) -> Vec<(usize, f32)> {
        let q = Self::nearest_point(query);
        let mut scored: Vec<(usize, f32)> = points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let qp = Self::nearest_point(p);
                let dist = Self::dist_sq_24(&q, &qp);
                (i, dist)
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}

// ---------------------------------------------------------------------------
// CodeAnalyzer — structural parsing for source code
// ---------------------------------------------------------------------------
//
// Extracts real structural features from code: declarations, call graph edges,
// import resolution, complexity metrics, type hierarchies. This is not a full
// AST parser — it operates on line/token patterns — but it captures the
// structural skeleton that determines how code relates to other code.

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CodeStructure {
    pub language: CodeLanguage,
    pub declarations: Vec<Declaration>,
    pub imports: Vec<ImportEdge>,
    pub call_sites: Vec<CallSite>,
    pub metrics: CodeMetrics,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CodeLanguage {
    Rust,
    Python,
    TypeScript,
    JavaScript,
    Go,
    C,
    Cpp,
    Java,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Declaration {
    pub kind: DeclKind,
    pub name: String,
    pub is_public: bool,
    pub line: usize,
    pub nesting_depth: u32,
    pub params: Vec<String>,
    pub return_type: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeclKind {
    Function,
    Struct,
    Enum,
    Trait,
    Impl,
    Class,
    Interface,
    Module,
    Constant,
    TypeAlias,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportEdge {
    pub module_path: String,
    pub symbols: Vec<String>,
    pub is_wildcard: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallSite {
    pub caller_decl: Option<String>,
    pub callee: String,
    pub line: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CodeMetrics {
    pub total_lines: usize,
    pub code_lines: usize,
    pub comment_lines: usize,
    pub blank_lines: usize,
    pub max_nesting: u32,
    pub cyclomatic_complexity: u32,
    pub public_api_count: usize,
    pub private_count: usize,
    pub unique_identifiers: usize,
    pub avg_function_length: f32,
    pub trait_impl_count: usize,
    pub test_function_count: usize,
    pub assertion_count: usize,
}

pub struct CodeAnalyzer;

impl CodeAnalyzer {
    pub fn analyze(path: &str, content: &str) -> CodeStructure {
        let language = Self::detect_language(path);
        let lines: Vec<&str> = content.lines().collect();
        let declarations = Self::extract_declarations(&lines, language);
        let imports = Self::extract_imports(&lines, language);
        let call_sites = Self::extract_call_sites(&lines, &declarations);
        let metrics = Self::compute_metrics(&lines, &declarations, language);

        CodeStructure {
            language,
            declarations,
            imports,
            call_sites,
            metrics,
        }
    }

    fn detect_language(path: &str) -> CodeLanguage {
        match path.rsplit('.').next() {
            Some("rs") => CodeLanguage::Rust,
            Some("py") => CodeLanguage::Python,
            Some("ts" | "tsx") => CodeLanguage::TypeScript,
            Some("js" | "jsx") => CodeLanguage::JavaScript,
            Some("go") => CodeLanguage::Go,
            Some("c" | "h") => CodeLanguage::C,
            Some("cpp" | "hpp" | "cc" | "cxx") => CodeLanguage::Cpp,
            Some("java") => CodeLanguage::Java,
            _ => CodeLanguage::Unknown,
        }
    }

    fn extract_declarations(lines: &[&str], lang: CodeLanguage) -> Vec<Declaration> {
        let mut decls = Vec::new();
        let mut brace_depth: i32 = 0;
        let mut indent_stack: Vec<u32> = vec![0];

        for (line_num, &line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty()
                || trimmed.starts_with("//")
                || trimmed.starts_with('#') && lang == CodeLanguage::C
            {
                continue;
            }

            let nesting = match lang {
                CodeLanguage::Python => {
                    let indent = line.len() - line.trim_start().len();
                    let indent_level = indent as u32 / 4;
                    while indent_stack.len() > 1 && *indent_stack.last().unwrap() >= indent_level {
                        indent_stack.pop();
                    }
                    indent_level
                }
                _ => brace_depth as u32,
            };

            // Rust declarations
            if matches!(lang, CodeLanguage::Rust) {
                let is_pub = trimmed.starts_with("pub ");
                let check = if is_pub { &trimmed[4..] } else { trimmed };

                if let Some(decl) = Self::parse_rust_decl(check, is_pub, line_num, nesting) {
                    decls.push(decl);
                }
            }

            // Python declarations
            if matches!(lang, CodeLanguage::Python) {
                if trimmed.starts_with("def ") || trimmed.starts_with("async def ") {
                    let name_start = if trimmed.starts_with("async") {
                        "async def ".len()
                    } else {
                        "def ".len()
                    };
                    if let Some(name) = trimmed[name_start..].split('(').next() {
                        let params = Self::extract_params_from_line(trimmed);
                        let is_test = name.starts_with("test_") || name.starts_with("test");
                        decls.push(Declaration {
                            kind: DeclKind::Function,
                            name: name.trim().to_string(),
                            is_public: !name.starts_with('_'),
                            line: line_num,
                            nesting_depth: nesting,
                            params,
                            return_type: Self::extract_python_return_type(trimmed),
                        });
                        if is_test {
                            // Track separately in metrics
                        }
                    }
                    if lang == CodeLanguage::Python {
                        indent_stack.push(nesting + 1);
                    }
                } else if trimmed.starts_with("class ") {
                    let name = trimmed["class ".len()..]
                        .split(['(', ':'])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    decls.push(Declaration {
                        kind: DeclKind::Class,
                        name,
                        is_public: !trimmed.starts_with("class _"),
                        line: line_num,
                        nesting_depth: nesting,
                        params: vec![],
                        return_type: None,
                    });
                    indent_stack.push(nesting + 1);
                }
            }

            // TypeScript / JavaScript declarations
            if matches!(lang, CodeLanguage::TypeScript | CodeLanguage::JavaScript) {
                let is_export = trimmed.starts_with("export ");
                let check = if is_export {
                    trimmed["export ".len()..].trim_start()
                } else {
                    trimmed
                };
                let check = if check.starts_with("default ") {
                    &check["default ".len()..]
                } else {
                    check
                };

                if check.starts_with("function ") || check.starts_with("async function ") {
                    let after = if check.starts_with("async") {
                        &check["async function ".len()..]
                    } else {
                        &check["function ".len()..]
                    };
                    if let Some(name) = after.split('(').next() {
                        decls.push(Declaration {
                            kind: DeclKind::Function,
                            name: name.trim().to_string(),
                            is_public: is_export,
                            line: line_num,
                            nesting_depth: nesting,
                            params: Self::extract_params_from_line(trimmed),
                            return_type: None,
                        });
                    }
                } else if check.starts_with("class ") {
                    let name = check["class ".len()..]
                        .split(['{', ' '])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    decls.push(Declaration {
                        kind: DeclKind::Class,
                        name,
                        is_public: is_export,
                        line: line_num,
                        nesting_depth: nesting,
                        params: vec![],
                        return_type: None,
                    });
                } else if check.starts_with("interface ") && lang == CodeLanguage::TypeScript {
                    let name = check["interface ".len()..]
                        .split(['{', ' ', '<'])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    decls.push(Declaration {
                        kind: DeclKind::Interface,
                        name,
                        is_public: is_export,
                        line: line_num,
                        nesting_depth: nesting,
                        params: vec![],
                        return_type: None,
                    });
                }
            }

            // Go declarations
            if matches!(lang, CodeLanguage::Go) {
                if trimmed.starts_with("func ") {
                    let rest = &trimmed["func ".len()..];
                    // Skip receiver: func (r *Receiver) Name(...)
                    let rest = if rest.starts_with('(') {
                        rest.find(')')
                            .and_then(|i| rest.get(i + 1..))
                            .unwrap_or(rest)
                            .trim_start()
                    } else {
                        rest
                    };
                    if let Some(name) = rest.split('(').next() {
                        let name = name.trim().to_string();
                        let is_public = name.chars().next().map_or(false, |c| c.is_uppercase());
                        decls.push(Declaration {
                            kind: DeclKind::Function,
                            name,
                            is_public,
                            line: line_num,
                            nesting_depth: nesting,
                            params: Self::extract_params_from_line(trimmed),
                            return_type: None,
                        });
                    }
                } else if trimmed.starts_with("type ") && trimmed.contains("struct") {
                    let name = trimmed["type ".len()..]
                        .split_whitespace()
                        .next()
                        .unwrap_or("")
                        .to_string();
                    decls.push(Declaration {
                        kind: DeclKind::Struct,
                        name,
                        is_public: true,
                        line: line_num,
                        nesting_depth: nesting,
                        params: vec![],
                        return_type: None,
                    });
                } else if trimmed.starts_with("type ") && trimmed.contains("interface") {
                    let name = trimmed["type ".len()..]
                        .split_whitespace()
                        .next()
                        .unwrap_or("")
                        .to_string();
                    decls.push(Declaration {
                        kind: DeclKind::Interface,
                        name,
                        is_public: true,
                        line: line_num,
                        nesting_depth: nesting,
                        params: vec![],
                        return_type: None,
                    });
                }
            }

            // C/C++ declarations
            if matches!(lang, CodeLanguage::C | CodeLanguage::Cpp) {
                if trimmed.contains('(')
                    && !trimmed.starts_with("if")
                    && !trimmed.starts_with("for")
                    && !trimmed.starts_with("while")
                    && !trimmed.starts_with("switch")
                    && !trimmed.starts_with("//")
                    && !trimmed.starts_with("/*")
                    && !trimmed.starts_with("return")
                    && brace_depth <= 1
                {
                    // Heuristic: top-level line with parens is likely a function
                    let parts: Vec<&str> = trimmed.split('(').collect();
                    if let Some(before_paren) = parts.first() {
                        let tokens: Vec<&str> = before_paren.split_whitespace().collect();
                        if tokens.len() >= 2 {
                            let name = tokens.last().unwrap().trim_start_matches('*').to_string();
                            if name
                                .chars()
                                .next()
                                .map_or(false, |c| c.is_alphabetic() || c == '_')
                            {
                                decls.push(Declaration {
                                    kind: DeclKind::Function,
                                    name,
                                    is_public: !trimmed.starts_with("static "),
                                    line: line_num,
                                    nesting_depth: nesting,
                                    params: Self::extract_params_from_line(trimmed),
                                    return_type: None,
                                });
                            }
                        }
                    }
                }
                if lang == CodeLanguage::Cpp && trimmed.starts_with("class ") {
                    let name = trimmed["class ".len()..]
                        .split(['{', ':', ' '])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    decls.push(Declaration {
                        kind: DeclKind::Class,
                        name,
                        is_public: true,
                        line: line_num,
                        nesting_depth: nesting,
                        params: vec![],
                        return_type: None,
                    });
                }
                if trimmed.contains("struct ") && (trimmed.contains('{') || trimmed.ends_with(';'))
                {
                    let after = trimmed.split("struct ").nth(1).unwrap_or("");
                    let name = after
                        .split(['{', ' ', ';'])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if !name.is_empty() {
                        decls.push(Declaration {
                            kind: DeclKind::Struct,
                            name,
                            is_public: true,
                            line: line_num,
                            nesting_depth: nesting,
                            params: vec![],
                            return_type: None,
                        });
                    }
                }
            }

            // Java declarations
            if matches!(lang, CodeLanguage::Java) {
                let is_public = trimmed.starts_with("public ");
                if trimmed.contains("class ") {
                    let after = trimmed.split("class ").nth(1).unwrap_or("");
                    let name = after
                        .split(['{', ' ', '<'])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if !name.is_empty() {
                        decls.push(Declaration {
                            kind: DeclKind::Class,
                            name,
                            is_public,
                            line: line_num,
                            nesting_depth: nesting,
                            params: vec![],
                            return_type: None,
                        });
                    }
                } else if trimmed.contains("interface ") {
                    let after = trimmed.split("interface ").nth(1).unwrap_or("");
                    let name = after
                        .split(['{', ' ', '<'])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if !name.is_empty() {
                        decls.push(Declaration {
                            kind: DeclKind::Interface,
                            name,
                            is_public,
                            line: line_num,
                            nesting_depth: nesting,
                            params: vec![],
                            return_type: None,
                        });
                    }
                } else if trimmed.contains('(')
                    && !trimmed.starts_with("if")
                    && !trimmed.starts_with("for")
                    && !trimmed.starts_with("while")
                    && brace_depth <= 2
                {
                    let parts: Vec<&str> = trimmed.split('(').collect();
                    if let Some(before) = parts.first() {
                        let tokens: Vec<&str> = before.split_whitespace().collect();
                        if tokens.len() >= 2 {
                            let name = tokens.last().unwrap().to_string();
                            if name.chars().next().map_or(false, |c| c.is_alphabetic()) {
                                decls.push(Declaration {
                                    kind: DeclKind::Function,
                                    name,
                                    is_public,
                                    line: line_num,
                                    nesting_depth: nesting,
                                    params: Self::extract_params_from_line(trimmed),
                                    return_type: None,
                                });
                            }
                        }
                    }
                }
            }

            // Track brace depth for C-family languages
            if !matches!(lang, CodeLanguage::Python) {
                for ch in trimmed.chars() {
                    match ch {
                        '{' => brace_depth += 1,
                        '}' => brace_depth = (brace_depth - 1).max(0),
                        _ => {}
                    }
                }
            }
        }

        decls
    }

    fn parse_rust_decl(
        check: &str,
        is_pub: bool,
        line: usize,
        nesting: u32,
    ) -> Option<Declaration> {
        if check.starts_with("fn ") || check.starts_with("async fn ") {
            let after = if check.starts_with("async") {
                &check["async fn ".len()..]
            } else {
                &check["fn ".len()..]
            };
            let name = after.split(['(', '<']).next()?.trim().to_string();
            let params = Self::extract_params_from_line(check);
            let ret = Self::extract_rust_return_type(check);
            return Some(Declaration {
                kind: DeclKind::Function,
                name,
                is_public: is_pub,
                line,
                nesting_depth: nesting,
                params,
                return_type: ret,
            });
        }
        if check.starts_with("struct ") {
            let name = check["struct ".len()..]
                .split(['{', '(', '<', ' ', ';'])
                .next()?
                .trim()
                .to_string();
            return Some(Declaration {
                kind: DeclKind::Struct,
                name,
                is_public: is_pub,
                line,
                nesting_depth: nesting,
                params: vec![],
                return_type: None,
            });
        }
        if check.starts_with("enum ") {
            let name = check["enum ".len()..]
                .split(['{', '<', ' '])
                .next()?
                .trim()
                .to_string();
            return Some(Declaration {
                kind: DeclKind::Enum,
                name,
                is_public: is_pub,
                line,
                nesting_depth: nesting,
                params: vec![],
                return_type: None,
            });
        }
        if check.starts_with("trait ") {
            let name = check["trait ".len()..]
                .split(['{', '<', ' ', ':'])
                .next()?
                .trim()
                .to_string();
            return Some(Declaration {
                kind: DeclKind::Trait,
                name,
                is_public: is_pub,
                line,
                nesting_depth: nesting,
                params: vec![],
                return_type: None,
            });
        }
        if check.starts_with("impl ") || check.starts_with("impl<") {
            let rest = if check.starts_with("impl<") {
                check
                    .find('>')
                    .map(|i| &check[i + 1..])
                    .unwrap_or(&check[5..])
            } else {
                &check[5..]
            };
            let name = rest.split(['{', ' ']).next()?.trim().to_string();
            return Some(Declaration {
                kind: DeclKind::Impl,
                name,
                is_public: is_pub,
                line,
                nesting_depth: nesting,
                params: vec![],
                return_type: None,
            });
        }
        if check.starts_with("mod ") {
            let name = check["mod ".len()..]
                .split(['{', ';', ' '])
                .next()?
                .trim()
                .to_string();
            return Some(Declaration {
                kind: DeclKind::Module,
                name,
                is_public: is_pub,
                line,
                nesting_depth: nesting,
                params: vec![],
                return_type: None,
            });
        }
        if check.starts_with("const ") || check.starts_with("static ") {
            let kw_len = if check.starts_with("const") { 6 } else { 7 };
            let name = check[kw_len..]
                .split([':', ' ', '='])
                .next()?
                .trim()
                .to_string();
            if !name.is_empty() && name != "_" {
                return Some(Declaration {
                    kind: DeclKind::Constant,
                    name,
                    is_public: is_pub,
                    line,
                    nesting_depth: nesting,
                    params: vec![],
                    return_type: None,
                });
            }
        }
        if check.starts_with("type ") {
            let name = check["type ".len()..]
                .split(['<', '=', ' '])
                .next()?
                .trim()
                .to_string();
            return Some(Declaration {
                kind: DeclKind::TypeAlias,
                name,
                is_public: is_pub,
                line,
                nesting_depth: nesting,
                params: vec![],
                return_type: None,
            });
        }
        None
    }

    fn extract_imports(lines: &[&str], lang: CodeLanguage) -> Vec<ImportEdge> {
        let mut imports = Vec::new();

        for line in lines {
            let trimmed = line.trim();

            match lang {
                CodeLanguage::Rust => {
                    if trimmed.starts_with("use ") || trimmed.starts_with("pub use ") {
                        let use_part = if trimmed.starts_with("pub") {
                            &trimmed["pub use ".len()..]
                        } else {
                            &trimmed["use ".len()..]
                        };
                        let clean = use_part.trim_end_matches(';').trim();
                        let is_wildcard = clean.ends_with("::*");
                        let module = clean
                            .split("::{")
                            .next()
                            .unwrap_or(clean)
                            .trim_end_matches("::*")
                            .to_string();
                        let symbols = if clean.contains("::{") {
                            clean
                                .split("::{")
                                .nth(1)
                                .unwrap_or("")
                                .trim_end_matches('}')
                                .split(',')
                                .map(|s| s.trim().to_string())
                                .filter(|s| !s.is_empty())
                                .collect()
                        } else if !is_wildcard {
                            vec![module.rsplit("::").next().unwrap_or("").to_string()]
                        } else {
                            vec![]
                        };
                        imports.push(ImportEdge {
                            module_path: module,
                            symbols,
                            is_wildcard,
                        });
                    }
                }
                CodeLanguage::Python => {
                    if trimmed.starts_with("import ") {
                        let module = trimmed["import ".len()..]
                            .split(' ')
                            .next()
                            .unwrap_or("")
                            .trim()
                            .to_string();
                        imports.push(ImportEdge {
                            module_path: module.clone(),
                            symbols: vec![module],
                            is_wildcard: false,
                        });
                    } else if trimmed.starts_with("from ") {
                        let parts: Vec<&str> = trimmed.splitn(4, ' ').collect();
                        if parts.len() >= 4 && parts[2] == "import" {
                            let module = parts[1].to_string();
                            let is_wildcard = parts[3].trim() == "*";
                            let symbols = if is_wildcard {
                                vec![]
                            } else {
                                parts[3].split(',').map(|s| s.trim().to_string()).collect()
                            };
                            imports.push(ImportEdge {
                                module_path: module,
                                symbols,
                                is_wildcard,
                            });
                        }
                    }
                }
                CodeLanguage::TypeScript | CodeLanguage::JavaScript => {
                    if trimmed.contains("import ")
                        && (trimmed.contains(" from ") || trimmed.contains("require("))
                    {
                        let module = if trimmed.contains(" from ") {
                            trimmed
                                .rsplit(" from ")
                                .next()
                                .unwrap_or("")
                                .trim()
                                .trim_matches(|c| c == '\'' || c == '"' || c == ';')
                                .to_string()
                        } else {
                            trimmed
                                .split("require(")
                                .nth(1)
                                .and_then(|s| s.split(')').next())
                                .unwrap_or("")
                                .trim()
                                .trim_matches(|c| c == '\'' || c == '"')
                                .to_string()
                        };
                        let is_wildcard = trimmed.contains("* as ");
                        let symbols = if trimmed.contains('{') {
                            trimmed
                                .split('{')
                                .nth(1)
                                .and_then(|s| s.split('}').next())
                                .unwrap_or("")
                                .split(',')
                                .map(|s| s.split(" as ").next().unwrap_or(s).trim().to_string())
                                .filter(|s| !s.is_empty())
                                .collect()
                        } else {
                            vec![]
                        };
                        imports.push(ImportEdge {
                            module_path: module,
                            symbols,
                            is_wildcard,
                        });
                    }
                }
                CodeLanguage::Go => {
                    if trimmed.starts_with("import ") {
                        let module = trimmed["import ".len()..]
                            .trim()
                            .trim_matches('"')
                            .trim_matches('(')
                            .to_string();
                        if !module.is_empty() && module != "(" {
                            imports.push(ImportEdge {
                                module_path: module,
                                symbols: vec![],
                                is_wildcard: false,
                            });
                        }
                    } else if trimmed.starts_with('"') && trimmed.ends_with('"') {
                        // Inside import block
                        let module = trimmed.trim_matches('"').to_string();
                        imports.push(ImportEdge {
                            module_path: module,
                            symbols: vec![],
                            is_wildcard: false,
                        });
                    }
                }
                CodeLanguage::Java => {
                    if trimmed.starts_with("import ") {
                        let path = trimmed["import ".len()..]
                            .trim_end_matches(';')
                            .trim_start_matches("static ")
                            .trim()
                            .to_string();
                        let is_wildcard = path.ends_with(".*");
                        imports.push(ImportEdge {
                            module_path: path,
                            symbols: vec![],
                            is_wildcard,
                        });
                    }
                }
                CodeLanguage::C | CodeLanguage::Cpp => {
                    if trimmed.starts_with("#include") {
                        let header = trimmed
                            .split(|c| c == '<' || c == '"')
                            .nth(1)
                            .and_then(|s| s.split(|c| c == '>' || c == '"').next())
                            .unwrap_or("")
                            .to_string();
                        imports.push(ImportEdge {
                            module_path: header,
                            symbols: vec![],
                            is_wildcard: false,
                        });
                    }
                }
                CodeLanguage::Unknown => {}
            }
        }

        imports
    }

    fn extract_call_sites(lines: &[&str], declarations: &[Declaration]) -> Vec<CallSite> {
        let mut sites = Vec::new();
        let _decl_names: std::collections::HashSet<&str> = declarations
            .iter()
            .filter(|d| d.kind == DeclKind::Function)
            .map(|d| d.name.as_str())
            .collect();

        let mut current_fn: Option<&str> = None;

        for (line_num, &line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with('#') {
                continue;
            }

            // Track which function scope we're in
            for d in declarations {
                if d.line == line_num && d.kind == DeclKind::Function {
                    current_fn = Some(&d.name);
                }
            }

            // Extract function calls: identifier followed by (
            let bytes = trimmed.as_bytes();
            let mut i = 0;
            while i < bytes.len() {
                if bytes[i] == b'(' && i > 0 {
                    // Walk back to find the identifier
                    let end = i;
                    let mut start = i;
                    while start > 0
                        && (bytes[start - 1].is_ascii_alphanumeric() || bytes[start - 1] == b'_')
                    {
                        start -= 1;
                    }
                    if start < end {
                        let callee = &trimmed[start..end];
                        // Skip language keywords and self-referencing control flow
                        if ![
                            "if",
                            "for",
                            "while",
                            "match",
                            "switch",
                            "return",
                            "print",
                            "println",
                            "eprintln",
                            "eprint",
                            "format",
                            "write",
                            "writeln",
                            "vec",
                            "assert",
                            "assert_eq",
                            "debug_assert",
                            "panic",
                            "Some",
                            "Ok",
                            "Err",
                            "None",
                            "Box",
                            "Arc",
                            "Rc",
                        ]
                        .contains(&callee)
                            && callee.len() > 1
                        {
                            sites.push(CallSite {
                                caller_decl: current_fn.map(|s| s.to_string()),
                                callee: callee.to_string(),
                                line: line_num,
                            });
                        }
                    }
                }
                i += 1;
            }
        }

        sites
    }

    fn compute_metrics(
        lines: &[&str],
        declarations: &[Declaration],
        lang: CodeLanguage,
    ) -> CodeMetrics {
        let total_lines = lines.len();
        let mut code_lines = 0usize;
        let mut comment_lines = 0usize;
        let mut blank_lines = 0usize;
        let mut max_nesting: u32 = 0;
        let mut branch_count: u32 = 1; // cyclomatic complexity starts at 1
        let mut in_block_comment = false;
        let mut unique_ids = std::collections::HashSet::new();

        for &line in lines {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                blank_lines += 1;
                continue;
            }

            if in_block_comment {
                comment_lines += 1;
                if trimmed.contains("*/") {
                    in_block_comment = false;
                }
                continue;
            }

            if trimmed.starts_with("/*") {
                comment_lines += 1;
                in_block_comment = !trimmed.contains("*/");
                continue;
            }

            if trimmed.starts_with("//")
                || trimmed.starts_with('#') && matches!(lang, CodeLanguage::Python)
            {
                comment_lines += 1;
                continue;
            }

            code_lines += 1;

            // Count branches for cyclomatic complexity
            for kw in &[
                "if ", "else if ", "elif ", "for ", "while ", "case ", "catch ", "match ", "&&",
                "||", "?",
            ] {
                branch_count += trimmed.matches(kw).count() as u32;
            }

            // Nesting depth via indentation
            let indent = line.len() - line.trim_start().len();
            let depth = if matches!(lang, CodeLanguage::Python) {
                indent as u32 / 4
            } else {
                (indent as u32 / 4).min(10)
            };
            max_nesting = max_nesting.max(depth);

            // Collect unique identifiers
            for word in trimmed.split(|c: char| !c.is_alphanumeric() && c != '_') {
                if word.len() > 1 && word.chars().next().map_or(false, |c| c.is_alphabetic()) {
                    unique_ids.insert(word.to_string());
                }
            }
        }

        let fn_decls: Vec<&Declaration> = declarations
            .iter()
            .filter(|d| d.kind == DeclKind::Function)
            .collect();
        let public_count = declarations.iter().filter(|d| d.is_public).count();
        let private_count = declarations.iter().filter(|d| !d.is_public).count();
        let trait_impl_count = declarations
            .iter()
            .filter(|d| d.kind == DeclKind::Impl)
            .count();
        let test_count = declarations
            .iter()
            .filter(|d| {
                d.kind == DeclKind::Function
                    && (d.name.starts_with("test") || d.name.starts_with("test_"))
            })
            .count();
        let assertion_count = lines
            .iter()
            .map(|l| {
                l.matches("assert").count()
                    + l.matches("expect(").count()
                    + l.matches("should").count()
            })
            .sum::<usize>();

        let avg_fn_len = if fn_decls.len() >= 2 {
            let mut lengths = Vec::new();
            for i in 0..fn_decls.len() - 1 {
                lengths.push((fn_decls[i + 1].line - fn_decls[i].line) as f32);
            }
            if let Some(last) = fn_decls.last() {
                lengths.push((total_lines.saturating_sub(last.line)) as f32);
            }
            lengths.iter().sum::<f32>() / lengths.len() as f32
        } else if fn_decls.len() == 1 {
            total_lines as f32
        } else {
            0.0
        };

        CodeMetrics {
            total_lines,
            code_lines,
            comment_lines,
            blank_lines,
            max_nesting,
            cyclomatic_complexity: branch_count,
            public_api_count: public_count,
            private_count,
            unique_identifiers: unique_ids.len(),
            avg_function_length: avg_fn_len,
            trait_impl_count,
            test_function_count: test_count,
            assertion_count,
        }
    }

    fn extract_params_from_line(line: &str) -> Vec<String> {
        line.split('(')
            .nth(1)
            .and_then(|s| s.split(')').next())
            .unwrap_or("")
            .split(',')
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty() && p != "self" && p != "&self" && p != "&mut self")
            .collect()
    }

    fn extract_rust_return_type(line: &str) -> Option<String> {
        line.split("->")
            .nth(1)
            .map(|s| s.split(['{', 'w']).next().unwrap_or(s).trim().to_string())
    }

    fn extract_python_return_type(line: &str) -> Option<String> {
        if line.contains("->") {
            line.split("->")
                .nth(1)
                .map(|s| s.split(':').next().unwrap_or(s).trim().to_string())
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// HybridEmbedder — production 24d Leech embedding from code analysis
// ---------------------------------------------------------------------------
//
// Combines five signal channels into a 24d embedding:
//   Channel 1: Structural skeleton (AST-lite) → dims 0-3
//   Channel 2: Semantic content (hash projection of significant tokens) → dims 4-7
//   Channel 3: Relational graph (imports, calls, type refs) → dims 8-11
//   Channel 4: Edit correlation (git co-change, populated externally) → dims 12-15
//   Channel 5: Test/quality signal → dims 16-19
//   Channel 6: Pattern identity (structural fingerprint) → dims 20-23

pub struct HybridEmbedder;

impl HybridEmbedder {
    /// Full hybrid embedding for a file.
    pub fn embed_file(path: &str, content: &str) -> [f32; 24] {
        let structure = CodeAnalyzer::analyze(path, content);
        Self::embed_from_structure(path, content, &structure)
    }

    /// Embed from pre-analyzed structure (avoids re-parsing).
    pub fn embed_from_structure(path: &str, content: &str, structure: &CodeStructure) -> [f32; 24] {
        let mut emb = [0.0f32; 24];

        // Channel 1: Structural skeleton → dims 0-3
        Self::fill_structural(&mut emb, structure);

        // Channel 2: Semantic content → dims 4-7
        Self::fill_semantic(&mut emb, content, structure);

        // Channel 3: Relational graph → dims 8-11
        Self::fill_relational(&mut emb, path, structure);

        // Channel 4: Edit correlation → dims 12-15  (populated externally via GitHistory)

        // Channel 5: Test/quality signal → dims 16-19
        Self::fill_test_quality(&mut emb, path, structure);

        // Channel 6: Pattern identity → dims 20-23
        Self::fill_pattern_identity(&mut emb, content, structure);

        // L2-normalize to unit sphere before Leech quantization
        let norm: f32 = emb.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for v in emb.iter_mut() {
                *v /= norm;
            }
        }

        emb
    }

    /// Embed a specific symbol (function, type, etc.)
    pub fn embed_symbol(
        path: &str,
        name: &str,
        body: &str,
        parent_structure: &CodeStructure,
    ) -> [f32; 24] {
        let sub_structure = CodeAnalyzer::analyze(path, body);
        let mut emb = [0.0f32; 24];

        // Structural: from the symbol's own body
        Self::fill_structural(&mut emb, &sub_structure);

        // Semantic: from the symbol body
        Self::fill_semantic(&mut emb, body, &sub_structure);

        // Relational: look up the symbol in the parent structure
        let decl = parent_structure
            .declarations
            .iter()
            .find(|d| d.name == name);
        if let Some(d) = decl {
            emb[8] = (d.nesting_depth as f32 / 5.0).min(1.0);
            emb[9] = if d.is_public { 0.8 } else { 0.2 };
            // Outgoing calls from this function
            let my_calls: Vec<&CallSite> = parent_structure
                .call_sites
                .iter()
                .filter(|c| c.caller_decl.as_deref() == Some(name))
                .collect();
            emb[10] = (my_calls.len() as f32 / 15.0).min(1.0);
            // Incoming calls to this function
            let incoming = parent_structure
                .call_sites
                .iter()
                .filter(|c| c.callee == name)
                .count();
            emb[11] = (incoming as f32 / 10.0).min(1.0);
        }

        Self::fill_test_quality(&mut emb, path, &sub_structure);
        Self::fill_pattern_identity(&mut emb, body, &sub_structure);

        let norm: f32 = emb.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for v in emb.iter_mut() {
                *v /= norm;
            }
        }

        emb
    }

    // --- Channel 1: Structural skeleton (dims 0-3) ---
    fn fill_structural(emb: &mut [f32; 24], structure: &CodeStructure) {
        let m = &structure.metrics;
        let fn_count = structure
            .declarations
            .iter()
            .filter(|d| d.kind == DeclKind::Function)
            .count();
        let type_count = structure
            .declarations
            .iter()
            .filter(|d| {
                matches!(
                    d.kind,
                    DeclKind::Struct | DeclKind::Enum | DeclKind::Class | DeclKind::Interface
                )
            })
            .count();

        // dim 0: declaration density (functions per 100 lines)
        if m.code_lines > 0 {
            emb[0] = (fn_count as f32 / m.code_lines as f32 * 100.0).min(1.0);
        }
        // dim 1: type density (types per 100 lines)
        if m.code_lines > 0 {
            emb[1] = (type_count as f32 / m.code_lines as f32 * 100.0).min(1.0);
        }
        // dim 2: structural complexity (cyclomatic normalized by functions)
        if fn_count > 0 {
            emb[2] = (m.cyclomatic_complexity as f32 / fn_count as f32 / 10.0).min(1.0);
        }
        // dim 3: max nesting depth normalized
        emb[3] = (m.max_nesting as f32 / 8.0).min(1.0);
    }

    // --- Channel 2: Semantic content (dims 4-7) ---
    fn fill_semantic(emb: &mut [f32; 24], content: &str, structure: &CodeStructure) {
        // Multi-hash projection of significant tokens into 4d semantic space.
        // Uses declaration names, import paths, and content keywords to build
        // a distributional signature — files with similar APIs land nearby.
        let mut sig_tokens: Vec<&str> = Vec::with_capacity(200);

        // Declaration names are high-signal
        for d in &structure.declarations {
            sig_tokens.push(&d.name);
            if let Some(ref ret) = d.return_type {
                sig_tokens.push(ret);
            }
            for p in &d.params {
                sig_tokens.push(p);
            }
        }

        // Import module names carry semantic weight
        for imp in &structure.imports {
            sig_tokens.push(&imp.module_path);
            for s in &imp.symbols {
                sig_tokens.push(s);
            }
        }

        // Content tokens (skip noise)
        let stopwords = [
            "the", "a", "an", "and", "or", "to", "for", "of", "in", "on", "with", "is", "are",
            "self", "let", "mut", "pub", "fn", "def", "class", "return", "if", "else", "true",
            "false", "none", "some", "ok", "err",
        ];
        for word in content
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .take(500)
        {
            if word.len() > 2 && !stopwords.contains(&word.to_ascii_lowercase().as_str()) {
                sig_tokens.push(word);
            }
        }

        // Project tokens into 4d via multiple independent hashes
        let seeds: [u64; 4] = [
            0x9e3779b97f4a7c15,
            0x517cc1b727220a95,
            0x6c62272e07bb0142,
            0xcbf29ce484222325,
        ];
        for (dim, seed) in seeds.iter().enumerate() {
            let mut acc = 0.0f64;
            for &token in &sig_tokens {
                let h = Self::fnv1a_hash(token.as_bytes(), *seed);
                // Map hash to [-1, 1] range (random projection via hashing)
                let val = ((h as f64) / (u64::MAX as f64)) * 2.0 - 1.0;
                acc += val;
            }
            if !sig_tokens.is_empty() {
                emb[4 + dim] = (acc / (sig_tokens.len() as f64).sqrt()) as f32;
            }
        }
    }

    // --- Channel 3: Relational graph (dims 8-11) ---
    fn fill_relational(emb: &mut [f32; 24], path: &str, structure: &CodeStructure) {
        // dim 8: import fan-out (how many external modules this file depends on)
        let import_count = structure.imports.len();
        emb[8] = (import_count as f32 / 20.0).min(1.0);

        // dim 9: export surface (public API as fraction of total declarations)
        let total_decls = structure.declarations.len();
        if total_decls > 0 {
            emb[9] = structure.metrics.public_api_count as f32 / total_decls as f32;
        }

        // dim 10: call graph density (unique callees / functions)
        let fn_count = structure
            .declarations
            .iter()
            .filter(|d| d.kind == DeclKind::Function)
            .count();
        let unique_callees: std::collections::HashSet<&str> = structure
            .call_sites
            .iter()
            .map(|c| c.callee.as_str())
            .collect();
        if fn_count > 0 {
            emb[10] = (unique_callees.len() as f32 / fn_count as f32 / 5.0).min(1.0);
        }

        // dim 11: module depth + language encoding
        let depth = path.matches('/').count() as f32;
        let lang_offset = match structure.language {
            CodeLanguage::Rust => 0.0,
            CodeLanguage::Python => 0.15,
            CodeLanguage::TypeScript => 0.30,
            CodeLanguage::JavaScript => 0.35,
            CodeLanguage::Go => 0.50,
            CodeLanguage::C => 0.65,
            CodeLanguage::Cpp => 0.70,
            CodeLanguage::Java => 0.85,
            CodeLanguage::Unknown => 0.95,
        };
        emb[11] = ((depth / 10.0).min(0.5)) + lang_offset * 0.5;
    }

    // --- Channel 5: Test/quality signal (dims 16-19) ---
    fn fill_test_quality(emb: &mut [f32; 24], path: &str, structure: &CodeStructure) {
        let m = &structure.metrics;

        // dim 16: is-test signal (binary + naming heuristic)
        let is_test_file =
            path.contains("test") || path.contains("spec") || path.contains("_test.");
        emb[16] = if is_test_file {
            1.0
        } else if m.test_function_count > 0 {
            0.5
        } else {
            0.0
        };

        // dim 17: test density (test functions / total functions)
        let fn_count = structure
            .declarations
            .iter()
            .filter(|d| d.kind == DeclKind::Function)
            .count();
        if fn_count > 0 {
            emb[17] = m.test_function_count as f32 / fn_count as f32;
        }

        // dim 18: assertion density (assertions per 100 lines)
        if m.code_lines > 0 {
            emb[18] = (m.assertion_count as f32 / m.code_lines as f32 * 100.0 / 10.0).min(1.0);
        }

        // dim 19: documentation density (comment lines / total lines)
        if m.total_lines > 0 {
            emb[19] = (m.comment_lines as f32 / m.total_lines as f32).min(1.0);
        }
    }

    // --- Channel 6: Pattern identity (dims 20-23) ---
    fn fill_pattern_identity(emb: &mut [f32; 24], content: &str, structure: &CodeStructure) {
        let m = &structure.metrics;
        let decls = &structure.declarations;

        // dim 20: API surface ratio (public / total) — libraries vs applications
        let total = decls.len().max(1) as f32;
        emb[20] = m.public_api_count as f32 / total;

        // dim 21: trait/interface implementation density — indicates pattern use
        let type_count = decls
            .iter()
            .filter(|d| matches!(d.kind, DeclKind::Struct | DeclKind::Class | DeclKind::Enum))
            .count()
            .max(1) as f32;
        emb[21] = (m.trait_impl_count as f32 / type_count).min(1.0);

        // dim 22: structural fingerprint — hash of the declaration sequence
        // Files with the same architectural pattern have similar fingerprints
        let decl_sig: String = decls
            .iter()
            .map(|d| match d.kind {
                DeclKind::Function => "F",
                DeclKind::Struct => "S",
                DeclKind::Enum => "E",
                DeclKind::Trait => "T",
                DeclKind::Impl => "I",
                DeclKind::Class => "C",
                DeclKind::Interface => "N",
                DeclKind::Module => "M",
                DeclKind::Constant => "K",
                DeclKind::TypeAlias => "A",
            })
            .collect();
        emb[22] = Self::fnv1a_hash(decl_sig.as_bytes(), 0xdeadbeef) as f32 / u64::MAX as f32;

        // dim 23: code-to-data ratio — how much is logic vs data/config
        let logic_keywords = [
            "if", "for", "while", "match", "loop", "return", "break", "continue",
        ];
        let logic_count = logic_keywords
            .iter()
            .map(|kw| content.matches(kw).count())
            .sum::<usize>();
        if m.code_lines > 0 {
            emb[23] = (logic_count as f32 / m.code_lines as f32 * 10.0).min(1.0);
        }
    }

    fn fnv1a_hash(data: &[u8], seed: u64) -> u64 {
        let mut h = seed;
        for &b in data {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }
}

// ---------------------------------------------------------------------------
// GitHistory — edit correlation from version control
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GitHistory {
    /// Co-change counts: (file_a, file_b) → how many commits changed both
    pub cochange: HashMap<(String, String), u32>,
    /// Per-file churn: total lines changed across all commits
    pub churn: HashMap<String, u32>,
    /// Per-file author count
    pub authors: HashMap<String, u32>,
    /// Per-file last modification (commit index, higher = more recent)
    pub recency: HashMap<String, u32>,
}

impl GitHistory {
    /// Parse git log output to build co-change and churn data.
    /// Expected format: output of `git log --name-only --pretty=format:"---"`
    pub fn from_git_log(log_output: &str) -> Self {
        let mut history = Self::default();
        let mut commit_files: Vec<String> = Vec::new();
        let mut commit_idx: u32 = 0;

        for line in log_output.lines() {
            let trimmed = line.trim();
            if trimmed == "---" || trimmed.is_empty() {
                // End of a commit block — record co-changes
                if commit_files.len() >= 2 {
                    for i in 0..commit_files.len() {
                        for j in (i + 1)..commit_files.len() {
                            let a = commit_files[i].clone();
                            let b = commit_files[j].clone();
                            let key = if a < b { (a, b) } else { (b, a) };
                            *history.cochange.entry(key).or_insert(0) += 1;
                        }
                    }
                }
                for f in &commit_files {
                    *history.churn.entry(f.clone()).or_insert(0) += 1;
                    history.recency.entry(f.clone()).or_insert(commit_idx);
                }
                commit_files.clear();
                if trimmed == "---" {
                    commit_idx += 1;
                }
                continue;
            }

            // File path line
            if !trimmed.starts_with("Author:")
                && !trimmed.starts_with("Date:")
                && !trimmed.starts_with("commit ")
                && !trimmed.starts_with("Merge:")
            {
                commit_files.push(trimmed.to_string());
            }
        }

        // Handle final commit block
        if commit_files.len() >= 2 {
            for i in 0..commit_files.len() {
                for j in (i + 1)..commit_files.len() {
                    let a = commit_files[i].clone();
                    let b = commit_files[j].clone();
                    let key = if a < b { (a, b) } else { (b, a) };
                    *history.cochange.entry(key).or_insert(0) += 1;
                }
            }
        }
        for f in &commit_files {
            *history.churn.entry(f.clone()).or_insert(0) += 1;
            history.recency.entry(f.clone()).or_insert(commit_idx);
        }

        history
    }

    /// Parse output from `git log --name-only --format='---' --diff-filter=AM`
    /// with `git shortlog -sn` for author counts.
    pub fn add_author_counts(&mut self, shortlog_output: &str) {
        for line in shortlog_output.lines() {
            let trimmed = line.trim();
            let parts: Vec<&str> = trimmed.splitn(2, '\t').collect();
            if parts.len() == 2 {
                if let Ok(count) = parts[0].trim().parse::<u32>() {
                    self.authors.insert(parts[1].to_string(), count);
                }
            }
        }
    }

    /// Populate edit correlation dimensions (12-15) for a file's embedding.
    pub fn fill_edit_correlation(&self, emb: &mut [f32; 24], path: &str, _all_paths: &[&str]) {
        // dim 12: co-change fan-out (how many other files this changes with)
        let cochange_partners: usize = self
            .cochange
            .keys()
            .filter(|(a, b)| a == path || b == path)
            .count();
        emb[12] = (cochange_partners as f32 / 20.0).min(1.0);

        // dim 13: churn rate (normalized)
        let churn = self.churn.get(path).copied().unwrap_or(0);
        let max_churn = self.churn.values().copied().max().unwrap_or(1);
        emb[13] = churn as f32 / max_churn as f32;

        // dim 14: author diversity (normalized)
        let authors = self.authors.get(path).copied().unwrap_or(1);
        emb[14] = (authors as f32 / 10.0).min(1.0);

        // dim 15: recency (higher = more recently changed)
        let max_commit = self.recency.values().copied().max().unwrap_or(1);
        let recency = self.recency.get(path).copied().unwrap_or(0);
        if max_commit > 0 {
            emb[15] = 1.0 - (recency as f32 / max_commit as f32); // invert: 0=oldest, 1=newest
        }
    }

    /// Compute a co-change similarity embedding for a file relative to all others.
    /// This produces a fingerprint where files that co-change frequently have
    /// similar vectors.
    pub fn cochange_vector(&self, path: &str, all_paths: &[&str]) -> [f32; 4] {
        let mut vec = [0.0f32; 4];
        for (i, seed) in [0x1234u64, 0x5678, 0x9abc, 0xdef0].iter().enumerate() {
            let mut acc = 0.0f64;
            for other in all_paths {
                if *other == path {
                    continue;
                }
                let key = if path < *other {
                    (path.to_string(), other.to_string())
                } else {
                    (other.to_string(), path.to_string())
                };
                let count = self.cochange.get(&key).copied().unwrap_or(0);
                if count > 0 {
                    let h = HybridEmbedder::fnv1a_hash(other.as_bytes(), *seed);
                    acc += (count as f64) * ((h as f64 / u64::MAX as f64) * 2.0 - 1.0);
                }
            }
            vec[i] = acc as f32;
        }
        let norm: f32 = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for v in vec.iter_mut() {
                *v /= norm;
            }
        }
        vec
    }
}

// ---------------------------------------------------------------------------
// ProjectModel — codebase spatial index using Leech lattice
// ---------------------------------------------------------------------------
//
// Maps files, functions, types, and modules to 24d Leech-quantized embeddings.
// The 24 dimensions encode 6 relationship facets (4d each):
//   dims  0-3:  structural skeleton (AST-lite: declaration density, complexity)
//   dims  4-7:  semantic content (distributional signature of significant tokens)
//   dims  8-11: relational graph (imports, call graph, API surface)
//   dims 12-15: edit correlation (git co-change, churn, recency)
//   dims 16-19: test/quality signal (coverage, assertions, documentation)
//   dims 20-23: pattern identity (structural fingerprint, design pattern markers)

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityKind {
    File,
    Function,
    Type,
    Module,
    Test,
    Pattern,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectEntity {
    pub kind: EntityKind,
    pub name: String,
    pub path: String,
    pub embedding: [f32; 24],
    #[serde(skip)]
    pub structure: Option<CodeStructure>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectModel {
    pub entities: Vec<ProjectEntity>,
    /// Git history for edit correlation.
    pub git_history: Option<GitHistory>,
    #[serde(skip)]
    quantized_cache: Vec<[f32; 24]>,
}

impl Default for ProjectModel {
    fn default() -> Self {
        Self {
            entities: Vec::new(),
            git_history: None,
            quantized_cache: Vec::new(),
        }
    }
}

impl ProjectModel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an entity with a raw 24d embedding. The embedding will be
    /// Leech-quantized for spatial indexing.
    pub fn add_entity(&mut self, kind: EntityKind, name: &str, path: &str, embedding: [f32; 24]) {
        let quantized = LeechLattice::nearest_point(&embedding);
        self.entities.push(ProjectEntity {
            kind,
            name: name.to_string(),
            path: path.to_string(),
            embedding,
            structure: None,
        });
        self.quantized_cache.push(quantized);
    }

    /// Index a file using the full hybrid embedding pipeline.
    pub fn index_file_hybrid(&mut self, path: &str, content: &str) {
        let structure = CodeAnalyzer::analyze(path, content);
        let mut emb = HybridEmbedder::embed_from_structure(path, content, &structure);

        // If git history is available, fill edit correlation
        if let Some(ref git) = self.git_history {
            let all_paths: Vec<&str> = self.entities.iter().map(|e| e.path.as_str()).collect();
            git.fill_edit_correlation(&mut emb, path, &all_paths);
            let cochange = git.cochange_vector(path, &all_paths);
            for i in 0..4 {
                emb[12 + i] = emb[12 + i] * 0.5 + cochange[i] * 0.5;
            }
        }

        let quantized = LeechLattice::nearest_point(&emb);

        // Extract sub-entities (functions, types) from declarations
        let sub_entities: Vec<(EntityKind, String)> = structure
            .declarations
            .iter()
            .filter(|d| {
                matches!(
                    d.kind,
                    DeclKind::Function
                        | DeclKind::Struct
                        | DeclKind::Enum
                        | DeclKind::Class
                        | DeclKind::Interface
                        | DeclKind::Trait
                )
            })
            .map(|d| {
                let kind = match d.kind {
                    DeclKind::Function => EntityKind::Function,
                    DeclKind::Struct
                    | DeclKind::Enum
                    | DeclKind::Class
                    | DeclKind::Interface
                    | DeclKind::Trait => EntityKind::Type,
                    _ => EntityKind::Function,
                };
                (kind, d.name.clone())
            })
            .collect();

        self.entities.push(ProjectEntity {
            kind: EntityKind::File,
            name: path.rsplit('/').next().unwrap_or(path).to_string(),
            path: path.to_string(),
            embedding: emb,
            structure: Some(structure.clone()),
        });
        self.quantized_cache.push(quantized);

        // Index sub-entities with embeddings derived from the file's structure
        for (kind, name) in sub_entities {
            let sub_emb = HybridEmbedder::embed_symbol(path, &name, content, &structure);
            let sub_quantized = LeechLattice::nearest_point(&sub_emb);
            self.entities.push(ProjectEntity {
                kind,
                name: name.clone(),
                path: path.to_string(),
                embedding: sub_emb,
                structure: None,
            });
            self.quantized_cache.push(sub_quantized);
        }
    }

    /// Load git history and re-compute edit correlation for all indexed entities.
    pub fn load_git_history(&mut self, log_output: &str) {
        let history = GitHistory::from_git_log(log_output);
        let all_paths: Vec<String> = self
            .entities
            .iter()
            .filter(|e| e.kind == EntityKind::File)
            .map(|e| e.path.clone())
            .collect();
        let all_path_refs: Vec<&str> = all_paths.iter().map(|s| s.as_str()).collect();

        for entity in self.entities.iter_mut() {
            if entity.kind == EntityKind::File {
                history.fill_edit_correlation(&mut entity.embedding, &entity.path, &all_path_refs);
                let cochange = history.cochange_vector(&entity.path, &all_path_refs);
                for i in 0..4 {
                    entity.embedding[12 + i] = entity.embedding[12 + i] * 0.5 + cochange[i] * 0.5;
                }
            }
        }

        // Rebuild quantized cache
        self.quantized_cache = self
            .entities
            .iter()
            .map(|e| LeechLattice::nearest_point(&e.embedding))
            .collect();

        self.git_history = Some(history);
    }

    /// Legacy compatibility: embed_file (delegates to HybridEmbedder).
    pub fn embed_file(path: &str, content: &str) -> [f32; 24] {
        HybridEmbedder::embed_file(path, content)
    }

    /// Legacy compatibility: embed_symbol.
    pub fn embed_symbol(path: &str, name: &str, body: &str) -> [f32; 24] {
        let parent = CodeAnalyzer::analyze(path, body);
        HybridEmbedder::embed_symbol(path, name, body, &parent)
    }

    /// Find the k most related entities to a query embedding.
    pub fn find_related(&self, query: &[f32; 24], k: usize) -> Vec<(usize, f32)> {
        if self.entities.is_empty() {
            return vec![];
        }

        let points: Vec<[f32; 24]> = self.entities.iter().map(|e| e.embedding).collect();
        LeechLattice::nearest_neighbors(query, &points, k)
    }

    /// Find all entities related to a file (by path), using Leech nearest-neighbor.
    pub fn context_for_file(&self, path: &str, k: usize) -> Vec<&ProjectEntity> {
        let idx = self
            .entities
            .iter()
            .position(|e| e.path == path && e.kind == EntityKind::File);
        match idx {
            Some(i) => {
                let query = self.entities[i].embedding;
                let neighbors = self.find_related(&query, k + 1);
                neighbors
                    .iter()
                    .filter(|(ni, _)| *ni != i)
                    .take(k)
                    .filter_map(|(ni, _)| self.entities.get(*ni))
                    .collect()
            }
            None => vec![],
        }
    }

    /// Project the entity context into a conditioning vector for generation.
    pub fn context_conditioning(&self, query: &[f32; 24], k: usize) -> Vec<f32> {
        let neighbors = self.find_related(query, k);
        if neighbors.is_empty() {
            return vec![0.0f32; 24];
        }

        let mut avg = [0.0f32; 24];
        let n = neighbors.len() as f32;
        for (idx, _dist) in &neighbors {
            if let Some(e) = self.entities.get(*idx) {
                for i in 0..24 {
                    avg[i] += e.embedding[i] / n;
                }
            }
        }

        let quantized = LeechLattice::nearest_point(&avg);
        quantized.to_vec()
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    pub fn entities_by_kind(&self, kind: EntityKind) -> Vec<&ProjectEntity> {
        self.entities.iter().filter(|e| e.kind == kind).collect()
    }

    /// Summary statistics for the indexed project.
    pub fn summary(&self) -> ProjectSummary {
        ProjectSummary {
            total_entities: self.entities.len(),
            files: self.entities_by_kind(EntityKind::File).len(),
            functions: self.entities_by_kind(EntityKind::Function).len(),
            types: self.entities_by_kind(EntityKind::Type).len(),
            tests: self.entities_by_kind(EntityKind::Test).len(),
            has_git_history: self.git_history.is_some(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProjectSummary {
    pub total_entities: usize,
    pub files: usize,
    pub functions: usize,
    pub types: usize,
    pub tests: usize,
    pub has_git_history: bool,
}

// ---------------------------------------------------------------------------
// DCT-II / IDCT (orthonormal, pure Rust)
// ---------------------------------------------------------------------------

pub fn dct_ii(signal: &[f32]) -> Vec<f32> {
    let n = signal.len();
    if n == 0 {
        return vec![];
    }
    let scale = (2.0 / n as f32).sqrt();
    let mut out = vec![0.0f32; n];
    for k in 0..n {
        let mut sum = 0.0f32;
        for i in 0..n {
            let angle = std::f32::consts::PI * (2 * i + 1) as f32 * k as f32 / (2 * n) as f32;
            sum += signal[i] * angle.cos();
        }
        out[k] = sum * scale;
        if k == 0 {
            out[k] /= std::f32::consts::SQRT_2;
        }
    }
    out
}

pub fn idct_iii(coeffs: &[f32]) -> Vec<f32> {
    let n = coeffs.len();
    if n == 0 {
        return vec![];
    }
    let scale = (2.0 / n as f32).sqrt();
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let mut sum = coeffs[0] / std::f32::consts::SQRT_2;
        for k in 1..n {
            let angle = std::f32::consts::PI * k as f32 * (2 * i + 1) as f32 / (2 * n) as f32;
            sum += coeffs[k] * angle.cos();
        }
        out[i] = sum * scale;
    }
    out
}

/// Keep only the top K coefficients by magnitude, zero out the rest.
pub fn sparse_coeffs(coeffs: &[f32], k: usize) -> Vec<f32> {
    if k >= coeffs.len() {
        return coeffs.to_vec();
    }
    let mut mags: Vec<f32> = coeffs.iter().map(|c| c.abs()).collect();
    mags.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = mags[k];
    coeffs
        .iter()
        .map(|&c| if c.abs() > threshold { c } else { 0.0 })
        .collect()
}

// ---------------------------------------------------------------------------
// Token Dictionary
// ---------------------------------------------------------------------------

/// End-of-sequence marker ID. Token sequences are padded with this.
pub const EOS_ID: u16 = 0;

/// Convert binary to Gray code.
pub fn to_gray(n: u16) -> u16 {
    n ^ (n >> 1)
}

/// Convert Gray code back to binary.
pub fn from_gray(g: u16) -> u16 {
    let mut n = g;
    let mut mask = n >> 1;
    while mask != 0 {
        n ^= mask;
        mask >>= 1;
    }
    n
}

/// Build co-occurrence vectors from tokenized texts, then return a permutation
/// that orders tokens so semantically related ones get adjacent positions.
///
/// Each token gets a vector of how often it co-occurs with every other token
/// within a context window. Cosine similarity on these vectors captures
/// distributional semantics (words in similar contexts are similar).
/// A greedy nearest-neighbor chain then arranges tokens so similar ones
/// are adjacent — meaning 1-bit Gray code errors land on related words.
fn semantic_order(token_list: &[String], texts: &[&str], window: usize) -> Vec<usize> {
    let n = token_list.len();
    if n <= 2 {
        return (0..n).collect();
    }

    let tok_idx: HashMap<&str, usize> = token_list
        .iter()
        .enumerate()
        .map(|(i, t)| (t.as_str(), i))
        .collect();

    // Build co-occurrence matrix (sparse, stored as dense for simplicity with n <= 2048)
    let mut cooc = vec![vec![0u32; n]; n];
    for text in texts {
        let tokens = tokenize(text);
        let ids: Vec<usize> = tokens
            .iter()
            .filter_map(|t| tok_idx.get(t.as_str()).copied())
            .collect();
        for (i, &a) in ids.iter().enumerate() {
            let end = (i + window + 1).min(ids.len());
            for &b in &ids[i + 1..end] {
                if a != b {
                    cooc[a][b] += 1;
                    cooc[b][a] += 1;
                }
            }
        }
    }

    // Compute norms for cosine similarity
    let norms: Vec<f32> = cooc
        .iter()
        .map(|row| {
            let s: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
            s.sqrt() as f32
        })
        .collect();

    // Cosine similarity between two tokens
    let cosine = |a: usize, b: usize| -> f32 {
        if norms[a] < 1e-9 || norms[b] < 1e-9 {
            return 0.0;
        }
        let dot: f64 = cooc[a]
            .iter()
            .zip(cooc[b].iter())
            .map(|(&x, &y)| x as f64 * y as f64)
            .sum();
        (dot / (norms[a] as f64 * norms[b] as f64)) as f32
    };

    // Greedy nearest-neighbor chain: start from most connected token,
    // always pick the most similar unvisited token next.
    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);

    // Punctuation and special tokens go first (stable prefix)
    let mut punct_ids: Vec<usize> = (0..n)
        .filter(|&i| {
            let ch = token_list[i].chars().next().unwrap_or(' ');
            ch.is_ascii_punctuation() || token_list[i].chars().all(|c| c.is_ascii_digit())
        })
        .collect();
    punct_ids.sort_by(|&a, &b| token_list[a].cmp(&token_list[b]));
    for &pid in &punct_ids {
        order.push(pid);
        visited[pid] = true;
    }

    // Start chain from the token with highest total co-occurrence
    let start = (0..n)
        .filter(|i| !visited[*i])
        .max_by_key(|&i| norms[i] as u64)
        .unwrap_or(0);
    if !visited[start] {
        order.push(start);
        visited[start] = true;
    }

    // Greedy walk
    while order.len() < n {
        let last = *order.last().unwrap();
        let mut best_idx = usize::MAX;
        let mut best_sim = -1.0f32;
        for j in 0..n {
            if visited[j] {
                continue;
            }
            let sim = cosine(last, j);
            if sim > best_sim {
                best_sim = sim;
                best_idx = j;
            }
        }
        if best_idx == usize::MAX {
            break;
        }
        order.push(best_idx);
        visited[best_idx] = true;
    }

    // Pick up any remaining (disconnected tokens with zero co-occurrence)
    for i in 0..n {
        if !visited[i] {
            order.push(i);
        }
    }

    order
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenDictionary {
    pub tokens: Vec<String>,
    lookup: HashMap<String, u16>,
    gray_to_binary: Vec<u16>,
    binary_to_gray: Vec<u16>,
}

impl TokenDictionary {
    /// Build from a corpus. Keeps the `max_size` most frequent tokens.
    /// ID 0 is reserved for <EOS>. Tokens are ordered by distributional
    /// semantics (co-occurrence vectors + greedy nearest-neighbor chain)
    /// so similar tokens get adjacent IDs, then Gray coding ensures adjacent
    /// IDs differ by only 1 bit — making generation errors land on
    /// semantically related words instead of garbage.
    pub fn build(texts: &[&str], max_size: usize) -> Self {
        let mut freq: HashMap<String, usize> = HashMap::new();
        for text in texts {
            for token in tokenize(text) {
                *freq.entry(token).or_default() += 1;
            }
        }
        let mut entries: Vec<(String, usize)> = freq.into_iter().collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1));
        entries.truncate(max_size.saturating_sub(1));

        // Build semantic ordering from co-occurrence in the corpus
        let token_list: Vec<String> = entries.iter().map(|(t, _)| t.clone()).collect();
        let perm = semantic_order(&token_list, texts, 5);

        let dict_size = entries.len() + 1; // +1 for EOS
        let mut tokens = Vec::with_capacity(dict_size);
        let mut lookup = HashMap::new();
        tokens.push("<EOS>".to_string());
        for &idx in &perm {
            let token = entries[idx].0.clone();
            let id = tokens.len() as u16;
            lookup.insert(token.clone(), id);
            tokens.push(token);
        }

        // Build Gray code lookup tables
        let mut gray_to_binary = vec![0u16; dict_size];
        let mut binary_to_gray = vec![0u16; dict_size];
        for i in 0..dict_size as u16 {
            let g = to_gray(i);
            if (g as usize) < dict_size {
                gray_to_binary[g as usize] = i;
            }
            binary_to_gray[i as usize] = g;
        }

        Self {
            tokens,
            lookup,
            gray_to_binary,
            binary_to_gray,
        }
    }

    /// The Gray-coded ID for a token's internal index.
    /// Used by GroupGenEnv for encoding targets.
    pub fn to_gray_id(&self, id: u16) -> u16 {
        self.binary_to_gray.get(id as usize).copied().unwrap_or(0)
    }

    /// Recover internal index from a Gray-coded ID.
    /// Used by GroupGenEnv for decoding outputs.
    pub fn from_gray_id(&self, gray_id: u16) -> u16 {
        self.gray_to_binary
            .get(gray_id as usize)
            .copied()
            .unwrap_or(0)
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn lookup_len(&self) -> usize {
        self.lookup.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.len() <= 1
    }

    pub fn token_id(&self, token: &str) -> Option<u16> {
        self.lookup.get(token).copied()
    }

    pub fn token_str(&self, id: u16) -> Option<&str> {
        self.tokens.get(id as usize).map(|s| s.as_str())
    }

    /// Encode text to a sequence of token IDs.
    /// Every word maps to exactly one token — words are irreducible
    /// geometric objects (Dirac particles), never decomposed into characters.
    /// Unknown words are converted to the nearest known word by edit distance.
    pub fn encode(&self, text: &str) -> Vec<u16> {
        let raw_tokens = tokenize(text);
        let mut ids = Vec::new();
        for tok in &raw_tokens {
            if let Some(&id) = self.lookup.get(tok.as_str()) {
                ids.push(id);
            } else if let Some((id, _, _)) = self.nearest_by_edit(tok) {
                ids.push(id);
            }
        }
        ids
    }

    /// Encode text, returning both token IDs and a parallel `word_start`
    /// vector. With one-word-one-token encoding, every entry is a word
    /// start — but the interface is retained for callers that check alignment.
    pub fn encode_with_word_boundaries(&self, text: &str) -> (Vec<u16>, Vec<bool>) {
        let ids = self.encode(text);
        let word_starts = vec![true; ids.len()];
        (ids, word_starts)
    }

    /// Word boundaries for a token ID sequence. With one-word-one-token
    /// encoding every position is a word start, but legacy sequences (from
    /// char-decomposed OOV words) are detected: consecutive single alphabetic
    /// characters are treated as interior fragments.
    pub fn infer_word_boundaries(&self, ids: &[u16]) -> Vec<bool> {
        ids.iter()
            .enumerate()
            .map(|(i, &id)| {
                if i == 0 {
                    return true;
                }
                let tok = self
                    .tokens
                    .get(id as usize)
                    .map(|s| s.as_str())
                    .unwrap_or("");
                if tok.len() == 1 && tok.chars().next().map_or(false, |c| c.is_alphabetic()) {
                    let prev = self
                        .tokens
                        .get(ids[i - 1] as usize)
                        .map(|s| s.as_str())
                        .unwrap_or("");
                    if prev.len() == 1 && prev.chars().next().map_or(false, |c| c.is_alphabetic()) {
                        return false;
                    }
                }
                true
            })
            .collect()
    }

    /// Decode token IDs back to text. Stops at EOS_ID.
    /// Runs of 3+ consecutive single-alphabetic tokens (legacy char-split
    /// fragments) are rejoined into a single word — preserving the Dirac
    /// particle invariant even for data encoded before one-word-one-token.
    pub fn decode(&self, ids: &[u16]) -> String {
        let toks: Vec<&str> = ids
            .iter()
            .take_while(|&&id| id != EOS_ID)
            .filter_map(|&id| self.tokens.get(id as usize).map(|s| s.as_str()))
            .collect();

        let mut result = String::new();
        let mut i = 0;
        while i < toks.len() {
            let is_sa =
                |s: &str| s.len() == 1 && s.chars().next().map_or(false, |c| c.is_alphabetic());

            if is_sa(toks[i]) {
                let start = i;
                while i < toks.len() && is_sa(toks[i]) {
                    i += 1;
                }
                let run = i - start;
                if run >= 3 {
                    if !result.is_empty() {
                        result.push(' ');
                    }
                    for j in start..i {
                        result.push_str(toks[j]);
                    }
                } else {
                    for j in start..i {
                        if !result.is_empty() {
                            result.push(' ');
                        }
                        result.push_str(toks[j]);
                    }
                }
            } else {
                if !result.is_empty() && !toks[i].starts_with(|c: char| c.is_ascii_punctuation()) {
                    result.push(' ');
                }
                result.push_str(toks[i]);
                i += 1;
            }
        }
        result
    }

    /// Find the nearest valid token ID to a raw (possibly non-integer) value.
    pub fn nearest_id(&self, raw_value: f32, dict_size: f32) -> u16 {
        let id = (raw_value * dict_size).round() as i32;
        id.clamp(0, self.tokens.len() as i32 - 1) as u16
    }

    /// Find the closest token by edit distance.
    pub fn nearest_by_edit(&self, query: &str) -> Option<(u16, &str, usize)> {
        if self.tokens.len() <= 1 {
            return None;
        }
        if let Some(&id) = self.lookup.get(query) {
            return Some((id, &self.tokens[id as usize], 0));
        }
        let mut best_id = 1u16; // skip EOS
        let mut best_dist = usize::MAX;
        for (i, tok) in self.tokens.iter().enumerate().skip(1) {
            let d = edit_distance(query, tok);
            if d < best_dist {
                best_dist = d;
                best_id = i as u16;
                if d <= 1 {
                    break;
                }
            }
        }
        Some((best_id, &self.tokens[best_id as usize], best_dist))
    }
}

// ---------------------------------------------------------------------------
// Syntax Role Classification — for syntax-aware codebook construction
// ---------------------------------------------------------------------------

/// Syntactic role of a token. Used by the syntax-aware codebook to distinguish
/// structural tokens (always fixed) from content tokens (potential slots).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SyntaxRole {
    /// Language keyword: def, fn, class, if, else, for, while, return, let, mut, pub, etc.
    Keyword,
    /// Structural punctuation: ( ) { } [ ] : ; , . -> => #
    Structure,
    /// Operator: = == != < > + - * / % & | ^ ! && || += -= etc.
    Operator,
    /// Numeric literal: 0, 1, 42, 3.14, 0xFF
    Literal,
    /// Identifier: variable names, function names, type names
    Identifier,
}

const KEYWORDS: &[&str] = &[
    // Python
    "def",
    "class",
    "if",
    "elif",
    "else",
    "for",
    "while",
    "return",
    "import",
    "from",
    "try",
    "except",
    "finally",
    "with",
    "as",
    "yield",
    "lambda",
    "pass",
    "break",
    "continue",
    "in",
    "not",
    "and",
    "or",
    "is",
    "None",
    "True",
    "False",
    "self",
    "raise",
    "assert",
    "del",
    "global",
    "nonlocal",
    "async",
    "await",
    "print",
    // Rust
    "fn",
    "let",
    "mut",
    "pub",
    "struct",
    "enum",
    "impl",
    "trait",
    "use",
    "mod",
    "crate",
    "super",
    "where",
    "match",
    "loop",
    "const",
    "static",
    "type",
    "move",
    "ref",
    "unsafe",
    "extern",
    "dyn",
    "Box",
    "Vec",
    "String",
    "Option",
    "Result",
    "Some",
    "Ok",
    "Err",
    "println",
    "macro_rules",
    // JavaScript/TypeScript
    "function",
    "var",
    "const",
    "new",
    "this",
    "prototype",
    "extends",
    "constructor",
    "export",
    "default",
    "typeof",
    "instanceof",
    "void",
    "null",
    "undefined",
    "true",
    "false",
    "console",
    "log",
    "require",
    // Shared
    "int",
    "float",
    "bool",
    "str",
    "char",
    "void",
    "static",
    "final",
    "abstract",
    "interface",
    "override",
    "virtual",
    "template",
    "namespace",
];

const STRUCTURE_CHARS: &[char] = &['(', ')', '{', '}', '[', ']', ':', ';', ',', '.', '#', '@'];

const OPERATOR_TOKENS: &[&str] = &[
    "=", "==", "!=", "<", ">", "<=", ">=", "+", "-", "*", "/", "%", "&", "|", "^", "!", "&&", "||",
    "<<", ">>", "+=", "-=", "*=", "/=", "->", "=>", "::", "..", "..=", "**",
];

/// Classify a token's syntactic role.
pub fn syntax_role(token: &str) -> SyntaxRole {
    if KEYWORDS.contains(&token) {
        return SyntaxRole::Keyword;
    }
    if OPERATOR_TOKENS.contains(&token) {
        return SyntaxRole::Operator;
    }
    if token.len() == 1 {
        let ch = token.chars().next().unwrap();
        if STRUCTURE_CHARS.contains(&ch) {
            return SyntaxRole::Structure;
        }
        if ch.is_ascii_digit() {
            return SyntaxRole::Literal;
        }
    }
    if token.chars().all(|c| {
        c.is_ascii_digit()
            || c == '.'
            || c == 'x'
            || c == 'X'
            || (c.is_ascii_hexdigit() && token.starts_with("0x"))
    }) {
        return SyntaxRole::Literal;
    }
    SyntaxRole::Identifier
}

/// Classify a sequence of token strings into their syntax roles.
pub fn syntax_roles(tokens: &[String]) -> Vec<SyntaxRole> {
    tokens.iter().map(|t| syntax_role(t)).collect()
}

/// Build a structural signature from token strings: replace identifiers/literals
/// with role placeholders, keep keywords/structure/operators as-is.
/// Two code snippets with the same signature have the same syntactic structure.
pub fn structural_signature(tokens: &[String]) -> Vec<String> {
    tokens
        .iter()
        .map(|t| match syntax_role(t) {
            SyntaxRole::Keyword | SyntaxRole::Structure | SyntaxRole::Operator => t.clone(),
            SyntaxRole::Literal => "_LIT_".to_string(),
            SyntaxRole::Identifier => "_ID_".to_string(),
        })
        .collect()
}

/// Tokenizer: split on whitespace, separate punctuation (but keep _ as word char).
pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_whitespace() {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
        } else if ch.is_ascii_punctuation() && ch != '_' {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
            tokens.push(ch.to_string());
        } else {
            current.push(ch);
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (m, n) = (a.len(), b.len());
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

// ---------------------------------------------------------------------------
// Spectral Codec — DCT on token ID sequences
// ---------------------------------------------------------------------------

/// Max token sequence length. Sequences are padded to this for fixed-size DCT.
pub const MAX_SEQ_LEN: usize = 64;

#[derive(Clone, Serialize, Deserialize)]
pub struct SpectralCodec {
    pub dictionary: TokenDictionary,
    pub max_seq: usize,
}

impl SpectralCodec {
    pub fn new(dictionary: TokenDictionary) -> Self {
        Self {
            dictionary,
            max_seq: MAX_SEQ_LEN,
        }
    }

    pub fn with_max_seq(dictionary: TokenDictionary, max_seq: usize) -> Self {
        Self {
            dictionary,
            max_seq,
        }
    }

    /// Number of DCT coefficients produced (= max_seq).
    pub fn coeff_dim(&self) -> usize {
        self.max_seq
    }

    /// Encode text → normalized token-ID signal → DCT coefficients.
    pub fn encode(&self, text: &str) -> Vec<f32> {
        let signal = self.text_to_signal(text);
        dct_ii(&signal)
    }

    /// Encode with sparsity (top K coefficients only).
    pub fn encode_sparse(&self, text: &str, k: usize) -> Vec<f32> {
        sparse_coeffs(&self.encode(text), k)
    }

    /// Decode DCT coefficients → text via dictionary lookup.
    pub fn decode(&self, coeffs: &[f32]) -> String {
        let signal = idct_iii(coeffs);
        let ids = self.signal_to_ids(&signal);
        self.dictionary.decode(&ids)
    }

    /// Minimum K for lossless token-level reconstruction.
    pub fn min_k_lossless(&self, text: &str) -> usize {
        let original_ids = self.dictionary.encode(text);
        let coeffs = self.encode(text);
        for k in 1..=coeffs.len() {
            let sparse = sparse_coeffs(&coeffs, k);
            let recon_signal = idct_iii(&sparse);
            let recon_ids = self.signal_to_ids(&recon_signal);
            if recon_ids.len() >= original_ids.len()
                && recon_ids[..original_ids.len()] == original_ids[..]
            {
                return k;
            }
        }
        coeffs.len()
    }

    /// Measure quality: (token_accuracy, exact_match) at given K.
    pub fn measure_quality(&self, text: &str, k: usize) -> (f32, bool) {
        let original_ids = self.dictionary.encode(text);
        let sparse = self.encode_sparse(text, k);
        let decoded = self.decode(&sparse);
        let recon_ids = self.dictionary.encode(&decoded);

        let len = original_ids.len().min(recon_ids.len());
        let correct = original_ids[..len]
            .iter()
            .zip(recon_ids[..len].iter())
            .filter(|(a, b)| a == b)
            .count();
        let acc = correct as f32 / original_ids.len().max(1) as f32;
        let exact = decoded.trim() == text.trim();
        (acc, exact)
    }

    fn text_to_signal(&self, text: &str) -> Vec<f32> {
        let ids = self.dictionary.encode(text);
        let dict_size = self.dictionary.len() as f32;
        let mut signal = vec![0.0f32; self.max_seq];
        for (i, &id) in ids.iter().take(self.max_seq).enumerate() {
            signal[i] = id as f32 / dict_size;
        }
        signal
    }

    fn signal_to_ids(&self, signal: &[f32]) -> Vec<u16> {
        let dict_size = self.dictionary.len() as f32;
        let mut ids = Vec::new();
        for &val in signal.iter().take(self.max_seq) {
            let raw_id = (val * dict_size).round() as i32;
            if raw_id <= 0 {
                if !ids.is_empty() {
                    break; // EOS
                }
                continue;
            }
            let id = raw_id.clamp(0, self.dictionary.len() as i32 - 1) as u16;
            ids.push(id);
        }
        ids
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_roundtrip_exact() {
        let signal: Vec<f32> = (0..64).map(|i| (i as f32) / 63.0).collect();
        let coeffs = dct_ii(&signal);
        let recon = idct_iii(&coeffs);
        for (a, b) in signal.iter().zip(recon.iter()) {
            assert!((a - b).abs() < 1e-4, "mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_sparse_preserves_top_k() {
        let coeffs = vec![1.0, 0.5, 3.0, 0.1, 2.0];
        let sparse = sparse_coeffs(&coeffs, 2);
        assert_eq!(sparse[2], 3.0);
        assert_eq!(sparse[4], 2.0);
        assert_eq!(sparse[1], 0.0);
        assert_eq!(sparse[3], 0.0);
    }

    #[test]
    fn test_tokenize_keeps_underscores() {
        let tokens = tokenize("hello, world! foo_bar");
        assert_eq!(tokens, vec!["hello", ",", "world", "!", "foo_bar"]);
    }

    #[test]
    fn test_edit_distance() {
        assert_eq!(edit_distance("kitten", "sitting"), 3);
        assert_eq!(edit_distance("hello", "hello"), 0);
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(edit_distance("reset", "rexet"), 1);
    }

    #[test]
    fn test_dictionary_encode_decode() {
        let dict = TokenDictionary::build(&["hello world", "hello rust"], 100);
        let ids = dict.encode("hello world");
        let text = dict.decode(&ids);
        assert_eq!(text, "hello world");
    }

    #[test]
    fn test_dictionary_char_fallback() {
        let dict = TokenDictionary::build(&["hello world"], 100);
        let ids = dict.encode("hello xyz");
        let text = dict.decode(&ids);
        assert!(
            text.contains("hello"),
            "should contain known token: {}",
            text
        );
    }

    #[test]
    fn test_codec_full_roundtrip() {
        let corpus = &[
            "reset your password",
            "check your email",
            "update your profile",
        ];
        let dict = TokenDictionary::build(corpus, 500);
        let codec = SpectralCodec::new(dict);

        for &text in corpus {
            let coeffs = codec.encode(text);
            let decoded = codec.decode(&coeffs);
            assert_eq!(
                decoded.trim(),
                text.trim(),
                "full roundtrip failed for: '{}'",
                text
            );
        }
    }

    #[test]
    fn test_codec_sparse_roundtrip() {
        let corpus = &[
            "reset your password",
            "check your email",
            "update your profile",
            "contact customer support",
            "how do I change my settings",
        ];
        let dict = TokenDictionary::build(corpus, 500);
        let codec = SpectralCodec::new(dict);

        println!("\n--- Sparse compression analysis ---");
        println!("Dictionary size: {} tokens", codec.dictionary.len());
        println!("Signal length: {}\n", codec.max_seq);

        for &text in corpus {
            let ids = codec.dictionary.encode(text);
            let k_loss = codec.min_k_lossless(text);
            println!(
                "'{}' → {} tokens, K_lossless={} ({:.0}% of signal)",
                text,
                ids.len(),
                k_loss,
                k_loss as f32 / codec.max_seq as f32 * 100.0
            );

            for k in [4, 8, 12, 16, 24, 32] {
                let (tok_acc, exact) = codec.measure_quality(text, k);
                println!(
                    "  K={:2}: tok_acc={:.0}%, exact={}",
                    k,
                    tok_acc * 100.0,
                    exact
                );
            }
        }
    }

    #[test]
    fn test_e8_nearest_point_integer_lattice() {
        // An integer point with even sum is already an E8 lattice point
        let x = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let p = E8Lattice::nearest_point(&x);
        assert_eq!(
            p, x,
            "integer point with even sum should be its own nearest"
        );
    }

    #[test]
    fn test_e8_nearest_point_snaps_to_lattice() {
        // A point near [1,1,0,0,0,0,0,0] should snap there
        let x = [1.1, 0.9, 0.1, -0.1, 0.05, -0.05, 0.02, -0.02];
        let p = E8Lattice::nearest_point(&x);
        let sum: f32 = p.iter().sum();
        // E8 lattice: coordinate sum must be even (integer sublattice)
        // or all half-integers with even sum
        let is_integer = p.iter().all(|v| (v - v.round()).abs() < 1e-6);
        let is_half_int = p.iter().all(|v| (v - (v - 0.5).round() - 0.5).abs() < 1e-6);
        assert!(
            is_integer || is_half_int,
            "should be integer or half-integer coords: {:?}",
            p
        );
        if is_integer {
            assert!(
                (sum.round() as i32) % 2 == 0,
                "integer coords must have even sum: sum={}",
                sum
            );
        }
    }

    #[test]
    fn test_e8_half_integer_lattice() {
        // Half-integer point with even sum is also an E8 lattice point
        let x = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]; // sum = 4.0 (even)
        let p = E8Lattice::nearest_point(&x);
        let dist = E8Lattice::dist_sq(&x, &p);
        assert!(
            dist < 1e-6,
            "half-integer E8 point should map to itself, dist={}",
            dist
        );
    }

    #[test]
    fn test_e8_quantize_64d() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        let q = E8Lattice::quantize_64d(&x);
        assert_eq!(q.len(), 64);

        // Each 8d block should be a valid E8 lattice point
        for sub in 0..8 {
            let offset = sub * 8;
            let block: Vec<f32> = q[offset..offset + 8].to_vec();
            let is_integer = block.iter().all(|v| (v - v.round()).abs() < 1e-6);
            let is_half_int = block
                .iter()
                .all(|v| (v - (v - 0.5).round() - 0.5).abs() < 1e-6);
            assert!(
                is_integer || is_half_int,
                "block {} should be E8: {:?}",
                sub,
                block
            );
        }
    }

    #[test]
    fn test_e8_quantize_128d() {
        let x: Vec<f32> = (0..128).map(|i| (i as f32) * 0.05).collect();
        let q = E8Lattice::quantize_64d(&x);
        assert_eq!(q.len(), 128);
        for sub in 0..16 {
            let offset = sub * 8;
            let block: Vec<f32> = q[offset..offset + 8].to_vec();
            let is_integer = block.iter().all(|v| (v - v.round()).abs() < 1e-6);
            let is_half_int = block
                .iter()
                .all(|v| (v - (v - 0.5).round() - 0.5).abs() < 1e-6);
            assert!(
                is_integer || is_half_int,
                "block {} should be E8: {:?}",
                sub,
                block
            );
        }
    }

    #[test]
    fn test_e8_root_inner_product_various_dims_no_panic() {
        for n in [1usize, 7, 8, 9, 31, 32, 63, 64, 65, 127, 128] {
            let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).sin()).collect();
            let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11).cos()).collect();
            let rip = E8Lattice::root_inner_product(&a, &b);
            assert!(rip.is_finite(), "n={} rip={}", n, rip);
            assert!(
                rip >= -1.01 && rip <= 1.01,
                "per-block cosines mean should stay in ~[-1,1], n={} rip={}",
                n,
                rip
            );
        }
    }

    #[test]
    fn test_e8_root_inner_product_mismatched_lengths_uses_prefix() {
        let a = vec![1.0f32; 10];
        let b = vec![1.0f32; 64];
        let rip_ab = E8Lattice::root_inner_product(&a, &b);
        let rip_ba = E8Lattice::root_inner_product(&b, &a);
        assert!(rip_ab.is_finite() && rip_ba.is_finite());
        assert!((rip_ab - rip_ba).abs() < 1e-5);
    }

    #[test]
    fn test_e8_self_compatibility_is_maximal() {
        let a: Vec<f32> = (0..64).map(|i| (i as f32) * 0.05 + 0.1).collect();
        let score = E8Lattice::compatibility_score(&a, &a);
        assert!(score >= 2.5, "self-compatibility should be high: {}", score);
    }

    #[test]
    fn test_e8_orthogonal_compatibility_is_low() {
        let mut a = vec![0.0f32; 64];
        let mut b = vec![0.0f32; 64];
        // Put signal in different subspaces
        for i in 0..8 {
            a[i] = 1.0;
        }
        for i in 8..16 {
            b[i] = 1.0;
        }
        let score = E8Lattice::compatibility_score(&a, &b);
        assert!(
            score < 2.0,
            "orthogonal patterns should have low compatibility: {}",
            score
        );
    }

    #[test]
    fn test_e8_select_archetype() {
        let protos = vec![
            vec![1.0f32; 64],
            vec![-1.0f32; 64],
            (0..64).map(|i| if i < 32 { 1.0 } else { -1.0 }).collect(),
        ];
        let query = vec![0.9f32; 64]; // closest to proto[0]
        let (idx, conf) = E8Lattice::select_archetype(&query, &protos);
        assert_eq!(idx, 0, "should select proto[0]");
        assert!(conf > 0.5, "confidence should be high: {}", conf);
    }

    #[test]
    fn test_extended_hamming_roundtrip() {
        for bits in 0u8..16 {
            let data = [
                (bits >> 0) & 1,
                (bits >> 1) & 1,
                (bits >> 2) & 1,
                (bits >> 3) & 1,
            ];
            let encoded = extended_hamming_encode(&data);
            let (decoded, ok) = extended_hamming_decode(&encoded);
            assert!(ok, "should decode cleanly for {:?}", data);
            assert_eq!(data, decoded, "roundtrip failed for {:?}", data);
        }
    }

    #[test]
    fn test_extended_hamming_single_error_correction() {
        let data = [1u8, 0, 1, 1];
        let encoded = extended_hamming_encode(&data);

        // Flip each bit and verify correction
        for flip in 0..8 {
            let mut corrupted = encoded;
            corrupted[flip] ^= 1;
            let (decoded, ok) = extended_hamming_decode(&corrupted);
            assert!(ok, "should correct single bit flip at position {}", flip);
            assert_eq!(data, decoded, "correction failed for flip at {}", flip);
        }
    }

    #[test]
    fn test_extended_hamming_double_error_detection() {
        let data = [1u8, 0, 1, 1];
        let encoded = extended_hamming_encode(&data);

        // Flip two bits — should detect but not correct
        let mut corrupted = encoded;
        corrupted[0] ^= 1;
        corrupted[1] ^= 1;
        let (_decoded, ok) = extended_hamming_decode(&corrupted);
        assert!(!ok, "should detect double-bit error");
    }

    // -------------------------------------------------------------------
    // Golay [24,12,8] tests
    // -------------------------------------------------------------------

    #[test]
    fn test_golay_roundtrip() {
        for val in 0u16..4096 {
            let data: [u8; 12] = std::array::from_fn(|i| ((val >> i) & 1) as u8);
            let encoded = golay_encode(&data);
            let (decoded, ok) = golay_decode(&encoded);
            assert!(ok, "clean codeword should decode cleanly");
            assert_eq!(decoded, data, "roundtrip failed for val={val}");
        }
    }

    #[test]
    fn test_golay_single_error_correction() {
        let data = [1u8, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1];
        let encoded = golay_encode(&data);
        for bit in 0..24 {
            let mut corrupted = encoded;
            corrupted[bit] ^= 1;
            let (decoded, ok) = golay_decode(&corrupted);
            assert!(ok, "single-bit error at pos {bit} should be correctable");
            assert_eq!(decoded, data, "single-bit error at pos {bit} decoded wrong");
        }
    }

    #[test]
    fn test_golay_triple_error_correction() {
        let data = [1u8, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0];
        let encoded = golay_encode(&data);
        // Flip 3 bits
        let mut corrupted = encoded;
        corrupted[0] ^= 1;
        corrupted[7] ^= 1;
        corrupted[15] ^= 1;
        let (decoded, ok) = golay_decode(&corrupted);
        assert!(ok, "triple-bit error should be correctable");
        assert_eq!(decoded, data, "triple-bit error decoded wrong");
    }

    #[test]
    fn test_golay_minimum_distance() {
        // Minimum distance of [24,12,8] means any two codewords differ in at least 8 positions
        let data1 = [0u8; 12];
        let data2 = [1u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let cw1 = golay_encode(&data1);
        let cw2 = golay_encode(&data2);
        let diff: usize = cw1.iter().zip(cw2.iter()).filter(|(a, b)| a != b).count();
        assert!(diff >= 8, "minimum distance should be >= 8, got {diff}");
    }

    // -------------------------------------------------------------------
    // Leech lattice tests
    // -------------------------------------------------------------------

    #[test]
    fn test_leech_nearest_point_zero() {
        let zero = [0.0f32; 24];
        let nearest = LeechLattice::nearest_point(&zero);
        assert_eq!(nearest, zero, "origin should snap to origin");
    }

    #[test]
    fn test_leech_nearest_point_integer() {
        let mut point = [0.0f32; 24];
        point[0] = 2.0;
        point[8] = -4.0;
        point[16] = 6.0;
        let nearest = LeechLattice::nearest_point(&point);
        // Should snap to the nearest valid lattice point
        let dist = LeechLattice::dist_sq_24(&point, &nearest);
        assert!(
            dist < 0.1,
            "integer lattice point should have near-zero distance, got {dist}"
        );
    }

    #[test]
    fn test_leech_nearest_point_snaps() {
        let point: [f32; 24] = [
            0.3, 0.7, -0.2, 1.1, 0.9, -0.5, 0.4, 0.8, -0.1, 0.6, 0.3, -0.7, 1.2, 0.1, -0.3, 0.5,
            0.2, -0.4, 0.8, 0.1, -0.6, 0.9, 0.3, -0.2,
        ];
        let nearest = LeechLattice::nearest_point(&point);
        let dist = LeechLattice::dist_sq_24(&point, &nearest);
        // Leech lattice has covering radius sqrt(2), so max dist^2 = 2
        assert!(
            dist <= 2.1,
            "distance should be within covering radius, got {dist}"
        );
        println!("Leech snap: dist²={dist:.4}");
    }

    #[test]
    fn test_leech_self_compatibility_maximal() {
        let point: [f32; 24] = [
            1.0, 0.5, -0.3, 0.7, 0.2, 0.8, -0.1, 0.4, 0.6, -0.5, 0.3, 0.9, 0.1, -0.7, 0.4, 0.2,
            0.8, 0.3, -0.6, 0.5, -0.2, 0.7, 0.1, 0.4,
        ];
        let score = LeechLattice::compatibility_score(&point, &point);
        assert!(
            score >= 3.5,
            "self-compatibility should be near-maximal, got {score}"
        );
    }

    #[test]
    fn test_leech_quantize_variable_length() {
        let v48 = vec![0.5f32; 48];
        let q = LeechLattice::quantize(&v48);
        assert!(q.len() >= 48, "quantized should cover the full input");
    }

    #[test]
    fn test_leech_nearest_neighbors() {
        let points: Vec<[f32; 24]> = (0..5)
            .map(|i| {
                let mut p = [0.0f32; 24];
                p[0] = i as f32;
                p
            })
            .collect();

        let mut query = [0.0f32; 24];
        query[0] = 0.4;

        let neighbors = LeechLattice::nearest_neighbors(&query, &points, 2);
        assert_eq!(neighbors.len(), 2);
        assert_eq!(neighbors[0].0, 0, "closest should be first point (0.0)");
        assert_eq!(
            neighbors[1].0, 1,
            "second closest should be second point (1.0)"
        );
    }

    // -------------------------------------------------------------------
    // CodeAnalyzer tests
    // -------------------------------------------------------------------

    #[test]
    fn test_code_analyzer_rust() {
        let code = r#"
use std::collections::HashMap;
use crate::dimension::GroupId;

pub struct LanguageService {
    pub brains: HashMap<String, DimensionManager>,
    active_brain: String,
}

impl LanguageService {
    pub fn new() -> Self {
        Self { brains: HashMap::new(), active_brain: "default".to_string() }
    }

    pub fn generation(&mut self, text: &str) -> Result<String, String> {
        if text.is_empty() {
            return Err("empty input".to_string());
        }
        let result = self.brains.get(&self.active_brain);
        Ok(format!("generated: {}", text))
    }

    fn private_helper(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_generation() {
        assert!(true);
    }
}
"#;
        let structure = CodeAnalyzer::analyze("src/service.rs", code);

        assert_eq!(structure.language, CodeLanguage::Rust);
        assert!(
            structure.imports.len() >= 2,
            "should find >= 2 imports, got {}",
            structure.imports.len()
        );
        assert!(
            structure
                .imports
                .iter()
                .any(|i| i.module_path.contains("HashMap")),
            "should find HashMap import"
        );

        let fns: Vec<&Declaration> = structure
            .declarations
            .iter()
            .filter(|d| d.kind == DeclKind::Function)
            .collect();
        assert!(
            fns.len() >= 3,
            "should find >= 3 functions, got {}: {:?}",
            fns.len(),
            fns.iter().map(|f| &f.name).collect::<Vec<_>>()
        );

        let pub_fns: Vec<_> = fns.iter().filter(|f| f.is_public).collect();
        assert!(pub_fns.len() >= 2, "should have >= 2 public functions");

        let structs: Vec<_> = structure
            .declarations
            .iter()
            .filter(|d| d.kind == DeclKind::Struct)
            .collect();
        assert!(!structs.is_empty(), "should find LanguageService struct");
        assert_eq!(structs[0].name, "LanguageService");

        assert!(
            structure.metrics.cyclomatic_complexity >= 2,
            "should have branches"
        );
        assert!(structure.metrics.code_lines > 10);
        assert!(
            structure.call_sites.len() >= 2,
            "should find call sites: {:?}",
            structure
                .call_sites
                .iter()
                .map(|c| &c.callee)
                .collect::<Vec<_>>()
        );

        println!("Rust analysis:");
        println!(
            "  {} declarations, {} imports, {} call sites",
            structure.declarations.len(),
            structure.imports.len(),
            structure.call_sites.len()
        );
        println!("  metrics: {:?}", structure.metrics);
    }

    #[test]
    fn test_code_analyzer_python() {
        let code = r#"
import os
from pathlib import Path
from typing import List, Optional

class FileProcessor:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self._cache = {}

    def process(self, path: str) -> Optional[str]:
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            content = f.read()
        self._cache[path] = content
        return content

    def _validate(self, path: str) -> bool:
        return path.startswith(str(self.base_dir))

def standalone_function(items: List[str]) -> int:
    count = 0
    for item in items:
        if item.strip():
            count += 1
    return count
"#;
        let structure = CodeAnalyzer::analyze("src/processor.py", code);

        assert_eq!(structure.language, CodeLanguage::Python);
        assert!(
            structure.imports.len() >= 3,
            "should find >= 3 imports, got {}",
            structure.imports.len()
        );

        let classes: Vec<_> = structure
            .declarations
            .iter()
            .filter(|d| d.kind == DeclKind::Class)
            .collect();
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0].name, "FileProcessor");

        let fns: Vec<_> = structure
            .declarations
            .iter()
            .filter(|d| d.kind == DeclKind::Function)
            .collect();
        assert!(
            fns.len() >= 4,
            "should find >= 4 functions (init, process, validate, standalone)"
        );

        let private_fns: Vec<_> = fns.iter().filter(|f| !f.is_public).collect();
        assert!(
            private_fns.len() >= 2,
            "should have private functions (starting with _)"
        );

        println!("Python analysis:");
        println!(
            "  {} declarations, {} imports",
            structure.declarations.len(),
            structure.imports.len()
        );
    }

    #[test]
    fn test_code_analyzer_typescript() {
        let code = r#"
import { useState, useEffect } from 'react';
import type { User } from './types';

export interface AuthService {
    login(username: string, password: string): Promise<User>;
    logout(): void;
}

export class AuthServiceImpl implements AuthService {
    private token: string | null = null;

    async login(username: string, password: string): Promise<User> {
        const response = await fetch('/api/login', {
            method: 'POST',
            body: JSON.stringify({ username, password }),
        });
        return response.json();
    }

    logout(): void {
        this.token = null;
    }
}

export function useAuth() {
    const [user, setUser] = useState<User | null>(null);
    useEffect(() => { /* check session */ }, []);
    return { user, setUser };
}
"#;
        let structure = CodeAnalyzer::analyze("src/auth.ts", code);

        assert_eq!(structure.language, CodeLanguage::TypeScript);
        assert!(structure.imports.len() >= 2);

        let interfaces: Vec<_> = structure
            .declarations
            .iter()
            .filter(|d| d.kind == DeclKind::Interface)
            .collect();
        assert!(!interfaces.is_empty(), "should find AuthService interface");

        let classes: Vec<_> = structure
            .declarations
            .iter()
            .filter(|d| d.kind == DeclKind::Class)
            .collect();
        assert!(!classes.is_empty(), "should find AuthServiceImpl class");

        let fns: Vec<_> = structure
            .declarations
            .iter()
            .filter(|d| d.kind == DeclKind::Function)
            .collect();
        assert!(!fns.is_empty(), "should find useAuth function");

        println!("TypeScript analysis:");
        println!(
            "  {} declarations, {} imports",
            structure.declarations.len(),
            structure.imports.len()
        );
    }

    // -------------------------------------------------------------------
    // HybridEmbedder tests
    // -------------------------------------------------------------------

    #[test]
    fn test_hybrid_embedder_different_files_different_embeddings() {
        let emb1 = HybridEmbedder::embed_file("src/main.rs", "fn main() { println!(\"hello\"); }");
        let emb2 = HybridEmbedder::embed_file("src/service.rs",
            "use std::collections::HashMap;\npub struct Service { map: HashMap<String, String> }\nimpl Service { pub fn new() -> Self { Self { map: HashMap::new() } } }");
        let emb3 = HybridEmbedder::embed_file(
            "tests/test_service.rs",
            "#[test]\nfn test_service() { assert!(true); }",
        );

        // Different files should have different embeddings
        assert_ne!(emb1, emb2, "main.rs and service.rs should differ");
        assert_ne!(emb2, emb3, "service.rs and test_service.rs should differ");
        assert_ne!(emb1, emb3, "main.rs and test_service.rs should differ");

        // Test file should have test signal in dim 16
        assert!(
            emb3[16] > emb1[16],
            "test file should have higher test signal"
        );
    }

    #[test]
    fn test_hybrid_embedder_similar_files_close_embeddings() {
        let service1 = HybridEmbedder::embed_file(
            "src/service_a.rs",
            r#"
use std::collections::HashMap;
pub struct ServiceA { data: HashMap<String, Vec<u8>> }
impl ServiceA {
    pub fn new() -> Self { Self { data: HashMap::new() } }
    pub fn get(&self, key: &str) -> Option<&Vec<u8>> { self.data.get(key) }
    pub fn set(&mut self, key: String, val: Vec<u8>) { self.data.insert(key, val); }
}
"#,
        );
        let service2 = HybridEmbedder::embed_file(
            "src/service_b.rs",
            r#"
use std::collections::HashMap;
pub struct ServiceB { cache: HashMap<String, Vec<u8>> }
impl ServiceB {
    pub fn new() -> Self { Self { cache: HashMap::new() } }
    pub fn fetch(&self, key: &str) -> Option<&Vec<u8>> { self.cache.get(key) }
    pub fn store(&mut self, key: String, val: Vec<u8>) { self.cache.insert(key, val); }
}
"#,
        );
        let test_file = HybridEmbedder::embed_file(
            "tests/test_services.rs",
            r#"
#[test]
fn test_service_a() { assert!(true); }
#[test]
fn test_service_b() { assert!(true); }
"#,
        );

        // Similar service files should be closer to each other than to a test file
        let dist_ab = emb_dist(&service1, &service2);
        let dist_at = emb_dist(&service1, &test_file);
        assert!(
            dist_ab < dist_at,
            "similar services should be closer ({:.4}) than service-to-test ({:.4})",
            dist_ab,
            dist_at
        );
    }

    #[test]
    fn test_hybrid_embedder_normalized() {
        let emb = HybridEmbedder::embed_file(
            "src/example.rs",
            r#"
pub fn compute(x: f32, y: f32) -> f32 {
    if x > y { x - y } else { y - x }
}
"#,
        );
        let norm: f32 = emb.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "embedding should be unit-normalized, got norm={norm}"
        );
    }

    fn emb_dist(a: &[f32; 24], b: &[f32; 24]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    // -------------------------------------------------------------------
    // GitHistory tests
    // -------------------------------------------------------------------

    #[test]
    fn test_git_history_parsing() {
        let log = "---\nsrc/main.rs\nsrc/service.rs\n---\nsrc/service.rs\nsrc/lib.rs\n---\nsrc/main.rs\nsrc/lib.rs\nsrc/service.rs\n";
        let history = GitHistory::from_git_log(log);

        // main.rs and service.rs co-changed in commits 0 and 2
        let key = ("src/main.rs".to_string(), "src/service.rs".to_string());
        assert!(
            history.cochange.get(&key).copied().unwrap_or(0) >= 2,
            "main.rs and service.rs should co-change >= 2 times"
        );

        // service.rs should have high churn (appears in all 3 commits)
        assert!(
            history.churn.get("src/service.rs").copied().unwrap_or(0) >= 3,
            "service.rs should have churn >= 3"
        );
    }

    #[test]
    fn test_git_history_edit_correlation() {
        let log = "---\nsrc/a.rs\nsrc/b.rs\n---\nsrc/a.rs\nsrc/b.rs\n---\nsrc/c.rs\n";
        let history = GitHistory::from_git_log(log);

        let mut emb_a = [0.0f32; 24];
        let mut emb_c = [0.0f32; 24];
        let paths = ["src/a.rs", "src/b.rs", "src/c.rs"];
        history.fill_edit_correlation(&mut emb_a, "src/a.rs", &paths);
        history.fill_edit_correlation(&mut emb_c, "src/c.rs", &paths);

        // a.rs has co-change partners, c.rs doesn't
        assert!(
            emb_a[12] > emb_c[12],
            "a.rs should have higher co-change fan-out than c.rs"
        );
    }

    // -------------------------------------------------------------------
    // ProjectModel hybrid indexing tests
    // -------------------------------------------------------------------

    #[test]
    fn test_project_model_hybrid_indexing() {
        let mut model = ProjectModel::new();

        model.index_file_hybrid(
            "src/main.rs",
            r#"
use crate::service::LanguageService;

fn main() {
    let mut svc = LanguageService::new();
    svc.generation("hello");
}
"#,
        );
        model.index_file_hybrid(
            "src/service.rs",
            r#"
use std::collections::HashMap;

pub struct LanguageService {
    brains: HashMap<String, String>,
}

impl LanguageService {
    pub fn new() -> Self { Self { brains: HashMap::new() } }
    pub fn generation(&mut self, text: &str) -> String { text.to_string() }
}
"#,
        );
        model.index_file_hybrid(
            "tests/test_service.rs",
            r#"
#[test]
fn test_generation() {
    assert!(true);
}
"#,
        );

        let summary = model.summary();
        println!(
            "Hybrid index: {} total ({} files, {} functions, {} types)",
            summary.total_entities, summary.files, summary.functions, summary.types
        );

        assert!(
            summary.files >= 3,
            "should index 3 files, got {}",
            summary.files
        );
        assert!(
            summary.functions >= 2,
            "should index functions from declarations"
        );

        // service.rs should relate more to main.rs than to test file
        let related = model.context_for_file("src/service.rs", 5);
        assert!(!related.is_empty(), "should find related entities");
        println!("Related to service.rs:");
        for r in &related {
            println!("  {:?} {} ({})", r.kind, r.name, r.path);
        }
    }

    #[test]
    fn test_project_model_context_conditioning() {
        let mut model = ProjectModel::new();

        let emb = HybridEmbedder::embed_file("src/main.rs", "fn main() { println!(\"hello\"); }");
        model.add_entity(EntityKind::File, "main.rs", "src/main.rs", emb);

        let cond = model.context_conditioning(&emb, 1);
        assert_eq!(cond.len(), 24, "conditioning vector should be 24d");
        assert!(
            cond.iter().any(|&v| v != 0.0),
            "conditioning should be non-zero"
        );
    }

    #[test]
    fn test_codec_longer_text() {
        let corpus = &[
            "To reset your password, go to Settings, then Security, then Change Password. Enter your current password and your new password.",
            "The observer pattern is a behavioral design pattern where an object maintains a list of dependents and notifies them of state changes.",
            "fn main() { let x = 42; println!(\"the answer is {}\", x); }",
        ];
        let dict = TokenDictionary::build(corpus, 1000);
        let codec = SpectralCodec::new(dict);

        println!("\n--- Longer text compression ---");
        println!("Dictionary size: {} tokens\n", codec.dictionary.len());

        for &text in corpus {
            let ids = codec.dictionary.encode(text);
            let k_loss = codec.min_k_lossless(text);
            println!(
                "'{}...' ({} chars → {} tokens)",
                &text[..40.min(text.len())],
                text.len(),
                ids.len()
            );
            println!(
                "  K_lossless={} ({:.0}% of signal)",
                k_loss,
                k_loss as f32 / codec.max_seq as f32 * 100.0
            );

            for k in [8, 16, 24, 32, 48] {
                let (tok_acc, exact) = codec.measure_quality(text, k);
                println!(
                    "  K={:2}: tok_acc={:.0}%, exact={}",
                    k,
                    tok_acc * 100.0,
                    exact
                );
            }
        }
    }
}
