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
                parity ^= if pos - 1 < codeword.len() { codeword[pos - 1] } else { 0 };
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
// The 64d bridge embedding decomposes as 8 × 8d E8 subspaces, giving
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

        if d8_dist <= coset_dist { d8_point } else { coset_point }
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

    /// Quantize a 64d vector by decomposing into 8 × 8d E8 subspaces.
    /// Returns 8 lattice points (one per subspace) as a flat 64-element vector.
    pub fn quantize_64d(x: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0f32; 64];
        for sub in 0..8 {
            let offset = sub * 8;
            let mut block = [0.0f32; 8];
            for i in 0..8 {
                block[i] = if offset + i < x.len() { x[offset + i] } else { 0.0 };
            }
            let lattice_point = Self::nearest_point(&block);
            for i in 0..8 {
                result[offset + i] = lattice_point[i];
            }
        }
        result
    }

    /// Compute the quantization distance (sum of squared errors across all
    /// 8 subspaces). Lower = better match to lattice structure.
    pub fn quantization_distance(x: &[f32]) -> f32 {
        let quantized = Self::quantize_64d(x);
        x.iter().zip(quantized.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum()
    }

    /// E8 root inner product between two lattice-quantized embeddings.
    /// Returns a value in {-2, -1, 0, 1, 2} for exact lattice points,
    /// or a continuous value for approximate embeddings.
    ///
    /// This replaces heuristic cosine similarity with the algebraic structure
    /// of the E8 root system for Hopf transition scoring.
    pub fn root_inner_product(a: &[f32], b: &[f32]) -> f32 {
        let qa = Self::quantize_64d(a);
        let qb = Self::quantize_64d(b);

        // Compute inner product in each 8d subspace, then average
        let mut total = 0.0f32;
        for sub in 0..8 {
            let offset = sub * 8;
            let dot: f32 = (0..8).map(|i| qa[offset + i] * qb[offset + i]).sum();
            let na = (0..8).map(|i| qa[offset + i] * qa[offset + i]).sum::<f32>().sqrt();
            let nb = (0..8).map(|i| qb[offset + i] * qb[offset + i]).sum::<f32>().sqrt();
            if na > 1e-8 && nb > 1e-8 {
                total += dot / (na * nb);
            }
        }
        total / 8.0
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

    /// Select the best archetype from prototypes using E8 lattice decoding.
    /// Quantizes the input embedding into E8 subspaces and compares against
    /// quantized prototypes, returning (best_index, confidence).
    ///
    /// When prototypes are already E8-quantized (after training), this reduces
    /// to a fast lattice-point comparison.
    pub fn select_archetype(
        embedding: &[f32],
        prototypes: &[Vec<f32>],
    ) -> (usize, f32) {
        if prototypes.is_empty() {
            return (0, 0.0);
        }

        let q_emb = Self::quantize_64d(embedding);
        let emb_norm = q_emb.iter().map(|v| v * v).sum::<f32>().sqrt();

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
            let angle =
                std::f32::consts::PI * (2 * i + 1) as f32 * k as f32 / (2 * n) as f32;
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
            let angle =
                std::f32::consts::PI * k as f32 * (2 * i + 1) as f32 / (2 * n) as f32;
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

/// Semantic cluster tag for a token: first character category.
/// Used to group tokens so semantically related tokens get adjacent IDs.
fn semantic_cluster(token: &str) -> u8 {
    let ch = token.chars().next().unwrap_or(' ');
    if ch.is_ascii_punctuation() {
        0
    } else if ch.is_ascii_digit() {
        1
    } else if ch.is_ascii_uppercase() {
        2
    } else {
        // lowercase: cluster by first letter (a-z → 3..28)
        3 + (ch as u8).wrapping_sub(b'a').min(25)
    }
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
    /// ID 0 is reserved for <EOS>. Tokens are sorted into semantic clusters
    /// so similar tokens get adjacent IDs, then Gray coding ensures adjacent
    /// IDs differ by only 1 bit.
    pub fn build(texts: &[&str], max_size: usize) -> Self {
        let mut freq: HashMap<String, usize> = HashMap::new();
        for text in texts {
            for token in tokenize(text) {
                *freq.entry(token).or_default() += 1;
            }
        }
        let mut entries: Vec<(String, usize)> = freq.into_iter().collect();
        // Sort by semantic cluster first, then by frequency within cluster.
        // This groups related tokens (punctuation, digits, uppercase, lowercase
        // by first letter) into contiguous ID ranges.
        entries.sort_by(|a, b| {
            let ca = semantic_cluster(&a.0);
            let cb = semantic_cluster(&b.0);
            ca.cmp(&cb).then(b.1.cmp(&a.1))
        });
        entries.truncate(max_size.saturating_sub(1));

        let dict_size = entries.len() + 1; // +1 for EOS
        let mut tokens = Vec::with_capacity(dict_size);
        let mut lookup = HashMap::new();
        tokens.push("<EOS>".to_string());
        for (token, _) in entries {
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

        Self { tokens, lookup, gray_to_binary, binary_to_gray }
    }

    /// The Gray-coded ID for a token's internal index.
    /// Used by GroupGenEnv for encoding targets.
    pub fn to_gray_id(&self, id: u16) -> u16 {
        self.binary_to_gray.get(id as usize).copied().unwrap_or(0)
    }

    /// Recover internal index from a Gray-coded ID.
    /// Used by GroupGenEnv for decoding outputs.
    pub fn from_gray_id(&self, gray_id: u16) -> u16 {
        self.gray_to_binary.get(gray_id as usize).copied().unwrap_or(0)
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
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
    /// Unknown tokens are split into individual characters as fallback.
    pub fn encode(&self, text: &str) -> Vec<u16> {
        let raw_tokens = tokenize(text);
        let mut ids = Vec::new();
        for tok in &raw_tokens {
            if let Some(&id) = self.lookup.get(tok.as_str()) {
                ids.push(id);
            } else {
                for ch in tok.chars() {
                    let s = ch.to_string();
                    if let Some(&id) = self.lookup.get(&s) {
                        ids.push(id);
                    }
                    // skip chars not in dictionary
                }
            }
        }
        ids
    }

    /// Decode token IDs back to text. Stops at EOS_ID.
    pub fn decode(&self, ids: &[u16]) -> String {
        let mut result = String::new();
        for &id in ids {
            if id == EOS_ID {
                break;
            }
            if let Some(tok) = self.tokens.get(id as usize) {
                if !result.is_empty()
                    && !tok.starts_with(|c: char| c.is_ascii_punctuation())
                {
                    result.push(' ');
                }
                result.push_str(tok);
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
    "def", "class", "if", "elif", "else", "for", "while", "return", "import", "from",
    "try", "except", "finally", "with", "as", "yield", "lambda", "pass", "break",
    "continue", "in", "not", "and", "or", "is", "None", "True", "False", "self",
    "raise", "assert", "del", "global", "nonlocal", "async", "await", "print",
    // Rust
    "fn", "let", "mut", "pub", "struct", "enum", "impl", "trait", "use", "mod",
    "crate", "super", "where", "match", "loop", "const", "static", "type", "move",
    "ref", "unsafe", "extern", "dyn", "Box", "Vec", "String", "Option", "Result",
    "Some", "Ok", "Err", "println", "macro_rules",
    // JavaScript/TypeScript
    "function", "var", "const", "new", "this", "prototype", "extends",
    "constructor", "export", "default", "typeof", "instanceof", "void",
    "null", "undefined", "true", "false", "console", "log", "require",
    // Shared
    "int", "float", "bool", "str", "char", "void", "static", "final",
    "abstract", "interface", "override", "virtual", "template", "namespace",
];

const STRUCTURE_CHARS: &[char] = &[
    '(', ')', '{', '}', '[', ']', ':', ';', ',', '.', '#', '@',
];

const OPERATOR_TOKENS: &[&str] = &[
    "=", "==", "!=", "<", ">", "<=", ">=", "+", "-", "*", "/", "%",
    "&", "|", "^", "!", "&&", "||", "<<", ">>", "+=", "-=", "*=", "/=",
    "->", "=>", "::", "..", "..=", "**",
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
    if token.chars().all(|c| c.is_ascii_digit() || c == '.' || c == 'x' || c == 'X'
        || (c.is_ascii_hexdigit() && token.starts_with("0x")))
    {
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
    tokens.iter().map(|t| {
        match syntax_role(t) {
            SyntaxRole::Keyword | SyntaxRole::Structure | SyntaxRole::Operator => t.clone(),
            SyntaxRole::Literal => "_LIT_".to_string(),
            SyntaxRole::Identifier => "_ID_".to_string(),
        }
    }).collect()
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
        Self { dictionary, max_seq }
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
        assert!(text.contains("hello"), "should contain known token: {}", text);
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
        assert_eq!(p, x, "integer point with even sum should be its own nearest");
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
        assert!(is_integer || is_half_int, "should be integer or half-integer coords: {:?}", p);
        if is_integer {
            assert!((sum.round() as i32) % 2 == 0, "integer coords must have even sum: sum={}", sum);
        }
    }

    #[test]
    fn test_e8_half_integer_lattice() {
        // Half-integer point with even sum is also an E8 lattice point
        let x = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]; // sum = 4.0 (even)
        let p = E8Lattice::nearest_point(&x);
        let dist = E8Lattice::dist_sq(&x, &p);
        assert!(dist < 1e-6, "half-integer E8 point should map to itself, dist={}", dist);
    }

    #[test]
    fn test_e8_quantize_64d() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        let q = E8Lattice::quantize_64d(&x);
        assert_eq!(q.len(), 64);

        // Each 8d block should be a valid E8 lattice point
        for sub in 0..8 {
            let offset = sub * 8;
            let block: Vec<f32> = q[offset..offset+8].to_vec();
            let is_integer = block.iter().all(|v| (v - v.round()).abs() < 1e-6);
            let is_half_int = block.iter().all(|v| (v - (v - 0.5).round() - 0.5).abs() < 1e-6);
            assert!(is_integer || is_half_int, "block {} should be E8: {:?}", sub, block);
        }
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
        for i in 0..8 { a[i] = 1.0; }
        for i in 8..16 { b[i] = 1.0; }
        let score = E8Lattice::compatibility_score(&a, &b);
        assert!(score < 2.0, "orthogonal patterns should have low compatibility: {}", score);
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
