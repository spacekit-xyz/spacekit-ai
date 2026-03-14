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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenDictionary {
    pub tokens: Vec<String>, // ID -> token. Index 0 = <EOS>
    lookup: HashMap<String, u16>,
}

impl TokenDictionary {
    /// Build from a corpus. Keeps the `max_size` most frequent tokens.
    /// ID 0 is reserved for <EOS>.
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

        let mut tokens = Vec::with_capacity(entries.len() + 1);
        let mut lookup = HashMap::new();
        tokens.push("<EOS>".to_string()); // ID 0 = end of sequence
        for (token, _) in entries {
            let id = tokens.len() as u16;
            lookup.insert(token.clone(), id);
            tokens.push(token);
        }
        Self { tokens, lookup }
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
                    && !result.ends_with(|c: char| c.is_ascii_punctuation())
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
