//! Arithmetic (range) coding — the concrete realisation of the
//! prediction ⇄ compression equivalence.
//!
//! A next-token model assigns probability `p` to the true token.  Shannon's
//! source-coding theorem says that token can be written in `−log₂ p` bits, and
//! arithmetic coding achieves this rate to within a couple of bits for the
//! whole message.  So a language model *is* a lossless compressor: drive an
//! arithmetic coder with the model's per-step distribution and you get an
//! optimal code for the data the model predicts well.
//!
//! This module is a small, dependency-free Witten–Neal–Cleary style coder with
//! 32-bit registers.  Frequencies are integers summing to `total` (a power of
//! two ≤ 2¹⁶); [`quantize`] turns a float distribution into such a table.
//!
//! See the tests for an end-to-end round trip that encodes a token sequence
//! using a real [`crate::v2::ModelStateV2`]'s distribution and decodes it back
//! bit-for-bit, asserting the encoded length matches `Σ −log₂ p`.

const TOP:   u64 = 1 << 32;
const MASK:  u64 = TOP - 1;        // 0xFFFF_FFFF
const HALF:  u64 = 1 << 31;
const QTR:   u64 = 1 << 30;
const TQTR:  u64 = 3 << 30;

/// Maximum frequency total (denominator).  Keeps `range * total` within u64.
pub const FREQ_BITS:  u32 = 16;
pub const FREQ_TOTAL: u64 = 1 << FREQ_BITS;

// ─── Quantisation ─────────────────────────────────────────────────────────────

/// Convert a probability distribution into integer frequencies that sum to
/// exactly `FREQ_TOTAL`, with every entry ≥ 1 (so no symbol is uncodable).
pub fn quantize(probs: &[f32]) -> Vec<u64> {
    let total = FREQ_TOTAL;
    let n = probs.len();
    let mut freq: Vec<u64> = probs
        .iter()
        .map(|&p| (((p as f64) * total as f64).round() as i64).max(1) as u64)
        .collect();

    // Make the sum exactly `total` by correcting the most probable symbol.
    let sum: i64 = freq.iter().map(|&f| f as i64).sum();
    let maxi = (0..n).max_by(|&a, &b| freq[a].cmp(&freq[b])).unwrap_or(0);
    let corrected = freq[maxi] as i64 + (total as i64 - sum);
    debug_assert!(corrected >= 1, "quantisation underflow; vocab too large for FREQ_TOTAL");
    freq[maxi] = corrected.max(1) as u64;
    freq
}

/// Cumulative `(low, high)` frequency for `symbol` given a frequency table.
pub fn cumulative(freq: &[u64], symbol: usize) -> (u64, u64) {
    let low: u64 = freq[..symbol].iter().sum();
    (low, low + freq[symbol])
}

// ─── Encoder ──────────────────────────────────────────────────────────────────

pub struct ArithmeticEncoder {
    low: u64,
    high: u64,
    pending: u64,
    bits: Vec<bool>,
}

impl Default for ArithmeticEncoder {
    fn default() -> Self { Self::new() }
}

impl ArithmeticEncoder {
    pub fn new() -> Self {
        Self { low: 0, high: MASK, pending: 0, bits: Vec::new() }
    }

    fn emit(&mut self, bit: bool) {
        self.bits.push(bit);
        while self.pending > 0 {
            self.bits.push(!bit);
            self.pending -= 1;
        }
    }

    /// Encode one symbol described by its cumulative interval over `total`.
    pub fn encode(&mut self, cum_low: u64, cum_high: u64, total: u64) {
        let range = self.high - self.low + 1;
        self.high = self.low + range * cum_high / total - 1;
        self.low  = self.low + range * cum_low / total;

        loop {
            if self.high < HALF {
                self.emit(false);
            } else if self.low >= HALF {
                self.emit(true);
                self.low  -= HALF;
                self.high -= HALF;
            } else if self.low >= QTR && self.high < TQTR {
                self.pending += 1;
                self.low  -= QTR;
                self.high -= QTR;
            } else {
                break;
            }
            self.low  = (self.low << 1) & MASK;
            self.high = ((self.high << 1) | 1) & MASK;
        }
    }

    /// Flush remaining state and return `(bytes, n_bits)`.
    pub fn finish(mut self) -> (Vec<u8>, usize) {
        self.pending += 1;
        if self.low < QTR { self.emit(false); } else { self.emit(true); }

        let n_bits = self.bits.len();
        let mut bytes = vec![0u8; n_bits.div_ceil(8)];
        for (i, &b) in self.bits.iter().enumerate() {
            if b {
                bytes[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        (bytes, n_bits)
    }
}

// ─── Decoder ──────────────────────────────────────────────────────────────────

pub struct ArithmeticDecoder<'a> {
    low: u64,
    high: u64,
    value: u64,
    bytes: &'a [u8],
    n_bits: usize,
    pos: usize,
}

impl<'a> ArithmeticDecoder<'a> {
    pub fn new(bytes: &'a [u8], n_bits: usize) -> Self {
        let mut d = Self { low: 0, high: MASK, value: 0, bytes, n_bits, pos: 0 };
        for _ in 0..32 {
            d.value = (d.value << 1) | d.next_bit();
        }
        d
    }

    fn next_bit(&mut self) -> u64 {
        let bit = if self.pos < self.n_bits {
            let byte = self.bytes[self.pos / 8];
            ((byte >> (7 - (self.pos % 8))) & 1) as u64
        } else {
            0
        };
        self.pos += 1;
        bit
    }

    /// Scaled value used to locate the next symbol within `total`.
    pub fn target(&self, total: u64) -> u64 {
        let range = self.high - self.low + 1;
        ((self.value - self.low + 1) * total - 1) / range
    }

    /// Advance past a symbol once its `(cum_low, cum_high)` interval is known.
    pub fn update(&mut self, cum_low: u64, cum_high: u64, total: u64) {
        let range = self.high - self.low + 1;
        self.high = self.low + range * cum_high / total - 1;
        self.low  = self.low + range * cum_low / total;

        loop {
            if self.high < HALF {
                // nothing
            } else if self.low >= HALF {
                self.value -= HALF;
                self.low   -= HALF;
                self.high  -= HALF;
            } else if self.low >= QTR && self.high < TQTR {
                self.value -= QTR;
                self.low   -= QTR;
                self.high  -= QTR;
            } else {
                break;
            }
            self.low   = (self.low << 1) & MASK;
            self.high  = ((self.high << 1) | 1) & MASK;
            self.value = ((self.value << 1) | self.next_bit()) & MASK;
        }
    }
}

/// Find the symbol whose cumulative interval contains `target`.
pub fn find_symbol(freq: &[u64], target: u64) -> usize {
    let mut acc = 0u64;
    for (s, &f) in freq.iter().enumerate() {
        if target < acc + f {
            return s;
        }
        acc += f;
    }
    freq.len() - 1
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::sample::softmax;
    use crate::v2::tape::model_forward_logits;
    use crate::v2::train_v2::{ModelStateV2, TrainConfigV2};

    /// Encode/decode a fixed integer sequence with static (non-model) tables.
    #[test]
    fn round_trip_static_distribution() {
        let probs = vec![0.5f32, 0.25, 0.125, 0.125];
        let freq = quantize(&probs);
        assert_eq!(freq.iter().sum::<u64>(), FREQ_TOTAL);

        let msg = [0usize, 1, 0, 3, 2, 0, 0, 1, 3, 2, 1, 0];

        let mut enc = ArithmeticEncoder::new();
        for &s in &msg {
            let (lo, hi) = cumulative(&freq, s);
            enc.encode(lo, hi, FREQ_TOTAL);
        }
        let (bytes, n_bits) = enc.finish();

        let mut dec = ArithmeticDecoder::new(&bytes, n_bits);
        let mut out = Vec::new();
        for _ in 0..msg.len() {
            let t = dec.target(FREQ_TOTAL);
            let s = find_symbol(&freq, t);
            let (lo, hi) = cumulative(&freq, s);
            dec.update(lo, hi, FREQ_TOTAL);
            out.push(s);
        }
        assert_eq!(out, msg);
    }

    /// The real equivalence test: encode a token sequence with a model's
    /// next-token distribution, decode it back losslessly, and check that the
    /// code length matches the model's information content `Σ −log₂ p`.
    #[test]
    fn round_trip_against_model_distribution() {
        // Tiny model; weights are random but fixed (seeded init).
        let mut cfg = TrainConfigV2::small(40);
        cfg.d_model = 8;
        cfg.n_heads = 2;
        cfg.d_ff = 16;
        cfg.n_blocks = 2;
        let state = ModelStateV2::new(cfg);

        // A fixed token sequence to transmit (ids within vocab).
        let seq: Vec<usize> = vec![2, 7, 13, 5, 7, 20, 31, 5, 7, 13, 9, 2, 18, 7];

        // Distribution for the token *after* `prefix` (last position's logits).
        let dist_at = |prefix: &[usize]| -> Vec<f32> {
            let logits = model_forward_logits(&state.alg, &state.model, prefix, true, state.cfg.dot_attention);
            softmax(logits.last().unwrap())
        };

        // ── Encode (token 0 is assumed known by the decoder, e.g. BOS) ──
        let mut enc = ArithmeticEncoder::new();
        let mut ideal_bits = 0.0f64;
        for p in 0..seq.len() - 1 {
            let probs = dist_at(&seq[..=p]);
            let freq = quantize(&probs);
            let target = seq[p + 1];
            let (lo, hi) = cumulative(&freq, target);
            enc.encode(lo, hi, FREQ_TOTAL);
            ideal_bits += -((freq[target] as f64 / FREQ_TOTAL as f64).log2());
        }
        let (bytes, n_bits) = enc.finish();

        // ── Decode: reconstruct context token-by-token, re-running the model ──
        let mut dec = ArithmeticDecoder::new(&bytes, n_bits);
        let mut decoded = vec![seq[0]];
        for _ in 0..seq.len() - 1 {
            let probs = dist_at(&decoded);
            let freq = quantize(&probs);
            let t = dec.target(FREQ_TOTAL);
            let s = find_symbol(&freq, t);
            let (lo, hi) = cumulative(&freq, s);
            dec.update(lo, hi, FREQ_TOTAL);
            decoded.push(s);
        }

        // Lossless: decoded sequence is identical to the original.
        assert_eq!(decoded, seq, "arithmetic round trip must be lossless");

        // Code length is within a small constant of the model's entropy.
        let overhead = n_bits as f64 - ideal_bits;
        assert!(
            overhead >= -1.0 && overhead < 32.0,
            "coded {n_bits} bits vs ideal {ideal_bits:.2} (overhead {overhead:.2}) out of range"
        );
    }
}
