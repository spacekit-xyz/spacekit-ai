//! Lightweight polarity probe: extracts sentiment axis features from the raw
//! encoder vector so the lattice conditioning has an explicit positive/negative
//! dimension — without adding external dependencies.
//!
//! The [`HashingLanguageEncoder`](super::language::HashingLanguageEncoder) now places
//! positive-sentiment energy at index 8 and negative-sentiment energy at index 9.
//! This module reads those two values and derives a small feature vector that is
//! appended into the zero-padded tail of the conditioning vector (indices 176–191
//! inside the 192-D `GEN_COND_DIM`).
//!
//! Layout of `polarity_features_from_raw` output (up to 16 floats):
//!
//! | idx | meaning |
//! |-----|---------|
//! | 0 | positive mass (v[8] rescaled) |
//! | 1 | negative mass (v[9] rescaled) |
//! | 2 | net polarity (pos − neg), signed |
//! | 3 | polarity magnitude (pos + neg) — high = opinionated |
//! | 4 | mixed indicator: min(pos,neg) / max(pos,neg) — high = both poles active |
//! | 5–15 | reserved (zero) |

/// Number of floats produced by [`polarity_features_from_raw`].
pub const POLARITY_FEATURE_DIM: usize = 16;

/// Positive anchor dimension in the raw encoder vector.
const POS_IDX: usize = 8;
/// Negative anchor dimension in the raw encoder vector.
const NEG_IDX: usize = 9;

/// Rescaling: the raw dimension accumulates `+4.0` per keyword hit and is then
/// L2-normalised with the rest of the vector. Values are typically small (0–0.4
/// range after normalisation). We amplify by this factor so the polarity signal
/// is competitive with the 128-D base + 48-D understanding block.
const AMPLIFY: f32 = 6.0;

/// Extract polarity features from the **raw** encoder output (before bridging).
///
/// Safe to call with any slice length (missing indices default to 0).
pub fn polarity_features_from_raw(h_raw: &[f32]) -> [f32; POLARITY_FEATURE_DIM] {
    let pos = h_raw.get(POS_IDX).copied().unwrap_or(0.0).max(0.0) * AMPLIFY;
    let neg = h_raw.get(NEG_IDX).copied().unwrap_or(0.0).max(0.0) * AMPLIFY;

    let net = pos - neg;
    let mag = pos + neg;
    let mixed = if mag > 1e-6 {
        pos.min(neg) / pos.max(neg)
    } else {
        0.0
    };

    let mut out = [0.0f32; POLARITY_FEATURE_DIM];
    out[0] = pos;
    out[1] = neg;
    out[2] = net;
    out[3] = mag;
    out[4] = mixed;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_raw_yields_zero_features() {
        let f = polarity_features_from_raw(&[]);
        assert!(f.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn positive_only_has_positive_net() {
        let mut raw = vec![0.0f32; 384];
        raw[POS_IDX] = 0.2;
        let f = polarity_features_from_raw(&raw);
        assert!(f[0] > 0.0, "pos mass");
        assert!(f[1] == 0.0, "neg mass zero");
        assert!(f[2] > 0.0, "net positive");
        assert!(f[4] == 0.0, "no mixed signal");
    }

    #[test]
    fn negative_only_has_negative_net() {
        let mut raw = vec![0.0f32; 384];
        raw[NEG_IDX] = 0.15;
        let f = polarity_features_from_raw(&raw);
        assert!(f[0] == 0.0);
        assert!(f[1] > 0.0);
        assert!(f[2] < 0.0, "net negative");
    }

    #[test]
    fn both_poles_yield_mixed_indicator() {
        let mut raw = vec![0.0f32; 384];
        raw[POS_IDX] = 0.1;
        raw[NEG_IDX] = 0.1;
        let f = polarity_features_from_raw(&raw);
        assert!((f[4] - 1.0).abs() < 1e-5, "equal poles → mixed=1.0");
    }

    #[test]
    fn opposite_prompts_differ_in_polarity() {
        use crate::dimension::language::{EncoderPreset, HashingLanguageEncoder, LanguageEncoder};

        let enc = HashingLanguageEncoder::new(EncoderPreset::MiniLmL6V2);
        let pos_raw = enc.encode("I love this product, it is amazing and wonderful");
        let neg_raw = enc.encode("I hate this product, it is terrible and awful");

        let pf_pos = polarity_features_from_raw(&pos_raw);
        let pf_neg = polarity_features_from_raw(&neg_raw);

        assert!(
            pf_pos[0] > pf_pos[1],
            "positive prompt: pos mass ({}) should exceed neg mass ({})",
            pf_pos[0],
            pf_pos[1]
        );
        assert!(
            pf_neg[1] > pf_neg[0],
            "negative prompt: neg mass ({}) should exceed pos mass ({})",
            pf_neg[1],
            pf_neg[0]
        );
        assert!(
            pf_pos[2] > 0.0,
            "positive prompt: net polarity should be positive"
        );
        assert!(
            pf_neg[2] < 0.0,
            "negative prompt: net polarity should be negative"
        );
    }
}
