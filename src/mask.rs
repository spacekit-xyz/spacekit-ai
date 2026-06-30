// mask.rs — Causal and padding masks for CliffordAttention
//
// The raw attention scores produced by CliffordAttention are a seq_len × seq_len
// matrix of f32 values.  Before softmax, we apply masks:
//
//   Causal mask:   score[i][j] = −∞  for  j > i   (no future attention)
//   Padding mask:  score[i][j] = −∞  for any j in a padding position
//
// After applying the mask, softmax assigns weight ≈ 0 to masked positions.

// ─── Causal mask ─────────────────────────────────────────────────────────────

/// A precomputed upper-triangular mask for a maximum sequence length.
///
/// `mask[i][j]` is `true` if position j should be *blocked* (j > i).
/// Precomputing avoids rebuilding the triangle on every forward pass.
pub struct CausalMask {
    max_len: usize,
    /// Flat row-major storage: `data[i * max_len + j]`
    data: Vec<bool>,
}

impl CausalMask {
    /// Build a causal mask for sequences up to `max_len` tokens.
    pub fn new(max_len: usize) -> Self {
        let data = (0..max_len)
            .flat_map(|i| (0..max_len).map(move |j| j > i))
            .collect();
        Self { max_len, data }
    }

    /// Return true if position j is masked (future) relative to query position i.
    #[inline]
    pub fn is_masked(&self, i: usize, j: usize) -> bool {
        self.data[i * self.max_len + j]
    }
}

/// Apply a causal mask to a mutable score matrix in place.
///
/// `scores[i][j]` is set to `NEG_INFINITY` wherever `j > i`.
///
/// `scores` — [query_len][key_len] row-major matrix
pub fn apply_causal_mask(scores: &mut Vec<Vec<f32>>) {
    for i in 0..scores.len() {
        for j in (i + 1)..scores[i].len() {
            scores[i][j] = f32::NEG_INFINITY;
        }
    }
}

/// Functional version: return a new masked score matrix without mutating.
pub fn causal_masked(scores: &[Vec<f32>]) -> Vec<Vec<f32>> {
    scores.iter().enumerate().map(|(i, row)| {
        row.iter().enumerate().map(|(j, &s)| {
            if j > i { f32::NEG_INFINITY } else { s }
        }).collect()
    }).collect()
}

// ─── Padding mask ─────────────────────────────────────────────────────────────

/// Apply a padding mask to a score matrix.
///
/// `padding` — boolean slice of length `key_len`, `true` = padding position.
///   Positions marked as padding receive score `NEG_INFINITY` in every query row.
pub fn apply_padding_mask(scores: &mut Vec<Vec<f32>>, padding: &[bool]) {
    for row in scores.iter_mut() {
        for (j, &is_pad) in padding.iter().enumerate() {
            if is_pad {
                row[j] = f32::NEG_INFINITY;
            }
        }
    }
}

// ─── Combined helper ──────────────────────────────────────────────────────────

/// Apply both causal and (optionally) padding masks to a score matrix.
///
/// This is the function you'd call inside CliffordAttention::forward before softmax.
///
/// `scores`  — mutable [query_len][key_len] score matrix
/// `padding` — optional &[bool] of length key_len; None skips padding masking
pub fn mask_scores(scores: &mut Vec<Vec<f32>>, padding: Option<&[bool]>) {
    apply_causal_mask(scores);
    if let Some(pad) = padding {
        apply_padding_mask(scores, pad);
    }
}

// ─── Batch helpers ────────────────────────────────────────────────────────────

/// Build a padding mask from a sequence of token ids, treating `pad_id` as padding.
///
/// Returns a Vec<bool> of length `seq_len` where `true` = padding.
pub fn padding_mask_from_ids(token_ids: &[usize], pad_id: usize) -> Vec<bool> {
    token_ids.iter().map(|&id| id == pad_id).collect()
}

/// Trim a score matrix to the actual sequence length in a padded batch.
/// Useful when the embedding table was built with a fixed `max_len` but the
/// actual input is shorter.
pub fn trim_scores(scores: Vec<Vec<f32>>, actual_len: usize) -> Vec<Vec<f32>> {
    scores.into_iter()
        .take(actual_len)
        .map(|row| row.into_iter().take(actual_len).collect())
        .collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_mask_upper_triangle() {
        let mask = CausalMask::new(4);
        // Lower triangle and diagonal should NOT be masked
        for i in 0..4 {
            for j in 0..=i {
                assert!(!mask.is_masked(i, j), "({i},{j}) should be visible");
            }
        }
        // Upper triangle should be masked
        for i in 0..4 {
            for j in (i+1)..4 {
                assert!(mask.is_masked(i, j), "({i},{j}) should be masked");
            }
        }
    }

    #[test]
    fn apply_causal_mask_sets_neg_inf() {
        let mut scores = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        apply_causal_mask(&mut scores);

        // Future positions should be NEG_INFINITY
        assert_eq!(scores[0][1], f32::NEG_INFINITY);
        assert_eq!(scores[0][2], f32::NEG_INFINITY);
        assert_eq!(scores[1][2], f32::NEG_INFINITY);

        // Past and present should be untouched
        assert_eq!(scores[0][0], 1.0);
        assert_eq!(scores[1][0], 4.0);
        assert_eq!(scores[2][2], 9.0);
    }

    #[test]
    fn padding_mask_blocks_pad_tokens() {
        let mut scores = vec![vec![1.0, 2.0, 3.0]; 1];
        let padding = vec![false, false, true]; // position 2 is padding
        apply_padding_mask(&mut scores, &padding);
        assert_eq!(scores[0][2], f32::NEG_INFINITY);
        assert_eq!(scores[0][0], 1.0);
    }

    #[test]
    fn mask_scores_combined() {
        let mut scores = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let padding = vec![false, false, true];
        mask_scores(&mut scores, Some(&padding));

        // Upper triangle (causal)
        assert_eq!(scores[0][1], f32::NEG_INFINITY);
        // Padding column
        assert_eq!(scores[1][2], f32::NEG_INFINITY); // would have been masked by causal anyway
        assert_eq!(scores[2][2], f32::NEG_INFINITY); // padding mask takes effect here (diagonal)
    }
}
