//! Word-boundary-aware keyword matching for topic/concept routing.
//!
//! Naive `str::contains` mis-fires on common English (`mean` ⊂ `meant`,
//! `quarter` ⊂ `headquarters`, `pr` ⊂ `presentation`). Single-token graph
//! keywords use ASCII alphanumerics + `_` as word characters.

#[inline]
fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// `lower` must already be lowercased.
///
/// - Multi-word `kw` (any whitespace): substring search (phrase match).
/// - Single token: match only at word boundaries.
pub fn keyword_matches_in_lower(lower: &str, kw: &str) -> bool {
    let kw = kw.trim();
    if kw.is_empty() {
        return false;
    }
    if kw.chars().any(char::is_whitespace) {
        return lower.contains(kw);
    }
    let hay = lower.as_bytes();
    for (idx, _) in lower.match_indices(kw) {
        let prev_ok = idx == 0 || !is_word_byte(hay[idx - 1]);
        let end = idx + kw.len();
        let next_ok = end >= hay.len() || !is_word_byte(hay[end]);
        if prev_ok && next_ok {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_not_meant() {
        let lower = "she meant more than anything".to_lowercase();
        assert!(!keyword_matches_in_lower(&lower, "mean"));
        assert!(keyword_matches_in_lower("the mean is 3", "mean"));
    }

    #[test]
    fn quarter_not_headquarters() {
        let lower = "we moved headquarters to austin".to_lowercase();
        assert!(!keyword_matches_in_lower(&lower, "quarter"));
        assert!(keyword_matches_in_lower("great quarter everyone", "quarter"));
    }

    #[test]
    fn phrase_unchanged() {
        let lower = "add two numbers together".to_lowercase();
        assert!(keyword_matches_in_lower(&lower, "add two"));
    }
}
