//! Lightweight connector heuristics for sentiment lattice retrieval.
//! Emits the same `gfcausal_t_*_c_*` tokens as [`crate::dimension::language::CausalAnnotation`]
//! in training JSONL so BM25 / lex-align can match without a separate causal brain.

use crate::dimension::language::causal_index_token;

/// Append at most a few causal index tokens derived from `intent_text` (lowercased, padded).
pub fn extend_subject_keywords_with_causal_tokens(intent_text: &str, subject_kw: &mut Vec<String>) {
    for tok in causal_bm25_tokens(intent_text) {
        if tok.len() > 2 && !subject_kw.iter().any(|x| x == &tok) {
            subject_kw.push(tok);
        }
    }
}

/// Heuristic connector → training-aligned tokens. Longer phrases first.
pub fn causal_bm25_tokens(intent_text: &str) -> Vec<String> {
    let padded = format!(
        " {} ",
        intent_text
            .to_ascii_lowercase()
            .replace(['\n', '\r', '\t'], " ")
    );
    let mut out: Vec<String> = Vec::new();

    if padded.contains(" even though ") {
        out.push(causal_index_token("contrastive", "even though"));
    } else if padded.contains(" at least ") {
        out.push(causal_index_token("compensatory", "at least"));
    } else if padded.contains(" would have ") || padded.contains(" would've ") {
        out.push(causal_index_token("counterfactual", "would have"));
    } else if padded.contains(" which means ") {
        out.push(causal_index_token("inferential", "which means"));
    } else if padded.contains(" despite ") {
        out.push(causal_index_token("contrastive", "despite"));
    } else if padded.contains(" although ") {
        out.push(causal_index_token("contrastive", "although"));
    } else if padded.contains(" because ") {
        out.push(causal_index_token("explanatory", "because"));
    } else if padded.contains(" therefore ") {
        out.push(causal_index_token("direct", "therefore"));
    } else if padded.contains(" however ") {
        out.push(causal_index_token("compensatory", "however"));
    } else if padded.contains(" but ") {
        out.push(causal_index_token("compensatory", "but"));
    } else if padded.contains(" so ") {
        out.push(causal_index_token("direct", "so"));
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_so_direct() {
        let v = causal_bm25_tokens("I lost big so I'm furious");
        assert!(v.iter().any(|t| t.contains("direct") && t.contains("so")));
    }

    #[test]
    fn longer_connector_wins_even_though() {
        let v = causal_bm25_tokens("Green even though volume died");
        assert!(v.iter().any(|t| t.contains("contrastive")));
        assert!(!v.iter().any(|t| t.contains("but")));
    }

    #[test]
    fn at_least_compensatory() {
        let v = causal_bm25_tokens("The tape was ugly at least funding cooled");
        assert!(v.iter().any(|t| t.contains("compensatory") && t.contains("at_least")));
    }
}
