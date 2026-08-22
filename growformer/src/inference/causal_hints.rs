//! Lightweight connector heuristics for sentiment lattice retrieval.
//! Emits the same `gfcausal_t_*_c_*` tokens as [`crate::dimension::language::CausalAnnotation`]
//! in training JSONL so BM25 / lex-align can match without a separate causal brain.

use crate::dimension::language::{causal_index_token, causal_subtype_index_token};

/// Append at most a few causal index tokens derived from `intent_text` (lowercased, padded).
pub fn extend_subject_keywords_with_causal_tokens(intent_text: &str, subject_kw: &mut Vec<String>) {
    for tok in causal_bm25_tokens(intent_text) {
        if tok.len() > 2 && !subject_kw.iter().any(|x| x == &tok) {
            subject_kw.push(tok);
        }
    }
}

/// Heuristic connector → training-aligned tokens. Longer phrases first.
///
/// Two passes:
///  1. Primary connector → `gfcausal_t_*_c_*` (first match wins).
///  2. Subtype overlay → `gfcausal_st_*` for retrospective / interventional hooks
///     (fires independently so both tokens can appear).
pub fn causal_bm25_tokens(intent_text: &str) -> Vec<String> {
    let padded = format!(
        " {} ",
        intent_text
            .to_ascii_lowercase()
            .replace(['\n', '\r', '\t'], " ")
    );
    let mut out: Vec<String> = Vec::new();

    // --- pass 1: primary connector (longest-phrase-first, first match wins) ---
    if padded.contains(" even though ") {
        out.push(causal_index_token("contrastive", "even though"));
    } else if padded.contains(" at least ") {
        out.push(causal_index_token("compensatory", "at least"));
    } else if padded.contains(" would have ") || padded.contains(" would've ") {
        out.push(causal_index_token("counterfactual", "would have"));
    } else if padded.contains(" wouldn't have ") || padded.contains(" wouldnt have ") {
        out.push(causal_index_token("counterfactual", "would have"));
    } else if padded.contains(" which means ") {
        out.push(causal_index_token("inferential", "which means"));
    } else if padded.contains(" which suggests ") {
        out.push(causal_index_token("inferential", "which suggests"));
    } else if padded.contains(" which implies ") {
        out.push(causal_index_token("inferential", "which implies"));
    } else if padded.contains(" despite ") {
        out.push(causal_index_token("contrastive", "despite"));
    } else if padded.contains(" although ") {
        out.push(causal_index_token("contrastive", "although"));
    } else if padded.contains(" because ") {
        out.push(causal_index_token("explanatory", "because"));
    } else if padded.contains(" triggered ") {
        out.push(causal_index_token("direct", "triggered"));
    } else if padded.contains(" caused ") {
        out.push(causal_index_token("direct", "caused"));
    } else if padded.contains(" causing ") {
        out.push(causal_index_token("direct", "causing"));
    } else if padded.contains(", pushing ") || padded.contains(" pushing ") {
        out.push(causal_index_token("direct", "pushing"));
    } else if padded.contains(" resulted in ") {
        out.push(causal_index_token("direct", "resulted in"));
    } else if padded.contains(" led to ") || padded.contains(" lead to ") {
        out.push(causal_index_token("direct", "led to"));
    } else if padded.contains(" therefore ") {
        out.push(causal_index_token("direct", "therefore"));
    } else if padded.contains(" however ") {
        out.push(causal_index_token("compensatory", "however"));
    } else if padded.contains(", implying ") || padded.contains(" implying ") {
        out.push(causal_index_token("inferential", "implying"));
    } else if padded.contains(", suggesting ") || padded.contains(" suggesting ") {
        out.push(causal_index_token("inferential", "suggesting"));
    } else if padded.contains(", meaning ") || padded.contains(" meaning ") {
        out.push(causal_index_token("inferential", "meaning"));
    } else if padded.contains(" yet somehow ") || padded.contains(" but somehow ") {
        out.push(causal_index_token("concessive", "yet somehow"));
    } else if padded.contains(" but still ") {
        out.push(causal_index_token("concessive", "but still"));
    } else if padded.contains(" yet ") || padded.contains(", yet ") {
        out.push(causal_index_token("contrastive", "yet"));
    } else if padded.contains(" but ") {
        out.push(causal_index_token("compensatory", "but"));
    } else if padded.contains(", so ") || padded.contains(". so ") || padded.contains("; so ") {
        out.push(causal_index_token("direct", "so"));
    } else if padded.contains(" so ") {
        out.push(causal_index_token("direct", "so"));
    } else if padded.contains(" since ") {
        out.push(causal_index_token("explanatory", "since"));
    } else if padded.contains(" thus ") || padded.contains(" thereby ") {
        out.push(causal_index_token("direct", "thus"));
    } else if padded.contains("; still,") || padded.contains("; still ") {
        out.push(causal_index_token("concessive", "still"));
    } else if padded.contains(" still ") || padded.contains(" somehow ") {
        out.push(causal_index_token("concessive", "still"));
    } else if padded.contains(" nevertheless ") {
        out.push(causal_index_token("concessive", "nevertheless"));
    }

    // --- pass 2: subtype overlay (independent of pass 1) ---
    if has_retrospective_hook(&padded) {
        let st = causal_subtype_index_token("retrospective_framing");
        if !st.is_empty() && !out.iter().any(|t| t == &st) {
            out.push(st);
        }
    }
    if has_interventional_hook(&padded) {
        let st = causal_subtype_index_token("interventional_counterfactual");
        if !st.is_empty() && !out.iter().any(|t| t == &st) {
            out.push(st);
        }
    }

    out
}

fn has_retrospective_hook(padded: &str) -> bool {
    const HOOKS: &[&str] = &[
        "looking back",
        "in hindsight",
        "in retrospect",
        "turned out",
        "ended up being",
        "was the best thing",
        "was a blessing",
        "now i realize",
        "now i realise",
        "it took me years to see",
        "only later did",
    ];
    HOOKS.iter().any(|h| padded.contains(h))
}

fn has_interventional_hook(padded: &str) -> bool {
    const HOOKS: &[&str] = &[
        "if i'd",
        "if i had",
        "if they'd",
        "if they had",
        "if we'd",
        "if we had",
        "had i ",
        "had they ",
        "had we ",
        "one more ",
        "if only",
        "i wish i'd",
        "i wish i had",
    ];
    HOOKS.iter().any(|h| padded.contains(h))
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
        assert!(v
            .iter()
            .any(|t| t.contains("compensatory") && t.contains("at_least")));
    }

    #[test]
    fn retrospective_looking_back() {
        let v = causal_bm25_tokens("Looking back, losing that job was the best thing.");
        assert!(v.iter().any(|t| t.contains("retrospective_framing")));
    }

    #[test]
    fn retrospective_in_hindsight() {
        let v = causal_bm25_tokens("In hindsight the crash taught us everything.");
        assert!(v.iter().any(|t| t.contains("retrospective_framing")));
    }

    #[test]
    fn retrospective_turned_out() {
        let v = causal_bm25_tokens("Turned out getting rejected was a blessing.");
        assert!(v.iter().any(|t| t.contains("retrospective_framing")));
    }

    #[test]
    fn interventional_if_id() {
        let v = causal_bm25_tokens("If I'd sold at the top I'd be fine now.");
        assert!(v
            .iter()
            .any(|t| t.contains("interventional_counterfactual")));
    }

    #[test]
    fn interventional_if_only() {
        let v = causal_bm25_tokens("If only they had listened to the audit.");
        assert!(v
            .iter()
            .any(|t| t.contains("interventional_counterfactual")));
    }

    #[test]
    fn both_connector_and_subtype_can_fire() {
        let v = causal_bm25_tokens(
            "I hated the renovation but looking back it doubled our home value.",
        );
        assert!(v
            .iter()
            .any(|t| t.contains("compensatory") && t.contains("but")));
        assert!(v.iter().any(|t| t.contains("retrospective_framing")));
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn concessive_but_somehow() {
        let v = causal_bm25_tokens("The market tanked but somehow we still came out ahead.");
        assert!(v.iter().any(|t| t.contains("concessive")));
    }

    #[test]
    fn since_explanatory() {
        let v = causal_bm25_tokens("I'm bullish since the ETF finally got approved.");
        assert!(v
            .iter()
            .any(|t| t.contains("explanatory") && t.contains("since")));
    }
}
