//! Rules-based causal relation tagger — runs deterministically on surface
//! connectors without lattice retrieval. Produces a structured
//! `(CausalRelation, connector, clause_boundary)` triple that the preempt
//! path in `lattice_shortcuts` can combine with independent sentiment scoring
//! per clause.
//!
//! Design: longest-phrase-first matching (same strategy as `causal_hints`),
//! but returns a structured result instead of BM25 index tokens. The two
//! modules are intentionally separate so BM25 retrieval tokens continue to
//! feed the lattice path when the preempt falls through.

use std::fmt;

/// The seven causal relation types plus `None` for no detected structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalRelation {
    Direct,
    Compensatory,
    Contrastive,
    Concessive,
    Explanatory,
    Inferential,
    Retrospective,
    Counterfactual,
    None,
}

impl CausalRelation {
    pub fn label(&self) -> &'static str {
        match self {
            CausalRelation::Direct => "Direct causal link",
            CausalRelation::Compensatory => "Compensatory framing",
            CausalRelation::Contrastive => "Contrastive tension",
            CausalRelation::Concessive => "Concessive persistence",
            CausalRelation::Explanatory => "Explanatory cause",
            CausalRelation::Inferential => "Inferential conclusion",
            CausalRelation::Retrospective => "Retrospective reassessment",
            CausalRelation::Counterfactual => "Counterfactual framing",
            CausalRelation::None => "No causal structure",
        }
    }

    /// Body template describing the structural role of the connector.
    pub fn body(&self, connector: &str) -> String {
        match self {
            CausalRelation::Direct => format!(
                "'{}' chains a cause event to an immediate effect with no mitigation",
                connector
            ),
            CausalRelation::Compensatory => format!(
                "'{}' frames a residual positive as partial offset to a negative backdrop",
                connector
            ),
            CausalRelation::Contrastive => format!(
                "'{}' introduces a second clause whose valence contradicts the first",
                connector
            ),
            CausalRelation::Concessive => format!(
                "'{}' concedes a weakness while asserting an unexpected positive outcome",
                connector
            ),
            CausalRelation::Explanatory => format!(
                "'{}' provides the reason behind a stated outcome or emotional state",
                connector
            ),
            CausalRelation::Inferential => format!(
                "'{}' draws an implication or strategic conclusion from the preceding fact",
                connector
            ),
            CausalRelation::Retrospective => format!(
                "'{}' signals a revised causal model where hindsight changes the evaluation",
                connector
            ),
            CausalRelation::Counterfactual => format!(
                "'{}' constructs a hypothetical world that didn't happen to highlight what did",
                connector
            ),
            CausalRelation::None => String::new(),
        }
    }

    /// Weight applied to the *second* clause when combining per-clause sentiment.
    /// Counterfactual clauses describe worlds that didn't happen, so their
    /// sentiment contribution is dampened.
    pub fn second_clause_weight(&self) -> f32 {
        match self {
            CausalRelation::Counterfactual => 0.3,
            CausalRelation::Retrospective => 0.7,
            _ => 1.0,
        }
    }
}

impl fmt::Display for CausalRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

pub struct CausalRelationResult {
    pub relation: CausalRelation,
    /// The surface connector that triggered the match (e.g. "yet somehow").
    pub connector: &'static str,
    /// Byte offset in the lowercased+padded text where the connector starts.
    /// Used to split the text into two clauses for independent sentiment scoring.
    pub clause_boundary: usize,
    /// Length of the connector string in the padded text.
    pub connector_len: usize,
    pub confidence: f32,
}

struct ConnectorRule {
    phrase: &'static str,
    relation: CausalRelation,
    confidence: f32,
}

/// Longest-phrase-first connector rules. Order matters: first match wins
/// within the same length tier, but we always prefer longer phrases.
const CONNECTOR_RULES: &[ConnectorRule] = &[
    // --- Retrospective (check first — often co-occurs with other connectors) ---
    ConnectorRule { phrase: "in retrospect", relation: CausalRelation::Retrospective, confidence: 1.0 },
    ConnectorRule { phrase: "looking back", relation: CausalRelation::Retrospective, confidence: 1.0 },
    ConnectorRule { phrase: "it turned out", relation: CausalRelation::Retrospective, confidence: 1.0 },
    ConnectorRule { phrase: "turned out", relation: CausalRelation::Retrospective, confidence: 0.95 },
    ConnectorRule { phrase: "in hindsight", relation: CausalRelation::Retrospective, confidence: 1.0 },
    ConnectorRule { phrase: "ended up being", relation: CausalRelation::Retrospective, confidence: 0.9 },
    ConnectorRule { phrase: "now i realize", relation: CausalRelation::Retrospective, confidence: 0.9 },
    ConnectorRule { phrase: "now i realise", relation: CausalRelation::Retrospective, confidence: 0.9 },
    // --- Counterfactual ---
    ConnectorRule { phrase: " would have ", relation: CausalRelation::Counterfactual, confidence: 0.9 },
    ConnectorRule { phrase: " would've ", relation: CausalRelation::Counterfactual, confidence: 0.9 },
    ConnectorRule { phrase: " wouldn't have ", relation: CausalRelation::Counterfactual, confidence: 0.95 },
    ConnectorRule { phrase: " wouldnt have ", relation: CausalRelation::Counterfactual, confidence: 0.95 },
    ConnectorRule { phrase: "if the ", relation: CausalRelation::Counterfactual, confidence: 0.7 },
    ConnectorRule { phrase: "if they ", relation: CausalRelation::Counterfactual, confidence: 0.7 },
    ConnectorRule { phrase: "if we ", relation: CausalRelation::Counterfactual, confidence: 0.7 },
    ConnectorRule { phrase: "without the ", relation: CausalRelation::Counterfactual, confidence: 0.85 },
    ConnectorRule { phrase: "had they ", relation: CausalRelation::Counterfactual, confidence: 0.9 },
    ConnectorRule { phrase: "had we ", relation: CausalRelation::Counterfactual, confidence: 0.9 },
    // --- Concessive (check before compensatory — "yet somehow" is concessive, not compensatory) ---
    ConnectorRule { phrase: " yet somehow ", relation: CausalRelation::Concessive, confidence: 1.0 },
    ConnectorRule { phrase: " but somehow ", relation: CausalRelation::Concessive, confidence: 1.0 },
    ConnectorRule { phrase: " but still ", relation: CausalRelation::Concessive, confidence: 0.95 },
    ConnectorRule { phrase: "; still, ", relation: CausalRelation::Concessive, confidence: 0.95 },
    ConnectorRule { phrase: "; still ", relation: CausalRelation::Concessive, confidence: 0.9 },
    ConnectorRule { phrase: " somehow ", relation: CausalRelation::Concessive, confidence: 0.7 },
    ConnectorRule { phrase: " nevertheless ", relation: CausalRelation::Concessive, confidence: 0.95 },
    // --- Contrastive ---
    ConnectorRule { phrase: " even though ", relation: CausalRelation::Contrastive, confidence: 1.0 },
    ConnectorRule { phrase: " despite ", relation: CausalRelation::Contrastive, confidence: 1.0 },
    ConnectorRule { phrase: " although ", relation: CausalRelation::Contrastive, confidence: 0.95 },
    // --- Compensatory ---
    ConnectorRule { phrase: " at least ", relation: CausalRelation::Compensatory, confidence: 0.95 },
    ConnectorRule { phrase: " however ", relation: CausalRelation::Compensatory, confidence: 0.9 },
    // --- Inferential ---
    ConnectorRule { phrase: " which suggests ", relation: CausalRelation::Inferential, confidence: 1.0 },
    ConnectorRule { phrase: " which means ", relation: CausalRelation::Inferential, confidence: 1.0 },
    ConnectorRule { phrase: " which implies ", relation: CausalRelation::Inferential, confidence: 1.0 },
    ConnectorRule { phrase: ", implying ", relation: CausalRelation::Inferential, confidence: 1.0 },
    ConnectorRule { phrase: ", suggesting ", relation: CausalRelation::Inferential, confidence: 0.95 },
    ConnectorRule { phrase: ", meaning ", relation: CausalRelation::Inferential, confidence: 0.95 },
    ConnectorRule { phrase: " meaning ", relation: CausalRelation::Inferential, confidence: 0.85 },
    ConnectorRule { phrase: " implying ", relation: CausalRelation::Inferential, confidence: 0.9 },
    ConnectorRule { phrase: " suggesting ", relation: CausalRelation::Inferential, confidence: 0.85 },
    // --- Explanatory ---
    ConnectorRule { phrase: " because ", relation: CausalRelation::Explanatory, confidence: 1.0 },
    ConnectorRule { phrase: " since ", relation: CausalRelation::Explanatory, confidence: 0.8 },
    // --- Direct (participial and explicit connectors) ---
    ConnectorRule { phrase: " triggered ", relation: CausalRelation::Direct, confidence: 1.0 },
    ConnectorRule { phrase: " caused ", relation: CausalRelation::Direct, confidence: 1.0 },
    ConnectorRule { phrase: " causing ", relation: CausalRelation::Direct, confidence: 1.0 },
    ConnectorRule { phrase: ", pushing ", relation: CausalRelation::Direct, confidence: 0.95 },
    ConnectorRule { phrase: " pushing ", relation: CausalRelation::Direct, confidence: 0.85 },
    ConnectorRule { phrase: " resulted in ", relation: CausalRelation::Direct, confidence: 1.0 },
    ConnectorRule { phrase: " led to ", relation: CausalRelation::Direct, confidence: 1.0 },
    ConnectorRule { phrase: " lead to ", relation: CausalRelation::Direct, confidence: 0.95 },
    ConnectorRule { phrase: " thereby ", relation: CausalRelation::Direct, confidence: 0.95 },
    ConnectorRule { phrase: " therefore ", relation: CausalRelation::Direct, confidence: 1.0 },
    ConnectorRule { phrase: " so that ", relation: CausalRelation::Direct, confidence: 0.9 },
    ConnectorRule { phrase: ", so ", relation: CausalRelation::Direct, confidence: 0.9 },
    ConnectorRule { phrase: ". so ", relation: CausalRelation::Direct, confidence: 0.9 },
    ConnectorRule { phrase: "; so ", relation: CausalRelation::Direct, confidence: 0.9 },
    ConnectorRule { phrase: " thus ", relation: CausalRelation::Direct, confidence: 0.9 },
    // --- Contrastive / Concessive fallback for bare "yet" / "but" ---
    // "yet" without "somehow"/"still" is contrastive, not concessive.
    ConnectorRule { phrase: " yet ", relation: CausalRelation::Contrastive, confidence: 0.8 },
    ConnectorRule { phrase: ", yet ", relation: CausalRelation::Contrastive, confidence: 0.85 },
    ConnectorRule { phrase: " but ", relation: CausalRelation::Compensatory, confidence: 0.7 },
    ConnectorRule { phrase: ", but ", relation: CausalRelation::Compensatory, confidence: 0.75 },
];

/// Detect the causal relation type from surface connectors in `intent_text`.
/// Returns `None` when no connector fires (relation == `CausalRelation::None`).
///
/// Matching is longest-phrase-first within the sorted rule table (the table is
/// pre-ordered by specificity; we iterate and pick the first hit whose phrase
/// is the longest among all matches).
pub fn detect(intent_text: &str) -> Option<CausalRelationResult> {
    let padded = format!(
        " {} ",
        intent_text
            .to_ascii_lowercase()
            .replace(['\n', '\r', '\t'], " ")
    );

    let mut best: Option<(usize, usize, &ConnectorRule)> = None; // (phrase_len, byte_offset, rule)

    for rule in CONNECTOR_RULES {
        if let Some(pos) = padded.find(rule.phrase) {
            let plen = rule.phrase.len();
            if best.as_ref().map_or(true, |(bl, _, _)| plen > *bl) {
                best = Some((plen, pos, rule));
            }
        }
    }

    let (_, byte_offset, rule) = best?;

    // Translate the byte offset in the padded string back to the original text.
    // The padded string has a 1-byte " " prefix, so subtract 1 (clamped to 0).
    let clause_boundary = byte_offset.saturating_sub(1);

    Some(CausalRelationResult {
        relation: rule.relation,
        connector: rule.phrase.trim(),
        clause_boundary,
        connector_len: rule.phrase.len(),
        confidence: rule.confidence,
    })
}

/// Split the original `intent_text` into two clauses at the connector boundary.
/// Returns `(clause_before, clause_after)`. Both are trimmed.
pub fn split_clauses<'a>(
    intent_text: &'a str,
    result: &CausalRelationResult,
) -> (&'a str, &'a str) {
    let lower_padded = format!(
        " {} ",
        intent_text
            .to_ascii_lowercase()
            .replace(['\n', '\r', '\t'], " ")
    );
    // Find the connector in the original-case text by locating it in the
    // lowered padded version (same offsets since to_ascii_lowercase preserves
    // byte length for ASCII) and mapping back.
    if let Some(pos) = lower_padded.find(result.connector) {
        // pos is in the padded string; subtract 1 for the leading space
        let orig_start = pos.saturating_sub(1);
        let orig_end = (orig_start + result.connector.len()).min(intent_text.len());
        let before = intent_text[..orig_start.min(intent_text.len())].trim();
        let after = intent_text[orig_end.min(intent_text.len())..].trim();
        (before, after)
    } else {
        (intent_text.trim(), "")
    }
}

// ---------------------------------------------------------------------------
// Per-clause sentiment scoring + structured output
// ---------------------------------------------------------------------------

/// Combined result of causal relation detection + per-clause sentiment.
pub struct CausalSentimentResult {
    pub relation: CausalRelation,
    pub connector: &'static str,
    pub sentiment_label: &'static str,
    pub relation_body: String,
    pub sentiment_body: String,
    pub confidence: f32,
}

fn polarity_to_coarse(topic: &str) -> &'static str {
    if topic.starts_with("positive") { "positive" }
    else if topic.starts_with("negative") { "negative" }
    else if topic == "mixed" { "mixed" }
    else { "neutral" }
}

fn sentiment_display_label(coarse: &str) -> &'static str {
    match coarse {
        "positive" => "POSITIVE",
        "negative" => "NEGATIVE",
        "mixed" => "MIXED",
        _ => "NEUTRAL",
    }
}

/// Score sentiment per clause using the existing `InferenceRulesRuntime`
/// lexical machinery, then combine based on the causal relation type.
///
/// Returns `Some(CausalSentimentResult)` when the relation is non-None.
pub fn score_with_relation(
    intent_text: &str,
) -> Option<CausalSentimentResult> {
    let cr = detect(intent_text)?;

    let rules = super::inference_toml::inference_rules_runtime();
    let lower_full = intent_text.to_ascii_lowercase();

    let (clause1_raw, clause2_raw) = split_clauses(intent_text, &cr);
    let c1 = clause1_raw.to_ascii_lowercase();
    let c2 = clause2_raw.to_ascii_lowercase();

    let pol1 = rules.lexical_polarity_signal(&c1);
    let pol2 = if c2.is_empty() {
        None
    } else {
        rules.lexical_polarity_signal(&c2)
    };

    let coarse1 = pol1.as_deref().map(polarity_to_coarse).unwrap_or("neutral");
    let coarse2 = pol2.as_deref().map(polarity_to_coarse).unwrap_or("neutral");

    // Also check contrastive + bipolar on the full text as a fallback
    let full_contrast = rules.has_contrastive_marker(&lower_full);
    let full_bipolar = rules.has_bipolar_lexicon(&lower_full);

    let combined = if coarse1 != coarse2
        && coarse1 != "neutral"
        && coarse2 != "neutral"
    {
        "mixed"
    } else if full_contrast && full_bipolar {
        "mixed"
    } else if coarse1 != "neutral" {
        coarse1
    } else if coarse2 != "neutral" {
        coarse2
    } else if rules.has_clear_evaluative_stance(&lower_full) {
        // Full-text evaluative check when per-clause scoring is empty
        if let Some(ref full_pol) = rules.lexical_polarity_signal(&lower_full) {
            polarity_to_coarse(full_pol)
        } else {
            "neutral"
        }
    } else {
        "neutral"
    };

    let sentiment_label = sentiment_display_label(combined);
    let relation_body = cr.relation.body(cr.connector);

    let sentiment_body = match combined {
        "mixed" => {
            let anchor1 = pol1.as_deref().unwrap_or("neutral");
            let anchor2 = pol2.as_deref().unwrap_or("neutral");
            if anchor1 != "neutral" && anchor2 != "neutral" {
                format!(
                    "Clause 1 reads as {} while clause 2 reads as {} — dual valence",
                    anchor1, anchor2,
                )
            } else {
                "Both positive and negative cues appear across clauses".to_string()
            }
        }
        "positive" => {
            let key = pol1
                .as_deref()
                .filter(|p| polarity_to_coarse(p) == "positive")
                .or_else(|| pol2.as_deref().filter(|p| polarity_to_coarse(p) == "positive"))
                .unwrap_or("positive_mild");
            match rules.anchor_phrase(&lower_full, key) {
                Some(a) => format!("{}. The overall tone reads as positive", a),
                None => "The overall tone reads as positive".to_string(),
            }
        }
        "negative" => {
            let key = pol1
                .as_deref()
                .filter(|p| polarity_to_coarse(p) == "negative")
                .or_else(|| pol2.as_deref().filter(|p| polarity_to_coarse(p) == "negative"))
                .unwrap_or("negative_mild");
            match rules.anchor_phrase(&lower_full, key) {
                Some(a) => format!("{}. The overall tone reads as negative", a),
                None => "The overall tone reads as negative".to_string(),
            }
        }
        _ => "No strong evaluative stance detected".to_string(),
    };

    Some(CausalSentimentResult {
        relation: cr.relation,
        connector: cr.connector,
        sentiment_label,
        relation_body,
        sentiment_body,
        confidence: cr.confidence,
    })
}

/// Format the final output line combining causal relation + sentiment.
pub fn format_causal_sentiment_line(
    r: &CausalSentimentResult,
    intent_text: &str,
) -> String {
    let excerpt = {
        let trimmed = intent_text.trim();
        if trimmed.chars().count() > 200 {
            let head: String = trimmed.chars().take(200).collect();
            format!("{}…", head)
        } else {
            trimmed.to_string()
        }
    };

    format!(
        "{} — {} ({sentiment}). {sent_body}. Grounded in the user's own words: \"{excerpt}\"",
        r.relation.label(),
        r.relation_body,
        sentiment = r.sentiment_label,
        sent_body = r.sentiment_body,
        excerpt = excerpt,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_so() {
        let r = detect("The server crashed, so every API returned 503.").unwrap();
        assert_eq!(r.relation, CausalRelation::Direct);
        assert_eq!(r.connector, ", so");
    }

    #[test]
    fn direct_triggered() {
        let r = detect("Inventory write-downs triggered a covenant breach.").unwrap();
        assert_eq!(r.relation, CausalRelation::Direct);
        assert_eq!(r.connector, "triggered");
    }

    #[test]
    fn concessive_yet_somehow() {
        let r = detect("Pipeline is bone-dry, yet somehow the team closed enough.").unwrap();
        assert_eq!(r.relation, CausalRelation::Concessive);
        assert!(r.connector.contains("yet somehow"));
    }

    #[test]
    fn inferential_implying() {
        let r = detect("The CFO sold half their holdings, implying limited upside.").unwrap();
        assert_eq!(r.relation, CausalRelation::Inferential);
        assert!(r.connector.contains("implying"));
    }

    #[test]
    fn contrastive_yet() {
        let r = detect("The company announced mass layoffs yet the stock closed at an all-time high.").unwrap();
        assert_eq!(r.relation, CausalRelation::Contrastive);
        assert!(r.connector.contains("yet"));
    }

    #[test]
    fn counterfactual_without() {
        let r = detect("Without the emergency credit line, payroll would have bounced.").unwrap();
        assert_eq!(r.relation, CausalRelation::Counterfactual);
    }

    #[test]
    fn retrospective_looking_back() {
        let r = detect("Looking back, ignoring the early churn signal was expensive.").unwrap();
        assert_eq!(r.relation, CausalRelation::Retrospective);
    }

    #[test]
    fn compensatory_at_least() {
        let r = detect("The UI is clunky, but at least the search is blazing fast.").unwrap();
        assert_eq!(r.relation, CausalRelation::Compensatory);
        assert!(r.connector.contains("at least"));
    }

    #[test]
    fn explanatory_because() {
        let r = detect("Churn doubled because the pricing overhaul confused users.").unwrap();
        assert_eq!(r.relation, CausalRelation::Explanatory);
        assert!(r.connector.contains("because"));
    }

    #[test]
    fn no_connector() {
        assert!(detect("Revenue grew 40% year-over-year.").is_none());
    }

    #[test]
    fn clause_split_works() {
        let text = "The server crashed, so every API returned 503.";
        let r = detect(text).unwrap();
        let (before, after) = split_clauses(text, &r);
        assert!(before.contains("crashed"));
        assert!(after.contains("API"));
    }

    #[test]
    fn concessive_still_semicolon() {
        let r = detect("Morale has cratered; still, nobody has actually quit yet.").unwrap();
        assert_eq!(r.relation, CausalRelation::Concessive);
    }

    #[test]
    fn direct_pushing() {
        let r = detect("Copper futures spiked, pushing manufacturing input costs to a five-year high.").unwrap();
        assert_eq!(r.relation, CausalRelation::Direct);
        assert!(r.connector.contains("pushing"));
    }

    #[test]
    fn contrastive_even_though() {
        let r = detect("Customer satisfaction improved even though response times got worse.").unwrap();
        assert_eq!(r.relation, CausalRelation::Contrastive);
    }

    #[test]
    fn concessive_but_still() {
        let r = detect("Budget was slashed across the board but still engineering shipped on time.").unwrap();
        assert_eq!(r.relation, CausalRelation::Concessive);
    }
}
