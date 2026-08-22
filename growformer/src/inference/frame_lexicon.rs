//! Compile-time frame lexicon — maps prompts to scenario topics before domain inference TOML.
//!
//! Phase 1 (declarative): absorbs repeated `headline_lexical_topic` scenario rules into one
//! auditable file. Domain TOML may still extend; frames run first in `sentiment_lexical_topic_key`.

use serde::Deserialize;

use super::inference_toml::{inference_rules_runtime, InferenceRulesRuntime};

#[derive(Debug, Clone, Deserialize)]
struct FrameLexiconFile {
    #[serde(default)]
    frames: Vec<FrameRow>,
    #[serde(default)]
    reject_frames: Vec<RejectFrameRow>,
}

#[derive(Debug, Clone, Deserialize)]
struct FrameRow {
    #[serde(default)]
    frame_id: String,
    scenario_topic: String,
    #[serde(default, alias = "keywords_cnf")]
    intent: Vec<Vec<String>>,
    #[serde(default)]
    min_trim_len: Option<usize>,
    #[serde(default)]
    exclude_first_person: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct RejectFrameRow {
    #[serde(default)]
    frame_id: String,
    #[serde(default, alias = "keywords_cnf")]
    intent: Vec<Vec<String>>,
}

fn cnf_groups_match(haystack: &str, groups: &[Vec<String>]) -> bool {
    !groups.is_empty()
        && groups
            .iter()
            .all(|or_alts| or_alts.iter().any(|p| haystack.contains(p)))
}

fn load_lexicon() -> FrameLexiconFile {
    const RAW: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/data/linguistics/frame_lexicon.toml"
    ));
    toml::from_str(RAW).unwrap_or_else(|e| {
        panic!("frame_lexicon.toml parse error: {e}");
    })
}

fn frame_rule_matches(row: &FrameRow, lower: &str) -> bool {
    if let Some(min) = row.min_trim_len {
        if lower.trim().len() < min {
            return false;
        }
    }
    if row.exclude_first_person
        && inference_rules_runtime().looks_like_first_person_finance_intent(lower)
    {
        return false;
    }
    cnf_groups_match(lower, &row.intent)
}

/// Best scenario topic from shared frame lexicon (first match wins).
pub fn resolve_scenario_topic(intent_text: &str) -> Option<String> {
    let lower = InferenceRulesRuntime::normalize_rules_text(intent_text);
    let lex = load_lexicon();
    for row in &lex.frames {
        if !row.scenario_topic.is_empty() && frame_rule_matches(row, lower.as_str()) {
            return Some(row.scenario_topic.clone());
        }
    }
    None
}

/// True when prompt matches a reject frame (e.g. counterfactual rally).
pub fn matches_reject_frame(intent_text: &str) -> bool {
    let lower = InferenceRulesRuntime::normalize_rules_text(intent_text);
    let lex = load_lexicon();
    lex.reject_frames
        .iter()
        .any(|row| cnf_groups_match(lower.as_str(), &row.intent))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn etf_delay_frame_routes() {
        assert_eq!(
            resolve_scenario_topic(
                "Regulators shelved the spot BTC ETF again and Bitcoin slid on the open"
            ),
            Some("etf_delay_bearish".into())
        );
    }

    #[test]
    fn mortgage_frame_routes() {
        assert_eq!(
            resolve_scenario_topic(
                "Wells Fargo lifted my mortgage APR overnight with no email alert"
            ),
            Some("mortgage_rate_complaint".into())
        );
    }

    #[test]
    fn counterfactual_reject() {
        assert!(matches_reject_frame(
            "I would have loved that rally if it had happened"
        ));
    }
}
