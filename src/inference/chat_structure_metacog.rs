//! Chat-structure metacognition for pet/companion compose paths.
//!
//! Unlike document-relevance [`crate::metacognition::MetaCognition`], this gate
//! only checks response-shaping contracts and compose-bleed policy. Outcomes:
//! Accept / Retry (resample compose) — Degrade is modeled as compose returning
//! `None` after retries are exhausted.

use crate::inference::harness::InferenceHarness;
use crate::inference::inference_toml::inference_toml_loaded;

/// Result of a single chat-structure evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatStructureOutcome {
    Accept,
    Retry { reason: &'static str },
}

/// Evaluate a composed candidate for pet-chat contract fit.
///
/// - shaping violation → Retry
/// - compose bleed (locale chat_policy) → Retry
/// - otherwise Accept
pub fn evaluate(
    intent_text: &str,
    response: &str,
    language_channel: Option<&str>,
    harness: &InferenceHarness,
) -> ChatStructureOutcome {
    if let Some(reason) = shaping_violation(response) {
        return ChatStructureOutcome::Retry { reason };
    }
    if harness
        .detect_compose_bleed(language_channel, intent_text, response)
        .is_some()
    {
        return ChatStructureOutcome::Retry {
            reason: "compose_bleed",
        };
    }
    ChatStructureOutcome::Accept
}

fn shaping_violation(text: &str) -> Option<&'static str> {
    let loaded = inference_toml_loaded();
    let shaping = loaded.response_shaping();
    let fragment = loaded.fragment_compose();
    let lower = text.to_ascii_lowercase();

    for phrase in &shaping.forbidden_phrases {
        if !phrase.is_empty() && lower.contains(&phrase.to_ascii_lowercase()) {
            return Some("forbidden_phrase");
        }
    }
    if shaping.forbid_asterisks && text.contains('*') {
        return Some("asterisk_action");
    }
    if shaping.require_sensory_or_vocalization && !has_required_signal(&lower, shaping, fragment) {
        return Some("missing_required_signal");
    }
    None
}

fn has_required_signal(
    lower: &str,
    shaping: &crate::inference::inference_toml::ResponseShapingConfig,
    fragment: &crate::inference::inference_toml::FragmentComposeConfig,
) -> bool {
    for v in &fragment.vocalizations {
        if !v.is_empty() && contains_word(lower, &v.to_ascii_lowercase()) {
            return true;
        }
    }
    const FALLBACK_VOCALS: &[&str] = &[
        "mrrp", "purr", "prrp", "chirp", "meow", "trill", "rumble", "thrum",
    ];
    if FALLBACK_VOCALS.iter().any(|tok| lower.contains(tok)) {
        return true;
    }
    for pat in &shaping.required_signal_patterns {
        let lit = pat.to_ascii_lowercase();
        let stripped = lit.replace(['^', '$', '\\'], "");
        if !stripped.is_empty() && lower.contains(stripped.trim()) {
            return true;
        }
    }
    false
}

fn contains_word(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    haystack
        .split(|c: char| !c.is_alphanumeric())
        .any(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::plugins::default_inference_harness;

    #[test]
    fn accept_clean_line_without_bleed() {
        let h = default_inference_harness();
        // Without a loaded pet TOML, bleed rules are empty; shaping may be loose.
        let out = evaluate(
            "hey luna",
            "There you are. I give you the slow blink. Purr.",
            Some("en"),
            &h,
        );
        assert_eq!(out, ChatStructureOutcome::Accept);
    }

    #[test]
    fn retry_on_asterisk_action() {
        let toml = r#"
[response_shaping]
forbid_asterisks = true
require_sensory_or_vocalization = false
"#;
        crate::inference::inference_toml::reload_inference_toml_from_str(toml).expect("reload");
        let h = default_inference_harness();
        let out = evaluate("hi", "*pounces*", Some("en"), &h);
        assert!(matches!(
            out,
            ChatStructureOutcome::Retry {
                reason: "asterisk_action"
            }
        ));
    }
}
