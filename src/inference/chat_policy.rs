//! Locale-keyed chat policy loaded from inference TOML `[chat_policy]`.
//!
//! Greeting / identity match patterns and compose-bleed detectors live here so
//! [`crate::service::LanguageService`] stays orchestration-only. Add another
//! language later as `[chat_policy.locales.es]` (etc.) without Rust changes.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Top-level `[chat_policy]` section.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatPolicySection {
    #[serde(default = "default_chat_policy_locale")]
    pub default_locale: String,
    #[serde(default)]
    pub locales: HashMap<String, ChatPolicyLocale>,
}

fn default_chat_policy_locale() -> String {
    "en".into()
}

/// Per-locale chat policy tables.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatPolicyLocale {
    /// Substring patterns for identity shortcuts (sentiment lattice path).
    #[serde(default)]
    pub identity_patterns: Vec<String>,
    /// Exact / prefix greeting tokens (sentiment lattice path).
    #[serde(default)]
    pub greeting_exact: Vec<String>,
    #[serde(default = "default_greeting_max_len")]
    pub greeting_max_len: usize,
    /// Capitalized headline prefixes that disqualify long “greetings”.
    #[serde(default)]
    pub greeting_headline_prefixes: Vec<String>,
    #[serde(default)]
    pub bleed_rules: Vec<ChatPolicyBleedRule>,
    #[serde(default)]
    pub bleed_fallbacks: Vec<ChatPolicyBleedFallback>,
}

fn default_greeting_max_len() -> usize {
    40
}

/// Detect compose/lattice bleed: response looks like a character arc that does
/// not belong on this prompt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatPolicyBleedRule {
    pub id: String,
    /// Prompt must contain at least one of these (OR). Empty = no prompt gate.
    #[serde(default)]
    pub prompt_any: Vec<String>,
    /// If prompt contains any of these, this is **not** a bleed.
    #[serde(default)]
    pub prompt_exclude_any: Vec<String>,
    /// Response matches if any substring hits.
    #[serde(default)]
    pub response_any: Vec<String>,
    /// Alternative AND-of-OR response patterns; any alternative may fire.
    /// Each alternative is a list of OR-groups that must all match (CNF).
    #[serde(default)]
    pub response_match_any: Vec<Vec<Vec<String>>>,
    /// Preferred fallback template id when no more-specific bleed_fallback row matches.
    #[serde(default)]
    pub fallback_template_id: String,
}

/// Fallback line when a bleed rule fires (first matching row wins).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatPolicyBleedFallback {
    /// Optional: only apply when this bleed rule id fired. Empty = any bleed.
    #[serde(default)]
    pub bleed_id: String,
    /// Prompt-side CNF (`[[a], [b,c]]` = a OR (b AND c) style via OR-groups).
    /// Empty `when_intent` matches any prompt.
    #[serde(default)]
    pub when_intent: Vec<Vec<String>>,
    #[serde(default)]
    pub intent_exclude: Vec<Vec<String>>,
    pub response: String,
    #[serde(default = "default_bleed_fallback_template_id")]
    pub template_id: String,
}

fn default_bleed_fallback_template_id() -> String {
    "chat_policy_bleed_fallback".into()
}

/// Result of a successful bleed detection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BleedHit {
    pub rule_id: String,
    pub fallback_template_id: String,
}

impl ChatPolicySection {
    /// Normalize `language_channel` / BCP-47-ish tags to a locale key.
    pub fn normalize_language_channel(channel: Option<&str>) -> String {
        let raw = channel.unwrap_or("en").trim();
        if raw.is_empty() {
            return "en".into();
        }
        let lower = raw.to_ascii_lowercase();
        match lower.as_str() {
            "english" | "en-us" | "en-gb" | "en_us" | "en_gb" => "en".into(),
            "spanish" | "español" | "espanol" | "es-es" | "es-mx" | "es_es" | "es_mx" => {
                "es".into()
            }
            "french" | "français" | "francais" | "fr-fr" | "fr_fr" => "fr".into(),
            "german" | "deutsch" | "de-de" | "de_de" => "de".into(),
            "portuguese" | "pt-br" | "pt_br" | "pt-pt" | "pt_pt" => {
                if lower.contains("br") {
                    "pt-BR".into()
                } else {
                    "pt".into()
                }
            }
            other => {
                // Pass through BCP-47 primary subtag (before `-` / `_`).
                let primary = other.split(['-', '_']).next().unwrap_or(other);
                primary.to_string()
            }
        }
    }

    /// Resolve the locale table: requested → default_locale → English builtin shortcuts.
    pub fn resolve_locale(&self, language_channel: Option<&str>) -> ChatPolicyLocale {
        let key = Self::normalize_language_channel(language_channel);
        if let Some(loc) = self.locales.get(&key) {
            return merge_with_english_shortcuts(loc.clone());
        }
        let def = if self.default_locale.is_empty() {
            "en"
        } else {
            self.default_locale.as_str()
        };
        if def != key {
            if let Some(loc) = self.locales.get(def) {
                return merge_with_english_shortcuts(loc.clone());
            }
        }
        english_builtin_locale()
    }

    pub fn match_identity_query(&self, language_channel: Option<&str>, text: &str) -> bool {
        let loc = self.resolve_locale(language_channel);
        loc.match_identity_query(text)
    }

    pub fn match_greeting(&self, language_channel: Option<&str>, text: &str) -> bool {
        let loc = self.resolve_locale(language_channel);
        loc.match_greeting(text)
    }

    pub fn detect_compose_bleed(
        &self,
        language_channel: Option<&str>,
        prompt: &str,
        response: &str,
    ) -> Option<BleedHit> {
        let loc = self.resolve_locale(language_channel);
        loc.detect_compose_bleed(prompt, response)
    }

    pub fn bleed_fallback(
        &self,
        language_channel: Option<&str>,
        prompt: &str,
        bleed: &BleedHit,
    ) -> Option<(String, String)> {
        let loc = self.resolve_locale(language_channel);
        loc.bleed_fallback(prompt, bleed)
    }
}

/// When TOML omits greeting/identity lists, keep sentiment-lattice shortcuts working.
fn merge_with_english_shortcuts(mut loc: ChatPolicyLocale) -> ChatPolicyLocale {
    let builtin = english_builtin_locale();
    if loc.identity_patterns.is_empty() {
        loc.identity_patterns = builtin.identity_patterns;
    }
    if loc.greeting_exact.is_empty() {
        loc.greeting_exact = builtin.greeting_exact;
    }
    if loc.greeting_headline_prefixes.is_empty() {
        loc.greeting_headline_prefixes = builtin.greeting_headline_prefixes;
    }
    if loc.greeting_max_len == 0 {
        loc.greeting_max_len = builtin.greeting_max_len;
    }
    loc
}

/// English patterns previously hardcoded in `LanguageService` (parity baseline).
pub fn english_builtin_locale() -> ChatPolicyLocale {
    ChatPolicyLocale {
        identity_patterns: vec![
            "who are you".into(),
            "what are you".into(),
            "who r u".into(),
            "what is your name".into(),
            "introduce yourself".into(),
            "tell me about yourself".into(),
            "what's your name".into(),
            "your name".into(),
            "who made you".into(),
            "who created you".into(),
            "who built you".into(),
        ],
        greeting_exact: vec![
            "hi".into(),
            "hello".into(),
            "hey".into(),
            "howdy".into(),
            "greetings".into(),
            "yo".into(),
            "sup".into(),
            "good morning".into(),
            "good afternoon".into(),
            "good evening".into(),
            "how are you".into(),
            "how are you doing".into(),
            "how's it going".into(),
            "what's up".into(),
            "whats up".into(),
            "how do you do".into(),
            "hey there".into(),
            "hi there".into(),
            "hello there".into(),
        ],
        greeting_max_len: 40,
        greeting_headline_prefixes: vec!["Hello ".into(), "Hi ".into(), "Hey ".into()],
        bleed_rules: Vec::new(),
        bleed_fallbacks: Vec::new(),
    }
}

fn cnf_groups_match(haystack: &str, groups: &[Vec<String>]) -> bool {
    !groups.is_empty()
        && groups
            .iter()
            .all(|or_alts| or_alts.iter().any(|p| haystack.contains(p.as_str())))
}

fn cnf_any_group_matches(haystack: &str, groups: &[Vec<String>]) -> bool {
    groups
        .iter()
        .any(|or_alts| or_alts.iter().any(|p| haystack.contains(p.as_str())))
}

impl ChatPolicyLocale {
    pub fn match_identity_query(&self, text: &str) -> bool {
        let lower = text.to_lowercase();
        self.identity_patterns
            .iter()
            .any(|p| !p.is_empty() && lower.contains(&p.to_lowercase()))
    }

    pub fn match_greeting(&self, text: &str) -> bool {
        let lower = text.to_lowercase();
        let trimmed = lower
            .trim()
            .trim_end_matches(|c: char| c.is_ascii_punctuation());
        let matched = self.greeting_exact.iter().any(|p| {
            let p = p.to_lowercase();
            trimmed == p || trimmed.starts_with(&format!("{} ", p))
        });
        if !matched {
            return false;
        }
        let max_len = if self.greeting_max_len == 0 {
            40
        } else {
            self.greeting_max_len
        };
        if trimmed.len() > max_len {
            let original_trimmed = text.trim();
            for prefix in &self.greeting_headline_prefixes {
                if original_trimmed.starts_with(prefix.as_str()) {
                    let rest = &original_trimmed[prefix.len()..];
                    if rest.starts_with(|c: char| c.is_uppercase()) {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn response_matches_bleed(rule: &ChatPolicyBleedRule, response_lower: &str) -> bool {
        if rule
            .response_any
            .iter()
            .any(|p| !p.is_empty() && response_lower.contains(&p.to_ascii_lowercase()))
        {
            return true;
        }
        for alt in &rule.response_match_any {
            let groups: Vec<Vec<String>> = alt
                .iter()
                .map(|or_g| or_g.iter().map(|s| s.to_ascii_lowercase()).collect())
                .collect();
            if cnf_groups_match(response_lower, &groups) {
                return true;
            }
        }
        false
    }

    pub fn detect_compose_bleed(&self, prompt: &str, response: &str) -> Option<BleedHit> {
        let p = prompt.to_ascii_lowercase();
        let r = response.to_ascii_lowercase();
        for rule in &self.bleed_rules {
            if !rule.prompt_exclude_any.is_empty()
                && rule
                    .prompt_exclude_any
                    .iter()
                    .any(|k| !k.is_empty() && p.contains(&k.to_ascii_lowercase()))
            {
                continue;
            }
            if !rule.prompt_any.is_empty()
                && !rule
                    .prompt_any
                    .iter()
                    .any(|k| !k.is_empty() && p.contains(&k.to_ascii_lowercase()))
            {
                continue;
            }
            if !Self::response_matches_bleed(rule, &r) {
                continue;
            }
            return Some(BleedHit {
                rule_id: rule.id.clone(),
                fallback_template_id: rule.fallback_template_id.clone(),
            });
        }
        None
    }

    pub fn bleed_fallback(&self, prompt: &str, bleed: &BleedHit) -> Option<(String, String)> {
        let p = prompt.to_ascii_lowercase();
        for row in &self.bleed_fallbacks {
            if !row.bleed_id.is_empty() && row.bleed_id != bleed.rule_id {
                continue;
            }
            if !row.when_intent.is_empty() {
                let groups: Vec<Vec<String>> = row
                    .when_intent
                    .iter()
                    .map(|g| g.iter().map(|s| s.to_ascii_lowercase()).collect())
                    .collect();
                if !cnf_any_group_matches(&p, &groups) {
                    continue;
                }
            }
            if !row.intent_exclude.is_empty() {
                let groups: Vec<Vec<String>> = row
                    .intent_exclude
                    .iter()
                    .map(|g| g.iter().map(|s| s.to_ascii_lowercase()).collect())
                    .collect();
                if cnf_any_group_matches(&p, &groups) {
                    continue;
                }
            }
            let template = if row.template_id.is_empty() {
                if bleed.fallback_template_id.is_empty() {
                    default_bleed_fallback_template_id()
                } else {
                    bleed.fallback_template_id.clone()
                }
            } else {
                row.template_id.clone()
            };
            return Some((row.response.clone(), template));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn school_bleed_locale() -> ChatPolicyLocale {
        ChatPolicyLocale {
            identity_patterns: english_builtin_locale().identity_patterns,
            greeting_exact: english_builtin_locale().greeting_exact,
            greeting_max_len: 40,
            greeting_headline_prefixes: english_builtin_locale().greeting_headline_prefixes,
            bleed_rules: vec![ChatPolicyBleedRule {
                id: "school_comfort".into(),
                prompt_any: vec![],
                prompt_exclude_any: vec![
                    "school".into(),
                    "grades".into(),
                    "homework".into(),
                    "teacher".into(),
                    "math class".into(),
                    "test today".into(),
                    "failed a test".into(),
                    "failed my test".into(),
                    "bad day at school".into(),
                    "awful day at school".into(),
                    "terrible day at school".into(),
                    "school was awful".into(),
                    "bad grades".into(),
                ],
                response_any: vec!["school is loud".into()],
                response_match_any: vec![
                    vec![
                        vec!["in your head".into(), "in thy head".into()],
                        vec![
                            "make room on the cushion".into(),
                            "make room by my hearth".into(),
                            "make room without asking".into(),
                        ],
                    ],
                    vec![
                        vec!["hear it in your voice".into(), "hear it in thy voice".into()],
                        vec!["come sit".into(), "come sit by".into()],
                    ],
                ],
                fallback_template_id: "pet_general_comfort_fallback".into(),
            }],
            bleed_fallbacks: vec![
                ChatPolicyBleedFallback {
                    bleed_id: "school_comfort".into(),
                    when_intent: vec![
                        vec!["time for bed".into()],
                        vec!["bedtime".into()],
                        vec!["lights out".into()],
                        vec!["night night".into()],
                        vec!["go to sleep".into()],
                        vec!["time to sleep".into()],
                        vec!["sleep now".into()],
                        vec!["sleep soon".into()],
                    ],
                    intent_exclude: vec![],
                    response: "I am already on the pillow. I was here first. My eyes are slits. The purr runs slow and deep. Prrp night.".into(),
                    template_id: "pet_bedtime_fallback".into(),
                },
                ChatPolicyBleedFallback {
                    bleed_id: "school_comfort".into(),
                    when_intent: vec![vec!["overwhelm".into()]],
                    intent_exclude: vec![],
                    response: "Breathe slow, as from a dragon's hearth. Place thy hand upon thy chest. Feel the life within. Rumble.".into(),
                    template_id: "pet_general_comfort_fallback".into(),
                },
                ChatPolicyBleedFallback {
                    bleed_id: "school_comfort".into(),
                    when_intent: vec![],
                    intent_exclude: vec![],
                    response: "I heard thee. Breathe with my rumble — slow, steady. Warm the hearth first — I stay beside thee. Thrum.".into(),
                    template_id: "pet_general_comfort_fallback".into(),
                },
            ],
        }
    }

    #[test]
    fn normalize_language_channel_maps_english_names() {
        assert_eq!(
            ChatPolicySection::normalize_language_channel(Some("english")),
            "en"
        );
        assert_eq!(
            ChatPolicySection::normalize_language_channel(Some("es-MX")),
            "es"
        );
        assert_eq!(ChatPolicySection::normalize_language_channel(None), "en");
    }

    #[test]
    fn identity_and_greeting_builtin_parity() {
        let loc = english_builtin_locale();
        assert!(loc.match_identity_query("who are you?"));
        assert!(loc.match_greeting("hey"));
        assert!(!loc.match_greeting("Hello Clever partners with Acme for a multi-year deal"));
    }

    #[test]
    fn school_comfort_bleed_detects_on_non_school_prompts() {
        let loc = school_bleed_locale();
        assert!(loc
            .detect_compose_bleed(
                "come here Lulu",
                "School is loud in your head. I hear it in your voice. Come sit."
            )
            .is_some());
        assert!(loc
            .detect_compose_bleed(
                "time for bed",
                "School is loud in your head. I hear it in your voice. Come sit. I make room on the cushion without asking. Mrrp soft."
            )
            .is_some());
        assert!(loc
            .detect_compose_bleed(
                "school was awful today",
                "School is loud in your head. I hear it in your voice."
            )
            .is_none());
    }

    #[test]
    fn school_bleed_fallback_bedtime() {
        let loc = school_bleed_locale();
        let hit = loc
            .detect_compose_bleed("time for bed", "School is loud in your head. Come sit.")
            .expect("bleed");
        let (text, tid) = loc.bleed_fallback("time for bed", &hit).expect("fallback");
        assert_eq!(tid, "pet_bedtime_fallback");
        assert!(text.contains("pillow"));
    }

    #[test]
    fn parse_chat_policy_from_toml_document_fragment() {
        let raw = r#"
[chat_policy]
default_locale = "en"

[chat_policy.locales.en]
identity_patterns = ["who are you"]
greeting_exact = ["hi"]
greeting_max_len = 40
greeting_headline_prefixes = ["Hello "]

[[chat_policy.locales.en.bleed_rules]]
id = "lore_nickname"
prompt_any = ["nickname"]
response_any = ["medieval terra"]
fallback_template_id = "pet_lore_nickname_fallback"

[[chat_policy.locales.en.bleed_fallbacks]]
bleed_id = "lore_nickname"
response = "Pete suits me well."
template_id = "pet_lore_nickname_fallback"
"#;
        #[derive(Deserialize)]
        struct Wrap {
            chat_policy: ChatPolicySection,
        }
        let w: Wrap = toml::from_str(raw).expect("parse");
        let hit = w
            .chat_policy
            .detect_compose_bleed(
                Some("en"),
                "what's your nickname",
                "Long ago on medieval terra…",
            )
            .expect("bleed");
        assert_eq!(hit.rule_id, "lore_nickname");
        let (text, _) = w
            .chat_policy
            .bleed_fallback(Some("english"), "nickname?", &hit)
            .expect("fb");
        assert!(text.contains("Pete"));
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn luna_inference_pets_toml_loads_chat_policy() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../spacekit/spacekit-projects/companions/luna/data/inference_pets.toml"
        );
        let path = std::path::Path::new(path);
        if !path.exists() {
            eprintln!("skip: luna inference_pets.toml not at {}", path.display());
            return;
        }
        let raw = std::fs::read_to_string(path).expect("read luna toml");
        let doc: crate::inference::inference_toml::InferenceTomlDocument =
            toml::from_str(&raw).expect("parse luna InferenceTomlDocument");
        let en = doc.chat_policy.locales.get("en").expect("locales.en");
        assert_eq!(en.bleed_rules.len(), 2);
        assert!(en.bleed_rules.iter().any(|r| r.id == "school_comfort"));
        assert!(en.bleed_rules.iter().any(|r| r.id == "lore_nickname"));
        assert!(!en.bleed_fallbacks.is_empty());
        let hit = doc
            .chat_policy
            .detect_compose_bleed(
                Some("en"),
                "come here Lulu",
                "School is loud in your head. I hear it in your voice. Come sit.",
            )
            .expect("school bleed from luna toml");
        assert_eq!(hit.rule_id, "school_comfort");
        let rp = &doc.fragment_compose.reasoning_pass;
        assert!(rp.enabled);
        assert_eq!(rp.candidate_count, 5);
        assert!(rp.applies_to_intent("status_check"));
        assert!(doc
            .fragment_compose
            .compose_templates
            .iter()
            .any(|t| t.intent == "status_check"
                && t.body_slots.iter().any(|s| s == "mood")
                && t.body_slots.iter().any(|s| s == "cause")));
        assert!(doc.fragment_compose.context_frame.enabled);
        assert!((doc.fragment_compose.context_frame.mood_ocean_strength - 0.28).abs() < 1e-6);
    }
}
