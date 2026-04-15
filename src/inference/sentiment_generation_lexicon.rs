//! Embedded locale lexicon for sentiment lattice generation (`IndexedGenEnv`):
//! witness weak-forms, excerpt stops, topic hints, routing-only coarse map, display labels,
//! **`hard_reject_substrings`** (substring match on ASCII-lowercased text; put lattice / Hopf
//! cross-domain soup phrases here rather than English conjunctions in Rust), and the
//! prompt-anchor marker string.
//!
//! Source: `data/sentiment/sentiment_generation_lexicon.toml`. Add `[locales.fr]` etc. when needed.

use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

#[derive(Debug, Deserialize, Clone)]
struct LocaleSentimentGenLex {
    prompt_anchor_marker: String,
    witness_weak_forms: Vec<String>,
    excerpt_stopwords: Vec<String>,
    internal_cue_prefixes: Vec<String>,
    sentiment_lattice_topic_hints: Vec<String>,
    #[serde(default)]
    routing_coarse: HashMap<String, String>,
    #[serde(default)]
    display_labels: HashMap<String, String>,
    hard_reject_substrings: Vec<String>,
    /// `topic_key` → rationale body when `lexical_polarity_override` wins in lattice shortcuts.
    #[serde(default)]
    lattice_lexical_override_bodies: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct LexiconFile {
    #[serde(default = "default_version")]
    version: u32,
    #[serde(default)]
    default_locale: String,
    locales: HashMap<String, LocaleSentimentGenLex>,
}

fn default_version() -> u32 {
    1
}

/// Loaded sentiment-generation lexicon (one locale; embedded `en` today).
#[derive(Debug)]
pub struct SentimentGenerationLexicon {
    prompt_anchor_marker_leak: &'static str,
    witness_weak: HashSet<String>,
    excerpt_stop: HashSet<String>,
    internal_prefixes: Vec<String>,
    topic_hints: HashSet<String>,
    routing_coarse: HashMap<String, String>,
    display_labels: HashMap<String, String>,
    hard_reject_substrings: Vec<String>,
    lattice_lexical_override_bodies: HashMap<String, String>,
}

impl SentimentGenerationLexicon {
    fn from_locale(loc: LocaleSentimentGenLex) -> Self {
        let marker: String = if loc.prompt_anchor_marker.trim().is_empty() {
            "No witness-matched lattice row for this wording".to_string()
        } else {
            loc.prompt_anchor_marker
        };
        let prompt_anchor_marker_leak: &'static str = Box::leak(marker.into_boxed_str());
        Self {
            prompt_anchor_marker_leak,
            witness_weak: loc.witness_weak_forms.into_iter().collect(),
            excerpt_stop: loc.excerpt_stopwords.into_iter().collect(),
            internal_prefixes: loc.internal_cue_prefixes,
            topic_hints: loc
                .sentiment_lattice_topic_hints
                .into_iter()
                .map(|s| s.to_ascii_lowercase())
                .collect(),
            routing_coarse: loc
                .routing_coarse
                .into_iter()
                .map(|(k, v)| (k.to_ascii_lowercase(), v.to_ascii_lowercase()))
                .collect(),
            display_labels: loc
                .display_labels
                .into_iter()
                .map(|(k, v)| (k.to_ascii_lowercase(), v))
                .collect(),
            hard_reject_substrings: loc.hard_reject_substrings,
            lattice_lexical_override_bodies: loc
                .lattice_lexical_override_bodies
                .into_iter()
                .map(|(k, v)| (k.to_ascii_lowercase(), v))
                .collect(),
        }
    }

    #[inline]
    pub fn prompt_anchor_marker(&self) -> &'static str {
        self.prompt_anchor_marker_leak
    }

    #[inline]
    pub fn is_witness_weak_form(&self, word: &str) -> bool {
        self.witness_weak
            .iter()
            .any(|w| word.eq_ignore_ascii_case(w))
    }

    #[inline]
    pub fn is_excerpt_stopword(&self, word: &str) -> bool {
        self.excerpt_stop
            .iter()
            .any(|w| word.eq_ignore_ascii_case(w))
    }

    /// True when cue token should be dropped (internal graph / money token noise).
    pub fn is_internal_cue_token(&self, t: &str) -> bool {
        let tl = t.to_ascii_lowercase();
        for p in &self.internal_prefixes {
            let pl = p.to_ascii_lowercase();
            if tl.starts_with(&pl) || tl.contains(&pl) {
                return true;
            }
        }
        false
    }

    #[inline]
    pub fn is_sentiment_lattice_topic_hint(&self, h: &str) -> bool {
        if h.eq_ignore_ascii_case("identity") {
            return false;
        }
        let hl = h.to_ascii_lowercase();
        if hl.contains("operation") || hl.contains("algorithm") {
            return false;
        }
        self.topic_hints.contains(&hl)
    }

    /// Coarse topic key for routing-only fallback; default `mixed` if unmapped.
    pub fn routing_coarse_for_hint(&self, hint: &str) -> &'static str {
        let h = hint.trim().to_ascii_lowercase();
        let s = self
            .routing_coarse
            .get(&h)
            .map(|s| s.as_str())
            .unwrap_or("mixed");
        // Leak coarse keys so callers match existing `&'static str` API on group_gen.
        match s {
            "mixed" => "mixed",
            "positive_mild" => "positive_mild",
            "positive_strong" => "positive_strong",
            "negative_mild" => "negative_mild",
            "negative_strong" => "negative_strong",
            "neutral" => "neutral",
            "neutral_chop" => "neutral_chop",
            "confused" => "confused",
            _ => "mixed",
        }
    }

    pub fn display_label_for_topic(&self, hint: &str) -> String {
        let k = hint.trim().to_ascii_lowercase();
        self.display_labels
            .get(&k)
            .cloned()
            .unwrap_or_else(|| hint.replace('_', " ").to_ascii_uppercase())
    }

    /// Hard-reject if any configured substring matches (ASCII-lowercased `text`).
    pub fn hard_reject_lexicon_substrings(&self, text_lower: &str) -> bool {
        self.hard_reject_substrings.iter().any(|s| {
            let sl = s.to_ascii_lowercase();
            text_lower.contains(sl.as_str())
        })
    }

    /// Rationale body for user-anchored line when longest `lexical_polarity` match overrides meta topic.
    #[inline]
    pub fn lattice_lexical_override_body(&self, topic_key: &str) -> Option<&str> {
        self.lattice_lexical_override_bodies
            .get(&topic_key.trim().to_ascii_lowercase())
            .map(|s| s.as_str())
    }
}

static EMBEDDED: OnceLock<SentimentGenerationLexicon> = OnceLock::new();

fn load_embedded() -> SentimentGenerationLexicon {
    let raw = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/data/sentiment/sentiment_generation_lexicon.toml"
    ));
    let file: LexiconFile =
        toml::from_str(raw).expect("parse embedded sentiment_generation_lexicon.toml");
    assert_eq!(
        file.version, 1,
        "unsupported sentiment_generation_lexicon.toml version"
    );
    let key = if file.default_locale.is_empty() {
        "en"
    } else {
        file.default_locale.as_str()
    };
    let loc = file
        .locales
        .get("en")
        .or_else(|| file.locales.get(key))
        .or_else(|| file.locales.values().next())
        .expect("sentiment_generation_lexicon.toml must define at least one [locales.*] table")
        .clone();
    SentimentGenerationLexicon::from_locale(loc)
}

pub fn global() -> &'static SentimentGenerationLexicon {
    EMBEDDED.get_or_init(load_embedded)
}

/// Stable marker substring for prompt-anchored OOD lines (embedded English).
#[inline]
pub fn prompt_anchor_marker() -> &'static str {
    global().prompt_anchor_marker()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_sentiment_gen_lexicon_loads() {
        let g = global();
        assert!(!g.prompt_anchor_marker().is_empty());
        assert!(g.is_witness_weak_form("even"));
        assert!(g.is_excerpt_stopword("the"));
        assert!(g.is_sentiment_lattice_topic_hint("negative_mild"));
        assert!(!g.is_sentiment_lattice_topic_hint("identity"));
        assert_eq!(g.routing_coarse_for_hint("copium"), "mixed");
        assert_eq!(g.routing_coarse_for_hint("identity"), "neutral");
        assert!(g.display_label_for_topic("negative_mild").contains("NEGATIVE"));
        assert!(g.hard_reject_lexicon_substrings("[mask]xx"));
        let paxos_garble = "consensus algorithms — corporate- hr funding; fast behavior reporting a neutral frustration,- news,'.";
        assert!(g.hard_reject_lexicon_substrings(&paxos_garble.to_ascii_lowercase()));
        assert!(g.hard_reject_lexicon_substrings("average operation — classic stop of public grievance"));
        assert!(g
            .lattice_lexical_override_body("neutral")
            .is_some_and(|s| !s.is_empty()));
        assert!(g
            .lattice_lexical_override_body("mixed")
            .is_some_and(|s| !s.is_empty()));
    }
}
