//! Locale-aware **lexical** retrieval helpers: stopwords, intent-action names, code-detection markers.
//!
//! Default English table: `data/inference/retrieval_lexicon.toml` (`locales.en`). Add `locales.fr`, …
//! and switch by `language_channel` / manifest in a follow-up.

use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

#[derive(Debug, Deserialize, Clone)]
struct LocaleLexToml {
    graph_stopwords: Vec<String>,
    lex_align_stopwords: Vec<String>,
    intent_implement: Vec<String>,
    intent_explain: Vec<String>,
    code_markers_bm25: Vec<String>,
    code_markers_rust: Vec<String>,
    code_markers_python: Vec<String>,
    code_python_return: String,
    code_python_exclude_arrow: String,
}

#[derive(Debug, Deserialize)]
struct LexiconFile {
    #[serde(default = "default_version")]
    version: u32,
    #[serde(default)]
    default_locale: String,
    locales: HashMap<String, LocaleLexToml>,
}

fn default_version() -> u32 {
    1
}

/// Loaded retrieval lexicon for one locale (currently embedded `en` only at runtime).
#[derive(Debug)]
pub struct RetrievalLexicon {
    graph_stop: HashSet<String>,
    lex_align_stop: HashSet<String>,
    intent_implement: HashSet<String>,
    intent_explain: HashSet<String>,
    code_markers_bm25: Vec<String>,
    code_markers_rust: Vec<String>,
    code_markers_python: Vec<String>,
    code_python_return: String,
    code_python_exclude_arrow: String,
}

impl RetrievalLexicon {
    fn from_toml_locale(loc: LocaleLexToml) -> Self {
        Self {
            graph_stop: loc.graph_stopwords.into_iter().collect(),
            lex_align_stop: loc.lex_align_stopwords.into_iter().collect(),
            intent_implement: loc.intent_implement.into_iter().collect(),
            intent_explain: loc.intent_explain.into_iter().collect(),
            code_markers_bm25: loc.code_markers_bm25,
            code_markers_rust: loc.code_markers_rust,
            code_markers_python: loc.code_markers_python,
            code_python_return: loc.code_python_return,
            code_python_exclude_arrow: loc.code_python_exclude_arrow,
        }
    }

    #[inline]
    pub fn is_graph_stop(&self, word: &str) -> bool {
        self.graph_stop.contains(word)
    }

    #[inline]
    pub fn is_lex_align_stop(&self, word: &str) -> bool {
        self.lex_align_stop.contains(word)
    }

    pub fn intent_prefers_code(&self, intent_action: &str) -> bool {
        self.intent_implement.contains(intent_action)
    }

    pub fn intent_prefers_prose(&self, intent_action: &str) -> bool {
        self.intent_explain.contains(intent_action)
    }

    pub fn program_has_code_markers_bm25(&self, text: &str) -> bool {
        self.code_markers_bm25
            .iter()
            .any(|m| text.contains(m.as_str()))
    }

    pub fn program_matches_rust_lang_hint(&self, text: &str) -> bool {
        self.code_markers_rust
            .iter()
            .any(|m| text.contains(m.as_str()))
    }

    /// Matches historical behavior: `def` or (`return` without `->`).
    pub fn program_matches_python_lang_hint(&self, text: &str) -> bool {
        self.code_markers_python
            .iter()
            .any(|m| text.contains(m.as_str()))
            || (text.contains(self.code_python_return.as_str())
                && !text.contains(self.code_python_exclude_arrow.as_str()))
    }
}

static EMBEDDED_EN: OnceLock<RetrievalLexicon> = OnceLock::new();

// TODO: Add Dynamic loading of lexicon files
fn load_embedded_default() -> RetrievalLexicon {
    let raw = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/data/inference/retrieval_lexicon.toml"
    ));
    let file: LexiconFile = toml::from_str(raw).expect("parse embedded retrieval_lexicon.toml");
    assert_eq!(
        file.version, 1,
        "unsupported retrieval_lexicon.toml version"
    );
    let key = if file.default_locale.is_empty() {
        "en".to_string()
    } else {
        file.default_locale.clone()
    };
    let loc = file
        .locales
        .get("en")
        .or_else(|| file.locales.get(&key))
        .or_else(|| file.locales.values().next())
        .expect("retrieval_lexicon.toml must define at least one [locales.*] table")
        .clone();
    RetrievalLexicon::from_toml_locale(loc)
}

/// English embedded lexicon (compile-time). Use [`global_for_locale`] when multi-locale tables exist.
pub fn global() -> &'static RetrievalLexicon {
    EMBEDDED_EN.get_or_init(load_embedded_default)
}

/// Select lexicon by BCP-47-ish tag; falls back to embedded English until more locales are shipped.
pub fn global_for_locale(locale: Option<&str>) -> &'static RetrievalLexicon {
    let _ = locale;
    // Phase 2: map `Some("fr")` → `locales.fr` once those tables ship.
    global()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_loads() {
        let g = global();
        assert!(g.is_graph_stop("the"));
        assert!(g.intent_prefers_code("implement"));
        assert!(g.program_has_code_markers_bm25("pub fn main() {}"));
    }
}
