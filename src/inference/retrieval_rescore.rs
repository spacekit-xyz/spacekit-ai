//! Language-neutral hooks for lattice retrieval rescoring (Phase 1–3 of the extensibility plan).
//!
//! Default sentiment/crypto tape adjustments are **declarative** in
//! `data/inference/sentiment_crypto_rescore.toml` — do not grow English-specific `if` chains in
//! `group_gen` for new scenarios; extend the TOML or a future WASM plugin (see
//! `docs/RETRIEVAL_EXTENSIBILITY.md`).

use serde::Deserialize;
use std::sync::OnceLock;

/// Per-candidate lexical view (already lowercased). Host fills this each retrieval step.
#[derive(Debug, Clone)]
pub struct RetrievalCandidateLexical<'a> {
    pub program_index: usize,
    pub text_lower: &'a str,
}

/// Query side for rescore: joined lowercase keywords / tokens from the user line.
#[derive(Debug, Clone)]
pub struct RetrievalQueryLexical<'a> {
    /// Space-joined query terms, lowercased (legacy `qjoin`).
    pub joined: &'a str,
    /// BCP-47-ish tag (e.g. `en`); reserved for locale-specific rule tables (Phase 2).
    pub locale: Option<&'a str>,
}

/// Optional extension point for WASM or custom hosts (Phase 4).
pub trait SentimentRetrievalRescoreExtension: Send + Sync {
    fn apply(
        &self,
        query: &RetrievalQueryLexical<'_>,
        candidate: &RetrievalCandidateLexical<'_>,
        base_score: f32,
    ) -> f32;
}

#[derive(Debug, Deserialize)]
struct SentimentCryptoRescoreFile {
    #[serde(default = "default_version")]
    version: u32,
    /// Declares table language; reserved for loading `sentiment_crypto_rescore.<locale>.toml` later.
    #[serde(default)]
    #[allow(dead_code)]
    locale: String,
    rules: Vec<RescoreRule>,
}

fn default_version() -> u32 {
    1
}

#[derive(Debug, Deserialize)]
struct RescoreRule {
    /// Stable name for diffs and future WASM codegen; matching does not use `id`.
    #[allow(dead_code)]
    id: String,
    #[serde(default)]
    when_query_all: Vec<String>,
    #[serde(default)]
    when_query_any: Vec<String>,
    #[serde(default)]
    unless_query_any: Vec<String>,
    /// If **all** of these substrings appear in the query, the rule is skipped (e.g. skip compression boost when breakdown+funding).
    #[serde(default)]
    unless_query_all: Vec<String>,
    #[serde(default)]
    when_program_all: Vec<String>,
    #[serde(default)]
    when_program_any: Vec<String>,
    #[serde(default)]
    unless_program_any: Vec<String>,
    delta: f32,
}

static DEFAULT_RESCORE_RULES: OnceLock<Vec<RescoreRule>> = OnceLock::new();

fn default_rules_table() -> &'static [RescoreRule] {
    DEFAULT_RESCORE_RULES.get_or_init(|| {
        let raw = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/data/inference/sentiment_crypto_rescore.toml"
        ));
        let file: SentimentCryptoRescoreFile =
            toml::from_str(raw).expect("parse embedded sentiment_crypto_rescore.toml");
        assert!(
            file.version == 1,
            "unsupported sentiment_crypto_rescore.toml version {}",
            file.version
        );
        file.rules
    })
}

fn query_matches(rule: &RescoreRule, qjoin: &str) -> bool {
    if !rule
        .when_query_all
        .iter()
        .all(|s| !s.is_empty() && qjoin.contains(s.as_str()))
    {
        return false;
    }
    if !rule.when_query_any.is_empty()
        && !rule
            .when_query_any
            .iter()
            .any(|s| !s.is_empty() && qjoin.contains(s.as_str()))
    {
        return false;
    }
    if rule
        .unless_query_any
        .iter()
        .any(|s| !s.is_empty() && qjoin.contains(s.as_str()))
    {
        return false;
    }
    if !rule.unless_query_all.is_empty()
        && rule
            .unless_query_all
            .iter()
            .all(|s| !s.is_empty() && qjoin.contains(s.as_str()))
    {
        return false;
    }
    true
}

fn program_matches(rule: &RescoreRule, tl: &str) -> bool {
    if !rule
        .when_program_all
        .iter()
        .all(|s| !s.is_empty() && tl.contains(s.as_str()))
    {
        return false;
    }
    if !rule.when_program_any.is_empty()
        && !rule
            .when_program_any
            .iter()
            .any(|s| !s.is_empty() && tl.contains(s.as_str()))
    {
        return false;
    }
    if rule
        .unless_program_any
        .iter()
        .any(|s| !s.is_empty() && tl.contains(s.as_str()))
    {
        return false;
    }
    true
}

/// Apply embedded TOML rules to `(program_idx, score)` pairs. `program_lower` supplies decoded text per index.
pub fn apply_embedded_sentiment_crypto_rescore(
    qjoin: &str,
    scored: &mut [(usize, f32)],
    mut program_lower: impl FnMut(usize) -> String,
) {
    let rules = default_rules_table();
    for (idx, sc) in scored.iter_mut() {
        let tl = program_lower(*idx);
        let tl = tl.to_ascii_lowercase();
        for rule in rules {
            if query_matches(rule, qjoin) && program_matches(rule, &tl) {
                *sc += rule.delta;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_toml_loads() {
        let rules = default_rules_table();
        assert!(!rules.is_empty());
        assert!(rules.iter().any(|r| r.id == "fintech_consumer_noise"));
    }

    #[test]
    fn breakdown_funding_penalizes_rotation_line() {
        let rules = default_rules_table();
        let q = "everyone breakdown funding mixed signals";
        let tl = "positive for btc but negative for altcoins; mixed sentiment";
        let mut matched = false;
        for rule in rules {
            if rule.id == "breakdown_funding_penalize_rotation_boilerplate_a"
                && query_matches(rule, q)
                && program_matches(rule, tl)
            {
                assert!(rule.delta < 0.0);
                matched = true;
            }
        }
        assert!(matched, "expected rotation penalty rule to match");
    }

    #[test]
    fn breakdown_funding_penalizes_altbtc_boilerplate_line() {
        let rules = default_rules_table();
        let q = "everyone calling breakdown funding mixed signals everywhere";
        let tl = "altcoin strength contrasted with btc weakness; mixed sentiment";
        let mut matched = false;
        for rule in rules {
            if rule.id == "breakdown_funding_penalize_altbtc_mismatch_line"
                && query_matches(rule, q)
                && program_matches(rule, tl)
            {
                assert!(rule.delta < 0.0);
                matched = true;
            }
        }
        assert!(matched, "expected alt/btc mismatch penalty rule to match");
    }
}
