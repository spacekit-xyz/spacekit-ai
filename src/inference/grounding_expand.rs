//! Layer 0 (MVP): data-driven **query expansion** before lattice BM25 / alignment.
//! Rules live in `data/inference/grounding_expand.toml` — add rows instead of new English in `group_gen`.
//!
//! This does not replace causal connectors or the program lattice; it nudges retrieval with
//! extra keywords when substring bundles fire (see `GROWFORMER_CAUSAL_AI.md` — World grounding).

use serde::Deserialize;
use std::sync::OnceLock;

#[derive(Debug, Deserialize)]
struct GroundingExpandFile {
    #[serde(default = "default_version")]
    version: u32,
    #[serde(default)]
    locale: String,
    #[serde(default)]
    rules: Vec<GroundingRuleToml>,
}

fn default_version() -> u32 {
    1
}

#[derive(Debug, Deserialize)]
struct GroundingRuleToml {
    when_all: Vec<String>,
    #[serde(default)]
    when_any: Vec<String>,
    #[serde(default)]
    unless_any: Vec<String>,
    add_keywords: Vec<String>,
}

#[derive(Debug)]
struct GroundingRule {
    when_all: Vec<String>,
    when_any: Vec<String>,
    unless_any: Vec<String>,
    add_keywords: Vec<String>,
}

impl GroundingRule {
    fn matches(&self, padded_lower: &str) -> bool {
        if !self
            .unless_any
            .iter()
            .all(|s| !subslice_hit(padded_lower, s))
        {
            return false;
        }
        if !self
            .when_all
            .iter()
            .all(|s| subslice_hit(padded_lower, s))
        {
            return false;
        }
        if !self.when_any.is_empty()
            && !self
                .when_any
                .iter()
                .any(|s| subslice_hit(padded_lower, s))
        {
            return false;
        }
        true
    }
}

/// True if `needle` appears as a substring in `hay` (both expected lowercase).
fn subslice_hit(hay: &str, needle: &str) -> bool {
    let n = needle.trim();
    if n.is_empty() {
        return true;
    }
    hay.contains(n)
}

fn padded_query(intent_text: &str) -> String {
    format!(
        " {} ",
        intent_text
            .to_ascii_lowercase()
            .replace(['\n', '\r', '\t'], " ")
    )
}

fn load_rules() -> Vec<GroundingRule> {
    let raw = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/data/inference/grounding_expand.toml"
    ));
    let file: GroundingExpandFile = toml::from_str(raw).expect("parse embedded grounding_expand.toml");
    assert_eq!(file.version, 1, "unsupported grounding_expand.toml version");
    file.rules
        .into_iter()
        .map(|r| GroundingRule {
            when_all: r.when_all,
            when_any: r.when_any,
            unless_any: r.unless_any,
            add_keywords: r.add_keywords,
        })
        .collect()
}

static RULES: OnceLock<Vec<GroundingRule>> = OnceLock::new();

fn rules() -> &'static [GroundingRule] {
    RULES.get_or_init(load_rules)
}

/// Append grounding keywords derived from `intent_text` (deduped, min length 3).
pub fn extend_subject_keywords_with_grounding(intent_text: &str, subject_kw: &mut Vec<String>) {
    let padded = padded_query(intent_text);
    for rule in rules() {
        if !rule.matches(&padded) {
            continue;
        }
        for kw in &rule.add_keywords {
            let k = kw.trim().to_ascii_lowercase();
            if k.len() > 2 && !subject_kw.iter().any(|x| x == &k) {
                subject_kw.push(k);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lost_on_game_adds_gambling_cluster() {
        let mut kw = vec!["i".to_string()];
        extend_subject_keywords_with_grounding("I lost $5000 on the game last night", &mut kw);
        assert!(kw.contains(&"gambling".to_string()));
        assert!(kw.contains(&"betting".to_string()));
    }

    #[test]
    fn video_game_excluded() {
        let mut kw: Vec<String> = Vec::new();
        extend_subject_keywords_with_grounding("I lost the game on Steam yesterday", &mut kw);
        assert!(!kw.iter().any(|x| x == "gambling"));
    }

    #[test]
    fn funding_negative_price_hint() {
        let mut kw = Vec::new();
        extend_subject_keywords_with_grounding(
            "Funding is negative but price won't drop. Weird.",
            &mut kw,
        );
        assert!(kw.iter().any(|x| x == "shorts" || x == "squeeze"));
    }
}
