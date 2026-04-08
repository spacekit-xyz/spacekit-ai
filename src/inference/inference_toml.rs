//! Generic inference TOML: flattened numeric gates + `[rules]` lists.
//!
//! **Native:** loaded at runtime from disk (not baked into the binary). Primary resolution:
//! 1. Paths from [`set_inference_toml_cli_paths`] (e.g. `--inference-toml` / `--project` *.gf.toml).
//! 2. `GROWFORMER_INFERENCE_TOML` or legacy `GROWFORMER_SENTIMENT_INFERENCE_TOML` (servers / automation).
//! 3. First readable, valid `data/sentiment/inference_sentiment.toml` under cwd, next to the exe, or
//!    `../data/sentiment/...` from the exe dir.
//!
//! Omitted or empty `[rules]` lists merge from: CLI defaults path, then `GROWFORMER_INFERENCE_DEFAULTS_TOML`,
//! then the first successful default-path document from (3).
//!
//! **wasm32:** no usable filesystem; a single compile-time include is used only for that target.

use serde::Deserialize;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use super::manifest::InferenceThresholds;

/// Default file location relative to cwd or the executable directory.
const INFERENCE_TOML_REL: &str = "data/sentiment/inference_sentiment.toml";

// ---------------------------------------------------------------------------
// CLI / host overrides (set once before first `inference_toml_loaded()`)
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Default)]
struct CliTomlPaths {
    primary: Option<PathBuf>,
    defaults: Option<PathBuf>,
}

#[cfg(not(target_arch = "wasm32"))]
static CLI_TOML_PATHS: OnceLock<CliTomlPaths> = OnceLock::new();

/// Register inference TOML paths from the growformer CLI or host (call before any inference load).
/// Passing `None` for both is a no-op (keeps search/env behavior).
#[cfg(not(target_arch = "wasm32"))]
pub fn set_inference_toml_cli_paths(primary: Option<PathBuf>, defaults: Option<PathBuf>) {
    if primary.is_none() && defaults.is_none() {
        return;
    }
    let _ = CLI_TOML_PATHS.set(CliTomlPaths { primary, defaults });
}

#[cfg(target_arch = "wasm32")]
pub fn set_inference_toml_cli_paths(_primary: Option<PathBuf>, _defaults: Option<PathBuf>) {}

#[cfg(not(target_arch = "wasm32"))]
fn cli_toml_paths() -> CliTomlPaths {
    CLI_TOML_PATHS.get().cloned().unwrap_or_default()
}

// ---------------------------------------------------------------------------
// On-disk / include! document (thresholds + rules)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct InferenceTomlDocument {
    #[serde(flatten)]
    pub thresholds: InferenceThresholds,
    #[serde(default)]
    pub rules: InferenceRulesSection,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct InferenceRulesSection {
    #[serde(default)]
    pub contrastive_markers: Vec<String>,
    /// Ordered: first matching phrase wins for that topic.
    #[serde(default)]
    pub lexical_polarity: Vec<LexicalPolarityRow>,
    /// Any single phrase match → sarcastic template.
    #[serde(default)]
    pub sarcasm_simple: Vec<String>,
    /// Each rule: every group must have at least one phrase match (OR within group, AND across groups).
    #[serde(default)]
    pub sarcasm_and: Vec<SarcasmAndRule>,
    #[serde(default)]
    pub positive_anchor_tokens: Vec<String>,
    #[serde(default)]
    pub negative_anchor_tokens: Vec<String>,
    #[serde(default)]
    pub bipolar_positive_tokens: Vec<String>,
    #[serde(default)]
    pub bipolar_negative_tokens: Vec<String>,
    #[serde(default)]
    pub ambiguous_disappointment_phrases: Vec<String>,
    #[serde(default)]
    pub ambiguous_neutral_hedge_phrases: Vec<String>,
    #[serde(default)]
    pub evaluative_words: Vec<String>,
    #[serde(default)]
    pub disappointment_words: Vec<String>,
    #[serde(default)]
    pub objective_fact_rules: Vec<ObjectiveFactRule>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LexicalPolarityRow {
    pub topic: String,
    pub phrases: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SarcasmAndRule {
    #[serde(default)]
    pub groups: Vec<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ObjectiveFactRule {
    #[serde(default)]
    pub requires_digit: bool,
    #[serde(default)]
    pub all: Vec<String>,
    #[serde(default)]
    pub any: Vec<String>,
}

impl InferenceRulesSection {
    /// Empty lists are replaced from `defaults` (shipped baseline).
    fn merge_empty_from(&self, defaults: &Self) -> Self {
        let mut s = self.clone();
        if s.contrastive_markers.is_empty() {
            s.contrastive_markers = defaults.contrastive_markers.clone();
        }
        if s.lexical_polarity.is_empty() {
            s.lexical_polarity = defaults.lexical_polarity.clone();
        }
        if s.sarcasm_simple.is_empty() {
            s.sarcasm_simple = defaults.sarcasm_simple.clone();
        }
        if s.sarcasm_and.is_empty() {
            s.sarcasm_and = defaults.sarcasm_and.clone();
        }
        if s.positive_anchor_tokens.is_empty() {
            s.positive_anchor_tokens = defaults.positive_anchor_tokens.clone();
        }
        if s.negative_anchor_tokens.is_empty() {
            s.negative_anchor_tokens = defaults.negative_anchor_tokens.clone();
        }
        if s.bipolar_positive_tokens.is_empty() {
            s.bipolar_positive_tokens = defaults.bipolar_positive_tokens.clone();
        }
        if s.bipolar_negative_tokens.is_empty() {
            s.bipolar_negative_tokens = defaults.bipolar_negative_tokens.clone();
        }
        if s.ambiguous_disappointment_phrases.is_empty() {
            s.ambiguous_disappointment_phrases = defaults.ambiguous_disappointment_phrases.clone();
        }
        if s.ambiguous_neutral_hedge_phrases.is_empty() {
            s.ambiguous_neutral_hedge_phrases = defaults.ambiguous_neutral_hedge_phrases.clone();
        }
        if s.evaluative_words.is_empty() {
            s.evaluative_words = defaults.evaluative_words.clone();
        }
        if s.disappointment_words.is_empty() {
            s.disappointment_words = defaults.disappointment_words.clone();
        }
        if s.objective_fact_rules.is_empty() {
            s.objective_fact_rules = defaults.objective_fact_rules.clone();
        }
        s
    }
}

// ---------------------------------------------------------------------------
// Runtime snapshot (Arc, used by lattice shortcut plugin)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct InferenceRulesRuntime {
    pub contrastive_markers: Vec<String>,
    pub lexical_polarity: Vec<(String, Vec<String>)>,
    pub sarcasm_simple: Vec<String>,
    pub sarcasm_and: Vec<Vec<Vec<String>>>,
    pub positive_anchor_tokens: Vec<String>,
    pub negative_anchor_tokens: Vec<String>,
    pub bipolar_positive_tokens: Vec<String>,
    pub bipolar_negative_tokens: Vec<String>,
    pub ambiguous_disappointment_phrases: Vec<String>,
    pub ambiguous_neutral_hedge_phrases: Vec<String>,
    pub evaluative_words: Vec<String>,
    pub disappointment_words: Vec<String>,
    pub objective_fact_rules: Vec<ObjectiveFactRule>,
}

impl InferenceRulesRuntime {
    fn from_section(s: InferenceRulesSection) -> Self {
        Self {
            contrastive_markers: s.contrastive_markers,
            lexical_polarity: s
                .lexical_polarity
                .into_iter()
                .map(|r| (r.topic, r.phrases))
                .collect(),
            sarcasm_simple: s.sarcasm_simple,
            sarcasm_and: s.sarcasm_and.into_iter().map(|r| r.groups).collect(),
            positive_anchor_tokens: s.positive_anchor_tokens,
            negative_anchor_tokens: s.negative_anchor_tokens,
            bipolar_positive_tokens: s.bipolar_positive_tokens,
            bipolar_negative_tokens: s.bipolar_negative_tokens,
            ambiguous_disappointment_phrases: s.ambiguous_disappointment_phrases,
            ambiguous_neutral_hedge_phrases: s.ambiguous_neutral_hedge_phrases,
            evaluative_words: s.evaluative_words,
            disappointment_words: s.disappointment_words,
            objective_fact_rules: s.objective_fact_rules,
        }
    }

    pub fn has_contrastive_marker(&self, lower: &str) -> bool {
        self.contrastive_markers.iter().any(|m| lower.contains(m.as_str()))
    }

    pub fn lexical_polarity_signal(&self, lower: &str) -> Option<String> {
        for (topic, phrases) in &self.lexical_polarity {
            if phrases.iter().any(|p| lower.contains(p.as_str())) {
                return Some(topic.clone());
            }
        }
        None
    }

    pub fn has_sarcasm_template(&self, lower: &str) -> bool {
        if self.sarcasm_simple.iter().any(|p| lower.contains(p.as_str())) {
            return true;
        }
        for groups in &self.sarcasm_and {
            if groups.is_empty() {
                continue;
            }
            if groups
                .iter()
                .all(|g| !g.is_empty() && g.iter().any(|p| lower.contains(p.as_str())))
            {
                return true;
            }
        }
        false
    }

    pub fn has_bipolar_lexicon(&self, lower: &str) -> bool {
        let tokens: std::collections::HashSet<&str> = lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .collect();
        let has_pos = self
            .bipolar_positive_tokens
            .iter()
            .any(|w| tokens.contains(w.as_str()));
        let has_neg = self
            .bipolar_negative_tokens
            .iter()
            .any(|w| tokens.contains(w.as_str()));
        has_pos && has_neg
    }

    pub fn has_clear_evaluative_stance(&self, lower: &str) -> bool {
        self.evaluative_words
            .iter()
            .any(|w| lower.contains(w.as_str()))
    }

    pub fn is_objective_factual_statement(&self, lower: &str) -> bool {
        if self.has_clear_evaluative_stance(lower) {
            return false;
        }
        let has_digit = lower.chars().any(|c| c.is_ascii_digit());
        for rule in &self.objective_fact_rules {
            if rule.requires_digit && !has_digit {
                continue;
            }
            let all_ok = rule.all.is_empty() || rule.all.iter().all(|p| lower.contains(p.as_str()));
            if !all_ok {
                continue;
            }
            let any_ok = rule.any.is_empty() || rule.any.iter().any(|p| lower.contains(p.as_str()));
            if any_ok {
                return true;
            }
        }
        false
    }

    pub fn ambiguous_valence_retarget(&self, lower: &str, key: &str) -> Option<&'static str> {
        if !matches!(key, "positive_strong" | "positive_mild") {
            return None;
        }
        if self
            .ambiguous_disappointment_phrases
            .iter()
            .any(|p| lower.contains(p.as_str()))
        {
            return Some("negative_mild");
        }
        let hedge = self
            .ambiguous_neutral_hedge_phrases
            .iter()
            .any(|p| lower.contains(p.as_str()));
        let lukewarm_okay = (lower.contains("it's okay") || lower.contains("it is okay"))
            && (hedge || lower.contains("suppose"));
        let okay_suppose = lower.contains("okay") && lower.contains("suppose");
        let fine_meh = (lower.contains("it's fine") || lower.contains("it is fine"))
            && (hedge || lower.contains("nothing special"));
        if hedge || lukewarm_okay || okay_suppose || fine_meh {
            return Some("neutral");
        }
        None
    }

    pub fn disappointment_positive_override(&self, lower: &str, key: &str) -> Option<&'static str> {
        if !matches!(key, "positive_strong" | "positive_mild") {
            return None;
        }
        if self
            .disappointment_words
            .iter()
            .any(|w| lower.contains(w.as_str()))
        {
            return Some("negative_mild");
        }
        None
    }

    pub fn anchor_phrase(&self, lower: &str, topic_key: &str) -> Option<String> {
        let tokens: std::collections::HashSet<&str> = lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .collect();
        match topic_key {
            "positive_strong" | "positive_mild" => {
                for w in &self.positive_anchor_tokens {
                    if tokens.contains(w.as_str()) {
                        let gloss = match w.as_str() {
                            "love" | "adore" | "treasure" => "strong positive affection",
                            "enjoy" | "like" | "prefer" => "warm approval / preference",
                            _ => "positive valence",
                        };
                        return Some(format!("Anchored by '{}' ({})", w, gloss));
                    }
                }
                None
            }
            "negative_strong" | "negative_mild" => {
                for w in &self.negative_anchor_tokens {
                    if tokens.contains(w.as_str()) {
                        let gloss = match w.as_str() {
                            "hate" | "despise" | "loathe" => "strong negative affect",
                            _ => "negative valence",
                        };
                        return Some(format!("Anchored by '{}' ({})", w, gloss));
                    }
                }
                None
            }
            _ => None,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn candidate_paths() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        v.push(cwd.join(INFERENCE_TOML_REL));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            v.push(dir.join(INFERENCE_TOML_REL));
            v.push(dir.join("../").join(INFERENCE_TOML_REL));
        }
    }
    v
}

#[cfg(not(target_arch = "wasm32"))]
fn read_first_parsed(paths: &[PathBuf]) -> Option<InferenceTomlDocument> {
    for p in paths {
        let Ok(s) = std::fs::read_to_string(p) else {
            continue;
        };
        match toml::from_str::<InferenceTomlDocument>(&s) {
            Ok(doc) => return Some(doc),
            Err(e) => eprintln!(
                "[inference-toml] skip unreadable/invalid {}: {}",
                p.display(),
                e
            ),
        }
    }
    None
}

/// Rules baseline for `merge_empty_from`.
#[cfg(not(target_arch = "wasm32"))]
fn baseline_rules_section() -> InferenceRulesSection {
    if let Some(p) = cli_toml_paths().defaults {
        match std::fs::read_to_string(&p) {
            Ok(s) => match toml::from_str::<InferenceTomlDocument>(&s) {
                Ok(doc) => return doc.rules,
                Err(e) => eprintln!(
                    "[inference-toml] --inference-defaults-toml {} invalid: {}",
                    p.display(),
                    e
                ),
            },
            Err(e) => eprintln!(
                "[inference-toml] could not read --inference-defaults-toml {}: {}",
                p.display(),
                e
            ),
        }
    }
    if let Ok(path) = std::env::var("GROWFORMER_INFERENCE_DEFAULTS_TOML") {
        let p = PathBuf::from(&path);
        if let Ok(s) = std::fs::read_to_string(&p) {
            if let Ok(doc) = toml::from_str::<InferenceTomlDocument>(&s) {
                return doc.rules;
            }
        }
        eprintln!(
            "[inference-toml] GROWFORMER_INFERENCE_DEFAULTS_TOML={} missing or invalid; using default-path baseline",
            path
        );
    }
    read_first_parsed(&candidate_paths())
        .map(|d| d.rules)
        .unwrap_or_default()
}

#[cfg(not(target_arch = "wasm32"))]
fn load_primary_document() -> Result<InferenceTomlDocument, String> {
    if let Some(p) = cli_toml_paths().primary {
        match std::fs::read_to_string(&p) {
            Ok(s) => match toml::from_str::<InferenceTomlDocument>(&s) {
                Ok(doc) => return Ok(doc),
                Err(e) => eprintln!(
                    "[inference-toml] failed to parse inference TOML {}: {} — trying env / default paths",
                    p.display(),
                    e
                ),
            },
            Err(e) => eprintln!(
                "[inference-toml] could not read inference TOML {}: {} — trying env / default paths",
                p.display(),
                e
            ),
        }
    }
    let env_path = std::env::var("GROWFORMER_INFERENCE_TOML")
        .ok()
        .or_else(|| std::env::var("GROWFORMER_SENTIMENT_INFERENCE_TOML").ok());
    if let Some(path) = env_path {
        let p = PathBuf::from(&path);
        match std::fs::read_to_string(&p) {
            Ok(s) => match toml::from_str::<InferenceTomlDocument>(&s) {
                Ok(doc) => return Ok(doc),
                Err(e) => eprintln!(
                    "[inference-toml] failed to parse GROWFORMER_INFERENCE_TOML={}: {} — trying default paths",
                    path, e
                ),
            },
            Err(e) => eprintln!(
                "[inference-toml] could not read GROWFORMER_INFERENCE_TOML={}: {} — trying default paths",
                path, e
            ),
        }
    }
    let paths = candidate_paths();
    read_first_parsed(&paths).ok_or_else(|| {
        format!(
            "[inference-toml] no inference TOML found. Use --inference-toml or --project *.gf.toml, set GROWFORMER_INFERENCE_TOML, or place {} where the process can read it (tried: {:?})",
            INFERENCE_TOML_REL,
            paths
        )
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn load_inference_toml_merged() -> InferenceTomlDocument {
    let mut file = load_primary_document().unwrap_or_else(|e| panic!("{}", e));
    let defaults = baseline_rules_section();
    file.rules = file.rules.merge_empty_from(&defaults);
    file
}

#[cfg(target_arch = "wasm32")]
fn load_inference_toml_merged() -> InferenceTomlDocument {
    const WASM_EMBED: &str = include_str!("../../data/sentiment/inference_sentiment.toml");
    toml::from_str(WASM_EMBED).expect("wasm embedded inference TOML invalid")
}

/// Single load: thresholds from file + merged rule lists.
#[derive(Debug)]
pub struct LoadedInferenceToml {
    pub thresholds: InferenceThresholds,
    rules: Arc<InferenceRulesRuntime>,
}

impl LoadedInferenceToml {
    pub fn rules(&self) -> Arc<InferenceRulesRuntime> {
        self.rules.clone()
    }
}

static FULL: OnceLock<Arc<LoadedInferenceToml>> = OnceLock::new();

pub fn inference_toml_loaded() -> Arc<LoadedInferenceToml> {
    FULL.get_or_init(|| {
        let file = load_inference_toml_merged();
        let rules = Arc::new(InferenceRulesRuntime::from_section(file.rules));
        Arc::new(LoadedInferenceToml {
            thresholds: file.thresholds,
            rules,
        })
    })
    .clone()
}

pub fn inference_rules_runtime() -> Arc<InferenceRulesRuntime> {
    inference_toml_loaded().rules()
}
