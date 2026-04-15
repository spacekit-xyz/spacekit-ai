//! Generic inference TOML: flattened numeric gates + `[rules]` lists.
//!
//! **Native:** loaded at runtime from disk (not baked into the binary). Primary resolution:
//! 1. Paths from [`set_inference_toml_cli_paths`] (e.g. `--inference-toml` / `--project` *.gf.toml).
//! 2. `GROWFORMER_INFERENCE_TOML` or legacy `GROWFORMER_SENTIMENT_INFERENCE_TOML` (servers / automation).
//! 3. First readable, valid file among **candidate relative paths** under cwd, next to the exe, and
//!    `../data/...` from the exe dir. Built-in order: `data/sentiment/inference_sentiment_core.toml`, then
//!    `data/fintech/inference_fintech.toml`. Prepend or override order with comma-separated
//!    **`GROWFORMER_INFERENCE_TOML_DEFAULT_RELS`** (paths relative to cwd / exe roots above).
//!
//! PR / wire headline → neutral routing: **`[[rules.pr_wire_neutral_prefix]]`** (in the core defaults)
//! plus optional **`[[rules.pr_wire_neutral_intent]]`** rows in domain / reference packs — not in Rust literals.
//!
//! Mixed-signal / ambiguous-valence / anchor gloss phrases live under `[rules]` keys such as
//! **`mixed_positive_outcome_phrases`**, **`mixed_skepticism_friction_phrases`**, **`ambiguous_neutral_conjunction_groups`**,
//! and **`[[rules.anchor_positive_gloss]]`** / **`[[rules.anchor_negative_gloss]]`** (see `InferenceRulesSection`).
//!
//! Omitted or empty `[rules]` lists merge from: CLI defaults path, then `GROWFORMER_INFERENCE_DEFAULTS_TOML`,
//! then **every** successful default-path document from (3) in search order, each pass filling only
//! fields still empty after the previous pass (so `inference_sentiment_core.toml` stays primary for
//! numeric gates and filled lists, and `inference_fintech.toml` can supply e.g. `headline_lexical_topic`).
//!
//! **wasm32:** no usable filesystem; a single compile-time include is used only for that target.
//!
//! **Guardrails JSONL** (optional): after merged TOML rules, [`crate::inference::inference_guardrails`]
//! appends `lexical_topic` / `lattice_misfire` lines from disk (CLI `--inference-guardrails-jsonl`,
//! `GROWFORMER_INFERENCE_GUARDRAILS_JSONL`, or default `data/*/inference_guardrails.jsonl` files).
//!
//! **Runtime:** [`InferenceRulesRuntime`] sentiment helpers are only consulted from [`crate::service::LanguageService`]
//! when [`crate::inference::plugins::lattice_shortcuts::sentiment_toml_lexical_guards_active`] is true
//! (sentiment-shaped gen lattice present; brain `inference_profile` not `off` / `none` / `disabled`).
//!
//! Phrase lists here are **guard rails** until retrieval and training reliably separate domains
//! (e.g. infra headlines vs tax-filing lattice rows). Matching is **O(total phrases)** per call;
//! if that becomes hot, precompile patterns or batch-check.

use aho_corasick::AhoCorasick;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use super::manifest::InferenceThresholds;

/// Shipped fallbacks when no CLI primary / env absolute path is set: each path is resolved relative
/// to cwd, then `exe_dir`, then `exe_dir/../` (same order as [`candidate_paths`]).
const INFERENCE_TOML_BUILTIN_DEFAULT_RELS: &[&str] = &[
    "data/sentiment/inference_sentiment_core.toml",
    "data/fintech/inference_fintech.toml",
];

/// Relative paths to try first (comma-separated), then [`INFERENCE_TOML_BUILTIN_DEFAULT_RELS`]
/// (deduped). Set e.g. `data/fintech/inference_fintech.toml` alone to prefer fintech when both exist.
fn inference_toml_rel_search_order() -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(env) = std::env::var("GROWFORMER_INFERENCE_TOML_DEFAULT_RELS") {
        for p in env.split(',') {
            let p = p.trim();
            if !p.is_empty() {
                out.push(p.to_string());
            }
        }
    }
    for p in INFERENCE_TOML_BUILTIN_DEFAULT_RELS {
        if !out.iter().any(|e| e == p) {
            out.push((*p).to_string());
        }
    }
    out
}

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

/// PR-wire headline: normalized text must start with `prefix`, meet `min_len`, and pass excludes / `require_any`.
#[derive(Debug, Clone, Deserialize)]
pub struct PrWireNeutralPrefixRule {
    pub prefix: String,
    #[serde(default = "default_pr_wire_prefix_min_len")]
    pub min_len: usize,
    #[serde(default)]
    pub exclude_prefixes: Vec<String>,
    #[serde(default)]
    pub require_any: Vec<String>,
    #[serde(default)]
    pub disallow_question_mark: bool,
}

fn default_pr_wire_prefix_min_len() -> usize {
    40
}

/// PR-wire headline: CNF substring match on trimmed normalized intent (`min_len` applies to full headline).
#[derive(Debug, Clone, Deserialize)]
pub struct PrWireNeutralIntentRow {
    #[serde(default)]
    pub min_len: Option<usize>,
    #[serde(default)]
    pub intent: Vec<Vec<String>>,
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
    /// Ordered headline overrides: first CNF `intent` match wins (`topic` sub-lattice key).
    #[serde(default)]
    pub headline_lexical_topic: Vec<HeadlineLexicalTopicRule>,
    /// Lattice output looks wrong for this intent (intent CNF ∧ response side).
    #[serde(default)]
    pub lattice_misfire: Vec<LatticeMisfireRule>,
    /// `How …` / `Why …` wire headlines: prefix + min length + excludes + optional `require_any`.
    #[serde(default)]
    pub pr_wire_neutral_prefix: Vec<PrWireNeutralPrefixRule>,
    /// Substring CNF rows → neutral (optional `min_len` on the whole headline).
    #[serde(default)]
    pub pr_wire_neutral_intent: Vec<PrWireNeutralIntentRow>,
    /// Substrings for [`InferenceRulesRuntime::has_crypto_or_broad_market_lexicon`].
    #[serde(default)]
    pub crypto_market_surface_tokens: Vec<String>,
    /// `starts_with` checks (e.g. `"eth "` headlines) — see [`InferenceRulesRuntime::has_crypto_or_broad_market_lexicon`].
    #[serde(default)]
    pub crypto_market_surface_prefixes: Vec<String>,
    /// Extra tape-style substrings OR‑joined in [`InferenceRulesRuntime::intent_text_suggests_crypto_market`].
    #[serde(default)]
    pub crypto_market_tape_tokens: Vec<String>,
    /// Word before `like` that marks perception idiom (“feels like”), not praise.
    #[serde(default)]
    pub like_perception_verbs: Vec<String>,
    /// Skip `like` anchor when any of these phrases hit (“like crazy”).
    #[serde(default)]
    pub like_intensity_exception_phrases: Vec<String>,
    /// Token hits → `positive_strong` before mild anchors.
    #[serde(default)]
    pub strong_positive_tokens: Vec<String>,
    /// Minimum `trim()` length before PR-wire prefix/intent tables apply (press headlines).
    #[serde(default)]
    pub pr_wire_press_min_trim_len: Option<usize>,
    /// Headline `topic` keys that count as plausible `mixed` without contrast/bipolar (see `sentiment_allow_forced_mixed_topic`).
    #[serde(default)]
    pub mixed_plausible_headline_topics: Vec<String>,
    /// Bad-clause substrings for silver-lining `mixed` (both sides must hit).
    #[serde(default)]
    pub mixed_silver_lining_bad_phrases: Vec<String>,
    #[serde(default)]
    pub mixed_silver_lining_good_phrases: Vec<String>,
    /// Require at least one anchor (e.g. `glad`) AND one of `mixed_fraud_relief_trigger_any`.
    #[serde(default)]
    pub mixed_fraud_relief_anchor_phrases: Vec<String>,
    #[serde(default)]
    pub mixed_fraud_relief_trigger_any: Vec<String>,
    #[serde(default)]
    pub mixed_implicit_followon_phrases: Vec<String>,
    #[serde(default)]
    pub mixed_implicit_unusual_token: String,
    #[serde(default)]
    pub mixed_implicit_unusual_context_any: Vec<String>,
    #[serde(default)]
    pub mixed_implicit_unusual_exclude_any: Vec<String>,
    #[serde(default)]
    pub mixed_positive_outcome_phrases: Vec<String>,
    #[serde(default)]
    pub mixed_skepticism_friction_phrases: Vec<String>,
    #[serde(default)]
    pub mixed_operational_decline_phrases: Vec<String>,
    #[serde(default)]
    pub mixed_operational_approve_phrases: Vec<String>,
    /// `ambiguous_valence_retarget`: lukewarm “okay” primary cues.
    #[serde(default)]
    pub ambiguous_lukewarm_okay_primary_phrases: Vec<String>,
    #[serde(default)]
    pub ambiguous_lukewarm_okay_supplement_phrases: Vec<String>,
    #[serde(default)]
    pub ambiguous_fine_meh_primary_phrases: Vec<String>,
    #[serde(default)]
    pub ambiguous_fine_meh_supplement_phrases: Vec<String>,
    /// Each inner list: **all** substrings must match for neutral retarget (e.g. `okay` + `suppose`).
    #[serde(default)]
    pub ambiguous_neutral_conjunction_groups: Vec<Vec<String>>,
    #[serde(default)]
    pub anchor_positive_gloss: Vec<AnchorTokenGlossRow>,
    #[serde(default)]
    pub anchor_negative_gloss: Vec<AnchorTokenGlossRow>,
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

/// Per-token gloss for [`InferenceRulesRuntime::anchor_phrase`] (substring / token hit on `w`).
#[derive(Debug, Clone, Deserialize)]
pub struct AnchorTokenGlossRow {
    pub token: String,
    pub gloss: String,
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

/// CNF on intent text: every group must match; within a group, any phrase match suffices (OR).
///
/// Optional gates (all default off / empty): `min_trim_len`, `exclude_first_person`,
/// `require_crypto_lexicon` (uses `[rules].crypto_market_surface_tokens` + `crypto_market_surface_prefixes`),
/// `unless_any_cnf` (skip rule if any inner CNF matches), `after_pr_wire` (run only after PR-wire tables),
/// `requires_mixed_positive_outcome_cue`, `require_question_mark`.
#[derive(Debug, Clone, Deserialize)]
pub struct HeadlineLexicalTopicRule {
    pub topic: String,
    /// Service may redirect to the sentiment lattice and bypass the knowledge floor.
    #[serde(default)]
    pub inclusion_redirect: bool,
    #[serde(default)]
    pub intent: Vec<Vec<String>>,
    /// Minimum `trim()` length of the full normalized intent (headline-style gates).
    #[serde(default)]
    pub min_trim_len: Option<usize>,
    /// Skip when [`InferenceRulesRuntime::looks_like_first_person_finance_user`] is true.
    #[serde(default)]
    pub exclude_first_person: bool,
    /// Require at least one substring from loaded `crypto_market_surface_tokens`.
    #[serde(default)]
    pub require_crypto_lexicon: bool,
    /// If any listed CNF matches, this rule is skipped (OR across CNFs).
    #[serde(default)]
    pub unless_any_cnf: Vec<Vec<Vec<String>>>,
    /// When true, evaluated only **after** PR-wire prefix/intent tables (`sentiment_fintech_press_headline_topic_key`).
    #[serde(default)]
    pub after_pr_wire: bool,
    #[serde(default)]
    pub requires_mixed_positive_outcome_cue: bool,
    #[serde(default)]
    pub require_question_mark: bool,
}

impl Default for HeadlineLexicalTopicRule {
    fn default() -> Self {
        Self {
            topic: String::new(),
            inclusion_redirect: false,
            intent: Vec::new(),
            min_trim_len: None,
            exclude_first_person: false,
            require_crypto_lexicon: false,
            unless_any_cnf: Vec::new(),
            after_pr_wire: false,
            requires_mixed_positive_outcome_cue: false,
            require_question_mark: false,
        }
    }
}

/// Detect composed lattice lines that contradict the headline (replace with routing-only fallback).
#[derive(Debug, Clone, Deserialize)]
pub struct LatticeMisfireRule {
    #[serde(default)]
    pub intent: Vec<Vec<String>>,
    /// Any listed substring in the response counts as a hit (OR). Combined with `response` when both set.
    #[serde(default)]
    pub response_any: Vec<String>,
    /// AND-of-OR groups on the response (substring match). Empty means ignore unless `response_any` set.
    #[serde(default)]
    pub response: Vec<Vec<String>>,
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
        if s.headline_lexical_topic.is_empty() {
            s.headline_lexical_topic = defaults.headline_lexical_topic.clone();
        }
        if s.lattice_misfire.is_empty() {
            s.lattice_misfire = defaults.lattice_misfire.clone();
        }
        if s.pr_wire_neutral_prefix.is_empty() {
            s.pr_wire_neutral_prefix = defaults.pr_wire_neutral_prefix.clone();
        }
        if s.pr_wire_neutral_intent.is_empty() {
            s.pr_wire_neutral_intent = defaults.pr_wire_neutral_intent.clone();
        }
        if s.crypto_market_surface_tokens.is_empty() {
            s.crypto_market_surface_tokens = defaults.crypto_market_surface_tokens.clone();
        }
        if s.crypto_market_surface_prefixes.is_empty() {
            s.crypto_market_surface_prefixes = defaults.crypto_market_surface_prefixes.clone();
        }
        if s.crypto_market_tape_tokens.is_empty() {
            s.crypto_market_tape_tokens = defaults.crypto_market_tape_tokens.clone();
        }
        if s.like_perception_verbs.is_empty() {
            s.like_perception_verbs = defaults.like_perception_verbs.clone();
        }
        if s.like_intensity_exception_phrases.is_empty() {
            s.like_intensity_exception_phrases = defaults.like_intensity_exception_phrases.clone();
        }
        if s.strong_positive_tokens.is_empty() {
            s.strong_positive_tokens = defaults.strong_positive_tokens.clone();
        }
        if s.pr_wire_press_min_trim_len.is_none() {
            s.pr_wire_press_min_trim_len = defaults.pr_wire_press_min_trim_len;
        }
        if s.mixed_plausible_headline_topics.is_empty() {
            s.mixed_plausible_headline_topics = defaults.mixed_plausible_headline_topics.clone();
        }
        if s.mixed_silver_lining_bad_phrases.is_empty() {
            s.mixed_silver_lining_bad_phrases = defaults.mixed_silver_lining_bad_phrases.clone();
        }
        if s.mixed_silver_lining_good_phrases.is_empty() {
            s.mixed_silver_lining_good_phrases = defaults.mixed_silver_lining_good_phrases.clone();
        }
        if s.mixed_fraud_relief_anchor_phrases.is_empty() {
            s.mixed_fraud_relief_anchor_phrases = defaults.mixed_fraud_relief_anchor_phrases.clone();
        }
        if s.mixed_fraud_relief_trigger_any.is_empty() {
            s.mixed_fraud_relief_trigger_any = defaults.mixed_fraud_relief_trigger_any.clone();
        }
        if s.mixed_implicit_followon_phrases.is_empty() {
            s.mixed_implicit_followon_phrases = defaults.mixed_implicit_followon_phrases.clone();
        }
        if s.mixed_implicit_unusual_token.is_empty() {
            s.mixed_implicit_unusual_token = defaults.mixed_implicit_unusual_token.clone();
        }
        if s.mixed_implicit_unusual_context_any.is_empty() {
            s.mixed_implicit_unusual_context_any = defaults.mixed_implicit_unusual_context_any.clone();
        }
        if s.mixed_implicit_unusual_exclude_any.is_empty() {
            s.mixed_implicit_unusual_exclude_any = defaults.mixed_implicit_unusual_exclude_any.clone();
        }
        if s.mixed_positive_outcome_phrases.is_empty() {
            s.mixed_positive_outcome_phrases = defaults.mixed_positive_outcome_phrases.clone();
        }
        if s.mixed_skepticism_friction_phrases.is_empty() {
            s.mixed_skepticism_friction_phrases = defaults.mixed_skepticism_friction_phrases.clone();
        }
        if s.mixed_operational_decline_phrases.is_empty() {
            s.mixed_operational_decline_phrases = defaults.mixed_operational_decline_phrases.clone();
        }
        if s.mixed_operational_approve_phrases.is_empty() {
            s.mixed_operational_approve_phrases = defaults.mixed_operational_approve_phrases.clone();
        }
        if s.ambiguous_lukewarm_okay_primary_phrases.is_empty() {
            s.ambiguous_lukewarm_okay_primary_phrases = defaults.ambiguous_lukewarm_okay_primary_phrases.clone();
        }
        if s.ambiguous_lukewarm_okay_supplement_phrases.is_empty() {
            s.ambiguous_lukewarm_okay_supplement_phrases = defaults.ambiguous_lukewarm_okay_supplement_phrases.clone();
        }
        if s.ambiguous_fine_meh_primary_phrases.is_empty() {
            s.ambiguous_fine_meh_primary_phrases = defaults.ambiguous_fine_meh_primary_phrases.clone();
        }
        if s.ambiguous_fine_meh_supplement_phrases.is_empty() {
            s.ambiguous_fine_meh_supplement_phrases = defaults.ambiguous_fine_meh_supplement_phrases.clone();
        }
        if s.ambiguous_neutral_conjunction_groups.is_empty() {
            s.ambiguous_neutral_conjunction_groups = defaults.ambiguous_neutral_conjunction_groups.clone();
        }
        if s.anchor_positive_gloss.is_empty() {
            s.anchor_positive_gloss = defaults.anchor_positive_gloss.clone();
        }
        if s.anchor_negative_gloss.is_empty() {
            s.anchor_negative_gloss = defaults.anchor_negative_gloss.clone();
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
    contrastive_markers_ac: Option<AhoCorasick>,
    pub lexical_polarity: Vec<(String, Vec<String>)>,
    lexical_polarity_automaton: Option<LexicalPolarityAutomaton>,
    /// Phrases from `lexical_polarity` rows with topic `negative_mild` / `negative_strong` — used to
    /// suppress misleading positive token anchors (e.g. token `like` inside “don’t like”).
    negative_lexical_suppress_phrases: Vec<String>,
    negative_lexical_suppress_ac: Option<AhoCorasick>,
    pub sarcasm_simple: Vec<String>,
    sarcasm_simple_ac: Option<AhoCorasick>,
    pub sarcasm_and: Vec<Vec<Vec<String>>>,
    pub positive_anchor_tokens: Vec<String>,
    pub negative_anchor_tokens: Vec<String>,
    pub bipolar_positive_tokens: Vec<String>,
    pub bipolar_negative_tokens: Vec<String>,
    pub ambiguous_disappointment_phrases: Vec<String>,
    ambiguous_disappointment_ac: Option<AhoCorasick>,
    pub ambiguous_neutral_hedge_phrases: Vec<String>,
    ambiguous_neutral_hedge_ac: Option<AhoCorasick>,
    pub evaluative_words: Vec<String>,
    pub disappointment_words: Vec<String>,
    pub objective_fact_rules: Vec<ObjectiveFactRule>,
    pub headline_lexical_topic: Vec<HeadlineLexicalTopicRule>,
    pub lattice_misfire: Vec<LatticeMisfireRule>,
    pub pr_wire_neutral_prefix: Vec<PrWireNeutralPrefixRule>,
    pub pr_wire_neutral_intent: Vec<PrWireNeutralIntentRow>,
    crypto_market_surface_tokens: Vec<String>,
    crypto_market_surface_prefixes: Vec<String>,
    crypto_market_tape_tokens: Vec<String>,
    like_perception_verbs: Vec<String>,
    like_intensity_exception_phrases: Vec<String>,
    like_intensity_exception_ac: Option<AhoCorasick>,
    strong_positive_tokens: Vec<String>,
    pr_wire_press_min_trim_len: usize,
    mixed_plausible_headline_topics: Vec<String>,
    mixed_silver_lining_bad_phrases: Vec<String>,
    mixed_silver_lining_bad_ac: Option<AhoCorasick>,
    mixed_silver_lining_good_phrases: Vec<String>,
    mixed_silver_lining_good_ac: Option<AhoCorasick>,
    mixed_fraud_relief_anchor_phrases: Vec<String>,
    mixed_fraud_relief_anchor_ac: Option<AhoCorasick>,
    mixed_fraud_relief_trigger_any: Vec<String>,
    mixed_fraud_relief_trigger_ac: Option<AhoCorasick>,
    mixed_implicit_followon_phrases: Vec<String>,
    mixed_implicit_followon_ac: Option<AhoCorasick>,
    mixed_implicit_unusual_token: String,
    mixed_implicit_unusual_context_any: Vec<String>,
    mixed_implicit_unusual_context_ac: Option<AhoCorasick>,
    mixed_implicit_unusual_exclude_any: Vec<String>,
    mixed_implicit_unusual_exclude_ac: Option<AhoCorasick>,
    mixed_positive_outcome_phrases: Vec<String>,
    mixed_positive_outcome_ac: Option<AhoCorasick>,
    mixed_skepticism_friction_phrases: Vec<String>,
    mixed_skepticism_friction_ac: Option<AhoCorasick>,
    mixed_operational_decline_phrases: Vec<String>,
    mixed_operational_approve_phrases: Vec<String>,
    mixed_operational_approve_ac: Option<AhoCorasick>,
    ambiguous_lukewarm_okay_primary_phrases: Vec<String>,
    ambiguous_lukewarm_okay_primary_ac: Option<AhoCorasick>,
    ambiguous_lukewarm_okay_supplement_phrases: Vec<String>,
    ambiguous_lukewarm_okay_supplement_ac: Option<AhoCorasick>,
    ambiguous_fine_meh_primary_phrases: Vec<String>,
    ambiguous_fine_meh_primary_ac: Option<AhoCorasick>,
    ambiguous_fine_meh_supplement_phrases: Vec<String>,
    ambiguous_fine_meh_supplement_ac: Option<AhoCorasick>,
    ambiguous_neutral_conjunction_groups: Vec<Vec<String>>,
    anchor_positive_gloss: HashMap<String, String>,
    anchor_negative_gloss: HashMap<String, String>,
}

fn cnf_groups_match(haystack: &str, groups: &[Vec<String>]) -> bool {
    !groups.is_empty() && groups.iter().all(|or_alts| or_alts.iter().any(|p| haystack.contains(p)))
}

fn anchor_gloss_map_from_rows(rows: &[AnchorTokenGlossRow]) -> HashMap<String, String> {
    rows.iter()
        .filter(|r| !r.token.is_empty())
        .map(|r| (r.token.to_ascii_lowercase(), r.gloss.clone()))
        .collect()
}

/// Build an Aho–Corasick matcher from non-empty patterns; `None` if there is nothing to match.
fn ac_from_strings(patterns: &[String]) -> Option<AhoCorasick> {
    let ps: Vec<&str> = patterns
        .iter()
        .map(|s| s.as_str())
        .filter(|p| !p.is_empty())
        .collect();
    if ps.is_empty() {
        return None;
    }
    AhoCorasick::new(ps).ok()
}

/// Longest phrase wins across all `lexical_polarity` rows (same semantics as the previous nested loop).
#[derive(Debug, Clone)]
struct LexicalPolarityAutomaton {
    ac: AhoCorasick,
    /// Per pattern index: topic key and phrase length (for tie-breaking by longest substring).
    pattern_topic: Vec<String>,
    pattern_len: Vec<usize>,
}

fn build_lexical_polarity_automaton(
    lexical_polarity: &[(String, Vec<String>)],
) -> Option<LexicalPolarityAutomaton> {
    let mut pats: Vec<String> = Vec::new();
    let mut pattern_topic: Vec<String> = Vec::new();
    let mut pattern_len: Vec<usize> = Vec::new();
    for (topic, phrases) in lexical_polarity {
        for phrase in phrases {
            if phrase.is_empty() {
                continue;
            }
            pattern_len.push(phrase.len());
            pattern_topic.push(topic.clone());
            pats.push(phrase.clone());
        }
    }
    let ps: Vec<&str> = pats.iter().map(|s| s.as_str()).collect();
    let ac = AhoCorasick::new(ps).ok()?;
    Some(LexicalPolarityAutomaton {
        ac,
        pattern_topic,
        pattern_len,
    })
}

#[inline]
fn ac_or_vec_contains(ac: &Option<AhoCorasick>, phrases: &[String], lower: &str) -> bool {
    if let Some(a) = ac {
        a.is_match(lower)
    } else {
        phrases.iter().any(|p| lower.contains(p.as_str()))
    }
}

fn pr_wire_prefix_rule_matches(t: &str, rule: &PrWireNeutralPrefixRule) -> bool {
    if !t.starts_with(rule.prefix.as_str()) {
        return false;
    }
    if t.len() < rule.min_len {
        return false;
    }
    if rule.disallow_question_mark && t.contains('?') {
        return false;
    }
    if rule
        .exclude_prefixes
        .iter()
        .any(|e:&String| t.starts_with(e.as_str()))
    {
        return false;
    }
    if !rule.require_any.is_empty()
        && !rule
            .require_any
            .iter()
            .any(|s:&String| t.contains(s.as_str()))
    {
        return false;
    }
    true
}

impl LatticeMisfireRule {
    fn response_side_matches(&self, response_lower: &str) -> bool {
        let any_hit = !self.response_any.is_empty()
            && self
                .response_any
                .iter()
                .any(|p| response_lower.contains(p.as_str()));
        let cnf_hit = cnf_groups_match(response_lower, &self.response);
        match (
            self.response_any.is_empty(),
            self.response.is_empty(),
        ) {
            (true, true) => false,
            (false, true) => any_hit,
            (true, false) => cnf_hit,
            (false, false) => any_hit || cnf_hit,
        }
    }
}

impl InferenceRulesRuntime {
    fn from_section(s: InferenceRulesSection) -> Self {
        let lexical_polarity: Vec<(String, Vec<String>)> = s
            .lexical_polarity
            .iter()
            .map(|r| (r.topic.clone(), r.phrases.clone()))
            .collect();
        let mut negative_lexical_suppress_phrases: Vec<String> = Vec::new();
        for (topic, phrases) in &lexical_polarity {
            if topic == "negative_mild" || topic == "negative_strong" {
                negative_lexical_suppress_phrases.extend(phrases.iter().cloned());
            }
        }
        let contrastive_markers_ac = ac_from_strings(&s.contrastive_markers);
        let lexical_polarity_automaton = build_lexical_polarity_automaton(&lexical_polarity);
        let negative_lexical_suppress_ac = ac_from_strings(&negative_lexical_suppress_phrases);
        let sarcasm_simple_ac = ac_from_strings(&s.sarcasm_simple);
        let ambiguous_disappointment_ac = ac_from_strings(&s.ambiguous_disappointment_phrases);
        let ambiguous_neutral_hedge_ac = ac_from_strings(&s.ambiguous_neutral_hedge_phrases);
        let ambiguous_lukewarm_okay_primary_ac = ac_from_strings(&s.ambiguous_lukewarm_okay_primary_phrases);
        let ambiguous_lukewarm_okay_supplement_ac =
            ac_from_strings(&s.ambiguous_lukewarm_okay_supplement_phrases);
        let ambiguous_fine_meh_primary_ac = ac_from_strings(&s.ambiguous_fine_meh_primary_phrases);
        let ambiguous_fine_meh_supplement_ac = ac_from_strings(&s.ambiguous_fine_meh_supplement_phrases);
        let like_intensity_exception_ac = ac_from_strings(&s.like_intensity_exception_phrases);
        let mixed_silver_lining_bad_ac = ac_from_strings(&s.mixed_silver_lining_bad_phrases);
        let mixed_silver_lining_good_ac = ac_from_strings(&s.mixed_silver_lining_good_phrases);
        let mixed_fraud_relief_anchor_ac = ac_from_strings(&s.mixed_fraud_relief_anchor_phrases);
        let mixed_fraud_relief_trigger_ac = ac_from_strings(&s.mixed_fraud_relief_trigger_any);
        let mixed_implicit_followon_ac = ac_from_strings(&s.mixed_implicit_followon_phrases);
        let mixed_implicit_unusual_context_ac = ac_from_strings(&s.mixed_implicit_unusual_context_any);
        let mixed_implicit_unusual_exclude_ac = ac_from_strings(&s.mixed_implicit_unusual_exclude_any);
        let mixed_positive_outcome_ac = ac_from_strings(&s.mixed_positive_outcome_phrases);
        let mixed_skepticism_friction_ac = ac_from_strings(&s.mixed_skepticism_friction_phrases);
        let mixed_operational_approve_ac = ac_from_strings(&s.mixed_operational_approve_phrases);
        Self {
            contrastive_markers: s.contrastive_markers,
            contrastive_markers_ac,
            lexical_polarity,
            lexical_polarity_automaton,
            negative_lexical_suppress_phrases,
            negative_lexical_suppress_ac,
            sarcasm_simple: s.sarcasm_simple,
            sarcasm_simple_ac,
            sarcasm_and: s.sarcasm_and.into_iter().map(|r| r.groups).collect(),
            positive_anchor_tokens: s.positive_anchor_tokens,
            negative_anchor_tokens: s.negative_anchor_tokens,
            bipolar_positive_tokens: s.bipolar_positive_tokens,
            bipolar_negative_tokens: s.bipolar_negative_tokens,
            ambiguous_disappointment_phrases: s.ambiguous_disappointment_phrases,
            ambiguous_disappointment_ac,
            ambiguous_neutral_hedge_phrases: s.ambiguous_neutral_hedge_phrases,
            ambiguous_neutral_hedge_ac,
            evaluative_words: s.evaluative_words,
            disappointment_words: s.disappointment_words,
            objective_fact_rules: s.objective_fact_rules,
            headline_lexical_topic: s.headline_lexical_topic,
            lattice_misfire: s.lattice_misfire,
            pr_wire_neutral_prefix: s.pr_wire_neutral_prefix,
            pr_wire_neutral_intent: s.pr_wire_neutral_intent,
            crypto_market_surface_tokens: s.crypto_market_surface_tokens,
            crypto_market_surface_prefixes: s.crypto_market_surface_prefixes,
            crypto_market_tape_tokens: s.crypto_market_tape_tokens,
            like_perception_verbs: s.like_perception_verbs,
            like_intensity_exception_phrases: s.like_intensity_exception_phrases,
            like_intensity_exception_ac,
            strong_positive_tokens: s.strong_positive_tokens,
            pr_wire_press_min_trim_len: s.pr_wire_press_min_trim_len.unwrap_or(32),
            mixed_plausible_headline_topics: s.mixed_plausible_headline_topics,
            mixed_silver_lining_bad_phrases: s.mixed_silver_lining_bad_phrases,
            mixed_silver_lining_bad_ac,
            mixed_silver_lining_good_phrases: s.mixed_silver_lining_good_phrases,
            mixed_silver_lining_good_ac,
            mixed_fraud_relief_anchor_phrases: s.mixed_fraud_relief_anchor_phrases,
            mixed_fraud_relief_anchor_ac,
            mixed_fraud_relief_trigger_any: s.mixed_fraud_relief_trigger_any,
            mixed_fraud_relief_trigger_ac,
            mixed_implicit_followon_phrases: s.mixed_implicit_followon_phrases,
            mixed_implicit_followon_ac,
            mixed_implicit_unusual_token: if s.mixed_implicit_unusual_token.is_empty() {
                "unusual".to_string()
            } else {
                s.mixed_implicit_unusual_token
            },
            mixed_implicit_unusual_context_any: s.mixed_implicit_unusual_context_any,
            mixed_implicit_unusual_context_ac,
            mixed_implicit_unusual_exclude_any: s.mixed_implicit_unusual_exclude_any,
            mixed_implicit_unusual_exclude_ac,
            mixed_positive_outcome_phrases: s.mixed_positive_outcome_phrases,
            mixed_positive_outcome_ac,
            mixed_skepticism_friction_phrases: s.mixed_skepticism_friction_phrases,
            mixed_skepticism_friction_ac,
            mixed_operational_decline_phrases: s.mixed_operational_decline_phrases,
            mixed_operational_approve_phrases: s.mixed_operational_approve_phrases,
            mixed_operational_approve_ac,
            ambiguous_lukewarm_okay_primary_phrases: s.ambiguous_lukewarm_okay_primary_phrases,
            ambiguous_lukewarm_okay_primary_ac,
            ambiguous_lukewarm_okay_supplement_phrases: s.ambiguous_lukewarm_okay_supplement_phrases,
            ambiguous_lukewarm_okay_supplement_ac,
            ambiguous_fine_meh_primary_phrases: s.ambiguous_fine_meh_primary_phrases,
            ambiguous_fine_meh_primary_ac,
            ambiguous_fine_meh_supplement_phrases: s.ambiguous_fine_meh_supplement_phrases,
            ambiguous_fine_meh_supplement_ac,
            ambiguous_neutral_conjunction_groups: s.ambiguous_neutral_conjunction_groups,
            anchor_positive_gloss: anchor_gloss_map_from_rows(&s.anchor_positive_gloss),
            anchor_negative_gloss: anchor_gloss_map_from_rows(&s.anchor_negative_gloss),
        }
    }

    fn headline_rule_matches(&self, rule: &HeadlineLexicalTopicRule, lower: &str) -> bool {
        if let Some(min) = rule.min_trim_len {
            if lower.trim().len() < min {
                return false;
            }
        }
        if rule.require_question_mark && !lower.contains('?') {
            return false;
        }
        if rule.exclude_first_person && Self::looks_like_first_person_finance_user(lower) {
            return false;
        }
        if rule.require_crypto_lexicon && !self.has_crypto_or_broad_market_lexicon(lower) {
            return false;
        }
        if rule.requires_mixed_positive_outcome_cue && !self.has_mixed_positive_outcome_cue(lower) {
            return false;
        }
        if rule
            .unless_any_cnf
            .iter()
            .any(|cnf| cnf_groups_match(lower, cnf))
        {
            return false;
        }
        cnf_groups_match(lower, &rule.intent)
    }

    fn scan_headline_lexical_topic(&self, lower: &str, after_pr_wire: bool) -> Option<String> {
        for rule in &self.headline_lexical_topic {
            if rule.after_pr_wire != after_pr_wire {
                continue;
            }
            if self.headline_rule_matches(rule, lower) {
                return Some(rule.topic.clone());
            }
        }
        None
    }

    pub fn has_contrastive_marker(&self, lower: &str) -> bool {
        ac_or_vec_contains(
            &self.contrastive_markers_ac,
            &self.contrastive_markers,
            lower,
        )
    }

    /// Open-finance / inclusion headline: redirect to sentiment lattice + knowledge-floor bypass.
    pub fn sentiment_inclusion_open_finance_headline_positive_raw(&self, intent_text: &str) -> bool {
        let lower = Self::normalize_for_rules(intent_text);
        let l = lower.as_str();
        self.headline_lexical_topic.iter().any(|r| {
            r.inclusion_redirect && self.headline_rule_matches(r, l)
        })
    }

    /// Headlines where low retrieval confidence should not force an honest decline (short, high-signal pack).
    pub fn sentiment_retrieval_confidence_floor_bypass(&self, intent_text: &str) -> bool {
        if self.sentiment_inclusion_open_finance_headline_positive_raw(intent_text) {
            return true;
        }
        let lower = Self::normalize_for_rules(intent_text);
        let s = lower.as_str();
        if s.contains("criticizing")
            && s.contains("trump")
            && s.contains("crypto")
            && (s.contains("investor") || s.contains("prominent"))
        {
            return true;
        }
        if s.contains("kraken") && s.contains("ipo") {
            return true;
        }
        false
    }

    pub fn lexical_polarity_signal(&self, lower: &str) -> Option<String> {
        if let Some(ref lp) = self.lexical_polarity_automaton {
            let mut best: Option<(usize, String)> = None;
            for m in lp.ac.find_overlapping_iter(lower) {
                let i = m.pattern().as_usize();
                let len = lp.pattern_len[i];
                let topic = lp.pattern_topic[i].clone();
                if best.as_ref().map_or(true, |(bl, _)| len > *bl) {
                    best = Some((len, topic));
                }
            }
            return best.map(|(_, t)| t);
        }
        let mut best: Option<(usize, String)> = None;
        for (topic, phrases) in &self.lexical_polarity {
            for p in phrases {
                if lower.contains(p.as_str()) {
                    let n = p.len();
                    if best.as_ref().map_or(true, |(best_len, _)| n > *best_len) {
                        best = Some((n, topic.clone()));
                    }
                }
            }
        }
        best.map(|(_, t)| t)
    }

    /// Lowercase + curly apostrophe normalization (keep aligned with `lattice_shortcuts`).
    fn normalize_for_rules(text: &str) -> String {
        let mut s = text.to_lowercase();
        s = s.replace('\u{2019}', "'");
        s = s.replace('\u{2018}', "'");
        s
    }

    /// Sub-lattice topic key from inference TOML only (no MetaBrain).
    ///
    /// Brains with expanded topic taxonomies fail [`crate::inference::plugins::lattice_shortcuts::is_lattice_shape`],
    /// so **user-anchored lattice preempt** does not run — but Layer‑0 keyword expansion and embedding
    /// routing still apply. Embedding routing can still follow domain
    /// words (e.g. “fee”) into a negative sub-lattice while the user clearly praises (“I love …”).
    /// When this returns [`Some`], callers may override `topic_hint` before retrieval.
    ///
    /// ## Where the English should live (training vs grounding vs data)
    ///
    /// - **Policy routing** (“if these phrases, bucket `mixed` / `neutral` / …”) belongs in **inference data**:
    ///   `[[rules.headline_lexical_topic]]`, PR-wire tables, and optional **guardrails JSONL** — not in
    ///   [`crate::inference::world_grounding`]. Grounding is for **graph nodes, aliases, and retrieval
    ///   expansion**, not for choosing among the seven sentiment topic keys.
    /// - **Shared lexicon** (ticker names, “crypto market” vocabulary) can be **sourced from grounding
    ///   aliases** in code later (e.g. replace [`Self::has_crypto_or_broad_market_lexicon`] with
    ///   activation over grounded tokens), while the **outcome** (which topic key) still lives in inference TOML/JSONL.
    /// - **Training** can eventually shrink hand-built lists when you have **labeled** intent→topic pairs
    ///   for the same buckets; until then, data files stay the auditable source of truth.
    ///
    /// Phrase-level routing uses `[[rules.headline_lexical_topic]]` with optional `after_pr_wire`,
    /// `exclude_first_person`, `unless_any_cnf`, etc. (see [`HeadlineLexicalTopicRule`]).
    pub fn sentiment_lexical_topic_key(&self, intent_text: &str) -> Option<String> {
        let lower = Self::normalize_for_rules(intent_text);
        let lower = lower.as_str();

        if let Some(t) = self.scan_headline_lexical_topic(lower, false) {
            return Some(t);
        }

        // PR / wire headlines (no I/we) — not operator tape; neutral lattice beats random sentiment retrieval.
        if let Some(k) = self.sentiment_fintech_press_headline_topic_key(lower) {
            return Some(k);
        }

        // Tape / third-party crypto copy / cautiously_positive rows with `after_pr_wire = true` in TOML/JSONL.
        if let Some(t) = self.scan_headline_lexical_topic(lower, true) {
            return Some(t);
        }

        let contrast = self.has_contrastive_marker(lower);
        let bipolar = contrast && self.has_bipolar_lexicon(lower);
        if contrast && bipolar {
            return Some("mixed".to_string());
        }

        // Friction + silver lining ("failed again, but at least support…")
        if contrast && self.has_mixed_silver_lining_pattern(lower) {
            return Some("mixed".to_string());
        }

        // Security friction + explicit relief ("flagged, but I'm glad…")
        if contrast && self.has_mixed_fraud_flag_relief_pattern(lower) {
            return Some("mixed".to_string());
        }

        // Good operational outcome + skepticism / friction (contrastive): avoid collapsing to one pole.
        if contrast
            && self.has_mixed_positive_outcome_cue(lower)
            && self.has_mixed_skepticism_or_friction_cue(lower)
        {
            return Some("mixed".to_string());
        }

        // Praise / relief in one sentence, hedged skepticism in the next (no "but").
        if self.has_mixed_positive_outcome_cue(lower) && self.has_mixed_implicit_followon_skeptic(lower) {
            return Some("mixed".to_string());
        }

        // Inconsistent terminal state (e.g. declined → approved → declined) without explicit "but".
        if self.has_operational_inconsistency_mixed_signal(lower) {
            return Some("mixed".to_string());
        }

        if let Some(k) = self.lexical_polarity_signal(lower) {
            return Some(k);
        }

        if self.has_sarcasm_template(lower) {
            return Some("sarcastic".to_string());
        }

        let pos_probe = "positive_mild";
        if let Some(k) = self.ambiguous_valence_retarget(lower, pos_probe) {
            return Some(k.to_string());
        }
        if let Some(k) = self.disappointment_positive_override(lower, pos_probe) {
            return Some(k.to_string());
        }

        if self.negated_eval_phrase_hit(lower) {
            return None;
        }

        let tokens: std::collections::HashSet<&str> = lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .collect();

        for w in &self.strong_positive_tokens {
            if tokens.contains(w.as_str()) {
                return Some("positive_strong".to_string());
            }
        }
        for w in &self.positive_anchor_tokens {
            if self.strong_positive_tokens.iter().any(|s| s == w) {
                continue;
            }
            if w == "like" {
                if self.like_token_is_perception_idiom(lower) {
                    continue;
                }
                if ac_or_vec_contains(
                    &self.like_intensity_exception_ac,
                    &self.like_intensity_exception_phrases,
                    lower,
                ) {
                    continue;
                }
            }
            if tokens.contains(w.as_str()) {
                return Some("positive_mild".to_string());
            }
        }

        None
    }

    /// Whether a **forced** `mixed` sub-lattice is linguistically plausible.
    ///
    /// MetaBrain / embeddings often match the prefix of a long contrastive training line (e.g. “X is slick,
    /// but I hate …”) and surface `mixed` even when the user only typed the positive clause. Requiring
    /// either a contrastive marker or both polarities in the bipolar lexicon avoids enumerating every
    /// praise adjective in TOML.
    pub fn sentiment_allow_forced_mixed_topic(&self, intent_text: &str) -> bool {
        let lower = Self::normalize_for_rules(intent_text);
        let l = lower.as_str();
        self.has_contrastive_marker(l)
            || self.has_bipolar_lexicon(l)
            // Oscillating auth/charge state without "but" — still a legitimate mixed bucket.
            || self.has_operational_inconsistency_mixed_signal(l)
            || self.mixed_plausible_headline_topics.iter().any(|topic| {
                self.headline_lexical_topic.iter().any(|r| {
                    r.topic == *topic && self.headline_rule_matches(r, l)
                })
            })
    }

    /// True when a composed line matches any `[[rules.lattice_misfire]]` row (intent ∧ response side).
    pub fn lattice_response_misfire_hit(&self, intent_text: &str, response: &str) -> bool {
        let il = Self::normalize_for_rules(intent_text);
        let l = il.as_str();
        let rl = response.to_ascii_lowercase();
        let rls = rl.as_str();
        self.lattice_misfire.iter().any(|rule| {
            cnf_groups_match(l, &rule.intent) && rule.response_side_matches(rls)
        })
    }

    /// After a lattice misfire strips a bad retrieved line, substitute a short canned witness when
    /// the intent shape is unambiguous (avoids routing-only for Kraken IPO vs M&A-stake collisions).
    pub fn sentiment_lattice_misfire_replacement_line(&self, intent_text: &str) -> Option<String> {
        let lower = Self::normalize_for_rules(intent_text);
        let l = lower.as_str();
        if l.contains("kraken") && l.contains("ipo") {
            return Some(
                "NEUTRAL — Listing / IPO pipeline wire; factual capital-markets news, not enforcement drama."
                    .to_string(),
            );
        }
        None
    }

    fn looks_like_first_person_finance_user(lower: &str) -> bool {
        let s = format!(
            " {} ",
            lower
                .replace('\u{2019}', "'")
                .replace('\u{2018}', "'")
                .chars()
                .map(|c| if c.is_alphanumeric() || c == '\'' {
                    c
                } else {
                    ' '
                })
                .collect::<String>()
        );
        s.contains(" my ")
            || s.contains(" i ")
            || s.contains(" i'm ")
            || s.contains(" im ")
            || s.contains(" i've ")
            || s.contains(" ive ")
            || s.contains(" i'd ")
            || s.contains(" id ")
            || s.contains(" our ")
            || s.contains(" we ")
            || s.contains(" me ")
            || lower.trim_start().starts_with("i ")
            || lower.trim_start().starts_with("i'm")
    }

    /// True when intent reads as direct consumer speech (my account, I want, we need).
    /// Third-party wire headlines return false so headline TOML can route to the sentiment lattice.
    pub fn looks_like_first_person_finance_intent(&self, intent_text: &str) -> bool {
        let lower = Self::normalize_for_rules(intent_text).to_lowercase();
        Self::looks_like_first_person_finance_user(lower.as_str())
    }

    /// `like` after perception verbs (“feels like”, “looks like a trap”) — not evaluative praise.
    fn like_token_is_perception_idiom(&self, lower: &str) -> bool {
        let tokens: Vec<&str> = lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .collect();
        for window in tokens.windows(2) {
            if window[1] == "like"
                && self
                    .like_perception_verbs
                    .iter()
                    .any(|v| v == window[0])
            {
                return true;
            }
        }
        false
    }

    /// PR / wire-style headlines (no first-person). Maps to `neutral` for seven-topic sentiment brains.
    /// Phrase lists: `[[rules.pr_wire_neutral_prefix]]` and `[[rules.pr_wire_neutral_intent]]` in inference TOML.
    fn sentiment_fintech_press_headline_topic_key(&self, lower: &str) -> Option<String> {
        if Self::looks_like_first_person_finance_user(lower) {
            return None;
        }
        let t = lower.trim();
        if t.len() < self.pr_wire_press_min_trim_len {
            return None;
        }
        for rule in &self.pr_wire_neutral_prefix {
            if pr_wire_prefix_rule_matches(t, rule) {
                return Some("neutral".to_string());
            }
        }
        for row in &self.pr_wire_neutral_intent {
            if let Some(min) = row.min_len {
                if t.len() < min {
                    continue;
                }
            }
            if cnf_groups_match(t, &row.intent) {
                return Some("neutral".to_string());
            }
        }
        None
    }

    /// Detects PR / wire copy that should use a neutral sentiment bucket (actual topic name may be `neutral_chop`, resolved in `LanguageService`).
    pub fn sentiment_pr_wire_neutral_key(&self, intent_text: &str) -> Option<String> {
        let lower = Self::normalize_for_rules(intent_text);
        self.sentiment_fintech_press_headline_topic_key(lower.as_str())
    }

    fn has_crypto_or_broad_market_lexicon(&self, lower: &str) -> bool {
        self.crypto_market_surface_tokens
            .iter()
            .any(|t| lower.contains(t.as_str()))
            || self
                .crypto_market_surface_prefixes
                .iter()
                .any(|p| lower.starts_with(p.as_str()))
    }

    /// Superset of surface lexicon for **retrieval gating** (tape / derivatives tokens from TOML).
    pub fn intent_text_suggests_crypto_market_impl(&self, lower: &str) -> bool {
        self.has_crypto_or_broad_market_lexicon(lower)
            || self
                .crypto_market_tape_tokens
                .iter()
                .any(|t| lower.contains(t.as_str()))
    }

    /// Delegates to the loaded inference rules (call after `inference_toml_loaded()` in hosts/tests).
    pub fn intent_text_suggests_crypto_market(lower: &str) -> bool {
        inference_rules_runtime().intent_text_suggests_crypto_market_impl(lower)
    }

    fn has_mixed_silver_lining_pattern(&self, lower: &str) -> bool {
        ac_or_vec_contains(
            &self.mixed_silver_lining_bad_ac,
            &self.mixed_silver_lining_bad_phrases,
            lower,
        ) && ac_or_vec_contains(
            &self.mixed_silver_lining_good_ac,
            &self.mixed_silver_lining_good_phrases,
            lower,
        )
    }

    fn has_mixed_fraud_flag_relief_pattern(&self, lower: &str) -> bool {
        ac_or_vec_contains(
            &self.mixed_fraud_relief_anchor_ac,
            &self.mixed_fraud_relief_anchor_phrases,
            lower,
        ) && ac_or_vec_contains(
            &self.mixed_fraud_relief_trigger_ac,
            &self.mixed_fraud_relief_trigger_any,
            lower,
        )
    }

    fn has_mixed_implicit_followon_skeptic(&self, lower: &str) -> bool {
        // "for once" / "not used to" + deposit → [`cautiously_positive`] (handled above).
        if ac_or_vec_contains(
            &self.mixed_implicit_followon_ac,
            &self.mixed_implicit_followon_phrases,
            lower,
        ) {
            return true;
        }
        let u = self.mixed_implicit_unusual_token.as_str();
        if !u.is_empty()
            && lower.contains(u)
            && ac_or_vec_contains(
                &self.mixed_implicit_unusual_context_ac,
                &self.mixed_implicit_unusual_context_any,
                lower,
            )
        {
            return !ac_or_vec_contains(
                &self.mixed_implicit_unusual_exclude_ac,
                &self.mixed_implicit_unusual_exclude_any,
                lower,
            );
        }
        false
    }

    /// Money-movement / approval cues that often co-occur with skepticism in the second clause.
    fn has_mixed_positive_outcome_cue(&self, lower: &str) -> bool {
        ac_or_vec_contains(
            &self.mixed_positive_outcome_ac,
            &self.mixed_positive_outcome_phrases,
            lower,
        )
    }

    fn has_mixed_skepticism_or_friction_cue(&self, lower: &str) -> bool {
        ac_or_vec_contains(
            &self.mixed_skepticism_friction_ac,
            &self.mixed_skepticism_friction_phrases,
            lower,
        )
    }

    /// Declined/approved oscillation or multiple declines — not a single-pole negative_mild story.
    fn has_operational_inconsistency_mixed_signal(&self, lower: &str) -> bool {
        let declined: usize = self
            .mixed_operational_decline_phrases
            .iter()
            .map(|p| lower.matches(p.as_str()).count())
            .sum();
        let approved = ac_or_vec_contains(
            &self.mixed_operational_approve_ac,
            &self.mixed_operational_approve_phrases,
            lower,
        );
        (declined >= 2) || (declined >= 1 && approved)
    }

    /// Substring match against phrases from `lexical_polarity` `negative_mild` / `negative_strong` rows.
    #[inline]
    pub fn negated_eval_phrase_hit(&self, lower: &str) -> bool {
        ac_or_vec_contains(
            &self.negative_lexical_suppress_ac,
            &self.negative_lexical_suppress_phrases,
            lower,
        )
    }

    pub fn has_sarcasm_template(&self, lower: &str) -> bool {
        if ac_or_vec_contains(&self.sarcasm_simple_ac, &self.sarcasm_simple, lower) {
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
        if ac_or_vec_contains(
            &self.ambiguous_disappointment_ac,
            &self.ambiguous_disappointment_phrases,
            lower,
        ) {
            return Some("negative_mild");
        }
        let hedge = ac_or_vec_contains(
            &self.ambiguous_neutral_hedge_ac,
            &self.ambiguous_neutral_hedge_phrases,
            lower,
        );
        let lukewarm_okay = ac_or_vec_contains(
            &self.ambiguous_lukewarm_okay_primary_ac,
            &self.ambiguous_lukewarm_okay_primary_phrases,
            lower,
        ) && (hedge
            || ac_or_vec_contains(
                &self.ambiguous_lukewarm_okay_supplement_ac,
                &self.ambiguous_lukewarm_okay_supplement_phrases,
                lower,
            ));
        let okay_suppose = self.ambiguous_neutral_conjunction_groups.iter().any(|group| {
            !group.is_empty() && group.iter().all(|p| lower.contains(p.as_str()))
        });
        let fine_meh = ac_or_vec_contains(
            &self.ambiguous_fine_meh_primary_ac,
            &self.ambiguous_fine_meh_primary_phrases,
            lower,
        ) && (hedge
            || ac_or_vec_contains(
                &self.ambiguous_fine_meh_supplement_ac,
                &self.ambiguous_fine_meh_supplement_phrases,
                lower,
            ));
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
                // Avoid “Anchored by 'like'” when the line is “don’t like …” / “wouldn’t use …”
                // (tokenizer splits “don’t” → don + t; “like” still matches as its own token).
                if self.negated_eval_phrase_hit(lower) {
                    return None;
                }
                for w in &self.positive_anchor_tokens {
                    if tokens.contains(w.as_str()) {
                        let wl = w.to_ascii_lowercase();
                        let gloss = self
                            .anchor_positive_gloss
                            .get(&wl)
                            .map(|s| s.as_str())
                            .unwrap_or("positive valence");
                        return Some(format!("Anchored by '{}' ({})", w, gloss));
                    }
                }
                None
            }
            "negative_strong" | "negative_mild" => {
                for w in &self.negative_anchor_tokens {
                    if tokens.contains(w.as_str()) {
                        let wl = w.to_ascii_lowercase();
                        let gloss = self
                            .anchor_negative_gloss
                            .get(&wl)
                            .map(|s| s.as_str())
                            .unwrap_or("negative valence");
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
    let rels = inference_toml_rel_search_order();
    let mut v = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        for rel in &rels {
            v.push(cwd.join(rel));
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            for rel in &rels {
                v.push(dir.join(rel));
                v.push(dir.join("../").join(rel));
            }
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

/// Merge `[rules]` from every readable candidate path in order (`merge_empty_from` each step).
/// Later packs (e.g. `data/fintech/inference_fintech.toml`) fill slices still empty after core.
#[cfg(not(target_arch = "wasm32"))]
fn accumulate_rules_from_candidate_paths() -> InferenceRulesSection {
    let mut acc = InferenceRulesSection::default();
    for p in candidate_paths() {
        let Ok(s) = std::fs::read_to_string(&p) else {
            continue;
        };
        let doc: InferenceTomlDocument = match toml::from_str(&s) {
            Ok(d) => d,
            Err(e) => {
                eprintln!(
                    "[inference-toml] skip unreadable/invalid {}: {}",
                    p.display(),
                    e
                );
                continue;
            }
        };
        acc = acc.merge_empty_from(&doc.rules);
    }
    acc
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
    accumulate_rules_from_candidate_paths()
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
            "[inference-toml] no inference TOML found. Use --inference-toml or --project *.gf.toml, set GROWFORMER_INFERENCE_TOML, set GROWFORMER_INFERENCE_TOML_DEFAULT_RELS, or place one of {:?} under cwd / next to the binary (tried: {:?})",
            inference_toml_rel_search_order(),
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
    const WASM_EMBED: &str = include_str!("../../data/sentiment/inference_sentiment_core.toml");
    toml::from_str(WASM_EMBED).expect("wasm embedded inference TOML invalid")
}

/// Single load: thresholds from file + merged rule lists.
#[derive(Debug)]
pub struct LoadedInferenceToml {
    pub thresholds: InferenceThresholds,
    rules: Arc<InferenceRulesRuntime>,
    /// JSONL guardrails merged after TOML (native only; empty on wasm).
    pub guardrails: super::inference_guardrails::GuardrailsDiskSummary,
}

impl LoadedInferenceToml {
    pub fn rules(&self) -> Arc<InferenceRulesRuntime> {
        self.rules.clone()
    }
}

/// Warm inference TOML + JSONL once and print a short summary to stdout (for `--train-brain` logs).
#[cfg(not(target_arch = "wasm32"))]
pub fn print_train_inference_disk_summary() {
    let loaded = inference_toml_loaded();
    let r = loaded.rules();
    println!(
        "  [inference] TOML + guardrails: {} headline_lexical_topic, {} lattice_misfire rows (runtime)",
        r.headline_lexical_topic.len(),
        r.lattice_misfire.len()
    );
    if loaded.guardrails.headline_rows_appended > 0 || loaded.guardrails.misfire_rows_appended > 0 {
        println!(
            "  [inference-guardrails] appended from JSONL: +{} headline, +{} misfire",
            loaded.guardrails.headline_rows_appended,
            loaded.guardrails.misfire_rows_appended
        );
    }
    for line in &loaded.guardrails.log_lines {
        println!("  [inference-guardrails] {}", line);
    }
}

#[cfg(target_arch = "wasm32")]
pub fn print_train_inference_disk_summary() {}

static FULL: OnceLock<Arc<LoadedInferenceToml>> = OnceLock::new();

pub fn inference_toml_loaded() -> Arc<LoadedInferenceToml> {
    FULL.get_or_init(|| {
        let file = load_inference_toml_merged();
        let mut rules = InferenceRulesRuntime::from_section(file.rules);
        #[cfg(not(target_arch = "wasm32"))]
        let guardrails = crate::inference::inference_guardrails::merge_guardrails_into_runtime(
            &mut rules.headline_lexical_topic,
            &mut rules.lattice_misfire,
        );
        #[cfg(target_arch = "wasm32")]
        let guardrails = crate::inference::inference_guardrails::GuardrailsDiskSummary::default();
        let rules = Arc::new(rules);
        Arc::new(LoadedInferenceToml {
            thresholds: file.thresholds,
            rules,
            guardrails,
        })
    })
    .clone()
}

pub fn inference_rules_runtime() -> Arc<InferenceRulesRuntime> {
    inference_toml_loaded().rules()
}

#[cfg(test)]
mod negation_tests {
    use super::*;

    #[test]
    fn intent_text_suggests_crypto_market_detects_tape_lexicon() {
        assert!(InferenceRulesRuntime::intent_text_suggests_crypto_market(
            "btc dominance is bleeding and funding is still positive"
        ));
        assert!(!InferenceRulesRuntime::intent_text_suggests_crypto_market(
            "my payment got flagged but at least they notified me instantly"
        ));
    }

    #[test]
    fn sentiment_headline_rescue_and_implicit_governance() {
        let raw = include_str!("../../data/sentiment/inference_sentiment_reference.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("fixture TOML");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let inc = "middle east leapfrogs to open finance to drive financial inclusion";
        assert!(rules.sentiment_inclusion_open_finance_headline_positive_raw(inc));
        assert_eq!(
            rules.sentiment_lexical_topic_key(inc).as_deref(),
            Some("positive_mild")
        );
        let macro_line =
            "open banking could unlock £43bn annually for the uk economy, new ey analysis reveals";
        assert_eq!(
            rules.sentiment_lexical_topic_key(macro_line).as_deref(),
            Some("positive_mild")
        );
        let gov = "corporate bitcoin adoption is growing. custody governance hasn't caught up";
        assert_eq!(rules.sentiment_lexical_topic_key(gov).as_deref(), Some("mixed"));
        assert!(rules.sentiment_allow_forced_mixed_topic(gov));
        let bleed_intent =
            "Midnight Network Goes Live as Privacy-Focused Blockchain Moves Into Mainnet Phase";
        let bleed_resp = "Data Privacy — Tighter filing … compliance burden for retail filers, not a tape read.";
        assert!(rules.lattice_response_misfire_hit(bleed_intent, bleed_resp));
        let bank_intent = "Monument and Midnight Bring Tokenised Deposits into UK Retail Banking";
        let bank_resp = "POSITIVE (mild) — Reliability appreciated. Positive sentiment from expectation being met.";
        assert!(rules.lattice_response_misfire_hit(bank_intent, bank_resp));
    }

    /// Default discovery merges core then fintech into empty `[rules]` slots; this mirrors that chain.
    #[test]
    fn merge_empty_from_chain_fills_fintech_headlines_after_core() {
        let core: InferenceTomlDocument = toml::from_str(include_str!(
            "../../data/sentiment/inference_sentiment_core.toml"
        ))
        .expect("core fixture");
        let fintech: InferenceTomlDocument = toml::from_str(include_str!(
            "../../data/fintech/inference_fintech.toml"
        ))
        .expect("fintech fixture");
        let mut acc = InferenceRulesSection::default();
        acc = acc.merge_empty_from(&core.rules);
        assert!(
            acc.headline_lexical_topic.is_empty(),
            "core pack should omit headline_lexical_topic"
        );
        acc = acc.merge_empty_from(&fintech.rules);
        assert!(
            !acc.headline_lexical_topic.is_empty(),
            "fintech pack should supply headline rows into empty slots"
        );
        let rules = InferenceRulesRuntime::from_section(acc);
        let sofi = "SoFi Technologies vs. Upstart: Which Fintech Stock Is the Better Long-Term Buy?";
        assert_eq!(
            rules.sentiment_lexical_topic_key(sofi).as_deref(),
            Some("neutral"),
            "vs-stock headline rule should apply after chained merge"
        );
    }

    #[test]
    fn fintech_headline_short_seller_enron_maps_negative_mild() {
        let raw = include_str!("../../data/fintech/inference_fintech.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("fintech fixture");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let h = "Short-seller Carson Block compares SoFi to Enron in a letter to the fintech firm's board";
        assert_eq!(
            rules.sentiment_lexical_topic_key(h).as_deref(),
            Some("negative_mild")
        );
    }

    #[test]
    fn fintech_headline_better_stock_vs_maps_neutral() {
        let raw = include_str!("../../data/fintech/inference_fintech.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("fintech fixture");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let h = "Better Fintech Stock for Growth Investors: Nu Holdings vs. SoFi";
        assert_eq!(rules.sentiment_lexical_topic_key(h).as_deref(), Some("neutral"));
    }

    #[test]
    fn fintech_headline_fca_open_finance_maps_neutral() {
        let raw = include_str!("../../data/fintech/inference_fintech.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("fintech fixture");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let h = "FCA Sets Out Vision for Open Finance to Empower Consumers and Businesses";
        assert_eq!(rules.sentiment_lexical_topic_key(h).as_deref(), Some("neutral"));
    }

    #[test]
    fn fintech_headline_kraken_ipo_wire_maps_neutral() {
        let raw = include_str!("../../data/fintech/inference_fintech.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("fintech fixture");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let h = "Crypto exchange Kraken confirms it has confidentially filed for an IPO";
        assert_eq!(rules.sentiment_lexical_topic_key(h).as_deref(), Some("neutral"));
    }

    #[test]
    fn fintech_headline_exchange_stake_maps_neutral() {
        let raw = include_str!("../../data/fintech/inference_fintech.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("fintech fixture");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let h = "Deutsche Börse Takes $200 Million Stake in Crypto Exchange Kraken";
        assert_eq!(rules.sentiment_lexical_topic_key(h).as_deref(), Some("neutral"));
    }

    #[test]
    fn fintech_headline_why_xrp_gaining_maps_neutral() {
        let raw = include_str!("../../data/fintech/inference_fintech.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("fintech fixture");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let h = "Why XRP Is Gaining Today";
        assert_eq!(rules.sentiment_lexical_topic_key(h).as_deref(), Some("neutral"));
    }

    #[test]
    fn lattice_misfire_kraken_ipo_vs_enforcement_witness() {
        let raw = include_str!("../../data/fintech/inference_fintech.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("fintech fixture");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let intent = "Crypto Exchange Kraken Prepares for IPO";
        let bad = "NEGATIVE (strong) — Landmark enforcement penalty signals severe compliance failure";
        assert!(rules.lattice_response_misfire_hit(intent, bad));
    }

    #[test]
    fn sentiment_lattice_misfire_replacement_kraken_ipo_line() {
        let raw = include_str!("../../data/fintech/inference_fintech.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("fintech fixture");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let s = rules
            .sentiment_lattice_misfire_replacement_line(
                "Crypto exchange Kraken confirms it has confidentially filed for an IPO",
            )
            .unwrap();
        assert!(s.contains("Listing / IPO pipeline"));
    }

    #[test]
    fn sentiment_retrieval_floor_bypass_trump_crypto_critic() {
        let raw = include_str!("../../data/fintech/inference_fintech.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("fintech fixture");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let h = "One of the most prominent investors in the Trump family's crypto company is now criticizing it";
        assert!(rules.sentiment_retrieval_confidence_floor_bypass(h));
    }

    #[test]
    fn fintech_headline_trump_investor_criticizing_maps_negative_mild() {
        let raw = include_str!("../../data/fintech/inference_fintech.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("fintech fixture");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let h = "One of the most prominent investors in the Trump family's crypto company is now criticizing it";
        assert_eq!(
            rules.sentiment_lexical_topic_key(h).as_deref(),
            Some("negative_mild")
        );
    }

    #[test]
    fn lattice_misfire_kraken_ipo_vs_stake_m_a_witness() {
        let raw = include_str!("../../data/fintech/inference_fintech.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("fintech fixture");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let intent = "Crypto exchange Kraken confirms it has confidentially filed for an IPO";
        let bad = "NEUTRAL — Strategic exchange stake / M&A headline; institutional markets story, not consumer praise.";
        assert!(rules.lattice_response_misfire_hit(intent, bad));
    }

    #[test]
    fn fintech_headline_superrich_resurrection_maps_neutral() {
        let raw = include_str!("../../data/fintech/inference_fintech.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("fintech fixture");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let h = "How A Credit Card Fintech Resurrected Itself By Targeting The Superrich";
        assert_eq!(rules.sentiment_lexical_topic_key(h).as_deref(), Some("neutral"));
    }

    #[test]
    fn fintech_headline_deplatforming_banned_maps_negative_strong() {
        let raw = include_str!("../../data/fintech/inference_fintech.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("fintech fixture");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let h = "AI Deplatforming Is The New Debanking As Claude Users Get Banned";
        assert_eq!(
            rules.sentiment_lexical_topic_key(h).as_deref(),
            Some("negative_strong")
        );
    }

    #[test]
    fn lattice_misfire_selective_investors_vs_growth_stat_witness() {
        let raw = include_str!("../../data/fintech/inference_fintech.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("fintech fixture");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let intent = "Fintech investors are getting more selective about AI";
        let bad_witness =
            "NEUTRAL — Macro growth statistic + interpretation; not first-person banking emotion.";
        assert!(rules.lattice_response_misfire_hit(intent, bad_witness));
        let ok_witness =
            "NEUTRAL — Investor selectivity / discipline headline; capital allocation caution.";
        assert!(!rules.lattice_response_misfire_hit(intent, ok_witness));
    }

    #[test]
    fn lexical_polarity_prefers_negation_over_positive_idiom() {
        let raw = include_str!("../../data/sentiment/inference_sentiment_reference.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("fixture TOML");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        assert_eq!(
            rules.lexical_polarity_signal("i don't like using google").as_deref(),
            Some("negative_mild")
        );
        assert_eq!(
            rules.lexical_polarity_signal("i wouldn't use google").as_deref(),
            Some("negative_mild")
        );
        assert_eq!(
            rules.lexical_polarity_signal("this blew me away").as_deref(),
            Some("positive_strong")
        );
        assert_eq!(
            rules
                .lexical_polarity_signal("i'm not happy with google search lately")
                .as_deref(),
            Some("negative_mild")
        );
    }

    #[test]
    fn anchor_suppresses_positive_tokens_when_negation_phrase_present() {
        let raw = include_str!("../../data/sentiment/inference_sentiment_reference.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("fixture TOML");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let lower = "i don't like using google";
        assert!(rules.negated_eval_phrase_hit(lower));
        assert!(rules.anchor_phrase(lower, "positive_mild").is_none());
    }

    #[test]
    fn lexical_polarity_negative_strong_and_litotes() {
        let raw = include_str!("../../data/sentiment/inference_sentiment_reference.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("fixture TOML");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        assert_eq!(
            rules
                .lexical_polarity_signal("this lender is a complete scam")
                .as_deref(),
            Some("negative_strong")
        );
        assert_eq!(
            rules
                .lexical_polarity_signal("the mobile app is not bad actually")
                .as_deref(),
            Some("positive_mild")
        );
        assert_eq!(
            rules
                .lexical_polarity_signal("support was slow but it could have been worse")
                .as_deref(),
            Some("positive_mild")
        );
    }

    #[test]
    fn sentiment_lexical_topic_key_loves_fee_transparency() {
        let raw = include_str!("../../data/sentiment/inference_sentiment_reference.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("fixture TOML");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        assert_eq!(
            rules.sentiment_lexical_topic_key("I love the fee transparency").as_deref(),
            Some("positive_strong")
        );
        assert_eq!(
            rules.sentiment_lexical_topic_key("I enjoy the clear fee breakdown").as_deref(),
            Some("positive_mild")
        );
        assert_eq!(
            rules.sentiment_lexical_topic_key("I love the fee transparency, but I hate the charges")
                .as_deref(),
            Some("mixed")
        );
        assert_eq!(
            rules.sentiment_lexical_topic_key("My deposit hit early").as_deref(),
            Some("positive_mild")
        );
        assert_eq!(
            rules
                .lexical_polarity_signal("my deposit hit early")
                .as_deref(),
            Some("positive_mild")
        );
        assert_eq!(
            rules.sentiment_lexical_topic_key("I got paid early").as_deref(),
            Some("positive_mild")
        );
        assert_eq!(
            rules.lexical_polarity_signal("i got paid early").as_deref(),
            Some("positive_mild")
        );
        // UX praise anchors + substring row beat mixed retrieval on a prefix of sent_fin_400.
        assert_eq!(
            rules
                .sentiment_lexical_topic_key("Instant card tokenization is slick")
                .as_deref(),
            Some("positive_mild")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "Instant card tokenization is slick, but I hate that fraud locks still hit my grocery runs."
                )
                .as_deref(),
            Some("mixed")
        );
        assert!(!rules.sentiment_allow_forced_mixed_topic(
            "Instant card tokenization is slick"
        ));
        assert!(rules.sentiment_allow_forced_mixed_topic(
            "Instant card tokenization is slick, but I hate the fraud locks"
        ));
        assert!(rules.sentiment_allow_forced_mixed_topic(
            "I love the UI and I hate the fees"
        ));
        // Contrast + good outcome + skepticism / friction → mixed (before single-pole lexical).
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "The transfer went through instantly today… which is weird because it never does."
                )
                .as_deref(),
            Some("mixed")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "My credit score jumped 18 points, but I still don't trust whatever system caused it."
                )
                .as_deref(),
            Some("mixed")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "The deposit hit early, but the balance didn't update, and then the app logged me out."
                )
                .as_deref(),
            Some("mixed")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "The card was declined, then approved, then declined again at the same store."
                )
                .as_deref(),
            Some("mixed")
        );
        // Startup accountability headlines (no I/we): steer away from false positive_strong on "even" / "hit".
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "The reputation of troubled YC startup Delve has gotten even worse"
                )
                .as_deref(),
            Some("negative_mild")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "Delve whistleblower strikes again, with alleged receipts about 'fake compliance'"
                )
                .as_deref(),
            Some("negative_strong")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "Mercor hit with 5 contractor lawsuits in a week over data breach"
                )
                .as_deref(),
            Some("negative_strong")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "Indian-origin founder breaks silence on allegations against startup Delve: 'We grew too fast'"
                )
                .as_deref(),
            Some("mixed")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "Glad the refund posted, though I still don't understand why the charge happened."
                )
                .as_deref(),
            Some("mixed")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "Nice that the transfer was instant today. Shame that it's usually a disaster."
                )
                .as_deref(),
            Some("mixed")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key("The app actually worked today. That's unusual.")
                .as_deref(),
            Some("cautiously_positive")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "My deposit hit early for once. I'm not used to this."
                )
                .as_deref(),
            Some("cautiously_positive")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "The payment finally cleared, but only after two days of errors. I'm relieved but annoyed."
                )
                .as_deref(),
            Some("mixed")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "The transfer failed again, but at least support responded quickly."
                )
                .as_deref(),
            Some("mixed")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "My card got flagged, but I'm glad the fraud system is paying attention."
                )
                .as_deref(),
            Some("mixed")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "My balance shows two different numbers. Not sure what to make of that."
                )
                .as_deref(),
            Some("confused")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "The app says the payment is complete, but the merchant says pending."
                )
                .as_deref(),
            Some("confused")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "Is it normal for my credit score to jump 18 points overnight?"
                )
                .as_deref(),
            Some("confused")
        );
        // Crypto tape / positioning (v2 fintech prompts): route before third-party headline path and `like` anchors.
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "BTC dominance is bleeding fast. Either altseason is real or we're about to nuke."
                )
                .as_deref(),
            Some("mixed")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "Liquidations are stacking on both sides. Feels like the calm before a big move."
                )
                .as_deref(),
            Some("mixed")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "Everyone's calling for a breakdown but funding is still positive. Mixed signals everywhere."
                )
                .as_deref(),
            Some("mixed")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "Volume is spiking but price isn't moving. Someone's absorbing like crazy."
                )
                .as_deref(),
            Some("cautiously_positive")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "BTC is coiling tighter than ever. This compression won't last long."
                )
                .as_deref(),
            Some("mixed")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "How Bolt's AI Pivot Showcases an Evolution in Fintech Hiring"
                )
                .as_deref(),
            Some("neutral")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key("Stripe's valuation soars 74% to $159 billion")
                .as_deref(),
            Some("neutral")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "Stripe alumni raise €30M Series A for Duna, backed by Stripe and Adyen execs"
                )
                .as_deref(),
            Some("neutral")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "Dormant wallets waking up after years… never a good sign."
                )
                .as_deref(),
            Some("negative_mild")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "Investigators recover $600K in stolen cryptocurrency, including some taken from CT resident"
                )
                .as_deref(),
            Some("cautiously_positive")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "Cryptocurrency, AI and Cyber Scams Cost U.S. Almost $21 Billion in 2025 According to New FBI Internet Crime Report"
                )
                .as_deref(),
            Some("negative_mild")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "MercadoLibre (MELI) to Discontinue Mercado Coin Cryptocurrency"
                )
                .as_deref(),
            Some("negative_mild")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "New law regulates cryptocurrency kiosks in Wisconsin to protect against scams"
                )
                .as_deref(),
            Some("cautiously_positive")
        );
        assert!(rules
            .sentiment_pr_wire_neutral_key("How do I dispute a charge on my statement")
            .is_none());
        assert_eq!(
            rules
                .sentiment_pr_wire_neutral_key(
                    "Why The 2026 Fintech Funding Boom Is About More Than AI"
                )
                .as_deref(),
            Some("neutral")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "Capital One completes acquisition of Brex, the fintech giant, in a $5.15 billion deal"
                )
                .as_deref(),
            Some("neutral")
        );
        assert!(rules.lattice_response_misfire_hit(
            "Monument and Midnight Bring Tokenised Deposits into UK Retail Banking",
            "POSITIVE causal Law enforcement recovery of stolen crypto with lack constructive enthusiasm in an otherwise harmful story",
        ));
        assert!(rules.lattice_response_misfire_hit(
            "Midnight Network Goes Live as Privacy-Focused Blockchain Moves Into Mainnet Phase",
            "Data Privacy — Direct — Law enforcement recovery print stolen crypto absorption lack constructive enthusiasm in an otherwise harmful story",
        ));
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "How StraitsX & KBank are Facilitating SE Asia's Tourism Boom"
                )
                .as_deref(),
            Some("neutral")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "Geopolitical drama reportedly stalls IPO of SoftBank-backed PayPay"
                )
                .as_deref(),
            Some("neutral")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "Alts are printing 20% candles out of nowhere. Feels like exit liquidity season."
                )
                .as_deref(),
            Some("cautiously_negative")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "Every pump gets sold instantly. Trend is clearly shifting."
                )
                .as_deref(),
            Some("negative_mild")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "People are euphoric over tiny green candles. That's usually the top signal."
                )
                .as_deref(),
            Some("cautiously_negative")
        );
        assert_eq!(
            rules.sentiment_lexical_topic_key("I really like the new dashboard").as_deref(),
            Some("positive_mild")
        );
        // Third-party crypto headlines / market copy (no first-person) → neutral or bearish, not consumer templates.
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "Bitcoin, XRP, And DOGE In Focus: Expert Points To Key Price Reversal In Crypto Market"
                )
                .as_deref(),
            Some("neutral")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "Dogecoin Stalls Inside The Kumo — Volatility Surge On The Horizon?"
                )
                .as_deref(),
            Some("neutral")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "Crypto markets are showing early signs that the worst may be over, following a prolonged decline that began with the industry's sharp sell-off back in October of last year."
                )
                .as_deref(),
            Some("neutral")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "XRP Holders Are Seeing Major Losses Since The Bull Market, And The Numbers Are Rising"
                )
                .as_deref(),
            Some("negative_mild")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "Cryptocurrency accounts seized in $2.3M money laundering scheme"
                )
                .as_deref(),
            Some("negative_mild")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "New rules for cryptocurrency investors who must now declare investments to HMRC"
                )
                .as_deref(),
            Some("negative_mild")
        );
        assert_eq!(
            rules
                .sentiment_lexical_topic_key(
                    "Where Will the Cryptocurrency XRP Be in 5 Years? | The Motley Fool"
                )
                .as_deref(),
            Some("neutral")
        );
        // First-person crypto stays off the headline → neutral shortcut (consumer utterance).
        assert_ne!(
            rules
                .sentiment_lexical_topic_key("My Bitcoin transfer is still pending on-chain.")
                .as_deref(),
            Some("neutral")
        );
    }
}

#[cfg(test)]
mod objective_fact_rules_tests {
    use super::*;

    #[test]
    fn objective_fact_storage_compact_dotted_comma() {
        let raw = include_str!("../../data/sentiment/inference_sentiment_reference.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("fixture TOML");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let l = |s: &str| s.to_lowercase();
        assert!(rules.is_objective_factual_statement(&l("spec sheet says 512gb ram")));
        assert!(rules.is_objective_factual_statement(&l("cap is 5 tb.")));
        assert!(rules.is_objective_factual_statement(&l("ordered 1tb, 512gb ssd")));
        assert!(!rules.is_objective_factual_statement(&l("I love this laptop")));
    }
}
