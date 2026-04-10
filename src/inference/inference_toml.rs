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
    /// Phrases from `lexical_polarity` rows with topic `negative_mild` / `negative_strong` — used to
    /// suppress misleading positive token anchors (e.g. token `like` inside “don’t like”).
    negative_lexical_suppress_phrases: Vec<String>,
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
        Self {
            contrastive_markers: s.contrastive_markers,
            lexical_polarity,
            negative_lexical_suppress_phrases,
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
    /// so user-anchored / lexical shortcuts never run — but embedding routing can still follow domain
    /// words (e.g. “fee”) into a negative sub-lattice while the user clearly praises (“I love …”).
    /// When this returns [`Some`], callers may override `topic_hint` before retrieval.
    pub fn sentiment_lexical_topic_key(&self, intent_text: &str) -> Option<String> {
        let lower = Self::normalize_for_rules(intent_text);
        let lower = lower.as_str();

        if self.sentiment_confused_lexical_topic_key(lower) {
            return Some("confused".to_string());
        }

        // Third-party crypto / macro market headlines & analysis — not first-person consumer sentiment.
        if let Some(k) = self.sentiment_third_party_crypto_market_copy_topic_key(lower) {
            return Some(k);
        }

        // Positive outcome + habitual distrust or backhanded surprise — not full MIXED (no competing event clause).
        // Topic must exist in the brain's `topic_subindex` or the service layer skips the override.
        if self.has_cautiously_positive_lexical_pattern(lower) {
            return Some("cautiously_positive".to_string());
        }

        let contrast = self.has_contrastive_marker(lower);
        let bipolar = self.has_bipolar_lexicon(lower);
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

        const STRONG_POS: &[&str] = &["love", "adore", "treasure", "obsessed"];
        for w in STRONG_POS {
            if tokens.contains(w) {
                return Some("positive_strong".to_string());
            }
        }
        for w in &self.positive_anchor_tokens {
            if STRONG_POS.contains(&w.as_str()) {
                continue;
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
            || self.sentiment_confused_lexical_topic_key(l)
            || self.has_cautiously_positive_lexical_pattern(l)
    }

    /// Deposit/timing wins + unfamiliarity, or “it worked” + surprise — hedged positivity, not bipolar MIXED.
    fn has_cautiously_positive_lexical_pattern(&self, lower: &str) -> bool {
        if self.has_mixed_positive_outcome_cue(lower)
            && lower.contains("deposit")
            && (lower.contains("for once") || lower.contains("not used to"))
        {
            return true;
        }
        lower.contains("actually worked") && lower.contains("unusual")
    }

    /// Informational / uncertainty lines — not a negative stance about a score drop.
    fn sentiment_confused_lexical_topic_key(&self, lower: &str) -> bool {
        let q = lower.contains('?');
        if q {
            let mentions_rise = lower.contains("jump")
                || lower.contains("jumps")
                || lower.contains("went up")
                || lower.contains("gone up")
                || lower.contains("increased")
                || lower.contains("increase ");
            let mentions_fall = lower.contains("drop")
                || lower.contains("dropped")
                || lower.contains("fell")
                || lower.contains("plummet")
                || lower.contains("decrease")
                || lower.contains("went down")
                || lower.contains("lower ");
            let benign_score_q = (lower.contains("is it normal")
                || lower.contains("is this normal")
                || lower.contains("should i worry")
                || lower.contains("should i be worried")
                || lower.contains("is that normal"))
                && mentions_rise
                && !mentions_fall;
            if benign_score_q {
                return true;
            }
            if lower.contains("am i missing") || lower.contains("am i crazy") {
                return true;
            }
        }
        if lower.contains("not sure what to make") {
            return true;
        }
        if lower.contains("two different numbers")
            || (lower.contains("two different") && lower.contains("balance"))
        {
            return true;
        }
        // Channel mismatch — app vs merchant
        (lower.contains("merchant") || lower.contains("vendor"))
            && lower.contains("pending")
            && (lower.contains("complete") || lower.contains("says"))
    }

    /// Headline-style crypto / asset-market copy (no I/my/we). Consumer wallet lines skip this path.
    fn sentiment_third_party_crypto_market_copy_topic_key(&self, lower: &str) -> Option<String> {
        if Self::looks_like_first_person_finance_user(lower) {
            return None;
        }
        if !Self::has_crypto_or_broad_market_lexicon(lower) {
            return None;
        }

        let recovery_narrative = lower.contains("worst may be over")
            || lower.contains("may be over")
            || lower.contains("early signs")
            || lower.contains("signs of recovery")
            || lower.contains("bottom may be in");

        let bearish = !recovery_narrative
            && (lower.contains("losses")
                || lower.contains("losing")
                || lower.contains("plunge")
                || lower.contains("plummet")
                || lower.contains("crash")
                || lower.contains("crashed")
                || lower.contains("sell-off")
                || lower.contains("selloff")
                || lower.contains("bear market")
                || lower.contains("major drawdown")
                || lower.contains("liquidations")
                || lower.contains("depeg")
                || lower.contains("insolvent")
                || lower.contains("bankruptcy"));
        if bearish {
            return Some("negative_mild".to_string());
        }

        let neutral_headline = lower.contains(" in focus")
            || lower.contains(" in focus:")
            || lower.contains("expert ")
            || lower.contains("analyst ")
            || lower.contains("analysts ")
            || lower.contains("crypto markets")
            || lower.contains("holders are")
            || lower.contains("holders face")
            || lower.contains(" on the horizon")
            || lower.contains("kumo")
            || lower.contains("volatility")
            || lower.contains("price reversal")
            || lower.contains("early signs")
            || lower.contains("worst may be over")
            || lower.contains("prolonged decline")
            || lower.contains("stalls inside")
            || lower.contains("surge on the");
        if neutral_headline {
            return Some("neutral".to_string());
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

    fn has_crypto_or_broad_market_lexicon(lower: &str) -> bool {
        lower.contains("bitcoin")
            || lower.contains("btc")
            || lower.contains("xrp")
            || lower.contains("doge")
            || lower.contains("dogecoin")
            || lower.contains("ethereum")
            || lower.contains(" eth ")
            || lower.starts_with("eth ")
            || lower.contains("solana")
            || lower.contains("crypto")
            || lower.contains("altcoin")
            || lower.contains("defi")
            || lower.contains("stablecoin")
            || lower.contains("blockchain")
            || lower.contains("token ")
            || lower.contains(" spot etf")
    }

    fn has_mixed_silver_lining_pattern(&self, lower: &str) -> bool {
        const BAD: &[&str] = &[
            "failed again",
            "failed twice",
            "crashed",
            "broke again",
            "two days of errors",
            "days of errors",
        ];
        const GOOD: &[&str] = &[
            "at least support",
            "at least they",
            "responded quickly",
            "saved my draft",
            "quick response",
        ];
        BAD.iter().any(|p| lower.contains(p)) && GOOD.iter().any(|p| lower.contains(p))
    }

    fn has_mixed_fraud_flag_relief_pattern(&self, lower: &str) -> bool {
        lower.contains("glad")
            && (lower.contains("flagged")
                || lower.contains("fraud alert")
                || lower.contains("fraud system")
                || lower.contains("froze")
                || lower.contains("frozen")
                || lower.contains("locked my"))
    }

    fn has_mixed_implicit_followon_skeptic(&self, lower: &str) -> bool {
        // "for once" / "not used to" + deposit → [`cautiously_positive`] (handled above).
        const PHRASES: &[&str] = &["that's unusual", "that is unusual", "thats unusual"];
        if PHRASES.iter().any(|p| lower.contains(p)) {
            return true;
        }
        // Non-deposit surprise hedges only; "actually worked" + "unusual" → cautious path.
        if lower.contains("unusual")
            && (lower.contains("that") || lower.contains("worked") || lower.contains("actually"))
        {
            return !lower.contains("actually worked");
        }
        false
    }

    /// Money-movement / approval cues that often co-occur with skepticism in the second clause.
    fn has_mixed_positive_outcome_cue(&self, lower: &str) -> bool {
        const PHRASES: &[&str] = &[
            "deposit hit early",
            "paycheck hit early",
            "payment hit early",
            "payment finally cleared",
            "finally cleared",
            "refund posted",
            "glad the refund",
            "nice that",
            "went through instantly",
            "went through today",
            "transfer went through",
            "transfer was instant",
            "instant today",
            "payment went through",
            "it went through",
            "cleared early",
            "posted early",
            "credited early",
            "credit score jumped",
            "score jumped",
            "points jumped",
            "actually worked",
            "app actually worked",
            "responded quickly",
            "at least support",
            "at least they",
            "fraud system is paying",
        ];
        PHRASES.iter().any(|p| lower.contains(p))
    }

    fn has_mixed_skepticism_or_friction_cue(&self, lower: &str) -> bool {
        const PHRASES: &[&str] = &[
            "weird",
            "strange",
            "odd that",
            "don't trust",
            "dont trust",
            "do not trust",
            "don't understand",
            "dont understand",
            "do not understand",
            "never does",
            "never did",
            "never has",
            "didn't update",
            "didnt update",
            "logged me out",
            "logged out",
            "looks wrong",
            "feel wrong",
            "whatever system",
            "suspicious",
            "usually a disaster",
            "shame that",
            " shame ",
            "two days of errors",
            "failed again",
            "why the charge",
            "why did the charge",
            "relieved but",
            " but annoyed",
            "days of errors",
        ];
        PHRASES.iter().any(|p| lower.contains(p))
    }

    /// Declined/approved oscillation or multiple declines — not a single-pole negative_mild story.
    fn has_operational_inconsistency_mixed_signal(&self, lower: &str) -> bool {
        let declined = lower.matches("declined").count();
        let approved = lower.contains("approved");
        (declined >= 2) || (declined >= 1 && approved)
    }

    /// Substring match against phrases from `lexical_polarity` `negative_mild` / `negative_strong` rows.
    #[inline]
    pub fn negated_eval_phrase_hit(&self, lower: &str) -> bool {
        self.negative_lexical_suppress_phrases
            .iter()
            .any(|p| lower.contains(p.as_str()))
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
                // Avoid “Anchored by 'like'” when the line is “don’t like …” / “wouldn’t use …”
                // (tokenizer splits “don’t” → don + t; “like” still matches as its own token).
                if self.negated_eval_phrase_hit(lower) {
                    return None;
                }
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

#[cfg(test)]
mod negation_tests {
    use super::*;

    #[test]
    fn lexical_polarity_prefers_negation_over_positive_idiom() {
        let raw = include_str!("../../data/sentiment/inference_sentiment.toml");
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
        let raw = include_str!("../../data/sentiment/inference_sentiment.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("fixture TOML");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let lower = "i don't like using google";
        assert!(rules.negated_eval_phrase_hit(lower));
        assert!(rules.anchor_phrase(lower, "positive_mild").is_none());
    }

    #[test]
    fn lexical_polarity_negative_strong_and_litotes() {
        let raw = include_str!("../../data/sentiment/inference_sentiment.toml");
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
        let raw = include_str!("../../data/sentiment/inference_sentiment.toml");
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
        let raw = include_str!("../../data/sentiment/inference_sentiment.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("fixture TOML");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let l = |s: &str| s.to_lowercase();
        assert!(rules.is_objective_factual_statement(&l("spec sheet says 512gb ram")));
        assert!(rules.is_objective_factual_statement(&l("cap is 5 tb.")));
        assert!(rules.is_objective_factual_statement(&l("ordered 1tb, 512gb ssd")));
        assert!(!rules.is_objective_factual_statement(&l("I love this laptop")));
    }
}
