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
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

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
// CLI / host overrides (updated when `--project` / `--inference-toml` is applied)
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Default)]
struct CliTomlPaths {
    primary: Option<PathBuf>,
    defaults: Option<PathBuf>,
}

#[cfg(not(target_arch = "wasm32"))]
static CLI_TOML_PATHS: std::sync::RwLock<CliTomlPaths> =
    std::sync::RwLock::new(CliTomlPaths {
        primary: None,
        defaults: None,
    });

/// Register inference TOML paths from the growformer CLI or host (call before any inference load).
/// Passing `None` for both is a no-op (keeps search/env behavior).
/// Later calls merge non-`None` fields so embedded defaults can be overridden by `--project`.
#[cfg(not(target_arch = "wasm32"))]
pub fn set_inference_toml_cli_paths(primary: Option<PathBuf>, defaults: Option<PathBuf>) {
    if primary.is_none() && defaults.is_none() {
        return;
    }
    {
        let mut guard = CLI_TOML_PATHS.write().unwrap();
        if primary.is_some() {
            guard.primary = primary;
        }
        if defaults.is_some() {
            guard.defaults = defaults;
        }
    }
    invalidate_native_inference_toml_cache();
}

#[cfg(not(target_arch = "wasm32"))]
fn invalidate_native_inference_toml_cache() {
    let mut guard = FULL.write().unwrap();
    *guard = None;
}

#[cfg(target_arch = "wasm32")]
pub fn set_inference_toml_cli_paths(_primary: Option<PathBuf>, _defaults: Option<PathBuf>) {}

#[cfg(not(target_arch = "wasm32"))]
fn cli_toml_paths() -> CliTomlPaths {
    CLI_TOML_PATHS.read().unwrap().clone()
}

// ---------------------------------------------------------------------------
// On-disk / include! document (thresholds + rules)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceTomlDocument {
    #[serde(flatten)]
    pub thresholds: InferenceThresholds,
    #[serde(default)]
    pub rules: InferenceRulesSection,
    #[serde(default)]
    pub generation: GenerationConfig,
    #[serde(default)]
    pub response_shaping: ResponseShapingConfig,
    #[serde(default)]
    pub validation: ValidationConfig,
    #[serde(default)]
    pub fragment_compose: FragmentComposeConfig,
}

/// Configuration for the generation/decoding stage, parsed from `[generation]` in inference TOML.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationConfig {
    #[serde(default = "default_generation_temperature")]
    pub temperature: f32,
    #[serde(default = "default_generation_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub top_p: f32,
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
    #[serde(default)]
    pub min_tokens: usize,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    #[serde(default)]
    pub enforce_min_tokens: bool,
    /// Enable stochastic top-k retrieval (when true, temperature controls sampling).
    #[serde(default = "default_stochastic_retrieval")]
    pub stochastic_retrieval: bool,
    /// Enable the homeostatic drive field: drives → neuromodulators → knob gains.
    /// Off by default so it is a clean A/B against base behavior.
    #[serde(default)]
    pub drive_field: bool,
    /// Enable the reflective field: unified Identity ⊕ Activity ⊕ Drive composition
    /// of the generation conditioning (one neuromodulator-coupled policy instead of
    /// scattered blend constants). Off by default for a clean A/B.
    #[serde(default)]
    pub reflective_field: bool,
    /// Enable the synthetic basal ganglia: value-weighted, neuromodulator-gated
    /// action selection over retrieval candidates (consolidates the coherence /
    /// anti-repeat / exploration heuristics). Off by default for a clean A/B.
    #[serde(default)]
    pub basal_ganglia: bool,
}

fn default_generation_temperature() -> f32 { 0.85 }
fn default_generation_top_k() -> usize { 4 }
fn default_repetition_penalty() -> f32 { 1.08 }
fn default_max_tokens() -> usize { 90 }
fn default_stochastic_retrieval() -> bool { true }

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            temperature: default_generation_temperature(),
            top_k: default_generation_top_k(),
            top_p: 0.92,
            repetition_penalty: default_repetition_penalty(),
            min_tokens: 18,
            max_tokens: default_max_tokens(),
            stop_sequences: Vec::new(),
            enforce_min_tokens: true,
            stochastic_retrieval: default_stochastic_retrieval(),
            drive_field: false,
            reflective_field: false,
            basal_ganglia: false,
        }
    }
}

/// Response shaping rules parsed from `[response_shaping]` section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseShapingConfig {
    #[serde(default = "default_min_response_chars")]
    pub min_response_chars: usize,
    #[serde(default = "default_max_response_chars")]
    pub max_response_chars: usize,
    #[serde(default)]
    pub require_first_person: bool,
    #[serde(default)]
    pub forbid_asterisks: bool,
    #[serde(default)]
    pub forbid_third_person_self_reference: bool,
    #[serde(default)]
    pub forbid_brain_metadata_in_output: bool,
    #[serde(default)]
    pub require_sensory_or_vocalization: bool,
    #[serde(default)]
    pub forbidden_phrases: Vec<String>,
    #[serde(default)]
    pub voice_violation_patterns: Vec<String>,
    #[serde(default)]
    pub required_signal_patterns: Vec<String>,
}

fn default_min_response_chars() -> usize { 60 }
fn default_max_response_chars() -> usize { 320 }

impl Default for ResponseShapingConfig {
    fn default() -> Self {
        Self {
            min_response_chars: default_min_response_chars(),
            max_response_chars: default_max_response_chars(),
            require_first_person: false,
            forbid_asterisks: false,
            forbid_third_person_self_reference: false,
            forbid_brain_metadata_in_output: false,
            require_sensory_or_vocalization: false,
            forbidden_phrases: Vec::new(),
            voice_violation_patterns: Vec::new(),
            required_signal_patterns: Vec::new(),
        }
    }
}

/// Validation pipeline config parsed from `[validation]` section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_max_retries")]
    pub max_retries: u8,
    #[serde(default = "default_retry_temperature_decay")]
    pub retry_temperature_decay: f32,
    #[serde(default)]
    pub fallback_strategy: String,
}

fn default_max_retries() -> u8 { 2 }
fn default_retry_temperature_decay() -> f32 { 0.15 }

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_retries: default_max_retries(),
            retry_temperature_decay: default_retry_temperature_decay(),
            fallback_strategy: "nearest_training_example".to_string(),
        }
    }
}

/// Typed fragment composition policy from `[fragment_compose]` in inference TOML.
/// Agent-specific vocal tokens, intent routing, and library paths live here —
/// not in the Rust runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentComposeConfig {
    #[serde(default)]
    pub enabled: bool,
    /// JSONL fragment library path, relative to the inference TOML directory.
    #[serde(default)]
    pub library: Option<String>,
    /// Vocal tokens checked on the terminal sentence of composed output.
    #[serde(default)]
    pub vocalizations: Vec<String>,
    /// Second tokens allowed after a vocal in a coda (e.g. `"now"` → `"Mrrp now."`).
    #[serde(default = "default_vocal_coda_modifiers")]
    pub vocal_coda_modifiers: Vec<String>,
    /// Intents that always try for a second body fragment from a different voice.
    #[serde(default)]
    pub force_second_body_intents: Vec<String>,
    /// Intents allowed to prepend a conversational opener fragment.
    /// When empty, a built-in greeting/reunion/comfort default list is used.
    #[serde(default)]
    pub opener_intents: Vec<String>,
    /// Baseline runtime state when no `agent_state` dimensions are supplied.
    #[serde(default)]
    pub default_neutral_state: std::collections::HashMap<String, f32>,
    /// Exact greeting phrases for the `greeting` intent rule.
    #[serde(default)]
    pub greeting_exact: Vec<String>,
    /// Prefixes before the agent name for name-based greetings (`"hey luna"`).
    #[serde(default)]
    pub agent_name_prefixes: Vec<String>,
    #[serde(default = "default_agent_name_greeting_max_len")]
    pub agent_name_greeting_max_len: usize,
    /// Ordered intent rules; first match wins. Include a terminal `fallback` rule.
    #[serde(default)]
    pub intent_rules: Vec<FragmentIntentRuleToml>,
    /// Offline decomposition heuristics (voice/opener/state-gate classification).
    #[serde(default)]
    pub decompose: FragmentDecomposeConfig,
    /// Intent-specific ordered body sub-slots for templated composition.
    #[serde(default)]
    pub compose_templates: Vec<ComposeTemplateToml>,
    /// Negative affinity: fragments to suppress when a given intent is active.
    #[serde(default)]
    pub intent_excludes: Vec<IntentExcludeRuleToml>,
}

/// One row in `[[fragment_compose.compose_templates]]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComposeTemplateToml {
    pub intent: String,
    #[serde(default)]
    pub body_slots: Vec<String>,
    #[serde(default = "default_template_min_bodies")]
    pub min_bodies: usize,
    #[serde(default = "default_require_distinct_voices")]
    pub require_distinct_voices: bool,
}

fn default_template_min_bodies() -> usize {
    1
}

fn default_require_distinct_voices() -> bool {
    true
}

/// One row in `[[fragment_compose.intent_excludes]]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentExcludeRuleToml {
    pub when_intent: String,
    #[serde(default)]
    pub exclude_body_slots: Vec<String>,
    #[serde(default)]
    pub exclude_keywords: Vec<String>,
    #[serde(default)]
    pub exclude_fragment_intents: Vec<String>,
}

/// Decomposition-only policy under `[fragment_compose.decompose]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentDecomposeConfig {
    /// Sentence prefixes classified as conversational openers.
    #[serde(default)]
    pub opener_prefixes: Vec<String>,
    /// Keywords scoring a clause as drive/state voice.
    #[serde(default)]
    pub drive_keywords: Vec<String>,
    /// Keywords scoring a clause as activity/action voice.
    #[serde(default)]
    pub activity_keywords: Vec<String>,
    /// Keywords scoring a clause as identity/persona voice.
    #[serde(default)]
    pub identity_keywords: Vec<String>,
    /// Any match forces drive voice (e.g. treat/meal context).
    #[serde(default)]
    pub drive_override_keywords: Vec<String>,
    /// Runtime dims merged into fragment `state_gate` ranges.
    #[serde(default = "default_state_gate_dims")]
    pub state_gate_dims: Vec<String>,
    /// Keywords classifying body fragments into compose sub-slots.
    #[serde(default)]
    pub body_slot_keywords: std::collections::HashMap<String, Vec<String>>,
}

fn default_state_gate_dims() -> Vec<String> {
    vec!["hunger".into(), "energy".into(), "mood".into()]
}

impl Default for FragmentDecomposeConfig {
    fn default() -> Self {
        Self {
            opener_prefixes: Vec::new(),
            drive_keywords: Vec::new(),
            activity_keywords: Vec::new(),
            identity_keywords: Vec::new(),
            drive_override_keywords: Vec::new(),
            state_gate_dims: default_state_gate_dims(),
            body_slot_keywords: std::collections::HashMap::new(),
        }
    }
}

fn default_vocal_coda_modifiers() -> Vec<String> {
    vec!["now".to_string()]
}

fn default_agent_name_greeting_max_len() -> usize {
    48
}

fn default_fragment_min_voices() -> usize {
    1
}

impl Default for FragmentComposeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            library: None,
            vocalizations: Vec::new(),
            vocal_coda_modifiers: default_vocal_coda_modifiers(),
            force_second_body_intents: Vec::new(),
            opener_intents: Vec::new(),
            default_neutral_state: std::collections::HashMap::new(),
            greeting_exact: Vec::new(),
            agent_name_prefixes: Vec::new(),
            agent_name_greeting_max_len: default_agent_name_greeting_max_len(),
            intent_rules: Vec::new(),
            decompose: FragmentDecomposeConfig::default(),
            compose_templates: Vec::new(),
            intent_excludes: Vec::new(),
        }
    }
}

/// One row in `[fragment_compose.intent_rules]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentIntentRuleToml {
    #[serde(default)]
    pub id: String,
    pub intent: String,
    #[serde(default)]
    pub anchors: Vec<String>,
    #[serde(default = "default_fragment_min_voices")]
    pub min_voices: usize,
    #[serde(default)]
    pub relaxed_parts: bool,
    /// `greeting` | `contains_any` | `starts_with_any` | `fallback`
    #[serde(default)]
    pub r#match: String,
    #[serde(default)]
    pub patterns: Vec<String>,
    #[serde(default)]
    pub max_len: Option<usize>,
}

/// Resolved intent hints for fragment eligibility / quality gate.
#[derive(Debug, Clone)]
pub struct FragmentIntentHint {
    pub intent: String,
    pub anchors: Vec<String>,
    pub min_voices: usize,
    pub relaxed_parts: bool,
}

impl FragmentComposeConfig {
    /// Intents that may prepend an opener fragment. Other intents compose body+coda only.
    pub fn effective_opener_intents(&self) -> Vec<String> {
        if !self.opener_intents.is_empty() {
            return self.opener_intents.clone();
        }
        [
            "greeting_check_in",
            "reunion_warm",
            "reunion",
            "identity_intro",
            "owner_absence",
            "emotional_support",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    pub fn should_use_opener(&self, intent: &str) -> bool {
        self.effective_opener_intents()
            .iter()
            .any(|i| i == intent)
    }

    /// Compose template for an intent, if configured.
    pub fn template_for_intent(&self, intent: &str) -> Option<&ComposeTemplateToml> {
        self.compose_templates.iter().find(|t| t.intent == intent)
    }

    /// Merge negative-affinity rules for the active intent.
    pub fn excludes_for_intent(&self, intent: &str) -> crate::fragment_composer::ComposeExcludes {
        use crate::fragment_composer::ComposeExcludes;
        let mut out = ComposeExcludes::default();
        for rule in &self.intent_excludes {
            if rule.when_intent != intent {
                continue;
            }
            for slot in &rule.exclude_body_slots {
                out.body_slots.insert(slot.clone());
            }
            for kw in &rule.exclude_keywords {
                out.keywords.push(kw.to_ascii_lowercase());
            }
            for fi in &rule.exclude_fragment_intents {
                out.fragment_intents.insert(fi.clone());
            }
        }
        out
    }

    /// Classify a body fragment into a compose sub-slot from decompose keywords.
    pub fn classify_body_slot(&self, text: &str, role: &str) -> Option<String> {
        if role != "body" || self.decompose.body_slot_keywords.is_empty() {
            return None;
        }
        let lower = text.to_ascii_lowercase();
        // Fixed priority so mealtime/grounding win over generic empathic/action.
        const PRIORITY: &[&str] = &[
            "mealtime", "preference", "lore", "stance", "grounding", "gratitude", "bonding", "refusal",
            "offer", "empathic", "action",
        ];
        for slot in PRIORITY {
            if let Some(keywords) = self.decompose.body_slot_keywords.get(*slot) {
                if keywords.iter().any(|kw| lower.contains(&kw.to_ascii_lowercase())) {
                    return Some(slot.to_string());
                }
            }
        }
        None
    }

    /// Match the user prompt against configured intent rules.
    pub fn match_intent(&self, text: &str, agent_name: &str) -> FragmentIntentHint {
        let lower = text.to_ascii_lowercase();
        for rule in &self.intent_rules {
            if self.rule_matches(&lower, text, agent_name, rule) {
                return FragmentIntentHint {
                    intent: rule.intent.clone(),
                    anchors: rule.anchors.clone(),
                    min_voices: rule.min_voices,
                    relaxed_parts: rule.relaxed_parts,
                };
            }
        }
        FragmentIntentHint {
            intent: "open_ended_chat".into(),
            anchors: Vec::new(),
            min_voices: 2,
            relaxed_parts: false,
        }
    }

    fn rule_matches(
        &self,
        lower: &str,
        original: &str,
        agent_name: &str,
        rule: &FragmentIntentRuleToml,
    ) -> bool {
        match rule.r#match.as_str() {
            "greeting" => self.matches_greeting(lower, original),
            "contains_any" => rule.patterns.iter().any(|p| lower.contains(&p.to_ascii_lowercase())),
            "starts_with_any" => rule.patterns.iter().any(|p| {
                let pat = p.to_ascii_lowercase();
                lower.starts_with(&pat)
                    && rule.max_len.map(|m| lower.len() < m).unwrap_or(true)
            }),
            "agent_name_greeting" => self.matches_agent_name_greeting(lower, agent_name),
            "fallback" => true,
            _ => false,
        }
    }

    fn matches_greeting(&self, lower: &str, original: &str) -> bool {
        let trimmed = lower.trim().trim_end_matches(|c: char| c.is_ascii_punctuation());
        let matched = self.greeting_exact.iter().any(|p| {
            let p = p.to_ascii_lowercase();
            trimmed == p || trimmed.starts_with(&format!("{p} "))
        });
        if !matched {
            return false;
        }
        if trimmed.len() > 40 {
            let original_trimmed = original.trim();
            for prefix in &["Hello ", "Hi ", "Hey "] {
                if original_trimmed.starts_with(prefix) {
                    let rest = &original_trimmed[prefix.len()..];
                    if rest.starts_with(|c: char| c.is_uppercase()) {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn matches_agent_name_greeting(&self, lower: &str, agent_name: &str) -> bool {
        let name = agent_name.trim().to_ascii_lowercase();
        if name.is_empty() {
            return false;
        }
        if self.agent_name_greeting_max_len > 0 && lower.len() >= self.agent_name_greeting_max_len {
            return false;
        }
        self.agent_name_prefixes.iter().any(|prefix| {
            let p = prefix.to_ascii_lowercase();
            if !lower.starts_with(&p) {
                return false;
            }
            let rest = lower[p.len()..].trim();
            rest == name || rest.starts_with(&format!("{name} "))
        })
    }

    /// True when the terminal sentence is vocal + disallowed modifier (`"Mrrp math."`).
    pub fn vocalization_tail_suspicious(&self, text: &str) -> bool {
        if self.vocalizations.is_empty() {
            return false;
        }
        let last = self.last_sentence(text);
        let ws: Vec<&str> = last.split_whitespace().collect();
        if ws.len() <= 1 {
            return false;
        }
        self.detect_vocalization(ws[0]).is_some() && !self.is_pure_vocal_coda(&last)
    }

    /// Load `[fragment_compose]` from an inference TOML file on disk.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_from_inference_toml_path(path: &std::path::Path) -> Result<Self, String> {
        let s = std::fs::read_to_string(path)
            .map_err(|e| format!("read inference TOML {}: {e}", path.display()))?;
        Self::load_from_inference_toml_str(&s)
    }

    /// Parse `[fragment_compose]` from a full inference TOML document string.
    pub fn load_from_inference_toml_str(toml_str: &str) -> Result<Self, String> {
        let doc: InferenceTomlDocument = toml::from_str(toml_str)
            .map_err(|e| format!("parse inference TOML: {e}"))?;
        if doc.fragment_compose.vocalizations.is_empty() {
            return Err(
                "fragment_compose.vocalizations is empty — add vocal tokens to inference TOML"
                    .into(),
            );
        }
        Ok(doc.fragment_compose)
    }

    /// First matching non-`fallback` intent rule for a user prompt, if any.
    pub fn prompt_intent_override(&self, text: &str, agent_name: &str) -> Option<FragmentIntentHint> {
        let lower = text.to_ascii_lowercase();
        for rule in &self.intent_rules {
            if rule.r#match == "fallback" {
                return None;
            }
            if self.rule_matches(&lower, text, agent_name, rule) {
                return Some(FragmentIntentHint {
                    intent: rule.intent.clone(),
                    anchors: rule.anchors.clone(),
                    min_voices: rule.min_voices,
                    relaxed_parts: rule.relaxed_parts,
                });
            }
        }
        None
    }

    /// Detect a configured vocal token at the start of `text` (longest match first).
    pub fn detect_vocalization(&self, text: &str) -> Option<String> {
        let lower = text.trim().trim_end_matches('.').to_ascii_lowercase();
        if lower.is_empty() {
            return None;
        }
        let first = lower.split_whitespace().next()?;
        let mut vocs: Vec<&str> = self.vocalizations.iter().map(String::as_str).collect();
        vocs.sort_by_key(|v| std::cmp::Reverse(v.len()));
        for v in vocs {
            if first == v {
                return Some(v.to_string());
            }
        }
        None
    }

    /// Trailing coda slot: bare vocalization or vocal + allowed modifier only.
    pub fn is_pure_vocal_coda(&self, text: &str) -> bool {
        let trimmed = text.trim().trim_end_matches('.').trim().to_ascii_lowercase();
        if trimmed.is_empty() {
            return false;
        }
        let ws: Vec<&str> = trimmed.split_whitespace().collect();
        match ws.len() {
            1 => self.detect_vocalization(ws[0]).is_some(),
            2 => {
                self.detect_vocalization(ws[0]).is_some()
                    && self
                        .vocal_coda_modifiers
                        .iter()
                        .any(|m| ws[1] == m.as_str())
            }
            _ => false,
        }
    }

    /// Whether a sentence begins with a configured vocal token.
    pub fn starts_with_vocalization(&self, text: &str) -> bool {
        let lower = text.to_ascii_lowercase();
        self.vocalizations
            .iter()
            .any(|v| lower.starts_with(v.as_str()))
    }

    /// Classify a decomposed sentence into identity / activity / drive voice.
    pub fn classify_voice(&self, text: &str, role: &str) -> &'static str {
        if role == "coda" {
            return "identity";
        }
        let lower = text.to_ascii_lowercase();
        let d = &self.decompose;
        if d
            .drive_override_keywords
            .iter()
            .any(|k| lower.contains(&k.to_ascii_lowercase()))
        {
            return "drive";
        }
        let drive_score = Self::score_keywords(&lower, &d.drive_keywords);
        let activity_score = Self::score_keywords(&lower, &d.activity_keywords);
        let identity_score = Self::score_keywords(&lower, &d.identity_keywords);
        if drive_score >= activity_score && drive_score >= identity_score && drive_score > 0 {
            "drive"
        } else if activity_score >= identity_score && activity_score > 0 {
            "activity"
        } else {
            "identity"
        }
    }

    /// Whether a lowercased sentence is a conversational opener prefix.
    pub fn is_opener(&self, lower: &str) -> bool {
        self.decompose
            .opener_prefixes
            .iter()
            .any(|p| lower.starts_with(&p.to_ascii_lowercase()))
    }

    /// Validate config has decomposition voice keywords.
    pub fn validate_for_decompose(&self) -> Result<(), String> {
        let d = &self.decompose;
        if d.drive_keywords.is_empty()
            && d.activity_keywords.is_empty()
            && d.identity_keywords.is_empty()
        {
            return Err(
                "fragment_compose.decompose voice keyword lists are empty — add drive/activity/identity keywords to inference TOML".into(),
            );
        }
        Ok(())
    }

    fn score_keywords(text: &str, keywords: &[String]) -> i32 {
        keywords
            .iter()
            .filter(|k| text.contains(&k.to_ascii_lowercase()))
            .count() as i32
    }

    fn last_sentence(&self, text: &str) -> String {
        let lower = text.to_ascii_lowercase();
        lower
            .split(|c| c == '.' || c == '!' || c == '?')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .last()
            .map(|s| s.to_string())
            .unwrap_or(lower)
    }

    /// Resolve the fragment library path: `GROWFORMER_FRAGMENTS` env, then
    /// `[fragment_compose].library` relative to the inference TOML / brain dirs.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn resolve_library_path(&self, brain_path: &str) -> Option<std::path::PathBuf> {
        use std::path::{Path, PathBuf};
        if let Ok(p) = std::env::var("GROWFORMER_FRAGMENTS") {
            let p = p.trim();
            if !p.is_empty() {
                return Some(PathBuf::from(p));
            }
        }
        if !self.enabled {
            return None;
        }
        let lib = self.library.as_ref()?.trim();
        if lib.is_empty() {
            return None;
        }
        let rel = PathBuf::from(lib);
        if rel.is_absolute() && rel.is_file() {
            return Some(rel);
        }
        if let Some(base) = inference_toml_directory() {
            let cand = base.join(&rel);
            if cand.is_file() {
                return Some(cand);
            }
        }
        if let Some(parent) = Path::new(brain_path).parent() {
            for cand in [parent.join(&rel), parent.join("data").join(&rel)] {
                if cand.is_file() {
                    return Some(cand);
                }
            }
        }
        if rel.is_file() {
            Some(rel)
        } else {
            None
        }
    }
}

/// Directory containing the active inference TOML (for resolving relative paths).
#[cfg(not(target_arch = "wasm32"))]
pub fn inference_toml_directory() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;
    if let Some(p) = cli_toml_paths().primary {
        return p.parent().map(|d| d.to_path_buf());
    }
    if let Ok(path) = std::env::var("GROWFORMER_INFERENCE_TOML") {
        return PathBuf::from(path).parent().map(|d| d.to_path_buf());
    }
    None
}

#[cfg(target_arch = "wasm32")]
pub fn inference_toml_directory() -> Option<std::path::PathBuf> {
    None
}

/// PR-wire headline: normalized text must start with `prefix`, meet `min_len`, and pass excludes / `require_any`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrWireNeutralIntentRow {
    #[serde(default)]
    pub min_len: Option<usize>,
    #[serde(default)]
    pub intent: Vec<Vec<String>>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
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
    /// Ordered fallback lines after a misfire in chat passthrough (first matching prompt row wins).
    #[serde(default)]
    pub lattice_misfire_fallback: Vec<LatticeMisfireFallbackRule>,
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
    /// Skip `like` anchor when the preceding token is a comparison-context noun
    /// ("stocks like Apple" → "such as", not sentiment).
    #[serde(default)]
    pub like_comparison_preceding_tokens: Vec<String>,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LexicalPolarityRow {
    pub topic: String,
    pub phrases: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SarcasmAndRule {
    #[serde(default)]
    pub groups: Vec<Vec<String>>,
}

/// Per-token gloss for [`InferenceRulesRuntime::anchor_phrase`] (substring / token hit on `w`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnchorTokenGlossRow {
    pub token: String,
    pub gloss: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeadlineLexicalTopicRule {
    pub topic: String,
    /// Service may redirect to the sentiment lattice and bypass the knowledge floor.
    #[serde(default)]
    pub inclusion_redirect: bool,
    #[serde(default, alias = "keywords_cnf")]
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatticeMisfireRule {
    /// Prompt-side CNF. When empty, any prompt matches (subject to `intent_exclude`).
    #[serde(default)]
    pub intent: Vec<Vec<String>>,
    /// When this CNF matches the prompt, the rule does not fire (e.g. school context on a school arc).
    #[serde(default)]
    pub intent_exclude: Vec<Vec<String>>,
    /// Any listed substring in the response counts as a hit (OR). Combined with `response` when both set.
    #[serde(default)]
    pub response_any: Vec<String>,
    /// AND-of-OR groups on the response (substring match). Empty means ignore unless `response_any` set.
    #[serde(default)]
    pub response: Vec<Vec<String>>,
    /// When non-empty, at least one prior agent turn must contain any listed substring
    /// before this rule can fire (cross-turn arc bleed).
    #[serde(default)]
    pub prior_response_any: Vec<String>,
}

/// Short canned line when chat passthrough detects a lattice/compose misfire (first matching row wins).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatticeMisfireFallbackRule {
    /// Prompt-side CNF. Empty `intent` matches any prompt (use as last-resort default row).
    #[serde(default)]
    pub intent: Vec<Vec<String>>,
    #[serde(default)]
    pub intent_exclude: Vec<Vec<String>>,
    pub response: String,
    #[serde(default = "default_lattice_misfire_fallback_template_id")]
    pub template_id: String,
}

fn default_lattice_misfire_fallback_template_id() -> String {
    "lattice_misfire_fallback".to_string()
}

impl InferenceRulesSection {
    /// Empty lists are replaced from `defaults` (shipped baseline).
    fn merge_empty_from(&self, defaults: &Self) -> Self {
        let mut s = self.clone();
        if s.contrastive_markers.is_empty() {
            s.contrastive_markers = defaults.contrastive_markers.clone();
        }
        // Lexical polarity is additive vocabulary — domain TOMls provide
        // domain-specific phrases, core TOML provides base coverage. Both
        // must be available to the Aho-Corasick automaton (longest-match-wins
        // resolves conflicts).
        if s.lexical_polarity.is_empty() {
            s.lexical_polarity = defaults.lexical_polarity.clone();
        } else if !defaults.lexical_polarity.is_empty() {
            s.lexical_polarity
                .extend(defaults.lexical_polarity.iter().cloned());
        }
        if s.sarcasm_simple.is_empty() {
            s.sarcasm_simple = defaults.sarcasm_simple.clone();
        } else if !defaults.sarcasm_simple.is_empty() {
            s.sarcasm_simple
                .extend(defaults.sarcasm_simple.iter().cloned());
        }
        if s.sarcasm_and.is_empty() {
            s.sarcasm_and = defaults.sarcasm_and.clone();
        } else if !defaults.sarcasm_and.is_empty() {
            s.sarcasm_and
                .extend(defaults.sarcasm_and.iter().cloned());
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
        } else if !defaults.objective_fact_rules.is_empty() {
            s.objective_fact_rules
                .extend(defaults.objective_fact_rules.iter().cloned());
        }
        if s.headline_lexical_topic.is_empty() {
            s.headline_lexical_topic = defaults.headline_lexical_topic.clone();
        } else if !defaults.headline_lexical_topic.is_empty() {
            // Core / reference may ship minimal rows; domain packs (fintech, etc.) append so both apply.
            s.headline_lexical_topic
                .extend(defaults.headline_lexical_topic.iter().cloned());
        }
        if s.lattice_misfire.is_empty() {
            s.lattice_misfire = defaults.lattice_misfire.clone();
        } else if !defaults.lattice_misfire.is_empty() {
            s.lattice_misfire
                .extend(defaults.lattice_misfire.iter().cloned());
        }
        if s.lattice_misfire_fallback.is_empty() {
            s.lattice_misfire_fallback = defaults.lattice_misfire_fallback.clone();
        } else if !defaults.lattice_misfire_fallback.is_empty() {
            s.lattice_misfire_fallback
                .extend(defaults.lattice_misfire_fallback.iter().cloned());
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
        if s.like_comparison_preceding_tokens.is_empty() {
            s.like_comparison_preceding_tokens = defaults.like_comparison_preceding_tokens.clone();
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
    pub lattice_misfire_fallback: Vec<LatticeMisfireFallbackRule>,
    pub pr_wire_neutral_prefix: Vec<PrWireNeutralPrefixRule>,
    pub pr_wire_neutral_intent: Vec<PrWireNeutralIntentRow>,
    crypto_market_surface_tokens: Vec<String>,
    crypto_market_surface_prefixes: Vec<String>,
    crypto_market_tape_tokens: Vec<String>,
    like_perception_verbs: Vec<String>,
    like_intensity_exception_phrases: Vec<String>,
    like_intensity_exception_ac: Option<AhoCorasick>,
    like_comparison_preceding_tokens: Vec<String>,
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

impl InferenceRulesRuntime {
    pub fn rules_summary_json(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"headline_lexical_topic\":{},",
                "\"lattice_misfire\":{},",
                "\"lexical_polarity\":{},",
                "\"sarcasm_simple\":{},",
                "\"sarcasm_and\":{},",
                "\"positive_anchor_tokens\":{},",
                "\"negative_anchor_tokens\":{},",
                "\"pr_wire_neutral_prefix\":{},",
                "\"pr_wire_neutral_intent\":{},",
                "\"objective_fact_rules\":{},",
                "\"contrastive_markers\":{},",
                "\"crypto_market_surface_tokens\":{},",
                "\"evaluative_words\":{},",
                "\"strong_positive_tokens\":{}",
                "}}"
            ),
            self.headline_lexical_topic.len(),
            self.lattice_misfire.len(),
            self.lexical_polarity.len(),
            self.sarcasm_simple.len(),
            self.sarcasm_and.len(),
            self.positive_anchor_tokens.len(),
            self.negative_anchor_tokens.len(),
            self.pr_wire_neutral_prefix.len(),
            self.pr_wire_neutral_intent.len(),
            self.objective_fact_rules.len(),
            self.contrastive_markers.len(),
            self.crypto_market_surface_tokens.len(),
            self.evaluative_words.len(),
            self.strong_positive_tokens.len(),
        )
    }
}

fn cnf_groups_match(haystack: &str, groups: &[Vec<String>]) -> bool {
    !groups.is_empty() && groups.iter().all(|or_alts| or_alts.iter().any(|p| haystack.contains(p)))
}

/// Any single OR-group match (used for lattice misfire `intent_exclude`).
fn cnf_any_group_matches(haystack: &str, groups: &[Vec<String>]) -> bool {
    groups
        .iter()
        .any(|or_alts| or_alts.iter().any(|p| haystack.contains(p)))
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

/// Returns true if the lowered text looks like bare metadata — timestamps,
/// location stubs, or wire-format datelines with no evaluative content.
///
/// Patterns detected:
/// - Time-only: "5:44am", "10:30 pm", "14:00 gmt"
/// - Dateline stubs: "headlines at", "update at", "briefing at"
/// - Location-only: "new york", "london", "tokyo" without an evaluative word
fn is_bare_metadata(lower: &str) -> bool {
    const TIME_SUFFIXES: &[&str] = &["am", "pm", "gmt", "est", "pst", "cst", "utc", "et", "pt", "ct"];
    const DATELINE_MARKERS: &[&str] = &[
        "headlines at", "headline at", "update at", "briefing at",
        "summary at", "roundup at", "wrap at", "recap at",
        "headlines from", "news at",
    ];
    for m in DATELINE_MARKERS {
        if lower.contains(m) {
            return true;
        }
    }
    let has_time_token = lower.split_whitespace().any(|tok| {
        let t = tok.trim_end_matches(|c: char| c == ',' || c == '.');
        TIME_SUFFIXES.iter().any(|sfx| {
            t.ends_with(sfx) && t.len() > sfx.len() && t[..t.len() - sfx.len()]
                .chars()
                .last()
                .map_or(false, |c| c.is_ascii_digit() || c == ':')
        })
    });
    if has_time_token {
        let alpha_tokens: Vec<&str> = lower
            .split_whitespace()
            .filter(|t| t.chars().any(|c| c.is_alphabetic()))
            .collect();
        if alpha_tokens.len() <= 6 {
            return true;
        }
    }
    false
}

const NEGATION_TOKENS: &[&str] = &[
    "not", "no", "never", "neither", "nor", "without",
    "hardly", "barely", "scarcely", "none",
];

/// Compound negation prefixes that extend the negation scope through an
/// entire subordinate clause: "not because it makes me happy" negates
/// "happy" even though "not" is 5+ tokens away.
const COMPOUND_NEGATION_PREFIXES: &[&str] = &[
    "not because", "not that", "not for", "not since",
    "not as", "not when", "not while", "not after",
];

/// Returns `true` when the word at `match_byte_start` in `lower` sits inside
/// a negation window — i.e. one of the preceding `window_words` tokens is a
/// negation word or a contracted negation (e.g. "don't", "isn't"), OR the
/// match falls inside a compound negation clause ("not because ... X").
fn is_in_negation_window(lower: &str, match_byte_start: usize, window_words: usize) -> bool {
    let prefix = &lower[..match_byte_start];
    for (i, w) in prefix.split_whitespace().rev().enumerate() {
        if i >= window_words {
            break;
        }
        let clean = w.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'');
        if NEGATION_TOKENS.iter().any(|neg| clean == *neg) {
            return true;
        }
        if clean.contains("n't") || clean.contains("n\u{2019}t") {
            return true;
        }
    }
    for cpat in COMPOUND_NEGATION_PREFIXES {
        if let Some(cp_start) = prefix.find(cpat) {
            if cp_start + cpat.len() <= match_byte_start {
                return true;
            }
        }
    }
    false
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
            lattice_misfire_fallback: s.lattice_misfire_fallback,
            pr_wire_neutral_prefix: s.pr_wire_neutral_prefix,
            pr_wire_neutral_intent: s.pr_wire_neutral_intent,
            crypto_market_surface_tokens: s.crypto_market_surface_tokens,
            crypto_market_surface_prefixes: s.crypto_market_surface_prefixes,
            crypto_market_tape_tokens: s.crypto_market_tape_tokens,
            like_perception_verbs: s.like_perception_verbs,
            like_intensity_exception_phrases: s.like_intensity_exception_phrases,
            like_intensity_exception_ac,
            like_comparison_preceding_tokens: s.like_comparison_preceding_tokens,
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
        let lower = Self::normalize_rules_text(intent_text);
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
        let lower = Self::normalize_rules_text(intent_text);
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
                if is_in_negation_window(lower, m.start(), 6) {
                    continue;
                }
                if best.as_ref().map_or(true, |(bl, _)| len > *bl) {
                    best = Some((len, topic));
                }
            }
            return best.map(|(_, t)| t);
        }
        let mut best: Option<(usize, String)> = None;
        for (topic, phrases) in &self.lexical_polarity {
            for p in phrases {
                if let Some(pos) = lower.find(p.as_str()) {
                    if is_in_negation_window(lower, pos, 6) {
                        continue;
                    }
                    let n = p.len();
                    if best.as_ref().map_or(true, |(best_len, _)| n > *best_len) {
                        best = Some((n, topic.clone()));
                    }
                }
            }
        }
        best.map(|(_, t)| t)
    }

    /// Like [`lexical_polarity_signal`] but also returns the byte length of the
    /// winning phrase. Used by the causal-relation preempt gate to distinguish
    /// curated full-sentence entries (long) from short token-level hits.
    pub fn lexical_polarity_signal_with_len(&self, lower: &str) -> Option<(String, usize)> {
        if let Some(ref lp) = self.lexical_polarity_automaton {
            let mut best: Option<(usize, String)> = None;
            for m in lp.ac.find_overlapping_iter(lower) {
                let i = m.pattern().as_usize();
                let len = lp.pattern_len[i];
                let topic = lp.pattern_topic[i].clone();
                if is_in_negation_window(lower, m.start(), 6) {
                    continue;
                }
                if best.as_ref().map_or(true, |(bl, _)| len > *bl) {
                    best = Some((len, topic));
                }
            }
            return best.map(|(l, t)| (t, l));
        }
        let mut best: Option<(usize, String)> = None;
        for (topic, phrases) in &self.lexical_polarity {
            for p in phrases {
                if let Some(pos) = lower.find(p.as_str()) {
                    if is_in_negation_window(lower, pos, 6) {
                        continue;
                    }
                    let n = p.len();
                    if best.as_ref().map_or(true, |(best_len, _)| n > *best_len) {
                        best = Some((n, topic.clone()));
                    }
                }
            }
        }
        best.map(|(l, t)| (t, l))
    }

    /// Lowercase + curly apostrophe + dash normalization (keep aligned with `lattice_shortcuts`).
    pub fn normalize_rules_text(text: &str) -> String {
        let mut s = text.to_lowercase();
        s = s.replace('\u{2019}', "'");
        s = s.replace('\u{2018}', "'");
        s = s.replace('\u{2014}', "-"); // em dash
        s = s.replace('\u{2013}', "-"); // en dash
        s
    }

    /// Sub-lattice topic key from inference TOML only (no MetaBrain).
    ///
    /// Brains missing any of the seven standard topic keys fail [`crate::inference::plugins::lattice_shortcuts::is_lattice_shape`];
    /// expanded taxonomies with **extra** topics still qualify when all seven keys are present,
    /// so **user-anchored lattice preempt** can still run. Layer‑0 keyword expansion and embedding
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
        let lower = Self::normalize_rules_text(intent_text);
        let lower = lower.as_str();

        if let Some(t) = super::frame_lexicon::resolve_scenario_topic(intent_text) {
            return Some(t);
        }

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
                if let Some(pos) = lower.find(w.as_str()) {
                    if is_in_negation_window(lower, pos, 6) {
                        continue;
                    }
                }
                return Some("positive_strong".to_string());
            }
        }
        for w in &self.positive_anchor_tokens {
            if self.strong_positive_tokens.iter().any(|s| s == w) {
                continue;
            }
            if w == "like" {
                if self.like_token_is_perception_idiom(lower)
                    || self.like_token_is_comparison(lower)
                {
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
            if self.weak_positive_token_in_contradicted_context(w, lower) {
                continue;
            }
            if tokens.contains(w.as_str()) {
                if let Some(pos) = lower.find(w.as_str()) {
                    if is_in_negation_window(lower, pos, 6) {
                        continue;
                    }
                }
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
        let lower = Self::normalize_rules_text(intent_text);
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

    fn lattice_misfire_prompt_matches(
        &self,
        prompt_lower: &str,
        rule: &LatticeMisfireRule,
        prior_agent_lower: Option<&str>,
    ) -> bool {
        if !rule.intent.is_empty() && !cnf_any_group_matches(prompt_lower, &rule.intent) {
            return false;
        }
        if !rule.intent_exclude.is_empty()
            && cnf_any_group_matches(prompt_lower, &rule.intent_exclude)
        {
            return false;
        }
        if !rule.prior_response_any.is_empty() {
            let prior = prior_agent_lower.unwrap_or("");
            if prior.is_empty()
                || !rule
                    .prior_response_any
                    .iter()
                    .any(|m| prior.contains(&m.to_ascii_lowercase()))
            {
                return false;
            }
        }
        true
    }

    /// True when a line matches any `[[rules.lattice_misfire]]` row (prompt side ∧ response side).
    pub fn lattice_response_misfire_hit(&self, intent_text: &str, response: &str) -> bool {
        self.lattice_response_misfire_hit_with_prior(intent_text, response, None)
    }

    /// Cross-turn variant: optional prior agent text enables `prior_response_any` rules.
    pub fn lattice_response_misfire_hit_with_prior(
        &self,
        intent_text: &str,
        response: &str,
        prior_agent_text: Option<&str>,
    ) -> bool {
        let il = Self::normalize_rules_text(intent_text);
        let l = il.as_str();
        let rl = response.to_ascii_lowercase();
        let rls = rl.as_str();
        let prior = prior_agent_text.map(|t| t.to_ascii_lowercase());
        let prior_ref = prior.as_deref();
        self.lattice_misfire.iter().any(|rule| {
            self.lattice_misfire_prompt_matches(l, rule, prior_ref)
                && rule.response_side_matches(rls)
        })
    }

    /// First matching `[[rules.lattice_misfire_fallback]]` row for chat passthrough recovery.
    pub fn lattice_misfire_fallback_line(&self, intent_text: &str) -> Option<(String, String)> {
        let l = Self::normalize_rules_text(intent_text);
        let prompt = l.as_str();
        for rule in &self.lattice_misfire_fallback {
            if !rule.intent.is_empty() && !cnf_any_group_matches(prompt, &rule.intent) {
                continue;
            }
            if !rule.intent_exclude.is_empty()
                && cnf_any_group_matches(prompt, &rule.intent_exclude)
            {
                continue;
            }
            let text = rule.response.trim();
            if text.is_empty() {
                continue;
            }
            let template_id = if rule.template_id.trim().is_empty() {
                default_lattice_misfire_fallback_template_id()
            } else {
                rule.template_id.clone()
            };
            return Some((text.to_string(), template_id));
        }
        None
    }

    /// After a lattice misfire strips a bad retrieved line, substitute a short canned witness when
    /// the intent shape is unambiguous (avoids routing-only for Kraken IPO vs M&A-stake collisions).
    pub fn sentiment_lattice_misfire_replacement_line(&self, intent_text: &str) -> Option<String> {
        let lower = Self::normalize_rules_text(intent_text);
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
        let lower = Self::normalize_rules_text(intent_text);
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

    /// `like` after a comparison-context noun ("stocks like Apple") → "such as", not sentiment.
    fn like_token_is_comparison(&self, lower: &str) -> bool {
        if self.like_comparison_preceding_tokens.is_empty() {
            return false;
        }
        let tokens: Vec<&str> = lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .collect();
        for window in tokens.windows(2) {
            if window[1] == "like"
                && self
                    .like_comparison_preceding_tokens
                    .iter()
                    .any(|v| v == window[0])
            {
                return true;
            }
        }
        false
    }

    /// Weak positive tokens (`fine`, `better`, `good`, `like`) should not anchor
    /// when the overall sentence contains strong **preceding negative context** —
    /// e.g. "fills were awful all morning; the desk keeps insisting liquidity is fine"
    /// should not fire POSITIVE on "fine".
    ///
    /// Heuristic: if `token` appears AND the text also contains a negative-anchor
    /// token **before** `token`'s position, skip `token` as an anchor. This prevents
    /// reported-speech or contradiction-clause hijacking.
    fn weak_positive_token_in_contradicted_context(&self, token: &str, lower: &str) -> bool {
        const WEAK_POSITIVES: &[&str] = &["fine", "good", "better", "like", "okay"];
        if !WEAK_POSITIVES.contains(&token) {
            return false;
        }
        let Some(tok_pos) = lower.find(token) else {
            return false;
        };
        let prefix = &lower[..tok_pos];
        self.negative_anchor_tokens.iter().any(|neg| {
            if let Some(neg_pos) = prefix.find(neg.as_str()) {
                !is_in_negation_window(lower, neg_pos, 6)
            } else {
                false
            }
        })
    }

    /// Post-key negative-affect override: when the MetaBrain (or lattice) resolves
    /// to a positive key but the text contains strong negative anchor tokens, the
    /// positive label is wrong — override to `negative_mild` (or `negative_strong`
    /// when the anchor is extreme). Prevents "deep liquidity" from hijacking a
    /// sentence that opens with "fills were awful."
    pub fn negative_affect_overrides_positive_key(&self, lower: &str, key: &str) -> Option<String> {
        if !key.starts_with("positive") {
            return None;
        }
        let tokens: Vec<&str> = lower
            .split(|c: char| !c.is_alphanumeric() && c != '\'')
            .filter(|t| !t.is_empty())
            .collect();
        const STRONG_NEG: &[&str] = &[
            "awful", "terrible", "horrible", "worst", "nightmare", "disaster",
            "furious", "livid", "devastated", "destroyed", "drained", "emptied",
        ];
        const MILD_NEG: &[&str] = &[
            "bad", "miserable", "depressing", "sucks", "hate", "despise",
        ];
        for &t in &tokens {
            if STRONG_NEG.iter().any(|&s| t == s) {
                return Some("negative_strong".to_string());
            }
        }
        for &t in &tokens {
            if MILD_NEG.iter().any(|&s| t == s) {
                return Some("negative_mild".to_string());
            }
        }
        if self.negative_anchor_tokens.iter().any(|neg| tokens.contains(&neg.as_str())) {
            return Some("negative_mild".to_string());
        }
        None
    }

    /// First-token affect floor: if the very first word is a strong emotional
    /// marker ("Furious:", "Devastated:", "Destroyed:"), the sentence cannot be
    /// neutral — floor it to `negative_strong`. Handles the common editorial
    /// pattern `Furious: <narrative>` where the lattice sees only the narrative
    /// body and misses the framing.
    pub fn first_token_affect_floor(&self, lower: &str, key: &str) -> Option<String> {
        if key.starts_with("negative") {
            return None;
        }
        let trimmed = lower.trim_start();
        const FLOOR_WORDS: &[&str] = &[
            "furious", "livid", "enraged", "outraged", "devastated", "destroyed",
            "heartbroken", "appalled", "disgusted", "gutted", "crushed",
        ];
        for &word in FLOOR_WORDS {
            if trimmed.starts_with(word) {
                let rest = &trimmed[word.len()..];
                if rest.is_empty()
                    || rest.starts_with(':')
                    || rest.starts_with(',')
                    || rest.starts_with(' ')
                    || rest.starts_with('.')
                {
                    return Some("negative_strong".to_string());
                }
            }
        }
        None
    }

    /// Desk-relief cap: prevents sentences that describe a scare followed by
    /// stability ("bid dropped … but the level held and nothing swept") from
    /// being classified as POSITIVE (strong). The outcome is neutral — the desk
    /// dodged a bullet but didn't gain. Cap to `neutral`.
    pub fn desk_relief_caps_positive(&self, lower: &str, key: &str) -> Option<String> {
        if key != "positive_strong" {
            return None;
        }
        const SCARE_TOKENS: &[&str] = &[
            "dropped", "fell", "slipped", "dipped", "declined", "plunged", "sank",
            "tanked", "cratered", "tumbled", "slid",
        ];
        const HOLD_PHRASES: &[&str] = &[
            "held", "nothing swept", "didn't break", "didn't move",
            "level held", "recovered", "came back", "bounced",
            "stabilized", "stabilised", "no follow-through",
        ];
        let has_scare = SCARE_TOKENS.iter().any(|s| lower.contains(s));
        let has_hold = HOLD_PHRASES.iter().any(|h| lower.contains(h));
        if has_scare && has_hold {
            Some("neutral".to_string())
        } else {
            None
        }
    }

    /// Headline confidence floor: when the text is third-person, very short,
    /// contains **no** evaluative anchor token, and has no narrative/experiential
    /// markers, cap any polar key back to `neutral`. Targets dry wire headlines
    /// like "MUFG modernises US ACH payments tech" or "Tabby obtains SVF licence."
    ///
    /// Conservative: only fires on text under 80 chars with no experiential cues.
    pub fn headline_neutral_floor(&self, lower: &str, key: &str) -> Option<String> {
        if key == "neutral" || key == "mixed" || key == "sarcastic" {
            return None;
        }
        if Self::looks_like_first_person_finance_user(lower) {
            return None;
        }
        if lower.len() > 80 {
            return None;
        }
        let tokens: Vec<&str> = lower
            .split(|c: char| !c.is_alphanumeric() && c != '\'')
            .filter(|t| !t.is_empty())
            .collect();
        let has_pos = self.positive_anchor_tokens.iter().any(|t| {
            tokens.contains(&t.as_str())
        });
        let has_neg = self.negative_anchor_tokens.iter().any(|t| {
            tokens.contains(&t.as_str())
        });
        if has_pos || has_neg {
            return None;
        }
        const EXPERIENTIAL_CUES: &[&str] = &[
            "spiked", "crashed", "failed", "stuck", "stressed", "destroyed",
            "furious", "nightmare", "doubled", "tripled", "halted", "expired",
            "slashed", "rejected", "plunged", "collapsed", "froze", "frozen",
            "wiped", "drained", "gutted", "priced out", "headcount",
        ];
        if EXPERIENTIAL_CUES.iter().any(|c| lower.contains(c)) {
            return None;
        }
        Some("neutral".to_string())
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
        let lower = Self::normalize_rules_text(intent_text);
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

    /// Apply degree-modifier promotion/demotion to a polarity key.
    /// "deeply disappointed" → promotes `negative_mild` to `negative_strong`.
    /// "slightly annoyed" → keeps `negative_mild` (no change).
    /// Only operates on `_mild`/`_strong` pairs; leaves `neutral`, `mixed`,
    /// `sarcastic` untouched.
    pub fn apply_degree_modifiers(&self, lower: &str, key: &str) -> String {
        const INTENSIFIERS: &[&str] = &[
            "deeply", "profoundly", "utterly", "absolutely", "completely",
            "thoroughly", "extremely", "incredibly", "exceptionally",
            "overwhelmingly", "insanely", "seriously", "genuinely",
        ];
        const DIMINISHERS: &[&str] = &[
            "slightly", "somewhat", "a bit", "a little", "mildly",
            "kind of", "kinda", "sort of", "sorta", "fairly",
        ];

        let has_intensifier = INTENSIFIERS.iter().any(|m| lower.contains(m));
        let has_diminisher = DIMINISHERS.iter().any(|m| lower.contains(m));

        if has_intensifier && !has_diminisher {
            match key {
                "negative_mild" => return "negative_strong".to_string(),
                "positive_mild" => return "positive_strong".to_string(),
                _ => {}
            }
        }
        if has_diminisher && !has_intensifier {
            match key {
                "negative_strong" => return "negative_mild".to_string(),
                "positive_strong" => return "positive_mild".to_string(),
                _ => {}
            }
        }
        key.to_string()
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

    /// Detects inputs with no evaluative or substantive content — bare
    /// timestamps, location-only stubs, metadata fragments, and other
    /// zero-content lines that should never be forced into a sentiment label.
    pub fn is_out_of_scope(&self, lower: &str) -> bool {
        if self.has_clear_evaluative_stance(lower) {
            return false;
        }
        // Strip whitespace / punctuation to get raw word tokens.
        let tokens: Vec<&str> = lower.split_whitespace().collect();
        if tokens.is_empty() {
            return true;
        }
        // Very short inputs that are purely time/location metadata.
        // "Middle Eastern Headlines at 5:44am GMT" — 6 tokens, no evaluative.
        if tokens.len() <= 10 && is_bare_metadata(lower) {
            return true;
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
                    if w == "like"
                        && (self.like_token_is_perception_idiom(lower)
                            || self.like_token_is_comparison(lower))
                    {
                        continue;
                    }
                    if self.weak_positive_token_in_contradicted_context(w, lower) {
                        continue;
                    }
                    if tokens.contains(w.as_str()) {
                        if let Some(pos) = lower.find(w.as_str()) {
                            if is_in_negation_window(lower, pos, 6) {
                                continue;
                            }
                        }
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
                        if let Some(pos) = lower.find(w.as_str()) {
                            if is_in_negation_window(lower, pos, 6) {
                                continue;
                            }
                        }
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

    /// True when the text contains at least one **praise-surface** token (love,
    /// great, best, wonderful, amazing, etc.) — used by the sarcasm validation
    /// gate to confirm that a sarcasm routing actually has praise/harm mismatch.
    pub fn text_has_praise_surface_token(&self, lower: &str) -> bool {
        const PRAISE_SURFACE: &[&str] = &[
            "love", "great", "amazing", "wonderful", "fantastic", "excellent",
            "perfect", "beautiful", "incredible", "best", "brilliant", "awesome",
            "good", "fine", "nice", "superb", "flawless",
        ];
        let tokens: std::collections::HashSet<&str> = lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .collect();
        PRAISE_SURFACE.iter().any(|p| tokens.contains(p))
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
                Ok(doc) => {
                    crate::infer_trace!(
                        "  [inference-toml] primary {} → {} lattice_misfire, {} lattice_misfire_fallback rows",
                        p.display(),
                        doc.rules.lattice_misfire.len(),
                        doc.rules.lattice_misfire_fallback.len()
                    );
                    return Ok(doc);
                }
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

/// Embedded core document for use as a merge baseline in [`reload_inference_toml_from_str`].
#[cfg(target_arch = "wasm32")]
fn wasm_core_document() -> InferenceTomlDocument {
    const WASM_EMBED: &str = include_str!("../../data/sentiment/inference_sentiment_core.toml");
    toml::from_str(WASM_EMBED).expect("wasm embedded inference TOML invalid")
}

/// Single load: thresholds from file + merged rule lists.
#[derive(Debug)]
pub struct LoadedInferenceToml {
    pub thresholds: InferenceThresholds,
    rules: Arc<InferenceRulesRuntime>,
    /// Serializable snapshot of the merged rules (for re-embedding in brain packages).
    pub rules_section: InferenceRulesSection,
    /// JSONL guardrails merged after TOML (native only; empty on wasm).
    pub guardrails: super::inference_guardrails::GuardrailsDiskSummary,
    /// Generation config from `[generation]` section.
    pub generation: GenerationConfig,
    /// Response shaping rules from `[response_shaping]` section.
    pub response_shaping: ResponseShapingConfig,
    /// Validation pipeline config from `[validation]` section.
    pub validation: ValidationConfig,
    /// Typed fragment composition policy from `[fragment_compose]`.
    pub fragment_compose: FragmentComposeConfig,
}

impl LoadedInferenceToml {
    pub fn rules(&self) -> Arc<InferenceRulesRuntime> {
        self.rules.clone()
    }
    pub fn generation_config(&self) -> &GenerationConfig {
        &self.generation
    }
    pub fn response_shaping(&self) -> &ResponseShapingConfig {
        &self.response_shaping
    }
    pub fn validation_config(&self) -> &ValidationConfig {
        &self.validation
    }
    pub fn fragment_compose(&self) -> &FragmentComposeConfig {
        &self.fragment_compose
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

// ---------------------------------------------------------------------------
// Native: RwLock so load_brain can replace with brain-embedded rules.
// ---------------------------------------------------------------------------
#[cfg(not(target_arch = "wasm32"))]
static FULL: std::sync::RwLock<Option<Arc<LoadedInferenceToml>>> =
    std::sync::RwLock::new(None);

#[cfg(not(target_arch = "wasm32"))]
fn build_native_default() -> Arc<LoadedInferenceToml> {
    let file = load_inference_toml_merged();
    let rules_section = file.rules.clone();
    let mut rules = InferenceRulesRuntime::from_section(file.rules);
    let guardrails = crate::inference::inference_guardrails::merge_guardrails_into_runtime(
        &mut rules.headline_lexical_topic,
        &mut rules.lattice_misfire,
    );
    let rules = Arc::new(rules);
    Arc::new(LoadedInferenceToml {
        thresholds: file.thresholds,
        rules,
        rules_section,
        guardrails,
        generation: file.generation,
        response_shaping: file.response_shaping,
        validation: file.validation,
        fragment_compose: file.fragment_compose,
    })
}

/// Replace the active native inference TOML from a provided string (parity with the
/// wasm `reload_inference_toml_from_str`). Useful for hosts/tests that inject a TOML
/// document directly instead of via `--inference-toml` discovery.
#[cfg(not(target_arch = "wasm32"))]
pub fn reload_inference_toml_from_str(toml_str: &str) -> Result<(), String> {
    let mut file: InferenceTomlDocument =
        toml::from_str(toml_str).map_err(|e| format!("invalid inference TOML: {}", e))?;
    let defaults = baseline_rules_section();
    file.rules = file.rules.merge_empty_from(&defaults);
    let rules_section = file.rules.clone();
    let mut rules = InferenceRulesRuntime::from_section(file.rules);
    let guardrails = crate::inference::inference_guardrails::merge_guardrails_into_runtime(
        &mut rules.headline_lexical_topic,
        &mut rules.lattice_misfire,
    );
    let loaded = Arc::new(LoadedInferenceToml {
        thresholds: file.thresholds,
        rules: Arc::new(rules),
        rules_section,
        guardrails,
        generation: file.generation,
        response_shaping: file.response_shaping,
        validation: file.validation,
        fragment_compose: file.fragment_compose,
    });
    let mut guard = FULL.write().unwrap();
    *guard = Some(loaded);
    Ok(())
}

/// Rebuild the native inference TOML cache from disk paths (CLI / env / discovery).
/// Call after [`crate::service::LanguageService::load_brain`] when a brain's embedded
/// `plugins_blob` has overwritten rules but `--project` / `--inference-toml` should win for local dev.
#[cfg(not(target_arch = "wasm32"))]
pub fn force_native_inference_rebuild_from_disk() {
    let loaded = build_native_default();
    let rules = loaded.rules();
    crate::infer_trace!(
        "  [inference] disk reload: {} lattice_misfire, {} lattice_misfire_fallback rows",
        rules.lattice_misfire.len(),
        rules.lattice_misfire_fallback.len()
    );
    let mut guard = FULL.write().unwrap();
    *guard = Some(loaded);
}

#[cfg(not(target_arch = "wasm32"))]
pub fn inference_toml_loaded() -> Arc<LoadedInferenceToml> {
    {
        let guard = FULL.read().unwrap();
        if let Some(ref cached) = *guard {
            return cached.clone();
        }
    }
    let loaded = build_native_default();
    let mut guard = FULL.write().unwrap();
    if guard.is_none() {
        *guard = Some(loaded.clone());
    }
    guard.as_ref().unwrap().clone()
}

// ---------------------------------------------------------------------------
// WASM: resettable RefCell so JS hosts can swap domain TOML at runtime.
// ---------------------------------------------------------------------------
#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;

#[cfg(target_arch = "wasm32")]
thread_local! {
    static WASM_FULL: RefCell<Option<Arc<LoadedInferenceToml>>> = RefCell::new(None);
}

#[cfg(target_arch = "wasm32")]
fn build_default_loaded() -> Arc<LoadedInferenceToml> {
    let file = load_inference_toml_merged();
    let rules_section = file.rules.clone();
    let rules = Arc::new(InferenceRulesRuntime::from_section(file.rules));
    let guardrails = super::inference_guardrails::GuardrailsDiskSummary::default();
    Arc::new(LoadedInferenceToml {
        thresholds: file.thresholds,
        rules,
        rules_section,
        guardrails,
        generation: file.generation,
        response_shaping: file.response_shaping,
        validation: file.validation,
        fragment_compose: file.fragment_compose,
    })
}

#[cfg(target_arch = "wasm32")]
pub fn inference_toml_loaded() -> Arc<LoadedInferenceToml> {
    WASM_FULL.with(|cell| {
        let mut opt = cell.borrow_mut();
        if let Some(ref cached) = *opt {
            return cached.clone();
        }
        let loaded = build_default_loaded();
        *opt = Some(loaded.clone());
        loaded
    })
}

/// Replace the active inference TOML on WASM with a domain-specific document.
///
/// The provided `toml_str` (e.g. the contents of `inference_fintech.toml`) is
/// parsed, then its rules are merged with the embedded core baseline using
/// [`merge_empty_from`] — exactly mirroring the native CLI merge path.
#[cfg(target_arch = "wasm32")]
pub fn reload_inference_toml_from_str(toml_str: &str) -> Result<(), String> {
    let domain: InferenceTomlDocument =
        toml::from_str(toml_str).map_err(|e| format!("invalid inference TOML: {}", e))?;
    let core = wasm_core_document();
    let merged_rules = domain.rules.merge_empty_from(&core.rules);
    let rules_section = merged_rules.clone();
    let rules = Arc::new(InferenceRulesRuntime::from_section(merged_rules));
    let guardrails = super::inference_guardrails::GuardrailsDiskSummary::default();
    let loaded = Arc::new(LoadedInferenceToml {
        thresholds: domain.thresholds,
        rules,
        rules_section,
        guardrails,
        generation: domain.generation,
        response_shaping: domain.response_shaping,
        validation: domain.validation,
        fragment_compose: domain.fragment_compose,
    });
    WASM_FULL.with(|cell| {
        *cell.borrow_mut() = Some(loaded);
    });
    Ok(())
}

pub fn inference_rules_runtime() -> Arc<InferenceRulesRuntime> {
    inference_toml_loaded().rules()
}

/// Replace the global inference rules with brain-embedded rules.
/// Called from [`crate::service::LanguageService::load_brain`] when the brain
/// package carries a `[rules]` section in its plugins manifest.
pub fn replace_loaded_from_rules_section(
    section: InferenceRulesSection,
    thresholds: InferenceThresholds,
) {
    let rules_section = section.clone();
    let rules = Arc::new(InferenceRulesRuntime::from_section(section));
    let guardrails = super::inference_guardrails::GuardrailsDiskSummary::default();
    let fragment_compose = {
        #[cfg(not(target_arch = "wasm32"))]
        {
            FULL.read()
                .ok()
                .and_then(|g| g.as_ref().map(|l| l.fragment_compose.clone()))
                .unwrap_or_default()
        }
        #[cfg(target_arch = "wasm32")]
        {
            WASM_FULL.with(|cell| {
                cell.borrow()
                    .as_ref()
                    .map(|l| l.fragment_compose.clone())
                    .unwrap_or_default()
            })
        }
    };
    let loaded = Arc::new(LoadedInferenceToml {
        thresholds,
        rules,
        rules_section,
        guardrails,
        generation: GenerationConfig::default(),
        response_shaping: ResponseShapingConfig::default(),
        validation: ValidationConfig::default(),
        fragment_compose,
    });

    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut guard = FULL.write().unwrap();
        *guard = Some(loaded);
    }

    #[cfg(target_arch = "wasm32")]
    {
        WASM_FULL.with(|cell| {
            *cell.borrow_mut() = Some(loaded);
        });
    }
}

#[cfg(test)]
mod fragment_compose_tests {
    use super::*;

    fn luna_like_config() -> FragmentComposeConfig {
        FragmentComposeConfig {
            enabled: true,
            vocalizations: vec!["mrrp".into(), "chirp".into(), "purr".into()],
            vocal_coda_modifiers: vec!["now".into()],
            greeting_exact: vec!["hi".into(), "hello".into(), "hey".into()],
            agent_name_prefixes: vec!["hey ".into(), "hi ".into()],
            agent_name_greeting_max_len: 48,
            intent_rules: vec![
                FragmentIntentRuleToml {
                    id: "greeting".into(),
                    intent: "greeting_check_in".into(),
                    anchors: vec!["greeting_check_in".into()],
                    min_voices: 1,
                    relaxed_parts: true,
                    r#match: "greeting".into(),
                    patterns: vec![],
                    max_len: None,
                },
                FragmentIntentRuleToml {
                    id: "agent".into(),
                    intent: "greeting_check_in".into(),
                    anchors: vec![],
                    min_voices: 1,
                    relaxed_parts: true,
                    r#match: "agent_name_greeting".into(),
                    patterns: vec![],
                    max_len: None,
                },
                FragmentIntentRuleToml {
                    id: "meal".into(),
                    intent: "mealtime_request".into(),
                    anchors: vec!["treat".into()],
                    min_voices: 1,
                    relaxed_parts: true,
                    r#match: "contains_any".into(),
                    patterns: vec!["treat".into()],
                    max_len: None,
                },
                FragmentIntentRuleToml {
                    id: "default".into(),
                    intent: "open_ended_chat".into(),
                    anchors: vec![],
                    min_voices: 2,
                    relaxed_parts: false,
                    r#match: "fallback".into(),
                    patterns: vec![],
                    max_len: None,
                },
            ],
            ..Default::default()
        }
    }

    #[test]
    fn match_intent_greeting_and_agent_name() {
        let cfg = luna_like_config();
        let h = cfg.match_intent("Hey there", "");
        assert_eq!(h.intent, "greeting_check_in");
        let h2 = cfg.match_intent("hey luna", "Luna");
        assert_eq!(h2.intent, "greeting_check_in");
    }

    #[test]
    fn match_intent_mealtime() {
        let cfg = luna_like_config();
        let h = cfg.match_intent("Want a treat?", "Luna");
        assert_eq!(h.intent, "mealtime_request");
    }

    #[test]
    fn vocalization_tail_from_config() {
        let cfg = luna_like_config();
        assert!(cfg.vocalization_tail_suspicious(
            "There you are. Mrrp greeting."
        ));
        assert!(!cfg.vocalization_tail_suspicious(
            "Chirp alert. I sit. Mrrp."
        ));
        assert!(!cfg.vocalization_tail_suspicious(
            "Kitchen. Mrrp now."
        ));
    }

    #[test]
    fn is_pure_vocal_coda_from_config() {
        let cfg = luna_like_config();
        assert!(cfg.is_pure_vocal_coda("Mrrp."));
        assert!(cfg.is_pure_vocal_coda("Mrrp now."));
        assert!(!cfg.is_pure_vocal_coda("Mrrp greeting."));
        assert!(cfg.starts_with_vocalization("Chirp alert."));
    }

    #[test]
    fn prompt_intent_override_skips_fallback() {
        let cfg = luna_like_config();
        assert!(cfg.prompt_intent_override("want a treat", "Luna").is_some());
        assert!(cfg.prompt_intent_override("random question", "Luna").is_none());
    }

    #[test]
    fn classify_voice_from_config() {
        let cfg = FragmentComposeConfig {
            decompose: FragmentDecomposeConfig {
                opener_prefixes: vec!["there you are".into()],
                drive_override_keywords: vec!["treat".into()],
                drive_keywords: vec!["hungry".into(), "stomach".into()],
                activity_keywords: vec!["chase".into(), "pounce".into()],
                identity_keywords: vec!["blink".into()],
                ..Default::default()
            },
            ..luna_like_config()
        };
        assert_eq!(cfg.classify_voice("My stomach has opinions.", "body"), "drive");
        assert_eq!(cfg.classify_voice("I pounce on the pen.", "body"), "activity");
        assert_eq!(cfg.classify_voice("I blink slow at you.", "body"), "identity");
        assert_eq!(cfg.classify_voice("Trill.", "coda"), "identity");
        assert!(cfg.is_opener("there you are, human"));
    }

    #[test]
    fn parses_fragment_compose_section() {
        let raw = r#"
mode = "chat"
[fragment_compose]
enabled = true
library = "fragments.jsonl"
vocalizations = ["mrrp", "chirp"]
[[fragment_compose.intent_rules]]
intent = "greeting_check_in"
match = "fallback"
"#;
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("fragment_compose TOML");
        assert!(doc.fragment_compose.enabled);
        assert_eq!(doc.fragment_compose.vocalizations.len(), 2);
    }
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

    #[test]
    fn reference_headline_sunset_average_instagram_neutral() {
        let raw = include_str!("../../data/sentiment/inference_sentiment_reference.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("fixture TOML");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let h = "The sunset was average. Just orange. Not everything has to be Instagram-worthy.";
        assert_eq!(rules.sentiment_lexical_topic_key(h).as_deref(), Some("neutral"));
    }

    #[test]
    fn core_headline_sunset_average_instagram_neutral() {
        let raw = include_str!("../../data/sentiment/inference_sentiment_core.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("fixture TOML");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let h = "The sunset was average. Just orange. Not everything has to be Instagram-worthy.";
        assert_eq!(rules.sentiment_lexical_topic_key(h).as_deref(), Some("neutral"));
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
            !acc.headline_lexical_topic.is_empty(),
            "core pack should ship minimal headline_lexical_topic rows"
        );
        let n_core_headlines = acc.headline_lexical_topic.len();
        acc = acc.merge_empty_from(&fintech.rules);
        assert!(
            acc.headline_lexical_topic.len() > n_core_headlines,
            "fintech headline rows should append after core headlines"
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
        let raw = include_str!("../../data/crypto/inference_crypto.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("crypto fixture");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let h = "Why XRP Is Gaining Today";
        assert_eq!(rules.sentiment_lexical_topic_key(h).as_deref(), Some("neutral"));
    }

    #[test]
    fn kitsu_inference_pets_toml_parses_and_detects_asakusa_bleed() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
            "../../spacekit/spacekit-projects/companions/kitsu/data/inference_pets.toml",
        );
        if !path.is_file() {
            eprintln!("skip kitsu fixture (path missing): {}", path.display());
            return;
        }
        let raw = std::fs::read_to_string(&path).expect("read kitsu inference_pets.toml");
        let doc: InferenceTomlDocument = toml::from_str(&raw).expect("parse kitsu inference_pets.toml");
        assert!(
            doc.rules.lattice_misfire.len() >= 10,
            "expected pet lattice_misfire rows, got {}",
            doc.rules.lattice_misfire.len()
        );
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        assert!(rules.lattice_response_misfire_hit(
            "will u be my friend?",
            "Asakusa, Tokyo. Narrow streets, shrine bells, food stalls — locals called me the little fox."
        ));
        let fb = rules.lattice_misfire_fallback_line("will u be my friend?");
        assert!(
            fb.as_ref()
                .map(|(t, _)| t.contains("Already am"))
                .unwrap_or(false),
            "expected bonding fallback, got {:?}",
            fb
        );
    }

    #[test]
    fn lattice_misfire_intent_exclude_and_response_driven_bleed() {
        let rules = InferenceRulesRuntime::from_section(InferenceRulesSection {
            lattice_misfire: vec![
                LatticeMisfireRule {
                    intent: vec![],
                    intent_exclude: vec![vec!["school".to_string()]],
                    response_any: vec!["school is loud".to_string()],
                    response: vec![],
                    prior_response_any: vec![],
                },
                LatticeMisfireRule {
                    intent: vec![vec!["who are you".to_string()]],
                    intent_exclude: vec![vec!["like to be called".to_string()]],
                    response_any: vec!["old wyrm".to_string()],
                    response: vec![],
                    prior_response_any: vec![],
                },
            ],
            lattice_misfire_fallback: vec![LatticeMisfireFallbackRule {
                intent: vec![vec!["i feel sad".to_string()]],
                intent_exclude: vec![],
                response: "Grounding line.".to_string(),
                template_id: "grounding_fallback".to_string(),
            }],
            ..InferenceRulesSection::default()
        });
        assert!(rules.lattice_response_misfire_hit(
            "come here",
            "School is loud in thy head."
        ));
        assert!(!rules.lattice_response_misfire_hit(
            "school was awful today",
            "School is loud in thy head."
        ));
        assert!(rules.lattice_response_misfire_hit(
            "who are you?",
            "Some call me old wyrm."
        ));
        assert!(!rules.lattice_response_misfire_hit(
            "what do you like to be called",
            "Some call me old wyrm."
        ));
        let fb = rules.lattice_misfire_fallback_line("I feel sad");
        assert_eq!(fb.as_ref().map(|(t, _)| t.as_str()), Some("Grounding line."));
        assert_eq!(
            fb.as_ref().map(|(_, id)| id.as_str()),
            Some("grounding_fallback")
        );
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
        let raw = include_str!("../../data/crypto/inference_crypto.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("crypto fixture");
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
    fn anchor_skips_like_in_feels_like_idiom() {
        let raw = include_str!("../../data/sentiment/inference_sentiment_reference.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("fixture TOML");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let lower = InferenceRulesRuntime::normalize_rules_text(
            "Nothing happened today. Genuinely nothing. And somehow that feels like exactly what I needed.",
        );
        assert!(rules.anchor_phrase(&lower, "positive_mild").is_none());
    }

    #[test]
    fn lexical_polarity_long_mixed_rows_match_core_phrases() {
        let raw = include_str!("../../data/sentiment/inference_sentiment_core.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("core TOML");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let acting = InferenceRulesRuntime::normalize_rules_text(
            "The acting was fine. The script was fine. The direction was fine. Nothing about it was memorable.",
        );
        assert_eq!(rules.lexical_polarity_signal(&acting).as_deref(), Some("mixed"));
        let zec = InferenceRulesRuntime::normalize_rules_text(
            "ZEC is either misunderstood or obsolete — I can't tell which anymore.",
        );
        assert_eq!(rules.lexical_polarity_signal(&zec).as_deref(), Some("mixed"));
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

    #[test]
    fn crypto_headline_funds_returned_victims_has_rule() {
        let raw = include_str!("../../data/crypto/inference_crypto.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("crypto fixture");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let h = "Funds returned to Florida victims after record-breaking cryptocurrency fraud operation";
        let result = rules.sentiment_lexical_topic_key(h);
        assert!(result.is_some(), "should match a headline rule");
    }

    #[test]
    fn crypto_headline_apple_pay_moment_maps_positive() {
        let raw = include_str!("../../data/crypto/inference_crypto.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("crypto fixture");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let h = "The 'Apple Pay' Moment for Web3: Mixin Integrates Coinbase to Make Fiat-to-Crypto Faster Than a Text Message";
        let result = rules.sentiment_lexical_topic_key(h);
        assert_eq!(result.as_deref(), Some("positive_mild"), "Apple Pay integration should be positive_mild, got {:?}", result);
    }

    #[test]
    fn crypto_headline_apple_pay_moment_maps_positive_merged() {
        let raw_crypto = include_str!("../../data/crypto/inference_crypto.toml");
        let raw_core = include_str!("../../data/sentiment/inference_sentiment_core.toml");
        let crypto_doc: InferenceTomlDocument = toml::from_str(raw_crypto).expect("crypto fixture");
        let core_doc: InferenceTomlDocument = toml::from_str(raw_core).expect("core fixture");
        let merged_rules = crypto_doc.rules.merge_empty_from(&core_doc.rules);
        let rules = InferenceRulesRuntime::from_section(merged_rules);
        let h = "The 'Apple Pay' Moment for Web3: Mixin Integrates Coinbase to Make Fiat-to-Crypto Faster Than a Text Message";
        let result = rules.sentiment_lexical_topic_key(h);
        assert_eq!(result.as_deref(), Some("positive_mild"), "Apple Pay (merged) should be positive_mild, got {:?}", result);
    }

    #[test]
    fn crypto_headline_scam_loss_advocacy_maps_mixed() {
        let raw = include_str!("../../data/crypto/inference_crypto.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("crypto fixture");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let h = "Wisconsin woman turns $4,400 crypto scam loss into advocacy";
        let result = rules.sentiment_lexical_topic_key(h);
        assert_eq!(result.as_deref(), Some("mixed"), "scam loss + advocacy should be mixed, got {:?}", result);
    }

    #[test]
    fn crypto_headline_rogue_ai_maps_negative_mild() {
        let raw = include_str!("../../data/crypto/inference_crypto.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("crypto fixture");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let h = "A Rogue AI Agent Started Mining Crypto, Which Left Scientists Concerned - SlashGear";
        let result = rules.sentiment_lexical_topic_key(h);
        assert_eq!(result.as_deref(), Some("negative_mild"), "rogue AI concern should be negative_mild, got {:?}", result);
    }

    #[test]
    fn crypto_headline_billions_theft_maps_negative_strong() {
        let raw = include_str!("../../data/crypto/inference_crypto.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("crypto fixture");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let h = "The $46 Million Government Crypto Theft That Put Billions at Risk";
        let result = rules.sentiment_lexical_topic_key(h);
        assert_eq!(result.as_deref(), Some("negative_strong"), "systemic theft + billions should be negative_strong, got {:?}", result);
    }

    #[test]
    fn crypto_headline_winds_down_new_gig_maps_mixed() {
        let raw = include_str!("../../data/crypto/inference_crypto.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("crypto fixture");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let h = "Crypto hedge fund Split Capital winds down as its founder nabs new gig as an exec at stablecoin startup Plasma | Fortune";
        let result = rules.sentiment_lexical_topic_key(h);
        assert_eq!(result.as_deref(), Some("mixed"), "winds down + nabs new gig should be mixed, got {:?}", result);
    }

    #[test]
    fn crypto_headline_401k_crypto_maps_positive_mild() {
        let raw = include_str!("../../data/crypto/inference_crypto.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("crypto fixture");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let h = "US Labor Department Proposes Opening 401(k) Plans to Crypto";
        let result = rules.sentiment_lexical_topic_key(h);
        assert_eq!(result.as_deref(), Some("positive_mild"), "401k + crypto opening should be positive_mild, got {:?}", result);
    }

    #[test]
    fn crypto_headline_misled_investors_maps_negative_mild() {
        let raw = include_str!("../../data/crypto/inference_crypto.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("crypto fixture");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let h = "Crypto Executive Misled Investors on Digital Currency, SEC Says";
        let result = rules.sentiment_lexical_topic_key(h);
        assert_eq!(result.as_deref(), Some("negative_mild"), "misled investors should be negative_mild, got {:?}", result);
    }

    #[test]
    fn crypto_headline_incentive_program_maps_positive_mild() {
        let raw = include_str!("../../data/crypto/inference_crypto.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("crypto fixture");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let h = "Crypto-powered incentive program introduced by Dallas homebuilder";
        let result = rules.sentiment_lexical_topic_key(h);
        assert_eq!(result.as_deref(), Some("positive_mild"), "incentive program should be positive_mild, got {:?}", result);
    }

    #[test]
    fn crypto_headline_deal_lockup_maps_neutral() {
        let raw = include_str!("../../data/crypto/inference_crypto.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("crypto fixture");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let h = "Payward's $550M Bitnomial deal aims to lock up U.S. crypto derivatives plumbing";
        let result = rules.sentiment_lexical_topic_key(h);
        assert_eq!(result.as_deref(), Some("neutral"), "M&A deal should be neutral, got {:?}", result);
    }

    #[test]
    fn crypto_headline_deal_lockup_maps_neutral_merged() {
        let raw_crypto = include_str!("../../data/crypto/inference_crypto.toml");
        let raw_core = include_str!("../../data/sentiment/inference_sentiment_core.toml");
        let crypto_doc: InferenceTomlDocument = toml::from_str(raw_crypto).expect("crypto fixture");
        let core_doc: InferenceTomlDocument = toml::from_str(raw_core).expect("core fixture");
        let merged_rules = crypto_doc.rules.merge_empty_from(&core_doc.rules);
        let rules = InferenceRulesRuntime::from_section(merged_rules);
        let h = "Payward's $550M Bitnomial deal aims to lock up U.S. crypto derivatives plumbing";
        let result = rules.sentiment_lexical_topic_key(h);
        assert_eq!(result.as_deref(), Some("neutral"), "M&A deal (merged, raw) should be neutral, got {:?}", result);
        // Intent parser normalizes $550M → money_usd_550000000; verify the rule still fires.
        let h_norm = "payward's money_usd_550000000 bitnomial deal aims to lock up u.s. crypto derivatives plumbing";
        let result_norm = rules.sentiment_lexical_topic_key(h_norm);
        assert_eq!(result_norm.as_deref(), Some("neutral"), "M&A deal (merged, intent-normalized) should be neutral, got {:?}", result_norm);
    }

    #[test]
    fn crypto_headline_russia_bill_criminalize_maps_neutral() {
        let raw = include_str!("../../data/crypto/inference_crypto.toml");
        let doc: InferenceTomlDocument = toml::from_str(raw).expect("crypto fixture");
        let rules = InferenceRulesRuntime::from_section(doc.rules);
        let h = "Russia Introduces Bill To Criminalize Unregistered Crypto Services";
        let result = rules.sentiment_lexical_topic_key(h);
        assert_eq!(result.as_deref(), Some("neutral"), "legislative bill should be neutral, got {:?}", result);
    }
}
