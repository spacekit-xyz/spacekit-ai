//! Topic-lattice shortcut plugin: user-anchored copy, weak meta-GK guard, keyword lists from TOML.
//! Thresholds: `[sentiment]` in brain `plugins_blob` (serde name) or embedded inference TOML.
//! Rules: `GROWFORMER_INFERENCE_TOML` / legacy env, or embedded default file.

use std::collections::HashSet;

use crate::brain::BrainPackageHeader;
use crate::dimension::DimensionManager;
use crate::growformer_lang::MetaConcept;
use crate::micro_brain::MetaResult;

use crate::inference::harness::{BrainInferencePlugin, GenerationPreemptOutcome, TemplatePostprocessFlags};
use crate::inference::inference_toml::inference_rules_runtime;
use crate::inference::manifest::{resolved_inference_thresholds, BrainPluginsManifest, InferenceThresholds};

/// `GeneratedResponse.template_id` when the user-anchored path is used.
pub const TEMPLATE_ID_USER_ANCHORED: &str = "sentiment_user_anchored";

/// Seven-topic sub-lattices (see checkpoint `topic_subindex` / training data layout).
pub const TOPIC_KEYS: &[&str] = &[
    "positive_mild",
    "negative_mild",
    "neutral",
    "negative_strong",
    "sarcastic",
    "positive_strong",
    "mixed",
];

/// True when the checkpoint exposes exactly the standard seven-topic sub-lattice in one group.
pub fn is_lattice_shape(dm: &DimensionManager) -> bool {
    if dm.group_gen_envs.len() != 1 {
        return false;
    }
    let Some(env) = dm.group_gen_envs.values().next() else {
        return false;
    };
    if env.topic_subindex.len() != TOPIC_KEYS.len() {
        return false;
    }
    let names: HashSet<String> = env
        .topic_subindex
        .iter()
        .map(|t| t.topic_name.to_ascii_lowercase())
        .collect();
    TOPIC_KEYS
        .iter()
        .all(|k| names.contains(&k.to_string()))
}

/// Lattice shortcuts require matching shape and are not explicitly disabled in the brain header.
pub fn shortcuts_enabled(dm: &DimensionManager, inference_profile: Option<&str>) -> bool {
    if !is_lattice_shape(dm) {
        return false;
    }
    match inference_profile.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("off") | Some("none") | Some("disabled") => false,
        _ => true,
    }
}

/// Ignore weak GeneralKnowledge meta projection for conditioning when thresholds allow it.
pub fn should_skip_weak_gk_for_meta_conditioning(
    dm: &DimensionManager,
    inference_profile: Option<&str>,
    thresholds_from_manifest: Option<&InferenceThresholds>,
    concept: MetaConcept,
    margin: f32,
    confidence: f32,
) -> bool {
    if !shortcuts_enabled(dm, inference_profile) {
        return false;
    }
    let cfg = resolved_inference_thresholds(thresholds_from_manifest);
    matches!(concept, MetaConcept::GeneralKnowledge)
        && margin < cfg.meta_gk_margin
        && confidence < cfg.meta_gk_confidence
}

fn label_header(topic_key: &str) -> &'static str {
    match topic_key {
        "positive_strong" => "POSITIVE (strong)",
        "positive_mild" => "POSITIVE (mild)",
        "negative_strong" => "NEGATIVE (strong)",
        "negative_mild" => "NEGATIVE (mild)",
        "neutral" => "NEUTRAL",
        "sarcastic" => "SARCASTIC",
        "mixed" => "MIXED",
        _ => "CLASS",
    }
}

fn normalize_match_text(text: &str) -> String {
    let mut s = text.to_lowercase();
    s = s.replace('\u{2019}', "'");
    s = s.replace('\u{2018}', "'");
    s
}

/// Prefer a user-anchored classification line over lattice paraphrase when shape/profile match
/// and MetaBrain topic/confidence allow it.
pub fn try_user_anchored_line(
    dm: &DimensionManager,
    inference_profile: Option<&str>,
    thresholds_from_manifest: Option<&InferenceThresholds>,
    intent_text: &str,
    meta_result: Option<&MetaResult>,
    topic_hint: Option<&str>,
) -> Option<(String, f32)> {
    if !shortcuts_enabled(dm, inference_profile) {
        return None;
    }
    let mr = meta_result?;
    let th = topic_hint?;
    if !TOPIC_KEYS.iter().any(|k| th.eq_ignore_ascii_case(k)) {
        return None;
    }
    let cfg = resolved_inference_thresholds(thresholds_from_manifest);
    let rules = inference_rules_runtime();
    let lower = normalize_match_text(intent_text);

    if rules.is_objective_factual_statement(&lower) {
        let header = label_header("neutral");
        let body = "Measurable fact, status, or time — no evaluative opinion; read as NEUTRAL (not praise or complaint)";
        let trimmed = intent_text.trim();
        let excerpt = if trimmed.chars().count() > 200 {
            let head: String = trimmed.chars().take(200).collect();
            format!("{}…", head)
        } else {
            trimmed.to_string()
        };
        println!("  [lattice-direct] objective fact / status → NEUTRAL (bypass meta conf gate)");
        let text = format!(
            "{} — {}. Grounded in the user's own words: \"{}\"",
            header, body, excerpt
        );
        return Some((text, cfg.ambiguous_line_confidence));
    }

    let lex_polar = rules.lexical_polarity_signal(&lower);
    if mr.confidence < cfg.min_meta_confidence_user_anchored && lex_polar.is_none() {
        return None;
    }
    let contrast = rules.has_contrastive_marker(&lower);
    let bipolar = rules.has_bipolar_lexicon(&lower);
    let force_mixed = contrast && bipolar;

    let mut key: String = TOPIC_KEYS
        .iter()
        .find(|k| th.eq_ignore_ascii_case(k))
        .copied()
        .unwrap_or(th)
        .to_string();
    if force_mixed {
        key = "mixed".to_string();
    }

    let mut lexical_polarity_override = false;
    if !force_mixed {
        if let Some(k) = lex_polar {
            key = k;
            lexical_polarity_override = true;
        }
    }

    let mut sarcasm_template_override = false;
    if !force_mixed && key != "sarcastic" && rules.has_sarcasm_template(&lower) {
        key = "sarcastic".to_string();
        sarcasm_template_override = true;
    }

    let mut ambiguous_override = false;
    let mut ambiguous_disappointment = false;
    if !force_mixed && !sarcasm_template_override {
        if let Some(new_key) = rules.ambiguous_valence_retarget(&lower, key.as_str()) {
            ambiguous_disappointment = new_key == "negative_mild";
            ambiguous_override = true;
            key = new_key.to_string();
        }
    }

    let mut disappointment_override = false;
    if !force_mixed && !sarcasm_template_override && !ambiguous_override {
        if let Some(new_key) = rules.disappointment_positive_override(&lower, key.as_str()) {
            key = new_key.to_string();
            disappointment_override = true;
        }
    }

    let header = label_header(key.as_str());
    let trimmed = intent_text.trim();
    let excerpt = if trimmed.chars().count() > 200 {
        let head: String = trimmed.chars().take(200).collect();
        format!("{}…", head)
    } else {
        trimmed.to_string()
    };

    let (body, conf) = if key.as_str() == "mixed" {
        let explain = if contrast && bipolar {
            "Contrastive marker (e.g. 'but') with both laudatory and critical wording — dual valence (MIXED), not a single pole"
        } else {
            "Positive and negative cues both appear; overall read is MIXED"
        };
        (
            explain.to_string(),
            if force_mixed {
                cfg.mixed_override_confidence
            } else {
                cfg.default_line_confidence
            },
        )
    } else if key.as_str() == "sarcastic" && sarcasm_template_override {
        (
            "Laudatory or stoic wording clashes with an obvious grievance (waits, silence, failure) — read as SARCASTIC / ironic, not literal negativity or praise".to_string(),
            cfg.mixed_override_confidence,
        )
    } else if ambiguous_override {
        let explain = if ambiguous_disappointment {
            "Mild letdown: the line implies reality fell short of expectation — NEGATIVE (mild), not enthusiastic praise"
        } else {
            "Hedged, lukewarm phrasing (e.g. guess / suppose / nothing special / okay) — read as NEUTRAL / ambiguous, not strongly positive"
        };
        (explain.to_string(), cfg.ambiguous_line_confidence)
    } else if disappointment_override {
        (
            "Explicit disappointment or letdown — NEGATIVE (mild), not a positive read".to_string(),
            cfg.ambiguous_line_confidence,
        )
    } else if lexical_polarity_override {
        let explain = if key.as_str() == "positive_strong" {
            "Strong positive idiom (e.g. 'blew me away') — read as enthusiastic praise despite weak topic scores"
        } else {
            "Clear non-repeat / rejection stance (e.g. wouldn't buy again) — negative on the product or experience, not praise"
        };
        let c = if key.as_str() == "positive_strong" {
            cfg.default_line_confidence
        } else {
            cfg.ambiguous_line_confidence
        };
        (explain.to_string(), c)
    } else {
        let anchor = rules.anchor_phrase(&lower, key.as_str());
        let tone = match key.as_str() {
            "positive_strong" | "positive_mild" => "The overall tone reads as clearly positive",
            "negative_strong" | "negative_mild" => "The overall tone reads as clearly negative",
            "neutral" => "The overall tone reads as mostly neutral",
            "sarcastic" => "The line may use irony or surface/actual mismatch",
            _ => "Classification is non-obvious from surface text alone",
        };
        let b = match &anchor {
            Some(a) => format!("{}. {}", a, tone),
            None => tone.to_string(),
        };
        (b, cfg.default_line_confidence)
    };

    if force_mixed {
        println!("  [lattice-direct] contrastive bipolar → MIXED (override meta topic)");
    }
    if sarcasm_template_override {
        println!("  [lattice-direct] sarcasm template → SARCASTIC (override meta topic)");
    }
    if ambiguous_override {
        println!(
            "  [lattice-direct] hedged/ambiguous → {} (override meta topic)",
            key
        );
    }
    if disappointment_override {
        println!("  [lattice-direct] disappointment cue → NEGATIVE (mild) (override meta topic)");
    }
    if lexical_polarity_override {
        println!(
            "  [lattice-direct] lexical polarity → {} (override meta topic / bypass low conf)",
            key
        );
    }

    let text = format!(
        "{} — {}. Grounded in the user's own words: \"{}\"",
        header, body, excerpt
    );
    Some((text, conf))
}

/// Zero-sized handle; behavior is in this module and TOML-backed [`InferenceRulesRuntime`].
pub struct LatticeShortcutsPlugin;

impl BrainInferencePlugin for LatticeShortcutsPlugin {
    fn skip_weak_gk_for_meta_conditioning(
        &self,
        dm: &DimensionManager,
        inference_profile: Option<&str>,
        thresholds_from_manifest: Option<&InferenceThresholds>,
        concept: MetaConcept,
        margin: f32,
        confidence: f32,
    ) -> bool {
        should_skip_weak_gk_for_meta_conditioning(
            dm,
            inference_profile,
            thresholds_from_manifest,
            concept,
            margin,
            confidence,
        )
    }

    fn extend_subject_keywords(
        &self,
        dm: &DimensionManager,
        inference_profile: Option<&str>,
        intent_text: &str,
        subject_kw: &mut Vec<String>,
    ) {
        if !shortcuts_enabled(dm, inference_profile) {
            return;
        }
        for w in intent_text.split_whitespace() {
            let lw = w
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_ascii_lowercase();
            if lw.len() > 2 && !subject_kw.iter().any(|x| x == &lw) {
                subject_kw.push(lw);
            }
        }
    }

    fn try_preempt_generation(
        &self,
        dm: &DimensionManager,
        inference_profile: Option<&str>,
        thresholds_from_manifest: Option<&InferenceThresholds>,
        intent_text: &str,
        meta_result: Option<&MetaResult>,
        topic_hint: Option<&str>,
    ) -> Option<GenerationPreemptOutcome> {
        try_user_anchored_line(
            dm,
            inference_profile,
            thresholds_from_manifest,
            intent_text,
            meta_result,
            topic_hint,
        )
        .map(|(text, confidence)| GenerationPreemptOutcome {
            text,
            confidence,
            template_id: TEMPLATE_ID_USER_ANCHORED,
        })
    }

    fn template_postprocess_flags(&self, template_id: &str) -> TemplatePostprocessFlags {
        if template_id == TEMPLATE_ID_USER_ANCHORED {
            TemplatePostprocessFlags {
                skip_coherence_truncate: true,
                skip_metacog: true,
            }
        } else {
            TemplatePostprocessFlags::default()
        }
    }

    fn export_brain_plugins(
        &self,
        dm: &DimensionManager,
        header: &mut BrainPackageHeader,
        manifest: &mut BrainPluginsManifest,
    ) -> bool {
        if !is_lattice_shape(dm) {
            return false;
        }
        header.inference_profile = Some("sentiment_lattice".to_string());
        if manifest.inference_thresholds.is_none() {
            manifest.inference_thresholds = Some(InferenceThresholds::default());
        }
        true
    }
}
