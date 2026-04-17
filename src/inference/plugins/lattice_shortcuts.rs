//! Topic-lattice shortcut plugin: user-anchored copy, weak meta-GK guard, keyword lists from TOML.
//! Thresholds: `[sentiment]` in brain `plugins_blob` (serde name) or embedded inference TOML.
//! Rules: `GROWFORMER_INFERENCE_TOML` / legacy env, or embedded default file.

use std::collections::HashSet;

use crate::infer_trace;
use crate::brain::BrainPackageHeader;
use crate::dimension::DimensionManager;
use crate::growformer_lang::MetaConcept;
use crate::micro_brain::MetaResult;

use crate::inference::causal_hints;
use crate::inference::grounding_expand;
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

/// True when the checkpoint exposes at least one generation group that includes every **standard**
/// seven-topic sub-lattice name (the historical sentiment lattice layout).
///
/// Single-group sentiment packs, multi-group fintech packs, and merged brains (sentiment ⊕ causal)
/// all qualify as long as **some** group matches. Extra topics (expanded taxonomies) are allowed —
/// user-anchored preempt and weak-GK shortcuts fire so long as the seven keys are present somewhere.
pub fn is_lattice_shape(dm: &DimensionManager) -> bool {
    if dm.group_gen_envs.is_empty() {
        return false;
    }
    dm.group_gen_envs.values().any(|env| {
        if env.topic_subindex.len() < TOPIC_KEYS.len() {
            return false;
        }
        let names: HashSet<String> = env
            .topic_subindex
            .iter()
            .map(|t| t.topic_name.to_ascii_lowercase())
            .collect();
        TOPIC_KEYS.iter().all(|k| names.contains(&k.to_string()))
    })
}

/// Topic names suggest a consumer / wire **sentiment** generation lattice (not a code-only brain).
///
/// Used for TOML headline guards, PR-wire neutralization, and lattice misfire sanitization — independent
/// of [`is_lattice_shape`] (which is single-group seven-topic only). Multi-group fintech+identity packs
/// still expose one sentiment-shaped group.
pub fn generation_env_looks_like_sentiment(dm: &DimensionManager, gidx: usize) -> bool {
    dm.group_gen_envs.get(&gidx).map_or(false, |env| {
        if env.topic_subindex.len() < 2 {
            return false;
        }
        const MARKERS: &[&str] = &[
            "positive_",
            "negative_",
            "neutral",
            "mixed",
            "sarcastic",
            "confused",
            "cautious",
            "hopium",
            "copium",
            "euphoric",
            "capitulation",
        ];
        env.topic_subindex.iter().any(|t| {
            let n = t.topic_name.to_ascii_lowercase();
            MARKERS.iter().any(|m| n.contains(m))
        })
    })
}

/// First **group env map key** (0..`group_order.len()`) whose gen env matches [`generation_env_looks_like_sentiment`].
///
/// This matches the historical `LanguageService` convention: `group_gen_envs` is keyed by group slot index,
/// not raw [`crate::types::GroupId`].
pub fn pick_sentiment_lattice_group_idx(dm: &DimensionManager) -> Option<usize> {
    (0..dm.main.group_order.len()).find(|&gidx| generation_env_looks_like_sentiment(dm, gidx))
}

/// When false, `LanguageService` skips inference-TOML PR-wire / headline lexical / mixed guards and
/// lattice misfire replacement (code brains, or `inference_profile` `off` / `none` / `disabled`).
///
/// True only if the loaded checkpoint exposes at least one sentiment-shaped gen lattice **and**
/// the brain header does not disable inference shortcuts by name.
pub fn sentiment_toml_lexical_guards_active(
    dm: &DimensionManager,
    inference_profile: Option<&str>,
) -> bool {
    match inference_profile.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("off") | Some("none") | Some("disabled") => return false,
        _ => {}
    }
    pick_sentiment_lattice_group_idx(dm).is_some()
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

pub fn label_header(topic_key: &str) -> &'static str {
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

/// Display header for a `semantic_intent` / topic key (matches seven-topic shortcuts, plus extended sentiment intents).
pub fn sentiment_display_header(topic_key: &str) -> String {
    match topic_key.to_ascii_lowercase().as_str() {
        "positive_strong" => "POSITIVE (strong)".to_string(),
        "positive_mild" => "POSITIVE (mild)".to_string(),
        "negative_strong" => "NEGATIVE (strong)".to_string(),
        "negative_mild" => "NEGATIVE (mild)".to_string(),
        "neutral" => "NEUTRAL".to_string(),
        "sarcastic" => "SARCASTIC".to_string(),
        "mixed" => "MIXED".to_string(),
        "neutral_chop" => "NEUTRAL (chop)".to_string(),
        "cautiously_negative" => "CAUTIOUSLY NEGATIVE".to_string(),
        "cautiously_positive" => "CAUTIOUSLY POSITIVE".to_string(),
        "confused" => "CONFUSED".to_string(),
        "hopium" => "HOPIUM".to_string(),
        "copium" => "COPIUM".to_string(),
        "capitulation" => "CAPITULATION".to_string(),
        "euphoric" => "EUPHORIC".to_string(),
        k => k
            .split('_')
            .map(|w| {
                let mut c = w.chars();
                c.next()
                    .map(|f| f.to_uppercase().collect::<String>() + c.as_str())
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>()
            .join(" "),
    }
}

/// True when `segment` looks like a display label (`MIXED`, `POSITIVE (mild)`, `NEGATIVE( sarcastic)`, `CAUTIOUSLY NEGATIVE`, …).
fn sentiment_segment_looks_like_label_header(seg: &str) -> bool {
    let h = seg.trim();
    if h.len() < 3 || h.len() > 88 {
        return false;
    }
    if !h.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
        return false;
    }
    let mut depth: i32 = 0;
    for c in h.chars() {
        match c {
            '(' => {
                depth += 1;
                if depth > 2 {
                    return false;
                }
            }
            ')' => {
                depth -= 1;
                if depth < 0 {
                    return false;
                }
            }
            _ if depth > 0 => {
                if !(c.is_ascii_alphanumeric() || c == ' ') {
                    return false;
                }
            }
            _ => {
                if !(c.is_ascii_uppercase() || c == ' ' || c == '-' || c == '/') {
                    return false;
                }
            }
        }
    }
    depth == 0
}

fn split_once_label_separator(s: &str) -> Option<(&str, &str)> {
    s.split_once(" — ")
        .or_else(|| s.split_once(" – "))
        .or_else(|| s.split_once(" - "))
}

/// Remove one or more leading `LABEL —` / `LABEL -` segments (fixes double headers when training text embedded old shortcut fragments).
pub fn strip_leading_sentiment_display_headers(s: &str) -> String {
    let mut cur = s.trim_start();
    loop {
        let Some((head, rest)) = split_once_label_separator(cur) else {
            return cur.to_string();
        };
        let r = rest.trim_start();
        if r.is_empty() {
            return cur.to_string();
        }
        if sentiment_segment_looks_like_label_header(head) {
            cur = r;
            continue;
        }
        return cur.to_string();
    }
}

/// True when `body` begins with a recognizable display label + separator (after stripping, would lose content).
pub fn sentiment_line_already_has_display_header(body: &str) -> bool {
    let Some((head, rest)) = split_once_label_separator(body.trim_start()) else {
        return false;
    };
    sentiment_segment_looks_like_label_header(head) && !rest.trim().is_empty()
}

/// Prefix lattice-retrieved rationale with `LABEL —` for parity with the user-anchored shortcut line format.
pub fn format_retrieved_sentiment_line(topic_key: &str, body: &str) -> String {
    let body = crate::dimension::language::strip_sentiment_lattice_witness_for_display(body);
    let cleaned = strip_leading_sentiment_display_headers(&body);
    let cleaned = if cleaned.trim().is_empty() {
        body.trim().to_string()
    } else {
        cleaned.trim_start().to_string()
    };
    let header = sentiment_display_header(topic_key);
    format!("{} — {}", header, cleaned)
}

fn normalize_match_text(text: &str) -> String {
    let mut s = text.to_lowercase();
    s = s.replace('\u{2019}', "'");
    s = s.replace('\u{2018}', "'");
    s = s.replace('\u{2014}', "-");
    s = s.replace('\u{2013}', "-");
    s
}

fn abbreviate_large_number(digits: &str) -> String {
    let n: u128 = match digits.parse() {
        Ok(v) => v,
        Err(_) => return digits.to_string(),
    };
    if n >= 1_000_000_000 && n % 1_000_000_000 == 0 {
        format!("{}B", n / 1_000_000_000)
    } else if n >= 1_000_000_000 {
        let whole = n / 1_000_000_000;
        let frac = (n % 1_000_000_000) / 100_000_000;
        if frac > 0 { format!("{}.{}B", whole, frac) } else { format!("{}B", whole) }
    } else if n >= 1_000_000 && n % 1_000_000 == 0 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000_000 {
        let whole = n / 1_000_000;
        let frac = (n % 1_000_000) / 100_000;
        if frac > 0 { format!("{}.{}M", whole, frac) } else { format!("{}M", whole) }
    } else {
        digits.to_string()
    }
}

pub fn detokenize_money_pub(text: &str) -> String {
    detokenize_money(text)
}

fn detokenize_money(text: &str) -> String {
    static CCY_SYMBOLS: &[(&str, &str)] = &[
        ("money_usd_", "$"), ("money_gbp_", "£"), ("money_eur_", "€"),
        ("money_jpy_", "¥"), ("money_krw_", "₩"), ("money_inr_", "₹"),
        ("money_btc_", "₿"), ("money_rub_", "₽"), ("money_php_", "₱"),
        ("money_vnd_", "₫"), ("money_try_", "₺"), ("money_uah_", "₴"),
        ("money_ngn_", "₦"), ("money_kzt_", "₸"), ("money_brl_", "R$"),
        ("money_sek_", "kr"),
    ];
    let mut out = text.to_string();
    for &(prefix, symbol) in CCY_SYMBOLS {
        while let Some(start) = out.find(prefix) {
            let num_start = start + prefix.len();
            let num_end = out[num_start..].find(|c: char| !c.is_ascii_digit())
                .map(|i| num_start + i)
                .unwrap_or(out.len());
            let amount = &out[num_start..num_end];
            let replacement = format!("{}{}", symbol, abbreviate_large_number(amount));
            out.replace_range(start..num_end, &replacement);
        }
    }
    out
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
    let cfg = resolved_inference_thresholds(thresholds_from_manifest);
    let rules = inference_rules_runtime();
    let lower = normalize_match_text(intent_text);
    let display_text = detokenize_money(intent_text);

    // Objective-fact / status-query preempt runs FIRST, before the topic-hint
    // gate. On merged brains the factual prompt often routes to a non-sentiment
    // topic (e.g. `general_knowledge`) so requiring a sentiment topic key up
    // front would skip the preempt and leak a MIXED fallback for pure factual
    // questions like "What nominal AC voltages..." or "List three ISO 27001...".
    if rules.is_objective_factual_statement(&lower) {
        let header = label_header("neutral");
        let body = "Measurable fact, status, or time — no evaluative opinion; read as NEUTRAL (not praise or complaint)";
        let trimmed = display_text.trim();
        let excerpt = if trimmed.chars().count() > 200 {
            let head: String = trimmed.chars().take(200).collect();
            format!("{}…", head)
        } else {
            trimmed.to_string()
        };
        infer_trace!("  [lattice-direct] objective fact / status → NEUTRAL (bypass meta conf gate)");
        let text = format!(
            "{} — {}. Grounded in the user's own words: \"{}\"",
            header, body, excerpt
        );
        return Some((text, cfg.ambiguous_line_confidence));
    }

    // Causal-relation preempt: when the brain has a causal group and the text
    // contains an explicit causal connector, produce a structured output that
    // separates the relation type ("Direct causal link", "Concessive
    // persistence", ...) from independent per-clause sentiment scoring. This
    // prevents retrieval misses from falling back to honest-unknown on lines
    // like "Inventory write-downs triggered a covenant breach" (no close
    // lattice row) or garbled multi-output on "CFO sold holdings, implying..."
    // where the lattice blends sentiment copy into the causal explanation.
    //
    // Gate: bare connectors like "but", "yet", "somehow", "because", "since"
    // are ambiguous between causal and sentiment contexts (e.g. "hospital food
    // was terrible but nurses were incredible" is sentiment-MIXED, not a causal
    // chain). Only fire when:
    //  (a) the connector confidence is >= 0.9 (unambiguously causal: "triggered",
    //      "in retrospect", "would have", "which suggests", etc.), OR
    //  (b) the topic_hint maps to a causal topic (meta-brain routed to causal),
    //      in which case any connector is trusted.
    if dm.find_causal_group().is_some() {
        let causal_topic_routed = topic_hint.map_or(false, |th| {
            const CAUSAL_TOPICS: &[&str] = &[
                "direct", "compensatory", "contrastive", "explanatory",
                "concessive", "inferential", "retrospective_framing",
                "interventional_counterfactual", "causal",
            ];
            CAUSAL_TOPICS.iter().any(|ct| th.eq_ignore_ascii_case(ct))
        });
        // When the TOML lexical_polarity list has a *long* curated phrase
        // match (>= 40 chars — full-sentence entries like "not because it makes
        // me happy — it makes me feel understood"), the sentiment path is
        // authoritative and the causal preempt yields. Short TOML hits like
        // "covenant breach" or "all-time high" don't suppress the preempt
        // since they may only cover one clause of a multi-clause causal line.
        let toml_long_sentiment_match = rules
            .lexical_polarity_signal_with_len(&lower)
            .map_or(false, |(_, len)| len >= 40);
        // Sarcasm templates are authoritative — "Oh great, the server crashed
        // because someone pushed to prod" should be SARCASTIC, not Explanatory.
        let sarcasm_fires = rules.has_sarcasm_template(&lower);
        if let Some(csr) = crate::inference::causal_relation::score_with_relation(intent_text) {
            if (csr.confidence >= 0.9 || causal_topic_routed) && !toml_long_sentiment_match && !sarcasm_fires {
                let text = crate::inference::causal_relation::format_causal_sentiment_line(
                    &csr,
                    intent_text,
                );
                infer_trace!(
                    "  [lattice-direct] causal-relation preempt → {} + {} (connector '{}', conf={:.2}, causal_routed={})",
                    csr.relation.label(),
                    csr.sentiment_label,
                    csr.connector,
                    csr.confidence,
                    causal_topic_routed,
                );
                return Some((text, csr.confidence.max(0.72)));
            }
        }
    }

    // Lexicon-authoritative preempt: when the configured `lexical_polarity`
    // phrase list or the contrastive-bipolar structural check matches, the
    // TOML is ground truth for the polarity label regardless of what the
    // meta-brain routed to. Without this, merged brains (Brain C) that route
    // personal anecdotes like "They promoted me and doubled my meetings. Be
    // careful what you wish for" to `topic_hint = identity` skip the whole
    // user-anchored path and fall back to the cue-dump. The preempt emits a
    // lexicon-override body when applicable and a contrastive-MIXED body
    // otherwise — same copy the in-flow path would produce.
    let lex_polar_early = rules.lexical_polarity_signal(&lower);
    let contrast_early = rules.has_contrastive_marker(&lower);
    let bipolar_early = rules.has_bipolar_lexicon(&lower);
    let topic_in_sentiment_keys = topic_hint
        .map(|th| TOPIC_KEYS.iter().any(|k| th.eq_ignore_ascii_case(k)))
        .unwrap_or(false);
    if !topic_in_sentiment_keys
        && (lex_polar_early.is_some() || (contrast_early && bipolar_early))
    {
        // Sarcasm templates override the raw polarity — "Love the customer
        // service" after a complaint should be SARCASTIC, not positive_mild.
        let sarcasm_early = rules.has_sarcasm_template(&lower);
        let key = if sarcasm_early {
            "sarcastic".to_string()
        } else if contrast_early && bipolar_early {
            "mixed".to_string()
        } else {
            let raw = lex_polar_early.clone().unwrap_or_else(|| "neutral".to_string());
            rules.apply_degree_modifiers(&lower, &raw)
        };
        let header = label_header(key.as_str());
        let trimmed = display_text.trim();
        let excerpt = if trimmed.chars().count() > 200 {
            let head: String = trimmed.chars().take(200).collect();
            format!("{}…", head)
        } else {
            trimmed.to_string()
        };
        // Prefer anchor+tone for single-pole polarity keys so a product-rejection override body
        // doesn't leak into unrelated contexts (e.g. a rehearsed apology, 7am drilling complaint).
        // Only `mixed` / `sarcastic` use their TOML override copy since those bodies are structural
        // ("contrastive dual-valence", "laudatory wording clashes with grievance") and fit any
        // phrase that triggered the structural check.
        let tone = match key.as_str() {
            "positive_strong" | "positive_mild" => "The overall tone reads as clearly positive",
            "negative_strong" | "negative_mild" => "The overall tone reads as clearly negative",
            "neutral" => "The overall tone reads as mostly neutral",
            "sarcastic" => "The line may use irony or surface/actual mismatch",
            _ => "Classification is non-obvious from surface text alone",
        };
        let body = if key == "mixed" || key == "sarcastic" {
            let lex = crate::inference::sentiment_generation_lexicon::global();
            lex.lattice_lexical_override_body(key.as_str())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .unwrap_or_else(|| {
                    if key == "mixed" {
                        "Contrastive marker (e.g. 'but') with both laudatory and critical wording — dual valence (MIXED), not a single pole".to_string()
                    } else {
                        "Laudatory or stoic wording clashes with an obvious grievance — read as SARCASTIC / ironic, not literal".to_string()
                    }
                })
        } else {
            match rules.anchor_phrase(&lower, key.as_str()) {
                Some(a) => format!("{}. {}", a, tone),
                None => tone.to_string(),
            }
        };
        let text = format!(
            "{} — {}. Grounded in the user's own words: \"{}\"",
            header, body, excerpt
        );
        let conf = if key == "mixed" {
            cfg.mixed_override_confidence
        } else if key == "positive_strong" {
            cfg.default_line_confidence
        } else {
            cfg.ambiguous_line_confidence
        };
        infer_trace!(
            "  [lattice-direct] lexicon-authoritative preempt → {} (topic hint {:?} bypassed)",
            key,
            topic_hint
        );
        return Some((text, conf));
    }

    let mr = meta_result?;
    let th = topic_hint?;
    if !TOPIC_KEYS.iter().any(|k| th.eq_ignore_ascii_case(k)) {
        return None;
    }

    let lex_polar = lex_polar_early;
    let mixed_structurally_ok = th.eq_ignore_ascii_case("mixed")
        && rules.sentiment_allow_forced_mixed_topic(intent_text);
    if mr.confidence < cfg.min_meta_confidence_user_anchored
        && lex_polar.is_none()
        && !mixed_structurally_ok
    {
        return None;
    }
    let contrast = contrast_early;
    let bipolar = bipolar_early;
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
            key = rules.apply_degree_modifiers(&lower, &k);
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
    let trimmed = display_text.trim();
    let excerpt = if trimmed.chars().count() > 200 {
        let head: String = trimmed.chars().take(200).collect();
        format!("{}…", head)
    } else {
        trimmed.to_string()
    };

    let (body, conf) = if key.as_str() == "mixed" && lexical_polarity_override {
        let lex = crate::inference::sentiment_generation_lexicon::global();
        let explain = lex
            .lattice_lexical_override_body("mixed")
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                if contrast && bipolar {
                    "Contrastive marker (e.g. 'but') with both laudatory and critical wording — dual valence (MIXED), not a single pole"
                } else {
                    "Positive and negative cues both appear; overall read is MIXED"
                }
                .to_string()
            });
        (
            explain,
            if force_mixed {
                cfg.mixed_override_confidence
            } else {
                cfg.default_line_confidence
            },
        )
    } else if key.as_str() == "mixed" {
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
        // For single-pole keys (positive/negative/neutral), the override body in the lexicon TOML
        // is tuned for product-review register ("wouldn't buy again", "commerce-rejection"). Prefer
        // anchor+tone for these so non-product contexts (rehearsed apology, drilling complaint,
        // gaming slang) don't inherit commerce framing. `mixed` and `sarcastic` still use the
        // override since their bodies are structural, not domain-flavored.
        let explain = if key.as_str() == "mixed" || key.as_str() == "sarcastic" {
            let lex = crate::inference::sentiment_generation_lexicon::global();
            lex.lattice_lexical_override_body(key.as_str())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .unwrap_or_else(|| key.clone())
        } else {
            let tone = match key.as_str() {
                "positive_strong" | "positive_mild" => "The overall tone reads as clearly positive",
                "negative_strong" | "negative_mild" => "The overall tone reads as clearly negative",
                "neutral" => "The overall tone reads as mostly neutral",
                _ => "Classification is non-obvious from surface text alone",
            };
            match rules.anchor_phrase(&lower, key.as_str()) {
                Some(a) => format!("{}. {}", a, tone),
                None => tone.to_string(),
            }
        };
        let c = if key.as_str() == "positive_strong" {
            cfg.default_line_confidence
        } else {
            cfg.ambiguous_line_confidence
        };
        (explain, c)
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
        infer_trace!("  [lattice-direct] contrastive bipolar → MIXED (override meta topic)");
    }
    if sarcasm_template_override {
        infer_trace!("  [lattice-direct] sarcasm template → SARCASTIC (override meta topic)");
    }
    if ambiguous_override {
        infer_trace!(
            "  [lattice-direct] hedged/ambiguous → {} (override meta topic)",
            key
        );
    }
    if disappointment_override {
        infer_trace!("  [lattice-direct] disappointment cue → NEGATIVE (mild) (override meta topic)");
    }
    if lexical_polarity_override {
        infer_trace!(
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
        // Full-line tokens + Layer‑0 / causal / rule expansion apply to **any** brain layout.
        // `shortcuts_enabled` (seven-topic single-group) gates only preempt / weak‑GK behavior
        // elsewhere — it must not skip world grounding or BM25 query enrichment (fintech has
        // identity + sentiment groups and many topic keys).
        if matches!(
            inference_profile.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
            Some("off") | Some("none") | Some("disabled")
        ) {
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
        causal_hints::extend_subject_keywords_with_causal_tokens(intent_text, subject_kw);
        grounding_expand::extend_subject_keywords_with_grounding(intent_text, subject_kw);
        crate::inference::world_grounding::extend_subject_keywords_with_world_graph(
            intent_text,
            subject_kw,
        );
        let _ = dm;
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

#[cfg(test)]
mod sentiment_format_tests {
    use super::{
        format_retrieved_sentiment_line, sentiment_display_header, sentiment_line_already_has_display_header,
    };

    #[test]
    fn display_header_extended_intents() {
        assert_eq!(sentiment_display_header("neutral_chop"), "NEUTRAL (chop)");
        assert_eq!(sentiment_display_header("cautiously_negative"), "CAUTIOUSLY NEGATIVE");
    }

    #[test]
    fn format_line_prefixes_body() {
        let out = format_retrieved_sentiment_line("neutral_chop", "Low activity; neutral chop.");
        assert!(out.starts_with("NEUTRAL (chop) — "));
        assert!(out.contains("Low activity"));
    }

    #[test]
    fn format_line_strips_joint_index_witness() {
        // Simulate decode collapsing spaces around the marker (still findable by core substring).
        let joint = "BTC dominance line.__GROWFORMER_SENT_WITNESS__ Rationale only.".to_string();
        assert!(joint.contains(crate::dimension::language::SENTIMENT_LATTICE_WITNESS_CORE));
        let out = format_retrieved_sentiment_line("mixed", &joint);
        assert!(out.starts_with("MIXED — "));
        assert!(out.contains("Rationale only."));
        assert!(!out.contains("BTC dominance"));
    }

    #[test]
    fn format_line_idempotent_when_prefixed() {
        let already = "MIXED — Two-sided valence.";
        assert!(sentiment_line_already_has_display_header(already));
        assert_eq!(format_retrieved_sentiment_line("mixed", already), already);
    }

    #[test]
    fn format_line_strips_chained_legacy_headers() {
        let body = "POSITIVE (mild) — POSITIVE( mild) — Security improvement acknowledged.";
        let out = format_retrieved_sentiment_line("positive_mild", body);
        assert_eq!(
            out,
            "POSITIVE (mild) — Security improvement acknowledged."
        );
        assert!(!out.contains("POSITIVE( mild)"));
    }
}
