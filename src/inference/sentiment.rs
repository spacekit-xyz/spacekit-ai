//! Sentiment lattice inference augment: user-anchored copy, weak meta-GK guard, richer keywords.
//! Configuration: optional TOML via `GROWFORMER_SENTIMENT_INFERENCE_TOML`.

use std::borrow::Cow;
use std::collections::HashSet;
use std::sync::{Arc, OnceLock};

use serde::{Deserialize, Serialize};

use crate::dimension::DimensionManager;
use crate::growformer_lang::MetaConcept;
use crate::micro_brain::MetaResult;

/// `GeneratedResponse.template_id` when the user-anchored path is used.
pub const TEMPLATE_ID_USER_ANCHORED: &str = "sentiment_user_anchored";

/// Seven-topic sentiment sub-lattices (see `data/sentiment/*.jsonl`).
pub const TOPIC_KEYS: &[&str] = &[
    "positive_mild",
    "negative_mild",
    "neutral",
    "negative_strong",
    "sarcastic",
    "positive_strong",
    "mixed",
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SentimentInferenceConfig {
    pub meta_gk_margin: f32,
    pub meta_gk_confidence: f32,
    pub min_meta_confidence_user_anchored: f32,
    pub mixed_override_confidence: f32,
    pub default_line_confidence: f32,
    /// Confidence for hedged / ambiguous lines (not a clear pole).
    pub ambiguous_line_confidence: f32,
}

impl Default for SentimentInferenceConfig {
    fn default() -> Self {
        Self {
            meta_gk_margin: 0.05,
            meta_gk_confidence: 0.55,
            min_meta_confidence_user_anchored: 0.28,
            mixed_override_confidence: 0.88,
            default_line_confidence: 0.92,
            ambiguous_line_confidence: 0.58,
        }
    }
}

static CONFIG: OnceLock<Arc<SentimentInferenceConfig>> = OnceLock::new();

pub fn config() -> Arc<SentimentInferenceConfig> {
    CONFIG
        .get_or_init(|| {
            let mut c = SentimentInferenceConfig::default();
            if let Ok(path) = std::env::var("GROWFORMER_SENTIMENT_INFERENCE_TOML") {
                if let Ok(s) = std::fs::read_to_string(&path) {
                    if let Ok(parsed) = toml::from_str::<SentimentInferenceConfig>(&s) {
                        c = parsed;
                    } else {
                        eprintln!(
                            "[sentiment-inference] failed to parse TOML at {}, using defaults",
                            path
                        );
                    }
                }
            }
            Arc::new(c)
        })
        .clone()
}

/// Brain-bundled `[sentiment]` TOML overrides env-based `GROWFORMER_SENTIMENT_INFERENCE_TOML` when present.
pub(crate) fn sentiment_cfg<'a>(
    brain_override: Option<&'a SentimentInferenceConfig>,
) -> Cow<'a, SentimentInferenceConfig> {
    match brain_override {
        Some(c) => Cow::Borrowed(c),
        None => Cow::Owned((*config()).clone()),
    }
}

/// True when the checkpoint exposes exactly the standard sentiment topic sub-lattice in one group.
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

/// Sentiment shortcuts (weak-GK guard, keyword merge, user-anchored line) require lattice shape
/// and are not explicitly disabled in the brain header.
pub fn shortcuts_enabled(dm: &DimensionManager, inference_profile: Option<&str>) -> bool {
    if !is_lattice_shape(dm) {
        return false;
    }
    match inference_profile.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("off") | Some("none") | Some("disabled") => false,
        _ => true,
    }
}

/// Ignore weak GeneralKnowledge meta projection for conditioning on sentiment brains.
pub fn skip_weak_gk_for_meta_conditioning(
    dm: &DimensionManager,
    inference_profile: Option<&str>,
    sentiment_from_manifest: Option<&SentimentInferenceConfig>,
    concept: MetaConcept,
    margin: f32,
    confidence: f32,
) -> bool {
    if !shortcuts_enabled(dm, inference_profile) {
        return false;
    }
    let cfg = sentiment_cfg(sentiment_from_manifest);
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
        _ => "SENTIMENT",
    }
}

fn anchor_phrase(lower: &str, topic_key: &str) -> Option<String> {
    let tokens: HashSet<&str> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();
    match topic_key {
        "positive_strong" | "positive_mild" => {
            const POS: &[&str] = &[
                "love", "adore", "treasure", "obsessed", "enjoy", "like", "prefer", "best", "great",
                "amazing", "wonderful", "fantastic", "excellent", "happy", "glad", "good", "nice",
                "fine", "better", "beautiful", "incredible", "perfect",
            ];
            for w in POS {
                if tokens.contains(w) {
                    let gloss = match *w {
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
            const NEG: &[&str] = &[
                "hate", "despise", "loathe", "awful", "terrible", "worst", "horrible", "bad",
                "sucks", "disgusting", "dislike", "miserable", "depressing",
            ];
            for w in NEG {
                if tokens.contains(w) {
                    let gloss = match *w {
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

fn has_contrastive_marker(lower: &str) -> bool {
    lower.contains(" but ")
        || lower.contains(", but ")
        || lower.contains(". but ")
        || lower.contains(" although ")
        || lower.contains(" even though ")
        || lower.contains(" however ")
        || lower.contains("; but ")
}

fn has_bipolar_lexicon(lower: &str) -> bool {
    let tokens: HashSet<&str> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();
    const POS: &[&str] = &[
        "love", "adore", "enjoy", "like", "prefer", "great", "good", "incredible", "amazing",
        "wonderful", "fantastic", "excellent", "beautiful", "best", "nice", "fine", "glad", "happy",
        "perfect", "flawless",
    ];
    const NEG: &[&str] = &[
        "hate", "despise", "awful", "terrible", "worst", "horrible", "bad", "sucks", "slow",
        "sloppy", "mediocre", "disappointing", "disappointed", "poor", "worse", "weak", "crash",
        "crashes", "crashed", "broken", "bug", "bugs", "ugly",
    ];
    let has_pos = POS.iter().any(|w| tokens.contains(w));
    let has_neg = NEG.iter().any(|w| tokens.contains(w));
    has_pos && has_neg
}

/// Discourse templates where laudatory wording usually means irony (MetaBrain often misfires as plain negative/positive).
fn has_sarcasm_template(lower: &str) -> bool {
    // --- "because X is exactly what I wanted" (stoic irony) ---
    if lower.contains("because") {
        let desired_outcome = lower.contains("exactly what")
            && (lower.contains("i wanted")
                || lower.contains("i needed")
                || lower.contains("we wanted")
                || lower.contains("we needed"));
        let just_what = lower.contains("just what")
            && (lower.contains("needed") || lower.contains("wanted"));
        if desired_outcome || just_what {
            return true;
        }
    }
    if lower.contains("yeah, because")
        || lower.contains("sure, because")
        || lower.contains("right, because")
    {
        return true;
    }

    if lower.contains("gotta love when") {
        return true;
    }
    if lower.contains("meritocracy at its finest") {
        return true;
    }
    if lower.contains("genius move") {
        return true;
    }
    if lower.contains("living the dream") {
        return true;
    }
    if lower.contains("isn't it just perfect") || lower.contains("isnt it just perfect") {
        return true;
    }
    if lower.contains("thanks for ghosting") || lower.contains("thanks for the zero notice") {
        return true;
    }
    if lower.contains("clearly i'm the problem") || lower.contains("clearly im the problem") {
        return true;
    }

    if (lower.contains("on hold") || lower.contains("been on hold")) && lower.contains("love") {
        return true;
    }

    if (lower.contains("oh great") || lower.contains("oh wonderful"))
        && (lower.contains("meeting")
            || lower.contains("homework")
            || lower.contains("another ")
            || lower.contains("more "))
    {
        return true;
    }

    if lower.contains("what a surprise")
        && (lower.contains("late")
            || lower.contains("again")
            || lower.contains("broken")
            || lower.contains("doesn't work")
            || lower.contains("doesnt work")
            || lower.contains("package"))
    {
        return true;
    }

    if lower.contains("so helpful")
        && (lower.contains("nobody")
            || lower.contains("never")
            || lower.contains("loops")
            || lower.contains("forever")
            || lower.contains("chatbot")
            || lower.contains(" to a human"))
    {
        return true;
    }

    if lower.contains("thrilled")
        && (lower.contains("died")
            || lower.contains("dies")
            || lower.contains("crash")
            || lower.contains("failed")
            || lower.contains("wrong"))
    {
        return true;
    }

    if lower.contains("brilliant") && lower.contains("crash") {
        return true;
    }

    if lower.contains("real helpful") && lower.contains("nobody") {
        return true;
    }

    if lower.contains("felt the love")
        && (lower.contains("thanks") || lower.contains("wow") || lower.contains("ghost"))
    {
        return true;
    }

    if lower.contains("so generous")
        && (lower.contains("voucher") || lower.contains("$") || lower.contains("hour"))
    {
        return true;
    }

    if lower.contains("great job") && lower.contains("dev") {
        return true;
    }
    if lower.contains("great job")
        && (lower.contains("really helpful") || lower.contains("alone again"))
    {
        return true;
    }

    if (lower.contains("you're doing great") || lower.contains("youre doing great"))
        && (lower.contains("email")
            || lower.contains("still no")
            || lower.contains("no answer")
            || lower.contains("waiting")
            || lower.contains("reply"))
    {
        return true;
    }

    // Praise + beat + "nailed" — common vent / ironic performance review tone
    if lower.contains("fantastic") && lower.contains("nail") {
        if lower.contains('…') || lower.contains("...") {
            return true;
        }
        if lower.contains("this time") || lower.contains(" again") || lower.ends_with("again.") {
            return true;
        }
    }

    // Grievance + bitter "praise" (long hold, silence, delays…)
    let gripe = lower.contains("waiting ")
        || lower.contains("waiting for")
        || lower.contains("waited ")
        || lower.contains("on hold")
        || lower.contains("minutes ")
        || lower.contains("minutes for")
        || lower.contains("still no")
        || lower.contains("no answer")
        || lower.contains("no reply")
        || lower.contains("delayed");
    let bitter_praise = lower.contains("love the customer")
        || lower.contains("love the service")
        || (lower.contains("doing great") && !lower.contains("you're not"))
        || (lower.contains("great job") && !lower.starts_with("great job on"));
    if gripe && bitter_praise {
        return true;
    }

    false
}

fn normalize_match_text(text: &str) -> String {
    let mut s = text.to_lowercase();
    s = s.replace('\u{2019}', "'");
    s = s.replace('\u{2018}', "'");
    s
}

/// MetaBrain often picks `positive_strong` on weak tokens ("fine", "better") while the line is hedged or disappointed.
fn ambiguous_valence_retarget(lower: &str, key: &str) -> Option<&'static str> {
    if !matches!(key, "positive_strong" | "positive_mild") {
        return None;
    }

    if lower.contains("thought it would be better")
        || lower.contains("thought it'd be better")
        || lower.contains("expected it to be better")
        || lower.contains("expected more")
        || lower.contains("hoped for better")
        || lower.contains("hoped it would be")
        || lower.contains("could have been better")
        || lower.contains("was hoping for")
    {
        return Some("negative_mild");
    }

    let hedge = lower.contains("i guess")
        || lower.contains("i suppose")
        || lower.contains("nothing special")
        || lower.contains("nothing great")
        || lower.contains("mediocre")
        || lower.contains("so-so")
        || lower.contains("so so")
        || lower.contains("could be worse")
        || lower.contains("not great")
        || lower.contains("not amazing");

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

/// Obvious opinion / affect words — if present, do not treat the line as a dry fact.
fn has_clear_evaluative_stance(lower: &str) -> bool {
    const EVAL: &[&str] = &[
        "hate", "love", "terrible", "awful", "amazing", "horrible", "fantastic", "disappointed",
        "disappointing", "thrilled", "annoyed", "frustrated", "excited", "worried", "anxious",
        "regret", "ashamed", "proud", "grateful", "annoying", "wonderful", "sucks", "best ever",
        "worst", "ridiculous", "outraged",
    ];
    EVAL.iter().any(|w| lower.contains(w))
}

/// Third-person / technical facts with no stance (file metrics, login status, delivery time).
fn is_objective_factual_statement(lower: &str) -> bool {
    if has_clear_evaluative_stance(lower) {
        return false;
    }
    let has_digit = lower.chars().any(|c| c.is_ascii_digit());

    if lower.contains("file size") && has_digit {
        return true;
    }
    if lower.contains("gigabyte")
        || lower.contains("megabytes")
        || lower.contains("megabyte")
        || lower.contains("terabytes")
        || lower.contains("terabyte")
        || lower.contains("kilobytes")
        || lower.contains("kilobyte")
    {
        return true;
    }
    if has_digit
        && (lower.contains(" gb")
            || lower.contains(" mb")
            || lower.contains(" kb")
            || lower.contains(" tb")
            || lower.contains("gb.")
            || lower.contains("mb."))
    {
        return true;
    }

    if (lower.contains("logged in")
        || lower.contains("log in")
        || lower.contains("signed in")
        || lower.contains("sign in"))
        && (lower.contains("success")
            || lower.contains("failed")
            || lower.contains("failure")
            || lower.contains("error"))
    {
        return true;
    }

    if (lower.contains("arrived")
        || lower.contains("delivered")
        || lower.contains("shipped")
        || lower.contains("departed"))
        && has_digit
        && (lower.contains(" at ")
            || lower.contains(" pm")
            || lower.contains(" am")
            || lower.contains(':'))
    {
        return true;
    }

    false
}

/// `disappointed` + MetaBrain `positive_*` is a common misfire.
fn disappointment_positive_override(lower: &str, key: &str) -> Option<&'static str> {
    if !matches!(key, "positive_strong" | "positive_mild") {
        return None;
    }
    if lower.contains("disappointed") || lower.contains("disappointing") {
        return Some("negative_mild");
    }
    None
}

/// Phrases where surface text fixes polarity regardless of MetaBrain (wrong pole or low confidence).
fn lexical_polarity_signal(lower: &str) -> Option<&'static str> {
    if lower.contains("blew me away")
        || lower.contains("blow me away")
        || lower.contains("blown away")
        || lower.contains("mind-blowing")
        || lower.contains("mind blowing")
        || lower.contains("mind blown")
    {
        return Some("positive_strong");
    }

    if lower.contains("wouldn't buy")
        || lower.contains("wouldnt buy")
        || lower.contains("won't buy again")
        || lower.contains("wont buy again")
        || lower.contains("never buying again")
        || lower.contains("not buying again")
        || lower.contains("won't purchase again")
        || lower.contains("wouldn't purchase again")
        || lower.contains("will not buy again")
        || lower.contains("won't be buying")
        || lower.contains("wont be buying")
        || lower.contains("not worth buying")
    {
        return Some("negative_mild");
    }

    None
}

/// Prefer a user-anchored sentiment line over lattice paraphrase when the brain matches the
/// sentiment profile and MetaBrain topic/confidence allow it.
pub fn try_user_anchored_line(
    dm: &DimensionManager,
    inference_profile: Option<&str>,
    sentiment_from_manifest: Option<&SentimentInferenceConfig>,
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
    let cfg = sentiment_cfg(sentiment_from_manifest);
    let lower = normalize_match_text(intent_text);

    // Dry facts often get weak / wrong MetaBrain scores; still answer NEUTRAL without lattice retrieval.
    if is_objective_factual_statement(&lower) {
        let header = label_header("neutral");
        let body = "Measurable fact, status, or time — no evaluative opinion; read as NEUTRAL (not praise or complaint)";
        let trimmed = intent_text.trim();
        let excerpt = if trimmed.chars().count() > 200 {
            let head: String = trimmed.chars().take(200).collect();
            format!("{}…", head)
        } else {
            trimmed.to_string()
        };
        println!("  [sentiment-direct] objective fact / status → NEUTRAL (bypass meta conf gate)");
        let text = format!(
            "{} — {}. Grounded in the user's own words: \"{}\"",
            header, body, excerpt
        );
        return Some((text, cfg.ambiguous_line_confidence));
    }

    let lex_polar = lexical_polarity_signal(&lower);
    if mr.confidence < cfg.min_meta_confidence_user_anchored && lex_polar.is_none() {
        return None;
    }
    let contrast = has_contrastive_marker(&lower);
    let bipolar = has_bipolar_lexicon(&lower);
    let force_mixed = contrast && bipolar;

    let mut key = TOPIC_KEYS
        .iter()
        .find(|k| th.eq_ignore_ascii_case(k))
        .copied()
        .unwrap_or(th);
    if force_mixed {
        key = "mixed";
    }

    let mut lexical_polarity_override = false;
    if !force_mixed {
        if let Some(k) = lex_polar {
            key = k;
            lexical_polarity_override = true;
        }
    }

    let mut sarcasm_template_override = false;
    if !force_mixed && key != "sarcastic" && has_sarcasm_template(&lower) {
        key = "sarcastic";
        sarcasm_template_override = true;
    }

    let mut ambiguous_override = false;
    let mut ambiguous_disappointment = false;
    if !force_mixed && !sarcasm_template_override {
        if let Some(new_key) = ambiguous_valence_retarget(&lower, key) {
            ambiguous_disappointment = new_key == "negative_mild";
            ambiguous_override = true;
            key = new_key;
        }
    }

    let mut disappointment_override = false;
    if !force_mixed
        && !sarcasm_template_override
        && !ambiguous_override
    {
        if let Some(new_key) = disappointment_positive_override(&lower, key) {
            key = new_key;
            disappointment_override = true;
        }
    }

    let header = label_header(key);
    let trimmed = intent_text.trim();
    let excerpt = if trimmed.chars().count() > 200 {
        let head: String = trimmed.chars().take(200).collect();
        format!("{}…", head)
    } else {
        trimmed.to_string()
    };

    let (body, conf) = if key == "mixed" {
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
    } else if key == "sarcastic" && sarcasm_template_override {
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
        let explain = if key == "positive_strong" {
            "Strong positive idiom (e.g. 'blew me away') — read as enthusiastic praise despite weak topic scores"
        } else {
            "Clear non-repeat / rejection stance (e.g. wouldn't buy again) — negative on the product or experience, not praise"
        };
        let c = if key == "positive_strong" {
            cfg.default_line_confidence
        } else {
            cfg.ambiguous_line_confidence
        };
        (explain.to_string(), c)
    } else {
        let anchor = anchor_phrase(&lower, key);
        let tone = match key {
            "positive_strong" | "positive_mild" => "The overall tone reads as clearly positive",
            "negative_strong" | "negative_mild" => "The overall tone reads as clearly negative",
            "neutral" => "The overall tone reads as mostly neutral",
            "sarcastic" => "The line may use irony or surface/actual mismatch",
            _ => "Sentiment is non-obvious from surface text alone",
        };
        let b = match &anchor {
            Some(a) => format!("{}. {}", a, tone),
            None => tone.to_string(),
        };
        (b, cfg.default_line_confidence)
    };

    if force_mixed {
        println!("  [sentiment-direct] contrastive bipolar → MIXED (override meta topic)");
    }
    if sarcasm_template_override {
        println!("  [sentiment-direct] sarcasm template → SARCASTIC (override meta topic)");
    }
    if ambiguous_override {
        println!(
            "  [sentiment-direct] hedged/ambiguous → {} (override meta topic)",
            key
        );
    }
    if disappointment_override {
        println!("  [sentiment-direct] disappointment cue → NEGATIVE (mild) (override meta topic)");
    }
    if lexical_polarity_override {
        println!(
            "  [sentiment-direct] lexical polarity → {} (override meta topic / bypass low conf)",
            key
        );
    }

    let text = format!(
        "{} — {}. Grounded in the user's own words: \"{}\"",
        header, body, excerpt
    );
    Some((text, conf))
}
