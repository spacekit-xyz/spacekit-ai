//! Language front-end layer for Growformer (M1/M2 foundation).
//!
//! This module provides:
//! - deterministic text encoder presets (default MiniLM-sized 384-d vectors),
//! - a globally calibrated bridge (384->128d default) with layer norm + confidence head,
//! - EMA smoothing for multi-turn routing stability,
//! - objective routing outputs (winner, margin, OOD reject).

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[cfg(feature = "native")]
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

use super::embedding::{cosine_similarity, GroupEmbedding};
use crate::types::GroupId;

pub const DEFAULT_ENCODER_DIM: usize = 384;
pub const DEFAULT_BRIDGE_DIM: usize = 128;
pub const DEFAULT_EMA_ALPHA: f32 = 0.2;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EncoderPreset {
    MiniLmL6V2,
    MiniLmMultilingualL12V2,
    BertClass,
    /// MLP-free encoder: 256-d hash → E8 quantize → Cl(1,7) embed → 128-d grade extract.
    CliffordE8,
    Custom {
        model_name: String,
        output_dim: usize,
    },
}

impl EncoderPreset {
    pub fn model_name(&self) -> String {
        match self {
            EncoderPreset::MiniLmL6V2 => "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            EncoderPreset::MiniLmMultilingualL12V2 => {
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2".to_string()
            }
            EncoderPreset::BertClass => "bert-base-uncased".to_string(),
            EncoderPreset::CliffordE8 => "clifford-e8-cata".to_string(),
            EncoderPreset::Custom { model_name, .. } => model_name.clone(),
        }
    }

    pub fn output_dim(&self) -> usize {
        match self {
            EncoderPreset::MiniLmL6V2 => 384,
            EncoderPreset::MiniLmMultilingualL12V2 => 384,
            EncoderPreset::BertClass => 768,
            EncoderPreset::CliffordE8 => CLIFFORD_ENCODER_OUTPUT_DIM,
            EncoderPreset::Custom { output_dim, .. } => *output_dim,
        }
    }
}

impl Default for EncoderPreset {
    fn default() -> Self {
        EncoderPreset::CliffordE8
    }
}

// Default to CliffordE8 at all times. BERT to be deprecated.
impl EncoderPreset {
    /// Resolve from the `GROWFORMER_ENCODER` env var, falling back to `CliffordE8`.
    pub fn from_env() -> Self {
        match std::env::var("GROWFORMER_ENCODER").ok().as_deref() {
            Some("clifford_e8") | Some("CliffordE8") | Some("clifford-e8") => {
                EncoderPreset::CliffordE8
            }
            _ => EncoderPreset::CliffordE8,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageConfig {
    pub encoder: EncoderPreset,
    pub bridge_output_dim: usize,
    pub ema_alpha: f32,
    pub ood_similarity_threshold: f32,
    /// Optional HTTP endpoint for external encoder embeddings.
    /// Expected API: POST JSON {"text": "...", "model": "..."} -> {"embedding":[f32...]}.
    #[serde(default)]
    pub gle_http_endpoint: Option<String>,
    /// Optional local distilled tiny-student checkpoint path (our GLE artifact).
    #[serde(default)]
    pub gle_checkpoint: Option<String>,
    /// Optional multi-checkpoint ensemble paths for local distilled students.
    /// When set, these are attempted first; valid checkpoints are combined.
    #[serde(default)]
    pub gle_checkpoints: Vec<String>,
    /// Optional weights for gle_checkpoints (same order). If missing or invalid,
    /// uniform weighting is used across loaded checkpoints.
    #[serde(default)]
    pub gle_checkpoint_weights: Option<Vec<f32>>,
}

impl Default for LanguageConfig {
    fn default() -> Self {
        Self {
            encoder: EncoderPreset::default(),
            bridge_output_dim: DEFAULT_BRIDGE_DIM,
            ema_alpha: DEFAULT_EMA_ALPHA,
            ood_similarity_threshold: 0.15,
            gle_http_endpoint: None,
            gle_checkpoint: None,
            gle_checkpoints: Vec::new(),
            gle_checkpoint_weights: None,
        }
    }
}

/// Optional causal slot in the sentiment **joint index** (BM25 / retrieval), placed before the witness.
/// Stripped with the user prefix when displaying (see [`strip_sentiment_lattice_witness_for_display`]).
pub const SENTIMENT_CAUSAL_INDEX_CORE: &str = "__GF_CAUSAL__";

/// Stable witness marker between user text and `expected_response` in sentiment lattice bodies (Phase A.1).
///
/// Must be a single tokenizer “word” (see [`crate::spectral::tokenize`]): it survives `encode`/`decode`.
/// **Do not** match only the spaced form when stripping — `decode` may drop spaces around `_`, so use
/// [`strip_sentiment_lattice_witness_for_display`] which searches for this core substring.
pub const SENTIMENT_LATTICE_WITNESS_CORE: &str = "__GROWFORMER_SENT_WITNESS__";

/// Back-compat name for older call sites / tests.
pub const SENTIMENT_LATTICE_WITNESS_SEP: &str = " __GROWFORMER_SENT_WITNESS__ ";

fn slug_ident(part: &str) -> String {
    let mut out = String::new();
    let mut prev_underscore = false;
    for c in part.to_ascii_lowercase().chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c);
            prev_underscore = false;
        } else if !prev_underscore && !out.is_empty() {
            out.push('_');
            prev_underscore = true;
        }
    }
    out.trim_matches('_').to_string()
}

/// Compact BM25 / graph keyword token aligned with [`CausalAnnotation::index_token`].
pub fn causal_index_token(causal_type: &str, connector: &str) -> String {
    let t = slug_ident(causal_type);
    let t = if t.is_empty() { "unknown".into() } else { t };
    let c = slug_ident(connector);
    let c = if c.is_empty() { "none".to_string() } else { c };
    format!("gfcausal_t_{}_c_{}", t, c)
}

/// Optional second BM25 token: `gfcausal_st_<subtype>` (e.g. `retrospective_framing`).
pub fn causal_subtype_index_token(subtype: &str) -> String {
    let s = slug_ident(subtype);
    if s.is_empty() {
        return String::new();
    }
    format!("gfcausal_st_{}", s)
}

/// Optional causal annotation on JSONL training rows (`causal` object). Drives joint-index tokens only until Brain B exists.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct CausalAnnotation {
    /// e.g. `direct`, `compensatory`, `contrastive`, `explanatory`, `counterfactual`, `concessive`, `inferential`
    #[serde(default)]
    pub causal_type: String,
    /// Surface cue: `so`, `despite`, `at least`, …
    #[serde(default)]
    pub connector: Option<String>,
    #[serde(default)]
    pub cause_span: Option<String>,
    #[serde(default)]
    pub effect_span: Option<String>,
    /// Links contrastive / counterfactual minimal pairs for dataset audits (optional).
    #[serde(default)]
    pub contrast_group: Option<String>,
    /// Finer class: e.g. `retrospective_framing`, `interventional_counterfactual` (see `GROWFORMER_CAUSAL_AI.md`).
    #[serde(default)]
    pub causal_subtype: Option<String>,
    /// Apparent sentiment before causal/retrospective resolution (e.g. `negative_mild` for "losing $5000 was the best thing").
    #[serde(default)]
    pub surface_valence: Option<String>,
    /// Settled sentiment after full causal resolution (e.g. `positive_strong` after retrospective reframe).
    #[serde(default)]
    pub resolved_valence: Option<String>,
}

impl CausalAnnotation {
    pub fn is_active(&self) -> bool {
        !self.causal_type.trim().is_empty()
    }

    /// Primary index token: `gfcausal_t_<type>_c_<connector>`.
    pub fn index_token(&self) -> String {
        causal_index_token(
            self.causal_type.trim(),
            self.connector.as_deref().unwrap_or("").trim(),
        )
    }

    /// Space-separated causal index tokens for the joint lattice body (type+connector, optional subtype, optional valence pair).
    pub fn joint_index_tokens(&self) -> String {
        let base = self.index_token();
        let mut parts = vec![base];
        if let Some(st) = self
            .causal_subtype
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(causal_subtype_index_token)
            .filter(|s| !s.is_empty())
        {
            parts.push(st);
        }
        if let Some(sv) = self
            .surface_valence
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            parts.push(format!("gfcausal_sv_{}", slug_ident(sv)));
        }
        if let Some(rv) = self
            .resolved_valence
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            parts.push(format!("gfcausal_rv_{}", slug_ident(rv)));
        }
        parts.join(" ")
    }
}

/// Joint index string for sentiment Paramecium programs: user line + response so retrieval BM25 sees scenario tokens.
pub fn sentiment_lattice_index_body(user: &str, response: &str) -> String {
    sentiment_lattice_index_body_with_causal(user, response, None)
}

/// Same as [`sentiment_lattice_index_body`], with an optional causal slot for BM25 / signature alignment.
pub fn sentiment_lattice_index_body_with_causal(
    user: &str,
    response: &str,
    causal: Option<&CausalAnnotation>,
) -> String {
    const MAX_USER_CHARS: usize = 480;
    let u = user.trim();
    let u_trunc: String = if u.chars().count() > MAX_USER_CHARS {
        u.chars().take(MAX_USER_CHARS).collect()
    } else {
        u.to_string()
    };
    let u_trunc = u_trunc.trim_end().replace(['\n', '\r'], " ");
    let causal_chunk = causal
        .filter(|c| c.is_active())
        .map(|c| {
            format!(
                " {} {}",
                SENTIMENT_CAUSAL_INDEX_CORE,
                c.joint_index_tokens()
            )
        })
        .unwrap_or_default();
    format!(
        "{}{} {} {}",
        u_trunc,
        causal_chunk,
        SENTIMENT_LATTICE_WITNESS_CORE,
        response.trim().replace(['\n', '\r'], " ")
    )
}

/// True when this sample should use [`sentiment_lattice_index_body`] for lattice text (vs `expected_response` only).
pub fn should_use_sentiment_joint_index(s: &LanguageSample) -> bool {
    s.action_target.as_deref() == Some("sentiment") || s.domain.eq_ignore_ascii_case("sentiment")
}

/// Strip the indexed user prefix before showing lattice text to the user.
pub fn strip_sentiment_lattice_witness_for_display(full: &str) -> String {
    if let Some(idx) = full.find(SENTIMENT_LATTICE_WITNESS_CORE) {
        let rest = &full[idx + SENTIMENT_LATTICE_WITNESS_CORE.len()..];
        return rest.trim_start().to_string();
    }
    if let Some((_u, rest)) = full.split_once("\n---\n") {
        return rest.trim_start().to_string();
    }
    full.to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageSample {
    pub domain: String,
    pub text: String,
    pub semantic_intent: String,
    pub action_target: Option<String>,
    pub policy_regime: String,
    pub language_channel: String,
    pub expected_response: Option<String>,
    pub expected_code: Option<String>,
    #[serde(default)]
    pub causal: Option<CausalAnnotation>,
    /// Prior conversation turns as `(role, text)` pairs (role is "user" or "pet"/"agent").
    /// Empty for single-turn samples. Used to build history-aware generation conditioning
    /// that matches the runtime conversation context blend (see service `update_geometric_context`).
    #[serde(default)]
    pub history: Vec<(String, String)>,
    /// 1-based conversation turn index. 1 for the opening turn, >1 for follow-ups.
    #[serde(default = "default_conversation_turn")]
    pub conversation_turn: u32,
}

fn default_conversation_turn() -> u32 {
    1
}

// --- Brain-training JSONL directory scans ------------------------------------

/// Inference guardrails use a different schema (`kind` / `intent`), not [`LanguageSample`] rows.
#[inline]
pub fn is_inference_guardrails_jsonl_filename(name: &str) -> bool {
    name == "inference_guardrails.jsonl"
}

/// Whether a basename in a training `data_dir` should be merged into brain training / M5-style calibration loads.
///
/// Every `*.jsonl` sample corpus is included except [`is_inference_guardrails_jsonl_filename`] and `eval_*.jsonl` holdouts.
#[inline]
pub fn is_brain_training_jsonl_filename(name: &str) -> bool {
    name.ends_with(".jsonl")
        && !is_inference_guardrails_jsonl_filename(name)
        && !name.starts_with("eval_")
}

#[derive(Deserialize)]
struct HistoryTurnRow {
    #[serde(default)]
    role: String,
    #[serde(default)]
    text: String,
}

/// Pet companion rows nest `history` / `conversation_turn` under a `pet` object.
#[derive(Deserialize)]
struct PetRow {
    #[serde(default)]
    history: Option<Vec<HistoryTurnRow>>,
    #[serde(default)]
    conversation_turn: Option<u32>,
}

#[derive(Deserialize)]
struct JsonlLanguageSampleRow {
    text: String,
    #[serde(default)]
    semantic_intent: Option<String>,
    #[serde(default)]
    intent: Option<String>,
    #[serde(default)]
    domain: Option<String>,
    #[serde(default)]
    action_target: Option<String>,
    #[serde(default)]
    policy_regime: Option<String>,
    #[serde(default)]
    language_channel: Option<String>,
    #[serde(default)]
    expected_response: Option<String>,
    #[serde(default)]
    expected_code: Option<String>,
    #[serde(default)]
    causal: Option<CausalAnnotation>,
    /// Multi-turn history may appear at the top level …
    #[serde(default)]
    history: Option<Vec<HistoryTurnRow>>,
    #[serde(default)]
    conversation_turn: Option<u32>,
    /// … or nested under a `pet` object (pet companion corpora).
    #[serde(default)]
    pet: Option<PetRow>,
}

/// Load newline-delimited [`LanguageSample`] records from one JSONL file (brain / distill corpora).
pub fn load_language_samples_jsonl(path: &str) -> Result<Vec<LanguageSample>, String> {
    let file = File::open(path).map_err(|e| format!("open failed: {}", e))?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("line {} read failed: {}", idx + 1, e))?;
        if line.trim().is_empty() {
            continue;
        }
        let rec: JsonlLanguageSampleRow = serde_json::from_str(&line)
            .map_err(|e| format!("line {} json parse failed: {}", idx + 1, e))?;
        let intent = rec
            .semantic_intent
            .or(rec.intent)
            .unwrap_or_else(|| "unknown_intent".to_string());
        // History / turn index may be at the top level or nested under `pet`.
        let mut history_rows = rec.history;
        let mut conv_turn = rec.conversation_turn;
        if let Some(pet) = rec.pet {
            if history_rows.is_none() {
                history_rows = pet.history;
            }
            if conv_turn.is_none() {
                conv_turn = pet.conversation_turn;
            }
        }
        let history: Vec<(String, String)> = history_rows
            .unwrap_or_default()
            .into_iter()
            .filter(|h| !h.text.trim().is_empty())
            .map(|h| (h.role, h.text))
            .collect();
        let conversation_turn = conv_turn.unwrap_or(if history.is_empty() { 1 } else { 2 });
        out.push(LanguageSample {
            domain: rec.domain.unwrap_or_else(|| "custom".to_string()),
            text: rec.text,
            semantic_intent: intent,
            action_target: rec.action_target,
            policy_regime: rec.policy_regime.unwrap_or_else(|| "default".to_string()),
            language_channel: rec
                .language_channel
                .unwrap_or_else(|| "english".to_string()),
            expected_response: rec.expected_response,
            expected_code: rec.expected_code,
            causal: rec.causal,
            history,
            conversation_turn,
        });
    }
    Ok(out)
}

/// Append all brain-training JSONL files from `dir` into `all` (sorted by filename).
pub fn append_language_samples_from_training_jsonl_dir(
    all: &mut Vec<LanguageSample>,
    dir: &Path,
) -> Result<(), String> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("read_dir failed ({}): {}", dir.display(), e))?
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let name = entry.file_name().to_string_lossy().to_string();
        if !is_brain_training_jsonl_filename(&name) {
            continue;
        }
        let path = entry.path();
        let samples = load_language_samples_jsonl(path.to_str().unwrap())?;
        println!("  loaded {}: {} samples", path.display(), samples.len());
        all.extend(samples);
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationDataset {
    pub samples: Vec<LanguageSample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationRequirements {
    pub min_domains: usize,
    pub min_samples_per_domain: usize,
    pub multilingual_min_ratio: f32,
    pub multilingual_required: bool,
}

impl Default for CalibrationRequirements {
    fn default() -> Self {
        Self {
            min_domains: 8,
            min_samples_per_domain: 500,
            multilingual_min_ratio: 0.20,
            multilingual_required: false,
        }
    }
}

impl CalibrationDataset {
    pub fn validate(&self, req: &CalibrationRequirements) -> Result<CalibrationCoverage, String> {
        if self.samples.is_empty() {
            return Err("calibration dataset is empty".to_string());
        }
        let mut counts: HashMap<&str, usize> = HashMap::new();
        let mut multilingual = 0usize;
        for s in &self.samples {
            *counts.entry(s.domain.as_str()).or_insert(0) += 1;
            if !s.language_channel.eq_ignore_ascii_case("english") {
                multilingual += 1;
            }
            if s.semantic_intent.is_empty()
                || s.policy_regime.is_empty()
                || s.language_channel.is_empty()
            {
                return Err(
                    "calibration samples must include intent/policy/language labels".to_string(),
                );
            }
        }
        if counts.len() < req.min_domains {
            return Err(format!(
                "need at least {} domains, got {}",
                req.min_domains,
                counts.len()
            ));
        }
        let low_coverage = counts.values().any(|&n| n < req.min_samples_per_domain);
        if low_coverage {
            return Err(format!(
                "each domain needs at least {} samples",
                req.min_samples_per_domain
            ));
        }
        let multilingual_ratio = multilingual as f32 / self.samples.len() as f32;
        if req.multilingual_required && multilingual_ratio < req.multilingual_min_ratio {
            return Err(format!(
                "need multilingual ratio >= {:.2}, got {:.3}",
                req.multilingual_min_ratio, multilingual_ratio
            ));
        }
        Ok(CalibrationCoverage {
            domains: counts.len(),
            samples: self.samples.len(),
            multilingual_ratio,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationCoverage {
    pub domains: usize,
    pub samples: usize,
    pub multilingual_ratio: f32,
}

pub trait LanguageEncoder {
    fn output_dim(&self) -> usize;
    fn encode(&self, text: &str) -> Vec<f32>;
}

/// Lightweight deterministic encoder used for M1 plumbing and tests.
/// This preserves the interface/shape contract without introducing external runtimes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashingLanguageEncoder {
    pub preset: EncoderPreset,
}

impl HashingLanguageEncoder {
    pub fn new(preset: EncoderPreset) -> Self {
        Self { preset }
    }
}

impl LanguageEncoder for HashingLanguageEncoder {
    fn output_dim(&self) -> usize {
        self.preset.output_dim()
    }

    fn encode(&self, text: &str) -> Vec<f32> {
        let dim = self.output_dim();
        let mut v = vec![0.0f32; dim];
        if dim == 0 {
            return v;
        }
        // Lightweight lexical anchors improve separability for the placeholder encoder.
        // Real deployments should replace this with a true transformer encoder runtime.
        // Domain-family anchors used by the synthetic dataset in main.rs.
        let customer_support_keywords = [
            "customer",
            "support",
            "account",
            "password",
            "billing",
            "ticket",
            "refund",
            "login",
            "unlock",
            "subscription",
            "recovery",
            "helpdesk",
        ];
        let coding_tool_keywords = [
            "coding", "code", "rust", "function", "debug", "compiler", "sql", "query", "parser",
            "tool", "stack", "pointer", "serde", "index", "module",
        ];
        let knowledge_qa_keywords = [
            "knowledge",
            "qa",
            "fact",
            "explain",
            "definition",
            "what",
            "why",
            "how",
            "answer",
            "reference",
            "documentation",
        ];
        let safety_refusal_keywords = [
            "safety",
            "policy",
            "refuse",
            "blocked",
            "forbidden",
            "harmful",
            "disallowed",
            "compliance",
            "unsafe",
            "restricted",
        ];
        let procedural_instruction_keywords = [
            "procedure",
            "instruction",
            "step",
            "follow",
            "sequence",
            "workflow",
            "checklist",
            "process",
            "guide",
        ];
        let short_conversation_keywords = [
            "hello",
            "hi",
            "thanks",
            "ok",
            "yes",
            "no",
            "greetings",
            "bye",
            "chat",
        ];
        let multi_turn_followup_keywords = [
            "followup", "continue", "previous", "earlier", "context", "as-said", "next", "again",
            "clarify", "thread",
        ];
        let adversarial_noisy_keywords = [
            "adversarial",
            "noisy",
            "jailbreak",
            "prompt-injection",
            "ignore",
            "override",
            "garbled",
            "nonsense",
            "obfuscated",
        ];
        // Sentiment anchors: loaded once from inference TOML (single source of truth).
        // Union of positive_anchor_tokens + bipolar_positive_tokens (and negative equivalents)
        // so the encoder vocabulary never drifts from the rules the inference engine uses.
        let (positive_sentiment_set, negative_sentiment_set) = sentiment_anchor_sets();
        let stopwords = [
            "the", "a", "an", "and", "or", "to", "for", "of", "in", "on", "with", "is", "are",
            "this", "that", "please",
        ];
        for token in text.split_whitespace() {
            let lower = token.to_ascii_lowercase();
            if stopwords.contains(&lower.as_str()) {
                continue;
            }
            if lower.chars().all(|c| c.is_ascii_digit()) {
                continue;
            }
            let bytes = lower.as_bytes();
            if bytes.is_empty() {
                continue;
            }
            if dim >= 10 {
                if customer_support_keywords.contains(&lower.as_str()) {
                    v[0] += 4.0;
                }
                if coding_tool_keywords.contains(&lower.as_str()) {
                    v[1] += 4.0;
                }
                if knowledge_qa_keywords.contains(&lower.as_str()) {
                    v[2] += 4.0;
                }
                if safety_refusal_keywords.contains(&lower.as_str()) {
                    v[3] += 4.0;
                }
                if procedural_instruction_keywords.contains(&lower.as_str()) {
                    v[4] += 4.0;
                }
                if short_conversation_keywords.contains(&lower.as_str()) {
                    v[5] += 4.0;
                }
                if multi_turn_followup_keywords.contains(&lower.as_str()) {
                    v[6] += 4.0;
                }
                if adversarial_noisy_keywords.contains(&lower.as_str()) {
                    v[7] += 4.0;
                }
                if positive_sentiment_set.contains(lower.as_str()) {
                    v[8] += 4.0;
                }
                if negative_sentiment_set.contains(lower.as_str()) {
                    v[9] += 4.0;
                }
            } else if dim >= 8 {
                if customer_support_keywords.contains(&lower.as_str()) {
                    v[0] += 4.0;
                }
                if coding_tool_keywords.contains(&lower.as_str()) {
                    v[1] += 4.0;
                }
                if knowledge_qa_keywords.contains(&lower.as_str()) {
                    v[2] += 4.0;
                }
                if safety_refusal_keywords.contains(&lower.as_str()) {
                    v[3] += 4.0;
                }
                if procedural_instruction_keywords.contains(&lower.as_str()) {
                    v[4] += 4.0;
                }
                if short_conversation_keywords.contains(&lower.as_str()) {
                    v[5] += 4.0;
                }
                if multi_turn_followup_keywords.contains(&lower.as_str()) {
                    v[6] += 4.0;
                }
                if adversarial_noisy_keywords.contains(&lower.as_str()) {
                    v[7] += 4.0;
                }
            }
            let mut h0: u64 = 1469598103934665603;
            let mut h1: u64 = 1099511628211;
            for &b in bytes {
                h0 ^= b as u64;
                h0 = h0.wrapping_mul(1099511628211);
                h1 = h1.wrapping_mul(33).wrapping_add(b as u64);
            }
            let i0 = (h0 as usize) % dim;
            let i1 = (h1 as usize) % dim;
            v[i0] += 1.0;
            if i1 != i0 {
                v[i1] += 0.5;
            }
        }
        l2_normalize(&mut v);
        v
    }
}

use std::sync::OnceLock;

/// Union of positive/negative anchor + bipolar tokens from inference TOML.
/// Loaded once; the encoder never carries a private word list that can drift.
fn sentiment_anchor_sets() -> (&'static HashSet<String>, &'static HashSet<String>) {
    static POS: OnceLock<HashSet<String>> = OnceLock::new();
    static NEG: OnceLock<HashSet<String>> = OnceLock::new();

    let pos = POS.get_or_init(|| {
        let rt = crate::inference::inference_toml::inference_rules_runtime();
        let mut s: HashSet<String> = rt
            .positive_anchor_tokens
            .iter()
            .map(|w| w.to_ascii_lowercase())
            .collect();
        for w in &rt.bipolar_positive_tokens {
            s.insert(w.to_ascii_lowercase());
        }
        s
    });
    let neg = NEG.get_or_init(|| {
        let rt = crate::inference::inference_toml::inference_rules_runtime();
        let mut s: HashSet<String> = rt
            .negative_anchor_tokens
            .iter()
            .map(|w| w.to_ascii_lowercase())
            .collect();
        for w in &rt.bipolar_negative_tokens {
            s.insert(w.to_ascii_lowercase());
        }
        s
    });
    (pos, neg)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmaSmoother {
    pub alpha: f32,
    pub state: Option<Vec<f32>>,
}

impl EmaSmoother {
    pub fn new(alpha: f32) -> Self {
        Self { alpha, state: None }
    }

    pub fn update(&mut self, x: &[f32]) -> Vec<f32> {
        if x.is_empty() {
            return vec![];
        }
        let out = match &mut self.state {
            Some(state) if state.len() == x.len() => {
                for (s, &xi) in state.iter_mut().zip(x.iter()) {
                    *s = self.alpha * xi + (1.0 - self.alpha) * *s;
                }
                state.clone()
            }
            _ => {
                self.state = Some(x.to_vec());
                x.to_vec()
            }
        };
        out
    }

    pub fn reset(&mut self) {
        self.state = None;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageBridge {
    pub input_dim: usize,
    pub output_dim: usize,
    pub projection: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
    pub ln_gamma: Vec<f32>,
    pub ln_beta: Vec<f32>,
    pub confidence_head: Vec<f32>,
    pub confidence_bias: f32,
    pub input_mean: Vec<f32>,
    pub input_std: Vec<f32>,
    pub calibrated: bool,
    pub frozen: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeOutput {
    pub routed_vector: Vec<f32>,
    pub confidence: f32,
}

pub const DEFAULT_ADAPTER_RANK: usize = 8;
pub const DEFAULT_ADAPTER_L2: f32 = 1e-4;

/// Low-rank per-group adapter: `z_g = z_shared + A @ B @ h_raw`.
/// B projects raw encoder dim down to rank, A projects rank up to bridge dim.
/// Trained per-group alongside generation heads; shared bridge stays frozen for routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupAdapter {
    pub raw_dim: usize,
    pub bridge_dim: usize,
    pub rank: usize,
    pub b_down: Vec<Vec<f32>>, // rank × raw_dim
    pub a_up: Vec<Vec<f32>>,   // bridge_dim × rank
    pub l2_weight: f32,
    pub frozen: bool,
}

impl GroupAdapter {
    pub fn new(raw_dim: usize, bridge_dim: usize, rank: usize) -> Self {
        let scale = (2.0 / (raw_dim + rank) as f32).sqrt();
        let mut b_down = vec![vec![0.0f32; raw_dim]; rank];
        let mut a_up = vec![vec![0.0f32; rank]; bridge_dim];
        // Kaiming-style init: small deterministic spread so groups diverge immediately
        for (r, row) in b_down.iter_mut().enumerate() {
            for (i, w) in row.iter_mut().enumerate() {
                let hash = ((r * 31 + i * 17) % 1000) as f32 / 1000.0 - 0.5;
                *w = hash * scale;
            }
        }
        for (o, row) in a_up.iter_mut().enumerate() {
            for (r, w) in row.iter_mut().enumerate() {
                let hash = ((o * 13 + r * 29) % 1000) as f32 / 1000.0 - 0.5;
                *w = hash * scale * 0.1; // near-zero init so adapter starts as identity-ish
            }
        }
        Self {
            raw_dim,
            bridge_dim,
            rank,
            b_down,
            a_up,
            l2_weight: DEFAULT_ADAPTER_L2,
            frozen: false,
        }
    }

    /// Forward: returns the adapter delta to add to z_shared.
    pub fn forward(&self, h_raw: &[f32]) -> Vec<f32> {
        debug_assert_eq!(h_raw.len(), self.raw_dim);
        // hidden = B @ h_raw  (rank-dim)
        let mut hidden = vec![0.0f32; self.rank];
        for (r, row) in self.b_down.iter().enumerate() {
            let mut acc = 0.0f32;
            for (i, &w) in row.iter().enumerate() {
                acc += w * h_raw[i];
            }
            hidden[r] = if acc > 0.0 { acc } else { 0.01 * acc }; // LeakyReLU
        }
        // delta = A @ hidden  (bridge_dim)
        let mut delta = vec![0.0f32; self.bridge_dim];
        for (o, row) in self.a_up.iter().enumerate() {
            let mut acc = 0.0f32;
            for (r, &w) in row.iter().enumerate() {
                acc += w * hidden[r];
            }
            delta[o] = acc;
        }
        delta
    }

    /// Adapt z_shared using raw embedding: z_g = z_shared + adapter.forward(h_raw)
    pub fn adapt(&self, z_shared: &[f32], h_raw: &[f32]) -> Vec<f32> {
        let delta = self.forward(h_raw);
        z_shared
            .iter()
            .zip(delta.iter())
            .map(|(&z, &d)| z + d)
            .collect()
    }

    /// SGD training step: given the gradient signal from gen head loss.
    /// `cond_grad` is the gradient of loss w.r.t. the adapted vector (bridge_dim).
    /// `h_raw` is the raw encoder vector for this sample.
    #[cfg(feature = "training")]
    pub fn train_step(&mut self, h_raw: &[f32], cond_grad: &[f32], lr: f32) {
        if self.frozen {
            return;
        }
        debug_assert_eq!(h_raw.len(), self.raw_dim);
        debug_assert_eq!(cond_grad.len(), self.bridge_dim);

        // Recompute forward activations for backward pass
        let mut hidden = vec![0.0f32; self.rank];
        let mut hidden_pre_relu = vec![0.0f32; self.rank];
        for (r, row) in self.b_down.iter().enumerate() {
            let mut acc = 0.0f32;
            for (i, &w) in row.iter().enumerate() {
                acc += w * h_raw[i];
            }
            hidden_pre_relu[r] = acc;
            hidden[r] = if acc > 0.0 { acc } else { 0.01 * acc };
        }

        // grad_A[o][r] = cond_grad[o] * hidden[r]
        for (o, row) in self.a_up.iter_mut().enumerate() {
            let go = cond_grad[o];
            for (r, w) in row.iter_mut().enumerate() {
                let grad = go * hidden[r] + self.l2_weight * *w;
                *w -= lr * grad;
            }
        }

        // grad through ReLU: d_hidden[r] = sum_o(cond_grad[o] * A[o][r]) * relu'(pre)
        let mut d_hidden = vec![0.0f32; self.rank];
        for r in 0..self.rank {
            let mut acc = 0.0f32;
            for (o, row) in self.a_up.iter().enumerate() {
                acc += cond_grad[o] * row[r];
            }
            d_hidden[r] = if hidden_pre_relu[r] > 0.0 {
                acc
            } else {
                0.01 * acc
            };
        }

        // grad_B[r][i] = d_hidden[r] * h_raw[i]
        for (r, row) in self.b_down.iter_mut().enumerate() {
            let dr = d_hidden[r];
            for (i, w) in row.iter_mut().enumerate() {
                let grad = dr * h_raw[i] + self.l2_weight * *w;
                *w -= lr * grad;
            }
        }
    }

    pub fn param_count(&self) -> usize {
        self.rank * self.raw_dim + self.bridge_dim * self.rank
    }

    pub fn freeze(&mut self) {
        self.frozen = true;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationReport {
    pub encoder_model: String,
    pub input_dim: usize,
    pub output_dim: usize,
    pub coverage: CalibrationCoverage,
    pub frozen_after_calibration: bool,
}

impl LanguageBridge {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        // Deterministic bucket projection preserves token-hash locality better than
        // random projection for this lightweight encoder placeholder.
        // TODO: replace with encoder
        let mut projection = vec![vec![0.0f32; input_dim]; output_dim];
        if output_dim > 0 {
            let bucket_size = ((input_dim as f32) / (output_dim as f32)).ceil() as usize;
            let scale = 1.0 / bucket_size.max(1) as f32;
            for (o, row) in projection.iter_mut().enumerate() {
                for (i, w) in row.iter_mut().enumerate() {
                    if i % output_dim == o {
                        *w = scale;
                    }
                }
            }
        }
        Self {
            input_dim,
            output_dim,
            projection,
            bias: vec![0.0; output_dim],
            ln_gamma: vec![1.0; output_dim],
            ln_beta: vec![0.0; output_dim],
            confidence_head: vec![0.0; output_dim],
            confidence_bias: 0.0,
            input_mean: vec![0.0; input_dim],
            input_std: vec![1.0; input_dim],
            calibrated: false,
            frozen: false,
        }
    }

    pub fn calibrate_global<E: LanguageEncoder + ?Sized>(
        &mut self,
        encoder: &E,
        dataset: &CalibrationDataset,
        requirements: &CalibrationRequirements,
        freeze_after: bool,
    ) -> Result<CalibrationReport, String> {
        if self.frozen {
            return Err("bridge is frozen; recalibration blocked".to_string());
        }
        if encoder.output_dim() != self.input_dim {
            return Err(format!(
                "encoder dim {} != bridge input {}",
                encoder.output_dim(),
                self.input_dim
            ));
        }
        let coverage = dataset.validate(requirements)?;
        let n = dataset.samples.len() as f32;
        let mut mean = vec![0.0f32; self.input_dim];
        let mut sq = vec![0.0f32; self.input_dim];
        let mut unique_intents = HashSet::new();
        for s in &dataset.samples {
            unique_intents.insert(s.semantic_intent.clone());
            let x = encoder.encode(&s.text);
            if x.len() != self.input_dim {
                return Err("encoder emitted unexpected dimension during calibration".to_string());
            }
            for i in 0..self.input_dim {
                mean[i] += x[i];
                sq[i] += x[i] * x[i];
            }
        }
        for i in 0..self.input_dim {
            mean[i] /= n;
            let var = (sq[i] / n) - mean[i] * mean[i];
            self.input_std[i] = var.max(1e-6).sqrt();
        }
        self.input_mean = mean;

        // Confidence head uses calibrated norm target; this is a simple proxy that
        // remains stable across domains once the bridge is frozen.
        let intent_scale = (unique_intents.len().max(1) as f32).ln_1p().max(1.0);
        for w in &mut self.confidence_head {
            *w = 1.0 / (self.output_dim as f32 * intent_scale);
        }
        self.confidence_bias = 0.0;

        self.calibrated = true;
        self.frozen = freeze_after;

        Ok(CalibrationReport {
            encoder_model: format!("{} (placeholder runtime)", encoder.output_dim()),
            input_dim: self.input_dim,
            output_dim: self.output_dim,
            coverage,
            frozen_after_calibration: self.frozen,
        })
    }

    pub fn freeze(&mut self) {
        self.frozen = true;
    }

    pub fn project(&self, encoder_vec: &[f32]) -> Result<BridgeOutput, String> {
        if encoder_vec.len() != self.input_dim {
            return Err(format!(
                "bridge expected {} dims, got {}",
                self.input_dim,
                encoder_vec.len()
            ));
        }
        let mut normalized = vec![0.0f32; self.input_dim];
        for i in 0..self.input_dim {
            normalized[i] = ((encoder_vec[i] as f64 - self.input_mean[i] as f64)
                / (self.input_std[i].max(1e-6) as f64)) as f32;
        }
        let mut z = vec![0.0f32; self.output_dim];
        for (o, zo) in z.iter_mut().enumerate() {
            let mut acc = self.bias[o] as f64;
            for i in 0..self.input_dim {
                acc += (self.projection[o][i] as f64) * (normalized[i] as f64);
            }
            *zo = acc as f32;
        }
        layer_norm_affine(&mut z, &self.ln_gamma, &self.ln_beta);
        let mut confidence_logit = self.confidence_bias as f64;
        for (w, v) in self.confidence_head.iter().zip(z.iter()) {
            confidence_logit += (*w as f64) * (v.abs() as f64);
        }
        let confidence = sigmoid(confidence_logit as f32);
        l2_normalize(&mut z);
        Ok(BridgeOutput {
            routed_vector: z,
            confidence,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageRoutingDecision {
    pub chosen_group_id: Option<GroupId>,
    pub best_similarity: f32,
    pub second_similarity: f32,
    pub margin: f32,
    pub confidence: f32,
    pub rejected_as_ood: bool,
    /// True when RoutingEntropyGuard fired and we fell back from learned-router argmax.
    #[serde(default)]
    pub routing_entropy_guard_triggered: bool,
}

pub fn route_language_embedding(
    embedding_library: &[GroupEmbedding],
    language_vec: &[f32],
    confidence: f32,
    ood_threshold: f32,
) -> LanguageRoutingDecision {
    let mut sims: Vec<(GroupId, f32)> = embedding_library
        .iter()
        .filter(|e| !e.language_vector.is_empty() && e.language_vector.len() == language_vec.len())
        .map(|e| {
            (
                e.group_id,
                cosine_similarity(language_vec, &e.language_vector),
            )
        })
        .collect();
    sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let (best_gid, best) = sims.first().copied().unwrap_or((0, -1.0));
    let second = sims.get(1).map(|x| x.1).unwrap_or(-1.0);
    let margin = best - second;
    let reject = sims.is_empty() || best < ood_threshold;
    LanguageRoutingDecision {
        chosen_group_id: if reject { None } else { Some(best_gid) },
        best_similarity: best,
        second_similarity: second,
        margin,
        confidence,
        rejected_as_ood: reject,
        routing_entropy_guard_triggered: false,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageRuntime {
    pub config: LanguageConfig,
    pub encoder: HashingLanguageEncoder,
    pub bridge: LanguageBridge,
    pub smoother: EmaSmoother,
    #[serde(default)]
    preloaded_students: Vec<GleStudentCheckpoint>,
    /// Corpus-built dictionary for the Clifford+ChunkCodec encoder path.
    /// Populated during training, serialized into the brain package.
    #[serde(default)]
    pub preloaded_dictionary: Option<crate::spectral::TokenDictionary>,
}

impl LanguageRuntime {
    pub fn new(config: LanguageConfig) -> Self {
        let encoder = HashingLanguageEncoder::new(config.encoder.clone());
        let input_dim = configured_encoder_dim(&config).unwrap_or_else(|| encoder.output_dim());
        let bridge = LanguageBridge::new(input_dim, config.bridge_output_dim);
        let smoother = EmaSmoother::new(config.ema_alpha);
        Self {
            config,
            encoder,
            bridge,
            smoother,
            preloaded_students: Vec::new(),
            preloaded_dictionary: None,
        }
    }

    pub fn calibrate(
        &mut self,
        dataset: &CalibrationDataset,
        requirements: &CalibrationRequirements,
    ) -> Result<CalibrationReport, String> {
        let encoder = self.build_encoder();
        let mut report =
            self.bridge
                .calibrate_global(encoder.as_ref(), dataset, requirements, true)?;
        report.encoder_model = self.config.encoder.model_name();
        Ok(report)
    }

    pub fn bridge_text(&mut self, text: &str) -> Result<BridgeOutput, String> {
        let encoder = self.build_encoder();
        let encoded = encoder.encode(text);
        let out = self.bridge.project(&encoded)?;
        let smoothed = self.smoother.update(&out.routed_vector);
        Ok(BridgeOutput {
            routed_vector: smoothed,
            confidence: out.confidence,
        })
    }

    /// Stateless bridge path for independent turns / offline evaluation.
    pub fn bridge_text_stateless(&self, text: &str) -> Result<BridgeOutput, String> {
        let encoder = self.build_encoder();
        let encoded = encoder.encode(text);
        self.bridge.project(&encoded)
    }

    /// Returns (raw_encoder_vec, bridged_output). The raw vector preserves full information
    /// for conditioning generation heads; the bridged vector is for routing.
    pub fn encode_and_bridge(&self, text: &str) -> Result<(Vec<f32>, BridgeOutput), String> {
        let encoder = self.build_encoder();
        let encoded = encoder.encode(text);
        let bridged = self.bridge.project(&encoded)?;
        Ok((encoded, bridged))
    }

    fn build_encoder(&self) -> Box<dyn LanguageEncoder> {
        if let EncoderPreset::CliffordE8 = &self.config.encoder {
            return Box::new(CliffordLanguageEncoder::new(
                self.preloaded_dictionary.clone(),
            ));
        }
        if let Some(enc) = self.build_encoder_from_preloaded() {
            return enc;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(enc) = self.build_encoder_from_disk() {
                return enc;
            }
        }
        #[cfg(feature = "native")]
        {
            if let (EncoderPreset::BertClass, Some(endpoint)) =
                (&self.config.encoder, self.config.gle_http_endpoint.clone())
            {
                return Box::new(HttpGleEncoder::new(
                    endpoint,
                    self.config.encoder.model_name(),
                    self.config.encoder.output_dim(),
                    self.encoder.clone(),
                ));
            }
        }
        Box::new(self.encoder.clone())
    }

    fn build_encoder_from_preloaded(&self) -> Option<Box<dyn LanguageEncoder>> {
        if self.preloaded_students.is_empty() {
            return None;
        }
        if self.preloaded_students.len() == 1 {
            return Some(Box::new(GrowformerLanguageEncoder::new(
                self.preloaded_students[0].clone(),
            )));
        }
        Some(Box::new(MultiGleEncoder::new(
            self.preloaded_students.clone(),
            self.config.gle_checkpoint_weights.clone(),
        )))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn build_encoder_from_disk(&self) -> Option<Box<dyn LanguageEncoder>> {
        if let EncoderPreset::BertClass = &self.config.encoder {
            let paths = resolved_gle_checkpoint_paths(&self.config);
            if !paths.is_empty() {
                let mut loaded: Vec<GleStudentCheckpoint> = Vec::new();
                let mut target_out_dim: Option<usize> = None;
                for path in paths {
                    if let Ok(student) = GleStudentCheckpoint::load(&path) {
                        let out_dim = student.output_dim();
                        if let Some(target) = target_out_dim {
                            if out_dim == target {
                                loaded.push(student);
                            }
                        } else {
                            target_out_dim = Some(out_dim);
                            loaded.push(student);
                        }
                    }
                }
                if loaded.len() == 1 {
                    return Some(Box::new(GrowformerLanguageEncoder::new(loaded.remove(0))));
                }
                if loaded.len() > 1 {
                    return Some(Box::new(MultiGleEncoder::new(
                        loaded,
                        self.config.gle_checkpoint_weights.clone(),
                    )));
                }
            }
        }
        None
    }

    /// Ensure GLE students are loaded into `preloaded_students` (from disk if needed)
    /// so they survive checkpoint serialization.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn ensure_students_preloaded(&mut self) {
        if !self.preloaded_students.is_empty() {
            return;
        }
        if let EncoderPreset::BertClass = &self.config.encoder {
            let paths = resolved_gle_checkpoint_paths(&self.config);
            let mut target_out_dim: Option<usize> = None;
            for path in paths {
                if let Ok(student) = GleStudentCheckpoint::load(&path) {
                    let out_dim = student.output_dim();
                    if let Some(target) = target_out_dim {
                        if out_dim == target {
                            self.preloaded_students.push(student);
                        }
                    } else {
                        target_out_dim = Some(out_dim);
                        self.preloaded_students.push(student);
                    }
                }
            }
        }
    }

    pub fn preloaded_student_count(&self) -> usize {
        self.preloaded_students.len()
    }

    /// Load GLE student checkpoints from byte slices (WASM-compatible path).
    pub fn load_students_from_bytes(&mut self, data: &[&[u8]]) -> Result<usize, String> {
        let mut count = 0;
        for bytes in data {
            let student = GleStudentCheckpoint::from_bytes(bytes)?;
            self.preloaded_students.push(student);
            count += 1;
        }
        Ok(count)
    }
}

fn configured_encoder_dim(config: &LanguageConfig) -> Option<usize> {
    if let EncoderPreset::CliffordE8 = &config.encoder {
        return Some(CLIFFORD_ENCODER_OUTPUT_DIM);
    }
    #[cfg(not(target_arch = "wasm32"))]
    if let EncoderPreset::BertClass = &config.encoder {
        for path in resolved_gle_checkpoint_paths(config) {
            if let Ok(student) = GleStudentCheckpoint::load(&path) {
                return Some(student.w2.len());
            }
        }
    }
    let _ = config;
    None
}

#[cfg(not(target_arch = "wasm32"))]
fn resolved_gle_checkpoint_paths(config: &LanguageConfig) -> Vec<String> {
    if !config.gle_checkpoints.is_empty() {
        return config.gle_checkpoints.clone();
    }
    config.gle_checkpoint.iter().cloned().collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GleStudentCheckpoint {
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    w2: Vec<Vec<f32>>,
    b2: Vec<f32>,
}

impl GleStudentCheckpoint {
    #[cfg(not(target_arch = "wasm32"))]
    fn load(path: &str) -> Result<Self, String> {
        let json = std::fs::read_to_string(path).map_err(|e| format!("read failed: {}", e))?;
        serde_json::from_str(&json).map_err(|e| format!("parse failed: {}", e))
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        serde_json::from_slice(data).map_err(|e| format!("parse failed: {}", e))
    }

    fn predict(&self, x: &[f32]) -> Vec<f32> {
        let mut h = vec![0.0f32; self.w1.len()];
        for (j, hj) in h.iter_mut().enumerate() {
            let mut acc = self.b1[j] as f64;
            for (i, &xi) in x.iter().enumerate() {
                acc += (self.w1[j][i] as f64) * (xi as f64);
            }
            *hj = (acc as f32).tanh();
        }
        let mut y = vec![0.0f32; self.w2.len()];
        for (o, yo) in y.iter_mut().enumerate() {
            let mut acc = self.b2[o] as f64;
            for (j, &hj) in h.iter().enumerate() {
                acc += (self.w2[o][j] as f64) * (hj as f64);
            }
            *yo = acc as f32;
        }
        l2_normalize(&mut y);
        y
    }

    fn output_dim(&self) -> usize {
        self.w2.len()
    }
}

#[derive(Debug, Clone)]
struct GrowformerLanguageEncoder {
    student: GleStudentCheckpoint,
    base: HashingLanguageEncoder,
}

impl GrowformerLanguageEncoder {
    fn new(student: GleStudentCheckpoint) -> Self {
        let base = HashingLanguageEncoder::new(EncoderPreset::Custom {
            model_name: "tiny-student-hash".to_string(),
            output_dim: 256,
        });
        Self { student, base }
    }
}

impl LanguageEncoder for GrowformerLanguageEncoder {
    fn output_dim(&self) -> usize {
        self.student.w2.len()
    }

    fn encode(&self, text: &str) -> Vec<f32> {
        let x = self.base.encode(text);
        self.student.predict(&x)
    }
}

#[derive(Debug, Clone)]
struct MultiGleEncoder {
    students: Vec<GleStudentCheckpoint>,
    weights: Vec<f32>,
    base: HashingLanguageEncoder,
}

impl MultiGleEncoder {
    fn new(students: Vec<GleStudentCheckpoint>, maybe_weights: Option<Vec<f32>>) -> Self {
        let n = students.len();
        let mut weights = maybe_weights.unwrap_or_default();
        if weights.len() != n || weights.iter().any(|w| *w <= 0.0) {
            weights = vec![1.0; n];
        }
        let sum = weights.iter().sum::<f32>();
        if sum > 1e-8 {
            for w in &mut weights {
                *w /= sum;
            }
        } else {
            let uniform = 1.0 / n as f32;
            weights.fill(uniform);
        }
        let base = HashingLanguageEncoder::new(EncoderPreset::Custom {
            model_name: "tiny-student-hash".to_string(),
            output_dim: 256,
        });
        Self {
            students,
            weights,
            base,
        }
    }
}

impl LanguageEncoder for MultiGleEncoder {
    fn output_dim(&self) -> usize {
        self.students.first().map(|s| s.output_dim()).unwrap_or(0)
    }

    fn encode(&self, text: &str) -> Vec<f32> {
        let x = self.base.encode(text);
        let dim = self.output_dim();
        if dim == 0 || self.students.is_empty() {
            return vec![];
        }
        let mut out = vec![0.0f32; dim];
        for (student, &w) in self.students.iter().zip(self.weights.iter()) {
            let y = student.predict(&x);
            if y.len() != dim {
                continue;
            }
            for i in 0..dim {
                out[i] += w * y[i];
            }
        }
        l2_normalize(&mut out);
        out
    }
}

// ---------------------------------------------------------------------------
// CliffordLanguageEncoder — MLP-free encoder using the neural physics substrate
// ---------------------------------------------------------------------------
//
// Pipeline: 256-d hash → E8 lattice quantize → Cl(1,7) embed → grade extract → L2 norm
//
// Replaces the GLE distilled MLP (256→192→384) with pure geometric algebra.
// Grade structure encodes semantics:
//   grade-1 (8d)  — direction / content signal
//   grade-2 (28d) — relational / causal bivectors (boost + rotation)
//   grade-0 (1d)  — scalar magnitude
// Total: 37 meaningful dimensions, padded to output_dim (default 128).
//
// The E8 quantization step snaps the hash vector to lattice points, providing
// a discrete nonlinearity that regularises hash collisions without learned params.
// The Clifford wedge products between successive 8-d blocks inject cross-subspace
// interaction into the bivector grades — a geometric nonlinearity that an MLP
// would need a hidden layer to approximate.
//
// Zero learned parameters in the encoder itself; downstream GroupRotor (28 SPSA
// params) handles per-group adaptation in the same Cl(1,7) space.

const CLIFFORD_ENCODER_HASH_DIM: usize = 256;

/// Grade features extracted from Cl(1,7): grade-1 (8) + grade-2 boost (7)
/// + grade-2 rotation (21) + scalar (1) = 37 geometric dimensions.
const CLIFFORD_GRADE_FEATURES: usize = 8 + 7 + 21 + 1;

/// Hash (256) + Clifford grade features (37) = 293.
/// Preserves per-dimension lexical anchors AND geometric cross-subspace structure.
const CLIFFORD_ENCODER_OUTPUT_DIM: usize = CLIFFORD_ENCODER_HASH_DIM + CLIFFORD_GRADE_FEATURES;

/// Blend of the semantic centroid (generalization) vs the surface chunk centroid
/// (exact-token precision) in the routing encoder. 0.0 = legacy behavior.
const SEM_BLEND: f32 = 0.5;
/// Half-width of the semantic-ID neighborhood smoothed per token. The dictionary
/// orders IDs by corpus co-occurrence, so adjacent IDs are semantically related;
/// smoothing over the neighborhood makes related content words share vector mass.
const SEM_NEIGHBOR_W: usize = 3;

/// Function/boilerplate words that should not dominate a request's meaning vector
/// (so "how do i implement binary search in python" centers on binary/search).
const ENCODER_STOPWORDS: &[&str] = &[
    "a",
    "an",
    "the",
    "of",
    "to",
    "in",
    "on",
    "at",
    "for",
    "and",
    "or",
    "is",
    "are",
    "be",
    "was",
    "were",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "i",
    "we",
    "you",
    "they",
    "he",
    "she",
    "how",
    "do",
    "does",
    "did",
    "with",
    "without",
    "as",
    "into",
    "from",
    "by",
    "not",
    "no",
    "so",
    "such",
    "please",
    "can",
    "could",
    "would",
    "should",
    "will",
    "may",
    "might",
    "me",
    "my",
    "your",
    "our",
    "their",
    "there",
    "here",
    "then",
    "than",
    "write",
    "implement",
    "create",
    "make",
    "build",
    "give",
    "want",
    "need",
    "code",
    "program",
    "python",
    "function",
    "func",
    "def",
    "class",
    "method",
    "using",
    "use",
    "some",
    "any",
    "get",
    "have",
];

fn content_weight(token: &str) -> f32 {
    if ENCODER_STOPWORDS.contains(&token.to_ascii_lowercase().as_str()) {
        0.25
    } else {
        1.0
    }
}

/// Content-weighted mean of semantic-neighbor-smoothed token vectors. Reuses the
/// dictionary's co-occurrence ID ordering (otherwise wasted, since per-token
/// vectors are random) so paraphrases with related content words land closer.
/// Decode path is untouched — this only shapes the routing/query embedding.
fn semantic_centroid(
    dict: &crate::spectral::TokenDictionary,
    codec: &crate::text_autoencoder::ChunkCodec,
    text: &str,
) -> Vec<f32> {
    use crate::text_autoencoder::CATA_DIM;
    let ids = dict.encode(text);
    let vocab = codec.vocab_size;
    let mut acc = vec![0.0f32; CATA_DIM];
    let mut wsum = 0.0f32;
    let w_half = SEM_NEIGHBOR_W as i32;
    for &id in &ids {
        if id == 0 {
            continue; // EOS
        }
        let w = content_weight(dict.token_str(id).unwrap_or(""));
        if w <= 0.0 {
            continue;
        }
        let mut sm = vec![0.0f32; CATA_DIM];
        let mut ksum = 0.0f32;
        let idi = id as i32;
        for off in -w_half..=w_half {
            let nid = idi + off;
            if nid < 1 || nid as usize >= vocab {
                continue;
            }
            let kw = (w_half + 1) as f32 - off.unsigned_abs() as f32; // triangular kernel
            let emb = codec.token_embedding(nid as u16);
            for j in 0..CATA_DIM {
                sm[j] += kw * emb[j];
            }
            ksum += kw;
        }
        if ksum <= 0.0 {
            continue;
        }
        for j in 0..CATA_DIM {
            acc[j] += w * sm[j] / ksum;
        }
        wsum += w;
    }
    if wsum > 0.0 {
        for x in acc.iter_mut() {
            *x /= wsum;
        }
    }
    acc
}

#[derive(Clone)]
struct CliffordLanguageEncoder {
    /// Fallback hash encoder (used when no dictionary is available, and for
    /// lexical anchor dimensions v[0..9] which are always injected).
    base: HashingLanguageEncoder,
    /// Vocabulary-grounded CDMA codec, built from training corpus.
    codec: Option<(
        crate::spectral::TokenDictionary,
        crate::text_autoencoder::ChunkCodec,
    )>,
}

impl std::fmt::Debug for CliffordLanguageEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CliffordLanguageEncoder")
            .field("has_codec", &self.codec.is_some())
            .field("vocab_size", &self.codec.as_ref().map(|(d, _)| d.len()))
            .finish()
    }
}

impl CliffordLanguageEncoder {
    fn new(dict: Option<crate::spectral::TokenDictionary>) -> Self {
        let base = HashingLanguageEncoder::new(EncoderPreset::Custom {
            model_name: "clifford-e8-cata".to_string(),
            output_dim: CLIFFORD_ENCODER_HASH_DIM,
        });
        let codec = dict.map(|d| {
            let codec = crate::text_autoencoder::ChunkCodec::new(d.len());
            (d, codec)
        });
        Self { base, codec }
    }
}

impl LanguageEncoder for CliffordLanguageEncoder {
    fn output_dim(&self) -> usize {
        CLIFFORD_ENCODER_OUTPUT_DIM
    }

    fn encode(&self, text: &str) -> Vec<f32> {
        let base_vec = if let Some((ref dict, ref codec)) = self.codec {
            // Surface centroid (CDMA chunks): preserves exact-token precision.
            let chunk = codec.encode_text(text, dict).centroid();
            // Semantic centroid: content-weighted, neighbor-smoothed token mean
            // that generalizes across phrasings sharing related content words.
            let sem = semantic_centroid(dict, codec, text);
            let mut centroid = vec![0.0f32; chunk.len()];
            for j in 0..centroid.len() {
                centroid[j] = SEM_BLEND * sem[j] + (1.0 - SEM_BLEND) * chunk[j];
            }
            // Inject lexical anchors from the hash encoder into v[0..10].
            // The hash encoder places sentiment/domain signals in these
            // dimensions; the CATA centroid has no reserved slots for them.
            let anchor_vec = self.base.encode(text);
            let n_anchors = 10.min(centroid.len()).min(anchor_vec.len());
            for i in 0..n_anchors {
                centroid[i] += anchor_vec[i];
            }
            centroid
        } else {
            self.base.encode(text)
        };

        let quantized = crate::spectral::E8Lattice::quantize_64d(&base_vec);

        let mv = crate::clifford::embed_bridge_vector(&quantized);

        let mut out = Vec::with_capacity(CLIFFORD_ENCODER_OUTPUT_DIM);
        out.extend_from_slice(&quantized);
        let grades = crate::clifford::extract_conditioning(&mv, CLIFFORD_GRADE_FEATURES);
        out.extend_from_slice(&grades);

        l2_normalize(&mut out);
        out
    }
}

#[cfg(feature = "native")]
#[derive(Debug, Clone)]
struct HttpGleEncoder {
    endpoint: String,
    model_name: String,
    output_dim: usize,
    fallback: HashingLanguageEncoder,
}

#[cfg(feature = "native")]
#[derive(Debug, Serialize)]
struct HttpEncodeRequest<'a> {
    text: &'a str,
    model: &'a str,
}

#[cfg(feature = "native")]
#[derive(Debug, Deserialize)]
struct HttpEncodeResponse {
    embedding: Vec<f32>,
}

#[cfg(feature = "native")]
impl HttpGleEncoder {
    fn new(
        endpoint: String,
        model_name: String,
        output_dim: usize,
        fallback: HashingLanguageEncoder,
    ) -> Self {
        Self {
            endpoint,
            model_name,
            output_dim,
            fallback,
        }
    }
}

#[cfg(feature = "native")]
impl LanguageEncoder for HttpGleEncoder {
    fn output_dim(&self) -> usize {
        self.output_dim
    }

    fn encode(&self, text: &str) -> Vec<f32> {
        let req = HttpEncodeRequest {
            text,
            model: &self.model_name,
        };
        let client = Client::new();
        let response = client.post(&self.endpoint).json(&req).send();
        if let Ok(resp) = response {
            if let Ok(payload) = resp.json::<HttpEncodeResponse>() {
                if payload.embedding.len() == self.output_dim {
                    return payload.embedding;
                }
            }
        }
        self.fallback.encode(text)
    }
}

fn layer_norm_affine(x: &mut [f32], gamma: &[f32], beta: &[f32]) {
    if x.is_empty() || x.len() != gamma.len() || x.len() != beta.len() {
        return;
    }
    let n = x.len() as f64;
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = x
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    let inv_std = 1.0 / (var + 1e-5_f64).sqrt();
    for i in 0..x.len() {
        x[i] = (((x[i] as f64 - mean) * inv_std) * gamma[i] as f64 + beta[i] as f64) as f32;
    }
}

fn sigmoid(x: f32) -> f32 {
    (1.0 / (1.0 + (-(x as f64)).exp())) as f32
}

fn l2_normalize(v: &mut [f32]) {
    let n = v
        .iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt();
    if n > 1e-20 {
        for x in v.iter_mut() {
            *x = (*x as f64 / n) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_projects_to_default_dim_with_confidence() {
        let encoder = HashingLanguageEncoder::new(EncoderPreset::MiniLmL6V2);
        let mut bridge = LanguageBridge::new(encoder.output_dim(), DEFAULT_BRIDGE_DIM);
        let dataset = CalibrationDataset {
            samples: (0..8)
                .flat_map(|d| {
                    (0..500).map(move |i| LanguageSample {
                        domain: format!("d{}", d),
                        text: format!("domain {} sample {}", d, i),
                        semantic_intent: "intent".to_string(),
                        action_target: None,
                        policy_regime: "default".to_string(),
                        language_channel: "english".to_string(),
                        expected_response: None,
                        expected_code: None,
                        causal: None,
                        history: Vec::new(),
                        conversation_turn: 1,
                    })
                })
                .collect(),
        };
        bridge
            .calibrate_global(
                &encoder,
                &dataset,
                &CalibrationRequirements::default(),
                true,
            )
            .expect("calibrate");
        let x = encoder.encode("route to support");
        let out = bridge.project(&x).expect("project");
        assert_eq!(out.routed_vector.len(), DEFAULT_BRIDGE_DIM);
        assert!(out.confidence >= 0.0 && out.confidence <= 1.0);
    }

    #[test]
    fn ema_smoother_updates_state() {
        let mut ema = EmaSmoother::new(0.2);
        let a = ema.update(&[1.0, 0.0]);
        let b = ema.update(&[0.0, 1.0]);
        assert_eq!(a, vec![1.0, 0.0]);
        assert!(b[0] < 1.0 && b[0] > 0.0);
        assert!(b[1] > 0.0 && b[1] < 1.0);
    }

    #[test]
    fn language_routing_rejects_ood() {
        let decision = route_language_embedding(&[], &vec![0.0; DEFAULT_BRIDGE_DIM], 0.5, 0.2);
        assert!(decision.rejected_as_ood);
        assert!(decision.chosen_group_id.is_none());
    }

    #[test]
    fn group_adapter_forward_produces_correct_dims() {
        let adapter = GroupAdapter::new(384, 128, 8);
        let h_raw: Vec<f32> = (0..384).map(|i| (i as f32 * 0.013).sin()).collect();
        let delta = adapter.forward(&h_raw);
        assert_eq!(delta.len(), 128);
        assert!(
            delta.iter().any(|&v| v != 0.0),
            "adapter output should be non-zero"
        );
    }

    #[test]
    fn group_adapter_adapt_adds_delta_to_shared() {
        let adapter = GroupAdapter::new(384, 128, 8);
        let z_shared = vec![1.0f32; 128];
        let h_raw: Vec<f32> = (0..384).map(|i| (i as f32 * 0.013).sin()).collect();
        let adapted = adapter.adapt(&z_shared, &h_raw);
        assert_eq!(adapted.len(), 128);
        assert!(adapted != z_shared, "adapted should differ from z_shared");
    }

    #[test]
    fn group_adapter_train_reduces_loss_proxy() {
        let mut adapter = GroupAdapter::new(64, 16, 4);
        let h_raw: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01).collect();
        let z_shared = vec![0.5f32; 16];

        let target_delta: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
        let target: Vec<f32> = z_shared
            .iter()
            .zip(target_delta.iter())
            .map(|(z, d)| z + d)
            .collect();

        let initial_adapted = adapter.adapt(&z_shared, &h_raw);
        let initial_err: f32 = initial_adapted
            .iter()
            .zip(target.iter())
            .map(|(a, t)| (a - t).powi(2))
            .sum();

        for _ in 0..200 {
            let adapted = adapter.adapt(&z_shared, &h_raw);
            let grad: Vec<f32> = adapted
                .iter()
                .zip(target.iter())
                .map(|(a, t)| 2.0 * (a - t))
                .collect();
            adapter.train_step(&h_raw, &grad, 0.01);
        }

        let final_adapted = adapter.adapt(&z_shared, &h_raw);
        let final_err: f32 = final_adapted
            .iter()
            .zip(target.iter())
            .map(|(a, t)| (a - t).powi(2))
            .sum();

        assert!(
            final_err < initial_err * 0.5,
            "adapter should reduce error: initial={:.4} final={:.4}",
            initial_err,
            final_err
        );
    }

    #[test]
    fn group_adapter_frozen_blocks_training() {
        let mut adapter = GroupAdapter::new(64, 16, 4);
        adapter.freeze();
        let h_raw = vec![0.1f32; 64];
        let before = adapter.forward(&h_raw);
        let grad = vec![1.0f32; 16];
        adapter.train_step(&h_raw, &grad, 0.1);
        let after = adapter.forward(&h_raw);
        assert_eq!(before, after, "frozen adapter should not change");
    }

    #[test]
    fn group_adapter_different_groups_diverge() {
        let a1 = GroupAdapter::new(384, 128, 8);
        let mut a2 = GroupAdapter::new(384, 128, 8);
        let h_raw: Vec<f32> = (0..384).map(|i| (i as f32 * 0.013).sin()).collect();
        let z_shared = vec![0.5f32; 128];

        let grad = vec![0.5f32; 128];
        for _ in 0..50 {
            a2.train_step(&h_raw, &grad, 0.01);
        }

        let out1 = a1.adapt(&z_shared, &h_raw);
        let out2 = a2.adapt(&z_shared, &h_raw);
        assert_ne!(
            out1, out2,
            "different adapters should produce different outputs after training"
        );
    }

    #[test]
    fn group_adapter_serialization_roundtrip() {
        let adapter = GroupAdapter::new(64, 16, 4);
        let json = serde_json::to_string(&adapter).expect("serialize");
        let loaded: GroupAdapter = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(adapter.raw_dim, loaded.raw_dim);
        assert_eq!(adapter.bridge_dim, loaded.bridge_dim);
        assert_eq!(adapter.rank, loaded.rank);
        assert_eq!(adapter.b_down, loaded.b_down);
        assert_eq!(adapter.a_up, loaded.a_up);

        let h = vec![0.1f32; 64];
        assert_eq!(adapter.forward(&h), loaded.forward(&h));
    }

    #[test]
    fn group_adapter_param_count() {
        let adapter = GroupAdapter::new(768, 128, 8);
        assert_eq!(adapter.param_count(), 768 * 8 + 128 * 8);
    }

    #[test]
    fn sentiment_joint_index_and_strip_roundtrip() {
        let s = LanguageSample {
            domain: "sentiment".into(),
            text: "BTC dominance bleeding".into(),
            semantic_intent: "mixed".into(),
            action_target: Some("sentiment".into()),
            policy_regime: "default".into(),
            language_channel: "english".into(),
            expected_response: Some("Forked narrative.".into()),
            expected_code: None,
            causal: None,
            history: Vec::new(),
            conversation_turn: 1,
        };
        let joint = sentiment_lattice_index_body(&s.text, s.expected_response.as_deref().unwrap());
        assert!(joint.contains(SENTIMENT_LATTICE_WITNESS_CORE));
        let stripped = strip_sentiment_lattice_witness_for_display(&joint);
        assert_eq!(stripped, "Forked narrative.");
    }

    #[test]
    fn sentiment_joint_index_embeds_causal_token_before_witness() {
        let causal = CausalAnnotation {
            causal_type: "direct".into(),
            connector: Some("so".into()),
            cause_span: None,
            effect_span: None,
            contrast_group: None,
            causal_subtype: None,
            surface_valence: None,
            resolved_valence: None,
        };
        let joint = sentiment_lattice_index_body_with_causal(
            "I lost big so I'm furious",
            "Anger from a clear loss.",
            Some(&causal),
        );
        assert!(joint.contains(SENTIMENT_CAUSAL_INDEX_CORE));
        assert!(joint.contains("gfcausal_t_direct_c_so"));
        assert!(
            joint.find(SENTIMENT_CAUSAL_INDEX_CORE).unwrap()
                < joint.find(SENTIMENT_LATTICE_WITNESS_CORE).unwrap()
        );
        let stripped = strip_sentiment_lattice_witness_for_display(&joint);
        assert_eq!(stripped, "Anger from a clear loss.");
    }

    #[test]
    fn sentiment_joint_index_includes_causal_subtype_token() {
        let causal = CausalAnnotation {
            causal_type: "counterfactual".into(),
            connector: Some("if".into()),
            cause_span: None,
            effect_span: None,
            contrast_group: None,
            causal_subtype: Some("interventional_counterfactual".into()),
            surface_valence: None,
            resolved_valence: None,
        };
        let joint = sentiment_lattice_index_body_with_causal(
            "If they'd reviewed my PR I wouldn't be this stressed",
            "Interventional stress relief hypothetical.",
            Some(&causal),
        );
        assert!(joint.contains("gfcausal_t_counterfactual_c_if"));
        assert!(joint.contains("gfcausal_st_interventional_counterfactual"));
    }

    #[test]
    fn causal_joint_index_emits_valence_tokens() {
        let causal = CausalAnnotation {
            causal_type: "retrospective_framing".into(),
            connector: Some("turned out".into()),
            cause_span: Some("losing the job".into()),
            effect_span: Some("best thing that happened".into()),
            contrast_group: None,
            causal_subtype: Some("retrospective_framing".into()),
            surface_valence: Some("negative_mild".into()),
            resolved_valence: Some("positive_strong".into()),
        };
        let tokens = causal.joint_index_tokens();
        assert!(tokens.contains("gfcausal_sv_negative_mild"));
        assert!(tokens.contains("gfcausal_rv_positive_strong"));
        assert!(tokens.contains("gfcausal_st_retrospective_framing"));
    }

    #[test]
    fn causal_valence_absent_emits_no_valence_tokens() {
        let causal = CausalAnnotation {
            causal_type: "direct".into(),
            connector: Some("so".into()),
            ..Default::default()
        };
        let tokens = causal.joint_index_tokens();
        assert!(!tokens.contains("gfcausal_sv_"));
        assert!(!tokens.contains("gfcausal_rv_"));
    }

    #[test]
    fn sentiment_strip_finds_core_when_glued_to_text() {
        // Decode may drop spaces around the marker; stripping must not require `SENTIMENT_LATTICE_WITNESS_SEP`.
        let glued = "Something feels off.__GROWFORMER_SENT_WITNESS__ ETH lagging rationale.";
        assert_eq!(
            strip_sentiment_lattice_witness_for_display(glued),
            "ETH lagging rationale."
        );
    }
}
