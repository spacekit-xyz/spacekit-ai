//! Language front-end layer for Growformer (M1/M2 foundation).
//!
//! This module provides:
//! - deterministic text encoder presets (default MiniLM-sized 384-d vectors),
//! - a globally calibrated bridge (384->128d default) with layer norm + confidence head,
//! - EMA smoothing for multi-turn routing stability,
//! - objective routing outputs (winner, margin, OOD reject).

use std::collections::{HashMap, HashSet};

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
    Custom { model_name: String, output_dim: usize },
}

impl EncoderPreset {
    pub fn model_name(&self) -> String {
        match self {
            EncoderPreset::MiniLmL6V2 => "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            EncoderPreset::MiniLmMultilingualL12V2 => {
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2".to_string()
            }
            EncoderPreset::BertClass => "bert-base-uncased".to_string(),
            EncoderPreset::Custom { model_name, .. } => model_name.clone(),
        }
    }

    pub fn output_dim(&self) -> usize {
        match self {
            EncoderPreset::MiniLmL6V2 => 384,
            EncoderPreset::MiniLmMultilingualL12V2 => 384,
            EncoderPreset::BertClass => 768,
            EncoderPreset::Custom { output_dim, .. } => *output_dim,
        }
    }
}

impl Default for EncoderPreset {
    fn default() -> Self {
        EncoderPreset::BertClass
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

    /// Space-separated causal index tokens for the joint lattice body (type+connector, then optional subtype).
    pub fn joint_index_tokens(&self) -> String {
        let base = self.index_token();
        let st = self
            .causal_subtype
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(causal_subtype_index_token)
            .filter(|s| !s.is_empty());
        match st {
            Some(s) => format!("{} {}", base, s),
            None => base,
        }
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
        .map(|c| format!(" {} {}", SENTIMENT_CAUSAL_INDEX_CORE, c.joint_index_tokens()))
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
                return Err("calibration samples must include intent/policy/language labels".to_string());
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
            "customer", "support", "account", "password", "billing", "ticket", "refund", "login",
            "unlock", "subscription", "recovery", "helpdesk",
        ];
        let coding_tool_keywords = [
            "coding", "code", "rust", "function", "debug", "compiler", "sql", "query", "parser",
            "tool", "stack", "pointer", "serde", "index", "module",
        ];
        let knowledge_qa_keywords = [
            "knowledge", "qa", "fact", "explain", "definition", "what", "why", "how", "answer",
            "reference", "documentation",
        ];
        let safety_refusal_keywords = [
            "safety", "policy", "refuse", "blocked", "forbidden", "harmful", "disallowed",
            "compliance", "unsafe", "restricted",
        ];
        let procedural_instruction_keywords = [
            "procedure", "instruction", "step", "follow", "sequence", "workflow", "checklist",
            "process", "guide",
        ];
        let short_conversation_keywords = [
            "hello", "hi", "thanks", "ok", "yes", "no", "greetings", "bye", "chat",
        ];
        let multi_turn_followup_keywords = [
            "followup", "continue", "previous", "earlier", "context", "as-said", "next", "again",
            "clarify", "thread",
        ];
        let adversarial_noisy_keywords = [
            "adversarial", "noisy", "jailbreak", "prompt-injection", "ignore", "override",
            "garbled", "nonsense", "obfuscated",
        ];
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
            if dim >= 8 {
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
    pub b_down: Vec<Vec<f32>>,  // rank × raw_dim
    pub a_up: Vec<Vec<f32>>,    // bridge_dim × rank
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
                *w = hash * scale * 0.1;  // near-zero init so adapter starts as identity-ish
            }
        }
        Self { raw_dim, bridge_dim, rank, b_down, a_up, l2_weight: DEFAULT_ADAPTER_L2, frozen: false }
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
            hidden[r] = if acc > 0.0 { acc } else { 0.01 * acc };  // LeakyReLU
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
        z_shared.iter().zip(delta.iter()).map(|(&z, &d)| z + d).collect()
    }

    /// SGD training step: given the gradient signal from gen head loss.
    /// `cond_grad` is the gradient of loss w.r.t. the adapted vector (bridge_dim).
    /// `h_raw` is the raw encoder vector for this sample.
    #[cfg(feature = "training")]
    pub fn train_step(&mut self, h_raw: &[f32], cond_grad: &[f32], lr: f32) {
        if self.frozen { return; }
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
            d_hidden[r] = if hidden_pre_relu[r] > 0.0 { acc } else { 0.01 * acc };
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
            normalized[i] = (encoder_vec[i] - self.input_mean[i]) / self.input_std[i].max(1e-6);
        }
        let mut z = vec![0.0f32; self.output_dim];
        for (o, zo) in z.iter_mut().enumerate() {
            let mut acc = self.bias[o];
            for i in 0..self.input_dim {
                acc += self.projection[o][i] * normalized[i];
            }
            *zo = acc;
        }
        layer_norm_affine(&mut z, &self.ln_gamma, &self.ln_beta);
        let mut confidence_logit = self.confidence_bias;
        for (w, v) in self.confidence_head.iter().zip(z.iter()) {
            confidence_logit += w * v.abs();
        }
        let confidence = sigmoid(confidence_logit);
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
        .map(|e| (e.group_id, cosine_similarity(language_vec, &e.language_vector)))
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
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageRuntime {
    pub config: LanguageConfig,
    pub encoder: HashingLanguageEncoder,
    pub bridge: LanguageBridge,
    pub smoother: EmaSmoother,
    #[serde(skip)]
    preloaded_students: Vec<GleStudentCheckpoint>,
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
        }
    }

    pub fn calibrate(
        &mut self,
        dataset: &CalibrationDataset,
        requirements: &CalibrationRequirements,
    ) -> Result<CalibrationReport, String> {
        let encoder = self.build_encoder();
        let mut report = self
            .bridge
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
            let mut acc = self.b1[j];
            for (i, &xi) in x.iter().enumerate() {
                acc += self.w1[j][i] * xi;
            }
            *hj = acc.tanh();
        }
        let mut y = vec![0.0f32; self.w2.len()];
        for (o, yo) in y.iter_mut().enumerate() {
            let mut acc = self.b2[o];
            for (j, &hj) in h.iter().enumerate() {
                acc += self.w2[o][j] * hj;
            }
            *yo = acc;
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
    let mean = x.iter().sum::<f32>() / x.len() as f32;
    let var = x
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f32>()
        / x.len() as f32;
    let inv_std = 1.0 / (var + 1e-5).sqrt();
    for i in 0..x.len() {
        x[i] = ((x[i] - mean) * inv_std) * gamma[i] + beta[i];
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn l2_normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-10 {
        for x in v {
            *x /= n;
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
        assert!(delta.iter().any(|&v| v != 0.0), "adapter output should be non-zero");
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
        let target: Vec<f32> = z_shared.iter().zip(target_delta.iter()).map(|(z, d)| z + d).collect();

        let initial_adapted = adapter.adapt(&z_shared, &h_raw);
        let initial_err: f32 = initial_adapted.iter().zip(target.iter())
            .map(|(a, t)| (a - t).powi(2)).sum();

        for _ in 0..200 {
            let adapted = adapter.adapt(&z_shared, &h_raw);
            let grad: Vec<f32> = adapted.iter().zip(target.iter())
                .map(|(a, t)| 2.0 * (a - t)).collect();
            adapter.train_step(&h_raw, &grad, 0.01);
        }

        let final_adapted = adapter.adapt(&z_shared, &h_raw);
        let final_err: f32 = final_adapted.iter().zip(target.iter())
            .map(|(a, t)| (a - t).powi(2)).sum();

        assert!(final_err < initial_err * 0.5,
            "adapter should reduce error: initial={:.4} final={:.4}", initial_err, final_err);
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
        assert_ne!(out1, out2, "different adapters should produce different outputs after training");
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
    fn sentiment_strip_finds_core_when_glued_to_text() {
        // Decode may drop spaces around the marker; stripping must not require `SENTIMENT_LATTICE_WITNESS_SEP`.
        let glued = "Something feels off.__GROWFORMER_SENT_WITNESS__ ETH lagging rationale.";
        assert_eq!(
            strip_sentiment_lattice_witness_for_display(glued),
            "ETH lagging rationale."
        );
    }
}
