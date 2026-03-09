//! Language front-end layer for Growformer (M1/M2 foundation).
//!
//! This module provides:
//! - deterministic text encoder presets (default MiniLM-sized 384-d vectors),
//! - a globally calibrated 384->64 bridge with layer norm + confidence head,
//! - EMA smoothing for multi-turn routing stability,
//! - objective routing outputs (winner, margin, OOD reject).

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use super::embedding::{cosine_similarity, GroupEmbedding};
use crate::types::GroupId;

pub const DEFAULT_ENCODER_DIM: usize = 384;
pub const DEFAULT_BRIDGE_DIM: usize = 64;
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
        EncoderPreset::MiniLmL6V2
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageConfig {
    pub encoder: EncoderPreset,
    pub bridge_output_dim: usize,
    pub ema_alpha: f32,
    pub ood_similarity_threshold: f32,
}

impl Default for LanguageConfig {
    fn default() -> Self {
        Self {
            encoder: EncoderPreset::default(),
            bridge_output_dim: DEFAULT_BRIDGE_DIM,
            ema_alpha: DEFAULT_EMA_ALPHA,
            ood_similarity_threshold: 0.15,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageSample {
    pub domain: String,
    pub text: String,
    pub semantic_intent: String,
    pub action_target: Option<String>,
    pub policy_regime: String,
    pub language_channel: String,
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
        for token in text.split_whitespace() {
            let bytes = token.as_bytes();
            if bytes.is_empty() {
                continue;
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
        let mut projection = vec![vec![0.0f32; input_dim]; output_dim];
        for (o, row) in projection.iter_mut().enumerate() {
            for (i, w) in row.iter_mut().enumerate() {
                let x = ((o as u64).wrapping_mul(1_000_003) ^ (i as u64).wrapping_mul(9176)) as f32;
                *w = ((x.sin() * 1000.0).fract() - 0.5) * 0.1;
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

    pub fn calibrate_global<E: LanguageEncoder>(
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
            encoder_model: EncoderPreset::default().model_name(),
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
}

impl LanguageRuntime {
    pub fn new(config: LanguageConfig) -> Self {
        let encoder = HashingLanguageEncoder::new(config.encoder.clone());
        let bridge = LanguageBridge::new(encoder.output_dim(), config.bridge_output_dim);
        let smoother = EmaSmoother::new(config.ema_alpha);
        Self {
            config,
            encoder,
            bridge,
            smoother,
        }
    }

    pub fn calibrate(
        &mut self,
        dataset: &CalibrationDataset,
        requirements: &CalibrationRequirements,
    ) -> Result<CalibrationReport, String> {
        self.bridge
            .calibrate_global(&self.encoder, dataset, requirements, true)
    }

    pub fn bridge_text(&mut self, text: &str) -> Result<BridgeOutput, String> {
        let encoded = self.encoder.encode(text);
        let out = self.bridge.project(&encoded)?;
        let smoothed = self.smoother.update(&out.routed_vector);
        Ok(BridgeOutput {
            routed_vector: smoothed,
            confidence: out.confidence,
        })
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
    fn bridge_projects_to_64_with_confidence() {
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
        assert_eq!(out.routed_vector.len(), 64);
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
        let decision = route_language_embedding(&[], &[0.0; 64], 0.5, 0.2);
        assert!(decision.rejected_as_ood);
        assert!(decision.chosen_group_id.is_none());
    }
}
