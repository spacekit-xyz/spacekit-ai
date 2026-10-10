//! The System 1 brain artifact: question schema + shared featurizer + fast
//! heads + calibration + router thresholds + (optional) open-model reference.
//! Serialised as JSON (optionally gzip, `*.s1.json.gz`), loadable from bytes
//! on every target including wasm.

use std::collections::BTreeMap;
use std::io::{Read, Write};

use serde::{Deserialize, Serialize};

use crate::calibration::{fit_temperature, metrics, probs_with_temperature, Metrics};
use crate::data::{split_holdout, LabeledRow};
use crate::features::{Featurizer, FeaturizerConfig, SparseVec};
use crate::head::{HeadTrainConfig, LinearHead};
use crate::prompt::TemplatePreset;
use crate::schema::{QuestionKind, QuestionSet};

pub const BRAIN_FORMAT: u32 = 1;

/// Supported open-model architectures (GGUF, run with candle).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LmArch {
    /// Llama 2/3.x, SmolLM2, TinyLlama, Mistral-style GGUFs.
    Llama,
    /// Qwen2 / Qwen2.5.
    Qwen2,
    /// Qwen3 (dense).
    Qwen3,
}

/// Open-model backend settings. Paths are relative to the brain file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LmConfig {
    pub arch: LmArch,
    /// GGUF weights.
    pub model: String,
    /// HF `tokenizer.json`.
    pub tokenizer: String,
    #[serde(default = "default_template")]
    pub template: TemplatePreset,
    /// System prompt (defaults to [`crate::prompt::DEFAULT_SYSTEM`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// State budget in tokens (head and tail kept).
    #[serde(default = "default_max_state_tokens")]
    pub max_state_tokens: usize,
}

fn default_template() -> TemplatePreset {
    TemplatePreset::Chatml
}

fn default_max_state_tokens() -> usize {
    2048
}

/// When to trust a head vs. ask the open model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RouterConfig {
    /// Use the head when its calibrated confidence is at least this (and an
    /// open model is attached; without one the head always answers).
    #[serde(default = "default_head_min_conf")]
    pub head_min_confidence: f32,
    /// Per-question overrides (written by `gf-system1 eval --tune-router`).
    #[serde(default)]
    pub per_question: BTreeMap<String, f32>,
}

fn default_head_min_conf() -> f32 {
    0.6
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            head_min_confidence: default_head_min_conf(),
            per_question: BTreeMap::new(),
        }
    }
}

impl RouterConfig {
    pub fn threshold(&self, qid: &str) -> f32 {
        self.per_question
            .get(qid)
            .copied()
            .unwrap_or(self.head_min_confidence)
    }
}

/// Held-out report for one question's head.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeadReport {
    pub n_train: usize,
    pub n_holdout: usize,
    pub temperature: f32,
    /// Held-out metrics before temperature scaling.
    pub raw: Metrics,
    /// Held-out metrics after temperature scaling.
    pub calibrated: Metrics,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct System1Brain {
    pub format: u32,
    pub name: String,
    pub questions: QuestionSet,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub featurizer: Option<Featurizer>,
    #[serde(default)]
    pub heads: BTreeMap<String, LinearHead>,
    #[serde(default)]
    pub head_temperature: BTreeMap<String, f32>,
    #[serde(default)]
    pub lm_temperature: BTreeMap<String, f32>,
    #[serde(default)]
    pub router: RouterConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lm: Option<LmConfig>,
    /// State budget for the heads, in characters.
    #[serde(default = "default_max_state_chars")]
    pub max_state_chars: usize,
    #[serde(default)]
    pub reports: BTreeMap<String, HeadReport>,
}

fn default_max_state_chars() -> usize {
    20_000
}

/// Head training knobs.
#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub holdout_frac: f64,
    pub seed: u64,
    /// Questions with fewer labelled rows get no head (the open model answers them).
    pub min_rows: usize,
    pub head: HeadTrainConfig,
    pub featurizer: FeaturizerConfig,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            holdout_frac: 0.2,
            seed: 7,
            min_rows: 8,
            head: HeadTrainConfig::default(),
            featurizer: FeaturizerConfig::default(),
        }
    }
}

/// Numeric value of each outcome (Score questions), for MAE / expected value.
pub fn score_values(kind: &QuestionKind) -> Option<Vec<f32>> {
    match kind {
        QuestionKind::Score { min, max, .. } => Some((*min..=*max).map(|v| v as f32).collect()),
        _ => None,
    }
}

impl System1Brain {
    pub fn new(name: impl Into<String>, questions: QuestionSet) -> Result<Self, String> {
        questions.validate()?;
        Ok(Self {
            format: BRAIN_FORMAT,
            name: name.into(),
            questions,
            featurizer: None,
            heads: BTreeMap::new(),
            head_temperature: BTreeMap::new(),
            lm_temperature: BTreeMap::new(),
            router: RouterConfig::default(),
            lm: None,
            max_state_chars: default_max_state_chars(),
            reports: BTreeMap::new(),
        })
    }

    /// Train one head per question that has at least `min_rows` labels:
    /// fit on the train split, fit a temperature on the held-out split, report
    /// held-out metrics, then refit on all rows (keeping that temperature).
    pub fn train_heads(&mut self, rows: &[LabeledRow], cfg: &TrainConfig) -> Result<(), String> {
        if rows.is_empty() {
            return Err("no labelled rows".into());
        }
        let texts: Vec<String> = rows
            .iter()
            .map(|r| crate::state::truncate_middle(&r.text, self.max_state_chars))
            .collect();
        let feat = Featurizer::fit(&texts, &cfg.featurizer);
        let xs: Vec<SparseVec> = texts.iter().map(|t| feat.transform(t)).collect();
        let (_train_idx, test_idx) = split_holdout(rows, cfg.holdout_frac, cfg.seed);
        let in_test: std::collections::HashSet<usize> = test_idx.iter().copied().collect();

        self.heads.clear();
        self.head_temperature.clear();
        self.reports.clear();
        for (qi, q) in self.questions.questions.iter().enumerate() {
            let labelled: Vec<(usize, &Vec<f32>, usize)> = rows
                .iter()
                .enumerate()
                .filter_map(|(ri, r)| {
                    r.targets
                        .iter()
                        .find(|(i, _, _)| *i == qi)
                        .map(|(_, t, g)| (ri, t, *g))
                })
                .collect();
            if labelled.len() < cfg.min_rows {
                continue;
            }
            let k = q.n_outcomes();
            let tr: Vec<_> = labelled
                .iter()
                .filter(|(ri, _, _)| !in_test.contains(ri))
                .collect();
            let te: Vec<_> = labelled
                .iter()
                .filter(|(ri, _, _)| in_test.contains(ri))
                .collect();

            let (temperature, report) = if te.len() >= 5 && tr.len() >= cfg.min_rows {
                let h = LinearHead::train(
                    &tr.iter().map(|(ri, _, _)| &xs[*ri]).collect::<Vec<_>>(),
                    &tr.iter().map(|(_, t, _)| (*t).clone()).collect::<Vec<_>>(),
                    k,
                    feat.dim(),
                    &cfg.head,
                )?;
                let logits: Vec<Vec<f32>> =
                    te.iter().map(|(ri, _, _)| h.logits(&xs[*ri])).collect();
                let gold: Vec<usize> = te.iter().map(|(_, _, g)| *g).collect();
                let t = fit_temperature(&logits, &gold);
                let sv = score_values(&q.kind);
                let raw: Vec<Vec<f32>> = logits
                    .iter()
                    .map(|z| probs_with_temperature(z, 1.0))
                    .collect();
                let cal: Vec<Vec<f32>> = logits
                    .iter()
                    .map(|z| probs_with_temperature(z, t))
                    .collect();
                let rep = HeadReport {
                    n_train: tr.len(),
                    n_holdout: te.len(),
                    temperature: t,
                    raw: metrics(&raw, &gold, sv.as_deref(), q.min_confidence),
                    calibrated: metrics(&cal, &gold, sv.as_deref(), q.min_confidence),
                };
                (t, Some(rep))
            } else {
                (1.0, None)
            };

            let head = LinearHead::train(
                &labelled
                    .iter()
                    .map(|(ri, _, _)| &xs[*ri])
                    .collect::<Vec<_>>(),
                &labelled
                    .iter()
                    .map(|(_, t, _)| (*t).clone())
                    .collect::<Vec<_>>(),
                k,
                feat.dim(),
                &cfg.head,
            )?;
            self.heads.insert(q.id.clone(), head);
            self.head_temperature.insert(q.id.clone(), temperature);
            if let Some(r) = report {
                self.reports.insert(q.id.clone(), r);
            }
        }
        self.featurizer = Some(feat);
        Ok(())
    }

    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|e| e.to_string())
    }

    /// Parse from JSON or gzip-compressed JSON bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let owned;
        let json: &[u8] = if bytes.starts_with(&[0x1f, 0x8b]) {
            let mut d = flate2::read::GzDecoder::new(bytes);
            let mut v = Vec::new();
            d.read_to_end(&mut v).map_err(|e| format!("gunzip: {e}"))?;
            owned = v;
            &owned
        } else {
            bytes
        };
        let mut b: Self = serde_json::from_slice(json).map_err(|e| format!("parse brain: {e}"))?;
        if b.format != BRAIN_FORMAT {
            return Err(format!("unsupported System 1 brain format {}", b.format));
        }
        b.questions.validate()?;
        if let Some(f) = b.featurizer.as_mut() {
            f.rebuild_index();
        }
        Ok(b)
    }

    /// Serialise; gzip when `gzip` is true.
    pub fn to_bytes(&self, gzip: bool) -> Result<Vec<u8>, String> {
        let json = self.to_json()?;
        if !gzip {
            return Ok(json.into_bytes());
        }
        let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        e.write_all(json.as_bytes()).map_err(|e| e.to_string())?;
        e.finish().map_err(|e| e.to_string())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        Self::from_bytes(&bytes)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        let gz = path.extension().map(|e| e == "gz").unwrap_or(false);
        if let Some(p) = path.parent() {
            if !p.as_os_str().is_empty() {
                std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
            }
        }
        std::fs::write(path, self.to_bytes(gz)?)
            .map_err(|e| format!("write {}: {e}", path.display()))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::data::parse_labeled_jsonl;

    pub(crate) fn toy_rows() -> (QuestionSet, Vec<LabeledRow>) {
        let qs = QuestionSet::parse(
            r#"
[[questions]]
id = "route"
text = "Which team should handle this?"
type = "choice"
options = ["Billing", "Tech"]

[[questions]]
id = "urgent"
text = "Is this urgent?"
type = "noul"
"#,
        )
        .unwrap();
        let mut lines = Vec::new();
        let billing = [
            "refund",
            "charged twice",
            "invoice wrong",
            "payment failed",
            "card declined",
            "billing question",
        ];
        let tech = [
            "app crashes",
            "cannot log in",
            "server error",
            "page will not load",
            "bug in editor",
            "sync broken",
        ];
        for i in 0..60 {
            let (txt, team) = if i % 2 == 0 {
                (billing[i / 2 % billing.len()], "Billing")
            } else {
                (tech[i / 2 % tech.len()], "Tech")
            };
            let urgent = i % 3 == 0;
            let suffix = if urgent {
                " ASAP this is urgent"
            } else {
                " when you have time"
            };
            lines.push(format!(
                r#"{{"state":"{txt}{suffix} #{i}","labels":{{"route":"{team}","urgent":{urgent}}}}}"#
            ));
        }
        let (rows, _) = parse_labeled_jsonl(&lines.join("\n"), &qs).unwrap();
        (qs, rows)
    }

    #[test]
    fn trains_reports_and_roundtrips() {
        let (qs, rows) = toy_rows();
        let mut b = System1Brain::new("toy", qs).unwrap();
        b.train_heads(&rows, &TrainConfig::default()).unwrap();
        assert_eq!(b.heads.len(), 2);
        let r = &b.reports["route"];
        assert!(r.calibrated.accuracy > 0.8, "{r:?}");
        let gz = b.to_bytes(true).unwrap();
        let back = System1Brain::from_bytes(&gz).unwrap();
        let f = back.featurizer.as_ref().unwrap();
        let x = f.transform("card declined again");
        assert_eq!(
            back.heads["route"].logits(&x),
            b.heads["route"].logits(
                &b.featurizer
                    .as_ref()
                    .unwrap()
                    .transform("card declined again")
            )
        );
    }
}
