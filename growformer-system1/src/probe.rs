//! Probe heads: small calibrated classifiers on the open model's hidden state.
//!
//! The open model reads the state **once** (no question text, no decoding)
//! and returns a feature vector ([`crate::decide::OptionScorer::embed`]). Each
//! question then has a multinomial logistic-regression head over those
//! features, trained on the same labelled rows as the fast heads and
//! temperature-calibrated on held-out rows.
//!
//! Compared with the TF-IDF heads, probes see what the text *means* (sarcasm,
//! counterfactual regret, "worked for once"), so they usually need far fewer
//! labels for the same accuracy. Compared with asking the model each question,
//! they learn your label definitions from examples and answer every question
//! from one pass.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::brain::{score_values, HeadReport};
use crate::calibration::{fit_temperature, metrics, probs_with_temperature};
use crate::data::{split_holdout, LabeledRow};
use crate::head::softmax_in_place;
use crate::schema::QuestionSet;

/// Dense multinomial logistic regression (soft targets, L2, full-batch Adam).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DenseHead {
    pub n_outcomes: usize,
    pub dim: usize,
    /// Row-major `[n_outcomes][dim]`.
    weights: Vec<f32>,
    bias: Vec<f32>,
    pub n_train: usize,
    /// Inverse L2 strength the head was trained with.
    pub c: f32,
}

impl DenseHead {
    pub fn train(
        xs: &[&[f32]],
        targets: &[Vec<f32>],
        k: usize,
        c: f32,
        epochs: usize,
        lr: f32,
    ) -> Result<Self, String> {
        let n = xs.len();
        if n == 0 {
            return Err("no training rows".into());
        }
        let d = xs[0].len();
        if xs.iter().any(|x| x.len() != d) {
            return Err("probe features have different lengths".into());
        }
        let nf = n as f32;
        let lambda = 1.0 / (c * nf);
        let mut w = vec![0.0f32; k * d];
        let mut b = vec![0.0f32; k];
        let mut prior = vec![1e-3f32; k];
        for t in targets {
            for (j, &p) in t.iter().enumerate() {
                prior[j] += p;
            }
        }
        let tot: f32 = prior.iter().sum();
        for j in 0..k {
            b[j] = (prior[j] / tot).ln();
        }
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let (mut mw, mut vw) = (vec![0.0f32; k * d], vec![0.0f32; k * d]);
        let (mut mb, mut vb) = (vec![0.0f32; k], vec![0.0f32; k]);
        let mut gw = vec![0.0f32; k * d];
        let mut gb = vec![0.0f32; k];
        let mut p = vec![0.0f32; k];
        for epoch in 1..=epochs {
            gw.iter_mut().for_each(|g| *g = 0.0);
            gb.iter_mut().for_each(|g| *g = 0.0);
            for (x, t) in xs.iter().zip(targets) {
                for j in 0..k {
                    p[j] = b[j] + dot(&w[j * d..(j + 1) * d], x);
                }
                softmax_in_place(&mut p);
                for j in 0..k {
                    let g = (p[j] - t[j]) / nf;
                    if g == 0.0 {
                        continue;
                    }
                    gb[j] += g;
                    for (gw, &xv) in gw[j * d..(j + 1) * d].iter_mut().zip(x.iter()) {
                        *gw += g * xv;
                    }
                }
            }
            let tt = epoch as f32;
            let (bc1, bc2) = (1.0 - b1.powf(tt), 1.0 - b2.powf(tt));
            for i in 0..k * d {
                let g = gw[i] + lambda * w[i];
                mw[i] = b1 * mw[i] + (1.0 - b1) * g;
                vw[i] = b2 * vw[i] + (1.0 - b2) * g * g;
                w[i] -= lr * (mw[i] / bc1) / ((vw[i] / bc2).sqrt() + eps);
            }
            for j in 0..k {
                let g = gb[j];
                mb[j] = b1 * mb[j] + (1.0 - b1) * g;
                vb[j] = b2 * vb[j] + (1.0 - b2) * g * g;
                b[j] -= lr * (mb[j] / bc1) / ((vb[j] / bc2).sqrt() + eps);
            }
        }
        Ok(Self {
            n_outcomes: k,
            dim: d,
            weights: w,
            bias: b,
            n_train: n,
            c,
        })
    }

    pub fn logits(&self, x: &[f32]) -> Vec<f32> {
        (0..self.n_outcomes)
            .map(|j| self.bias[j] + dot(&self.weights[j * self.dim..(j + 1) * self.dim], x))
            .collect()
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Probe training knobs.
#[derive(Clone, Debug)]
pub struct ProbeTrainConfig {
    /// Share of rows held out to pick `C` and fit the temperature.
    pub holdout_frac: f64,
    pub seed: u64,
    /// Questions with fewer labelled rows get no probe.
    pub min_rows: usize,
    /// Inverse-L2 values tried; the best held-out log-loss wins.
    pub c_grid: Vec<f32>,
    pub epochs: usize,
    pub lr: f32,
}

impl Default for ProbeTrainConfig {
    fn default() -> Self {
        Self {
            holdout_frac: 0.2,
            seed: 7,
            min_rows: 8,
            c_grid: vec![0.003, 0.03, 0.3],
            epochs: 150,
            lr: 0.01,
        }
    }
}

/// Probe heads for one open model + layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Probe {
    /// Which model/template produced the features (`OptionScorer::model_key`).
    /// Probes only work with that model.
    pub model_key: String,
    /// Depth of the tapped layer (0 = first, 1 = last).
    pub layer_frac: f32,
    pub dim: usize,
    /// Per-feature standardisation fitted on the training rows.
    mean: Vec<f32>,
    inv_std: Vec<f32>,
    pub heads: BTreeMap<String, DenseHead>,
    pub temperature: BTreeMap<String, f32>,
    #[serde(default)]
    pub reports: BTreeMap<String, HeadReport>,
}

impl Probe {
    pub fn standardize(&self, x: &[f32]) -> Vec<f32> {
        x.iter()
            .zip(&self.mean)
            .zip(&self.inv_std)
            .map(|((v, m), s)| (v - m) * s)
            .collect()
    }

    /// Calibrated probabilities from *standardised* features.
    pub fn probs(&self, qid: &str, x_std: &[f32]) -> Option<Vec<f32>> {
        let h = self.heads.get(qid)?;
        let t = self.temperature.get(qid).copied().unwrap_or(1.0);
        Some(probs_with_temperature(&h.logits(x_std), t))
    }

    /// Train one probe head per question. `feats[i]` are the raw features of
    /// `rows[i]`. Per question: pick `C` and fit a temperature on the held-out
    /// split, report held-out metrics, then refit on every row.
    pub fn train(
        questions: &QuestionSet,
        rows: &[LabeledRow],
        feats: &[Vec<f32>],
        model_key: String,
        layer_frac: f32,
        cfg: &ProbeTrainConfig,
    ) -> Result<Self, String> {
        if rows.is_empty() || rows.len() != feats.len() {
            return Err("probe training needs one feature vector per row".into());
        }
        let dim = feats[0].len();
        let n = feats.len() as f64;
        let mut mean = vec![0.0f64; dim];
        for f in feats {
            for (m, v) in mean.iter_mut().zip(f) {
                *m += *v as f64 / n;
            }
        }
        let mut var = vec![0.0f64; dim];
        for f in feats {
            for ((s, v), m) in var.iter_mut().zip(f).zip(&mean) {
                *s += (*v as f64 - m).powi(2) / n;
            }
        }
        let mut probe = Probe {
            model_key,
            layer_frac,
            dim,
            mean: mean.iter().map(|&m| m as f32).collect(),
            // Constant features get weight 0 rather than blowing up.
            inv_std: var
                .iter()
                .map(|&v| {
                    if v > 1e-12 {
                        (1.0 / v.sqrt()) as f32
                    } else {
                        0.0
                    }
                })
                .collect(),
            heads: BTreeMap::new(),
            temperature: BTreeMap::new(),
            reports: BTreeMap::new(),
        };
        let xs: Vec<Vec<f32>> = feats.iter().map(|f| probe.standardize(f)).collect();
        let (_, test_idx) = split_holdout(rows, cfg.holdout_frac, cfg.seed);
        let in_test: std::collections::HashSet<usize> = test_idx.into_iter().collect();

        for (qi, q) in questions.questions.iter().enumerate() {
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
            type Row<'a> = (usize, &'a Vec<f32>, usize);
            let (tr, te): (Vec<&Row>, Vec<&Row>) =
                labelled.iter().partition(|r| !in_test.contains(&r.0));
            let fit = |set: &[&Row], c: f32| {
                DenseHead::train(
                    &set.iter()
                        .map(|(ri, _, _)| xs[*ri].as_slice())
                        .collect::<Vec<_>>(),
                    &set.iter().map(|(_, t, _)| (*t).clone()).collect::<Vec<_>>(),
                    k,
                    c,
                    cfg.epochs,
                    cfg.lr,
                )
            };
            let (mut c_best, mut temp, mut report) = (cfg.c_grid[0], 1.0f32, None);
            if te.len() >= 5 && tr.len() >= cfg.min_rows {
                let gold: Vec<usize> = te.iter().map(|(_, _, g)| *g).collect();
                let mut best: Option<(f64, f32, Vec<Vec<f32>>)> = None;
                for &c in &cfg.c_grid {
                    let h = fit(&tr, c)?;
                    let logits: Vec<Vec<f32>> =
                        te.iter().map(|(ri, _, _)| h.logits(&xs[*ri])).collect();
                    let t = fit_temperature(&logits, &gold);
                    let p: Vec<Vec<f32>> = logits
                        .iter()
                        .map(|z| probs_with_temperature(z, t))
                        .collect();
                    let nll = metrics(&p, &gold, None, None).nll;
                    if best.as_ref().map(|b| nll < b.0).unwrap_or(true) {
                        best = Some((nll, c, logits));
                    }
                }
                let (_, c, logits) = best.expect("non-empty C grid");
                c_best = c;
                temp = fit_temperature(&logits, &gold);
                let sv = score_values(&q.kind);
                let raw: Vec<Vec<f32>> = logits
                    .iter()
                    .map(|z| probs_with_temperature(z, 1.0))
                    .collect();
                let cal: Vec<Vec<f32>> = logits
                    .iter()
                    .map(|z| probs_with_temperature(z, temp))
                    .collect();
                report = Some(HeadReport {
                    n_train: tr.len(),
                    n_holdout: te.len(),
                    temperature: temp,
                    raw: metrics(&raw, &gold, sv.as_deref(), q.min_confidence),
                    calibrated: metrics(&cal, &gold, sv.as_deref(), q.min_confidence),
                });
            }
            let all: Vec<&Row> = labelled.iter().collect();
            probe.heads.insert(q.id.clone(), fit(&all, c_best)?);
            probe.temperature.insert(q.id.clone(), temp);
            if let Some(r) = report {
                probe.reports.insert(q.id.clone(), r);
            }
        }
        Ok(probe)
    }
}

/// Toy feature extractor for tests: hashed bag of words, so probe heads have
/// something learnable without a real model.
#[cfg(test)]
pub(crate) fn toy_embed(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    for w in text.to_lowercase().split_whitespace() {
        let h = crate::brain::row_fingerprint(w);
        v[(h % dim as u64) as usize] += 1.0;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_learns_and_calibrates() {
        let (qs, rows) = crate::brain::tests::toy_rows();
        let feats: Vec<Vec<f32>> = rows.iter().map(|r| toy_embed(&r.text, 64)).collect();
        let p = Probe::train(
            &qs,
            &rows,
            &feats,
            "toy".into(),
            0.66,
            &ProbeTrainConfig::default(),
        )
        .unwrap();
        assert!(!p.heads.is_empty());
        for (qid, rep) in &p.reports {
            assert!(rep.calibrated.accuracy > 0.8, "{qid}: {rep:?}");
        }
        let x = p.standardize(&feats[0]);
        let (qi, _, gold) = rows[0].targets[0];
        let probs = p.probs(&qs.questions[qi].id, &x).unwrap();
        assert_eq!(crate::calibration::argmax(&probs).0, gold);
        let json = serde_json::to_string(&p).unwrap();
        let back: Probe = serde_json::from_str(&json).unwrap();
        assert_eq!(back.heads.len(), p.heads.len());
    }
}
