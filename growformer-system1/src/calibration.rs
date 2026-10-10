//! Calibration: per-question temperature scaling, plus the metrics that say
//! whether confidences can be trusted (accuracy, NLL, Brier, ECE).

use serde::{Deserialize, Serialize};

use crate::head::softmax_in_place;

/// Softmax of `logits / temperature`.
pub fn probs_with_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
    let t = if temperature > 1e-4 { temperature } else { 1.0 };
    let mut p: Vec<f32> = logits.iter().map(|z| z / t).collect();
    softmax_in_place(&mut p);
    p
}

fn mean_nll(logits: &[Vec<f32>], gold: &[usize], t: f32) -> f64 {
    let mut s = 0.0f64;
    for (z, &g) in logits.iter().zip(gold) {
        let p = probs_with_temperature(z, t);
        s -= (p[g].max(1e-12) as f64).ln();
    }
    s / logits.len().max(1) as f64
}

/// Fit the temperature that minimises held-out NLL (golden-section search on
/// `ln T` in `[ln 0.05, ln 20]`). Returns 1.0 with fewer than 5 rows.
pub fn fit_temperature(logits: &[Vec<f32>], gold: &[usize]) -> f32 {
    if logits.len() < 5 {
        return 1.0;
    }
    let (mut a, mut b) = ((0.05f64).ln(), (20.0f64).ln());
    let phi = (5f64.sqrt() - 1.0) / 2.0;
    let f = |lt: f64| mean_nll(logits, gold, lt.exp() as f32);
    let mut c = b - phi * (b - a);
    let mut d = a + phi * (b - a);
    let (mut fc, mut fd) = (f(c), f(d));
    for _ in 0..60 {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - phi * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + phi * (b - a);
            fd = f(d);
        }
    }
    (((a + b) / 2.0).exp()) as f32
}

/// Quality report for one question / backend.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Metrics {
    pub n: usize,
    pub accuracy: f64,
    /// Mean negative log-likelihood of the gold outcome.
    pub nll: f64,
    /// Multiclass Brier score (mean squared error of the probability vector; 0 = perfect).
    pub brier: f64,
    /// Expected calibration error over 10 confidence bins (0 = perfectly calibrated).
    pub ece: f64,
    /// Score questions: mean absolute error of the expected value vs the gold point.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mae: Option<f64>,
    /// Share of rows at/above the question's `min_confidence` (1.0 if unset).
    pub coverage: f64,
    /// Accuracy on the covered (non-abstained) rows.
    pub selective_accuracy: f64,
}

/// Compute metrics from calibrated probability vectors.
/// `score_values` = the numeric value of each outcome for Score questions.
pub fn metrics(
    probs: &[Vec<f32>],
    gold: &[usize],
    score_values: Option<&[f32]>,
    min_confidence: Option<f32>,
) -> Metrics {
    let n = probs.len();
    if n == 0 {
        return Metrics::default();
    }
    let (mut correct, mut nll, mut brier, mut mae) = (0usize, 0.0f64, 0.0f64, 0.0f64);
    let (mut cov_n, mut cov_ok) = (0usize, 0usize);
    let mut bins = [(0usize, 0.0f64, 0.0f64); 10]; // (count, sum conf, sum correct)
    for (p, &g) in probs.iter().zip(gold) {
        let (arg, conf) = argmax(p);
        let ok = arg == g;
        correct += ok as usize;
        nll -= (p[g].max(1e-12) as f64).ln();
        for (i, &pi) in p.iter().enumerate() {
            let y = if i == g { 1.0 } else { 0.0 };
            brier += ((pi as f64) - y).powi(2);
        }
        let bi = ((conf * 10.0) as usize).min(9);
        bins[bi].0 += 1;
        bins[bi].1 += conf as f64;
        bins[bi].2 += ok as u8 as f64;
        if let Some(vals) = score_values {
            let ev: f64 = p.iter().zip(vals).map(|(&a, &v)| a as f64 * v as f64).sum();
            mae += (ev - vals[g] as f64).abs();
        }
        if conf >= min_confidence.unwrap_or(0.0) {
            cov_n += 1;
            cov_ok += ok as usize;
        }
    }
    let ece = bins
        .iter()
        .filter(|b| b.0 > 0)
        .map(|b| (b.0 as f64 / n as f64) * ((b.1 - b.2) / b.0 as f64).abs())
        .sum();
    Metrics {
        n,
        accuracy: correct as f64 / n as f64,
        nll: nll / n as f64,
        brier: brier / n as f64,
        ece,
        mae: score_values.map(|_| mae / n as f64),
        coverage: cov_n as f64 / n as f64,
        selective_accuracy: if cov_n > 0 {
            cov_ok as f64 / cov_n as f64
        } else {
            0.0
        },
    }
}

/// `(index, value)` of the largest entry.
pub fn argmax(p: &[f32]) -> (usize, f32) {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, &v) in p.iter().enumerate() {
        if v > best.1 {
            best = (i, v);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_softens_overconfident_logits() {
        // Logits are 10× too sharp: correct only 60% of the time but ~100% confident.
        let mut logits = Vec::new();
        let mut gold = Vec::new();
        for i in 0..100 {
            logits.push(vec![10.0, 0.0]);
            gold.push(if i % 5 < 3 { 0 } else { 1 });
        }
        let t = fit_temperature(&logits, &gold);
        assert!(t > 5.0, "expected strong softening, got T={t}");
        let p = probs_with_temperature(&logits[0], t);
        assert!((p[0] - 0.6).abs() < 0.05, "{p:?}");
        let before = metrics(
            &logits
                .iter()
                .map(|z| probs_with_temperature(z, 1.0))
                .collect::<Vec<_>>(),
            &gold,
            None,
            None,
        );
        let after = metrics(
            &logits
                .iter()
                .map(|z| probs_with_temperature(z, t))
                .collect::<Vec<_>>(),
            &gold,
            None,
            None,
        );
        assert!(after.ece < before.ece && after.nll < before.nll);
    }

    #[test]
    fn score_mae_and_coverage() {
        let probs = vec![vec![0.1, 0.8, 0.1], vec![0.4, 0.3, 0.3]];
        let m = metrics(&probs, &[1, 2], Some(&[0.0, 1.0, 2.0]), Some(0.5));
        assert!((m.coverage - 0.5).abs() < 1e-9);
        assert!((m.selective_accuracy - 1.0).abs() < 1e-9);
        assert!(m.mae.unwrap() > 0.0);
    }
}
