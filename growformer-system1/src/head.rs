//! Per-question linear head: multinomial logistic regression over the shared
//! features, trained with full-batch Adam + L2 on **soft targets** (one-hot
//! for human labels, a probability distribution for distilled labels).

use serde::{Deserialize, Serialize};

use crate::features::SparseVec;

#[derive(Clone, Debug)]
pub struct HeadTrainConfig {
    /// Inverse L2 strength (sklearn-style C).
    pub c: f32,
    pub epochs: usize,
    pub lr: f32,
}

impl Default for HeadTrainConfig {
    fn default() -> Self {
        Self {
            c: 10.0,
            epochs: 300,
            lr: 0.05,
        }
    }
}

/// Linear head over the shared feature space. Weights are stored sparsely
/// (`|w| < 1e-4` dropped after training) to keep brain files small.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinearHead {
    pub n_outcomes: usize,
    /// Per outcome: sparse `(feature, weight)` list, sorted by feature.
    weights: Vec<Vec<(u32, f32)>>,
    bias: Vec<f32>,
    pub n_train: usize,
}

pub(crate) fn softmax_in_place(z: &mut [f32]) {
    let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0f32;
    for v in z.iter_mut() {
        *v = (*v - m).exp();
        s += *v;
    }
    if s > 0.0 {
        for v in z.iter_mut() {
            *v /= s;
        }
    }
}

impl LinearHead {
    /// Train on `(features, target distribution)` pairs.
    pub fn train(
        xs: &[&SparseVec],
        targets: &[Vec<f32>],
        n_outcomes: usize,
        n_features: usize,
        cfg: &HeadTrainConfig,
    ) -> Result<Self, String> {
        if xs.is_empty() {
            return Err("no training rows".into());
        }
        if xs.len() != targets.len() {
            return Err("features/targets length mismatch".into());
        }
        let k = n_outcomes;
        let n = xs.len() as f32;
        let lambda = 1.0 / (cfg.c * n);
        let mut w = vec![vec![0.0f32; n_features]; k];
        let mut b = vec![0.0f32; k];
        // Initialise bias to log class priors (faster, better calibrated start).
        let mut prior = vec![1e-3f32; k];
        for t in targets {
            for (c, &p) in t.iter().enumerate() {
                prior[c] += p;
            }
        }
        let tot: f32 = prior.iter().sum();
        for c in 0..k {
            b[c] = (prior[c] / tot).ln();
        }

        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let mut mw = vec![vec![0.0f32; n_features]; k];
        let mut vw = vec![vec![0.0f32; n_features]; k];
        let (mut mb, mut vb) = (vec![0.0f32; k], vec![0.0f32; k]);
        let mut gw = vec![vec![0.0f32; n_features]; k];
        let mut gb = vec![0.0f32; k];
        // Features actually touched by data: only those (and their decay) need updates.
        let mut active = vec![false; n_features];
        for x in xs {
            for &(i, _) in x.iter() {
                active[i as usize] = true;
            }
        }
        let active_idx: Vec<usize> = (0..n_features).filter(|&j| active[j]).collect();

        for epoch in 1..=cfg.epochs {
            for row in gw.iter_mut() {
                for &j in &active_idx {
                    row[j] = 0.0;
                }
            }
            gb.iter_mut().for_each(|g| *g = 0.0);
            for (x, t) in xs.iter().zip(targets) {
                let mut p: Vec<f32> = (0..k)
                    .map(|c| b[c] + x.iter().map(|&(i, v)| w[c][i as usize] * v).sum::<f32>())
                    .collect();
                softmax_in_place(&mut p);
                for c in 0..k {
                    let d = (p[c] - t[c]) / n;
                    if d == 0.0 {
                        continue;
                    }
                    gb[c] += d;
                    for &(i, v) in x.iter() {
                        gw[c][i as usize] += d * v;
                    }
                }
            }
            let tt = epoch as f32;
            let (bc1, bc2) = (1.0 - b1.powf(tt), 1.0 - b2.powf(tt));
            for c in 0..k {
                for &j in &active_idx {
                    let g = gw[c][j] + lambda * w[c][j];
                    mw[c][j] = b1 * mw[c][j] + (1.0 - b1) * g;
                    vw[c][j] = b2 * vw[c][j] + (1.0 - b2) * g * g;
                    w[c][j] -= cfg.lr * (mw[c][j] / bc1) / ((vw[c][j] / bc2).sqrt() + eps);
                }
                let g = gb[c];
                mb[c] = b1 * mb[c] + (1.0 - b1) * g;
                vb[c] = b2 * vb[c] + (1.0 - b2) * g * g;
                b[c] -= cfg.lr * (mb[c] / bc1) / ((vb[c] / bc2).sqrt() + eps);
            }
        }

        let weights = w
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .enumerate()
                    .filter(|(_, v)| v.abs() >= 1e-4)
                    .map(|(i, v)| (i as u32, v))
                    .collect()
            })
            .collect();
        Ok(Self {
            n_outcomes: k,
            weights,
            bias: b,
            n_train: xs.len(),
        })
    }

    /// Raw logits (pre-temperature) for one feature vector.
    pub fn logits(&self, x: &SparseVec) -> Vec<f32> {
        self.weights
            .iter()
            .zip(&self.bias)
            .map(|(row, &b)| b + sparse_dot(row, x))
            .collect()
    }
}

fn sparse_dot(a: &[(u32, f32)], b: &[(u32, f32)]) -> f32 {
    let (mut i, mut j, mut s) = (0usize, 0usize, 0.0f32);
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                s += a[i].1 * b[j].1;
                i += 1;
                j += 1;
            }
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{Featurizer, FeaturizerConfig};

    #[test]
    fn learns_soft_and_hard_targets() {
        let texts: Vec<String> = [
            "refund my payment",
            "charged twice on my card",
            "billing error on invoice",
            "app crashes on login",
            "cannot connect to server",
            "error 500 when saving",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let f = Featurizer::fit(&texts, &FeaturizerConfig::default());
        let xs: Vec<SparseVec> = texts.iter().map(|t| f.transform(t)).collect();
        let refs: Vec<&SparseVec> = xs.iter().collect();
        let targets: Vec<Vec<f32>> = (0..6)
            .map(|i| {
                if i < 3 {
                    vec![0.9, 0.1]
                } else {
                    vec![0.0, 1.0]
                }
            })
            .collect();
        let h =
            LinearHead::train(&refs, &targets, 2, f.dim(), &HeadTrainConfig::default()).unwrap();
        let mut p = h.logits(&f.transform("refund the double charge on my card"));
        softmax_in_place(&mut p);
        assert!(p[0] > 0.5, "{p:?}");
        let mut q = h.logits(&f.transform("server error when I login"));
        softmax_in_place(&mut q);
        assert!(q[1] > 0.5, "{q:?}");
    }
}
