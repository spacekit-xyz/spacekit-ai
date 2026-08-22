//! Adjustable-cone router for two frozen LM specialists (CL-1).
//! Ported from `growformer::dimension::cone_router` with LM-oriented naming.

use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// Oracle-free features from two specialist confidence scalars in `[0,1]`.
pub fn lm_cone_features(scalar_a: f32, scalar_b: f32) -> Vec<f32> {
    let s = scalar_a - 0.5;
    let c = scalar_b - 0.5;
    vec![s, c, s - c, (s - c).abs(), s.abs() - c.abs()]
}

pub const LM_CONE_FEATURE_DIM: usize = 5;

#[derive(Clone, Debug)]
pub struct LmConeSample {
    pub features: Vec<f32>,
    /// Route to specialist A when true (else B).
    pub route_a: bool,
    /// Train-only ambiguity proxy for boundary curriculum (not used at inference).
    pub ambiguity: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct LmConeConfig {
    pub inner_radius: f32,
    pub annulus_eps: f32,
    pub hidden: usize,
    pub epochs: usize,
    pub lr: f32,
    pub l2: f32,
    pub tau_narrow: f32,
    pub curriculum_boost: f32,
    pub balance_lambda: f32,
    pub seed: u64,
}

impl Default for LmConeConfig {
    fn default() -> Self {
        Self {
            inner_radius: 0.4,
            annulus_eps: 0.08,
            hidden: 8,
            epochs: 1200,
            lr: 0.3,
            l2: 1e-4,
            tau_narrow: 0.5,
            curriculum_boost: 3.0,
            balance_lambda: 0.5,
            seed: 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LmConeDecision {
    pub weight_a: f32,
    pub cone_width: f32,
    pub margin: f32,
    pub wide: bool,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct LmAdjustableConeRouter {
    boundary: Mlp,
    fast: Mlp,
    piecewise: Mlp,
    tau_narrow: f32,
    feat_dim: usize,
}

impl LmAdjustableConeRouter {
    pub fn train(samples: &[LmConeSample], cfg: LmConeConfig) -> Self {
        let feat_dim = samples
            .first()
            .map(|s| s.features.len())
            .unwrap_or(LM_CONE_FEATURE_DIM);
        let mut rng = StdRng::seed_from_u64(cfg.seed.wrapping_mul(2_654_435_761).wrapping_add(1));

        let near: Vec<bool> = samples
            .iter()
            .map(|s| (s.ambiguity - cfg.inner_radius).abs() < cfg.annulus_eps)
            .collect();

        let n_near = near.iter().filter(|&&b| b).count().max(1);
        let n_far = samples.len().saturating_sub(n_near).max(1);
        let w_near = samples.len() as f32 / (2.0 * n_near as f32);
        let w_far = samples.len() as f32 / (2.0 * n_far as f32);
        let mut boundary = Mlp::new(feat_dim, cfg.hidden, &mut rng);
        for _ in 0..cfg.epochs {
            let mut grad = MlpGrad::zeros(feat_dim, cfg.hidden);
            let mut count = 0.0f32;
            for (s, &is_near) in samples.iter().zip(near.iter()) {
                let (a1, logit) = boundary.forward(&s.features);
                let p = sigmoid(logit);
                let y = if is_near { 1.0 } else { 0.0 };
                let w = if is_near { w_near } else { w_far };
                boundary.accumulate(&s.features, &a1, w * (p - y), &mut grad);
                count += 1.0;
            }
            boundary.apply(&grad, cfg.lr / count.max(1.0), cfg.l2);
        }

        let mut fast = Mlp::new(feat_dim, cfg.hidden, &mut rng);
        for _ in 0..cfg.epochs {
            let mut grad = MlpGrad::zeros(feat_dim, cfg.hidden);
            let mut count = 0.0f32;
            for s in samples {
                let (a1, logit) = fast.forward(&s.features);
                let p = sigmoid(logit);
                let y = if s.route_a { 1.0 } else { 0.0 };
                fast.accumulate(&s.features, &a1, p - y, &mut grad);
                count += 1.0;
            }
            fast.apply(&grad, cfg.lr / count.max(1.0), cfg.l2);
        }

        let pw_dim = feat_dim + 1;
        let mut piecewise = Mlp::new(pw_dim, cfg.hidden, &mut rng);
        let pw_inputs: Vec<Vec<f32>> = samples
            .iter()
            .map(|s| {
                let p_near = sigmoid(boundary.forward(&s.features).1);
                let mut f = s.features.clone();
                f.push(p_near);
                f
            })
            .collect();
        for _ in 0..cfg.epochs {
            let mut grad = MlpGrad::zeros(pw_dim, cfg.hidden);
            let mut sum_p_annulus = 0.0f32;
            let mut n_annulus = 0.0f32;
            for (x, &is_near) in pw_inputs.iter().zip(near.iter()) {
                if is_near {
                    sum_p_annulus += sigmoid(piecewise.forward(x).1);
                    n_annulus += 1.0;
                }
            }
            let mean_p_annulus = if n_annulus > 0.0 {
                sum_p_annulus / n_annulus
            } else {
                0.5
            };
            let mut count = 0.0f32;
            for ((x, s), &is_near) in pw_inputs.iter().zip(samples.iter()).zip(near.iter()) {
                let (a1, logit) = piecewise.forward(x);
                let p = sigmoid(logit);
                let y = if s.route_a { 1.0 } else { 0.0 };
                let w = if is_near {
                    1.0 + cfg.curriculum_boost
                } else {
                    1.0
                };
                let mut dlogit = w * (p - y);
                if is_near {
                    dlogit += cfg.balance_lambda * (mean_p_annulus - 0.5);
                }
                piecewise.accumulate(x, &a1, dlogit, &mut grad);
                count += 1.0;
            }
            piecewise.apply(&grad, cfg.lr / count.max(1.0), cfg.l2);
        }

        Self {
            boundary,
            fast,
            piecewise,
            tau_narrow: cfg.tau_narrow,
            feat_dim,
        }
    }

    pub fn cone_width(&self, features: &[f32]) -> f32 {
        sigmoid(self.boundary.forward(features).1)
    }

    pub fn decide(&self, features: &[f32]) -> LmConeDecision {
        let cone_width = self.cone_width(features);
        if cone_width < self.tau_narrow {
            let margin = self.fast.forward(features).1;
            LmConeDecision {
                weight_a: if margin >= 0.0 { 1.0 } else { 0.0 },
                cone_width,
                margin,
                wide: false,
            }
        } else {
            let mut x = features.to_vec();
            x.push(cone_width);
            let margin = self.piecewise.forward(&x).1;
            LmConeDecision {
                weight_a: sigmoid(margin),
                cone_width,
                margin,
                wide: true,
            }
        }
    }

    pub fn route_index(&self, features: &[f32]) -> usize {
        if self.decide(features).weight_a >= 0.5 {
            0
        } else {
            1
        }
    }

    pub fn weight_a(&self, features: &[f32]) -> f32 {
        self.decide(features).weight_a
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct Mlp {
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: f32,
    in_dim: usize,
    hidden: usize,
}

impl Mlp {
    fn new(in_dim: usize, hidden: usize, rng: &mut StdRng) -> Self {
        let scale = (1.0 / in_dim.max(1) as f32).sqrt();
        let w1 = (0..hidden)
            .map(|_| (0..in_dim).map(|_| rng.gen_range(-scale..scale)).collect())
            .collect();
        let w2 = (0..hidden).map(|_| rng.gen_range(-scale..scale)).collect();
        Self {
            w1,
            b1: vec![0.0; hidden],
            w2,
            b2: 0.0,
            in_dim,
            hidden,
        }
    }

    fn forward(&self, x: &[f32]) -> (Vec<f32>, f32) {
        let mut a1 = vec![0.0f32; self.hidden];
        let mut logit = self.b2;
        for j in 0..self.hidden {
            let mut z = self.b1[j];
            let row = &self.w1[j];
            for i in 0..self.in_dim.min(x.len()) {
                z += row[i] * x[i];
            }
            let a = z.tanh();
            a1[j] = a;
            logit += self.w2[j] * a;
        }
        (a1, logit)
    }

    fn accumulate(&self, x: &[f32], a1: &[f32], dlogit: f32, grad: &mut MlpGrad) {
        grad.b2 += dlogit;
        for j in 0..self.hidden {
            grad.w2[j] += dlogit * a1[j];
            let dz = dlogit * self.w2[j] * (1.0 - a1[j] * a1[j]);
            grad.b1[j] += dz;
            let row = &mut grad.w1[j];
            for i in 0..self.in_dim.min(x.len()) {
                row[i] += dz * x[i];
            }
        }
    }

    fn apply(&mut self, grad: &MlpGrad, lr_over_n: f32, l2: f32) {
        self.b2 -= lr_over_n * grad.b2;
        for j in 0..self.hidden {
            self.w2[j] -= lr_over_n * (grad.w2[j] + l2 * self.w2[j]);
            self.b1[j] -= lr_over_n * grad.b1[j];
            for i in 0..self.in_dim {
                self.w1[j][i] -= lr_over_n * (grad.w1[j][i] + l2 * self.w1[j][i]);
            }
        }
    }
}

struct MlpGrad {
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: f32,
}

impl MlpGrad {
    fn zeros(in_dim: usize, hidden: usize) -> Self {
        Self {
            w1: vec![vec![0.0; in_dim]; hidden],
            b1: vec![0.0; hidden],
            w2: vec![0.0; hidden],
            b2: 0.0,
        }
    }
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}
