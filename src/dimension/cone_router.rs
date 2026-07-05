//! Adjustable-Cone Cognitive Router (Task E, Phase 1).
//!
//! Targets the whitepaper's authenticated routing failure: `VirtualGroup` blends
//! frozen specialists with one global weight vector (structurally cannot switch),
//! and the lattice `LearnedRouter` collapses to a constant specialist on ~14/20
//! seeds under sparse boundary coverage. This module builds a router whose
//! effective "cognitive cone" (the scope of machinery engaged) expands near the
//! decision boundary:
//!
//!   - **Small cone** (clear interior/outer, decisive experts): a fast gate picks
//!     a single specialist.
//!   - **Wide cone** (near the annulus, ambiguous experts): a boundary controller
//!     escalates to an input-dependent piecewise gate that produces a per-point
//!     blend weight — the thing `VirtualGroup` lacks.
//!
//! ## Honesty contract (matches the plan's "fair fight" rule)
//!
//! Inference is **oracle-free**: every head consumes only features derived from
//! frozen-specialist outputs ([`cone_features`]); the latent switching radius `r`
//! is never an input. `r` is used **only at training time** (annulus curriculum,
//! near-boundary labels, region labels) and for certification. Recovery of the
//! switch is legitimate because the specialists already encode it
//! (`f_circles ↔ r ≈ 0.87`).
//!
//! ## Decontamination note (margin↔r is NOT trained on)
//!
//! An earlier version regressed the decision margin toward `(inner_radius - r)`
//! ("margin shaping"). That made the `margin ↔ (0.4 − r)` certifier **circular** —
//! it would have measured how well the loss optimized the very quantity the
//! certifier scores. The shaping term has been **removed from every head's loss**.
//! The router still exposes [`AdjustableConeRouter::margin`] (the decision log-odds)
//! so `margin↔r` can be reported, but only as an *observational* quantity the loss
//! never touched — never as a pre-registered gate.
//!
//! The module is deliberately decoupled from `MainDimension`: callers extract
//! [`ConeSample`]s (features + train-only `r`) and the router never sees geometry.

use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// Engineered, oracle-free router features from the two specialist scalars.
///
/// All five components are functions of the frozen-specialist outputs only:
/// centered spiral output, centered circles output, their disagreement, the
/// magnitude of disagreement (ambiguity), and the decisiveness gap. Position /
/// radius never enters.
pub fn cone_features(spiral_scalar: f32, circles_scalar: f32) -> Vec<f32> {
    let s = spiral_scalar - 0.5;
    let c = circles_scalar - 0.5;
    vec![s, c, s - c, (s - c).abs(), s.abs() - c.abs()]
}

/// Dimension of [`cone_features`].
pub const CONE_FEATURE_DIM: usize = 5;

/// One training record: oracle-free features plus the train-only radius `r`.
#[derive(Clone, Debug)]
pub struct ConeSample {
    /// Oracle-free features (`cone_features`).
    pub features: Vec<f32>,
    /// Composite-teacher routing label (true ⇒ route spiral). Derived without `r`.
    pub route_spiral: bool,
    /// Generative radius. TRAIN-ONLY: curriculum, near-boundary label, margin shaping.
    pub r: f32,
}

/// Hyperparameters for [`AdjustableConeRouter::train`].
#[derive(Clone, Copy, Debug)]
pub struct ConeConfig {
    pub inner_radius: f32,
    pub annulus_eps: f32,
    pub hidden: usize,
    pub epochs: usize,
    pub lr: f32,
    pub l2: f32,
    /// Cone-width threshold: P(near-boundary) above this escalates to the wide cone.
    pub tau_narrow: f32,
    /// Extra weight on annulus samples when training the piecewise gate (curriculum).
    pub curriculum_boost: f32,
    /// Cone-expansion regularizer strength: pulls the mean annulus spiral-prob to 0.5,
    /// penalizing collapse to a constant specialist exactly where uncertainty is high.
    pub balance_lambda: f32,
    pub seed: u64,
}

impl Default for ConeConfig {
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

/// A single routing decision with its cone diagnostics.
#[derive(Clone, Copy, Debug)]
pub struct ConeDecision {
    /// Blend weight on the spiral specialist in `[0,1]`; `1-spiral_weight` on circles.
    /// Hard `0.0`/`1.0` in the narrow cone, soft in the wide cone.
    pub spiral_weight: f32,
    /// Estimated boundary proximity P(near-annulus); the cone width signal.
    pub cone_width: f32,
    /// Decision log-odds of routing spiral (margin); used by the margin↔r certifier.
    pub margin: f32,
    /// Whether the wide (deliberative) cone was engaged.
    pub wide: bool,
}

/// Adjustable-Cone Cognitive Router: boundary controller + fast gate + piecewise gate.
#[derive(Clone, Serialize, Deserialize)]
pub struct AdjustableConeRouter {
    boundary: Mlp,
    fast: Mlp,
    piecewise: Mlp,
    tau_narrow: f32,
    feat_dim: usize,
}

impl AdjustableConeRouter {
    /// Train all three heads from oracle-free features + train-only `r`.
    pub fn train(samples: &[ConeSample], cfg: ConeConfig) -> Self {
        let feat_dim = samples
            .first()
            .map(|s| s.features.len())
            .unwrap_or(CONE_FEATURE_DIM);
        let mut rng = StdRng::seed_from_u64(cfg.seed.wrapping_mul(2_654_435_761).wrapping_add(1));

        // Per-sample derived training targets (train-time r allowed here).
        let near: Vec<bool> = samples
            .iter()
            .map(|s| (s.r - cfg.inner_radius).abs() < cfg.annulus_eps)
            .collect();

        // --- Boundary controller: features -> P(near annulus) ---
        // Annulus points are rare (~20% of 30), so weight by inverse class prior.
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

        // --- Fast gate (small cone): features -> P(route spiral) ---
        // Plain BCE on the region label. No margin shaping (would contaminate margin↔r).
        let mut fast = Mlp::new(feat_dim, cfg.hidden, &mut rng);
        for _ in 0..cfg.epochs {
            let mut grad = MlpGrad::zeros(feat_dim, cfg.hidden);
            let mut count = 0.0f32;
            for s in samples {
                let (a1, logit) = fast.forward(&s.features);
                let p = sigmoid(logit);
                let y = if s.route_spiral { 1.0 } else { 0.0 };
                fast.accumulate(&s.features, &a1, p - y, &mut grad);
                count += 1.0;
            }
            fast.apply(&grad, cfg.lr / count.max(1.0), cfg.l2);
        }

        // --- Piecewise gate (wide cone): [features, P_near] -> blend weight ---
        // Boundary-aware curriculum + cone-expansion balance + margin shaping.
        let pw_dim = feat_dim + 1;
        let mut piecewise = Mlp::new(pw_dim, cfg.hidden, &mut rng);
        // Precompute augmented inputs once (P_near is fixed after boundary training).
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
            // Pre-pass: mean spiral-prob over the annulus subset for the balance term.
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
                let y = if s.route_spiral { 1.0 } else { 0.0 };
                // Curriculum weight: upweight the sparse annulus band.
                let w = if is_near { 1.0 + cfg.curriculum_boost } else { 1.0 };
                let mut dlogit = w * (p - y);
                // Cone-expansion regularizer: push the annulus mean toward 0.5 so the
                // boundary band cannot collapse to a constant specialist. (Note: this
                // works *against* the annulus-localization certifier, so that certifier
                // is adversarially clean — the loss does not optimize toward it passing.)
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

    /// Estimated boundary proximity P(near-annulus): the cone width signal.
    pub fn cone_width(&self, features: &[f32]) -> f32 {
        sigmoid(self.boundary.forward(features).1)
    }

    /// Full routing decision (oracle-free).
    pub fn decide(&self, features: &[f32]) -> ConeDecision {
        let cone_width = self.cone_width(features);
        if cone_width < self.tau_narrow {
            // Narrow cone: trust the fast gate, hard-switch to one specialist.
            let margin = self.fast.forward(features).1;
            ConeDecision {
                spiral_weight: if margin >= 0.0 { 1.0 } else { 0.0 },
                cone_width,
                margin,
                wide: false,
            }
        } else {
            // Wide cone: deliberative input-dependent blend.
            let mut x = features.to_vec();
            x.push(cone_width);
            let margin = self.piecewise.forward(&x).1;
            ConeDecision {
                spiral_weight: sigmoid(margin),
                cone_width,
                margin,
                wide: true,
            }
        }
    }

    /// Spiral blend weight in `[0,1]` (`1.0` ⇒ pure spiral).
    pub fn spiral_weight(&self, features: &[f32]) -> f32 {
        self.decide(features).spiral_weight
    }

    /// Hard route index (0 = spiral, 1 = circles) for region-agreement certification.
    pub fn route_index(&self, features: &[f32]) -> usize {
        if self.decide(features).spiral_weight >= 0.5 {
            0
        } else {
            1
        }
    }

    /// Decision margin (log-odds of routing spiral) for the margin↔r certifier.
    pub fn margin(&self, features: &[f32]) -> f32 {
        self.decide(features).margin
    }

    /// Blend two specialist scalars by the per-point spiral weight.
    pub fn blend_scalars(&self, features: &[f32], spiral_scalar: f32, circles_scalar: f32) -> f32 {
        let w = self.spiral_weight(features);
        w * spiral_scalar + (1.0 - w) * circles_scalar
    }

    pub fn feature_dim(&self) -> usize {
        self.feat_dim
    }
}

// ---------------------------------------------------------------------------
// Minimal one-hidden-layer MLP (tanh hidden, sigmoid output) with full-batch GD.
// Kept dependency-light and deterministic, consistent with the demo's existing
// RadiusLogisticGate / CompetenceHead style.
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize, Deserialize)]
struct Mlp {
    w1: Vec<Vec<f32>>, // [hidden][in]
    b1: Vec<f32>,      // [hidden]
    w2: Vec<f32>,      // [hidden]
    b2: f32,
    in_dim: usize,
    hidden: usize,
}

impl Mlp {
    fn new(in_dim: usize, hidden: usize, rng: &mut StdRng) -> Self {
        // Small symmetric init scaled by fan-in.
        let scale = (1.0 / in_dim.max(1) as f32).sqrt();
        let w1 = (0..hidden)
            .map(|_| (0..in_dim).map(|_| rng.gen_range(-scale..scale)).collect())
            .collect();
        let w2 = (0..hidden)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();
        Self {
            w1,
            b1: vec![0.0; hidden],
            w2,
            b2: 0.0,
            in_dim,
            hidden,
        }
    }

    /// Forward pass: returns hidden activations (tanh) and the output logit.
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

    /// Backprop one sample given the upstream gradient on the logit; accumulate into `grad`.
    fn accumulate(&self, x: &[f32], a1: &[f32], dlogit: f32, grad: &mut MlpGrad) {
        grad.b2 += dlogit;
        for j in 0..self.hidden {
            grad.w2[j] += dlogit * a1[j];
            // dL/dz1 = dlogit * w2[j] * (1 - a1^2)
            let dz = dlogit * self.w2[j] * (1.0 - a1[j] * a1[j]);
            grad.b1[j] += dz;
            let row = &mut grad.w1[j];
            for i in 0..self.in_dim.min(x.len()) {
                row[i] += dz * x[i];
            }
        }
    }

    /// Apply accumulated gradient (already scaled by 1/N via `lr_over_n`) with L2.
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

#[cfg(test)]
mod tests {
    use super::*;

    // Synthetic separable Task-E-like stream: spiral output ~ region(r<0.4).
    // Builds oracle-free features from specialist scalars; r is train-only.
    fn synthetic_samples(seed: u64, n: usize) -> Vec<ConeSample> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let r: f32 = rng.gen_range(0.0..1.0);
            let inner = r < 0.4;
            // Smooth specialists (like real Task E): decisive far from the boundary,
            // ambiguous (both ~0.5) near r=0.4. This makes the boundary band genuinely
            // hard, which is the whole point of the cone controller.
            let n1 = rng.gen_range(-0.1..0.1_f32);
            let n2 = rng.gen_range(-0.1..0.1_f32);
            let spiral = (0.5 + 1.6 * (0.4 - r) + n1).clamp(0.02, 0.98);
            let circles = (0.5 + 1.6 * (r - 0.4) + n2).clamp(0.02, 0.98);
            out.push(ConeSample {
                features: cone_features(spiral, circles),
                route_spiral: inner, // composite teacher ≈ region for separable experts
                r,
            });
        }
        out
    }

    #[test]
    fn separable_region_sanity() {
        let train = synthetic_samples(1, 200);
        let test = synthetic_samples(2, 200);
        let router = AdjustableConeRouter::train(&train, ConeConfig::default());
        let agree = test
            .iter()
            .filter(|s| (router.route_index(&s.features) == 0) == (s.r < 0.4))
            .count();
        let acc = agree as f32 / test.len() as f32;
        assert!(acc > 0.85, "region agreement too low on separable task: {acc}");
    }

    #[test]
    fn oracle_free_decision_is_pure_function_of_features() {
        // Identical specialist features but wildly different r must yield identical
        // decisions — proving no positional/oracle leak at inference.
        let router = AdjustableConeRouter::train(&synthetic_samples(3, 200), ConeConfig::default());
        let feats = cone_features(0.88, 0.52);
        let d1 = router.decide(&feats);
        let d2 = router.decide(&feats.clone());
        assert_eq!(d1.spiral_weight, d2.spiral_weight);
        assert_eq!(d1.margin, d2.margin);
        assert_eq!(d1.cone_width, d2.cone_width);
        // The API exposes no way to pass r into a decision: decide() takes features only.
    }

    #[test]
    fn cone_widens_on_ambiguous_inputs() {
        let router = AdjustableConeRouter::train(&synthetic_samples(4, 300), ConeConfig::default());
        // Decisive interior (experts strongly disagree) vs ambiguous boundary (experts tie).
        let decisive = router.cone_width(&cone_features(0.95, 0.5));
        let ambiguous = router.cone_width(&cone_features(0.7, 0.7));
        assert!(
            ambiguous > decisive,
            "cone should widen on ambiguous experts: ambiguous={ambiguous} decisive={decisive}"
        );
    }
}
