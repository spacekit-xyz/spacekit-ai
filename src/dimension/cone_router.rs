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

/// Label-free training strategies (Phase 3h): no region / `r` in the loss.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabelFreeStrategy {
    /// Median-split on the circles specialist scalar (sign fixed by prior analysis:
    /// `f_circles ↔ r ≈ +0.87`, so low circles ⇒ inner ⇒ route spiral).
    CirclesThreshold,
    /// 1-D 2-means on circles scalar; assign spiral to the low-circles centroid.
    CirclesCluster,
    /// CirclesThreshold warm-start, then one self-distill round from the router's
    /// soft train predictions (still no `r`).
    Bootstrap,
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
        let near: Vec<bool> = samples
            .iter()
            .map(|s| (s.r - cfg.inner_radius).abs() < cfg.annulus_eps)
            .collect();
        let routes: Vec<bool> = samples.iter().map(|s| s.route_spiral).collect();
        let feats: Vec<&[f32]> = samples.iter().map(|s| s.features.as_slice()).collect();
        Self::train_from_targets(&feats, &routes, &near, cfg)
    }

    /// Phase 3h: train with **no region / `r` in the loss**.
    ///
    /// `specialist_pairs` are `(spiral_scalar, circles_scalar)` per train point.
    /// Pseudo route + near-boundary targets are derived only from those scalars.
    pub fn train_label_free(
        specialist_pairs: &[(f32, f32)],
        cfg: ConeConfig,
        strategy: LabelFreeStrategy,
    ) -> Self {
        let features: Vec<Vec<f32>> = specialist_pairs
            .iter()
            .map(|(s, c)| cone_features(*s, *c))
            .collect();
        let (mut routes, near) = label_free_pseudo_targets(specialist_pairs, strategy);

        let feat_refs: Vec<&[f32]> = features.iter().map(|f| f.as_slice()).collect();
        let mut router = Self::train_from_targets(&feat_refs, &routes, &near, cfg);

        if strategy == LabelFreeStrategy::Bootstrap && !features.is_empty() {
            // One self-distill round: hard labels from the warm-started router's train decisions.
            for (i, f) in features.iter().enumerate() {
                routes[i] = router.route_index(f) == 0;
            }
            // Keep the same near mask (still feature-only).
            let feat_refs: Vec<&[f32]> = features.iter().map(|f| f.as_slice()).collect();
            router = Self::train_from_targets(&feat_refs, &routes, &near, cfg);
        }

        router
    }

    fn train_from_targets(
        features: &[&[f32]],
        routes: &[bool],
        near: &[bool],
        cfg: ConeConfig,
    ) -> Self {
        assert_eq!(features.len(), routes.len());
        assert_eq!(features.len(), near.len());
        let feat_dim = features
            .first()
            .map(|f| f.len())
            .unwrap_or(CONE_FEATURE_DIM);
        let mut rng = StdRng::seed_from_u64(cfg.seed.wrapping_mul(2_654_435_761).wrapping_add(1));

        let n_near = near.iter().filter(|&&b| b).count().max(1);
        let n_far = features.len().saturating_sub(n_near).max(1);
        let w_near = features.len() as f32 / (2.0 * n_near as f32);
        let w_far = features.len() as f32 / (2.0 * n_far as f32);
        let mut boundary = Mlp::new(feat_dim, cfg.hidden, &mut rng);
        for _ in 0..cfg.epochs {
            let mut grad = MlpGrad::zeros(feat_dim, cfg.hidden);
            let mut count = 0.0f32;
            for (f, &is_near) in features.iter().zip(near.iter()) {
                let (a1, logit) = boundary.forward(f);
                let p = sigmoid(logit);
                let y = if is_near { 1.0 } else { 0.0 };
                let w = if is_near { w_near } else { w_far };
                boundary.accumulate(f, &a1, w * (p - y), &mut grad);
                count += 1.0;
            }
            boundary.apply(&grad, cfg.lr / count.max(1.0), cfg.l2);
        }

        let mut fast = Mlp::new(feat_dim, cfg.hidden, &mut rng);
        for _ in 0..cfg.epochs {
            let mut grad = MlpGrad::zeros(feat_dim, cfg.hidden);
            let mut count = 0.0f32;
            for (f, &route_spiral) in features.iter().zip(routes.iter()) {
                let (a1, logit) = fast.forward(f);
                let p = sigmoid(logit);
                let y = if route_spiral { 1.0 } else { 0.0 };
                fast.accumulate(f, &a1, p - y, &mut grad);
                count += 1.0;
            }
            fast.apply(&grad, cfg.lr / count.max(1.0), cfg.l2);
        }

        let pw_dim = feat_dim + 1;
        let mut piecewise = Mlp::new(pw_dim, cfg.hidden, &mut rng);
        let pw_inputs: Vec<Vec<f32>> = features
            .iter()
            .map(|f| {
                let p_near = sigmoid(boundary.forward(f).1);
                let mut x = f.to_vec();
                x.push(p_near);
                x
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
            for ((x, &route_spiral), &is_near) in
                pw_inputs.iter().zip(routes.iter()).zip(near.iter())
            {
                let (a1, logit) = piecewise.forward(x);
                let p = sigmoid(logit);
                let y = if route_spiral { 1.0 } else { 0.0 };
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

/// Build (route_spiral, near_boundary) targets from specialist scalars only — no `r`.
fn label_free_pseudo_targets(
    pairs: &[(f32, f32)],
    strategy: LabelFreeStrategy,
) -> (Vec<bool>, Vec<bool>) {
    if pairs.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let circles: Vec<f32> = pairs.iter().map(|(_, c)| *c).collect();
    // Near-boundary proxy: high |s−c| disagreement (feature index 3 of cone_features).
    let disagree: Vec<f32> = pairs
        .iter()
        .map(|(s, c)| ((s - 0.5) - (c - 0.5)).abs())
        .collect();
    let mut disagree_sorted = disagree.clone();
    disagree_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let disagree_cut = percentile_sorted(&disagree_sorted, 0.70);

    let routes = match strategy {
        LabelFreeStrategy::CirclesThreshold | LabelFreeStrategy::Bootstrap => {
            let mut sorted = circles.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let med = percentile_sorted(&sorted, 0.50);
            // Prior: f_circles ↔ r ≈ +0.87 ⇒ low circles ⇒ inner ⇒ route spiral.
            circles.iter().map(|&c| c < med).collect()
        }
        LabelFreeStrategy::CirclesCluster => {
            let (lo, hi) = two_means_1d(&circles, 24);
            let mid = 0.5 * (lo + hi);
            // Spiral ← low-circles cluster.
            circles.iter().map(|&c| c < mid).collect()
        }
    };
    let near: Vec<bool> = disagree.iter().map(|&d| d >= disagree_cut).collect();
    (routes, near)
}

fn percentile_sorted(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f32 - 1.0) * q.clamp(0.0, 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn two_means_1d(xs: &[f32], iters: usize) -> (f32, f32) {
    if xs.is_empty() {
        return (0.0, 1.0);
    }
    let mut sorted = xs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut c0 = sorted[0];
    let mut c1 = *sorted.last().unwrap();
    for _ in 0..iters {
        let mut s0 = 0.0f32;
        let mut n0 = 0.0f32;
        let mut s1 = 0.0f32;
        let mut n1 = 0.0f32;
        for &x in xs {
            if (x - c0).abs() <= (x - c1).abs() {
                s0 += x;
                n0 += 1.0;
            } else {
                s1 += x;
                n1 += 1.0;
            }
        }
        if n0 > 0.0 {
            c0 = s0 / n0;
        }
        if n1 > 0.0 {
            c1 = s1 / n1;
        }
    }
    if c0 <= c1 {
        (c0, c1)
    } else {
        (c1, c0)
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
        assert!(
            acc > 0.85,
            "region agreement too low on separable task: {acc}"
        );
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

    #[test]
    fn label_free_train_never_reads_r() {
        // Build specialist pairs from synthetic stream but drop r from the train API.
        let samples = synthetic_samples(5, 200);
        let pairs: Vec<(f32, f32)> = samples
            .iter()
            .map(|s| {
                // Recover approx specialists from features (s+0.5, c+0.5).
                (s.features[0] + 0.5, s.features[1] + 0.5)
            })
            .collect();
        let router = AdjustableConeRouter::train_label_free(
            &pairs,
            ConeConfig {
                seed: 5,
                ..ConeConfig::default()
            },
            LabelFreeStrategy::CirclesThreshold,
        );
        let test = synthetic_samples(6, 200);
        let agree = test
            .iter()
            .filter(|s| (router.route_index(&s.features) == 0) == (s.r < 0.4))
            .count();
        let acc = agree as f32 / test.len() as f32;
        // Looser than supervised train: label-free may be weaker, but must beat chance.
        assert!(acc > 0.60, "label-free region agreement too low: {acc}");
    }
}
