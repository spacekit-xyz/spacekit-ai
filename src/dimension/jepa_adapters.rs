//! JEPA-like frozen encoder + promotable predictor adapters (world-model Task E toy).
//!
//! Contract (parameter-isolation preserving):
//! - The sensory encoder is **frozen and hash-pinned** after construction.
//! - Only predictor / affinity adapter parameters train in a Mirror and promote to Main.
//! - No gradient path into the encoder or into previously promoted predictors.
//!
//! This is *not* a claim of full LeCun AMI. It is the minimal predictive Task-E
//! analogue: two dynamics regimes, two promoted predictors, authenticated cone
//! routing on affinity scalars (oracle-free at inference).

use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use super::cone_router::{cone_features, AdjustableConeRouter, ConeConfig, ConeSample};

/// Observation dimension for the toy world (`[x, y, vx, vy]`).
pub const WM_OBS_DIM: usize = 4;
/// Latent dimension after the frozen encoder.
pub const WM_LATENT_DIM: usize = 8;
/// Inner-disk radius for the regime switch (same geometry as Task E).
pub const WM_INNER_RADIUS: f32 = 0.4;

/// Frozen JEPA-like encoder: fixed linear map + tanh. Never trained after init.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrozenJepaEncoder {
    /// `[latent][obs]` weights.
    pub w: Vec<Vec<f32>>,
    pub b: Vec<f32>,
    /// FNV-style fingerprint of weights; promotion must match this pin.
    pub fingerprint: u64,
}

impl FrozenJepaEncoder {
    pub fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9).wrapping_add(17));
        let scale = (2.0 / WM_OBS_DIM as f32).sqrt();
        let w: Vec<Vec<f32>> = (0..WM_LATENT_DIM)
            .map(|_| {
                (0..WM_OBS_DIM)
                    .map(|_| rng.gen_range(-scale..scale))
                    .collect()
            })
            .collect();
        let b: Vec<f32> = (0..WM_LATENT_DIM).map(|_| rng.gen_range(-0.05..0.05)).collect();
        let fingerprint = fingerprint_encoder(&w, &b);
        Self { w, b, fingerprint }
    }

    pub fn encode(&self, obs: &[f32]) -> Vec<f32> {
        assert_eq!(obs.len(), WM_OBS_DIM);
        self.w
            .iter()
            .zip(self.b.iter())
            .map(|(row, &bias)| {
                let mut s = bias;
                for (j, &x) in obs.iter().enumerate() {
                    s += row[j] * x;
                }
                s.tanh()
            })
            .collect()
    }

    pub fn assert_pinned(&self, expected: u64) {
        assert_eq!(
            self.fingerprint, expected,
            "encoder fingerprint drift: got {:#x}, pinned {:#x}",
            self.fingerprint, expected
        );
    }
}

fn fingerprint_encoder(w: &[Vec<f32>], b: &[f32]) -> u64 {
    let mut h = DefaultHasher::new();
    for row in w {
        for &v in row {
            v.to_bits().hash(&mut h);
        }
    }
    for &v in b {
        v.to_bits().hash(&mut h);
    }
    h.finish()
}

/// Promotable predictor adapter: next-latent dynamics + regime affinity head.
///
/// Trained only in Mirror on a single dynamics regime; then frozen into Main.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PredictorAdapter {
    pub name: String,
    pub regime_is_inner: bool,
    /// Dynamics: latent → Δlatent (residual), then `z_next = z + Δ`.
    dyn_w1: Vec<Vec<f32>>,
    dyn_b1: Vec<f32>,
    dyn_w2: Vec<Vec<f32>>,
    dyn_b2: Vec<f32>,
    /// Affinity: latent → logit of P(home regime).
    aff_w1: Vec<Vec<f32>>,
    aff_b1: Vec<f32>,
    aff_w2: Vec<f32>,
    aff_b2: f32,
    hidden: usize,
    /// Encoder fingerprint required at promote time.
    pub encoder_pin: u64,
}

impl PredictorAdapter {
    pub fn new(name: &str, regime_is_inner: bool, encoder_pin: u64, seed: u64) -> Self {
        let hidden = 16;
        let mut rng = StdRng::seed_from_u64(seed);
        let scale_in = (1.0 / WM_LATENT_DIM as f32).sqrt();
        let scale_h = (1.0 / hidden as f32).sqrt();
        let dyn_w1 = rand_mat(hidden, WM_LATENT_DIM, scale_in, &mut rng);
        let dyn_b1 = vec![0.0; hidden];
        let dyn_w2 = rand_mat(WM_LATENT_DIM, hidden, scale_h, &mut rng);
        let dyn_b2 = vec![0.0; WM_LATENT_DIM];
        let aff_w1 = rand_mat(hidden, WM_LATENT_DIM, scale_in, &mut rng);
        let aff_b1 = vec![0.0; hidden];
        let aff_w2 = (0..hidden).map(|_| rng.gen_range(-scale_h..scale_h)).collect();
        Self {
            name: name.to_string(),
            regime_is_inner,
            dyn_w1,
            dyn_b1,
            dyn_w2,
            dyn_b2,
            aff_w1,
            aff_b1,
            aff_w2,
            aff_b2: 0.0,
            hidden,
            encoder_pin,
        }
    }

    pub fn predict_next(&self, z: &[f32]) -> Vec<f32> {
        let h = relu_forward(z, &self.dyn_w1, &self.dyn_b1);
        let delta = linear_forward(&h, &self.dyn_w2, &self.dyn_b2);
        z.iter().zip(delta.iter()).map(|(a, b)| a + b).collect()
    }

    /// Oracle-free regime affinity in `[0, 1]` from current latent only.
    pub fn affinity(&self, z: &[f32]) -> f32 {
        let h = relu_forward(z, &self.aff_w1, &self.aff_b1);
        let mut logit = self.aff_b2;
        for (i, &a) in h.iter().enumerate() {
            logit += self.aff_w2[i] * a;
        }
        sigmoid(logit)
    }

    pub fn prediction_mse(&self, z: &[f32], z_next: &[f32]) -> f32 {
        let pred = self.predict_next(z);
        let mut s = 0.0f32;
        for (a, b) in pred.iter().zip(z_next.iter()) {
            let d = a - b;
            s += d * d;
        }
        s / WM_LATENT_DIM as f32
    }

    /// Train dynamics + affinity on regime-pure pairs. Encoder stays frozen.
    pub fn train(
        &mut self,
        pairs: &[(Vec<f32>, Vec<f32>)],
        contrast_z: &[Vec<f32>],
        epochs: usize,
        lr: f32,
    ) {
        if pairs.is_empty() {
            return;
        }
        for _ in 0..epochs {
            // Dynamics: full-batch GD on residual MLP.
            let mut g_w1 = zeros_mat(self.hidden, WM_LATENT_DIM);
            let mut g_b1 = vec![0.0; self.hidden];
            let mut g_w2 = zeros_mat(WM_LATENT_DIM, self.hidden);
            let mut g_b2 = vec![0.0; WM_LATENT_DIM];
            let n = pairs.len() as f32;
            for (z, z_next) in pairs {
                let h = relu_forward(z, &self.dyn_w1, &self.dyn_b1);
                let delta = linear_forward(&h, &self.dyn_w2, &self.dyn_b2);
                let pred: Vec<f32> = z.iter().zip(delta.iter()).map(|(a, b)| a + b).collect();
                let mut d_pred = vec![0.0f32; WM_LATENT_DIM];
                for i in 0..WM_LATENT_DIM {
                    d_pred[i] = 2.0 * (pred[i] - z_next[i]) / (WM_LATENT_DIM as f32 * n);
                }
                // d_delta = d_pred
                for o in 0..WM_LATENT_DIM {
                    g_b2[o] += d_pred[o];
                    for j in 0..self.hidden {
                        g_w2[o][j] += d_pred[o] * h[j];
                    }
                }
                let mut d_h = vec![0.0f32; self.hidden];
                for j in 0..self.hidden {
                    let mut s = 0.0f32;
                    for o in 0..WM_LATENT_DIM {
                        s += self.dyn_w2[o][j] * d_pred[o];
                    }
                    d_h[j] = if h[j] > 0.0 { s } else { 0.0 };
                }
                for j in 0..self.hidden {
                    g_b1[j] += d_h[j];
                    for i in 0..WM_LATENT_DIM {
                        g_w1[j][i] += d_h[j] * z[i];
                    }
                }
            }
            apply_mat(&mut self.dyn_w1, &g_w1, lr, 1e-4);
            apply_vec(&mut self.dyn_b1, &g_b1, lr, 1e-4);
            apply_mat(&mut self.dyn_w2, &g_w2, lr, 1e-4);
            apply_vec(&mut self.dyn_b2, &g_b2, lr, 1e-4);

            // Affinity: home pairs → 1, contrast latents → 0.
            let mut ag_w1 = zeros_mat(self.hidden, WM_LATENT_DIM);
            let mut ag_b1 = vec![0.0; self.hidden];
            let mut ag_w2 = vec![0.0; self.hidden];
            let mut ag_b2 = 0.0f32;
            let mut count = 0.0f32;
            for (z, _) in pairs {
                accumulate_affinity_grad(
                    z,
                    1.0,
                    &self.aff_w1,
                    &self.aff_b1,
                    &self.aff_w2,
                    self.aff_b2,
                    &mut ag_w1,
                    &mut ag_b1,
                    &mut ag_w2,
                    &mut ag_b2,
                );
                count += 1.0;
            }
            for z in contrast_z {
                accumulate_affinity_grad(
                    z,
                    0.0,
                    &self.aff_w1,
                    &self.aff_b1,
                    &self.aff_w2,
                    self.aff_b2,
                    &mut ag_w1,
                    &mut ag_b1,
                    &mut ag_w2,
                    &mut ag_b2,
                );
                count += 1.0;
            }
            if count > 0.0 {
                let scale = lr / count;
                apply_mat(&mut self.aff_w1, &ag_w1, scale, 1e-4);
                apply_vec(&mut self.aff_b1, &ag_b1, scale, 1e-4);
                apply_vec(&mut self.aff_w2, &ag_w2, scale, 1e-4);
                self.aff_b2 -= scale * ag_b2 + 1e-4 * self.aff_b2;
            }
        }
    }
}

/// Promotion record: encoder pin + frozen predictors. Documents the CL contract.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JepaPromotionBundle {
    pub encoder_fingerprint: u64,
    pub predictors: Vec<PredictorAdapter>,
}

impl JepaPromotionBundle {
    /// Promote predictors only; encoder must match the pin (never updated here).
    pub fn promote(
        encoder: &FrozenJepaEncoder,
        predictors: Vec<PredictorAdapter>,
    ) -> Result<Self, String> {
        for p in &predictors {
            if p.encoder_pin != encoder.fingerprint {
                return Err(format!(
                    "predictor '{}' pin {:#x} ≠ encoder {:#x}",
                    p.name, p.encoder_pin, encoder.fingerprint
                ));
            }
        }
        Ok(Self {
            encoder_fingerprint: encoder.fingerprint,
            predictors,
        })
    }

    pub fn verify_encoder(&self, encoder: &FrozenJepaEncoder) -> Result<(), String> {
        if encoder.fingerprint != self.encoder_fingerprint {
            return Err(format!(
                "encoder drift: live {:#x} ≠ pinned {:#x}",
                encoder.fingerprint, self.encoder_fingerprint
            ));
        }
        Ok(())
    }
}

/// One transition in the toy world.
#[derive(Clone, Debug)]
pub struct WmTransition {
    pub obs: Vec<f32>,
    pub obs_next: Vec<f32>,
    pub z: Vec<f32>,
    pub z_next: Vec<f32>,
    pub r: f32,
    pub regime_inner: bool,
}

/// Inner: rotation dynamics. Outer: radial expand + shear. Switch at `WM_INNER_RADIUS`.
pub fn step_dynamics(obs: &[f32], dt: f32) -> Vec<f32> {
    let x = obs[0];
    let y = obs[1];
    let vx = obs[2];
    let vy = obs[3];
    let r = (x * x + y * y).sqrt();
    if r < WM_INNER_RADIUS {
        // Rotation in the plane + mild damping of velocity.
        let ang = 0.35 * dt;
        let (c, s) = (ang.cos(), ang.sin());
        let nx = c * x - s * y;
        let ny = s * x + c * y;
        let nvx = c * vx - s * vy;
        let nvy = s * vx + c * vy;
        vec![nx, ny, nvx * 0.98, nvy * 0.98]
    } else {
        // Radial push + shear.
        let scale = 1.0 + 0.12 * dt;
        let nx = x * scale + 0.05 * y * dt;
        let ny = y * scale - 0.05 * x * dt;
        let nvx = vx * 0.95 + 0.08 * x * dt;
        let nvy = vy * 0.95 + 0.08 * y * dt;
        vec![nx, ny, nvx, nvy]
    }
}

pub fn generate_transitions(
    encoder: &FrozenJepaEncoder,
    n: usize,
    balanced: bool,
    rng: &mut StdRng,
) -> Vec<WmTransition> {
    let mut out = Vec::with_capacity(n);
    let half = n / 2;
    for i in 0..n {
        let want_inner = if balanced {
            i < half
        } else {
            rng.gen::<bool>()
        };
        let obs = sample_obs(want_inner, rng);
        let obs_next = step_dynamics(&obs, 1.0);
        let z = encoder.encode(&obs);
        let z_next = encoder.encode(&obs_next);
        let r = (obs[0] * obs[0] + obs[1] * obs[1]).sqrt();
        out.push(WmTransition {
            obs,
            obs_next,
            z,
            z_next,
            r,
            regime_inner: r < WM_INNER_RADIUS,
        });
    }
    out
}

fn sample_obs(want_inner: bool, rng: &mut StdRng) -> Vec<f32> {
    for _ in 0..64 {
        let ang = rng.gen_range(0.0..std::f32::consts::TAU);
        let rad = if want_inner {
            rng.gen_range(0.05..WM_INNER_RADIUS * 0.95)
        } else {
            rng.gen_range(WM_INNER_RADIUS * 1.05..0.95)
        };
        let x = rad * ang.cos();
        let y = rad * ang.sin();
        let vx = rng.gen_range(-0.2..0.2);
        let vy = rng.gen_range(-0.2..0.2);
        let obs = vec![x, y, vx, vy];
        let r = (x * x + y * y).sqrt();
        if (r < WM_INNER_RADIUS) == want_inner {
            return obs;
        }
    }
    if want_inner {
        vec![0.1, 0.0, 0.05, 0.0]
    } else {
        vec![0.7, 0.0, 0.05, 0.0]
    }
}

/// Stratified split by regime for composite train/held-out.
pub fn stratified_wm_split(
    data: &[WmTransition],
    train_n: usize,
    rng: &mut StdRng,
) -> (Vec<WmTransition>, Vec<WmTransition>) {
    let mut inner: Vec<_> = data.iter().filter(|t| t.regime_inner).cloned().collect();
    let mut outer: Vec<_> = data.iter().filter(|t| !t.regime_inner).cloned().collect();
    shuffle(&mut inner, rng);
    shuffle(&mut outer, rng);
    let n_in = train_n / 2;
    let n_out = train_n - n_in;
    let mut train = Vec::with_capacity(train_n);
    train.extend(inner.iter().take(n_in).cloned());
    train.extend(outer.iter().take(n_out).cloned());
    let mut heldout = Vec::new();
    heldout.extend(inner.into_iter().skip(n_in));
    heldout.extend(outer.into_iter().skip(n_out));
    shuffle(&mut train, rng);
    shuffle(&mut heldout, rng);
    (train, heldout)
}

/// Run one seed of the world-model Task E protocol.
#[derive(Clone, Debug)]
pub struct WmSeedResult {
    pub train_n: usize,
    pub vg_mse: f32,
    pub conf_mse: f32,
    pub cone_mse: f32,
    pub regime_agreement: f32,
    pub entropy_bits: f32,
    pub degenerate: bool,
    pub encoder_fingerprint: u64,
}

pub fn run_wm_task_e_seed(seed: u64, train_n: usize) -> WmSeedResult {
    let encoder = FrozenJepaEncoder::new(seed);
    let pin = encoder.fingerprint;
    let mut data_rng = StdRng::seed_from_u64(seed.wrapping_mul(17).wrapping_add(3));

    // Specialist corpora (regime-pure Mirrors) + balanced composite pool.
    let mut pure_rng = StdRng::seed_from_u64(seed.wrapping_mul(31).wrapping_add(5));
    let mut inner_pure = Vec::with_capacity(180);
    let mut outer_pure = Vec::with_capacity(180);
    while inner_pure.len() < 180 || outer_pure.len() < 180 {
        let batch = generate_transitions(&encoder, 100, true, &mut pure_rng);
        for t in batch {
            if t.regime_inner && inner_pure.len() < 180 {
                inner_pure.push(t);
            } else if !t.regime_inner && outer_pure.len() < 180 {
                outer_pure.push(t);
            }
        }
    }

    let mut pred_inner = PredictorAdapter::new("inner_rot", true, pin, seed.wrapping_add(11));
    let mut pred_outer = PredictorAdapter::new("outer_rad", false, pin, seed.wrapping_add(13));
    let inner_pairs: Vec<_> = inner_pure
        .iter()
        .map(|t| (t.z.clone(), t.z_next.clone()))
        .collect();
    let outer_pairs: Vec<_> = outer_pure
        .iter()
        .map(|t| (t.z.clone(), t.z_next.clone()))
        .collect();
    let outer_z: Vec<_> = outer_pure.iter().map(|t| t.z.clone()).collect();
    let inner_z: Vec<_> = inner_pure.iter().map(|t| t.z.clone()).collect();
    pred_inner.train(&inner_pairs, &outer_z, 400, 0.15);
    pred_outer.train(&outer_pairs, &inner_z, 400, 0.15);

    let bundle = JepaPromotionBundle::promote(
        &encoder,
        vec![pred_inner.clone(), pred_outer.clone()],
    )
    .expect("promote");
    bundle.verify_encoder(&encoder).expect("pin");
    encoder.assert_pinned(pin);

    let composite = generate_transitions(&encoder, 400, true, &mut data_rng);
    let mut split_rng = StdRng::seed_from_u64(seed.wrapping_mul(131).wrapping_add(train_n as u64));
    let (train, heldout) = stratified_wm_split(&composite, train_n, &mut split_rng);

    let cone_train: Vec<ConeSample> = train
        .iter()
        .map(|t| {
            let a = pred_inner.affinity(&t.z);
            let b = pred_outer.affinity(&t.z);
            ConeSample {
                features: cone_features(a, b),
                route_spiral: t.regime_inner, // "spiral" slot = inner predictor
                r: t.r,
            }
        })
        .collect();
    let cfg = ConeConfig {
        seed,
        inner_radius: WM_INNER_RADIUS,
        ..ConeConfig::default()
    };
    let router = AdjustableConeRouter::train(&cone_train, cfg);

    let mut vg_err = 0.0f32;
    let mut conf_err = 0.0f32;
    let mut cone_err = 0.0f32;
    let mut region_hits = 0usize;
    let mut route_choices = Vec::with_capacity(heldout.len());
    let n = heldout.len().max(1) as f32;

    for t in &heldout {
        let a = pred_inner.affinity(&t.z);
        let b = pred_outer.affinity(&t.z);
        let mse_a = pred_inner.prediction_mse(&t.z, &t.z_next);
        let mse_b = pred_outer.prediction_mse(&t.z, &t.z_next);
        // VirtualGroup: average of both predictions.
        let pred_a = pred_inner.predict_next(&t.z);
        let pred_b = pred_outer.predict_next(&t.z);
        let vg: Vec<f32> = pred_a
            .iter()
            .zip(pred_b.iter())
            .map(|(u, v)| 0.5 * (u + v))
            .collect();
        vg_err += mse_vec(&vg, &t.z_next);
        // Confidence argmax: pick lower MSE affinity proxy (higher affinity).
        let conf_pred = if a >= b { &pred_a } else { &pred_b };
        conf_err += mse_vec(conf_pred, &t.z_next);

        let feats = cone_features(a, b);
        let decision = router.decide(&feats);
        let blended: Vec<f32> = pred_a
            .iter()
            .zip(pred_b.iter())
            .map(|(u, v)| decision.spiral_weight * u + (1.0 - decision.spiral_weight) * v)
            .collect();
        cone_err += mse_vec(&blended, &t.z_next);

        let route_inner = decision.spiral_weight >= 0.5;
        route_choices.push(if route_inner { 0 } else { 1 });
        if route_inner == t.regime_inner {
            region_hits += 1;
        }
        let _ = (mse_a, mse_b);
    }

    let regime_agreement = region_hits as f32 / n;
    let entropy_bits = routing_entropy(&route_choices);
    let degenerate = (regime_agreement - 0.5).abs() < 0.01 || entropy_bits < 0.3;

    WmSeedResult {
        train_n,
        vg_mse: vg_err / n,
        conf_mse: conf_err / n,
        cone_mse: cone_err / n,
        regime_agreement,
        entropy_bits,
        degenerate,
        encoder_fingerprint: pin,
    }
}

fn routing_entropy(choices: &[usize]) -> f32 {
    if choices.is_empty() {
        return 0.0;
    }
    let n = choices.len() as f32;
    let mut c0 = 0usize;
    for &c in choices {
        if c == 0 {
            c0 += 1;
        }
    }
    let p0 = c0 as f32 / n;
    let p1 = 1.0 - p0;
    let h = |p: f32| {
        if p <= 1e-8 || p >= 1.0 - 1e-8 {
            0.0
        } else {
            -p * p.log2()
        }
    };
    h(p0) + h(p1)
}

fn mse_vec(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for (u, v) in a.iter().zip(b.iter()) {
        let d = u - v;
        s += d * d;
    }
    s / a.len().max(1) as f32
}

// --- tiny linear algebra helpers ---

fn rand_mat(rows: usize, cols: usize, scale: f32, rng: &mut StdRng) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|_| (0..cols).map(|_| rng.gen_range(-scale..scale)).collect())
        .collect()
}

fn zeros_mat(rows: usize, cols: usize) -> Vec<Vec<f32>> {
    vec![vec![0.0; cols]; rows]
}

fn relu_forward(x: &[f32], w: &[Vec<f32>], b: &[f32]) -> Vec<f32> {
    w.iter()
        .zip(b.iter())
        .map(|(row, &bias)| {
            let mut s = bias;
            for (j, &xj) in x.iter().enumerate() {
                s += row[j] * xj;
            }
            s.max(0.0)
        })
        .collect()
}

fn linear_forward(x: &[f32], w: &[Vec<f32>], b: &[f32]) -> Vec<f32> {
    w.iter()
        .zip(b.iter())
        .map(|(row, &bias)| {
            let mut s = bias;
            for (j, &xj) in x.iter().enumerate() {
                s += row[j] * xj;
            }
            s
        })
        .collect()
}

fn apply_mat(w: &mut [Vec<f32>], g: &[Vec<f32>], lr: f32, l2: f32) {
    for (row, grow) in w.iter_mut().zip(g.iter()) {
        for (cell, &gc) in row.iter_mut().zip(grow.iter()) {
            *cell -= lr * (gc + l2 * *cell);
        }
    }
}

fn apply_vec(v: &mut [f32], g: &[f32], lr: f32, l2: f32) {
    for (cell, &gc) in v.iter_mut().zip(g.iter()) {
        *cell -= lr * (gc + l2 * *cell);
    }
}

fn accumulate_affinity_grad(
    z: &[f32],
    y: f32,
    w1: &[Vec<f32>],
    b1: &[f32],
    w2: &[f32],
    b2: f32,
    g_w1: &mut [Vec<f32>],
    g_b1: &mut [f32],
    g_w2: &mut [f32],
    g_b2: &mut f32,
) {
    let h = relu_forward(z, w1, b1);
    let mut logit = b2;
    for (i, &a) in h.iter().enumerate() {
        logit += w2[i] * a;
    }
    let p = sigmoid(logit);
    let dlogit = p - y;
    *g_b2 += dlogit;
    for i in 0..h.len() {
        g_w2[i] += dlogit * h[i];
    }
    for j in 0..h.len() {
        let dh = if h[j] > 0.0 { dlogit * w2[j] } else { 0.0 };
        g_b1[j] += dh;
        for i in 0..z.len() {
            g_w1[j][i] += dh * z[i];
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn shuffle<T>(xs: &mut [T], rng: &mut StdRng) {
    for i in (1..xs.len()).rev() {
        let j = rng.gen_range(0..=i);
        xs.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_fingerprint_stable() {
        let e1 = FrozenJepaEncoder::new(42);
        let e2 = FrozenJepaEncoder::new(42);
        assert_eq!(e1.fingerprint, e2.fingerprint);
        let e3 = FrozenJepaEncoder::new(43);
        assert_ne!(e1.fingerprint, e3.fingerprint);
    }

    #[test]
    fn promote_rejects_pin_mismatch() {
        let enc = FrozenJepaEncoder::new(7);
        let bad = PredictorAdapter::new("x", true, enc.fingerprint ^ 1, 1);
        assert!(JepaPromotionBundle::promote(&enc, vec![bad]).is_err());
    }

    #[test]
    fn promote_accepts_matching_pin() {
        let enc = FrozenJepaEncoder::new(7);
        let p = PredictorAdapter::new("x", true, enc.fingerprint, 1);
        let bundle = JepaPromotionBundle::promote(&enc, vec![p]).unwrap();
        assert!(bundle.verify_encoder(&enc).is_ok());
    }

    #[test]
    fn wm_task_e_seed_beats_chance_regime() {
        let r = run_wm_task_e_seed(42, 30);
        assert!(
            r.regime_agreement > 0.55,
            "regime agreement too low: {}",
            r.regime_agreement
        );
        assert!(!r.degenerate, "degenerate routing");
        assert!(
            r.cone_mse <= r.vg_mse + 0.05,
            "cone mse {} should not be much worse than vg {}",
            r.cone_mse,
            r.vg_mse
        );
    }
}
