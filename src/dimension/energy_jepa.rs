//! Energy-based JEPA adapters (Phase 3j).
//!
//! Generalizes Phase 3i predictors into promotable **energy landscapes**
//! \(E_\theta(z_t, z_{t+1})\): true transitions are low-energy, contrasts are
//! high-energy. Proposal heads decode a next-latent for planning/MSE; affinity
//! heads keep oracle-free cone routing.
//!
//! Distinct from metabolic synapse `energy_budget` in the Growformer physics
//! substrate — this is Hopfield / EB-JEPA style latent energy.
//!
//! Contract: same as [`super::jepa_adapters`] — encoder frozen + hash-pinned;
//! only energy / proposal / affinity params promote.

use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use super::cone_router::{cone_features, AdjustableConeRouter, ConeConfig, ConeSample};
use super::jepa_adapters::{
    generate_transitions, stratified_wm_split, FrozenJepaEncoder, WM_INNER_RADIUS, WM_LATENT_DIM,
};

/// Feature dim for energy: `[z; z_next; z_next - z]`.
const ENERGY_IN: usize = WM_LATENT_DIM * 3;

/// Promotable energy-based world-model adapter.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnergyAdapter {
    pub name: String,
    pub regime_is_inner: bool,
    pub encoder_pin: u64,
    hidden: usize,
    // Energy MLP: ENERGY_IN → hidden → 1 (softplus output).
    e_w1: Vec<Vec<f32>>,
    e_b1: Vec<f32>,
    e_w2: Vec<f32>,
    e_b2: f32,
    // Proposal residual (same shape as Phase 3i dynamics).
    p_w1: Vec<Vec<f32>>,
    p_b1: Vec<f32>,
    p_w2: Vec<Vec<f32>>,
    p_b2: Vec<f32>,
    // Affinity: z → P(home).
    a_w1: Vec<Vec<f32>>,
    a_b1: Vec<f32>,
    a_w2: Vec<f32>,
    a_b2: f32,
}

impl EnergyAdapter {
    pub fn new(name: &str, regime_is_inner: bool, encoder_pin: u64, seed: u64) -> Self {
        let hidden = 24;
        let mut rng = StdRng::seed_from_u64(seed);
        let s_e = (1.0 / ENERGY_IN as f32).sqrt();
        let s_z = (1.0 / WM_LATENT_DIM as f32).sqrt();
        let s_h = (1.0 / hidden as f32).sqrt();
        Self {
            name: name.to_string(),
            regime_is_inner,
            encoder_pin,
            hidden,
            e_w1: rand_mat(hidden, ENERGY_IN, s_e, &mut rng),
            e_b1: vec![0.0; hidden],
            e_w2: (0..hidden).map(|_| rng.gen_range(-s_h..s_h)).collect(),
            e_b2: 0.5,
            p_w1: rand_mat(hidden, WM_LATENT_DIM, s_z, &mut rng),
            p_b1: vec![0.0; hidden],
            p_w2: rand_mat(WM_LATENT_DIM, hidden, s_h, &mut rng),
            p_b2: vec![0.0; WM_LATENT_DIM],
            a_w1: rand_mat(hidden, WM_LATENT_DIM, s_z, &mut rng),
            a_b1: vec![0.0; hidden],
            a_w2: (0..hidden).map(|_| rng.gen_range(-s_h..s_h)).collect(),
            a_b2: 0.0,
        }
    }

    fn energy_feats(z: &[f32], z_next: &[f32]) -> Vec<f32> {
        let mut f = Vec::with_capacity(ENERGY_IN);
        f.extend_from_slice(z);
        f.extend_from_slice(z_next);
        for i in 0..WM_LATENT_DIM {
            f.push(z_next[i] - z[i]);
        }
        f
    }

    /// Latent-pair energy (≥ 0 via softplus).
    pub fn energy(&self, z: &[f32], z_next: &[f32]) -> f32 {
        let x = Self::energy_feats(z, z_next);
        let h = relu_forward(&x, &self.e_w1, &self.e_b1);
        let mut logit = self.e_b2;
        for (i, &a) in h.iter().enumerate() {
            logit += self.e_w2[i] * a;
        }
        softplus(logit)
    }

    pub fn propose_next(&self, z: &[f32]) -> Vec<f32> {
        let h = relu_forward(z, &self.p_w1, &self.p_b1);
        let delta = linear_forward(&h, &self.p_w2, &self.p_b2);
        z.iter().zip(delta.iter()).map(|(a, b)| a + b).collect()
    }

    pub fn affinity(&self, z: &[f32]) -> f32 {
        let h = relu_forward(z, &self.a_w1, &self.a_b1);
        let mut logit = self.a_b2;
        for (i, &a) in h.iter().enumerate() {
            logit += self.a_w2[i] * a;
        }
        sigmoid(logit)
    }

    pub fn proposal_mse(&self, z: &[f32], z_next: &[f32]) -> f32 {
        mse_vec(&self.propose_next(z), z_next)
    }

    /// Train energy (contrastive), proposal (low energy + MSE), affinity.
    pub fn train(
        &mut self,
        pairs: &[(Vec<f32>, Vec<f32>)],
        contrast_pairs: &[(Vec<f32>, Vec<f32>)],
        contrast_z: &[Vec<f32>],
        epochs: usize,
        lr: f32,
        rng: &mut StdRng,
    ) {
        if pairs.is_empty() {
            return;
        }
        let margin = 2.5f32;
        for _ in 0..epochs {
            // --- Energy contrastive: E(true) low, E(false) ≥ E(true)+margin ---
            let mut eg_w1 = zeros_mat(self.hidden, ENERGY_IN);
            let mut eg_b1 = vec![0.0; self.hidden];
            let mut eg_w2 = vec![0.0; self.hidden];
            let mut eg_b2 = 0.0f32;
            let mut ecount = 0.0f32;

            for (z, z_next) in pairs {
                let e_pos = self.energy(z, z_next);
                // Stronger pull on positives (home transitions must be low-energy).
                accumulate_energy_grad(
                    z, z_next, e_pos, 2.0, &self.e_w1, &self.e_b1, &self.e_w2, self.e_b2,
                    &mut eg_w1, &mut eg_b1, &mut eg_w2, &mut eg_b2,
                );
                ecount += 1.0;

                // Two negatives: cross-regime transition + shuffled next in-regime.
                for neg_i in 0..2 {
                    let (zn, zn_next) = if neg_i == 0 && !contrast_pairs.is_empty() {
                        let j = rng.gen_range(0..contrast_pairs.len());
                        (&contrast_pairs[j].0, &contrast_pairs[j].1)
                    } else {
                        let j = rng.gen_range(0..pairs.len());
                        (z, &pairs[j].1)
                    };
                    let e_neg = self.energy(zn, zn_next);
                    let gap = e_neg - e_pos;
                    if gap < margin {
                        accumulate_energy_grad(
                            zn, zn_next, e_neg, -2.0, &self.e_w1, &self.e_b1, &self.e_w2,
                            self.e_b2, &mut eg_w1, &mut eg_b1, &mut eg_w2, &mut eg_b2,
                        );
                        accumulate_energy_grad(
                            z, z_next, e_pos, 1.0, &self.e_w1, &self.e_b1, &self.e_w2, self.e_b2,
                            &mut eg_w1, &mut eg_b1, &mut eg_w2, &mut eg_b2,
                        );
                        ecount += 1.0;
                    }
                }
            }
            if ecount > 0.0 {
                let s = lr / ecount;
                apply_mat(&mut self.e_w1, &eg_w1, s, 1e-4);
                apply_vec(&mut self.e_b1, &eg_b1, s, 1e-4);
                apply_vec(&mut self.e_w2, &eg_w2, s, 1e-4);
                self.e_b2 -= s * eg_b2 + 1e-4 * self.e_b2;
            }

            // --- Proposal: MSE to true next + keep E(z, propose(z)) small ---
            let mut pg_w1 = zeros_mat(self.hidden, WM_LATENT_DIM);
            let mut pg_b1 = vec![0.0; self.hidden];
            let mut pg_w2 = zeros_mat(WM_LATENT_DIM, self.hidden);
            let mut pg_b2 = vec![0.0; WM_LATENT_DIM];
            let n = pairs.len() as f32;
            for (z, z_next) in pairs {
                let h = relu_forward(z, &self.p_w1, &self.p_b1);
                let delta = linear_forward(&h, &self.p_w2, &self.p_b2);
                let pred: Vec<f32> = z.iter().zip(delta.iter()).map(|(a, b)| a + b).collect();
                let mut d_pred = vec![0.0f32; WM_LATENT_DIM];
                for i in 0..WM_LATENT_DIM {
                    d_pred[i] = 2.0 * (pred[i] - z_next[i]) / (WM_LATENT_DIM as f32 * n);
                }
                // Energy pull: approximate ∂E/∂z' ≈ (pred - z_next) direction already;
                // add small residual toward lowering energy of proposal.
                let e_prop = self.energy(z, &pred);
                let e_true = self.energy(z, z_next);
                if e_prop > e_true {
                    for i in 0..WM_LATENT_DIM {
                        d_pred[i] += 0.1 * (pred[i] - z_next[i]) / n;
                    }
                }
                for o in 0..WM_LATENT_DIM {
                    pg_b2[o] += d_pred[o];
                    for j in 0..self.hidden {
                        pg_w2[o][j] += d_pred[o] * h[j];
                    }
                }
                let mut d_h = vec![0.0f32; self.hidden];
                for j in 0..self.hidden {
                    let mut s = 0.0f32;
                    for o in 0..WM_LATENT_DIM {
                        s += self.p_w2[o][j] * d_pred[o];
                    }
                    d_h[j] = if h[j] > 0.0 { s } else { 0.0 };
                }
                for j in 0..self.hidden {
                    pg_b1[j] += d_h[j];
                    for i in 0..WM_LATENT_DIM {
                        pg_w1[j][i] += d_h[j] * z[i];
                    }
                }
            }
            apply_mat(&mut self.p_w1, &pg_w1, lr, 1e-4);
            apply_vec(&mut self.p_b1, &pg_b1, lr, 1e-4);
            apply_mat(&mut self.p_w2, &pg_w2, lr, 1e-4);
            apply_vec(&mut self.p_b2, &pg_b2, lr, 1e-4);

            // --- Affinity ---
            let mut ag_w1 = zeros_mat(self.hidden, WM_LATENT_DIM);
            let mut ag_b1 = vec![0.0; self.hidden];
            let mut ag_w2 = vec![0.0; self.hidden];
            let mut ag_b2 = 0.0f32;
            let mut ac = 0.0f32;
            for (z, _) in pairs {
                accumulate_affinity_grad(
                    z, 1.0, &self.a_w1, &self.a_b1, &self.a_w2, self.a_b2, &mut ag_w1, &mut ag_b1,
                    &mut ag_w2, &mut ag_b2,
                );
                ac += 1.0;
            }
            for z in contrast_z {
                accumulate_affinity_grad(
                    z, 0.0, &self.a_w1, &self.a_b1, &self.a_w2, self.a_b2, &mut ag_w1, &mut ag_b1,
                    &mut ag_w2, &mut ag_b2,
                );
                ac += 1.0;
            }
            if ac > 0.0 {
                let s = lr / ac;
                apply_mat(&mut self.a_w1, &ag_w1, s, 1e-4);
                apply_vec(&mut self.a_b1, &ag_b1, s, 1e-4);
                apply_vec(&mut self.a_w2, &ag_w2, s, 1e-4);
                self.a_b2 -= s * ag_b2 + 1e-4 * self.a_b2;
            }
        }
    }
}

/// Promotion bundle for energy adapters (encoder pin required).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnergyPromotionBundle {
    pub encoder_fingerprint: u64,
    pub adapters: Vec<EnergyAdapter>,
}

impl EnergyPromotionBundle {
    pub fn promote(
        encoder: &FrozenJepaEncoder,
        adapters: Vec<EnergyAdapter>,
    ) -> Result<Self, String> {
        for a in &adapters {
            if a.encoder_pin != encoder.fingerprint {
                return Err(format!(
                    "energy adapter '{}' pin {:#x} ≠ encoder {:#x}",
                    a.name, a.encoder_pin, encoder.fingerprint
                ));
            }
        }
        Ok(Self {
            encoder_fingerprint: encoder.fingerprint,
            adapters,
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

/// One seed of Phase 3j energy-WM Task E.
#[derive(Clone, Debug)]
pub struct EnergyWmSeedResult {
    pub train_n: usize,
    pub vg_mse: f32,
    pub conf_mse: f32,
    pub cone_mse: f32,
    pub vg_energy: f32,
    pub cone_energy: f32,
    pub energy_margin: f32,
    pub regime_agreement: f32,
    pub entropy_bits: f32,
    pub degenerate: bool,
    pub encoder_fingerprint: u64,
}

pub fn run_energy_wm_task_e_seed(seed: u64, train_n: usize) -> EnergyWmSeedResult {
    let encoder = FrozenJepaEncoder::new(seed);
    let pin = encoder.fingerprint;
    let mut data_rng = StdRng::seed_from_u64(seed.wrapping_mul(19).wrapping_add(7));
    let mut pure_rng = StdRng::seed_from_u64(seed.wrapping_mul(41).wrapping_add(9));

    let mut inner_pure = Vec::with_capacity(180);
    let mut outer_pure = Vec::with_capacity(180);
    while inner_pure.len() < 180 || outer_pure.len() < 180 {
        for t in generate_transitions(&encoder, 100, true, &mut pure_rng) {
            if t.regime_inner && inner_pure.len() < 180 {
                inner_pure.push(t);
            } else if !t.regime_inner && outer_pure.len() < 180 {
                outer_pure.push(t);
            }
        }
    }

    let mut e_inner = EnergyAdapter::new("inner_energy", true, pin, seed.wrapping_add(21));
    let mut e_outer = EnergyAdapter::new("outer_energy", false, pin, seed.wrapping_add(23));
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

    let mut train_rng = StdRng::seed_from_u64(seed.wrapping_add(99));
    e_inner.train(
        &inner_pairs,
        &outer_pairs,
        &outer_z,
        500,
        0.15,
        &mut train_rng,
    );
    e_outer.train(
        &outer_pairs,
        &inner_pairs,
        &inner_z,
        500,
        0.15,
        &mut train_rng,
    );

    let bundle = EnergyPromotionBundle::promote(&encoder, vec![e_inner.clone(), e_outer.clone()])
        .expect("promote");
    bundle.verify_encoder(&encoder).expect("pin");
    encoder.assert_pinned(pin);

    let composite = generate_transitions(&encoder, 400, true, &mut data_rng);
    let mut split_rng = StdRng::seed_from_u64(seed.wrapping_mul(131).wrapping_add(train_n as u64));
    let (train, heldout) = stratified_wm_split(&composite, train_n, &mut split_rng);

    let cone_train: Vec<ConeSample> = train
        .iter()
        .map(|t| ConeSample {
            features: cone_features(e_inner.affinity(&t.z), e_outer.affinity(&t.z)),
            route_spiral: t.regime_inner,
            r: t.r,
        })
        .collect();
    let router = AdjustableConeRouter::train(
        &cone_train,
        ConeConfig {
            seed,
            inner_radius: WM_INNER_RADIUS,
            ..ConeConfig::default()
        },
    );

    let mut vg_err = 0.0f32;
    let mut conf_err = 0.0f32;
    let mut cone_err = 0.0f32;
    let mut vg_e = 0.0f32;
    let mut cone_e = 0.0f32;
    let mut margin_sum = 0.0f32;
    let mut region_hits = 0usize;
    let mut route_choices = Vec::with_capacity(heldout.len());
    let n = heldout.len().max(1) as f32;

    for t in &heldout {
        let a = e_inner.affinity(&t.z);
        let b = e_outer.affinity(&t.z);
        let pred_a = e_inner.propose_next(&t.z);
        let pred_b = e_outer.propose_next(&t.z);
        let vg: Vec<f32> = pred_a
            .iter()
            .zip(pred_b.iter())
            .map(|(u, v)| 0.5 * (u + v))
            .collect();
        vg_err += mse_vec(&vg, &t.z_next);
        let conf_pred = if a >= b { &pred_a } else { &pred_b };
        conf_err += mse_vec(conf_pred, &t.z_next);

        let decision = router.decide(&cone_features(a, b));
        let blended: Vec<f32> = pred_a
            .iter()
            .zip(pred_b.iter())
            .map(|(u, v)| decision.spiral_weight * u + (1.0 - decision.spiral_weight) * v)
            .collect();
        cone_err += mse_vec(&blended, &t.z_next);

        // Energy of true transition under each landscape; cone picks by route weight.
        let e_a = e_inner.energy(&t.z, &t.z_next);
        let e_b = e_outer.energy(&t.z, &t.z_next);
        vg_e += 0.5 * (e_a + e_b);
        cone_e += decision.spiral_weight * e_a + (1.0 - decision.spiral_weight) * e_b;

        let home_e = if t.regime_inner { e_a } else { e_b };
        let away_e = if t.regime_inner { e_b } else { e_a };
        margin_sum += away_e - home_e;

        let route_inner = decision.spiral_weight >= 0.5;
        route_choices.push(if route_inner { 0 } else { 1 });
        if route_inner == t.regime_inner {
            region_hits += 1;
        }
    }

    let regime_agreement = region_hits as f32 / n;
    let entropy_bits = routing_entropy(&route_choices);
    let degenerate = (regime_agreement - 0.5).abs() < 0.01 || entropy_bits < 0.3;

    EnergyWmSeedResult {
        train_n,
        vg_mse: vg_err / n,
        conf_mse: conf_err / n,
        cone_mse: cone_err / n,
        vg_energy: vg_e / n,
        cone_energy: cone_e / n,
        energy_margin: margin_sum / n,
        regime_agreement,
        entropy_bits,
        degenerate,
        encoder_fingerprint: pin,
    }
}

// --- helpers (local; keep energy module self-contained) ---

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn softplus_grad_from_output(e: f32) -> f32 {
    // e = softplus(logit) ⇒ sigmoid(logit) = 1 - exp(-e) for e>0
    1.0 - (-e).exp()
}

fn accumulate_energy_grad(
    z: &[f32],
    z_next: &[f32],
    energy_val: f32,
    d_energy: f32,
    w1: &[Vec<f32>],
    b1: &[f32],
    w2: &[f32],
    b2: f32,
    g_w1: &mut [Vec<f32>],
    g_b1: &mut [f32],
    g_w2: &mut [f32],
    g_b2: &mut f32,
) {
    let x = EnergyAdapter::energy_feats(z, z_next);
    let h = relu_forward(&x, w1, b1);
    let mut logit = b2;
    for (i, &a) in h.iter().enumerate() {
        logit += w2[i] * a;
    }
    let _ = logit;
    let dlogit = d_energy * softplus_grad_from_output(energy_val);
    *g_b2 += dlogit;
    for i in 0..h.len() {
        g_w2[i] += dlogit * h[i];
    }
    for j in 0..h.len() {
        let dh = if h[j] > 0.0 { dlogit * w2[j] } else { 0.0 };
        g_b1[j] += dh;
        for i in 0..x.len() {
            g_w1[j][i] += dh * x[i];
        }
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

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn mse_vec(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for (u, v) in a.iter().zip(b.iter()) {
        let d = u - v;
        s += d * d;
    }
    s / a.len().max(1) as f32
}

fn routing_entropy(choices: &[usize]) -> f32 {
    if choices.is_empty() {
        return 0.0;
    }
    let n = choices.len() as f32;
    let c0 = choices.iter().filter(|&&c| c == 0).count() as f32 / n;
    let c1 = 1.0 - c0;
    let h = |p: f32| {
        if p <= 1e-8 || p >= 1.0 - 1e-8 {
            0.0
        } else {
            -p * p.log2()
        }
    };
    h(c0) + h(c1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn energy_promote_pin() {
        let enc = FrozenJepaEncoder::new(3);
        let bad = EnergyAdapter::new("x", true, enc.fingerprint ^ 1, 1);
        assert!(EnergyPromotionBundle::promote(&enc, vec![bad]).is_err());
        let good = EnergyAdapter::new("x", true, enc.fingerprint, 1);
        assert!(EnergyPromotionBundle::promote(&enc, vec![good]).is_ok());
    }

    #[test]
    fn energy_wm_seed_sane() {
        let r = run_energy_wm_task_e_seed(42, 30);
        assert!(
            r.regime_agreement > 0.55,
            "regime agree {}",
            r.regime_agreement
        );
        assert!(!r.degenerate);
        assert!(
            r.energy_margin > 0.0,
            "home energy should beat away: margin={}",
            r.energy_margin
        );
        assert!(r.cone_mse <= r.vg_mse + 0.05);
    }
}
