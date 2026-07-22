//! Phases 3k / 3ℓ / 3m — successors on the energy-JEPA substrate.
//!
//! - **3k Geometric:** frozen Clifford grade-1 latent manifold; energy still scores pairs.
//! - **3ℓ Probabilistic:** temperature + ensemble over promoted energy heads; abstain.
//! - **3m Neuro-symbolic:** typed rule penalties constrain \(E\) (RiJEPA-style), not a second reasoner.
//!
//! All keep [JEPA_ADAPTER_PROMOTION](../../docs/JEPA_ADAPTER_PROMOTION.md): encoder pinned;
//! only adapter params promote.

use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::clifford::{
    embed_bridge_vector, minkowski_interval, Multivector, CL8_VECTOR_DIM, GRADE_OFFSETS,
};

use super::cone_router::{cone_features, AdjustableConeRouter, ConeConfig, ConeSample};
use super::energy_jepa::{EnergyAdapter, EnergyPromotionBundle};
use super::jepa_adapters::{
    generate_transitions, step_dynamics, stratified_wm_split, FrozenJepaEncoder, WM_INNER_RADIUS,
    WM_LATENT_DIM, WM_OBS_DIM,
};

// =============================================================================
// 3k — Geometric (Clifford) encoder
// =============================================================================

/// Frozen encoder: obs → Cl(1,7) grade-1 vector (8D = [`WM_LATENT_DIM`]).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrozenGeometricEncoder {
    /// Linear lift obs→8 before Clifford embed (frozen).
    pub w: Vec<Vec<f32>>,
    pub b: Vec<f32>,
    pub fingerprint: u64,
}

impl FrozenGeometricEncoder {
    pub fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0xC2B2_AE3D).wrapping_add(11));
        let scale = (2.0 / WM_OBS_DIM as f32).sqrt();
        let w: Vec<Vec<f32>> = (0..CL8_VECTOR_DIM)
            .map(|_| {
                (0..WM_OBS_DIM)
                    .map(|_| rng.gen_range(-scale..scale))
                    .collect()
            })
            .collect();
        let b: Vec<f32> = (0..CL8_VECTOR_DIM)
            .map(|_| rng.gen_range(-0.05..0.05))
            .collect();
        let fingerprint = {
            let mut h = DefaultHasher::new();
            for row in &w {
                for &v in row {
                    v.to_bits().hash(&mut h);
                }
            }
            for &v in &b {
                v.to_bits().hash(&mut h);
            }
            h.finish()
        };
        Self { w, b, fingerprint }
    }

    pub fn encode(&self, obs: &[f32]) -> Vec<f32> {
        assert_eq!(obs.len(), WM_OBS_DIM);
        let mut lifted = vec![0.0f32; CL8_VECTOR_DIM];
        for (i, row) in self.w.iter().enumerate() {
            let mut s = self.b[i];
            for (j, &x) in obs.iter().enumerate() {
                s += row[j] * x;
            }
            lifted[i] = s.tanh();
        }
        // Enrich with polar cues in unused capacity (already 8); blend r into e0-ish slot.
        let r = (obs[0] * obs[0] + obs[1] * obs[1]).sqrt();
        lifted[0] = 0.7 * lifted[0] + 0.3 * (2.0 * r - 1.0).tanh();
        let mv = embed_bridge_vector(&lifted);
        let start = GRADE_OFFSETS[1];
        mv.components[start..start + CL8_VECTOR_DIM].to_vec()
    }

    pub fn assert_pinned(&self, expected: u64) {
        assert_eq!(self.fingerprint, expected, "geometric encoder pin drift");
    }
}

/// Minkowski interval between two grade-1 latents.
pub fn latent_interval(z: &[f32], z_next: &[f32]) -> f32 {
    let mut a = [0.0f32; 8];
    let mut b = [0.0f32; 8];
    for i in 0..8.min(z.len()) {
        a[i] = z[i];
    }
    for i in 0..8.min(z_next.len()) {
        b[i] = z_next[i];
    }
    minkowski_interval(&Multivector::vector(&a), &Multivector::vector(&b))
}

/// Energy with geometric regularizer: neural E + λ·regime-interval mismatch.
pub fn geometric_energy(adapter: &EnergyAdapter, z: &[f32], z_next: &[f32]) -> f32 {
    let e = adapter.energy(z, z_next);
    let s2 = latent_interval(z, z_next);
    // Inner (rotation) ↔ more timelike/near-null; outer (radial) ↔ spacelike-ish.
    let geo = if adapter.regime_is_inner {
        (s2).max(0.0) // penalize spacelike for inner
    } else {
        (-s2).max(0.0) // penalize timelike for outer
    };
    e + 0.15 * geo
}

fn transitions_geometric(
    enc: &FrozenGeometricEncoder,
    n: usize,
    balanced: bool,
    rng: &mut StdRng,
) -> Vec<GeoTransition> {
    let mut out = Vec::with_capacity(n);
    let half = n / 2;
    for i in 0..n {
        let want_inner = if balanced { i < half } else { rng.gen() };
        let obs = sample_obs(want_inner, rng);
        let obs_next = step_dynamics(&obs, 1.0);
        let z = enc.encode(&obs);
        let z_next = enc.encode(&obs_next);
        let r = (obs[0] * obs[0] + obs[1] * obs[1]).sqrt();
        out.push(GeoTransition {
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

#[derive(Clone, Debug)]
struct GeoTransition {
    obs: Vec<f32>,
    obs_next: Vec<f32>,
    z: Vec<f32>,
    z_next: Vec<f32>,
    r: f32,
    regime_inner: bool,
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
        let r = (x * x + y * y).sqrt();
        if (r < WM_INNER_RADIUS) == want_inner {
            return vec![x, y, vx, vy];
        }
    }
    if want_inner {
        vec![0.1, 0.0, 0.05, 0.0]
    } else {
        vec![0.7, 0.0, 0.05, 0.0]
    }
}

fn stratified_geo(
    data: &[GeoTransition],
    train_n: usize,
    rng: &mut StdRng,
) -> (Vec<GeoTransition>, Vec<GeoTransition>) {
    let mut inner: Vec<_> = data.iter().filter(|t| t.regime_inner).cloned().collect();
    let mut outer: Vec<_> = data.iter().filter(|t| !t.regime_inner).cloned().collect();
    shuffle(&mut inner, rng);
    shuffle(&mut outer, rng);
    let n_in = train_n / 2;
    let n_out = train_n - n_in;
    let mut train = Vec::new();
    train.extend(inner.iter().take(n_in).cloned());
    train.extend(outer.iter().take(n_out).cloned());
    let mut held = Vec::new();
    held.extend(inner.into_iter().skip(n_in));
    held.extend(outer.into_iter().skip(n_out));
    shuffle(&mut train, rng);
    shuffle(&mut held, rng);
    (train, held)
}

#[derive(Clone, Debug)]
pub struct GeoWmSeedResult {
    pub train_n: usize,
    pub vg_mse: f32,
    pub conf_mse: f32,
    pub cone_mse: f32,
    pub energy_margin: f32,
    pub regime_agreement: f32,
    pub geo_sign_agree: f32,
    pub degenerate: bool,
    pub encoder_fingerprint: u64,
}

pub fn run_phase3k_geo_seed(seed: u64, train_n: usize) -> GeoWmSeedResult {
    let enc = FrozenGeometricEncoder::new(seed);
    let pin = enc.fingerprint;
    debug_assert_eq!(WM_LATENT_DIM, CL8_VECTOR_DIM);

    let mut pure_rng = StdRng::seed_from_u64(seed.wrapping_mul(7).wrapping_add(3));
    let (inner_pure, outer_pure) = collect_geo_pure(&enc, 180, &mut pure_rng);

    let mut e_in = EnergyAdapter::new("geo_inner", true, pin, seed.wrapping_add(31));
    let mut e_out = EnergyAdapter::new("geo_outer", false, pin, seed.wrapping_add(33));
    let in_pairs: Vec<_> = inner_pure
        .iter()
        .map(|t| (t.z.clone(), t.z_next.clone()))
        .collect();
    let out_pairs: Vec<_> = outer_pure
        .iter()
        .map(|t| (t.z.clone(), t.z_next.clone()))
        .collect();
    let out_z: Vec<_> = outer_pure.iter().map(|t| t.z.clone()).collect();
    let in_z: Vec<_> = inner_pure.iter().map(|t| t.z.clone()).collect();
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(101));
    e_in.train(&in_pairs, &out_pairs, &out_z, 450, 0.15, &mut rng);
    e_out.train(&out_pairs, &in_pairs, &in_z, 450, 0.15, &mut rng);

    enc.assert_pinned(pin);
    for a in [&e_in, &e_out] {
        assert_eq!(
            a.encoder_pin, pin,
            "geometric encoder pin must match promoted energy adapters"
        );
    }

    let mut data_rng = StdRng::seed_from_u64(seed.wrapping_mul(13).wrapping_add(5));
    let composite = transitions_geometric(&enc, 400, true, &mut data_rng);
    let mut split_rng = StdRng::seed_from_u64(seed.wrapping_mul(131).wrapping_add(train_n as u64));
    let (train, heldout) = stratified_geo(&composite, train_n, &mut split_rng);

    let cone_train: Vec<ConeSample> = train
        .iter()
        .map(|t| ConeSample {
            features: cone_features(e_in.affinity(&t.z), e_out.affinity(&t.z)),
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

    let mut vg_err = 0.0;
    let mut conf_err = 0.0;
    let mut cone_err = 0.0;
    let mut margin = 0.0;
    let mut region_hits = 0usize;
    let mut geo_hits = 0usize;
    let mut routes = Vec::new();
    let n = heldout.len().max(1) as f32;

    for t in &heldout {
        let a = e_in.affinity(&t.z);
        let b = e_out.affinity(&t.z);
        let pa = e_in.propose_next(&t.z);
        let pb = e_out.propose_next(&t.z);
        let vg: Vec<f32> = pa.iter().zip(pb.iter()).map(|(u, v)| 0.5 * (u + v)).collect();
        vg_err += mse(&vg, &t.z_next);
        conf_err += mse(if a >= b { &pa } else { &pb }, &t.z_next);
        let d = router.decide(&cone_features(a, b));
        let blend: Vec<f32> = pa
            .iter()
            .zip(pb.iter())
            .map(|(u, v)| d.spiral_weight * u + (1.0 - d.spiral_weight) * v)
            .collect();
        cone_err += mse(&blend, &t.z_next);

        let ea = geometric_energy(&e_in, &t.z, &t.z_next);
        let eb = geometric_energy(&e_out, &t.z, &t.z_next);
        let home = if t.regime_inner { ea } else { eb };
        let away = if t.regime_inner { eb } else { ea };
        margin += away - home;

        let route_inner = d.spiral_weight >= 0.5;
        routes.push(if route_inner { 0 } else { 1 });
        if route_inner == t.regime_inner {
            region_hits += 1;
        }
        // Geometric energy prefers the home landscape (interval regularizer active).
        if home < away {
            geo_hits += 1;
        }
        let _ = latent_interval(&t.z, &t.z_next);
    }

    let regime_agreement = region_hits as f32 / n;
    let entropy = routing_entropy(&routes);
    GeoWmSeedResult {
        train_n,
        vg_mse: vg_err / n,
        conf_mse: conf_err / n,
        cone_mse: cone_err / n,
        energy_margin: margin / n,
        regime_agreement,
        geo_sign_agree: geo_hits as f32 / n,
        degenerate: (regime_agreement - 0.5).abs() < 0.01 || entropy < 0.3,
        encoder_fingerprint: pin,
    }
}

fn collect_geo_pure(
    enc: &FrozenGeometricEncoder,
    n: usize,
    rng: &mut StdRng,
) -> (Vec<GeoTransition>, Vec<GeoTransition>) {
    let mut inner = Vec::with_capacity(n);
    let mut outer = Vec::with_capacity(n);
    while inner.len() < n || outer.len() < n {
        for t in transitions_geometric(enc, 80, true, rng) {
            if t.regime_inner && inner.len() < n {
                inner.push(t);
            } else if !t.regime_inner && outer.len() < n {
                outer.push(t);
            }
        }
    }
    (inner, outer)
}

// =============================================================================
// 3ℓ — Probabilistic ensemble + temperature
// =============================================================================

#[derive(Clone, Debug)]
pub struct ProbWmSeedResult {
    pub train_n: usize,
    pub vg_mse: f32,
    pub conf_mse: f32,
    pub cone_mse: f32,
    pub energy_margin: f32,
    pub regime_agreement: f32,
    pub abstain_rate: f32,
    pub abstain_annulus_frac: f32,
    pub degenerate: bool,
    pub encoder_fingerprint: u64,
}

const ENSEMBLE_K: usize = 3;
const ABSTAIN_TAU: f32 = 0.12;

pub fn run_phase3l_prob_seed(seed: u64, train_n: usize) -> ProbWmSeedResult {
    let encoder = FrozenJepaEncoder::new(seed);
    let pin = encoder.fingerprint;
    let mut pure_rng = StdRng::seed_from_u64(seed.wrapping_mul(23).wrapping_add(2));
    let (inner_pure, outer_pure) = collect_jepa_pure(&encoder, 160, &mut pure_rng);

    let in_pairs: Vec<_> = inner_pure
        .iter()
        .map(|t| (t.z.clone(), t.z_next.clone()))
        .collect();
    let out_pairs: Vec<_> = outer_pure
        .iter()
        .map(|t| (t.z.clone(), t.z_next.clone()))
        .collect();
    let out_z: Vec<_> = outer_pure.iter().map(|t| t.z.clone()).collect();
    let in_z: Vec<_> = inner_pure.iter().map(|t| t.z.clone()).collect();

    let mut ens_in = Vec::with_capacity(ENSEMBLE_K);
    let mut ens_out = Vec::with_capacity(ENSEMBLE_K);
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(200));
    for k in 0..ENSEMBLE_K {
        let mut a = EnergyAdapter::new(
            &format!("prob_in_{k}"),
            true,
            pin,
            seed.wrapping_add(40 + k as u64),
        );
        let mut b = EnergyAdapter::new(
            &format!("prob_out_{k}"),
            false,
            pin,
            seed.wrapping_add(60 + k as u64),
        );
        a.train(&in_pairs, &out_pairs, &out_z, 320, 0.14, &mut rng);
        b.train(&out_pairs, &in_pairs, &in_z, 320, 0.14, &mut rng);
        ens_in.push(a);
        ens_out.push(b);
    }
    EnergyPromotionBundle::promote(&encoder, ens_in.iter().chain(ens_out.iter()).cloned().collect())
        .expect("promote");

    let temperature = 0.75f32;
    let mut data_rng = StdRng::seed_from_u64(seed.wrapping_mul(29).wrapping_add(4));
    let composite = generate_transitions(&encoder, 400, true, &mut data_rng);
    let mut split_rng = StdRng::seed_from_u64(seed.wrapping_mul(131).wrapping_add(train_n as u64));
    let (train, heldout) = stratified_wm_split(&composite, train_n, &mut split_rng);

    let aff = |ens: &[EnergyAdapter], z: &[f32]| -> f32 {
        ens.iter().map(|a| a.affinity(z)).sum::<f32>() / ens.len() as f32
    };
    let ener = |ens: &[EnergyAdapter], z: &[f32], zn: &[f32]| -> f32 {
        ens.iter().map(|a| a.energy(z, zn)).sum::<f32>() / ens.len() as f32
    };
    let propose = |ens: &[EnergyAdapter], z: &[f32]| -> Vec<f32> {
        let mut acc = vec![0.0; WM_LATENT_DIM];
        for a in ens {
            let p = a.propose_next(z);
            for i in 0..WM_LATENT_DIM {
                acc[i] += p[i];
            }
        }
        for v in &mut acc {
            *v /= ens.len() as f32;
        }
        acc
    };

    let cone_train: Vec<ConeSample> = train
        .iter()
        .map(|t| ConeSample {
            features: cone_features(aff(&ens_in, &t.z), aff(&ens_out, &t.z)),
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

    let mut vg_err = 0.0;
    let mut conf_err = 0.0;
    let mut cone_err = 0.0;
    let mut margin = 0.0;
    let mut region_hits = 0usize;
    let mut abstain_n = 0usize;
    let mut abstain_annulus = 0usize;
    let mut routes = Vec::new();
    let n = heldout.len().max(1) as f32;
    let annulus_eps = 0.08f32;

    for t in &heldout {
        let a0 = aff(&ens_in, &t.z);
        let b0 = aff(&ens_out, &t.z);
        // Temperature soft affinities for reporting / abstain.
        let ea = (-a0 / temperature).exp();
        let eb = (-b0 / temperature).exp(); // use affinity as logit-ish via temp on gap
        let _ = (ea, eb);
        let gap = (a0 - b0).abs();
        let abstain = gap < ABSTAIN_TAU;

        let pa = propose(&ens_in, &t.z);
        let pb = propose(&ens_out, &t.z);
        let vg: Vec<f32> = pa.iter().zip(pb.iter()).map(|(u, v)| 0.5 * (u + v)).collect();
        vg_err += mse(&vg, &t.z_next);
        conf_err += mse(if a0 >= b0 { &pa } else { &pb }, &t.z_next);

        let d = router.decide(&cone_features(a0, b0));
        let mut w = d.spiral_weight;
        if abstain {
            // Wide cone: blend toward 0.5 under uncertainty.
            w = 0.5 * w + 0.5 * 0.5;
            abstain_n += 1;
            if (t.r - WM_INNER_RADIUS).abs() < annulus_eps {
                abstain_annulus += 1;
            }
        }
        let blend: Vec<f32> = pa
            .iter()
            .zip(pb.iter())
            .map(|(u, v)| w * u + (1.0 - w) * v)
            .collect();
        cone_err += mse(&blend, &t.z_next);

        let e_in = ener(&ens_in, &t.z, &t.z_next);
        let e_out = ener(&ens_out, &t.z, &t.z_next);
        let home = if t.regime_inner { e_in } else { e_out };
        let away = if t.regime_inner { e_out } else { e_in };
        margin += away - home;

        let route_inner = w >= 0.5;
        routes.push(if route_inner { 0 } else { 1 });
        if route_inner == t.regime_inner {
            region_hits += 1;
        }
    }

    let regime_agreement = region_hits as f32 / n;
    let entropy = routing_entropy(&routes);
    let abstain_rate = abstain_n as f32 / n;
    let abstain_annulus_frac = if abstain_n > 0 {
        abstain_annulus as f32 / abstain_n as f32
    } else {
        0.0
    };

    ProbWmSeedResult {
        train_n,
        vg_mse: vg_err / n,
        conf_mse: conf_err / n,
        cone_mse: cone_err / n,
        energy_margin: margin / n,
        regime_agreement,
        abstain_rate,
        abstain_annulus_frac,
        degenerate: (regime_agreement - 0.5).abs() < 0.01 || entropy < 0.3,
        encoder_fingerprint: pin,
    }
}

fn collect_jepa_pure(
    enc: &FrozenJepaEncoder,
    n: usize,
    rng: &mut StdRng,
) -> (
    Vec<super::jepa_adapters::WmTransition>,
    Vec<super::jepa_adapters::WmTransition>,
) {
    let mut inner = Vec::with_capacity(n);
    let mut outer = Vec::with_capacity(n);
    while inner.len() < n || outer.len() < n {
        for t in generate_transitions(enc, 80, true, rng) {
            if t.regime_inner && inner.len() < n {
                inner.push(t);
            } else if !t.regime_inner && outer.len() < n {
                outer.push(t);
            }
        }
    }
    (inner, outer)
}

// =============================================================================
// 3m — Neuro-symbolic rule constraints on energy
// =============================================================================

/// Typed rules over observations (RiJEPA-style constraints on E).
#[derive(Clone, Copy, Debug)]
pub enum WorldRule {
    /// Inner disk: angular motion should dominate radial drift.
    InnerRotationDominates,
    /// Outer annulus: radius should expand (radial push).
    OuterRadialExpand,
}

impl WorldRule {
    pub fn penalty(&self, obs: &[f32], obs_next: &[f32]) -> f32 {
        let r0 = (obs[0] * obs[0] + obs[1] * obs[1]).sqrt();
        let r1 = (obs_next[0] * obs_next[0] + obs_next[1] * obs_next[1]).sqrt();
        let ang0 = obs[1].atan2(obs[0]);
        let ang1 = obs_next[1].atan2(obs_next[0]);
        let mut dang = ang1 - ang0;
        while dang > std::f32::consts::PI {
            dang -= std::f32::consts::TAU;
        }
        while dang < -std::f32::consts::PI {
            dang += std::f32::consts::TAU;
        }
        let dr = (r1 - r0).abs();
        match self {
            WorldRule::InnerRotationDominates => (dr - dang.abs()).max(0.0),
            WorldRule::OuterRadialExpand => (0.02 - (r1 - r0)).max(0.0),
        }
    }
}

pub fn symbolic_energy(
    adapter: &EnergyAdapter,
    z: &[f32],
    z_next: &[f32],
    obs: &[f32],
    obs_next: &[f32],
    lambda: f32,
) -> f32 {
    let mut e = adapter.energy(z, z_next);
    if adapter.regime_is_inner {
        e += lambda * WorldRule::InnerRotationDominates.penalty(obs, obs_next);
    } else {
        e += lambda * WorldRule::OuterRadialExpand.penalty(obs, obs_next);
    }
    e
}

#[derive(Clone, Debug)]
pub struct SymWmSeedResult {
    pub train_n: usize,
    pub vg_mse: f32,
    pub conf_mse: f32,
    pub cone_mse: f32,
    pub energy_margin: f32,
    pub regime_agreement: f32,
    pub rule_violation_rate: f32,
    pub degenerate: bool,
    pub encoder_fingerprint: u64,
}

pub fn run_phase3m_sym_seed(seed: u64, train_n: usize) -> SymWmSeedResult {
    let encoder = FrozenJepaEncoder::new(seed);
    let pin = encoder.fingerprint;
    let mut pure_rng = StdRng::seed_from_u64(seed.wrapping_mul(37).wrapping_add(6));
    let (inner_pure, outer_pure) = collect_jepa_pure(&encoder, 180, &mut pure_rng);

    let in_pairs: Vec<_> = inner_pure
        .iter()
        .map(|t| (t.z.clone(), t.z_next.clone()))
        .collect();
    let out_pairs: Vec<_> = outer_pure
        .iter()
        .map(|t| (t.z.clone(), t.z_next.clone()))
        .collect();
    let out_z: Vec<_> = outer_pure.iter().map(|t| t.z.clone()).collect();
    let in_z: Vec<_> = inner_pure.iter().map(|t| t.z.clone()).collect();

    let mut e_in = EnergyAdapter::new("sym_inner", true, pin, seed.wrapping_add(71));
    let mut e_out = EnergyAdapter::new("sym_outer", false, pin, seed.wrapping_add(73));
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(300));
    e_in.train(&in_pairs, &out_pairs, &out_z, 450, 0.15, &mut rng);
    e_out.train(&out_pairs, &in_pairs, &in_z, 450, 0.15, &mut rng);
    EnergyPromotionBundle::promote(&encoder, vec![e_in.clone(), e_out.clone()]).expect("promote");

    let lambda = 0.35f32;
    let mut data_rng = StdRng::seed_from_u64(seed.wrapping_mul(43).wrapping_add(8));
    let composite = generate_transitions(&encoder, 400, true, &mut data_rng);
    let mut split_rng = StdRng::seed_from_u64(seed.wrapping_mul(131).wrapping_add(train_n as u64));
    let (train, heldout) = stratified_wm_split(&composite, train_n, &mut split_rng);

    let cone_train: Vec<ConeSample> = train
        .iter()
        .map(|t| ConeSample {
            features: cone_features(e_in.affinity(&t.z), e_out.affinity(&t.z)),
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

    let mut vg_err = 0.0;
    let mut conf_err = 0.0;
    let mut cone_err = 0.0;
    let mut margin = 0.0;
    let mut region_hits = 0usize;
    let mut violations = 0usize;
    let mut routes = Vec::new();
    let n = heldout.len().max(1) as f32;

    for t in &heldout {
        let a = e_in.affinity(&t.z);
        let b = e_out.affinity(&t.z);
        let pa = e_in.propose_next(&t.z);
        let pb = e_out.propose_next(&t.z);
        let vg: Vec<f32> = pa.iter().zip(pb.iter()).map(|(u, v)| 0.5 * (u + v)).collect();
        vg_err += mse(&vg, &t.z_next);
        conf_err += mse(if a >= b { &pa } else { &pb }, &t.z_next);
        let d = router.decide(&cone_features(a, b));
        let blend: Vec<f32> = pa
            .iter()
            .zip(pb.iter())
            .map(|(u, v)| d.spiral_weight * u + (1.0 - d.spiral_weight) * v)
            .collect();
        cone_err += mse(&blend, &t.z_next);

        let ea = symbolic_energy(&e_in, &t.z, &t.z_next, &t.obs, &t.obs_next, lambda);
        let eb = symbolic_energy(&e_out, &t.z, &t.z_next, &t.obs, &t.obs_next, lambda);
        let home = if t.regime_inner { ea } else { eb };
        let away = if t.regime_inner { eb } else { ea };
        margin += away - home;

        let route_inner = d.spiral_weight >= 0.5;
        routes.push(if route_inner { 0 } else { 1 });
        if route_inner == t.regime_inner {
            region_hits += 1;
        }

        let home_rule = if t.regime_inner {
            WorldRule::InnerRotationDominates
        } else {
            WorldRule::OuterRadialExpand
        };
        if home_rule.penalty(&t.obs, &t.obs_next) > 1e-3 {
            violations += 1;
        }
    }

    let regime_agreement = region_hits as f32 / n;
    let entropy = routing_entropy(&routes);
    SymWmSeedResult {
        train_n,
        vg_mse: vg_err / n,
        conf_mse: conf_err / n,
        cone_mse: cone_err / n,
        energy_margin: margin / n,
        regime_agreement,
        rule_violation_rate: violations as f32 / n,
        degenerate: (regime_agreement - 0.5).abs() < 0.01 || entropy < 0.3,
        encoder_fingerprint: pin,
    }
}

// --- helpers ---

fn mse(a: &[f32], b: &[f32]) -> f32 {
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
    let p0 = choices.iter().filter(|&&c| c == 0).count() as f32 / n;
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
    fn geo_seed_sane() {
        let r = run_phase3k_geo_seed(42, 30);
        assert!(r.regime_agreement > 0.55, "{}", r.regime_agreement);
        assert!(!r.degenerate);
        assert!(r.energy_margin > 0.0, "{}", r.energy_margin);
    }

    #[test]
    fn prob_seed_sane() {
        let r = run_phase3l_prob_seed(42, 30);
        assert!(r.regime_agreement > 0.55, "{}", r.regime_agreement);
        assert!(!r.degenerate);
    }

    #[test]
    fn sym_seed_sane() {
        let r = run_phase3m_sym_seed(42, 30);
        assert!(r.regime_agreement > 0.55, "{}", r.regime_agreement);
        assert!(!r.degenerate);
        assert!(r.energy_margin > 0.0);
    }

    #[test]
    fn rules_fire_on_mismatched_dynamics() {
        let obs = vec![0.1, 0.0, 0.0, 0.0];
        // Pure radial move in inner — should penalize InnerRotationDominates.
        let next = vec![0.25, 0.0, 0.0, 0.0];
        assert!(WorldRule::InnerRotationDominates.penalty(&obs, &next) > 0.05);
    }
}
