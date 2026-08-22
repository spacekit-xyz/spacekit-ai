//! Phases 3n–3q: grow the energy substrate beyond the basic toy.
//!
//! - **3n** Action-conditioned \(E(z,a,z')\) + short rollout planner
//! - **3o** Composed stack: geometric \(z\) + ensemble + rule penalties
//! - **3p** Harder / higher-D dynamics transfer (same certifiers)
//! - **3q** Deployment contract: serialize pinned bundle + `deploy_step` (not Luna)

use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::path::Path;

use super::cone_router::{cone_features, AdjustableConeRouter, ConeConfig, ConeSample};
use super::energy_jepa::EnergyAdapter;
use super::jepa_adapters::{WM_INNER_RADIUS, WM_LATENT_DIM};
use super::wm_frontier::{geometric_energy, FrozenGeometricEncoder, WorldRule};

pub const ACTION_DIM: usize = 4;
pub const HARD_OBS_DIM: usize = 8;
const ACT_ENERGY_IN: usize = WM_LATENT_DIM * 2 + ACTION_DIM + WM_LATENT_DIM; // z + a + z' + Δz... wait
                                                                             // feats = [z; a_onehot; z_next; z_next-z] = 8+4+8+8 = 28
const ACT_FEAT: usize = WM_LATENT_DIM + ACTION_DIM + WM_LATENT_DIM + WM_LATENT_DIM;

// =============================================================================
// 3n — Action-conditioned energy
// =============================================================================

/// Discrete actions: 0 tangential+, 1 radial+, 2 radial−, 3 brake.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WmAction {
    Tangential = 0,
    RadialOut = 1,
    RadialIn = 2,
    Brake = 3,
}

impl WmAction {
    pub fn from_u8(a: u8) -> Self {
        match a % 4 {
            0 => Self::Tangential,
            1 => Self::RadialOut,
            2 => Self::RadialIn,
            _ => Self::Brake,
        }
    }
    pub fn one_hot(self) -> [f32; ACTION_DIM] {
        let mut v = [0.0; ACTION_DIM];
        v[self as usize] = 1.0;
        v
    }
}

/// Apply action impulse then passive step (rotation/radial by regime).
pub fn step_dynamics_action(obs: &[f32], action: WmAction, dt: f32) -> Vec<f32> {
    let mut x = obs[0];
    let mut y = obs[1];
    let mut vx = obs[2];
    let mut vy = obs[3];
    let r = (x * x + y * y).sqrt().max(1e-4);
    let ux = x / r;
    let uy = y / r;
    let tx = -uy;
    let ty = ux;
    match action {
        WmAction::Tangential => {
            vx += 0.12 * tx;
            vy += 0.12 * ty;
        }
        WmAction::RadialOut => {
            vx += 0.12 * ux;
            vy += 0.12 * uy;
        }
        WmAction::RadialIn => {
            vx -= 0.10 * ux;
            vy -= 0.10 * uy;
        }
        WmAction::Brake => {
            vx *= 0.7;
            vy *= 0.7;
        }
    }
    // Passive regime dynamics on the updated state.
    let ang = if r < WM_INNER_RADIUS {
        0.30 * dt
    } else {
        0.05 * dt
    };
    let (c, s) = (ang.cos(), ang.sin());
    let nx = c * x - s * y + vx * dt * 0.15;
    let ny = s * x + c * y + vy * dt * 0.15;
    let scale = if r < WM_INNER_RADIUS {
        1.0
    } else {
        1.0 + 0.08 * dt
    };
    vec![
        nx * scale,
        ny * scale,
        (c * vx - s * vy) * 0.97,
        (s * vx + c * vy) * 0.97,
    ]
}

/// Promotable \(E(z, a, z')\) + action-conditioned proposal + planning rank head.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActionEnergyAdapter {
    pub name: String,
    pub regime_is_inner: bool,
    pub encoder_pin: u64,
    hidden: usize,
    e_w1: Vec<Vec<f32>>,
    e_b1: Vec<f32>,
    e_w2: Vec<f32>,
    e_b2: f32,
    // propose: [z; a] → Δz
    p_w1: Vec<Vec<f32>>,
    p_b1: Vec<f32>,
    p_w2: Vec<Vec<f32>>,
    p_b2: Vec<f32>,
    a_w1: Vec<Vec<f32>>,
    a_b1: Vec<f32>,
    a_w2: Vec<f32>,
    a_b2: f32,
    /// Planning landscape on \([z; a]\) (no \(z'\) — avoids flat propose-energy collapse).
    #[serde(default)]
    r_w1: Vec<Vec<f32>>,
    #[serde(default)]
    r_b1: Vec<f32>,
    #[serde(default)]
    r_w2: Vec<f32>,
    #[serde(default)]
    r_b2: f32,
}

impl ActionEnergyAdapter {
    pub fn new(name: &str, regime_is_inner: bool, encoder_pin: u64, seed: u64) -> Self {
        let hidden = 28;
        let mut rng = StdRng::seed_from_u64(seed);
        let s_e = (1.0 / ACT_FEAT as f32).sqrt();
        let pin_dim = WM_LATENT_DIM + ACTION_DIM;
        let s_p = (1.0 / pin_dim as f32).sqrt();
        let s_h = (1.0 / hidden as f32).sqrt();
        let s_z = (1.0 / WM_LATENT_DIM as f32).sqrt();
        Self {
            name: name.to_string(),
            regime_is_inner,
            encoder_pin,
            hidden,
            e_w1: rand_mat(hidden, ACT_FEAT, s_e, &mut rng),
            e_b1: vec![0.0; hidden],
            e_w2: (0..hidden).map(|_| rng.gen_range(-s_h..s_h)).collect(),
            e_b2: 0.5,
            p_w1: rand_mat(hidden, pin_dim, s_p, &mut rng),
            p_b1: vec![0.0; hidden],
            p_w2: rand_mat(WM_LATENT_DIM, hidden, s_h, &mut rng),
            p_b2: vec![0.0; WM_LATENT_DIM],
            a_w1: rand_mat(hidden, WM_LATENT_DIM, s_z, &mut rng),
            a_b1: vec![0.0; hidden],
            a_w2: (0..hidden).map(|_| rng.gen_range(-s_h..s_h)).collect(),
            a_b2: 0.0,
            r_w1: rand_mat(hidden, pin_dim, s_p, &mut rng),
            r_b1: vec![0.0; hidden],
            r_w2: (0..hidden).map(|_| rng.gen_range(-s_h..s_h)).collect(),
            r_b2: 0.5,
        }
    }

    fn feats(z: &[f32], a: &[f32], z_next: &[f32]) -> Vec<f32> {
        let mut f = Vec::with_capacity(ACT_FEAT);
        f.extend_from_slice(z);
        f.extend_from_slice(a);
        f.extend_from_slice(z_next);
        for i in 0..WM_LATENT_DIM {
            f.push(z_next[i] - z[i]);
        }
        f
    }

    pub fn energy(&self, z: &[f32], action: WmAction, z_next: &[f32]) -> f32 {
        let a = action.one_hot();
        let x = Self::feats(z, &a, z_next);
        let h = relu(&x, &self.e_w1, &self.e_b1);
        softplus(dot(&h, &self.e_w2) + self.e_b2)
    }

    pub fn propose(&self, z: &[f32], action: WmAction) -> Vec<f32> {
        let mut xa = z.to_vec();
        xa.extend_from_slice(&action.one_hot());
        let h = relu(&xa, &self.p_w1, &self.p_b1);
        let delta = linear(&h, &self.p_w2, &self.p_b2);
        z.iter().zip(delta.iter()).map(|(u, v)| u + v).collect()
    }

    pub fn affinity(&self, z: &[f32]) -> f32 {
        let h = relu(z, &self.a_w1, &self.a_b1);
        sigmoid(dot(&h, &self.a_w2) + self.a_b2)
    }

    /// Planning energy \(E_\mathrm{plan}(z,a)\) — what the rollout planner minimizes.
    /// Linear score (unbounded): softplus saturates at 0 and kills CE gradients.
    pub fn planning_energy(&self, z: &[f32], action: WmAction) -> f32 {
        if self.r_w1.is_empty() {
            // Legacy adapters without rank head: fall back to propose-pair energy.
            let zn = self.propose(z, action);
            return self.energy(z, action, &zn);
        }
        let mut xa = z.to_vec();
        xa.extend_from_slice(&action.one_hot());
        let h = relu(&xa, &self.r_w1, &self.r_b1);
        dot(&h, &self.r_w2) + self.r_b2
    }

    pub fn train(
        &mut self,
        triples: &[(Vec<f32>, WmAction, Vec<f32>)],
        contrast: &[(Vec<f32>, WmAction, Vec<f32>)],
        contrast_z: &[Vec<f32>],
        epochs: usize,
        lr: f32,
        rng: &mut StdRng,
    ) {
        if triples.is_empty() {
            return;
        }
        let margin = 2.0f32;
        for _ in 0..epochs {
            // Contrastive energy (finite-diff style GD on softplus MLP via numeric-ish accumulate).
            for (z, act, zn) in triples {
                let e_pos = self.energy(z, *act, zn);
                nudge_energy(self, z, *act, zn, e_pos, 2.0 * lr);
                let (cz, ca, czn) = if !contrast.is_empty() && rng.gen_bool(0.5) {
                    let j = rng.gen_range(0..contrast.len());
                    (&contrast[j].0, contrast[j].1, &contrast[j].2)
                } else {
                    let j = rng.gen_range(0..triples.len());
                    let wrong_a = WmAction::from_u8(rng.gen_range(0..4));
                    (z, wrong_a, &triples[j].2)
                };
                let e_neg = self.energy(cz, ca, czn);
                if e_neg - e_pos < margin {
                    nudge_energy(self, cz, ca, czn, e_neg, -2.0 * lr);
                    nudge_energy(self, z, *act, zn, e_pos, lr);
                }
            }
            // Proposal MSE
            for (z, act, zn) in triples {
                let pred = self.propose(z, *act);
                let mut d = vec![0.0; WM_LATENT_DIM];
                for i in 0..WM_LATENT_DIM {
                    d[i] = 2.0 * (pred[i] - zn[i]) / WM_LATENT_DIM as f32;
                }
                // one-step GD on proposal head
                let mut xa = z.to_vec();
                xa.extend_from_slice(&act.one_hot());
                let h = relu(&xa, &self.p_w1, &self.p_b1);
                for o in 0..WM_LATENT_DIM {
                    self.p_b2[o] -= lr * d[o];
                    for j in 0..self.hidden {
                        self.p_w2[o][j] -= lr * (d[o] * h[j] + 1e-4 * self.p_w2[o][j]);
                    }
                }
                for j in 0..self.hidden {
                    let mut dh = 0.0f32;
                    for o in 0..WM_LATENT_DIM {
                        dh += self.p_w2[o][j] * d[o];
                    }
                    if h[j] <= 0.0 {
                        dh = 0.0;
                    }
                    self.p_b1[j] -= lr * dh;
                    for i in 0..xa.len() {
                        self.p_w1[j][i] -= lr * (dh * xa[i] + 1e-4 * self.p_w1[j][i]);
                    }
                }
            }
            // Affinity
            for (z, _, _) in triples {
                nudge_affinity(self, z, 1.0, lr);
            }
            for z in contrast_z {
                nudge_affinity(self, z, 0.0, lr);
            }

            // Explicit action-ranking on planning energy E(z,a,propose(z,a)).
            for (z, act, _) in triples {
                let zn_star = self.propose(z, *act);
                let e_star = self.energy(z, *act, &zn_star);
                for ai in 0..ACTION_DIM {
                    let wrong = WmAction::from_u8(ai as u8);
                    if wrong == *act {
                        continue;
                    }
                    let zn_w = self.propose(z, wrong);
                    let e_w = self.energy(z, wrong, &zn_w);
                    if e_w < e_star + margin {
                        nudge_energy(self, z, wrong, &zn_w, e_w, -2.0 * lr);
                        nudge_energy(self, z, *act, &zn_star, e_star, 2.0 * lr);
                    }
                }
            }
        }
    }

    /// CE-only update of the planning-rank head (no proposal / pair-energy side effects).
    pub fn train_rank_only(
        &mut self,
        states: &[(Vec<f32>, WmAction, Vec<(WmAction, Vec<f32>)>)],
        epochs: usize,
        lr: f32,
    ) {
        if states.is_empty() || self.r_w1.is_empty() {
            return;
        }
        for _ in 0..epochs {
            for (z, oracle, _) in states {
                let mut es = [0.0f32; ACTION_DIM];
                for a in 0..ACTION_DIM {
                    let e = self.planning_energy(z, WmAction::from_u8(a as u8));
                    es[a] = if e.is_finite() {
                        e.clamp(-20.0, 20.0)
                    } else {
                        0.0
                    };
                }
                // softmin via stable log-sum-exp on −E
                let mut m = es[0];
                for &e in &es {
                    m = m.min(e);
                }
                let mut exps = [0.0f32; ACTION_DIM];
                let mut zsum = 0.0f32;
                for a in 0..ACTION_DIM {
                    exps[a] = (-(es[a] - m)).exp();
                    zsum += exps[a];
                }
                if !zsum.is_finite() || zsum < 1e-12 {
                    continue;
                }
                let oi = *oracle as usize;
                for a in 0..ACTION_DIM {
                    let p = exps[a] / zsum;
                    let target = if a == oi { 1.0 } else { 0.0 };
                    let grad = ((target - p) * lr).clamp(-0.1, 0.1);
                    if grad.is_finite() {
                        nudge_rank(self, z, WmAction::from_u8(a as u8), es[a], grad);
                    }
                }
            }
            // Hard clip rank weights to keep linear head finite.
            for row in self.r_w1.iter_mut() {
                for w in row.iter_mut() {
                    if !w.is_finite() {
                        *w = 0.0;
                    } else {
                        *w = w.clamp(-3.0, 3.0);
                    }
                }
            }
            for w in self.r_b1.iter_mut() {
                *w = if w.is_finite() {
                    w.clamp(-3.0, 3.0)
                } else {
                    0.0
                };
            }
            for w in self.r_w2.iter_mut() {
                *w = if w.is_finite() {
                    w.clamp(-3.0, 3.0)
                } else {
                    0.0
                };
            }
            self.r_b2 = if self.r_b2.is_finite() {
                self.r_b2.clamp(-3.0, 3.0)
            } else {
                0.0
            };
        }
    }

    /// Strong ranking for planning landscapes.
    ///
    /// 1. Proposal MSE on every action's true next (dynamics).
    /// 2. Action-conditioned contrast on the **oracle next**:  
    ///    \(E(z,a^\*,z'^\*) + m < E(z,a_w,z'^\*)\) (same \(z'\), different \(a\)).
    /// 3. Planning hinge on \(E(z,a,\mathrm{propose}(z,a))\) so the planner's
    ///    landscape ranks the oracle below wrongs.
    ///
    /// Note: ranking \(E(z,a,z'_a)\) after per-action MSE is near-flat by construction;
    /// do not use that as a certifier.
    pub fn train_true_next_ranked(
        &mut self,
        states: &[(Vec<f32>, WmAction, Vec<(WmAction, Vec<f32>)>)],
        contrast_z: &[Vec<f32>],
        epochs: usize,
        lr: f32,
    ) {
        if states.is_empty() {
            return;
        }
        let margin = 4.0f32;
        for _ in 0..epochs {
            for (z, oracle, nexts) in states {
                let Some((_, zn_star)) = nexts.iter().find(|(a, _)| *a == *oracle) else {
                    continue;
                };
                // Proposal MSE for every action's true next.
                for (act, zn) in nexts {
                    let pred = self.propose(z, *act);
                    let mut d = vec![0.0f32; WM_LATENT_DIM];
                    for i in 0..WM_LATENT_DIM {
                        d[i] = 2.0 * (pred[i] - zn[i]) / WM_LATENT_DIM as f32;
                    }
                    let mut xa = z.to_vec();
                    xa.extend_from_slice(&act.one_hot());
                    let h = relu(&xa, &self.p_w1, &self.p_b1);
                    for o in 0..WM_LATENT_DIM {
                        self.p_b2[o] -= lr * d[o];
                        for j in 0..self.hidden {
                            self.p_w2[o][j] -= lr * (d[o] * h[j] + 1e-4 * self.p_w2[o][j]);
                        }
                    }
                    for j in 0..self.hidden {
                        let mut dh = 0.0f32;
                        for o in 0..WM_LATENT_DIM {
                            dh += self.p_w2[o][j] * d[o];
                        }
                        if h[j] <= 0.0 {
                            dh = 0.0;
                        }
                        self.p_b1[j] -= lr * dh;
                        for i in 0..xa.len() {
                            self.p_w1[j][i] -= lr * (dh * xa[i] + 1e-4 * self.p_w1[j][i]);
                        }
                    }
                }
                // Action-conditioned contrast: same oracle next, wrong action → high E.
                let e_star = self.energy(z, *oracle, zn_star);
                nudge_energy(self, z, *oracle, zn_star, e_star, 4.0 * lr);
                for (act, _) in nexts {
                    if *act == *oracle {
                        continue;
                    }
                    let e_w = self.energy(z, *act, zn_star);
                    if e_w < e_star + margin {
                        nudge_energy(self, z, *act, zn_star, e_w, -4.0 * lr);
                        nudge_energy(self, z, *oracle, zn_star, e_star, 2.0 * lr);
                    }
                }
                nudge_affinity(self, z, 1.0, lr);
            }
            for z in contrast_z {
                nudge_affinity(self, z, 0.0, lr);
            }
        }
    }
}

/// Greedy one-step (or horizon-2) planner: pick action minimizing planning energy.
pub fn plan_action(adapter: &ActionEnergyAdapter, z: &[f32], horizon: usize) -> (WmAction, f32) {
    let mut best = WmAction::Brake;
    let mut best_e = f32::INFINITY;
    for a in 0..ACTION_DIM {
        let act = WmAction::from_u8(a as u8);
        let mut e = 0.0f32;
        let mut zt = z.to_vec();
        for _ in 0..horizon.max(1) {
            e += adapter.planning_energy(&zt, act);
            zt = adapter.propose(&zt, act);
        }
        // Strict < so flat energy does not collapse to action 0.
        if e < best_e - 1e-6 {
            best_e = e;
            best = act;
        }
    }
    (best, best_e)
}

#[derive(Clone, Debug)]
pub struct ActionWmSeedResult {
    pub plan_acc: f32,
    pub regime_agreement: f32,
    pub energy_margin: f32,
    /// Fraction of eval points where oracle action has lower planning energy than mean wrong.
    pub energy_rank_frac: f32,
    pub rollout_mse: f32,
    pub random_plan_acc: f32,
    pub degenerate: bool,
    pub encoder_fingerprint: u64,
}

pub fn run_phase3n_action_seed(seed: u64) -> ActionWmSeedResult {
    let enc = FrozenGeometricEncoder::new(seed);
    let pin = enc.fingerprint;
    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(11).wrapping_add(3));

    let mut inner_t = Vec::new();
    let mut outer_t = Vec::new();
    while inner_t.len() < 160 || outer_t.len() < 160 {
        let obs = sample_obs4(rng.gen_bool(0.5), &mut rng);
        let act = WmAction::from_u8(rng.gen_range(0..4));
        let next = step_dynamics_action(&obs, act, 1.0);
        let z = enc.encode(&obs);
        let zn = enc.encode(&next);
        let r = (obs[0] * obs[0] + obs[1] * obs[1]).sqrt();
        let trip = (z, act, zn, obs, next, r);
        if r < WM_INNER_RADIUS {
            if inner_t.len() < 160 {
                inner_t.push(trip);
            }
        } else if outer_t.len() < 160 {
            outer_t.push(trip);
        }
    }

    let mut a_in = ActionEnergyAdapter::new("act_in", true, pin, seed + 1);
    let mut a_out = ActionEnergyAdapter::new("act_out", false, pin, seed + 2);
    let in_triples: Vec<_> = inner_t
        .iter()
        .map(|(z, a, zn, ..)| (z.clone(), *a, zn.clone()))
        .collect();
    let out_triples: Vec<_> = outer_t
        .iter()
        .map(|(z, a, zn, ..)| (z.clone(), *a, zn.clone()))
        .collect();
    let out_z: Vec<_> = outer_t.iter().map(|(z, ..)| z.clone()).collect();
    let in_z: Vec<_> = inner_t.iter().map(|(z, ..)| z.clone()).collect();
    a_in.train(&in_triples, &out_triples, &out_z, 200, 0.10, &mut rng);
    a_out.train(&out_triples, &in_triples, &in_z, 200, 0.10, &mut rng);

    // Close stretch: true-next ranking + propose distillation (Phase 3r / §8).
    let make_ranked = |trips: &[(Vec<f32>, WmAction, Vec<f32>, Vec<f32>, Vec<f32>, f32)],
                       inner: bool|
     -> Vec<(Vec<f32>, WmAction, Vec<(WmAction, Vec<f32>)>)> {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        for (_z, _a, _zn, obs, _next, _r) in trips {
            let key = (
                (obs[0] * 1000.0) as i32,
                (obs[1] * 1000.0) as i32,
                (obs[2] * 1000.0) as i32,
                (obs[3] * 1000.0) as i32,
            );
            if !seen.insert(key) {
                continue;
            }
            let z = enc.encode(obs);
            let mut nexts = Vec::with_capacity(ACTION_DIM);
            let mut oracle = WmAction::Brake;
            let mut best = f32::NEG_INFINITY;
            for ai in 0..ACTION_DIM {
                let act = WmAction::from_u8(ai as u8);
                let next = step_dynamics_action(obs, act, 1.0);
                let zn = enc.encode(&next);
                // Latent-visible goal so planning rank head can fit from z.
                let n0: f32 = z.iter().map(|x| x * x).sum::<f32>().sqrt();
                let n1: f32 = zn.iter().map(|x| x * x).sum::<f32>().sqrt();
                let s = (n1 - n0)
                    + 0.35 * (zn[0] - z[0])
                    + 0.15 * dynamics_goal_score(obs, &next, inner);
                if s > best {
                    best = s;
                    oracle = act;
                }
                nexts.push((act, zn));
            }
            out.push((z, oracle, nexts));
        }
        out
    };
    let ranked_in = make_ranked(&inner_t, true);
    let ranked_out = make_ranked(&outer_t, false);
    a_in.train_true_next_ranked(&ranked_in, &out_z, 40, 0.08);
    a_out.train_true_next_ranked(&ranked_out, &in_z, 40, 0.08);
    a_in.train_rank_only(&ranked_in, 80, 0.08);
    a_out.train_rank_only(&ranked_out, 80, 0.08);

    // Held-out: oracle best action = lowest true next-step latent MSE under encoder
    // vs planner pick; also regime routing on affinity.
    let mut plan_ok = 0usize;
    let mut rand_ok = 0usize;
    let mut region_ok = 0usize;
    let mut margin = 0.0f32;
    let mut rank_ok = 0usize;
    let mut mse_sum = 0.0f32;
    let mut routes = Vec::new();
    let n_eval = 200usize;
    for _ in 0..n_eval {
        let obs = sample_obs4(rng.gen_bool(0.5), &mut rng);
        let z = enc.encode(&obs);
        let r = (obs[0] * obs[0] + obs[1] * obs[1]).sqrt();
        let inner = r < WM_INNER_RADIUS;
        let adapter = if inner { &a_in } else { &a_out };

        // Oracle: action minimizing encoded next-step distance to true dynamics.
        let mut oracle = WmAction::Brake;
        let mut best_d = f32::INFINITY;
        for ai in 0..ACTION_DIM {
            let act = WmAction::from_u8(ai as u8);
            let next = step_dynamics_action(&obs, act, 1.0);
            let zn = enc.encode(&next);
            let d = mse(&zn, &adapter.propose(&z, act));
            // Prefer action whose true next is well-modeled — also score true progress goal:
            // maximize |Δr| for outer, |Δθ| for inner as soft oracle.
            let score = dynamics_goal_score(&obs, &next, inner) - 0.1 * d;
            if score > best_d || best_d.is_infinite() {
                // maximize score
                if !best_d.is_finite() || score > best_d {
                    best_d = score;
                    oracle = act;
                }
            }
        }
        // Oracle: latent-visible goal (aligned with rank-head training) + light dynamics prior.
        let mut oracle_score = f32::NEG_INFINITY;
        for ai in 0..ACTION_DIM {
            let act = WmAction::from_u8(ai as u8);
            let next = step_dynamics_action(&obs, act, 1.0);
            let zn = enc.encode(&next);
            let n0: f32 = z.iter().map(|x| x * x).sum::<f32>().sqrt();
            let n1: f32 = zn.iter().map(|x| x * x).sum::<f32>().sqrt();
            let s =
                (n1 - n0) + 0.35 * (zn[0] - z[0]) + 0.15 * dynamics_goal_score(&obs, &next, inner);
            if s > oracle_score {
                oracle_score = s;
                oracle = act;
            }
        }

        let (planned, _) = plan_action(adapter, &z, 1);
        if planned == oracle {
            plan_ok += 1;
        }
        let rand_a = WmAction::from_u8(rng.gen_range(0..4));
        if rand_a == oracle {
            rand_ok += 1;
        }

        let next = step_dynamics_action(&obs, planned, 1.0);
        let zn = enc.encode(&next);
        mse_sum += mse(&zn, &adapter.propose(&z, planned));

        let ai = a_in.affinity(&z);
        let ao = a_out.affinity(&z);
        let route_inner = ai >= ao;
        routes.push(if route_inner { 0 } else { 1 });
        if route_inner == inner {
            region_ok += 1;
        }
        // Planning energy margin — what the planner minimizes.
        let e_oracle = adapter.planning_energy(&z, oracle);
        let mut e_wrong = 0.0f32;
        let mut n_wrong = 0.0f32;
        for ai in 0..ACTION_DIM {
            let act = WmAction::from_u8(ai as u8);
            if act == oracle {
                continue;
            }
            e_wrong += adapter.planning_energy(&z, act);
            n_wrong += 1.0;
        }
        let m = e_wrong / n_wrong.max(1.0) - e_oracle;
        margin += m;
        if m > 0.0 {
            rank_ok += 1;
        }
    }
    let n = n_eval as f32;
    let regime_agreement = region_ok as f32 / n;
    let entropy = routing_entropy(&routes);
    ActionWmSeedResult {
        plan_acc: plan_ok as f32 / n,
        regime_agreement,
        energy_margin: margin / n,
        energy_rank_frac: rank_ok as f32 / n,
        rollout_mse: mse_sum / n,
        random_plan_acc: rand_ok as f32 / n,
        degenerate: (regime_agreement - 0.5).abs() < 0.01 || entropy < 0.3,
        encoder_fingerprint: pin,
    }
}

fn dynamics_goal_score(obs: &[f32], next: &[f32], inner: bool) -> f32 {
    let r0 = (obs[0] * obs[0] + obs[1] * obs[1]).sqrt();
    let r1 = (next[0] * next[0] + next[1] * next[1]).sqrt();
    let a0 = obs[1].atan2(obs[0]);
    let a1 = next[1].atan2(next[0]);
    let mut da = a1 - a0;
    while da > std::f32::consts::PI {
        da -= std::f32::consts::TAU;
    }
    while da < -std::f32::consts::PI {
        da += std::f32::consts::TAU;
    }
    if inner {
        da.abs() - 0.5 * (r1 - r0).abs()
    } else {
        (r1 - r0) - 0.3 * da.abs()
    }
}

// =============================================================================
// 3o — Composed geometric + ensemble + rules
// =============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComposedWmBundle {
    pub encoder: FrozenGeometricEncoder,
    pub ens_inner: Vec<EnergyAdapter>,
    pub ens_outer: Vec<EnergyAdapter>,
    pub lambda_rule: f32,
    pub abstain_tau: f32,
    pub encoder_fingerprint: u64,
}

impl ComposedWmBundle {
    pub fn verify(&self) -> Result<(), String> {
        if self.encoder.fingerprint != self.encoder_fingerprint {
            return Err("encoder fingerprint drift".into());
        }
        for a in self.ens_inner.iter().chain(self.ens_outer.iter()) {
            if a.encoder_pin != self.encoder_fingerprint {
                return Err(format!("adapter {} pin mismatch", a.name));
            }
        }
        Ok(())
    }

    pub fn affinity_pair(&self, z: &[f32]) -> (f32, f32) {
        let a = mean_aff(&self.ens_inner, z);
        let b = mean_aff(&self.ens_outer, z);
        (a, b)
    }

    pub fn energy_total(
        &self,
        z: &[f32],
        z_next: &[f32],
        obs: &[f32],
        obs_next: &[f32],
        inner: bool,
    ) -> f32 {
        let ens = if inner {
            &self.ens_inner
        } else {
            &self.ens_outer
        };
        let mut e = ens
            .iter()
            .map(|a| geometric_energy(a, z, z_next))
            .sum::<f32>()
            / ens.len().max(1) as f32;
        let rule = if inner {
            WorldRule::InnerRotationDominates
        } else {
            WorldRule::OuterRadialExpand
        };
        e += self.lambda_rule * rule.penalty(obs, obs_next);
        e
    }
}

fn mean_aff(ens: &[EnergyAdapter], z: &[f32]) -> f32 {
    ens.iter().map(|a| a.affinity(z)).sum::<f32>() / ens.len().max(1) as f32
}

fn mean_propose(ens: &[EnergyAdapter], z: &[f32]) -> Vec<f32> {
    let mut acc = vec![0.0; WM_LATENT_DIM];
    for a in ens {
        let p = a.propose_next(z);
        for i in 0..WM_LATENT_DIM {
            acc[i] += p[i];
        }
    }
    let k = ens.len().max(1) as f32;
    for v in &mut acc {
        *v /= k;
    }
    acc
}

#[derive(Clone, Debug)]
pub struct ComposeWmSeedResult {
    pub regime_agreement: f32,
    pub energy_margin: f32,
    pub cone_mse: f32,
    pub vg_mse: f32,
    pub abstain_rate: f32,
    pub geo_home_frac: f32,
    pub degenerate: bool,
    pub encoder_fingerprint: u64,
}

pub fn train_composed_bundle(seed: u64) -> ComposedWmBundle {
    let enc = FrozenGeometricEncoder::new(seed);
    let pin = enc.fingerprint;
    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(17).wrapping_add(5));
    let (inner, outer) = collect_geo_transitions(&enc, 160, &mut rng);

    let in_pairs: Vec<_> = inner
        .iter()
        .map(|t| (t.z.clone(), t.z_next.clone()))
        .collect();
    let out_pairs: Vec<_> = outer
        .iter()
        .map(|t| (t.z.clone(), t.z_next.clone()))
        .collect();
    let out_z: Vec<_> = outer.iter().map(|t| t.z.clone()).collect();
    let in_z: Vec<_> = inner.iter().map(|t| t.z.clone()).collect();

    let mut ens_in = Vec::new();
    let mut ens_out = Vec::new();
    for k in 0..3 {
        let mut a = EnergyAdapter::new(&format!("c_in_{k}"), true, pin, seed + 10 + k);
        let mut b = EnergyAdapter::new(&format!("c_out_{k}"), false, pin, seed + 20 + k);
        a.train(&in_pairs, &out_pairs, &out_z, 300, 0.14, &mut rng);
        b.train(&out_pairs, &in_pairs, &in_z, 300, 0.14, &mut rng);
        ens_in.push(a);
        ens_out.push(b);
    }
    ComposedWmBundle {
        encoder: enc,
        ens_inner: ens_in,
        ens_outer: ens_out,
        lambda_rule: 0.35,
        abstain_tau: 0.12,
        encoder_fingerprint: pin,
    }
}

pub fn run_phase3o_compose_seed(seed: u64) -> ComposeWmSeedResult {
    let bundle = train_composed_bundle(seed);
    bundle.verify().expect("pin");
    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(19).wrapping_add(7));
    let data = gen_geo_balanced(&bundle.encoder, 400, &mut rng);
    let mut split_rng = StdRng::seed_from_u64(seed.wrapping_mul(131).wrapping_add(30));
    let (train, held) = strat_geo(&data, 30, &mut split_rng);

    let cone_train: Vec<ConeSample> = train
        .iter()
        .map(|t| {
            let (a, b) = bundle.affinity_pair(&t.z);
            ConeSample {
                features: cone_features(a, b),
                route_spiral: t.regime_inner,
                r: t.r,
            }
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

    let mut region = 0usize;
    let mut margin = 0.0;
    let mut cone_e = 0.0;
    let mut vg_e = 0.0;
    let mut abstain_n = 0usize;
    let mut geo_ok = 0usize;
    let mut routes = Vec::new();
    let n = held.len().max(1) as f32;

    for t in &held {
        let (a, b) = bundle.affinity_pair(&t.z);
        let gap = (a - b).abs();
        let abstain = gap < bundle.abstain_tau;
        if abstain {
            abstain_n += 1;
        }
        let d = router.decide(&cone_features(a, b));
        let mut w = d.spiral_weight;
        if abstain {
            w = 0.5 * w + 0.25;
        }
        let pa = mean_propose(&bundle.ens_inner, &t.z);
        let pb = mean_propose(&bundle.ens_outer, &t.z);
        let blend: Vec<_> = pa
            .iter()
            .zip(pb.iter())
            .map(|(u, v)| w * u + (1.0 - w) * v)
            .collect();
        let vg: Vec<_> = pa
            .iter()
            .zip(pb.iter())
            .map(|(u, v)| 0.5 * (u + v))
            .collect();
        cone_e += mse(&blend, &t.z_next);
        vg_e += mse(&vg, &t.z_next);

        let route_inner = w >= 0.5;
        routes.push(if route_inner { 0 } else { 1 });
        if route_inner == t.regime_inner {
            region += 1;
        }

        let eh = bundle.energy_total(&t.z, &t.z_next, &t.obs, &t.obs_next, t.regime_inner);
        let ea = bundle.energy_total(&t.z, &t.z_next, &t.obs, &t.obs_next, !t.regime_inner);
        margin += ea - eh;
        if eh < ea {
            geo_ok += 1;
        }
    }
    let regime_agreement = region as f32 / n;
    ComposeWmSeedResult {
        regime_agreement,
        energy_margin: margin / n,
        cone_mse: cone_e / n,
        vg_mse: vg_e / n,
        abstain_rate: abstain_n as f32 / n,
        geo_home_frac: geo_ok as f32 / n,
        degenerate: (regime_agreement - 0.5).abs() < 0.01 || routing_entropy(&routes) < 0.3,
        encoder_fingerprint: bundle.encoder_fingerprint,
    }
}

// =============================================================================
// 3p — Harder / higher-D dynamics
// =============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrozenHardEncoder {
    pub w: Vec<Vec<f32>>,
    pub b: Vec<f32>,
    pub fingerprint: u64,
}

impl FrozenHardEncoder {
    pub fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0xA5A5).wrapping_add(9));
        let scale = (2.0 / HARD_OBS_DIM as f32).sqrt();
        let w: Vec<Vec<f32>> = (0..WM_LATENT_DIM)
            .map(|_| {
                (0..HARD_OBS_DIM)
                    .map(|_| rng.gen_range(-scale..scale))
                    .collect()
            })
            .collect();
        let b = vec![0.0; WM_LATENT_DIM];
        let mut h = DefaultHasher::new();
        for row in &w {
            for &v in row {
                v.to_bits().hash(&mut h);
            }
        }
        Self {
            w,
            b,
            fingerprint: h.finish(),
        }
    }

    pub fn encode(&self, obs: &[f32]) -> Vec<f32> {
        assert_eq!(obs.len(), HARD_OBS_DIM);
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
}

/// Higher-D: [x,y,vx,vy, ox,oy, phase, clutter] with 3 regimes (inner/mid/outer).
pub fn step_hard(obs: &[f32], dt: f32) -> Vec<f32> {
    let (x, y, vx, vy) = (obs[0], obs[1], obs[2], obs[3]);
    let (ox, oy) = (obs[4], obs[5]);
    let phase = obs[6];
    let clutter = obs[7];
    let r = (x * x + y * y).sqrt();
    let mut nx = x;
    let mut ny = y;
    let mut nvx = vx;
    let mut nvy = vy;
    if r < 0.35 {
        // Strong rotation (distinct from mid/outer).
        let ang = (0.55 + 0.15 * phase.sin()) * dt;
        let (c, s) = (ang.cos(), ang.sin());
        nx = c * x - s * y;
        ny = s * x + c * y;
        nvx = (c * vx - s * vy) * 0.98;
        nvy = (s * vx + c * vy) * 0.98;
    } else if r < 0.65 {
        // Mid: strong shear + obstacle chase (no rotation, no radial expand).
        nx = x + 0.22 * y * dt + 0.08 * (ox - x) * dt;
        ny = y - 0.22 * x * dt + 0.08 * (oy - y) * dt;
        nvx = vx * 0.90 + 0.05 * (ox - x);
        nvy = vy * 0.90 + 0.05 * (oy - y);
    } else {
        // Outer: radial expand + clutter drive (no shear).
        let scale = 1.0 + (0.18 + 0.10 * clutter) * dt;
        nx = x * scale;
        ny = y * scale;
        nvx = vx * 0.92 + 0.12 * x * dt;
        nvy = vy * 0.92 + 0.12 * y * dt;
    }
    vec![
        nx,
        ny,
        nvx,
        nvy,
        ox,
        oy,
        (phase + 0.07 * dt).rem_euclid(std::f32::consts::TAU),
        clutter,
    ]
}

fn regime_hard(obs: &[f32]) -> usize {
    let r = (obs[0] * obs[0] + obs[1] * obs[1]).sqrt();
    if r < 0.35 {
        0
    } else if r < 0.65 {
        1
    } else {
        2
    }
}

#[derive(Clone, Debug)]
pub struct HardWmSeedResult {
    pub regime_agreement: f32,
    pub energy_margin: f32,
    pub cone_mse: f32,
    pub vg_mse: f32,
    pub degenerate: bool,
    pub encoder_fingerprint: u64,
}

pub fn run_phase3p_hard_seed(seed: u64) -> HardWmSeedResult {
    let enc = FrozenHardEncoder::new(seed);
    let pin = enc.fingerprint;
    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(41).wrapping_add(2));

    let mut pools: [Vec<(Vec<f32>, Vec<f32>)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    while pools.iter().any(|p| p.len() < 200) {
        let obs = sample_obs8(&mut rng);
        let next = step_hard(&obs, 1.0);
        let z = enc.encode(&obs);
        let zn = enc.encode(&next);
        let reg = regime_hard(&obs);
        if pools[reg].len() < 200 {
            pools[reg].push((z, zn));
        }
    }

    let mut adapters: Vec<EnergyAdapter> = (0..3)
        .map(|k| {
            EnergyAdapter::new(
                &format!("hard_{k}"),
                k == 0,
                pin,
                seed.wrapping_add(50 + k as u64),
            )
        })
        .collect();
    for k in 0..3 {
        let contrast: Vec<_> = pools[(k + 1) % 3]
            .iter()
            .chain(pools[(k + 2) % 3].iter())
            .cloned()
            .collect();
        let cz: Vec<_> = contrast.iter().map(|(z, _)| z.clone()).collect();
        adapters[k].train(&pools[k], &contrast, &cz, 550, 0.16, &mut rng);
    }

    // Composite held-out
    let mut held = Vec::new();
    while held.len() < 300 {
        let obs = sample_obs8(&mut rng);
        let next = step_hard(&obs, 1.0);
        held.push((
            enc.encode(&obs),
            enc.encode(&next),
            regime_hard(&obs),
            (obs[0] * obs[0] + obs[1] * obs[1]).sqrt(),
        ));
    }
    let train_n = 45usize; // 15 per regime approx
    let mut train = Vec::new();
    let mut counts = [0usize; 3];
    for h in &held {
        if counts[h.2] < train_n / 3 {
            train.push(h.clone());
            counts[h.2] += 1;
        }
    }
    let held: Vec<_> = held.into_iter().skip(train.len()).collect();

    // Pairwise cone between 0 vs rest approximated: route among 3 by affinity argmax;
    // train binary cone for 0 vs not-0 as anti-collapse probe + report 3-way agree.
    let cone_train: Vec<ConeSample> = train
        .iter()
        .map(|(z, _, reg, r)| {
            let a0 = adapters[0].affinity(z);
            let a1 = adapters[1].affinity(z).max(adapters[2].affinity(z));
            ConeSample {
                features: cone_features(a0, a1),
                route_spiral: *reg == 0,
                r: *r,
            }
        })
        .collect();
    let router = AdjustableConeRouter::train(
        &cone_train,
        ConeConfig {
            seed,
            inner_radius: 0.35,
            ..ConeConfig::default()
        },
    );

    let mut region = 0usize;
    let mut margin = 0.0;
    let mut cone_mse = 0.0;
    let mut vg_mse = 0.0;
    let mut routes = Vec::new();
    let n = held.len().max(1) as f32;
    for (z, zn, reg, _) in &held {
        let preds: Vec<Vec<f32>> = adapters.iter().map(|a| a.propose_next(z)).collect();
        // Route by lowest energy of (z, propose_k(z)) — energy substrate, not affinity alone.
        let energies: Vec<f32> = adapters
            .iter()
            .enumerate()
            .map(|(i, a)| a.energy(z, &preds[i]))
            .collect();
        let pred_idx = energies
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        if pred_idx == *reg {
            region += 1;
        }
        routes.push(pred_idx);

        let vg: Vec<f32> = (0..WM_LATENT_DIM)
            .map(|i| preds.iter().map(|p| p[i]).sum::<f32>() / 3.0)
            .collect();
        vg_mse += mse(&vg, zn);
        cone_mse += mse(&preds[pred_idx], zn);

        let e_home = adapters[*reg].energy(z, zn);
        let e_away = adapters
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != *reg)
            .map(|(_, a)| a.energy(z, zn))
            .sum::<f32>()
            / 2.0;
        margin += e_away - e_home;
        let _ = router; // trained as anti-collapse probe; routing uses energy argmin
    }

    // degeneracy: collapse to one class
    let mut hist = [0usize; 3];
    for &r in &routes {
        hist[r.min(2)] += 1;
    }
    let maxc = *hist.iter().max().unwrap() as f32 / n;
    HardWmSeedResult {
        regime_agreement: region as f32 / n,
        energy_margin: margin / n,
        cone_mse: cone_mse / n,
        vg_mse: vg_mse / n,
        degenerate: maxc > 0.92 || (region as f32 / n - 0.333).abs() < 0.02 && maxc > 0.9,
        encoder_fingerprint: pin,
    }
}

fn sample_obs8(rng: &mut StdRng) -> Vec<f32> {
    let reg = rng.gen_range(0..3);
    let ang = rng.gen_range(0.0..std::f32::consts::TAU);
    let rad = match reg {
        0 => rng.gen_range(0.05..0.34),
        1 => rng.gen_range(0.36..0.64),
        _ => rng.gen_range(0.66..0.95),
    };
    vec![
        rad * ang.cos(),
        rad * ang.sin(),
        rng.gen_range(-0.2..0.2),
        rng.gen_range(-0.2..0.2),
        rng.gen_range(-0.5..0.5),
        rng.gen_range(-0.5..0.5),
        rng.gen_range(0.0..std::f32::consts::TAU),
        rng.gen_range(0.0..1.0),
    ]
}

// =============================================================================
// 3q — Deployment contract
// =============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeployDecision {
    pub route_inner: bool,
    pub abstain: bool,
    pub affinity_inner: f32,
    pub affinity_outer: f32,
    pub energy_inner: f32,
    pub energy_outer: f32,
    pub proposed_z: Vec<f32>,
    pub encoder_fingerprint: u64,
}

pub fn save_composed_bundle(path: &Path, bundle: &ComposedWmBundle) -> Result<(), String> {
    bundle.verify()?;
    let json = serde_json::to_string_pretty(bundle).map_err(|e| e.to_string())?;
    std::fs::write(path, json).map_err(|e| e.to_string())
}

pub fn load_composed_bundle(path: &Path) -> Result<ComposedWmBundle, String> {
    let s = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let bundle: ComposedWmBundle = serde_json::from_str(&s).map_err(|e| e.to_string())?;
    bundle.verify()?;
    Ok(bundle)
}

/// Inference-only step for a deployed composed world-model bundle.
pub fn deploy_step(bundle: &ComposedWmBundle, obs: &[f32]) -> Result<DeployDecision, String> {
    bundle.verify()?;
    if obs.len() != 4 {
        return Err("deploy_step expects 4D obs [x,y,vx,vy]".into());
    }
    let z = bundle.encoder.encode(obs);
    let (ai, ao) = bundle.affinity_pair(&z);
    let abstain = (ai - ao).abs() < bundle.abstain_tau;
    let route_inner = ai >= ao;
    let ens = if route_inner {
        &bundle.ens_inner
    } else {
        &bundle.ens_outer
    };
    let proposed = mean_propose(ens, &z);
    // Score true-pair energy proxy using proposal as z'
    let e_in = bundle.energy_total(&z, &proposed, obs, obs, true);
    let e_out = bundle.energy_total(&z, &proposed, obs, obs, false);
    Ok(DeployDecision {
        route_inner,
        abstain,
        affinity_inner: ai,
        affinity_outer: ao,
        energy_inner: e_in,
        energy_outer: e_out,
        proposed_z: proposed,
        encoder_fingerprint: bundle.encoder_fingerprint,
    })
}

#[derive(Clone, Debug)]
pub struct DeploySeedResult {
    pub roundtrip_ok: bool,
    pub pin_stable: bool,
    pub deploy_regime_agree: f32,
    pub decisions: usize,
}

pub fn run_phase3q_deploy_seed(seed: u64, path: &Path) -> DeploySeedResult {
    let bundle = train_composed_bundle(seed);
    save_composed_bundle(path, &bundle).expect("save");
    let loaded = load_composed_bundle(path).expect("load");
    let pin_stable =
        loaded.encoder_fingerprint == bundle.encoder_fingerprint && loaded.verify().is_ok();

    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(999));
    let mut ok = 0usize;
    let n = 80usize;
    for _ in 0..n {
        let obs = sample_obs4(rng.gen_bool(0.5), &mut rng);
        let r = (obs[0] * obs[0] + obs[1] * obs[1]).sqrt();
        let inner = r < WM_INNER_RADIUS;
        let d = deploy_step(&loaded, &obs).expect("step");
        if d.route_inner == inner {
            ok += 1;
        }
    }
    DeploySeedResult {
        roundtrip_ok: pin_stable,
        pin_stable,
        deploy_regime_agree: ok as f32 / n as f32,
        decisions: n,
    }
}

// =============================================================================
// Shared helpers
// =============================================================================

#[derive(Clone)]
struct GeoT {
    obs: Vec<f32>,
    obs_next: Vec<f32>,
    z: Vec<f32>,
    z_next: Vec<f32>,
    r: f32,
    regime_inner: bool,
}

fn sample_obs4(want_inner: bool, rng: &mut StdRng) -> Vec<f32> {
    for _ in 0..48 {
        let ang = rng.gen_range(0.0..std::f32::consts::TAU);
        let rad = if want_inner {
            rng.gen_range(0.05..WM_INNER_RADIUS * 0.95)
        } else {
            rng.gen_range(WM_INNER_RADIUS * 1.05..0.95)
        };
        let x = rad * ang.cos();
        let y = rad * ang.sin();
        let r = (x * x + y * y).sqrt();
        if (r < WM_INNER_RADIUS) == want_inner {
            return vec![x, y, rng.gen_range(-0.2..0.2), rng.gen_range(-0.2..0.2)];
        }
    }
    if want_inner {
        vec![0.1, 0.0, 0.05, 0.0]
    } else {
        vec![0.7, 0.0, 0.05, 0.0]
    }
}

fn collect_geo_transitions(
    enc: &FrozenGeometricEncoder,
    n: usize,
    rng: &mut StdRng,
) -> (Vec<GeoT>, Vec<GeoT>) {
    let mut inner = Vec::new();
    let mut outer = Vec::new();
    while inner.len() < n || outer.len() < n {
        let obs = sample_obs4(rng.gen_bool(0.5), rng);
        let next = {
            // reuse wm_frontier step via jepa step_dynamics
            super::jepa_adapters::step_dynamics(&obs, 1.0)
        };
        let r = (obs[0] * obs[0] + obs[1] * obs[1]).sqrt();
        let t = GeoT {
            z: enc.encode(&obs),
            z_next: enc.encode(&next),
            obs,
            obs_next: next,
            r,
            regime_inner: r < WM_INNER_RADIUS,
        };
        if t.regime_inner && inner.len() < n {
            inner.push(t);
        } else if !t.regime_inner && outer.len() < n {
            outer.push(t);
        }
    }
    (inner, outer)
}

fn gen_geo_balanced(enc: &FrozenGeometricEncoder, n: usize, rng: &mut StdRng) -> Vec<GeoT> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let obs = sample_obs4(i < n / 2, rng);
        let next = super::jepa_adapters::step_dynamics(&obs, 1.0);
        let r = (obs[0] * obs[0] + obs[1] * obs[1]).sqrt();
        out.push(GeoT {
            z: enc.encode(&obs),
            z_next: enc.encode(&next),
            obs,
            obs_next: next,
            r,
            regime_inner: r < WM_INNER_RADIUS,
        });
    }
    out
}

fn strat_geo(data: &[GeoT], train_n: usize, rng: &mut StdRng) -> (Vec<GeoT>, Vec<GeoT>) {
    let mut inner: Vec<_> = data.iter().filter(|t| t.regime_inner).cloned().collect();
    let mut outer: Vec<_> = data.iter().filter(|t| !t.regime_inner).cloned().collect();
    shuffle(&mut inner, rng);
    shuffle(&mut outer, rng);
    let n_in = train_n / 2;
    let mut train = Vec::new();
    train.extend(inner.iter().take(n_in).cloned());
    train.extend(outer.iter().take(train_n - n_in).cloned());
    let mut held = Vec::new();
    held.extend(inner.into_iter().skip(n_in));
    held.extend(outer.into_iter().skip(train_n - n_in));
    (train, held)
}

fn nudge_energy(
    ad: &mut ActionEnergyAdapter,
    z: &[f32],
    act: WmAction,
    zn: &[f32],
    e: f32,
    scale: f32,
) {
    let a = act.one_hot();
    let x = ActionEnergyAdapter::feats(z, &a, zn);
    let h = relu(&x, &ad.e_w1, &ad.e_b1);
    let dlogit = scale * (1.0 - (-e).exp()); // d softplus
    ad.e_b2 -= dlogit;
    for i in 0..ad.hidden {
        ad.e_w2[i] -= dlogit * h[i] + 1e-4 * ad.e_w2[i] * scale.signum().max(0.0);
        let dh = if h[i] > 0.0 { dlogit * ad.e_w2[i] } else { 0.0 };
        ad.e_b1[i] -= dh;
        for j in 0..x.len() {
            ad.e_w1[i][j] -= dh * x[j];
        }
    }
}

fn nudge_rank(ad: &mut ActionEnergyAdapter, z: &[f32], act: WmAction, _e: f32, scale: f32) {
    if ad.r_w1.is_empty() {
        return;
    }
    let mut xa = z.to_vec();
    xa.extend_from_slice(&act.one_hot());
    let h = relu(&xa, &ad.r_w1, &ad.r_b1);
    // Linear head: dE/dlogit = 1. Positive scale lowers energy (same convention as nudge_energy).
    let dlogit = scale;
    ad.r_b2 -= dlogit;
    for i in 0..ad.hidden {
        ad.r_w2[i] -= dlogit * h[i];
        let dh = if h[i] > 0.0 { dlogit * ad.r_w2[i] } else { 0.0 };
        ad.r_b1[i] -= dh;
        for j in 0..xa.len() {
            ad.r_w1[i][j] -= dh * xa[j];
        }
    }
}

fn nudge_affinity(ad: &mut ActionEnergyAdapter, z: &[f32], y: f32, lr: f32) {
    let h = relu(z, &ad.a_w1, &ad.a_b1);
    let p = sigmoid(dot(&h, &ad.a_w2) + ad.a_b2);
    let d = (p - y) * lr;
    ad.a_b2 -= d;
    for i in 0..ad.hidden {
        ad.a_w2[i] -= d * h[i];
        let dh = if h[i] > 0.0 { d * ad.a_w2[i] } else { 0.0 };
        ad.a_b1[i] -= dh;
        for j in 0..z.len() {
            ad.a_w1[i][j] -= dh * z[j];
        }
    }
}

fn rand_mat(rows: usize, cols: usize, scale: f32, rng: &mut StdRng) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|_| (0..cols).map(|_| rng.gen_range(-scale..scale)).collect())
        .collect()
}

fn relu(x: &[f32], w: &[Vec<f32>], b: &[f32]) -> Vec<f32> {
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

fn linear(x: &[f32], w: &[Vec<f32>], b: &[f32]) -> Vec<f32> {
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

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(u, v)| u * v).sum()
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

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
    let mut counts = std::collections::HashMap::new();
    for &c in choices {
        *counts.entry(c).or_insert(0usize) += 1;
    }
    counts
        .values()
        .map(|&c| {
            let p = c as f32 / n;
            if p <= 1e-8 {
                0.0
            } else {
                -p * p.log2()
            }
        })
        .sum()
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
    use std::env::temp_dir;

    #[test]
    fn action_seed_beats_random() {
        let r = run_phase3n_action_seed(42);
        assert!(
            r.plan_acc > r.random_plan_acc + 0.05,
            "plan {} vs rand {}",
            r.plan_acc,
            r.random_plan_acc
        );
        assert!(r.regime_agreement > 0.55);
    }

    #[test]
    fn compose_seed_sane() {
        let r = run_phase3o_compose_seed(42);
        assert!(r.regime_agreement > 0.55);
        assert!(r.energy_margin > 0.0);
        assert!(!r.degenerate);
    }

    #[test]
    fn hard_seed_above_chance() {
        let r = run_phase3p_hard_seed(42);
        assert!(
            r.regime_agreement > 0.45,
            "3-way chance~33%, got {}",
            r.regime_agreement
        );
        assert!(r.energy_margin > 0.0, "margin {}", r.energy_margin);
        assert!(!r.degenerate);
    }

    #[test]
    fn deploy_roundtrip() {
        let path = temp_dir().join(format!("wm_deploy_{}.json", std::process::id()));
        let r = run_phase3q_deploy_seed(42, &path);
        let _ = std::fs::remove_file(&path);
        assert!(r.roundtrip_ok);
        assert!(r.deploy_regime_agree > 0.55);
    }
}
