//! Phase 3v — Spatial / SpaceTime scene-graph world model (WORLD_MODELS WM-1).
//!
//! Explicit objects + typed relations (not spiral toys, not Luna chat).
//! Frozen scene encoder; promoted energy adapters; structure ablation kill-gate.

use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;

use super::energy_jepa::EnergyAdapter;
use super::jepa_adapters::WM_LATENT_DIM;
use super::wm_transfer::{plan_action, ActionEnergyAdapter, WmAction, ACTION_DIM};

pub const SCENE_MAX_NODES: usize = 4; // table + up to 3 blocks
pub const SCENE_FEAT_DIM: usize = SCENE_MAX_NODES * 6 + 16; // poses + edge bag

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeKind {
    Table = 0,
    Block = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeKind {
    Supports = 0,
    Contacts = 1,
    LeftOf = 2,
    OnTop = 3,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneNode {
    pub kind: NodeKind,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneEdge {
    pub src: usize,
    pub dst: usize,
    pub kind: EdgeKind,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneGraph {
    pub nodes: Vec<SceneNode>,
    pub edges: Vec<SceneEdge>,
    /// Regime A = stack-stable friction; B = slippery slide-apart.
    pub regime_stable: bool,
}

impl SceneGraph {
    pub fn features(&self) -> Vec<f32> {
        let mut f = vec![0.0f32; SCENE_FEAT_DIM];
        for (i, n) in self.nodes.iter().take(SCENE_MAX_NODES).enumerate() {
            let o = i * 6;
            f[o] = match n.kind {
                NodeKind::Table => 1.0,
                NodeKind::Block => -1.0,
            };
            f[o + 1] = n.x;
            f[o + 2] = n.y;
            f[o + 3] = n.w;
            f[o + 4] = n.h;
            // Slot reserved; regime label is eval-only (never leaked into features).
            f[o + 5] = 0.0;
        }
        let edge_base = SCENE_MAX_NODES * 6;
        for e in &self.edges {
            let k = e.kind as usize;
            if k < 4 {
                f[edge_base + k] += 1.0;
                f[edge_base + 4 + k] += (e.src as f32 + 1.0) * 0.1;
                f[edge_base + 8 + k] += (e.dst as f32 + 1.0) * 0.1;
            }
        }
        // normalize soft counts
        for i in edge_base..edge_base + 4 {
            f[i] = (f[i] / 4.0).tanh();
        }
        f
    }

    /// Shuffle edge types (structure ablation — adapters should fail if they used structure).
    pub fn with_shuffled_edges(&self, rng: &mut StdRng) -> Self {
        let mut g = self.clone();
        for e in &mut g.edges {
            e.kind = match rng.gen_range(0..4) {
                0 => EdgeKind::Supports,
                1 => EdgeKind::Contacts,
                2 => EdgeKind::LeftOf,
                _ => EdgeKind::OnTop,
            };
        }
        g
    }
}

// =============================================================================
// Frozen scene encoder
// =============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrozenSceneEncoder {
    pub w1: Vec<Vec<f32>>,
    pub b1: Vec<f32>,
    pub w2: Vec<Vec<f32>>,
    pub b2: Vec<f32>,
    pub fingerprint: u64,
    pub note: String,
}

impl FrozenSceneEncoder {
    pub fn new(seed: u64) -> Self {
        let hidden = 40;
        let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x5CE_11E).wrapping_add(2));
        let s1 = (2.0 / SCENE_FEAT_DIM as f32).sqrt();
        let s2 = (2.0 / hidden as f32).sqrt();
        let w1 = rand_mat(hidden, SCENE_FEAT_DIM, s1, &mut rng);
        let b1 = vec![0.0; hidden];
        let w2 = rand_mat(WM_LATENT_DIM, hidden, s2, &mut rng);
        let b2 = vec![0.0; WM_LATENT_DIM];
        let fingerprint = fp_mats(&[&w1, &w2], &[&b1, &b2]);
        Self {
            w1,
            b1,
            w2,
            b2,
            fingerprint,
            note: "frozen scene-graph encoder; never trained after construction".into(),
        }
    }

    pub fn encode(&self, scene: &SceneGraph) -> Vec<f32> {
        let x = scene.features();
        let h: Vec<f32> = self
            .w1
            .iter()
            .zip(self.b1.iter())
            .map(|(row, &bias)| {
                let mut s = bias;
                for (j, &xj) in x.iter().enumerate() {
                    s += row[j] * xj;
                }
                s.tanh()
            })
            .collect();
        self.w2
            .iter()
            .zip(self.b2.iter())
            .map(|(row, &bias)| {
                let mut s = bias;
                for (j, &hj) in h.iter().enumerate() {
                    s += row[j] * hj;
                }
                s.tanh()
            })
            .collect()
    }
}

// =============================================================================
// Scene dynamics + relations
// =============================================================================

fn recompute_edges(nodes: &[SceneNode]) -> Vec<SceneEdge> {
    let mut edges = Vec::new();
    for i in 0..nodes.len() {
        for j in 0..nodes.len() {
            if i == j {
                continue;
            }
            let a = &nodes[i];
            let b = &nodes[j];
            if a.kind == NodeKind::Table && b.kind == NodeKind::Block {
                if b.y - b.h * 0.5 <= a.y + a.h * 0.5 + 0.08
                    && (b.x - a.x).abs() < (a.w + b.w) * 0.5
                {
                    edges.push(SceneEdge {
                        src: i,
                        dst: j,
                        kind: EdgeKind::Supports,
                    });
                }
            }
            if a.kind == NodeKind::Block && b.kind == NodeKind::Block {
                let dx = (a.x - b.x).abs();
                let dy = (a.y - b.y).abs();
                if dx < (a.w + b.w) * 0.55 && dy < (a.h + b.h) * 0.55 {
                    edges.push(SceneEdge {
                        src: i,
                        dst: j,
                        kind: EdgeKind::Contacts,
                    });
                }
                if a.x + a.w * 0.4 < b.x - b.w * 0.4 {
                    edges.push(SceneEdge {
                        src: i,
                        dst: j,
                        kind: EdgeKind::LeftOf,
                    });
                }
                if (a.x - b.x).abs() < (a.w + b.w) * 0.4
                    && a.y > b.y + b.h * 0.3
                    && a.y < b.y + b.h * 1.2
                {
                    edges.push(SceneEdge {
                        src: i,
                        dst: j,
                        kind: EdgeKind::OnTop,
                    });
                }
            }
        }
    }
    edges
}

pub fn sample_scene(want_stable: bool, rng: &mut StdRng) -> SceneGraph {
    let table = SceneNode {
        kind: NodeKind::Table,
        x: 0.0,
        y: -0.7,
        w: 1.6,
        h: 0.15,
    };
    let mut nodes = vec![table];
    if want_stable {
        // Stacked tower
        let x = rng.gen_range(-0.2..0.2);
        nodes.push(SceneNode {
            kind: NodeKind::Block,
            x,
            y: -0.45,
            w: 0.25,
            h: 0.22,
        });
        nodes.push(SceneNode {
            kind: NodeKind::Block,
            x: x + rng.gen_range(-0.03..0.03),
            y: -0.18,
            w: 0.22,
            h: 0.22,
        });
        if rng.gen_bool(0.6) {
            nodes.push(SceneNode {
                kind: NodeKind::Block,
                x: x + rng.gen_range(-0.03..0.03),
                y: 0.08,
                w: 0.20,
                h: 0.20,
            });
        }
    } else {
        // Spread / sliding layout
        for k in 0..3 {
            nodes.push(SceneNode {
                kind: NodeKind::Block,
                x: rng.gen_range(-0.7..0.7),
                y: -0.45 + k as f32 * 0.05,
                w: 0.22,
                h: 0.20,
            });
        }
    }
    let edges = recompute_edges(&nodes);
    SceneGraph {
        nodes,
        edges,
        regime_stable: want_stable,
    }
}

/// Discrete act: nudge block index (1..) in cardinal direction (WmAction mapping).
pub fn step_scene(scene: &SceneGraph, block_idx: usize, action: WmAction) -> SceneGraph {
    let mut nodes = scene.nodes.clone();
    if block_idx == 0 || block_idx >= nodes.len() {
        let edges = recompute_edges(&nodes);
        return SceneGraph {
            nodes,
            edges,
            regime_stable: scene.regime_stable,
        };
    }
    let impulse = if scene.regime_stable { 0.04 } else { 0.12 };
    {
        let n = &mut nodes[block_idx];
        match action {
            WmAction::Tangential => n.x -= impulse, // left
            WmAction::RadialOut => n.x += impulse,  // right
            WmAction::RadialIn => n.y += impulse,   // up
            WmAction::Brake => n.y -= impulse,      // down
        }
        n.x = n.x.clamp(-0.95, 0.95);
        n.y = n.y.clamp(-0.65, 0.9);
    }
    // Gravity / support settle
    for i in 1..nodes.len() {
        if scene.regime_stable {
            // snap down onto support
            let mut support_y = -0.7 + 0.075;
            for j in 0..nodes.len() {
                if j == i {
                    continue;
                }
                if (nodes[i].x - nodes[j].x).abs() < (nodes[i].w + nodes[j].w) * 0.45
                    && nodes[j].y < nodes[i].y
                {
                    support_y =
                        f32::max(support_y, nodes[j].y + nodes[j].h * 0.5 + nodes[i].h * 0.5);
                }
            }
            nodes[i].y = nodes[i].y.max(support_y);
        } else {
            // slippery: drift apart from nearest block
            let mut push = 0.0f32;
            let xi = nodes[i].x;
            for j in 1..nodes.len() {
                if j == i {
                    continue;
                }
                let dx = xi - nodes[j].x;
                if dx.abs() < 0.35 {
                    push += dx.signum() * 0.03;
                }
            }
            nodes[i].x = (xi + push).clamp(-0.95, 0.95);
            nodes[i].y = (nodes[i].y - 0.02).max(-0.55);
        }
    }
    let edges = recompute_edges(&nodes);
    SceneGraph {
        nodes,
        edges,
        regime_stable: scene.regime_stable,
    }
}

fn stack_height(scene: &SceneGraph) -> f32 {
    scene
        .nodes
        .iter()
        .filter(|n| n.kind == NodeKind::Block)
        .map(|n| n.y)
        .fold(f32::NEG_INFINITY, f32::max)
}

fn block_xs(scene: &SceneGraph) -> Vec<f32> {
    scene
        .nodes
        .iter()
        .filter(|n| n.kind == NodeKind::Block)
        .map(|n| n.x)
        .collect()
}

fn mean_spread(xs: &[f32]) -> f32 {
    if xs.len() < 2 {
        return 0.0;
    }
    let mut s = 0.0f32;
    let mut c = 0.0f32;
    for i in 0..xs.len() {
        for j in i + 1..xs.len() {
            s += (xs[i] - xs[j]).abs();
            c += 1.0;
        }
    }
    s / c
}

fn support_score(scene: &SceneGraph) -> f32 {
    scene
        .edges
        .iter()
        .filter(|e| e.kind == EdgeKind::Supports || e.kind == EdgeKind::OnTop)
        .count() as f32
}

pub fn goal_scene(before: &SceneGraph, after: &SceneGraph) -> f32 {
    if before.regime_stable {
        // Keep tower: height + supports − lateral spread
        let h0 = stack_height(before);
        let h1 = stack_height(after);
        let s0 = support_score(before);
        let s1 = support_score(after);
        let sp0 = mean_spread(&block_xs(before));
        let sp1 = mean_spread(&block_xs(after));
        2.0 * (h1 - h0) + 0.5 * (s1 - s0) - 1.5 * (sp1 - sp0)
    } else {
        // Slippery: maximize lateral spread; mild penalty for collapsing height
        let sp0 = mean_spread(&block_xs(before));
        let sp1 = mean_spread(&block_xs(after));
        let h0 = stack_height(before);
        let h1 = stack_height(after);
        3.0 * (sp1 - sp0) - 0.25 * (h1 - h0).abs()
    }
}

/// Geometric 1-step oracle action (acting certifier labels).
fn oracle_action(scene: &SceneGraph, block_idx: usize) -> WmAction {
    let mut best = WmAction::RadialIn;
    let mut best_s = f32::NEG_INFINITY;
    for a in 0..ACTION_DIM {
        let act = WmAction::from_u8(a as u8);
        let g1 = step_scene(scene, block_idx, act);
        let s = goal_scene(scene, &g1);
        if s > best_s + 1e-6 {
            best_s = s;
            best = act;
        }
    }
    best
}

// =============================================================================
// Serializable scene WM bundle (SpaceKit deploy artifact)
// =============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneWmBundle {
    pub encoder: FrozenSceneEncoder,
    pub encoder_fingerprint: u64,
    pub abstain_tau: f32,
    pub energy_stable: EnergyAdapter,
    pub energy_slip: EnergyAdapter,
    pub act_stable: ActionEnergyAdapter,
    pub act_slip: ActionEnergyAdapter,
    pub note: String,
}

impl SceneWmBundle {
    pub fn verify(&self) -> Result<(), String> {
        if self.encoder.fingerprint != self.encoder_fingerprint {
            return Err("scene encoder fingerprint drift".into());
        }
        for (name, pin) in [
            ("energy_stable", self.energy_stable.encoder_pin),
            ("energy_slip", self.energy_slip.encoder_pin),
            ("act_stable", self.act_stable.encoder_pin),
            ("act_slip", self.act_slip.encoder_pin),
        ] {
            if pin != self.encoder_fingerprint {
                return Err(format!("{name} adapter pin drift"));
            }
        }
        Ok(())
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        self.verify()?;
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
        }
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, json).map_err(|e| e.to_string())
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let s = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let b: Self = serde_json::from_str(&s).map_err(|e| e.to_string())?;
        b.verify()?;
        Ok(b)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneStepDecision {
    pub route_stable: bool,
    pub abstain: bool,
    pub energy_stable: f32,
    pub energy_slip: f32,
    pub affinity_stable: f32,
    pub affinity_slip: f32,
    pub proposed_z: Vec<f32>,
    pub encoder_fingerprint: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneActDecision {
    pub action: u8,
    pub block_idx: usize,
    pub planning_energy: f32,
    pub route_stable: bool,
    pub abstain: bool,
    pub encoder_fingerprint: u64,
}

/// Predictive route / energy on a scene graph (deploy_step analog).
pub fn scene_deploy_step(
    bundle: &SceneWmBundle,
    scene: &SceneGraph,
) -> Result<SceneStepDecision, String> {
    bundle.verify()?;
    let z = bundle.encoder.encode(scene);
    let ps = bundle.energy_stable.propose_next(&z);
    let pl = bundle.energy_slip.propose_next(&z);
    let es = bundle.energy_stable.energy(&z, &ps);
    let el = bundle.energy_slip.energy(&z, &pl);
    let as_ = bundle.energy_stable.affinity(&z);
    let al = bundle.energy_slip.affinity(&z);
    let route_stable = es < el;
    let abstain = (as_ - al).abs() < bundle.abstain_tau;
    let proposed_z = if route_stable { ps } else { pl };
    Ok(SceneStepDecision {
        route_stable,
        abstain,
        energy_stable: es,
        energy_slip: el,
        affinity_stable: as_,
        affinity_slip: al,
        proposed_z,
        encoder_fingerprint: bundle.encoder_fingerprint,
    })
}

/// Route → plan_action on scene graph (product act surface).
///
/// Default route matches [`scene_deploy_step`] (proposal-energy). Optional sticky
/// keeps the specialist for an episode (avoids mid-horizon thrash).
pub fn scene_act_step(
    bundle: &SceneWmBundle,
    scene: &SceneGraph,
    block_idx: usize,
) -> Result<SceneActDecision, String> {
    scene_act_step_routed(bundle, scene, block_idx, None)
}

pub fn scene_act_step_routed(
    bundle: &SceneWmBundle,
    scene: &SceneGraph,
    block_idx: usize,
    sticky_route_stable: Option<bool>,
) -> Result<SceneActDecision, String> {
    bundle.verify()?;
    if block_idx == 0 || block_idx >= scene.nodes.len() {
        return Err("block_idx must address a Block node".into());
    }
    let z = bundle.encoder.encode(scene);
    let as_ = bundle.energy_stable.affinity(&z);
    let al = bundle.energy_slip.affinity(&z);
    let abstain = (as_ - al).abs() < bundle.abstain_tau;
    let route_stable = match sticky_route_stable {
        Some(s) => s,
        None => {
            let ps = bundle.energy_stable.propose_next(&z);
            let pl = bundle.energy_slip.propose_next(&z);
            bundle.energy_stable.energy(&z, &ps) < bundle.energy_slip.energy(&z, &pl)
        }
    };
    let ad = if route_stable {
        &bundle.act_stable
    } else {
        &bundle.act_slip
    };
    let (act, e) = plan_action(ad, &z, 1);
    Ok(SceneActDecision {
        action: act as u8,
        block_idx,
        planning_energy: e,
        route_stable,
        abstain,
        encoder_fingerprint: bundle.encoder_fingerprint,
    })
}

pub fn pick_block(scene: &SceneGraph, rng: &mut StdRng) -> usize {
    if scene.nodes.len() <= 1 {
        return 0;
    }
    rng.gen_range(1..scene.nodes.len())
}

/// Train frozen-encoder + energy/act adapters (3v/3w shared).
pub fn train_scene_wm_bundle(seed: u64) -> SceneWmBundle {
    let enc = FrozenSceneEncoder::new(seed.wrapping_add(0x5CE_11E));
    let pin = enc.fingerprint;
    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(41).wrapping_add(7));

    let mut stable_pairs = Vec::new();
    let mut slippery_pairs = Vec::new();
    while stable_pairs.len() < 140 || slippery_pairs.len() < 140 {
        let stable = rng.gen_bool(0.5);
        let g0 = sample_scene(stable, &mut rng);
        let bi = pick_block(&g0, &mut rng);
        let act = WmAction::from_u8(rng.gen_range(0..4));
        let g1 = step_scene(&g0, bi, act);
        let z = enc.encode(&g0);
        let zn = enc.encode(&g1);
        if stable {
            if stable_pairs.len() < 140 {
                stable_pairs.push((z, zn));
            }
        } else if slippery_pairs.len() < 140 {
            slippery_pairs.push((z, zn));
        }
    }

    let mut a_s = EnergyAdapter::new("scene_stable", true, pin, seed + 1);
    let mut a_l = EnergyAdapter::new("scene_slip", false, pin, seed + 2);
    let cz_l: Vec<_> = slippery_pairs.iter().map(|(z, _)| z.clone()).collect();
    let cz_s: Vec<_> = stable_pairs.iter().map(|(z, _)| z.clone()).collect();
    a_s.train(&stable_pairs, &slippery_pairs, &cz_l, 350, 0.14, &mut rng);
    a_l.train(&slippery_pairs, &stable_pairs, &cz_s, 350, 0.14, &mut rng);

    let mut ranked_s = Vec::new();
    let mut ranked_l = Vec::new();
    while ranked_s.len() < 160 || ranked_l.len() < 160 {
        let stable = ranked_s.len() < 160;
        let g0 = sample_scene(stable, &mut rng);
        let z = enc.encode(&g0);
        let bi = pick_block(&g0, &mut rng);
        let best = oracle_action(&g0, bi);
        let mut nexts = Vec::with_capacity(ACTION_DIM);
        for a in 0..ACTION_DIM {
            let act = WmAction::from_u8(a as u8);
            let g1 = step_scene(&g0, bi, act);
            nexts.push((act, enc.encode(&g1)));
        }
        let row = (z, best, nexts);
        if stable {
            ranked_s.push(row);
        } else {
            ranked_l.push(row);
        }
    }
    let mut act_s = ActionEnergyAdapter::new("scene_act_s", true, pin, seed + 3);
    let mut act_l = ActionEnergyAdapter::new("scene_act_l", false, pin, seed + 4);
    let z_l: Vec<_> = ranked_l.iter().map(|(z, ..)| z.clone()).collect();
    let z_s: Vec<_> = ranked_s.iter().map(|(z, ..)| z.clone()).collect();
    act_s.train_true_next_ranked(&ranked_s, &z_l, 50, 0.07);
    act_l.train_true_next_ranked(&ranked_l, &z_s, 50, 0.07);
    act_s.train_rank_only(&ranked_s, 220, 0.10);
    act_l.train_rank_only(&ranked_l, 220, 0.10);

    SceneWmBundle {
        encoder: enc,
        encoder_fingerprint: pin,
        abstain_tau: 0.08,
        energy_stable: a_s,
        energy_slip: a_l,
        act_stable: act_s,
        act_slip: act_l,
        note: "Phase 3v/3w scene-graph WM — frozen encoder; adapters only; not Luna/chat".into(),
    }
}

// =============================================================================
// Seed runner
// =============================================================================

#[derive(Clone, Debug)]
pub struct SceneWmSeedResult {
    pub regime_agreement: f32,
    pub energy_margin: f32,
    pub selected_mse: f32,
    pub vg_mse: f32,
    pub return_wm: f32,
    pub return_random: f32,
    pub return_vg: f32,
    pub structure_ablation_drop: f32,
    pub pin_stable: bool,
    pub degenerate: bool,
    pub encoder_fingerprint: u64,
    pub chat_metric_used: bool,
}

/// Certifier eval on a (possibly reloaded) scene bundle — shared by 3v/3w.
pub fn evaluate_scene_wm_bundle(bundle: &SceneWmBundle, seed: u64) -> SceneWmSeedResult {
    let pin = bundle.encoder_fingerprint;
    let pin_after = fp_mats(
        &[&bundle.encoder.w1, &bundle.encoder.w2],
        &[&bundle.encoder.b1, &bundle.encoder.b2],
    );
    let pin_stable = pin_after == pin && bundle.verify().is_ok();
    let enc = &bundle.encoder;
    let a_s = &bundle.energy_stable;
    let a_l = &bundle.energy_slip;
    let act_s = &bundle.act_stable;
    let act_l = &bundle.act_slip;
    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(91).wrapping_add(13));

    let mut region = 0usize;
    let mut margin = 0.0f32;
    let mut sel = 0.0f32;
    let mut vg = 0.0f32;
    let mut routes = Vec::new();
    let n = 200usize;
    for _ in 0..n {
        let stable = rng.gen_bool(0.5);
        let g0 = sample_scene(stable, &mut rng);
        let bi = pick_block(&g0, &mut rng);
        let act = WmAction::from_u8(rng.gen_range(0..4));
        let g1 = step_scene(&g0, bi, act);
        let z = enc.encode(&g0);
        let zn = enc.encode(&g1);
        let ps = a_s.propose_next(&z);
        let pl = a_l.propose_next(&z);
        let es = a_s.energy(&z, &ps);
        let el = a_l.energy(&z, &pl);
        let pick_stable = es < el;
        routes.push(if pick_stable { 0 } else { 1 });
        if pick_stable == stable {
            region += 1;
        }
        let pred = if pick_stable { &ps } else { &pl };
        let avg: Vec<f32> = ps
            .iter()
            .zip(pl.iter())
            .map(|(u, v)| 0.5 * (u + v))
            .collect();
        sel += mse(pred, &zn);
        vg += mse(&avg, &zn);
        let e_home = if stable {
            a_s.energy(&z, &zn)
        } else {
            a_l.energy(&z, &zn)
        };
        let e_away = if stable {
            a_l.energy(&z, &zn)
        } else {
            a_s.energy(&z, &zn)
        };
        margin += e_away - e_home;
    }

    let mut mse_clean = 0.0f32;
    let mut mse_shuf = 0.0f32;
    for _ in 0..n {
        let stable = rng.gen_bool(0.5);
        let g0 = sample_scene(stable, &mut rng);
        let bi = pick_block(&g0, &mut rng);
        let act = WmAction::from_u8(rng.gen_range(0..4));
        let g1 = step_scene(&g0, bi, act);
        let zn = enc.encode(&g1);
        let z = enc.encode(&g0);
        let ps = a_s.propose_next(&z);
        let pl = a_l.propose_next(&z);
        let pred = if a_s.energy(&z, &ps) < a_l.energy(&z, &pl) {
            &ps
        } else {
            &pl
        };
        mse_clean += mse(pred, &zn);

        let g0s = g0.with_shuffled_edges(&mut rng);
        let zs = enc.encode(&g0s);
        let pss = a_s.propose_next(&zs);
        let pls = a_l.propose_next(&zs);
        let preds = if a_s.energy(&zs, &pss) < a_l.energy(&zs, &pls) {
            &pss
        } else {
            &pls
        };
        mse_shuf += mse(preds, &zn);
    }
    let regime_agreement = region as f32 / n as f32;
    let structure_ablation_drop = (mse_shuf - mse_clean) / n as f32;

    let episodes = 48usize;
    let horizon = 6usize;
    let mut ret_wm = 0.0f32;
    let mut ret_rand = 0.0f32;
    let mut ret_vg = 0.0f32;
    for _ in 0..episodes {
        let stable = rng.gen_bool(0.5);
        let g0 = sample_scene(stable, &mut rng);
        let bi = pick_block(&g0, &mut rng);
        let mut g = g0.clone();
        let mut g_r = g0.clone();
        let mut g_v = g0.clone();
        for _ in 0..horizon {
            let z = enc.encode(&g);
            let ad = if stable { act_s } else { act_l };
            let (act, _) = plan_action(ad, &z, 1);
            let g1 = step_scene(&g, bi, act);
            ret_wm += goal_scene(&g, &g1);
            g = g1;

            let ar = WmAction::from_u8(rng.gen_range(0..4));
            let gr1 = step_scene(&g_r, bi, ar);
            ret_rand += goal_scene(&g_r, &gr1);
            g_r = gr1;

            let zv = enc.encode(&g_v);
            let mut best = WmAction::Brake;
            let mut best_e = f32::INFINITY;
            for a in 0..ACTION_DIM {
                let act = WmAction::from_u8(a as u8);
                let e = 0.5 * (act_s.planning_energy(&zv, act) + act_l.planning_energy(&zv, act));
                if e < best_e - 1e-6 {
                    best_e = e;
                    best = act;
                }
            }
            let gv1 = step_scene(&g_v, bi, best);
            ret_vg += goal_scene(&g_v, &gv1);
            g_v = gv1;
        }
    }

    let c0 = routes.iter().filter(|&&c| c == 0).count() as f32 / n as f32;
    SceneWmSeedResult {
        regime_agreement,
        energy_margin: margin / n as f32,
        selected_mse: sel / n as f32,
        vg_mse: vg / n as f32,
        return_wm: ret_wm / episodes as f32,
        return_random: ret_rand / episodes as f32,
        return_vg: ret_vg / episodes as f32,
        structure_ablation_drop,
        pin_stable,
        degenerate: c0.max(1.0 - c0) > 0.95 || (regime_agreement - 0.5).abs() < 0.01,
        encoder_fingerprint: pin,
        chat_metric_used: false,
    }
}

pub fn run_phase3v_scene_seed(seed: u64) -> SceneWmSeedResult {
    let bundle = train_scene_wm_bundle(seed);
    evaluate_scene_wm_bundle(&bundle, seed)
}

pub fn save_scene_bundle_fingerprint(path: &Path, fp: u64) -> Result<(), String> {
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
    }
    std::fs::write(path, format!("{fp:#x}\n")).map_err(|e| e.to_string())
}

fn mse(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for (u, v) in a.iter().zip(b.iter()) {
        let d = u - v;
        s += d * d;
    }
    s / a.len().max(1) as f32
}

fn rand_mat(rows: usize, cols: usize, scale: f32, rng: &mut StdRng) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|_| (0..cols).map(|_| rng.gen_range(-scale..scale)).collect())
        .collect()
}

fn fp_mats(mats: &[&Vec<Vec<f32>>], biases: &[&Vec<f32>]) -> u64 {
    let mut h = DefaultHasher::new();
    for m in mats {
        for row in *m {
            for &v in row {
                v.to_bits().hash(&mut h);
            }
        }
    }
    for b in biases {
        for &v in *b {
            v.to_bits().hash(&mut h);
        }
    }
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_transfer_and_structure() {
        let r = run_phase3v_scene_seed(42);
        assert!(!r.chat_metric_used);
        assert!(r.pin_stable);
        assert!(r.regime_agreement > 0.55, "regime {}", r.regime_agreement);
        assert!(r.energy_margin > 0.0);
        assert!(
            r.structure_ablation_drop > 1e-4,
            "structure MSE ablation {}",
            r.structure_ablation_drop
        );
        assert!(r.return_wm > r.return_random);
        assert!(r.selected_mse <= r.vg_mse + 5e-4);
    }
}
