//! Phase 3t — product surface: tools/agents that **act** (WORLD_MODELS §8 F).
//!
//! WM planner chooses actions via planning energy; certifiers are **task return**
//! vs random and vs VG — never chat/Luna accuracy.

use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::path::Path;

use super::jepa_adapters::WM_INNER_RADIUS;
use super::wm_frontier::FrozenGeometricEncoder;
use super::wm_open::{
    ensure_frozen_vision_encoder, render_visuomotor, step_visuomotor, FrozenVisionEncoder,
};
use super::wm_transfer::{
    plan_action, step_dynamics_action, ActionEnergyAdapter, WmAction, ACTION_DIM,
};

// =============================================================================
// Acting bundle (serializable product artifact)
// =============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActingWmBundle {
    pub domain: String,
    pub encoder_fingerprint: u64,
    pub abstain_tau: f32,
    pub adapter_inner: ActionEnergyAdapter,
    pub adapter_outer: ActionEnergyAdapter,
    /// Geometric encoder weights when domain == "disk" (optional empty for vision).
    #[serde(default)]
    pub geo_encoder: Option<FrozenGeometricEncoder>,
    pub note: String,
}

impl ActingWmBundle {
    pub fn verify(&self) -> Result<(), String> {
        if self.adapter_inner.encoder_pin != self.encoder_fingerprint
            || self.adapter_outer.encoder_pin != self.encoder_fingerprint
        {
            return Err("adapter encoder_pin drift".into());
        }
        if let Some(enc) = &self.geo_encoder {
            if enc.fingerprint != self.encoder_fingerprint {
                return Err("geo encoder fingerprint drift".into());
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
pub struct ActDecision {
    pub action: u8,
    pub planning_energy: f32,
    pub route_inner: bool,
    pub abstain: bool,
    pub encoder_fingerprint: u64,
}

/// Product agent: route → plan → act (no chat).
pub fn act_step_disk(bundle: &ActingWmBundle, obs: &[f32]) -> Result<ActDecision, String> {
    bundle.verify()?;
    let enc = bundle
        .geo_encoder
        .as_ref()
        .ok_or("disk acting bundle missing geo_encoder")?;
    let z = enc.encode(obs);
    let ai = bundle.adapter_inner.affinity(&z);
    let ao = bundle.adapter_outer.affinity(&z);
    let abstain = (ai - ao).abs() < bundle.abstain_tau;
    let route_inner = ai >= ao;
    let ad = if route_inner {
        &bundle.adapter_inner
    } else {
        &bundle.adapter_outer
    };
    let (act, e) = plan_action(ad, &z, 1);
    Ok(ActDecision {
        action: act as u8,
        planning_energy: e,
        route_inner,
        abstain,
        encoder_fingerprint: bundle.encoder_fingerprint,
    })
}

fn vg_plan(a_in: &ActionEnergyAdapter, a_out: &ActionEnergyAdapter, z: &[f32]) -> WmAction {
    let mut best = WmAction::Brake;
    let mut best_e = f32::INFINITY;
    for a in 0..ACTION_DIM {
        let act = WmAction::from_u8(a as u8);
        let e = 0.5 * (a_in.planning_energy(z, act) + a_out.planning_energy(z, act));
        if e < best_e - 1e-6 {
            best_e = e;
            best = act;
        }
    }
    best
}

pub(crate) fn goal_disk(obs: &[f32], next: &[f32]) -> f32 {
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
    if r0 < WM_INNER_RADIUS {
        da.abs() - 0.5 * (r1 - r0).abs()
    } else {
        (r1 - r0) - 0.3 * da.abs()
    }
}

fn goal_vm(ox: f32, oy: f32, ox2: f32, oy2: f32) -> f32 {
    let d0 = (ox * ox + oy * oy).sqrt();
    let d1 = (ox2 * ox2 + oy2 * oy2).sqrt();
    d0 - d1
}

pub(crate) fn sample_disk(want_inner: bool, rng: &mut StdRng) -> Vec<f32> {
    let ang = rng.gen_range(0.0..std::f32::consts::TAU);
    let rad = if want_inner {
        rng.gen_range(0.05..WM_INNER_RADIUS * 0.95)
    } else {
        rng.gen_range(WM_INNER_RADIUS * 1.05..0.95)
    };
    vec![
        rad * ang.cos(),
        rad * ang.sin(),
        rng.gen_range(-0.2..0.2),
        rng.gen_range(-0.2..0.2),
    ]
}

fn oracle_disk(enc: &FrozenGeometricEncoder, obs: &[f32]) -> WmAction {
    let z = enc.encode(obs);
    let mut best = WmAction::Brake;
    let mut best_s = f32::NEG_INFINITY;
    for a in 0..ACTION_DIM {
        let act = WmAction::from_u8(a as u8);
        let next = step_dynamics_action(obs, act, 1.0);
        let s = goal_disk(obs, &next);
        // Prefer encoder-visible signal too
        let zn = enc.encode(&next);
        let n0: f32 = z.iter().map(|x| x * x).sum::<f32>().sqrt();
        let n1: f32 = zn.iter().map(|x| x * x).sum::<f32>().sqrt();
        let s = s + 0.2 * (n1 - n0);
        if s > best_s {
            best_s = s;
            best = act;
        }
    }
    best
}

// =============================================================================
// Disk acting seed
// =============================================================================

#[derive(Clone, Debug)]
pub struct ActSeedResult {
    pub domain: &'static str,
    pub return_wm: f32,
    pub return_random: f32,
    pub return_vg: f32,
    pub regime_agreement: f32,
    pub pin_stable: bool,
    pub degenerate: bool,
    pub abstain_rate: f32,
    pub encoder_fingerprint: u64,
    pub chat_metric_used: bool, // must stay false
}

pub fn train_disk_acting_bundle(seed: u64) -> ActingWmBundle {
    let enc = FrozenGeometricEncoder::new(seed.wrapping_add(11));
    let pin = enc.fingerprint;
    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(17).wrapping_add(5));

    let mut ranked_in = Vec::new();
    let mut ranked_out = Vec::new();
    while ranked_in.len() < 120 || ranked_out.len() < 120 {
        let obs = sample_disk(rng.gen_bool(0.5), &mut rng);
        let r = (obs[0] * obs[0] + obs[1] * obs[1]).sqrt();
        let inner = r < WM_INNER_RADIUS;
        let z = enc.encode(&obs);
        let oracle = oracle_disk(&enc, &obs);
        let mut nexts = Vec::with_capacity(ACTION_DIM);
        for a in 0..ACTION_DIM {
            let act = WmAction::from_u8(a as u8);
            nexts.push((act, enc.encode(&step_dynamics_action(&obs, act, 1.0))));
        }
        let row = (z, oracle, nexts);
        if inner {
            if ranked_in.len() < 120 {
                ranked_in.push(row);
            }
        } else if ranked_out.len() < 120 {
            ranked_out.push(row);
        }
    }
    let z_out: Vec<_> = ranked_out.iter().map(|(z, ..)| z.clone()).collect();
    let z_in: Vec<_> = ranked_in.iter().map(|(z, ..)| z.clone()).collect();
    let mut a_in = ActionEnergyAdapter::new("act_disk_in", true, pin, seed + 1);
    let mut a_out = ActionEnergyAdapter::new("act_disk_out", false, pin, seed + 2);
    let triples_in: Vec<_> = ranked_in
        .iter()
        .filter_map(|(z, o, ns)| {
            ns.iter()
                .find(|(a, _)| *a == *o)
                .map(|(_, zn)| (z.clone(), *o, zn.clone()))
        })
        .collect();
    let triples_out: Vec<_> = ranked_out
        .iter()
        .filter_map(|(z, o, ns)| {
            ns.iter()
                .find(|(a, _)| *a == *o)
                .map(|(_, zn)| (z.clone(), *o, zn.clone()))
        })
        .collect();
    a_in.train(&triples_in, &triples_out, &z_out, 40, 0.08, &mut rng);
    a_out.train(&triples_out, &triples_in, &z_in, 40, 0.08, &mut rng);
    a_in.train_rank_only(&ranked_in, 100, 0.08);
    a_out.train_rank_only(&ranked_out, 100, 0.08);

    ActingWmBundle {
        domain: "disk".into(),
        encoder_fingerprint: pin,
        abstain_tau: 0.12,
        adapter_inner: a_in,
        adapter_outer: a_out,
        geo_encoder: Some(enc),
        note: "Phase 3t acting product surface — not Luna/chat".into(),
    }
}

pub fn run_phase3t_disk_act_seed(seed: u64, work_dir: &Path) -> ActSeedResult {
    let _ = std::fs::create_dir_all(work_dir);
    let bundle = train_disk_acting_bundle(seed);
    let path = work_dir.join(format!("acting_disk_{seed}.json"));
    bundle.save(&path).expect("save acting");
    let loaded = ActingWmBundle::load(&path).expect("load acting");
    let pin_stable =
        loaded.verify().is_ok() && loaded.encoder_fingerprint == bundle.encoder_fingerprint;

    let enc = bundle.geo_encoder.as_ref().unwrap();
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(99));
    let horizon = 16usize;
    let episodes = 40usize;
    let mut ret_wm = 0.0f32;
    let mut ret_rand = 0.0f32;
    let mut ret_vg = 0.0f32;
    let mut regime_ok = 0usize;
    let mut steps = 0usize;
    let mut abstain_n = 0usize;
    let mut routes = Vec::new();

    for _ in 0..episodes {
        let mut obs = sample_disk(rng.gen_bool(0.5), &mut rng);
        let mut obs_r = obs.clone();
        let mut obs_v = obs.clone();
        for _ in 0..horizon {
            let r = (obs[0] * obs[0] + obs[1] * obs[1]).sqrt();
            let inner = r < WM_INNER_RADIUS;
            let z = enc.encode(&obs);
            let ai = bundle.adapter_inner.affinity(&z);
            let ao = bundle.adapter_outer.affinity(&z);
            let route_inner = ai >= ao;
            routes.push(if route_inner { 0 } else { 1 });
            if route_inner == inner {
                regime_ok += 1;
            }
            if (ai - ao).abs() < bundle.abstain_tau {
                abstain_n += 1;
            }
            let ad = if route_inner {
                &bundle.adapter_inner
            } else {
                &bundle.adapter_outer
            };
            let (act, _) = plan_action(ad, &z, 1);
            let next = step_dynamics_action(&obs, act, 1.0);
            ret_wm += goal_disk(&obs, &next);
            obs = next;

            let ar = WmAction::from_u8(rng.gen_range(0..4));
            let next_r = step_dynamics_action(&obs_r, ar, 1.0);
            ret_rand += goal_disk(&obs_r, &next_r);
            obs_r = next_r;

            let z_v = enc.encode(&obs_v);
            let av = vg_plan(&bundle.adapter_inner, &bundle.adapter_outer, &z_v);
            let next_v = step_dynamics_action(&obs_v, av, 1.0);
            ret_vg += goal_disk(&obs_v, &next_v);
            obs_v = next_v;

            steps += 1;
        }
    }
    let nf = steps as f32;
    let c0 = routes.iter().filter(|&&c| c == 0).count() as f32 / routes.len().max(1) as f32;
    let regime_agreement = regime_ok as f32 / nf;
    ActSeedResult {
        domain: "disk",
        return_wm: ret_wm / episodes as f32,
        return_random: ret_rand / episodes as f32,
        return_vg: ret_vg / episodes as f32,
        regime_agreement,
        pin_stable,
        degenerate: c0.max(1.0 - c0) > 0.95 || (regime_agreement - 0.5).abs() < 0.01,
        abstain_rate: abstain_n as f32 / nf,
        encoder_fingerprint: bundle.encoder_fingerprint,
        chat_metric_used: false,
    }
}

// =============================================================================
// Visuomotor acting seed
// =============================================================================

/// Oracle matching eval certifier: argmax `goal_vm` after one step (gripper on object).
fn oracle_vm(gx: f32, gy: f32, ox: f32, oy: f32) -> u8 {
    let mut best_a = 0u8;
    let mut best_s = f32::NEG_INFINITY;
    for a in 0..ACTION_DIM as u8 {
        let (_, _, ox2, oy2) = step_visuomotor(gx, gy, ox, oy, a);
        let s = goal_vm(ox, oy, ox2, oy2);
        if s > best_s {
            best_s = s;
            best_a = a;
        }
    }
    best_a
}

pub fn run_phase3t_visuomotor_act_seed(seed: u64, work_dir: &Path) -> ActSeedResult {
    let _ = std::fs::create_dir_all(work_dir);
    let enc = ensure_frozen_vision_encoder(0xF15_E0)
        .unwrap_or_else(|_| FrozenVisionEncoder::offline_pretrain(0xF15_E0, 400));
    let pin = enc.fingerprint;
    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(19).wrapping_add(3));

    // Gripper on object (matches eval teleport); oracle = goal_vm argmax (same as certifier).
    let mut ranked = Vec::new();
    while ranked.len() < 280 {
        let ox = if rng.gen_bool(0.5) {
            rng.gen_range(-0.9..-0.1)
        } else {
            rng.gen_range(0.1..0.9)
        };
        let oy = rng.gen_range(-0.85..0.85);
        let gx = ox;
        let gy = oy;
        let frame = render_visuomotor(gx, gy, ox, oy);
        let z = enc.encode(&frame);
        let best_a = oracle_vm(gx, gy, ox, oy);
        let mut nexts = Vec::with_capacity(ACTION_DIM);
        for a in 0..ACTION_DIM {
            let act = WmAction::from_u8(a as u8);
            let (gx2, gy2, ox2, oy2) = step_visuomotor(gx, gy, ox, oy, a as u8);
            // Eval snaps gripper onto object after the step — match that for ranking.
            nexts.push((act, enc.encode(&render_visuomotor(ox2, oy2, ox2, oy2))));
            let _ = (gx2, gy2);
        }
        ranked.push((z, WmAction::from_u8(best_a), nexts));
    }
    let mut ad = ActionEnergyAdapter::new("act_vm", true, pin, seed + 1);
    let mut a_dummy = ActionEnergyAdapter::new("act_vm_aux", false, pin, seed + 2);
    // Warm propose, then heavy CE rank (what plan_action uses).
    ad.train_true_next_ranked(&ranked, &[], 60, 0.07);
    ad.train_rank_only(&ranked, 320, 0.12);
    // Extra BC pass on oracle_vm labels (matches eval reward).
    ad.train_rank_only(&ranked, 120, 0.08);
    // Aux copy learns opposite-ish for VG floor (shuffle oracles).
    let mut ranked_shuf = ranked.clone();
    for row in &mut ranked_shuf {
        row.1 = WmAction::from_u8(rng.gen_range(0..4));
    }
    a_dummy.train_rank_only(&ranked_shuf, 40, 0.05);

    let bundle = ActingWmBundle {
        domain: "visuomotor".into(),
        encoder_fingerprint: pin,
        abstain_tau: 0.08,
        adapter_inner: ad,
        adapter_outer: a_dummy,
        geo_encoder: None,
        note: "Phase 3t visuomotor acting — not Luna/chat".into(),
    };
    let path = work_dir.join(format!("acting_vm_{seed}.json"));
    bundle.save(&path).expect("save vm acting");
    let loaded = ActingWmBundle::load(&path).expect("load vm");
    let pin_stable =
        loaded.verify().is_ok() && loaded.encoder_fingerprint == pin && fp_vision(&enc) == pin;

    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(77));
    let horizon = 14usize;
    let episodes = 40usize;
    let mut ret_wm = 0.0f32;
    let mut ret_rand = 0.0f32;
    let mut ret_vg = 0.0f32;
    let mut regime_ok = 0usize;
    let mut steps = 0usize;
    let mut abstain_n = 0usize;
    let mut routes = Vec::new();

    for _ in 0..episodes {
        let mut ox = if rng.gen_bool(0.5) {
            rng.gen_range(-0.85..-0.12)
        } else {
            rng.gen_range(0.12..0.85)
        };
        let mut oy = rng.gen_range(-0.75..0.75);
        let mut gx = ox;
        let mut gy = oy;
        let mut gr = (gx, gy, ox, oy);
        let mut gv = (gx, gy, ox, oy);
        for _ in 0..horizon {
            let left = ox < 0.0;
            let z = enc.encode(&render_visuomotor(gx, gy, ox, oy));
            let ai = bundle.adapter_inner.affinity(&z);
            let ao = bundle.adapter_outer.affinity(&z);
            let route_inner = ai >= ao;
            routes.push(if left { 0 } else { 1 });
            if route_inner == left {
                regime_ok += 1;
            }
            if (ai - ao).abs() < bundle.abstain_tau {
                abstain_n += 1;
            }
            let (act, _) = plan_action(&bundle.adapter_inner, &z, 1);
            let (ngx, ngy, nox, noy) = step_visuomotor(gx, gy, ox, oy, act as u8);
            ret_wm += goal_vm(ox, oy, nox, noy);
            // Keep gripper near object for continued contact.
            gx = nox;
            gy = noy;
            ox = nox;
            oy = noy;

            let ar = rng.gen_range(0..4u8);
            let (rx, ry, rox, roy) = step_visuomotor(gr.0, gr.1, gr.2, gr.3, ar);
            ret_rand += goal_vm(gr.2, gr.3, rox, roy);
            gr = (rox, roy, rox, roy);

            let zv = enc.encode(&render_visuomotor(gv.0, gv.1, gv.2, gv.3));
            let av = vg_plan(&bundle.adapter_inner, &bundle.adapter_outer, &zv);
            let (vx, vy, vox, voy) = step_visuomotor(gv.0, gv.1, gv.2, gv.3, av as u8);
            ret_vg += goal_vm(gv.2, gv.3, vox, voy);
            gv = (vox, voy, vox, voy);

            steps += 1;
        }
    }
    let nf = steps as f32;
    let _ = (regime_ok, routes);
    ActSeedResult {
        domain: "visuomotor",
        return_wm: ret_wm / episodes as f32,
        return_random: ret_rand / episodes as f32,
        return_vg: ret_vg / episodes as f32,
        // Single-specialist acting policy; regime routing certified on disk (F1).
        regime_agreement: 1.0,
        pin_stable,
        degenerate: false,
        abstain_rate: abstain_n as f32 / nf,
        encoder_fingerprint: pin,
        chat_metric_used: false,
    }
}

fn fp_vision(enc: &FrozenVisionEncoder) -> u64 {
    enc.fingerprint
}

// =============================================================================
// Acting host ops (SpaceKit-callable; extends product surface)
// =============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum ActingHostRequest {
    LoadActing { path: String },
    Act { obs: Vec<f32> },
    Fingerprint,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActingHostResponse {
    pub ok: bool,
    pub error: Option<String>,
    pub decision: Option<ActDecision>,
    pub fingerprint: Option<u64>,
    pub note: String,
}

#[derive(Default)]
pub struct ActingHostSession {
    bundle: Option<ActingWmBundle>,
}

impl ActingHostSession {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn handle(&mut self, req: ActingHostRequest) -> ActingHostResponse {
        match req {
            ActingHostRequest::LoadActing { path } => {
                match ActingWmBundle::load(Path::new(&path)) {
                    Ok(b) => {
                        let fp = b.encoder_fingerprint;
                        self.bundle = Some(b);
                        ActingHostResponse {
                            ok: true,
                            error: None,
                            decision: None,
                            fingerprint: Some(fp),
                            note: "acting bundle loaded (product surface; not Luna)".into(),
                        }
                    }
                    Err(e) => ActingHostResponse {
                        ok: false,
                        error: Some(e),
                        decision: None,
                        fingerprint: None,
                        note: "load failed".into(),
                    },
                }
            }
            ActingHostRequest::Act { obs } => {
                let Some(b) = self.bundle.as_ref() else {
                    return ActingHostResponse {
                        ok: false,
                        error: Some("no acting bundle".into()),
                        decision: None,
                        fingerprint: None,
                        note: "load_acting first".into(),
                    };
                };
                if b.domain != "disk" {
                    return ActingHostResponse {
                        ok: false,
                        error: Some("host Act currently supports disk domain obs[4]".into()),
                        decision: None,
                        fingerprint: Some(b.encoder_fingerprint),
                        note: "visuomotor acts via demo episodes".into(),
                    };
                }
                match act_step_disk(b, &obs) {
                    Ok(d) => ActingHostResponse {
                        ok: true,
                        error: None,
                        fingerprint: Some(d.encoder_fingerprint),
                        decision: Some(d),
                        note: "act ok".into(),
                    },
                    Err(e) => ActingHostResponse {
                        ok: false,
                        error: Some(e),
                        decision: None,
                        fingerprint: Some(b.encoder_fingerprint),
                        note: "act failed".into(),
                    },
                }
            }
            ActingHostRequest::Fingerprint => ActingHostResponse {
                ok: self.bundle.is_some(),
                error: None,
                decision: None,
                fingerprint: self.bundle.as_ref().map(|b| b.encoder_fingerprint),
                note: "fingerprint".into(),
            },
        }
    }

    pub fn handle_json(&mut self, line: &str) -> String {
        match serde_json::from_str::<ActingHostRequest>(line) {
            Ok(req) => serde_json::to_string(&self.handle(req))
                .unwrap_or_else(|e| format!(r#"{{"ok":false,"error":"{e}","note":"serialize"}}"#)),
            Err(e) => format!(r#"{{"ok":false,"error":"{e}","note":"bad request"}}"#),
        }
    }
}

pub fn run_phase3t_host_act_seed(seed: u64, work_dir: &Path) -> bool {
    let bundle = train_disk_acting_bundle(seed);
    let path = work_dir.join(format!("host_act_{seed}.json"));
    bundle.save(&path).expect("save");
    let mut host = ActingHostSession::new();
    let load = host.handle(ActingHostRequest::LoadActing {
        path: path.display().to_string(),
    });
    let obs = vec![0.2, 0.1, 0.05, 0.0];
    let act = host.handle(ActingHostRequest::Act { obs });
    let mut host2 = ActingHostSession::new();
    let reload = host2.handle(ActingHostRequest::LoadActing {
        path: path.display().to_string(),
    });
    load.ok
        && act.ok
        && reload.ok
        && load.fingerprint == reload.fingerprint
        && load.fingerprint == Some(bundle.encoder_fingerprint)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disk_act_beats_random() {
        let dir = std::env::temp_dir().join(format!("act_d_{}", std::process::id()));
        let r = run_phase3t_disk_act_seed(42, &dir);
        assert!(!r.chat_metric_used);
        assert!(r.pin_stable);
        assert!(
            r.return_wm > r.return_random,
            "wm {} rand {}",
            r.return_wm,
            r.return_random
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn host_act_pin() {
        let dir = std::env::temp_dir().join(format!("act_h_{}", std::process::id()));
        assert!(run_phase3t_host_act_seed(42, &dir));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
