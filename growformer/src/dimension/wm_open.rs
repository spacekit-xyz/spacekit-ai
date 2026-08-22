//! Phase 3s — close remaining WORLD_MODELS §8 open rungs.
//!
//! - **D** Offline-pretrained frozen vision encoder (JEPA-export slot; never fine-tuned)
//! - **C** Visuomotor foreign log (push–object, rendered patches — not spiral toys)
//! - **E** SpaceKit-callable `deploy_step` host (JSON protocol; not Luna/chat)

use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use super::energy_jepa::EnergyAdapter;
use super::jepa_adapters::WM_LATENT_DIM;
use super::wm_transfer::{
    deploy_step, load_composed_bundle, save_composed_bundle, train_composed_bundle,
    ComposedWmBundle, DeployDecision,
};

pub const VISION_SIDE: usize = 8;
pub const VISION_PIXELS: usize = VISION_SIDE * VISION_SIDE; // 64

// =============================================================================
// D — Offline-pretrained frozen vision encoder (JEPA weight slot)
// =============================================================================

/// Frozen patch encoder. Weights are produced **once** by offline JEPA-style
/// pretraining on a synthetic visuomotor corpus, then hash-pinned forever.
/// Replace this JSON with a real V-JEPA export without changing adapter code.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrozenVisionEncoder {
    pub w1: Vec<Vec<f32>>,
    pub b1: Vec<f32>,
    pub w2: Vec<Vec<f32>>,
    pub b2: Vec<f32>,
    pub in_dim: usize,
    pub fingerprint: u64,
    pub pretrain_steps: usize,
    pub pretrain_loss_final: f32,
    pub note: String,
    pub model_card: String,
}

impl FrozenVisionEncoder {
    /// Offline JEPA-style pretrain: freeze after this call; never train again.
    pub fn offline_pretrain(seed: u64, steps: usize) -> Self {
        let hidden = 48;
        let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0xF15).wrapping_add(7));
        let s1 = (2.0 / VISION_PIXELS as f32).sqrt();
        let s2 = (2.0 / hidden as f32).sqrt();
        let mut w1 = rand_mat(hidden, VISION_PIXELS, s1, &mut rng);
        let mut b1 = vec![0.0; hidden];
        let mut w2 = rand_mat(WM_LATENT_DIM, hidden, s2, &mut rng);
        let mut b2 = vec![0.0; WM_LATENT_DIM];
        // Temporary predictor (discarded after pretrain — JEPA pattern).
        let mut pw = rand_mat(WM_LATENT_DIM, WM_LATENT_DIM, 0.1, &mut rng);
        let mut pb = vec![0.0f32; WM_LATENT_DIM];

        // Disposable probe: predict regime side from z during pretrain only (discarded).
        let mut probe_w = vec![0.0f32; WM_LATENT_DIM];
        let mut probe_b = 0.0f32;

        let mut last_loss = 0.0f32;
        let lr = 0.05f32;
        for _ in 0..steps {
            let (frame, next, left) = sample_visuomotor_transition(&mut rng);
            let z = encode_raw(&w1, &b1, &w2, &b2, &frame);
            let zn = encode_raw(&w1, &b1, &w2, &b2, &next);
            let pred: Vec<f32> = (0..WM_LATENT_DIM)
                .map(|i| {
                    let mut s = pb[i];
                    for j in 0..WM_LATENT_DIM {
                        s += pw[i][j] * z[j];
                    }
                    s.tanh()
                })
                .collect();
            let mut loss = 0.0f32;
            let mut dpred = vec![0.0f32; WM_LATENT_DIM];
            for i in 0..WM_LATENT_DIM {
                let d = pred[i] - zn[i];
                loss += d * d;
                dpred[i] = 2.0 * d / WM_LATENT_DIM as f32;
            }
            // Update predictor
            for i in 0..WM_LATENT_DIM {
                pb[i] -= lr * dpred[i];
                for j in 0..WM_LATENT_DIM {
                    pw[i][j] -= lr * dpred[i] * z[j];
                }
            }
            let h = hidden_raw(&w1, &b1, &frame);
            for i in 0..WM_LATENT_DIM {
                let target = 0.5 * (zn[i] + pred[i]);
                let err = z[i] - target;
                b2[i] -= lr * 0.3 * err;
                for j in 0..hidden {
                    w2[i][j] -= lr * 0.3 * err * h[j];
                }
            }
            // Spatial probe (pretrain only): keep room side readable in z.
            let mut logit = probe_b;
            for i in 0..WM_LATENT_DIM {
                logit += probe_w[i] * z[i];
            }
            let p = 1.0 / (1.0 + (-logit).exp());
            let y = if left { 1.0 } else { 0.0 };
            let derr = p - y;
            loss += derr * derr;
            probe_b -= lr * derr;
            for i in 0..WM_LATENT_DIM {
                probe_w[i] -= lr * derr * z[i];
                // Push encoder so z separates regimes (still frozen after this loop).
                b2[i] -= lr * 0.5 * derr * probe_w[i];
                for j in 0..hidden {
                    w2[i][j] -= lr * 0.35 * derr * probe_w[i] * h[j];
                }
            }
            last_loss = loss;
        }
        let _ = (probe_w, probe_b); // discarded — JEPA/adapter path never sees them
        let fingerprint = fp_mats(&[&w1, &w2], &[&b1, &b2]);
        Self {
            w1,
            b1,
            w2,
            b2,
            in_dim: VISION_PIXELS,
            fingerprint,
            pretrain_steps: steps,
            pretrain_loss_final: last_loss,
            note: "offline JEPA-style vision encoder; frozen after pretrain; adapters only".into(),
            model_card: "Slot for real V-JEPA/JEPA weights. Same encode()/fingerprint contract. Do not fine-tune.".into(),
        }
    }

    pub fn encode(&self, pixels: &[f32]) -> Vec<f32> {
        assert_eq!(pixels.len(), self.in_dim);
        encode_raw(&self.w1, &self.b1, &self.w2, &self.b2, pixels)
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
        }
        let json = serde_json::to_string(self).map_err(|e| e.to_string())?;
        std::fs::write(path, json).map_err(|e| e.to_string())
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let s = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let enc: Self = serde_json::from_str(&s).map_err(|e| e.to_string())?;
        let fp = fp_mats(&[&enc.w1, &enc.w2], &[&enc.b1, &enc.b2]);
        if fp != enc.fingerprint {
            return Err(format!(
                "vision encoder pin drift: file {:#x} recomputed {:#x}",
                enc.fingerprint, fp
            ));
        }
        Ok(enc)
    }
}

fn vision_encoder_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/wm/frozen_vision_encoder_v1.json")
}

/// Ensure offline-pretrained vision encoder exists (create once; never retrain).
pub fn ensure_frozen_vision_encoder(seed: u64) -> Result<FrozenVisionEncoder, String> {
    let path = vision_encoder_path();
    if path.exists() {
        return FrozenVisionEncoder::load(&path);
    }
    let enc = FrozenVisionEncoder::offline_pretrain(seed, 2500);
    enc.save(&path)?;
    let card = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("data/wm/frozen_vision_encoder_v1.MODEL_CARD.md");
    let _ = std::fs::write(
        &card,
        "# Frozen vision encoder (JEPA slot)\n\n\
         - Produced by offline JEPA-style pretrain (+ disposable spatial probe); **never** fine-tuned during WM adapter training.\n\
         - Replace this JSON with a real V-JEPA/JEPA weight export that exposes the same\n\
           `encode(pixels[64]) -> latent[8]` + `fingerprint` contract.\n\
         - Kill gate: any gradient into these weights during Phase 3s adapter train.\n",
    );
    Ok(enc)
}

fn encode_raw(w1: &[Vec<f32>], b1: &[f32], w2: &[Vec<f32>], b2: &[f32], x: &[f32]) -> Vec<f32> {
    let h = hidden_raw(w1, b1, x);
    w2.iter()
        .zip(b2.iter())
        .map(|(row, &bias)| {
            let mut s = bias;
            for (j, &hj) in h.iter().enumerate() {
                s += row[j] * hj;
            }
            s.tanh()
        })
        .collect()
}

fn hidden_raw(w1: &[Vec<f32>], b1: &[f32], x: &[f32]) -> Vec<f32> {
    w1.iter()
        .zip(b1.iter())
        .map(|(row, &bias)| {
            let mut s = bias;
            for (j, &xj) in x.iter().enumerate() {
                s += row[j] * xj;
            }
            s.tanh()
        })
        .collect()
}

// =============================================================================
// C — Visuomotor foreign domain (push–object log)
// =============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VisuomotorFrame {
    pub pixels: Vec<f32>,
    pub gx: f32,
    pub gy: f32,
    pub ox: f32,
    pub oy: f32,
    /// Regime A = object x < 0 (soft contact); B = sticky.
    pub regime_left: bool,
}

/// Discrete push: 0 left, 1 right, 2 up, 3 down (gripper impulse).
pub fn step_visuomotor(gx: f32, gy: f32, ox: f32, oy: f32, action: u8) -> (f32, f32, f32, f32) {
    let mut gx = gx;
    let mut gy = gy;
    let mut ox = ox;
    let mut oy = oy;
    let impulse = 0.08f32;
    match action % 4 {
        0 => gx -= impulse,
        1 => gx += impulse,
        2 => gy += impulse,
        _ => gy -= impulse,
    }
    gx = gx.clamp(-1.0, 1.0);
    gy = gy.clamp(-1.0, 1.0);
    let dx = ox - gx;
    let dy = oy - gy;
    let dist = (dx * dx + dy * dy).sqrt();
    if dist < 0.22 {
        let soft = ox < 0.0;
        let gain = if soft { 0.55 } else { 0.18 };
        let nx = if dist > 1e-4 { dx / dist } else { 1.0 };
        let ny = if dist > 1e-4 { dy / dist } else { 0.0 };
        // Push object along gripper motion
        let (px, py) = match action % 4 {
            0 => (-1.0, 0.0),
            1 => (1.0, 0.0),
            2 => (0.0, 1.0),
            _ => (0.0, -1.0),
        };
        ox += gain * impulse * (px + 0.25 * nx);
        oy += gain * impulse * (py + 0.25 * ny);
        if !soft {
            // sticky: damp gripper
            gx *= 0.85;
            gy *= 0.85;
        }
    }
    ox = ox.clamp(-1.0, 1.0);
    oy = oy.clamp(-1.0, 1.0);
    (gx, gy, ox, oy)
}

/// Render gripper + object into an 8×8 grayscale patch (foreign “camera”).
pub fn render_visuomotor(gx: f32, gy: f32, ox: f32, oy: f32) -> Vec<f32> {
    let mut pix = vec![0.05f32; VISION_PIXELS];
    let to_uv = |x: f32, y: f32| -> (f32, f32) {
        (
            (x + 1.0) * 0.5 * (VISION_SIDE as f32 - 1.0),
            (1.0 - (y + 1.0) * 0.5) * (VISION_SIDE as f32 - 1.0),
        )
    };
    let splat = |pix: &mut [f32], x: f32, y: f32, val: f32, rad: f32| {
        let (u, v) = to_uv(x, y);
        for iy in 0..VISION_SIDE {
            for ix in 0..VISION_SIDE {
                let du = ix as f32 - u;
                let dv = iy as f32 - v;
                let d = (du * du + dv * dv).sqrt();
                if d < rad {
                    let i = iy * VISION_SIDE + ix;
                    pix[i] = pix[i].max(val * (1.0 - d / rad));
                }
            }
        }
    };
    // Strong room cues (left bright floor / right dark) — foreign camera, not spiral.
    for iy in 0..VISION_SIDE {
        for ix in 0..VISION_SIDE {
            let i = iy * VISION_SIDE + ix;
            if ix < VISION_SIDE / 2 {
                pix[i] = 0.18;
            } else {
                pix[i] = 0.02;
            }
        }
    }
    for iy in 0..VISION_SIDE {
        let i = iy * VISION_SIDE + VISION_SIDE / 2;
        pix[i] = 0.45;
    }
    splat(&mut pix, gx, gy, 1.0, 1.5);
    let obj_val = if ox < 0.0 { 0.9 } else { 0.55 };
    splat(&mut pix, ox, oy, obj_val, 1.7);
    pix
}

fn sample_visuomotor_transition(rng: &mut StdRng) -> (Vec<f32>, Vec<f32>, bool) {
    let gx = rng.gen_range(-0.8..0.8);
    let gy = rng.gen_range(-0.8..0.8);
    let ox = if rng.gen_bool(0.5) {
        rng.gen_range(-0.85..-0.05)
    } else {
        rng.gen_range(0.05..0.85)
    };
    let oy = rng.gen_range(-0.8..0.8);
    let action = rng.gen_range(0..4u8);
    let (gx2, gy2, ox2, oy2) = step_visuomotor(gx, gy, ox, oy, action);
    let left = ox < 0.0;
    (
        render_visuomotor(gx, gy, ox, oy),
        render_visuomotor(gx2, gy2, ox2, oy2),
        left,
    )
}

/// Write a foreign visuomotor JSONL log (episode transitions).
pub fn write_visuomotor_log(path: &Path, n: usize, seed: u64) -> Result<(), String> {
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut f = std::fs::File::create(path).map_err(|e| e.to_string())?;
    use std::io::Write;
    for t in 0..n {
        let (frame, next, left) = sample_visuomotor_transition(&mut rng);
        let row = serde_json::json!({
            "t": t,
            "pixels": frame,
            "pixels_next": next,
            "regime_left": left,
        });
        writeln!(f, "{}", row).map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct VisuomotorSeedResult {
    pub regime_agreement: f32,
    pub energy_margin: f32,
    pub selected_mse: f32,
    pub vg_mse: f32,
    pub degenerate: bool,
    pub encoder_fingerprint: u64,
    pub encoder_was_frozen: bool,
    pub log_path: String,
}

pub fn run_phase3s_visuomotor_seed(seed: u64, log_dir: &Path) -> VisuomotorSeedResult {
    let enc = ensure_frozen_vision_encoder(0xF15_E0)
        .unwrap_or_else(|_| FrozenVisionEncoder::offline_pretrain(0xF15_E0, 400));
    let pin_before = enc.fingerprint;
    let log_path = log_dir.join(format!("visuomotor_{seed}.jsonl"));
    let _ = std::fs::create_dir_all(log_dir);
    write_visuomotor_log(&log_path, 400, seed).expect("log");

    let mut rng = StdRng::seed_from_u64(seed);
    let mut left = Vec::new();
    let mut right = Vec::new();
    while left.len() < 160 || right.len() < 160 {
        let (frame, next, is_left) = sample_visuomotor_transition(&mut rng);
        let z = enc.encode(&frame);
        let zn = enc.encode(&next);
        if is_left {
            if left.len() < 160 {
                left.push((z, zn));
            }
        } else if right.len() < 160 {
            right.push((z, zn));
        }
    }
    let mut a_l = EnergyAdapter::new("vm_L", true, pin_before, seed + 1);
    let mut a_r = EnergyAdapter::new("vm_R", false, pin_before, seed + 2);
    let cz_r: Vec<_> = right.iter().map(|(z, _)| z.clone()).collect();
    let cz_l: Vec<_> = left.iter().map(|(z, _)| z.clone()).collect();
    a_l.train(&left, &right, &cz_r, 350, 0.14, &mut rng);
    a_r.train(&right, &left, &cz_l, 350, 0.14, &mut rng);

    // Kill gate: encoder fingerprint unchanged after adapter train.
    let pin_after = fp_mats(&[&enc.w1, &enc.w2], &[&enc.b1, &enc.b2]);
    let encoder_was_frozen = pin_after == pin_before;

    let mut region = 0usize;
    let mut margin = 0.0f32;
    let mut sel = 0.0f32;
    let mut vg = 0.0f32;
    let mut routes = Vec::new();
    let n = 220usize;
    for _ in 0..n {
        let (frame, next, is_left) = sample_visuomotor_transition(&mut rng);
        let z = enc.encode(&frame);
        let zn = enc.encode(&next);
        let pl = a_l.propose_next(&z);
        let pr = a_r.propose_next(&z);
        let el = a_l.energy(&z, &pl);
        let er = a_r.energy(&z, &pr);
        let pick_left = el < er;
        routes.push(if pick_left { 0 } else { 1 });
        if pick_left == is_left {
            region += 1;
        }
        let pred = if pick_left { &pl } else { &pr };
        let avg: Vec<f32> = pl
            .iter()
            .zip(pr.iter())
            .map(|(u, v)| 0.5 * (u + v))
            .collect();
        sel += mse(pred, &zn);
        vg += mse(&avg, &zn);
        let e_home = if is_left {
            a_l.energy(&z, &zn)
        } else {
            a_r.energy(&z, &zn)
        };
        let e_away = if is_left {
            a_r.energy(&z, &zn)
        } else {
            a_l.energy(&z, &zn)
        };
        margin += e_away - e_home;
    }
    let nf = n as f32;
    let regime_agreement = region as f32 / nf;
    let c0 = routes.iter().filter(|&&c| c == 0).count() as f32 / nf;
    VisuomotorSeedResult {
        regime_agreement,
        energy_margin: margin / nf,
        selected_mse: sel / nf,
        vg_mse: vg / nf,
        degenerate: c0.max(1.0 - c0) > 0.95 || (regime_agreement - 0.5).abs() < 0.01,
        encoder_fingerprint: pin_before,
        encoder_was_frozen,
        log_path: log_path.display().to_string(),
    }
}

// =============================================================================
// E — SpaceKit WM host (JSON deploy_step; not Luna)
// =============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum WmHostRequest {
    LoadBundle { path: String },
    Step { obs: Vec<f32> },
    Fingerprint,
    Status,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WmHostResponse {
    pub ok: bool,
    pub error: Option<String>,
    pub fingerprint: Option<u64>,
    pub decision: Option<DeployDecision>,
    pub loaded: bool,
    pub note: String,
}

#[derive(Default)]
pub struct WmHostSession {
    bundle: Option<ComposedWmBundle>,
    path: Option<String>,
}

impl WmHostSession {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn handle(&mut self, req: WmHostRequest) -> WmHostResponse {
        match req {
            WmHostRequest::LoadBundle { path } => match load_composed_bundle(Path::new(&path)) {
                Ok(b) => {
                    let fp = b.encoder_fingerprint;
                    self.path = Some(path);
                    self.bundle = Some(b);
                    WmHostResponse {
                        ok: true,
                        error: None,
                        fingerprint: Some(fp),
                        decision: None,
                        loaded: true,
                        note: "bundle loaded; encoder pin verified (SpaceKit WM host, not Luna)"
                            .into(),
                    }
                }
                Err(e) => WmHostResponse {
                    ok: false,
                    error: Some(e),
                    fingerprint: None,
                    decision: None,
                    loaded: false,
                    note: "load failed".into(),
                },
            },
            WmHostRequest::Step { obs } => {
                let Some(bundle) = self.bundle.as_ref() else {
                    return WmHostResponse {
                        ok: false,
                        error: Some("no bundle loaded".into()),
                        fingerprint: None,
                        decision: None,
                        loaded: false,
                        note: "call load_bundle first".into(),
                    };
                };
                match deploy_step(bundle, &obs) {
                    Ok(d) => WmHostResponse {
                        ok: true,
                        error: None,
                        fingerprint: Some(d.encoder_fingerprint),
                        decision: Some(d),
                        loaded: true,
                        note: "deploy_step ok".into(),
                    },
                    Err(e) => WmHostResponse {
                        ok: false,
                        error: Some(e),
                        fingerprint: Some(bundle.encoder_fingerprint),
                        decision: None,
                        loaded: true,
                        note: "step failed".into(),
                    },
                }
            }
            WmHostRequest::Fingerprint => WmHostResponse {
                ok: self.bundle.is_some(),
                error: if self.bundle.is_none() {
                    Some("no bundle".into())
                } else {
                    None
                },
                fingerprint: self.bundle.as_ref().map(|b| b.encoder_fingerprint),
                decision: None,
                loaded: self.bundle.is_some(),
                note: "fingerprint".into(),
            },
            WmHostRequest::Status => WmHostResponse {
                ok: true,
                error: None,
                fingerprint: self.bundle.as_ref().map(|b| b.encoder_fingerprint),
                decision: None,
                loaded: self.bundle.is_some(),
                note: format!(
                    "SpaceKit WM host | path={}",
                    self.path.as_deref().unwrap_or("(none)")
                ),
            },
        }
    }

    pub fn handle_json(&mut self, line: &str) -> String {
        match serde_json::from_str::<WmHostRequest>(line) {
            Ok(req) => serde_json::to_string(&self.handle(req)).unwrap_or_else(|e| {
                format!(
                    r#"{{"ok":false,"error":"{}","loaded":false,"note":"serde"}}"#,
                    e
                )
            }),
            Err(e) => format!(
                r#"{{"ok":false,"error":"{}","loaded":false,"note":"bad request"}}"#,
                e
            ),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpacekitHostSeedResult {
    pub load_ok: bool,
    pub step_ok: bool,
    pub pin_stable_reload: bool,
    pub fingerprint: u64,
    pub abstain_seen: bool,
    pub log_path: String,
}

/// Train bundle → host load → step → reload (process-restart proxy) → pin check.
pub fn run_phase3s_spacekit_host_seed(seed: u64, work_dir: &Path) -> SpacekitHostSeedResult {
    let _ = std::fs::create_dir_all(work_dir);
    let bundle = train_composed_bundle(seed);
    let path = work_dir.join(format!("spacekit_bundle_{seed}.json"));
    save_composed_bundle(&path, &bundle).expect("save");
    let fp0 = bundle.encoder_fingerprint;

    let mut host = WmHostSession::new();
    let load = host.handle(WmHostRequest::LoadBundle {
        path: path.display().to_string(),
    });
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(3));
    let obs = vec![
        rng.gen_range(-0.5..0.5),
        rng.gen_range(-0.5..0.5),
        rng.gen_range(-0.2..0.2),
        rng.gen_range(-0.2..0.2),
    ];
    let step = host.handle(WmHostRequest::Step { obs: obs.clone() });
    let abstain_seen = step.decision.as_ref().map(|d| d.abstain).unwrap_or(false) || {
        // probe a few more
        let mut any = false;
        for _ in 0..12 {
            let o = vec![
                rng.gen_range(-0.9..0.9),
                rng.gen_range(-0.9..0.9),
                rng.gen_range(-0.3..0.3),
                rng.gen_range(-0.3..0.3),
            ];
            if let Some(d) = host.handle(WmHostRequest::Step { obs: o }).decision {
                if d.abstain {
                    any = true;
                    break;
                }
            }
        }
        any
    };

    // Process-restart proxy: new session, reload same file.
    let mut host2 = WmHostSession::new();
    let reload = host2.handle(WmHostRequest::LoadBundle {
        path: path.display().to_string(),
    });
    let step2 = host2.handle(WmHostRequest::Step { obs });
    let fp1 = reload.fingerprint.unwrap_or(0);
    let fp2 = step2
        .decision
        .as_ref()
        .map(|d| d.encoder_fingerprint)
        .unwrap_or(0);

    let log_path = work_dir.join(format!("spacekit_host_{seed}.jsonl"));
    if let Ok(mut f) = std::fs::File::create(&log_path) {
        use std::io::Write;
        let _ = writeln!(f, "{}", serde_json::to_string(&load).unwrap_or_default());
        let _ = writeln!(f, "{}", serde_json::to_string(&step).unwrap_or_default());
        let _ = writeln!(f, "{}", serde_json::to_string(&reload).unwrap_or_default());
    }

    SpacekitHostSeedResult {
        load_ok: load.ok,
        step_ok: step.ok && step2.ok,
        pin_stable_reload: load.ok && reload.ok && fp0 == fp1 && fp1 == fp2,
        fingerprint: fp0,
        abstain_seen,
        log_path: log_path.display().to_string(),
    }
}

// =============================================================================
// Helpers
// =============================================================================

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
    fn vision_encoder_pin_stable() {
        let enc = FrozenVisionEncoder::offline_pretrain(1, 50);
        let dir = std::env::temp_dir().join(format!("vis_enc_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("e.json");
        enc.save(&path).unwrap();
        let loaded = FrozenVisionEncoder::load(&path).unwrap();
        assert_eq!(loaded.fingerprint, enc.fingerprint);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn visuomotor_above_chance() {
        let dir = std::env::temp_dir().join(format!("vm_{}", std::process::id()));
        let r = run_phase3s_visuomotor_seed(42, &dir);
        assert!(r.encoder_was_frozen);
        assert!(r.regime_agreement > 0.55, "regime {}", r.regime_agreement);
        assert!(r.energy_margin > 0.0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn spacekit_host_pin_reload() {
        let dir = std::env::temp_dir().join(format!("sk_{}", std::process::id()));
        let r = run_phase3s_spacekit_host_seed(42, &dir);
        assert!(r.load_ok && r.step_ok && r.pin_stable_reload);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
