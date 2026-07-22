//! Phase 3u — real V-JEPA export bridge into the frozen encoder slot.
//!
//! Loads a pinned feature bank produced by `scripts/export_vjepa_features.py`
//! (`--mode hf` for Meta V-JEPA 2, `--mode mock` for CI). Adapters train;
//! projector / student / backbone never do.

use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use super::energy_jepa::EnergyAdapter;
use super::jepa_adapters::WM_LATENT_DIM;
use super::wm_open::{render_visuomotor, step_visuomotor, VISION_PIXELS};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VjepaFrame {
    pub pixels: Vec<f32>,
    pub pixels_next: Vec<f32>,
    pub regime_left: bool,
    pub z: Vec<f32>,
    pub z_next: Vec<f32>,
}

/// Pinned V-JEPA export: teacher projector + distilled student for live encode.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrozenVjepaExport {
    pub source_model: String,
    pub export_mode: String,
    pub jepa_dim: usize,
    pub latent_dim: usize,
    pub fingerprint: u64,
    pub projector_w: Vec<Vec<f32>>,
    pub projector_b: Vec<f32>,
    pub student_w1: Vec<Vec<f32>>,
    pub student_b1: Vec<f32>,
    pub student_w2: Vec<Vec<f32>>,
    pub student_b2: Vec<f32>,
    pub note: String,
    pub frames: Vec<VjepaFrame>,
}

impl FrozenVjepaExport {
    pub fn load(path: &Path) -> Result<Self, String> {
        let s = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let exp: Self = serde_json::from_str(&s).map_err(|e| e.to_string())?;
        if exp.latent_dim != WM_LATENT_DIM {
            return Err(format!(
                "latent_dim {} != WM_LATENT_DIM {}",
                exp.latent_dim, WM_LATENT_DIM
            ));
        }
        let fp = exp.recompute_fingerprint();
        if fp != exp.fingerprint {
            return Err(format!(
                "V-JEPA export pin drift: file {:#x} recomputed {:#x}",
                exp.fingerprint, fp
            ));
        }
        Ok(exp)
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
        }
        let json = serde_json::to_string(self).map_err(|e| e.to_string())?;
        std::fs::write(path, json).map_err(|e| e.to_string())
    }

    fn recompute_fingerprint(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.source_model.hash(&mut h);
        self.jepa_dim.hash(&mut h);
        for row in &self.projector_w {
            for &v in row {
                v.to_bits().hash(&mut h);
            }
        }
        for &v in &self.projector_b {
            v.to_bits().hash(&mut h);
        }
        for row in &self.student_w1 {
            for &v in row {
                v.to_bits().hash(&mut h);
            }
        }
        for &v in &self.student_b1 {
            v.to_bits().hash(&mut h);
        }
        for row in &self.student_w2 {
            for &v in row {
                v.to_bits().hash(&mut h);
            }
        }
        for &v in &self.student_b2 {
            v.to_bits().hash(&mut h);
        }
        h.finish()
    }

    /// Live encode via frozen distilled student (no teacher / no fine-tune).
    pub fn encode(&self, pixels: &[f32]) -> Vec<f32> {
        assert_eq!(pixels.len(), VISION_PIXELS);
        let h: Vec<f32> = self
            .student_w1
            .iter()
            .zip(self.student_b1.iter())
            .map(|(row, &bias)| {
                let mut s = bias;
                for (j, &x) in pixels.iter().enumerate() {
                    s += row[j] * x;
                }
                s.tanh()
            })
            .collect();
        self.student_w2
            .iter()
            .zip(self.student_b2.iter())
            .map(|(row, &bias)| {
                let mut s = bias;
                for (j, &x) in h.iter().enumerate() {
                    s += row[j] * x;
                }
                s.tanh()
            })
            .collect()
    }
}

fn default_export_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/wm/vjepa_export_v1.json")
}

/// Ensure export exists: prefer on-disk artifact; else build Rust mock bank (same schema).
pub fn ensure_vjepa_export(seed: u64) -> Result<FrozenVjepaExport, String> {
    let path = default_export_path();
    if path.exists() {
        return FrozenVjepaExport::load(&path);
    }
    let exp = build_rust_mock_export(seed, 256);
    exp.save(&path)?;
    let card = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("data/wm/vjepa_export_v1.MODEL_CARD.md");
    let _ = std::fs::write(
        &card,
        "# V-JEPA export bank (Phase 3u)\n\n\
         - Replace via: `python3 scripts/export_vjepa_features.py --mode hf --model facebook/vjepa2-vitl-fpc64-256`\n\
         - Or refresh mock: `python3 scripts/export_vjepa_features.py --mode mock`\n\
         - Kill gate: fine-tuning projector, student, or backbone.\n",
    );
    Ok(exp)
}

/// Deterministic mock teacher export (identical contract to the Python mock path).
pub fn build_rust_mock_export(seed: u64, n: usize) -> FrozenVjepaExport {
    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x4EFA).wrapping_add(9));
    let jepa_dim = 256usize;
    // Frozen mock teacher weights
    let tw1 = rand_mat(128, VISION_PIXELS, 0.15, &mut StdRng::seed_from_u64(0x4EFA));
    let tb1 = vec![0.0f32; 128];
    let tw2 = rand_mat(jepa_dim, 128, 0.1, &mut StdRng::seed_from_u64(0x4EFA + 1));
    let tb2 = vec![0.0f32; jepa_dim];

    let mut teacher_zs = Vec::new();
    let mut frames_raw = Vec::new();
    for _ in 0..n {
        let gx = rng.gen_range(-0.8..0.8);
        let gy = rng.gen_range(-0.8..0.8);
        let ox = if rng.gen_bool(0.5) {
            rng.gen_range(-0.85..-0.05)
        } else {
            rng.gen_range(0.05..0.85)
        };
        let oy = rng.gen_range(-0.8..0.8);
        let action = rng.gen_range(0..4u8);
        let pix = render_visuomotor(gx, gy, ox, oy);
        let (gx2, gy2, ox2, oy2) = step_visuomotor(gx, gy, ox, oy, action);
        let pix_n = render_visuomotor(gx2, gy2, ox2, oy2);
        let zt = mock_teacher_encode(&tw1, &tb1, &tw2, &tb2, &pix);
        let ztn = mock_teacher_encode(&tw1, &tb1, &tw2, &tb2, &pix_n);
        teacher_zs.push(zt.clone());
        frames_raw.push((pix, pix_n, ox < 0.0, zt, ztn));
    }

    let (pw, pb) = fit_projector(&teacher_zs, WM_LATENT_DIM);
    let mut targets = Vec::new();
    let mut pixels = Vec::new();
    let mut frames = Vec::new();
    for (pix, pix_n, left, zt, ztn) in frames_raw {
        let z = project(&pw, &pb, &zt);
        let zn = project(&pw, &pb, &ztn);
        targets.push(z.clone());
        pixels.push(pix.clone());
        frames.push(VjepaFrame {
            pixels: pix,
            pixels_next: pix_n,
            regime_left: left,
            z,
            z_next: zn,
        });
    }
    let (sw1, sb1, sw2, sb2) = fit_student(&pixels, &targets, 500, seed + 3);

    let mut exp = FrozenVjepaExport {
        source_model: "mock-vjepa-teacher-v1".into(),
        export_mode: "mock".into(),
        jepa_dim,
        latent_dim: WM_LATENT_DIM,
        fingerprint: 0,
        projector_w: pw,
        projector_b: pb,
        student_w1: sw1,
        student_b1: sb1,
        student_w2: sw2,
        student_b2: sb2,
        note: "Frozen V-JEPA export bank (Rust mock). Replace with HF export for Meta weights."
            .into(),
        frames,
    };
    exp.fingerprint = exp.recompute_fingerprint();
    exp
}

fn mock_teacher_encode(
    w1: &[Vec<f32>],
    b1: &[f32],
    w2: &[Vec<f32>],
    b2: &[f32],
    x: &[f32],
) -> Vec<f32> {
    let h: Vec<f32> = w1
        .iter()
        .zip(b1.iter())
        .map(|(row, &bias)| {
            let mut s = bias;
            for (j, &xj) in x.iter().enumerate() {
                s += row[j] * xj;
            }
            s.tanh()
        })
        .collect();
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

fn fit_projector(zs: &[Vec<f32>], out_dim: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let d = zs[0].len();
    let n = zs.len() as f32;
    let mut mean = vec![0.0f32; d];
    for z in zs {
        for i in 0..d {
            mean[i] += z[i] / n;
        }
    }
    // Power-iteration style top directions (cheap PCA stand-in).
    let mut rng = StdRng::seed_from_u64(0xC0_A1);
    let mut w = Vec::with_capacity(out_dim);
    for _ in 0..out_dim {
        let mut v: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();
        for _ in 0..40 {
            let mut acc = vec![0.0f32; d];
            for z in zs {
                let mut dot = 0.0f32;
                for i in 0..d {
                    dot += (z[i] - mean[i]) * v[i];
                }
                for i in 0..d {
                    acc[i] += (z[i] - mean[i]) * dot;
                }
            }
            let norm = acc.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            for i in 0..d {
                v[i] = acc[i] / norm;
            }
        }
        w.push(v);
    }
    let mut b = vec![0.0f32; out_dim];
    for i in 0..out_dim {
        let mut s = 0.0f32;
        for j in 0..d {
            s += w[i][j] * mean[j];
        }
        b[i] = -s;
    }
    (w, b)
}

fn project(w: &[Vec<f32>], b: &[f32], z: &[f32]) -> Vec<f32> {
    w.iter()
        .zip(b.iter())
        .map(|(row, &bias)| {
            let mut s = bias;
            for (j, &zj) in z.iter().enumerate() {
                s += row[j] * zj;
            }
            s.tanh()
        })
        .collect()
}

fn fit_student(
    pixels: &[Vec<f32>],
    targets: &[Vec<f32>],
    steps: usize,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<f32>, Vec<Vec<f32>>, Vec<f32>) {
    let hidden = 48;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut w1 = rand_mat(hidden, VISION_PIXELS, 0.2, &mut rng);
    let mut b1 = vec![0.0f32; hidden];
    let mut w2 = rand_mat(WM_LATENT_DIM, hidden, 0.2, &mut rng);
    let mut b2 = vec![0.0f32; WM_LATENT_DIM];
    let lr = 0.05f32;
    let n = pixels.len();
    for _ in 0..steps {
        let i = rng.gen_range(0..n);
        let x = &pixels[i];
        let t = &targets[i];
        let h: Vec<f32> = w1
            .iter()
            .zip(b1.iter())
            .map(|(row, &bias)| {
                let mut s = bias;
                for (j, &xj) in x.iter().enumerate() {
                    s += row[j] * xj;
                }
                s.tanh()
            })
            .collect();
        let y: Vec<f32> = w2
            .iter()
            .zip(b2.iter())
            .map(|(row, &bias)| {
                let mut s = bias;
                for (j, &hj) in h.iter().enumerate() {
                    s += row[j] * hj;
                }
                s.tanh()
            })
            .collect();
        let mut dy = vec![0.0f32; WM_LATENT_DIM];
        for o in 0..WM_LATENT_DIM {
            dy[o] = 2.0 * (y[o] - t[o]) / WM_LATENT_DIM as f32 * (1.0 - y[o] * y[o]);
        }
        for o in 0..WM_LATENT_DIM {
            b2[o] -= lr * dy[o];
            for j in 0..hidden {
                w2[o][j] -= lr * dy[o] * h[j];
            }
        }
        let mut dh = vec![0.0f32; hidden];
        for j in 0..hidden {
            let mut s = 0.0f32;
            for o in 0..WM_LATENT_DIM {
                s += w2[o][j] * dy[o];
            }
            dh[j] = s * (1.0 - h[j] * h[j]);
        }
        for j in 0..hidden {
            b1[j] -= lr * dh[j];
            for i in 0..VISION_PIXELS {
                w1[j][i] -= lr * dh[j] * x[i];
            }
        }
    }
    (w1, b1, w2, b2)
}

fn rand_mat(rows: usize, cols: usize, scale: f32, rng: &mut StdRng) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|_| (0..cols).map(|_| rng.gen_range(-scale..scale)).collect())
        .collect()
}

fn mse(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for (u, v) in a.iter().zip(b.iter()) {
        let d = u - v;
        s += d * d;
    }
    s / a.len().max(1) as f32
}

#[derive(Clone, Debug)]
pub struct VjepaWmSeedResult {
    pub source_model: String,
    pub export_mode: String,
    pub regime_agreement: f32,
    pub energy_margin: f32,
    pub selected_mse: f32,
    pub vg_mse: f32,
    pub degenerate: bool,
    pub encoder_frozen: bool,
    pub fingerprint: u64,
}

pub fn run_phase3u_vjepa_seed(seed: u64) -> VjepaWmSeedResult {
    let exp = ensure_vjepa_export(0x4E_FA_E0).unwrap_or_else(|_| build_rust_mock_export(seed, 200));
    let pin_before = exp.fingerprint;
    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(31).wrapping_add(5));

    let mut left = Vec::new();
    let mut right = Vec::new();
    for fr in &exp.frames {
        if fr.regime_left {
            if left.len() < 140 {
                left.push((fr.z.clone(), fr.z_next.clone()));
            }
        } else if right.len() < 140 {
            right.push((fr.z.clone(), fr.z_next.clone()));
        }
    }
    // Top up with live student encodes if bank thin.
    while left.len() < 140 || right.len() < 140 {
        let gx = rng.gen_range(-0.8..0.8);
        let gy = rng.gen_range(-0.8..0.8);
        let ox = if rng.gen_bool(0.5) {
            rng.gen_range(-0.85..-0.05)
        } else {
            rng.gen_range(0.05..0.85)
        };
        let oy = rng.gen_range(-0.8..0.8);
        let action = rng.gen_range(0..4u8);
        let pix = render_visuomotor(gx, gy, ox, oy);
        let (gx2, gy2, ox2, oy2) = step_visuomotor(gx, gy, ox, oy, action);
        let pix_n = render_visuomotor(gx2, gy2, ox2, oy2);
        let z = exp.encode(&pix);
        let zn = exp.encode(&pix_n);
        if ox < 0.0 {
            if left.len() < 140 {
                left.push((z, zn));
            }
        } else if right.len() < 140 {
            right.push((z, zn));
        }
    }

    let mut a_l = EnergyAdapter::new("vjepa_L", true, pin_before, seed + 1);
    let mut a_r = EnergyAdapter::new("vjepa_R", false, pin_before, seed + 2);
    let cz_r: Vec<_> = right.iter().map(|(z, _)| z.clone()).collect();
    let cz_l: Vec<_> = left.iter().map(|(z, _)| z.clone()).collect();
    a_l.train(&left, &right, &cz_r, 350, 0.14, &mut rng);
    a_r.train(&right, &left, &cz_l, 350, 0.14, &mut rng);

    let pin_after = exp.recompute_fingerprint();
    let encoder_frozen = pin_after == pin_before;

    let mut region = 0usize;
    let mut margin = 0.0f32;
    let mut sel = 0.0f32;
    let mut vg = 0.0f32;
    let mut routes = Vec::new();
    let n = 200usize;
    for _ in 0..n {
        let gx = rng.gen_range(-0.8..0.8);
        let gy = rng.gen_range(-0.8..0.8);
        let ox = if rng.gen_bool(0.5) {
            rng.gen_range(-0.85..-0.05)
        } else {
            rng.gen_range(0.05..0.85)
        };
        let oy = rng.gen_range(-0.8..0.8);
        let action = rng.gen_range(0..4u8);
        let pix = render_visuomotor(gx, gy, ox, oy);
        let (gx2, gy2, ox2, oy2) = step_visuomotor(gx, gy, ox, oy, action);
        let pix_n = render_visuomotor(gx2, gy2, ox2, oy2);
        let z = exp.encode(&pix);
        let zn = exp.encode(&pix_n);
        let left_reg = ox < 0.0;
        let pl = a_l.propose_next(&z);
        let pr = a_r.propose_next(&z);
        let el = a_l.energy(&z, &pl);
        let er = a_r.energy(&z, &pr);
        let pick_left = el < er;
        routes.push(if pick_left { 0 } else { 1 });
        if pick_left == left_reg {
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
        let e_home = if left_reg {
            a_l.energy(&z, &zn)
        } else {
            a_r.energy(&z, &zn)
        };
        let e_away = if left_reg {
            a_r.energy(&z, &zn)
        } else {
            a_l.energy(&z, &zn)
        };
        margin += e_away - e_home;
    }
    let nf = n as f32;
    let regime_agreement = region as f32 / nf;
    let c0 = routes.iter().filter(|&&c| c == 0).count() as f32 / nf;
    VjepaWmSeedResult {
        source_model: exp.source_model.clone(),
        export_mode: exp.export_mode.clone(),
        regime_agreement,
        energy_margin: margin / nf,
        selected_mse: sel / nf,
        vg_mse: vg / nf,
        degenerate: c0.max(1.0 - c0) > 0.95 || (regime_agreement - 0.5).abs() < 0.01,
        encoder_frozen,
        fingerprint: pin_before,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_pin_and_transfer() {
        let exp = build_rust_mock_export(42, 80);
        let dir = std::env::temp_dir().join(format!("vjepa_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("e.json");
        exp.save(&path).unwrap();
        let loaded = FrozenVjepaExport::load(&path).unwrap();
        assert_eq!(loaded.fingerprint, exp.fingerprint);
        let r = run_phase3u_vjepa_seed(42);
        assert!(r.encoder_frozen);
        assert!(r.regime_agreement > 0.55, "regime {}", r.regime_agreement);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
