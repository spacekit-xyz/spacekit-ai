//! PathMNIST: colorectal cancer histology classification through Cl(1,7).
//!
//! Loads the MedMNIST v2 PathMNIST dataset (28×28 RGB tissue patches,
//! 9 classes) and classifies using the same Clifford algebra encoder
//! that handles MNIST digits and language.
//!
//! Data format: raw uint8 binary files extracted from PathMNIST zip.
//!   images: N × 28 × 28 × 3 bytes (RGB, row-major)
//!   labels: N × 1 bytes (class 0–8)

use rand::rngs::StdRng;
use rand::Rng;
use std::fs;
use std::path::Path;

pub const PATH_IMAGE_H: usize = 28;
pub const PATH_IMAGE_W: usize = 28;
pub const PATH_CHANNELS: usize = 3;
pub const PATH_RGB_DIM: usize = PATH_IMAGE_H * PATH_IMAGE_W * PATH_CHANNELS; // 2352
pub const PATH_GRAY_DIM: usize = PATH_IMAGE_H * PATH_IMAGE_W; // 784
pub const PATH_NUM_CLASSES: usize = 9;

pub const CLASS_NAMES: [&str; 9] = [
    "adipose",        // 0 — fat cells
    "background",     // 1 — no tissue
    "debris",         // 2 — necrotic/dead
    "lymphocytes",    // 3 — immune response
    "mucus",          // 4 — secretory
    "smooth_muscle",  // 5 — structural
    "normal_mucosa",  // 6 — healthy baseline
    "cancer_stroma",  // 7 — cancer indicator
    "adenocarcinoma", // 8 — primary cancer
];

pub fn is_cancer_class(label: u8) -> bool {
    label == 7 || label == 8
}

pub struct PathMNISTDataset {
    pub images_rgb: Vec<Vec<f32>>,
    pub images_gray: Vec<Vec<f32>>,
    pub labels: Vec<u8>,
    pub n: usize,
    pub height: usize,
    pub width: usize,
}

impl PathMNISTDataset {
    /// Load a split from raw binary .npy files (headerless uint8).
    /// Auto-detects resolution from file size.
    pub fn load(data_dir: &Path, split: &str) -> Self {
        Self::load_with_resolution(data_dir, split, PATH_IMAGE_H, PATH_IMAGE_W)
    }

    /// Load with explicit resolution (for 64x64, 224x224, etc.)
    pub fn load_with_resolution(data_dir: &Path, split: &str, h: usize, w: usize) -> Self {
        let img_path = data_dir.join(format!("{}_images.npy", split));
        let lbl_path = data_dir.join(format!("{}_labels.npy", split));

        let raw_img = fs::read(&img_path)
            .unwrap_or_else(|e| panic!("Cannot read {}: {}", img_path.display(), e));
        let raw_lbl = fs::read(&lbl_path)
            .unwrap_or_else(|e| panic!("Cannot read {}: {}", lbl_path.display(), e));

        let n = raw_lbl.len();
        let rgb_dim = h * w * PATH_CHANNELS;
        assert_eq!(
            raw_img.len(),
            n * rgb_dim,
            "Image file size mismatch: {} bytes for {} samples at {}x{} (expected {})",
            raw_img.len(),
            n,
            h,
            w,
            n * rgb_dim
        );

        let labels: Vec<u8> = raw_lbl.to_vec();

        let mut images_rgb: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut images_gray: Vec<Vec<f32>> = Vec::with_capacity(n);

        for rgb in raw_img.chunks_exact(rgb_dim) {
            let rgb_f: Vec<f32> = rgb.iter().map(|&b| b as f32 / 255.0).collect();
            let gray: Vec<f32> = rgb
                .chunks_exact(3)
                .map(|px| {
                    0.299 * px[0] as f32 / 255.0
                        + 0.587 * px[1] as f32 / 255.0
                        + 0.114 * px[2] as f32 / 255.0
                })
                .collect();
            images_rgb.push(rgb_f);
            images_gray.push(gray);
        }

        assert_eq!(images_rgb.len(), n);
        PathMNISTDataset {
            images_rgb,
            images_gray,
            labels,
            n,
            height: h,
            width: w,
        }
    }

    pub fn class_distribution(&self) -> [usize; PATH_NUM_CLASSES] {
        let mut counts = [0usize; PATH_NUM_CLASSES];
        for &l in &self.labels {
            if (l as usize) < PATH_NUM_CLASSES {
                counts[l as usize] += 1;
            }
        }
        counts
    }
}

/// Histology-specific augmentation: flips + 90° rotations + brightness.
/// Tissue patches have no canonical orientation.
pub fn augment_histology(image: &[f32], rng: &mut StdRng) -> Vec<f32> {
    let mut img = image.to_vec();

    // Horizontal flip (50%)
    if rng.gen_bool(0.5) {
        for y in 0..PATH_IMAGE_H {
            let row_start = y * PATH_IMAGE_W;
            img[row_start..row_start + PATH_IMAGE_W].reverse();
        }
    }

    // Vertical flip (50%)
    if rng.gen_bool(0.5) {
        for y in 0..PATH_IMAGE_H / 2 {
            for x in 0..PATH_IMAGE_W {
                let a = y * PATH_IMAGE_W + x;
                let b = (PATH_IMAGE_H - 1 - y) * PATH_IMAGE_W + x;
                img.swap(a, b);
            }
        }
    }

    // 90° rotation (0, 90, 180, 270)
    let rot: u32 = rng.gen_range(0..4);
    if rot > 0 {
        let mut rotated = vec![0.0f32; PATH_GRAY_DIM];
        for y in 0..PATH_IMAGE_H {
            for x in 0..PATH_IMAGE_W {
                let (ny, nx) = match rot {
                    1 => (x, PATH_IMAGE_H - 1 - y),                    // 90°
                    2 => (PATH_IMAGE_H - 1 - y, PATH_IMAGE_W - 1 - x), // 180°
                    3 => (PATH_IMAGE_W - 1 - x, y),                    // 270°
                    _ => unreachable!(),
                };
                rotated[ny * PATH_IMAGE_W + nx] = img[y * PATH_IMAGE_W + x];
            }
        }
        img = rotated;
    }

    // Brightness jitter (±10%) — simulates staining batch variation
    let brightness: f32 = 1.0 + rng.gen_range(-0.1..0.1);
    for v in &mut img {
        *v = (*v * brightness).clamp(0.0, 1.0);
    }

    img
}

/// RGB augmentation: flips + 90° rotations + per-channel color jitter.
/// Operates on 2352D (28×28×3) interleaved RGB images.
pub fn augment_histology_rgb(image: &[f32], rng: &mut StdRng) -> Vec<f32> {
    let w = PATH_IMAGE_W;
    let h = PATH_IMAGE_H;
    let mut img = image.to_vec();

    // Horizontal flip (50%) — swap 3-channel pixels
    if rng.gen_bool(0.5) {
        for y in 0..h {
            for x in 0..w / 2 {
                let a = (y * w + x) * 3;
                let b = (y * w + (w - 1 - x)) * 3;
                for c in 0..3 {
                    img.swap(a + c, b + c);
                }
            }
        }
    }

    // Vertical flip (50%)
    if rng.gen_bool(0.5) {
        for y in 0..h / 2 {
            for x in 0..w {
                let a = (y * w + x) * 3;
                let b = ((h - 1 - y) * w + x) * 3;
                for c in 0..3 {
                    img.swap(a + c, b + c);
                }
            }
        }
    }

    // 90° rotation
    let rot: u32 = rng.gen_range(0..4);
    if rot > 0 {
        let mut rotated = vec![0.0f32; PATH_RGB_DIM];
        for y in 0..h {
            for x in 0..w {
                let (ny, nx) = match rot {
                    1 => (x, h - 1 - y),
                    2 => (h - 1 - y, w - 1 - x),
                    3 => (w - 1 - x, y),
                    _ => unreachable!(),
                };
                let src = (y * w + x) * 3;
                let dst = (ny * w + nx) * 3;
                rotated[dst] = img[src];
                rotated[dst + 1] = img[src + 1];
                rotated[dst + 2] = img[src + 2];
            }
        }
        img = rotated;
    }

    // Per-channel color jitter (±10% independently) — simulates staining variation
    let jr: f32 = 1.0 + rng.gen_range(-0.1..0.1);
    let jg: f32 = 1.0 + rng.gen_range(-0.1..0.1);
    let jb: f32 = 1.0 + rng.gen_range(-0.1..0.1);
    for i in (0..img.len()).step_by(3) {
        img[i] = (img[i] * jr).clamp(0.0, 1.0);
        img[i + 1] = (img[i + 1] * jg).clamp(0.0, 1.0);
        img[i + 2] = (img[i + 2] * jb).clamp(0.0, 1.0);
    }

    img
}

/// Split 2352D interleaved RGB into three 784D channels.
pub fn split_rgb_channels(rgb: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = rgb.len() / 3;
    let mut r = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for chunk in rgb.chunks_exact(3) {
        r.push(chunk[0]);
        g.push(chunk[1]);
        b.push(chunk[2]);
    }
    (r, g, b)
}

/// Cancer detection metrics: sensitivity, specificity, F1.
pub struct CancerMetrics {
    pub sensitivity: f32,
    pub specificity: f32,
    pub f1: f32,
    pub stroma_recall: f32, // class 7
    pub adeno_recall: f32,  // class 8
}

pub fn compute_cancer_metrics(predictions: &[u8], labels: &[u8]) -> CancerMetrics {
    let mut tp = 0u32;
    let mut fp = 0u32;
    let mut fn_ = 0u32;
    let mut tn = 0u32;
    let mut stroma_correct = 0u32;
    let mut stroma_total = 0u32;
    let mut adeno_correct = 0u32;
    let mut adeno_total = 0u32;

    for (&pred, &label) in predictions.iter().zip(labels.iter()) {
        let pred_cancer = is_cancer_class(pred);
        let label_cancer = is_cancer_class(label);
        match (pred_cancer, label_cancer) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fn_ += 1,
            (false, false) => tn += 1,
        }
        if label == 7 {
            stroma_total += 1;
            if pred == 7 {
                stroma_correct += 1;
            }
        }
        if label == 8 {
            adeno_total += 1;
            if pred == 8 {
                adeno_correct += 1;
            }
        }
    }

    let sensitivity = if tp + fn_ > 0 {
        tp as f32 / (tp + fn_) as f32
    } else {
        0.0
    };
    let specificity = if tn + fp > 0 {
        tn as f32 / (tn + fp) as f32
    } else {
        0.0
    };
    let f1 = if 2 * tp + fp + fn_ > 0 {
        2.0 * tp as f32 / (2 * tp + fp + fn_) as f32
    } else {
        0.0
    };
    let stroma_recall = if stroma_total > 0 {
        stroma_correct as f32 / stroma_total as f32
    } else {
        0.0
    };
    let adeno_recall = if adeno_total > 0 {
        adeno_correct as f32 / adeno_total as f32
    } else {
        0.0
    };

    CancerMetrics {
        sensitivity,
        specificity,
        f1,
        stroma_recall,
        adeno_recall,
    }
}
