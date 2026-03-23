//! MNIST loading and random projection for Split MNIST benchmark.
//! Requires MNIST data in `./data` (train-images-idx3-ubyte.gz etc.) or set base_path.

use mnist::Mnist;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

pub const MNIST_INPUT: usize = 28 * 28;
pub const MNIST_PROJECTED: usize = 64;

/// One sample: input vector (784 raw or 64 after projection) and binary target.
pub type MnistSample = (Vec<f32>, [f32; 1]);

/// Load MNIST and return normalized float images (0-1) and labels. Images are 784-dim.
/// Expects data in directory at `data_path` (e.g. "data" with train-images-idx3-ubyte.gz etc.).
pub fn load_mnist_normalized(data_path: &str) -> (Vec<Vec<f32>>, Vec<u8>, Vec<Vec<f32>>, Vec<u8>) {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = mnist::MnistBuilder::new()
        .label_format_digit()
        .base_path(data_path)
        .finalize();

    let train_images = chunk_and_normalize(&trn_img, MNIST_INPUT);
    let train_labels = trn_lbl;
    let test_images = chunk_and_normalize(&tst_img, MNIST_INPUT);
    let test_labels = tst_lbl;

    (train_images, train_labels, test_images, test_labels)
}

fn chunk_and_normalize(raw: &[u8], size: usize) -> Vec<Vec<f32>> {
    raw.chunks_exact(size)
        .map(|chunk| chunk.iter().map(|&p| p as f32 / 255.0).collect())
        .collect()
}

/// Filter to one digit pair and binary labels. Returns (input 784-dim, target 0 or 1).
pub fn filter_digit_pair(
    images: &[Vec<f32>],
    labels: &[u8],
    digit_a: u8,
    digit_b: u8,
) -> Vec<MnistSample> {
    let mut out = Vec::new();
    for (img, &lbl) in images.iter().zip(labels.iter()) {
        if lbl == digit_a {
            out.push((img.clone(), [0.0]));
        } else if lbl == digit_b {
            out.push((img.clone(), [1.0]));
        }
    }
    out
}

/// Filter to one digit pair, preserving original labels (not binary).
/// Returns (image 784-dim, original_label).
pub fn filter_digit_pair_raw(
    images: &[Vec<f32>],
    labels: &[u8],
    digit_a: u8,
    digit_b: u8,
) -> Vec<(Vec<f32>, u8)> {
    images.iter().zip(labels.iter())
        .filter(|(_, &lbl)| lbl == digit_a || lbl == digit_b)
        .map(|(img, &lbl)| (img.clone(), lbl))
        .collect()
}

/// Fixed random projection 784 -> 64 with L2 normalization. Same projection for all tasks.
pub struct RandomProjection {
    matrix: Vec<Vec<f32>>,
    out_dim: usize,
}

impl RandomProjection {
    pub fn new(in_dim: usize, out_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let matrix = (0..in_dim)
            .map(|_| (0..out_dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        RandomProjection { matrix, out_dim }
    }

    pub fn project(&self, x: &[f32]) -> Vec<f32> {
        let mut y = vec![0.0f32; self.out_dim];
        for (i, row) in self.matrix.iter().enumerate() {
            let xi = x.get(i).copied().unwrap_or(0.0);
            for (j, &w) in row.iter().enumerate() {
                y[j] += xi * w;
            }
        }
        let norm: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for v in &mut y {
                *v /= norm;
            }
        }
        y
    }
}

/// Project a dataset from 784 to 64 dimensions.
pub fn project_dataset(proj: &RandomProjection, data: &[MnistSample]) -> Vec<MnistSample> {
    data.iter()
        .map(|(x, t)| (proj.project(x), *t))
        .collect()
}
