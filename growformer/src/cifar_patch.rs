//! Frozen CIFAR patch encoder (Phase 4f).
//!
//! Non-trainable local patch bank + random projection. Fingerprint-pinned;
//! only promote–freeze adapters (task MLPs) train. Better than gray→64d.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::cifar10::{Cifar10Sample, CIFAR_SIDE};
use crate::types::Sample;

pub const PATCH: usize = 8;
pub const PATCH_PIXELS: usize = PATCH * PATCH * 3; // 192
pub const N_PATCHES: usize = (CIFAR_SIDE / PATCH) * (CIFAR_SIDE / PATCH); // 16
pub const PATCH_OUT: usize = 8;
pub const FROZEN_FEAT: usize = N_PATCHES * PATCH_OUT; // 128

#[derive(Clone, Debug)]
pub struct FrozenCifarPatchEncoder {
    /// PATCH_PIXELS × PATCH_OUT projection (row-major).
    weights: Vec<f32>,
    pub fingerprint: u64,
    pub out_dim: usize,
}

fn fp_weights(w: &[f32]) -> u64 {
    let mut h = DefaultHasher::new();
    for &x in w {
        x.to_bits().hash(&mut h);
    }
    h.finish()
}

impl FrozenCifarPatchEncoder {
    pub fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0xC1F4_A701));
        let n = PATCH_PIXELS * PATCH_OUT;
        let scale = (2.0 / PATCH_PIXELS as f32).sqrt();
        let weights: Vec<f32> = (0..n).map(|_| rng.gen_range(-scale..scale)).collect();
        let fingerprint = fp_weights(&weights);
        Self {
            weights,
            fingerprint,
            out_dim: FROZEN_FEAT,
        }
    }

    pub fn verify_pin(&self, pin: u64) -> bool {
        self.fingerprint == pin && fp_weights(&self.weights) == pin
    }

    /// Encode planar RGB CIFAR sample → L2-normalized `FROZEN_FEAT` vector.
    pub fn encode(&self, s: &Cifar10Sample) -> Vec<f32> {
        let mut out = vec![0.0f32; FROZEN_FEAT];
        let mut p = 0usize;
        for py in 0..(CIFAR_SIDE / PATCH) {
            for px in 0..(CIFAR_SIDE / PATCH) {
                let mut patch = [0.0f32; PATCH_PIXELS];
                let mut k = 0usize;
                for c in 0..3 {
                    for dy in 0..PATCH {
                        for dx in 0..PATCH {
                            let y = py * PATCH + dy;
                            let x = px * PATCH + dx;
                            patch[k] = s.pixels[c * CIFAR_SIDE * CIFAR_SIDE + y * CIFAR_SIDE + x];
                            k += 1;
                        }
                    }
                }
                // Local contrast normalize
                let mean = patch.iter().sum::<f32>() / PATCH_PIXELS as f32;
                let var = patch.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>()
                    / PATCH_PIXELS as f32;
                let std = (var + 1e-4).sqrt();
                for v in &mut patch {
                    *v = (*v - mean) / std;
                }
                let base = p * PATCH_OUT;
                for o in 0..PATCH_OUT {
                    let mut acc = 0.0f32;
                    for i in 0..PATCH_PIXELS {
                        acc += patch[i] * self.weights[i * PATCH_OUT + o];
                    }
                    out[base + o] = acc.tanh();
                }
                p += 1;
            }
        }
        let n = out.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        for v in &mut out {
            *v /= n;
        }
        out
    }
}

pub fn filter_class_pair_frozen(
    data: &[Cifar10Sample],
    a: u8,
    b: u8,
    enc: &FrozenCifarPatchEncoder,
    limit: usize,
) -> Vec<Sample> {
    let mut out = Vec::new();
    for s in data {
        if s.label == a {
            out.push((enc.encode(s), [0.0]));
        } else if s.label == b {
            out.push((enc.encode(s), [1.0]));
        }
        if out.len() >= limit {
            break;
        }
    }
    out
}
