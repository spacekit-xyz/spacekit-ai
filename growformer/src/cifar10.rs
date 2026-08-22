//! CIFAR-10 flat binary loader (torchvision export).
//!
//! Record layout (3073 bytes): `<label u8><3072 RGB pixels>` (R,G,B planes).
//! Produced by [`scripts/export_cifar10.py`](../../scripts/export_cifar10.py).

use std::fs::File;
use std::io::{Read, Result as IoResult};
use std::path::{Path, PathBuf};

use crate::mnist::RandomProjection;
use crate::types::Sample;

pub const CIFAR_SIDE: usize = 32;
pub const CIFAR_PIXELS: usize = CIFAR_SIDE * CIFAR_SIDE * 3; // 3072
pub const CIFAR10_RECORD: usize = 1 + CIFAR_PIXELS; // 3073
pub const CIFAR_PROJECTED: usize = 64;
pub const CIFAR_GRAY: usize = CIFAR_SIDE * CIFAR_SIDE; // 1024

#[derive(Clone, Debug)]
pub struct Cifar10Sample {
    pub label: u8,
    pub pixels: Vec<f32>, // 3072, [0,1]
}

fn resolve_dir(root: &Path) -> Option<PathBuf> {
    let candidates = [
        root.join("cifar10_export"),
        root.to_path_buf(),
        root.join("data/cifar10_export"),
    ];
    for c in candidates {
        if c.join("train.bin").exists() {
            return Some(c);
        }
    }
    None
}

pub fn cifar10_available(root: &Path) -> bool {
    resolve_dir(root).is_some()
}

fn load_bin(path: &Path) -> IoResult<Vec<Cifar10Sample>> {
    let mut f = File::open(path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    let n = buf.len() / CIFAR10_RECORD;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let o = i * CIFAR10_RECORD;
        let label = buf[o];
        let mut pixels = Vec::with_capacity(CIFAR_PIXELS);
        for j in 0..CIFAR_PIXELS {
            pixels.push(buf[o + 1 + j] as f32 / 255.0);
        }
        out.push(Cifar10Sample { label, pixels });
    }
    Ok(out)
}

pub fn load_cifar10(root: &Path) -> Result<(Vec<Cifar10Sample>, Vec<Cifar10Sample>), String> {
    let dir = resolve_dir(root).ok_or_else(|| {
        String::from(
            "CIFAR-10 export not found (expect train.bin). Run: python3 scripts/export_cifar10.py",
        )
    })?;
    let train = load_bin(&dir.join("train.bin")).map_err(|e| e.to_string())?;
    let test = load_bin(&dir.join("test.bin")).map_err(|e| e.to_string())?;
    Ok((train, test))
}

pub fn project_cifar_gray(proj: &RandomProjection, s: &Cifar10Sample) -> Vec<f32> {
    let mut gray = vec![0.0f32; CIFAR_GRAY];
    for i in 0..CIFAR_GRAY {
        let r = s.pixels[i];
        let g = s.pixels[1024 + i];
        let b = s.pixels[2048 + i];
        gray[i] = (r + g + b) / 3.0;
    }
    proj.project(&gray)
}

/// Binary task from two class labels.
pub fn filter_class_pair(
    data: &[Cifar10Sample],
    a: u8,
    b: u8,
    proj: &RandomProjection,
    limit: usize,
) -> Vec<Sample> {
    let mut out = Vec::new();
    for s in data {
        if s.label == a {
            out.push((project_cifar_gray(proj, s), [0.0]));
        } else if s.label == b {
            out.push((project_cifar_gray(proj, s), [1.0]));
        }
        if out.len() >= limit {
            break;
        }
    }
    out
}
