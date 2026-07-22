//! CIFAR-100 binary loader (coarse + fine labels).
//!
//! Record layout (3074 bytes): `<coarse u8><fine u8><3072 RGB pixels>` (R,G,B planes).

use std::fs::File;
use std::io::{Read, Result as IoResult};
use std::path::{Path, PathBuf};

use crate::mnist::RandomProjection;
use crate::types::Sample;

pub const CIFAR_SIDE: usize = 32;
pub const CIFAR_PIXELS: usize = CIFAR_SIDE * CIFAR_SIDE * 3; // 3072
pub const CIFAR_RECORD: usize = 2 + CIFAR_PIXELS; // 3074
pub const CIFAR_PROJECTED: usize = 64;
pub const CIFAR_GRAY: usize = CIFAR_SIDE * CIFAR_SIDE; // 1024

#[derive(Clone, Debug)]
pub struct CifarSample {
    pub coarse: u8,
    pub fine: u8,
    pub pixels: Vec<f32>, // 3072, [0,1]
}

fn resolve_dir(root: &Path) -> Option<PathBuf> {
    let candidates = [
        root.join("cifar-100-binary"),
        root.to_path_buf(),
        root.join("data/cifar-100-binary"),
    ];
    for c in candidates {
        if c.join("train.bin").exists() {
            return Some(c);
        }
    }
    None
}

pub fn cifar100_available(root: &Path) -> bool {
    resolve_dir(root).is_some()
}

fn load_bin(path: &Path) -> IoResult<Vec<CifarSample>> {
    let mut f = File::open(path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    let n = buf.len() / CIFAR_RECORD;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let o = i * CIFAR_RECORD;
        let coarse = buf[o];
        let fine = buf[o + 1];
        let mut pixels = Vec::with_capacity(CIFAR_PIXELS);
        for j in 0..CIFAR_PIXELS {
            pixels.push(buf[o + 2 + j] as f32 / 255.0);
        }
        out.push(CifarSample {
            coarse,
            fine,
            pixels,
        });
    }
    Ok(out)
}

pub fn load_cifar100(root: &Path) -> Result<(Vec<CifarSample>, Vec<CifarSample>), String> {
    let dir = resolve_dir(root).ok_or_else(|| {
        String::from(
            "CIFAR-100 binary not found (expect train.bin). Run scripts/download_cifar100.sh",
        )
    })?;
    let train = load_bin(&dir.join("train.bin")).map_err(|e| e.to_string())?;
    let test = load_bin(&dir.join("test.bin")).map_err(|e| e.to_string())?;
    Ok((train, test))
}

/// Grayscale mean of RGB planes → 1024, then random-project to [`CIFAR_PROJECTED`].
pub fn project_cifar_gray(proj: &RandomProjection, s: &CifarSample) -> Vec<f32> {
    let mut gray = vec![0.0f32; CIFAR_GRAY];
    for i in 0..CIFAR_GRAY {
        let r = s.pixels[i];
        let g = s.pixels[1024 + i];
        let b = s.pixels[2048 + i];
        gray[i] = (r + g + b) / 3.0;
    }
    proj.project(&gray)
}

/// Binary task from two coarse labels (superclass split).
pub fn filter_coarse_pair(
    data: &[CifarSample],
    a: u8,
    b: u8,
    proj: &RandomProjection,
    limit: usize,
) -> Vec<Sample> {
    let mut out = Vec::new();
    for s in data {
        if s.coarse == a {
            out.push((project_cifar_gray(proj, s), [0.0]));
        } else if s.coarse == b {
            out.push((project_cifar_gray(proj, s), [1.0]));
        }
        if out.len() >= limit {
            break;
        }
    }
    out
}
