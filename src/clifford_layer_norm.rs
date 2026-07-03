//! Metric-aware layer norm over `d_model` multivectors (16·d_model components per position).
//!
//! Mean and variance use per-blade weights [`BLADE_METRIC_WEIGHT`] so normalization
//! respects the STA blade metric instead of treating all 16 components as Euclidean.

use crate::blade::BLADE_METRIC_WEIGHT;
use crate::Multivector;

/// Stats recorded for backward.
#[derive(Clone, Debug)]
pub struct FlatNormStats {
    pub x_hat: Vec<f32>,
    pub mean:  f32,
    pub std:   f32,
}

#[inline]
fn component_weight(flat_idx: usize) -> f32 {
    BLADE_METRIC_WEIGHT[flat_idx % 16]
}

#[inline]
fn sum_weights(n: usize) -> f32 {
    (0..n).map(component_weight).sum()
}

/// Forward: metric-weighted mean/var over flattened multivectors, then γ/β.
pub fn forward_flat(
    flat:  &[f32],
    gamma: &[f32],
    beta:  &[f32],
    eps:   f32,
) -> (Vec<f32>, FlatNormStats) {
    debug_assert_eq!(flat.len(), gamma.len());
    debug_assert_eq!(flat.len(), beta.len());

    let n = flat.len();
    let sum_w = sum_weights(n);
    let mean = if sum_w > 0.0 {
        flat.iter()
            .enumerate()
            .map(|(i, v)| component_weight(i) * v)
            .sum::<f32>()
            / sum_w
    } else {
        0.0
    };

    let var = if sum_w > 0.0 {
        flat.iter()
            .enumerate()
            .map(|(i, v)| {
                let d = v - mean;
                component_weight(i) * d * d
            })
            .sum::<f32>()
            / sum_w
    } else {
        0.0
    };
    let std = (var + eps).sqrt();

    let x_hat: Vec<f32> = flat.iter().map(|v| (v - mean) / std).collect();
    let output: Vec<f32> = x_hat
        .iter()
        .enumerate()
        .map(|(i, &x)| x * gamma[i] + beta[i])
        .collect();

    (
        output,
        FlatNormStats {
            x_hat,
            mean,
            std,
        },
    )
}

pub fn forward_multivectors(
    ln_gamma: &[f32],
    ln_beta:  &[f32],
    ln_eps:   f32,
    x:        &[Multivector],
) -> (Vec<Multivector>, FlatNormStats) {
    let flat: Vec<f32> = x.iter().flat_map(|mv| mv.c).collect();
    let (normalised, stats) = forward_flat(&flat, ln_gamma, ln_beta, ln_eps);
    let output: Vec<Multivector> = normalised
        .chunks(16)
        .map(|chunk| {
            let mut c = [0.0f32; 16];
            c.copy_from_slice(chunk);
            Multivector { c }
        })
        .collect();
    (output, stats)
}

/// Backward through metric-weighted layer norm.
pub fn backward_flat(
    x_hat:    &[f32],
    gamma:    &[f32],
    grad_out: &[f32],
    std:      f32,
) -> Vec<f32> {
    let n = x_hat.len();
    let sum_w = sum_weights(n);

    let dl_dg: Vec<f32> = grad_out.iter().zip(gamma).map(|(&g, &gam)| g * gam).collect();

    let mean_dl_dg: f32 = if sum_w > 0.0 {
        dl_dg
            .iter()
            .enumerate()
            .map(|(i, d)| component_weight(i) * d)
            .sum::<f32>()
            / sum_w
    } else {
        0.0
    };

    let mean_dl_dg_xhat: f32 = if sum_w > 0.0 {
        dl_dg
            .iter()
            .zip(x_hat.iter())
            .enumerate()
            .map(|(i, (d, xh))| component_weight(i) * d * xh)
            .sum::<f32>()
            / sum_w
    } else {
        0.0
    };

    dl_dg
        .iter()
        .zip(x_hat.iter())
        .map(|(&d, &x)| (d - mean_dl_dg - x * mean_dl_dg_xhat) / std)
        .collect()
}
