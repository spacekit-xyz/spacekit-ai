//! Standard (Euclidean) layer norm for the row-2 vanilla transformer.

#[derive(Clone, Debug)]
pub struct StandardLayerNorm {
    pub gamma: Vec<f32>,
    pub beta:  Vec<f32>,
}

impl StandardLayerNorm {
    pub fn new(d_model: usize) -> Self {
        Self {
            gamma: vec![1.0; d_model],
            beta:  vec![0.0; d_model],
        }
    }

    pub fn n_scalars(&self) -> usize {
        2 * self.gamma.len()
    }
}

#[derive(Clone, Debug)]
pub struct StandardNormStats {
    pub x_hat: Vec<f32>,
    pub mean:  f32,
    pub std:   f32,
}

pub fn forward(x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> (Vec<f32>, StandardNormStats) {
    let n = x.len();
    let mean = x.iter().sum::<f32>() / n as f32;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    let std = (var + eps).sqrt();
    let x_hat: Vec<f32> = x.iter().map(|v| (v - mean) / std).collect();
    let y: Vec<f32> = x_hat
        .iter()
        .zip(gamma.iter())
        .zip(beta.iter())
        .map(|((&xh, &g), &b)| xh * g + b)
        .collect();
    (
        y,
        StandardNormStats {
            x_hat,
            mean,
            std,
        },
    )
}

/// Backward for `y = x_hat * gamma + beta`.
pub fn backward(
    x_hat:    &[f32],
    gamma:    &[f32],
    grad_out: &[f32],
    std:      f32,
) -> Vec<f32> {
    let n = x_hat.len();
    let mut grad_x_hat = vec![0.0f32; n];
    for i in 0..n {
        grad_x_hat[i] = grad_out[i] * gamma[i];
    }
    let sum1: f32 = grad_x_hat.iter().sum();
    let sum2: f32 = grad_x_hat.iter().zip(x_hat.iter()).map(|(g, x)| g * x).sum();
    let inv_std = 1.0 / std;
    let inv_n = 1.0 / n as f32;
    (0..n)
        .map(|i| {
            inv_std * (grad_x_hat[i] - inv_n * sum1 - inv_n * x_hat[i] * sum2)
        })
        .collect()
}
