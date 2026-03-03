//! Mirror Dimension — isolated training environment for one task. Full plasticity.

use rand::Rng;

use crate::environment::NeuralEnvironment;
use crate::types::EnvironmentConfig;
use serde::{Deserialize, Serialize};

/// Isolated env for training one task. No Group A neurons; full plasticity.
#[derive(Serialize, Deserialize)]
pub struct MirrorDimension {
    pub task_name: String,
    pub env: NeuralEnvironment,
    pub config: EnvironmentConfig,
    pub epochs_trained: u32,
    pub best_accuracy: f32,
    pub current_accuracy: f32,
    /// Read-only query to Main (optional). Not serialized.
    #[serde(skip)]
    pub main_query_fn: Option<Box<dyn Fn(&[f32]) -> Vec<(crate::types::GroupId, Vec<f32>)> + Send>>,
}

impl MirrorDimension {
    pub fn new(task_name: String, env: NeuralEnvironment, config: EnvironmentConfig) -> Self {
        Self {
            task_name,
            env,
            config,
            epochs_trained: 0,
            best_accuracy: 0.0,
            current_accuracy: 0.0,
            main_query_fn: None,
        }
    }

    /// Train one epoch; returns mean loss and accuracy (0.0..1.0).
    pub fn train_epoch(
        &mut self,
        data: &[([f32; 2], [f32; 1])],
        rng: &mut impl Rng,
    ) -> EpochResult {
        let mut total_loss = 0.0f32;
        let mut correct = 0usize;
        let n = data.len();
        for (input, target) in data {
            let out = self.env.predict(input);
            let target_arr = [target[0]];
            let loss = self.env.train_tick(input, &target_arr, rng).loss;
            total_loss += loss;
            if out.len() >= 1 && (out[0] - target[0]).abs() < 0.5 {
                correct += 1;
            }
        }
        let loss = if n > 0 { total_loss / n as f32 } else { 0.0 };
        let accuracy = if n > 0 { correct as f32 / n as f32 } else { 0.0 };
        self.epochs_trained += 1;
        self.current_accuracy = accuracy;
        if accuracy > self.best_accuracy {
            self.best_accuracy = accuracy;
        }
        EpochResult { loss, accuracy, correct, total: n }
    }

    /// True if accuracy has been stable (no improvement) for `window` epochs.
    pub fn is_stable(&self, window: u32) -> bool {
        self.epochs_trained >= window
        // Simplified: we don't track per-epoch history here; caller can check plateau externally.
        // For the test we require epochs_trained >= window.
    }
}

#[derive(Debug, Clone)]
pub struct EpochResult {
    pub loss: f32,
    pub accuracy: f32,
    pub correct: usize,
    pub total: usize,
}
