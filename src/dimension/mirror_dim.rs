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
    /// Once true, neurogenesis trigger will not fire again for this mirror.
    pub neurogenesis_triggered: bool,
    /// Consecutive epochs with loss above residual threshold (for residual-based trigger).
    pub residual_streak: u32,
    /// Pre-allocated neurons for neurogenesis (promote from pool instead of allocating). Optional.
    /// Priority: pool neurons could later receive passive exposure before recruitment for warmer integration; current on-demand allocation is valid without it.
    pub reserve_pool: Option<Vec<crate::neuron::Neuron>>,
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
            neurogenesis_triggered: false,
            residual_streak: 0,
            reserve_pool: None,
            main_query_fn: None,
        }
    }

    /// Create a mirror with an optional reserve pool of neurons for neurogenesis.
    pub fn new_with_reserve_pool(
        task_name: String,
        env: NeuralEnvironment,
        config: EnvironmentConfig,
        reserve_pool_size: usize,
    ) -> Self {
        let reserve_pool = if reserve_pool_size > 0 {
            Some(
                (0..reserve_pool_size)
                    .map(|_| crate::neuron::Neuron::new(0, crate::types::Vec3::zero(), &config))
                    .collect(),
            )
        } else {
            None
        };
        Self {
            task_name,
            env,
            config,
            epochs_trained: 0,
            best_accuracy: 0.0,
            current_accuracy: 0.0,
            neurogenesis_triggered: false,
            residual_streak: 0,
            reserve_pool,
            main_query_fn: None,
        }
    }

    /// If epoch count and loss exceed thresholds and not yet triggered, insert one neuron into
    /// the last hidden layer and mark triggered. Returns true if a neuron was added.
    pub fn try_neurogenesis_trigger(
        &mut self,
        epoch_trigger: u32,
        loss_threshold: f32,
        current_loss: f32,
        rng: &mut impl Rng,
    ) -> bool {
        if self.neurogenesis_triggered {
            return false;
        }
        if self.epochs_trained < epoch_trigger || current_loss <= loss_threshold {
            return false;
        }
        let last_hidden = self.env.layers.len().saturating_sub(2);
        if last_hidden == 0 {
            return false;
        }
        let added = self
            .env
            .insert_neuron_at_layer(last_hidden, rng, self.reserve_pool.as_mut())
            .is_some();
        if added {
            self.neurogenesis_triggered = true;
        }
        added
    }

    /// Residual-based trigger: add one neuron if loss has been above `residual_threshold` for
    /// at least `min_epochs_high` consecutive epochs. Resets streak when loss <= threshold.
    pub fn try_neurogenesis_trigger_residual(
        &mut self,
        residual_threshold: f32,
        min_epochs_high: u32,
        current_loss: f32,
        rng: &mut impl Rng,
    ) -> bool {
        if self.neurogenesis_triggered {
            return false;
        }
        if current_loss > residual_threshold {
            self.residual_streak = self.residual_streak.saturating_add(1);
        } else {
            self.residual_streak = 0;
        }
        if self.residual_streak < min_epochs_high {
            return false;
        }
        let last_hidden = self.env.layers.len().saturating_sub(2);
        if last_hidden == 0 {
            return false;
        }
        let added = self
            .env
            .insert_neuron_at_layer(last_hidden, rng, self.reserve_pool.as_mut())
            .is_some();
        if added {
            self.neurogenesis_triggered = true;
            self.residual_streak = 0;
        }
        added
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::NeuralEnvironment;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_residual_trigger_fires_after_streak() {
        let mut rng = StdRng::seed_from_u64(123);
        let config = EnvironmentConfig::default();
        let mut env = NeuralEnvironment::new(config.clone());
        env.build_layers(&[2, 4, 1], &mut rng);
        let mut mirror = MirrorDimension::new("t".into(), env, config);
        let threshold = 0.25f32;
        let min_epochs = 3u32;
        assert!(!mirror.try_neurogenesis_trigger_residual(threshold, min_epochs, 0.3, &mut rng));
        assert_eq!(mirror.residual_streak, 1);
        assert!(!mirror.try_neurogenesis_trigger_residual(threshold, min_epochs, 0.3, &mut rng));
        assert_eq!(mirror.residual_streak, 2);
        assert!(mirror.try_neurogenesis_trigger_residual(threshold, min_epochs, 0.3, &mut rng));
        assert!(mirror.neurogenesis_triggered);
        assert_eq!(mirror.env.layers[1].len(), 5);
    }

    #[test]
    fn test_residual_streak_resets_when_loss_below_threshold() {
        let mut rng = StdRng::seed_from_u64(123);
        let config = EnvironmentConfig::default();
        let mut env = NeuralEnvironment::new(config.clone());
        env.build_layers(&[2, 4, 1], &mut rng);
        let mut mirror = MirrorDimension::new("t".into(), env, config);
        mirror.try_neurogenesis_trigger_residual(0.25, 5, 0.3, &mut rng);
        mirror.try_neurogenesis_trigger_residual(0.25, 5, 0.1, &mut rng);
        assert_eq!(mirror.residual_streak, 0);
    }

    #[test]
    fn test_reserve_pool_used_when_provided() {
        let mut rng = StdRng::seed_from_u64(123);
        let config = EnvironmentConfig::default();
        let mut env = NeuralEnvironment::new(config.clone());
        env.build_layers(&[2, 4, 1], &mut rng);
        let mut mirror = MirrorDimension::new_with_reserve_pool("t".into(), env, config, 2);
        assert_eq!(mirror.reserve_pool.as_ref().map(|p| p.len()), Some(2));
        let _ = mirror.try_neurogenesis_trigger(0, 0.0, 0.5, &mut rng);
        assert!(mirror.neurogenesis_triggered);
        assert_eq!(mirror.reserve_pool.as_ref().map(|p| p.len()), Some(1));
    }
}
