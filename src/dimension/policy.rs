//! Policy — small MLP for state → action.
//! - **Policy**: discrete actions (logits → argmax). Same pattern as LearnedRouter.
//! - **ContinuousPolicy**: continuous action vector; output dim = action_dim, trained with MSE.

use crate::environment::NeuralEnvironment;
use crate::types::EnvironmentConfig;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Policy MLP: state_dim → hidden → num_actions. Output = action logits; argmax = chosen action.
#[derive(Clone, Serialize, Deserialize)]
pub struct Policy {
    pub env: NeuralEnvironment,
    pub num_actions: usize,
    pub state_dim: usize,
}

impl Policy {
    /// Build policy MLP: state_dim -> hidden -> num_actions. Same init pattern as LearnedRouter.
    pub fn new(
        state_dim: usize,
        num_actions: usize,
        hidden_size: usize,
        rng: &mut impl Rng,
    ) -> Self {
        let mut config = EnvironmentConfig::default();
        config.learning_rate = 0.15;
        let mut env = NeuralEnvironment::new(config);
        env.build_layers(&[state_dim, hidden_size, num_actions], rng);
        Policy {
            env,
            num_actions,
            state_dim,
        }
    }

    /// Action logits. Empty if state len != state_dim.
    pub fn predict_logits(&mut self, state: &[f32]) -> Vec<f32> {
        if state.len() != self.state_dim || self.num_actions == 0 {
            return vec![];
        }
        self.env.predict(state)
    }

    /// One training step: one-hot target for action index. Returns MSE loss.
    #[cfg(feature = "training")]
    pub fn train_step(
        &mut self,
        state: &[f32],
        target_action: usize,
        _rng: &mut impl Rng,
    ) -> f32 {
        if state.len() != self.state_dim || target_action >= self.num_actions {
            return 0.0;
        }
        let output = self.env.forward(state);
        let mut target = vec![0.0f32; self.num_actions];
        target[target_action] = 1.0;
        self.env.backprop(&output, &target)
    }

    /// Argmax action index. None if no actions or invalid state.
    pub fn choose_action(&mut self, state: &[f32]) -> Option<usize> {
        let logits = self.predict_logits(state);
        if logits.is_empty() {
            return None;
        }
        let (idx, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;
        Some(idx)
    }
}

/// Continuous-action policy: state_dim → hidden → action_dim. Output = action vector; train with MSE to target.
#[derive(Clone, Serialize, Deserialize)]
pub struct ContinuousPolicy {
    pub env: NeuralEnvironment,
    pub action_dim: usize,
    pub state_dim: usize,
}

impl ContinuousPolicy {
    /// Build MLP: state_dim -> hidden -> action_dim. Same init pattern as Policy.
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_size: usize,
        rng: &mut impl Rng,
    ) -> Self {
        let mut config = EnvironmentConfig::default();
        config.learning_rate = 0.15;
        let mut env = NeuralEnvironment::new(config);
        env.build_layers(&[state_dim, hidden_size, action_dim], rng);
        ContinuousPolicy {
            env,
            action_dim,
            state_dim,
        }
    }

    /// Action vector. Empty if state len != state_dim.
    pub fn predict(&mut self, state: &[f32]) -> Vec<f32> {
        if state.len() != self.state_dim || self.action_dim == 0 {
            return vec![];
        }
        self.env.predict(state)
    }

    /// One training step: MSE to target action. Returns MSE loss.
    #[cfg(feature = "training")]
    pub fn train_step(
        &mut self,
        state: &[f32],
        target: &[f32],
        _rng: &mut impl Rng,
    ) -> f32 {
        if state.len() != self.state_dim
            || target.len() != self.action_dim
        {
            return 0.0;
        }
        let output = self.env.forward(state);
        self.env.backprop(&output, target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_policy_new_and_predict_logits_shape() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut policy = Policy::new(2, 3, 8, &mut rng);
        let logits = policy.predict_logits(&[0.5, -0.3]);
        assert_eq!(logits.len(), 3);
    }

    #[test]
    fn test_policy_predict_logits_wrong_state_dim_returns_empty() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut policy = Policy::new(2, 3, 8, &mut rng);
        assert!(policy.predict_logits(&[0.5]).is_empty());
        assert!(policy.predict_logits(&[0.5, 0.0, 0.0]).is_empty());
    }

    #[test]
    fn test_policy_choose_action_returns_argmax() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut policy = Policy::new(2, 3, 8, &mut rng);
        let a = policy.choose_action(&[0.1, 0.2]);
        assert!(a.is_some());
        let logits = policy.predict_logits(&[0.1, 0.2]);
        let expected = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i);
        assert_eq!(a, expected);
    }

    #[test]
    fn test_policy_train_step_valid_returns_loss() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut policy = Policy::new(2, 3, 8, &mut rng);
        let loss = policy.train_step(&[0.5, -0.3], 1, &mut rng);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_policy_train_step_invalid_no_op() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut policy = Policy::new(2, 3, 8, &mut rng);
        assert_eq!(policy.train_step(&[0.5], 0, &mut rng), 0.0);
        assert_eq!(policy.train_step(&[0.5, 0.0], 3, &mut rng), 0.0);
    }

    #[test]
    fn test_policy_training_moves_toward_target_action() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut policy = Policy::new(2, 2, 8, &mut rng);
        let state = [0.3f32, 0.7];
        for _ in 0..200 {
            policy.train_step(&state, 1, &mut rng);
        }
        let chosen = policy.choose_action(&state);
        assert_eq!(chosen, Some(1));
    }

    // --- ContinuousPolicy ---

    #[test]
    fn test_continuous_policy_new_and_predict_shape() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut policy = ContinuousPolicy::new(2, 1, 8, &mut rng);
        let out = policy.predict(&[0.5, -0.3]);
        assert_eq!(out.len(), 1);
        let mut policy2 = ContinuousPolicy::new(3, 2, 8, &mut rng);
        let out2 = policy2.predict(&[0.0, 0.0, 1.0]);
        assert_eq!(out2.len(), 2);
    }

    #[test]
    fn test_continuous_policy_predict_wrong_state_dim_returns_empty() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut policy = ContinuousPolicy::new(2, 1, 8, &mut rng);
        assert!(policy.predict(&[0.5]).is_empty());
        assert!(policy.predict(&[0.5, 0.0, 0.0]).is_empty());
    }

    #[test]
    fn test_continuous_policy_train_step_valid_returns_loss() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut policy = ContinuousPolicy::new(2, 1, 8, &mut rng);
        let loss = policy.train_step(&[0.5, -0.3], &[0.7], &mut rng);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_continuous_policy_train_step_invalid_no_op() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut policy = ContinuousPolicy::new(2, 1, 8, &mut rng);
        assert_eq!(policy.train_step(&[0.5], &[0.0], &mut rng), 0.0);
        assert_eq!(policy.train_step(&[0.5, 0.0], &[0.0, 0.0], &mut rng), 0.0);
    }

    #[test]
    fn test_continuous_policy_training_moves_toward_target() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut policy = ContinuousPolicy::new(2, 1, 8, &mut rng);
        let state = [0.3f32, 0.7];
        let target = [0.8f32];
        for _ in 0..300 {
            policy.train_step(&state, &target, &mut rng);
        }
        let out = policy.predict(&state);
        assert_eq!(out.len(), 1);
        assert!((out[0] - target[0]).abs() < 0.3, "output {:?} should be near target {:?}", out, target);
    }
}
