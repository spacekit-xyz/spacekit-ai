//! Trained MLP classifier: embedding -> action_type logits.
//! Replaces keyword-based `infer_action_type_from_text`.

use serde::{Deserialize, Serialize};

use crate::types::GroupId;

use super::action::ActionType;

pub const NUM_ACTION_TYPES: usize = 4;

fn action_type_index(at: &ActionType) -> usize {
    match at {
        ActionType::SupportTicket => 0,
        ActionType::CodingAssist => 1,
        ActionType::GeneralAssist => 2,
        ActionType::Fallback => 3,
    }
}

/// One-hot encoding of action type for conditioning generation heads. Returns a 4-element vector.
pub fn action_type_one_hot(at: &ActionType) -> [f32; NUM_ACTION_TYPES] {
    let mut out = [0.0f32; NUM_ACTION_TYPES];
    out[action_type_index(at)] = 1.0;
    out
}

/// One-hot encoding of routed group for conditioning generation heads (region binding).
/// Returns a vector of length `num_dims` (typically `group_order.len()`). Used so the head
/// receives an explicit region signal and can select the correct attractor per group.
pub fn group_id_one_hot(group_id: Option<GroupId>, group_order: &[GroupId], num_dims: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; num_dims];
    if let Some(gid) = group_id {
        if let Some(idx) = group_order.iter().position(|&g| g == gid) {
            if idx < num_dims {
                v[idx] = 1.0;
            }
        }
    }
    v
}

fn index_to_action_type(idx: usize) -> ActionType {
    match idx {
        0 => ActionType::SupportTicket,
        1 => ActionType::CodingAssist,
        2 => ActionType::GeneralAssist,
        _ => ActionType::Fallback,
    }
}

/// Small 2-layer MLP for action classification.
/// Architecture: input_dim -> hidden -> NUM_ACTION_TYPES
#[derive(Clone, Serialize, Deserialize)]
pub struct ActionClassifier {
    pub w1: Vec<Vec<f32>>,
    pub b1: Vec<f32>,
    pub w2: Vec<Vec<f32>>,
    pub b2: Vec<f32>,
    pub input_dim: usize,
    pub hidden_dim: usize,
}

impl ActionClassifier {
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        let mut w1 = vec![vec![0.0f32; input_dim]; hidden_dim];
        let mut w2 = vec![vec![0.0f32; hidden_dim]; NUM_ACTION_TYPES];
        for (h, row) in w1.iter_mut().enumerate() {
            for (i, w) in row.iter_mut().enumerate() {
                *w = (((h as u64 * 2654435761 + i as u64 * 7919) % 1000) as f32 / 1000.0 - 0.5)
                    * 0.1;
            }
        }
        for (o, row) in w2.iter_mut().enumerate() {
            for (h, w) in row.iter_mut().enumerate() {
                *w = (((o as u64 * 2654435761 + h as u64 * 7919) % 1000) as f32 / 1000.0 - 0.5)
                    * 0.1;
            }
        }
        Self {
            w1,
            b1: vec![0.0; hidden_dim],
            w2,
            b2: vec![0.0; NUM_ACTION_TYPES],
            input_dim,
            hidden_dim,
        }
    }

    fn forward(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut h = vec![0.0f32; self.hidden_dim];
        for (j, hj) in h.iter_mut().enumerate() {
            let mut acc = self.b1[j];
            for (i, &xi) in x.iter().enumerate() {
                if i < self.w1[j].len() {
                    acc += self.w1[j][i] * xi;
                }
            }
            *hj = acc.tanh();
        }
        let mut logits = vec![0.0f32; NUM_ACTION_TYPES];
        for (o, lo) in logits.iter_mut().enumerate() {
            let mut acc = self.b2[o];
            for (j, &hj) in h.iter().enumerate() {
                acc += self.w2[o][j] * hj;
            }
            *lo = acc;
        }
        (h, logits)
    }

    pub fn predict(&self, x: &[f32]) -> ActionType {
        let (_, logits) = self.forward(x);
        let idx = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(3);
        index_to_action_type(idx)
    }

    pub fn predict_with_confidence(&self, x: &[f32]) -> (ActionType, f32) {
        let (_, logits) = self.forward(x);
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
        let probs: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp() / exp_sum).collect();
        let idx = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(3);
        (index_to_action_type(idx), probs[idx])
    }

    /// Train one step with softmax cross-entropy loss. Returns loss.
    pub fn train_step(&mut self, x: &[f32], target: &ActionType, lr: f32) -> f32 {
        let target_idx = action_type_index(target);
        let (h, logits) = self.forward(x);

        // Softmax
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum_exp: f32 = exp.iter().sum();
        let probs: Vec<f32> = exp.iter().map(|e| e / sum_exp).collect();

        let loss = -(probs[target_idx].max(1e-10).ln());

        // d_logits = probs - one_hot(target)
        let mut d_logits = probs.clone();
        d_logits[target_idx] -= 1.0;

        // Backprop through w2, b2
        let mut d_h = vec![0.0f32; self.hidden_dim];
        for (o, &dl) in d_logits.iter().enumerate() {
            self.b2[o] -= lr * dl;
            for (j, &hj) in h.iter().enumerate() {
                d_h[j] += self.w2[o][j] * dl;
                self.w2[o][j] -= lr * dl * hj;
            }
        }

        // Backprop through tanh: d_pre_h = d_h * (1 - h^2)
        for (j, dh) in d_h.iter().enumerate() {
            let d_pre = dh * (1.0 - h[j] * h[j]);
            self.b1[j] -= lr * d_pre;
            for (i, &xi) in x.iter().enumerate() {
                if i < self.w1[j].len() {
                    self.w1[j][i] -= lr * d_pre * xi;
                }
            }
        }

        loss
    }
}

pub fn action_target_to_type(target: &str) -> ActionType {
    match target {
        "support" => ActionType::SupportTicket,
        "coding" => ActionType::CodingAssist,
        "safety" => ActionType::Fallback,
        _ => ActionType::GeneralAssist,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classifier_trains_and_predicts() {
        let mut clf = ActionClassifier::new(8, 16);
        let support_emb = vec![1.0, 0.0, 0.0, 0.0, 0.5, 0.1, 0.0, 0.0];
        let coding_emb = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.1];
        for _ in 0..100 {
            clf.train_step(&support_emb, &ActionType::SupportTicket, 0.1);
            clf.train_step(&coding_emb, &ActionType::CodingAssist, 0.1);
        }
        assert_eq!(clf.predict(&support_emb), ActionType::SupportTicket);
        assert_eq!(clf.predict(&coding_emb), ActionType::CodingAssist);
    }
}
