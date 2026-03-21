//! Paramecium-based action classifier: embedding -> action_type.
//! One-pass `develop()` replaces iterative backprop MLP training.

use serde::{Deserialize, Serialize};

use crate::types::GroupId;
use crate::spectral::TokenDictionary;
use crate::dimension::paramecium::InfraciliaryLattice;

use super::action::ActionType;

pub const NUM_ACTION_TYPES: usize = 5;

fn action_type_index(at: &ActionType) -> usize {
    match at {
        ActionType::SupportTicket => 0,
        ActionType::CodingAssist => 1,
        ActionType::GeneralAssist => 2,
        ActionType::ToolCall => 3,
        ActionType::Fallback => 4,
    }
}

/// One-hot encoding of action type for conditioning generation heads.
pub fn action_type_one_hot(at: &ActionType) -> [f32; NUM_ACTION_TYPES] {
    let mut out = [0.0f32; NUM_ACTION_TYPES];
    out[action_type_index(at)] = 1.0;
    out
}

/// One-hot encoding of routed group for conditioning generation heads.
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
        3 => ActionType::ToolCall,
        _ => ActionType::Fallback,
    }
}

const ACTION_LABELS: [&str; NUM_ACTION_TYPES] = [
    "action_support", "action_coding", "action_general", "action_tool", "action_fallback",
];

/// Paramecium lattice-based action classifier.
/// Replaces the hand-rolled MLP. One-pass `develop()` training.
#[derive(Clone, Serialize, Deserialize)]
pub struct ActionClassifier {
    pub lattice: InfraciliaryLattice,
    pub input_dim: usize,
    pub hidden_dim: usize,
}

impl ActionClassifier {
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        let dict = TokenDictionary::build(&ACTION_LABELS, 64);
        let lattice = InfraciliaryLattice::new(dict);
        Self {
            lattice,
            input_dim,
            hidden_dim,
        }
    }

    /// Build classifier from labeled data in one pass.
    pub fn build(input_dim: usize, samples: &[(Vec<f32>, ActionType)]) -> Self {
        let dict = TokenDictionary::build(&ACTION_LABELS, 64);
        let pairs: Vec<(Vec<f32>, String)> = samples.iter()
            .map(|(emb, at)| {
                let idx = action_type_index(at);
                (emb.clone(), ACTION_LABELS[idx].to_string())
            })
            .collect();
        let mut lattice = InfraciliaryLattice::new(dict);
        lattice.develop(&pairs, 0.90);
        Self {
            lattice,
            input_dim,
            hidden_dim: 0,
        }
    }

    /// Expand output — no-op for lattice (programs grow automatically).
    pub fn ensure_output_dim(&mut self) {}

    pub fn predict(&mut self, x: &[f32]) -> ActionType {
        self.predict_shared(x)
    }

    /// Immutable prediction — no EMA centroid drift.
    pub fn predict_shared(&self, x: &[f32]) -> ActionType {
        if self.lattice.programs.is_empty() {
            return ActionType::Fallback;
        }
        let mut best_idx = 0;
        let mut best_sim = f32::NEG_INFINITY;
        for (i, prog) in self.lattice.programs.iter().enumerate() {
            let sim = cosine_sim(x, &prog.ema_centroid);
            if sim > best_sim { best_sim = sim; best_idx = i; }
        }
        let text = self.lattice.dictionary.decode(&self.lattice.programs[best_idx].token_sequence);
        Self::parse_action(&text).unwrap_or(ActionType::Fallback)
    }

    pub fn predict_with_confidence(&mut self, x: &[f32]) -> (ActionType, f32) {
        self.predict_with_confidence_shared(x)
    }

    pub fn predict_with_confidence_shared(&self, x: &[f32]) -> (ActionType, f32) {
        if self.lattice.programs.is_empty() {
            return (ActionType::Fallback, 0.0);
        }
        let mut best_idx = 0;
        let mut best_sim = f32::NEG_INFINITY;
        for (i, prog) in self.lattice.programs.iter().enumerate() {
            let sim = cosine_sim(x, &prog.ema_centroid);
            if sim > best_sim { best_sim = sim; best_idx = i; }
        }
        let text = self.lattice.dictionary.decode(&self.lattice.programs[best_idx].token_sequence);
        let at = Self::parse_action(&text).unwrap_or(ActionType::Fallback);
        (at, best_sim.max(0.0))
    }

    /// Train one step: develop lattice with this (input, action) pair.
    pub fn train_step(&mut self, x: &[f32], target: &ActionType, _lr: f32) -> f32 {
        let idx = action_type_index(target);
        let pairs = vec![(x.to_vec(), ACTION_LABELS[idx].to_string())];
        self.lattice.develop(&pairs, 0.90);

        let resp = self.lattice.respond(x);
        let predicted = Self::parse_action(&resp.text);
        if predicted.as_ref() == Some(target) { 0.0 } else { 1.0 }
    }

    pub fn program_count(&self) -> usize {
        self.lattice.program_count()
    }

    fn parse_action(text: &str) -> Option<ActionType> {
        match text {
            "action_support" => Some(ActionType::SupportTicket),
            "action_coding" => Some(ActionType::CodingAssist),
            "action_general" => Some(ActionType::GeneralAssist),
            "action_tool" => Some(ActionType::ToolCall),
            "action_fallback" => Some(ActionType::Fallback),
            _ => None,
        }
    }
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-10 { 0.0 } else { dot / denom }
}

pub fn action_target_to_type(target: &str) -> ActionType {
    match target {
        "support" => ActionType::SupportTicket,
        "coding" => ActionType::CodingAssist,
        "patterns" | "concepts" | "math" | "reasoning" | "general" => ActionType::GeneralAssist,
        "safety" => ActionType::Fallback,
        _ => ActionType::GeneralAssist,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classifier_build_and_predict() {
        let samples: Vec<(Vec<f32>, ActionType)> = vec![
            (vec![1.0, 0.0, 0.0, 0.0, 0.5, 0.1, 0.0, 0.0], ActionType::SupportTicket),
            (vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.1], ActionType::CodingAssist),
            (vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.1], ActionType::GeneralAssist),
        ];
        let mut clf = ActionClassifier::build(8, &samples);
        assert_eq!(clf.predict(&samples[0].0), ActionType::SupportTicket);
        assert_eq!(clf.predict(&samples[1].0), ActionType::CodingAssist);
        assert_eq!(clf.predict(&samples[2].0), ActionType::GeneralAssist);
    }

    #[test]
    fn test_classifier_trains_and_predicts() {
        let mut clf = ActionClassifier::new(8, 16);
        let support_emb = vec![1.0, 0.0, 0.0, 0.0, 0.5, 0.1, 0.0, 0.0];
        let coding_emb = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.1];
        for _ in 0..3 {
            clf.train_step(&support_emb, &ActionType::SupportTicket, 0.1);
            clf.train_step(&coding_emb, &ActionType::CodingAssist, 0.1);
        }
        assert_eq!(clf.predict(&support_emb), ActionType::SupportTicket);
        assert_eq!(clf.predict(&coding_emb), ActionType::CodingAssist);
    }
}
