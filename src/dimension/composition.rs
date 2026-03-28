//! Phase 3c: VirtualGroup (blend frozen groups) and EpisodicMemory (recall compositions).
//!
//! VirtualGroup runs input through two or more frozen groups and blends outputs with learned
//! scalar weights. Only the weights train — a tiny problem on top of frozen representations.
//! EpisodicMemory stores successful compositions for zero-shot recall on re-presentation.

use crate::types::GroupId;
use serde::{Deserialize, Serialize};

use super::embedding::cosine_similarity;
use super::main_dim::MainDimension;

// ---------------------------------------------------------------------------
// VirtualGroup — composition by blending group outputs
// ---------------------------------------------------------------------------

/// Blends outputs of two or more frozen groups with learned weights. Only weights train.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VirtualGroup {
    /// Group IDs to query (order matches blend_weights).
    pub group_ids: Vec<GroupId>,
    /// Scalar weight per group; trained by gradient on (blended - target)^2.
    pub blend_weights: Vec<f32>,
}

impl VirtualGroup {
    /// New composition with equal weights 1/n per group.
    pub fn new(group_ids: Vec<GroupId>) -> Self {
        let n = group_ids.len().max(1);
        let blend_weights = vec![1.0 / n as f32; n];
        VirtualGroup {
            group_ids,
            blend_weights,
        }
    }

    /// Run input through each group and blend outputs: out[i] = sum_g (w_g * group_g_output[i]).
    pub fn predict(&self, main: &mut MainDimension, input: &[f32]) -> Vec<f32> {
        let outputs = main.query(input, &self.group_ids);
        if outputs.is_empty() || outputs.len() != self.blend_weights.len() {
            return vec![];
        }
        let len = outputs[0].1.len();
        let mut out = vec![0.0f32; len];
        for ((_gid, o), &w) in outputs.iter().zip(self.blend_weights.iter()) {
            if o.len() != len {
                return vec![];
            }
            for (j, &v) in o.iter().enumerate() {
                out[j] += w * v;
            }
        }
        out
    }

    /// One gradient step on blend weights. Loss = (blended[0] - target[0])^2 for single output.
    /// Returns loss.
    #[cfg(feature = "training")]
    pub fn train_step(
        &mut self,
        main: &mut MainDimension,
        input: &[f32],
        target: &[f32],
        lr: f32,
    ) -> f32 {
        let outputs = main.query(input, &self.group_ids);
        if outputs.len() != self.blend_weights.len() {
            return 0.0;
        }
        let blended: Vec<f32> = (0..outputs[0].1.len())
            .map(|j| {
                outputs
                    .iter()
                    .zip(self.blend_weights.iter())
                    .map(|((_, o), &w)| w * o.get(j).copied().unwrap_or(0.0))
                    .sum()
            })
            .collect();
        let t0 = target.get(0).copied().unwrap_or(0.0);
        let b0 = blended.get(0).copied().unwrap_or(0.0);
        let loss = (b0 - t0) * (b0 - t0);
        let grad_scale = 2.0 * (b0 - t0);
        for (i, (_gid, o)) in outputs.iter().enumerate() {
            let o0 = o.get(0).copied().unwrap_or(0.0);
            self.blend_weights[i] -= lr * grad_scale * o0;
        }
        // Clamp to non-negative then renormalize to sum to 1
        for w in &mut self.blend_weights {
            *w = w.max(0.0);
        }
        let sum: f32 = self.blend_weights.iter().sum();
        if sum > 1e-10 {
            for w in &mut self.blend_weights {
                *w /= sum;
            }
        }
        loss
    }
}

// ---------------------------------------------------------------------------
// EpisodicMemory — store and retrieve compositions by input signature
// ---------------------------------------------------------------------------

/// One stored composition: signature, group IDs, blend weights, and metrics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Episode {
    /// Mean activation pattern of task examples (e.g. mean hidden over calibration).
    pub input_signature: Vec<f32>,
    pub group_ids: Vec<GroupId>,
    pub blend_weights: Vec<f32>,
    pub accuracy: f32,
    /// How much was left unexplained by single best group (residual).
    pub residual: f32,
}

/// Retrieves episodes by cosine similarity to a query signature.
#[derive(Default, Clone, Serialize, Deserialize)]
pub struct EpisodicMemory {
    pub episodes: Vec<Episode>,
}

impl EpisodicMemory {
    pub fn new() -> Self {
        Self {
            episodes: Vec::new(),
        }
    }

    pub fn store(&mut self, episode: Episode) {
        self.episodes.push(episode);
    }

    /// Best matching episode if cosine(signature, episode.input_signature) >= threshold.
    pub fn retrieve(&self, signature: &[f32], threshold: f32) -> Option<&Episode> {
        let mut best: Option<(&Episode, f32)> = None;
        for ep in &self.episodes {
            if ep.input_signature.len() != signature.len() {
                continue;
            }
            let sim = cosine_similarity(&ep.input_signature, signature);
            if sim >= threshold {
                if best.map_or(true, |(_, s)| sim > s) {
                    best = Some((ep, sim));
                }
            }
        }
        best.map(|(ep, _)| ep)
    }
}
