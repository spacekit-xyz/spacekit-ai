//! Promotion Gate — gatekeeper between Mirror and Main Dimension.

use crate::types::GroupId;
use serde::{Deserialize, Serialize};

use super::embedding::{compute_group_embedding, cosine_similarity, GroupEmbedding};
use super::main_dim::MainDimension;
use super::mirror_dim::MirrorDimension;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotionGateConfig {
    pub accuracy_threshold: f32,
    pub redundancy_threshold: f32,  // cosine above this = reject (redundant)
    pub stability_window_epochs: u32,
}

impl Default for PromotionGateConfig {
    fn default() -> Self {
        Self {
            accuracy_threshold: 0.85,
            redundancy_threshold: 0.80,
            stability_window_epochs: 50,
        }
    }
}

#[derive(Debug, Clone)]
pub enum PromotionDecision {
    Promote,
    ContinueTraining { reason: String },
    Reject { reason: String, similar_to: Option<GroupId> },
}

/// Evaluate a mirror for promotion. May run forward passes on mirror.env (activations only).
pub fn evaluate_promotion(
    mirror: &mut MirrorDimension,
    main: &MainDimension,
    calibration_data: &[([f32; 2], [f32; 1])],
    config: &PromotionGateConfig,
) -> PromotionDecision {
    if mirror.best_accuracy < config.accuracy_threshold {
        return PromotionDecision::ContinueTraining {
            reason: format!("accuracy {:.1}% below threshold {:.0}%",
                mirror.best_accuracy * 100.0, config.accuracy_threshold * 100.0),
        };
    }
    let mirror_vec = compute_group_embedding(&mut mirror.env, calibration_data);
    if mirror_vec.is_empty() {
        return PromotionDecision::ContinueTraining {
            reason: "empty embedding".to_string(),
        };
    }
    for emb in &main.embedding_library {
        let sim = cosine_similarity(&mirror_vec, &emb.vector);
        if sim > config.redundancy_threshold {
            return PromotionDecision::Reject {
                reason: format!("redundant with existing group (cosine {:.3})", sim),
                similar_to: Some(emb.group_id),
            };
        }
    }
    if !mirror.is_stable(config.stability_window_epochs) {
        return PromotionDecision::ContinueTraining {
            reason: "stability window not met".to_string(),
        };
    }
    PromotionDecision::Promote
}

/// Consume mirror: freeze env, compute embedding, register with main. Returns new group_id.
pub fn promote(
    mirror: MirrorDimension,
    main: &mut MainDimension,
    calibration_data: &[([f32; 2], [f32; 1])],
    next_group_id: GroupId,
) -> GroupId {
    let mut env = mirror.env;
    env.freeze_all();
    let vector = compute_group_embedding(&mut env, calibration_data);
    let embedding = GroupEmbedding {
        group_id: next_group_id,
        vector,
        task_name: mirror.task_name.clone(),
        accuracy: mirror.best_accuracy,
        intrinsic_dim: None,
    };
    main.register_group(
        next_group_id,
        mirror.task_name,
        env,
        embedding,
        mirror.best_accuracy,
        mirror.epochs_trained as u64,
    );
    next_group_id
}
