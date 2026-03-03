//! GlobalObserver — routes inference, gates promotion, maintains coherence.

use std::collections::{HashMap, VecDeque};

use crate::types::GroupId;
use serde::{Deserialize, Serialize};

use super::embedding::{cosine_similarity, hidden_activation_vector, GroupEmbedding};
use super::main_dim::MainDimension;
use super::mirror_dim::MirrorDimension;
use super::promotion::{evaluate_promotion, promote, PromotionDecision, PromotionGateConfig};

#[derive(Clone, Serialize, Deserialize)]
pub struct GlobalObserver {
    pub embedding_library: Vec<GroupEmbedding>,
    pub group_activity: HashMap<GroupId, VecDeque<f32>>,
    pub activity_window: usize,
    pub promotion_gate_config: PromotionGateConfig,
    pub coherence: f32,
    /// Set each infer(); used for logging which group was routed.
    #[serde(skip)]
    pub last_chosen_group_id: Option<GroupId>,
    /// Per-group (self_sim, cross_sim, margin, score) from last infer(); for margin reporting.
    #[serde(skip)]
    pub last_routing_scores: Option<Vec<(GroupId, f32, f32, f32, f32)>>,
}

impl Default for GlobalObserver {
    fn default() -> Self {
        Self {
            embedding_library: Vec::new(),
            group_activity: HashMap::new(),
            activity_window: 1000,
            promotion_gate_config: PromotionGateConfig::default(),
            coherence: 1.0,
            last_chosen_group_id: None,
            last_routing_scores: None,
        }
    }
}

impl GlobalObserver {
    pub fn new(config: PromotionGateConfig) -> Self {
        Self {
            promotion_gate_config: config,
            ..Self::default()
        }
    }

    /// Route by margin + confidence: (self_sim - cross_sim) + 2*|out-0.5|.
    /// Scales to 1..N groups: no special cases; tie-break by lower group id.
    /// For very large N, consider pre-filtering by embedding retrieval (top-k) before full score.
    pub fn infer(
        &mut self,
        input: &[f32],
        main: &mut MainDimension,
    ) -> Vec<f32> {
        if main.group_order.is_empty() || main.embedding_library.is_empty() {
            self.last_chosen_group_id = None;
            return vec![];
        }
        // Collect (gid, hidden, out) for each group
        let mut group_hidden_out: Vec<(GroupId, Vec<f32>, Vec<f32>)> = Vec::new();
        for &gid in &main.group_order {
            let fg = match main.groups.get_mut(&gid) {
                Some(f) => f,
                None => continue,
            };
            let out = fg.env.predict(input);
            let hidden = hidden_activation_vector(&fg.env);
            group_hidden_out.push((gid, hidden, out));
            self.update_activity(gid, 1.0);
        }

        // Score = margin (self_sim - cross_sim) + strong confidence term.
        let mut best_gid = main.group_order[0];
        let mut best_score = -2.0f32;
        let mut best_out = vec![];
        let mut routing_scores: Vec<(GroupId, f32, f32, f32, f32)> = Vec::new();

        for (gid, hidden, out) in &group_hidden_out {
            let self_sim: f32 = main.embedding_library.iter()
                .find(|e| e.group_id == *gid)
                .map(|e| cosine_similarity(hidden, &e.vector))
                .unwrap_or(-1.0);
            let cross_sim: f32 = main.embedding_library.iter()
                .filter(|e| e.group_id != *gid)
                .map(|e| cosine_similarity(hidden, &e.vector))
                .fold(-1.0, |a, b| a.max(b));
            let margin = self_sim - cross_sim;
            let confidence = out.first().map(|&x| (x - 0.5).abs()).unwrap_or(0.0);
            let score = margin + 2.0 * confidence;
            routing_scores.push((*gid, self_sim, cross_sim, margin, score));
            if score > best_score || (score >= best_score - 0.05 && *gid < best_gid) {
                best_score = score;
                best_gid = *gid;
                best_out = out.clone();
            }
        }
        self.last_routing_scores = Some(routing_scores);

        // No two-group flip: report actual scorer choice. Scores show group 1 often wins
        // on both inputs (higher self_sim/margin); margin gap is wide but assignment can be wrong.
        // Use last_routing_scores in the demo to interpret robustness vs fragility.

        self.last_chosen_group_id = Some(best_gid);
        best_out
    }

    fn update_activity(&mut self, group_id: GroupId, value: f32) {
        let q = self.group_activity.entry(group_id).or_insert_with(VecDeque::new);
        q.push_back(value);
        while q.len() > self.activity_window {
            q.pop_front();
        }
    }

    /// Evaluate all active mirrors for promotion; promote those that pass.
    /// Returns names of mirrors that were promoted (and thus removed).
    pub fn evaluate_mirrors(
        &mut self,
        mirrors: &mut std::collections::HashMap<String, MirrorDimension>,
        main: &mut MainDimension,
        calibration_data: &[([f32; 2], [f32; 1])],
        next_group_id: &mut GroupId,
    ) -> Vec<String> {
        let mut to_promote = Vec::new();
        for (name, mirror) in mirrors.iter_mut() {
            match evaluate_promotion(
                mirror,
                main,
                calibration_data,
                &self.promotion_gate_config,
            ) {
                PromotionDecision::Promote => to_promote.push(name.clone()),
                _ => {}
            }
        }
        let mut promoted = Vec::new();
        for name in to_promote {
            if let Some(mirror) = mirrors.remove(&name) {
                let gid = *next_group_id;
                *next_group_id = next_group_id.saturating_add(1);
                promote(mirror, main, calibration_data, gid);
                self.embedding_library = main.embedding_library.clone();
                promoted.push(name);
            }
        }
        promoted
    }
}
