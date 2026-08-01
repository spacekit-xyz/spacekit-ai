//! GlobalObserver — routes inference, gates promotion, maintains coherence.

use std::collections::{HashMap, VecDeque};

use crate::types::GroupId;
use serde::{Deserialize, Serialize};

use super::composition::RoutingEntropyGuard;
use super::embedding::{
    build_tag_vector, cosine_similarity, hidden_activation_vector, GroupEmbedding, TAG_VECTOR_DIM,
};
use super::main_dim::MainDimension;
use super::mirror_dim::MirrorDimension;
use super::promotion::{evaluate_promotion, promote, PromotionDecision, PromotionGateConfig};
use super::router::{apply_stickiness, LearnedRouter};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoutingConfig {
    pub stickiness: f32,
    pub tag_rerank_weight: f32,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            stickiness: 0.15,
            tag_rerank_weight: 0.3,
        }
    }
}

fn default_routing_entropy_guard() -> RoutingEntropyGuard {
    RoutingEntropyGuard::new(64, 0.3)
}

#[derive(Clone, Serialize, Deserialize)]
pub struct GlobalObserver {
    pub embedding_library: Vec<GroupEmbedding>,
    pub group_activity: HashMap<GroupId, VecDeque<f32>>,
    pub activity_window: usize,
    pub promotion_gate_config: PromotionGateConfig,
    pub coherence: f32,
    pub routing_config: RoutingConfig,
    pub learned_router: Option<LearnedRouter>,
    /// Detects constant-specialist collapse on discrete argmax routes (phase3f §8).
    #[serde(default = "default_routing_entropy_guard")]
    pub routing_entropy_guard: RoutingEntropyGuard,
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
            routing_config: RoutingConfig::default(),
            last_chosen_group_id: None,
            last_routing_scores: None,
            learned_router: None,
            routing_entropy_guard: default_routing_entropy_guard(),
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

    /// When context_tags provided: route by tag only. Else: score = cosine(input_activation, group_embedding) + stickiness.
    pub fn infer(
        &mut self,
        input: &[f32],
        main: &mut MainDimension,
        context_tags: Option<&[String]>,
    ) -> Vec<f32> {
        if main.group_order.is_empty() || main.embedding_library.is_empty() {
            self.last_chosen_group_id = None;
            return vec![];
        }
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

        let mut routing_scores: Vec<(GroupId, f32, f32, f32, f32)> = Vec::new();
        let mut scores: Vec<(GroupId, f32)> = Vec::new();

        let query_opt = context_tags.map(|tags| build_tag_vector(&tags.to_vec(), TAG_VECTOR_DIM));
        let use_tag_only = query_opt.as_ref().map_or(false, |q| !q.is_empty());

        if use_tag_only {
            let query = query_opt.as_ref().unwrap();
            for emb in &main.embedding_library {
                let tag_sim = if emb.tag_vector.is_empty() || emb.tag_vector.len() != query.len() {
                    -1.0
                } else {
                    cosine_similarity(&emb.tag_vector, query)
                };
                scores.push((emb.group_id, tag_sim));
                routing_scores.push((emb.group_id, tag_sim, 0.0, tag_sim, tag_sim));
            }
        } else {
            let use_router = self.learned_router.as_ref().map_or(false, |r| {
                r.num_groups == main.group_order.len() && input.len() == r.input_dim
            });
            let mut router_route_idx: Option<usize> = None;
            if use_router {
                let router = self.learned_router.as_mut().unwrap();
                let logits = router.predict_logits(input);
                if logits.len() == main.group_order.len() {
                    for (i, &gid) in main.group_order.iter().enumerate() {
                        let s = logits[i];
                        routing_scores.push((gid, s, 0.0, s, s));
                        scores.push((gid, s));
                    }
                    router_route_idx = scores
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i);
                }
            }
            if scores.is_empty() {
                for (gid, hidden, _out) in &group_hidden_out {
                    let self_sim: f32 = main
                        .embedding_library
                        .iter()
                        .find(|e| e.group_id == *gid)
                        .map(|e| cosine_similarity(hidden, &e.vector))
                        .unwrap_or(-1.0);
                    let cross_sim: f32 = main
                        .embedding_library
                        .iter()
                        .filter(|e| e.group_id != *gid)
                        .map(|e| cosine_similarity(hidden, &e.vector))
                        .fold(-1.0, |a, b| a.max(b));
                    let margin = self_sim - cross_sim;
                    routing_scores.push((*gid, self_sim, cross_sim, margin, self_sim));
                    scores.push((*gid, self_sim));
                }
            } else if let Some(idx) = router_route_idx {
                if self.routing_entropy_guard.observe(idx) {
                    scores.clear();
                    routing_scores.clear();
                    for (gid, hidden, _out) in &group_hidden_out {
                        let self_sim: f32 = main
                            .embedding_library
                            .iter()
                            .find(|e| e.group_id == *gid)
                            .map(|e| cosine_similarity(hidden, &e.vector))
                            .unwrap_or(-1.0);
                        let cross_sim: f32 = main
                            .embedding_library
                            .iter()
                            .filter(|e| e.group_id != *gid)
                            .map(|e| cosine_similarity(hidden, &e.vector))
                            .fold(-1.0, |a, b| a.max(b));
                        let margin = self_sim - cross_sim;
                        routing_scores.push((*gid, self_sim, cross_sim, margin, self_sim));
                        scores.push((*gid, self_sim));
                    }
                }
            }
        }
        self.last_routing_scores = Some(routing_scores);

        apply_stickiness(
            &mut scores,
            self.last_chosen_group_id,
            self.routing_config.stickiness,
        );

        let mut best_gid = main.group_order[0];
        let mut best_score = -2.0f32;
        for (gid, s) in &scores {
            if *s > best_score || (*s >= best_score - 0.05 && *gid < best_gid) {
                best_score = *s;
                best_gid = *gid;
            }
        }
        let best_out = group_hidden_out
            .iter()
            .find(|(gid, _, _)| *gid == best_gid)
            .map(|(_, _, out)| out.clone())
            .unwrap_or_default();

        self.last_chosen_group_id = Some(best_gid);
        best_out
    }

    fn update_activity(&mut self, group_id: GroupId, value: f32) {
        let q = self
            .group_activity
            .entry(group_id)
            .or_insert_with(VecDeque::new);
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
        calibration_data: &[crate::types::Sample],
        next_group_id: &mut GroupId,
    ) -> Vec<String> {
        let mut to_promote = Vec::new();
        for (name, mirror) in mirrors.iter_mut() {
            match evaluate_promotion(mirror, main, calibration_data, &self.promotion_gate_config) {
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
