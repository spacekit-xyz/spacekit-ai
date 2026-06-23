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

    /// Fit blend weights in one forward pass over data (least squares on frozen group outputs).
    #[cfg(feature = "training")]
    pub fn fit_blend_weights_one_pass(
        group_ids: &[GroupId],
        main: &mut MainDimension,
        data: &[crate::types::Sample],
    ) -> Self {
        let g = group_ids.len();
        let n = data.len();
        let mut vg = Self::new(group_ids.to_vec());
        if n == 0 || g == 0 {
            return vg;
        }

        let mut rows: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);
        for (input, target) in data {
            let outputs = main.query(input.as_slice(), group_ids);
            if outputs.len() != g {
                return vg;
            }
            rows.push(
                outputs
                    .iter()
                    .map(|(_, o)| o.get(0).copied().unwrap_or(0.0))
                    .collect(),
            );
            targets.push(target.get(0).copied().unwrap_or(0.0));
        }

        let ridge = 1e-4f32;
        let mut oto = vec![vec![0.0f32; g]; g];
        let mut oty = vec![0.0f32; g];
        for (row, &y) in rows.iter().zip(targets.iter()) {
            for a in 0..g {
                oty[a] += row[a] * y;
                for b in 0..g {
                    oto[a][b] += row[a] * row[b];
                }
            }
        }
        for i in 0..g {
            oto[i][i] += ridge;
        }

        vg.blend_weights =
            solve_small_linear_system(&oto, &oty).unwrap_or_else(|| vec![1.0 / g as f32; g]);
        for w in &mut vg.blend_weights {
            *w = w.max(0.0);
        }
        let sum: f32 = vg.blend_weights.iter().sum();
        if sum > 1e-10 {
            for w in &mut vg.blend_weights {
                *w /= sum;
            }
        }
        vg
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

/// Solve `A x = b` for small dense systems (g ≤ 8) via Gaussian elimination.
#[cfg(feature = "training")]
fn solve_small_linear_system(a: &[Vec<f32>], b: &[f32]) -> Option<Vec<f32>> {
    let n = b.len();
    if n == 0 || a.len() != n || a.iter().any(|row| row.len() != n) {
        return None;
    }
    let mut m: Vec<Vec<f32>> = a.iter().map(|row| row.clone()).collect();
    let mut rhs = b.to_vec();

    for col in 0..n {
        let mut pivot = col;
        let mut best = m[col][col].abs();
        for r in (col + 1)..n {
            let v = m[r][col].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best < 1e-12 {
            return None;
        }
        if pivot != col {
            m.swap(col, pivot);
            rhs.swap(col, pivot);
        }
        let div = m[col][col];
        for j in col..n {
            m[col][j] /= div;
        }
        rhs[col] /= div;
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = m[r][col];
            if factor.abs() < 1e-12 {
                continue;
            }
            for j in col..n {
                m[r][j] -= factor * m[col][j];
            }
            rhs[r] -= factor * rhs[col];
        }
    }
    Some(rhs)
}

// ---------------------------------------------------------------------------
// Routing entropy guard — runtime collapse detector (§8, COMPETENCE_ROUTING_SPEC)
// ---------------------------------------------------------------------------

/// Shannon entropy (bits) of discrete route choices over a batch/window.
pub fn routing_entropy_bits(route_choices: &[usize]) -> f32 {
    if route_choices.is_empty() {
        return 0.0;
    }
    let n = route_choices.len() as f32;
    let mut counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for &k in route_choices {
        *counts.entry(k).or_insert(0) += 1;
    }
    let mut h = 0.0f32;
    for &c in counts.values() {
        let p = c as f32 / n;
        if p > 0.0 {
            h -= p * p.log2();
        }
    }
    h
}

/// True when routing distribution is dangerously collapsed (constant-specialist mode).
pub fn routing_entropy_degenerate(route_choices: &[usize], min_bits: f32) -> bool {
    routing_entropy_bits(route_choices) < min_bits
}

/// Rolling window over recent discrete route indices; triggers fallback when collapsed.
#[derive(Clone, Debug)]
pub struct RoutingEntropyGuard {
    window: std::collections::VecDeque<usize>,
    capacity: usize,
    min_bits: f32,
    triggered: bool,
}

impl RoutingEntropyGuard {
    pub fn new(window_size: usize, min_bits: f32) -> Self {
        Self {
            window: std::collections::VecDeque::with_capacity(window_size.max(1)),
            capacity: window_size.max(1),
            min_bits,
            triggered: false,
        }
    }

    /// Record a route choice; returns true if guard fires (use ensemble / abstain fallback).
    pub fn observe(&mut self, route_k: usize) -> bool {
        if self.window.len() >= self.capacity {
            self.window.pop_front();
        }
        self.window.push_back(route_k);
        if self.window.len() < self.capacity / 2 {
            return false;
        }
        let choices: Vec<usize> = self.window.iter().copied().collect();
        self.triggered = routing_entropy_degenerate(&choices, self.min_bits);
        self.triggered
    }

    pub fn is_triggered(&self) -> bool {
        self.triggered
    }

    pub fn reset(&mut self) {
        self.window.clear();
        self.triggered = false;
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
