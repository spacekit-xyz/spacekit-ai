// ── growformer.rs ─────────────────────────────────────────────────────────────
// Curriculum-aware trainer: real bifunctor forward through the parse Pythagoras
// tree, trainable linear heads, three-term disentanglement, cross-branch dropout.

use crate::category::curriculum::{AuxLabelLoss, CurriculumScheduler};
use crate::category::disentanglement::{
    combined_loss_full, cosine_sim,
    LossBreakdown, SimpleRng,
};
use crate::category::forward::{
    align_to_dim, apply_weight_grad_sgd, bifunctor_branch_vectors,
    bifunctor_branch_vectors_backward_acc, char_hash_embed, record_embedding, zero_weight_clone,
};
use crate::category::inference::{infer_from_embedding, InferenceDetail, InferenceResult};
use crate::category::linear_head::LinearHead;
use crate::category::node::{CategoricalNode, NodeMetadata};
use crate::category::pythagoras::{nearest_pythagorean_split, PythagorasNode};
use crate::category::sentiment::{entity_to_aux_category, ParsedInput, SentimentFunctor};
use crate::category::{Layer, NodeId};
use crate::category::training::{AuxCategory, SentimentLabel, TrainingBatch};
use crate::clifford::{embed_bridge_vector, temporal_ordering_loss};
use std::collections::HashMap;

/// Small random, partially anti-correlated leaf weights so new branches are not identical
/// (avoids degenerate ortho / cosine right after [`GrowformerNode::try_grow`]).
fn grow_child_weights_pair(rng: &mut SimpleRng, a: usize, b: usize) -> (Vec<f32>, Vec<f32>) {
    let mut left = vec![0.0f32; a];
    let mut right = vec![0.0f32; b];
    let m = a.min(b);
    for i in 0..m {
        let u = (rng.gen_f32() * 2.0 - 1.0) * 0.02f32;
        left[i] = u;
        right[i] = -u + (rng.gen_f32() * 2.0 - 1.0) * 0.005f32;
    }
    for i in m..a {
        left[i] = (rng.gen_f32() * 2.0 - 1.0) * 0.02f32;
    }
    for i in m..b {
        right[i] = (rng.gen_f32() * 2.0 - 1.0) * 0.02f32;
    }
    (left, right)
}

// ── TrainerConfig ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TrainerConfig {
    /// `GrowformerNode` id used for bifunctor branch extraction.
    pub parse_node_id: usize,
    pub embed_dim: usize,
    /// Fixed width for sentiment / entity heads and disentanglement vectors.
    pub branch_dim: usize,
    pub lr: f64,
    pub head_seed: u64,
    /// When both are set (`sample_count > 0`, `every_steps > 0`), log branch norms + cosine
    /// for the first `sample_count` batch rows every `every_steps` global steps (stderr).
    /// Use to verify whether bifunctor outputs change (they stay fixed until tree weights train).
    pub branch_stats_sample_count: usize,
    pub branch_stats_every_steps: usize,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            parse_node_id: 0,
            embed_dim: 64,
            branch_dim: 32,
            lr: 0.08,
            head_seed: 42,
            branch_stats_sample_count: 0,
            branch_stats_every_steps: 50,
        }
    }
}

// ── GrowthPolicy (stage-aware) ────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GrowthPolicy {
    pub min_steps_before_grow: usize,
    /// Compared against **mean task (sentiment) CE** on the node, not combined loss.
    /// HARDEN uses a large default so [`CurriculumScheduler`]'s disentanglement gate is the
    /// primary switch; lower this to also require a very small CE before growing.
    pub grow_loss_threshold: f32,
    pub max_depth: usize,
    pub pythagorean_tolerance: f64,
    pub disentanglement_lambda: f32,
}

impl Default for GrowthPolicy {
    fn default() -> Self {
        Self {
            min_steps_before_grow: 100,
            grow_loss_threshold: 0.3,
            max_depth: 6,
            pythagorean_tolerance: 0.1,
            disentanglement_lambda: 0.1,
        }
    }
}

impl GrowthPolicy {
    pub fn scaffold() -> Self {
        Self {
            min_steps_before_grow: usize::MAX,
            grow_loss_threshold: 0.0,
            max_depth: 1,
            pythagorean_tolerance: 0.1,
            disentanglement_lambda: 0.05,
        }
    }
    pub fn loosen() -> Self {
        Self {
            min_steps_before_grow: usize::MAX,
            grow_loss_threshold: 0.0,
            max_depth: 3,
            pythagorean_tolerance: 0.1,
            disentanglement_lambda: 0.15,
        }
    }
    pub fn harden() -> Self {
        Self {
            min_steps_before_grow: 50,
            // Do not block growth on combined loss; curriculum uses dis_total < harden_grow_threshold.
            grow_loss_threshold: 10.0,
            max_depth: 6,
            pythagorean_tolerance: 0.1,
            disentanglement_lambda: 0.1,
        }
    }
}

// ── NodeState ─────────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct NodeState {
    pub steps: usize,
    /// Last step's combined loss (task + dis + aux) for logging.
    pub last_loss: f32,
    /// Mean sentiment CE used for growth gating (see [`GrowthPolicy::grow_loss_threshold`]).
    pub last_task_loss: f32,
    pub last_dis_breakdown: LossBreakdown,
    pub grow_count: usize,
    pub prune_count: usize,
}

// ── GrowformerNode ────────────────────────────────────────────────────────────

pub struct GrowformerNode {
    pub node: CategoricalNode<Vec<f32>, Vec<f32>, Vec<f32>>,
    pub state: NodeState,
}

impl GrowformerNode {
    pub fn new(id: NodeId, label: impl Into<String>, dim: usize, weights: Vec<f32>) -> Self {
        debug_assert_eq!(
            weights.len(),
            dim,
            "GrowformerNode: weight vector length must equal dim"
        );
        let meta = NodeMetadata::new(label, dim, dim);
        let composition = PythagorasNode::leaf(weights, dim);
        Self {
            node: CategoricalNode::new(id, meta, composition),
            state: NodeState::default(),
        }
    }

    pub fn try_grow(
        &mut self,
        left_w: Vec<f32>,
        right_w: Vec<f32>,
        policy: &GrowthPolicy,
        growth_enabled: bool,
    ) -> bool {
        if !growth_enabled {
            return false;
        }
        if self.state.steps < policy.min_steps_before_grow {
            return false;
        }
        if self.state.last_task_loss > policy.grow_loss_threshold {
            return false;
        }
        if self.node.composition_depth() >= policy.max_depth {
            return false;
        }
        if !self.node.is_leaf_node() {
            return false;
        }

        match self.node.grow(left_w, right_w, policy.pythagorean_tolerance) {
            Ok(()) => {
                self.state.grow_count += 1;
                println!("[grow]  {}", self.node.summary());
                true
            }
            Err(e) => {
                println!("[grow-err] {}: {}", self.node.meta.label, e);
                false
            }
        }
    }

    pub fn try_prune(&mut self, prune_enabled: bool) -> bool {
        if !prune_enabled {
            return false;
        }
        if self.node.composition_depth() <= 1 {
            return false;
        }
        if self.state.last_dis_breakdown.ortho < 0.3 {
            return false;
        }
        self.node.prune();
        self.state.prune_count += 1;
        println!("[prune] {}", self.node.summary());
        true
    }

    pub fn record_step(&mut self, combined_loss: f32, task_loss: f32, breakdown: LossBreakdown) {
        self.state.last_loss = combined_loss;
        self.state.last_task_loss = task_loss;
        self.state.last_dis_breakdown = breakdown;
        self.state.steps += 1;
    }

    pub fn summary(&self) -> String {
        format!(
            "{} | steps={} total={:.4} task={:.4} dis={:.4} ortho={:.4} grows={} prunes={}",
            self.node.summary(),
            self.state.steps,
            self.state.last_loss,
            self.state.last_task_loss,
            self.state.last_dis_breakdown.total,
            self.state.last_dis_breakdown.ortho,
            self.state.grow_count,
            self.state.prune_count,
        )
    }
}

// ── StepMetrics ───────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct StepMetrics {
    pub step: usize,
    pub stage: String,
    pub task_loss: f32,
    pub total_loss: f32,
    pub dis_breakdown: LossBreakdown,
    pub aux_loss: f32,
    pub branch_dropout_p: f32,
    pub growth_events: usize,
    pub prune_events: usize,
}

impl StepMetrics {
    pub fn display(&self) {
        println!(
            "Step {:4} [{}] task={:.4} total={:.4} aux={:.4} dropout={:.2}  grows={}  prunes={}",
            self.step,
            self.stage,
            self.task_loss,
            self.total_loss,
            self.aux_loss,
            self.branch_dropout_p,
            self.growth_events,
            self.prune_events,
        );
        self.dis_breakdown.display();
    }
}

// ── GrowformerTrainer ─────────────────────────────────────────────────────────

pub struct GrowformerTrainer {
    pub nodes: HashMap<usize, GrowformerNode>,
    pub curriculum: CurriculumScheduler,
    pub config: TrainerConfig,
    pub global_step: usize,
    pub history: Vec<StepMetrics>,
    pub sentiment_head: LinearHead,
    pub aux_head: LinearHead,
    rng: SimpleRng,
}

impl GrowformerTrainer {
    pub fn new(curriculum: CurriculumScheduler) -> Self {
        Self::with_config(curriculum, TrainerConfig::default())
    }

    pub fn with_config(curriculum: CurriculumScheduler, config: TrainerConfig) -> Self {
        let mut hrng = SimpleRng::new(config.head_seed);
        let sentiment_head = LinearHead::new_random(
            config.branch_dim,
            SentimentLabel::num_classes(),
            &mut hrng,
        );
        let aux_head = LinearHead::new_random(
            config.branch_dim,
            AuxCategory::num_classes(),
            &mut hrng,
        );
        Self {
            nodes: HashMap::new(),
            curriculum,
            config,
            global_step: 0,
            history: Vec::new(),
            sentiment_head,
            aux_head,
            rng: SimpleRng::new(99),
        }
    }

    pub fn add_node(&mut self, node: GrowformerNode) {
        self.nodes.insert(node.node.id.0, node);
    }

    fn parse_node(&self) -> Option<&GrowformerNode> {
        self.nodes.get(&self.config.parse_node_id)
    }

    /// Run one curriculum-aware training step over a batch (heads + disentanglement + curriculum).
    pub fn step(&mut self, batch: &TrainingBatch) {
        if batch.is_empty() {
            return;
        }

        self.global_step += 1;
        let step = self.global_step;

        let last_dis = self
            .history
            .last()
            .map(|m| m.dis_breakdown.total)
            .unwrap_or(1.0);
        let config = self.curriculum.config(step, last_dis);

        if self.parse_node().is_none() {
            eprintln!(
                "[GrowformerTrainer] missing parse node id {} — no step",
                self.config.parse_node_id
            );
            return;
        }
        let parse_id = self.config.parse_node_id;
        let mut acc = zero_weight_clone(
            &self
                .nodes
                .get(&parse_id)
                .expect("parse node")
                .node
                .composition,
        );

        let lr = self.config.lr as f32;
        let branch_dim = self.config.branch_dim;
        let embed_dim = self.config.embed_dim;

        let mut sent_batch: Vec<Vec<f32>> = Vec::with_capacity(batch.len());
        let mut ent_batch: Vec<Vec<f32>> = Vec::with_capacity(batch.len());
        let labels: Vec<SentimentLabel> = batch.records.iter().map(|r| r.sentiment.clone()).collect();

        let mut sum_sent_ce = 0.0f32;
        let mut sum_aux = 0.0f32;
        let n = batch.len() as f32;

        let aux_helper = AuxLabelLoss::new(config.aux_label_lambda);

        for r in &batch.records {
            let emb = record_embedding(r, embed_dim);
            let comp = &self.nodes[&parse_id].node.composition;
            let (s0, e0) = bifunctor_branch_vectors(comp, &emb, branch_dim);

            let out = crate::category::disentanglement::cross_branch_dropout(
                s0.clone(),
                e0.clone(),
                config.branch_dropout_p,
                &mut self.rng,
            );
            let s = out.left.unwrap_or(s0);
            let e = out.right.unwrap_or(e0);

            let si = r.sentiment.class_index();
            sum_sent_ce += self.sentiment_head.cross_entropy(&s, si);
            let gs = self.sentiment_head.grad_input_ce(&s, si);
            self.sentiment_head.step_ce(&s, si, lr);

            let ge = if aux_helper.is_active() {
                let logits = self.aux_head.forward(&e);
                sum_aux += aux_helper.compute(&logits, &r.resolved_aux_category().one_hot());
                let ai = r.resolved_aux_category().class_index();
                let g = self.aux_head.grad_input_ce(&e, ai);
                self.aux_head
                    .step_ce(&e, ai, lr * config.aux_label_lambda.max(0.05));
                g.into_iter()
                    .map(|x| x * config.aux_label_lambda)
                    .collect()
            } else {
                vec![0.0f32; branch_dim]
            };

            let comp = &self.nodes[&parse_id].node.composition;
            bifunctor_branch_vectors_backward_acc(comp, &emb, branch_dim, &gs, &ge, &mut acc);

            sent_batch.push(s);
            ent_batch.push(e);
        }

        if let Some(gn) = self.nodes.get_mut(&parse_id) {
            apply_weight_grad_sgd(&mut gn.node.composition, &acc, lr / n);
        }

        let mean_sent_ce = sum_sent_ce / n;
        let mean_aux = if aux_helper.is_active() {
            sum_aux / n
        } else {
            0.0
        };

        let (total_loss, breakdown) = combined_loss_full(
            mean_sent_ce,
            &sent_batch,
            &ent_batch,
            &labels,
            &config.disentanglement,
        );

        let causal_lambda = 0.15f32;
        let causal_margin = 0.5f32;
        let causal_ordering_term = {
            let mut sum = 0.0f32;
            let mut cnt = 0u32;
            for r in &batch.records {
                let Some(ref ca) = r.causal else { continue };
                let (Some(ref cs), Some(ref es)) = (&ca.cause_span, &ca.effect_span) else {
                    continue;
                };
                let cause_emb = char_hash_embed(cs, embed_dim);
                let effect_emb = char_hash_embed(es, embed_dim);
                let cause_mv = embed_bridge_vector(&cause_emb);
                let effect_mv = embed_bridge_vector(&effect_emb);
                let is_retro = ca
                    .causal_subtype
                    .as_deref()
                    .map_or(false, |s| s == "retrospective_framing");
                sum += temporal_ordering_loss(&cause_mv, &effect_mv, !is_retro, causal_margin);
                cnt += 1;
            }
            if cnt > 0 { sum / cnt as f32 } else { 0.0 }
        };

        let combined = total_loss + mean_aux + causal_lambda * causal_ordering_term;

        let log_branch = self.config.branch_stats_sample_count > 0
            && self.config.branch_stats_every_steps > 0
            && step % self.config.branch_stats_every_steps == 0;
        if log_branch {
            let n = self
                .config
                .branch_stats_sample_count
                .min(sent_batch.len());
            eprintln!(
                "[branch] step={}  (s,e change with parse-tree SGD + dropout; frozen only if train stalled)",
                step
            );
            for i in 0..n {
                let s = &sent_batch[i];
                let e = &ent_batch[i];
                let s_norm: f32 = s.iter().map(|x| x * x).sum::<f32>().sqrt();
                let e_norm: f32 = e.iter().map(|x| x * x).sum::<f32>().sqrt();
                let c = cosine_sim(s, e);
                eprintln!(
                    "  sample[{}] s_norm={:.6} e_norm={:.6} cosine={:.6}",
                    i, s_norm, e_norm, c
                );
            }
        }

        let mut growth_events = 0usize;
        let mut prune_events = 0usize;

        for (_, gnode) in self.nodes.iter_mut() {
            gnode.record_step(combined, mean_sent_ce, breakdown.clone());

            let parse_dim = gnode.node.composition.dimension;
            let (a, b) = nearest_pythagorean_split(parse_dim);
            let (left_w, right_w) = grow_child_weights_pair(&mut self.rng, a, b);
            if gnode.try_grow(
                left_w,
                right_w,
                &config.growth_policy,
                config.growth_enabled,
            ) {
                growth_events += 1;
            }
            if gnode.try_prune(config.prune_enabled) {
                prune_events += 1;
            }
        }

        self.history.push(StepMetrics {
            step,
            stage: config.stage.name().to_string(),
            task_loss: mean_sent_ce,
            total_loss: combined,
            dis_breakdown: breakdown,
            aux_loss: mean_aux,
            branch_dropout_p: config.branch_dropout_p,
            growth_events,
            prune_events,
        });
    }

    /// Full training loop.
    pub fn train(&mut self, batch: &TrainingBatch, num_steps: usize) {
        println!("╔══════════════════════════════════════════════════════╗");
        println!("║         Growformer — Curriculum Training              ║");
        println!("╚══════════════════════════════════════════════════════╝\n");
        self.curriculum.summary();
        println!();
        println!("{}", batch.coverage_report());
        let warnings = batch.validate_coverage(3);
        for w in &warnings {
            println!("[WARN] {}", w);
        }
        println!();

        for step in 0..num_steps {
            self.step(batch);
            if step == 0 || (step + 1) % 50 == 0 || step == num_steps - 1 {
                if let Some(m) = self.history.last() {
                    m.display();
                }
                for (_, gnode) in &self.nodes {
                    println!("  {}", gnode.summary());
                }
                println!();
            }
        }
        println!("=== Training Complete ({} steps) ===\n", num_steps);
    }

    /// Inference using trained heads + parse tree (deterministic hash embedding).
    /// `inferred_category` comes from the **aux head** (same space as training), not string rules.
    pub fn infer_head(&self, input: &str) -> Result<InferenceResult, &'static str> {
        let emb = char_hash_embed(input, self.config.embed_dim);
        self.infer_head_with_embedding(input, &emb)
    }

    /// Same as [`Self::infer_head`], but with a caller-supplied sentence vector (e.g. your encoder).
    /// Length may differ from `embed_dim`; it is padded or truncated to match training.
    pub fn infer_head_with_embedding(
        &self,
        input: &str,
        embedding: &[f32],
    ) -> Result<InferenceResult, &'static str> {
        self.infer_head_detail_with_embedding(input, embedding)
            .map(|d| d.to_result())
    }

    /// Logits, softmax, confidence, and heuristic vs head aux for debugging / calibration.
    pub fn infer_head_detail(&self, input: &str) -> Result<InferenceDetail, &'static str> {
        let emb = char_hash_embed(input, self.config.embed_dim);
        self.infer_head_detail_with_embedding(input, &emb)
    }

    pub fn infer_head_detail_with_embedding(
        &self,
        input: &str,
        embedding: &[f32],
    ) -> Result<InferenceDetail, &'static str> {
        let parse = self.parse_node().ok_or("parse node not registered")?;
        infer_from_embedding(
            input,
            embedding,
            self.config.embed_dim,
            self.config.branch_dim,
            &parse.node.composition,
            &self.sentiment_head,
            &self.aux_head,
        )
    }

    /// Batch over strings (each row uses hash embedding). Fails only per-item if parse node missing.
    pub fn infer_head_batch<S: AsRef<str>>(&self, inputs: &[S]) -> Vec<Result<InferenceResult, &'static str>> {
        inputs.iter().map(|s| self.infer_head(s.as_ref())).collect()
    }

    pub fn infer_head_detail_batch<S: AsRef<str>>(
        &self,
        inputs: &[S],
    ) -> Vec<Result<InferenceDetail, &'static str>> {
        inputs.iter().map(|s| self.infer_head_detail(s.as_ref())).collect()
    }

    /// Classify using `SentimentFunctor` only (no linear heads).
    /// `inferred_category` is the **string heuristic** (`entity_to_aux_category`), not the trained aux head.
    pub fn infer(&self, input: &str, functor: &SentimentFunctor) -> InferenceResult {
        let entity = input
            .split_whitespace()
            .last()
            .unwrap_or("")
            .trim_end_matches('s');
        let category = entity_to_aux_category(entity);
        let embedding = char_hash_embed(input, self.config.embed_dim);
        let parsed = ParsedInput::new(input, embedding).with_category(category.clone());
        let label = functor.forward(parsed);
        InferenceResult {
            input: input.to_string(),
            sentiment: label,
            entity: entity.to_string(),
            inferred_category: category,
        }
    }

    /// When the parse tree has exactly two **leaf** morphisms (typical after one grow),
    /// cosine similarity between their weight vectors (padded to a common length for comparison).
    ///
    /// Interpretation: values near **1.0** suggest redundant child weights (CE alone may not
    /// enforce a sentiment/entity split); well below **~0.5** often means the two branches
    /// diverged in parameter space (still not guaranteed semantics without dis→tree gradients).
    pub fn parse_leaf_pair_weight_cosine(&self) -> Option<f32> {
        let g = self.parse_node()?;
        let leaves = g.node.composition.collect_leaves();
        if leaves.len() != 2 {
            return None;
        }
        let d = leaves[0].len().max(leaves[1].len());
        let a = align_to_dim(leaves[0], d);
        let b = align_to_dim(leaves[1], d);
        Some(cosine_sim(&a, &b))
    }

    /// Print a stage-transition summary from training history.
    pub fn stage_summary(&self) {
        println!("\n── Stage Transition Summary ──");
        let mut last_stage = "";
        for m in &self.history {
            if m.stage != last_stage {
                println!(
                    "  Step {:4}: → {} (task={:.4} dis={:.4})",
                    m.step, m.stage, m.task_loss, m.dis_breakdown.total
                );
                last_stage = &m.stage;
            }
        }
        if let Some(sim) = self.parse_leaf_pair_weight_cosine() {
            println!(
                "\n── Parse bifunctor leaf weights (exactly 2 leaves) ──\n  cosine(left, right | aligned) = {:.4}",
                sim
            );
        }
    }
}

/// Back-compat alias: deterministic embedding used when no `TrainingRecord.embedding`.
#[inline]
pub fn mock_embed(input: &str, dim: usize) -> Vec<f32> {
    crate::category::forward::char_hash_embed(input, dim)
}
