// ── curriculum.rs ─────────────────────────────────────────────────────────────
// Three-stage training curriculum for Growformer:
//
//   Stage 1 — SCAFFOLD   (steps 0..scaffold_end)
//     Soft auxiliary category labels at λ=0.3
//     Light disentanglement pressure
//     No cross-branch dropout
//     Goal: seed initial region formation
//
//   Stage 2 — LOOSEN     (steps scaffold_end..loosen_end)
//     Auxiliary labels dropped entirely
//     Full disentanglement loss stack active
//     Cross-branch dropout at p=0.2
//     Contrastive alignment on sentiment branch
//     Goal: force sentiment morphism to become entity-agnostic
//
//   Stage 3 — HARDEN     (steps loosen_end..)
//     Grow Pythagoras nodes only when disentanglement loss < threshold
//     Prune nodes where branches have collapsed
//     Tighter orthogonality weight, lower contrastive weight
//     Goal: stabilise compositionality, not memorisation
//
// The scheduler is queried each step; it returns a CurriculumConfig that
// the trainer applies to the loss computation and growth policy.

use crate::category::disentanglement::{DisentanglementLoss, DisentanglementWeights};
use crate::category::growformer::GrowthPolicy;

// ── Stage ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stage {
    Scaffold,
    Loosen,
    Harden,
}

impl Stage {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Scaffold => "SCAFFOLD",
            Self::Loosen   => "LOOSEN",
            Self::Harden   => "HARDEN",
        }
    }
}

// ── CurriculumConfig ──────────────────────────────────────────────────────────

/// Everything the trainer needs to know for the current step.
#[derive(Debug, Clone)]
pub struct CurriculumConfig {
    pub stage: Stage,
    pub step: usize,
    /// λ for the auxiliary entity-category label loss (0.0 = disabled).
    pub aux_label_lambda: f32,
    /// Cross-branch dropout probability (0.0 = disabled).
    pub branch_dropout_p: f32,
    /// Whether Pythagoras nodes are allowed to grow this step.
    pub growth_enabled: bool,
    /// Whether Pythagoras nodes are allowed to prune this step.
    pub prune_enabled: bool,
    /// Disentanglement loss configuration for this stage.
    pub disentanglement: DisentanglementLoss,
    /// Growth policy overrides for this stage.
    pub growth_policy: GrowthPolicy,
}

impl CurriculumConfig {
    pub fn display(&self) {
        println!(
            "[Curriculum | {}] step={} aux_λ={:.2} dropout={:.2} grow={} prune={}",
            self.stage.name(),
            self.step,
            self.aux_label_lambda,
            self.branch_dropout_p,
            self.growth_enabled,
            self.prune_enabled,
        );
    }
}

// ── CurriculumScheduler ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CurriculumScheduler {
    /// Step at which Stage 1 (Scaffold) ends and Stage 2 (Loosen) begins.
    pub scaffold_end: usize,
    /// Step at which Stage 2 (Loosen) ends and Stage 3 (Harden) begins.
    pub loosen_end: usize,
    /// Disentanglement loss below which growth is permitted in Stage 3.
    pub harden_grow_threshold: f32,
    /// Disentanglement loss above which pruning is triggered in Stage 3.
    pub harden_prune_threshold: f32,
}

impl Default for CurriculumScheduler {
    fn default() -> Self {
        Self {
            scaffold_end: 100,
            loosen_end:   250,
            harden_grow_threshold:  0.15,
            harden_prune_threshold: 0.40,
        }
    }
}

impl CurriculumScheduler {
    pub fn new(scaffold_end: usize, loosen_end: usize) -> Self {
        Self {
            scaffold_end,
            loosen_end,
            ..Default::default()
        }
    }

    /// Determine current stage from global step count.
    pub fn stage(&self, step: usize) -> Stage {
        if step < self.scaffold_end {
            Stage::Scaffold
        } else if step < self.loosen_end {
            Stage::Loosen
        } else {
            Stage::Harden
        }
    }

    /// Return a full CurriculumConfig for the given step and last measured
    /// disentanglement loss (used to gate growth/pruning in Stage 3).
    pub fn config(&self, step: usize, last_dis_loss: f32) -> CurriculumConfig {
        let stage = self.stage(step);
        match stage {
            Stage::Scaffold => CurriculumConfig {
                stage,
                step,
                aux_label_lambda: 0.3,
                branch_dropout_p: 0.0,
                growth_enabled:   false, // too early — regions not yet stable
                prune_enabled:    false,
                disentanglement:  DisentanglementLoss::new(DisentanglementWeights::stage_1()),
                growth_policy:    GrowthPolicy::scaffold(),
            },
            Stage::Loosen => CurriculumConfig {
                stage,
                step,
                aux_label_lambda: 0.0,  // labels dropped
                branch_dropout_p: 0.2,
                growth_enabled:   false, // still building independence
                prune_enabled:    false,
                disentanglement:  DisentanglementLoss::new(DisentanglementWeights::stage_2()),
                growth_policy:    GrowthPolicy::loosen(),
            },
            Stage::Harden => {
                // Growth only when disentanglement is tight enough
                let growth_enabled = last_dis_loss < self.harden_grow_threshold;
                // Prune when branches have collapsed (loss too high despite depth)
                let prune_enabled  = last_dis_loss > self.harden_prune_threshold;
                CurriculumConfig {
                    stage,
                    step,
                    aux_label_lambda: 0.0,
                    branch_dropout_p: 0.1, // light dropout to maintain robustness
                    growth_enabled,
                    prune_enabled,
                    disentanglement:  DisentanglementLoss::new(DisentanglementWeights::stage_3()),
                    growth_policy:    GrowthPolicy::harden(),
                }
            }
        }
    }

    /// Progress fraction within the current stage [0.0, 1.0].
    pub fn stage_progress(&self, step: usize) -> f32 {
        match self.stage(step) {
            Stage::Scaffold => step as f32 / self.scaffold_end as f32,
            Stage::Loosen   => {
                (step - self.scaffold_end) as f32
                    / (self.loosen_end - self.scaffold_end) as f32
            }
            Stage::Harden   => 1.0, // open-ended
        }
    }

    pub fn summary(&self) {
        println!(
            "CurriculumScheduler: scaffold=[0,{}] loosen=[{},{}] harden=[{},∞]",
            self.scaffold_end,
            self.scaffold_end, self.loosen_end,
            self.loosen_end,
        );
        println!(
            "  grow_threshold={:.3} prune_threshold={:.3}",
            self.harden_grow_threshold,
            self.harden_prune_threshold,
        );
    }
}

// ── AuxLabelLoss ──────────────────────────────────────────────────────────────

/// Auxiliary entity-category label loss — used only during Stage 1 (Scaffold).
/// Provides weak structural signal to seed region formation without imposing
/// hard symbolic categories. Disabled (λ=0) in Stages 2 and 3.
pub struct AuxLabelLoss {
    pub lambda: f32,
}

impl AuxLabelLoss {
    pub fn new(lambda: f32) -> Self {
        Self { lambda }
    }

    /// Soft cross-entropy over entity category logits.
    /// `logits`: model output for entity category (one-hot targets).
    /// `target`:  one-hot encoded ground truth (soft label).
    pub fn compute(&self, logits: &[f32], target: &[f32]) -> f32 {
        if self.lambda == 0.0 {
            return 0.0;
        }
        // Softmax + cross-entropy
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
        let sum_exp: f32 = exp.iter().sum();
        let log_softmax: Vec<f32> = exp.iter().map(|e| (e / sum_exp).ln()).collect();

        let ce: f32 = target.iter().zip(log_softmax.iter())
            .map(|(t, ls)| -t * ls)
            .sum();

        self.lambda * ce
    }

    /// Disabled: returns 0 immediately without any computation.
    pub fn is_active(&self) -> bool {
        self.lambda > 0.0
    }
}
